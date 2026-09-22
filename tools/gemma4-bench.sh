#!/usr/bin/env bash
# Direct Gemma benchmark runner: one requested mode, no prerequisites or preflight.
set -euo pipefail

usage() {
  echo "usage: $0 MODE CTX GEN PROMPT_FILE [REPEATS] [LOG_DIR] [QA_ARGS...]"
  echo "       $0 report LOG_FILE"
  echo "       $0 oracle LOG_FILE [LOG_FILE...]"
  echo "modes: mac-cpu mac-metal ec2-cpu cuda-full"
}

oracle_logs() {
  "${GEMMA4_PYTHON:-python3}" - "$@" <<'PY'
import re, sys
from pathlib import Path

rows = []
for name in sys.argv[1:]:
    text = Path(name).read_text(encoding="utf-8", errors="replace")
    line = next((line for line in text.splitlines()
                 if line.startswith("GEMMA4_DPR_PROOF")), "")
    fields = dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))
    draft_line = next((line for line in text.splitlines()
                       if line.startswith("dpr_draft_ids=")), "")
    serial_line = next((line for line in text.splitlines()
                        if line.startswith("dpr_serial_ids=")), "")
    if not fields or not draft_line or not serial_line:
        raise SystemExit(f"ORACLE FAIL {name}: missing DPR proof identity")
    draft = tuple(map(int, draft_line.split("=", 1)[1].split(",")))
    serial = tuple(map(int, serial_line.split("=", 1)[1].split(",")))
    block = int(fields["block"])
    bonus = int(fields["bonus"])
    checks = {
        "accepted": int(fields["accepted"]) == block,
        "compute_ids": draft == serial[:block],
        "bonus": len(serial) == block + 1 and bonus == serial[block],
        "state_kv_bytes": int(fields["differing_bytes"]) == 0,
        "state_geometry": int(fields["state_bytes"]) > 0 and int(fields["kv_arena_bytes"]) > 0,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise SystemExit(f"ORACLE FAIL {name}: {','.join(failed)}")
    rows.append((name, block, bonus, draft, serial,
                 int(fields["state_bytes"]), int(fields["kv_arena_bytes"])))

anchor = rows[0][1:]
for row in rows[1:]:
    if row[1:] != anchor:
        raise SystemExit(f"ORACLE FAIL {row[0]}: cross-platform identity mismatch")
print(f"ORACLE PASS logs={len(rows)} block={anchor[0]} bonus={anchor[1]} "
      f"compute_ids=IDENTICAL state_kv=IDENTICAL "
      f"state_gb={anchor[4] / 1_000_000_000:.3f} "
      f"kv_arena_gb={anchor[5] / 1_000_000_000:.3f}")
PY
}

report_log() {
  "${GEMMA4_PYTHON:-python3}" - "$1" <<'PY'
import re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
def marker(prefix):
    line = next((line for line in text.splitlines() if line.startswith(prefix)), "")
    return dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))
start = marker("GEMMA4_QA_EXECUTION_START")
done = marker("GEMMA4_QA_EXECUTION_DONE")
dpr = marker("GEMMA4_DPR_PROOF")
if not start or (not done and not dpr):
    raise SystemExit("REPORT unavailable: run did not complete")
if dpr and not done:
    accepted = int(dpr["accepted"])
    verify = float(dpr["verify_s"])
    serial = float(dpr["serial_verify_s"])
    memory_lines = [dict(re.findall(r"([A-Za-z0-9_]+)=([^ ]+)", line))
                    for line in text.splitlines() if line.startswith("GEMMA4_MEMORY")]
    memory = memory_lines[-1]
    print("WATERFALL")
    print(f"  load/startup              {float(dpr['load_s']):9.3f} s")
    prefill = float(dpr["prefill_s"])
    prompt = int(start["prompt_tokens"])
    print(f"  prefill ({prompt} input)       {prefill:9.3f} s  {prefill * 1000 / prompt:9.3f} ms/input")
    for label, key in (("snapshot", "snapshot_s"), ("draft", "draft_s"),
                       ("recovery", "recover_s"), ("pending", "pending_s")):
        print(f"  {label:27s} {float(dpr[key]):9.3f} s")
    print(f"  target N                  {int(dpr['block']):9d}  accepted={accepted} bonus={dpr['bonus']}")
    print(f"  target verify             {verify:9.3f} s  {verify * 1000 / accepted:9.3f} ms/accepted")
    print(f"  serial verify control     {serial:9.3f} s  {serial * 1000 / accepted:9.3f} ms/accepted")
    print(f"  target speedup            {serial / verify:9.3f}x")
    print(f"  state/KV identity         {'PASS' if int(dpr['differing_bytes']) == 0 else 'FAIL':>9s}")
    print(f"  submit/fence/publish      {dpr['engine_submissions']}/{dpr['completion_fences']}/{dpr['intermediate_publications']}+{dpr['target_frontier']}")
    print(f"  target GPU program        {dpr['target_gpu_program']}  CUDA={dpr['target_cuda_program']} ROCm={dpr['target_rocm_program']}")
    peak = int(memory["resident_peak_bytes"])
    current = int(memory["current_rss_bytes"])
    print(f"  RSS peak                  {peak / 1_000_000_000:9.3f} GB")
    print(f"  RSS final                 {current / 1_000_000_000:9.3f} GB")
    raise SystemExit(0)
load = float(start["load_s"])
prompt = int(done["prompt_tokens"])
outputs = int(done["output_steps"])
prefill = float(done["prefill_s"])
ttft = float(done["first_token_ready_s"])
total = float(done["total_s"])
post_steps = max(outputs - 1, 0)
post = max(total - ttft, 0.0)
def gb(key): return int(done.get(key, 0)) / 1_000_000_000
def integer(key): return int(done.get(key, 0))
print("WATERFALL")
print(f"  load/startup              {load:9.3f} s")
print(f"  prefill ({prompt} input)       {prefill:9.3f} s  {prefill * 1000 / prompt:9.3f} ms/input")
print(f"  TTFT from execution       {ttft:9.3f} s")
print(f"  fresh-process TTFT        {load + ttft:9.3f} s")
if post_steps:
    print(f"  post-first decode ({post_steps})  {post:9.3f} s  {post * 1000 / post_steps:9.3f} ms/step  {post_steps / post:7.3f} step/s")
print(f"  request wall              {total:9.3f} s")
print(f"  startup + request         {load + total:9.3f} s")
print(f"  output                    {outputs:9d} steps  stop={done['stop_token']}")
print(f"  RSS peak                  {gb('resident_peak_bytes'):9.3f} GB")
print(f"  RSS final                 {gb('current_rss_bytes'):9.3f} GB")
print(f"  physical footprint final  {gb('physical_footprint_bytes'):9.3f} GB")
print(f"  startup forecast          {gb('startup_forecast_bytes'):9.3f} GB")
print(f"  expert cache capacity     {gb('expert_cache_capacity_bytes'):9.3f} GB")
print(f"  GPU submissions           dense={integer('gpu_dense_submissions')} router={integer('gpu_router_submissions')} gate_up={integer('gpu_expert_gate_up_submissions')} down={integer('gpu_expert_down_submissions')} head={integer('gpu_head_submissions')}")
print(f"  GPU failures/copies       failures={integer('gpu_failures')} copied_weights={gb('gpu_weight_copied_bytes'):.3f} GB")
print(f"  proof/state               package_proof={done['package_proof']} envelope={done['state_envelope']}")
if dpr:
    accepted = int(dpr["accepted"])
    verify = float(dpr["verify_s"])
    serial = float(dpr["serial_verify_s"])
    print(f"  target N                  {int(dpr['block']):9d}  accepted={accepted} bonus={dpr['bonus']}")
    print(f"  target verify             {verify:9.3f} s  {verify * 1000 / accepted:9.3f} ms/accepted")
    print(f"  serial verify control     {serial:9.3f} s  {serial * 1000 / accepted:9.3f} ms/accepted")
    print(f"  target speedup            {serial / verify:9.3f}x")
    print(f"  target GPU program        {dpr['target_gpu_program']}  CUDA={dpr['target_cuda_program']} ROCm={dpr['target_rocm_program']}")
begin = text.rfind("GEMMA4_QA_RESPONSE_BEGIN\n")
end = text.find("\nGEMMA4_QA_RESPONSE_END", begin)
if begin >= 0 and end > begin:
    print("RESPONSE")
    print(text[begin + len("GEMMA4_QA_RESPONSE_BEGIN\n"):end])
PY
}

if [[ ${1:-} == -h || ${1:-} == --help ]]; then usage; exit 0; fi
if [[ ${1:-} == report ]]; then [[ $# == 2 ]] || { usage >&2; exit 2; }; report_log "$2"; exit; fi
if [[ ${1:-} == oracle ]]; then [[ $# -ge 2 ]] || { usage >&2; exit 2; }; shift; oracle_logs "$@"; exit; fi
[[ $# -ge 4 ]] || { usage >&2; exit 2; }

mode=$1
ctx=$2
gen=$3
prompt=$4
repeats=${5:-1}
log_dir=${6:-/tmp/salt-gemma4-bench}
shift $(( $# >= 6 ? 6 : $# ))
extra=("$@")

case "$mode" in
  mac-cpu)   recipe=mac ;;
  mac-metal) recipe=mac-metal ;;
  ec2-cpu)   recipe=ec2 ;;
  cuda-full) recipe=spark ;;
  *) echo "unknown mode: $mode" >&2; usage >&2; exit 2 ;;
esac
[[ $ctx =~ ^[1-9][0-9]*$ && $gen =~ ^[1-9][0-9]*$ ]] || { echo "CTX and GEN must be positive integers" >&2; exit 2; }
[[ $repeats =~ ^[1-9][0-9]*$ ]] || { echo "REPEATS must be a positive integer" >&2; exit 2; }
[[ -f $prompt ]] || { echo "prompt not found: $prompt" >&2; exit 2; }

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target_n=
while IFS='=' read -r key value; do
  [[ $key != SALT_TARGET_ROUTE_N ]] || target_n=$value
done < "$repo/models/gemma4-26b-a4b/configs/$recipe/engine.config"
: "${target_n:?recipe does not define SALT_TARGET_ROUTE_N}"
: "${GEMMA4_SOURCE_DIR:?set GEMMA4_SOURCE_DIR}"
: "${GEMMA4_POOL:?set GEMMA4_POOL}"
: "${GEMMA4_AUTH_RECEIPT:?set GEMMA4_AUTH_RECEIPT}"
receipt=${GEMMA4_RECEIPT:-${GEMMA4_POOL}.import.json}
runner=${GEMMA4_RUNNER:-$repo/gemma4-qa}
python=${GEMMA4_PYTHON:-python3}
workers=${GEMMA4_WORKERS:-8}
mkdir -p "$log_dir"

for name in ${!SALT_@} ${!SPECTRUM_@}; do unset "$name"; done
export SALT_GEMMA_PLATFORM_RECIPE=$recipe
export SALT_WATERFALL=1
[[ $mode != cuda-full ]] || export SALT_CUDA_PAGEABLE_MMAP=0

command=(env -u PYTHONPATH "$python" "$repo/tools/gemma4-qa.py"
  --source-dir "$GEMMA4_SOURCE_DIR" --pool "$GEMMA4_POOL"
  --receipt "$receipt" --auth-receipt "$GEMMA4_AUTH_RECEIPT"
  --runner "$runner" --ctx "$ctx" --gen "$gen" --workers "$workers"
  --prompt-file "$prompt" --dpr-proof "$target_n" --target-frontier
  "${extra[@]}")

for ((repeat=1; repeat<=repeats; repeat++)); do
  log="$log_dir/${mode}-ctx${ctx}-o${gen}-r${repeat}.log"
  echo "RUN mode=$mode recipe=$recipe target_n=$target_n ctx=$ctx gen=$gen repeat=$repeat/$repeats log=$log"
  "${command[@]}" 2>&1 | tee "$log"
  report_log "$log"
  oracle_logs "$log"
done
