# Maple preview — experimental Mac CPU floor

The native CPU route is runnable through the normal Salt server. This is a
scoped Mac qualification, not cross-platform promotion; the model registry's
`runtime_ready` marker remains false.

## Current resource recipe and 2 GB expert trial

The current experimental recipe retains up to 2,504 demand-selected expert slots
(1,999,994,880 payload bytes; 1.999994880 decimal GB) with a 3.5 decimal-GB native
RSS ceiling, CTX2048, O1000 cap, W8 and PREFILL B16. This is bounded zero-copy
mapping, not preload or a copied payload arena. Earlier sub-1GB floor results
below retain their original eight-slot scope.

The same six normal HTTP cells (three per execution mode) preserve output IDs
and all 88,688,216 exported state bytes against the eight-slot authorities.
Long P1031/O1000: mode 0 measured PREFILL 19.282942 / decode 40.033231 ms/tk,
59.924712 seconds HTTP wall, 2,981,494,784 bytes (2.981494784 GB) peak native RSS.
Mode 2 measured 20.370404 / 49.940280 ms/tk, 70.948803 seconds HTTP wall and
3,050,094,592 bytes (3.050094592 GB) peak. Increased residency helped this observed
workload but did not make collective mode faster than mode 0. No team mode was
promoted; sampled answer quality is unchanged and remains unqualified.
See `qualification/mac-cpu-expert-2gb.json` (private reference; not exported).

## All-worker collective graph — exact, performance RED, default off

`SALT_MAPLE_CPU_TEAM=2` enters one graph traversal on every existing compute
worker. Each worker advances the immutable node index itself and executes its
retained tensor slice or attention heads directly; there is no per-node pool
submission/function mailbox. Single-owner resource preparation and canonical
serial folds rotate by node. Existing SaltCache slots/leases, fixed startup
scratch requirement hooks, exact kernels and KV/commit remain authoritative.
Mode `0` is still the default; mode `1` below remains the rejected control.

Strict C99 Mac/Clang and Linux/GCC gates pass W1/W2/W8, every-worker node visits,
dependency/skew/failure/reuse, complete canonical compute/KV/state, and 12 experts
over eight bounded slots. Normal Mac HTTP haiku, persistent continuation and
P1031/O1000 library requests match mode 0 and prior authority byte-for-byte,
including all 88,688,216 exported state bytes for each request.

The matched-build long comparison is performance **RED**: PREFILL 23.208577 ->
27.959184 ms/tk, decode including close 47.053845 -> 54.619835 ms/tk, HTTP wall
70.989516 -> 83.453351 seconds (off -> collective). Native high-water RSS was
986,742,784 bytes (0.986742784 GB) off versus 968,638,464 bytes (0.968638464 GB)
collective, including earlier explicit exports. All-worker execution is proven;
lower synchronization toll is not. Barrier/resource versus arithmetic time was
not decomposed. This is one sequence per mode, not sustained performance data.
The live Acer deployment was not replaced; Linux evidence is component/build
only. See `qualification/mac-cpu-collective-graph.json` (private reference; not exported).

## Rejected worker-0 mailbox control — default off, not promoted

`SALT_MAPLE_CPU_TEAM=1` opts into a whole-program physical scope on the existing
resident compute pool. Worker 0 runs the same core interpreter within W; ready
operator phases use atomic publication/completion and bounded polling, then the
existing condition variables for parking. This does not add another graph, pool,
OpenMP dependency, arena, resource authority or arithmetic implementation.
The default is `0`, preserving the prior path. TensorOps, kernels, cache, sampler,
tokenizer and KV/commit implementations remain unchanged by this candidate.

Mac/Clang and Acer/Linux/GCC component gates pass W1/W2/W8, variable widths,
parking/skew, recursive refusal, failed-flow reuse, canonical compute arena and
KV identity. The default-off normal HTTP sequence matches prior full state.
The first team-on HTTP request failed the 1 GB ceiling at 1,180,827,648 bytes
(1.180827648 GB). An unchanged short repeat passed tokens and all 88,688,216
state bytes at 839,483,392 bytes (0.839483392 GB), but decode was 49.383119 ms/tk
versus 46.639085 ms/tk for the control. Its first failure remains unexplained;
the repeat does not establish a memory fix or performance improvement. Long
team-on/continuation/cross-host normal inference remains unrun. The existing
Acer deployment was not changed. See
`qualification/mac-cpu-continuous-team.json` (private reference; not exported).

## Current core SIMD/batch realization

The configured limits are now CTX2048 and an output cap of 1000. The earlier
HTTP raw-completion test consumed P1021/O512: 24.165465 ms/input token,
43.004967 ms/output token, 46.697011 seconds HTTP wall, and 839,204,864 bytes
native inference RSS. The same native executable ran this cell; only limits
and fixture changed. Execution, memory, and export integrity passed, but answer
quality was **RED**: repetition, unsupported policy claims, and incomplete topic
coverage. Long-context repeat/cross-width identity was not run. See
`qualification/mac-cpu-ctx2048-long-po.json` (private reference; not exported).
The CTX512 comparisons below remain the earlier, separately scoped controls.

The INT2 representation now uses SALT's existing eight-lane, unfused arithmetic
order through its SIMD module and bounded weight-stationary token tiles. The
core quant row-batch operator owns disjoint row assignment. The model binds the
QKV/expert batches already supplied by the core, using existing canonical
offsets, bounded SaltCache acquire/release-many, and the existing persistent
pool. TensorOps, scheduler, KV/publication, cache/pool implementations, numerical
policy and memory geometry were not changed in this optimization round.

Matched CTX512/W8/B16/eight-slot P12/O16 normal HTTP requests, proof off;
both rows below use warm OS pages but start from **empty native KV**:

| Metric | Preserved baseline | Core SIMD/batch candidate |
|---|---:|---:|
| PREFILL ms/input token | 69.612917 | 20.743500 |
| Decode ms/output token, including native close | 102.902250 | 39.895313 |
| HTTP wall, seconds | 2.483373 | 0.888174 |
| Combined input + output tk/s | 11.274988 | 31.525354 |
| Peak native RSS, bytes | 821,493,760 | 838,320,128 |

The first fresh-process candidate request separately measured 97.707333 ms/input,
56.632625 ms/output, and 2.079805 seconds HTTP wall. Do not mix its first-touch
boundary with the reset/warm-page comparison above. This is one fresh and one
reset request per binary, not a sustained/quiet-host performance matrix.

Strict C99 and existing contracts pass, as do 110 exact scalar/SIMD shape cells
and the real bounded-cache 12-expert/8-slot binding test. All four HTTP cells
matched output IDs and all 50,939,480 exported state bytes. Existing SIMD bodies
were preserved byte-for-byte; x86 INT2 still uses the portable scalar realization.
Gemma's logical compatibility identity remains unchanged. No new arena, worker,
payload copy/repack, state controller, or cache budget was introduced.

Source/build identities, timestamps and full receipts:
`qualification/mac-cpu-shape.json` (private reference; not exported).
The earlier floor evidence below remains the preserved pre-optimization control.

## Initial floor-qualified cell (historical control)

Apple M4 Pro arm64, CTX512, W8, PREFILL B16, eight bounded-mmap expert slots,
1,000,000,000-byte native RSS ceiling. A fresh request used 12 actual input
and 16 actual output tokens. The source snapshot is dirty relative to the
local plan checkpoint; the evidence pins every compiled source/header and
its actual executable SHA.

- Strict C99 build and existing engine/state contract gates: PASS.
- Packed INT2 versus the BF16 decoding oracle, disjoint row/batch tests: PASS.
- Final-row-only and headless CPU PREFILL tests: PASS.
- Repeated W8/B16 versus fresh W1/B8: identical output IDs, final logits,
  complete KV, and all 50,939,480 exported state bytes.
- Proof-off request followed by a separate export: byte-identical to the
  proof-on authority.
- Gemma's existing projected compatibility identity: unchanged; its
  adversarial projection test passes. This does not replace real Gemma/Qwen
  cross-platform regression qualification of the changed source.

Proof-off speed observation, 2026-10-05 07:12:59–07:13:02 UTC:

| Metric | Result |
|---|---:|
| PREFILL | 84.338333 ms/input token |
| Decode, including native finalization/cap closure | 115.608375 ms/output token |
| Total HTTP wall | 2.862903 s |
| Combined `(input + output) / HTTP wall` | 9.780282 tk/s |
| Native inference RSS/high-water reported by the engine | 779,452,416 bytes (0.779452416 GB) |
| Expert retention ceiling | 6,389,760 bytes |
| Allocated live-KV payload capacity | 50,331,648 bytes |

This is one short-request measurement with warm OS pages, not a sustained
or quiet-host speed claim. The host had substantial compressor occupancy.
Export and request-scoped state hashing are outside this speed measurement;
RSS refers to the native process, not a summed Python/native footprint.
The raw answer begins `Paris`, then continues into control-token/reasoning
text. Clean chat formatting and broader semantic quality remain unqualified.

Evidence: `qualification/mac-cpu-floor.json` (private reference; not exported).
The evidence contains complete output IDs, state/file digests, real timings,
source identity, and the three real proof cells.

## Native sampled repetition experiment

The HTTP route now accepts opt-in `repetition_penalty` in `[1, 2]`; values above
one require positive temperature. The default is one (off). C99 applies the
historical SALT `logit / penalty^count` policy over the last 64 tokens generated
in this request, excluding the prompt. **Both signs are divided**; this is not
the conventional sign-aware/Hugging Face repetition penalty. Python only
validates/transports scalars. Existing scratch logits and output IDs are reused;
raw logits, transformer arithmetic, TensorOps, resource pools and KV are unchanged.

The P1021/O512 trial at temperature 0.95, penalty 1.4, top-k 40, seed zero measured
22.158620 ms/input token, 43.049506 ms/output token including native close,
44.673739 seconds HTTP wall, and 849,920,000 bytes (0.849920000 GB) native RSS.
Same-process reset/replay matched tokens and all 88,688,216 exported state bytes.
Default/off and explicit-one CTX512 requests matched all 50,939,480 bytes of the
prior state authority. Strict C99 and the existing compatibility/contracts pass.

Quality remains **RED**: the old repetitive loop is absent in this sample, but
all 512 output tokens are unfinished `<think>` planning with unsupported
restoration details, not a delivered answer. Neither these internal exactness
checks nor sampling establish source-model fidelity or cross-platform coverage.
See `qualification/mac-cpu-repetition.json` (private reference; not exported)
and `qualification/long-po-repetition.json` (private reference; not exported).

## Official thinking-template comparison

A fixture rendered from the original Maple `chat_template.jinja` (including
`<|im_start|>assistant\n<think>\n`) was submitted through normal HTTP completions.
The original user text and sampled settings stayed unchanged; the prompt became
P1031 and the output cap increased to 1000 within CTX2048. This is a pre-rendered
fixture comparison, **not** an update to the server's chat adapter.

The executable remained byte-identical. The request generated O1000 in
69.214639 seconds: PREFILL 23.673201 ms/input token, decode including native close
44.799877 ms/output token, and peak native RSS 845,578,240 bytes (0.845578240 GB).
The 88,688,216-byte state export passed integrity verification after timing;
new repeat/cross-platform identity was not tested in this single settings trial.

Thinking closed at output token 109, followed by 891 further tokens. Semantic
quality remained **RED**: invented reservation rules and a policy section,
fourteen weeks instead of fourteen days, and a mid-sentence cap truncation.
Neither template correctness nor internal reproducibility establishes upstream
model fidelity. Evidence: `qualification/mac-cpu-official-template.json` (private reference; not exported).

## Native top-p and author-recipe trial

The normal HTTP API accepts `top_p` in `(0, 1]`, default `1` (off), alongside
the existing `top_k` range `[1, 256]`. The native C99 sampler filters the full
untempered vocabulary mass before intersecting top-k and applying temperature,
matching the inspected official MLX filter order. It reuses the existing bounded
candidate arrays and Salt counter RNG; Python only transports validated scalars.
This does not claim numerical or RNG equivalence with MLX. The original sampler
configuration ABI and disabled path are preserved; no TensorOps, KV geometry,
resource pools, model arithmetic, SIMD or engine-recipe change was made.

Strict C99, 30,720 nucleus-reference cells, 6,912 saved-prior-sampler parity
cells, scheduler gates and existing model/resource/compatibility tests passed.
The normal P1031/O1000 `top_p=1` control matched the previous official-template
response and all 88,688,216 exported state bytes. The author-recipe requests
used the official pre-rendered template, temperature 1.0, top-p 0.95, top-k 20,
seed zero and no repetition penalty, resetting the same native server per cell.

| Request | Actual P/O | PREFILL ms/tk | Decode ms/tk incl. close | HTTP s | Native RSS high-water bytes |
|---|---:|---:|---:|---:|---:|
| Official haiku prompt | 19/59 | 29.831737 | 46.083153 | 3.287206 | 866,189,312 |
| Library explanation | 1031/1000 | 23.271940 | 44.676265 | 68.678123 | 870,400,000 |
| Haiku reset/replay | 19/59 | 33.348421 | 42.510525 | 3.143154 | 870,400,000 |

The haiku is on-topic and naturally stops, with identical IDs and complete state
on replay, but does not meet its own claimed classic syllabic form. The library
answer remains **RED**: unsupported fine-policy details, confusing patron/staff
roles, and output-cap truncation. There is no semantic-recovery claim. RSS is
native process high-water, not an independent per-request or combined-process
measurement. Existing sampling defaults remain unchanged. See
`qualification/mac-cpu-top-p.json` (private reference; not exported).

## Acerbox deployment and official CPU reference

The same source snapshot was deployed in a dedicated host-local Acerbox workspace
(GB10 ARM64 Linux), using CPU W8/B16/CTX2048/eight expert slots. The
Mac and Acer greedy haiku/library requests matched all output IDs and complete
88,688,216-byte state exports. This is two-host ARM64 evidence, not x86 coverage.

The comparison with DeepGrove's official llama.cpp fork and TQ2_0/FP16-head model
found a source-tokenizer mismatch: the original Hugging Face and GGUF tokenizers
produce P1009 for the long fixture, while Salt produces P1031 by splitting
newline/punctuation merges. Even on identical prompt IDs, outputs diverge after
65 shared haiku tokens and 69 shared library tokens. Float/representation
differences remain confounded; this is not an original-BF16 numerical oracle.
Both implementations also produced semantic defects/output truncation in the
long fixture. No tokenizer or arithmetic repair was performed in this round.

Matched P1031/O1000 library timing on Acer: Salt PREFILL 37.73 / decode 63.22
ms/tk versus reference 3.45 / 16.94 ms/tk, at different memory operating points:
Salt native peak 914,874,368 bytes (0.914874368 GB); reference lifetime peak
6,792,388,608 bytes (6.792388608 GB). Salt decode includes native finalization.
See `qualification/acer-cpu-reference-comparison.md` (private reference; not exported)
and its JSON receipt for full identities, actual outputs and qualification limits.

## Setup and launch

Use absolute paths to the selected project Python and dedicated workspace:

```bash
export MAPLE_ROOT=/path/to/dedicated-maple-workspace
export SALT_BUILD_PYTHON=/absolute/path/to/project/python3
export SALT_SERVER_PYTHON="$SALT_BUILD_PYTHON"
export MAPLE_BIN_DIR="$MAPLE_ROOT/bin"
cd "$MAPLE_ROOT/repo"

"$SALT_BUILD_PYTHON" models/maple-preview/setup.py fetch \
  --source "$MAPLE_ROOT/weights/colibri-int2"
"$SALT_BUILD_PYTHON" models/maple-preview/setup.py import \
  --source "$MAPLE_ROOT/weights/colibri-int2" \
  --package "$MAPLE_ROOT/packages/int2-v1"
"$SALT_BUILD_PYTHON" models/maple-preview/setup.py prepare-tokenizer \
  --package "$MAPLE_ROOT/packages/int2-v1"
bash models/maple-preview/build.sh
SALT_GPU=0 bash server/serve-launch.sh --backend maple \
  --host 127.0.0.1 --port 18095 \
  --maple-package "$MAPLE_ROOT/packages/int2-v1" \
  --maple-runner "$MAPLE_ROOT/bin/maple-server"
```

Import and runtime-tokenizer preparation refuse existing outputs; do not
rerun them over a materialized package or overwrite a receipt. Normal startup
checks authenticated identities without rescanning weight payloads.

The floor tokenizer is a model-local derived artifact: merge arrays become
unambiguous `left right` strings already supported by the existing C tokenizer.
Every vocabulary ID, merge rank, added token and other tokenizer field is
preserved and the complete parsed document round-trips exactly. Original
source files, tokenizer/parser code, model weights and inference math are
unchanged by that representation change. The derived receipt binds both the
original tokenizer SHA and the new file identity/hash.

Issue an ordinary completion request:

```bash
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  --data-binary @models/maple-preview/qualification/bench.json \
  http://127.0.0.1:18095/v1/completions
```

The server is loopback-only and experimental. CPU is the only admitted
backend. `/v1/state/reset`, `/v1/state/export`, and `/v1/state/import` operate
on the native session; export/import require an explicit named action.
A context/output change does not authorize replacing a healthy live session.
