# Gemma 4 26B-A4B package

Model/package authority for Salt’s `gemma-4-26b-a4b-it` backend.

```text
native arithmetic authority: portable C99
operating default: CPU
mixed compute node: fail-closed
full compute node: explicit CUDA/Spark and ROCm/V520 candidates; Metal qualification
portable writable state: G4KVC006 (G4KVC005/G4KVC004 legacy-read-only)
mindset default without attachment: empty prefix (`mindset_end=0`)
protected-prefix C + default-CPU and V520 boot attachment/recovery: qualified in their named scopes; full-context GB10 HMM boot attachment: unqualified
catalog selection/live swapping: unwired
runtime_ready=true
```

The transition Nomogram (`ND`) and Mindset Attention Nomogram (`NM`) are
orthogonal engine mechanisms. Current Gemma evidence covers `ND_ONLY` plus a
focused-GREEN A0 NM expert-prefetch adaptation for `NM_ONLY` and `ND_NM`. NM
remains default-off and performance-unqualified. The historical Mentor-family
selector path must not be advertised as NM; public CLI booleans own both
activations.
`--gemma-dpr-root` selects this model's isolated DPR collection; collections are
not shared across models.

This readiness marker is model-scoped. The general DPR engine is independently
READY. Gemma's authenticated MLX affine-Q4 default CPU runtime and persistent
text/one-image API are ready; optional accelerators, catalog selection, and live
state swapping retain their own gates.

Full-text pure-GPU HMM engine exactness and repeat closure are fixed; one
explicit CTX262144/O128 four-engine Acerbox serving candidate also passed its
session/reset/spill/SSE/real-use boundary. The mode remains default-off. Current
server code aligns multimodal package-role authentication with the qualified
readiness marker; it does not widen readiness or add HTTP client
authentication/TLS.

Detailed operating procedure: `SOP.md` (private reference; not exported). HTTP/server contract:
`../../server/LATEST_WORKING.md` (private reference; not exported). Canonical server setup and
API client: `configs/server/README.md` (private reference; not exported). Current
repository boundary: `../../BREAKING-POINT.md` (private reference; not exported).

## Source tracks

### MLX affine-Q4

Pinned source contract: [`mlx4-source/source-contract.json`](mlx4-source/source-contract.json).

- External authenticated MLX encoding; MLX is not the runtime backend.
- CPU is the operating path on Mac/Linux.
- Imported expert pool remains the canonical routed payload.
- State compatibility: [`kv-cache-compat.json`](kv-cache-compat.json).

### NVFP4

Pinned source contract: [`nvfp4-source/source-contract.json`](nvfp4-source/source-contract.json), revision `985ef63dfdaf51a74df35bac3ae54a69d7bbf351`.

- External compressed-tensors source; repository retains bounded authority metadata, not the payload.
- Portable C99 owns E2M1/E4M3FN, activation group-16 Q/DQ, reciprocal divisor semantics, and canonical projection fold.
- Exact source-backed CUDA is qualified on GB10 for focused/full-graph evidence.
- State compatibility: [`kv-cache-compat-nvfp4.json`](kv-cache-compat-nvfp4.json).

The tracks are not state-compatible. Cross-track loading rejects.

## Package map

| Path | Role |
|---|---|
| `SOP.md` | current model operation and qualification procedure |
| `engine.config` | default/compatibility blueprint |
| `configs/mac/engine.config` | Darwin CPU recipe |
| `configs/ec2/engine.config` | Linux/x86 CPU recipe |
| `configs/spark/engine.config` | CUDA/Spark qualification recipe |
| `configs/spark-hmm/engine.config` | low-resident full-text pageable-CUDA qualification recipe |
| `configs/spark-hmm-vision/engine.config` | runtime-ready CPU-text/pageable-CUDA-vision recipe |
| `configs/mac-metal/engine.config` | bounded full-Metal qualification recipe |
| `configs/rocm/engine.config` | conservative AMD ROCm full-GPU qualification recipe |
| `configs/server/setup.sh` | scheduler/existing-server HTTP client utility; launch actions fail closed to the portable launcher |
| `configs/server/README.md` | server-only setup and operational procedure |
| `package-plan.json` | architecture/package contract |
| `mlx4-source/` | MLX-Q4 source authority |
| `nvfp4-source/` | NVFP4 source authority |
| `mlx4-package/source-role-contract.json` | imported package role closure |
| `kv-cache-compat*.json` | quant-scoped state compatibility closures |
| `server.config` | serving defaults and limits |
| `qualification/` | retained fixed workloads and accepted/scoped evidence |
| `qualification/ec2-cpu-bnf-6g14g-2026-08-29.md` | EC2 CPU 6-GB expert / 14-GB RSS B/N/F baseline and equal-work verdict |
| `qualification/gpu-bnf-metal-cuda-mixed-2026-08-29.md` | Metal and diagnostic CUDA B/N/F profiles, mixed fail-closed result, and AMD full-GPU reachability blocker |
| `qualification/nfq-n16f1q4-cross-platform-2026-08-31.md` | retained 2026-08-31 N/F/Q cross-platform exactness and performance evidence |
| `qualification/rocm-backend-implementation-2026-08-29.md` | gfx1011 HIP backend compile/link checkpoint, conservative resource model, and runtime blocker |
| `qualification/dpr-aiayn-ec2/` | portable AIAYN-004 Native/ND/NM/ND+NM fixture bundle and benchmark harness |
| `qualification/finegrain-200x30.*` | canonical exact 200-input/30-output five-cell fine-grained phase matrix |

Production payloads, tokenizers, imported pools, and receipts remain external and read-only.

## Platform defaults and flag meanings

Every recipe is merged over this model's `engine.config`; explicit values are
model/platform policy, not portable-C constants.

| Recipe | Compute | Prefill B | NFQ N/F/Q | Recipe X maximum | Expert budget | RSS ceiling | Target-program default |
|---|---|---:|---:|---:|---:|---:|---:|
| `mac` | CPU | 512 | 64/1/4 | 64 | 1 GB | 4.5 GB | off |
| `ec2` | CPU | 512 | 32/1/4 | 32 | 6 GB | 14 GB | off |
| `spark` | full CUDA | 256 | 48/1/4 | 48 | 13 GB + preload | 60 GB | on |
| `spark-hmm` | full-text pageable CUDA candidate | 256 | 48/1/4 | 48 | 2 GB | 16 GB | on |
| `spark-hmm-vision` | CPU text + CUDA-HMM vision | 512 | 32/1/4 | 32 | 1 GB | 7 GB | off |
| `mac-metal` | full Metal qualification | 512 | 64/1/4 | 64 | 1 GB | 4.5 GB | on |
| `rocm` | full HIP low-cost candidate | 128 | 104/1/4 | 16 | 3 GB | 7.5 GB | on |

- `SALT_PREFILL_B` is the prompt-chunk cap; actual prompt chunks may be smaller.
- `SALT_TARGET_ROUTE_N/F/Q` is the model/platform NFQ cheap-check policy. It is
  not prefill width or causal model-row geometry; total candidate checks are
  `N×F×Q`. Q is the bounded alternative lifetime for each NFQ tile.
  Native startup owns policy and executor selection. Fixed candidate IDs
  are a QA input, not a production server API. DPR is optional and cannot select
  the base serving method; transport code may not construct candidates or choose
  an ordinary B1 fallback.
- `SALT_TARGET_X` is the independent causal X-TARGET row count. X, not N,
  sets the recipe ceiling on consecutive target rows. Normal serving forces X1
  below `SALT_TARGET_KV_WARMUP_ROWS=512` pre-request live-KV rows and permits at
  most `SALT_TARGET_WARM_X=4` afterward. Current-prompt PREFILL is excluded.
- `SALT_TARGET_GPU_PROGRAM` selects the complete target-program realization.
  ROCm enables it for the explicit slower V520 tier. The current eight-position-
  worker attention path is state-exact at `345.003 ms/accepted` for historical
  X27 evidence; this is
  not a Mac/GB10 parity claim.
- `SALT_TEXT_FINE_TOKEN` selects the complete 689-cell ordinary B1 control. It
  remains off because CUDA measured `190.892 ms/token` versus `~65–69 ms` for the
  accepted ordinary path; it is not a generic "continuous mode" switch.
- `SALT_TARGET_CPU_GRAPH` selects the retained CPU realization of the 689-cell
  program for admitted target/fine-token work. `SALT_TOKEN_CPU_GRAPH` is a
  separate ordinary-token control. Both default off; neither is implied by N/F/Q.
- `SALT_PREFILL_OPERATION_FLOW` wraps the existing prefill operation in the
  retained CPU flow only where explicitly enabled; it does not alter arithmetic
  or turn prefill into X-TARGET.
- `SALT_PREFILL_DECODE_LOOKAHEAD=0` keeps the engine-level cross-phase resource
  preparation candidate default-off. It is independent of DPR/NM, seeds
  canonical layer-0 file read-ahead after the PREFILL head, and rolls next-layer
  weight advice through the first ordinary DECODE without starting arithmetic
  before PREFILL publication. The corrected Mac CPU release-policy cell was
  exact and faster, but Mac CPU already retains dense layers and accelerator
  modes remain separately unqualified, so it is not an operating default.
- `SALT_GEMMA_MEMORY_LIMIT_GB` is an all-inclusive decimal current-RSS ceiling;
  resident file-backed pages count. Mac recipes use 4.5 GB.

The closed [PR #91](https://github.com/rhCat/salt/pull/91) implementation and
acceptance record is
`../../docs/workorder/wo0908resourcepool.md` (private reference; not exported).
It records the pure-GPU HMM fix (`50f2f00`, merged-equivalent runtime/config
`13fa8f6c`), the 12/12 four-context repeat gate, later `896ef6c`/`988cc887`
identity, and the focused CUDA/Metal TARGET closures. HMM remains
explicit/default-off pending serving/E2E promotion; that remaining work is not
an engine exactness repair. Latest fresh HMM measurements are `152.792 ms/post-first token`, with `0.967 s`
warm TTFT and `3.271811072 GB` peak for a historical tool-owned B1 conversation;
that row proves state continuity only and is rejected as serving-route evidence.
Canonical X-TARGET-48 is exact at `30.975 ms/accepted token` and
`3.719786496 GB` peak. These B1 and TARGET rows measure different work.

The scoped predecessor resource-lifecycle record remains archived at
`../../docs/archive/workorders/2026-09-08/workorder0905-hmm-resource-lifecycle.md` (private reference; not exported).
The older cross-phase scheduler order remains archived at
`../../docs/waveform/archive/workorder0903-cross-phase-resource-wavefront.md` (private reference; not exported).

## Model contract

- 30 text layers: 25 sliding-window and five full-attention layers;
- hidden width 2816;
- parallel dense and routed GeGLU branches;
- 128 experts, top-k 8, routed width 704, dense width 2112;
- Gemma-specific router normalization, residual/post-norm order, and tied head;
- BF16 vision tower and affine-Q4 vision-to-text projection;
- exact image-token insertion for canonical 384×384 fixtures;
- configured terminal IDs 1, 50, and 106.

Malformed roles, dtype/shape/extent drift, unsupported preprocessing, nonfinite execution, or compatibility mismatch fail closed.

## Validate authority

```bash
python3 tools/gemma4-package.py validate-plan
python3 tools/gemma4-nvfp4-source.py validate-authority
```

Standard fine-grained phase contract and matrix:

```bash
make test-gemma4-finegrain-contract

python3 tools/test/gemma4-finegrain-matrix.py run \
  --cell mac-cpu --repo . --recipe mac --workers 3 \
  --source-dir /path/to/authenticated/source \
  --pool /path/to/pool.bin \
  --receipt /path/to/pool.bin.import.json \
  --auth-receipt /path/to/package.auth.json \
  --output /path/to/evidence/mac-cpu
```

Run the required Acer CPU, Acer CUDA, Mac CPU, Mac Metal, and EC2 CPU
cells independently, then pass their five `summary.json` files to the tool's
`aggregate` command. Host transport, credentials, and lifecycle remain outside
the repository; the checked-in manifest owns the exact fixture, workload,
required cell IDs, response-token identity, and aggregate shape.

Explicit full NVFP4 source audit:

```bash
python3 tools/gemma4-nvfp4-source.py validate-download \
  --source-dir /path/to/pinned/nvfp4/package
```

Build native runners and receipts:

```bash
make gemma4-build-receipts
```

Build, authenticate, and serve one persistent engine in the foreground:

```bash
server/test-gemma4-launch.sh \
  --source-dir /path/to/pinned-source \
  --pool /path/to/imported/pool.bin
```

## Serve

Use `server/test-gemma4-launch.sh` for machine-local setup, build,
authentication, and foreground launch. After READY, do not invoke a native
runner or backend directly. `configs/server/setup.sh` remains a scheduler and
existing-server API client only; if an endpoint is missing, stop instead of
creating a bypass.

```bash
server/test-gemma4-launch.sh \
  --source-dir /path/to/pinned-source \
  --pool /path/to/imported/pool.bin
```

The generic server defaults to one persistent engine and delegates inference to
the Gemma model-owned native adapter. A measured platform may select a
bounded session-affine pool with `GEMMA4_ENGINE_PROCESSES=N` and
`bash server/model/gemma4-endpoint-launch.sh`. Every unchanged child server keeps
private live KV and native COMMIT authority; the transport scheduler only
assigns sessions, spills/restores only idle committed sessions when needed, and
estimates remaining work from observed DPR coverage. Production serving source
does not live under `tools/`; QA wrappers remain non-production.

For full-context GB10 hosting, use `server/model/gemma4-endpoint-launch.sh` with the
explicit `spark-hmm` recipe, CTX262144/O128, W8, four engines, and one heavy
request at a time. For the low-cost V520 candidate, use the same endpoint entry
with `ROCM=1`, recipe `rocm`, one engine, W8, CTX5120/O128,
N104/F1/Q4 and X16, a 3-GB
expert budget, and the 7.5-GB RSS ceiling. Rebuild explicitly for `gfx1011`
before launch.

A dedicated informational appliance may add:

```bash
GEMMA4_KV_CACHE_ROOT=/private/session-root \
GEMMA4_STARTUP_KV_MANIFEST=/private/startup-kv.json \
GEMMA4_DPR_ROOT=/private/dpr-root \
GEMMA4_DPR_MODE=dynamic \
  bash server/model/gemma4-endpoint-launch.sh
```

The startup manifest is an operator file with no group/other permission bits
(normally mode 0600) that names an authenticated `G4KVC006` cache under the
private root. It is loaded before READY and is not a
client-facing cache selector.

The API has no built-in client authentication or TLS. Keep it on loopback or
behind an authenticated gateway. Gemma's model-scoped `runtime_ready=true`
marker applies only to the authenticated admitted runtime described here.

## Mindset and replaceable facts

The persistent `--gemma-shared-kv` path attaches an explicit immutable mindset
`0,M)`. Native C also supports replaceable private factual
rows `[M,F)`. Facts must come from a complete `G4KVC006` reference compiled
under the exact mindset; every parent K/V byte is compared before suffix commit.
Clear returns to `M`, and combined export is canonical. No server/catalog/HTTP
live selector swaps this state; persistent startup attaches one operator-chosen
protected artifact.

## Test and qualification

No-model contracts:

```bash
make test-gemma4-e2e-contract
make test-gemma4-phase5-strict
make test-gemma4-mindset-kv
```

Platform builds:

```bash
make CC=clang test-gemma4-phase5
make CC=gcc test-gemma4-phase5
```

Expensive authenticated E2E:

```bash
make test-gemma4-e2e \
  GEMMA4_SOURCE_DIR=/path/to/pinned-source \
  GEMMA4_POOL=/path/to/imported/pool.bin \
  GEMMA4_AUTH_RECEIPT=/path/to/package.auth.json
```

Generated reports remain under `/tmp` unless deliberately copied into a dated archive.

## Authority and evidence index

- Current runtime boundary: `../../BREAKING-POINT.md` (private reference; not exported).
- Current model operation and controls: `SOP.md` (private reference; not exported).
- Closed PR #91 TARGET resource-pool/HMM closure record: `../../docs/workorder/wo0908resourcepool.md` (private reference; not exported).
- Terminal tool-call KV reproducibility: `tool-call-reproducibility/RESULT.md` (private reference; not exported).
- V520 learned-DPR grand scheduler: `tool-call-reproducibility/experiments/dpr-terminal/V520-GRAND-SCHEDULER-REPORT.md` (private reference; not exported).
- V520 scheduler stress boundary: `tool-call-reproducibility/experiments/dpr-terminal/V520-GRAND-SCHEDULER-STRESS-REPORT.md` (private reference; not exported).
- Runtime readiness: `qualification/runtime-ready-2026-09-02/README.md` (private reference; not exported).
- Archived scoped HMM resource lifecycle: `../../docs/archive/workorders/2026-09-08/workorder0905-hmm-resource-lifecycle.md` (private reference; not exported).
- NVFP4 C1 exact operation: `qualification/nvfp4-821/C1-RESULT.md` (private reference; not exported).
- Fixed E2E cases: `qualification/e2e-cases.json` (private reference; not exported).
- Persistent-state architecture: `../../docs/architecture/gemma4-persistent-state.md` (private reference; not exported).
- Three-platform retained report: `../../docs/reports/gemma4-three-platform-benchmark-2026-08-21.md` (private reference; not exported).
- Formal session model: `../../docs/formal/gemma4-persistent-session/` (private reference; not exported).
- Execution-continuity history: `../../docs/waveform/archive/workorder0831-execution-continuity-closure.md` (private reference; not exported).
- Archived cross-phase scheduler order: `../../docs/waveform/archive/workorder0903-cross-phase-resource-wavefront.md` (private reference; not exported).

## Shelved work

C2a weight-stationary prompt tiling was exact but slower and was removed. C3–C9 are shelved under `../../docs/backlog/` (private reference; not exported). Backlog text is not active authorization.

The archived pre-clean-slate model README is `../../docs/archive/status/GEMMA4-MODEL-README-2026-08-21-pre-timemachine.md` (private reference; not exported).
## Source ownership

Gemma-specific C implementation lives with the model package:

```text
models/gemma4-26b-a4b/gemma4.c
models/gemma4-26b-a4b/gemma4_text.c
models/gemma4-26b-a4b/gemma4_vision.c
models/gemma4-26b-a4b/server.c
```

The repository `src/` directory contains generic engine implementation. Generic
text chunking, layer order, release order, finish, validation, and publication
remain in `src/text_exec.c`; the Gemma files supply model arithmetic callbacks.
The generic HTTP entry remains `server/serve.py`; production serving logic never
belongs in `tools/`.
