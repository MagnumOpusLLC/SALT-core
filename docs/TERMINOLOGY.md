# Salt terminology

This file is the canonical vocabulary for active Salt architecture, runtime,
configuration, tests, and reports. Source-field names and historical documents
do not redefine these terms. Dated evidence keeps its original scope, but new
work must use the definitions below.

## NFQ

NFQ is an engine-core execution policy shared by CPU, Metal, CUDA, and HIP. Its
semantic meaning is identical on every backend; only the physical realization
changes.

| Symbol | Canonical meaning |
|---|---|
| `N` | **NFQ tile number.** The number of base NFQ work tiles admitted in each F lane. N participates directly in the NFQ candidate/check count; it is not causal TARGET row width. |
| `F` | **Fanout lanes / latent universes.** The number of concurrent isolated tentative universes. Each lane has its own parent state, logits, ancestry, and tentative destination until TARGET resolves it. |
| `Q` | **Search attention span / worker lifetime.** The bounded alternatives each `(n,f)` tile may explore before resolution, retirement, or reassignment. |
| `W` | **Physical worker capacity.** Persistent CPU workers or backend execution lanes available to realize admitted NFQ work. `W` is a resource quantity, not part of NFQ semantics. |

For lane `f`, the engine defines:

```text
base tile            = (n,f)
candidate/check      = (n,f,q)
base worker surface  = N x F = n x W
candidate_count      = N x F x Q
```

For example, `N30/F1/Q4` means 30 admitted NFQ tiles in one F lane and four Q
alternatives for every tile: 120 candidate checks. It never means four total
candidates. W freelancers consume the N×F base surface; Q extends each tile's
bounded lifetime. None of N, F, or Q specifies how many causal rows one TARGET
attempt executes.

At or above the live-KV warmup threshold, when retained/DPR proposal state is
unavailable, one ordinary authoritative B1 bootstrap establishes the first
candidate distribution:

```text
L(S) -> [A, B, C, D]
T(S,A) -> SA + L(SA)
T(S,B) -> SB + L(SB)
```

`SA` and `SB` are distinct tentative universes. Their successor proposals are
cheap because TARGET has already produced `L(SA)` and `L(SB)`; deriving later Q
alternatives uses those retained logits/projections rather than another proposal
rollout. Below 512 pre-request live-KV rows, the separate runtime admission gate
forces X1 and ordinary B1 may repeat after a proposal miss or with DPR off;
authenticated exact reuse and one-row DPR verification remain valid.
Current-prompt PREFILL does not count toward the threshold.

`F2` means two such isolated universes. A backend may batch two ready lane
transitions as physical B=2, but B=2 does not define F2 and duplicate rows do not
create distinct universes. Completion order never chooses a lane, alternative,
token, or commit.

`W` workers are freelancers. The orchestrator owns the shared dependency-ready
work pool and dispatches one bounded `(n,f,q)` claim to any free worker. Worker
identity never owns a lane, Q runway, tile, expert, output range, or winner.

### Backend invariance

CPU and GPU use the same NFQ terminology and core policy:

- CPU may realize tiles with persistent workers and output/expert-row tiling.
- GPU may realize the same tiles as batched matrices and backend command graphs.
- Physical occupancy, workgroup shape, row tiling, command encoding, and fences
  may differ without changing N, F, Q, ancestry, target selection, or commit.

Do not redefine NFQ by backend. In particular:

- `N` does not become token depth, proposal length, or accepted-token count.
- `F` does not become alternative rank, duplicate batch rows, or worker count.
- `Q` does not become tokenless queue capacity, generic scheduler depth,
  accepted-token count, or speculative commit length.

### Inference role

NFQ is the normal engine-core decode mechanism on CPU and GPU. DPR is an
optional exact-reuse or candidate source; it does not enable NFQ and does not
change N/F/Q semantics.

At an authoritative token boundary:

1. an authenticated exact transition hit may be reused directly;
2. otherwise Q alternatives come from authoritative lane logits, retained target
   projections, DPR/ND, or the bounded initial bootstrap fallback;
3. the engine admits those candidates into NFQ;
4. TARGET executes tentative work and selects the exact accepted path;
5. only target-authorized state/KV is committed.

Below 512 pre-request live-KV rows, effective X is one: no-DPR or proposal-miss
decode uses ordinary B1, while authenticated exact reuse and one-row DPR
verification remain valid. At or above 512 rows, an ordinary transition may
bootstrap the first candidate state when DPR or a retained projection is
unavailable; after that bootstrap, non-exact-hit boundaries return to the
bounded X4 cascade.

## X-TARGET

`X` is the configured causal row width of one TARGET attempt. It is independent
of the NFQ N/F/Q checking surface: N/F/Q governs proposal/search work, while X
governs how many causally ordered candidate-token rows TARGET may verify in one
attempt.

For normal serving:

```text
request_x = 1                                      when position_before < 512
request_x = min(configured_x, warm_x=4)            when position_before >= 512
active_x  = min(request_x, output_remaining)
```

`position_before` is the live-KV position before the current prompt, so prompt
PREFILL cannot unlock wider X. `active_x` may shorten the final output-tail
attempt without changing configured N/F/Q or configured X. One X-TARGET attempt
ends at its first mismatch. TARGET commits the accepted prefix and returns the
exact correction/pending boundary; the rejected remainder is not re-admitted as
part of the same attempt.

Inside one attempt the proposed rows are computed as one causal batch and the
comparisons form a validity vector `V[0..X-1]`; the commit boundary is its first
false entry `j`. `V[0]` uses the existing parent logits; `V[i]` uses the target
logits produced after the proposed prefix through `i-1`. Locally matching entries
after `j` cannot commit. The target-computed state and logits at `j-1` are
retained as the next episode's parent; they are never recomputed.

## Projection cascade (proposal sources)

An **episode** is one boundary's proposal → X-TARGET → commit sequence. The root
is the parent's greedy token (zero model work). Proposals for the remaining rows
come from exactly one of three sources, in this precedence:

| Source | Meaning | Cost |
|---|---|---|
| Cache route | DPR trajectory when enabled and applicable; otherwise the **parent history match**: the longest, most recent earlier occurrence of the committed suffix ending in the root, whose followers are proposed (prompt/session tokens, then this request's outputs; match length ≤ 16). | zero model work |
| Local memory | Retained rows of the previous X-TARGET block past its accepted prefix; one argmax per already-computed speculative row. Exists only after a rejection; a full hit leaves only the bonus root. | zero model work |
| True cold | Root only: one ordinary transition (the bootstrap). Same cost as B1; never the loop. | one B1 |

A proposal is token IDs only and never state. Forbidden proposers: a sequential
`TK[i-1] -> TK[i]` chain through the model layers (that is B1), vocabulary ranks
of one parent distribution used as successive tokens, N−1 serial B1 steps, and
any use of retained logits to commit a row the target has not computed.

Cost rule: `cost per accepted token = (toll + per_row × X) / accepted`. X is a
cap; the useful depth is the proposal's expected run length. Large X pays only
for strong sources.

## Related terms

| Term | Meaning |
|---|---|
| `B` | Row width admitted for a compute stage, such as a PREFILL chunk. `B` is separate from N/F/Q. |
| `X` | Configured causal X-TARGET row ceiling. Normal serving applies the pre-request live-KV gate and output-tail clamp before an attempt. X is independent of N/F/Q. |
| Candidate | One deterministic check identified by `(n,f,q)`. It has no commit authority. |
| Candidate-work cell | One `(n,f,q)` unit in the configured `N x F x Q` checking surface. |
| Tile | One bounded NFQ work item counted by `N`; never token depth or causal TARGET row width. |
| Lane | One isolated tentative latent universe counted by `F`, with its own state, logits, ancestry, and canonical seat. |
| Alternative | One deterministic possibility selected from a lane's already-computed logits and indexed by `Q`. |
| Search attention span | The bounded Q alternatives a lane may explore before target resolution, retirement, or reassignment. This is unrelated to the model's attention operator. |
| X-TARGET | One causal TARGET attempt of up to X rows. It stops at the first mismatch and alone authorizes accepted-prefix, correction, pending-token, KV/state, and commit outcomes. |
| TARGET | Canonical model execution and authority used by X-TARGET verification. |
| Episode | One boundary's proposal → X-TARGET → commit sequence. |
| Parent-exact hit | A DPR reuse of already-computed KV/logits at the same parent: the only tier that skips model compute. Distinct from a proposal hit, which only supplies token IDs. |
| Toll | Per-submission cost of a TARGET block that does not scale with rows (fixed per block). Measured, backend-specific; a realization property, never arithmetic. |
| DPR | Optional authenticated exact-transition/state reuse and proposal source. DPR does not infer, enable NFQ, or commit target state. |
| Ordinary bootstrap | At or above the warmup threshold, the bounded initial transition used when candidate state is unavailable. Below the threshold, no-DPR or proposal-miss X1 uses ordinary B1; exact reuse and one-row DPR verification remain eligible. |
| Commit | Publication of the exact target-authorized transition into canonical live session state/KV. Completion order never determines it. |

## Gemma execution and recipe flags

These flags are model-owned startup policy. They select geometry, resource
envelopes, or a physical realization of the same portable-C semantic program;
they do not transfer token selection, KV/state, TARGET, or COMMIT authority to a
wrapper or backend. A recipe value is qualified only for its named model,
artifact identity, host class, and workshape. Unset values inherit the model's
base `engine.config`; they are not synonyms for zero.

The active Gemma recipes are:

| Recipe | Intended physical realization |
|---|---|
| `mac` | Apple Silicon portable-CPU operation. |
| `mac-metal` | Bounded full-Metal qualification and TARGET execution. |
| `ec2` | Portable x86-64/AArch64 CPU operation, including the DGX Spark CPU control. |
| `spark` | DGX Spark registered full-CUDA operation. |
| `spark-hmm` | DGX Spark bounded pageable-HMM CUDA operation. |
| `spark-hmm-vision` | DGX Spark CPU-text operation with pageable-HMM CUDA vision. |
| `rocm` | Radeon Pro V520 bounded-window, device-local-trunk HIP operation. |

### Recipe and compute selection

| Flag | Canonical meaning and use |
|---|---|
| `SALT_GEMMA_PLATFORM_RECIPE` | Names the selected platform policy. It chooses defaults; it is not a backend implementation or state-compatibility identity. |
| `SALT_GEMMA_COMPUTE_NODE` | Selects physical compute intent: `cpu` requires CPU realization; `full` requires the admitted complete GPU realization and fails closed rather than silently substituting CPU model work. |
| `SALT_FULL_MIN_RAM_GB` | Minimum host-memory admission prerequisite for the recipe. It is not an allocation request or RSS measurement. |
| `SALT_GEMMA_MEMORY_LIMIT_GB` | All-inclusive decimal-GB current-RSS ceiling, including resident file-backed pages. Footprint, compressed charge, and device telemetry remain separate diagnostics. |

### Server stderr and measurement policy

`SALT_SERVER_STDERR` controls native stderr visibility for every persistent Gemma
engine child without changing inference arithmetic or selective in-memory
diagnostic retention:

| Value | Meaning |
|---|---|
| `summary` | Default. Print one bounded `[gemma4-server-ms-tk]` record per completed request with PREFILL ms/input, decode ms/output, TARGET/proposal denominators, live-KV reuse, DPR parent/Nomogram/PREFILL hits, misses, cycles, and request wall. Do not stream raw native meters. |
| `diagnostic` | Print the bounded operational native diagnostic subset in addition to the always-on summary. |
| `waterfall` | Enable the complete native meter set (`SALT_WATERFALL`, TARGET profiling, GPU diagnostics, HIP cell/synchronization timing and path summary, and CPU FFN detail), print all native stderr, and emit the complete endpoint waterfall in addition to the summary. Use only for bounded diagnosis because event timing, meter work, and output volume perturb operation. |

Printing and durable log storage are separate service concerns. The server writes
the selected stream to stderr; the operator or service supervisor decides whether
and how to retain, rotate, or discard it. Recipe files must not enable
`SALT_WATERFALL` directly: only explicit `SALT_SERVER_STDERR=waterfall` admits the
full meter.

### PREFILL policy

PREFILL consumes the prompt into the live session KV and produces the parent
logits used by the immediately following scheduler/decode boundary. Separate
phase timers do not imply separate model instances or state.

| Flag | Canonical meaning and use |
|---|---|
| `SALT_PREFILL_CHUNK` | Enables chunked/layer-major PREFILL. It does not define the chunk width. |
| `SALT_PREFILL_B` | Model-owned logical PREFILL row cap. It is independent of NFQ N/F/Q and X-TARGET. |
| `SALT_GPU_PREFILL_B` | Backend physical PREFILL batch cap. It may differ from logical `SALT_PREFILL_B` without changing prompt order or KV. |
| `SALT_GPU_B_QKV` | Backend physical row cap for PREFILL Q/K/V projections. |
| `SALT_GPU_B_Z` | Backend physical row cap for the configured intermediate/Z projection stage. |
| `SALT_GPU_B_O` | Backend physical row cap for the attention output projection. |
| `SALT_PREFILL_OPERATION_FLOW` | Selects the retained PREFILL operation-flow control. `0` keeps the qualified direct phase realization; this flag does not create a second scheduler. |
| `SALT_PREFILL_RETAIN_LAYERS` | Resource lifetime policy. `1` retains completed operation resources across layers where admitted; `0` releases per layer for bounded-window/pageable operation. It does not retain or discard semantic KV. |
| `SALT_PREFILL_DECODE_LOOKAHEAD` | Allows engine-owned readiness overlap from the PREFILL tail toward DECODE. It never changes sequential state semantics and is incompatible with fine-token ownership of the handoff. |
| `SALT_PREFILL_SHARED_ARENAS` | Reuses the startup-owned compatible phase arenas. It does not merge canonical state or authorize hot-path allocation. |
| `SALT_PREFILL_GPU_ATTENTION` | Selects GPU physical realization of PREFILL attention. It does not move attention authority or KV geometry into the backend. |
| `SALT_M2_BATCH` | Enables the existing batched M2/matrix projection path for eligible rows. It does not define PREFILL B, NFQ N, X-TARGET width, or serving concurrency. |
| `SALT_QKV_BATCH` | Enables the existing batched Q/K/V projection path for eligible rows. It does not change operation order or KV layout. |
| `SALT_ATTN_THREADS` | CPU worker width available to the attention body. It is a physical resource cap, not N, F, Q, or serving concurrency. |

### Scheduler, bootstrap, proposal, and TARGET policy

Normal production order is:

```text
PREFILL live KV/logits
    -> exact DPR parent reuse, when available
    -> DPR/ND proposal, when eligible
    -> retained TARGET projection, when sufficient causal rows remain
    -> one authoritative native bootstrap when no proposal state exists
    -> TARGET verification
    -> commit and retain the resulting projection/pending boundary
```

QA may replace only the proposal source with an authenticated fixed candidate
trajectory. It still uses the same live PREFILL state and TARGET verifier.

| Flag | Canonical meaning and use |
|---|---|
| `SALT_TARGET_ROUTE_N` | NFQ tile count N. It is NFQ policy only: not causal TARGET row width, accepted-token count, or a proposal generator. |
| `SALT_TARGET_ROUTE_F` | NFQ isolated-lane count F. It is not worker count or duplicate batch rows. |
| `SALT_TARGET_ROUTE_Q` | NFQ bounded alternative/worker-lifetime span Q per `(n,f)` tile. It is not queue capacity or speculative commit length. |
| `SALT_TARGET_X` | Configured causal X-TARGET row ceiling. The normal-serving live-KV gate and output tail may reduce it; it does not redefine N/F/Q. |
| `SALT_TARGET_KV_WARMUP_ROWS` | Pre-request live-KV threshold below which normal serving forces X1. Current-prompt PREFILL rows are excluded. Current Gemma recipes use 512. |
| `SALT_TARGET_WARM_X` | Maximum normal-serving X after the warmup threshold. It is still clamped by `SALT_TARGET_X` and output remaining. Current Gemma recipes use 4. |
| `SALT_DPR_DRAFT_N` | Maximum DPR/ND token horizon offered to TARGET. DPR remains optional; this value does not enable DPR or redefine NFQ N or X-TARGET width. |
| `SALT_TARGET_AREA_WORKERS` | Persistent physical worker budget for target area scans and CPU realization. Worker identity never owns a candidate, lane, or winner. |
| `SALT_TARGET_GPU_PROGRAM` | Makes the complete backend TARGET program available. It does not enable DPR, manufacture candidates, or force normal generation into TARGET. |
| `SALT_TARGET_CPU_GRAPH` | Selects the retained CPU graph realization for admitted TARGET/fine-token work. `0` retains the qualified graph-off interpreter; the graph is a physical control, not a scheduler. |
| `SALT_TARGET_N_PARALLEL` | Legacy-named control selecting the optional physical X-row parallel realization. It never changes configured X, causal ancestry, target selection, or commit. |
| `SALT_TARGET_MATRIX_FLOW` | Selects the optional matrix-flow realization for TARGET projections. It changes dispatch shape only. |
| `SALT_TARGET_GPU_EXPERTS` | Selects the compatibility GPU-expert TARGET path where explicitly qualified. It is separate from the complete GPU TARGET program and must not silently introduce CPU fallback. |
| `SALT_TEXT_FINE_TOKEN` | Selects the complete compiled 689-cell B=1 executor for one scheduler-selected known token. It is a physical bootstrap/correction control, not a proposer, N/F/Q switch, DPR switch, or generic continuous mode. It remains off where measured slower; `spark-hmm` uses it for complete pageable-GPU one-token execution and `rocm` uses it to keep the bounded cold bootstrap/correction inside the complete HIP program before returning to retained-projection/NFQ TARGET. |
| `SALT_TOKEN_CPU_FLOW` | Selects the retained CPU ordinary-token flow control. It affects physical realization only. |
| `SALT_TOKEN_CPU_GRAPH` | Selects the retained CPU ordinary-token graph control. It is distinct from `SALT_TARGET_CPU_GRAPH`. |
| `SALT_EXPERT_MATRIX_WAVES` | Selects the optional persistent-worker expert matrix-wave realization. It does not change routing, expert arithmetic order, or the layer fence. |

Native telemetry preserves the split. `GEMMA4_SERVER_POLICY` reports configured
`nfq_n`, `nfq_f`, `nfq_q`, and `target_x`; `GEMMA4_SERVER_X_TARGET` reports the
attempt's `active_x` and execution/acceptance counters. Neither surface permits
deriving X from N.

`SALT_TEXT_FINE_TOKEN=1` sends a known token through
`salt_text_execute_all()` with one complete submission/fence and canonical KV
publication. It is not universally enabled because B=1 cannot amortize the
complete program: CUDA measured `190.892 ms/token` versus approximately
`65-69 ms/token` for the accepted path, and the attempted CPU default was
rejected as a regression. A shared semantic scheduler does not require one
universally fastest physical B1 implementation.

### Weight addressability, trunk, and expert residency

| Flag | Canonical meaning and use |
|---|---|
| `SALT_GPU_TRUNK` | Enables GPU realization of the fixed dense/non-routed trunk for the engine lifecycle. It does not preload the routed expert pool. |
| `SALT_GPU_TRUNK_B` | Physical eligibility threshold for trunk GPU execution. `1` admits B1; a larger value keeps narrow work on the qualified non-GPU path. It is not PREFILL B or X. |
| `SALT_GPU_BOUNDED_WEIGHTS` | Requires bounded learned-weight addressability/residency instead of assuming the entire source/expert pool is GPU-resident. |
| `SALT_GPU_WEIGHT_ADDRESSABILITY` | Selects the physical source-address contract, including `pageable` or `bounded-window`. It does not change tensor identity or authorize copied/repacked weights. |
| `SALT_GPU_EXPERT_VIEW` | Selects expert view policy. `selected` exposes only router-selected experts through canonical slots/leases. |
| `SALT_GPU_TRUNK_LAYER_VIEW` | Allows or forbids per-layer trunk views. Current bounded recipes use `0` to avoid per-layer trunk authority. |
| `SALT_GPU_TRUNK_SHARED_POOL` | Selects whether exact trunk views refer to one process-lifetime shared device pool. It is physical storage policy, not a second tensor map. |
| `SALT_GPU_TRUNK_EXACT_VIEWS` | Requires exact canonical component views into the retained trunk realization. It does not permit repacking. |
| `SALT_GPU_RESIDENCY` | Selects an explicitly declared GPU residency realization. `device-local` permits bounded one-time permanent payload population; undeclared or hot-path copies remain failures. |
| `SALT_GPU_RESIDENCY_EXPERT_GB` | Fixed device-local expert-residency arena capacity for the residency backend. It is a physical arena bound, separate from retention policy. |
| `SALT_GPU_RESIDENCY_STAGING_MB` | Size of each preallocated backend staging slot. It does not authorize unbounded staging or payload duplication. |
| `SALT_GPU_RESIDENCY_STAGING_SLOTS` | Count of fixed startup-owned staging slots. No hot-path slot growth is allowed. |
| `SALT_EXPERT_BUDGET_GB` | Demand-selected expert retention budget. It is an upper bound on retained hot experts, not forced RSS, preload, or the total model size. |
| `SALT_EXPERT_PRELOAD` | Enables the recipe's explicitly qualified expert preload policy. `0` requires demand population; it does not disable expert execution. |
| `SALT_EXPERT_PRELOAD_LAYERS` | Optional model-owned layer selector for full expert-layer preload. `none` means no layer-specific full-hot selection. It is distinct from demand-selected retention. |
| `SALT_FETCH_TOUCH_BYTES` | Bounded source-touch extent used by the qualified fetch/proof contract. It is not copied-weight size; ordinary consumers reuse canonical resident-slot addresses. |

### KV and attention physical controls

| Flag | Canonical meaning and use |
|---|---|
| `SALT_GEMMA_GPU_KV_RING` | Selects Gemma's device-live hybrid KV geometry: sliding layers use their bounded ring and full-attention layers remain absolute. It does not alter canonical state meaning. |
| `SALT_DECODE_GPU_ATTENTION` | Keeps ordinary decode attention on the selected GPU realization and consumes the same device-live KV published by PREFILL. It is physical continuity, not a second KV authority. |
| `SALT_CUDA_PAGEABLE_MMAP` | Selects CUDA pageable/HMM access to canonical mapped payloads. It does not mean CPU staging or H2D copies. |

### Metal physical controls

| Flag | Canonical meaning and use |
|---|---|
| `SALT_METAL_LIBRARY` | Path to the authenticated prebuilt Metal library consumed by the backend. Recipe selection does not authorize rebuilding it. |
| `SALT_METAL_Q4_SIMDGROUPS` | Selects the optional Metal Q4 SIMD-group realization. `0` keeps it disabled. Arithmetic/state contracts remain unchanged. |
| `SALT_METAL_Q4_MAX_DESCRIPTORS` | Recipe cap for optional Metal Q4 descriptor use. `0` imposes no recipe-enabled descriptor path. |
| `SALT_METAL_Q4_WEIGHT_STATIONARY_MIN_B` | Minimum physical row width for the Metal weight-stationary Q4 path. It is a crossover, not semantic B, NFQ N, or X. |
| `SALT_GEMMA_METAL_EXACT_CELLS` | Synchronous T4 exact-cell proof harness. It must be `0` for serving, recurring qualification, and performance runs; `1` is allowed only for one bounded setup proof. |

### HIP physical controls

These flags are AMD-private workgroup/kernel-selection policy. They may change
occupancy and wall time only; they never redefine model geometry, routing,
arithmetic order, state, KV, winner selection, or COMMIT.

| Flag | Canonical meaning and use |
|---|---|
| `SALT_HIP_TEXT_SELECTED_WAVE` | Enables the qualified HIP selected-expert wave realization. |
| `SALT_HIP_TEXT_NROW` | Enables the qualified HIP N-row kernel realization. |
| `SALT_HIP_TEXT_PAIR` | Enables the qualified paired-row HIP realization where geometry admits it. |
| `SALT_HIP_TEXT_SYSTEM_FENCE` | Selects the optional HIP system-scope fence. `0` keeps the qualified narrower completion contract. |
| `SALT_HIP_TEXT_WAVE_GROUPS` | Physical HIP wave-group count for the text kernels. |
| `SALT_HIP_TEXT_BF16_WAVE_GROUPS` | Physical HIP wave-group count for BF16 text operations. |
| `SALT_HIP_TEXT_NROW_MIN_B` | Minimum physical width before HIP selects its legacy-named N-row realization. It is not NFQ N or configured X. |

### Current cross-recipe geometry

This table records current checked-in policy, not a universal optimum:

| Recipe | Compute | PREFILL B | NFQ N/F/Q | X | Fine token | TARGET GPU | Expert budget | Memory ceiling |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `mac` | CPU | 512 | N64/F1/Q4 | X64 | 0 | 0 | 1 GB | 4.5 GB |
| `mac-metal` | Metal | 512 | N64/F1/Q4 | X64 | 0 | 1 | 1 GB | 4.5 GB |
| `ec2` | CPU | 512 | N32/F1/Q4 | X32 | 0 | 0 | 6 GB | 14 GB |
| `spark` | registered CUDA | 256 | N48/F1/Q4 | X48 | 0 | 1 | 13 GB, preload | 60 GB |
| `spark-hmm` | pageable CUDA | 256 | N48/F1/Q4 | X48 | 1 | 1 | 2 GB | 16 GB |
| `spark-hmm-vision` | CPU text + pageable-HMM CUDA vision | 512 | N32/F1/Q4 | X32 | 0 | 0 | 1 GB | 7 GB |
| `rocm` | HIP | 128 | N104/F1/Q4 | X16 | 1 | 1 | 3 GB | 7.5 GB |

Do not transfer a measured B, N/F/Q, X, residency mode, or memory envelope to a
different model or host merely because the portable semantic engine is shared.
