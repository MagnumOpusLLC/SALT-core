# Salt documentation

This index separates current authority from scoped evidence, formal models,
closed work-order records, future backlog, and historical material.

## Start here

1. [`COGNITIVE-STATE-RISK.md`](COGNITIVE-STATE-RISK.md) — KV Cache and Persistent
   Agent State Statement: beyond-prompt state, rights, consent, risks, and the
   separation of computational continuity from authority to act.
2. [`ONBOARDING.md`](ONBOARDING.md) — latest-engine ownership, package, extension,
   and qualification onboarding.
3. [`../BREAKING-POINT.md`](../BREAKING-POINT.md) — current runtime/readiness and
   next-gate boundary.
4. [`ARCHITECTURE.md`](ARCHITECTURE.md) — authoritative storage, execution, and
   state ownership.
5. [`TERMINOLOGY.md`](TERMINOLOGY.md) — canonical NFQ N/F/Q, X-TARGET, and
   related engine vocabulary.
6. [`SOP.md`](SOP.md) — build, operate, qualify, measure, and publish.
7. [`../README.md`](../README.md) — repository entry point.

The current boundary is [`../BREAKING-POINT.md`](../BREAKING-POINT.md).
[PR #91](https://github.com/rhCat/salt/pull/91) merged the CUDA/Metal TARGET
resource-pool implementation; its
[`workorder/wo0908resourcepool.md`](workorder/wo0908resourcepool.md) is closed and
retained as an implementation/acceptance record, not active authorization.
[Work order 0909](workorder/wo0909serverFix.md) is the completed generic-serving,
model-owned native adapter, and `G4KVC006` shared-projection compression record;
it is not active implementation authorization.
[Work order 0910](workorder/wo0910-full-context-hmm-serving.md) records the
authenticated Acerbox CTX262144 four-engine pageable-HMM serving candidate,
including demand-paged KV, reset/spill, continuation, SSE, real-use behavior,
and the explicit arbitrary-prompt TARGET boundary.
The current server path accepts the qualified Gemma source-role readiness marker
during package authentication. Prior HMM and decode work orders remain archived under
[`archive/workorders/2026-09-08/`](archive/workorders/2026-09-08/). The latest
formal publication remains `8838dca` (PR #54). The general DPR engine is
**READY**; Qwen and Gemma are `runtime_ready=true`. Gemma's frozen default-runtime release matrix is
[`../models/gemma4-26b-a4b/qualification/runtime-ready-2026-09-02/README.md`](../models/gemma4-26b-a4b/qualification/runtime-ready-2026-09-02/README.md).
Model recipes and controls are authoritative in
[`../models/gemma4-26b-a4b/SOP.md`](../models/gemma4-26b-a4b/SOP.md).

Only the current authority documents and checked-in model/server configuration
define admitted operation. Reports are scoped evidence; formal proof, backlog,
research, completed work orders, and archive material do not override runtime
qualification.

## Latest merged engine evidence

- **PR #91 — CUDA/Metal TARGET resource-pool closure (`988cc887`, `2be1fd1`):**
  [`workorder/wo0908resourcepool.md`](workorder/wo0908resourcepool.md) records the
  closed focused gates and retained fixed HMM lineage. It is not an active work
  order; current status remains in `BREAKING-POINT.md`.
- **PR #52 — model adapters separated from engine core (`d4a7f21`):**
  [`reports/gemma4-detaildag-delineation-2026-08-25.md`](reports/gemma4-detaildag-delineation-2026-08-25.md)
  records the pure-core source boundary, Qwen canonical relocation, and final
  three-platform Gemma text/vision qualification.
- **PR #53 — generic text prefill control plane (`ea564711`):**
  [`formal/core-runtime/README.md`](formal/core-runtime/README.md) maps the
  engine-owned phase cursor, fail-closed release, validation, and publication
  boundary. The completed work order is archived below.
- **PR #54 — latest-engine formal refinement (`8838dca`):**
  [`formal/latest-engine-refinement.md`](formal/latest-engine-refinement.md)
  separates CI, five-model TLC results, authoritative engine-route TLAPS results,
  supplemental Qwen/Gemma refinements, and real inference evidence.

## Current implementation and work-order records

- [`workorder/wo0908resourcepool.md`](workorder/wo0908resourcepool.md)
  — closed PR #91 CUDA/Metal TARGET resource-pool implementation and acceptance
  record; not active authorization.
- [`workorder/wo0909serverFix.md`](workorder/wo0909serverFix.md)
  — completed documentation, source-ownership, generic-serving, CTX2048, and
  portable-state compression acceptance record; not active authorization.
- [`workorder/wo0910-full-context-hmm-serving.md`](workorder/wo0910-full-context-hmm-serving.md)
  — completed Acerbox full-context pageable-HMM serving candidate and exact
  memory/session/route boundary; not broader default promotion.
- [`archive/workorders/2026-09-08/`](archive/workorders/2026-09-08/)
  — historical HMM lifecycle, decode-lowering, and superseded benchmark orders.
- [`waveform/README.md`](waveform/README.md)
  — waveform execution, NFQ N-row tiling, X-TARGET causal rows, matrix-flow
  ownership, N/F/Q search,
  CPU/Metal/CUDA parallel dispatch, reports, and archived predecessor work.
- [`waveform/archive/workorder0903-cross-phase-resource-wavefront.md`](waveform/archive/workorder0903-cross-phase-resource-wavefront.md)
  — closed historical cross-phase scheduler order; exact/default-off results are
  retained, with no provider expansion carried forward.
The Aug 24 CUDA, Aug 26 KV/SoA, completed/superseded Aug 27 attention order,
and rejected Aug 27 routed-FFN order are archived under
[`archive/workorders/2026-08-27/`](archive/workorders/2026-08-27/).

### Current GPU-gate status

- PR #91 closed the focused TARGET position-collaboration gates: CUDA
  `988cc887` retained complete HMM identity, and Metal `2be1fd1` was exact at
  scheduled X16 (historically labeled N16) while improving the counterbalanced
  P774 gate by 14.122%.
  Metal ordinary B1 remains a separate performance-RED qualification path.
- The full-text pure-GPU HMM mechanism and repeat gate are fixed (`50f2f00`,
  merged-equivalent runtime/config `13fa8f6c`; 12/12 across CTX512/1024/2048/4096),
  with complete identity retained by `896ef6c` and `988cc887`. One authenticated
  Acerbox full-context four-engine serving/E2E candidate now also passes its
  memory, reset/spill, continuation, SSE, and real-use gates. HMM remains
  explicit/default-off; broader promotion and unmatched-prompt proposal coverage
  remain open rather than engine exactness.
- Canonical HMM `TARGET-N48` (the historical evidence label, where N and X were
  both 48) is exact at `30.975 ms/accepted token` and `3.719786496 GB` peak.
  Current N/F/Q is NFQ-only; X is the X-TARGET causal-row count, independent of
  optional DPR. The historical
  persistent B1 conversation is rejected as serving-route evidence; B1 is
  explicit-only and must not activate as a DPR-off, miss, or no-candidate
  fallback.
- Historical pre-split V520 HIP X2/X3/X8/X27 evidence (then labeled
  N2/N3/N8/N27) is exact with address digest `8f411c73fe52e753`.
  Fresh historical X27 is `216.998 ms/accepted`, 659 kernels, `1/1/0`, zero
  fallback, and remains default-off against the `<50 ms` gate.

The current one-pass cross-board result is
[`bench/gemma4-latest-target-cross-board-2026-09-02.md`](bench/gemma4-latest-target-cross-board-2026-09-02.md).
Use [`architecture/gemma4-gpu-native-prefill-gate.md`](architecture/gemma4-gpu-native-prefill-gate.md)
for the consolidated accelerator gate. GPU progress does not silently change
the ready default CPU mode or promote a mode-specific candidate.

## Current authority and sub-indexes

### Current formal correspondence

- [`formal/current-engine-696a06d4.md`](formal/current-engine-696a06d4.md) —
  source/transmutation pin, symbolic model bounds, six-model TLC result,
  partial current TLAPS status, contradiction audit, Rosarium limitation, and
  open candidate-zero server refinement mismatch.
- [`formal/README.md`](formal/README.md) — active formal model index and
  historical-proof boundary.

### Architecture and engine mechanisms

- [`architecture/README.md`](architecture/README.md) — architecture sub-index.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — canonical runtime architecture and source
  map.
- [`architecture/compute-theory.md`](architecture/compute-theory.md) —
  deterministic compute model.
- [`architecture/memory-architecture.md`](architecture/memory-architecture.md) —
  mindset/delta and factual-KV ownership.
- [`architecture/modular-matrix.md`](architecture/modular-matrix.md) — math ×
  quant × model composition.
- [`waveform/architecture/cache-first-nfq-matrix-search.md`](waveform/architecture/cache-first-nfq-matrix-search.md)
  — proposed cache-first exact-transition admission and CPU/GPU `N×F×Q`
  matrix-search execution; the current route-matrix foundation remains accepted
  while this extension is implemented and qualified.
- [`architecture/gemma4-persistent-state.md`](architecture/gemma4-persistent-state.md)
  — Gemma C-owned state/session architecture.
- [`mentor/README.md`](mentor/README.md) — current DPR/Nomogram authority and
  evidence plus clearly labeled historical Mentor material. Use this sub-index
  instead of treating the superseded Mentor C1–C5 sequence as current operation.

### Operations

- [`operations/README.md`](operations/README.md) — operations sub-index.
- [`SOP.md`](SOP.md) — repository-wide operating procedure.
- [`operations/memory-accounting.md`](operations/memory-accounting.md) — hard
  limit and resident/touched reporting.
- [`operations/gpu-offload.md`](operations/gpu-offload.md) — current Qwen
  CPU/Metal policy.
- [`operations/cache-modes.md`](operations/cache-modes.md) — expert
  transport/cache semantics.
- [`operations/degenerate-guard.md`](operations/degenerate-guard.md) — turn-reset
  and loop-failure behavior.
- [`../server/LATEST_WORKING.md`](../server/LATEST_WORKING.md) — Gemma API/operator contract.
- [`../models/gemma4-26b-a4b/SOP.md`](../models/gemma4-26b-a4b/SOP.md) —
  model-specific authority.

### Formal models

- [`formal/README.md`](formal/README.md) — current core, Qwen, Gemma, DPR, and
  KV/expert formal sub-index.
- [`formal/latest-engine-refinement.md`](formal/latest-engine-refinement.md) —
  source-bound latest-engine TLC/TLAPS result and proof boundary.

Formal invariants and runtime qualification are separate evidence classes.

### Reports

- [`reports/README.md`](reports/README.md) — all retained current reports,
  including the PR #52 detail-DAG/source-delineation report.
- [`bench/gemma4-latest-target-cross-board-2026-09-02.md`](bench/gemma4-latest-target-cross-board-2026-09-02.md)
  — frozen 2026-09-02 direct one-pass Gemma TARGET performance and oracle result;
  retained evidence, not the current status authority.
- [`../models/gemma4-26b-a4b/qualification/runtime-ready-2026-09-02/README.md`](../models/gemma4-26b-a4b/qualification/runtime-ready-2026-09-02/README.md)
  — frozen repeat-three Gemma state/text/vision runtime-readiness evidence.

Reports retain their named identity and scope; they are not automatic current
readiness claims.

### Reference and research

- [`reference/README.md`](reference/README.md) — model setup, live interface
  discovery, and quant-format references.
- [`../BENCHMARK.md`](../BENCHMARK.md) — frozen Qwen pure-CPU baseline.
- [`research/README.md`](research/README.md) — hypotheses and research evidence,
  not admitted operation.

## Archived execution-continuity work orders

- [`waveform/archive/workorder0831-execution-continuity-closure.md`](waveform/archive/workorder0831-execution-continuity-closure.md)
  — implemented/rejected 0831 execution-continuity evidence; retained only as
  historical predecessor material.
- [`waveform/archive/AUG27-ADAPTIVE-N-TILED-TARGET-EXECUTION-WORK-ORDER.md`](waveform/archive/AUG27-ADAPTIVE-N-TILED-TARGET-EXECUTION-WORK-ORDER.md)
  and [`waveform/archive/AUG28-PREFILL-NROW-MATRIX-FLOW-WORK-ORDER.md`](waveform/archive/AUG28-PREFILL-NROW-MATRIX-FLOW-WORK-ORDER.md)
  — archived candidate-source-neutral target and prefill/expert matrix evidence.
- [`waveform/archive/AUG28-HARDENED-SEQUENTIAL-DISPATCH-WORK-ORDER.md`](waveform/archive/AUG28-HARDENED-SEQUENTIAL-DISPATCH-WORK-ORDER.md)
  — completed sequential dispatch hardening through Step 15.

## Completed Aug 25 work orders

These completed instructions are historical context, not active authorization:

- [`archive/workorders/2026-08-25/AUG25-DETAILDAG-WORK-ORDER.md`](archive/workorders/2026-08-25/AUG25-DETAILDAG-WORK-ORDER.md)
- [`archive/workorders/2026-08-25/AUG25-TEXT-CONTROL-PLANE.md`](archive/workorders/2026-08-25/AUG25-TEXT-CONTROL-PLANE.md)

## Backlog and archive

- [`backlog/README.md`](backlog/README.md) — inactive ranked future work; a new
  explicit GO is required.
- [`archive/backlog/roadmap-2026-08-19.md`](archive/backlog/roadmap-2026-08-19.md),
  [`archive/backlog/optimization-architecture.md`](archive/backlog/optimization-architecture.md),
  and [`archive/backlog/legacy-plans/README.md`](archive/backlog/legacy-plans/README.md)
  — archived roadmap, optimization architecture, and legacy plan index.
- [`archive/README.md`](archive/README.md) — historical evidence and release
  bundles.
- [`archive/status/`](archive/status/) — pre-clean-slate authority snapshots.

Archived text preserves historical claims and may retain old relative paths. It
is never current authority.

## Documentation policy

A documentation-only change requires relative-link/anchor validation, a stale
current-claim sweep, whitespace/control-byte checks, and final Git scope
inspection. `CORE-FILES.sha256` covers its declared 56-file core surface only;
it is not repository-wide documentation integrity. Documentation-only work does
not run inference.
