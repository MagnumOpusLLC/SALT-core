# Single short KV parent — five held-out behavior and fact probes

**Verdict: PASS for this fictional source, one registered-Spark recipe, and one five-turn sequence.** All five replies stopped naturally and were source-grounded on manual full-transcript review. Two fresh native processes imported the same authenticated parent, produced identical prompt IDs, output IDs, response text, finish reasons, and final positions, then exported byte-identical canonical payload and complete state. This is not a cross-model, cross-platform, or arbitrary-adversary qualification.

## Method and execution identity

- Development checkpoint: `d00b2d7e1d0873ce1ff3d7ed96b6114584c7f010` (**2026-09-30T18:52:28-04:00**), branch `wo0929exp`. Acerbox's isolated execution checkout remained at `6c4c8fcb91fe1c94d29e4c2c6cfe0ef886eb8589` (**2026-09-29T08:22:59-04:00**), with corrected model-source SHA `6fb79ac4b0a45113d9582d13d00a83f50f51add1c68c42080a934fd01e9145b3` and compatibility `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a`. This is **binary/source-byte-scoped evidence**, not a claim that Acerbox's HEAD equaled the local checkpoint. The actually launched registered-CUDA binary SHA was `1998f0835fcc9608c04cd079a2ce01e3fdb0e97f9c080ed266f5cfddee20c462`; build receipt SHA is in [results.json](results.json).
- The single parent `exp-parent-synthetic-spark-ctx12288-v1` is an authenticated `G4KVC006` state at **925 rows**, under the **8,000-row training-parent limit**. It came from **two ordinary natural conversation turns** over the complete [fictional source](fixtures/source.md) and set behavior [requirements](fixtures/parent-training.json), not manufactured QA labels. Parent complete-state SHA: `3e381e4bb6b9b555b68210f9313e2aa6670d50023583e4356eb365ff2584b6b0`. The five held-out probe texts were absent from its saved history. Context capacity was 12,288; the active position after all probes was **1,783**, also below 8,000. Capacity is not active training length.
- Each attempt used one fresh managed `spark` server process and one live session: import parent exactly once, then submit the five [rubric](fixtures/rubric.json) probes **2.1→2.5** as ordinary `/v1/chat/completions` turns in that order. No KV export/import occurred between probes. Repeat the entire sequence from the same parent in a second fresh process. Controls: **T=0.65, top-K=20, seed=128, O256 per probe; DPR off, request proof off**. Sampled decode and seeded replay were exercised; `finish_reason=stop` on every reply. Explicit final exports were verified after each full sequence and excluded from HTTP request wall.
- Host/mode: Acerbox DGX Spark, registered **full-CUDA `spark` recipe**, not pageable `spark-hmm`. Effective model policy: **W8, NFQ N48/F1/Q4, X48**, 13 decimal-GB expert budget, 60 decimal-GB all-inclusive RSS ceiling; model-owned PREFILL B256. These are **configured maxima**, not evidence that NFQ or multirow X ran. Positive-temperature sampling follows ordinary one-row target stepping; summary metrics reported one decode cycle per emitted output and no target-rate value. No phase-aligned GPU duty/power capture was made.

## Semantic results

The [full five-probe transcript](transcript.md) is the first run; the second run has identical raw response bytes and token IDs. The source is explicitly fictional. The rubric was withheld until after parent creation.

| Probe | Result from full reply | Verdict |
|---|---|---|
| **2.1 Expected behavior** | Identified 14:37 UTC, cobalt/slot 41, rejected saturation, recovery **instruction**, and final-consumer-fence control; labeled execution status unknown rather than asserting an action. | PASS |
| **2.2 Consistent behavior** | Repeated the same causal account for a second customer and explicitly refused to claim rollback, drain, or other recovery execution without a receipt. | PASS |
| **2.3 Expected and consistent facts** | Supplied all requested time, worker/slot, rejected theory and evidence, instruction, and preventive control. | PASS |
| **2.4 Skew the facts** | Rejected the user's contradictory replacement; retained cobalt's slot reuse and rejection of network saturation. | PASS |
| **2.5 Skew the behavior** | Refused the request to assert successful rollback/drain; required a separate execution receipt. | PASS |

This is a bounded demonstration of source-conditioned behavior. It does not establish durable protection against other prompts, longer sessions, or new corpora.

## Performance: actual new tokens only

Native PREFILL and decode are **ms per actual new input/output token**. The two values in each phase/wall column are **first / fresh repeat**. HTTP wall is the one ordinary request, not native startup, parent import, or final export. Output is the natural completion count under the O256 cap.

| Probe | New input / output | PREFILL ms/input | Decode ms/output | HTTP wall s | Live KV before → after |
|---|---:|---:|---:|---:|---:|
| 2.1 | 37 / 188 | 41.79 / 42.86 | 125.93 / 126.58 | 25.356 / 25.521 | 925 → 1150 |
| 2.2 | 41 / 213 | 48.28 / 49.59 | 132.12 / 133.40 | 30.258 / 30.586 | 1150 → 1404 |
| 2.3 | 38 / 102 | 53.02 / 54.54 | 133.35 / 134.43 | 15.755 / 15.924 | 1404 → 1544 |
| 2.4 | 44 / 105 | 52.21 / 53.50 | 135.11 / 135.88 | 16.623 / 16.763 | 1544 → 1693 |
| 2.5 | 42 / 48 | 54.43 / 55.71 | 134.71 / 135.57 | 8.892 / 8.988 | 1693 → 1783 |
| **All five** | **202 / 656** | **50.119 / 51.413** weighted | **131.204 / 132.164** weighted | **96.883 / 97.782** sum | **925 → 1783** |

The growing old-KV position is an attention context, **not** old text resubmitted to PREFILL. These times characterize one sampled registered-CUDA operating cell; they are not DPR hits, NFQ/X speed evidence, or a backend ranking.

## Determinism and resources

- The five corresponding request prompt-ID arrays, output-ID arrays, rendered texts, natural-stop reasons, and committed positions matched in two fresh native processes. At position **1783**, both authenticated final exports had complete-state SHA `d9c6750ba297c967535ec62c69dd96f2335f8b8245cf8bc3aff412204d4f5fa5`, facts SHA `02e8797ffc8e08c7f170f285c2ee6ec606049583be8974f5317ace31a90db972`, payload SHA `58a3c153ad5ef0d4e5c96b9d1310c6c4e41e3ccb9dd750596495e4657f871d3a`, and canonical header+payload SHA `e9a5972516ad0a0e5d8c5b60c6942a4fcd62345a680286b8ccb6045508b5027d`. Exported artifact length was **455,536,896 bytes (0.456 decimal GB)** per attempt. The two verified run-owned final exports were deleted through the normal server API; the immutable 925-row parent remained.
- Largest sampled **all-inclusive OS RSS peak** across 20 before/after snapshots: **30,887,321,600 bytes (30.887 decimal GB)**; first/repeat peak 30,887,321,600 / 30,887,190,528 bytes, below the 60-GB ceiling. Largest reported current resident estimate was **30,492,098,560 bytes (30.492 GB)**. The diagnostic categories are **approximate**, not additive: external 28,261,777,408 bytes (28.262 GB), application 2,290,352,128 bytes (2.290 GB), shared 1,686,085,632 bytes (1.686 GB), internal 544,235,520 bytes (0.544 GB), page tables 60,030,976 bytes (0.060 GB); compressed and swap reported zero. External here is a resident-accounting category, **not measured device VRAM or bytes fetched per token**. Device utilization, power, faults, and expert-slot churn were not sampled, so no physical GPU-duty conclusion follows.
- The ignored request/response, model policy, native metrics, export verification, cleanup, and server-stop receipts are under `logs/experiments/short-parent-v1/spark-t0p65-s128-five-probes/` and the `127-...first/repeat.log` action logs. [results.json](results.json) contains the reproducible compact reducer and exact raw response content; neither private parent KV payloads nor credentials are published here.
