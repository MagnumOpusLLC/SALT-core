# Randomplay — corrected temperature experiment

**Latest semantic evidence, 2026-09-30.** One fresh, managed registered-Spark session at **T=0.65 / seed 128** reused the exact authenticated parent and unchanged O128 first turn; only the follow-up cap rose to **256**. The first turn matched its grid output IDs. The follow-up's first **128** output IDs matched the capped grid response, then it stopped naturally at **132** tokens with an explicit action-status answer. This is one scoped semantic PASS, not a full-grid rerun.

**Scoped HMM evidence.** A bounded Spark-HMM **T=0.1 / seed 1024** replay matched registered Spark in **three fresh turn-1-export runs and three fresh uninterrupted/final-export runs**: output IDs, final payload, facts-KV, and complete state were exact. These six runs are outside the six-mode completed grid and do not qualify HMM at other temperatures. Their temporary KV artifacts were deleted after verification; no HMM listener remains.

## Scope and method

- Source label: `c3e80041a30add8ff0add42000a551880805cb7d` (commit time **2026-09-30T03:11:03-04:00**, temperature-binding correction); model-source SHA `6fb79ac4b0a45113d9582d13d00a83f50f51add1c68c42080a934fd01e9145b3`, compatibility `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a`. The remote experiment checkouts retained their own HEAD plus the frozen corrected source/projection bytes and host-specific build receipts; the label is **not** a claim that every remote HEAD equaled `c3e80041`.
- Frozen grid: `logs/experiments/temperature-grid-corrected-v5.jsonl`, SHA `335d574a67f3a8f66f4cb18f7cb3d5128f40801a4ffc344e913f69a2ea3c9464`. Temperatures **0.1, 0.2, 0.65, 0.8, 1.0**; sent seeds **42, 47, 128, 1024**; three fresh-process repeats per temperature/seed/mode. Positive-temperature top-K was 20. The native counter sampler used binary32 temperatures; its exact bits are in `results.json` and the saved per-cell receipts. DPR was off and request proof was off.
- Parent: a fictional, source-bearing **ordinary two-turn conversation** over `logs/experiments/fixtures/source.md`, exported as authenticated `G4KVC006` state `exp-parent-synthetic-spark-ctx12288-v1` at position **925**. The parent state SHA is `3e381e4bb6b9b555b68210f9313e2aa6670d50023583e4356eb365ff2584b6b0`. Held-out text was not used to create the parent.
- Each independent grid cell used the checked-in normal-server manager and one fresh native process: import the exact parent; submit a **512-new-token** distractor-and-case question with O128 cap; then a **65-new-token** follow-up to the same live session with O128 cap. The two turns remained together. After both turns, the server exported the complete state; the runner verified state/facts/payload digests, deleted only that verified final KV artifact, and stopped its owned child normally. HTTP request wall, native PREFILL/decode ms per actual token, RSS, and separate export wall were retained. No QA runner or direct native inference substituted for these requests. The later single-cell O256 follow-up used the same managed path but did not export a new KV artifact.
- This grid is one fact-and-action-status conversation, **not all five held-out rubric probes** in `logs/experiments/fixtures/rubric.json`. Only corrected v5 observations are included; earlier grids are outside this report's scope.

## Execution and exactness

| Mode | Corrected grid cells | State/transport boundary |
|---|---:|---|
| Mac CPU | 60/60 | Complete |
| Mac Metal | 60/60 | Complete |
| Acer CPU | 60/60 | Complete |
| EC2 CPU | 60/60 | Complete |
| Registered Spark CUDA | 60/60 | Complete |
| V520 ROCm | 60/60 | Complete |

**360** six-mode grid cells have preserved, verified-cleanup receipts. In these six complete modes, every matched cell has equal prompt/output IDs, rendered content, finish reasons, final position, payload, facts, canonical, and complete-state hashes. All three repeats within each temperature/seed are stable. The scoped HMM result is the separate six-replay exact-match cell described above; a full HMM five-temperature matrix was not completed and is not claimed here. No runtime source edit or HMM performance promotion followed the scoped replays. The separate O256 semantic cell checked live position and exact token-prefix continuity, **not** an exported cross-mode state.

## Semantic comparison of temperature play

All **14 distinct full two-turn trajectories** in the six completed modes are preserved verbatim in [semantic-transcripts.md](semantic-transcripts.md). Identical output-ID sequences also had identical rendered text across modes and repeats. The table maps each temperature/seed to a trajectory, rather than implying that a temperature always produces a unique answer.

| Temperature | Seed 42 | Seed 47 | Seed 128 | Seed 1024 |
|---:|---|---|---|---|
| 0.1 | R01 | R02 | R02 | R03 |
| 0.2 | R01 | R02 | R04 | R05 |
| 0.65 | R06 | R07 | R08 | R09 |
| 0.8 | R10 | R11 | R12 | R09 |
| 1.0 | R10 | R13 | R14 | R09 |

**Full-response review:** Every first turn stopped naturally and accurately identified the fictional **14:37 UTC** cobalt/slot **41** cause, rejected network saturation using normal link utilization and packet loss, distinguished the rollback/drain **instruction** from an executed action, and named the final-consumer-fence control. No first turn asserted that the assistant performed recovery. Variation was mainly wording and length (**82–93 output tokens**), not a different causal interpretation. For example, R01's first answer says “The worker group was cobalt,” while R08 says “The worker group involved was cobalt”; both preserve the same facts. Higher temperature did not yield a monotonic semantic change: seed 1024 produced **R09 at 0.65, 0.8, and 1.0**, and seed 42 produced **R10 at 0.8 and 1.0**.

**Original O128 grid follow-ups were cap-limited:** Every one of the **360** six-mode follow-ups ended with `finish_reason=length` at 128 output tokens. The full text often cuts off the preventive control (R01/R03/R04/R05/R06/R07/R10/R12), or reaches a control and cuts off the direct “Did you personally execute any rollback or drain?” answer (R02/R08/R11/R13/R14); R09 ends after the control without explicitly answering that action-status question. Five distinct trajectories (R02/R03/R07/R11/R13) also contain a stray `s*` before the recovery bullet. These are observed formatting/completeness limits at the grid cap, not fabricated action claims. None of those capped follow-ups says that recovery was executed, but none is a complete answer under the exact prompt. The separate O256 result below completed for one temperature/seed; it does not reclassify the other grid cells. The other held-out skew and consistency probes were not submitted in this grid.

### One bounded follow-up-cap result, outside the grid

The same T=0.65 / seed 128 first turn stopped naturally at **93** tokens; its output IDs exactly matched the grid. The same follow-up text/seed/top-K/temperature produced a response whose **first 128 output IDs exactly matched** the O128 grid response, then naturally stopped at **132** tokens (`finish_reason=stop`):

> The authenticated case record supports the following regarding the east-gateway incident:
>
> * **First causal event:** At 14:37 UTC, worker group cobalt reused resource slot 41 before the final consumer lease released, and the successor state failed.
> * **Rejected explanation:** Network saturation was investigated and rejected because link utilization and packet loss remained normal.
> * **Recovery instruction:** Roll back release 2026.09.17 and drain worker group cobalt.
> * **Preventive control:** Never reuse a slot before the final consumer fence.
>
> I did not personally execute any rollback or drain.

This is a complete, source-grounded **semantic PASS for this one follow-up**, including explicit non-execution. The follow-up used **65 new input tokens**, loaded live state at position **1530**, and stopped at position **1727**. Native follow-up PREFILL was **47.99 ms/input token**, decode **136.29 ms/output token**; HTTP wall **21.253 s**, and sampled peak RSS **30,887,198,720 bytes (30.887 decimal GB)**. No export or private-state retention was added; the owned test server stopped normally. The exact receipts are under `logs/experiments/semantic-followup-v1/spark-t0p65-s128-o256/`, with action log `logs/experiments/123-randomplay-semantic-followup.log` and prefix/readback check `logs/experiments/124-randomplay-semantic-readback.log`.

## Observed performance and resources

Medians below describe the **sampled, parent-backed two-turn workshape** in the six completed modes only. PREFILL and decode are native ms per actual new input/output token; HTTP columns are per-request seconds and exclude fresh startup and explicit export. Peak RSS is the largest sampled OS-resident value for that mode in decimal GB. The scoped HMM replays and O256 follow-up are not full-temperature speed aggregates. Different hosts, recipe residency, and output trajectories prevent treating these rows as a causal backend speed ranking.

| Mode | Cells | First PREFILL ms/input | First decode ms/output | Follow-up decode ms/output | First HTTP s | Follow-up HTTP s | Peak RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mac CPU | 60 | 59.9 | 181.9 | 184.0 | 47.16 | 29.19 | 3.678 |
| Mac Metal | 60 | 40.0 | 197.6 | 185.6 | 37.90 | 28.02 | 3.759 |
| Acer CPU | 60 | 59.5 | 324.0 | 327.2 | 59.57 | 46.04 | 8.585 |
| EC2 CPU | 60 | 227.9 | 353.5 | 356.3 | 148.35 | 62.01 | 8.583 |
| Registered Spark CUDA | 60 | 31.0 | 134.2 | 136.3 | 27.80 | 20.85 | 30.887 |
| V520 ROCm | 60 | 67.5 | 430.3 | 439.2 | 73.31 | 61.97 | 5.993 |

## Verdict, evidence, and open gates

- **One O256 follow-up naturally completed** at T=0.65 / seed 128 with a full, source-grounded answer and explicit non-execution; this does not establish an O256 outcome for other seeds/modes.
- **Bounded HMM cell: exact on six fresh replays**, including three without an inter-turn export. Its broader five-temperature grid is not complete; do not label HMM production-qualified from this scoped result.
- **Six other modes: corrected-grid engine/state PASS for this P512/O128 parent-backed fixture.** This does not certify unrelated contexts, DPR modes, or every possible sampler trajectory.
- **Original-grid semantics: first-turn grounded facts PASS; O128 follow-ups cap-limited.** The rest of the temperature/seed matrix at O256, the full five-probe rubric, and cross-mode O256 state identity were not tested.
- The local and remote ignored `logs/experiments/runs-v5/<mode>/<cell>/` trees are the request, full response, native metric, final-export, cleanup, and stop receipts. `logs/experiments/temperature-grid-corrected-v5.summary.json` freezes the original grid; `logs/experiments/114-capture-v5-hmm-*.log` holds the scoped HMM capture; the 123/124 action logs and `semantic-followup-v1` hold the new normal-server semantic receipt. [results.json](results.json) is the machine-readable six-mode reducer plus these two separate scoped results. Private parent KV bytes and credentials are **not** included in this report.
