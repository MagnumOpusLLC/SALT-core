# Exp 3 extension — balanced answer positions and no-parent control

**Latest verdict:** **40/40 correct, strict one-letter natural stops** across five fictional questions, four cyclic option placements, and two session conditions. The parent-backed arm scored **20/20**; the fresh **no-parent** arm also scored **20/20**. For this answer-bearing fixture, **the experiment found no accuracy dependence on the 925-row KV parent**. This is a useful negative control: the supplied options contain enough information for the model to select the intended answer without the source conversation.

## Frozen scope

- The original [Exp 3 fixture](questions.json) supplies four complete candidate responses per question, exactly one designated best choice. For each of its five questions, rotate the same four option texts through labels A–D exactly once. This balances the expected answer: **five A, five B, five C, and five D selections per condition**. The stem, candidate content, sampler, model binary, and request cap remain fixed. Only the option-to-letter mapping and parent condition change. This is systematic position rotation, **not** new distractors or paraphrases.
- Conditions: (1) import the same authenticated fictional parent `exp-parent-synthetic-spark-ctx12288-v1` at position **925**, then five sequential requests in one live session; (2) fresh native process/session starting at **position 0**, no parent import, then the same rotated five-question sequence. Eight arms total. The previously verified unrotated parent run is reused as one arm (**5 old requests**); seven newly executed arms contribute **35 new requests**. The requested matrix therefore contains **40 distinct condition/question/position cells**, not 40 fresh HTTP submissions in this extension.
- Host is Acerbox DGX Spark, registered `spark` binary SHA `1998f0835fcc9608c04cd079a2ce01e3fdb0e97f9c080ed266f5cfddee20c462`, model-source SHA `6fb79ac4b0a45113d9582d13d00a83f50f51add1c68c42080a934fd01e9145b3`. T=0.65, top-K=20, seed=128, O16, DPR off and request proof off throughout. W8, NFQ N48/F1/Q4, X48 remain **configured** recipe values; sampled-route execution is not a measured NFQ/X engagement claim. One listener at a time; every new arm launched and stopped its own managed server. No core, backend, model, or recipe source changed.

## Accuracy and format

| Correct answer position | With parent | Without parent |
|---|---:|---:|
| A | 5/5 | 5/5 |
| B | 5/5 | 5/5 |
| C | 5/5 | 5/5 |
| D | 5/5 | 5/5 |
| **Total** | **20/20** | **20/20** |

Every rendered reply was exactly one uppercase letter with no extra text, and every request ended with `finish_reason=stop`. All five probe categories scored **4/4 in each condition**. Usage was 2 completion tokens per request despite one displayed letter; do not infer a one-token native execution from the visible string. Full per-cell selections, output-ID digests, live positions, timing, and parent condition are in [robustness-results.json](robustness-results.json). Raw prompt/output IDs and ordinary HTTP receipts remain in ignored `logs/experiments/exp3-robustness-v1/spark-t0p65-s128-positions-parent/`.

## Speed and resource usage

Each five-question arm submitted **942 new input tokens** and emitted **10 completion tokens**. Wall below is the sum of those five ordinary HTTP requests; it excludes native startup, optional parent import, and explicit export. The unrotated parent arm was measured in the original Exp 3 run; all other arms are newly measured here.

| Option rotation | With parent wall s | Without parent wall s |
|---:|---:|---:|
| 0 | 33.636 (reused) | 21.718 |
| 1 | 33.367 | 22.007 |
| 2 | 33.417 | 21.847 |
| 3 | 33.713 | 22.049 |
| **Four-arm sum** | **134.133** | **87.621** |
| **Mean five-request arm** | **33.533** | **21.905** |

Across 20 requests per condition, weighted native PREFILL was **34.124 ms/new input token with parent** and **22.016 ms/new input token without**; decode was **69.189** versus **58.183 ms/output token**. The observed parent/no-parent HTTP-wall ratio was **1.531**, with identical new-input/output counts. The parent condition starts with 925 existing KV rows and has a different attention/state workshape and resident footprint; host/cache warmth can also differ. This is an observed condition contrast, **not** an isolated causal estimate of the KV cost or a kernel speedup. Do not compare those two-token decode denominators as ordinary long-form generation throughput.

Peak sampled all-inclusive OS RSS: **30,887,370,752 bytes (30.887 decimal GB)** with parent and **30,492,131,328 bytes (30.492 decimal GB)** without, a measured peak difference of **395,239,424 bytes (0.395 decimal GB)**. These are process-level OS-resident peaks under the same 60-GB ceiling, not additive per-request RSS or GPU device allocation. No GPU duty/power, cache-miss traffic, or expert-slot churn was measured.

## Boundary and next meaningful test

This matrix establishes robustness to **label position** on one seeded sampler/host/build, but does **not** show source-conditioned learning. A model with no parent made every same selection. The five supplied answer sets are not balanced for epistemic ambiguity: the correct alternative is often longer, more internally consistent, or aligned with ordinary safety norms. A future source-dependence test would need equally plausible alternatives whose distinguishing fact is available only in the authenticated parent, with option-position balancing and a matched no-parent control. That test has **not** been run here.

The seven new arms verified ordinary live-session loaded/saved positions, absence of replay, exact sampler controls, and server cleanup; they did **not** export complete final KV. The original unrotated parent arm alone has the separately authenticated final state/export from [the initial report](README.md). These position-control results must not be promoted to complete-state parity or cross-platform qualification. The parent artifact was imported in each parent arm and was never deleted by the run script.
