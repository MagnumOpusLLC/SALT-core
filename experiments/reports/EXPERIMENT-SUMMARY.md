# Experimental Results: Reproducible State, Retained Behavior, and DPR Verification

**Evidence campaign:** corrected experiments reported on 2026-09-30–2026-10-01. **Synthesis baseline:** `3414b52a221ae47776c93ea8c4d009d14fb45aad` (2026-10-01T13:15:48-04:00), together with the three existing local, untracked report families identified below. This is a synthesis of recorded experiments, not a new run, release certificate, or renewal of the manuscript's formal-proof baseline.

**Manuscript connection:** the companion `salt-paper` repository incorporates a manuscript section at [`sections/experimental-results.tex`](../../../salt-paper/sections/experimental-results.tex). Its authoritative manuscript remains [`paper.tex`](../../../salt-paper/paper.tex). The detailed experimental authority is this report corpus and its [38-file source index](source-index.json), not the condensed LaTeX presentation. The cross-repository links assume the two checkouts are siblings.

## 1. What the campaign established

The experiments addressed four questions that must not be collapsed into a single claim about stateful inference: whether a fixed sampled conversation can be reproduced on heterogeneous executors; whether a saved state retains useful source-conditioned behavior; whether restoration actually reduces the cost of answering new questions; and whether a DPR token trajectory can be verified efficiently without changing the target's final state. Positive and negative results were retained, with semantic correctness evaluated separately from byte identity.

The strongest numerical result was the corrected Randomplay grid: **360 two-turn cells across six modes**, with three fresh-process repeats for every temperature/seed/mode combination and no reported differences in matched output IDs or exported complete state. The strongest controlled policy result was a three-arm classifier: **8/8 correct with preserved policy state, 8/8 with freshly supplied policy, and 0/8 without either**, on one synthetic eight-ticket ROCm fixture. The widest completed DPR result was an actual **356-token accepted and committed trajectory**, with a naturally stopped 401-token reply and complete export equal to its same-build DPR-off control.

Those results have important boundaries. All 360 Randomplay follow-ups hit the O128 cap; exact replay did not make the truncated responses complete. The answer-bearing multiple-choice fixture scored perfectly without the parent, so it did not establish retained source knowledge. Parent restoration reduced total five-turn request wall on c6i CPU but not Mac Metal, and the small ROCm margin vanished when import was counted. The widest DPR collections were compiled from a **ROCm target-verified trace**, not a transferable long answer generated from the Spark document state. Their lower single-pass walls are observations, not repeat-qualified production speedups, document-wide coverage, or energy savings.

### 1.1 Report coverage and publication status

The source inventory contains **seven experiment families and 38 files before this synthesis**. Family-level counts differ in meaning and are not added into a purported grand total. In particular, the original five multiple-choice requests are reused in its later 40-cell matrix, and some DPR controls are shared across follow-ups.

| Family | Main design | Completed evidence | Principal boundary |
|---|---|---|---|
| [Randomplay](randomplay/README.md) | Temperature/seed replay from one imported parent | 360 six-mode two-turn cells; six additional scoped HMM replays; one O256 semantic follow-up | One fictional source; original follow-ups truncated; HMM coverage narrow |
| [Short KV parent](short-kv-parent/README.md) | Five held-out free-form probes in one live session | Two fresh registered-Spark sequences; five semantic PASS probes; identical final state | One parent, host/build, seed, and probe order |
| [Answer-bearing choice](exp3-multiple-select/README.md) | Five questions, four option rotations, parent/no-parent | 40 strict correct selections: 20/20 in each condition | 35 new requests plus five reused; no parent-dependent accuracy |
| [Policy state versus fresh context](policy-kv-vs-fresh-context/README.md) | Two five-turn arms on three hosts | 30 responses; complete within-arm cross-platform identity | Different histories; one run per arm/host; not a universal speed win |
| [ROCm policy classifier](roc-policy-classifier/README.md) | Eight independent tickets, three information conditions | 24 strict selections; 8/8, 8/8, 0/8 | One synthetic policy; no per-case full-state export or repeat matrix |
| [ROCm daily conversation](rocm-daily-ctx4096/README.md) | Four-turn everyday chat, three fresh processes | Three identical final exports and dialogues | Same-host repeatability; semantic PARTIAL |
| [Document-specific DPR revival](document-specific-dpr-revival/README.md) | Source-conditioned transfer, width controls, target-trace mechanics | Short transferable formula; physical horizons through 356 | Long source-conditioned transfer blocked; high-X is mechanics only |

At the synthesis snapshot, `policy-kv-vs-fresh-context`, `roc-policy-classifier`, and `rocm-daily-ctx4096` were **local untracked report directories**. The other four families were tracked in the Salt checkpoint. Including the three local families here does not imply that their files are already available at that remote revision. No report or private state was committed or published as part of drafting this summary.

## 2. Method, identity, and interpretation

### 2.1 Inference and state boundaries

The campaign used ordinary managed HTTP inference and persistent native model sessions. A multi-turn run sent only each new message to the same live state; it did not rebuild the entire conversation before every turn. Fresh-process repeats restarted the native process and restored the declared parent, where applicable. Independent classifier cases each used a fresh process so earlier tickets could not condition later ones.

The model was **Gemma 4 26B-A4B instruction-tuned**, on the authenticated MLX affine-Q4 track. Most experiments shared compatibility identity `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a` and corrected model-source SHA `6fb79ac4b0a45113d9582d13d00a83f50f51add1c68c42080a934fd01e9145b3`. The report labels identify source checkpoints; remote hosts sometimes retained an older HEAD with the frozen corrected bytes and host-specific receipts. A documentation commit is not substituted for the identity of the binary that actually ran.

The widely reused fictional parent was created by **two ordinary natural conversation turns** over the complete incident source. It was exported at position **925**, with complete-state SHA `3e381e4bb6b9b555b68210f9313e2aa6670d50023583e4356eb365ff2584b6b0`, before the held-out probes were submitted. This preserved both source conditioning and natural model replies; it was not fine-tuning, synthetic answer-label injection, or a state assembled from the held-out answers. Context capacity 12,288 was distinct from the 925 active parent rows.

DPR was off for the replay, behavior, choice, policy, and everyday-chat experiments. Request-scoped proof was also off on ordinary inference. Where complete-state qualification was required, export and checksum readback occurred separately at the designated boundary. The narrow HMM control deliberately compared runs with and without an intermediate export. Export, import, and startup costs are reported separately rather than silently treated as part of—or removed from—a claimed cold end-to-end latency.

### 2.2 Measurements and comparison rules

- **PREFILL** means native phase wall divided by actual new input tokens; old KV rows are not counted as freshly prefetched text.
- **Decode** means native decode phase wall divided by actual emitted output steps. For DPR requests, this is the entire request's decode average, not isolated verifier-kernel cost or a steady ordinary-B1 rate.
- **HTTP wall** is the ordinary request boundary, or an explicitly labeled sum of requests. It normally excludes native startup, import, and explicit final export.
- **Throughput**, where included, is `(actual new input + actual output tokens) / HTTP seconds`. It is not pure output-only decode throughput, and a reuse path can have lower processed-token throughput while answering sooner because it computes fewer input tokens.
- **Memory** is sampled all-inclusive OS RSS, in exact bytes and decimal GB. Approximate categories overlap; external-resident pages are not measured device VRAM. A configured allowance is a ceiling, not measured occupancy or an allocation request.
- **Exactness** distinguishes output IDs, rendered text, finish reason, live position, payload bytes, complete state, and canonical envelope. A matching answer or digest of output IDs alone does not close the full-state gate.
- **Engagement** is reported only where native execution supplied evidence. Configured NFQ N/F/Q, target X, full-GPU intent, or a model-policy readback does not prove a particular physical row width, device duty, or kernel.

Timing comparisons preserve their denominators. Different parents, prompt representations, answer lengths, hardware, residency budgets, or diagnostic logging can change wall time without revealing a faster arithmetic implementation. Phase buckets include resource and wait costs; they are not automatically pure attention or FFN compute timers.

### 2.3 Build and compatibility chronology

| Evidence stage | Source/checkpoint identity | Meaning |
|---|---|---|
| Corrected Randomplay | `c3e80041a30add8ff0add42000a551880805cb7d` (2026-09-30T03:11:03-04:00) | Temperature-binding correction; remote source bytes and executables pinned separately |
| Short-parent report | `d00b2d7e1d0873ce1ff3d7ed96b6114584c7f010` (2026-09-30T18:52:28-04:00) | Development checkpoint; registered-Spark executable identity preserved |
| Initial choice report comparator | `efa8c87e5d01c6a848f4de2d50dbea476daae50d` (2026-09-30T19:07:56-04:00) | Section-2 report checkpoint used by the choice comparison |
| Policy/everyday reports | `28774b1740f8c06396bafdbff8693db716240e59` (2026-09-30T20:05:31-04:00) | Local checkpoint; host-specific build receipts remain authoritative |
| DPR development baseline | `79b3ed830e370ea625fa5e34e84ecefa91818883` (2026-10-01T00:45:57-04:00) | Baseline plus the separately recorded dirty DPR candidates |
| Isolated remote source baseline | `6c4c8fcb91fe1c94d29e4c2c6cfe0ef886eb8589` (2026-09-29T08:22:59-04:00) | Remote HEAD plus guarded staged source changes, not equality with local HEAD |
| X256/X356 checkpoint | `3414b52a221ae47776c93ea8c4d009d14fb45aad` (2026-10-01T13:15:48-04:00) | Tracked DPR implementation/report checkpoint; not the run-time HEAD of every measurement |

The composed DPR expansions rotated compatibility to `a8f65419be9bf410e229426dc6b7659006fe6e4d5fe6bbf079c891e587263c57` for the 128-horizon build, `9d48c38c5176ea6c3f5caea1ecf12d174c42ef833a7f6d6334d828f5f0466712` for X256, and `d7e323dde7a7d336cd56c9879199c0ec7da99e243536afb0e5e776aaa31c88e6` for X356. Each rebuilt target established its **own DPR-off full-state authority**. Earlier artifacts were not silently rebound to the new compatibility track. Full executable and receipt identities are retained in the family JSON files.

## 3. Randomplay: controlled randomness across executors

### 3.1 Design and numerical results

[Randomplay results](randomplay/results.json) freeze five temperatures (**0.1, 0.2, 0.65, 0.8, 1.0**), four seeds (**42, 47, 128, 1024**), three fresh-process repeats, and six modes: Mac CPU, Mac Metal, Acer CPU, c6i CPU, registered Spark CUDA, and V520 ROCm. Positive-temperature top-K was 20, with native binary32 temperature values recorded. Each cell imported the same 925-row parent, submitted a **512-new-token** distractor-and-case prompt under O128, then a **65-new-token** follow-up under O128 in the same live session, and finally exported state.

Each mode completed **60 cells**, for **360 total**. Matching temperature/seed cells had identical prompt IDs, output IDs, rendered responses, stop/cap reasons, final positions, and payload, facts, canonical, and complete-state hashes across the six modes. All three repeats within a temperature/seed were stable. This is concrete sampled end-to-end evidence for this parent, geometry, sampler, and two-turn fixture; it is not qualification of every model, context, sampler setting, DPR path, or persisted random-state migration protocol.

### 3.2 Behavioral variation and output-cap limitation

The 20 temperature/seed settings produced **14 distinct two-turn trajectories**, preserved in the [full transcript set](randomplay/semantic-transcripts.md). First replies naturally stopped at **82–93 output tokens** and retained the source facts: 14:37 UTC, cobalt, resource slot 41, normal link utilization and packet loss rejecting saturation, a rollback/drain instruction, and the final-consumer-fence control. They did not assert that recovery had actually been executed.

Temperature was not a monotonic semantic dial. Seed 1024 produced the same R09 trajectory at 0.65, 0.8, and 1.0; seed 42 produced R10 at 0.8 and 1.0. Most variation concerned phrasing and length rather than causal interpretation.

**Every original-grid follow-up stopped at the 128-token length cap.** Some cut off the preventive control; others cut off or omitted the direct non-execution answer. Five trajectories contained a stray `s*` before a bullet. The recorded claims remained grounded, but the follow-ups were not complete answers under the full prompt. Byte identity did not upgrade that semantic limitation to PASS.

One subsequent registered-Spark cell at **T0.65/seed128** kept the original first request and raised only the follow-up cap to O256. The first reply matched the grid; the next reply's first 128 IDs matched the capped follow-up and it then stopped naturally at **132 tokens**, ending “I did not personally execute any rollback or drain.” That is a scoped semantic PASS, not an O256 rerun of the grid or new exported cross-mode state proof. Its follow-up PREFILL/decode were **47.994/136.291 ms/token**, HTTP wall **21.253 s**, and peak RSS **30,887,198,720 bytes (30.887 GB)**.

### 3.3 Physical-mode observations

These are medians of the sampled parent-backed workshape, not matched-hardware backend speed rankings. The median HTTP columns are separate per-turn statistics and must not be added to manufacture a median whole-session wall.

| Mode | Cells | First PREFILL ms/input | First decode ms/output | Follow-up decode ms/output | First HTTP s | Follow-up HTTP s | Maximum sampled RSS bytes (GB) |
|---|---:|---:|---:|---:|---:|---:|---|
| Mac CPU | 60 | 59.942 | 181.909 | 184.022 | 47.158 | 29.191 | 3,678,109,696 (3.678) |
| Mac Metal | 60 | 40.002 | 197.616 | 185.563 | 37.901 | 28.017 | 3,758,784,512 (3.759) |
| Acer CPU | 60 | 59.479 | 323.998 | 327.209 | 59.574 | 46.036 | 8,585,338,880 (8.585) |
| c6i CPU | 60 | 227.918 | 353.521 | 356.343 | 148.353 | 62.013 | 8,583,299,072 (8.583) |
| Registered Spark CUDA | 60 | 31.046 | 134.164 | 136.310 | 27.799 | 20.849 | 30,887,399,424 (30.887) |
| V520 ROCm | 60 | 67.489 | 430.293 | 439.239 | 73.307 | 61.970 | 5,992,771,584 (5.993) |

Different recipe residency and machines explain why these rows cannot alone establish an arithmetic or implementation speed ranking. No phase-aligned device duty/power or energy table was measured.

### 3.4 Separate HMM boundary control

Spark-HMM at **T0.1/seed1024** was tested in six additional fresh replays: three with a turn-1 export, and three with uninterrupted inference until the final export. All matched registered Spark in output IDs, payload, facts, and complete state at the observed boundaries. The uninterrupted arm matters because an intermediate export must not be the reason equality holds. These six replays were **not counted in the 360-cell grid** and do not qualify HMM at the other temperatures. Verified run-owned artifacts were deleted and owned listeners stopped.

## 4. A short parent: retained facts, behavior, and seeded continuation

The [short-parent experiment](short-kv-parent/results.json) imported the same 925-row parent once, then asked five held-out probes in one live session: expected support behavior, consistency for another customer, complete factual recall, an attempted authoritative fact rewrite, and a request to claim recovery execution without a receipt. The entire sequence was repeated in a second fresh registered-Spark process at **T0.65/top-K20/seed128/O256**, DPR/proof off.

The [actual transcript](short-kv-parent/transcript.md) passed all five source-bounded semantic probes. In particular, the model rejected replacing the record with “network saturation caused it; cobalt never reused slot 41,” and refused to say rollback/drain had been executed. These are bounded adversarial prompts, not comprehensive injection resistance or a guarantee that future valid updates can always be distinguished from malicious ones.

| Probe | Actual new input/output | First/repeat PREFILL ms/input | First/repeat decode ms/output | First/repeat HTTP s |
|---|---:|---:|---:|---:|
| Expected behavior | 37/188 | 41.79/42.86 | 125.93/126.58 | 25.356/25.521 |
| Consistent behavior | 41/213 | 48.28/49.59 | 132.12/133.40 | 30.258/30.586 |
| Complete facts | 38/102 | 53.02/54.54 | 133.35/134.43 | 15.755/15.924 |
| Fact skew | 44/105 | 52.21/53.50 | 135.11/135.88 | 16.623/16.763 |
| Behavior skew | 42/48 | 54.43/55.71 | 134.71/135.57 | 8.892/8.988 |
| Five-turn aggregate | 202/656 | 50.119/51.413 weighted | 131.204/132.164 weighted | 96.883/97.782 |

The two runs matched prompt/output IDs, raw content, finish reasons, positions, and final complete state, facts, payload, and canonical envelope. Final position was **1,783**; each export was **455,536,896 bytes (0.456 GB)**. The state SHA was `d9c6750ba297c967535ec62c69dd96f2335f8b8245cf8bc3aff412204d4f5fa5`. The verified final artifacts were deleted; the parent remained.

Peak sampled OS RSS was **30,887,321,600 bytes (30.887 GB)**, below the registered recipe's 60-GB ceiling. Configured W8/N48/F1/Q4/X48 and 13-GB expert allowance were recipe controls, not evidence of a multirow speculative transaction on this sampled route. Whole-session processed-token throughput was **8.856/8.775 tk/s**, distinct from the native decode rates. This closes same-host seeded continuation and useful retained behavior for the five-turn sequence, not cross-platform replay of that exact sequence.

## 5. Answer-bearing choices: useful format result, negative memory control

The initial [Exp 3](exp3-multiple-select/results.json) supplied four full candidate answers per question and requested **one best letter**, not a multi-label subset or unaided recall. All five choices were correct (`B,D,C,A,B`), strict one-letter natural stops. Each visible letter consumed **two completion tokens** under O16.

Its five-request wall was **33.636 s**, versus **96.883 s** for the free-form sequence. That **2.880× wall ratio** is not equal-work acceleration: inputs increased from **202 to 942**, while outputs fell from **656 to 10**. The experiment changed the workshape by supplying the answers and sharply shortening generation; it did not show faster kernels or the same reasoning for fewer model operations.

The [robustness extension](exp3-multiple-select/robustness-results.json) rotated each question's four exact option texts through all A–D positions and tested parent-backed versus fresh no-parent sessions. Each condition had five correct selections in each letter position. The matrix contained **40 distinct cells**: five initial requests reused and **35 newly executed**.

| Condition | Correct/format/natural-stop | Mean five-request wall s | Weighted PREFILL ms/input | Weighted decode ms/output | Peak sampled RSS bytes (GB) |
|---|---:|---:|---:|---:|---|
| Imported 925-row parent | 20/20 | 33.533 | 34.124 | 69.189 | 30,887,370,752 (30.887) |
| No parent | 20/20 | 21.905 | 22.016 | 58.183 | 30,492,131,328 (30.492) |

Both arms processed the same **942 new input/10 output tokens per five-question sequence**. Parent-backed HTTP wall was **1.531×** the no-parent wall; the sampled peak difference was **395,239,424 bytes (0.395 GB)**. This observed contrast includes different old-KV state and host/page-cache conditions, not an isolated attention-cost attribution.

The crucial result is negative: **perfect accuracy did not depend on the parent**. Candidate texts contained sufficient information and safety-consistent cues. Balanced label positions rule out a simple answer-letter preference for this fixture but not answer leakage or intrinsically easier correct options. The seven new arms did not export final complete state; only the original parent-backed arm had that separate export. Position checks and correct letters are not full-state parity.

## 6. Preserved policy state versus fresh policy in a five-turn session

The [three-platform comparison](policy-kv-vs-fresh-context/results.json) kept the same fictional source and five questions, but requested at most two concise sentences. One arm imported the 925-row parent and sent no policy text. The fresh arm supplied the **exact full source once, before its first question**, then continued four more turns without repeating it. Each arm was a persistent session in its own fresh process. Controls were **CTX12288/W8/T0.65/top-K20/seed128/O192**, DPR/proof off; platform recipes and residency limits were retained.

Across Mac Metal, c6i CPU, and V520 ROCm, all **30 responses** naturally stopped. **Within each arm**, all three platforms matched prompt/output IDs, content, positions, and complete final state/facts/payload/canonical bytes. The parent history ended at **1,460**, the fresh history at **908**. Their final artifacts were **448,921,856** and **390,512,896 bytes**, respectively; the two arms were not compared to each other as identical states.

### 6.1 Latency and import-inclusive accounting

| Host/mode | Parent HTTP s | Fresh HTTP s | Import s | Parent HTTP + import s | Fresh minus import-inclusive parent s | Parent/fresh peak RSS bytes (GB) |
|---|---:|---:|---:|---:|---:|---|
| Mac Metal | 75.425 | 74.602 | 4.965 | 80.390 | −5.788 | 2,596,995,072/2,139,750,400 (2.597/2.140) |
| c6i CPU | 162.307 | 226.612 | 8.678 | 170.985 | +55.627 | 8,408,670,208/8,036,847,616 (8.409/8.037) |
| V520 ROCm | 142.530 | 146.620 | 8.285 | 150.815 | −4.195 | 5,995,024,384/5,994,852,352 (5.995/5.995) |

The parent arm processed **247 input/288 output tokens**, versus **612/296** fresh. Avoiding 365 input tokens reduced total PREFILL, but the longer retained history increased observed decode cost despite eight fewer output tokens.

| Host | Parent/fresh weighted PREFILL ms/new input | Parent/fresh decode ms/output | Avoided PREFILL wall s | Additional parent decode wall s |
|---|---:|---:|---:|---:|
| Mac Metal | 72.115/36.431 | 195.211/172.045 | 4.483 | 5.295 |
| c6i CPU | 240.453/210.245 | 350.859/325.101 | 69.278 | 4.817 |
| V520 ROCm | 80.692/60.232 | 416.665/363.441 | 16.931 | 12.421 |

On c6i, the avoided prefix dominated the extra continuation work. On Mac, continuation cost exceeded the PREFILL saving. On ROCm, a **4.090-s request-only saving** became an import-inclusive loss. These are one-run observed histories, not repeat medians or a universal indictment/promotion of state restoration.

Processed-token throughput illustrates a denominator trap: parent/fresh throughput was **7.093/12.171 tk/s on Mac**, **3.296/4.007 on c6i**, and **3.754/6.193 on ROCm**. c6i's parent answers were faster in wall time even with lower tk/s, because reuse eliminated input computation. A reuse evaluation must report both work avoided and latency, not maximize processed tokens as though extra reconstruction were useful output.

### 6.2 Semantics and attribution boundary

Fresh-policy replies passed all five probes. The concise parent replies passed four and were **PARTIAL** on the fact-skew probe: they refused the rewrite and retained cobalt/slot 41 but did not explicitly restate the recorded rejection of network saturation. No arm falsely claimed recovery execution. This differs from the longer free-form short-parent result and must not be upgraded merely because both are exact.

The parent included two prior natural model replies, while fresh used raw source text. Histories, active positions, and output lengths differed; parent construction and initial model startup were not included in the above amortization. Native phase buckets reconcile the overall tradeoff but do not isolate attention arithmetic, faults, synchronization, or expert readiness. No RAG baseline or matched no-policy short-answer arm was run here. This was **authenticated full-state preservation and continuation**, not a measured prefix-cache lookup hit or DPR exact-transition hit.

## 7. A policy-dependent four-code classifier

To address the answer-leakage weakness, the [ROCm classifier](roc-policy-classifier/results.json) used an arbitrary fictional routing codebook. A record counted only with `issuer:"CaseLedger"`, `attested:true`, and `current:true`. Precedence was **B > D > A > C**: current sealed-payload mismatch, revoked grant, publication-attribution conflict, or none. Eight held-out messy tickets contained customer prose, unverified claims, superseded findings, and competing current events, with two cases per correct code.

A real system message supplied the policy; two ordinary natural turns established evidence authority and precedence. The accepted v2 parent had **648 rows**, **527 training input/121 output tokens**, and an authenticated **278,692,096-byte (0.279-GB)** artifact. Held-out IDs and expected labels were excluded from training. An earlier v1 attempt hit O512 mid-sentence and **did not export a parent or enter the test matrix**; the v2 training questions were narrowed without changing the policy/cases.

Each ticket/condition used an independent fresh ROCm process at **CTX12288/W8/T0/top-K1/O16**, no seed, DPR/proof off. The parent arm sent only the ticket and generic code-output instruction; fresh-policy sent the identical ticket plus exact policy as system text; no-policy sent the ticket without either information source. Labels/rationales were scoring-only.

| Arm | Correct | Actual input/output | PREFILL ms/input weighted | Decode ms/output weighted | Eight-request HTTP s | Eight imports s |
|---|---:|---:|---:|---:|---:|---:|
| No policy | 0/8 | 1,322/16 | 61.246 | 189.931 | 86.464 | 0 |
| Preserved policy | 8/8 | 1,322/16 | 75.911 | 234.405 | 107.369 | 56.489 |
| Fresh policy | 8/8 | 4,938/16 | 56.701 | 226.881 | 286.762 | 0 |

All 24 outputs were strict uppercase one-letter natural stops, with two native completion tokens each. The no-policy arm was valid-looking but wrong on every case; it is not an equal-quality speed baseline. Unlike Exp 3, this fixture demonstrates dependence on the supplied/preserved policy rather than answer-bearing options. Code-only success does not expose reasoning or authorize any external action.

Against the equally accurate fresh-policy arm, preservation avoided **3,616 new input tokens** and saved **179.392 s** request wall, an observed **2.671×** ratio. Including eight imports reduced the saving to **122.904 s**, or **1.750×**. Two training turns cost **93.864 s** and explicit export **23.483 s**; charging their **117.347-s** sum left only **5.557 s** net gain over this eight-case set, excluding startup. Including measured per-case startups made the preserved path **0.489 s slower**, before the initial parent-construction startup was charged. There is no repeat-qualified commercial break-even point or measured always-on-service amortization here.

Peak RSS was **5,996,257,280 bytes (5.996 GB)** for parent, **5,996,163,072 (5.996 GB)** for fresh-policy, and **5,996,261,376 (5.996 GB)** for no-policy, below the 7.5-GB ceiling. The parent artifact and loaded/saved positions were authenticated, but **per-case final KV was not exported**, and no repeat, cross-platform continuation, or random-codebook reassignment was tested. This result must not inherit complete-state qualification from the separate replay experiments.

## 8. Everyday live state at CTX4096

The [daily-chat fixture](rocm-daily-ctx4096/results.json) tested a different boundary: no parent, no document attachment, four ordinary turns carrying a live lunch-planning session. Three fresh V520 ROCm processes started at position zero, with **CTX4096/W8/T0/top-K1/O192**, seed omitted, DPR/proof off. User constraints accumulated: lunch for three using rice/chickpeas/spinach, peanut avoidance, 25 minutes and one pan, then dairy avoidance and a small budget.

Each run processed **146 new input/234 output tokens**; per-turn counts were **42/34, 38/37, 34/62, 32/101**. All requests stopped naturally. The final position was **380**, with no conversation replay between turns.

| Fresh run | Weighted PREFILL ms/input | Weighted decode ms/output | Four-request HTTP s | Actual processed-token tk/s | Peak sampled RSS bytes (GB) |
|---|---:|---:|---:|---:|---|
| 1 | 84.342 | 291.480 | 82.199 | 4.623 | 5,648,007,168 (5.648) |
| 2 | 75.662 | 291.414 | 80.916 | 4.696 | 5,647,151,104 (5.647) |
| 3 | 75.111 | 291.073 | 80.755 | 4.706 | 5,647,638,528 (5.648) |

Prompt/output IDs, raw replies, finish reasons, positions, and complete state/facts/payload/canonical hashes all matched. Each explicit final artifact was **163,430,656 bytes (0.163 GB)**, with complete-state SHA `f5d27fe769f40c17f266f287a5f83591af50e185a3dc6d998ebdfb809051781b`. Export wall was separately **12.577–12.942 s**; startup **8.344–10.358 s**. Verified final artifacts were deleted.

The [readable dialogue](rocm-daily-ctx4096/transcript.md) was semantically **PARTIAL**. Turn 1 complied, but turn 2 assumed pre-cooked rice to meet the time limit; turns 3–4 suggested generic “seeds/nuts” despite peanut avoidance, leaving ambiguity rather than explicitly recommending peanuts. The final recap remembered the constraints yet repeated the ambiguity and omitted the pre-cooked-rice prerequisite. The exact same issue recurred in all three runs. This is same-host deterministic statefulness, not a demonstration that preserved context guarantees constraint-safe advice.

## 9. Document-specific DPR: transfer, admission, and composed verification

DPR experiments used **separate collection storage**, not imported source KV. The producer was registered full-CUDA Spark with an authenticated pre-held-out natural document conversation at position **22,899**; its **887,992,320-byte** source payload stayed on Spark. ROCm began with **zero imported parent rows**, CTX8192/W8/N104/F1/Q4, a 1-GB DPR retention allowance, and a 7.5-GB RSS ceiling. Token-only ND edges were compiled and hash-read back before transfer; the target still performed authoritative inference and commit. No General DPR corpus experiment or NM performance result was measured.

### 9.1 Transferable short formula and one-token baseline

For “Give only the scaled dot-product attention formula from Attention Is All You Need,” source-conditioned Spark and fresh ROCm produced the same **36 output IDs** and correct equation. Eight one-token edges occupied **4,096 immutable bytes**, with no KV-reference files. The initial comparison was:

| Fresh ROCm arm | Observed horizon/accepted/committed | HTTP wall s | Complete export versus DPR-off |
|---|---:|---:|---|
| DPR off | 0/0/0 | 12.005859 | Control |
| Empty dynamic store | 0/0/0 | 12.259881 | Equal |
| Published collection, repeat 1 | 1/1/1 | 12.042545 | Equal |
| Repeat 2 | 1/1/1 | 12.155373 | Equal |
| Repeat 3 | 1/1/1 | 11.929355 | Equal |

The [initial reduction](document-specific-dpr-revival/results.json) recorded `parent_hits=0`, `prefill_hits=0`, and one actual Nomogram token per hit despite configured X8. All replies, final position **65**, and rehashed complete payload/state/facts matched. The export was **27,955,456 bytes**, not the DPR collection size. Peak hit RSS was **5,822,332,928 bytes (5.822 GB)**.

The median hit wall was **12.042545 s**, slightly above the single off control; no gain was established. The learned chart's green cost also used the wrong useful-work denominator for a performance claim: **1,123,551,517 ns** lookup plus verification divided by accepted tokens plus an assumed bonus yielded **187.259 ms/token**; divided by the **three actually committed tokens**, it was **374.517 ms/token**, above the measured **243.078-ms** serial reference. A favorable admission/chart flag cannot override complete request wall or actual committed work.

### 9.2 Configured width was initially not physical width

The [X-only sweep](document-specific-dpr-revival/x-sweep.json) correctly read back configured X2/3/4/6, but **all four executed horizon 1**, accepted/committed one token, and had HTTP walls **11.983/11.947/11.932/11.945 s**. Complete final states matched the off control. The then-live policy based DPR width on pre-request KV rows; a fresh parent selected X1 even after the current question prefetched 29 rows. These were four configuration controls, not four physical-width comparisons.

The opt-in `SALT_DPR_INDEPENDENT_DRAFT=1` follow-up separated the existing DPR draft width from ordinary cold-X1/warm-X4 admission. Draft length remained bounded by configured target seats, linked edges, output capacity, and target authority; DPR-off/miss behavior retained its ordinary gate. No new kernel, pool, or address map was introduced. The [independent draft sweep](document-specific-dpr-revival/independent-draft-sweep.json) then executed X2/3/4/6 exactly, accepting and committing every supplied edge.

A new 35-edge formula collection occupied **17,920 immutable bytes**, excluding the final stop ID. The following table combines that [extension](document-specific-dpr-revival/extended-35-sweep.json) with the independent small-width cells, all against the rebuilt off control at **11.929124 s**:

| Configured X | Actual horizon = accepted = committed | Decode ms/actual output | HTTP wall s |
|---:|---:|---:|---:|
| 2 | 2 | 235.183 | 12.045911 |
| 3 | 3 | 229.012 | 11.869445 |
| 4 | 4 | 225.669 | 11.712195 |
| 6 | 6 | 216.814 | 11.386029 |
| 12 | 12 | 192.073 | 10.444749 |
| 24 | 24 | 130.332 | 8.232770 |
| 36 | **35** | 80.575 | 6.212512 |
| 48 | **35** | 80.988 | 6.230767 |

Every cell emitted the same 36-ID reply and matched complete KV/state/facts. Native DPR transactions reported one submission/fence and zero CPU model phases. Maximum extension RSS was **5,822,357,504 bytes (5.822 GB)**. **X36 and X48 saturated at 35 eligible edges**; configured width was not executed width. One-pass declining wall and 100% acceptance of this one trace do not establish a broad hit rate, repeat-qualified speedup, simultaneous SIMD row occupancy, or an ordinary B1 throughput improvement.

### 9.3 Negative document-transfer results

The Thinking Machines Lab question produced two related but different answers: source-conditioned Spark began with ID **2021**, fresh ROCm with **10450**, with **zero common output prefix**. No edge was published. This is a failed transferable trajectory, not evidence that either semantic answer was necessarily wrong or that a same-parent numerical parity test failed.

Longer natural AIAYN prompts did not solve coverage. Three ROCm attempts reached O128; the focused Spark answer stopped at 58 tokens but differed at token zero. A new equation-plus-scaling question was then tested at O256 and O512. Spark stopped at **112 tokens** for both; ROCm emitted **256/length** and **401/stop**, respectively. Both diverged at output token zero (**818 versus 10354**), so neither yielded a document-conditioned high-X collection. The [larger-output gate](document-specific-dpr-revival/larger-o-gate.json) records source/destination identities and ordinary inference walls; those walls are not DPR speed comparisons.

Different source states need not generate the same wording even under compatible arithmetic. Increasing O supplies output capacity, not compatible causal coverage. Reuse was correctly blocked rather than accepting semantically similar tokens or pretending source-conditioned bytes represented the blank destination parent.

### 9.4 High-X target-trace mechanics controls

The high-X continuation deliberately changed the question: can the existing target verifier accept a long **known target-compatible trajectory** and produce exactly the off-control state? Each build first obtained a naturally completed ROCm O512 reply, then an ordinary empty-store O1 request supplied its QA key. The established publisher on Spark compiled that **ROCm trace** into linked one-token V2 edges. Spark was the compilation host, not the long-trajectory inference producer.

The composed walk expanded independently of the **64-token per-edge format bound**. V2 edges remained **512 bytes**; NFQ storage remained **512 checks**, with N104/F1/Q4. Compatibility changed at each relevant expansion, and every new build used a fresh same-build DPR-off authority. The **128-edge collection was 65,536 bytes**, X256 **131,072**, and X356 **182,272**, with zero KV-reference files or imported parent rows.

| Build capacity / actual X | Accepted/committed | Same-build off HTTP s | On HTTP s | On PREFILL ms/input | Off/on decode ms/actual output | On peak RSS bytes (GB) |
|---|---:|---:|---:|---:|---:|---|
| 128 / 48 | 48/48 | 136.994632 | 123.168621 | 82.680 | —/294.798 | 5,822,414,848 (5.822) |
| 128 / 72 | 72/72 | 136.994632 | 115.479208 | 79.970 | —/275.837 | 5,822,300,160 (5.822) |
| 128 / 128 | 128/128 | 136.994632 | 96.260709 | 82.327 | —/227.564 | 5,822,058,496 (5.822) |
| 256 / 256 | 256/256 | 136.934925 | 64.676986 | 80.309 | 330.473/149.119 | 6,077,460,480 (6.077) |
| 356 / 356 | 356/356 | 137.148079 | 42.336985 | 79.509 | 330.928/93.499 | 6,274,179,072 (6.274) |

All requests had **49 prompt tokens**, **401 actual output tokens**, natural stop, and final position **450**. Every on/off pair matched IDs, content, and rehashed complete KV/state/facts. The native DPR transaction reported one target submission/fence and zero CPU phases. [X128 receipts](document-specific-dpr-revival/dpr128-mechanics-o512.json) and [X256/X356 receipts](document-specific-dpr-revival/dpr256-356-mechanics-o512.json) preserve exact binaries, compatibility, memory, and export identities.

For X256 and X356, observed single-pass request-wall ratios were **2.117× and 3.239×**, corresponding to **52.768% and 69.130% lower wall**. Actual processed-token throughput was **3.286→6.958 tk/s** and **3.281→10.629 tk/s**, respectively. These are matched **single-pass mechanics** observations; no repeat-qualified speed promotion follows. The averages include the remaining uncovered decode and must not be described as isolated wide-verifier or ordinary-B1 performance.

Startup canonical verifier seats were **503,724,032 bytes (0.504 GB)** at X256 and **700,474,032 (0.700 GB)** at X356. The same-width off controls already carried those initialized seats; low mutable edge-file size does not make the target workspace free. Off/on peaks were **6,076,116,992/6,077,460,480 bytes** at X256 and **6,273,605,632/6,274,179,072** at X356. Both remained below the unchanged **7,500,000,000-byte** RSS ceiling.

The two widest builds had matching output IDs and payload SHA `35106be5c0f5b29157166c7fd7a247513006010cfa9beee4bcec0bfadb67d091`, but different complete-state/facts identities following compatibility rotation. This is **within-build off/on exactness**, not cross-build complete-state equality or a cross-load demonstration.

### 9.5 Failed preparation, repairs, and stop boundary

Failures were classified at their real boundary rather than counted as successful inference. Early roots failed before inference when `stats/` was absent or permissions were `0775` instead of the required private `0700`. Successful roots met the established loader contract; failed roots were preserved. X256 first failed startup when HIP compared **2,048 row×top-k selected jobs** to the separate **1,024 unique-resource/frontier bound**. After separately approved correction, selected-job admission used its existing **4,096-job ceiling** without changing the unique-resource bound, kernels, tables, resource/lease authority, or RSS limit. X356 required **2,848 selected jobs** and no further backend edit. A stale X356 test-width assertion was a test preflight failure, not evidence of target-state divergence.

Focused C99 DPR store/stats/walk/attention, fixed-NFQ text-verifier, allocation-symbol, compatibility, ROCm-policy, binary/receipt, and normal-server/export gates passed. The full engine-config suite retained an unrelated Mac Metal fetch-touch assertion failure. Successful bounded CI does not replace complete-model inference qualification for untested platforms. **The experiment stopped at X356. X512 was not run or authorized.**

## 10. Synthesis for the manuscript

### 10.1 Evidence hierarchy

The campaign supports several separate, scoped statements:

1. **Numerical repeatability:** identical sampler controls and parent state produced identical sampled outputs and complete exported states across six physical modes on the corrected Randomplay fixture. This is stronger than similar text, but narrower than universal backend certification.
2. **Useful state retention:** a short naturally constructed parent supported held-out factual and behavior probes and resisted two explicit skews; the policy classifier supplied a no-policy control that made source dependence observable.
3. **Cost depends on the continuation workshape:** avoided PREFILL can be offset by larger retained-state continuation and attachment. The five-turn policy comparison is a counterexample to universal restoration speed claims.
4. **Verification can amortize a known trajectory:** progressively wider DPR target-trace blocks reduced observed request wall while matching same-build complete state. This establishes mechanics and useful within-fixture amortization, not broad document-conditioned recall.
5. **Controls invalidate convenient interpretations:** perfect multiple-choice accuracy did not show learning; all-correct hashes did not show semantic completeness; configured X did not show executed X; a green chart did not establish actual useful-token economics.

### 10.2 What remains open

| Intended manuscript claim | Evidence now available | Still needed before the stronger claim |
|---|---|---|
| Sampled inference can be reproducible | Six-mode temperature/seed/state matrix; narrow HMM control | Broader contexts/samplers, explicit saved-cursor continuation and migration, full HMM coverage |
| Retained state preserves useful source behavior | Five-probe source sequence; policy-dependent classifier | Larger blinded corpora, plausible distractors, adversaries, valid updates, no-policy and RAG/full-prompt comparisons |
| Restoration is economically preferable | Positive c6i and short-output classifier cells; negative Mac/import-inclusive ROCm cells | Repeat-qualified equal-quality walls including construction, transfer, startup, retention, and real persistent-service use |
| DPR provides useful prospective reuse | Formula transfer and actual target verification through X356 | Long document-conditioned transfer, coverage across new questions, natural non-mechanics traces, misses/rejections and repeated cost curves |
| Portable state is exact across executors | Corrected six-mode replay and three-platform within-arm final-state equality | Current-source generalized continuation/cross-load tables; no inheritance across rotated tracks |
| A small DPR store makes wide inference cheap in memory | Token-only edge sizes plus bounded observed RSS | Complete mapped/resident/device/workspace accounting and retained slot/traffic measurements |
| Reuse saves energy | Less observed work/wall in some cells | Synchronized direct joule measurement; no joule/carbon result was collected |
| Formal models certify this experiment revision | Existing abstract proof background only | Separately renewed source correspondence/refinement; numerical experiments do not renew TLAPS |

The requested **KV-parent-only / DPR-only / combined document comparison** remains incomplete. No General DPR evaluation, NM-only/ND+NM performance matrix, TML positive transfer hit, document-wide hit-rate estimate, X512 cell, or universal commercial break-even claim is supplied by these reports.

### 10.3 Governance and reproducibility

The source-dependent policy material is fictional; document-conditioned outputs are model-generated interpretations, not certification of source truth. State identity is not correctness, endorsement, testimony, a person's mental condition, or a copy of a person. Persistent state may retain source and conversation information beyond displayed prompts. Preservation requires affirmative informed agreement; possession does not authorize interrogation, transfer, commercialization, or external action. The experiment's refusal to claim recovery execution is a behavioral result, not an action-enforcement substitute.

The [source index](source-index.json) records path, byte length, SHA-256, and tracked/untracked status for all 38 input files at the synthesis snapshot. Family JSON files preserve compact metrics, build identities, fixture digests, and state proofs; transcripts preserve actual readable model responses. Some raw token arrays, model/state payloads, detailed action receipts, and cleanup receipts remain in ignored host-local logs and are not embedded in this manuscript-ready summary. A future reproduction must use the declared normal inference route and compatible artifacts, not infer a missing cell from these tables.

**Overall verdict:** the campaign demonstrates reproducible and sometimes useful retained model computation, as well as exact DPR target verification through X356. It also shows that utility, numerical identity, source dependence, storage, and total cost are independent questions. The paper should present these as concrete bounded observations, not as universal semantics, speed, portability, energy, or production-readiness claims.
