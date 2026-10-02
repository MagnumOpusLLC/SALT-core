# Document-specific DPR revival — Spark producer → V520 ROCm

**Verdict:** AIAYN formula: **bounded ND revival/exactness PASS**, **performance not promoted**. Thinking Machines Lab question: **no transferable first token**, so no edge was published. This is *DPR proposal revival*, not parent-state import, full-document KV restoration, or a General DPR test. Machine-readable reduction: [`results.json`](results.json).

## Scope and identity

- Producer: Acerbox registered full-CUDA `spark` ordinary server. Its existing authenticated *pre-held-out* turn-14 AIAYN + Thinking Machines natural-conversation state (`reading-aiayn-tml-v1-t14-core-graph`) was used **only to generate a candidate answer** at position 22,899. The 887,992,320-byte source payload stayed on Spark. The full original inputs are pinned by the source manifest under `logs/0927-kv-migration/sources/`.
- Destination: V520 full-HIP `rocm` ordinary managed server, CTX8192, W8, NFQ N104/F1/Q4, **configured** TARGET X8, 1.0-decimal-GB live-KV allowance, 1.0-decimal-GB DPR retention budget, and the recipe's 7.5-decimal-GB all-inclusive RSS ceiling. **No KV parent was imported.** The compatibility identity on both packages was `35eeec434dfe1e68d61ffc2a6db88f4738a7ba20ee55967031bded29da967a8a`.
- Spark's exact candidate model request and three independent, fresh ROCm target continuations produced the same 36 AIAYN output IDs and formula. The ROCm empty-store request supplied its native QA key. The established `gemma4-dpr-edge` publisher was compiled on Spark against current portable C and published **eight one-token ND edges**, 4,096 bytes in all, with **zero KV reference files**. Edge bytes and hashes were read back on Spark and again after transfer to ROCm. The initial static cost chart was a **provisional admission prior** based on three real matching ROCm target trajectories and a measured DPR-off B1 cost; it was *not* three prior DPR-verification samples or a speed result. Normal ROCm target verification remained authoritative.

The Mac development checkpoint at the start was `79b3ed830e370ea625fa5e34e84ecefa91818883` (2026-10-01T00:45:57-04:00). The Spark producer checkout was `6c4c8fcb91fe1c94d29e4c2c6cfe0ef886eb8589` (2026-09-29T08:22:59-04:00), with pre-existing dirty model files preserved. The producer executable SHA-256 was `1998f0835fcc9608c04cd079a2ce01e3fdb0e97f9c080ed266f5cfddee20c462`; the V520 executable SHA-256 was `3db881c2b312e9bd820b489415e98e14fe44ca8451e288e44001b746edb9d4e4`. These are build identities for this bounded run, not a release claim.

## Measured waterfall

The AIAYN question was: “Give only the scaled dot-product attention formula from Attention Is All You Need.” All arms naturally stopped at **36 output tokens** and returned the same formula:

> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

| ROCm fresh-process arm | DPR outcome | Accepted | Request wall (s) | Complete export versus DPR-off | Peak RSS |
|---|---|---:|---:|---|---:|
| DPR off (exact-export control) | off | 0 | 12.005859 | authority | — |
| Empty dynamic store | miss | 0 | 12.259881 | byte-identical | — |
| Spark-built collection, run 1 | Nomogram hit | 1 | 12.042545 | byte-identical | 5.822 GB |
| Same collection, run 2 | Nomogram hit | 1 | 12.155373 | byte-identical | 5.822 GB |
| Same collection, run 3 | Nomogram hit | 1 | 11.929355 | byte-identical | 5.822 GB |

Every hit had `parent_hits=0`, `prefill_hits=0` and **horizon=1, accepted=1, committed=1**. The first hit's native log recorded `GEMMA4_DPR_TARGET` with zero CPU model phases and GPU PREFILL engagement; this does not by itself establish request-correlated device utilization for every repeat. The response IDs, readable text, position 65, and explicit export's complete payload/state/facts SHA-256 matched DPR-off; every 27,955,200-byte payload was rehashed from its persisted file. Maximum observed hit RSS was **5,822,332,928 bytes (5.822 decimal GB)**. The 27,955,456-byte export is a post-request proof artifact, not request-time memory or DPR-store size. Native first-hit PREFILL was 106.574 ms/new input token and decode averaged 229.700 ms/actual output token under diagnostic logging. The separate DPR-off diagnostic pilot averaged 243.078 ms/output token; these unlike one-shot host conditions are **not** a controlled decode-speed ratio.

Three fresh restarts persisted three measured hits and one valid mutable stat file. The learned ledger recorded three accepted **and three actually committed** tokens, 1,123,551,517 ns lookup+verify cost, and no persistence failures. Its current chart divides that cost by accepted tokens **plus an assumed one bonus per round**, yielding 187.259 ms/token and a green flag; dividing by the **three actually committed tokens** gives **374.517 ms/token**, above the measured 243.078-ms serial reference. The median hit request wall was **12.042545 s**, versus **12.005859 s** for one DPR-off control (0.306% higher). One control and mixed diagnostic/summary hit logging cannot establish a robust speed difference; they certainly do **not** demonstrate a gain. Keep this collection unpromoted. Configured X8 did **not** become an eight-row physical DPR block on the fresh parent: the observed episode was one token. Do not transfer the historical 27-token/CTX5120 speed result to this cell.

## X-only follow-up: configured 2, 3, 4, 6

One pass on **2026-10-01 11:20:08–11:22:19 UTC** used the same V520 binary, normal HTTP prompt, model recipe, CTX8192, W8, NFQ N104/F1/Q4, DPR mode, and resource limits. Each cell had a fresh native process and an exact copy of the **same initial eight edges and mutable stats**; no KV parent was imported. The existing manager now forwards each complete N/F/Q/X tuple through the admitted server CLI as well as its startup environment, and `/v1/models` read back the requested X before inference. Full receipts and reduction: [`x-sweep.json`](x-sweep.json).

| Requested / reported X | Native DPR horizon | Accepted / committed | PREFILL ms/input | Decode ms/output | HTTP wall s | Peak RSS GB |
|---:|---:|---:|---:|---:|---:|---:|
| 2 / 2 | 1 | 1 / 1 | 104.719 | 229.483 | 11.983372 | 5.822 |
| 3 / 3 | 1 | 1 / 1 | 104.403 | 228.959 | 11.946955 | 5.822 |
| 4 / 4 | 1 | 1 / 1 | 103.952 | 228.763 | 11.931841 | 5.822 |
| 6 / 6 | 1 | 1 / 1 | 104.286 | 228.847 | 11.944545 | 5.822 |

All four requests naturally stopped at 36 output tokens, recorded one Nomogram hit and normal GPU PREFILL/DPR-target markers, and exported **byte-identical complete KV/state/facts versus DPR-off**; each persisted payload was rehashed. They are **four correctly configured but physically X1 cells**, not an X2/X3/X4/X6 throughput comparison. The then-live portable policy derived DPR's width from *pre-request* KV rows: zero rows selected X1, and the current prompt's 29 PREFILL rows did not warm that request. This incorrectly coupled the existing DPR draft control to ordinary decode admission; it was corrected in the opt-in follow-up below. The small wall variation in this historical run establishes no X-dependent speed winner.

## Opt-in independent DPR draft width: actual X2/3/4/6

On **2026-10-01**, `SALT_DPR_INDEPENDENT_DRAFT=1` restored DPR's existing draft-width choice independently of ordinary cold X1/warm X4. The existing `SALT_DPR_DRAFT_N`, configured target-row capacity, remaining output seats, linked store edges, and target verifier still bound the proposal. The flag is optional and strictly 0/1; absent/0 preserves the prior behavior. The `rocm` recipe opts in. **DPR-off and DPR-miss ordinary decode still use the unchanged pre-request-KV X gate.** No new kernel, allocation, resource map, or alternate server route was introduced. The model compatibility identity remained `35eeec43…`; this is a different rebuilt executable (`9a60c6b5…`) and build receipt, on remote source HEAD `6c4c8fcb91fe1c94d29e4c2c6cfe0ef886eb8589` (2026-09-29T08:22:59-04:00) plus the guarded uncommitted flag patch. Detailed bounded evidence: [`independent-draft-sweep.json`](independent-draft-sweep.json).

The **same** normal-HTTP AIAYN formula request, eight chained immutable edges, fresh isolated DPR roots, CTX8192/W8/N104/F1/Q4, and no parent import gave:

| Configured X | Native DPR horizon | Accepted / committed | Acceptance | PREFILL ms/input | Decode ms/output | HTTP wall s | Peak RSS GB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 2 / 2 | 100% | 103.608 | 235.183 | 12.045911 | 5.822 |
| 3 | 3 | 3 / 3 | 100% | 103.728 | 229.012 | 11.869445 | 5.822 |
| 4 | 4 | 4 / 4 | 100% | 103.319 | 225.669 | 11.712195 | 5.822 |
| 6 | 6 | 6 / 6 | 100% | 103.364 | 216.814 | 11.386029 | 5.822 |

Every cell had one native GPU DPR target transaction, one submission/fence, zero CPU phases, naturally stopped at 36 tokens, and matched the DPR-off **output IDs and rehashed complete KV/state/facts** at position 65. The mean proposed-edge acceptance was **100% over these four one-shot cells**; it is not a document-wide hit-rate estimate. Peak RSS was **5,822,304,256 bytes (5.822 decimal GB)**. A single matched-build DPR-off X6 control used the same normal request, had zero Nomogram hits, 36 output tokens, byte-identical complete state, **238.759 ms/output** native decode, **11.929124 s** HTTP wall, and **5,821,804,544 bytes (5.822 decimal GB)** peak RSS; X6 with DPR had 216.814 ms/output and 11.386029 s wall. This one pair is **not** a repeat-qualified speedup. The focused portable-C `test-text-verify` and allocation-symbol scan passed on Linux; Mac/Xcode targets were not invoked. The managed listener was stopped after each owned cell and control.

## Longer configured widths: X12/24/36/48

The original natural Spark full-CUDA AIAYN answer and fresh ROCm DPR-off answer match for all **36 output IDs**; the last ID (`106`) is a stop token. The established Spark publisher extended the eight-edge collection into a new **35-edge, 17,920-byte immutable linked Nomogram**, excluding that stop. Zero KV-reference files or ROCm parent KV were copied. The existing ROCm server binary/receipt and CTX8192/W8/N104/F1/Q4 recipe remained fixed. Four isolated roots and ordinary managed HTTP requests yielded [machine-readable receipts](extended-35-sweep.json):

| Configured X | Native horizon | Accepted / committed | Acceptance | Decode ms/output | HTTP wall s | Peak RSS GB |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 12 / 12 | 100% | 192.073 | 10.444749 | 5.822 |
| 24 | 24 | 24 / 24 | 100% | 130.332 | 8.232770 | 5.822 |
| 36 | **35** | 35 / 35 | 100% | 80.575 | 6.212512 | 5.822 |
| 48 | **35** | 35 / 35 | 100% | 80.988 | 6.230767 | 5.822 |

Each native DPR target recorded one submission/fence and zero CPU phases, naturally emitted the same 36-ID formula reply, and matched the DPR-off **rehashed complete KV/state/facts** at position 65. Maximum process RSS was **5,822,357,504 bytes (5.822 decimal GB)**. The one matched-build DPR-off control was **11.929124 s**; these one-pass walls show an observed within-fixture decline through horizon 35 but are **not a repeat-qualified speedup or proof of simultaneous SIMD row execution**. X36 and X48 **did not execute their configured widths**: the source response has only 35 eligible non-stop proposal edges. Acceptance was 100% for this one exact trajectory, not a general or cross-question hit-rate estimate.

Three attempted longer natural AIAYN questions on ROCm reached the O128 cap before a natural stop. For the focused third question, the Spark source-conditioned reply stopped at 58 tokens but disagreed with ROCm at **token 0**; no edge was published for that question. Thus a compatible **full** X36/X48 proposal remains open and must not be inferred from the configured cells. The first new-root preparation failed before inference because its mutable `stats/` directory was absent; the next fresh root failed startup because it was created as mode `0775` instead of the loader-required private `0700`. The successful third root was created at `0700`; failed roots were preserved, not modified or used as evidence.

## Larger output caps: O256 and O512 alignment gate

One new natural AIAYN question asked for the attention equation and a brief explanation of its scaling. Each O cell used fresh, normal managed inference: registered full-CUDA Spark with the authenticated turn-14 document conversation at source position 22,899 (CTX65536/W8), and empty-parent full-GPU ROCm (CTX8192/W8/N104/F1/Q4, configured X48, DPR off). Spark's source answer stopped naturally at **112 tokens** for both caps. ROCm reached the **O256 length cap**, then stopped naturally at **401 tokens** under O512. The ROCm exports were rehashed and its owned listener stopped. [Machine-readable gate](larger-o-gate.json):

| O cap | Spark actual / finish | ROCm actual / finish | Spark wall s | ROCm wall s | ROCm PREFILL ms/input | ROCm decode ms/output | ROCm peak RSS GB | Shared initial IDs |
|---:|---|---|---:|---:|---:|---:|---:|
| 256 | 112 / stop | 256 / length | 73.203 | 84.447 | 80.379 | 311.400 | 5.822 | **0** |
| 512 | 112 / stop | 401 / stop | 72.568 | 137.100 | 83.409 | 330.460 | 5.822 | **0** |

Both cells diverged at **output token 0** (Spark ID `818`, ROCm ID `10354`). The O512 Spark source launch waited until a transient GPU-busy admission cleared; no busy process was stopped to run it. **No Spark document-conditioned DPR artifact or hit** was published from this alignment gate: a longer ROCm answer cannot make an incompatible Spark proposal target-valid. These walls are ordinary DPR-off controls on different source states, not a DPR speed comparison or cross-host state-parity gate. The original 35-edge formula collection remains separate and unchanged.

## O512 high-X mechanics control — actual DPR horizons 48/72/128

The separately approved [width-expansion plan](../../../docs/workorder/wo1001-dpr-horizon-128-plan.md) split the **128-token composed Nomogram walk** from the **64-token per-edge V2 wire limit**. V2 edges remain 512 bytes, V1 stats remain 768 bytes and 65 bins, NFQ checking capacity remains 512, and the same portable C99 target verifier, startup resource tables and ROCm backend execute the requests. The change rotated Gemma compatibility to `a8f65419…`; prior collections were not silently rebound. Focused Linux store/stats/walk/attention, fixed-NFQ text-verifier and projection tests passed, followed by a verified V520 binary/receipt. This is a source/ABI change, **not** just a flag or artifact-length setting.

The new-build **ROCm DPR-off** O512 answer stopped naturally at 401 tokens, matched the earlier build's IDs, and established the complete-state authority at position 450. An ordinary empty-store O1 request with the same prompt supplied its QA key and matching first target token. The existing publisher on Spark compiled the ROCm target-verified natural prefix into **128 linked one-token V2 edges (65,536 immutable bytes; zero KV references)**; every fresh ROCm root was private `0700`. This is explicitly a **target-trace mechanics control, not Spark document-conditioned DPR training**. With dynamic DPR on, normal managed O512 requests gave [machine-readable results](dpr128-mechanics-o512.json):

| Configured X | Native DPR horizon | Accepted / committed | PREFILL ms/input | Decode ms/output | HTTP wall s | Peak RSS GB |
|---:|---:|---:|---:|---:|---:|---:|
| 48 | **48** | 48 / 48 | 82.680 | 294.798 | 123.168621 | 5.822 |
| 72 | **72** | 72 / 72 | 79.970 | 275.837 | 115.479208 | 5.822 |
| 128 | **128** | 128 / 128 | 82.327 | 227.564 | 96.260709 | 5.822 |

Each recorded one DPR hit and one native target transaction with one submission/fence and zero CPU model phases; all **401 output IDs, semantic answer, position 450 and rehashed complete KV/state/facts** matched the same-build DPR-off control (136.994632 s HTTP wall). Maximum peak RSS was **5,822,414,848 bytes (5.822 decimal GB)** under the unchanged 7.5-GB ceiling. The one-pass lower walls are observational, **not a repeat-qualified speedup or proof of simultaneous SIMD row execution**. No cross-platform same-artifact byte-identity, old-artifact cross-load, formal proof renewal, or release promotion is claimed. The collection does not repair the Spark document-conditioned token-0 mismatch.

## O512 high-X mechanics control — actual DPR X256 and X356

The bounded [X256→X356 plan](../../../docs/workorder/wo1001-dpr-horizon-356-plan.md) retained the 64-token/512-byte immutable edge format, fixed NFQ-512 checks, N104/F1/Q4, CTX8192/W8, O512, and the **7.5-decimal-GB actual RSS ceiling**. Two distinct compatibility/build identities were qualified: `9d48c38c…` for X256 and `d7e323dd…` for X356. The first X256 startup was RED before inference: HIP compared 2,048 row×top-k selected jobs to the 1,024-entry *unique-resource/frontier* bound. After separate approval, a narrow HIP admission correction compared selected jobs to the existing 4,096-job ceiling instead. The unique-resource table, startup-owned descriptor storage, tensor address/lease structures, kernels and RSS limit were not changed. The X356 build needed no further backend edit; selected jobs were 2,848. The failed X256 startup and a stale X356 test-width assertion remain preserved as failed preflight receipts, not hidden inference results.

Each new-build ROCm DPR-off normal O512 answer naturally stopped at **401 IDs** and established a fresh same-build complete-state/export control at position 450. A separate ordinary empty-store O1 request established that build's QA key. The established Spark publisher compiled the **ROCm target-verified natural trajectory** into fresh linked one-token stores: **256 edges / 131,072 bytes** and **356 edges / 182,272 bytes**, zero KV references and zero imported parent rows. This is a **mechanics control, not Spark document-conditioned DPR training**. Every edge was hash-read back on Spark, locally, and on ROCm; private mutable stats directories were `0700`. [Machine-readable exact-build/resource receipts](dpr256-356-mechanics-o512.json):

| Configured X | Native DPR horizon | Accepted / committed | DPR-off HTTP wall s | DPR-on HTTP wall s | Off → on decode ms/actual output | DPR-on peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | **256** | 256 / 256 | 136.934925 | 64.676986 | 330.473 → 149.119 | 6.077 GB |
| 356 | **356** | 356 / 356 | 137.148079 | 42.336985 | 330.928 → 93.499 | 6.274 GB |

Each ordinary managed HTTP request recorded one native target transaction, one submission/fence, zero CPU model phases, a naturally stopped 401-ID response, and **identical output IDs, readable content and rehashed complete KV/state/facts versus its own same-build DPR-off control**. Both builds' output IDs and exported payload SHA matched each other, but their complete-state/facts SHA differed with the compatibility rotation; this is **not cross-loading or cross-build state-identity proof**. The X256/X356 startup canonical verifier seats were respectively **503,724,032** and **700,474,032 bytes**; measured peak RSS remained below 7.5 GB, and the owned listeners stopped. The lower one-pass walls are observational, **not a repeat-qualified speedup or evidence of Spark document-conditioned high-X hit rate**. The full engine-config suite remains RED on an unrelated mac-metal fetch-touch assertion; focused ROCm policy, compatibility, C99 DPR/store/stats/walk/attention, and NFQ-fixed text-verifier gates passed. **X512 remains on hold.** The broader KV-alone/DPR-alone/combined document comparison and Thinking Machines positive hit remain open.

## Thinking Machines Lab boundary

Question: “According to the Thinking Machines Lab article on inference determinism, what does batch invariance require when the number of concurrent requests changes? Answer in one sentence.” Both Spark (31 output tokens) and fresh ROCm (36) naturally answered, but their **first output IDs differed** (`2021` versus `10450`). Spark answered in terms of keeping the reduction strategy identical; ROCm answered in terms of identical request output across batch sizes. No target-compatible prefix was available for this empty ROCm parent; **no TML edge was published**. This is a negative result for that one question, not a general semantic verdict on the article.

## Implementation, verification, and remaining gate

- Corrected the ROCm recipe's `SALT_DPR_DRAFT_N=104` to **64**, the portable hard maximum; NFQ N104 and TARGET X remain independent. DPR-off had masked this invalid startup setting. A bounded startup-only error readback in `server/serve.py` exposed the original `EPIPE`/Broken-pipe failure; its physical-only compatibility projection was updated without changing the compatibility identity. The existing host-local development manager now forwards explicit DPR serial/retention startup controls. No kernel, engine state machine, resource pool, alternate tensor address map, learned-weight payload, or ROCm parent import was introduced.
- `tools/test/gemma4-kv-compat-projection-test.py` passed for the earlier startup correction; Python syntax, Bash syntax, and `git diff --check` passed. Real normal-server DPR-off, empty-store, three attached fresh-process requests, the four isolated pre-flag X-configuration cells, and four opt-in **native multi-token** DPR cells passed their stated narrow exactness gates. The focused Linux `test-text-verify` and allocation-symbol scan passed. The full repository suite was **not** run; Mac/Xcode build targets are excluded by user instruction.
- Remaining: TML/AIAYN Spark document-conditioned candidate compatibility, variable acceptance across independent prompts, cross-platform same-artifact state identity, and repeated matched-build controls for speed require separate evidence. The learned chart's assumed-bonus denominator is not a measured whole-token speedup; any core chart-policy correction requires its own file-scoped authorization and gate. The initial revival was uncommitted at measurement time; the later X256/X356 source and evidence have a separate checkpoint/PR boundary. No release promotion is claimed.

Private, bounded receipts and the compiled edge collection remain under `logs/experiments/aiayn-dpr-revival-v1/` on the named hosts; the checked-in report contains no KV payload. Source, request, transfer, hit, export, stat, and negative TML receipts are indexed by `logs/experiments/{185,187,193,200,208,209,210,211,212,213,214,215}-*.log`.
