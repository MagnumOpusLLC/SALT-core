# Performance ledger

Measured performance, grouped by the **source commit actually built and tested**.
Append a new commit section for each measured revision; retain earlier results
and failed controls. Do not assign old measurements to a documentation-only
commit or silently combine different execution modes, fixtures, or run rounds.

## Columns and measurement rules

- **Platform:** physical host/platform, not merely the recipe name. Acerbox is a DGX Spark host; its CPU lane uses the `ec2` CPU recipe.
- **Mode:** actual CPU, GPU, or mixed realization and heavy decode route. `B1` means ordinary one-token computation, not X-TARGET configured to one row.
- **Workers W:** requested worker budget; not a measurement of active cores or GPU occupancy.
- **Expert arena/cap GB:** configured `expert_budget_gb` from each run's saved effective model configuration. This bounds the expert cache/arena; it is not measured resident weight, and bounded-mmap modes need not allocate that many bytes up front.
- **KV cache seat/budget GB:** actual fixed live-KV capacity from `kv_layout.capacity_bytes` (decimal GB) / configured `kv_budget_gb`. This is not a per-request export size or an expert-cache allocation.
- **Peak RSS bytes (GB):** maximum of sampled process RSS and Linux HWM over the row's completed repeats, in exact bytes and decimal GB. File-backed resident pages count; RSS is not device VRAM or the sum of several processes' host use.
- **RSS limit GB:** configured all-inclusive host-RSS ceiling from the saved effective model configuration, not the observed peak.
- **NFQ:** configured `N/F/Q`. Admitted cheap-check capacity is `N × F × Q`; this is not the number of checks actually executed. NFQ selection and heavy model computation are separate.
- **X:** executed X-TARGET status, with the configured maximum in parentheses. `off (48)` means no X-TARGET transactions, despite configured X48. Ordinary B1 is shown in Mode rather than mislabeled as X1 TARGET.
- **P:** actual prompt/input tokens.
- **O:** requested output cap / actual emitted completion tokens. Natural stopping below the cap is valid.
- **Prefill ms/tk:** native PREFILL milliseconds per input token.
- **Decode ms/tk:** native reported decode milliseconds per actual completion token. Preserve the source metric's denominator; do not relabel it as a different post-first-step measurement.
- **Throughput tk/s:** **(actual input tokens + actual output tokens) / total HTTP request wall in seconds**. The timed request includes PREFILL, decode, request-scoped proof, and HTTP/control overhead; separately performed server startup is excluded. Use actual output count, not the output cap. This is combined input/output throughput, not pure decode speed. Higher is better.
- **n:** completed repeats used in the row. Rates are arithmetic means of the per-request normalized measurements, not a pooled ratio.
- **Verdict:** observed exactness for this cell. A failed cell's timings are retained as rejected evidence, not a qualified performance result. PASS does not imply universal platform certification or full semantic completeness of a capped answer.

Use `—` for missing measurements; never estimate them from another backend,
commit, candidate source, or workload. Keep ordinary HTTP measurements separate
from fixed-candidate TARGET controls and instrumented diagnostic timings.

## `03eaa01b9de7285321d1031838f2a224f199dddd` — LFM2.5 CUDA P4096/O100 checkpoint

**Commit timestamp:** `2026-10-16T21:43:35-04:00`.
**Run:** `2026-10-17T02:13:06Z` on Acerbox DGX Spark / NVIDIA GB10
cc=12.1, CUDA pageable mmap, authenticated package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`,
B32, W8, 1,504 bounded demand-retention slots, CTX4196, native greedy,
five fresh processes. The exact synthetic P4096 fixture has SHA-256
`3370b866c3290235229b768296bbc6734adff19d0378c9d0b9c389698dd02e80`.
A test-local 16.0-decimal-GB RSS cutoff admits the long-context working set;
the checked-in 10-GB P480 policy is unchanged.

| Platform | Mode | B | W | Expert slots | RSS limit | P | O | n | PREFILL ms/tk | Decode ms/tk | Decode tok/s | Throughput tk/s | Peak RSS | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Acerbox DGX Spark | CUDA pageable-mmap, bounded demand retention | 32 | 8 | 1,504 | 16.0 GB | 4,096 | 100 | 5 | 10.771189 | 45.394559 | 22.029072 | 86.235304 | 11,277,672,448 B (11.278 GB) | PASS: identical IDs/text/scalar state/KV/recurrent state/logits |

Decode spans `45.286718–45.523488 ms/token` (0.523% spread); PREFILL
spans `10.726112–10.842090 ms/input-token` (1.081% spread); request wall
spans `48.468843–48.954640 s`. Every request emits 7,700 backend graph
launches, 155,018 physical kernel nodes, and 93,818 host launch calls. Compared
with the source-identical P480/O8 bounded cell (`34.646442 ms/token`), P4096
decode is 31.02% slower. This makes P480/O8 an exactness/regression fixture,
not an external decode-performance denominator.

Complete state is byte-identical across all five runs: scalar-state SHA-256
`c18b0f7b713a653e8de89df3aa311435febdf7ce8d7e84b0bd1b0669337bba2b`,
KV SHA-256 `229888c977ba71d2251602204fd033ba6f1453ce09033fae4f8de74e1b861f87`,
recurrent/model-state SHA-256
`0d843179c5a86265bee3738726a0570ffae4691363d367af4cf1d213e875f463`,
and logits SHA-256
`f0d9b0440dd4a2fea3dadd31cf2380bac3e3acaf3e37659538962ef4f8670d69`.
Tracked evidence and the exact prompt are in
`experiments/reports/lfm2-cuda-p4096-o100/` (private reference; not exported).

## `03952c0c9319705e77c8bec562d705f1fed82957` — LFM2.5 router GEMV and bounded extent graphs

**Commit timestamp:** `2026-10-16T19:53:01-04:00`.
**Workload:** the same authenticated P480/O8, B32, W8 Acerbox DGX Spark
bounded-CUDA cell as the preceding LFM section, with 256 zero-copy expert slots
and three fresh native processes.

| Mode | n | PREFILL ms/tk | Decode ms/tk | Decode tok/s | Request throughput tk/s | Peak RSS | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Prior selected bounded CUDA | 3 | 7.330 | 46.311 | 21.593 | 125.514 | 9.495 GB | PASS, superseded |
| F32 router tile only | 3 | 7.204 | 43.104 | 23.199 | 128.402 | 9.495 GB | PASS, intermediate |
| F32 router tile + partial extent graphs | 3 | 6.965 | 41.435 | 24.134 | 132.810 | 9,753,657,344 B (9.754 GB) | PASS, selected |
| Same-build full-resident pageable control | 3 | 6.678 | 35.501 | 28.169 | 140.042 | 13.928 GB | PASS diagnostic control; not selected |

The selected bounded result is 10.5% below the preceding 46.311 ms checkpoint
and 25.2% below the original matched 55.385 ms control. Full residency still
buys only 5.935 ms/token while violating the bounded product envelope, so it
remains a diagnostic control.

The model-owned exact F32 router kernel forms 64 products concurrently and has
thread 0 consume them in the original ascending column order. Wider tiles 128
and 256 and a shuffle-based warp form were exact but slower. CUDA bounded mode
now opts into the engine's existing extent graphs through the generic
`SALT_TEXT_MODEL_GPU_EXTENT_GRAPHS` capability bit. Capture is partial:
uncapturable extents retain the established cell loop rather than failing model
startup. The final request emits stable existing backend counters in all three
repeats: `backend_graph_launches=616`, `backend_kernel_nodes=15602`, and
`backend_host_launch_calls=10706`. A zero-node extent-elision candidate was
exact but slower (`42.203 ms/token`) and was removed.

Registered standalone evidence now uses four distinct authenticated experts per
layer rather than one synthetic payload. The complete 38-layer selected-expert
sweep reads `806,879,232` active payload bytes and measures about 5.57 ms warm;
the complete B1 convolution block measures about 0.16 ms/layer. Both F32
projection and three-launch expert-chain callbacks pass direct CUDA graph
capture/instantiate/launch checks. Exact P480 attention remains only
0.0737 ms per attention layer.

Evidence:

- `logs/lfm2-dgx-cuda-bounded-p480/2b33cdf9b5e53303690a39b9ad0730026a24cf22/20261016T235219Z/`;
- `logs/lfm2-dgx-cuda-full-resident-p480/2b33cdf9b5e53303690a39b9ad0730026a24cf22/20261016T233423Z-pageable1/`;
- `logs/lfm-attention/20261016T233406Z/`;
- rejected zero-node control:
  `logs/lfm2-dgx-cuda-bounded-p480/2b33cdf9b5e53303690a39b9ad0730026a24cf22/20261016T234611Z/`.

The strict model gate, Mac Metal build and exact request, CUDA direct parity,
real expert sweep, and callback graph-capture probes pass. The complete state
hashes remain those recorded in the preceding section.

## `cb65e955003dfa827e27942d156f82c51bdd0229` — LFM2.5 exact GPU decode

**Commit timestamp:** `2026-10-16T14:07:42-04:00`.
**Workload:** authenticated package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`,
P480/O8, W8, B32 unless noted, three fresh native processes per cell. CUDA
uses Acerbox DGX Spark / NVIDIA GB10 cc=12.1, pageable HMM, and the bounded
256-slot zero-copy expert path. Metal uses the local Apple M4 Pro. All retained
cells produced IDs `6515,1164,39246,53560,5175,31057,1657,26403`, text
`Stateful inference preserves exact computation while bounded`, scalar-state
SHA-256 `a56e38e2dfbd10c4c5ce88a8dcc66f634725188f948cabca24e28547aac02269`,
KV SHA-256 `7cffb0db92d2daf7224d568ce5a309ed9722e43d2bf0b42b72492ff4f3663fad`,
recurrent-state SHA-256
`ab2a07e59e732c45b6e48a550084f5f6a51c0026c31d45c3040a9f534a6e0792`,
and logits SHA-256
`4590c72eb0ecb9ba761d901f3ecd993a8347ec9e136c8e7653b259ab9c09b4f8`.

| Platform | Mode | B | n | PREFILL ms/tk | Decode ms/tk | Throughput tk/s | Peak RSS | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DGX Spark | CUDA bounded, prior matched control | 32 | 3 | 8.465 | 55.385 | — | — | PASS, superseded |
| DGX Spark | CUDA bounded, exact model chain/router/Q6 | 32 | 3 | 7.330 | 46.311 | 125.514 | 9,494,855,680 B (9.495 GB) | PASS, selected |
| Apple M4 Pro | Metal, prior matched control | 32 | 3 | 31.986 | 165.774 | — | — | PASS, superseded |
| Apple M4 Pro | Metal, parallel router/grouped Q6 metadata | 32 | 3 | 30.668 | 122.279 | 31.088 | 1,311,621,120 B (1.312 GB) | PASS, selected |
| DGX Spark | CUDA bounded B512 qualification | 512 | 3 | 8.075 | 48.578 | 114.451 | 9,960,222,720 B (9.960 GB) | PASS; B512 supported, not selected |
| DGX Spark | CUDA full-resident pageable control | 32 | 3 | 6.677 | 39.440 | 138.628 | 13,928,009,728 B (13.928 GB) | PASS diagnostic control; not selected |

The selected bounded CUDA decode is 16.4% lower than the matched 55.385 ms
control; Metal is 26.2% lower than its 165.774 ms control. The full-resident
control leaves 39.440 ms/token and consumes 13.928 GB RSS, so residency alone
is not the answer and remains outside the selected bounded product path. B512
is exact and improves the DGX CPU PREFILL control, but is 10.2% slower than B32
on CUDA for P480; the model default is therefore B32 while the backend retains
512-row capacity.

Retained CUDA changes are the generic model expert-chain seam, model-owned
three-launch exact routed chain, one-time parallel sigmoid scores, and exact
64-product Q6 tile. Retained Metal changes are parallel sigmoid scores and
64-value Q6 metadata reuse; the older tiled-attention and warp controls remain
default-off. Rejected candidates include direct-source/groupless routing,
shared-memory router fusion, Metal encoder-boundary substitution, and the
Metal 64-product Q6 tile; each was exact but neutral or slower end to end.

Evidence:

- `logs/lfm2-dgx-cuda-bounded-p480/aa3b2ef1ba9ed1a14566240781d20cfdd9cd11eb/20261016T175844Z/`;
- `logs/lfm2-current-mac-p480/20261016T174515Z-b32/`;
- `logs/lfm2-current-dgx-p480/aa3b2ef1ba9ed1a14566240781d20cfdd9cd11eb/20261016T175916Z-b512/`;
- `logs/lfm2-dgx-cuda-full-resident-p480/aa3b2ef1ba9ed1a14566240781d20cfdd9cd11eb/20261016T175406Z-pageable1/`.

`CUDA_ONLY=1 bash tools/lfm-gpu-parity/run.sh` passes exact Q4/Q6,
activation, router, mapped/device expert-chain, convolution, and state cells.
`models/lfm2-24b-a2b/test.sh` passes. Root `make test` still fails only on the
pre-existing Gemma live-source compatibility pin and independent mac-metal
`SALT_FETCH_TOUCH_BYTES` recipe assertion after the generic GPU gates pass.

## `6b298e34b92310b531123388bd4af9e274f5d5a0` — LFM2.5 exact tiled attention

**Commit timestamp:** `2026-10-15T22:31:04-04:00`.
**Run:** `2026-10-16T02:15:53Z` through `2026-10-16T02:22:43Z`;
authenticated package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`,
Acerbox DGX Spark / NVIDIA GB10 cc=12.1, CUDA pageable HMM, B32, W8.
Design checkpoint: `99d90da22b0277a71fc2d1c361c3db8f0d37bc0a` at
`2026-10-15T21:53:52-04:00`. Qualified CUDA binary SHA-256:
`16584f5d7378069175fac326b0d2a51142c4df098a3fed53d99c5341f2bc2ad0`.

The model-owned realization uses a startup-compiled compute I/O mapping ledger:
every parallel score/product/output job has a fixed owner and canonical
address. Exact products are generated in 64-position shared-memory tiles, then
each output column consumes them in the original ascending-position `fadd_rn`
order. There are no partial-sum reductions, runtime allocations, peer address
maps, KV copies, host attention readbacks, or intermediate commits.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Acerbox DGX Spark | CUDA pageable-HMM, bounded 256-slot demand cache, exact tiled attention | 8 | 1.359 | 0.084 / — | 9,492,828,160 (9.493) | 10.0 | — | off | 480 | 8 / 8 | 8.247 | 56.232 | 110.705 | 3 | PASS: exact IDs/text/complete state across repeats |
| Acerbox DGX Spark | CUDA pageable-HMM, all 2,432 experts resident, exact tiled attention | 8 | 12.910 | 0.671 / — | 14,514,249,728 (14.514) | 20.0 | — | off | 12,478 | 192 / 13 | 22.907 | 76.299 | 43.549 | 1 | PASS: exact accepted-baseline IDs/text/state/KV/recurrent/logits |
| Acerbox DGX Spark | CUDA pageable-HMM, all 2,432 experts resident, exact tiled attention | 8 | 12.910 | 0.671 / — | 14,514,274,304 (14.514) | 20.0 | — | off | 12,310 | 192 / 192 | 23.433 | 83.015 | 41.071 | 1 | PASS: full cap and exact accepted-baseline IDs/state/KV/recurrent/logits |

Against the accepted full-resident P12,478/O13 baseline, PREFILL falls from
31.389 to 22.907 ms/input-token (27.0%), decode from 124.381 to 76.299
ms/output-token (38.7%), request wall from 393.285 to 286.828 seconds (27.1%),
and combined throughput rises from 31.761 to 43.549 tk/s (37.1%). At
P12,310/O192, PREFILL falls from 31.091 to 23.433 (24.6%), decode from 133.886
to 83.015 (38.0%), wall from 408.436 to 304.399 seconds (25.5%), and throughput
rises from 30.609 to 41.071 tk/s (34.2%). RSS and device allocation remain in
the accepted bands.

The standalone registered probe passed 15/15 byte-identical cells at positions
1, 480, 12,310, 25,344, and 131,071 with row shapes 1, 8, and 32. At P12,310
B1 one attention layer falls from 6.637 to 2.518 ms (2.64x); at P131,071 it
falls from 77.007 to 29.590 ms (2.60x). Evidence:

- `logs/lfm-attention/20261016T020924Z/` — rejected exact GQA-sharing control;
- `logs/lfm-attention/20261016T021232Z/` — exact score-split control;
- `logs/lfm-attention/20261016T021355Z/` — accepted exact product-tile gate;
- `logs/lfm2-cuda-long-candidate/99d90da22b0277a71fc2d1c361c3db8f0d37bc0a/20261016T021553Z/`;
- `logs/lfm2-cuda-pressure/16584f5d7378069175fac326b0d2a51142c4df098a3fed53d99c5341f2bc2ad0/20261016T021642Z-p12000/`;
- `logs/lfm2-cuda-pressure/16584f5d7378069175fac326b0d2a51142c4df098a3fed53d99c5341f2bc2ad0/20261016T022243Z-p12000-aiayn/`.

Mac CPU, Mac Metal, DGX CPU, and DGX CUDA produce IDs
`278,3270,302,4741,20,643,355,278`, scalar-state SHA-256
`69ed522e5058e00682cbc6d252964176a120f69db3d2a6346c1e44776e1599af`,
KV SHA-256 `bf3d5474ee4673e912cc97ec5185dc4f75512a2a55f34a250bfa03f78362e498`,
recurrent-state SHA-256
`d9a2810e9201e91c78e7b040bedf5e3a4e893a7f259e26f8b56ef8c3520654c0`,
and logits SHA-256
`608cfbe6e8bf86ce4fb5a7a9b1c9bc71e00080f2f48625d776d74f2413c2c7bf`.
The canonical state gate reports `GATE_DONE ok=631 fail=0`. `make test` reaches
and passes generic GPU resource/trunk gates, then remains RED on the pre-existing
Gemma live-source compatibility mismatch and the independent mac-metal
`SALT_FETCH_TOUCH_BYTES` recipe assertion.

## `16d57c7a018644e32bdad03aa43fc501d656b93f` — LFM2.5 chunked long PREFILL

**Commit timestamp:** `2026-10-15T08:29:31-04:00`.
**Run:** `2026-10-15T12:30:30Z`; authenticated package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`,
tracked `qualification/prefill-long.prompt`, P480/O8, CTX2048, B32, W8,
256 expert slots, Apple M4 Pro and Acerbox/DGX Spark. The model runtime commits
15 B32 PREFILL chunks into one live state and publishes logits only from the
final chunk. Each repeat used a fresh process; OS page-cache warmth was
uncontrolled.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac M4 Pro | CPU native, chunked B32 | 8 | 1.359 | 0.084 / — | 3,136,372,736 (3.136) | 6.0 | — | off | 480 | 8 / 8 | 52.408 | 107.455 | 18.849 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Mac M4 Pro | Metal GPU native, chunked B32 | 8 | 1.359 | 0.084 / — | 961,363,968 (0.961) | 6.0 | — | off | 480 | 8 / 8 | 30.702 | 158.282 | 30.497 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | Arm CPU native, chunked B32 | 8 | 1.359 | 0.084 / — | 2,101,178,368 (2.101) | 6.0 | — | off | 480 | 8 / 8 | 40.582 | 178.768 | 23.473 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | CUDA pageable-HMM full GPU, chunked B32 | 8 | 1.359 | 0.084 / — | 9,491,922,944 (9.492) | 6.0 native / 10.0 external | — | off | 480 | 8 / 8 | 13.708 | 49.206 | 69.980 | 3 | PASS exactness; resource PASS at 10.0 GB, RED at 6.0 GB |

Repeat spread (`(max-min)/min`): Mac CPU PREFILL 17.644%, decode 5.596%;
Mac Metal PREFILL 2.247%, decode 3.268%; DGX CPU PREFILL 15.470%, decode
117.686%; DGX CUDA PREFILL 1.346%, decode 1.197%. The DGX CPU decode mean is
therefore observational; both accelerator PREFILL lanes are narrow.

All 12 requests produced IDs `6515,1164,39246,53560,5175,31057,1657,26403`,
text `Stateful inference preserves exact computation while bounded`, scalar
state SHA-256
`a56e38e2dfbd10c4c5ce88a8dcc66f634725188f948cabca24e28547aac02269`,
committed KV `19,988,480` bytes / SHA-256
`7cffb0db92d2daf7224d568ce5a309ed9722e43d2bf0b42b72492ff4f3663fad`,
recurrent state `737,280` bytes / SHA-256
`ab2a07e59e732c45b6e48a550084f5f6a51c0026c31d45c3040a9f534a6e0792`,
and logits SHA-256
`4590c72eb0ecb9ba761d901f3ecd993a8347ec9e136c8e7653b259ab9c09b4f8`.

A separate direct-process RSS safety pass used one fresh request per lane with a
kill-on-limit monitor. Mac CPU peaked at `2,207,858,688` bytes, Mac Metal at
`1,149,370,368`, and DGX CPU at `2,102,857,728`, each below 6.0 decimal GB.
CUDA peaked at `9,493,131,264` bytes below its explicit 10.0 decimal-GB HMM
envelope. A preserved P120/B32 CUDA control under a 6.0 decimal-GB direct-RSS
cutoff was rejected at `7,222,640,640` bytes; a cgroup-only pass was not accepted
because shared file pages escaped that cgroup's accounting. This is a resource
RED for the 6.0-GB CUDA-HMM cell, not an arithmetic RED.

Build identities: Mac CPU
`d2d52e867af4f92098b49ac03b261fa6c9974f25454701163fc590ec997f5161`,
Mac Metal `3f4d3a5e2cbd39e9cd6b9fe5cb3426975a285879ec184de8afe6fd54845094c0`,
DGX CPU `b8efebeca86d0cba692a5cb3dbc97b0cc93865c1a499683c7e170afaf6d5cbd1`,
and DGX CUDA
`769755b5c5ae98ab938a033fc2ab9fa4bfa32a06093c88dbc64c553e47db34e0`.
Evidence roots:
`logs/lfm2-long-prefill/16d57c7a018644e32bdad03aa43fc501d656b93f/20261015T123030Z/`
and
`logs/lfm2-long-safety/16d57c7a018644e32bdad03aa43fc501d656b93f/20261015T124048Z/`.

## `81cf5043df81d3ad1c367b348326b1ce35f4c5b9` — LFM2.5 model-owned accelerator restoration

**Commit timestamp:** `2026-10-15T08:18:19-04:00`.
**Run:** `2026-10-15T12:18:25Z`; authenticated package above, P5/O8,
CTX2048, B32, W8, 256 expert slots, 6.0 decimal-GB native limit, fresh process
per repeat. This is the semantic/decode cell, not long-PREFILL evidence.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac M4 Pro | CPU native, persistent pool + wide Q4 NEON | 8 | 1.359 | 0.084 / — | 2,930,606,080 (2.931) | 6.0 | — | off | 5 | 8 / 8 | 172.169 | 145.647 | 6.508 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Mac M4 Pro | Metal GPU, model-owned kernels | 8 | 1.359 | 0.084 / — | 2,036,924,416 (2.037) | 6.0 | — | off | 5 | 8 / 8 | 137.279 | 151.395 | 6.918 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | Arm CPU native, persistent pool + wide Q4 NEON | 8 | 1.359 | 0.084 / — | 2,056,347,648 (2.056) | 6.0 | — | off | 5 | 8 / 8 | 109.380 | 202.381 | 6.219 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | CUDA pageable-HMM, model-owned Q6/routing kernels | 8 | 1.359 | 0.084 / — | 5,300,498,432 (5.300) | 6.0 | — | off | 5 | 8 / 8 | 68.365 | 50.525 | 17.452 | 3 | PASS: IDs, text, logits, KV, recurrent state |

The corrected tree moves Q6, SiLU, biased-sigmoid routing, routed-map/gather,
and short-convolution implementations under `models/lfm2-24b-a2b/`; shared
backends carry only generic descriptor/callback dispatch. The CUDA Q6 warp and
parallel routed gather preserve the CPU expression tree and restore decode from
the rejected serial model-owned control's `160.980` to `50.525 ms/token`.

All 12 requests produced IDs `278,3270,302,4741,20,643,355,278`, text
` the city of Paris, which is the`, scalar-state SHA-256
`69ed522e5058e00682cbc6d252964176a120f69db3d2a6346c1e44776e1599af`,
committed KV `532,480` bytes / SHA-256
`bf3d5474ee4673e912cc97ec5185dc4f75512a2a55f34a250bfa03f78362e498`,
recurrent state `737,280` bytes / SHA-256
`d9a2810e9201e91c78e7b040bedf5e3a4e893a7f259e26f8b56ef8c3520654c0`,
and logits SHA-256
`608cfbe6e8bf86ce4fb5a7a9b1c9bc71e00080f2f48625d776d74f2413c2c7bf`.

Build identities: Mac CPU
`8ce4b144644102f8db46661cd44046cacb622972f971de728c3875bb88f92101`,
Mac Metal `9daf5f8a21385e6d008744d4b5ce3cca52f5408b94ef74399386e19419f781c5`,
DGX CPU `71769cfd4ea7406f51f8a59ea6467f374d7fb78f7fb27955a559309509b8f0b1`,
and DGX CUDA
`3882d5a26f86a3e2712f415895cb67a0af7100b02c05aa877f835f0dfc861a7c`.
Evidence root:
`logs/lfm2-performance/81cf5043df81d3ad1c367b348326b1ce35f4c5b9/20261015T121825Z/`.
The strict model gate passed, the canonical state gate reported
`GATE_DONE ok=631 fail=0`, and the repository-wide gate remains independently
RED on the pre-existing Gemma compatibility pin and `SALT_FETCH_TOUCH_BYTES`
recipe assertion.

## `4408ffbb376deeafb7132da5fd9aaeeebb8aff5d` — LFM2.5 wide-Q4 SIMD completion

**Commit timestamp:** `2026-10-14T23:17:28-04:00`.
**Run:** `2026-10-15T03:17:37Z`; same authenticated model/package,
P5/O8 fixture, CTX2048, B32, W8, 256 expert slots, 6.0 decimal-GB RSS
limit, Apple M4 Pro, and Acerbox/DGX Spark as `bb5f7bd...` and
`6f161615...`. Each repeat used a fresh native process; OS page-cache warmth
was uncontrolled.

This revision completes the remaining CPU arithmetic realization: affine-Q4
rows wider than the 128-group metadata scratch stream BF16 scale/bias groups
through the existing NEON/AVX2 32-accumulator graph instead of falling back to
scalar per-row arithmetic. Wide rows retain the canonical non-fused multiply,
add, multiply, accumulator-add order; smaller qualified rows keep their
existing SIMD path unchanged.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac M4 Pro | CPU native, persistent worker pool + wide Q4 NEON | 8 | 1.359 | 0.084 / — | 3,164,143,616 (3.164) | 6.0 | — | off | 5 | 8 / 8 | 170.875 | 111.574 | 8.537 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Mac M4 Pro | Metal GPU native | 8 | 1.359 | 0.084 / — | 1,955,987,456 (1.956) | 6.0 | — | off | 5 | 8 / 8 | 141.800 | 140.028 | 7.250 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | Arm CPU native, persistent worker pool + wide Q4 NEON | 8 | 1.359 | 0.084 / — | 2,056,216,576 (2.056) | 6.0 | — | off | 5 | 8 / 8 | 104.781 | 179.954 | 7.176 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | CUDA pageable-HMM full GPU native | 8 | 1.359 | 0.084 / — | 5,300,543,488 (5.301) | 6.0 | — | off | 5 | 8 / 8 | 60.859 | 49.411 | 18.585 | 3 | PASS: IDs, text, logits, KV, recurrent state |

Against the original `6f161615...` baseline, Mac CPU PREFILL is 2.15x
faster, decode is 4.21x faster, and combined throughput is 3.60x higher; DGX
CPU is 2.84x, 3.17x, and 3.34x faster respectively. Against the immediately
preceding `bb5f7bd...` scheduler revision, wide-Q4 SIMD improves Mac CPU decode
by 1.10x and combined throughput by 1.15x. CUDA remains the DGX leader at
1.72x CPU PREFILL, 3.64x CPU decode, and 2.59x CPU combined throughput.
CUDA sampled peak device memory remains 360,710,144 bytes (0.361 GB).

Repeat spread (`(max-min)/min`): Mac CPU PREFILL 308.788%, decode 48.840%;
Mac Metal PREFILL 72.353%, decode 27.890%; DGX CPU PREFILL 69.561%, decode
89.099%; DGX CUDA PREFILL 8.388%, decode 2.758%. The arithmetic means follow
this ledger's rule; the wide CPU/PREFILL spread is preserved rather than
presented as a stable narrow band.

All 12 requests produced IDs `278,3270,302,4741,20,643,355,278`, text
` the city of Paris, which is the`, scalar-state SHA-256
`69ed522e5058e00682cbc6d252964176a120f69db3d2a6346c1e44776e1599af`,
committed KV `532,480` bytes / SHA-256
`bf3d5474ee4673e912cc97ec5185dc4f75512a2a55f34a250bfa03f78362e498`,
recurrent state `737,280` bytes / SHA-256
`d9a2810e9201e91c78e7b040bedf5e3a4e893a7f259e26f8b56ef8c3520654c0`,
and logits SHA-256
`608cfbe6e8bf86ce4fb5a7a9b1c9bc71e00080f2f48625d776d74f2413c2c7bf`.

Build identities: Mac CPU
`c6e3e126676dc2c2159d826b8b6a56f1d3b599a03b00bae48a3a7d73249416ed`,
Mac Metal `eca523b1b5d9270f3b69bf17559feb9e808049bdb2c0d6302e7d6325039baf73`,
DGX CPU `37a43c5517369f3daff63bd260fc825f013615e0a08c719c39fa761b53c1e62b`,
and DGX CUDA
`b416e56e9256f3a34d945b1f514ce3ab87b0a21c24eaa2d9c1a862e31fe65944`.
Evidence root:
`logs/lfm2-performance/4408ffbb376deeafb7132da5fd9aaeeebb8aff5d/20261015T031737Z/`.
The strict model gate and all native builds passed; the canonical-state gate
reported `GATE_DONE ok=631 fail=0`.

## `bb5f7bd5bb0f17cc6b3c1f0afe1e1859e23d20fa` — LFM2.5 amortized CPU execution

**Commit timestamp:** `2026-10-14T23:06:32-04:00`.
**Run:** `2026-10-15T03:06:49Z`; the same authenticated package, hardware,
model geometry, text-only P5/O8 fixture, CTX2048, B32, W8, 256 expert slots,
and 6.0 decimal-GB RSS limit as the `6f161615...` baseline below. Every
repeat used a fresh native process. OS page-cache warmth remained uncontrolled;
CPU and Metal PREFILL and request-wall spreads therefore remain large.

This revision removes the CPU serial executor, runs disjoint output rows through
the existing persistent eight-worker pool, acquires each layer's selected
experts once for the complete gate/up/down job set, batches convolution input
and output projections across active rows, parallelizes wide exact Q4 fallback
rows, and uses model-owned affine-Q6 SIMD for the tied 128,000-row head. CUDA
page readiness is partitioned across that same startup pool and an already-paid
completion fence is consumed rather than repeated. No model arithmetic moved
into portable core; the only core Q4 change exposes the existing exact scalar
row realization to the generic row-range seam when a row exceeds SIMD scratch.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac M4 Pro | CPU native, persistent worker pool | 8 | 1.359 | 0.084 / — | 3,395,682,304 (3.396) | 6.0 | — | off | 5 | 8 / 8 | 168.202 | 122.217 | 7.451 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Mac M4 Pro | Metal GPU native | 8 | 1.359 | 0.084 / — | 2,237,366,272 (2.237) | 6.0 | — | off | 5 | 8 / 8 | 147.773 | 146.554 | 6.989 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | Arm CPU native, persistent worker pool | 8 | 1.359 | 0.084 / — | 2,056,286,208 (2.056) | 6.0 | — | off | 5 | 8 / 8 | 102.066 | 181.187 | 6.984 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | CUDA pageable-HMM full GPU native | 8 | 1.359 | 0.084 / — | 5,300,080,640 (5.300) | 6.0 | — | off | 5 | 8 / 8 | 61.094 | 49.301 | 18.592 | 3 | PASS: IDs, text, logits, KV, recurrent state |

Against `6f161615...`, Mac CPU PREFILL is 2.19x faster, decode is 3.84x
faster, and combined throughput is 3.15x higher; DGX CPU is 2.92x, 3.15x,
and 3.25x faster respectively. Mac CPU now leads Metal by 1.20x in decode and
1.07x in combined throughput for this small fixture, while Metal retains a
1.14x PREFILL lead. CUDA leads DGX CPU by 1.67x in PREFILL, 3.68x in decode,
and 2.66x in combined throughput. CUDA sampled peak device memory remained
360,710,144 bytes (0.361 GB).

Repeat spread (`(max-min)/min`): Mac CPU PREFILL 158.146%, decode 23.122%;
Mac Metal PREFILL 104.733%, decode 23.736%; DGX CPU PREFILL 84.663%, decode
107.144%; DGX CUDA PREFILL 12.895%, decode 3.659%. These wide CPU spreads are
reported rather than hidden; the arithmetic means follow this ledger's rule.

All 12 requests produced IDs `278,3270,302,4741,20,643,355,278`, text
` the city of Paris, which is the`, and the same complete accepted boundary as
the baseline: scalar state
`69ed522e5058e00682cbc6d252964176a120f69db3d2a6346c1e44776e1599af`,
committed KV `532,480` bytes / SHA-256
`bf3d5474ee4673e912cc97ec5185dc4f75512a2a55f34a250bfa03f78362e498`,
recurrent state `737,280` bytes / SHA-256
`d9a2810e9201e91c78e7b040bedf5e3a4e893a7f259e26f8b56ef8c3520654c0`,
and logits
`608cfbe6e8bf86ce4fb5a7a9b1c9bc71e00080f2f48625d776d74f2413c2c7bf`.
An additional fresh pre-matrix CUDA process also produced this complete state.

Build identities: Mac CPU
`d679d2f661c19b31c65da7123c1835673bd9389ce8e8061d41199c7ac52d173a`,
Mac Metal `ddf4af62c18d933e949d14e7b0440ee6b8fd5942897ec2864c0fa6fb8d0ea596`,
DGX CPU `6795e52e5da407b96a6748e9cf368ce3cc9be7fe45931a846d3faa01ba771bff`,
and DGX CUDA
`0f40e035e7f5d2cfcab89d97467ef9d09c6b28fb89cab47432728be336b922a7`.
Evidence root:
`logs/lfm2-performance/bb5f7bd5bb0f17cc6b3c1f0afe1e1859e23d20fa/20261015T030649Z/`.
The strict model gate and all native builds passed; the canonical-state gate
reported `GATE_DONE ok=631 fail=0`.

## `6f16161565185545a7dc06ea2b5b0d963dcba7b9` — LFM2.5 CPU, Metal, and CUDA qualification

**Commit timestamp:** `2026-10-14T21:07:21-04:00`.
**Run:** `2026-10-15T01:07:29Z`; fresh authenticated native process per
repeat on an Apple M4 Pro Mac and Acerbox/DGX Spark (`NVIDIA GB10`, compute
capability 12.1). **Model:** LiquidAI LFM2.5-24B-A2B, source revision
`667408d494bde191c2cb9dd4f4549827816b9275`, package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`
(`salt-lfm25-affine4-6-f32-conv-v1`). The checkpoint is text-only; there is no
vision lane.

The fixture is the five-token prompt `The capital of France is`, greedy output
cap 8, CTX2048, B32, W8, 256 expert slots, and a 6.0 decimal-GB host-RSS limit.
All rows use the model-owned authenticated native entry rather than HTTP;
throughput uses the same ledger formula `(actual input + actual output) /
native request wall`, excluding model startup. The slot cap is exactly
1,358,954,496 bytes (1.359 GB); the fixed live-KV seat is 83,886,080 bytes
(0.084 GB). OS page-cache warmth was not controlled, which is visible in the
Mac PREFILL spread.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac M4 Pro | CPU native | 8 | 1.359 | 0.084 / — | 3,652,927,488 (3.653) | 6.0 | — | off | 5 | 8 / 8 | 367.782 | 469.397 | 2.369 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Mac M4 Pro | Metal GPU native | 8 | 1.359 | 0.084 / — | 2,208,595,968 (2.209) | 6.0 | — | off | 5 | 8 / 8 | 160.558 | 146.811 | 6.776 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | Arm CPU native | 8 | 1.359 | 0.084 / — | 2,055,860,224 (2.056) | 6.0 | — | off | 5 | 8 / 8 | 297.623 | 570.638 | 2.148 | 3 | PASS: IDs, text, logits, KV, recurrent state |
| Acerbox DGX Spark | CUDA pageable-HMM full GPU native | 8 | 1.359 | 0.084 / — | 5,300,670,464 (5.301) | 6.0 | — | off | 5 | 8 / 8 | 63.853 | 48.940 | 18.292 | 3 | PASS: IDs, text, logits, KV, recurrent state |

Repeat spread (`(max-min)/min`): Mac CPU PREFILL 146.718%, decode 0.690%;
Mac Metal PREFILL 148.724%, decode 18.002%; DGX CPU PREFILL 2.151%, decode
1.314%; DGX CUDA PREFILL 5.789%, decode 0.103%. Metal versus Mac CPU is
2.29x faster in PREFILL, 3.20x faster in decode, and 2.86x higher combined
throughput. CUDA versus DGX CPU is 4.66x faster in PREFILL, 11.66x faster in
decode, and 8.52x higher combined throughput. CUDA sampled peak device memory
was 360,710,144 bytes (0.361 GB); pageable source pages remain host RSS.

Every one of the 12 measured requests produced IDs
`278,3270,302,4741,20,643,355,278` and text
` the city of Paris, which is the`. The complete accepted boundary was identical
across all four lanes and repeats: position 13, generation 9, scalar state SHA-256
`69ed522e5058e00682cbc6d252964176a120f69db3d2a6346c1e44776e1599af`,
committed KV 532,480 bytes with SHA-256
`bf3d5474ee4673e912cc97ec5185dc4f75512a2a55f34a250bfa03f78362e498`,
recurrent state 737,280 bytes with SHA-256
`d9a2810e9201e91c78e7b040bedf5e3a4e893a7f259e26f8b56ef8c3520654c0`,
and full logit-row SHA-256
`608cfbe6e8bf86ce4fb5a7a9b1c9bc71e00080f2f48625d776d74f2413c2c7bf`.

Build identities: Mac CPU
`e0fc40891d9eef207be9e9516435a692d50e2b444e38af767704635eee158065`,
Mac Metal `dfcd45066a4ee46e81f36c422577740a6f20b5fded65be9b6698e729898be757`,
DGX CPU `3dddc18429b09d3eaa2fa0fbf0fa249e4637715a68c3cb324a76f24a1e6a3b`,
and DGX CUDA
`dbdf02545a3cd5154eca65d1198da2796892b4920c0de1bd25ffa0f32cca9314`.
Evidence root:
`logs/lfm2-performance/6f16161565185545a7dc06ea2b5b0d963dcba7b9/20261015T010729Z/`.
The model gate and strict native builds passed; the canonical-state gate reported
`GATE_DONE ok=631 fail=0`.

## `ca634c59d2632122c55f4a25ef86338d2144a697` — expert-retention response curves

**Commit timestamp:** `2026-10-06T23:47:09-04:00`.
**Model:** Gemma 4 26B A4B, authenticated MLX affine Q4.
**Execution windows (UTC):** Acerbox CUDA-HMM `2026-10-07T06:43:23Z`–`06:51:13Z`;
EC2 V520 ROCm `2026-10-07T06:43:39Z`–`06:57:58Z`; EC2 c6i CPU
`2026-10-07T13:00:57Z`–`13:32:49Z`.
All three native builds used the same committed source archive; 777 source
files were verified unchanged. Build identities and bounded request receipts
are retained separately from this measured source pin.

Normal managed HTTP serving, P500/O128, CTX1024, W8, greedy, DPR off and request
proof off. Each budget had one fresh-process request followed by two identical
requests after KV-only clear on the same PID, preserving expert retention.
**The table averages only those two retained-expert requests (`n=2`).** The
initial observations and repeat ranges remain in the linked report. All 500
input tokens were processed each time: no prefix/DPR hits or response replay.
Clear, startup and final state export are outside HTTP timing. Fresh process
does not mean cold OS page cache.

Heavy decode was ordinary B1 under the existing prior-KV admission policy,
not X-TARGET or a KV-hit path. PREFILL B was 512 on CPU, 256 on CUDA-HMM and 128
on ROCm; N/F/Q and X maxima stayed recipe-owned. c6i provides eight vCPUs on
four physical Xeon Platinum 8375C cores. Peak RSS below is the native reported
process high-water maximum across all three requests in that process.
ROCm's **GPU expert arena stayed fixed at 3.5 GB**; its host-slot quota varied
usable retention. Spark-HMM uses canonical pageable views, not a separately
allocated expert VRAM copy. The actual KV seat is 461,373,440 bytes; the 2 GB
allowance is not a 2 GB allocation.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X: active (configured) | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| EC2 c6i.2xlarge | CPU / B1 | 8 | 1 | 0.461373440 / 2.0 | 3,095,433,216 (3.095) | 14 | 32/1/4 | off (32) | 500 | 128 / 128 | 202.677 | 318.425 | 4.398 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 c6i.2xlarge | CPU / B1 | 8 | 2 | 0.461373440 / 2.0 | 4,096,389,120 (4.096) | 14 | 32/1/4 | off (32) | 500 | 128 / 128 | 202.658 | 314.787 | 4.413 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 c6i.2xlarge | CPU / B1 | 8 | 4 | 0.461373440 / 2.0 | 6,099,853,312 (6.100) | 14 | 32/1/4 | off (32) | 500 | 128 / 128 | 202.768 | 310.923 | 4.426 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 c6i.2xlarge | CPU / B1 | 8 | 8 | 0.461373440 / 2.0 | 10,106,068,992 (10.106) | 14 | 32/1/4 | off (32) | 500 | 128 / 128 | 202.463 | 305.263 | 4.454 | 2 | PASS IDs/content ×3; final KV/state |
| Acerbox / DGX Spark | CUDA HMM full-GPU / B1 | 8 | 1 | 0.461373440 / 2.0 | 4,360,364,032 (4.360) | 16 | 48/1/4 | off (48) | 500 | 128 / 128 | 20.526 | 153.488 | 20.774 | 2 | PASS IDs/content ×3; final KV/state |
| Acerbox / DGX Spark | CUDA HMM full-GPU / B1 | 8 | 2 | 0.461373440 / 2.0 | 5,346,590,720 (5.347) | 16 | 48/1/4 | off (48) | 500 | 128 / 128 | 20.388 | 140.932 | 22.008 | 2 | PASS IDs/content ×3; final KV/state |
| Acerbox / DGX Spark | CUDA HMM full-GPU / B1 | 8 | 4 | 0.461373440 / 2.0 | 7,338,487,808 (7.338) | 16 | 48/1/4 | off (48) | 500 | 128 / 128 | 20.082 | 130.981 | 23.181 | 2 | PASS IDs/content ×3; final KV/state |
| Acerbox / DGX Spark | CUDA HMM full-GPU / B1 | 8 | 8 | 0.461373440 / 2.0 | 11,317,329,920 (11.317) | 16 | 48/1/4 | off (48) | 500 | 128 / 128 | 19.858 | 112.601 | 25.530 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 V520 | ROCm full-GPU / B1 | 8 | 1 | 0.461373440 / 2.0 | 3,965,616,128 (3.966) | 8 | 104/1/4 | off (356) | 500 | 128 / 128 | 51.595 | 472.540 | 7.198 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 V520 | ROCm full-GPU / B1 | 8 | 2 | 0.461373440 / 2.0 | 4,967,477,248 (4.967) | 8 | 104/1/4 | off (356) | 500 | 128 / 128 | 49.004 | 391.942 | 8.320 | 2 | PASS IDs/content ×3; final KV/state |
| EC2 V520 | ROCm full-GPU / B1 | 8 | 3 | 0.461373440 / 2.0 | 5,969,047,552 (5.969) | 8 | 104/1/4 | off (356) | 500 | 128 / 128 | 46.687 | 344.539 | 9.214 | 2 | PASS IDs/content ×3; final KV/state |

### Finding: balance expert availability with available compute

**CPU improved little; GPU decode improved materially.** Increasing retention
from 1 to 8 GB reduced CPU decode latency by **4.13%**, PREFILL by **0.11%** and
HTTP wall by **1.26%**. The same endpoints on Spark reduced decode by **26.64%**
and HTTP wall by **18.64%**; V520's 1-to-3 GB usable-retention range reduced decode
by **27.09%** and HTTP wall by **21.87%**. GPU PREFILL gains were smaller: 3.25%
on Spark and 9.51% on V520.

The optimization question is the **balance between ready expert/data availability
and the compute resources available to consume it**. Retention can improve data
readiness; execution shape and usable CPU/GPU capacity determine how much that
readiness helps. A larger arena alone neither supplies more compute capacity nor
proves better utilization. The CPU host had 16.467 GB RAM, and retained-expert
passes recorded zero native major faults and storage-read bytes: OS page cache
could serve weights outside Salt's retained slots. These curves therefore do
not establish the same small CPU benefit on a smaller-RAM or disk-cold host.

This interpretation is consistent with the measured curves, not a measured
arithmetic-versus-wait decomposition. CPU saturation, a hardware compute ceiling,
and individual GPU mapping/transfer/synchronization costs were not established.
No new kernels, resource policy, recipe defaults or runtime code were introduced.

**Qualification:** 33 timed requests across 11 platform-budget cells; all prompt/output IDs and
content matched. Complete native state, facts and KV payload were checked on
the final pass of each budget (11 exports), not separately on every earlier
pass. The x86 CPU and GPU exports matched one another and the compatible
historical Mac CPU reference; no fresh Arm CPU run or cross-load was performed.
The capped output preserves the grounded core facts but not a complete long
technical report. All owned servers and temporary deployment roots were removed;
the CPU instance was stopped and its temporary SSH rule revoked.

**Evidence:** full report (`experiments/reports/expert-retention/README.md`, not exported),
frozen experiment (`experiments/reports/expert-retention/experiment.json`, not exported),
per-pass results (`experiments/reports/expert-retention/results.json`, not exported),
CSV (`experiments/reports/expert-retention/results.csv`, not exported), and
cleanup receipt (`experiments/reports/expert-retention/cleanup.json`, not exported).
Detailed commands and native/HTTP receipts remain under
`logs/expert-retention-sweep/`.

## `541ee7eec22e1c3d18baa3766d1af52c5eae84b7` — CUDA PREFILL-ring repair

**Commit timestamp:** `2026-09-26T15:34:23-04:00`.
**Run:** Acerbox (DGX Spark), 2026-09-26; native full-CUDA `spark` registered
and `spark-hmm` pageable recipes, each through the ordinary managed HTTP server.
The remote source was a host-local overlay on an earlier Git HEAD, **not** a
clean checkout of this commit: ten selected code/recipe files checked against
the Mac checkout of this commit were byte-identical (`logs/0926-hmm-state/source-match.sh`).
Binary/build receipts remain host-local and separate from that source check.
The same four versioned requests and **source-specific** CPU references are
now available in `tools/grounding/fixtures/`; `tools/grounding/grounding.sh` is the host-local
reproduction control, not the measurement source for these earlier runs.

CTX12288, W8, KV budget 1 decimal GB (fixed live seat 922,746,880 bytes),
greedy, DPR off, request proof on, and fresh process/session per repeat.
`spark` has a 13 GB registered expert budget and 60 GB RSS ceiling;
`spark-hmm` has a 2 GB selected-expert budget and 16 GB RSS ceiling. Both
effective recipes report `compute_node=full`, NFQ 48/1/4 and target X48,
but these requests executed ordinary B1, not X-TARGET. The optional KV artifact
root was present for the six-repeat HMM group and disabled for the other HMM
groups; `prefill_hits=0` and `parent_hits=0` in the native measurements.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X: active (configured) | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Acerbox | CUDA registered `spark` / B1 | 8 | 13 | 0.923 / 1.0 | 29,793,038,336 (29.793) | 60 | 48/1/4 | off (48) | 384 | 128 / 128 | 23.891 | 75.497 | 24.034 | 1 | PASS: IDs, content, complete state/facts |
| Acerbox | CUDA pageable `spark-hmm` / B1; KV artifact root enabled | 8 | 2 | 0.923 / 1.0 | 5,830,868,992 (5.831) | 16 | 48/1/4 | off (48) | 2048 | 512 / 322 | 29.376 | 187.179 | 18.850 | 6 | PASS observed; earlier RED unresolved |
| Acerbox | CUDA pageable `spark-hmm` / B1; KV artifact root disabled | 8 | 2 | 0.923 / 1.0 | 5,829,009,408 (5.829) | 16 | 48/1/4 | off (48) | 2048 | 512 / 322 | 29.440 | 187.669 | 18.806 | 3 | PASS observed; earlier RED unresolved |
| Acerbox | CUDA pageable `spark-hmm` / B1; KV artifact root disabled | 8 | 2 | 0.923 / 1.0 | 5,828,640,768 (5.829) | 16 | 48/1/4 | off (48) | 4096 | 1024 / 314 | 36.671 | 208.925 | 19.870 | 1 | PASS observed; earlier RED unresolved |

`spark`'s repaired P384/O128 first output ID is 10354 versus 2021 in the
earlier rejected run; all 128 IDs and complete hashes matched the CPU oracle.
Its large registered-source RSS is not the HMM memory floor. The nine HMM
P2048/O512 repeats matched 322 IDs, content, complete state/facts and final
position 2370; the longer-P HMM request matched 314 IDs, content, complete
state/facts and position 4410. O1024 is a configured cap, **not** 1024 emitted
tokens. The earlier HMM P2048 state/facts-only RED on different source and the
September 25 HMM P4096 RED are retained; these later PASS observations do not
identify the faulty operation or promote those earlier failures. GPU physical
engagement and other unrun platform/cell combinations are not certified by
these identity checks.

**Receipts:** private evidence under `logs/0926-spark-ring-commit/` and the
host-local Acerbox `0926-platform-r3` run roots; those operational artifacts are
not exported in SALT-core.
Read-only aggregation, including exact resource columns and per-run output:
`logs/0926-hmm-state/readback-performance.sh` and
`logs/0926-hmm-state/performance-readback.jsonl`. The arithmetic means above
normalize each individual request before averaging; startup and separate KV
export are excluded from HTTP wall. These are new-source observations, not a
retroactive correction to either earlier source section.

## `66beda33ed5d816b92d71a4d13c049ff2623f73e`

**Commit timestamp:** `2026-09-25T08:15:18-04:00`
**Run:** final ordinary-B1 matrix, 2026-09-25; controller finished `2026-09-25T14:56:20Z`.
**Model:** Gemma 4 26B A4B, authenticated MLX affine Q4 package.
**Protocol:** normal HTTP via `server/serve-launch.sh`; CTX12288, W8, greedy,
DPR off, request proof enabled, fresh native process/model/session per repeat.
Fresh processes do not imply cold OS page caches. CPU/GPU lanes on the same
host ran sequentially; independent hosts ran concurrently.

PREFILL was separately batched. All heavy decode below was ordinary B1, with
X-TARGET disabled. Live source retains cheap NFQ selection before B1; the
summary-mode receipts establish configured geometry, not a per-boundary
executed-check census. Mac Metal is explicitly mixed ordinary decode; Acer
CUDA HMM is full-GPU text arithmetic with pageable weight access and shared
pinned/mapped live KV.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X: active (configured) | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Mac | CPU / B1 | 8 | 1 | 0.923 / 1.0 | 2,715,680,768 (2.716) | 4.5 | 64/1/4 | off (64) | 384 | 128 / 128 | 51.921 | 179.584 | 11.515 | 3 | PASS observed |
| Mac | CPU / B1 | 8 | 1 | 0.923 / 1.0 | 3,369,943,040 (3.370) | 4.5 | 64/1/4 | off (64) | 1024 | 128 / 128 | 55.534 | 204.027 | 13.440 | 3 | PASS observed |
| Mac | CPU / B1 | 8 | 1 | 0.923 / 1.0 | 2,827,501,568 (2.828) | 4.5 | 64/1/4 | off (64) | 2048 | 512 / 322 | 54.276 | 177.777 | 13.841 | 3 | PASS observed |
| Mac | CPU / B1 | 8 | 1 | 0.923 / 1.0 | 3,153,690,624 (3.154) | 4.5 | 64/1/4 | off (64) | 4096 | 1024 / 314 | 58.365 | 196.812 | 14.500 | 3 | PASS observed |
| Mac | Metal mixed / B1 | 8 | 1 | 0.923 / 1.0 | 2,829,860,864 (2.830) | 4.5 | 64/1/4 | off (64) | 384 | 128 / 128 | 27.430 | 186.705 | 14.248 | 3 | PASS observed |
| Mac | Metal mixed / B1 | 8 | 1 | 0.923 / 1.0 | 2,794,864,640 (2.795) | 4.5 | 64/1/4 | off (64) | 1024 | 128 / 128 | 30.508 | 219.707 | 18.556 | 3 | PASS observed |
| Mac | Metal mixed / B1 | 8 | 1 | 0.923 / 1.0 | 3,205,414,912 (3.205) | 4.5 | 64/1/4 | off (64) | 2048 | 512 / 322 | 39.174 | 230.329 | 15.064 | 3 | PASS observed |
| Mac | Metal mixed / B1 | 8 | 1 | 0.923 / 1.0 | 3,089,334,272 (3.089) | 4.5 | 64/1/4 | off (64) | 4096 | 1024 / 314 | 46.471 | 259.304 | 16.030 | 3 | PASS observed |
| Acerbox | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 7,849,357,312 (7.849) | 14 | 32/1/4 | off (32) | 384 | 128 / 128 | 47.080 | 301.557 | 8.603 | 3 | PASS observed |
| Acerbox | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,129,314,816 (8.129) | 14 | 32/1/4 | off (32) | 1024 | 128 / 128 | 51.824 | 306.805 | 11.840 | 3 | PASS observed |
| Acerbox | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,179,257,344 (8.179) | 14 | 32/1/4 | off (32) | 2048 | 512 / 322 | 67.239 | 279.489 | 10.176 | 3 | PASS observed |
| Acerbox | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,262,946,816 (8.263) | 14 | 32/1/4 | off (32) | 4096 | 1024 / 314 | 62.105 | 302.058 | 12.403 | 3 | PASS observed |
| Acerbox | CUDA HMM full-GPU / B1 | 8 | 2 | 0.923 / 1.0 | 5,828,636,672 (5.829) | 16 | 48/1/4 | off (48) | 384 | 128 / 128 | 22.322 | 144.353 | 17.281 | 3 | PASS observed |
| Acerbox | CUDA HMM full-GPU / B1 | 8 | 2 | 0.923 / 1.0 | 5,828,890,624 (5.829) | 16 | 48/1/4 | off (48) | 1024 | 128 / 128 | 24.151 | 170.189 | 22.391 | 3 | PASS observed |
| Acerbox | CUDA HMM full-GPU / B1 | 8 | 2 | 0.923 / 1.0 | 5,830,004,736 (5.830) | 16 | 48/1/4 | off (48) | 2048 | 512 / 322 | 29.503 | 186.352 | 18.844 | 3 | PASS observed |
| Acerbox | CUDA HMM full-GPU / B1 | 8 | 2 | 0.923 / 1.0 | 5,826,383,872 (5.826) | 16 | 48/1/4 | off (48) | 4096 | 1024 / 357 | 36.487 | 202.747 | 19.525 | 1 | **RED** |
| EC2 c6i | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 7,837,732,864 (7.838) | 14 | 32/1/4 | off (32) | 384 | 128 / 128 | 202.462 | 308.734 | 4.266 | 3 | PASS observed |
| EC2 c6i | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,127,963,136 (8.128) | 14 | 32/1/4 | off (32) | 1024 | 128 / 128 | 211.526 | 346.517 | 4.332 | 3 | PASS observed |
| EC2 c6i | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,177,901,568 (8.178) | 14 | 32/1/4 | off (32) | 2048 | 512 / 322 | 221.392 | 363.055 | 4.118 | 3 | PASS observed |
| EC2 c6i | CPU / B1 | 8 | 6 | 0.923 / 1.0 | 8,261,636,096 (8.262) | 14 | 32/1/4 | off (32) | 4096 | 1024 / 314 | 233.099 | 388.048 | 4.074 | 3 | PASS observed |
| EC2 V520 | ROCm GPU / B1 | 8 | 3 | 0.923 / 1.0 | 5,999,468,544 (5.999) | 7.5 | 104/1/4 | off (16) | 384 | 128 / 128 | 50.268 | 338.166 | 7.831 | 3 | PASS observed |
| EC2 V520 | ROCm GPU / B1 | 8 | 3 | 0.923 / 1.0 | 5,999,345,664 (5.999) | 7.5 | 104/1/4 | off (16) | 1024 | 128 / 128 | 55.490 | 412.815 | 10.036 | 3 | PASS observed |
| EC2 V520 | ROCm GPU / B1 | 8 | 3 | 0.923 / 1.0 | 5,997,998,080 (5.998) | 7.5 | 104/1/4 | off (16) | 2048 | 512 / 322 | 62.222 | 465.333 | 8.390 | 3 | PASS observed |
| EC2 V520 | ROCm GPU / B1 | 8 | 3 | 0.923 / 1.0 | 5,996,843,008 (5.997) | 7.5 | 104/1/4 | off (16) | 4096 | 1024 / 314 | 69.363 | 536.419 | 9.616 | 3 | PASS observed |

### Exactness and limitations

- 72 requests planned; 70 completed. Of the completed requests, 69 matched canonical output IDs and complete native state/facts hashes. One Acer HMM request diverged; the remaining two HMM P4096 repeats were interrupted/skipped.
- The failed HMM request first diverged visibly at output index 63. Its 357-token result is not equivalent work to the canonical 314-token result; do not rank its timing as a speed win.
- A subsequent uninstrumented HMM replay matched all 314 tokens and both complete hashes. Three tools-only observed runs also matched, with identical bounded layer captures. These are separate diagnostic runs, excluded from the performance means above. No repair was demonstrated; HMM's P4096 qualification remains RED.
- All completed requests passed the bounded grounded-core check. Natural early stops are valid; capped responses are not automatically complete answers.
- Proof-enabled wall includes observational proof overhead. These rows are not proof-off serving latency, export/cross-load qualification, or a claim of recovered historical peak speed.

### Evidence and build identity

- Versioned phase summary: aggregate.json (`docs/workorder/evidence/wo0925final-b1/aggregate.json`, not exported).
- Versioned diagnostic comparison: probe-analysis.json (`docs/workorder/evidence/wo0925final-b1/probe-analysis.json`, not exported).
- Requirements, caller audit, and chronology: wo0925final.md (`docs/workorder/wo0925final.md`, not exported).
- Individual measured records: `logs/wo0925final-matrix/<host>/results/<recipe>/<cell>/r*/result.json`.
- Throughput derivation and phase/configuration reconciliation: `logs/wo0925final-matrix/performance-ledger-input.sh` and `performance-ledger-input.log`. Throughput uses `(usage.prompt_tokens + usage.completion_tokens) / http_wall_s` for each request; phase rates use native `metrics` fields.
- Resource-column reconciliation: `logs/0926-platform-r3/performance-resource-reconcile.sh` and its `.log`; each peak comes from the same three measured requests (one for the RED row), while expert/KV/RSS limits come from those requests' saved `/v1/models` effective configuration.

Host-local native build identities, distinct from the portable source commit:

| Platform / recipe | Build identity SHA-256 |
|---|---|
| Mac / `mac` | `51a50ffaf9e4bb0527cb96627bd62f9c9580048e850b511a21439f5a467a180b` |
| Mac / `mac-metal` | `5e77177d08d148465d9046f76da83872f8e13a35d88068530cb3e829815ed432` |
| Acerbox / `ec2` | `5544de985bc6d81c60bb4dac546771be1f8eabd1c99a6587bec655949b064ed3` |
| Acerbox / `spark-hmm` | `e6ca6857bc5a8ae5130d71fa252f56a02bd1522cd744a76258d49de7dc492786` |
| EC2 c6i / `ec2` | `366a64687b6ba171a58ca6fa45dccd2a9e9525aa80d4254b1a0c353acb167efc` |
| EC2 V520 / `rocm` | `5c4d89bb13617c3a273a5492d3fb52f7a290736d7b21d5eb266a7a35ec43d6f5` |

## Separate registered CUDA full-resident diagnostic: frozen uncommitted source

This is **not** a row from the six-lane September 25 matrix or a measurement of
the clean commit alone. The later seven-lane attempt used HEAD
`c7b25c77a02da28dee2aa417939a99def543b96a`
(`2026-09-25T13:53:10-04:00`) **plus** frozen tracked changes with patch SHA-256
`562d33c0b84f7e5e98589209d9080332600e4cda28e75c48b050a600d146bd4a`.
Registered `spark` and pageable `spark-hmm` are distinct CUDA recipes. The first
registered-CUDA startup attempt did not qualify; its `r1-a2` normal-HTTP retry
returned a complete response but diverged at output token 0 (reference 10354,
observed 2021), failed grounded semantics and full-state/facts identity, and
did not show compiled GPU-PREFILL engagement. The saved timing is **rejected
diagnostic evidence**, not a qualified full-GPU throughput claim.

| Platform | Mode | Workers W | Expert arena/cap GB | KV cache seat/budget GB | Peak RSS bytes (GB) | RSS limit GB | NFQ (N/F/Q) | X: active (configured) | P | O: cap / actual | Prefill ms/tk | Decode ms/tk | Throughput tk/s | n | Verdict |
|---|---|---:|---:|---|---:|---:|---|---|---:|---|---:|---:|---:|---:|---|
| Acerbox | CUDA registered full-resident intent / B1 | 8 | 13 | 0.923 / 1.0 | 29,793,030,144 (29.793) | 60 | 48/1/4 | off (48) | 384 | 128 / 128 | 23.799* | 73.029* | 24.472* | 1 | **RED: token 0, state/facts, semantics; GPU PREFILL unproved** |

`*` Actual values from the rejected `r1-a2` request only; exclude them from
backend rankings. Its `spark` build identity is
`34eb14ca234808cafa23d6bf1439f35bacc38725482d453d2a34c714f3c077eb`.
Receipt: `logs/0926-platform-r3/acerbox/results/spark/p384-o128/r1-a2/result.json`;
seven-lane disposition: `logs/0926-platform-r3/aggregate-held.json` (68 PASS,
2 RED, 14 dependent requests unrun). No full-resident CUDA cell passed that
round; HMM timings cannot stand in for registered CUDA.

## Subsequent checkpoints without clean-commit performance measurements

`c7b25c77a02da28dee2aa417939a99def543b96a`
(`2026-09-25T13:53:10-04:00`) records documentation, evidence summaries, and the
tools-only observer. It changes no production runtime. No separate performance
numbers are assigned to it; the measured source remains the commit above.

The qualified table indexes the final September 25 matrix, not every historical
experiment. The distinct dirty-source CUDA row above remains a rejected control.
Add older or future measured commits as separate sections only after their
source identity, execution mode, workload, and denominators are reconciled
with their receipts.
