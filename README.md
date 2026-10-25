![SALT logo](salt-logo-white.svg)

# SALT - State Assured Latent-Trasition engine

**Paper:** [SALT: Reusable Computation for Stateful Language-Model Inference](https://zenodo.org/records/23111764) (Zenodo preprint, 2026).

> **Release scope:** Gemma 4 CPU/GPU text, LFM2.5 text in the named CPU/Metal/CUDA
> cells, and qualified one-image CUDA vision. Qwen retains its scoped adapter and
> Maple remains experimental. A model or recipe appearing in the tree is not a
> claim that every backend, context length, or workload is qualified.

SALT Core is a portable C99 state-assured latent-transition engine. Its public
contract centers deterministic arithmetic, authenticated model packages, live
state/KV ownership, portable state export/import, and subordinate CPU, Metal,
CUDA, and HIP physical backends.

## What SALT does

SALT makes model computation and its resulting state reusable across compatible
machines instead of rebuilding every session from a prompt. Its seven roots are
stateful inference, bit-identical cross-platform compute, portable C99,
deterministic math, rigorous resource conservation, disk-mapped weight offload,
and controlled randomness. The aim is useful inference on existing cloud and
edge hardware without requiring the machine that consumes a session to be the
one that originally computed it.

Authenticated compressed files remain the canonical weight source. The
portable engine owns arithmetic order, resource and address bookkeeping,
sampling, KV/state transitions, and commit; model adapters supply geometry and
package policy. CPU, Metal, CUDA, and HIP execute admitted work without becoming
separate state authorities. Export/import preserves compatible model-native KV
and session state. Optional Dynamic Performance Reference (DPR) can reuse exact
transitions or propose cached trajectories, but the live model target alone
verifies and commits new state. See [Architecture](docs/ARCHITECTURE.md) and
[Terminology](docs/TERMINOLOGY.md) for the contracts and vocabulary.

> [!WARNING]
> Preserved KV/state is more than a saved prompt: it can carry the accumulated
> effects of an agent's automated activity and support continued or branched
> execution. It may also expose sensitive information. These risks exist with
> or without SALT. Retention permission is not unrestricted reuse permission;
> restoring state does not grant authority to act. Read the
> [KV Cache and Persistent Agent State Statement](docs/COGNITIVE-STATE-RISK.md).
> This is a disclosure and responsible-use standard, not a claim of implemented
> safeguards or an amendment to the software license.

## Models and execution options

Model weights are **not** included. Each model owns its package format,
geometry, quantization, runtime defaults, and qualification boundary under
[`models/<model>/`](models/README.md). Never combine a tokenizer, weight package,
receipt, or portable state from different model/quant tracks.

| Model | Native entry | Exported execution choices | Current boundary |
|---|---|---|---|
| [Gemma 4 26B-A4B](models/gemma4-26b-a4b/README.md) | `gemma4-server` through `server/serve-launch.sh` | Mac/Linux CPU, Mac Metal, registered CUDA, pageable-HMM CUDA, HIP/ROCm; separate CPU-text/CUDA-vision recipe | Runtime-ready in the named recipes; MLX affine-Q4 and NVFP4 are separate state tracks |
| [LFM2.5-24B-A2B](models/lfm2-24b-a2b/README.md) | `lfm-server` | CPU; Mac Metal PREFILL + CPU B1 decode; CUDA; HIP build path | Runtime-ready; Mac Metal above position 8,192 remains RED; HIP has no current promoted real-model cell |
| [Qwen3.6 A3B](models/qwen36/qualification/README.md) | `salt`, optionally through the Qwen server adapter | CPU default; bounded Metal PREFILL with CPU decode above the model-owned B≥200 crossover | Runtime-ready adapter with scoped historical qualification; root asset paths are development defaults, not a portable package |
| [Maple preview](models/maple-preview/README.md) | Salt model adapter | Mac CPU INT2 | Experimental; `runtime_ready=false`; sampled answer quality remains RED |

### Gemma 4: package, authenticate, and serve

Obtain the pinned MLX affine-Q4 source named in
[`mlx4-source/source-contract.json`](models/gemma4-26b-a4b/mlx4-source/source-contract.json)
with its tokenizer and metadata, and follow the model license. Keep source and
materialized files outside the checkout:

```bash
SOURCE=/absolute/path/to/pinned-gemma4-mlx4-source
POOL=/absolute/path/to/gemma4-pool.bin
export SALT_GEMMA_PLATFORM_RECIPE=ec2  # checked-in Linux CPU recipe

python3 tools/gemma4-mlx4-source.py validate-source
python3 tools/gemma4-mlx4-source.py validate-download --source-dir "$SOURCE"
python3 tools/gemma4-mlx4-source.py materialize-expert-pool \
  --source-dir "$SOURCE" --output "$POOL"
make gemma4-build-receipts

# One-time authentication/materialization check; no listener.
bash server/serve-launch.sh --backend gemma4 \
  --gemma-source-dir "$SOURCE" --gemma-pool "$POOL" \
  --gemma-context-tokens 512 --gemma-max-output-tokens 128 \
  --gemma-verify-only

# One persistent native session on loopback.
bash server/serve-launch.sh --backend gemma4 \
  --gemma-source-dir "$SOURCE" --gemma-pool "$POOL" \
  --gemma-context-tokens 512 --gemma-max-output-tokens 128
```

In another terminal:

```bash
curl -sS http://127.0.0.1:8090/v1/models
curl -sS http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-26b-a4b-it","messages":[{"role":"user","content":"Hello from Magnum Opus."}],"max_tokens":32}'
```

#### Optional committed-token logits

A Gemma rotation engine using native protocol v6 advertises
`salt_token_logits` under `openai_chat_completions.response_formats` in
`GET /v1/models`. Request up to eight target-authoritative entries per
committed output token:

```bash
curl -sS http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gemma-4-26b-a4b-it",
    "messages":[{
      "role":"user",
      "content":"Which planet is known as the Red Planet? A. Venus B. Mars C. Jupiter D. Mercury. Answer with only the option letter."
    }],
    "temperature":0,
    "top_k":1,
    "max_tokens":8,
    "response_format":{"type":"salt_token_logits","top_k":8}
  }'
```

The ordinary response text and `usage` fields are unchanged. Metadata is added
under `x_salt.token_logits`; streaming responses carry it in the final SSE
event. An abbreviated record is:

```json
{
  "sequence": 1,
  "token_id": 236799,
  "authority": "target",
  "distribution": "full_vocabulary",
  "calibrated": false,
  "selected": {
    "token_id": 236799,
    "logit_bits": "419d94e0",
    "logit": 19.69769287109375,
    "probability_ppm": 1000000,
    "probability_pct": 100.0
  },
  "top": [
    {"token_id": 236799, "logit_bits": "419d94e0",
     "logit": 19.69769287109375,
     "probability_ppm": 1000000, "probability_pct": 100.0}
  ]
}
```

The engine scans the authoritative logits already used for selection, stores a
bounded record in initialization-owned memory, and publishes it with the
existing token commit. The option does **not** constrain or mask logits, change
the sampler, alter PREFILL/DECODE/TARGET shape, add a model step, or add tokens
to `usage`. One visible answer letter normally still counts the committed turn
close token, so `completion_tokens` may be two. Idempotent replay returns the
same records. Requests fail closed when the native protocol is older than v6 or
when DPR is enabled.

These are raw full-vocabulary softmax values rounded to integer parts per
million. They are not calibrated factual confidence and are not renormalized
over the displayed choices. Do not infer semantic ambiguity from a low or high
token probability. If ambiguity matters, include explicit options such as
`Cannot be determined`, `Conflicting evidence`, or `Not applicable`, then map
the selected option to that application status.

Keep the listener on loopback: the API has no built-in client authentication or
TLS. Changing a context/output request does not require restarting a healthy
server unless its startup capacity is insufficient.

### LFM2.5: native CPU or accelerator build

The LFM importer authenticates its external source and produces `trunk.bin`,
`pool.bin`, `index.txt`, tokenizer runtime data, and `auth.json`. Build the same
model package for a selected physical backend:

```bash
export LFM_ROOT=/absolute/path/to/lfm-workspace
export LFM_BIN_DIR="$LFM_ROOT/bin"
export SALT_BUILD_PYTHON=/absolute/path/to/python3

make lfm-server                         # portable CPU
LFM_BACKEND=metal make lfm-server       # Darwin: B256 PREFILL, CPU B1 decode
LFM_BACKEND=cuda make lfm-server        # Linux/NVIDIA
LFM_BACKEND=hip make lfm-server         # Linux/AMD build path
```

`lfm-server` takes already-open authenticated package descriptors. The checked-in
default CPU cell is CTX2048, B32, W8, 376 expert slots, and a 3.5-decimal-GB RSS
ceiling. Consult the model README for the exact descriptor invocation.

### Qwen and Maple

Qwen requires one complete compatible package: trunk, expert pool, head,
embedding, layout/offset manifests, and tokenizer. The root `engine.config`
contains host-specific development defaults, so pass portable package paths
explicitly. CPU is the default; `--gpu-prefill` selects the qualified bounded
Metal-PREFILL/CPU-decode crossover. Maple is an experimental CPU-only preview;
its presence in the registry is not production promotion.

## Platform recipes and resource limits

Gemma recipes are merged over the model blueprint. B, NFQ, X, expert capacity,
and RSS policy are model/platform inputs rather than portable-engine constants:

| Recipe | Physical realization | PREFILL B | NFQ N/F/Q | X max | Expert budget | RSS ceiling |
|---|---|---:|---:|---:|---:|---:|
| `mac` | Apple CPU | 512 | 64/1/4 | 64 | 1 GB | 4.5 GB |
| `mac-metal` | Apple Metal + bounded host resources | 512 | 64/1/4 | 64 | 1 GB | 4.5 GB |
| `ec2` | Linux/x86 CPU | 512 | 32/1/4 | 32 | 6 GB | 14 GB |
| `spark` | registered full CUDA | 256 | 48/1/4 | 48 | 13 GB + preload | 60 GB |
| `spark-hmm` | full-text pageable-HMM CUDA | 256 | 48/1/4 | 48 | 2 GB | 16 GB |
| `spark-hmm-vision` | CPU text + pageable-HMM CUDA vision | 512 | 32/1/4 | 32 | 1 GB | 7 GB |
| `rocm` | HIP/ROCm candidate | 128 | 104/1/4 | 356 | 3 GB host / 3.5 GB device | 8 GB |

LFM model defaults:

| Backend | PREFILL | Decode | Expert slots / payload ceiling | Process RSS ceiling |
|---|---|---|---:|---:|
| CPU | B32 | CPU B1 | 376 / 1,995,964,416 bytes (1.996 GB) | 3.5 GB |
| Metal | B256 | portable CPU B1 | 376 / 1,995,964,416 bytes (1.996 GB) | 3.5 GB |
| CUDA | B32 | CUDA | 448 / 2,378,170,368 bytes (2.378 GB) | 4.0 GB |
| HIP | caller-selected build cell | HIP | explicit runtime argument | no promoted current cell |

Gemma launch overrides are explicit and range-checked:

```bash
--gemma-context-tokens N
--gemma-max-output-tokens N
--gemma-kv-budget-gb N
--gemma-expert-budget-gb N
--gemma-memory-limit-gb N
```

Memory numbers are decimal GB. `SALT_GEMMA_MEMORY_LIMIT_GB` is an all-inclusive
current-process RSS ceiling, including resident file-backed pages. Expert budgets
are capacity ceilings, not promises that every byte is resident. KV capacity,
expert capacity, process RSS, device memory, and macOS application/compressor
figures are different quantities and must not be added together. LFM passes
context, B, workers, expert slots, and exact RSS bytes to its native entry.

## Build and test options

```bash
make -n test all                       # dependency/command closure; stderr must be empty
make all                               # Qwen `salt` + Gemma server
make test                              # curated public smoke suite
make gemma4-build-receipts             # Gemma native runners and authenticated receipts
make lfm-server                        # CPU LFM build + receipt
LFM_BACKEND=metal make lfm-server      # Darwin Framework-Metal build
python3 tools/test/gemma4-kv-compat-projection-test.py
shasum -a 256 -c CORE-FILES.sha256
```

On Darwin, `make test` also runs the mapped routed-MoE and mapped-trunk Metal
lifecycle gates. These commands do not invoke `xcodebuild`, `xcrun`, or the
offline Metal compiler. Linux CUDA/HIP builds require their platform compilers.
A component/build PASS is not an inference qualification: release evidence also
requires an authenticated package, ordinary native/server inference, exact
tokens, complete KV/state where proof is requested, memory enforcement, and a
clean process exit.

## Representative measured cells

The historical ledger is [`PERFORMANCE.md`](PERFORMANCE.md); the `f58d6fd5`
rows below are the newer private-source evidence for the exported runtime and
are not yet duplicated there. These rows are not interchangeable: each keeps
its model, source, host, workload, mode, and proof boundary.

| Source/model/platform | Mode and workload | PREFILL ms/tk | Decode ms/tk | Combined tk/s | Peak RSS | Verdict |
|---|---|---:|---:|---:|---:|---|
| `f58d6fd5`, Gemma 4, Apple M4 Pro | Metal P12,059/O478 AIAYN | **50.215** | 258.004 | **17.185** | 2,123,923,456 B (2.124 GB) | HTTP 200; CPU/Metal IDs, text, response SHA, position and facts geometry exact; request proof off |
| `f58d6fd5`, Gemma 4, Apple M4 Pro | Metal P500/O128, n=3 | **25.386** | 198.896 | 16.276 | ≤3,577,643,008 B (3.578 GB) | exact repeated output; full wall 0.91% below the retained `3e107ca` cell |
| `f58d6fd5`, LFM2.5, Apple M4 Pro | Metal B256 PREFILL + CPU B1, P480/O64, n=3 | **19.953** | **93.980** | **34.947** | 2,310,963,200 B (2.311 GB) | exact IDs/text/complete state across repeats |
| `03eaa01b`, LFM2.5, DGX Spark | bounded CUDA P4,096/O100, n=5 | 10.771 | 45.395 | 86.235 | 11,277,672,448 B (11.278 GB) | exact complete state; historical 1,504-slot/16-GB cell, not the current 448-slot default |
| Maple preview, Apple M4 Pro | CPU mode 0, P1,031/O1,000 | 19.283 | 40.033 | 33.893 | 2,981,494,784 B (2.981 GB) | exact state against its control; sampled answer quality remains RED |

The current Gemma Metal score-workspace repair computes every QK score once in
startup-owned scratch and retains ordered maximum, denominator, and AV folds.
Against the rejected repeated-score AIAYN candidate it cut wall by 46.26% and
PREFILL ms/token by 50.55%; it used no new runtime arena. LFM Mac Metal remains
RED beyond position 8,192 because its model-program submission is rejected
before any GPU kernel launch. Neither result promotes an untested context.

## Repository map

- `include/salt/` and `src/`: portable engine, state/resources, and physical backends.
- `models/`: model adapters, package contracts, build entries, and policy.
- `server/`: OpenAI-shaped HTTP admission and persistent native-session supervision.
- `tools/`: package preparation and bounded public checks.
- `docs/FORMAL-PROOFS.md` and `experiments/reports/`: scoped proof and experimental
  evidence, not substitutes for authenticated real-model inference.

The curated runtime source now includes private Salt review commit
`f58d6fd5059e9046cb462a7eaff7441b19bf56eb`
(`2026-10-22T01:19:21-04:00`) for the exported runtime paths, based on merged
private `main` revision `90b9c82b11eb18116f6dc4f18146c0e36100f58b`. This updates
source bytes, not the historical proof pins or platform qualification. It
contains no model weights. Model artifacts retain their own licenses and must
be obtained separately.

The scoped [formal proof bundle](docs/FORMAL-PROOFS.md) contains executable TLA+
models and TLAPS proof sources. Its documented proof boundary is not a claim
that the complete runtime or concrete KV bytes are formally verified.

The exported tree passed isolated Linux CPU and CUDA builds, focused
KV-compatibility tests, and the curated public `make test` smoke suite;
`make all` and `make -n test all` passed on Linux. A clean SALT-core c6i build
and authenticated package served ordinary Gemma 4 HTTP requests and matched
tokens and complete KV/state after export, clear, import, and continuation in
both the same and a fresh native process.

Cross-platform inference evidence comes from the separately pinned private
Salt qualification: the corrected six-mode CPU/Metal/CUDA/ROCm text grid
matched output IDs and complete exported state in 360/360 cells (see
[Randomplay](experiments/reports/randomplay/README.md)). The Acerbox
`spark-hmm-vision` CPU-text/CUDA-vision recipe also passed its scoped 12/12
text-and-image E2E repeat gate and a persistent one-image API request. Those
records establish their tested configurations; they are not independent reruns
of every accelerator and vision cell from this exact SALT-core commit, nor a
claim of universal performance or engine-wide formal verification. Model
weights and their licenses remain separate from this source release.
