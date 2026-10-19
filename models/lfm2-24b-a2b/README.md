# LFM2.5-24B-A2B — pinned source authority and package contract

`Lfm2MoeForCausalLM`, 24.0B total / 2.45B active. 40 layers: 30 double-gated
short-convolution blocks and 10 GQA layers, 64 experts top-4 per MoE block,
128,000-token vocabulary, 131,072-token context, tied output head.

Native CPU text inference is runtime-ready on the authenticated MLX affine
package. The qualified path compiles the complete 40-layer program, keeps conv
recurrence and live attention KV across PREFILL/decode, uses model-owned
sigmoid/bias routing and affine Q6 through generic core seams, and publishes the
tied-head vocabulary logits for deterministic compiled-C sampling.

## Source authority

- Container: <https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit>
- Frozen revision: `667408d494bde191c2cb9dd4f4549827816b9275`
- Architecture reference: `LiquidAI/LFM2.5-24B-A2B` `config.json`
- Semantics reference: `transformers` `models/lfm2_moe/modeling_lfm2_moe.py`
  (`Lfm2MoeShortConv`, `Lfm2MoeAttention`, `Lfm2MoeTopKRouter`,
  `Lfm2MoeSparseMoeBlock`, `Lfm2MoeDecoderLayer`, `Lfm2MoeModel`)
- License declared by the container: Apache-2.0 (see `LICENSE`).

The importer pins every source file by size and SHA-256, reads all three shard
headers with bounded range reads, and refuses missing, extra, duplicate,
mis-typed or mis-shaped tensors. Source payload bytes are copied verbatim:
`payload_copied: false`, `requantized: false`.

## Quantization

One container, two affine tracks, both `group_size 64 / mode affine`:

| track | scope | weight | scales / biases |
|---|---|---|---|
| 4-bit | every projection | `U32` packed | `BF16` |
| 6-bit | `model.embed_tokens` only | `U32` packed (`[128000, 384]`) | `BF16` (`[128000, 32]`) |

The 6-bit embedding is also the tied output head (`tie_word_embeddings: true`),
so the package binds one payload and references it — no duplicated table.

## Model semantics (pinned from the reference implementation)

Per layer:

```
residual = h
h = operator_norm(h)                    # RMSNorm, eps = norm_eps = 1e-5
h = conv(h)  or  self_attn(h)
h = h + residual
h = h + feed_forward(ffn_norm(h))
```

There is no pre-embedding norm and no logit softcap. The top-level norm named
`model.embedding_norm` is applied **after** the last layer (it is the final
norm); the head is the tied embedding table.

- **ShortConv** (`layer_types[i] == "conv"`): `in_proj -> (B, C, x)` chunk,
  `h = B * x`, depthwise causal `conv1d` (`groups = hidden`, `kernel = 3`,
  `padding = 2`, truncated to `seq_len`), `y = C * h`, `out_proj`. No bias
  anywhere (`conv_bias: false`). Conv state is `[hidden, kernel]` per layer.
- **Attention** (`layer_types[i] == "full_attention"`, exactly `i % 4 == 2`):
  GQA 32 query heads / 8 KV heads, `head_dim 64`; per-head RMSNorm on Q and K
  (`q_layernorm`, `k_layernorm`, eps = `norm_eps`); full RoPE on `head_dim`
  with `theta = 1e7`; scaling `64**-0.5`; causal softmax; `repeat_kv` 4x.
- **Dense FFN** (`layer_idx < num_dense_layers`, layers 0 and 1):
  `w2(silu(w1(x)) * w3(x))` with `gate_proj = w1`, `up_proj = w3`,
  `down_proj = w2`, intermediate 11776.
- **MoE FFN** (layers 2..39): `gate.weight` is a plain `F32 [64, 2048]` router;
  `expert_bias` is a plain `F32 [64]` routing bias.
  `routing_weights = sigmoid(router_logits)` — **sigmoid, not softmax**.
  Selection is `topk(routing_weights + expert_bias, k=4)`; the weights carried
  forward are the *unbiased* sigmoid values gathered at the selected indices.
  `norm_topk_prob` divides by `(sum + 1e-6)`, then multiplies by
  `routed_scaling_factor` (= 1.0). Each expert is `down(silu(gate(x)) * up(x))`
  with intermediate 1536.

## Package layout

```
trunk.bin   embeddings, final norm, per-layer norms, conv/attention/dense
            projections, router and expert-bias vectors, conv kernels
pool.bin    LFMPOOL1 header (expert_bytes, geometry, per-layer expert base)
            followed by 2,432 expert slots, 5,308,416 bytes each
index.txt   LFMIDX1: geometry line, 40 L lines (layer -> expert base),
            T lines (trunk tensors, absolute offsets in trunk.bin),
            E lines (expert tensors, absolute offsets in pool.bin)
auth.json   authenticated receipt: per-file identity + SHA-256, package_id
tokenizer.runtime.json   compact-merges-v1 derived artifact (+ .auth.json)
```

Expert slot contents, in order: `gate` weight/scales/biases, `up`
weight/scales/biases, `down` weight/scales/biases.

The two dense layers (0, 1) have `L <layer> -1`; their expert space is unused,
so the pool holds only the 2,432 real experts and never a padded 40x64 grid.

## Setup and launch

Use absolute paths to the selected project Python and the dedicated workspace:

```bash
export LFM_ROOT=/path/to/dedicated-lfm-workspace
export SALT_BUILD_PYTHON=/absolute/path/to/project/python3
export LFM_BIN_DIR="$LFM_ROOT/bin"
cd "$LFM_ROOT/repo"

"$SALT_BUILD_PYTHON" models/lfm2-24b-a2b/setup.py fetch \
  --source "$LFM_ROOT/weights/LFM2.5-24B-A2B-MLX-4bit"
"$SALT_BUILD_PYTHON" models/lfm2-24b-a2b/setup.py import \
  --source "$LFM_ROOT/weights/LFM2.5-24B-A2B-MLX-4bit" \
  --package "$LFM_ROOT/packages/mlx4-6e-v1"
"$SALT_BUILD_PYTHON" models/lfm2-24b-a2b/setup.py prepare-tokenizer \
  --package "$LFM_ROOT/packages/mlx4-6e-v1"
bash models/lfm2-24b-a2b/build.sh

exec 3<"$LFM_ROOT/packages/mlx4-6e-v1/trunk.bin"
exec 4<"$LFM_ROOT/packages/mlx4-6e-v1/pool.bin"
exec 5<"$LFM_ROOT/packages/mlx4-6e-v1/index.txt"
exec 6<"$LFM_ROOT/weights/LFM2.5-24B-A2B-MLX-4bit/tokenizer.json"
"$LFM_BIN_DIR/lfm-server" 3 4 5 6 \
  a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e \
  2048 32 8 256 3221225472
```

`fetch`, `import` and `prepare-tokenizer` refuse existing outputs; do not rerun
them over a materialized package or overwrite a receipt. Normal startup checks
authenticated identities without rescanning weight payloads.

## Build and gate

```bash
make lfm-server   # -> $LFM_BIN_DIR/lfm-server + receipt
make test         # curated public smoke, including the LFM descriptor gate
```

`lfm-server` is the authenticated native CPU inference entry. It validates the
pool header and per-layer expert bases, audits `index.txt` against the declared
geometry, maps the bounded package, compiles the complete engine program, and
runs a tokenizer-backed semantic PREFILL/decode probe. Its build receipt records
`entry_scope: authenticated-native-inference`, `inference_ready: true`, and
`runtime_qualified: true`. CPU is the default; accelerator builds require their
explicit backend selection and platform toolchain.

## 6-bit affine track: verified against the unquantized checkpoint

`models/lfm2-24b-a2b/lfm_quant.c` implements MLX affine group-64 6-bit as the
model-owned `SaltQuant` module. MLX packs 6-bit as **4 values per 3 bytes**
(`get_pack_factor<6>() = 4`, `get_bytes_per_pack<6>() = 3`), which is a
contiguous little-endian bit stream, and the core holds only the seam: one
declaration in `include/salt/quant.h` and one registry case in `src/quant.c`.

Verified against the unquantized `LiquidAI/LFM2.5-24B-A2B` checkpoint, read by
range request, with the container's own stored group scale and bias:

| token | BF16 oracle min / max / mean | module min / max / mean |
|---|---|---|
| 0 | -0.064941 / 0.091309 / 0.000253375 | -0.064941 / 0.091309 / 0.000269078 |
| 124900 | -0.097656 / 0.113281 / -0.000351308 | -0.097656 / 0.113281 / -0.000354160 |

Min and max are exact on both rows; the mean agrees within 5% and 0.8% relative.
`bias[0] = 0.039062` is exactly the group max and `|scale[0]| = 0.001083` matches
the implied `span/63 = 0.001085`. 100 of 2048 codes differ from the value derived
as `round((oracle - bias)/scale)` by exactly ±1, spread uniformly over every bit
position and group — rounding-boundary noise in the derived code, not a layout
defect. The residual against the oracle is quantization error.

Caution for anyone re-measuring this: Python's `struct` has no bfloat16 code and
`'e'` is IEEE **FP16**. Reading a BF16 tensor with `'e'` yields plausible-looking
but wrong numbers and produced a false "the packing is wrong" verdict. Decode
BF16 explicitly as `uint16 << 16`.

## Private-source qualification boundary

The following evidence is pinned to merged private Salt source; private test
scripts and raw receipts are not exported in SALT-core. Measured on Acerbox
(GB10, aarch64) against package
`a7625428a15a5c1b83f4fecaf5dbaddb0e553a087d90e94557c97dca7146e41e`:

- `models/lfm2-24b-a2b/test.sh`: **644 ok / 0 FAIL**.
- All 35 engine translation units compile under `-Werror -pedantic`.
- The descriptor admits all 40 layers: 30 conv / 10 full-attention; conv layers
  carry zero KV, attention layers carry Q/K norms and live key/value planes.
- Maple INT2 and state-control regressions pass inside the same model gate.
- Two fresh native processes produced identical token IDs
  `[278, 3270, 302, 4741, 20, 643, 355, 278]` for
  `The capital of France is`, decoding to
  ` the city of Paris, which is the`.
- The canonical result-record SHA-256 for both runs was
  `ce32f0a06065d85988d4edfc482f880f5bd51b86baf238f77c160c50c15a3c53`.

The core additions remain seams: a generic conv cell dispatch, an optional
model-owned router callback, affine-Q6 encoding/registry admission, pinned dense
layer planning, and zero-KV handling. Short-convolution arithmetic, sigmoid
routing, package mapping, Q6 realization, tokenizer use, runtime and native
entry remain under `models/lfm2-24b-a2b/`.
