#ifndef SALT_LFM_H
#define SALT_LFM_H

/* LFM2.5-24B-A2B (Lfm2MoeForCausalLM) geometry and package contract.
 *
 * Single source for the model descriptor, the native entry and the tests.
 * Every value here is pinned from the container's config.json and from
 * transformers models/lfm2_moe/modeling_lfm2_moe.py; none is inherited from
 * Gemma, Qwen or Maple. */

#include <stdint.h>

#define LFM_NAME "lfm2-24b-a2b"

#define LFM_LAYERS 40
#define LFM_HIDDEN 2048
#define LFM_VOCAB 128000
#define LFM_EXPERTS 64
#define LFM_TOP_K 4
#define LFM_MOE_INTERMEDIATE 1536
#define LFM_DENSE_INTERMEDIATE 11776
#define LFM_DENSE_LAYERS 2
#define LFM_HEADS 32
#define LFM_KV_HEADS 8
#define LFM_HEAD_DIM 64
#define LFM_CONV_L 3
#define LFM_GROUP 64
#define LFM_PERIOD 4
#define LFM_FULL_SLOT 2

#define LFM_PAD 124893
#define LFM_BOS 124894
#define LFM_EOS 124900
#define LFM_THINK_OPEN 124901
#define LFM_THINK_CLOSE 124902

#define LFM_ROPE_THETA 10000000.0
#define LFM_NORM_EPS 1e-5f
/* 64 ** -0.5, exactly representable. */
#define LFM_SCORE_SCALE 0.125f

/* Layers 0 and 1 are dense; the expert space exists only for layers 2..39. */
#define LFM_MOE_LAYERS (LFM_LAYERS - LFM_DENSE_LAYERS)
#define LFM_EXPERT_COUNT (LFM_MOE_LAYERS * LFM_EXPERTS)

/* gate w/s/b + up w/s/b + down w/s/b, verbatim source bytes, no padding. */
#define LFM_EXPERT_BYTES 5308416u
#define LFM_POOL_HEADER_BYTES 4096u
#define LFM_POOL_BYTES \
    ((uint64_t)LFM_POOL_HEADER_BYTES + (uint64_t)LFM_EXPERT_COUNT * LFM_EXPERT_BYTES)

#define LFM_POOL_MAGIC "LFMPOOL1"
#define LFM_INDEX_MAGIC "LFMIDX1"
#define LFM_INDEX_ABI "salt-lfm25-affine4-6-f32-conv-v1"
#define LFM_PACKAGE_SCHEMA "salt.lfm25.package.v1"

/* Encodings in index.txt are the engine's SaltTensorEncoding values
 * (include/salt/tensorops.h). LFM_TENSOR_ENCODING_AFFINE_Q6 does not exist in
 * that enum yet: it is the one quant-track addition this model requires, and
 * until it lands salt_tensor_storage_validate refuses every 6-bit tensor. */
#define LFM_TENSOR_ENCODING_F32 1
#define LFM_TENSOR_ENCODING_BF16 2
#define LFM_TENSOR_ENCODING_AFFINE_Q4 3
#define LFM_TENSOR_ENCODING_AFFINE_Q6 7

/* Trunk tensor roles, in the order the importer emits them. */
typedef enum {
    LFM_ROLE_EMBEDDING = 0,
    LFM_ROLE_FINAL_NORM,
    LFM_ROLE_OPERATOR_NORM,
    LFM_ROLE_FFN_NORM,
    LFM_ROLE_Q,
    LFM_ROLE_K,
    LFM_ROLE_V,
    LFM_ROLE_O,
    LFM_ROLE_Q_NORM,
    LFM_ROLE_K_NORM,
    LFM_ROLE_CONV_IN,
    LFM_ROLE_CONV_OUT,
    LFM_ROLE_CONV_KERNEL,
    LFM_ROLE_DENSE_GATE,
    LFM_ROLE_DENSE_UP,
    LFM_ROLE_DENSE_DOWN,
    LFM_ROLE_ROUTER,
    LFM_ROLE_EXPERT_BIAS,
    LFM_ROLE_OUTPUT_HEAD_TIED,
    LFM_ROLE_COUNT
} LfmTrunkRole;

static const char *const LFM_ROLE_NAMES[LFM_ROLE_COUNT] = {
    "embedding", "final_norm", "operator_norm", "ffn_norm",
    "q", "k", "v", "o", "q_norm", "k_norm",
    "conv_in", "conv_out", "conv_kernel",
    "dense_gate", "dense_up", "dense_down",
    "router", "expert_bias", "output_head_tied",
};

/* Expected trunk role census. Attention layers are exactly i % LFM_PERIOD ==
 * LFM_FULL_SLOT, so 10 of the 40 layers carry q/k/v/o and 30 carry conv. */
#define LFM_ATTENTION_LAYERS 10
#define LFM_CONV_LAYERS 30

/* Expert tensor roles, in slot order. */
typedef enum {
    LFM_EXPERT_ROLE_GATE = 0,
    LFM_EXPERT_ROLE_UP,
    LFM_EXPERT_ROLE_DOWN,
    LFM_EXPERT_ROLE_COUNT
} LfmExpertRole;

static const char *const LFM_EXPERT_ROLE_NAMES[LFM_EXPERT_ROLE_COUNT] = {
    "expert_gate", "expert_up", "expert_down",
};

/* Live state geometry, in float elements.
 *
 * Attention layers keep full causal K/V; conv layers keep a kernel-sized
 * recurrent window per channel. These are deliberately separate quantities:
 * the conv window is not KV and must never be folded into the KV plan. */
static inline uint64_t lfm_kv_floats(uint64_t tokens) {
    return tokens * (uint64_t)LFM_ATTENTION_LAYERS * 2u
         * (uint64_t)LFM_KV_HEADS * (uint64_t)LFM_HEAD_DIM;
}

static inline uint64_t lfm_conv_state_floats(void) {
    return (uint64_t)LFM_CONV_LAYERS * (uint64_t)LFM_HIDDEN * (uint64_t)LFM_CONV_L;
}

/* Every LFM norm stores the multiplier directly -- no 1+w shift.
 *
 * Lfm2MoeRMSNorm returns `self.weight * hidden_states` and is documented as
 * equivalent to T5LayerNorm, so the stored tensor IS the gain. Gemma's MLX
 * container stores 1+w because Gemma's own norm computes (1 + weight); that is
 * an architecture convention, not an MLX-wide one, and copying it here would
 * shift every norm by exactly 1.0 and change logits and KV.
 *
 * Applies to operator_norm, ffn_norm, embedding_norm and q_layernorm alike. The
 * modules still take the convention as a field and test both paths, so the
 * value below is the pin, not an assumption baked into the arithmetic. */
#define LFM_NORM_SHIFTED 0

static inline int lfm_is_attention_layer(int layer) {
    return layer % LFM_PERIOD == LFM_FULL_SLOT;
}

static inline int lfm_is_conv_layer(int layer) {
    return layer >= 0 && layer < LFM_LAYERS && !lfm_is_attention_layer(layer);
}

#endif /* SALT_LFM_H */
