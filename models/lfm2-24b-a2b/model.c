#include "salt/model.h"
#include "lfm.h"

/* LFM2.5-24B-A2B descriptor.
 *
 * The complete runtime is qualified on the source-bound MLX affine package:
 * model-owned short convolution and sigmoid/bias routing enter through generic
 * engine seams, affine Q6 is a registered tensor encoding, and the ordinary
 * engine cells own attention, FFN traversal, KV/state commit and tied-head
 * publication. The declared schedule is period 4/full slot 2 with dense layers
 * 0 and 1, full-head RoPE on attention layers, raw SiLU, no logit softcap,
 * layer scale, parallel dense branch or residual postnorm. */

static const SaltStateModelDesc STATE = {
    .schema_version = SALT_STATE_SCHEMA_VERSION,
    .mindset_kind = SALT_MINDSET_PREFIX,
    .facts_kind = SALT_FACTS_KV_ROWS,
    .facts_optional = 1, .clear_facts_supported = 1,
    .hard_reset_supported = 1, .default_mindset_end = 0,
};

static const SaltTextGraphDesc TEXT = {
    .period = LFM_PERIOD, .full_slot = LFM_FULL_SLOT,
    .dense_intermediate = LFM_DENSE_INTERMEDIATE,
    /* The dense branch exists only on layers 0 and 1, and those two are PINNED:
     * resident rather than streamed, so they carry no routed experts. */
    .dense_layers = LFM_DENSE_LAYERS,
    .pinned_layers = LFM_DENSE_LAYERS,
    .expert_activation = SALT_ACT_SILU,
    .router_rmsnorm = 0, .parallel_dense_routed = 0, .residual_postnorm = 0,
    .final_layer_scale = 0, .tied_embeddings = 1, .vision_bidirectional_local = 0,
    .logit_softcap = 0.0f, .activation_limit = 0.0f, .router_rank_order = 0,
    .router_kind = SALT_ROUTER_MODEL,
    .router_normalize_topk = 1, .router_scale = 1.0f,
    .eos = {LFM_EOS}, .n_eos = 1,
    /* The 30 non-attention layers are double-gated short-convolution blocks:
     * no heads, no RoPE and no score scale, with the kernel size in `window`.
     * The block arithmetic is model-owned (lfm_conv.c); this declares the
     * geometry so the layer plan and the KV plan are correct. */
    .sliding = {
        .kind = SALT_ATTN_MODEL, .causal = 1,
        .n_heads = 0, .n_kv_heads = 0, .head_dim = 0,
        .rope_kind = SALT_ROPE_NONE, .rope_dim = 0, .rope_base_dim = 0,
        .window = LFM_CONV_L, .raw_values = 1,
    },
    .full = {
        .kind = SALT_ATTN_FULL, .causal = 1,
        .n_heads = LFM_HEADS, .n_kv_heads = LFM_KV_HEADS, .head_dim = LFM_HEAD_DIM,
        .rope_dim = LFM_HEAD_DIM, .rope_base_dim = LFM_HEAD_DIM,
        .rope_kind = SALT_ROPE_DEFAULT, .window = 0,
        .rope_theta = LFM_ROPE_THETA, .score_scale = LFM_SCORE_SCALE,
        .raw_values = 1,
        /* The container carries self_attn.q_layernorm.weight AND
         * self_attn.k_layernorm.weight, (64,) each, one pair per attention
         * layer. Both are required; k_norm_absent stays 0. */
    },
};

static const SaltModelDesc MODEL = {
    .name = LFM_NAME, .runtime_ready = 1,
    .n_layers = LFM_LAYERS, .n_experts = LFM_EXPERTS, .topk = LFM_TOP_K,
    .n_shared = 0,
    .hidden = LFM_HIDDEN, .latent = LFM_HIDDEN, .moe_inter = LFM_MOE_INTERMEDIATE,
    .n_heads = LFM_HEADS, .n_kv_heads = LFM_KV_HEADS, .qk_rope = LFM_HEAD_DIM,
    .rope_theta = LFM_ROPE_THETA,
    .think_open = LFM_THINK_OPEN, .think_close = LFM_THINK_CLOSE,
    .text_graph = &TEXT, .state = &STATE,
};

const SaltModelDesc *salt_lfm2_model_descriptor(void) { return &MODEL; }
