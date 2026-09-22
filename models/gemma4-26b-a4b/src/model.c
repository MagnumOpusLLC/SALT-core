#include "salt/model.h"

static const SaltStateModelDesc GEMMA4_STATE = {
    .schema_version = SALT_STATE_SCHEMA_VERSION,
    .mindset_kind = SALT_MINDSET_PREFIX,
    .facts_kind = SALT_FACTS_KV_ROWS,
    .facts_optional = 1,
    .clear_facts_supported = 1,
    .hard_reset_supported = 1,
    .default_mindset_end = 0,
};

static const SaltTextGraphDesc GEMMA4_TEXT = {
    .period = 6,
    .full_slot = 5,
    .dense_intermediate = 2112,
    .expert_activation = SALT_ACT_GELU_TANH,
    .router_rmsnorm = 1,
    .parallel_dense_routed = 1,
    .residual_postnorm = 1,
    .final_layer_scale = 1,
    .tied_embeddings = 1,
    .vision_bidirectional_local = 1,
    .logit_softcap = 30.0f,
    .eos = {1, 106, 50, 0},
    .n_eos = 3,
    .sliding = {
        .kind = SALT_ATTN_SLIDING,
        .causal = 1,
        .n_heads = 16,
        .n_kv_heads = 8,
        .head_dim = 256,
        .rope_dim = 256,
        .rope_base_dim = 256,
        .rope_kind = SALT_ROPE_DEFAULT,
        .window = 1024,
        .shared_kv_projection = 0,
        .rope_theta = 10000.0,
        .score_scale = 1.0f,
    },
    .full = {
        .kind = SALT_ATTN_FULL,
        .causal = 1,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 512,
        .rope_dim = 128,
        .rope_base_dim = 512,
        .rope_kind = SALT_ROPE_PROPORTIONAL,
        .window = 0,
        .shared_kv_projection = 1,
        .rope_theta = 1000000.0,
        .score_scale = 1.0f,
    },
};

static const SaltVisionGraphDesc GEMMA4_VISION = {
    .n_layers = 27,
    .hidden = 1152,
    .intermediate = 4304,
    .n_heads = 16,
    .head_dim = 72,
    .patch_size = 16,
    .position_embedding_size = 10240,
    .pooling_kernel_size = 3,
    .default_soft_tokens = 280,
    .projection_output = 2816,
    .image_token_id = 258880,
    .boi_token_id = 255999,
    .eoi_token_id = 258882,
    .weights_bf16 = 1,
    .lazy = 1,
    .rope_theta = 100.0,
};

static const SaltModelDesc GEMMA4_MODEL = {
    .name = "gemma4-26b-a4b",
    .runtime_ready = 1,
    .n_layers = 30,
    .n_experts = 128,
    .topk = 8,
    .n_shared = 1,
    .hidden = 2816,
    .latent = 4096,
    .moe_inter = 704,
    .n_heads = 16,
    .n_kv_heads = 8,
    .qk_rope = 256,
    .rope_theta = 10000.0,
    .text_graph = &GEMMA4_TEXT,
    .vision_graph = &GEMMA4_VISION,
    .state = &GEMMA4_STATE,
};

const SaltModelDesc *salt_gemma4_model_descriptor(void) {
    return &GEMMA4_MODEL;
}
