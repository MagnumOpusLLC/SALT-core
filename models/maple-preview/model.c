#include "salt/model.h"

static const SaltStateModelDesc STATE = {
    .schema_version = SALT_STATE_SCHEMA_VERSION,
    .mindset_kind = SALT_MINDSET_PREFIX,
    .facts_kind = SALT_FACTS_KV_ROWS,
    .facts_optional = 1, .clear_facts_supported = 1,
    .hard_reset_supported = 1, .default_mindset_end = 0,
};
static const SaltTextGraphDesc TEXT = {
    .period = 4, .full_slot = 3, .dense_intermediate = 0,
    .expert_activation = SALT_ACT_SILU_CLAMPED, .activation_limit = 7.0f,
    .router_rank_order = 1, .eos = {151645}, .n_eos = 1,
    .sliding = {
        .kind = SALT_ATTN_SLIDING, .causal = 1,
        .n_heads = 16, .n_kv_heads = 4, .head_dim = 128,
        .rope_dim = 64, .rope_base_dim = 64,
        .rope_kind = SALT_ROPE_PARTIAL_F32, .window = 512,
        .rope_theta = 10000.0, .score_scale = 0.08838834764831845f,
        .raw_values = 1,
    },
    .full = {
        .kind = SALT_ATTN_FULL, .causal = 1,
        .n_heads = 16, .n_kv_heads = 4, .head_dim = 128,
        .rope_kind = SALT_ROPE_NONE, .rope_theta = 10000.0,
        .score_scale = 0.08838834764831845f, .raw_values = 1,
    },
};
static const SaltModelDesc MODEL = {
    .name = "maple-preview", .runtime_ready = 0,
    .n_layers = 24, .n_experts = 256, .topk = 8, .n_shared = 0,
    .hidden = 2048, .latent = 2048, .moe_inter = 512,
    .n_heads = 16, .n_kv_heads = 4, .qk_rope = 64,
    .rope_theta = 10000.0, .text_graph = &TEXT, .state = &STATE,
};
const SaltModelDesc *salt_maple_model_descriptor(void) { return &MODEL; }
