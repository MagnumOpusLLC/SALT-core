#include "salt/model.h"

static const SaltStateModelDesc QWEN36_STATE = {
    .schema_version = SALT_STATE_SCHEMA_VERSION,
    .mindset_kind = SALT_MINDSET_RECURRENT,
    .facts_kind = SALT_FACTS_KV_ROWS,
    .facts_optional = 1,
    .clear_facts_supported = 1,
    .hard_reset_supported = 1,
    .default_mindset_end = 0,
};

static const SaltModelDesc QWEN36_MODEL = {
    .name = "qwen36",
    .runtime_ready = 1,
    .n_layers = 40,
    .n_experts = 256,
    .topk = 8,
    .n_shared = 1,
    .hidden = 2048,
    .latent = 2048,
    .moe_inter = 512,
    .n_heads = 16,
    .n_kv_heads = 2,
    .qk_rope = 64,
    .linear_num_key_heads = 16,
    .linear_num_value_heads = 32,
    .linear_key_head_dim = 128,
    .linear_value_head_dim = 128,
    .rope_theta = 10000000.0,
    .think_open = 248068,
    .think_close = 248069,
    .state = &QWEN36_STATE,
};

const SaltModelDesc *salt_qwen36_model_descriptor(void) {
    return &QWEN36_MODEL;
}
