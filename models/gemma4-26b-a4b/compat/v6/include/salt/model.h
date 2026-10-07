#ifndef SALT_MODEL_H
#define SALT_MODEL_H

#include "salt/salt.h"   /* SaltCfg */
#include "salt/state.h"

#include <stddef.h>

/* The model registry: the named model entries as DATA. The engine
 * loads the geometry from the config JSON when present and fills
 * the gaps from the registry entry -- a new model is a new entry,
 * zero engine changes. The layer registry (salt/layer.h) and the
 * quant registry (salt/quant.h) are the other two surfaces; this
 * is the geometry surface. */

typedef enum {
    SALT_ATTN_NONE = 0,
    SALT_ATTN_SLIDING = 1,
    SALT_ATTN_FULL = 2
} SaltAttentionKind;

typedef enum {
    SALT_ACT_NONE = 0,
    SALT_ACT_SILU = 1,
    SALT_ACT_GELU_TANH = 2
} SaltActivationKind;

typedef enum {
    SALT_ROPE_NONE = 0,
    SALT_ROPE_DEFAULT = 1,
    SALT_ROPE_PROPORTIONAL = 2
} SaltRopeKind;

typedef struct {
    SaltAttentionKind kind;
    int causal;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int rope_dim;
    int rope_base_dim;
    SaltRopeKind rope_kind;
    int window;
    int shared_kv_projection;
    double rope_theta;
    float score_scale;
} SaltAttentionDesc;

typedef struct {
    int period;
    int full_slot;
    int dense_intermediate;
    SaltActivationKind expert_activation;
    int router_rmsnorm;
    int parallel_dense_routed;
    int residual_postnorm;
    int final_layer_scale;
    int tied_embeddings;
    int vision_bidirectional_local;
    float logit_softcap;
    int eos[4];
    int n_eos;
    SaltAttentionDesc sliding;
    SaltAttentionDesc full;
} SaltTextGraphDesc;

typedef struct {
    size_t offset_floats;
    size_t token_capacity;
    int k_width;
    int v_width;
} SaltKvLayerPlan;

/* Tensor-index-free execution policy for one text layer. This is the boundary
 * between model semantics and an eventual package-specific loader/dispatcher.
 * runtime_ready is carried into every row so a qualified descriptor cannot be
 * mistaken for a runnable model package. */
typedef struct {
    int layer;
    int runtime_ready;
    int hidden;
    SaltAttentionDesc attention;
    int n_experts;
    int top_k_experts;
    int expert_intermediate;
    int dense_intermediate;
    SaltActivationKind activation;
    int router_rmsnorm;
    int parallel_dense_routed;
    int residual_postnorm;
    int final_layer_scale;
    int tied_embeddings;
    int vision_bidirectional_local;
    float logit_softcap;
} SaltTextLayerPlan;

typedef struct {
    int n_layers;
    int hidden;
    int intermediate;
    int n_heads;
    int head_dim;
    int patch_size;
    int position_embedding_size;
    int pooling_kernel_size;
    int default_soft_tokens;
    int projection_output;
    int image_token_id;
    int boi_token_id;
    int eoi_token_id;
    int weights_bf16;
    int lazy;
    double rope_theta;
} SaltVisionGraphDesc;

typedef struct {
    const char *name;        /* "qwen36" */
    int n_layers;
    int n_experts, topk, n_shared;
    int hidden, latent, moe_inter;
    int n_heads, n_kv_heads, qk_rope;
    int linear_num_key_heads, linear_num_value_heads;
    int linear_key_head_dim, linear_value_head_dim;
    double rope_theta;
    int think_open, think_close;  /* the reasoning tokens (0 = none) */
    /* Extensions stay after the legacy descriptor prefix above. */
    int runtime_ready;       /* 0 = descriptor/tests only; cfg load refuses */
    const SaltTextGraphDesc *text_graph;
    const SaltVisionGraphDesc *vision_graph;
    const SaltStateModelDesc *state;
} SaltModelDesc;

/* the named registry lookup (NULL = unknown) */
const SaltModelDesc *salt_model_get(const char *name);
const char *salt_model_default_name(void);
const SaltStateModelDesc *salt_model_state(const SaltModelDesc *model);

/* fill a SaltCfg's geometry fields from the entry (the caller's
 * JSON-loaded values win -- the registry fills only the gaps). */
void salt_model_apply(const SaltModelDesc *m, void *cfg);

/* Validate and return model-owned linear-attention geometry. qkv_rows is
 * 2 * key_heads * key_dim + value_heads * value_dim. */
int salt_model_linear_geometry(const SaltCfg *cfg, int *key_heads,
                               int *value_heads, int *key_dim,
                               int *value_dim, int *qkv_rows);

/* Validate model-owned GQA geometry against active tensor dimensions. */
int salt_model_gqa_geometry(const SaltCfg *cfg, int q_rows, int k_rows,
                            int v_rows, int o_rows, int o_cols,
                            int *heads, int *kv_heads, int *head_dim,
                            int *rope_dim);

/* Resolve a model-owned attention layer. This is deliberately separate from
 * the legacy one-geometry GQA helper: mixed-geometry models must not be
 * coerced through Qwen's doubled/gated Q projection contract. */
int salt_model_attention(const SaltModelDesc *m, int layer,
                         SaltAttentionDesc *out);

/* Build model-owned text-layer policy without binding any trunk tensor index.
 * The batch form returns m->n_layers and leaves the complete caller array
 * unchanged on failure. A runtime-blocked model may be planned for structural
 * qualification; consumers must honor each row's runtime_ready gate. */
int salt_model_text_layer_plan(const SaltModelDesc *m, int layer,
                               SaltTextLayerPlan *out);
int salt_model_text_plan(const SaltModelDesc *m, SaltTextLayerPlan *out,
                         size_t plan_len);

/* Plan heterogeneous persistent per-layer K/V storage. Before appending the
 * current query, sliding layers retain at most window - 1 prior tokens; full
 * layers retain max_tokens. Offsets and the total are in float elements, not
 * bytes. */
int salt_model_kv_plan(const SaltModelDesc *m, size_t max_tokens,
                       SaltKvLayerPlan *plan, size_t plan_len,
                       size_t *total_floats);

#endif /* SALT_MODEL_H */