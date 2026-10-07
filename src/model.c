#include "salt/model.h"
#include "salt/salt.h"
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

const SaltStateModelDesc *salt_model_state(const SaltModelDesc *model) {
    return model ? model->state : NULL;
}

void salt_model_apply(const SaltModelDesc *m, void *cfgv) {
    SaltCfg *cfg = (SaltCfg *)cfgv;
    if (!m || !cfg) return;
    if (cfg->n_layers <= 0)   cfg->n_layers = m->n_layers;
    if (cfg->n_experts <= 0)  cfg->n_experts = m->n_experts;
    if (cfg->topk <= 0)       cfg->topk = m->topk;
    if (cfg->n_shared <= 0)   cfg->n_shared = m->n_shared;
    if (cfg->hidden <= 0)     cfg->hidden = m->hidden;
    if (cfg->latent <= 0)     cfg->latent = m->latent;
    if (cfg->moe_inter <= 0)  cfg->moe_inter = m->moe_inter;
    if (cfg->n_heads <= 0)    cfg->n_heads = m->n_heads;
    if (cfg->n_kv_heads <= 0) cfg->n_kv_heads = m->n_kv_heads;
    if (cfg->qk_rope <= 0)    cfg->qk_rope = m->qk_rope;
    if (cfg->linear_num_key_heads <= 0)
        cfg->linear_num_key_heads = m->linear_num_key_heads;
    if (cfg->linear_num_value_heads <= 0)
        cfg->linear_num_value_heads = m->linear_num_value_heads;
    if (cfg->linear_key_head_dim <= 0)
        cfg->linear_key_head_dim = m->linear_key_head_dim;
    if (cfg->linear_value_head_dim <= 0)
        cfg->linear_value_head_dim = m->linear_value_head_dim;
    if (cfg->rope_theta <= 0.0) cfg->rope_theta = m->rope_theta;
}

int salt_model_linear_geometry(const SaltCfg *cfg, int *key_heads,
                               int *value_heads, int *key_dim,
                               int *value_dim, int *qkv_rows) {
    int64_t kh, vh, kd, vd, rows;
    if (!cfg || !key_heads || !value_heads || !key_dim || !value_dim ||
        !qkv_rows)
        return -1;
    kh = cfg->linear_num_key_heads;
    vh = cfg->linear_num_value_heads;
    kd = cfg->linear_key_head_dim;
    vd = cfg->linear_value_head_dim;
    if (kh < 1 || vh < 1 || kd < 1 || vd < 1 || vh % kh != 0 ||
        kh > INT_MAX || vh > INT_MAX || kd > INT_MAX || vd > INT_MAX ||
        kh > INT_MAX / (2 * kd))
        return -1;
    rows = 2 * kh * kd;
    if (vh > (INT_MAX - rows) / vd)
        return -1;
    rows += vh * vd;
    *key_heads = (int)kh;
    *value_heads = (int)vh;
    *key_dim = (int)kd;
    *value_dim = (int)vd;
    *qkv_rows = (int)rows;
    return 0;
}

int salt_model_gqa_geometry(const SaltCfg *cfg, int q_rows, int k_rows,
                            int v_rows, int o_rows, int o_cols,
                            int *heads, int *kv_heads, int *head_dim,
                            int *rope_dim) {
    int64_t nh, nkv, hd, rd;
    if (!cfg || !heads || !kv_heads || !head_dim || !rope_dim)
        return -1;
    nh = cfg->n_heads;
    nkv = cfg->n_kv_heads;
    rd = cfg->qk_rope;
    if (cfg->hidden < 1 || nh < 1 || nkv < 1 || nh % nkv != 0 ||
        k_rows < 1 || k_rows % nkv != 0)
        return -1;
    hd = k_rows / nkv;
    if (hd < 1 || hd > INT_MAX || 2 * nh * hd != q_rows ||
        nkv * hd != v_rows || o_rows != cfg->hidden ||
        nh * hd != o_cols || rd < 0 || rd > hd || rd % 2 != 0 ||
        (rd > 0 && (!(cfg->rope_theta > 0.0) ||
                    cfg->rope_theta > DBL_MAX)))
        return -1;
    *heads = (int)nh;
    *kv_heads = (int)nkv;
    *head_dim = (int)hd;
    *rope_dim = (int)rd;
    return 0;
}

static int attention_valid(const SaltAttentionDesc *a) {
    if (!a || (a->kind != SALT_ATTN_SLIDING &&
               a->kind != SALT_ATTN_FULL) ||
        (a->causal != 0 && a->causal != 1) ||
        a->n_heads < 1 || a->n_kv_heads < 1 ||
        a->n_heads % a->n_kv_heads != 0 || a->head_dim < 1 ||
        a->rope_dim < 0 || a->rope_dim > a->head_dim ||
        a->rope_dim % 2 != 0 || a->rope_base_dim < a->rope_dim ||
        a->rope_base_dim > a->head_dim ||
        (a->rope_kind != SALT_ROPE_DEFAULT &&
         a->rope_kind != SALT_ROPE_PROPORTIONAL &&
         a->rope_kind != SALT_ROPE_PARTIAL_F32 &&
         a->rope_kind != SALT_ROPE_NONE) ||
        (a->rope_kind == SALT_ROPE_NONE && a->rope_dim != 0) ||
        (a->raw_values != 0 && a->raw_values != 1) ||
        !(a->rope_theta > 0.0) || !isfinite(a->rope_theta) ||
        !(a->score_scale > 0.0f) || !isfinite(a->score_scale))
        return 0;
    if (a->kind == SALT_ATTN_SLIDING)
        return a->window > 0;
    return a->window == 0;
}

int salt_model_attention(const SaltModelDesc *m, int layer,
                         SaltAttentionDesc *out) {
    const SaltTextGraphDesc *g;
    if (!m || !out || layer < 0 || layer >= m->n_layers ||
        !(g = m->text_graph) || g->period < 1 || g->full_slot < 0 ||
        g->full_slot >= g->period)
        return -1;
    *out = (layer % g->period == g->full_slot) ? g->full : g->sliding;
    return attention_valid(out) ? 0 : -1;
}

static int bool_field(int value) {
    return value == 0 || value == 1;
}

int salt_model_text_layer_plan(const SaltModelDesc *m, int layer,
                               SaltTextLayerPlan *out) {
    const SaltTextGraphDesc *g;
    SaltTextLayerPlan plan;
    if (!m || !out || layer < 0 || layer >= m->n_layers ||
        !(g = m->text_graph) || !bool_field(m->runtime_ready) ||
        m->hidden < 1 || m->n_experts < 1 || m->topk < 1 ||
        m->topk > m->n_experts || m->moe_inter < 1 ||
        g->dense_intermediate < 0 ||
        (g->dense_intermediate == 0 && g->parallel_dense_routed) ||
        (g->expert_activation != SALT_ACT_SILU &&
         g->expert_activation != SALT_ACT_GELU_TANH &&
         g->expert_activation != SALT_ACT_SILU_CLAMPED) ||
        (g->expert_activation == SALT_ACT_SILU_CLAMPED &&
         (!(g->activation_limit > 0.0f) || !isfinite(g->activation_limit))) ||
        !bool_field(g->router_rank_order) ||
        !bool_field(g->router_rmsnorm) ||
        !bool_field(g->parallel_dense_routed) ||
        !bool_field(g->residual_postnorm) ||
        !bool_field(g->final_layer_scale) ||
        !bool_field(g->tied_embeddings) ||
        !bool_field(g->vision_bidirectional_local) ||
        !(g->logit_softcap >= 0.0f) || !isfinite(g->logit_softcap) ||
        g->n_eos < 1 || g->n_eos > 4)
        return -1;
    for (int i = 0; i < g->n_eos; i++)
        if (g->eos[i] < 0)
            return -1;
    memset(&plan, 0, sizeof plan);
    if (salt_model_attention(m, layer, &plan.attention) != 0)
        return -1;
    plan.layer = layer;
    plan.runtime_ready = m->runtime_ready;
    plan.hidden = m->hidden;
    plan.n_experts = m->n_experts;
    plan.top_k_experts = m->topk;
    plan.expert_intermediate = m->moe_inter;
    plan.dense_intermediate = g->dense_intermediate;
    plan.activation = g->expert_activation;
    plan.router_rmsnorm = g->router_rmsnorm;
    plan.parallel_dense_routed = g->parallel_dense_routed;
    plan.residual_postnorm = g->residual_postnorm;
    plan.final_layer_scale = g->final_layer_scale;
    plan.tied_embeddings = g->tied_embeddings;
    plan.vision_bidirectional_local = g->vision_bidirectional_local;
    plan.logit_softcap = g->logit_softcap;
    plan.activation_limit = g->activation_limit;
    plan.router_rank_order = g->router_rank_order;
    *out = plan;
    return 0;
}

int salt_model_text_plan(const SaltModelDesc *m, SaltTextLayerPlan *out,
                         size_t plan_len) {
    SaltTextLayerPlan *tmp;
    size_t count;
    if (!m || !out || m->n_layers < 1 ||
        plan_len < (size_t)m->n_layers ||
        (size_t)m->n_layers > SIZE_MAX / sizeof *tmp)
        return -1;
    count = (size_t)m->n_layers;
    tmp = (SaltTextLayerPlan *)malloc(count * sizeof *tmp);
    if (!tmp)
        return -1;
    for (int layer = 0; layer < m->n_layers; layer++) {
        if (salt_model_text_layer_plan(m, layer, &tmp[layer]) != 0) {
            free(tmp);
            return -1;
        }
    }
    memcpy(out, tmp, count * sizeof *tmp);
    free(tmp);
    return m->n_layers;
}

int salt_model_kv_plan(const SaltModelDesc *m, size_t max_tokens,
                       SaltKvLayerPlan *plan, size_t plan_len,
                       size_t *total_floats) {
    size_t off = 0;
    if (!m || !plan || !total_floats || max_tokens == 0 ||
        m->n_layers < 1 || plan_len < (size_t)m->n_layers)
        return -1;
    for (int layer = 0; layer < m->n_layers; layer++) {
        SaltAttentionDesc a;
        size_t tokens, kw, vw, width, count;
        if (salt_model_attention(m, layer, &a) != 0)
            return -1;
        tokens = max_tokens;
        if (a.kind == SALT_ATTN_SLIDING) {
            size_t retained;
            if (a.window < 2)
                return -1;
            retained = (size_t)a.window - 1;
            if (tokens > retained)
                tokens = retained;
        }
        kw = (size_t)a.n_kv_heads * (size_t)a.head_dim;
        vw = kw;
        if (kw > INT_MAX || vw > INT_MAX || kw > SIZE_MAX - vw)
            return -1;
        width = kw + vw;
        if (tokens > SIZE_MAX / width)
            return -1;
        count = tokens * width;
        if (off > SIZE_MAX - count)
            return -1;
        plan[layer].offset_floats = off;
        plan[layer].token_capacity = tokens;
        plan[layer].k_width = (int)kw;
        plan[layer].v_width = (int)vw;
        off += count;
    }
    *total_floats = off;
    return 0;
}
