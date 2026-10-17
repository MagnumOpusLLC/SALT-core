#include "lfm_router.h"

#include "salt/bitmath.h"

float lfm_sigmoid(float x) {
    return 1.0f / (1.0f + salt_expf(-x));
}

int lfm_router_topk(const float *logits, const LfmRouterConfig *cfg,
                    int *out_index, float *out_weight) {
    int k, i, e;

    if (!logits || !cfg || !out_index || !out_weight)
        return -1;
    if (!cfg->expert_bias || cfg->n_experts < 1)
        return -1;
    k = cfg->top_k;
    if (k < 1 || k > cfg->n_experts)
        return -1;

    for (i = 0; i < k; i++) {
        out_index[i] = -1;
        out_weight[i] = 0.0f;
    }

    /* Select on sigmoid(logits) + expert_bias. Strict `>` keeps the lower
     * index on an exact tie. */
    for (i = 0; i < k; i++) {
        int best = -1;
        float best_score = 0.0f;
        for (e = 0; e < cfg->n_experts; e++) {
            float score;
            int taken = 0;
            for (int j = 0; j < i; j++)
                if (out_index[j] == e) {
                    taken = 1;
                    break;
                }
            if (taken)
                continue;
            score = lfm_sigmoid(logits[e]) + cfg->expert_bias[e];
            if (best < 0 || score > best_score) {
                best = e;
                best_score = score;
            }
        }
        if (best < 0)
            return -1;
        out_index[i] = best;
    }

    /* The weight is the unbiased sigmoid of the same logits. */
    for (i = 0; i < k; i++)
        out_weight[i] = lfm_sigmoid(logits[out_index[i]]);

    if (cfg->norm_topk_prob) {
        float sum = 0.0f;
        for (i = 0; i < k; i++)
            sum += out_weight[i];
        if (sum != 0.0f)
            for (i = 0; i < k; i++)
                out_weight[i] /= sum;
    }
    if (cfg->routed_scaling != 0.0f)
        for (i = 0; i < k; i++)
            out_weight[i] *= cfg->routed_scaling;

    return 0;
}

int lfm_router_topk_seam(void *context, const float *logits,
                         const float *expert_bias, uint32_t experts,
                         uint32_t top_k, int32_t *selected, float *weights) {
    LfmRouterConfig cfg;
    int ids[LFM_TOP_K];
    float routed_weights[LFM_TOP_K];
    (void)context;
    if (!logits || !expert_bias || !selected || !weights ||
        experts < 1u || experts > INT32_MAX || top_k < 1u ||
        top_k > LFM_TOP_K || top_k > experts)
        return -1;
    cfg.expert_bias = expert_bias;
    cfg.n_experts = (int)experts;
    cfg.top_k = (int)top_k;
    cfg.routed_scaling = 1.0f;
    cfg.norm_topk_prob = 1;
    if (lfm_router_topk(logits, &cfg, ids, routed_weights) != 0)
        return -1;
    for (uint32_t rank = 0u; rank < top_k; rank++) {
        selected[rank] = (int32_t)ids[rank];
        weights[rank] = routed_weights[rank];
    }
    for (uint32_t left = 0u; left < top_k; left++)
        for (uint32_t right = left + 1u; right < top_k; right++)
            if (selected[right] < selected[left]) {
                int32_t id = selected[left];
                float weight = weights[left];
                selected[left] = selected[right];
                weights[left] = weights[right];
                selected[right] = id;
                weights[right] = weight;
            }
    return 0;
}
