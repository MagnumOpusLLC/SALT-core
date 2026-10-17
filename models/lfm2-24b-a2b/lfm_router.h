/* LFM2.5-24B-A2B routed-expert selection.
 *
 * Two things here differ from the usual MoE router, and both are pinned from
 * Lfm2MoeTopkRouter / the reference implementation rather than assumed:
 *
 *   1. Selection is on the BIASED score -- sigmoid(logits) + expert_bias.
 *   2. The gathered weights are the UNBIASED sigmoid of the same logits. The
 *      bias steers which experts are chosen; it does not scale their weight.
 *
 * Selection is then normalised over the chosen set when norm_topk_prob is set,
 * and scaled by routed_scaling_factor (1.0 for this model).
 *
 * Determinism: the scan uses a strict `>` so an exact tie keeps the lower
 * expert index. No unstable sort, no hash order, no floating-point comparison
 * that could differ across a recompile.
 *
 * The model owns this arithmetic. The engine owns traversal, resources and
 * publication. */

#ifndef SALT_LFM_ROUTER_H
#define SALT_LFM_ROUTER_H

#include <stdint.h>

#include "lfm.h"

typedef struct {
    const float *expert_bias; /* [n_experts] */
    int n_experts;
    int top_k;
    float routed_scaling;
    int norm_topk_prob;
} LfmRouterConfig;

/* The engine's deterministic exp, so a recompile cannot move the result. */
float lfm_sigmoid(float x);

/* Writes top_k indices in descending selection order and their weights in the
 * same order. Both arrays must hold top_k entries. Returns 0 or -1. */
int lfm_router_topk(const float *logits, const LfmRouterConfig *cfg,
                    int *out_index, float *out_weight);

/* Generic engine-seam shape. Selection/weights remain model-owned; this wrapper
 * only adapts fixed-width IDs and orders the chosen experts by canonical expert
 * id for deterministic engine grouping/reduction. */
int lfm_router_topk_seam(void *context, const float *logits,
                         const float *expert_bias, uint32_t experts,
                         uint32_t top_k, int32_t *selected, float *weights);

#endif /* SALT_LFM_ROUTER_H */
