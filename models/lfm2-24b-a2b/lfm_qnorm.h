/* LFM2.5-24B-A2B per-head Q layernorm.
 *
 * `self_attn.q_layernorm.weight` is (64,) -- head_dim -- and applies to Q
 * BEFORE attention, independently per head. There is no K norm in this
 * container; the earlier note claiming a QK norm was wrong.
 *
 * This is model-owned because the engine's attention path has no per-head
 * pre-normalisation stage. The arithmetic uses the engine's deterministic
 * salt_rsqrtf, so it is the same reciprocal square root as every other norm in
 * the engine and not a second implementation of it.
 *
 * Storage convention is explicit, exactly as in lfm_head: some containers store
 * the gain w, others store 1+w, and the difference changes logits and KV. */

#ifndef SALT_LFM_QNORM_H
#define SALT_LFM_QNORM_H

#include "lfm.h"

typedef struct {
    const float *gain; /* [head_dim] */
    int n_heads;
    int head_dim;
    float eps;
    int shifted; /* 1 = the stored gain is 1+w; 0 = it is w itself */
} LfmQNorm;

/* Normalises each head of q independently, in place. `q` holds n_heads *
 * head_dim floats. */
int lfm_q_layernorm_inplace(const LfmQNorm *n, float *q);

/* Same, into a separate buffer, so a caller can keep the input. */
int lfm_q_layernorm(const LfmQNorm *n, const float *q, float *out);

#endif /* SALT_LFM_QNORM_H */
