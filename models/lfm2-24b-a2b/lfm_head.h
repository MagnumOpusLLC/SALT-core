/* LFM2.5-24B-A2B output head and final norm.
 *
 * The head is TIED: the container has no lm_head tensor, so the 6-bit affine
 * embedding matrix is the output projection. That is confirmed from the
 * safetensors headers, not assumed.
 *
 * The final norm runs after the last layer (embedding_norm). Its storage
 * convention is NOT assumed here: some containers store the gain w, others
 * store 1+w, and the two differ by exactly 1.0 on every element. Because that
 * difference changes logits and therefore KV, the convention is an explicit
 * field rather than a guess, and the gate tests both paths.
 *
 * Model-owned. The engine owns traversal, resources and publication. */

#ifndef SALT_LFM_HEAD_H
#define SALT_LFM_HEAD_H

#include "lfm.h"

typedef int (*LfmAffineMatvec)(const void *vals, const void *scales,
                               const void *bias, int rows, int cols,
                               const float *x, float *y);

typedef struct {
    /* The tied head: [vocab, hidden] affine, the same tensor as the embedding. */
    const void *emb_vals, *emb_scales, *emb_biases;
    const float *norm_weight; /* [hidden] */
    int vocab;
    int hidden;
    float eps;
    /* 1 = the stored norm gain is 1+w; 0 = it is w itself. */
    int norm_shifted;
} LfmHeadWeights;

/* normed must hold hidden floats; logits must hold vocab floats. Both are
 * allocated by the caller at initialization. */
int lfm_head_logits(const LfmHeadWeights *w, LfmAffineMatvec matvec,
                    const float *x, float *normed, float *logits);

/* The final norm alone, so the two stages can be tested apart. */
int lfm_head_final_norm(const LfmHeadWeights *w, const float *x, float *y);

#endif /* SALT_LFM_HEAD_H */
