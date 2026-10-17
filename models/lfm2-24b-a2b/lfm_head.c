#include "lfm_head.h"

#include "salt/bitmath.h"

int lfm_head_final_norm(const LfmHeadWeights *w, const float *x, float *y) {
    float mean = 0.0f;
    float inv;
    int i;

    if (!w || !x || !y || !w->norm_weight || w->hidden < 1)
        return -1;

    /* Deterministic: one fixed accumulation order, rsqrt through the engine. */
    for (i = 0; i < w->hidden; i++)
        mean += x[i] * x[i];
    mean /= (float)w->hidden;
    inv = salt_rsqrtf(mean + w->eps);
    if (inv != inv) /* NaN from a bad eps or an all-zero row */
        return -1;

    for (i = 0; i < w->hidden; i++) {
        float g = w->norm_weight[i];
        if (!w->norm_shifted)
            g = 1.0f + g;
        y[i] = x[i] * inv * g;
    }
    return 0;
}

int lfm_head_logits(const LfmHeadWeights *w, LfmAffineMatvec matvec,
                    const float *x, float *normed, float *logits) {
    if (!w || !matvec || !x || !normed || !logits)
        return -1;
    if (w->vocab < 1 || w->hidden < 1)
        return -1;
    if (!w->emb_vals || !w->emb_scales || !w->emb_biases)
        return -1;

    if (lfm_head_final_norm(w, x, normed))
        return -1;
    /* The tied head: the embedding matrix read as the output projection. */
    return matvec(w->emb_vals, w->emb_scales, w->emb_biases,
                  w->vocab, w->hidden, normed, logits) ? -1 : 0;
}
