#include <stddef.h>

#include "lfm_qnorm.h"

#include "salt/bitmath.h"

static int qnorm_valid(const LfmQNorm *n) {
    return n && n->gain && n->n_heads > 0 && n->head_dim > 0;
}

/* One head: mean of squares over head_dim only, one fixed accumulation order,
 * the engine's rsqrt. */
static int norm_head(const LfmQNorm *n, const float *x, float *y) {
    float mean = 0.0f;
    float inv;
    int i;

    for (i = 0; i < n->head_dim; i++)
        mean += x[i] * x[i];
    mean /= (float)n->head_dim;
    inv = salt_rsqrtf(mean + n->eps);
    if (inv != inv)
        return -1;
    for (i = 0; i < n->head_dim; i++) {
        float g = n->gain[i];
        if (!n->shifted)
            g = 1.0f + g;
        y[i] = x[i] * inv * g;
    }
    return 0;
}

int lfm_q_layernorm(const LfmQNorm *n, const float *q, float *out) {
    int h;

    if (!qnorm_valid(n) || !q || !out)
        return -1;
    for (h = 0; h < n->n_heads; h++) {
        size_t off = (size_t)h * (size_t)n->head_dim;
        if (norm_head(n, q + off, out + off))
            return -1;
    }
    return 0;
}

int lfm_q_layernorm_inplace(const LfmQNorm *n, float *q) {
    int h;

    if (!qnorm_valid(n) || !q)
        return -1;
    for (h = 0; h < n->n_heads; h++) {
        size_t off = (size_t)h * (size_t)n->head_dim;
        if (norm_head(n, q + off, q + off))
            return -1;
    }
    return 0;
}
