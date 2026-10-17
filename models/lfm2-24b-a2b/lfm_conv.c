/* LFM2.5-24B-A2B model-owned short-convolution block.
 *
 * Arithmetic contract: per channel, an ascending tap sum over the window with
 * the most recent input last, then the second gate. The build uses
 * -ffp-contract=off, so no multiply-add is contracted and the sequence is
 * reproducible. Nothing here is shared with any other model: a conv block is
 * LFM's, and the engine carries no conv arithmetic at all. */

#include "lfm_conv.h"
#include "lfm_router.h"

#include <string.h>

size_t lfm_conv_window_bytes(int hidden, int kernel) {
    if (hidden < 1 || kernel < 1)
        return 0;
    return (size_t)hidden * (size_t)kernel * sizeof(float);
}

int lfm_conv_window_init(LfmConvWindow *w, void *memory, size_t bytes,
                         int hidden, int kernel) {
    if (!w || !memory || hidden < 1 || kernel < 1)
        return -1;
    if (bytes < lfm_conv_window_bytes(hidden, kernel))
        return -1;
    w->hidden = hidden;
    w->kernel = kernel;
    w->window = (float *)memory;
    memset(w->window, 0, lfm_conv_window_bytes(hidden, kernel));
    return 0;
}

void lfm_conv_window_reset(LfmConvWindow *w) {
    if (w && w->window)
        memset(w->window, 0, lfm_conv_window_bytes(w->hidden, w->kernel));
}

int lfm_conv_step(const float *b, const float *c, const float *x,
                  const float *kernel, LfmConvWindow *w, float *y) {
    if (!b || !c || !x || !kernel || !y || !w || !w->window)
        return -1;
    for (int ch = 0; ch < w->hidden; ch++) {
        float *win = w->window + (size_t)ch * (size_t)w->kernel;
        const float *k = kernel + (size_t)ch * (size_t)w->kernel;
        float acc = 0.0f;
        /* Shift the window and append the gated input. */
        for (int j = 0; j < w->kernel - 1; j++)
            win[j] = win[j + 1];
        win[w->kernel - 1] = b[ch] * x[ch];
        /* Causal FIR over the window, ascending taps. */
        for (int j = 0; j < w->kernel; j++)
            acc += k[j] * win[j];
        y[ch] = c[ch] * acc;
    }
    return 0;
}

int lfm_conv_sequence(const float *b, const float *c, const float *x,
                      const float *kernel, LfmConvWindow *w, int rows, float *y) {
    if (!b || !c || !x || !kernel || !y || !w || rows < 0)
        return -1;
    for (int t = 0; t < rows; t++) {
        size_t off = (size_t)t * (size_t)w->hidden;
        if (lfm_conv_step(b + off, c + off, x + off, kernel, w, y + off))
            return -1;
    }
    return 0;
}

size_t lfm_conv_layer_scratch_bytes(int hidden) {
    if (hidden < 1)
        return 0;
    return (size_t)4 * (size_t)hidden * sizeof(float);
}

int lfm_conv_layer_forward(const LfmLayerWeights *weights, LfmAffineMatvec matvec,
                           LfmConvWindow *w, const float *x, float *y,
                           float *scratch) {
    float *bcx, *gated;
    if (!weights || !matvec || !w || !x || !y || !scratch)
        return -1;
    if (weights->hidden < 1 || weights->kernel_size < 1 ||
        !weights->in_vals || !weights->in_scales || !weights->in_biases ||
        !weights->out_vals || !weights->out_scales || !weights->out_biases ||
        !weights->kernel)
        return -1;
    if (w->hidden != weights->hidden || w->kernel != weights->kernel_size)
        return -1;

    bcx = scratch;                     /* 3 * hidden */
    gated = scratch + 3 * (size_t)weights->hidden;  /* hidden */

    /* in_proj(x) -> (B, C, x), exactly as the pinned chunk(3, dim=-1) order. */
    if (matvec(weights->in_vals, weights->in_scales, weights->in_biases,
               3 * weights->hidden, weights->hidden, x, bcx))
        return -1;
    /* h = B * x; h = causal_conv(h); y = C * h */
    if (lfm_conv_step(bcx, bcx + weights->hidden, bcx + 2 * (size_t)weights->hidden,
                      weights->kernel, w, gated))
        return -1;
    /* out_proj(y) */
    if (matvec(weights->out_vals, weights->out_scales, weights->out_biases,
                  weights->hidden, weights->hidden, gated, y))
        return -1;
    return 0;
}

/* ---- the registered seam ------------------------------------------------ */

static int lfm_ops_validate(void *context, uint32_t layer,
                            const SaltTextLayerPlan *plan,
                            const SaltTextLayerExecDesc *descriptor);
static int lfm_ops_forward(void *context, uint32_t layer, uint32_t rows,
                           const float *x, float *y,
                           float *scratch, size_t scratch_float_capacity,
                           void *state, size_t state_bytes);
static LfmLayerOps g_ops = {
    .validate = lfm_ops_validate,
    .forward = lfm_ops_forward,
    .router_topk = lfm_router_topk_seam,
    .context = NULL,
};

static int lfm_ops_validate(void *context, uint32_t layer,
                            const SaltTextLayerPlan *plan,
                            const SaltTextLayerExecDesc *descriptor) {
    const LfmLayerContext *c = (const LfmLayerContext *)context;
    if (!plan || !descriptor || layer >= LFM_LAYERS ||
        !lfm_is_conv_layer((int)layer) ||
        (c && c->hidden != LFM_HIDDEN) ||
        plan->attention.kind != SALT_ATTN_MODEL ||
        plan->attention.window != LFM_CONV_L ||
        descriptor->model_input_projection.rows != 3u * LFM_HIDDEN ||
        descriptor->model_input_projection.cols != LFM_HIDDEN ||
        descriptor->model_kernel.rows != LFM_HIDDEN ||
        descriptor->model_kernel.cols != LFM_CONV_L ||
        descriptor->model_output_projection.rows != LFM_HIDDEN ||
        descriptor->model_output_projection.cols != LFM_HIDDEN ||
        descriptor->model_scratch_floats_per_row != 4u * LFM_HIDDEN ||
        descriptor->model_state_bytes !=
            (size_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float))
        return -1;
    return 0;
}

static int lfm_ops_forward(void *context, uint32_t layer, uint32_t rows,
                           const float *x, float *y,
                           float *scratch, size_t scratch_float_capacity,
                           void *state, size_t state_bytes) {
    const LfmLayerContext *c = (const LfmLayerContext *)context;
    const LfmLayerWeights *weights;
    LfmConvWindow window;
    float *bcx, *gated;
    size_t required;
    if (!c || !c->matvec || !c->matvec_batch || c->hidden < 1 ||
        !scratch || !state)
        return -1;
    if (!lfm_is_conv_layer((int)layer) || rows < 1u || !x || !y)
        return -1;
    if ((size_t)rows > SIZE_MAX / (4u * (size_t)c->hidden))
        return -1;
    required = (size_t)rows * 4u * (size_t)c->hidden;
    if (scratch_float_capacity < required ||
        state_bytes != lfm_conv_window_bytes(c->hidden, LFM_CONV_L))
        return -1;
    weights = &c->weights[layer];
    window.hidden = c->hidden;
    window.kernel = LFM_CONV_L;
    window.window = (float *)state;
    bcx = scratch;
    gated = scratch + (size_t)rows * 3u * (size_t)c->hidden;
    if (c->matvec_batch(c->matvec_context,
            weights->in_vals, weights->in_scales,
            weights->in_biases, 3 * c->hidden, c->hidden, (int)rows,
            x, bcx) != 0)
        return -1;
    for (uint32_t row = 0; row < rows; row++) {
        float *row_bcx = bcx + (size_t)row * 3u * (size_t)c->hidden;
        if (lfm_conv_step(row_bcx, row_bcx + c->hidden,
                row_bcx + 2u * (size_t)c->hidden,
                weights->kernel, &window,
                gated + (size_t)row * (size_t)c->hidden) != 0)
            return -1;
    }
    return c->matvec_batch(c->matvec_context,
        weights->out_vals, weights->out_scales,
        weights->out_biases, c->hidden, c->hidden, (int)rows,
        gated, y);
}

int lfm_layer_ops_init(const LfmLayerContext *context) {
    if (!context || !context->matvec || !context->matvec_batch ||
        context->hidden < 1)
        return -1;
    for (int layer = 0; layer < LFM_LAYERS; layer++) {
        const LfmLayerWeights *weights;
        if (!lfm_is_conv_layer(layer))
            continue;
        weights = &context->weights[layer];
        if (weights->hidden != context->hidden ||
            weights->kernel_size != LFM_CONV_L || !weights->in_vals ||
            !weights->in_scales || !weights->in_biases ||
            !weights->out_vals || !weights->out_scales ||
            !weights->out_biases || !weights->kernel)
            return -1;
    }
    g_ops.context = (void *)context;
    return 0;
}

const LfmLayerOps *lfm_layer_ops(void) {
    return &g_ops;
}
