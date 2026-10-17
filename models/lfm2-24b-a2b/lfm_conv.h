#ifndef SALT_LFM_CONV_H
#define SALT_LFM_CONV_H

/* LFM2.5-24B-A2B model-owned short-convolution block.
 *
 * The engine owns no conv arithmetic. This module owns the block semantics --
 * the double gate and the depthwise causal convolution with its recurrent
 * window -- and reuses the engine's existing quantized matvec for the two
 * projections. It never implements a matvec of its own.
 *
 * Pinned from transformers models/lfm2_moe/modeling_lfm2_moe.py, class
 * Lfm2MoeShortConv and the causal_conv1d_fn/update kernels:
 *
 *   BCx = in_proj(x)                  # 3 * hidden rows
 *   B, C, x = BCx.chunk(3, dim=-1)
 *   h = B * x
 *   h = depthwise_causal_conv1d(h, weight[hidden][kernel], bias=None,
 *                               padding=kernel-1)[:, :, :seq_len]
 *   y = C * h
 *   y = out_proj(y)
 *
 * conv1d with padding = kernel-1 truncated to seq_len is a causal FIR:
 *
 *   out[t] = sum_j weight[c][j] * in[t - (kernel-1) + j]
 *
 * so position t reads x[t-kernel+1 .. t] and the recurrent window is the last
 * kernel-1 inputs per channel. There is no bias (conv_bias is false). */

#include <stddef.h>

#include "salt/text_verify.h" /* SaltTextModelLayerOps: the engine's seam type */
#include "lfm.h"

#define LFM_CONV_KERNEL 3

typedef struct {
    int hidden;
    int kernel;
    /* [hidden][kernel], most recent in the last slot; zeroed at session start. */
    float *window;
} LfmConvWindow;

/* Bytes one window needs, so a caller can size its arena at initialization. */
size_t lfm_conv_window_bytes(int hidden, int kernel);

/* Attach preallocated startup memory. The caller owns the storage; this module
 * never allocates. */
int lfm_conv_window_init(LfmConvWindow *w, void *memory, size_t bytes,
                         int hidden, int kernel);

/* Clear the window to session start (all zeros). */
void lfm_conv_window_reset(LfmConvWindow *w);

/* One position. b, c and x are the three projected thirds (each `hidden`
 * floats, already computed by the engine's matvec); kernel is the depthwise
 * weight laid out [hidden][kernel]; y is `hidden` floats of output. The window
 * advances. Returns 0 on success. */
int lfm_conv_step(const float *b, const float *c, const float *x,
                  const float *kernel, LfmConvWindow *w, float *y);

/* `rows` consecutive positions from a zero-initialized window, matching the
 * prefill path where the convolution runs over the whole prompt. */
int lfm_conv_sequence(const float *b, const float *c, const float *x,
                      const float *kernel, LfmConvWindow *w, int rows, float *y);

/* ---- the layer seam the core calls ---------------------------------------
 *
 * One complete LFM short-convolution layer. The core owns traversal, resource
 * readiness, dependencies, canonical destinations and publication; it calls
 * this and gets the layer's output. All of the layer's arithmetic is here, in
 * the model's src -- the core carries no conv logic at all.
 *
 * The projections are supplied as the engine's own affine matvec through a
 * function pointer, so this module never implements a second matvec and the
 * quantized track stays the engine's. */

typedef int (*LfmAffineMatvec)(const void *vals, const void *scales,
                               const void *bias, int rows, int cols,
                               const float *x, float *y);
typedef int (*LfmAffineMatvecBatch)(void *context,
                                    const void *vals, const void *scales,
                                    const void *bias, int rows, int cols,
                                    int batch, const float *inputs,
                                    float *outputs);

typedef struct {
    /* in_proj: [3 * hidden, hidden]; out_proj: [hidden, hidden]; both affine. */
    const void *in_vals, *in_scales, *in_biases;
    const void *out_vals, *out_scales, *out_biases;
    /* Depthwise kernel, [hidden][kernel_size], already expanded to float. */
    const float *kernel;
    int hidden;
    int kernel_size;
} LfmLayerWeights;

/* scratch must hold 4 * hidden floats; the caller allocates it at startup. */
int lfm_conv_layer_forward(const LfmLayerWeights *weights, LfmAffineMatvec matvec,
                           LfmConvWindow *w, const float *x, float *y,
                           float *scratch);

/* Bytes one layer's scratch needs, so a caller can size its arena. */
size_t lfm_conv_layer_scratch_bytes(int hidden);

/* ---- the seam the core calls --------------------------------------------
 *
 * The core holds a pointer to this table and invokes forward(). It never
 * inspects the layer kind, the weights, the window or the gating -- it passes
 * a layer index, a row count and canonical addresses, and gets an output. All
 * of the model's knowledge stays on this side of the call. */
typedef SaltTextModelLayerOps LfmLayerOps;

/* The model's own per-layer state. The core never reads it. */
typedef struct {
    LfmLayerWeights weights[LFM_LAYERS];
    LfmAffineMatvec matvec;
    LfmAffineMatvecBatch matvec_batch;
    void *matvec_context;
    int hidden;
} LfmLayerContext;

/* Model-side startup: bind the context and return the table the core holds.
 * lfm_layer_ops() returns NULL until this has run, so the core can fail closed
 * rather than call through an unbound seam. */
int lfm_layer_ops_init(const LfmLayerContext *context);
const LfmLayerOps *lfm_layer_ops(void);

#endif /* SALT_LFM_CONV_H */
