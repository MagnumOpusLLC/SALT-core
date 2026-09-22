/*
 * simd.h -- accelerated kernel paths (issue #5).
 *
 * Two implementations of the hot kernels, both verified against the
 * scalar reference in kernels.c:
 *   - NEON on aarch64 (Apple M-series, this box)
 *   - AVX2 on x86-64 (the acer box, CI)
 *
 * Decode is a pure per-element function, so SIMD decode must be
 * BIT-IDENTICAL to scalar -- the gate enforces it. Matvec accumulation
 * order differs (vector lanes), so SIMD matvec is verified within a
 * tight fp32 tolerance; run-to-run determinism holds per backend.
 */
#ifndef SALT_SIMD_H
#define SALT_SIMD_H

#include <stdint.h>

/* Fixed per-row affine-Q4 scale/bias scratch. The current widest qualified
 * projection is Gemma 4 full-attention O with C=8192 (128 groups). Wider
 * rows stay on the scalar path until separately qualified. */
#define SALT_SIMD_Q4_MAX_GROUPS_PER_ROW 128

/* Non-zero when a SIMD path is compiled in AND available at runtime. */
int salt_simd_available(void);
/* Qualification control for x86 AVX2 independent-row packing. ARM and scalar
 * backends accept the setting but retain their existing realization. */
int salt_simd_set_q4_row_pair(int enabled);

/* Same contracts as the kernels.c scalar versions. */
void salt_simd_mx4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out);
void salt_simd_mx4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch);
void salt_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y);

/* SIMD matvecs (issue #6 step 4): I8 (int8 + optional E8M0 block
 * scales) and F8_E4M3 (two-table decode + masked subnormal/inf fixup).
 * The vector path is used when the scale blocks are 16-aligned
 * (SC % 16 == 0) or a single per-row scale (SC == 1); otherwise the
 * caller falls back to the scalar kernels.c path. */
void salt_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y);
void salt_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1);
void salt_simd_f8_matvec_bf16(const uint8_t *W, const uint16_t *scales,
                              int R, int C, int SR, int SC, const float *x,
                              float *y, int r0, int r1);
/* MLX 4-bit matvec (SIMD). rows [r0, r1). C % 8 == 0 and at most
 * SALT_SIMD_Q4_MAX_GROUPS_PER_ROW groups per row are required. */
void salt_simd_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                           const uint16_t *biases, int R, int C,
                           const float *x, float *y, int r0, int r1);

/* Weighted value accumulation for one attention head. output is already
 * zeroed or contains the preceding canonical value-position segment. SIMD
 * lanes own distinct output dimensions; every lane retains score order and
 * executes separate multiply then add operations. */
void salt_simd_weighted_value_accumulate(
    float *output, const float *values, int value_stride,
    const float *scores, int count, int dimension, float denominator);

#endif /* SALT_SIMD_H */
