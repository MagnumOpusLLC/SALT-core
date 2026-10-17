#ifndef SALT_LFM_QUANT_H
#define SALT_LFM_QUANT_H

/* LFM2.5-24B-A2B model-owned quant module: MLX affine group-64 6-bit.
 *
 * The core owns the SaltQuant interface and the registry (include/salt/quant.h,
 * src/quant.c); this module owns the format. The core never contains 6-bit
 * arithmetic: src/quant.c includes this header for the declaration and the
 * module itself is filled here.
 *
 * Storage: `rows` groups of 64 values. Each group of 64 six-bit codes is packed
 * little-endian into 12 consecutive uint32 words (64 * 6 = 384 bits), so one
 * row of `cols` values is `cols * 6 / 8 / 4` words. Scales and biases are BF16
 * with `cols / 64` entries per row, and dequantization is
 * `value = code * scale + bias` per group. */

#include "salt/quant.h"

#include <stdint.h>

#define LFM_AFFINE6_BITS 6
#define LFM_AFFINE6_GROUP 64
#define LFM_AFFINE6_WORDS_PER_GROUP 12

/* One 6-bit code, little-endian bit order within the uint32 stream. */
static inline unsigned lfm_affine6_code(const uint32_t *words, unsigned index) {
    unsigned bit = index * LFM_AFFINE6_BITS;
    unsigned shift = bit & 31u;
    uint64_t acc = words[bit >> 5];
    if (shift + LFM_AFFINE6_BITS > 32u)
        acc |= (uint64_t)words[(bit >> 5) + 1] << 32;
    return (unsigned)((acc >> shift) & 0x3fu);
}

/* The registry entry. `bits` is 6 so salt_quant_get(6) resolves it. */
extern const SaltQuant salt_quant_affine6;

/* Standalone dequantization of one row, for probes and oracles. Returns 0 on
 * success. `cols` must be a multiple of the group size. */
int lfm_affine6_dequant_row(const void *vals, const void *scales,
                            const void *biases, int cols, float *out);

/* The matvec pair, exposed so a probe can call the module directly. */
int lfm_affine6_matvec(const void *vals, const void *scales, const void *biases,
                       int rows, int cols, const float *x, float *y);
int lfm_affine6_matvec_batch(const void *vals, const void *scales,
                             const void *biases, int rows, int cols, int batch,
                             const float *xs, float *ys);
int lfm_affine6_matvec_batch_rows(const void *vals, const void *scales,
                                  const void *biases, int rows, int cols,
                                  int batch, const float *xs, float *ys,
                                  int row_start, int row_end);

#endif /* SALT_LFM_QUANT_H */
