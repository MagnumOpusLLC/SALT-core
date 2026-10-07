#include "salt/quant.h"
#include "salt/kernels.h"
#include "salt/simd.h"
#include "salt/int2.h"
#include <stddef.h>

/* The quant modules: thin adapters over the per-format matvecs so
 * the core never branches on the format. The row order, staged
 * dequant and FMA sequence are the per-format contracts -- the
 * adapters only carry the type casts. */

static int q4_matvec(const void *vals, const void *scales, const void *bias,
                     int R, int C, const float *x, float *y) {
    salt_q4_matvec((const uint32_t *)vals, (const uint16_t *)scales,
                   (const uint16_t *)bias, R, C, x, y);
    return 0;
}
static int q4_batch(const void *vals, const void *scales, const void *bias,
                    int R, int C, int B, const float *xs, float *ys) {
    return salt_q4_matvec_batch((const uint32_t *)vals,
                                (const uint16_t *)scales,
                                (const uint16_t *)bias, R, C, B, xs, ys);
}
static int q8_matvec(const void *vals, const void *scales, const void *bias,
                     int R, int C, const float *x, float *y) {
    salt_q8_matvec((const uint32_t *)vals, (const uint16_t *)scales,
                   (const uint16_t *)bias, R, C, x, y);
    return 0;
}
static int bf16_matvec(const void *vals, const void *scales, const void *bias,
                       int R, int C, const float *x, float *y) {
    (void)scales;
    salt_bf16_matvec((const uint16_t *)vals, R, C, x,
                     (const float *)bias, y);
    return 0;
}
static int f8_matvec_bf16(const void *vals, const void *scales,
                          const void *bias, int R, int C,
                          const float *x, float *y) {
    (void)bias;
    /* the official Qwen FP8 experts: BF16 weight_scale_inv block
     * scales, the pool's [SR, SC] grid (the moe's fmt-2 path). */
    salt_f8_matvec_bf16((const uint8_t *)vals, (const uint16_t *)scales,
                        R, C, salt_quant_fp8.sr, salt_quant_fp8.sc, x, y);
    return 0;
}

static int int2_batch(const void *v, const void *s, const void *bias,
                     int R, int C, int B, const float *x, float *y) {
    if (bias) return -1;
    return salt_int2_matvec_batch_rows(v, s, R, C, B, x, y, 0, R);
}
static int int2_matvec(const void *v, const void *s, const void *bias,
                      int R, int C, const float *x, float *y) {
    return int2_batch(v, s, bias, R, C, 1, x, y);
}
const SaltQuant salt_quant_int2 = {
    2, 0, 8, 0, 0, "row-int2", int2_matvec, int2_batch };

const SaltQuant salt_quant_q4 = {
    4, 1769472, 32, 0, 0, "mlx4", q4_matvec, q4_batch };
const SaltQuant salt_quant_q8 = {
    8, 3342336, 0, 0, 0, "mlx8", q8_matvec, NULL };
const SaltQuant salt_quant_bf16 = {
    16, 4096, 0, 0, 0, "bf16", bf16_matvec, NULL };
const SaltQuant salt_quant_fp8 = {
    8, 1769472, 0, 4, 16, "fp8", f8_matvec_bf16, NULL };

const SaltQuant *salt_quant_get(int bits) {
    switch (bits) {
        case 2:  return &salt_quant_int2;
        case 4:  return &salt_quant_q4;
        case 8:  return &salt_quant_q8;
        case 16: return &salt_quant_bf16;
        default: return NULL;
    }
}
