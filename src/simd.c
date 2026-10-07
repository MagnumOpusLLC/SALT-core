/* simd.c -- NEON (aarch64) and AVX2 (x86-64) kernel paths (issue #5).
 *
 * Layout reminder (mxfp4-pool-v1): element i lives in byte i/2, nibble
 * (i&1 ? high : low); one E8M0 scale byte per bsize elements. Nibble
 * 0-7 magnitude table {0,0.5,1,1.5,2,3,4,6}, bit 3 = negative sign.
 *
 * SIMD strategy: process 16-element groups (8 bytes = 16 nibbles).
 * Even elements are the low nibbles, odd the high nibbles. A signed
 * LUT (2x magnitude, sign folded in) is indexed via pshufb/vtbl, then
 * byte->float widened, halved, scaled. Decode is order-free -> must be
 * bit-identical to the scalar kernel. Matvec uses mul+add (never FMA)
 * per lane so only accumulation ORDER differs from scalar.
 */
#include "salt/simd.h"
#include "compiler.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

/* 2x magnitudes with sign folded in (nibble 8..15 = negative 0..7). */
static const uint8_t MAG2S[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,           /* +0, +0.5, +1, +1.5, +2, +3, +4, +6 */
    0, 255, 254, 253, 252, 250, 248, 244 /* -0, -0.5, -1, -1.5, -2, -3, -4, -6 */
};

static int g_q4_row_pair = 1;

int salt_simd_set_q4_row_pair(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    g_q4_row_pair = enabled;
    return 0;
}

static float e8m0f(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);   /* 2^(b-127); b=0 -> 2^-127 */
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

#if !defined(__aarch64__)
static float f8_e4m3f(uint8_t byte) {
    int exponent = (byte >> 3) & 0x0f;
    int mantissa = byte & 7;
    float value;
    if (exponent == 0)
        value = (float)mantissa * 0.001953125f;
    else if (exponent == 0x0f)
        value = 448.0f;
    else
        value = (1.0f + (float)mantissa * 0.125f) *
                ldexpf(1.0f, exponent - 7);
    return (byte & 0x80u) ? -value : value;
}

static void scalar_f8_matvec_bf16(
        const uint8_t *weights, const uint16_t *scales,
        int rows, int columns, int scale_rows, int scale_columns,
        const float *input, float *output, int row_start, int row_end) {
    if (row_start < 0) row_start = 0;
    if (row_end > rows) row_end = rows;
    if (scale_rows < 1) scale_rows = 1;
    if (scale_columns < 1) scale_columns = 1;
    for (int row = row_start; row < row_end; row++) {
        int scale_row = (int)(((int64_t)row * scale_rows) / rows);
        const uint8_t *weight_row = weights + (size_t)row * columns;
        float accumulator[16] = {0};
        int column = 0;
        for (; column + 15 < columns; column += 16) {
            int scale_column =
                (int)(((int64_t)column * scale_columns) / columns);
            float scale = scales ?
                bf16_f(scales[scale_row * scale_columns + scale_column]) : 1.0f;
            for (int lane = 0; lane < 16; lane++) {
                float term = f8_e4m3f(weight_row[column + lane]) * scale;
                term = term * input[column + lane];
                accumulator[lane] += term;
            }
        }
        float pair0 = accumulator[0] + accumulator[2];
        float pair1 = accumulator[1] + accumulator[3];
        float pair2 = accumulator[4] + accumulator[6];
        float pair3 = accumulator[5] + accumulator[7];
        float sum0 = pair0 + pair2;
        float sum1 = pair1 + pair3;
        sum0 += accumulator[8] + accumulator[10];
        sum1 += accumulator[9] + accumulator[11];
        sum0 += accumulator[12] + accumulator[14];
        sum1 += accumulator[13] + accumulator[15];
        float sum = sum0 + sum1;
        for (; column < columns; column++) {
            int scale_column =
                (int)(((int64_t)column * scale_columns) / columns);
            float scale = scales ?
                bf16_f(scales[scale_row * scale_columns + scale_column]) : 1.0f;
            sum += f8_e4m3f(weight_row[column]) * scale * input[column];
        }
        output[row] = sum;
    }
}
#endif

static uint16_t load_u16_unaligned(const uint16_t *base, size_t index) {
    uint16_t value;
    memcpy(&value, (const unsigned char *)(const void *)base + index * 2,
           sizeof value);
    return value;
}

static uint32_t load_u32_unaligned(const uint32_t *base, size_t index) {
    uint32_t value;
    memcpy(&value, (const unsigned char *)(const void *)base + index * 4,
           sizeof value);
    return value;
}

/* ------------------------------------------------------------------ */
/* scalar tail (any length)                                            */
/* ------------------------------------------------------------------ */
#if !defined(__aarch64__) && !defined(__x86_64__) && !defined(__i386__)
static void scalar_decode(const uint8_t *vals, const uint8_t *scales,
                          int n, int bsize, float *out) {
    for (int i = 0; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}
#endif

/* MLX 4-bit group size (mirrors kernels.h SALT_MLX4_GROUP). File
 * scope: used by BOTH the NEON and AVX2 matvec paths. */
#define SALT_MLX4_GROUP_LOCAL 64

/* ------------------------------------------------------------------ */
/* NEON (aarch64)                                                      */
/* ------------------------------------------------------------------ */
#if defined(__aarch64__)
#include <alloca.h>
#include <arm_neon.h>
#include <stdlib.h>

/* 8 bytes = 16 nibbles = 16 elements, one scale. */
static void neon_decode16(const uint8_t *v8, float scale, float *o) {
    uint8x8_t b = vld1_u8(v8);
    uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
    uint8x8_t hi = vshr_n_u8(b, 4);
    uint8x16_t lut = vld1q_u8(MAG2S);   /* 16-byte table: nibbles 8-15 ok */
    uint8x16_t ml = vqtbl1q_u8(lut, vcombine_u8(lo, lo));  /* even */
    uint8x16_t mh = vqtbl1q_u8(lut, vcombine_u8(hi, hi));  /* odd */
    uint8x8_t z0 = vzip1_u8(vget_low_u8(ml), vget_low_u8(mh));   /* 0..7 */
    uint8x8_t z1 = vzip2_u8(vget_low_u8(ml), vget_low_u8(mh));   /* 8..15 */

    int16x8_t s0 = vmovl_s8(vreinterpret_s8_u8(z0));
    int16x8_t s1 = vmovl_s8(vreinterpret_s8_u8(z1));
    int32x4_t a0 = vmovl_s16(vget_low_s16(s0));
    int32x4_t a1 = vmovl_s16(vget_high_s16(s0));
    int32x4_t b0 = vmovl_s16(vget_low_s16(s1));
    int32x4_t b1 = vmovl_s16(vget_high_s16(s1));
    vst1q_f32(o, vmulq_n_f32(vcvtq_f32_s32(a0), scale * 0.5f));
    vst1q_f32(o + 4, vmulq_n_f32(vcvtq_f32_s32(a1), scale * 0.5f));
    vst1q_f32(o + 8, vmulq_n_f32(vcvtq_f32_s32(b0), scale * 0.5f));
    vst1q_f32(o + 12, vmulq_n_f32(vcvtq_f32_s32(b1), scale * 0.5f));
}

void salt_simd_mx4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out) {
    int i = 0;
    for (; i + 15 < n; i += 16)
        neon_decode16(vals + (i >> 1), e8m0f(scales[i / bsize]), out + i);
    for (; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}

void salt_simd_mx4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch) {
    salt_simd_mx4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < C; c += 8) {
            acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(wr + c),
                                              vld1q_f32(x + c)));
            acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(wr + c + 4),
                                              vld1q_f32(x + c + 4)));
        }
        float32x2_t t = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}

void salt_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
        int c = 0;
        for (; c + 7 < C; c += 8) {
            uint16x8_t h = vld1q_u16(wr + c);
            uint32x4_t h0 = vshll_n_u16(vget_low_u16(h), 16);
            uint32x4_t h1 = vshll_n_u16(vget_high_u16(h), 16);
            acc0 = vaddq_f32(acc0, vmulq_f32(vreinterpretq_f32_u32(h0),
                                              vld1q_f32(x + c)));
            acc1 = vaddq_f32(acc1, vmulq_f32(vreinterpretq_f32_u32(h1),
                                              vld1q_f32(x + c + 4)));
        }
        float32x2_t t = vadd_f32(vget_low_f32(acc0), vget_high_f32(acc0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(acc1), vget_high_f32(acc1)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}

void salt_simd_weighted_value_accumulate(
        float *output, const float *values, int value_stride,
        const float *scores, int count, int dimension, float denominator) {
    for (int position = 0; position < count; position++) {
        const float *value = values + (size_t)position * value_stride;
        float weight = scores[position] / denominator;
        float32x4_t weight4 = vdupq_n_f32(weight);
        int d = 0;
        for (; d + 3 < dimension; d += 4) {
            float32x4_t product = vmulq_f32(vld1q_f32(value + d), weight4);
            float32x4_t accumulated = vaddq_f32(
                vld1q_f32(output + d), product);
            vst1q_f32(output + d, accumulated);
        }
        for (; d < dimension; d++)
            output[d] += weight * value[d];
    }
}

void salt_simd_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                           const uint16_t *biases, int R, int C,
                           const float *x, float *y, int r0, int r1) {
    /* MLX 4-bit: U32 packs 8 nibbles (low first); BF16 scale+bias per
     * SALT_MLX4_GROUP (64) elements. Vector math per 8-element word:
     * extract nibbles -> widen -> float, q*s+b, FMA with x. C%8==0
     * and groups aligned to words (64 % 8 == 0). The fixed group scratch
     * covers qualified rows through the Gemma 4 full-attention C=8192 O
     * projection; the format-aware caller keeps wider rows scalar. */
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    float srow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    float brow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    float32x4_t acc[8];
    for (int r = r0; r < r1; r++) {
        const uint32_t *vr = vals + (size_t)r * (C / 8);
        int ng = (C + SALT_MLX4_GROUP_LOCAL - 1) / SALT_MLX4_GROUP_LOCAL;
        for (int g = 0; g < ng; g++) {
            /* true absolute group: (r*C + g*64)/64. This is r*ng+g
             * ONLY when C%64==0 -- the general formula is needed for
             * fixture shapes like C=200 (test 13). */
            size_t absg = ((size_t)r * C + (size_t)g * SALT_MLX4_GROUP_LOCAL)
                          / SALT_MLX4_GROUP_LOCAL;
            uint32_t sb = (uint32_t)load_u16_unaligned(scales, absg) << 16;
            uint32_t bb = biases ?
                (uint32_t)load_u16_unaligned(biases, absg) << 16 : 0;
            memcpy(&srow[g], &sb, 4);
            memcpy(&brow[g], &bb, 4);
        }
        /* FUSED decode+FMA: no scratch row. Decode 16 elements to
         * d0..d3, FMA straight into the accumulators. The accumulator
         * mapping (c>>2)&7 is 0 or 4 for c%16==0, so acc[a] always
         * owns columns {a, a+8, ...} mod 32 -- bit-identical to the
         * old pass-1/pass-2 split, minus the row round-trip. */
        for (int a = 0; a < 8; a++) acc[a] = vdupq_n_f32(0.0f);
        int w = 0;
        int c = 0;
        /* Logical width is fixed at 32. Physical vector width is selected by
         * the compiled substrate; request-time environment cannot change the
         * arithmetic DAG. */
        const int g_dw = 32;
        if (g_dw >= 32) {
        for (; c + 31 < C; c += 32, w += 4) {
            int g = c / SALT_MLX4_GROUP_LOCAL;
            float32x4_t sv = vld1q_dup_f32(&srow[g]);
            float32x4_t bv = vld1q_dup_f32(&brow[g]);
            uint8x16_t packed =
                vreinterpretq_u8_u32(vld1q_u32(&vr[w]));
            uint8x16_t lo = vandq_u8(packed, vdupq_n_u8(0x0F));
            uint8x16_t hi = vshrq_n_u8(packed, 4);
            uint8x16x2_t nz = vzipq_u8(lo, hi);
            uint8x8_t n0 = vget_low_u8(nz.val[0]);
            uint8x8_t n1 = vget_high_u8(nz.val[0]);
            uint8x8_t n2 = vget_low_u8(nz.val[1]);
            uint8x8_t n3 = vget_high_u8(nz.val[1]);
            uint32x4_t q00 = vmovl_u16(vget_low_u16(vmovl_u8(n0)));
            uint32x4_t q01 = vmovl_u16(vget_high_u16(vmovl_u8(n0)));
            uint32x4_t q10 = vmovl_u16(vget_low_u16(vmovl_u8(n1)));
            uint32x4_t q11 = vmovl_u16(vget_high_u16(vmovl_u8(n1)));
            uint32x4_t q20 = vmovl_u16(vget_low_u16(vmovl_u8(n2)));
            uint32x4_t q21 = vmovl_u16(vget_high_u16(vmovl_u8(n2)));
            uint32x4_t q30 = vmovl_u16(vget_low_u16(vmovl_u8(n3)));
            uint32x4_t q31 = vmovl_u16(vget_high_u16(vmovl_u8(n3)));
            float32x4_t d0 = vmlaq_f32(bv, vcvtq_f32_u32(q00), sv);
            float32x4_t d1 = vmlaq_f32(bv, vcvtq_f32_u32(q01), sv);
            float32x4_t d2 = vmlaq_f32(bv, vcvtq_f32_u32(q10), sv);
            float32x4_t d3 = vmlaq_f32(bv, vcvtq_f32_u32(q11), sv);
            float32x4_t d4 = vmlaq_f32(bv, vcvtq_f32_u32(q20), sv);
            float32x4_t d5 = vmlaq_f32(bv, vcvtq_f32_u32(q21), sv);
            float32x4_t d6 = vmlaq_f32(bv, vcvtq_f32_u32(q30), sv);
            float32x4_t d7 = vmlaq_f32(bv, vcvtq_f32_u32(q31), sv);
            int ab = (c >> 2) & 7;   /* 0 for c%32==0 */
            acc[ab + 0] = vmlaq_f32(acc[ab + 0], d0, vld1q_f32(x + c));
            acc[ab + 1] = vmlaq_f32(acc[ab + 1], d1, vld1q_f32(x + c + 4));
            acc[ab + 2] = vmlaq_f32(acc[ab + 2], d2, vld1q_f32(x + c + 8));
            acc[ab + 3] = vmlaq_f32(acc[ab + 3], d3, vld1q_f32(x + c + 12));
            acc[ab + 4] = vmlaq_f32(acc[ab + 4], d4, vld1q_f32(x + c + 16));
            acc[ab + 5] = vmlaq_f32(acc[ab + 5], d5, vld1q_f32(x + c + 20));
            acc[ab + 6] = vmlaq_f32(acc[ab + 6], d6, vld1q_f32(x + c + 24));
            acc[ab + 7] = vmlaq_f32(acc[ab + 7], d7, vld1q_f32(x + c + 28));
        }
        }
        for (; c + 15 < C; c += 16, w += 2) {
            int g = c / SALT_MLX4_GROUP_LOCAL;
            float32x4_t sv = vld1q_dup_f32(&srow[g]);
            float32x4_t bv = vld1q_dup_f32(&brow[g]);
            uint32_t u0 = load_u32_unaligned(vr, (size_t)w);
            uint32_t u1 = load_u32_unaligned(vr, (size_t)w + 1);
            uint8x8_t b0 = vreinterpret_u8_u32(vdup_n_u32(u0));
            uint8x8_t b1 = vreinterpret_u8_u32(vdup_n_u32(u1));
            uint8x8_t lo0 = vand_u8(b0, vdup_n_u8(0x0F));
            uint8x8_t hi0 = vand_u8(vshr_n_u8(b0, 4), vdup_n_u8(0x0F));
            uint8x8_t lo1 = vand_u8(b1, vdup_n_u8(0x0F));
            uint8x8_t hi1 = vand_u8(vshr_n_u8(b1, 4), vdup_n_u8(0x0F));
            uint8x8_t n0 = vzip_u8(lo0, hi0).val[0];
            uint8x8_t n1 = vzip_u8(lo1, hi1).val[0];
            uint32x4_t q00 = vmovl_u16(vget_low_u16(vmovl_u8(n0)));
            uint32x4_t q01 = vmovl_u16(vget_high_u16(vmovl_u8(n0)));
            uint32x4_t q10 = vmovl_u16(vget_low_u16(vmovl_u8(n1)));
            uint32x4_t q11 = vmovl_u16(vget_high_u16(vmovl_u8(n1)));
            float32x4_t d0 = vmlaq_f32(bv, vcvtq_f32_u32(q00), sv);
            float32x4_t d1 = vmlaq_f32(bv, vcvtq_f32_u32(q01), sv);
            float32x4_t d2 = vmlaq_f32(bv, vcvtq_f32_u32(q10), sv);
            float32x4_t d3 = vmlaq_f32(bv, vcvtq_f32_u32(q11), sv);
            int ab = (c >> 2) & 7;   /* 0 or 4 for c%16==0 */
            acc[ab + 0] = vmlaq_f32(acc[ab + 0], d0, vld1q_f32(x + c));
            acc[ab + 1] = vmlaq_f32(acc[ab + 1], d1, vld1q_f32(x + c + 4));
            acc[ab + 2] = vmlaq_f32(acc[ab + 2], d2, vld1q_f32(x + c + 8));
            acc[ab + 3] = vmlaq_f32(acc[ab + 3], d3, vld1q_f32(x + c + 12));
        }
        /* reduce the accumulators FIRST, then the scalar tail --
         * same summation order as the original pass-1/pass-2 split,
         * so results stay bit-identical (incl. fixture shapes like
         * C=200 where the tail is non-empty) */
        float32x4_t s4 = vdupq_n_f32(0.0f);
        for (int a = 0; a < 8; a++) s4 = vaddq_f32(s4, acc[a]);
        float32x2_t t = vadd_f32(vget_low_f32(s4), vget_high_f32(s4));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        /* scalar tail for C % 16 != 0: decode + FMA in one step */
        for (; c < C; c++) {
            long k = (size_t)r * C + c;
            int gl = (int)((c) / SALT_MLX4_GROUP_LOCAL);
            long wl = (k - (size_t)r * C) >> 3;
            int q = (int)((load_u32_unaligned(vr, (size_t)wl) >>
                           (4 * (k & 7))) & 0xFu);
            s += ((float)q * srow[gl] + brow[gl]) * x[c];
        }
        y[r] = s;
    }
}

/* I8 matvec: 16 int8 -> 16 floats, per-16-group scale (SC % 16 == 0
 * or SC == 1, so the scale is constant within each 16-group). */
void salt_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    const int8_t *Ws = (const int8_t *)W;
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = Ws + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            float32x4_t sv = vdupq_n_f32(s);
            int8x16_t w = vld1q_s8(wr + c);
            int16x8_t wl = vmovl_s8(vget_low_s8(w));
            int16x8_t wh = vmovl_s8(vget_high_s8(w));
            float32x4_t f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wl)));
            float32x4_t f1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wl)));
            float32x4_t f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(wh)));
            float32x4_t f3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(wh)));
            a0 = vmlaq_f32(a0, f0, vmulq_f32(vld1q_f32(x + c), sv));
            a1 = vmlaq_f32(a1, f1, vmulq_f32(vld1q_f32(x + c + 4), sv));
            a2 = vmlaq_f32(a2, f2, vmulq_f32(vld1q_f32(x + c + 8), sv));
            a3 = vmlaq_f32(a3, f3, vmulq_f32(vld1q_f32(x + c + 12), sv));
        }
        float32x2_t t = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a1), vget_high_f32(a1)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a2), vget_high_f32(a2)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a3), vget_high_f32(a3)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}

/* F8_E4M3 matvec. Two-table-free decode: P = 2^(e-7) via float bits
 * ((e+120) << 23), M = m/8, val = P*(1+M), with masked fixups for
 * e==0 (subnormal: m*2^-9) and e==15 (clamp 448), then sign. Exact
 * match to the scalar LUT in kernels.c. */
void salt_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            float32x4_t sv = vdupq_n_f32(s);
            uint8x16_t b = vld1q_u8(wr + c);
            uint8x16_t e8 = vandq_u8(vshrq_n_u8(b, 3), vdupq_n_u8(15));
            uint8x16_t m8 = vandq_u8(b, vdupq_n_u8(7));
            uint16x8_t el = vmovl_u8(vget_low_u8(e8));
            uint16x8_t eh = vmovl_u8(vget_high_u8(e8));
            uint16x8_t ml = vmovl_u8(vget_low_u8(m8));
            uint16x8_t mh = vmovl_u8(vget_high_u8(m8));
            uint32x4_t E0 = vmovl_u16(vget_low_u16(el));
            uint32x4_t E1 = vmovl_u16(vget_high_u16(el));
            uint32x4_t E2 = vmovl_u16(vget_low_u16(eh));
            uint32x4_t E3 = vmovl_u16(vget_high_u16(eh));
            float32x4_t F0 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(ml))), 0.125f);
            float32x4_t F1 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(ml))), 0.125f);
            float32x4_t F2 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(mh))), 0.125f);
            float32x4_t F3 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(mh))), 0.125f);
            float32x4_t V0 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E0, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F0));
            float32x4_t V1 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E1, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F1));
            float32x4_t V2 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E2, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F2));
            float32x4_t V3 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E3, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F3));
            /* e==0 fixup: scalar LUT says v = m * 2^-9. The old
             * arithmetic fixup (V += -2^-7 - F*2^-10) computed
             * m*7*2^-13 -- WRONG for every subnormal byte. Replace
             * with the exact subnormal value via masked select. */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(0));
                float32x4_t S0 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_low_u16(ml))), 0.001953125f);
                V0 = vbslq_f32(m0, S0, V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(0));
                float32x4_t S1 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_high_u16(ml))), 0.001953125f);
                V1 = vbslq_f32(m1, S1, V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(0));
                float32x4_t S2 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_low_u16(mh))), 0.001953125f);
                V2 = vbslq_f32(m2, S2, V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(0));
                float32x4_t S3 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_high_u16(mh))), 0.001953125f);
                V3 = vbslq_f32(m3, S3, V3);
            }
            /* e==15 fixup: scalar LUT says v = 448.0f (exact). */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(15));
                V0 = vbslq_f32(m0, vdupq_n_f32(448.0f), V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(15));
                V1 = vbslq_f32(m1, vdupq_n_f32(448.0f), V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(15));
                V2 = vbslq_f32(m2, vdupq_n_f32(448.0f), V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(15));
                V3 = vbslq_f32(m3, vdupq_n_f32(448.0f), V3);
            }
            /* sign per element (byte bit 7) */
            {
                uint8x16_t sg = vshrq_n_u8(b, 7);
                uint16x8_t sgl = vmovl_u8(vget_low_u8(sg));
                uint16x8_t sgh = vmovl_u8(vget_high_u8(sg));
                uint32x4_t n0 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgl)), 0x80000000u);
                uint32x4_t n1 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgl)), 0x80000000u);
                uint32x4_t n2 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgh)), 0x80000000u);
                uint32x4_t n3 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgh)), 0x80000000u);
                V0 = vbslq_f32(n0, vnegq_f32(V0), V0);
                V1 = vbslq_f32(n1, vnegq_f32(V1), V1);
                V2 = vbslq_f32(n2, vnegq_f32(V2), V2);
                V3 = vbslq_f32(n3, vnegq_f32(V3), V3);
            }
            a0 = vmlaq_f32(a0, V0, vmulq_f32(vld1q_f32(x + c), sv));
            a1 = vmlaq_f32(a1, V1, vmulq_f32(vld1q_f32(x + c + 4), sv));
            a2 = vmlaq_f32(a2, V2, vmulq_f32(vld1q_f32(x + c + 8), sv));
            a3 = vmlaq_f32(a3, V3, vmulq_f32(vld1q_f32(x + c + 12), sv));
        }
        float32x2_t t = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a1), vget_high_f32(a1)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a2), vget_high_f32(a2)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a3), vget_high_f32(a3)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
    }
}

/* F8_E4M3 matvec with BF16 block scales (the OFFICIAL Qwen FP8 layout:
 * weight_scale_inv is BF16 [SR,SC], unlike the E8M0 variant above).
 * Identical E4M3 SIMD decode; the only difference is the scale read
 * (2-byte BF16 instead of 1-byte E8M0). Bit-exact to the scalar
 * salt_f8_matvec_bf16 in kernels.c (same decode + accumulation order
 * per 16-block; the scalar tail matches the scalar kernel). */
void salt_simd_f8_matvec_bf16(const uint8_t *W, const uint16_t *scales,
                              int R, int C, int SR, int SC, const float *x,
                              float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
        float32x4_t a2 = vdupq_n_f32(0), a3 = vdupq_n_f32(0);
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? bf16_f(scales[sr * SC + sc]) : 1.0f;
            float32x4_t sv = vdupq_n_f32(s);
            uint8x16_t b = vld1q_u8(wr + c);
            uint8x16_t e8 = vandq_u8(vshrq_n_u8(b, 3), vdupq_n_u8(15));
            uint8x16_t m8 = vandq_u8(b, vdupq_n_u8(7));
            uint16x8_t el = vmovl_u8(vget_low_u8(e8));
            uint16x8_t eh = vmovl_u8(vget_high_u8(e8));
            uint16x8_t ml = vmovl_u8(vget_low_u8(m8));
            uint16x8_t mh = vmovl_u8(vget_high_u8(m8));
            uint32x4_t E0 = vmovl_u16(vget_low_u16(el));
            uint32x4_t E1 = vmovl_u16(vget_high_u16(el));
            uint32x4_t E2 = vmovl_u16(vget_low_u16(eh));
            uint32x4_t E3 = vmovl_u16(vget_high_u16(eh));
            float32x4_t F0 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(ml))), 0.125f);
            float32x4_t F1 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(ml))), 0.125f);
            float32x4_t F2 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_low_u16(mh))), 0.125f);
            float32x4_t F3 = vmulq_n_f32(vcvtq_f32_u32(
                vmovl_u16(vget_high_u16(mh))), 0.125f);
            float32x4_t V0 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E0, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F0));
            float32x4_t V1 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E1, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F1));
            float32x4_t V2 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E2, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F2));
            float32x4_t V3 = vmulq_f32(
                vreinterpretq_f32_u32(vshlq_n_u32(
                    vaddq_u32(E3, vdupq_n_u32(120)), 23)),
                vaddq_f32(vdupq_n_f32(1.0f), F3));
            /* e==0 fixup: scalar LUT says v = m * 2^-9. The old
             * arithmetic fixup (V += -2^-7 - F*2^-10) computed
             * m*7*2^-13 -- WRONG for every subnormal byte. Replace
             * with the exact subnormal value via masked select. */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(0));
                float32x4_t S0 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_low_u16(ml))), 0.001953125f);
                V0 = vbslq_f32(m0, S0, V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(0));
                float32x4_t S1 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_high_u16(ml))), 0.001953125f);
                V1 = vbslq_f32(m1, S1, V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(0));
                float32x4_t S2 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_low_u16(mh))), 0.001953125f);
                V2 = vbslq_f32(m2, S2, V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(0));
                float32x4_t S3 = vmulq_n_f32(vcvtq_f32_u32(
                    vmovl_u16(vget_high_u16(mh))), 0.001953125f);
                V3 = vbslq_f32(m3, S3, V3);
            }
            /* e==15 fixup: scalar LUT says v = 448.0f (exact). */
            {
                uint32x4_t m0 = vceqq_u32(E0, vdupq_n_u32(15));
                V0 = vbslq_f32(m0, vdupq_n_f32(448.0f), V0);
                uint32x4_t m1 = vceqq_u32(E1, vdupq_n_u32(15));
                V1 = vbslq_f32(m1, vdupq_n_f32(448.0f), V1);
                uint32x4_t m2 = vceqq_u32(E2, vdupq_n_u32(15));
                V2 = vbslq_f32(m2, vdupq_n_f32(448.0f), V2);
                uint32x4_t m3 = vceqq_u32(E3, vdupq_n_u32(15));
                V3 = vbslq_f32(m3, vdupq_n_f32(448.0f), V3);
            }
            /* sign per element (byte bit 7) */
            {
                uint8x16_t sg = vshrq_n_u8(b, 7);
                uint16x8_t sgl = vmovl_u8(vget_low_u8(sg));
                uint16x8_t sgh = vmovl_u8(vget_high_u8(sg));
                uint32x4_t n0 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgl)), 0x80000000u);
                uint32x4_t n1 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgl)), 0x80000000u);
                uint32x4_t n2 = vmulq_n_u32(
                    vmovl_u16(vget_low_u16(sgh)), 0x80000000u);
                uint32x4_t n3 = vmulq_n_u32(
                    vmovl_u16(vget_high_u16(sgh)), 0x80000000u);
                V0 = vbslq_f32(n0, vnegq_f32(V0), V0);
                V1 = vbslq_f32(n1, vnegq_f32(V1), V1);
                V2 = vbslq_f32(n2, vnegq_f32(V2), V2);
                V3 = vbslq_f32(n3, vnegq_f32(V3), V3);
            }
            /* accumulate with the SAME rounding as the scalar kernel:
             * t1 = V*s (rounded), t2 = t1*x (rounded), a += t2
             * (rounded). NOT vmlaq (fused = 2 roundings, different
             * bits). -ffp-contract=off keeps these separate ops. */
            {
                float32x4_t xv0 = vld1q_f32(x + c);
                float32x4_t xv1 = vld1q_f32(x + c + 4);
                float32x4_t xv2 = vld1q_f32(x + c + 8);
                float32x4_t xv3 = vld1q_f32(x + c + 12);
                float32x4_t t0 = vmulq_f32(V0, sv);   /* V*s */
                float32x4_t t1 = vmulq_f32(V1, sv);
                float32x4_t t2 = vmulq_f32(V2, sv);
                float32x4_t t3 = vmulq_f32(V3, sv);
                t0 = vmulq_f32(t0, xv0);              /* (V*s)*x */
                t1 = vmulq_f32(t1, xv1);
                t2 = vmulq_f32(t2, xv2);
                t3 = vmulq_f32(t3, xv3);
                a0 = vaddq_f32(a0, t0);               /* a += t */
                a1 = vaddq_f32(a1, t1);
                a2 = vaddq_f32(a2, t2);
                a3 = vaddq_f32(a3, t3);
            }
        }
        /* final lane sum, EXACT pairwise tree matching the scalar:
         * t = (a0.lo+a0.hi, a0.lo2+a0.hi2); t += a1 pairs; += a2;
         * += a3; s = t[0] + t[1]. */
        float32x2_t t = vadd_f32(vget_low_f32(a0), vget_high_f32(a0));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a1), vget_high_f32(a1)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a2), vget_high_f32(a2)));
        t = vadd_f32(t, vadd_f32(vget_low_f32(a3), vget_high_f32(a3)));
        float s = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? bf16_f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
    }
}

int salt_simd_available(void) { return 1; }

/* ------------------------------------------------------------------ */
/* AVX2 (x86-64)                                                       */
/* ------------------------------------------------------------------ */
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>

static int avx2_ok(void) {
    return salt_compiler_avx2_available();
}

/* 8 bytes = 16 nibbles = 16 elements, one scale. */
static SALT_TARGET_AVX2 void avx2_decode16(
        const uint8_t *v8, float scale, float *o) {
    __m128i b = _mm_loadl_epi64((const __m128i *)v8);
    __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
    __m128i hi = _mm_srli_epi16(
        _mm_and_si128(b, _mm_set1_epi8((char)0xF0)), 4);
    __m128i lut = _mm_loadu_si128((const __m128i *)MAG2S);
    __m128i ml = _mm_shuffle_epi8(lut, lo);   /* even elements */
    __m128i mh = _mm_shuffle_epi8(lut, hi);   /* odd elements */
    /* Eight packed bytes contain exactly sixteen nibbles. Interleave the low
     * eight LUT results into one 16-byte vector, then widen its four quartets.
     * Reading unpackhi would consume bytes 8..15 from the NEXT block. */
    __m128i zlo = _mm_unpacklo_epi8(ml, mh);

    __m128i d0 = _mm_cvtepi8_epi32(zlo);                       /* 0-3  */
    __m128i d1 = _mm_cvtepi8_epi32(_mm_srli_si128(zlo, 4));    /* 4-7  */
    __m128i d2 = _mm_cvtepi8_epi32(_mm_srli_si128(zlo, 8));    /* 8-11 */
    __m128i d3 = _mm_cvtepi8_epi32(_mm_srli_si128(zlo, 12));   /* 12-15 */
    __m128 f0 = _mm_mul_ps(_mm_cvtepi32_ps(d0), _mm_set1_ps(scale * 0.5f));
    __m128 f1 = _mm_mul_ps(_mm_cvtepi32_ps(d1), _mm_set1_ps(scale * 0.5f));
    __m128 f2 = _mm_mul_ps(_mm_cvtepi32_ps(d2), _mm_set1_ps(scale * 0.5f));
    __m128 f3 = _mm_mul_ps(_mm_cvtepi32_ps(d3), _mm_set1_ps(scale * 0.5f));
    _mm_storeu_ps(o, f0);
    _mm_storeu_ps(o + 4, f1);
    _mm_storeu_ps(o + 8, f2);
    _mm_storeu_ps(o + 12, f3);
}

SALT_TARGET_AVX2 void salt_simd_mx4_decode(
        const uint8_t *vals, const uint8_t *scales,
        int n, int bsize, float *out) {
    int i = 0;
    for (; i + 15 < n; i += 16)
        avx2_decode16(vals + (i >> 1), e8m0f(scales[i / bsize]), out + i);
    for (; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        out[i] = (float)((int8_t)MAG2S[nib]) * 0.5f * e8m0f(scales[i / bsize]);
    }
}

SALT_TARGET_AVX2 void salt_simd_mx4_matvec(
        const uint8_t *vals, const uint8_t *scales,
        int R, int C, int bsize, const float *x,
        float *y, float *scratch) {
    salt_simd_mx4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 7 < C; c += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(wr + c),
                                                   _mm256_loadu_ps(x + c)));
        /* reduce -- EXACTLY the aarch64 tree (simd.c:110-112):
         * t = (acc0.low+acc0.high) + (acc1.low+acc1.high), then
         * s = t[0]+t[1]. The aarch64 kernel uses TWO 4-lane accs
         * (acc0 = cols c..c+3, acc1 = cols c+4..c+7 per 8-block);
         * the 8-lane __m256 holds the same lanes, so the tree is:
         *   t0 = acc[0]+acc[2]  (acc0.low+acc0.high)
         *   t1 = acc[1]+acc[3]  (acc0.lanes 0,1 pairwise)
         *   t2 = acc[4]+acc[6]  (acc1.low+acc1.high)
         *   t3 = acc[5]+acc[7]  (acc1 lanes)
         *   s = (t0+t2) + (t1+t3)   -- NEON: t = (l0+h0)+(l1+h1),
         *   then s = t[0]+t[1]. Volatile stores keep gcc from
         *   reassociating the pure-add tree. */
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        volatile float v0 = tmp[0], v1 = tmp[1], v2 = tmp[2], v3 = tmp[3];
        volatile float v4 = tmp[4], v5 = tmp[5], v6 = tmp[6], v7 = tmp[7];
        volatile float t0 = v0 + v2;   /* acc0.low + acc0.high */
        volatile float t1 = v1 + v3;
        volatile float t2 = v4 + v6;   /* acc1.low + acc1.high */
        volatile float t3 = v5 + v7;
        volatile float u0 = t0 + t2;   /* t = (l0+h0) + (l1+h1) */
        volatile float u1 = t1 + t3;
        float s = u0 + u1;             /* s = t[0] + t[1] */
        for (; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}

SALT_TARGET_AVX2 void salt_simd_bf16_matvec(
        const uint16_t *W, int R, int C,
        const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        __m256 acc = _mm256_setzero_ps();
        int c = 0;
        for (; c + 7 < C; c += 8) {
            __m128i h = _mm_loadu_si128((const __m128i *)(wr + c));
            __m256i w = _mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16);
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_castsi256_ps(w),
                                                   _mm256_loadu_ps(x + c)));
        }
        /* reduce -- EXACTLY the aarch64 tree (simd.c:133-135):
         * t = (acc0.low+acc0.high) + (acc1.low+acc1.high), then
         * s = t[0]+t[1]. The aarch64 kernel uses TWO 4-lane accs
         * (acc0 = cols c..c+3, acc1 = cols c+4..c+7 per 8-block);
         * the 8-lane __m256 holds the same lanes, so:
         *   t0 = tmp[0]+tmp[2] (acc0.low+acc0.high)
         *   t1 = tmp[1]+tmp[3]
         *   t2 = tmp[4]+tmp[6] (acc1.low+acc1.high)
         *   t3 = tmp[5]+tmp[7]
         *   s = (t0+t2) + (t1+t3). Volatile stores keep gcc from
         * reassociating the pure-add tree. */
        float tmp[8];
        _mm256_storeu_ps(tmp, acc);
        volatile float v0 = tmp[0], v1 = tmp[1], v2 = tmp[2], v3 = tmp[3];
        volatile float v4 = tmp[4], v5 = tmp[5], v6 = tmp[6], v7 = tmp[7];
        volatile float t0 = v0 + v2;   /* acc0.low + acc0.high */
        volatile float t1 = v1 + v3;
        volatile float t2 = v4 + v6;   /* acc1.low + acc1.high */
        volatile float t3 = v5 + v7;
        volatile float u0 = t0 + t2;   /* t = (l0+h0) + (l1+h1) */
        volatile float u1 = t1 + t3;
        float s = u0 + u1;             /* s = t[0] + t[1] */
        for (; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}

SALT_TARGET_AVX2 void salt_simd_weighted_value_accumulate(
        float *output, const float *values, int value_stride,
        const float *scores, int count, int dimension, float denominator) {
    for (int position = 0; position < count; position++) {
        const float *value = values + (size_t)position * value_stride;
        float weight = scores[position] / denominator;
        __m256 weight8 = _mm256_set1_ps(weight);
        int d = 0;
        for (; d + 7 < dimension; d += 8) {
            __m256 product = _mm256_mul_ps(_mm256_loadu_ps(value + d), weight8);
            __m256 accumulated = _mm256_add_ps(
                _mm256_loadu_ps(output + d), product);
            _mm256_storeu_ps(output + d, accumulated);
        }
        for (; d < dimension; d++)
            output[d] += weight * value[d];
    }
}

SALT_TARGET_AVX2 void salt_simd_i8_matvec(
        const uint8_t *W, const uint8_t *scales,
        int R, int C, int SR, int SC, const float *x, float *y) {
    const int8_t *Ws = (const int8_t *)W;
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = Ws + (size_t)r * C;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            __m256 sv = _mm256_set1_ps(s);
            __m128i b = _mm_loadu_si128((const __m128i *)(wr + c));
            __m256i w01 = _mm256_cvtepi8_epi16(b);
            __m256i w0 = _mm256_cvtepi16_epi32(
                _mm256_castsi256_si128(w01));
            __m256i w1 = _mm256_cvtepi16_epi32(
                _mm256_extracti128_si256(w01, 1));
            __m256 f0 = _mm256_cvtepi32_ps(w0);
            __m256 f1 = _mm256_cvtepi32_ps(w1);
            a0 = _mm256_fmadd_ps(f0, _mm256_mul_ps(
                _mm256_loadu_ps(x + c), sv), a0);
            a1 = _mm256_fmadd_ps(f1, _mm256_mul_ps(
                _mm256_loadu_ps(x + c + 8), sv), a1);
        }
        float s = 0.0f;
        float t0[8], t1[8];
        _mm256_storeu_ps(t0, a0);
        _mm256_storeu_ps(t1, a1);
        for (int q = 0; q < 8; q++) s += t0[q] + t1[q];
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}

SALT_TARGET_AVX2 void salt_simd_f8_matvec(
        const uint8_t *W, const uint8_t *scales,
        int R, int C, int SR, int SC, const float *x,
        float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            __m256 sv = _mm256_set1_ps(s);
            __m128i b = _mm_loadu_si128((const __m128i *)(wr + c));
            __m128i e8 = _mm_and_si128(
                _mm_srli_epi16(_mm_and_si128(b, _mm_set1_epi8(0x78)), 3),
                _mm_set1_epi8(0x0F));
            __m128i m8 = _mm_and_si128(b, _mm_set1_epi8(0x07));
            __m256i e01 = _mm256_cvtepu8_epi16(e8);
            __m256i m01 = _mm256_cvtepu8_epi16(m8);
            __m256i E0 = _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(e01));
            __m256i E1 = _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(e01, 1));
            __m256i M0 = _mm256_cvtepu16_epi32(
                _mm256_castsi256_si128(m01));
            __m256i M1 = _mm256_cvtepu16_epi32(
                _mm256_extracti128_si256(m01, 1));
            __m256 F0 = _mm256_mul_ps(_mm256_cvtepi32_ps(M0),
                                      _mm256_set1_ps(0.125f));
            __m256 F1 = _mm256_mul_ps(_mm256_cvtepi32_ps(M1),
                                      _mm256_set1_ps(0.125f));
            __m256 V0 = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_add_epi32(E0, _mm256_set1_epi32(120)), 23)),
                _mm256_add_ps(_mm256_set1_ps(1.0f), F0));
            __m256 V1 = _mm256_mul_ps(
                _mm256_castsi256_ps(_mm256_slli_epi32(
                    _mm256_add_epi32(E1, _mm256_set1_epi32(120)), 23)),
                _mm256_add_ps(_mm256_set1_ps(1.0f), F1));
            /* e==0 fixup: val += M*2^-10 - 2^-7 */
            {
                __m256 msk0 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E0, _mm256_setzero_si256()));
                __m256 C0 = _mm256_sub_ps(
                    _mm256_mul_ps(F0, _mm256_set1_ps(0.0009765625f)),
                    _mm256_set1_ps(0.0078125f));
                V0 = _mm256_blendv_ps(V0, _mm256_add_ps(V0, C0), msk0);
                __m256 msk1 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E1, _mm256_setzero_si256()));
                __m256 C1 = _mm256_sub_ps(
                    _mm256_mul_ps(F1, _mm256_set1_ps(0.0009765625f)),
                    _mm256_set1_ps(0.0078125f));
                V1 = _mm256_blendv_ps(V1, _mm256_add_ps(V1, C1), msk1);
            }
            /* e==15 fixup: val += 448 - 256*(1+M) */
            {
                __m256 msk0 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E0, _mm256_set1_epi32(15)));
                __m256 C0 = _mm256_sub_ps(_mm256_set1_ps(448.0f),
                    _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), F0),
                                  _mm256_set1_ps(256.0f)));
                V0 = _mm256_blendv_ps(V0, _mm256_add_ps(V0, C0), msk0);
                __m256 msk1 = _mm256_castsi256_ps(
                    _mm256_cmpeq_epi32(E1, _mm256_set1_epi32(15)));
                __m256 C1 = _mm256_sub_ps(_mm256_set1_ps(448.0f),
                    _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), F1),
                                  _mm256_set1_ps(256.0f)));
                V1 = _mm256_blendv_ps(V1, _mm256_add_ps(V1, C1), msk1);
            }
            /* sign per element (byte bit 7) */
            {
                __m128i sg = _mm_srli_epi16(
                    _mm_and_si128(b, _mm_set1_epi8((char)0x80)), 7);
                __m256i sg01 = _mm256_cvtepu8_epi16(sg);
                __m256i n0 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(
                    _mm256_castsi256_si128(sg01)), 31);
                __m256i n1 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(
                    _mm256_extracti128_si256(sg01, 1)), 31);
                V0 = _mm256_blendv_ps(V0, _mm256_sub_ps(
                    _mm256_setzero_ps(), V0), _mm256_castsi256_ps(n0));
                V1 = _mm256_blendv_ps(V1, _mm256_sub_ps(
                    _mm256_setzero_ps(), V1), _mm256_castsi256_ps(n1));
            }
            /* UNFUSED mul+add: Apple clang compiles NEON vmlaq_f32 as
             * fmul+fadd (verified: 0 fmla in the Mac binary), so the
             * AVX2 path must NOT use _mm256_fmadd_ps or the rounding
             * differs cross-ISA (the q4 kernel bug, same class). */
            a0 = _mm256_add_ps(a0, _mm256_mul_ps(V0, _mm256_mul_ps(
                _mm256_loadu_ps(x + c), sv)));
            a1 = _mm256_add_ps(a1, _mm256_mul_ps(V1, _mm256_mul_ps(
                _mm256_loadu_ps(x + c + 8), sv)));
        }
        /* reduce -- EXACTLY the aarch64 tree (simd.c:379-383):
         * t = (a0.low+a0.high) + (a1.low+a1.high) + (a2.low+a2.high)
         *     + (a3.low+a3.high), then s = t[0]+t[1]. The aarch64
         * kernel uses FOUR 4-lane accs; the two 8-lane __m256 hold
         * the same lanes (a0 = cols c..c+7, a1 = cols c+8..c+15),
         * so: t0 = a0[0]+a0[2] (a0.low+a0.high), etc. Volatile stores
         * keep gcc from reassociating the pure-add tree. */
        float t0[8], t1[8];
        _mm256_storeu_ps(t0, a0);
        _mm256_storeu_ps(t1, a1);
        volatile float v0 = t0[0], v1 = t0[1], v2 = t0[2], v3 = t0[3];
        volatile float v4 = t0[4], v5 = t0[5], v6 = t0[6], v7 = t0[7];
        volatile float w0 = t1[0], w1 = t1[1], w2 = t1[2], w3 = t1[3];
        volatile float w4 = t1[4], w5 = t1[5], w6 = t1[6], w7 = t1[7];
        volatile float p0 = v0 + v2;   /* a0.low + a0.high */
        volatile float p1 = v1 + v3;
        volatile float p2 = v4 + v6;   /* a1.low + a1.high */
        volatile float p3 = v5 + v7;
        volatile float p4 = w0 + w2;   /* a2.low + a2.high */
        volatile float p5 = w1 + w3;
        volatile float p6 = w4 + w6;   /* a3.low + a3.high */
        volatile float p7 = w5 + w7;
        /* NEON: t = vadd_f32(t, ...) chained FOUR times = SERIAL left
         * fold: t[0] = ((p0 + p2) + p4) + p6, t[1] = ((p1 + p3) + p5)
         * + p7. NOT pairwise -- the association matters for rounding. */
        volatile float r0 = (p0 + p2) + p4;   /* t[0] fold step 1-2 */
        volatile float r1 = (p1 + p3) + p5;
        volatile float q0 = r0 + p6;          /* t[0] = fold + a3 */
        volatile float q1 = r1 + p7;          /* t[1] */
        float s = q0 + q1;                    /* s = t[0] + t[1] */
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
    }
}

void salt_simd_f8_matvec_bf16(
        const uint8_t *weights, const uint16_t *scales,
        int rows, int columns, int scale_rows, int scale_columns,
        const float *input, float *output, int row_start, int row_end) {
    scalar_f8_matvec_bf16(
        weights, scales, rows, columns, scale_rows, scale_columns,
        input, output, row_start, row_end);
}

int salt_simd_available(void) { return avx2_ok(); }

static SALT_TARGET_AVX2 void avx2_q4_matvec_rows2(
        const uint32_t *vals, const uint16_t *scales,
        const uint16_t *biases, int C, const float *x,
        float *y, int row, int ng) {
    const uint32_t *vr0 = vals + (size_t)row * (C / 8);
    const uint32_t *vr1 = vals + (size_t)(row + 1) * (C / 8);
    float srow0[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    float brow0[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    float srow1[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    float brow1[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
    __m256 acc[8];
    const __m256i mask4 = _mm256_set1_epi32(0x0F);
    const __m256i sh04 = _mm256_setr_epi32(0, 4, 8, 12, 0, 4, 8, 12);
    const __m256i sh16 = _mm256_setr_epi32(16, 20, 24, 28,
                                           16, 20, 24, 28);
    for (int g = 0; g < ng; g++) {
        size_t absg0 = (size_t)row * (size_t)ng + (size_t)g;
        size_t absg1 = (size_t)(row + 1) * (size_t)ng + (size_t)g;
        uint32_t sb0 = (uint32_t)load_u16_unaligned(scales, absg0) << 16;
        uint32_t bb0 = biases ?
            (uint32_t)load_u16_unaligned(biases, absg0) << 16 : 0;
        uint32_t sb1 = (uint32_t)load_u16_unaligned(scales, absg1) << 16;
        uint32_t bb1 = biases ?
            (uint32_t)load_u16_unaligned(biases, absg1) << 16 : 0;
        memcpy(&srow0[g], &sb0, sizeof(float));
        memcpy(&brow0[g], &bb0, sizeof(float));
        memcpy(&srow1[g], &sb1, sizeof(float));
        memcpy(&brow1[g], &bb1, sizeof(float));
    }
    for (int a = 0; a < 8; a++) acc[a] = _mm256_setzero_ps();
    for (int c = 0, w = 0; c < C; c += 32, w += 4) {
        int g = c / SALT_MLX4_GROUP_LOCAL;
        __m256 sv = _mm256_castps128_ps256(_mm_set1_ps(srow0[g]));
        __m256 bv = _mm256_castps128_ps256(_mm_set1_ps(brow0[g]));
        __m256i words[4];
        __m256 dequant[8];
        sv = _mm256_insertf128_ps(sv, _mm_set1_ps(srow1[g]), 1);
        bv = _mm256_insertf128_ps(bv, _mm_set1_ps(brow1[g]), 1);
        for (int word = 0; word < 4; word++) {
            __m128i low = _mm_set1_epi32(
                (int)load_u32_unaligned(vr0, (size_t)w + (size_t)word));
            __m128i high = _mm_set1_epi32(
                (int)load_u32_unaligned(vr1, (size_t)w + (size_t)word));
            words[word] = _mm256_castsi128_si256(low);
            words[word] = _mm256_inserti128_si256(words[word], high, 1);
            dequant[2 * word] = _mm256_add_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_srlv_epi32(words[word], sh04), mask4)), sv), bv);
            dequant[2 * word + 1] = _mm256_add_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_and_si256(
                    _mm256_srlv_epi32(words[word], sh16), mask4)), sv), bv);
        }
        for (int a = 0; a < 8; a++) {
            __m128 x4 = _mm_loadu_ps(x + c + 4 * a);
            __m256 x8 = _mm256_castps128_ps256(x4);
            x8 = _mm256_insertf128_ps(x8, x4, 1);
            acc[a] = _mm256_add_ps(acc[a], _mm256_mul_ps(dequant[a], x8));
        }
    }
    {
        __m256 sum = _mm256_setzero_ps();
        float lanes[8];
        for (int a = 0; a < 8; a++) sum = _mm256_add_ps(sum, acc[a]);
        _mm256_storeu_ps(lanes, sum);
        volatile float a0 = lanes[0], a1 = lanes[1];
        volatile float a2 = lanes[2], a3 = lanes[3];
        volatile float b0 = lanes[4], b1 = lanes[5];
        volatile float b2 = lanes[6], b3 = lanes[7];
        volatile float ap0 = a0 + a2;
        volatile float ap1 = a1 + a3;
        volatile float bp0 = b0 + b2;
        volatile float bp1 = b1 + b3;
        y[row] = ap0 + ap1;
        y[row + 1] = bp0 + bp1;
    }
}

SALT_TARGET_AVX2 void salt_simd_q4_matvec(
        const uint32_t *vals, const uint16_t *scales,
        const uint16_t *biases, int R, int C,
        const float *x, float *y, int r0, int r1) {
    /* AVX2, CANONICAL 8x4-lane structure: mirrors the aarch64 non-tiled
     * batch worker (kernels.c) EXACTLY so arm64 and x86 produce
     * byte-identical results -- same decode (fused), same 8-acc map,
     * same lane-wise+pairwise reduce tree. Cross-ISA KV identity is the
     * default requirement ("knowledge carved in stone"). */
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    const int ng = (C + SALT_MLX4_GROUP_LOCAL - 1) / SALT_MLX4_GROUP_LOCAL;
    int r = r0;
    if (g_q4_row_pair && (C % SALT_MLX4_GROUP_LOCAL) == 0)
        for (; r + 1 < r1; r += 2)
            avx2_q4_matvec_rows2(
                vals, scales, biases, C, x, y, r, ng);
    for (; r < r1; r++) {
        const uint32_t *vr = vals + (size_t)r * (C / 8);
        /* scale/bias rows (absolute group index, like kernels.c) */
        float srow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        float brow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        for (int g = 0; g < ng; g++) {
            size_t absg = ((size_t)r * C + (size_t)g * SALT_MLX4_GROUP_LOCAL)
                          / SALT_MLX4_GROUP_LOCAL;
            uint32_t sb = (uint32_t)load_u16_unaligned(scales, absg) << 16;
            uint32_t bb = biases ?
                (uint32_t)load_u16_unaligned(biases, absg) << 16 : 0;
            float sf, bf;
            memcpy(&sf, &sb, 4);
            memcpy(&bf, &bb, 4);
            srow[g] = sf;
            brow[g] = bf;
        }
        /* 8 x 4-lane accumulators, (c>>2)&7 map -- the aarch64 layout */
        __m128 acc[8];
        for (int a = 0; a < 8; a++) acc[a] = _mm_setzero_ps();
        int c = 0, w = 0;
        const __m128i mask4 = _mm_set1_epi32(0x0F);
        const __m128i sh04 = _mm_setr_epi32(0, 4, 8, 12);
        const __m128i sh16 = _mm_setr_epi32(16, 20, 24, 28);
        for (; c + 31 < C; c += 32, w += 4) {
            int g = c / SALT_MLX4_GROUP_LOCAL;
            __m128 sv = _mm_set1_ps(srow[g]);
            __m128 bv = _mm_set1_ps(brow[g]);
            __m128i u0 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w));
            __m128i u1 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w + 1));
            __m128i u2 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w + 2));
            __m128i u3 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w + 3));
            __m128 f0 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u0, sh04), mask4)),
                sv), bv);
            __m128 f1 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u0, sh16), mask4)),
                sv), bv);
            __m128 f2 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u1, sh04), mask4)),
                sv), bv);
            __m128 f3 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u1, sh16), mask4)),
                sv), bv);
            __m128 f4 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u2, sh04), mask4)),
                sv), bv);
            __m128 f5 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u2, sh16), mask4)),
                sv), bv);
            __m128 f6 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u3, sh04), mask4)),
                sv), bv);
            __m128 f7 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u3, sh16), mask4)),
                sv), bv);
            acc[0] = _mm_add_ps(acc[0],
                _mm_mul_ps(f0, _mm_loadu_ps(x + c)));
            acc[1] = _mm_add_ps(acc[1],
                _mm_mul_ps(f1, _mm_loadu_ps(x + c + 4)));
            acc[2] = _mm_add_ps(acc[2],
                _mm_mul_ps(f2, _mm_loadu_ps(x + c + 8)));
            acc[3] = _mm_add_ps(acc[3],
                _mm_mul_ps(f3, _mm_loadu_ps(x + c + 12)));
            acc[4] = _mm_add_ps(acc[4],
                _mm_mul_ps(f4, _mm_loadu_ps(x + c + 16)));
            acc[5] = _mm_add_ps(acc[5],
                _mm_mul_ps(f5, _mm_loadu_ps(x + c + 20)));
            acc[6] = _mm_add_ps(acc[6],
                _mm_mul_ps(f6, _mm_loadu_ps(x + c + 24)));
            acc[7] = _mm_add_ps(acc[7],
                _mm_mul_ps(f7, _mm_loadu_ps(x + c + 28)));
        }
        for (; c + 15 < C; c += 16, w += 2) {
            int g = c / SALT_MLX4_GROUP_LOCAL;
            __m128 sv = _mm_set1_ps(srow[g]);
            __m128 bv = _mm_set1_ps(brow[g]);
            __m128i u0 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w));
            __m128i u1 = _mm_set1_epi32(
                (int)load_u32_unaligned(vr, (size_t)w + 1));
            __m128 f0 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u0, sh04), mask4)),
                sv), bv);
            __m128 f1 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u0, sh16), mask4)),
                sv), bv);
            __m128 f2 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u1, sh04), mask4)),
                sv), bv);
            __m128 f3 = _mm_add_ps(_mm_mul_ps(
                _mm_cvtepi32_ps(_mm_and_si128(_mm_srlv_epi32(u1, sh16), mask4)),
                sv), bv);
            int ab = (c >> 2) & 7;
            acc[ab + 0] = _mm_add_ps(acc[ab + 0],
                _mm_mul_ps(f0, _mm_loadu_ps(x + c)));
            acc[ab + 1] = _mm_add_ps(acc[ab + 1],
                _mm_mul_ps(f1, _mm_loadu_ps(x + c + 4)));
            acc[ab + 2] = _mm_add_ps(acc[ab + 2],
                _mm_mul_ps(f2, _mm_loadu_ps(x + c + 8)));
            acc[ab + 3] = _mm_add_ps(acc[ab + 3],
                _mm_mul_ps(f3, _mm_loadu_ps(x + c + 12)));
        }
        /* reduce: lane-wise vaddq across 8 accs, low+high pairwise,
         * then scalar -- EXACTLY the aarch64 tree (kernels.c:554-558):
         * vadd_f32(low,high) pairs (0,2) and (1,3), then t[0]+t[1].
         * Volatile stores kill any compiler reassociation of the
         * adds (gcc was free to reorder the pure-add tree). */
        __m128 s4 = _mm_setzero_ps();
        for (int a = 0; a < 8; a++) s4 = _mm_add_ps(s4, acc[a]);
        float t[4];
        _mm_storeu_ps(t, s4);
        volatile float v0 = t[0], v1 = t[1], v2 = t[2], v3 = t[3];
        volatile float p0 = v0 + v2;   /* vadd_f32(low,high).lane0 */
        volatile float p1 = v1 + v3;   /* vadd_f32(low,high).lane1 */
        float s = p0 + p1;             /* vget_lane(t,0)+vget_lane(t,1) */
        /* scalar tail (C % 16 != 0): same as NEON tail */
        for (; c < C; c++) {
            long k = (size_t)r * C + c;
            int gl = (int)((c) / SALT_MLX4_GROUP_LOCAL);
            long wl = (k - (size_t)r * C) >> 3;
            int q = (int)((load_u32_unaligned(vr, (size_t)wl) >>
                           (4 * (k & 7))) & 0xFu);
            s += ((float)q * srow[gl] + brow[gl]) * x[c];
        }
        y[r] = s;
    }
}

/* ------------------------------------------------------------------ */
/* fallback (portable)                                                 */
/* ------------------------------------------------------------------ */
#else
void salt_simd_mx4_decode(const uint8_t *vals, const uint8_t *scales,
                            int n, int bsize, float *out) {
    scalar_decode(vals, scales, n, bsize, out);
}
void salt_simd_mx4_matvec(const uint8_t *vals, const uint8_t *scales,
                            int R, int C, int bsize, const float *x,
                            float *y, float *scratch) {
    scalar_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        const float *wr = scratch + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += wr[c] * x[c];
        y[r] = s;
    }
}
void salt_simd_bf16_matvec(const uint16_t *W, int R, int C,
                           const float *x, const float *bias, float *y) {
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) s += bf16_f(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}
void salt_simd_weighted_value_accumulate(
        float *output, const float *values, int value_stride,
        const float *scores, int count, int dimension, float denominator) {
    for (int position = 0; position < count; position++) {
        const float *value = values + (size_t)position * value_stride;
        float weight = scores[position] / denominator;
        for (int d = 0; d < dimension; d++)
            output[d] += weight * value[d];
    }
}
void salt_simd_i8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const int8_t *wr = (const int8_t *)W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            s += (float)wr[c] * sl * x[c];
        }
        y[r] = s;
    }
}
void salt_simd_f8_matvec(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        float s = 0.0f;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float sl = scales ? e8m0f(scales[sr * SC + sc]) : 1.0f;
            uint8_t b = wr[c];
            int e = (b >> 3) & 0xF, m = b & 7;
            float v;
            if (e == 0) v = (float)m * 0.001953125f;
            else if (e == 0xF) v = 448.0f;
            else v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
            s += ((b & 0x80) ? -v : v) * sl * x[c];
        }
        y[r] = s;
    }
}
void salt_simd_f8_matvec_bf16(
        const uint8_t *weights, const uint16_t *scales,
        int rows, int columns, int scale_rows, int scale_columns,
        const float *input, float *output, int row_start, int row_end) {
    scalar_f8_matvec_bf16(
        weights, scales, rows, columns, scale_rows, scale_columns,
        input, output, row_start, row_end);
}

int salt_simd_available(void) { return 0; }
#endif

/* INT2 loads, SALT's unfused eight-lane BF16 fold, and its bounded
 * weight-stationary token-tile shape. No decoded payload or worker storage. */
int salt_simd_int2_matvec_batch_rows(const uint8_t *values, const void *scales,
    int rows, int cols, int batch, const float *inputs, float *outputs,
    int first_row, int end_row) {
#if defined(__aarch64__)
    enum { TOKEN_TILE = 4 };
    const int32_t shift_data[4] = {0, -2, -4, -6};
    int32x4_t shifts = vld1q_s32(shift_data);
    uint32x4_t mask = vdupq_n_u32(3u);
    int32x4_t center = vdupq_n_s32(2);
    if (!values || !scales || !inputs || !outputs || rows < 1 || cols < 1 ||
        batch < 1 || first_row < 0 || end_row < first_row || end_row > rows)
        return -1;
    size_t stride = ((size_t)cols + 3u) / 4u;
    for (int r = first_row; r < end_row; r++) {
        const uint8_t *sp = (const uint8_t *)scales + (size_t)r * 4u;
        uint32_t bits = (uint32_t)sp[0] | (uint32_t)sp[1] << 8 |
            (uint32_t)sp[2] << 16 | (uint32_t)sp[3] << 24;
        float scale;
        memcpy(&scale, &bits, sizeof scale);
        if (!isfinite(scale) || scale < 0.0f) return -1;
    }
    for (int r = first_row; r < end_row; r++) {
        const uint8_t *wr = values + (size_t)r * stride;
        const uint8_t *sp = (const uint8_t *)scales + (size_t)r * 4u;
        uint32_t bits = (uint32_t)sp[0] | (uint32_t)sp[1] << 8 |
            (uint32_t)sp[2] << 16 | (uint32_t)sp[3] << 24;
        float scale;
        memcpy(&scale, &bits, sizeof scale);
        for (int b0 = 0; b0 < batch; b0 += TOKEN_TILE) {
            int count = batch - b0 < TOKEN_TILE ? batch - b0 : TOKEN_TILE;
            float32x4_t lo[TOKEN_TILE], hi[TOKEN_TILE];
            for (int b = 0; b < TOKEN_TILE; b++) {
                lo[b] = vdupq_n_f32(0.0f);
                hi[b] = vdupq_n_f32(0.0f);
            }
            int c = 0;
            for (; c + 7 < cols; c += 8) {
                uint32x4_t q0 = vandq_u32(vshlq_u32(vdupq_n_u32(wr[c / 4]), shifts), mask);
                uint32x4_t q1 = vandq_u32(vshlq_u32(vdupq_n_u32(wr[c / 4 + 1]), shifts), mask);
                float32x4_t w0 = vmulq_n_f32(vcvtq_f32_s32(
                    vsubq_s32(vreinterpretq_s32_u32(q0), center)), scale);
                float32x4_t w1 = vmulq_n_f32(vcvtq_f32_s32(
                    vsubq_s32(vreinterpretq_s32_u32(q1), center)), scale);
                for (int b = 0; b < TOKEN_TILE; b++) if (b < count) {
                    const float *x = inputs + (size_t)(b0 + b) * cols + c;
                    lo[b] = vaddq_f32(lo[b], vmulq_f32(w0, vld1q_f32(x)));
                    hi[b] = vaddq_f32(hi[b], vmulq_f32(w1, vld1q_f32(x + 4)));
                }
            }
            for (int b = 0; b < count; b++) {
                float32x2_t sum2 = vadd_f32(vget_low_f32(lo[b]), vget_high_f32(lo[b]));
                sum2 = vadd_f32(sum2, vadd_f32(vget_low_f32(hi[b]), vget_high_f32(hi[b])));
                float sum = vget_lane_f32(sum2, 0) + vget_lane_f32(sum2, 1);
                const float *x = inputs + (size_t)(b0 + b) * cols;
                for (int tail = c; tail < cols; tail++) {
                    int code = (wr[tail / 4] >> (2 * (tail % 4))) & 3;
                    float weight = (float)(code - 2) * scale;
                    float product = weight * x[tail];
                    sum += product;
                }
                if (!isfinite(sum)) return -1;
                outputs[(size_t)(b0 + b) * rows + r] = sum;
            }
        }
    }
    return 0;
#else
    (void)values; (void)scales; (void)rows; (void)cols; (void)batch;
    (void)inputs; (void)outputs; (void)first_row; (void)end_row;
    return 1;
#endif
}
