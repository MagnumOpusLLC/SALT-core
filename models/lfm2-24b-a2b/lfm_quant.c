/* LFM2.5-24B-A2B model-owned MLX affine group-64 6-bit module.
 *
 * Arithmetic contract, matching the other SaltQuant modules: one row at a
 * time, groups in ascending order, values within a group in ascending index
 * order, staged dequantization (code * scale + bias) before the multiply, and
 * an ascending accumulation into y. The build uses -ffp-contract=off, so no
 * multiply-add is contracted. Nothing here is shared with the 4-bit module:
 * different representations produce different values by construction. */

#include "lfm_quant.h"

#include <stddef.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

static float lfm_bf16(uint16_t bits) {
    uint32_t wide = (uint32_t)bits << 16;
    float value;
    /* BF16 to F32 is a representation conversion, not arithmetic. */
    __builtin_memcpy(&value, &wide, sizeof value);
    return value;
}

static uint32_t lfm_load24(const unsigned char *bytes) {
    return (uint32_t)bytes[0] |
           ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u);
}

int lfm_affine6_dequant_row(const void *vals, const void *scales,
                            const void *biases, int cols, float *out) {
    const uint32_t *words = (const uint32_t *)vals;
    const uint16_t *sc = (const uint16_t *)scales;
    const uint16_t *bi = (const uint16_t *)biases;
    int groups, index = 0;
    if (!words || !sc || !bi || !out || cols < 1 || cols % LFM_AFFINE6_GROUP)
        return -1;
    groups = cols / LFM_AFFINE6_GROUP;
    for (int g = 0; g < groups; g++) {
        float scale = lfm_bf16(sc[g]);
        float bias = lfm_bf16(bi[g]);
        for (int k = 0; k < LFM_AFFINE6_GROUP; k++, index++)
            out[index] = (float)lfm_affine6_code(words, (unsigned)index) * scale + bias;
    }
    return 0;
}

static void lfm_affine6_rows4(const uint32_t *vals,
        const uint16_t *scales, const uint16_t *biases,
        int rows, int cols, int batch, const float *xs, float *ys,
        int row) {
    size_t bytes_per_row = (size_t)cols * LFM_AFFINE6_BITS / 8u;
    size_t groups_per_row = (size_t)cols / LFM_AFFINE6_GROUP;
    const unsigned char *packed_row[4];
    const uint16_t *scale[4], *bias[4];
    for (int lane = 0; lane < 4; lane++) {
        packed_row[lane] = (const unsigned char *)(const void *)vals +
            (size_t)(row + lane) * bytes_per_row;
        scale[lane] = scales + (size_t)(row + lane) * groups_per_row;
        bias[lane] = biases + (size_t)(row + lane) * groups_per_row;
    }
    for (int b = 0; b < batch; b++) {
        const float *input = xs + (size_t)b * (size_t)cols;
#if defined(__aarch64__)
        float32x4_t accumulator = vdupq_n_f32(0.0f);
        for (int group = 0; group < cols / LFM_AFFINE6_GROUP; group++) {
            float sv_data[4], bv_data[4];
            for (int lane = 0; lane < 4; lane++) {
                sv_data[lane] = lfm_bf16(scale[lane][group]);
                bv_data[lane] = lfm_bf16(bias[lane][group]);
            }
            float32x4_t sv = vld1q_f32(sv_data);
            float32x4_t bv = vld1q_f32(bv_data);
            int first = group * LFM_AFFINE6_GROUP;
            for (int pack = 0; pack < LFM_AFFINE6_GROUP / 4; pack++) {
                uint32_t bits[4];
                for (int lane = 0; lane < 4; lane++)
                    bits[lane] = lfm_load24(
                        packed_row[lane] + (size_t)group * 48u +
                        (size_t)pack * 3u);
                for (int sub = 0; sub < 4; sub++) {
                    float qv_data[4];
                    for (int lane = 0; lane < 4; lane++)
                        qv_data[lane] =
                            (float)((bits[lane] >> (6u * (unsigned)sub)) & 63u);
                    int column = first + pack * 4 + sub;
                    float32x4_t qv = vld1q_f32(qv_data);
                    float32x4_t xv = vdupq_n_f32(input[column]);
                    float32x4_t left = vmulq_f32(qv, sv);
                    left = vmulq_f32(left, xv);
                    float32x4_t right = vmulq_f32(bv, xv);
                    accumulator = vaddq_f32(
                        accumulator, vaddq_f32(left, right));
                }
            }
        }
        vst1q_f32(ys + (size_t)b * (size_t)rows + (size_t)row,
                   accumulator);
#elif defined(__x86_64__) || defined(_M_X64)
        __m128 accumulator = _mm_setzero_ps();
        for (int group = 0; group < cols / LFM_AFFINE6_GROUP; group++) {
            float sv_data[4], bv_data[4];
            for (int lane = 0; lane < 4; lane++) {
                sv_data[lane] = lfm_bf16(scale[lane][group]);
                bv_data[lane] = lfm_bf16(bias[lane][group]);
            }
            __m128 sv = _mm_loadu_ps(sv_data);
            __m128 bv = _mm_loadu_ps(bv_data);
            int first = group * LFM_AFFINE6_GROUP;
            for (int pack = 0; pack < LFM_AFFINE6_GROUP / 4; pack++) {
                uint32_t bits[4];
                for (int lane = 0; lane < 4; lane++)
                    bits[lane] = lfm_load24(
                        packed_row[lane] + (size_t)group * 48u +
                        (size_t)pack * 3u);
                for (int sub = 0; sub < 4; sub++) {
                    float qv_data[4];
                    for (int lane = 0; lane < 4; lane++)
                        qv_data[lane] =
                            (float)((bits[lane] >> (6u * (unsigned)sub)) & 63u);
                    int column = first + pack * 4 + sub;
                    __m128 qv = _mm_loadu_ps(qv_data);
                    __m128 xv = _mm_set1_ps(input[column]);
                    __m128 left = _mm_mul_ps(qv, sv);
                    left = _mm_mul_ps(left, xv);
                    __m128 right = _mm_mul_ps(bv, xv);
                    accumulator = _mm_add_ps(
                        accumulator, _mm_add_ps(left, right));
                }
            }
        }
        _mm_storeu_ps(ys + (size_t)b * (size_t)rows + (size_t)row,
                      accumulator);
#else
        for (int lane = 0; lane < 4; lane++) {
            float accumulator = 0.0f;
            for (int group = 0; group < cols / LFM_AFFINE6_GROUP; group++) {
                float group_scale = lfm_bf16(scale[lane][group]);
                float group_bias = lfm_bf16(bias[lane][group]);
                int first = group * LFM_AFFINE6_GROUP;
                for (int pack = 0; pack < LFM_AFFINE6_GROUP / 4; pack++) {
                    uint32_t bits = lfm_load24(
                        packed_row[lane] + (size_t)group * 48u +
                        (size_t)pack * 3u);
                    for (int sub = 0; sub < 4; sub++) {
                        int column = first + pack * 4 + sub;
                        float left =
                            (float)((bits >> (6u * (unsigned)sub)) & 63u) *
                            group_scale;
                        left *= input[column];
                        float right = group_bias * input[column];
                        accumulator += left + right;
                    }
                }
            }
            ys[(size_t)b * (size_t)rows + (size_t)(row + lane)] = accumulator;
        }
#endif
    }
}

int lfm_affine6_matvec_batch_rows(const void *vals, const void *scales,
        const void *biases, int rows, int cols, int batch,
        const float *xs, float *ys, int row_start, int row_end) {
    const uint32_t *values = (const uint32_t *)vals;
    const uint16_t *scale = (const uint16_t *)scales;
    const uint16_t *bias = (const uint16_t *)biases;
    size_t words_per_row, groups_per_row;
    if (!values || !scale || !bias || !xs || !ys || rows < 1 || cols < 1 ||
        batch < 1 || cols % LFM_AFFINE6_GROUP || row_start < 0 ||
        row_start > row_end || row_end > rows)
        return -1;
    words_per_row = (size_t)cols * LFM_AFFINE6_BITS / 32u;
    groups_per_row = (size_t)cols / LFM_AFFINE6_GROUP;
    int row = row_start;
    for (; row + 3 < row_end; row += 4)
        lfm_affine6_rows4(values, scale, bias, rows, cols, batch, xs, ys, row);
    for (; row < row_end; row++) {
        const uint32_t *row_words = values + (size_t)row * words_per_row;
        const uint16_t *row_scales = scale + (size_t)row * groups_per_row;
        const uint16_t *row_biases = bias + (size_t)row * groups_per_row;
        for (int b = 0; b < batch; b++) {
            const float *input = xs + (size_t)b * (size_t)cols;
            float accumulator = 0.0f;
            for (int group = 0; group < cols / LFM_AFFINE6_GROUP; group++) {
                float group_scale = lfm_bf16(row_scales[group]);
                float group_bias = lfm_bf16(row_biases[group]);
                int first = group * LFM_AFFINE6_GROUP;
                int last = first + LFM_AFFINE6_GROUP;
                for (int column = first; column < last; column++) {
                    float left = (float)lfm_affine6_code(
                        row_words, (unsigned)column) * group_scale;
                    left *= input[column];
                    float right = group_bias * input[column];
                    accumulator += left + right;
                }
            }
            ys[(size_t)b * (size_t)rows + (size_t)row] = accumulator;
        }
    }
    return 0;
}

int lfm_affine6_matvec(const void *vals, const void *scales, const void *biases,
                       int rows, int cols, const float *x, float *y) {
    return lfm_affine6_matvec_batch_rows(
        vals, scales, biases, rows, cols, 1, x, y, 0, rows);
}

int lfm_affine6_matvec_batch(const void *vals, const void *scales,
                             const void *biases, int rows, int cols, int batch,
                             const float *xs, float *ys) {
    return lfm_affine6_matvec_batch_rows(
        vals, scales, biases, rows, cols, batch, xs, ys, 0, rows);
}

const SaltQuant salt_quant_affine6 = {
    LFM_AFFINE6_BITS, 0, 0, 0, 0, "affine6", lfm_affine6_matvec,
    lfm_affine6_matvec_batch };
