#include "salt/nvfp4.h"

#include <float.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

static int nvfp4_simd_enabled = 1;

static const float nvfp4_e2m1_positive[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

void salt_nvfp4_set_simd(int enabled) { nvfp4_simd_enabled = enabled ? 1 : 0; }
int salt_nvfp4_simd(void) { return nvfp4_simd_enabled; }

float salt_nvfp4_e2m1(uint8_t code) {
    float value = nvfp4_e2m1_positive[code & 7u];
    if (value == 0.0f) return 0.0f;
    return (code & 8u) ? -value : value;
}

float salt_nvfp4_e4m3fn(uint8_t code) {
    int sign = (code >> 7) & 1;
    int exponent = (code >> 3) & 15;
    int mantissa = code & 7;
    float value;
    if (exponent == 0) {
        value = (float)mantissa * 0.001953125f; /* m * 2^-9 */
    } else if (exponent == 15 && mantissa == 7) {
        return NAN;
    } else {
        value = (1.0f + (float)mantissa * 0.125f) *
                ldexpf(1.0f, exponent - 7);
    }
    return sign ? -value : value;
}

int salt_nvfp4_e4m3fn_encode(float value, uint8_t *code) {
    uint8_t sign;
    float magnitude;
    uint8_t best = 0;
    float best_distance = FLT_MAX;
    if (!code || isnan(value)) return -1;
    sign = signbit(value) ? 0x80u : 0u;
    magnitude = fabsf(value);
    if (magnitude > 448.0f) magnitude = 448.0f;
    /* Positive finite E4M3FN codes are 0x00..0x7e; 0x7f is NaN. */
    for (unsigned int candidate = 0; candidate <= 0x7eu; candidate++) {
        float decoded = salt_nvfp4_e4m3fn((uint8_t)candidate);
        float distance = fabsf(magnitude - decoded);
        if (distance < best_distance ||
            (distance == best_distance &&
             ((candidate & 1u) == 0u) && ((best & 1u) != 0u))) {
            best_distance = distance;
            best = (uint8_t)candidate;
        }
    }
    *code = (uint8_t)(best | sign);
    return 0;
}

static uint8_t nvfp4_e2m1_encode(float value) {
    float magnitude = fabsf(value);
    uint8_t best = 0;
    float best_distance = FLT_MAX;
    for (unsigned int candidate = 0; candidate < 8u; candidate++) {
        float distance = fabsf(magnitude - nvfp4_e2m1_positive[candidate]);
        if (distance < best_distance ||
            (distance == best_distance &&
             ((candidate & 1u) == 0u) && ((best & 1u) != 0u))) {
            best_distance = distance;
            best = (uint8_t)candidate;
        }
    }
    if (signbit(value)) best |= 8u;
    return best;
}

int salt_nvfp4_quantize_input_reference(
    const float *input, int C, float input_global_scale,
    uint8_t *packed, uint8_t *scales, float *dequant) {
    if (!input || !packed || !scales || !dequant || C < 16 ||
        (C & 15) != 0 || !isfinite(input_global_scale) ||
        input_global_scale <= 0.0f)
        return -1;
    for (int block = 0; block < C / 16; block++) {
        int base = block * 16;
        float maximum = 0.0f;
        float scaled_maximum;
        float raw_scale;
        uint8_t scale_code;
        float scale;
        float output_scale;
        float dequant_scale;
        for (int lane = 0; lane < 16; lane++) {
            float magnitude = fabsf(input[base + lane]);
            if (!isfinite(magnitude)) return -1;
            if (magnitude > maximum) maximum = magnitude;
        }
        scaled_maximum = maximum * (1.0f / 6.0f);
        raw_scale = input_global_scale * scaled_maximum;
        if (raw_scale > 448.0f) raw_scale = 448.0f;
        if (salt_nvfp4_e4m3fn_encode(raw_scale, &scale_code) != 0) return -1;
        scales[block] = scale_code;
        scale = salt_nvfp4_e4m3fn(scale_code);
        output_scale = scale == 0.0f ? 0.0f : input_global_scale / scale;
        dequant_scale = scale / input_global_scale;
        for (int lane = 0; lane < 16; lane += 2) {
            float first = input[base + lane] * output_scale;
            float second = input[base + lane + 1] * output_scale;
            uint8_t first_code;
            uint8_t second_code;
            if (first > 6.0f) first = 6.0f;
            if (first < -6.0f) first = -6.0f;
            if (second > 6.0f) second = 6.0f;
            if (second < -6.0f) second = -6.0f;
            first_code = nvfp4_e2m1_encode(first);
            second_code = nvfp4_e2m1_encode(second);
            packed[(base + lane) / 2] =
                (uint8_t)(first_code | (uint8_t)(second_code << 4));
            dequant[base + lane] = salt_nvfp4_e2m1(first_code) * dequant_scale;
            dequant[base + lane + 1] =
                salt_nvfp4_e2m1(second_code) * dequant_scale;
        }
    }
    return 0;
}

int salt_nvfp4_dequantize_weights_reference(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float stored_weight_global_divisor, int R, int C, float *output) {
    float weight_global_scale;
    if (!weight_packed || !weight_scales || !output || R < 1 || C < 16 ||
        (C & 15) != 0 || !isfinite(stored_weight_global_divisor) ||
        stored_weight_global_divisor == 0.0f)
        return -1;
    weight_global_scale = 1.0f / stored_weight_global_divisor;
    for (int row = 0; row < R; row++) {
        for (int col = 0; col < C; col++) {
            size_t index = (size_t)row * (size_t)C + (size_t)col;
            uint8_t packed = weight_packed[index >> 1u];
            uint8_t code = (index & 1u) ? (uint8_t)(packed >> 4) :
                                          (uint8_t)(packed & 15u);
            size_t scale_index = (size_t)row * (size_t)(C / 16) +
                                 (size_t)(col / 16);
            float scale = salt_nvfp4_e4m3fn(weight_scales[scale_index]);
            float scaled = scale * weight_global_scale;
            output[index] = salt_nvfp4_e2m1(code) * scaled;
        }
    }
    return 0;
}

static float nvfp4_fold_accumulators(const float acc[32]) {
    float lane0 = 0.0f, lane1 = 0.0f;
    float lane2 = 0.0f, lane3 = 0.0f;
    for (int a = 0; a < 8; a++) {
        lane0 += acc[4 * a];
        lane1 += acc[4 * a + 1];
        lane2 += acc[4 * a + 2];
        lane3 += acc[4 * a + 3];
    }
    {
        float t0 = lane0 + lane2;
        float t1 = lane1 + lane3;
        return t0 + t1;
    }
}

static float nvfp4_row_scalar(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float weight_global_scale, int row, int C, const float *x_dequant) {
    float acc[32];
    for (int lane = 0; lane < 32; lane++) acc[lane] = 0.0f;
    for (int col = 0; col < C; col++) {
        size_t index = (size_t)row * (size_t)C + (size_t)col;
        uint8_t packed = weight_packed[index >> 1u];
        uint8_t code = (index & 1u) ? (uint8_t)(packed >> 4) :
                                      (uint8_t)(packed & 15u);
        size_t scale_index = (size_t)row * (size_t)(C / 16) +
                             (size_t)(col / 16);
        float scale = salt_nvfp4_e4m3fn(weight_scales[scale_index]);
        float scaled = scale * weight_global_scale;
        float weight = salt_nvfp4_e2m1(code) * scaled;
        float product = weight * x_dequant[col];
        acc[col & 31] += product;
    }
    return nvfp4_fold_accumulators(acc);
}

#if defined(__aarch64__)
static float nvfp4_row_neon(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float weight_global_scale, int row, int C, const float *x_dequant) {
    float32x4_t accumulators[8];
    float acc[32], decoded[32];
    for (int lane = 0; lane < 8; lane++)
        accumulators[lane] = vdupq_n_f32(0.0f);
    for (int base = 0; base < C; base += 32) {
        for (int lane = 0; lane < 32; lane++) {
            int col = base + lane;
            size_t index = (size_t)row * (size_t)C + (size_t)col;
            uint8_t packed = weight_packed[index >> 1u];
            uint8_t code = (index & 1u) ? (uint8_t)(packed >> 4) :
                                              (uint8_t)(packed & 15u);
            size_t scale_index = (size_t)row * (size_t)(C / 16) +
                                 (size_t)(col / 16);
            float scale = salt_nvfp4_e4m3fn(weight_scales[scale_index]);
            float scaled = scale * weight_global_scale;
            decoded[lane] = salt_nvfp4_e2m1(code) * scaled;
        }
        for (int vector = 0; vector < 8; vector++) {
            float32x4_t weights = vld1q_f32(decoded + vector * 4);
            float32x4_t inputs = vld1q_f32(x_dequant + base + vector * 4);
            float32x4_t products = vmulq_f32(weights, inputs);
            accumulators[vector] = vaddq_f32(accumulators[vector], products);
        }
    }
    for (int vector = 0; vector < 8; vector++)
        vst1q_f32(acc + vector * 4, accumulators[vector]);
    return nvfp4_fold_accumulators(acc);
}
#endif

int salt_nvfp4_matvec_reference(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float stored_weight_global_divisor, float input_global_scale,
    int R, int C, const float *input, float *output,
    uint8_t *x_packed, uint8_t *x_scales, float *x_dequant) {
    if (!weight_packed || !weight_scales || !input || !output ||
        !x_packed || !x_scales || !x_dequant || R < 1 || C < 16 ||
        (C & 15) != 0 || !isfinite(stored_weight_global_divisor) ||
        stored_weight_global_divisor == 0.0f)
        return -1;
    float weight_global_scale = 1.0f / stored_weight_global_divisor;
    if (salt_nvfp4_quantize_input_reference(
            input, C, input_global_scale,
            x_packed, x_scales, x_dequant) != 0)
        return -1;
    for (int row = 0; row < R; row++) {
#if defined(__aarch64__)
        if (nvfp4_simd_enabled && C % 32 == 0)
            output[row] = nvfp4_row_neon(
                weight_packed, weight_scales, weight_global_scale,
                row, C, x_dequant);
        else
#endif
            output[row] = nvfp4_row_scalar(
                weight_packed, weight_scales, weight_global_scale,
                row, C, x_dequant);
    }
    return 0;
}
