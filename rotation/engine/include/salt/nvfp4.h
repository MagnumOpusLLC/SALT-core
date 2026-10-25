#ifndef SALT_NVFP4_H
#define SALT_NVFP4_H

#include <stddef.h>
#include <stdint.h>

/* NVIDIA FP4 E2M1 codebook. code is one unpacked nibble. */
float salt_nvfp4_e2m1(uint8_t code);

/* IEEE-style finite FP8 E4M3FN used for NVFP4 block scales. Returns NaN for
 * exponent=15,mantissa=7. This is intentionally distinct from Salt's older
 * clamped FP8 model format. */
float salt_nvfp4_e4m3fn(uint8_t code);

/* Select the optional architecture SIMD realization. Unsupported hosts retain
 * scalar C99. The operation and output bytes are unchanged. */
void salt_nvfp4_set_simd(int enabled);
int salt_nvfp4_simd(void);

/* Round one finite FP32 value to E4M3FN, round-to-nearest-even, saturating to
 * +/-448. NaN is rejected as the canonical NVFP4 scale path never emits it. */
int salt_nvfp4_e4m3fn_encode(float value, uint8_t *code);

/* Dynamic group-16 activation Q/DQ from the compressed-tensors NVFP4
 * reference. All output arrays are caller-owned and fixed at engine init:
 * packed must hold C/2 bytes, scales C/16 bytes, dequant C floats. */
int salt_nvfp4_quantize_input_reference(
    const float *input, int C, float input_global_scale,
    uint8_t *packed, uint8_t *scales, float *dequant);

/* Dequantize a complete U8/F8_E4M3FN/F32 weight matrix without changing
 * orientation or packing. output must hold R*C floats. */
int salt_nvfp4_dequantize_weights_reference(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float stored_weight_global_divisor, int R, int C, float *output);

/* Canonical portable projection reference. Weight layout is U8 packed E2M1,
 * low nibble then high nibble, scales [R][C/16] E4M3FN. Input is dynamically
 * group-16 Q/DQ first. x_dequant, x_packed and x_scales are caller scratch.
 * Per-row accumulation uses the established 32 canonical accumulators and
 * exact fixed fold; compile with -ffp-contract=off. */
int salt_nvfp4_matvec_reference(
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    float stored_weight_global_divisor, float input_global_scale,
    int R, int C, const float *input, float *output,
    uint8_t *x_packed, uint8_t *x_scales, float *x_dequant);

#endif /* SALT_NVFP4_H */
