#include "salt/kernels.h"
#include "salt/simd.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng_state = 0x9e3779b9u;
static uint32_t rng32(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}

static uint16_t bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static int run_shape(int R, int C) {
    enum { BATCH = 3 };
    size_t elements = (size_t)R * (size_t)C;
    size_t words = elements / 8u;
    size_t groups = elements / SALT_MLX4_GROUP;
    uint32_t *values = (uint32_t *)malloc(words * sizeof *values);
    uint16_t *scales = (uint16_t *)malloc(groups * sizeof *scales);
    uint16_t *biases = (uint16_t *)malloc(groups * sizeof *biases);
    float *x = (float *)malloc((size_t)C * sizeof *x);
    float *scalar = (float *)malloc((size_t)R * sizeof *scalar);
    float *simd = (float *)malloc((size_t)R * sizeof *simd);
    float *batch_x = (float *)malloc(
        (size_t)BATCH * (size_t)C * sizeof *batch_x);
    float *batch_expected = (float *)malloc(
        (size_t)BATCH * (size_t)R * sizeof *batch_expected);
    float *batch_actual = (float *)malloc(
        (size_t)BATCH * (size_t)R * sizeof *batch_actual);
    float *batch_tiled = (float *)calloc(
        (size_t)BATCH * (size_t)R, sizeof *batch_tiled);
    if (!values || !scales || !biases || !x || !scalar || !simd ||
        !batch_x || !batch_expected || !batch_actual || !batch_tiled) {
        free(batch_tiled);
        free(batch_actual); free(batch_expected); free(batch_x);
        free(simd); free(scalar); free(x); free(biases); free(scales); free(values);
        return -1;
    }
    for (size_t i = 0; i < words; i++) values[i] = rng32();
    for (size_t i = 0; i < groups; i++) {
        scales[i] = bf16((float)((i % 7u) + 1u) * 0.037f);
        biases[i] = bf16((float)((int)(i % 9u) - 4) * 0.071f);
    }
    for (int c = 0; c < C; c++)
        x[c] = (float)((int)(rng32() % 257u) - 128) * 0.0078123f;
    salt_kernels_set_simd(0);
    if (salt_q4_matvec_status(values, scales, biases, R, C, x, scalar) != 0)
        return -1;
    salt_kernels_set_simd(1);
    if (salt_simd_set_q4_row_pair(0) != 0 ||
        salt_q4_matvec_status(values, scales, biases, R, C, x, simd) != 0)
        return -1;
    if (memcmp(scalar, simd, (size_t)R * sizeof *scalar) != 0) {
        for (int r = 0; r < R; r++) {
            if (memcmp(&scalar[r], &simd[r], sizeof(float)) != 0) {
                fprintf(stderr,
                        "FAIL q4 %dx%d row=%d scalar=%.9g simd=%.9g\n",
                        R, C, r, scalar[r], simd[r]);
                break;
            }
        }
        free(simd); free(scalar); free(x); free(biases); free(scales); free(values);
        return -1;
    }
    if (salt_simd_set_q4_row_pair(1) != 0 ||
        salt_q4_matvec_status(values, scales, biases, R, C, x, simd) != 0 ||
        memcmp(scalar, simd, (size_t)R * sizeof *scalar) != 0) {
        for (int r = 0; r < R; r++)
            if (memcmp(&scalar[r], &simd[r], sizeof(float)) != 0) {
                fprintf(stderr,
                        "FAIL q4-row-pair %dx%d row=%d scalar=%.9g "
                        "simd=%.9g\n",
                        R, C, r, scalar[r], simd[r]);
                break;
            }
        free(simd); free(scalar); free(x); free(biases); free(scales); free(values);
        return -1;
    }
    printf("PASS q4 %dx%d single/pair memcmp=0\n", R, C);
    for (int b = 0; b < BATCH; b++) {
        float *input = batch_x + (size_t)b * C;
        float *expected = batch_expected + (size_t)b * R;
        for (int c = 0; c < C; c++)
            input[c] = (float)((int)(rng32() % 257u) - 128) * 0.0078123f;
        if (salt_q4_matvec_status(
                values, scales, biases, R, C, input, expected) != 0)
            return -1;
    }
    if (salt_q4_matvec_batch(values, scales, biases, R, C, BATCH,
                             batch_x, batch_actual) != 0 ||
        memcmp(batch_expected, batch_actual,
               (size_t)BATCH * (size_t)R * sizeof(float)) != 0) {
        for (int b = 0; b < BATCH; b++)
            for (int r = 0; r < R; r++) {
                size_t i = (size_t)b * R + r;
                if (memcmp(&batch_expected[i], &batch_actual[i],
                           sizeof(float)) != 0) {
                    fprintf(stderr,
                        "FAIL q4-batch %dx%d B=%d token=%d row=%d "
                        "single=%.9g batch=%.9g\n",
                        R, C, BATCH, b, r,
                        batch_expected[i], batch_actual[i]);
                    b = BATCH;
                    break;
                }
            }
        free(batch_tiled); free(batch_actual); free(batch_expected); free(batch_x);
        free(simd); free(scalar); free(x); free(biases); free(scales); free(values);
        return -1;
    }
    printf("PASS q4-batch %dx%d B=%d memcmp=0\n", R, C, BATCH);
    {
        int split_b = BATCH > 1 ? BATCH / 2 : BATCH;
        int split_r = R > 1 ? R / 2 : R;
        if (salt_q4_matvec_batch_tile(values, scales, biases,
                R, C, BATCH, batch_x, batch_tiled,
                0, split_b, 0, split_r) != 0 ||
            salt_q4_matvec_batch_tile(values, scales, biases,
                R, C, BATCH, batch_x, batch_tiled,
                0, split_b, split_r, R) != 0 ||
            salt_q4_matvec_batch_tile(values, scales, biases,
                R, C, BATCH, batch_x, batch_tiled,
                split_b, BATCH, 0, split_r) != 0 ||
            salt_q4_matvec_batch_tile(values, scales, biases,
                R, C, BATCH, batch_x, batch_tiled,
                split_b, BATCH, split_r, R) != 0 ||
            memcmp(batch_actual, batch_tiled,
                (size_t)BATCH * (size_t)R * sizeof(float)) != 0) {
            fprintf(stderr, "FAIL q4-area-tile %dx%d B=%d\n", R, C, BATCH);
            free(batch_tiled); free(batch_actual); free(batch_expected);
            free(batch_x); free(simd); free(scalar); free(x);
            free(biases); free(scales); free(values);
            return -1;
        }
        printf("PASS q4-area-tile %dx%d B=%d memcmp=0\n", R, C, BATCH);
    }
    free(batch_tiled); free(batch_actual); free(batch_expected); free(batch_x);
    free(simd); free(scalar); free(x); free(biases); free(scales); free(values);
    return 0;
}

static int run_weighted_value_shape(int count, int dimension, int padding) {
    int stride = dimension + padding;
    int split = count > 1 ? count / 2 : count;
    size_t value_count = (size_t)count * (size_t)stride;
    float *values = (float *)malloc(value_count * sizeof *values);
    float *scores = (float *)malloc((size_t)count * sizeof *scores);
    float *scalar = (float *)calloc((size_t)dimension, sizeof *scalar);
    float *simd = (float *)calloc((size_t)dimension, sizeof *simd);
    float denominator = 0.0f;
    if (!values || !scores || !scalar || !simd) {
        free(simd); free(scalar); free(scores); free(values);
        return -1;
    }
    for (size_t i = 0; i < value_count; i++)
        values[i] = (float)((int)(rng32() % 2049u) - 1024) * 0.0009765625f;
    for (int position = 0; position < count; position++) {
        scores[position] =
            (float)((rng32() % 1023u) + 1u) * 0.0009765625f;
        denominator += scores[position];
    }
    for (int position = 0; position < count; position++) {
        const float *value = values + (size_t)position * stride;
        float weight = scores[position] / denominator;
        for (int d = 0; d < dimension; d++)
            scalar[d] += weight * value[d];
    }
    salt_simd_weighted_value_accumulate(
        simd, values, stride, scores, split, dimension, denominator);
    if (split < count)
        salt_simd_weighted_value_accumulate(
            simd, values + (size_t)split * stride, stride,
            scores + split, count - split, dimension, denominator);
    if (memcmp(scalar, simd, (size_t)dimension * sizeof *scalar) != 0) {
        for (int d = 0; d < dimension; d++)
            if (memcmp(&scalar[d], &simd[d], sizeof(float)) != 0) {
                fprintf(stderr,
                    "FAIL weighted-value count=%d dimension=%d padding=%d "
                    "d=%d scalar=%.9g simd=%.9g\n",
                    count, dimension, padding, d, scalar[d], simd[d]);
                break;
            }
        free(simd); free(scalar); free(scores); free(values);
        return -1;
    }
    printf("PASS weighted-value count=%d dimension=%d padding=%d memcmp=0\n",
           count, dimension, padding);
    free(simd); free(scalar); free(scores); free(values);
    return 0;
}

int main(void) {
    static const int shapes[][2] = {
        {704, 2816},
        {705, 2816},
        {2048, 2816},
        {2816, 8192},
    };
    if (!salt_simd_available()) {
        puts("SKIP q4 bit identity: no SIMD substrate");
        return 0;
    }
    if (salt_simd_set_q4_row_pair(2) == 0)
        return 1;
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        if (run_shape(shapes[i][0], shapes[i][1]) != 0)
            return 1;
    if (run_weighted_value_shape(1, 1, 0) != 0 ||
        run_weighted_value_shape(7, 3, 2) != 0 ||
        run_weighted_value_shape(200, 256, 3) != 0 ||
        run_weighted_value_shape(1024, 256, 0) != 0 ||
        run_weighted_value_shape(1500, 512, 3) != 0)
        return 1;
    puts("q4 canonical substrate identity: PASS");
    return 0;
}
