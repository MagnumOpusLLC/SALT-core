#include "salt/gpu.h"
#include "salt/kernels.h"
#include "salt/salt.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    TEST_R = 64,
    TEST_C = 64,
    TEST_GROUP = 64,
    TEST_VAL_WORDS = (TEST_R * TEST_C) / 8,
    TEST_GROUPS = (TEST_R * TEST_C) / TEST_GROUP,
    TEST_VAL_BYTES = TEST_VAL_WORDS * 4,
    TEST_META_BYTES = TEST_GROUPS * 2,
    TEST_LAYER_BYTES = TEST_VAL_BYTES + 2 * TEST_META_BYTES,
    TEST_SOURCE_BYTES = 70000
};

static void fill_layer(uint8_t *layer, int seed) {
    uint32_t *vals = (uint32_t *)(void *)layer;
    uint16_t *scales = (uint16_t *)(void *)(layer + TEST_VAL_BYTES);
    uint16_t *biases = (uint16_t *)(void *)(
        layer + TEST_VAL_BYTES + TEST_META_BYTES);
    memset(layer, 0, TEST_LAYER_BYTES);
    for (uint64_t k = 0; k < (uint64_t)TEST_R * TEST_C; k++) {
        uint32_t q = (uint32_t)((k + (uint64_t)seed) & 15u);
        vals[k >> 3] |= q << ((k & 7u) * 4u);
    }
    for (int group = 0; group < TEST_GROUPS; group++) {
        scales[group] = 0x3e00u; /* bf16 0.125 */
        biases[group] = 0xbf00u; /* bf16 -0.5 */
    }
}

static float bf16_to_f32(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}

static void q4_reference(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, const float *x, float *y) {
    for (int row = 0; row < TEST_R; row++) {
        float acc = 0.0f;
        for (int col = 0; col < TEST_C; col++) {
            uint64_t k = (uint64_t)(unsigned)row * TEST_C +
                         (uint64_t)(unsigned)col;
            uint64_t group = k / TEST_GROUP;
            uint32_t q = (vals[k >> 3] >> ((k & 7u) * 4u)) & 15u;
            float weight = (float)q * bf16_to_f32(scales[group]);
            weight += bf16_to_f32(biases[group]);
            acc += weight * x[col];
        }
        y[row] = acc;
    }
}

static int write_fixture(const char *offsets_path, const char *bin_path,
                         const uint64_t offsets[2]) {
    uint8_t *source = (uint8_t *)calloc(TEST_SOURCE_BYTES, 1);
    FILE *file = NULL;
    uint64_t n_layers = 2;
    uint64_t pair[2];
    if (!source) return -1;
    fill_layer(source + offsets[0], 1);
    fill_layer(source + offsets[1], 7);

    file = fopen(bin_path, "wb");
    if (!file || fwrite(source, 1, TEST_SOURCE_BYTES, file) !=
                     TEST_SOURCE_BYTES || fclose(file) != 0) {
        if (file) fclose(file);
        free(source);
        return -1;
    }
    file = fopen(offsets_path, "wb");
    if (!file || fwrite(&n_layers, sizeof n_layers, 1, file) != 1) {
        if (file) fclose(file);
        free(source);
        return -1;
    }
    for (int layer = 0; layer < 2; layer++) {
        pair[0] = offsets[layer];
        pair[1] = TEST_LAYER_BYTES;
        if (fwrite(pair, sizeof pair, 1, file) != 1) {
            fclose(file);
            free(source);
            return -1;
        }
    }
    if (fclose(file) != 0) {
        free(source);
        return -1;
    }
    free(source);
    return 0;
}

#define MIX_B 6
#define MIX_GUARD 16

static int mixed_case(const uint32_t *v, const uint16_t *s,
                      const uint16_t *b, const float *xs,
                      const float *expected, SaltGpuSharedBuffer *output,
                      const void *key, const char *order,
                      int gpu_first_job, const char *label) {
    uint32_t *bits = (uint32_t *)output->contents;
    const size_t body = (size_t)MIX_B * TEST_R;
    const size_t total = body + 2u * MIX_GUARD;
    const uint32_t guard = UINT32_C(0x7fc12345);
    for (size_t i = 0; i < total; i++) bits[i] = guard;
    if (order) setenv("SALT_GPU_MIX_TEST_ORDER", order, 1);
    else unsetenv("SALT_GPU_MIX_TEST_ORDER");
    if (salt_gpu_mixed_proj_batch(v, s, b, TEST_R, TEST_C, MIX_B,
                                  gpu_first_job, MIX_B / 2, xs, output,
                                  MIX_GUARD, key) != 0 ||
        salt_gpu_sync() != 0) {
        fprintf(stderr, "%s: mixed submit/sync failed\n", label);
        unsetenv("SALT_GPU_MIX_TEST_ORDER");
        return -1;
    }
    unsetenv("SALT_GPU_MIX_TEST_ORDER");
    for (size_t i = 0; i < MIX_GUARD; i++) {
        if (bits[i] != guard || bits[MIX_GUARD + body + i] != guard) {
            fprintf(stderr, "%s: destination guard changed at %zu\n",
                    label, i);
            return -1;
        }
    }
    if (memcmp((float *)output->contents + MIX_GUARD, expected,
               body * sizeof(float)) != 0) {
        fprintf(stderr, "%s: mixed output is not byte exact\n", label);
        return -1;
    }
    return 0;
}

static int mixed_indexed_test(const uint32_t *v, const uint16_t *s,
                              const uint16_t *b, const void *key) {
    float xs[MIX_B * TEST_C];
    float expected[MIX_B * TEST_R];
    SaltGpuSharedBuffer output = {0};
    SaltGpuBatchStats before, after;
    size_t body = (size_t)MIX_B * TEST_R;
    size_t total = body + 2u * MIX_GUARD;
    int prior_scope, rc;
    for (int j = 0; j < MIX_B; j++)
        for (int c = 0; c < TEST_C; c++)
            xs[(size_t)j * TEST_C + c] =
                (float)(((j + 3) * 29 + c * 11) % 61 - 30) / 19.0f;
    prior_scope = salt_kernels_cpu_expert_scope();
    salt_kernels_set_cpu_expert_scope(1);
    rc = salt_q4_matvec_batch(v, s, b, TEST_R, TEST_C, MIX_B,
                              xs, expected);
    salt_kernels_set_cpu_expert_scope(prior_scope);
    if (rc != 0 || salt_gpu_shared_buffer_alloc(
            &output, total * sizeof(float)) != 0) {
        fprintf(stderr, "mixed-indexed: reference/allocation failed\n");
        return -1;
    }
    salt_gpu_batch_stats_get(&before);
    if (mixed_case(v, s, b, xs, expected, &output, key,
                   "gpu-first", 0, "mixed-front-gpu-first") != 0 ||
        mixed_case(v, s, b, xs, expected, &output, key,
                   "cpu-first", 0, "mixed-front-cpu-first") != 0 ||
        mixed_case(v, s, b, xs, expected, &output, key,
                   "gpu-first", MIX_B / 2, "mixed-back-gpu-first") != 0 ||
        mixed_case(v, s, b, xs, expected, &output, key,
                   "cpu-first", MIX_B / 2, "mixed-back-cpu-first") != 0 ||
        mixed_case(v, s, b, xs, expected, &output, key,
                   NULL, 0, "mixed-concurrent") != 0) {
        (void)salt_gpu_shared_buffer_free(&output);
        return -1;
    }
    salt_gpu_batch_stats_get(&after);
    if (after.direct_output_batches - before.direct_output_batches != 5 ||
        after.direct_output_jobs - before.direct_output_jobs !=
            5u * (MIX_B / 2)) {
        fprintf(stderr, "mixed-indexed: direct counters mismatch\n");
        (void)salt_gpu_shared_buffer_free(&output);
        return -1;
    }
    /* The full canonical B*R destination does not fit from this offset.
     * Rejection must occur before any command is submitted. */
    if (salt_gpu_proj_batch_indexed(v, s, b, TEST_R, TEST_C, MIX_B,
                                    0, MIX_B / 2, xs, &output,
                                    total, key) == 0) {
        fprintf(stderr, "mixed-indexed: out-of-range destination accepted\n");
        (void)salt_gpu_sync();
        (void)salt_gpu_shared_buffer_free(&output);
        return -1;
    }
    {
        const size_t wrapping_offsets[2] = {
            SIZE_MAX - body / 2u,
            (size_t)UINT32_MAX - body / 2u
        };
        SaltGpuBatchStats rejected_before, rejected_after;
        salt_gpu_batch_stats_get(&rejected_before);
        for (size_t i = 0; i < 2; i++) {
            if (salt_gpu_proj_batch_indexed(
                    v, s, b, TEST_R, TEST_C, MIX_B, 0, MIX_B / 2,
                    xs, &output, wrapping_offsets[i], key) == 0 ||
                salt_gpu_mixed_proj_batch(
                    v, s, b, TEST_R, TEST_C, MIX_B, 0, MIX_B / 2,
                    xs, &output, wrapping_offsets[i], key) == 0) {
                fprintf(stderr,
                        "mixed-indexed: wrapping destination accepted\n");
                (void)salt_gpu_sync();
                (void)salt_gpu_shared_buffer_free(&output);
                return -1;
            }
        }
        salt_gpu_batch_stats_get(&rejected_after);
        if (rejected_after.direct_output_batches !=
                rejected_before.direct_output_batches ||
            rejected_after.direct_output_jobs !=
                rejected_before.direct_output_jobs) {
            fprintf(stderr,
                    "mixed-indexed: rejected destination was submitted\n");
            (void)salt_gpu_shared_buffer_free(&output);
            return -1;
        }
    }
    if (salt_gpu_shared_buffer_free(&output) != 0) {
        fprintf(stderr, "mixed-indexed: synchronized free failed\n");
        return -1;
    }
    return 0;
}

static int mapping_is_exact(const SaltTrunk *trunk, int layer,
                            const uint8_t *payload, const uint8_t *base,
                            size_t mapped_nbytes) {
    long page_long = sysconf(_SC_PAGESIZE);
    uint64_t page, source_offset, source_nbytes, map_offset, data_offset;
    if (!trunk || layer < 0 || layer >= trunk->n_layers || !payload || !base ||
        payload < base || page_long <= 0)
        return 0;
    page = (uint64_t)page_long;
    source_offset = (uint64_t)trunk->lay[layer].off;
    source_nbytes = (uint64_t)trunk->lay[layer].nbytes;
    map_offset = source_offset - source_offset % page;
    data_offset = source_offset - map_offset;
    if (source_nbytes > (uint64_t)SIZE_MAX - data_offset)
        return 0;
    return (uint64_t)(payload - base) == data_offset &&
           mapped_nbytes == (size_t)(data_offset + source_nbytes);
}

static int run_second_cycle(const char *offsets_path, const char *bin_path,
                            const float *input) {
    SaltTrunk trunk;
    const uint8_t *payload = NULL, *base = NULL;
    const uint32_t *vals;
    const uint16_t *scales, *biases;
    float cpu[TEST_R], gpu[TEST_R];
    size_t mapped_nbytes = 0;
    int trunk_open = 0, device_active = 0, retained = 0, rc = -1;

    if (salt_trunk_open(&trunk, bin_path, offsets_path) != 0)
        goto done;
    trunk_open = 1;
    if (salt_trunk_start_mapped(&trunk) != 0)
        goto done;
    payload = salt_trunk_bind(&trunk, 1);
    base = (const uint8_t *)salt_trunk_mapping(&trunk, &mapped_nbytes);
    if (!mapping_is_exact(&trunk, 1, payload, base, mapped_nbytes))
        goto done;
    salt_gpu_set_mapped_only(1);
    if (salt_gpu_resident_trunk_map(base, mapped_nbytes) != 0)
        goto done;
    device_active = 1;
    vals = (const uint32_t *)(const void *)payload;
    scales = (const uint16_t *)(const void *)(payload + TEST_VAL_BYTES);
    biases = (const uint16_t *)(const void *)(
        payload + TEST_VAL_BYTES + TEST_META_BYTES);
    q4_reference(vals, scales, biases, input, cpu);
    memset(gpu, 0, sizeof gpu);
    if (salt_gpu_q4_matvec(vals, scales, biases, TEST_R, TEST_C,
                           input, gpu) != 0 ||
        memcmp(cpu, gpu, sizeof cpu) != 0)
        goto done;
    if (salt_gpu_resident_trunk_unmap() != 0)
        goto done;
    device_active = 0;
    salt_gpu_set_mapped_only(0);
    if (salt_trunk_unbind(&trunk) != 0 || salt_trunk_close(&trunk) != 0)
        goto done;
    trunk_open = 0;
    rc = 0;

done:
    if (device_active) {
        if (salt_gpu_resident_trunk_unmap() != 0)
            retained = 1;
        else
            device_active = 0;
    }
    salt_gpu_set_mapped_only(0);
    if (trunk_open && !retained && salt_trunk_unbind(&trunk) != 0)
        retained = 1;
    if (trunk_open && !retained && salt_trunk_close(&trunk) != 0)
        retained = 1;
    return retained ? -2 : rc;
}

int main(void) {
    char offsets_path[128];
    char bin_path[128];
    const uint64_t offsets[2] = {100u, 40000u};
    SaltTrunk trunk;
    SaltGpuBatchStats initial, before, after;
    float input[TEST_C], cpu[TEST_R], gpu[TEST_R];
    uint32_t missing_vals[TEST_VAL_WORDS];
    uint16_t missing_scales[TEST_GROUPS], missing_biases[TEST_GROUPS];
    int device_active = 0;
    int trunk_open = 0;
    int gpu_ready = 0;
    int resource_retained = 0;
    int warm_arena_key = 0;
    int trunk_key = 0;
    int missing_key = 0;
    int rc = 1;

    snprintf(offsets_path, sizeof offsets_path,
             "/tmp/salt-gpu-trunk-%ld.offsets", (long)getpid());
    snprintf(bin_path, sizeof bin_path,
             "/tmp/salt-gpu-trunk-%ld.bin", (long)getpid());
    if (salt_gpu_init() != 0) {
        if (getenv("SALT_REQUIRE_GPU")) {
            fprintf(stderr, "gpu-trunk mapped projection: GPU required\n");
            return 2;
        }
        printf("gpu-trunk mapped projection: SKIP (backend unavailable)\n");
        return 0;
    }
    gpu_ready = 1;
    if (write_fixture(offsets_path, bin_path, offsets) != 0) {
        fprintf(stderr, "fixture write failed\n");
        goto done;
    }
    if (salt_trunk_open(&trunk, bin_path, offsets_path) != 0 ||
        salt_trunk_start_mapped(&trunk) != 0) {
        fprintf(stderr, "mapped trunk open failed\n");
        goto done;
    }
    trunk_open = 1;
    for (int col = 0; col < TEST_C; col++)
        input[col] = (float)((col % 13) - 6) * (1.0f / 16.0f);

    memset(missing_vals, 0, sizeof missing_vals);
    memset(missing_scales, 0, sizeof missing_scales);
    memset(missing_biases, 0, sizeof missing_biases);
    salt_gpu_set_defer(0);
    salt_gpu_set_mapped_only(0);
    salt_gpu_batch_stats_get(&initial);
    {
        const uint32_t *value_jobs[1] = {missing_vals};
        const uint16_t *scale_jobs[1] = {missing_scales};
        const uint16_t *bias_jobs[1] = {missing_biases};
        const float *input_jobs[1] = {input};
        float *output_jobs[1] = {gpu};
        const void *keys[1] = {&warm_arena_key};
        int rows[1] = {TEST_R};
        if (salt_gpu_q4_batch(value_jobs, scale_jobs, bias_jobs,
                              input_jobs, output_jobs, keys, rows,
                              TEST_C, 1) != 0) {
            fprintf(stderr, "arena cache prehistory setup failed\n");
            goto done;
        }
        salt_gpu_batch_stats_get(&before);
        if (before.arena_batches != initial.arena_batches + 1) {
            fprintf(stderr, "arena cache prehistory was not dispatched\n");
            goto done;
        }
        salt_gpu_set_mapped_only(1);
        if (salt_gpu_q4_batch(value_jobs, scale_jobs, bias_jobs,
                              input_jobs, output_jobs, keys, rows,
                              TEST_C, 1) == 0) {
            fprintf(stderr, "warmed arena cache bypassed mapped-only\n");
            goto done;
        }
        salt_gpu_batch_stats_get(&after);
        if (after.arena_batches != before.arena_batches ||
            after.mapped_only_misses != before.mapped_only_misses + 1) {
            fprintf(stderr, "warmed arena rejection accounting failed\n");
            goto done;
        }
        before = after;
    }
    {
        enum { EXPERT_R = 512, EXPERT_B = 2 };
        size_t value_words = (size_t)EXPERT_R * TEST_C / 8u;
        size_t groups = ((size_t)EXPERT_R * TEST_C + 63u) / 64u;
        uint32_t *expert_vals = (uint32_t *)calloc(value_words,
                                                    sizeof *expert_vals);
        uint16_t *expert_scales = (uint16_t *)calloc(groups,
                                                     sizeof *expert_scales);
        uint16_t *expert_biases = (uint16_t *)calloc(groups,
                                                     sizeof *expert_biases);
        float *expert_xs = (float *)calloc((size_t)EXPERT_B * TEST_C,
                                           sizeof *expert_xs);
        float *expert_ys = (float *)calloc((size_t)EXPERT_B * EXPERT_R,
                                           sizeof *expert_ys);
        SaltGpuBatchStats expert_before, expert_after;
        int prior_scope = salt_kernels_cpu_expert_scope();
        int batch_rc;
        if (!expert_vals || !expert_scales || !expert_biases || !expert_xs ||
            !expert_ys || setenv("SALT_GPU", "1", 1) != 0 ||
            setenv("SALT_GPU_TRUNK", "1", 1) != 0 ||
            setenv("SALT_GPU_MOE", "0", 1) != 0 ||
            setenv("SALT_GPU_TRUNK_B", "1", 1) != 0) {
            free(expert_vals); free(expert_scales); free(expert_biases);
            free(expert_xs); free(expert_ys);
            fprintf(stderr, "CPU expert-scope fixture setup failed\n");
            goto done;
        }
        salt_gpu_batch_stats_get(&expert_before);
        salt_kernels_set_cpu_expert_scope(1);
        batch_rc = salt_q4_matvec_batch(
            expert_vals, expert_scales, expert_biases,
            EXPERT_R, TEST_C, EXPERT_B, expert_xs, expert_ys);
        salt_kernels_set_cpu_expert_scope(prior_scope);
        salt_gpu_batch_stats_get(&expert_after);
        free(expert_vals); free(expert_scales); free(expert_biases);
        free(expert_xs); free(expert_ys);
        if (batch_rc != 0 ||
            expert_after.arena_batches != expert_before.arena_batches ||
            expert_after.trunk_batches != expert_before.trunk_batches ||
            expert_after.expert_layer_batches !=
                expert_before.expert_layer_batches ||
            expert_after.mapped_only_misses !=
                expert_before.mapped_only_misses) {
            fprintf(stderr,
                    "CPU expert scope entered mapped GPU routing\n");
            goto done;
        }
    }
    for (int layer = 0; layer < 2; layer++) {
        const uint8_t *payload = salt_trunk_bind(&trunk, layer);
        size_t mapped_nbytes = 0;
        const uint8_t *base = (const uint8_t *)salt_trunk_mapping(
            &trunk, &mapped_nbytes);
        const uint32_t *vals;
        const uint16_t *scales, *biases;
        if (!mapping_is_exact(&trunk, layer, payload, base, mapped_nbytes) ||
            salt_trunk_bind(&trunk, 1 - layer) != NULL) {
            fprintf(stderr, "bounded host lease failed at layer %d\n", layer);
            goto done;
        }
        if (salt_gpu_resident_trunk_map(base, mapped_nbytes) != 0) {
            fprintf(stderr, "Metal trunk registration failed at layer %d\n",
                    layer);
            goto done;
        }
        device_active = 1;
        vals = (const uint32_t *)(const void *)payload;
        scales = (const uint16_t *)(const void *)(payload + TEST_VAL_BYTES);
        biases = (const uint16_t *)(const void *)(
            payload + TEST_VAL_BYTES + TEST_META_BYTES);
        q4_reference(vals, scales, biases, input, cpu);
        memset(gpu, 0, sizeof gpu);
        {
            const uint32_t *value_jobs[1] = {vals};
            const uint16_t *scale_jobs[1] = {scales};
            const uint16_t *bias_jobs[1] = {biases};
            const float *input_jobs[1] = {input};
            float *output_jobs[1] = {gpu};
            const void *keys[1] = {&trunk_key};
            int rows[1] = {TEST_R};
            if (salt_gpu_q4_batch(value_jobs, scale_jobs, bias_jobs,
                                  input_jobs, output_jobs, keys, rows,
                                  TEST_C, 1) != 0 ||
                memcmp(cpu, gpu, sizeof cpu) != 0) {
                fprintf(stderr, "mapped trunk projection mismatch at layer %d\n",
                        layer);
                goto done;
            }
        }
        memset(gpu, 0, sizeof gpu);
        if (salt_gpu_q4_matvec(vals, scales, biases, TEST_R, TEST_C,
                               input, gpu) != 0 ||
            memcmp(cpu, gpu, sizeof cpu) != 0) {
            fprintf(stderr,
                    "mapped trunk singleton mismatch at layer %d\n", layer);
            goto done;
        }
        if (layer == 0 && mixed_indexed_test(
                vals, scales, biases, &trunk_key) != 0)
            goto done;
        {
            SaltBatchJob jobs[2];
            float fused_a[TEST_R], fused_b[TEST_R];
            memset(jobs, 0, sizeof jobs);
            jobs[0].vals = vals;
            jobs[0].scales = scales;
            jobs[0].biases = biases;
            jobs[0].R = TEST_R;
            jobs[0].C = TEST_C;
            jobs[0].B = 1;
            jobs[0].xs = input;
            jobs[0].ys = fused_a;
            jobs[1] = jobs[0];
            jobs[1].ys = fused_b;
            if (salt_q4_matvec2(vals, scales, biases, TEST_R,
                                vals, scales, biases, TEST_R, TEST_C,
                                input, fused_a, fused_b) == 0 ||
                salt_q4_multi_batch(jobs, 2) == 0) {
                fprintf(stderr,
                        "fused CPU helper bypassed mapped-only routing\n");
                goto done;
            }
        }
        if (salt_gpu_resident_trunk_unmap() != 0) {
            fprintf(stderr, "Metal trunk release failed at layer %d\n", layer);
            goto done;
        }
        device_active = 0;
        if (salt_trunk_unbind(&trunk) != 0) {
            fprintf(stderr, "host trunk munmap failed at layer %d\n", layer);
            goto done;
        }
        if (salt_trunk_mapping(&trunk, &mapped_nbytes) != NULL ||
            mapped_nbytes != 0) {
            fprintf(stderr, "host trunk release failed at layer %d\n", layer);
            goto done;
        }
    }
    salt_gpu_batch_stats_get(&after);
    if (after.trunk_batches != before.trunk_batches + 9 ||
        after.arena_batches != before.arena_batches) {
        fprintf(stderr, "trunk/arena resource accounting failed\n");
        goto done;
    }

    {
        const uint32_t *value_jobs[1] = {missing_vals};
        const uint16_t *scale_jobs[1] = {missing_scales};
        const uint16_t *bias_jobs[1] = {missing_biases};
        const float *input_jobs[1] = {input};
        float *output_jobs[1] = {gpu};
        const void *keys[1] = {&missing_key};
        int rows[1] = {TEST_R};
        if (salt_gpu_q4_batch(value_jobs, scale_jobs, bias_jobs,
                              input_jobs, output_jobs, keys, rows,
                              TEST_C, 1) == 0) {
            fprintf(stderr, "unregistered trunk reached GPU dispatch\n");
            goto done;
        }
    }
    salt_gpu_batch_stats_get(&after);
    if (after.mapped_only_misses != before.mapped_only_misses + 1 ||
        after.arena_batches != before.arena_batches) {
        fprintf(stderr, "mapped-only rejection accounting failed\n");
        goto done;
    }

    rc = 0;

done:
    if (device_active) {
        if (salt_gpu_resident_trunk_unmap() != 0) {
            fprintf(stderr, "Metal cleanup retained the trunk mapping\n");
            resource_retained = 1;
            rc = 1;
        } else {
            device_active = 0;
        }
    }
    if (trunk_open && !resource_retained) {
        if (salt_trunk_unbind(&trunk) != 0) {
            fprintf(stderr, "host trunk cleanup unbind failed\n");
            resource_retained = 1;
            rc = 1;
        }
    }
    if (trunk_open && !resource_retained) {
        if (salt_trunk_close(&trunk) != 0) {
            fprintf(stderr, "host trunk synchronized close failed\n");
            resource_retained = 1;
            rc = 1;
        }
    }
    salt_gpu_set_mapped_only(0);
    if (gpu_ready && !resource_retained) {
        if (salt_gpu_free() != 0) {
            fprintf(stderr, "Metal synchronized cleanup failed\n");
            resource_retained = 1;
            rc = 1;
        }
        gpu_ready = 0;
    }
    if (rc == 0) {
        if (salt_gpu_init() != 0) {
            fprintf(stderr, "Metal reinitialization after cleanup failed\n");
            rc = 1;
        } else {
            gpu_ready = 1;
            int cycle_rc = run_second_cycle(offsets_path, bin_path, input);
            if (cycle_rc != 0) {
                fprintf(stderr, "Metal second full lifecycle failed\n");
                if (cycle_rc == -2) resource_retained = 1;
                rc = 1;
            }
            if (!resource_retained && salt_gpu_free() != 0) {
                fprintf(stderr, "Metal second cleanup failed\n");
                rc = 1;
            }
            if (!resource_retained) gpu_ready = 0;
        }
    }
    if (rc == 0)
        printf("gpu-trunk mapped projection: PASS layers=2 R=%d C=%d "
               "batch+singleton=4 guarded-arena=0 warm-cache-rejected=1 "
               "cpu-expert-scope=1 direct-mixed=5 completion-matrix=4 "
               "second-lifecycle=1\n", TEST_R, TEST_C);
    if (!resource_retained) {
        unlink(offsets_path);
        unlink(bin_path);
    }
    return rc;
}
