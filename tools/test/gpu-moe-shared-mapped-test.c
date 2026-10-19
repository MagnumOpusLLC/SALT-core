#define _POSIX_C_SOURCE 200809L

#include "salt/gpu.h"
#include "salt/kernels.h"
#include "salt/moe.h"
#include "salt/salt.h"

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef PATH_MAX
#define PATH_MAX 1024
#endif

enum {
    H = 512,
    M = 512,
    B = 4,
    TOPK = 4,
    SOURCE_OFFSET = 100,
    N_SHARED_TENSORS = 12
};

static uint16_t f32_to_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    return (uint16_t)(bits >> 16);
}

static size_t align4(size_t value) { return (value + 3u) & ~(size_t)3u; }

static size_t add_trunk_q4(SaltTrunkLayout *tl, int vi, int si, int bi,
                           size_t off, int R, int C) {
    size_t elements = (size_t)R * (size_t)C;
    size_t values = elements / 2u;
    size_t groups = elements / 64u;
    off = align4(off);
    tl->t[vi].off = (long)off;
    tl->t[vi].dims[0] = R;
    tl->t[vi].dims[1] = C;
    tl->t[vi].rank = 2;
    tl->t[vi].bits = 4;
    off += values;
    tl->t[si].off = (long)off;
    tl->t[si].dims[0] = (long)groups;
    tl->t[si].rank = 1;
    off += groups * sizeof(uint16_t);
    tl->t[bi].off = (long)off;
    tl->t[bi].dims[0] = (long)groups;
    tl->t[bi].rank = 1;
    off += groups * sizeof(uint16_t);
    return off;
}

static void fill_trunk_q4(uint8_t *payload, const SaltTrunkLayout *tl,
                          int vi, int si, int bi, int R, int C) {
    size_t elements = (size_t)R * (size_t)C;
    size_t groups = elements / 64u;
    uint32_t *values = (uint32_t *)(void *)(payload + tl->t[vi].off);
    uint16_t *scales = (uint16_t *)(void *)(payload + tl->t[si].off);
    uint16_t *biases = (uint16_t *)(void *)(payload + tl->t[bi].off);
    for (size_t i = 0; i < elements / 8u; i++)
        values[i] = 0x11111111u;
    for (size_t i = 0; i < groups; i++) {
        scales[i] = f32_to_bf16(1.0f / 512.0f);
        biases[i] = f32_to_bf16(0.0f);
    }
}

static size_t add_expert_q4(SaltMoETensor *tensor, size_t off,
                            int R, int C) {
    size_t elements = (size_t)R * (size_t)C;
    size_t groups = elements / 64u;
    memset(tensor, 0, sizeof *tensor);
    tensor->dims[0] = R;
    tensor->dims[1] = C;
    tensor->rank = 2;
    tensor->rel_v = (long)align4(off);
    tensor->v_nbytes = (long)(elements / 2u);
    off = (size_t)tensor->rel_v + elements / 2u;
    tensor->rel_s = (long)off;
    tensor->s_nbytes = (long)(groups * sizeof(uint16_t));
    off += groups * sizeof(uint16_t);
    tensor->rel_b = (long)off;
    off += groups * sizeof(uint16_t);
    tensor->fmt = 1;
    tensor->bits = 4;
    tensor->bsize = 64;
    return off;
}

static int write_fixture(const char *offsets_path, const char *bin_path,
                         const uint8_t *payload, size_t payload_nbytes) {
    FILE *file = NULL;
    uint8_t zeros[SOURCE_OFFSET];
    uint64_t n_layers = 1;
    uint64_t pair[2] = {SOURCE_OFFSET, 0};
    memset(zeros, 0, sizeof zeros);
    pair[1] = (uint64_t)payload_nbytes;

    file = fopen(bin_path, "wb");
    if (!file) return -1;
    if (fwrite(zeros, 1, sizeof zeros, file) != sizeof zeros ||
        fwrite(payload, 1, payload_nbytes, file) != payload_nbytes ||
        fclose(file) != 0)
        return -1;

    file = fopen(offsets_path, "wb");
    if (!file) return -1;
    if (fwrite(&n_layers, sizeof n_layers, 1, file) != 1 ||
        fwrite(pair, sizeof pair, 1, file) != 1 ||
        fclose(file) != 0)
        return -1;
    return 0;
}

static int states_equal(const float a[B][H], const float b[B][H]) {
    return memcmp(a, b, (size_t)B * H * sizeof(float)) == 0;
}

int main(void) {
    SaltCfg cfg;
    SaltTrunkLayout tl;
    SaltTrunkTensor tensors[N_SHARED_TENSORS];
    int tensor_offsets[2] = {0, N_SHARED_TENSORS};
    SaltPoolLayout pl;
    SaltExpertLayout expert;
    SaltTrunk trunk;
    SaltGpuBatchStats before, after_sync, after_gpu;
    char offsets_path[PATH_MAX];
    char bin_path[PATH_MAX];
    uint8_t *payload = NULL;
    uint8_t *slot = NULL;
    const uint8_t *tr = NULL;
    const uint8_t *slots[B * TOPK];
    int selected[B * TOPK];
    float weights[B * TOPK];
    float xins[H * B];
    float cpu_state[B][H], gpu_state[B][H];
    float *cpu_states[B], *gpu_states[B];
    size_t payload_nbytes = 0, slot_nbytes = 0, mapped_nbytes = 0;
    const uint8_t *mapping = NULL;
    int gpu_ready = 0, trunk_open = 0, device_active = 0;
    int resource_retained = 0, rc = 1;
    int64_t n_matvec = 0, n_decode = 0;

    if (snprintf(offsets_path, sizeof offsets_path,
                 "/tmp/salt-shared-offsets.%ld", (long)getpid()) >=
            (int)sizeof offsets_path ||
        snprintf(bin_path, sizeof bin_path,
                 "/tmp/salt-shared-trunk.%ld", (long)getpid()) >=
            (int)sizeof bin_path)
        return 1;

    memset(&cfg, 0, sizeof cfg);
    cfg.hidden = H;
    cfg.latent = H;
    cfg.moe_inter = M;
    cfg.n_layers = 1;
    cfg.n_experts = 1;
    cfg.topk = TOPK;

    memset(&tl, 0, sizeof tl);
    memset(tensors, 0, sizeof tensors);
    tl.n_layers = 1;
    tl.t = tensors;
    tl.t_off = tensor_offsets;
    tl.hc_ffn_fn[0] = -1;
    tl.up[0] = -1;
    tl.ffn_norm[0] = -1;
    tl.se_g[0] = 0; tl.se_gs[0] = 1; tl.se_gb[0] = 2;
    tl.se_u[0] = 3; tl.se_us[0] = 4; tl.se_ub[0] = 5;
    tl.se_d[0] = 6; tl.se_ds[0] = 7; tl.se_db[0] = 8;
    tl.se_r[0] = 9; tl.se_rs[0] = 10; tl.se_rb[0] = 11;
    payload_nbytes = add_trunk_q4(&tl, 0, 1, 2, payload_nbytes, M, H);
    payload_nbytes = add_trunk_q4(&tl, 3, 4, 5, payload_nbytes, M, H);
    payload_nbytes = add_trunk_q4(&tl, 6, 7, 8, payload_nbytes, H, M);
    payload_nbytes = add_trunk_q4(&tl, 9, 10, 11, payload_nbytes, 1, H);
    payload = (uint8_t *)calloc(payload_nbytes, 1);
    if (!payload) goto done;
    fill_trunk_q4(payload, &tl, 0, 1, 2, M, H);
    fill_trunk_q4(payload, &tl, 3, 4, 5, M, H);
    fill_trunk_q4(payload, &tl, 6, 7, 8, H, M);
    fill_trunk_q4(payload, &tl, 9, 10, 11, 1, H);

    memset(&expert, 0, sizeof expert);
    expert.n = 3;
    expert.chain = 1;
    slot_nbytes = add_expert_q4(&expert.t[0], slot_nbytes, M, H);
    slot_nbytes = add_expert_q4(&expert.t[1], slot_nbytes, M, H);
    slot_nbytes = add_expert_q4(&expert.t[2], slot_nbytes, H, M);
    slot = (uint8_t *)calloc(slot_nbytes, 1);
    if (!slot) goto done;
    memset(&pl, 0, sizeof pl);
    pl.n_layers = 1;
    pl.n_experts = 1;
    pl.expert_nbytes = (int64_t)slot_nbytes;
    pl.max_rc = H * M;
    pl.exp = &expert;

    for (int b = 0; b < B; b++) {
        cpu_states[b] = cpu_state[b];
        gpu_states[b] = gpu_state[b];
        for (int i = 0; i < H; i++) {
            float value = (float)(1 + b + i % 7) / 64.0f;
            cpu_state[b][i] = value;
            gpu_state[b][i] = value;
            xins[(size_t)i * B + b] = value;
        }
        for (int j = 0; j < TOPK; j++) {
            size_t k = (size_t)b * TOPK + j;
            selected[k] = 0;
            weights[k] = 0.0f;
            slots[k] = slot;
        }
    }

    if (write_fixture(offsets_path, bin_path, payload, payload_nbytes) != 0)
        goto done;
    if (setenv("SALT_GPU", "1", 1) != 0 ||
        setenv("SALT_GPU_TRUNK", "1", 1) != 0 ||
        setenv("SALT_GPU_MOE", "0", 1) != 0 ||
        setenv("SALT_GPU_TRUNK_B", "1", 1) != 0 ||
        setenv("SALT_ATTN_THREADS", "1", 1) != 0)
        goto done;

    if (salt_gpu_init() != 0) {
        if (getenv("SALT_REQUIRE_GPU")) {
            fprintf(stderr, "shared-MoE mapped test requires Metal\n");
            goto done;
        }
        printf("gpu-moe shared mapped: SKIP (GPU unavailable)\n");
        rc = 0;
        goto done;
    }
    gpu_ready = 1;
    if (salt_trunk_open(&trunk, bin_path, offsets_path) != 0)
        goto done;
    trunk_open = 1;
    if (salt_trunk_start_mapped(&trunk) != 0)
        goto done;
    tr = salt_trunk_bind(&trunk, 0);
    mapping = (const uint8_t *)salt_trunk_mapping(&trunk, &mapped_nbytes);
    if (!tr || !mapping || mapped_nbytes !=
            (size_t)SOURCE_OFFSET + payload_nbytes ||
        (size_t)(tr - mapping) != SOURCE_OFFSET)
        goto done;
    salt_gpu_set_mapped_only(1);
    if (salt_gpu_resident_trunk_map(mapping, mapped_nbytes) != 0)
        goto done;
    device_active = 1;

    salt_gpu_batch_stats_get(&before);
    if (salt_gpu_set_defer(0) != 0)
        goto done;
    if (salt_moe_step_batch(&cfg, &tl, 0, tr, &pl, xins, B, TOPK,
                            selected, weights, slots, cpu_states,
                            &n_matvec, &n_decode) != 0) {
        fprintf(stderr, "shared-MoE synchronous reference failed\n");
        goto done;
    }
    salt_gpu_batch_stats_get(&after_sync);
    if (salt_gpu_defer() != 0 ||
        after_sync.trunk_batches < before.trunk_batches + 3 ||
        after_sync.arena_batches != before.arena_batches ||
        after_sync.expert_layer_batches != before.expert_layer_batches ||
        after_sync.mapped_only_misses != before.mapped_only_misses ||
        salt_kernels_cpu_expert_scope() != 0)
        goto done;

    n_matvec = n_decode = 0;
    if (salt_gpu_set_defer(1) != 0) {
        fprintf(stderr, "shared-MoE could not enable defer\n");
        goto done;
    }
    {
        int batch_rc = salt_moe_step_batch(
            &cfg, &tl, 0, tr, &pl, xins, B, TOPK,
            selected, weights, slots, gpu_states,
            &n_matvec, &n_decode);
        if (batch_rc != 0) {
            salt_gpu_batch_stats_get(&after_gpu);
            fprintf(stderr,
                    "shared-MoE mapped execution failed rc=%d "
                    "trunk=%llu arena=%llu expert=%llu misses=%llu "
                    "defer=%d scope=%d\n",
                    batch_rc,
                    (unsigned long long)(after_gpu.trunk_batches -
                                         after_sync.trunk_batches),
                    (unsigned long long)(after_gpu.arena_batches -
                                         after_sync.arena_batches),
                    (unsigned long long)(after_gpu.expert_layer_batches -
                                         after_sync.expert_layer_batches),
                    (unsigned long long)(after_gpu.mapped_only_misses -
                                         after_sync.mapped_only_misses),
                    salt_gpu_defer(), salt_kernels_cpu_expert_scope());
            goto done;
        }
    }
    if (salt_gpu_defer() != 1 || salt_gpu_sync() != 0) {
        fprintf(stderr, "shared-MoE defer policy/completion failed\n");
        goto done;
    }
    salt_gpu_batch_stats_get(&after_gpu);
    if (!states_equal(cpu_state, gpu_state) ||
        after_gpu.trunk_batches < after_sync.trunk_batches + 3 ||
        after_gpu.arena_batches != after_sync.arena_batches ||
        after_gpu.expert_layer_batches != after_sync.expert_layer_batches ||
        after_gpu.mapped_only_misses != after_sync.mapped_only_misses ||
        salt_kernels_cpu_expert_scope() != 0) {
        size_t ndiff = 0, first = 0;
        float max_diff = 0.0f;
        for (size_t i = 0; i < (size_t)B * H; i++) {
            float diff = fabsf(((const float *)cpu_state)[i] -
                               ((const float *)gpu_state)[i]);
            if (memcmp((const float *)cpu_state + i,
                       (const float *)gpu_state + i, sizeof(float)) != 0) {
                if (ndiff == 0) first = i;
                ndiff++;
            }
            if (diff > max_diff) max_diff = diff;
        }
        fprintf(stderr,
                "shared-MoE mismatch ndiff=%zu max|d|=%.9g first=%zu "
                "cpu=%.9g gpu=%.9g trunk=%llu arena=%llu expert=%llu "
                "misses=%llu\n",
                ndiff, max_diff, first,
                ((const float *)cpu_state)[first],
                ((const float *)gpu_state)[first],
                (unsigned long long)(after_gpu.trunk_batches -
                                     after_sync.trunk_batches),
                (unsigned long long)(after_gpu.arena_batches -
                                     after_sync.arena_batches),
                (unsigned long long)(after_gpu.expert_layer_batches -
                                     after_sync.expert_layer_batches),
                (unsigned long long)(after_gpu.mapped_only_misses -
                                     after_sync.mapped_only_misses));
        goto done;
    }

    if (salt_gpu_set_defer(0) != 0 ||
        salt_gpu_resident_trunk_unmap() != 0) {
        resource_retained = 1;
        goto done;
    }
    device_active = 0;
    salt_gpu_set_mapped_only(0);
    if (salt_trunk_unbind(&trunk) != 0 || salt_trunk_close(&trunk) != 0)
        goto done;
    trunk_open = 0;
    if (salt_gpu_free() != 0)
        goto done;
    gpu_ready = 0;
    printf("gpu-moe shared mapped: PASS trunk-batches=%llu defer-restored=1\n",
           (unsigned long long)(after_gpu.trunk_batches -
                                after_sync.trunk_batches));
    rc = 0;

done:
    salt_kernels_set_cpu_expert_scope(0);
    if (device_active) {
        if (salt_gpu_set_defer(0) != 0 ||
            salt_gpu_resident_trunk_unmap() != 0)
            resource_retained = 1;
        else
            device_active = 0;
    }
    salt_gpu_set_mapped_only(0);
    if (trunk_open && !resource_retained && salt_trunk_unbind(&trunk) != 0)
        resource_retained = 1;
    if (trunk_open && !resource_retained && salt_trunk_close(&trunk) != 0)
        resource_retained = 1;
    if (gpu_ready && !resource_retained && salt_gpu_free() != 0)
        resource_retained = 1;
    free(payload);
    free(slot);
    if (!resource_retained) {
        unlink(offsets_path);
        unlink(bin_path);
    }
    return resource_retained ? 1 : rc;
}
