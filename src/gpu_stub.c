#include "salt/gpu.h"
#include "salt/gpu_residency.h"

#include <string.h>

static int stub_defer = 0;
static int stub_mapped_only = 0;

const SaltGpuResidencyBackendOps *salt_gpu_residency_backend_ops(void) {
    return NULL;
}

int salt_gpu_startup_requirements(
    SaltGpuStartupRequirements *requirements, size_t requirements_size) {
    if (requirements && requirements_size == sizeof *requirements)
        memset(requirements, 0, sizeof *requirements);
    return -1;
}

int salt_gpu_init(void) { return -1; }
int salt_gpu_free(void) {
    stub_defer = 0;
    stub_mapped_only = 0;
    return 0;
}

int salt_gpu_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                         const uint16_t *biases, int R, int C,
                         const float *x, float *y) {
    (void)vals; (void)scales; (void)biases;
    (void)R; (void)C; (void)x; (void)y;
    return -1;
}

int salt_gpu_q8_matvec(const uint32_t *vals, const uint16_t *scales,
                       const uint16_t *biases, int R, int C,
                       const float *x, float *y) {
    (void)vals; (void)scales; (void)biases;
    (void)R; (void)C; (void)x; (void)y;
    return -1;
}

int salt_gpu_nvfp4_qdq(const float *input, int C, float input_global_scale,
                       uint8_t *packed, uint8_t *scales, float *dequant) {
    (void)input; (void)C; (void)input_global_scale;
    (void)packed; (void)scales; (void)dequant;
    return -1;
}

int salt_gpu_nvfp4_matvec(const void *key, int R, int C,
                          const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return -1;
}

int salt_gpu_nvfp4_matvec_batch(const void *key, int R, int C, int B,
                                const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return -1;
}

int salt_gpu_nvfp4_batch(const void *const *keys,
                         const float *const *inputs, float *const *outputs,
                         const int *batches, int R, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches;
    (void)R; (void)C; (void)njobs;
    return -1;
}

int salt_gpu_nvfp4_mixed_batch(const void *const *keys,
                               const float *const *inputs,
                               float *const *outputs, const int *batches,
                               const int *rows, int C, int njobs) {
    (void)keys; (void)inputs; (void)outputs; (void)batches; (void)rows;
    (void)C; (void)njobs;
    return -1;
}

int salt_gpu_bf16_resource_slot(uint32_t kind, uint32_t resource_id,
                                const void *key, const uint16_t *weights,
                                int R, int C) {
    (void)kind; (void)resource_id; (void)key; (void)weights; (void)R; (void)C;
    return -1;
}
int salt_gpu_bf16_matvec(const void *key, int R, int C,
                         const float *input, float *output) {
    (void)key; (void)R; (void)C; (void)input; (void)output;
    return -1;
}

int salt_gpu_bf16_matvec_batch(const void *key, int R, int C, int B,
                               const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return -1;
}

int salt_gpu_q4_batch(const uint32_t *const *vals,
                        const uint16_t *const *scales,
                        const uint16_t *const *biases,
                        const float *const *xs, float *const *ys,
                        const void *const *ids,
                        const int *Rj, int C, int njobs) {
    (void)vals; (void)scales; (void)biases; (void)xs; (void)ys;
    (void)ids; (void)Rj; (void)C; (void)njobs;
    return -1;
}

int salt_gpu_q8_batch(const uint32_t *const *vals,
                      const uint16_t *const *scales,
                      const uint16_t *const *biases,
                      const float *const *xs, float *const *ys,
                      const void *const *ids,
                      const int *Rj, int C, int njobs) {
    (void)vals; (void)scales; (void)biases; (void)xs; (void)ys;
    (void)ids; (void)Rj; (void)C; (void)njobs;
    return -1;
}

void salt_gpu_set_mapped_only(int on) { stub_mapped_only = on ? 1 : 0; }
int salt_gpu_mapped_only(void) { return stub_mapped_only; }

void salt_gpu_batch_stats_get(SaltGpuBatchStats *stats) {
    if (stats) memset(stats, 0, sizeof *stats);
}

int salt_gpu_batch_stats_get_v2(SaltGpuBatchStatsV2 *stats,
                                size_t stats_size) {
    if (!stats || stats_size != sizeof *stats) return -1;
    memset(stats, 0, sizeof *stats);
    return 0;
}

int salt_gpu_proj_batch(const uint32_t *vals, const uint16_t *scales,
                        const uint16_t *biases, int R, int C, int B,
                        const float *const *xs, float *const *ys,
                        const void *id) {
    (void)vals; (void)scales; (void)biases; (void)R; (void)C;
    (void)B; (void)xs; (void)ys; (void)id;
    return -1;
}

int salt_gpu_shared_buffer_alloc(SaltGpuSharedBuffer *buffer, size_t nbytes) {
    (void)nbytes;
    if (buffer) memset(buffer, 0, sizeof *buffer);
    return -1;
}

int salt_gpu_exact_cells(const SaltGpuExactCell *cells, int count) {
    (void)cells;
    (void)count;
    return -1;
}

int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer) {
    if (!buffer) return -1;
    memset(buffer, 0, sizeof *buffer);
    return 0;
}

int salt_gpu_attention_batch(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const SaltGpuSharedBuffer *queries, size_t query_float_offset,
    const SaltGpuSharedBuffer *kv, size_t key_float_offset,
    size_t value_float_offset, int start_position, int batch,
    int query_stride, int kv_stride, SaltGpuSharedBuffer *outputs,
    size_t output_float_offset) {
    (void)full_attention; (void)n_heads; (void)n_kv_heads;
    (void)head_dim; (void)window; (void)queries; (void)query_float_offset;
    (void)kv; (void)key_float_offset; (void)value_float_offset;
    (void)start_position; (void)batch; (void)query_stride; (void)kv_stride;
    (void)outputs; (void)output_float_offset;
    return -1;
}

int salt_gpu_attention_prepare(const SaltGpuSharedBuffer *kv) {
    (void)kv;
    return -1;
}

int salt_gpu_attention_bind_kv(size_t key_float_offset,
                               size_t value_float_offset, int kv_stride) {
    (void)key_float_offset; (void)value_float_offset; (void)kv_stride;
    return -1;
}
int salt_gpu_attention_materialize(SaltGpuSharedBuffer *kv, int position,
                                   uint64_t *copied_bytes) {
    (void)kv; (void)position; if (copied_bytes) *copied_bytes = 0; return -1;
}
int salt_gpu_attention_import(SaltGpuSharedBuffer *kv, int position,
                              uint64_t *copied_bytes) {
    (void)kv; (void)position; if (copied_bytes) *copied_bytes = 0; return -1;
}
int salt_gpu_attention_rewind(int position) {
    (void)position;
    return -1;
}

int salt_gpu_attention_transform(
    int n_heads, int n_kv_heads, int head_dim, int rope_dim,
    int start_position, int batch, int query_stride, int kv_stride, float eps,
    SaltGpuSharedBuffer *queries, size_t query_float_offset,
    SaltGpuSharedBuffer *keys, size_t key_float_offset,
    SaltGpuSharedBuffer *values, size_t value_float_offset,
    const float *q_weight, const float *k_weight,
    const float *cosines, const float *sines, int rope_pairs,
    SaltGpuSharedBuffer *kv, size_t key_cache_float_offset,
    size_t value_cache_float_offset) {
    (void)n_heads; (void)n_kv_heads; (void)head_dim; (void)rope_dim;
    (void)start_position; (void)batch; (void)query_stride; (void)kv_stride;
    (void)eps; (void)queries; (void)query_float_offset; (void)keys;
    (void)key_float_offset; (void)values; (void)value_float_offset;
    (void)q_weight; (void)k_weight; (void)cosines; (void)sines;
    (void)rope_pairs; (void)kv; (void)key_cache_float_offset;
    (void)value_cache_float_offset;
    return -1;
}

int salt_gpu_q4_selected_shared(
        const SaltGpuSelectedProjectionJob *jobs, int job_count,
        SaltGpuSharedBuffer *arena) {
    (void)jobs;
    (void)job_count;
    (void)arena;
    return -1;
}

int salt_gpu_q4_moe_chain(const SaltGpuMoeExpert *experts, int count,
                          int hidden, int routed,
                          const float *inputs, float *outputs,
                          float *gate_scratch, float *up_scratch,
                          size_t scratch_float_capacity) {
    (void)experts; (void)count; (void)hidden; (void)routed;
    (void)inputs; (void)outputs;
    (void)gate_scratch; (void)up_scratch; (void)scratch_float_capacity;
    return -1;
}

int salt_gpu_q4_moe_chain_selected(const SaltGpuMoeExpert *experts, int count,
                                   int hidden, int routed,
                                   const float *inputs, float *outputs,
                                   float *gate_scratch, float *up_scratch,
                                   size_t scratch_float_capacity) {
    return salt_gpu_q4_moe_chain(experts, count, hidden, routed,
        inputs, outputs, gate_scratch, up_scratch, scratch_float_capacity);
}

int salt_gpu_selected_resources_prepare(int capacity, int logical_capacity) {
    (void)capacity; (void)logical_capacity;
    return -1;
}

int salt_gpu_selected_resource_bind(int slot, uint64_t logical_resource_id,
                                    const void *base, size_t nbytes,
                                    const void *payload) {
    (void)slot; (void)logical_resource_id; (void)base; (void)nbytes;
    (void)payload;
    return -1;
}

int salt_gpu_selected_resource_bind_resident(
        int slot, uint64_t logical_resource_id,
        const void *payload, size_t nbytes,
        uintptr_t device_address, uint64_t generation) {
    (void)slot; (void)logical_resource_id; (void)payload; (void)nbytes;
    (void)device_address; (void)generation;
    return -1;
}

int salt_gpu_selected_resources_fence(void) { return -1; }

int salt_gpu_selected_resource_unbind(int slot,
                                      uint64_t logical_resource_id) {
    (void)slot; (void)logical_resource_id;
    return -1;
}

int salt_gpu_proj_batch_indexed(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id) {
    (void)vals; (void)scales; (void)biases; (void)R; (void)C; (void)B;
    (void)first_job; (void)job_count; (void)xs; (void)output;
    (void)y_float_offset; (void)id;
    return -1;
}

int salt_gpu_proj_batch_indexed_exact_views(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id) {
    (void)vals; (void)scales; (void)biases; (void)R; (void)C; (void)B;
    (void)first_job; (void)job_count; (void)xs; (void)output;
    (void)y_float_offset; (void)id;
    return -1;
}

int salt_gpu_resident_slot(const void *key, const uint32_t *vals,
                           const uint16_t *scales, const uint16_t *biases,
                           int R, int C) {
    (void)key; (void)vals; (void)scales; (void)biases;
    (void)R; (void)C;
    return -1;
}

int salt_gpu_expert_resource_bind(uint32_t resource_id,
                                  const void *base, size_t nbytes) {
    (void)resource_id; (void)base; (void)nbytes;
    return -1;
}

int salt_gpu_expert_resource_slot(uint32_t resource_id, const void *key,
                                  const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases, int R, int C) {
    (void)resource_id; (void)key; (void)vals; (void)scales; (void)biases;
    (void)R; (void)C;
    return -1;
}

int salt_gpu_expert_resource_unbind(uint32_t resource_id) {
    (void)resource_id;
    return -1;
}

int salt_gpu_weight_resource_describe(uint32_t kind, uint32_t resource_id,
                                      const void *base, size_t nbytes) {
    (void)kind; (void)resource_id; (void)base; (void)nbytes;
    return -1;
}

int salt_gpu_weight_resource_activate(uint32_t kind, uint32_t resource_id,
                                      SaltGpuWeightAddressability policy) {
    (void)kind; (void)resource_id; (void)policy;
    return -1;
}

int salt_gpu_weight_resource_pool_bind(uint32_t kind, uint32_t resource_id) {
    (void)kind; (void)resource_id;
    return -1;
}

int salt_gpu_weight_resource_deactivate(uint32_t kind, uint32_t resource_id) {
    (void)kind; (void)resource_id;
    return -1;
}

int salt_gpu_weight_resource_usage(SaltGpuWeightResourceUsage *usage,
                                   size_t usage_size) {
    if (!usage || usage_size != sizeof *usage) return -1;
    memset(usage, 0, sizeof *usage);
    return 0;
}

int salt_gpu_weight_resource_bind(uint32_t kind, uint32_t resource_id,
                                  const void *base, size_t nbytes) {
    (void)kind; (void)resource_id; (void)base; (void)nbytes;
    return -1;
}

int salt_gpu_weight_window_bind(uint32_t kind, uint32_t resource_id,
                                uint64_t offset, size_t nbytes) {
    (void)kind; (void)resource_id; (void)offset; (void)nbytes;
    return -1;
}

int salt_gpu_weight_window_unbind(uint32_t kind, uint32_t resource_id) {
    (void)kind; (void)resource_id;
    return -1;
}

int salt_gpu_weight_resource_slot(uint32_t kind, uint32_t resource_id,
                                  const void *key, const uint32_t *vals,
                                  const uint16_t *scales,
                                  const uint16_t *biases,
                                  int bits, int R, int C) {
    (void)kind; (void)resource_id; (void)key; (void)vals; (void)scales;
    (void)biases; (void)bits; (void)R; (void)C;
    return -1;
}

int salt_gpu_nvfp4_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    const float *weight_global_scale, const float *input_global_scale,
    int R, int C) {
    (void)kind; (void)resource_id; (void)key;
    (void)weight_packed; (void)weight_scales;
    (void)weight_global_scale; (void)input_global_scale;
    (void)R; (void)C;
    return -1;
}

int salt_gpu_weight_resource_unbind(uint32_t kind, uint32_t resource_id) {
    (void)kind; (void)resource_id;
    return -1;
}

int salt_gpu_resident_trunk_map(const void *base, size_t nbytes) {
    (void)base; (void)nbytes;
    return -1;
}

int salt_gpu_resident_trunk_unmap(void) { return 0; }

int salt_gpu_resident_pool_map(const void *base, size_t nbytes) {
    (void)base; (void)nbytes;
    return -1;
}

int salt_gpu_set_defer(int on) { stub_defer = on ? 1 : 0; return 0; }
int salt_gpu_defer(void) { return stub_defer; }
int salt_gpu_sync(void) { return 0; }

const struct SaltTextGpuProgramOps *salt_gpu_tensor_program_ops(void) {
    return NULL;
}

const SaltGpuDispatch *salt_dispatch_get(void) { return NULL; }

/* the CUDA-runtime probe (the non-Darwin stub): no CUDA here. */
int salt_gpu_cuda_present(void) { return 0; }
int salt_gpu_rocm_present(void) { return 0; }
int salt_gpu_pageable_mmap_active(void) { return 0; }
