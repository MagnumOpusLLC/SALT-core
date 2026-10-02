#include "salt/gpu.h"
#include "salt/gpu_residency.h"
#include "salt/gpu_resource.h"
#include "salt/text_verify.h"

#include <hip/hip_runtime.h>

#if defined(__HIP_PLATFORM_AMD__)
#define __shfl_sync(mask, value, source) __shfl((value), (source), 32)
#endif

#include <chrono>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HIP_MAX_RESOURCES 16
#define HIP_MAX_TENSORS 16384
#define HIP_MAX_COMPONENT_REGIONS (HIP_MAX_TENSORS * 4)
#define HIP_MAX_C 8192
#define HIP_MAX_OUTPUTS (4096u * 2816u)
#define HIP_MAX_BATCH_JOBS 4096
#define HIP_FORMAT_NVFP4 40
#define HIP_SHARED_SLOTS 8

typedef struct HipResource {
    int used;
    int active;
    int registered;
    uint32_t kind, id;
    SaltGpuWeightAddressability policy;
    const unsigned char *host;
    unsigned char *device;
    size_t nbytes;
    uint64_t window_offset;
    size_t window_nbytes;
    int component_pool;
} HipResource;

typedef struct HipComponentRegion {
    int resource_index;
    int registered;
    const unsigned char *host;
    unsigned char *device;
    size_t nbytes;
    uint64_t source_offset;
    uint64_t residency_offset;
    uint64_t populated_bytes;
    int resident;
} HipComponentRegion;

typedef struct HipSelectedResource {
    int active;
    int registered;
    int resident;
    uint64_t logical_resource_id;
    uint64_t generation;
    const unsigned char *host;
    const unsigned char *payload;
    unsigned char *device;
    size_t nbytes;
} HipSelectedResource;

typedef struct HipSharedSlot {
    void *contents;
    void *device;
    size_t nbytes;
} HipSharedSlot;

typedef struct HipResidencyBackendState {
    SaltGpuResidencyPlan plan;
    unsigned char *device_arena;
    unsigned char *staging_arena;
    hipStream_t stream;
    hipEvent_t *slot_events;
    hipEvent_t permanent_event;
    uint64_t *slot_tickets;
    uint64_t *staging_tickets;
    uint32_t *slot_staging;
    uint32_t event_count;
    uint64_t permanent_layout_bytes;
    uint64_t permanent_max_span_bytes;
    uint32_t permanent_span_count;
    int permanent_published;
    int ready;
    int poisoned;
} HipResidencyBackendState;

typedef struct HipTensor {
    int used, bits, rows, cols;
    const void *key;
    uint32_t kind, resource_id;
    uint64_t value_offset, scale_offset, bias_offset, aux_offset;
    unsigned char *resident_device;
    uint64_t resident_value_offset, resident_scale_offset;
    uint64_t resident_bias_offset, resident_aux_offset;
    int resident;
} HipTensor;

typedef struct HipBatchDesc {
    const unsigned char *resource;
    uint64_t value_offset, scale_offset, bias_offset, aux_offset;
    uint32_t input_offset, output_offset;
    int rows, cols, batch;
    int32_t resource_slot;
    uint32_t selected;
    uint64_t logical_resource_id;
} HipBatchDesc;

#define HIP_TEXT_EXPERT_PROJECTIONS 3u
#define HIP_TEXT_STATE_MAGIC UINT64_C(0x4355545854505247)

typedef struct HipAttentionKvSync HipAttentionKvSync;

typedef struct HipTextTensorRef {
    const unsigned char *resource;
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t encoding;
    uint32_t reserved;
} HipTextTensorRef;

typedef struct HipTextKvRef {
    float *keys;
    float *values;
    float *device_keys;
    float *device_values;
    float *host_keys;
    float *host_values;
    HipAttentionKvSync *sync;
    size_t key_float_offset;
    size_t value_float_offset;
    size_t tentative_key_float_offset;
    size_t tentative_value_float_offset;
    uint32_t width;
    uint32_t private_mode;
    uint32_t row_capacity;
} HipTextKvRef;

typedef struct HipTextLayerRefs {
    HipTextTensorRef q, k, v, o;
    HipTextTensorRef dense_gate, dense_up, dense_down, router;
    HipTextTensorRef pre_attention_norm, q_norm, k_norm;
    HipTextTensorRef post_attention_norm;
    HipTextTensorRef pre_ffn_norm_1, pre_ffn_norm_2;
    HipTextTensorRef post_ffn_norm_1, post_ffn_norm_2;
    HipTextTensorRef post_ffn_norm, router_scale, per_expert_scale;
    HipTextTensorRef layer_scalar;
    HipTextKvRef kv;
} HipTextLayerRefs;

typedef struct HipTextCellTemplate {
    const SaltTextExecutionCell *cell;
    uint32_t index;
    uint32_t kind;
    uint32_t unit;
    uint32_t layer;
    uint32_t logical_capacity;
    uint32_t dependency_epoch;
    uint32_t completion_epoch;
    uint32_t hidden;
    uint32_t dense;
    uint32_t routed;
    uint32_t experts;
    uint32_t topk;
    size_t destination_offset;
    size_t destination_stride;
    uint32_t flags;
} HipTextCellTemplate;

typedef struct HipTextProgramState {
    uint64_t magic;
    const SaltTextVerifyProgram *program;
    SaltGpuSharedBuffer canonical;
    size_t canonical_bytes;
    size_t status_offset;
    unsigned char *device_canonical;
    size_t device_canonical_bytes;
    hipStream_t stream;
    HipTextTensorRef *device_experts;
    HipTextKvRef *device_kv;
    HipBatchDesc *host_selected_desc;
    HipBatchDesc *device_selected_desc;
    HipTextTensorRef embedding, final_norm, output_head;
    HipTextLayerRefs *layers;
    HipTextTensorRef *expert_refs;
    HipTextCellTemplate *templates;
    hipEvent_t *cell_event_start;
    hipEvent_t *cell_event_end;
    uint32_t source_position_host;
    uint32_t parent_rows_host[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t depths_host[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t *device_source_position;
    uint64_t address_matrix_digest;
    uint32_t address_matrix_cells;
    uint32_t template_count;
    uint32_t layer_capacity;
    uint32_t experts_per_layer;
    uint32_t expert_ref_capacity;
    uint32_t template_capacity;
    uint32_t cell_event_capacity;
    int cell_timing_enabled;
    uint32_t selected_job_capacity;
    int ready;
} HipTextProgramState;

enum HipTextCommandPhase {
    HIP_TEXT_PHASE_IDLE = 0,
    HIP_TEXT_PHASE_PREFIX,
    HIP_TEXT_PHASE_RESOURCE_PENDING,
    HIP_TEXT_PHASE_EXPERT,
    HIP_TEXT_PHASE_SUFFIX,
    HIP_TEXT_PHASE_SUBMITTED,
    HIP_TEXT_PHASE_FINISHED,
};

typedef struct HipTextProgramCommand {
    uint64_t generation;
    uint32_t source_position;
    uint32_t input_count;
    uint32_t maximum_depth;
    uint32_t attention_score_rows;
    uint32_t next_cell;
    uint32_t encoded_cells;
    uint32_t barrier_count;
    uint32_t highest_completion;
    uint32_t ffn_norm_wave_layer;
    uint32_t active_resource_layer;
    uint32_t resource_request_count;
    uint32_t selected_descriptor_count;
    uint32_t selected_max_occupancy;
    uint32_t qk_wave_pending;
    uint32_t dense_wave_pending;
    uint32_t phase;
    uint32_t frontier;
    uint32_t authoritative;
    uint32_t authoritative_output_rows;
    int32_t resource_experts[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    int32_t resource_slots[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t expert_occupancies[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    SaltTextTouchedSpan touched_spans[SALT_TEXT_MAX_TOUCHED_SPANS];
    uint32_t touched_span_count;
    uint64_t touched_span_bytes;
    uint64_t resource_wait_started_ns;
    uint64_t resource_wait_ns;
    uint64_t prefix_sync_ns;
    uint64_t expert_sync_ns;
    uint64_t final_sync_ns;
    uint32_t resource_wait_calls;
    uint32_t begun;
    uint32_t submitted;
    uint32_t finished;
    SaltTextVerifyBackendStats stats;
} HipTextProgramCommand;

static int hip_ready, hip_mapped_only, hip_defer;
static int hip_pageable_mmap;
static int hip_nvfp4_telemetry_enabled = 1;
static int hip_nvfp4_timing_enabled;
static hipEvent_t hip_nvfp4_event_start, hip_nvfp4_event_qdq;
static hipEvent_t hip_nvfp4_event_projection, hip_nvfp4_event_d2h;
static hipEvent_t hip_moe_event_start, hip_moe_event_gate_up;
static hipEvent_t hip_moe_event_activation, hip_moe_event_down;
static int hip_moe_timing_enabled;
static HipResource hip_resources[HIP_MAX_RESOURCES];
static HipComponentRegion hip_component_regions[HIP_MAX_COMPONENT_REGIONS];
static int hip_component_region_count;
static int hip_component_pool_finalized;
static uint64_t hip_component_pool_bytes;
static uint64_t hip_component_pool_peak_bytes;
static int hip_component_test_failure_fired;
static int hip_component_test_unreg_failure_fired;
static HipSelectedResource *hip_selected_resources;
static unsigned char **hip_selected_resource_slots;
static int32_t *hip_selected_logical_slots;
static uint32_t *hip_selected_payload_offsets;
static unsigned char **hip_selected_device_resources;
static int32_t *hip_selected_device_logical_slots;
static uint32_t *hip_selected_device_payload_offsets;
static int hip_selected_capacity, hip_selected_logical_capacity;
static HipSharedSlot hip_shared_slots[HIP_SHARED_SLOTS];
static HipResidencyBackendState hip_residency;
static HipTensor hip_tensors[HIP_MAX_TENSORS];
static int hip_tensor_count;
static float *hip_x, *hip_y;
static float *hip_attention_kv;
static size_t hip_attention_kv_nbytes;
static const void *hip_attention_kv_host;
static int *hip_attention_status;
struct HipAttentionKvSync {
    size_t key_offset, value_offset;
    size_t row_capacity;
    int kv_stride;
    int device_positions;
    int host_positions;
};
static HipAttentionKvSync hip_attention_kv_sync[64];
static int hip_attention_kv_sync_count;
typedef struct HipAttentionStage {
    const void *query_host;
    size_t query_offset, key_offset, value_offset;
    int start, batch, query_stride, kv_stride;
    int deferred_publish;
    int valid;
} HipAttentionStage;
static HipAttentionStage hip_attention_stage;
static float *hip_nvfp4_xdq;
static uint8_t *hip_nvfp4_packed, *hip_nvfp4_scales;
static float *hip_host_x, *hip_host_y;
static HipBatchDesc *hip_host_desc, *hip_device_desc;
static uint32_t hip_job_output_offset[HIP_MAX_BATCH_JOBS];
static int hip_job_rows[HIP_MAX_BATCH_JOBS];
static SaltGpuBatchStatsV2 hip_stats;
static uint64_t hip_q4_hetero_logical_input_bytes;
static uint64_t hip_q4_hetero_transfer_input_bytes;
static uint64_t hip_q4_hetero_output_bytes;
static uint64_t hip_q4_hetero_descriptor_bytes;
static uint64_t hip_q4_hetero_reused_inputs;
static uint64_t hip_q4_hetero_kernel_launches;
static uint64_t hip_q4_hetero_completion_fences;
static uint64_t hip_q4_hetero_h2d_ns;
static uint64_t hip_q4_hetero_launch_ns;
static uint64_t hip_q4_hetero_completion_ns;
static uint64_t hip_q4_hetero_scatter_ns;
static uint64_t hip_selected_bind_calls;
static uint64_t hip_selected_bind_reuses;
static uint64_t hip_selected_register_ns;
static uint64_t hip_selected_pointer_ns;
static uint64_t hip_selected_unbind_calls;
static uint64_t hip_selected_unbind_sync_calls;
static uint64_t hip_selected_unbind_sync_skips;
static uint64_t hip_selected_unbind_sync_ns;
static uint64_t hip_selected_unregister_ns;
static uint64_t hip_selected_read_epoch;
static uint64_t hip_selected_complete_epoch;
static uint64_t hip_selected_shared_calls;
static uint64_t hip_selected_shared_jobs;
static uint64_t hip_selected_shared_h2d_ns;
static uint64_t hip_selected_shared_completion_ns;
static uint64_t hip_selected_gate_up_gpu_ns;
static uint64_t hip_selected_activation_gpu_ns;
static uint64_t hip_selected_down_gpu_ns;
static uint64_t hip_consumer_sync_calls;
static uint64_t hip_consumer_sync_ns;
static uint64_t hip_indexed_calls;
static uint64_t hip_indexed_jobs;
static uint64_t hip_indexed_h2d_ns;
static uint64_t hip_indexed_launch_ns;
static uint64_t hip_peak_window_bytes;

enum HipPathId {
    HIP_PATH_NONE = 0,
    HIP_PATH_Q4_SINGLE_EXACT,
    HIP_PATH_Q8_SINGLE_EXACT,
    HIP_PATH_Q4_BATCH_WARP,
    HIP_PATH_Q4_BATCH_EXACT,
    HIP_PATH_Q8_BATCH_TILE,
    HIP_PATH_Q8_BATCH_EXACT,
    HIP_PATH_BF16_WARP_EXACT,
    HIP_PATH_NVFP4_QDQ,
    HIP_PATH_NVFP4_PROJECT,
    HIP_PATH_NVFP4_HET_QDQ,
    HIP_PATH_NVFP4_HET_PROJECT,
    HIP_PATH_Q4_HET_WARP,
    HIP_PATH_Q4_HET_WARP_BATCH,
    HIP_PATH_Q4_HET_TOKEN,
    HIP_PATH_ATTN_TRANSFORM,
    HIP_PATH_ATTN_BODY,
    HIP_PATH_MOE_GATE_UP_WARP_BATCH,
    HIP_PATH_MOE_GATE_UP_WARP_EXACT,
    HIP_PATH_MOE_ACTIVATION,
    HIP_PATH_MOE_DOWN_WARP_BATCH,
    HIP_PATH_MOE_DOWN_WARP_EXACT,
    HIP_PATH_MOE_HET_GATE_UP,
    HIP_PATH_MOE_HET_ACTIVATION,
    HIP_PATH_MOE_HET_DOWN,
    HIP_PATH_MOE_SELECTED_GATE_UP_WARP,
    HIP_PATH_MOE_SELECTED_GATE_UP_TOKEN,
    HIP_PATH_MOE_SELECTED_ACTIVATION,
    HIP_PATH_MOE_SELECTED_DOWN_WARP,
    HIP_PATH_MOE_SELECTED_DOWN_TOKEN,
    HIP_PATH_SELECTED_SHARED_SINGLE,
    HIP_PATH_SELECTED_SHARED_WAVE,
    HIP_PATH_INDEXED_Q4_BATCH,
    HIP_PATH_INDEXED_Q4_EXACT,
    HIP_PATH_TEXT_Q4,
    HIP_PATH_TEXT_Q8,
    HIP_PATH_TEXT_BF16,
    HIP_PATH_TEXT_EMBEDDING,
    HIP_PATH_TEXT_NORM,
    HIP_PATH_TEXT_ATTN_TRANSFORM,
    HIP_PATH_TEXT_ATTN_BODY,
    HIP_PATH_TEXT_COMBINE,
    HIP_PATH_TEXT_ACTIVATION,
    HIP_PATH_TEXT_ROUTER_INPUT,
    HIP_PATH_TEXT_TOPK,
    HIP_PATH_TEXT_GROUP_MAPS,
    HIP_PATH_TEXT_ROUTED_GATHER,
    HIP_PATH_TEXT_EXPERT_Q4,
    HIP_PATH_TEXT_EXPERT_REDUCE,
    HIP_PATH_TEXT_SOFTCAP,
    HIP_PATH_TEXT_SYSTEM_FENCE,
    HIP_PATH_TEXT_SUBMIT,
    HIP_PATH_TEXT_SYNC,
    HIP_PATH_CONSUMER_SYNC,
    HIP_PATH_COUNT,
};

typedef struct HipPathCounter {
    uint64_t calls;
    uint64_t logical_jobs;
    uint64_t logical_rows;
    uint64_t physical_launches;
    uint64_t consumers;
} HipPathCounter;

static const char *const hip_path_names[HIP_PATH_COUNT] = {
    "none",
    "q4_single_exact",
    "q8_single_exact",
    "q4_batch_warp",
    "q4_batch_exact",
    "q8_batch_tile",
    "q8_batch_exact",
    "bf16_warp_exact",
    "nvfp4_qdq",
    "nvfp4_project",
    "nvfp4_heterogeneous_qdq",
    "nvfp4_heterogeneous_project",
    "q4_heterogeneous_warp",
    "q4_heterogeneous_warp_batch",
    "q4_heterogeneous_token",
    "attention_transform",
    "attention_body",
    "moe_gate_up_warp_batch",
    "moe_gate_up_warp_exact",
    "moe_activation",
    "moe_down_warp_batch",
    "moe_down_warp_exact",
    "moe_heterogeneous_gate_up",
    "moe_heterogeneous_activation",
    "moe_heterogeneous_down",
    "moe_selected_gate_up_warp",
    "moe_selected_gate_up_token",
    "moe_selected_activation",
    "moe_selected_down_warp",
    "moe_selected_down_token",
    "selected_shared_single",
    "selected_shared_wave",
    "indexed_q4_batch",
    "indexed_q4_exact",
    "text_q4",
    "text_q8",
    "text_bf16",
    "text_embedding",
    "text_norm",
    "text_attention_transform",
    "text_attention_body",
    "text_combine",
    "text_activation",
    "text_router_input",
    "text_topk",
    "text_group_maps",
    "text_routed_gather",
    "text_expert_q4",
    "text_expert_reduce",
    "text_softcap",
    "text_system_fence",
    "text_submit",
    "text_sync",
    "consumer_sync",
};

static HipPathCounter hip_path_counters[HIP_PATH_COUNT];
static int hip_path_enabled;
static uint64_t hip_selected_legacy_refusals;
static uint64_t hip_api_failures;
static int hip_text_selected_wave_enabled;
static int hip_text_nrow_enabled;
static int hip_text_pair_enabled;
static int hip_text_system_fence_enabled;
static uint32_t hip_text_wave_groups;
static uint32_t hip_text_bf16_wave_groups;
static uint32_t hip_text_nrow_min_b;

static void hip_path_record(enum HipPathId path, uint64_t logical_jobs,
        uint64_t logical_rows, uint64_t physical_launches,
        uint64_t consumers) {
    HipPathCounter *counter;
    if (!hip_path_enabled || path <= HIP_PATH_NONE || path >= HIP_PATH_COUNT)
        return;
    counter = &hip_path_counters[path];
    __atomic_fetch_add(&counter->calls, UINT64_C(1), __ATOMIC_RELAXED);
    __atomic_fetch_add(&counter->logical_jobs, logical_jobs, __ATOMIC_RELAXED);
    __atomic_fetch_add(&counter->logical_rows, logical_rows, __ATOMIC_RELAXED);
    __atomic_fetch_add(&counter->physical_launches, physical_launches,
        __ATOMIC_RELAXED);
    __atomic_fetch_add(&counter->consumers, consumers, __ATOMIC_RELAXED);
}

static void hip_path_publish(void) {
    HipPathCounter total = {0, 0, 0, 0, 0};
    if (!hip_path_enabled) return;
    for (int path = HIP_PATH_NONE + 1; path < HIP_PATH_COUNT; path++) {
        HipPathCounter value;
        value.calls = __atomic_load_n(&hip_path_counters[path].calls,
            __ATOMIC_RELAXED);
        value.logical_jobs = __atomic_load_n(
            &hip_path_counters[path].logical_jobs, __ATOMIC_RELAXED);
        value.logical_rows = __atomic_load_n(
            &hip_path_counters[path].logical_rows, __ATOMIC_RELAXED);
        value.physical_launches = __atomic_load_n(
            &hip_path_counters[path].physical_launches, __ATOMIC_RELAXED);
        value.consumers = __atomic_load_n(&hip_path_counters[path].consumers,
            __ATOMIC_RELAXED);
        if (value.calls == 0 && value.physical_launches == 0) continue;
        fprintf(stderr,
            "SALT_HIP_PATH id=%d name=%s calls=%llu logical_jobs=%llu "
            "logical_rows=%llu launches=%llu consumers=%llu\n",
            path, hip_path_names[path],
            (unsigned long long)value.calls,
            (unsigned long long)value.logical_jobs,
            (unsigned long long)value.logical_rows,
            (unsigned long long)value.physical_launches,
            (unsigned long long)value.consumers);
        total.calls += value.calls;
        total.logical_jobs += value.logical_jobs;
        total.logical_rows += value.logical_rows;
        total.physical_launches += value.physical_launches;
        total.consumers += value.consumers;
    }
    fprintf(stderr,
        "SALT_HIP_PATH_TOTAL calls=%llu logical_jobs=%llu logical_rows=%llu "
        "launches=%llu consumers=%llu\n",
        (unsigned long long)total.calls,
        (unsigned long long)total.logical_jobs,
        (unsigned long long)total.logical_rows,
        (unsigned long long)total.physical_launches,
        (unsigned long long)total.consumers);
}

typedef struct HipNvfp4Telemetry {
    uint64_t packed_weight_bytes;
    uint64_t weight_scale_bytes;
    uint64_t activation_input_bytes;
    uint64_t activation_qdq_bytes;
    uint64_t projection_output_bytes;
    uint64_t h2d_ns;
    uint64_t qdq_ns;
    uint64_t projection_ns;
    uint64_t d2h_ns;
} HipNvfp4Telemetry;

static int hip_success(hipError_t error, const char *where);
static int hip_residency_permanent_layout(
    void *context, SaltGpuResidencyPermanentLayout *layout,
    size_t layout_size);
static int hip_residency_permanent_span(
    void *context, uint32_t span_index,
    SaltGpuResidencyPermanentSpan *span, size_t span_size);
static int hip_residency_permanent_copy(
    void *context, uint32_t span_index, uint64_t device_offset,
    uint32_t staging_slot, size_t nbytes, uint64_t ticket);
static int hip_residency_permanent_publish(void *context);

static uint64_t hip_host_now_ns(void) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int hip_event_elapsed_ns(hipEvent_t start, hipEvent_t end,
                                 uint64_t *result) {
    float elapsed_ms = 0.0f;
    if (!result || hip_success(hipEventElapsedTime(&elapsed_ms, start, end),
                                "hipEventElapsedTime NVFP4") != 0 ||
        !(elapsed_ms >= 0.0f) || elapsed_ms > (float)UINT64_MAX / 1000000.0f)
        return -1;
    *result = (uint64_t)((double)elapsed_ms * 1000000.0 + 0.5);
    return 0;
}

static int hip_nvfp4_d2h_timing(hipEvent_t producer,
                                 HipNvfp4Telemetry *telemetry) {
    if (!hip_nvfp4_timing_enabled) return 0;
    if (!telemetry ||
        hip_success(hipEventRecord(hip_nvfp4_event_d2h),
                     "hipEventRecord NVFP4 D2H") != 0 ||
        hip_success(hipEventSynchronize(hip_nvfp4_event_d2h),
                     "hipEventSynchronize NVFP4 D2H") != 0)
        return -1;
    return hip_event_elapsed_ns(producer, hip_nvfp4_event_d2h,
                                 &telemetry->d2h_ns);
}

static int hip_u64_mul(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || (a != 0 && b > UINT64_MAX / a)) return -1;
    *result = a * b;
    return 0;
}

static int hip_u64_add(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || b > UINT64_MAX - a) return -1;
    *result = a + b;
    return 0;
}

static void hip_u64_add_saturating(uint64_t *counter, uint64_t delta) {
    if (!counter) return;
    *counter = delta > UINT64_MAX - *counter ? UINT64_MAX : *counter + delta;
}

static void hip_selected_read_submitted(void) {
    hip_u64_add_saturating(&hip_selected_read_epoch, UINT64_C(1));
}

static void hip_selected_reads_completed(void) {
    hip_selected_complete_epoch = hip_selected_read_epoch;
}

static void hip_selected_epoch_state_reset(void) {
    hip_selected_unbind_sync_calls = 0;
    hip_selected_unbind_sync_skips = 0;
    hip_selected_read_epoch = 0;
    hip_selected_complete_epoch = 0;
}

static int hip_nvfp4_telemetry(int R, int C, int B,
                                HipNvfp4Telemetry *telemetry) {
    uint64_t rows = (uint64_t)(unsigned int)R;
    uint64_t cols = (uint64_t)(unsigned int)C;
    uint64_t batch = (uint64_t)(unsigned int)B;
    uint64_t rc, bc, br, qdq_per_row, value;
    if (!telemetry) return -1;
    memset(telemetry, 0, sizeof *telemetry);
    if (R < 1 || C < 16 || (C & 15) != 0 || B < 1 ||
        hip_u64_mul(rows, cols, &rc) != 0 ||
        hip_u64_mul(batch, cols, &bc) != 0 ||
        hip_u64_mul(batch, rows, &br) != 0 ||
        hip_u64_mul(batch, rc / UINT64_C(2),
                     &telemetry->packed_weight_bytes) != 0 ||
        hip_u64_mul(batch, rc / UINT64_C(16),
                     &telemetry->weight_scale_bytes) != 0 ||
        hip_u64_mul(bc, (uint64_t)sizeof(float),
                     &telemetry->activation_input_bytes) != 0 ||
        hip_u64_add(cols / UINT64_C(2), cols / UINT64_C(16),
                     &qdq_per_row) != 0 ||
        hip_u64_mul(cols, (uint64_t)sizeof(float), &value) != 0 ||
        hip_u64_add(qdq_per_row, value, &qdq_per_row) != 0 ||
        hip_u64_mul(batch, qdq_per_row,
                     &telemetry->activation_qdq_bytes) != 0 ||
        hip_u64_mul(br, (uint64_t)sizeof(float),
                     &telemetry->projection_output_bytes) != 0)
        return -1;
    return 0;
}

static int hip_nvfp4_telemetry_add(HipNvfp4Telemetry *total,
                                    const HipNvfp4Telemetry *delta) {
    HipNvfp4Telemetry next;
    if (!total || !delta) return -1;
    next = *total;
    if (hip_u64_add(total->packed_weight_bytes, delta->packed_weight_bytes,
                     &next.packed_weight_bytes) != 0 ||
        hip_u64_add(total->weight_scale_bytes, delta->weight_scale_bytes,
                     &next.weight_scale_bytes) != 0 ||
        hip_u64_add(total->activation_input_bytes,
                     delta->activation_input_bytes,
                     &next.activation_input_bytes) != 0 ||
        hip_u64_add(total->activation_qdq_bytes,
                     delta->activation_qdq_bytes,
                     &next.activation_qdq_bytes) != 0 ||
        hip_u64_add(total->projection_output_bytes,
                     delta->projection_output_bytes,
                     &next.projection_output_bytes) != 0)
        return -1;
    *total = next;
    return 0;
}

static void hip_nvfp4_telemetry_publish(
        const HipNvfp4Telemetry *telemetry, uint64_t launches) {
    if (!telemetry || !hip_nvfp4_telemetry_enabled) return;
    hip_u64_add_saturating(&hip_stats.nvfp4_logical_packed_weight_bytes,
                            telemetry->packed_weight_bytes);
    hip_u64_add_saturating(&hip_stats.nvfp4_logical_weight_scale_bytes,
                            telemetry->weight_scale_bytes);
    hip_u64_add_saturating(&hip_stats.nvfp4_activation_input_bytes,
                            telemetry->activation_input_bytes);
    hip_u64_add_saturating(&hip_stats.nvfp4_activation_qdq_bytes,
                            telemetry->activation_qdq_bytes);
    hip_u64_add_saturating(&hip_stats.nvfp4_projection_output_bytes,
                            telemetry->projection_output_bytes);
    hip_u64_add_saturating(&hip_stats.nvfp4_kernel_launches, launches);
    hip_u64_add_saturating(&hip_stats.nvfp4_completion_fences, UINT64_C(1));
    hip_u64_add_saturating(&hip_stats.nvfp4_h2d_ns, telemetry->h2d_ns);
    hip_u64_add_saturating(&hip_stats.nvfp4_qdq_ns, telemetry->qdq_ns);
    hip_u64_add_saturating(&hip_stats.nvfp4_projection_ns,
                            telemetry->projection_ns);
    hip_u64_add_saturating(&hip_stats.nvfp4_d2h_ns, telemetry->d2h_ns);
}

static int hip_success(hipError_t error, const char *where) {
    if (error == hipSuccess) return 0;
    __atomic_fetch_add(&hip_api_failures, UINT64_C(1), __ATOMIC_RELAXED);
    fprintf(stderr, "gpu-hip: %s: %s\n", where, hipGetErrorString(error));
    return -1;
}

static int hip_residency_release_storage(void) {
    int failed = 0;
    if (hip_residency.permanent_published) {
        for (int index = 0; index < hip_tensor_count; index++) {
            hip_tensors[index].resident_device = NULL;
            hip_tensors[index].resident_value_offset = 0;
            hip_tensors[index].resident_scale_offset = 0;
            hip_tensors[index].resident_bias_offset = 0;
            hip_tensors[index].resident_aux_offset = 0;
            hip_tensors[index].resident = 0;
        }
        for (int index = 0; index < hip_component_region_count; index++) {
            hip_component_regions[index].device = NULL;
            hip_component_regions[index].residency_offset = 0;
            hip_component_regions[index].populated_bytes = 0;
            hip_component_regions[index].resident = 0;
        }
        hip_component_pool_bytes = 0;
        hip_component_pool_finalized = 0;
    }
    if (hip_residency.stream &&
        hip_success(hipStreamSynchronize(hip_residency.stream),
            "hipStreamSynchronize residency destroy") != 0)
        failed = 1;
    if (hip_residency.permanent_event &&
        hip_success(hipEventDestroy(hip_residency.permanent_event),
            "hipEventDestroy permanent residency") != 0)
        failed = 1;
    for (uint32_t index = hip_residency.event_count; index > 0; index--)
        if (hip_residency.slot_events[index - 1u] &&
            hip_success(hipEventDestroy(
                hip_residency.slot_events[index - 1u]),
                "hipEventDestroy residency") != 0)
            failed = 1;
    if (hip_residency.stream &&
        hip_success(hipStreamDestroy(hip_residency.stream),
            "hipStreamDestroy residency") != 0)
        failed = 1;
    if (hip_residency.staging_arena &&
        hip_success(hipHostFree(hip_residency.staging_arena),
            "hipHostFree residency staging") != 0)
        failed = 1;
    if (hip_residency.device_arena &&
        hip_success(hipFree(hip_residency.device_arena),
            "hipFree residency arena") != 0)
        failed = 1;
    free(hip_residency.slot_staging);
    free(hip_residency.staging_tickets);
    free(hip_residency.slot_tickets);
    free(hip_residency.slot_events);
    memset(&hip_residency, 0, sizeof hip_residency);
    return failed ? -1 : 0;
}

static int hip_residency_create(
        void *context, const SaltGpuResidencyPlan *plan) {
    uint64_t staging_total;
    size_t device_bytes, staging_bytes;
    (void)context;
    if (!hip_ready || hip_residency.ready || hip_residency.device_arena ||
        !plan || salt_gpu_residency_plan_validate(plan) != 0 ||
        plan->device_budget_bytes > SIZE_MAX ||
        plan->staging_bytes > UINT64_MAX / plan->staging_count)
        return -1;
    staging_total = plan->staging_bytes * plan->staging_count;
    if (staging_total > SIZE_MAX ||
        plan->slot_count > SIZE_MAX / sizeof(hipEvent_t) ||
        plan->slot_count > SIZE_MAX / sizeof(uint64_t) ||
        plan->slot_count > SIZE_MAX / sizeof(uint32_t) ||
        plan->staging_count > SIZE_MAX / sizeof(uint64_t))
        return -1;
    device_bytes = (size_t)plan->device_budget_bytes;
    staging_bytes = (size_t)staging_total;
    hip_residency.plan = *plan;
    hip_residency.slot_events = (hipEvent_t *)calloc(
        plan->slot_count, sizeof *hip_residency.slot_events);
    hip_residency.slot_tickets = (uint64_t *)calloc(
        plan->slot_count, sizeof *hip_residency.slot_tickets);
    hip_residency.staging_tickets = (uint64_t *)calloc(
        plan->staging_count, sizeof *hip_residency.staging_tickets);
    hip_residency.slot_staging = (uint32_t *)calloc(
        plan->slot_count, sizeof *hip_residency.slot_staging);
    if (!hip_residency.slot_events || !hip_residency.slot_tickets ||
        !hip_residency.staging_tickets || !hip_residency.slot_staging)
        goto fail;
    for (uint32_t slot = 0; slot < plan->slot_count; slot++)
        hip_residency.slot_staging[slot] = UINT32_MAX;
    if (hip_success(hipMalloc((void **)&hip_residency.device_arena,
            device_bytes), "hipMalloc residency arena") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_residency.staging_arena,
            staging_bytes, hipHostMallocDefault),
            "hipHostMalloc residency staging") != 0 ||
        hip_success(hipStreamCreateWithFlags(&hip_residency.stream,
            hipStreamNonBlocking), "hipStreamCreate residency") != 0 ||
        hip_success(hipEventCreateWithFlags(&hip_residency.permanent_event,
            hipEventDisableTiming), "hipEventCreate permanent residency") != 0)
        goto fail;
    for (uint32_t slot = 0; slot < plan->slot_count; slot++) {
        if (hip_success(hipEventCreateWithFlags(
                &hip_residency.slot_events[slot], hipEventDisableTiming),
                "hipEventCreate residency") != 0)
            goto fail;
        hip_residency.event_count++;
    }
    hip_residency.ready = 1;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_RESIDENCY_CREATE device_bytes=%llu "
            "permanent_bytes=%llu slot_base=%llu slot_bytes=%llu "
            "slots=%u staging_bytes=%llu staging_slots=%u\n",
            (unsigned long long)plan->device_budget_bytes,
            (unsigned long long)plan->permanent_bytes,
            (unsigned long long)plan->slot_base_offset,
            (unsigned long long)plan->slot_bytes, plan->slot_count,
            (unsigned long long)staging_total, plan->staging_count);
    return 0;
fail:
    hip_residency_release_storage();
    return -1;
}

static int hip_residency_staging(
        void *context, uint32_t staging_slot,
        void **address, size_t *capacity) {
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned ||
        !address || !capacity ||
        staging_slot >= hip_residency.plan.staging_count ||
        hip_residency.plan.staging_bytes > SIZE_MAX)
        return -1;
    *address = hip_residency.staging_arena +
        (size_t)staging_slot * (size_t)hip_residency.plan.staging_bytes;
    *capacity = (size_t)hip_residency.plan.staging_bytes;
    return 0;
}

static int hip_residency_populate(
        void *context, uint32_t physical_slot, uint64_t device_offset,
        uint32_t staging_slot, size_t nbytes, uint64_t ticket) {
    uint64_t expected_offset;
    hipError_t error;
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned ||
        ticket == 0 || nbytes == 0 ||
        physical_slot >= hip_residency.plan.slot_count ||
        staging_slot >= hip_residency.plan.staging_count ||
        nbytes > hip_residency.plan.slot_bytes ||
        hip_residency.slot_tickets[physical_slot] != 0 ||
        hip_residency.staging_tickets[staging_slot] != 0)
        return -1;
    expected_offset = hip_residency.plan.slot_base_offset +
        (uint64_t)physical_slot * hip_residency.plan.slot_bytes;
    if (device_offset != expected_offset ||
        device_offset > hip_residency.plan.device_budget_bytes ||
        nbytes > hip_residency.plan.device_budget_bytes - device_offset)
        return -1;
    error = hipMemcpyAsync(hip_residency.device_arena + (size_t)device_offset,
        hip_residency.staging_arena +
            (size_t)staging_slot * (size_t)hip_residency.plan.staging_bytes,
        nbytes, hipMemcpyHostToDevice, hip_residency.stream);
    if (error != hipSuccess)
        return hip_success(error, "hipMemcpyAsync residency populate");
    error = hipEventRecord(
        hip_residency.slot_events[physical_slot], hip_residency.stream);
    if (error != hipSuccess) {
        if (hip_success(hipStreamSynchronize(hip_residency.stream),
                "hipStreamSynchronize residency rollback") != 0 &&
            hip_success(hipDeviceSynchronize(),
                "hipDeviceSynchronize residency rollback") != 0)
            hip_residency.poisoned = 1;
        hip_success(error, "hipEventRecord residency");
        return -1;
    }
    hip_residency.slot_tickets[physical_slot] = ticket;
    hip_residency.staging_tickets[staging_slot] = ticket;
    hip_residency.slot_staging[physical_slot] = staging_slot;
    return 0;
}

static int hip_residency_poll(
        void *context, uint32_t physical_slot,
        uint64_t ticket, int *complete) {
    hipError_t error;
    uint32_t staging_slot;
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned ||
        !complete || ticket == 0 ||
        physical_slot >= hip_residency.plan.slot_count ||
        hip_residency.slot_tickets[physical_slot] != ticket)
        return -1;
    error = hipEventQuery(hip_residency.slot_events[physical_slot]);
    if (error == hipErrorNotReady) {
        *complete = 0;
        return 0;
    }
    if (error != hipSuccess)
        return hip_success(error, "hipEventQuery residency");
    staging_slot = hip_residency.slot_staging[physical_slot];
    if (staging_slot >= hip_residency.plan.staging_count ||
        hip_residency.staging_tickets[staging_slot] != ticket)
        return -1;
    hip_residency.slot_tickets[physical_slot] = 0;
    hip_residency.staging_tickets[staging_slot] = 0;
    hip_residency.slot_staging[physical_slot] = UINT32_MAX;
    *complete = 1;
    return 0;
}

static int hip_residency_resolve(
        void *context, uint64_t device_offset,
        size_t nbytes, uintptr_t *address) {
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned ||
        !address || nbytes == 0 ||
        device_offset > hip_residency.plan.device_budget_bytes ||
        nbytes > hip_residency.plan.device_budget_bytes - device_offset)
        return -1;
    *address = (uintptr_t)(void *)(hip_residency.device_arena +
        (size_t)device_offset);
    return *address ? 0 : -1;
}

static int hip_residency_retire(
        void *context, uint32_t physical_slot, uint64_t generation) {
    uint64_t ticket;
    uint32_t staging_slot;
    (void)context;
    (void)generation;
    if (!hip_residency.ready || hip_residency.poisoned ||
        physical_slot >= hip_residency.plan.slot_count)
        return -1;
    ticket = hip_residency.slot_tickets[physical_slot];
    staging_slot = hip_residency.slot_staging[physical_slot];
    if (ticket != 0) {
        if (hip_success(hipEventSynchronize(
                hip_residency.slot_events[physical_slot]),
                "hipEventSynchronize residency retire") != 0)
            return -1;
        if (staging_slot >= hip_residency.plan.staging_count ||
            hip_residency.staging_tickets[staging_slot] != ticket)
            return -1;
        hip_residency.staging_tickets[staging_slot] = 0;
    } else if (staging_slot != UINT32_MAX) {
        return -1;
    }
    hip_residency.slot_tickets[physical_slot] = 0;
    hip_residency.slot_staging[physical_slot] = UINT32_MAX;
    return 0;
}

static int hip_residency_destroy(void *context) {
    (void)context;
    if (!hip_residency.ready) return -1;
    return hip_residency_release_storage();
}

static const SaltGpuResidencyBackendOps hip_residency_ops = {
    SALT_GPU_RESIDENCY_ABI_VERSION,
    sizeof(SaltGpuResidencyBackendOps),
    hip_residency_create,
    hip_residency_staging,
    hip_residency_populate,
    hip_residency_poll,
    hip_residency_resolve,
    hip_residency_retire,
    hip_residency_destroy,
    hip_residency_permanent_layout,
    hip_residency_permanent_span,
    hip_residency_permanent_copy,
    hip_residency_permanent_publish,
};

extern "C" const SaltGpuResidencyBackendOps *
salt_gpu_residency_backend_ops(void) {
    return &hip_residency_ops;
}

static int hip_env_u32(const char *name, uint32_t minimum,
        uint32_t maximum, uint32_t fallback, uint32_t *value_out) {
    const char *text;
    char *end = NULL;
    unsigned long value;
    if (!name || !value_out || minimum > maximum || fallback < minimum ||
        fallback > maximum)
        return -1;
    text = getenv(name);
    if (!text || !*text) {
        *value_out = fallback;
        return 0;
    }
    value = strtoul(text, &end, 10);
    if (!end || *end || value < minimum || value > maximum)
        return -1;
    *value_out = (uint32_t)value;
    return 0;
}

__device__ static uint16_t hip_load_u16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ static uint32_t hip_load_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

__device__ static uint32_t hip_load_u32_aligned(const unsigned char *p) {
    uintptr_t address = (uintptr_t)p;
    const uint32_t *aligned = (const uint32_t *)(address & ~(uintptr_t)3u);
    unsigned int shift = (unsigned int)(address & 3u) * 8u;
    uint32_t low = aligned[0];
    if (shift == 0) return low;
    uint32_t high = aligned[1];
    return (low >> shift) | (high << (32u - shift));
}

__device__ static float hip_bf16(const unsigned char *p) {
    return __uint_as_float((uint32_t)hip_load_u16(p) << 16);
}

__device__ static float hip_f32(const unsigned char *p) {
    return __uint_as_float(hip_load_u32(p));
}

__device__ static float hip_nvfp4_e2m1(uint8_t code) {
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                             2.0f, 3.0f, 4.0f, 6.0f};
    float value = values[code & 7u];
    if (value == 0.0f) return 0.0f;
    return (code & 8u) ? -value : value;
}

__device__ static float hip_nvfp4_e4m3fn(uint8_t code) {
    int sign = (code >> 7) & 1;
    int exponent = (code >> 3) & 15;
    int mantissa = code & 7;
    float value;
    if (exponent == 0)
        value = __fmul_rn((float)mantissa, 0.001953125f);
    else {
        float power = __uint_as_float((uint32_t)(exponent + 120) << 23);
        float fraction = __fadd_rn(1.0f,
                                   __fmul_rn((float)mantissa, 0.125f));
        value = __fmul_rn(power, fraction);
    }
    return sign ? -value : value;
}

__device__ static uint8_t hip_nvfp4_e4m3fn_encode(float value) {
    uint8_t sign = (__float_as_uint(value) >> 31) ? 0x80u : 0u;
    float magnitude = fabsf(value);
    uint8_t best = 0;
    float best_distance = __uint_as_float(0x7f800000u);
    if (magnitude > 448.0f) magnitude = 448.0f;
    for (unsigned int candidate = 0; candidate <= 0x7eu; candidate++) {
        float decoded = hip_nvfp4_e4m3fn((uint8_t)candidate);
        float distance = fabsf(__fadd_rn(magnitude, -decoded));
        if (distance < best_distance ||
            (distance == best_distance && (candidate & 1u) == 0u &&
             (best & 1u) != 0u)) {
            best_distance = distance;
            best = (uint8_t)candidate;
        }
    }
    return (uint8_t)(best | sign);
}

__device__ static uint8_t hip_nvfp4_e2m1_encode(float value) {
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                             2.0f, 3.0f, 4.0f, 6.0f};
    uint8_t sign = (__float_as_uint(value) >> 31) ? 8u : 0u;
    float magnitude = fabsf(value);
    uint8_t best = 0;
    float best_distance = __uint_as_float(0x7f800000u);
    for (unsigned int candidate = 0; candidate < 8u; candidate++) {
        float distance = fabsf(__fadd_rn(magnitude, -values[candidate]));
        if (distance < best_distance ||
            (distance == best_distance && (candidate & 1u) == 0u &&
             (best & 1u) != 0u)) {
            best_distance = distance;
            best = (uint8_t)candidate;
        }
    }
    return (uint8_t)(best | sign);
}

__global__ static void hip_nvfp4_qdq_exact(
    const float *input, int C, int B, float input_global_scale,
    uint8_t *packed, uint8_t *scales, float *dequant) {
    int block = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int base = block * 16;
    if (base >= B * C) return;
    float maximum = 0.0f;
    for (int lane = 0; lane < 16; lane++) {
        float magnitude = fabsf(input[base + lane]);
        if (magnitude > maximum) maximum = magnitude;
    }
    float scaled_maximum = __fmul_rn(maximum, 1.0f / 6.0f);
    float raw_scale = __fmul_rn(input_global_scale, scaled_maximum);
    if (raw_scale > 448.0f) raw_scale = 448.0f;
    uint8_t scale_code = hip_nvfp4_e4m3fn_encode(raw_scale);
    scales[block] = scale_code;
    float scale = hip_nvfp4_e4m3fn(scale_code);
    float output_scale = scale == 0.0f ? 0.0f :
                         __fdiv_rn(input_global_scale, scale);
    float dequant_scale = __fdiv_rn(scale, input_global_scale);
    for (int lane = 0; lane < 16; lane += 2) {
        float first = __fmul_rn(input[base + lane], output_scale);
        float second = __fmul_rn(input[base + lane + 1], output_scale);
        if (first > 6.0f) first = 6.0f;
        if (first < -6.0f) first = -6.0f;
        if (second > 6.0f) second = 6.0f;
        if (second < -6.0f) second = -6.0f;
        uint8_t first_code = hip_nvfp4_e2m1_encode(first);
        uint8_t second_code = hip_nvfp4_e2m1_encode(second);
        packed[(base + lane) / 2] =
            (uint8_t)(first_code | (uint8_t)(second_code << 4));
        dequant[base + lane] =
            __fmul_rn(hip_nvfp4_e2m1(first_code), dequant_scale);
        dequant[base + lane + 1] =
            __fmul_rn(hip_nvfp4_e2m1(second_code), dequant_scale);
    }
}

__global__ static void hip_q4_exact(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C) {
    int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int batch = (int)blockIdx.y;
    if (r >= R) return;
    x += (size_t)batch * (size_t)C;
    y += (size_t)batch * (size_t)R;
    float acc[32];
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    for (int c = 0; c < C; c++) {
        uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C + (uint32_t)c;
        uint64_t g = k / 64u;
        uint32_t packed = hip_load_u32(resource + voff + (k >> 3u) * 4u);
        uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
        float scale = hip_bf16(resource + soff + g * 2u);
        float bias = hip_bf16(resource + boff + g * 2u);
        float t = __fmul_rn((float)q, scale);
        t = __fadd_rn(t, bias);
        float u = __fmul_rn(t, x[c]);
        acc[c & 31] = __fadd_rn(acc[c & 31], u);
    }
    float lane0 = 0.0f, lane1 = 0.0f;
    float lane2 = 0.0f, lane3 = 0.0f;
    for (int a = 0; a < 8; a++) {
        lane0 = __fadd_rn(lane0, acc[4 * a]);
        lane1 = __fadd_rn(lane1, acc[4 * a + 1]);
        lane2 = __fadd_rn(lane2, acc[4 * a + 2]);
        lane3 = __fadd_rn(lane3, acc[4 * a + 3]);
    }
    float t0 = __fadd_rn(lane0, lane2);
    float t1 = __fadd_rn(lane1, lane3);
    y[r] = __fadd_rn(t0, t1);
}

__global__ static void hip_q4_warp_exact(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    float acc = 0.0f;
    if (r < R) {
        const float *input = x + (size_t)batch * (size_t)C;
        for (int c = lane; c < C; c += 32) {
            uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C +
                         (uint32_t)c;
            uint64_t g = k / 64u;
            uint32_t packed = hip_load_u32_aligned(
                resource + voff + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = hip_bf16(resource + soff + g * 2u);
            float bias = hip_bf16(resource + boff + g * 2u);
            float dequant = __fmul_rn((float)q, scale);
            dequant = __fadd_rn(dequant, bias);
            float product = __fmul_rn(dequant, input[c]);
            acc = __fadd_rn(acc, product);
        }
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    float lane_sum = 0.0f;
    for (int a = 0; a < 8; a++)
        lane_sum = __fadd_rn(
            lane_sum, __shfl_sync(mask, acc, base + 4 * a));
    float lane0 = __shfl_sync(mask, lane_sum, 0);
    float lane1 = __shfl_sync(mask, lane_sum, 1);
    float lane2 = __shfl_sync(mask, lane_sum, 2);
    float lane3 = __shfl_sync(mask, lane_sum, 3);
    float t0 = __fadd_rn(lane0, lane2);
    float t1 = __fadd_rn(lane1, lane3);
    if (lane == 0 && r < R)
        y[(size_t)batch * (size_t)R + (size_t)r] = __fadd_rn(t0, t1);
}

__global__ static void hip_q8_exact(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C) {
    int r = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int batch = (int)blockIdx.y;
    if (r >= R) return;
    x += (size_t)batch * (size_t)C;
    y += (size_t)batch * (size_t)R;
    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C + (uint32_t)c;
        uint64_t g = k / 64u;
        uint32_t q = resource[voff + k];
        float scale = hip_bf16(resource + soff + g * 2u);
        float bias = hip_bf16(resource + boff + g * 2u);
        float t = __fmul_rn((float)q, scale);
        t = __fadd_rn(t, bias);
        float u = __fmul_rn(t, x[c]);
        acc = __fadd_rn(acc, u);
    }
    y[r] = acc;
}

__global__ static void hip_q8_warp_exact(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    float acc = 0.0f;
    const float *input = x + (size_t)batch * (size_t)C;
    unsigned int mask = 0xffffffffu;
    for (int cb = 0; cb < C; cb += 32) {
        int c = cb + lane;
        float product = 0.0f;
        if (r < R && c < C) {
            uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C +
                         (uint32_t)c;
            uint64_t g = k / 64u;
            uint32_t q = resource[voff + k];
            float scale = hip_bf16(resource + soff + g * 2u);
            float bias = hip_bf16(resource + boff + g * 2u);
            float dequant = __fmul_rn((float)q, scale);
            dequant = __fadd_rn(dequant, bias);
            product = __fmul_rn(dequant, input[c]);
        }
        int count = C - cb;
        if (count > 32) count = 32;
        for (int source = 0; source < count; source++) {
            float selected = __shfl_sync(mask, product, source);
            if (lane == 0) acc = __fadd_rn(acc, selected);
        }
    }
    if (lane == 0 && r < R)
        y[(size_t)batch * (size_t)R + (size_t)r] = acc;
}

__global__ static void hip_bf16_warp_exact(
    const unsigned char *resource, uint64_t value_offset,
    const float *input, float *output, int R, int C) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * 4 + warp;
    float acc = 0.0f;
    if (row < R && lane < 8) {
        const unsigned char *weights = resource + value_offset +
            (uint64_t)(uint32_t)row * (uint32_t)C * 2u;
        for (int col = lane; col < C; col += 8) {
            float weight = hip_bf16(weights + (size_t)col * 2u);
            float product = __fmul_rn(weight, input[col]);
            acc = __fadd_rn(acc, product);
        }
    }
    unsigned int mask = 0xffffffffu;
    float a0 = __shfl_sync(mask, acc, 0);
    float a1 = __shfl_sync(mask, acc, 1);
    float a2 = __shfl_sync(mask, acc, 2);
    float a3 = __shfl_sync(mask, acc, 3);
    float a4 = __shfl_sync(mask, acc, 4);
    float a5 = __shfl_sync(mask, acc, 5);
    float a6 = __shfl_sync(mask, acc, 6);
    float a7 = __shfl_sync(mask, acc, 7);
    float even01 = __fadd_rn(a0, a2);
    float even23 = __fadd_rn(a4, a6);
    float odd01 = __fadd_rn(a1, a3);
    float odd23 = __fadd_rn(a5, a7);
    float even = __fadd_rn(even01, even23);
    float odd = __fadd_rn(odd01, odd23);
    if (lane == 0 && row < R)
        output[row] = __fadd_rn(even, odd);
}

__global__ static void hip_bf16_warp_batch_exact(
    const unsigned char *resource, uint64_t value_offset,
    const float *input, float *output, int R, int C, int B) {
    __shared__ float weight_tile[8];
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int warps = (int)blockDim.x >> 5;
    int row = (int)blockIdx.x;
    int token_span = warps * 8;
    int token_base = (int)blockIdx.y * token_span;
    float acc[8];
    if (row >= R || warps < 1 || warps > 8) return;
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;
    const unsigned char *weights = resource + value_offset +
        (uint64_t)(uint32_t)row * (uint32_t)C * 2u;
    for (int cb = 0; cb < C; cb += 8) {
        if (warp == 0 && lane < 8) {
            int col = cb + lane;
            weight_tile[lane] = col < C
                ? hip_bf16(weights + (size_t)col * 2u) : 0.0f;
        }
        __syncthreads();
        int index = 0;
        for (int token = token_base + warp;
             token < B && token < token_base + token_span;
             token += warps, index++) {
            int col = cb + lane;
            if (lane < 8 && col < C) {
                float product = __fmul_rn(weight_tile[lane],
                    input[(size_t)token * (size_t)C + (size_t)col]);
                acc[index] = __fadd_rn(acc[index], product);
            }
        }
        __syncthreads();
    }
    unsigned int mask = 0xffffffffu;
    int index = 0;
    for (int token = token_base + warp;
         token < B && token < token_base + token_span;
         token += warps, index++) {
        float a0 = __shfl_sync(mask, acc[index], 0);
        float a1 = __shfl_sync(mask, acc[index], 1);
        float a2 = __shfl_sync(mask, acc[index], 2);
        float a3 = __shfl_sync(mask, acc[index], 3);
        float a4 = __shfl_sync(mask, acc[index], 4);
        float a5 = __shfl_sync(mask, acc[index], 5);
        float a6 = __shfl_sync(mask, acc[index], 6);
        float a7 = __shfl_sync(mask, acc[index], 7);
        float even01 = __fadd_rn(a0, a2);
        float even23 = __fadd_rn(a4, a6);
        float odd01 = __fadd_rn(a1, a3);
        float odd23 = __fadd_rn(a5, a7);
        float even = __fadd_rn(even01, even23);
        float odd = __fadd_rn(odd01, odd23);
        if (lane == 0)
            output[(size_t)token * (size_t)R + (size_t)row] =
                __fadd_rn(even, odd);
    }
}

__global__ static void hip_nvfp4_warp_exact(
    const unsigned char *resource, uint64_t value_offset,
    uint64_t scale_offset, uint64_t weight_global_offset,
    const float *input_dequant, float *output, int R, int C, int B) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    float acc = 0.0f;
    if (row < R && batch < B) {
        float stored_weight_global = hip_f32(resource + weight_global_offset);
        float weight_global = __fdiv_rn(1.0f, stored_weight_global);
        const float *input = input_dequant + (size_t)batch * (size_t)C;
        for (int col = lane; col < C; col += 32) {
            uint64_t index = (uint64_t)(uint32_t)row * (uint32_t)C +
                             (uint32_t)col;
            uint8_t packed = resource[value_offset + (index >> 1u)];
            uint8_t code = (index & 1u) ? (uint8_t)(packed >> 4) :
                                          (uint8_t)(packed & 15u);
            uint64_t scale_index =
                (uint64_t)(uint32_t)row * (uint32_t)(C / 16) +
                (uint32_t)(col / 16);
            float scale = hip_nvfp4_e4m3fn(
                resource[scale_offset + scale_index]);
            float scaled = __fmul_rn(scale, weight_global);
            float weight = __fmul_rn(hip_nvfp4_e2m1(code), scaled);
            float product = __fmul_rn(weight, input[col]);
            acc = __fadd_rn(acc, product);
        }
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    float lane_sum = 0.0f;
    for (int a = 0; a < 8; a++)
        lane_sum = __fadd_rn(
            lane_sum, __shfl_sync(mask, acc, base + 4 * a));
    float lane0 = __shfl_sync(mask, lane_sum, 0);
    float lane1 = __shfl_sync(mask, lane_sum, 1);
    float lane2 = __shfl_sync(mask, lane_sum, 2);
    float lane3 = __shfl_sync(mask, lane_sum, 3);
    float t0 = __fadd_rn(lane0, lane2);
    float t1 = __fadd_rn(lane1, lane3);
    if (lane == 0 && row < R && batch < B)
        output[(size_t)batch * (size_t)R + (size_t)row] =
            __fadd_rn(t0, t1);
}

__global__ static void hip_nvfp4_heterogeneous_qdq(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *input, float *dequant) {
    int descriptor_index = (int)blockIdx.z;
    int group = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (descriptor_index >= descriptor_count) return;
    HipBatchDesc descriptor = descriptors[descriptor_index];
    int group_count = descriptor.batch * descriptor.cols / 16;
    if (group >= group_count) return;
    int base = group * 16;
    const float *source = input + descriptor.input_offset + base;
    float *target = dequant + descriptor.input_offset + base;
    float input_global_scale =
        hip_f32(descriptor.resource + descriptor.aux_offset);
    float maximum = 0.0f;
    for (int lane = 0; lane < 16; lane++) {
        float magnitude = fabsf(source[lane]);
        if (magnitude > maximum) maximum = magnitude;
    }
    float scaled_maximum = __fmul_rn(maximum, 1.0f / 6.0f);
    float raw_scale = __fmul_rn(input_global_scale, scaled_maximum);
    if (raw_scale > 448.0f) raw_scale = 448.0f;
    uint8_t scale_code = hip_nvfp4_e4m3fn_encode(raw_scale);
    float scale = hip_nvfp4_e4m3fn(scale_code);
    float output_scale = scale == 0.0f ? 0.0f :
                         __fdiv_rn(input_global_scale, scale);
    float dequant_scale = __fdiv_rn(scale, input_global_scale);
    for (int lane = 0; lane < 16; lane++) {
        float quantized = __fmul_rn(source[lane], output_scale);
        if (quantized > 6.0f) quantized = 6.0f;
        if (quantized < -6.0f) quantized = -6.0f;
        uint8_t code = hip_nvfp4_e2m1_encode(quantized);
        target[lane] = __fmul_rn(hip_nvfp4_e2m1(code), dequant_scale);
    }
}

__global__ static void hip_nvfp4_heterogeneous_warp(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *input_dequant, float *output) {
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    HipBatchDesc descriptor = {0};
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    float acc = 0.0f;
    if (descriptor_index < descriptor_count && row < descriptor.rows &&
        batch < descriptor.batch) {
        float stored_weight_global =
            hip_f32(descriptor.resource + descriptor.bias_offset);
        float weight_global = __fdiv_rn(1.0f, stored_weight_global);
        const float *input = input_dequant + descriptor.input_offset +
            (size_t)batch * (size_t)descriptor.cols;
        for (int col = lane; col < descriptor.cols; col += 32) {
            uint64_t index = (uint64_t)(uint32_t)row *
                (uint32_t)descriptor.cols + (uint32_t)col;
            uint8_t packed = descriptor.resource[
                descriptor.value_offset + (index >> 1u)];
            uint8_t code = (index & 1u) ? (uint8_t)(packed >> 4) :
                                          (uint8_t)(packed & 15u);
            uint64_t scale_index = (uint64_t)(uint32_t)row *
                (uint32_t)(descriptor.cols / 16) + (uint32_t)(col / 16);
            float scale = hip_nvfp4_e4m3fn(descriptor.resource[
                descriptor.scale_offset + scale_index]);
            float scaled = __fmul_rn(scale, weight_global);
            float weight = __fmul_rn(hip_nvfp4_e2m1(code), scaled);
            float product = __fmul_rn(weight, input[col]);
            acc = __fadd_rn(acc, product);
        }
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    float lane_sum = 0.0f;
    for (int a = 0; a < 8; a++)
        lane_sum = __fadd_rn(
            lane_sum, __shfl_sync(mask, acc, base + 4 * a));
    float lane0 = __shfl_sync(mask, lane_sum, 0);
    float lane1 = __shfl_sync(mask, lane_sum, 1);
    float lane2 = __shfl_sync(mask, lane_sum, 2);
    float lane3 = __shfl_sync(mask, lane_sum, 3);
    float t0 = __fadd_rn(lane0, lane2);
    float t1 = __fadd_rn(lane1, lane3);
    if (lane == 0 && descriptor_index < descriptor_count &&
        row < descriptor.rows && batch < descriptor.batch)
        output[descriptor.output_offset + (size_t)batch *
            (size_t)descriptor.rows + (size_t)row] = __fadd_rn(t0, t1);
}

__global__ static void hip_q4_warp_batch(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C, int B) {
    __shared__ float dequant[1024];
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x;
    float acc[16];
    if (r >= R) return;
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int tile = 0; tile < C; tile += 1024) {
        int loaded_column = tile + (int)threadIdx.x;
        if (loaded_column < C) {
            uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C +
                         (uint32_t)loaded_column;
            uint64_t g = k / 64u;
            uint32_t packed = hip_load_u32(
                resource + voff + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = hip_bf16(resource + soff + g * 2u);
            float bias = hip_bf16(resource + boff + g * 2u);
            float value = __fmul_rn((float)q, scale);
            dequant[threadIdx.x] = __fadd_rn(value, bias);
        } else {
            dequant[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        int index = 0;
        for (int token = warp; token < B; token += 32, index++) {
            for (int offset = 0; offset < 1024; offset += 32) {
                int c = tile + offset + lane;
                if (c < C) {
                    float product = __fmul_rn(
                        dequant[offset + lane],
                        x[(size_t)token * (size_t)C + (size_t)c]);
                    acc[index] = __fadd_rn(acc[index], product);
                }
            }
        }
        __syncthreads();
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    int index = 0;
    for (int token = warp; token < B; token += 32, index++) {
        float lane_sum = 0.0f;
        for (int a = 0; a < 8; a++)
            lane_sum = __fadd_rn(
                lane_sum, __shfl_sync(mask, acc[index], base + 4 * a));
        float lane0 = __shfl_sync(mask, lane_sum, 0);
        float lane1 = __shfl_sync(mask, lane_sum, 1);
        float lane2 = __shfl_sync(mask, lane_sum, 2);
        float lane3 = __shfl_sync(mask, lane_sum, 3);
        float t0 = __fadd_rn(lane0, lane2);
        float t1 = __fadd_rn(lane1, lane3);
        if (lane == 0)
            y[(size_t)token * (size_t)R + (size_t)r] = __fadd_rn(t0, t1);
    }
}

__device__ static float hip_salt_expf(float x) {
    if (x != x) return x;
    if (x > 88.722839f) return 3.402823466e+38f;
    if (x < -87.336548f) return 0.0f;
    double n = round((double)x *
        1.442695040888963407359924681001892137426645954152985934135449406931);
    double r = (double)x - n *
        0.693147180559945309417232121458176568075500134360255254120680009;
    float rf = (float)r;
    float p = 2.48015873015873e-05f;
    p = __fadd_rn(__fmul_rn(p, rf), 1.98412698412698e-04f);
    p = __fadd_rn(__fmul_rn(p, rf), 1.38888888888889e-03f);
    p = __fadd_rn(__fmul_rn(p, rf), 8.33333333333333e-03f);
    p = __fadd_rn(__fmul_rn(p, rf), 4.16666666666667e-02f);
    p = __fadd_rn(__fmul_rn(p, rf), 1.66666666666667e-01f);
    p = __fadd_rn(__fmul_rn(p, rf), 5.0e-01f);
    p = __fadd_rn(__fmul_rn(p, rf), 1.0f);
    p = __fadd_rn(__fmul_rn(p, rf), 1.0f);
    return ldexpf(p, (int)n);
}

__device__ static float hip_round_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
        bits &= UINT32_C(0xffff0000);
    }
    return __uint_as_float(bits);
}

__device__ static float hip_bf16_mul(float a, float b) {
    return hip_round_bf16(__fmul_rn(hip_round_bf16(a),
                                     hip_round_bf16(b)));
}

__device__ static float hip_bf16_add(float a, float b) {
    return hip_round_bf16(__fadd_rn(hip_round_bf16(a),
                                     hip_round_bf16(b)));
}

__device__ static float hip_salt_rsqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f) return 1.0f / 0.0f;
    if (x == 1.0f) return 1.0f;
    if (x < 0x1p-126f)
        return __fmul_rn(hip_salt_rsqrtf(__fmul_rn(x, 0x1p24f)),
                         0x1p12f);
    uint32_t bits = __float_as_uint(x);
    bits = UINT32_C(0x5f3759df) - (bits >> 1);
    double y = (double)__uint_as_float(bits);
    double xd = (double)x;
    for (int i = 0; i < 6; i++) {
        double yy = __dmul_rn(y, y);
        double half_product = __dmul_rn(0.5, __dmul_rn(xd, yy));
        double correction = __dadd_rn(1.5, -half_product);
        y = __dmul_rn(y, correction);
    }
    return (float)y;
}

__global__ static void hip_attention_transform_exact(
    int n_heads, int n_kv_heads, int head_dim, int rope_dim,
    int start_position, int batch, int query_stride, int kv_stride,
    int row_capacity, int publish_cache, float eps,
    float *queries, float *keys, float *values,
    const float *q_weight, const float *k_weight,
    const float *cosines, const float *sines, int rope_pairs,
    float *key_cache, float *value_cache, int *failure) {
    int tasks_per_token = n_heads + 2 * n_kv_heads;
    int task = (int)blockIdx.x;
    int token = task / tasks_per_token;
    int local = task - token * tasks_per_token;
    int kind, head;
    float *row;
    const float *weight = NULL;
    __shared__ float inverse;
    if (token >= batch) return;
    if (row_capacity < 1) {
        if (threadIdx.x == 0) atomicExch(failure, 1);
        return;
    }
    if (local < n_heads) {
        kind = 0;
        head = local;
        row = queries + (size_t)token * (size_t)query_stride +
            (size_t)head * (size_t)head_dim;
        weight = q_weight;
    } else if (local < n_heads + n_kv_heads) {
        kind = 1;
        head = local - n_heads;
        row = keys + (size_t)token * (size_t)kv_stride +
            (size_t)head * (size_t)head_dim;
        weight = k_weight;
    } else {
        kind = 2;
        head = local - n_heads - n_kv_heads;
        row = values + (size_t)token * (size_t)kv_stride +
            (size_t)head * (size_t)head_dim;
    }
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            float x = row[i];
            if (!isfinite(x) || (weight && !isfinite(weight[i]))) {
                atomicExch(failure, 1);
            }
            ss = __fadd_rn(ss, __fmul_rn(x, x));
        }
        float mean = __fadd_rn(__fdiv_rn(ss, (float)head_dim), eps);
        inverse = hip_salt_rsqrtf(mean);
        if (!isfinite(inverse)) atomicExch(failure, 1);
    }
    __syncthreads();
    for (int i = (int)threadIdx.x; i < head_dim; i += (int)blockDim.x) {
        float y = __fmul_rn(row[i], inverse);
        if (weight) y = __fmul_rn(y, weight[i]);
        row[i] = y;
        if (!isfinite(y)) atomicExch(failure, 1);
    }
    __syncthreads();
    if (kind != 2) {
        for (int i = (int)threadIdx.x; i < head_dim; i += (int)blockDim.x)
            row[i] = hip_round_bf16(row[i]);
        __syncthreads();
        int half = head_dim / 2;
        for (int pair = (int)threadIdx.x; pair < rope_dim / 2;
             pair += (int)blockDim.x) {
            float c = hip_round_bf16(
                cosines[(size_t)token * (size_t)rope_pairs + (size_t)pair]);
            float s = hip_round_bf16(
                sines[(size_t)token * (size_t)rope_pairs + (size_t)pair]);
            float a = row[pair];
            float b = row[half + pair];
            float ac = hip_bf16_mul(a, c);
            float bs = hip_bf16_mul(b, s);
            float bc = hip_bf16_mul(b, c);
            float as = hip_bf16_mul(a, s);
            row[pair] = hip_bf16_add(ac, -bs);
            row[half + pair] = hip_bf16_add(bc, as);
        }
    }
    __syncthreads();
    size_t destination = (size_t)(start_position + token) %
        (size_t)row_capacity;
    if (publish_cache && kind == 1)
        for (int i = (int)threadIdx.x; i < head_dim; i += (int)blockDim.x)
            key_cache[destination * (size_t)kv_stride +
                (size_t)head * (size_t)head_dim + (size_t)i] = row[i];
    else if (publish_cache && kind == 2)
        for (int i = (int)threadIdx.x; i < head_dim; i += (int)blockDim.x)
            value_cache[destination * (size_t)kv_stride +
                (size_t)head * (size_t)head_dim + (size_t)i] = row[i];
}

__global__ static void hip_attention_exact(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const float *queries, const float *keys, const float *values,
    int start_position, int batch, int query_stride, int kv_stride,
    int row_capacity, const float *new_keys, const float *new_values,
    int new_start_position, int new_count, float *outputs, int *failure) {
    extern __shared__ float scores[];
    __shared__ float denominator;
    __shared__ int score_count;
    int task = (int)blockIdx.x;
    int token = task / n_heads;
    int head = task % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv_head = head / groups;
    if (row_capacity < 1) {
        if (threadIdx.x == 0) atomicExch(failure, 1);
        return;
    }
    int position = start_position + token;
    int first = full_attention ? 0 : position - window + 1;
    if (first < 0) first = 0;
    int count = position - first + 1;
    const float *query = queries + (size_t)token * (size_t)query_stride +
        (size_t)head * (size_t)head_dim;
    if (threadIdx.x == 0) {
        float maximum = -1.0f / 0.0f;
        for (int i = 0, position_k = first; position_k <= position;
             position_k++, i++) {
            size_t physical_position = (size_t)position_k %
                (size_t)row_capacity;
            const float *key = new_keys && position_k >= new_start_position &&
                    position_k < new_start_position + new_count
                ? new_keys + (size_t)(position_k - new_start_position) *
                    (size_t)kv_stride + (size_t)kv_head * (size_t)head_dim
                : keys + physical_position * (size_t)kv_stride +
                    (size_t)kv_head * (size_t)head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++)
                score = __fadd_rn(score, __fmul_rn(query[d], key[d]));
            scores[i] = score;
            if (score > maximum) maximum = score;
        }
        float sum = 0.0f;
        for (int i = 0; i < count; i++) {
            scores[i] = hip_salt_expf(__fadd_rn(scores[i], -maximum));
            sum = __fadd_rn(sum, scores[i]);
        }
        denominator = sum;
        score_count = (!(sum > 0.0f) || !isfinite(sum)) ? -1 : count;
        if (score_count < 0) atomicExch(failure, 1);
    }
    __syncthreads();
    if (score_count < 0) return;
    float *output = outputs + (size_t)token * (size_t)query_stride +
        (size_t)head * (size_t)head_dim;
    for (int d = (int)threadIdx.x; d < head_dim; d += (int)blockDim.x) {
        float accumulator = 0.0f;
        for (int i = 0, position_v = first; i < score_count;
             i++, position_v++) {
            size_t physical_position = (size_t)position_v %
                (size_t)row_capacity;
            const float *value = new_values && position_v >= new_start_position &&
                    position_v < new_start_position + new_count
                ? new_values + (size_t)(position_v - new_start_position) *
                    (size_t)kv_stride + (size_t)kv_head * (size_t)head_dim
                : values + physical_position * (size_t)kv_stride +
                    (size_t)kv_head * (size_t)head_dim;
            float weight = __fdiv_rn(scores[i], denominator);
            accumulator = __fadd_rn(
                accumulator, __fmul_rn(weight, value[d]));
        }
        output[d] = accumulator;
    }
}

__global__ static void hip_attention_publish_exact(
        const float *keys, const float *values,
        float *key_cache, float *value_cache,
        int start_position, int batch, int kv_stride, int row_capacity) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * (size_t)kv_stride;
    if (index >= count || row_capacity < 1) return;
    size_t row = index / (size_t)kv_stride;
    size_t column = index - row * (size_t)kv_stride;
    size_t destination = (size_t)(start_position + (int)row) %
        (size_t)row_capacity;
    key_cache[destination * (size_t)kv_stride + column] = keys[index];
    value_cache[destination * (size_t)kv_stride + column] = values[index];
}

__global__ static void hip_text_publish_committed_kv(
        const HipTextKvRef *layers, uint32_t layer_count,
        const float *canonical, uint32_t source_position,
        uint32_t row_count, const int *status) {
    uint32_t layer = (uint32_t)blockIdx.z;
    uint32_t row = (uint32_t)blockIdx.y;
    if (!layers || !canonical || !status || *status != 0 ||
        layer >= layer_count || row >= row_count)
        return;
    const HipTextKvRef *kv = &layers[layer];
    uint32_t position = source_position + row;
    uint32_t destination_row;
    if (kv->private_mode == SALT_TEXT_KV_PRIVATE_ABSOLUTE) {
        if (position >= kv->row_capacity) return;
        destination_row = position;
    } else if (kv->private_mode == SALT_TEXT_KV_PRIVATE_RING) {
        if (kv->row_capacity == 0u) return;
        destination_row = position % kv->row_capacity;
    } else {
        return;
    }
    const float *source_keys = canonical +
        kv->tentative_key_float_offset + (size_t)row * kv->width;
    const float *source_values = canonical +
        kv->tentative_value_float_offset + (size_t)row * kv->width;
    float *destination_keys = kv->device_keys +
        (size_t)destination_row * kv->width;
    float *destination_values = kv->device_values +
        (size_t)destination_row * kv->width;
    for (uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
         column < kv->width; column += gridDim.x * blockDim.x) {
        destination_keys[column] = source_keys[column];
        destination_values[column] = source_values[column];
    }
}

__device__ static float hip_salt_tanhf(float x) {
    if (x != x) return x;
    if (x >= 10.0f) return 1.0f;
    if (x <= -10.0f) return -1.0f;
    float magnitude = x < 0.0f ? -x : x;
    float exponential = hip_salt_expf(__fmul_rn(2.0f, magnitude));
    float result = __fdiv_rn(__fadd_rn(exponential, -1.0f),
                             __fadd_rn(exponential, 1.0f));
    return x < 0.0f ? -result : result;
}

__device__ static float hip_gemma4_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    float cubic = __fmul_rn(0.044715f, x);
    cubic = __fmul_rn(cubic, x);
    cubic = __fmul_rn(cubic, x);
    float inner = __fadd_rn(x, cubic);
    float tanh_value = hip_salt_tanhf(__fmul_rn(k, inner));
    float half_x = __fmul_rn(0.5f, x);
    return __fmul_rn(half_x, __fadd_rn(1.0f, tanh_value));
}

__device__ static double hip_text_invsqrt64(float x) {
    uint32_t bits = __float_as_uint(x);
    bits = UINT32_C(0x5f3759df) - (bits >> 1);
    double y = (double)__uint_as_float(bits);
    double xd = (double)x;
    for (int i = 0; i < 6; i++) {
        double yy = __dmul_rn(y, y);
        double correction = __dadd_rn(
            1.5, -__dmul_rn(0.5, __dmul_rn(xd, yy)));
        y = __dmul_rn(y, correction);
    }
    return y;
}

__device__ static float hip_text_sqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f || x == 1.0f) return x;
    if (x < 0x1p-126f)
        return __fmul_rn(hip_text_sqrtf(__fmul_rn(x, 0x1p24f)), 0x1p-12f);
    return (float)__dmul_rn((double)x, hip_text_invsqrt64(x));
}

__device__ static float hip_text_logf(float x) {
    if (x != x) return x;
    if (x <= 0.0f) return -1.0f / 0.0f;
    int exponent;
    float mantissa = frexpf(x, &exponent);
    double u = __ddiv_rn(__dadd_rn((double)mantissa, -1.0),
                           __dadd_rn((double)mantissa, 1.0));
    double s = __dmul_rn(u, u);
    double z = 1.0 / 25.0;
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 23.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 21.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 19.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 17.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 15.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 13.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 11.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 9.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 7.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 5.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0 / 3.0);
    z = __dadd_rn(__dmul_rn(z, s), 1.0);
    double logarithm = __dmul_rn(2.0, __dmul_rn(u, z));
    return (float)__dadd_rn(logarithm,
        __dmul_rn((double)exponent,
                  0.693147180559945309417232121458176568));
}

__device__ static float hip_text_powf(float x, float y) {
    if (x == 1.0f || y == 0.0f) return 1.0f;
    if (x > 0.0f && y == -0.5f) return hip_salt_rsqrtf(x);
    if (x > 0.0f) return hip_salt_expf(__fmul_rn(y, hip_text_logf(x)));
    return x;
}

__device__ static void hip_text_sincos_reduce(
        float x, int *quadrant, float *reduced) {
    const double pio2_hi = 1.57079632679489655800e+00;
    const double pio2_lo = 6.12323399573676603587e-17;
    const double inv_pio2 = 6.36619772367581382433e-01;
    double n = round(__dmul_rn((double)x, inv_pio2));
    double r = __dadd_rn((double)x, -__dmul_rn(n, pio2_hi));
    r = __dadd_rn(r, -__dmul_rn(n, pio2_lo));
    long long q = (long long)n % 4ll;
    if (q < 0) q += 4;
    *quadrant = (int)q;
    *reduced = (float)r;
}

__device__ static float hip_text_sin_poly(float x) {
    float x2 = __fmul_rn(x, x);
    float z = __fadd_rn(__fmul_rn(x2, -1.95152958910081473e-04f),
                          8.33302275008926213e-03f);
    z = __fadd_rn(__fmul_rn(x2, z), -1.66666666641626524e-01f);
    z = __fmul_rn(x2, z);
    return __fadd_rn(x, __fmul_rn(x, z));
}

__device__ static float hip_text_cos_poly(float x) {
    float x2 = __fmul_rn(x, x);
    float z = __fadd_rn(__fmul_rn(x2, 2.44331571180994839e-05f),
                         -1.38873162549376522e-03f);
    z = __fadd_rn(__fmul_rn(x2, z), 4.16666679023301001e-02f);
    z = __fadd_rn(__fmul_rn(x2, z), -0.5f);
    return __fadd_rn(1.0f, __fmul_rn(x2, z));
}

__device__ static float hip_text_sinf(float x) {
    if (x != x) return x;
    int quadrant;
    float reduced;
    hip_text_sincos_reduce(x, &quadrant, &reduced);
    if (quadrant == 0) return hip_text_sin_poly(reduced);
    if (quadrant == 1) return hip_text_cos_poly(reduced);
    if (quadrant == 2) return -hip_text_sin_poly(reduced);
    return -hip_text_cos_poly(reduced);
}

__device__ static float hip_text_cosf(float x) {
    if (x != x) return x;
    int quadrant;
    float reduced;
    hip_text_sincos_reduce(x, &quadrant, &reduced);
    if (quadrant == 0) return hip_text_cos_poly(reduced);
    if (quadrant == 1) return -hip_text_sin_poly(reduced);
    if (quadrant == 2) return -hip_text_cos_poly(reduced);
    return hip_text_sin_poly(reduced);
}

__device__ static void hip_text_fail(int *status) {
    atomicExch(status, 1);
}

__global__ static void hip_text_system_fence(void) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        __threadfence_system();
}

__device__ static void hip_text_norm_block(
        const float *input, const float *weight, float *output,
        uint32_t width, float epsilon, int with_scale,
        float *inverse, int *status) {
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = input[column];
            if (!isfinite(value) || (with_scale && !isfinite(weight[column])))
                hip_text_fail(status);
            sum = __fadd_rn(sum, __fmul_rn(value, value));
        }
        *inverse = hip_salt_rsqrtf(
            __fadd_rn(__fdiv_rn(sum, (float)width), epsilon));
        if (!isfinite(*inverse)) hip_text_fail(status);
    }
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(input[column], *inverse);
        if (with_scale) value = __fmul_rn(value, weight[column]);
        output[column] = value;
        if (!isfinite(value)) hip_text_fail(status);
    }
    __syncthreads();
}

__global__ static void hip_text_rmsnorm_rows(
        const float *input, size_t input_stride, const float *weight,
        float *output, size_t output_stride, uint32_t rows, uint32_t width,
        float epsilon, int with_scale, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    if (row >= rows) return;
    hip_text_norm_block(input + (size_t)row * input_stride, weight,
        output + (size_t)row * output_stride, width, epsilon,
        with_scale, &inverse, status);
}

__global__ static void hip_text_rmsnorm_pair_rows(
        const float *input, size_t input_stride,
        const float *weight0, const float *weight1,
        float *output0, float *output1, size_t output_stride,
        uint32_t rows, uint32_t width, float epsilon, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t plane = (uint32_t)blockIdx.y;
    const float *weight = plane == 0u ? weight0 : weight1;
    float *output = plane == 0u ? output0 : output1;
    __shared__ float inverse;
    if (row >= rows || plane >= 2u || !weight || !output) return;
    hip_text_norm_block(input + (size_t)row * input_stride, weight,
        output + (size_t)row * output_stride, width, epsilon,
        1, &inverse, status);
}

__global__ static void hip_text_embedding_q4(
        const unsigned char *resource, uint64_t value_offset,
        uint64_t scale_offset, uint64_t bias_offset,
        const int32_t *token_ids, uint32_t token_count,
        uint32_t vocabulary, uint32_t width, float scale,
        float *output, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)token_count * width;
    if (index >= elements) return;
    uint32_t row = (uint32_t)(index / width);
    uint32_t column = (uint32_t)(index - (size_t)row * width);
    int32_t token = token_ids[row];
    if (token < 0 || (uint32_t)token >= vocabulary) {
        hip_text_fail(status);
        return;
    }
    uint64_t packed_index = (uint64_t)(uint32_t)token * width + column;
    uint64_t group = packed_index / 64u;
    uint32_t packed = hip_load_u32_aligned(
        resource + value_offset + (packed_index >> 3u) * 4u);
    uint32_t quantized =
        (packed >> ((packed_index & 7u) * 4u)) & 15u;
    float value = __fmul_rn((float)quantized,
        hip_bf16(resource + scale_offset + group * 2u));
    value = __fadd_rn(value,
        hip_bf16(resource + bias_offset + group * 2u));
    value = __fmul_rn(value, scale);
    output[index] = value;
    if (!isfinite(value)) hip_text_fail(status);
}

__global__ static void hip_text_copy(
        const float *input, float *output, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index];
}

__global__ static void hip_text_attention_transform(
        uint32_t n_heads, uint32_t n_kv_heads, uint32_t head_dim,
        uint32_t rope_dim, uint32_t rope_base_dim, float rope_theta,
        const uint32_t *source_position_pointer, const uint32_t *depths,
        uint32_t rows,
        uint32_t query_width, uint32_t kv_width, float epsilon,
        int shared_kv_projection,
        float *queries, float *keys, float *values,
        const float *q_weight, const float *k_weight,
        float *tentative_keys, float *tentative_values, int *status) {
    uint32_t tasks_per_row = n_heads + 2u * n_kv_heads;
    if (!source_position_pointer || !depths) {
        hip_text_fail(status);
        return;
    }
    uint32_t source_position = *source_position_pointer;
    uint32_t task = (uint32_t)blockIdx.x;
    uint32_t row_index = task / tasks_per_row;
    uint32_t local = task - row_index * tasks_per_row;
    uint32_t kind, head;
    float *row;
    const float *weight = NULL;
    __shared__ float inverse;
    if (row_index >= rows) return;
    if (local < n_heads) {
        kind = 0u;
        head = local;
        row = queries + (size_t)row_index * query_width +
            (size_t)head * head_dim;
        weight = q_weight;
    } else if (local < n_heads + n_kv_heads) {
        kind = 1u;
        head = local - n_heads;
        row = keys + (size_t)row_index * kv_width +
            (size_t)head * head_dim;
        weight = k_weight;
    } else {
        kind = 2u;
        head = local - n_heads - n_kv_heads;
        row = values + (size_t)row_index * kv_width +
            (size_t)head * head_dim;
        if (shared_kv_projection)
            for (uint32_t column = threadIdx.x; column < head_dim;
                 column += blockDim.x)
                row[column] = keys[(size_t)row_index * kv_width +
                    (size_t)head * head_dim + column];
    }
    __syncthreads();
    hip_text_norm_block(row, weight, row, head_dim, epsilon,
                         kind != 2u, &inverse, status);
    if (kind != 2u) {
        for (uint32_t column = threadIdx.x; column < head_dim;
             column += blockDim.x)
            row[column] = hip_round_bf16(row[column]);
        __syncthreads();
        uint32_t half = head_dim / 2u;
        uint32_t position = source_position + depths[row_index];
        for (uint32_t pair = threadIdx.x; pair < rope_dim / 2u;
             pair += blockDim.x) {
            float exponent = __fdiv_rn((float)(2u * pair),
                                       (float)rope_base_dim);
            float angle = __fdiv_rn((float)position,
                hip_text_powf(rope_theta, exponent));
            float cosine = hip_round_bf16(hip_text_cosf(angle));
            float sine = hip_round_bf16(hip_text_sinf(angle));
            float left = row[pair];
            float right = row[half + pair];
            float lc = hip_bf16_mul(left, cosine);
            float rs = hip_bf16_mul(right, sine);
            float rc = hip_bf16_mul(right, cosine);
            float ls = hip_bf16_mul(left, sine);
            row[pair] = hip_bf16_add(lc, -rs);
            row[half + pair] = hip_bf16_add(rc, ls);
            if (!isfinite(row[pair]) || !isfinite(row[half + pair]))
                hip_text_fail(status);
        }
    }
    __syncthreads();
    if (kind == 1u)
        for (uint32_t column = threadIdx.x; column < head_dim;
             column += blockDim.x)
            tentative_keys[(size_t)row_index * kv_width +
                (size_t)head * head_dim + column] = row[column];
    else if (kind == 2u)
        for (uint32_t column = threadIdx.x; column < head_dim;
             column += blockDim.x)
            tentative_values[(size_t)row_index * kv_width +
                (size_t)head * head_dim + column] = row[column];
}

__device__ static const float *hip_text_frontier_row(
        const float *tentative, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows, uint32_t current,
        uint32_t target_depth, uint32_t stride) {
    uint32_t row = current, steps = 0u;
    if (!tentative || !parents || !depths || current >= rows) return NULL;
    while (depths[row] > target_depth) {
        if (++steps > rows) return NULL;
        row = parents[row];
        if (row == UINT32_MAX || row >= rows) return NULL;
    }
    if (depths[row] != target_depth) return NULL;
    return tentative + (size_t)row * stride;
}

__device__ static const float *hip_text_committed_row(
        const float *rows, uint32_t position, uint32_t source_position,
        uint32_t width, uint32_t mode, uint32_t capacity) {
    uint32_t index;
    if (!rows || width == 0u || capacity == 0u ||
        position >= source_position)
        return NULL;
    if (mode == SALT_TEXT_KV_PRIVATE_ABSOLUTE) {
        if (position >= capacity) return NULL;
        index = position;
    } else if (mode == SALT_TEXT_KV_PRIVATE_RING) {
        uint32_t oldest = source_position > capacity
            ? source_position - capacity : 0u;
        if (position < oldest) return NULL;
        index = position % capacity;
    } else {
        return NULL;
    }
    return rows + (size_t)index * width;
}

__global__ static void hip_text_attention_body(
        int full_attention, uint32_t n_heads, uint32_t n_kv_heads,
        uint32_t head_dim, uint32_t window, float score_scale,
        const uint32_t *source_position_pointer, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t private_mode, uint32_t private_capacity,
        const float *queries, const float *committed_keys,
        const float *committed_values, const float *tentative_keys,
        const float *tentative_values, float *output, int *status) {
    extern __shared__ float scores[];
    if (!source_position_pointer || !parents || !depths) {
        hip_text_fail(status);
        return;
    }
    uint32_t source_position = *source_position_pointer;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ int score_count;
    __shared__ int score_failed;
    uint32_t task = (uint32_t)blockIdx.x;
    uint32_t row = task / n_heads;
    uint32_t head = task - row * n_heads;
    uint32_t groups = n_heads / n_kv_heads;
    uint32_t kv_head = head / groups;
    if (row >= rows) return;
    uint32_t position = source_position + depths[row];
    uint32_t first = 0;
    if (!full_attention && position + 1u > window)
        first = position + 1u - window;
    const float *query = queries + (size_t)row * query_width +
        (size_t)head * head_dim;
    uint32_t count = position - first + 1u;
    uint32_t score_workers = count < 8u ? count : 8u;
    if (threadIdx.x == 0) {
        score_failed = 0;
        score_count = (int)count;
    }
    __syncthreads();
    for (uint32_t index = threadIdx.x;
         threadIdx.x < score_workers && index < count;
         index += score_workers) {
        uint32_t key_position = first + index;
        const float *key = NULL;
        if (key_position < source_position) {
            key = hip_text_committed_row(committed_keys, key_position,
                source_position, kv_width, private_mode, private_capacity);
        } else {
            key = hip_text_frontier_row(tentative_keys, parents, depths,
                rows, row, key_position - source_position, kv_width);
        }
        if (!key) {
            atomicExch(&score_failed, 1);
            hip_text_fail(status);
            scores[index] = 0.0f;
            continue;
        }
        key += (size_t)kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t column = 0; column < head_dim; column++)
            score = __fadd_rn(score,
                __fmul_rn(query[column], key[column]));
        scores[index] = __fmul_rn(score, score_scale);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (score_failed) {
            score_count = -1;
        } else {
            maximum = -1.0f / 0.0f;
            for (uint32_t index = 0; index < count; index++)
                if (scores[index] > maximum) maximum = scores[index];
        }
    }
    __syncthreads();
    if (score_count < 0) return;
    for (uint32_t index = threadIdx.x; index < count;
         index += blockDim.x)
        scores[index] = hip_salt_expf(
            __fadd_rn(scores[index], -maximum));
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t index = 0; index < count; index++) {
            sum = __fadd_rn(sum, scores[index]);
        }
        denominator = sum;
        score_count = (!(sum > 0.0f) || !isfinite(sum)) ? -1 : (int)count;
        if (score_count < 0) hip_text_fail(status);
    }
    __syncthreads();
    if (score_count < 0) return;
    float *head_output = output + (size_t)row * query_width +
        (size_t)head * head_dim;
    for (uint32_t column = threadIdx.x; column < head_dim;
         column += blockDim.x) {
        float accumulator = 0.0f;
        uint32_t value_position = first;
        for (int index = 0; index < score_count;
             index++, value_position++) {
            const float *value = NULL;
            if (value_position < source_position) {
                value = hip_text_committed_row(committed_values,
                    value_position, source_position, kv_width,
                    private_mode, private_capacity);
            } else {
                value = hip_text_frontier_row(tentative_values, parents,
                    depths, rows, row, value_position - source_position,
                    kv_width);
            }
            if (!value) {
                hip_text_fail(status);
                return;
            }
            value += (size_t)kv_head * head_dim;
            float probability = __fdiv_rn(scores[index], denominator);
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(probability, value[column]));
        }
        head_output[column] = accumulator;
        if (!isfinite(accumulator)) hip_text_fail(status);
    }
}

__global__ static void hip_text_residual_postnorm_rows(
        const float *residual, float *branch, const float *weight,
        float *output, uint32_t rows, uint32_t width, float epsilon,
        int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    if (row >= rows) return;
    residual += (size_t)row * width;
    branch += (size_t)row * width;
    output += (size_t)row * width;
    hip_text_norm_block(branch, weight, branch, width, epsilon, 1,
                         &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fadd_rn(residual[column], branch[column]);
        output[column] = value;
        if (!isfinite(value)) hip_text_fail(status);
    }
}

__global__ static void hip_text_router_input_rows(
        const float *input, const float *weight, float *output,
        uint32_t rows, uint32_t width, float epsilon, float root,
        int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    __shared__ float effective_root;
    if (row >= rows) return;
    input += (size_t)row * width;
    output += (size_t)row * width;
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            if (!isfinite(input[column]) || !isfinite(weight[column]))
                hip_text_fail(status);
            sum = __fadd_rn(sum,
                __fmul_rn(input[column], input[column]));
        }
        inverse = hip_salt_rsqrtf(
            __fadd_rn(__fdiv_rn(sum, (float)width), epsilon));
        effective_root = root > 0.0f ? root :
            __fdiv_rn(1.0f, hip_text_sqrtf((float)width));
        if (!isfinite(inverse) || !isfinite(effective_root))
            hip_text_fail(status);
    }
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(input[column], inverse);
        value = __fmul_rn(value, weight[column]);
        value = __fmul_rn(value, effective_root);
        output[column] = value;
        if (!isfinite(value)) hip_text_fail(status);
    }
}

__global__ static void hip_text_topk_rows(
        const float *logits, const float *scales,
        int32_t *selected, float *weights, uint32_t rows,
        uint32_t experts, uint32_t topk, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    logits += (size_t)row * experts;
    selected += (size_t)row * topk;
    weights += (size_t)row * topk;
    float maximum = -1.0f / 0.0f;
    float all_sum = 0.0f, selected_sum = 0.0f;
    for (uint32_t expert = 0; expert < experts; expert++) {
        if (!isfinite(logits[expert]) || !isfinite(scales[expert]))
            hip_text_fail(status);
        if (logits[expert] > maximum) maximum = logits[expert];
    }
    for (uint32_t rank = 0; rank < topk; rank++) {
        selected[rank] = -1;
        weights[rank] = -1.0f / 0.0f;
    }
    for (uint32_t expert = 0; expert < experts; expert++) {
        int rank = (int)topk - 1;
        float score = logits[expert];
        while (rank >= 0 &&
               (selected[rank] < 0 || score > weights[rank])) rank--;
        if (rank < (int)topk - 1) {
            for (int move = (int)topk - 2; move > rank; move--) {
                selected[move + 1] = selected[move];
                weights[move + 1] = weights[move];
            }
            selected[rank + 1] = (int32_t)expert;
            weights[rank + 1] = score;
        }
    }
    for (uint32_t expert = 0; expert < experts; expert++)
        all_sum = __fadd_rn(all_sum,
            hip_salt_expf(__fadd_rn(logits[expert], -maximum)));
    if (!(all_sum > 0.0f) || !isfinite(all_sum)) hip_text_fail(status);
    for (uint32_t rank = 0; rank < topk; rank++) {
        weights[rank] = __fdiv_rn(
            hip_salt_expf(__fadd_rn(weights[rank], -maximum)), all_sum);
        selected_sum = __fadd_rn(selected_sum, weights[rank]);
    }
    if (!(selected_sum > 0.0f) || !isfinite(selected_sum))
        hip_text_fail(status);
    for (uint32_t rank = 0; rank < topk; rank++)
        weights[rank] = __fmul_rn(__fdiv_rn(weights[rank], selected_sum),
                                  scales[(uint32_t)selected[rank]]);
    for (uint32_t left = 0; left < topk; left++)
        for (uint32_t right = left + 1u; right < topk; right++)
            if (selected[right] < selected[left]) {
                int32_t id = selected[left];
                float weight = weights[left];
                selected[left] = selected[right];
                weights[left] = weights[right];
                selected[right] = id;
                weights[right] = weight;
            }
}

__global__ static void hip_text_group_maps(
        const int32_t *selected, uint32_t rows, uint32_t experts,
        uint32_t topk, int32_t *grouped_to_canonical,
        int32_t *canonical_to_grouped, int32_t *scratch, int *status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t jobs = rows * topk;
    for (uint32_t expert = 0; expert < experts; expert++) scratch[expert] = 0;
    for (uint32_t canonical = 0; canonical < jobs; canonical++) {
        int32_t expert = selected[canonical];
        if (expert < 0 || (uint32_t)expert >= experts ||
            scratch[expert] == INT32_MAX) {
            hip_text_fail(status);
            return;
        }
        scratch[expert]++;
    }
    uint32_t next = 0;
    for (uint32_t expert = 0; expert < experts; expert++) {
        uint32_t population = (uint32_t)scratch[expert];
        scratch[expert] = (int32_t)next;
        next += population;
    }
    if (next != jobs) {
        hip_text_fail(status);
        return;
    }
    for (uint32_t canonical = 0; canonical < jobs; canonical++) {
        int32_t expert = selected[canonical];
        uint32_t grouped = (uint32_t)scratch[expert]++;
        if (grouped >= jobs) {
            hip_text_fail(status);
            return;
        }
        grouped_to_canonical[grouped] = (int32_t)canonical;
    }
    for (uint32_t grouped = 0; grouped < jobs; grouped++) {
        uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
        if (canonical >= jobs) {
            hip_text_fail(status);
            return;
        }
        canonical_to_grouped[canonical] = (int32_t)grouped;
    }
}

__global__ static void hip_text_routed_gather(
        const int32_t *grouped_to_canonical, const float *rows_input,
        float *routed_input, uint32_t jobs, uint32_t topk,
        uint32_t hidden, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t elements = (size_t)jobs * hidden;
    if (index >= elements) return;
    uint32_t grouped = (uint32_t)(index / hidden);
    uint32_t column = (uint32_t)(index - (size_t)grouped * hidden);
    uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
    if (canonical >= jobs) {
        hip_text_fail(status);
        return;
    }
    uint32_t row = canonical / topk;
    routed_input[index] = rows_input[(size_t)row * hidden + column];
}

__global__ static void hip_text_activate(
        const float *gate, const float *up, float *output,
        size_t elements, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = __fmul_rn(hip_gemma4_gelu_tanh(gate[index]), up[index]);
    output[index] = value;
    if (!isfinite(value)) hip_text_fail(status);
}

__global__ static void hip_text_expert_q4(
        const HipTextTensorRef *experts, uint32_t layer, uint32_t phase,
        const int32_t *selected, const int32_t *grouped_to_canonical,
        uint32_t jobs, uint32_t topk, uint32_t experts_per_layer,
        const float *input,
        float *output, float *paired_output,
        uint32_t expected_rows, uint32_t expected_cols,
        int *status) {
    if (blockIdx.z != 0u) {
        phase += (uint32_t)blockIdx.z;
        output = paired_output;
    }
    uint32_t grouped = (uint32_t)blockIdx.y;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    uint32_t row = (uint32_t)blockIdx.x * 4u + (uint32_t)warp;
    HipTextTensorRef ref = {0};
    int valid = grouped < jobs;
    if (valid) {
        uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
        if (canonical >= jobs) valid = 0;
        else {
            int32_t expert = selected[canonical];
            if (expert < 0 || (uint32_t)expert >= experts_per_layer)
                valid = 0;
            else
                ref = experts[((size_t)layer * experts_per_layer +
                    (uint32_t)expert) * HIP_TEXT_EXPERT_PROJECTIONS + phase];
        }
    }
    if (!valid || ref.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        ref.rows != expected_rows || ref.cols != expected_cols ||
        !ref.resource) {
        if (lane == 0 && warp == 0) hip_text_fail(status);
        return;
    }
    float accumulator = 0.0f;
    if (row < ref.rows) {
        const float *row_input = input + (size_t)grouped * ref.cols;
        for (uint32_t column = (uint32_t)lane; column < ref.cols;
             column += 32u) {
            uint64_t k = (uint64_t)row * ref.cols + column;
            uint64_t group = k / 64u;
            uint32_t packed = hip_load_u32_aligned(
                ref.resource + ref.value_offset + (k >> 3u) * 4u);
            uint32_t quantized = (packed >> ((k & 7u) * 4u)) & 15u;
            float value = __fmul_rn((float)quantized,
                hip_bf16(ref.resource + ref.scale_offset + group * 2u));
            value = __fadd_rn(value,
                hip_bf16(ref.resource + ref.bias_offset + group * 2u));
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(value, row_input[column]));
        }
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    float lane_sum = 0.0f;
    for (int item = 0; item < 8; item++)
        lane_sum = __fadd_rn(lane_sum,
            __shfl_sync(mask, accumulator, base + 4 * item));
    float lane0 = __shfl_sync(mask, lane_sum, 0);
    float lane1 = __shfl_sync(mask, lane_sum, 1);
    float lane2 = __shfl_sync(mask, lane_sum, 2);
    float lane3 = __shfl_sync(mask, lane_sum, 3);
    float even = __fadd_rn(lane0, lane2);
    float odd = __fadd_rn(lane1, lane3);
    if (lane == 0 && row < ref.rows)
        output[(size_t)grouped * ref.rows + row] = __fadd_rn(even, odd);
}

__global__ static void hip_text_expert_reduce(
        const float *weights, const int32_t *canonical_to_grouped,
        const float *job_outputs, float *output, uint32_t rows,
        uint32_t topk, uint32_t hidden, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    if (row >= rows) return;
    for (uint32_t column = threadIdx.x; column < hidden;
         column += blockDim.x) {
        float accumulator = 0.0f;
        for (uint32_t rank = 0; rank < topk; rank++) {
            uint32_t canonical = row * topk + rank;
            uint32_t grouped = (uint32_t)canonical_to_grouped[canonical];
            if (grouped >= rows * topk) {
                hip_text_fail(status);
                return;
            }
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(weights[canonical],
                    job_outputs[(size_t)grouped * hidden + column]));
        }
        output[(size_t)row * hidden + column] = accumulator;
        if (!isfinite(accumulator)) hip_text_fail(status);
    }
}

__global__ static void hip_text_parallel_combine_rows(
        const float *residual, const float *dense, const float *routed,
        const float *dense_weight, const float *routed_weight,
        const float *final_weight, float *scratch_a, float *scratch_b,
        float *output, uint32_t rows, uint32_t width, float epsilon,
        float scalar, const float *scalar_pointer, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    __shared__ float effective_scalar;
    if (row >= rows) return;
    if (threadIdx.x == 0) {
        effective_scalar = scalar_pointer ? scalar_pointer[0] : scalar;
        if (!isfinite(effective_scalar)) hip_text_fail(status);
    }
    __syncthreads();
    residual += (size_t)row * width;
    dense += (size_t)row * width;
    routed += (size_t)row * width;
    scratch_a += (size_t)row * width;
    scratch_b += (size_t)row * width;
    output += (size_t)row * width;
    hip_text_norm_block(dense, dense_weight, scratch_a, width, epsilon, 1,
                         &inverse, status);
    hip_text_norm_block(routed, routed_weight, scratch_b, width, epsilon, 1,
                         &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        scratch_a[column] = __fadd_rn(scratch_a[column], scratch_b[column]);
        if (!isfinite(scratch_a[column])) hip_text_fail(status);
    }
    __syncthreads();
    hip_text_norm_block(scratch_a, final_weight, scratch_b,
                         width, epsilon, 1, &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(
            __fadd_rn(residual[column], scratch_b[column]), effective_scalar);
        output[column] = value;
        if (!isfinite(value)) hip_text_fail(status);
    }
}

__global__ static void hip_text_softcap(
        float *values, size_t elements, float cap, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = values[index];
    if (!isfinite(value)) {
        hip_text_fail(status);
        return;
    }
    values[index] = __fmul_rn(cap,
        hip_salt_tanhf(__fdiv_rn(value, cap)));
}

__global__ static void hip_moe_activate(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *gate_up, float *chains) {
    int descriptor_index = (int)blockIdx.z;
    if (descriptor_index >= descriptor_count) return;
    HipBatchDesc descriptor = descriptors[descriptor_index];
    size_t elements = (size_t)descriptor.batch * (size_t)descriptor.rows;
    size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float gate = gate_up[descriptor.input_offset + index];
    float up = gate_up[descriptor.output_offset + index];
    chains[descriptor.aux_offset + index] =
        __fmul_rn(hip_gemma4_gelu_tanh(gate), up);
}

__global__ static void hip_moe_activate_contiguous(
    const float *gate, const float *up, float *chain, size_t elements) {
    size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index < elements)
        chain[index] = __fmul_rn(hip_gemma4_gelu_tanh(gate[index]), up[index]);
}

__global__ static void hip_q8_shared_tile(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C, int B) {
    __shared__ float dequant[32];
    int r = (int)blockIdx.x;
    int token = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    float acc = 0.0f;
    if (r >= R) return;
    for (int cb = 0; cb < C; cb += 32) {
        int lane = (int)threadIdx.x;
        if (lane < 32) {
            int c = cb + lane;
            if (c < C) {
                uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C +
                             (uint32_t)c;
                uint64_t g = k / 64u;
                uint32_t q = resource[voff + k];
                float scale = hip_bf16(resource + soff + g * 2u);
                float bias = hip_bf16(resource + boff + g * 2u);
                float t = __fmul_rn((float)q, scale);
                dequant[lane] = __fadd_rn(t, bias);
            } else {
                dequant[lane] = 0.0f;
            }
        }
        __syncthreads();
        if (token < B) {
            int count = C - cb;
            if (count > 32) count = 32;
            const float *input = x + (size_t)token * (size_t)C + cb;
            for (int i = 0; i < count; i++) {
                float product = __fmul_rn(dequant[i], input[i]);
                acc = __fadd_rn(acc, product);
            }
        }
        __syncthreads();
    }
    if (token < B)
        y[(size_t)token * (size_t)R + (size_t)r] = acc;
}

__global__ static void hip_q4_heterogeneous(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y,
    unsigned char *const *selected_resources,
    const int32_t *logical_slots, const uint32_t *payload_offsets) {
    __shared__ float dequant[32];
    int descriptor_index = (int)blockIdx.z;
    int r = (int)blockIdx.x;
    int token = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (descriptor_index >= descriptor_count) return;
    HipBatchDesc descriptor = descriptors[descriptor_index];
    const unsigned char *resource = descriptor.resource;
    uint64_t payload = 0;
    if (descriptor.selected == 1u && !descriptor.resource) {
        int32_t slot = logical_slots[descriptor.logical_resource_id];
        if (slot < 0 || slot != descriptor.resource_slot ||
            !selected_resources[slot])
            return;
        resource = selected_resources[slot];
        payload = payload_offsets[descriptor.logical_resource_id];
    }
    if (r >= descriptor.rows ||
        (int)blockIdx.y * (int)blockDim.x >= descriptor.batch)
        return;
    float acc[32];
#pragma unroll
    for (int lane = 0; lane < 32; lane++) acc[lane] = 0.0f;
    for (int cb = 0; cb < descriptor.cols; cb += 32) {
        int lane = (int)threadIdx.x;
        if (lane < 32) {
            int c = cb + lane;
            if (c < descriptor.cols) {
                uint64_t k = (uint64_t)(uint32_t)r *
                             (uint32_t)descriptor.cols + (uint32_t)c;
                uint64_t g = k / 64u;
                uint32_t packed = hip_load_u32(resource + payload +
                    descriptor.value_offset + (k >> 3u) * 4u);
                uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
                float scale = hip_bf16(resource + payload +
                    descriptor.scale_offset + g * 2u);
                float bias = hip_bf16(resource + payload +
                    descriptor.bias_offset + g * 2u);
                float t = __fmul_rn((float)q, scale);
                dequant[lane] = __fadd_rn(t, bias);
            } else {
                dequant[lane] = 0.0f;
            }
        }
        __syncthreads();
        if (token < descriptor.batch) {
            int count = descriptor.cols - cb;
            if (count > 32) count = 32;
            const float *input = x + descriptor.input_offset +
                (size_t)token * (size_t)descriptor.cols + cb;
#pragma unroll
            for (int i = 0; i < 32; i++)
                if (i < count) {
                    float product = __fmul_rn(dequant[i], input[i]);
                    acc[i] = __fadd_rn(acc[i], product);
                }
        }
        __syncthreads();
    }
    if (token < descriptor.batch) {
        float lane0 = 0.0f, lane1 = 0.0f;
        float lane2 = 0.0f, lane3 = 0.0f;
#pragma unroll
        for (int a = 0; a < 8; a++) {
            lane0 = __fadd_rn(lane0, acc[4 * a]);
            lane1 = __fadd_rn(lane1, acc[4 * a + 1]);
            lane2 = __fadd_rn(lane2, acc[4 * a + 2]);
            lane3 = __fadd_rn(lane3, acc[4 * a + 3]);
        }
        float t0 = __fadd_rn(lane0, lane2);
        float t1 = __fadd_rn(lane1, lane3);
        y[descriptor.output_offset + (size_t)token *
            (size_t)descriptor.rows + (size_t)r] = __fadd_rn(t0, t1);
    }
}

__global__ static void hip_q4_heterogeneous_warp_batch(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y) {
    __shared__ float dequant[1024];
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x;
    int token_base = (int)blockIdx.y * 512;
    HipBatchDesc descriptor = {0};
    float acc[16];
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    if (descriptor_index >= descriptor_count || r >= descriptor.rows ||
        token_base >= descriptor.batch)
        return;
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int tile = 0; tile < descriptor.cols; tile += 1024) {
        int loaded_column = tile + (int)threadIdx.x;
        if (loaded_column < descriptor.cols) {
            uint64_t k = (uint64_t)(uint32_t)r *
                         (uint32_t)descriptor.cols + (uint32_t)loaded_column;
            uint64_t g = k / 64u;
            uint32_t packed = hip_load_u32(descriptor.resource +
                descriptor.value_offset + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = hip_bf16(descriptor.resource +
                descriptor.scale_offset + g * 2u);
            float bias = hip_bf16(descriptor.resource +
                descriptor.bias_offset + g * 2u);
            float value = __fmul_rn((float)q, scale);
            dequant[threadIdx.x] = __fadd_rn(value, bias);
        } else {
            dequant[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        int index = 0;
        for (int token = token_base + warp;
             token < descriptor.batch && token < token_base + 512;
             token += 32, index++) {
            for (int offset = 0; offset < 1024; offset += 32) {
                int c = tile + offset + lane;
                if (c < descriptor.cols) {
                    float product = __fmul_rn(
                        dequant[offset + lane],
                        x[descriptor.input_offset + (size_t)token *
                            (size_t)descriptor.cols + (size_t)c]);
                    acc[index] = __fadd_rn(acc[index], product);
                }
            }
        }
        __syncthreads();
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    int index = 0;
    for (int token = token_base + warp;
         token < descriptor.batch && token < token_base + 512;
         token += 32, index++) {
        float lane_sum = 0.0f;
        for (int a = 0; a < 8; a++)
            lane_sum = __fadd_rn(
                lane_sum, __shfl_sync(mask, acc[index], base + 4 * a));
        float lane0 = __shfl_sync(mask, lane_sum, 0);
        float lane1 = __shfl_sync(mask, lane_sum, 1);
        float lane2 = __shfl_sync(mask, lane_sum, 2);
        float lane3 = __shfl_sync(mask, lane_sum, 3);
        float t0 = __fadd_rn(lane0, lane2);
        float t1 = __fadd_rn(lane1, lane3);
        if (lane == 0)
            y[descriptor.output_offset + (size_t)token *
                (size_t)descriptor.rows + (size_t)r] = __fadd_rn(t0, t1);
    }
}

__global__ static void hip_q4_selected_warp_batch(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y,
    unsigned char *const *selected_resources,
    const int32_t *logical_slots, const uint32_t *payload_offsets) {
    __shared__ float dequant[256];
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int warps = (int)blockDim.x >> 5;
    int r = (int)blockIdx.x;
    int token_span = warps * 16;
    int token_base = (int)blockIdx.y * token_span;
    HipBatchDesc descriptor = {0};
    float acc[16];
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    if (warps < 1 || warps > 8 || descriptor_index >= descriptor_count ||
        descriptor.selected != 1u || descriptor.resource ||
        descriptor.resource_slot < 0 || r >= descriptor.rows ||
        token_base >= descriptor.batch)
        return;
    int32_t slot = logical_slots[descriptor.logical_resource_id];
    if (slot != descriptor.resource_slot || !selected_resources[slot]) return;
    const unsigned char *resource = selected_resources[slot];
    uint64_t payload = payload_offsets[descriptor.logical_resource_id];
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int tile = 0; tile < descriptor.cols; tile += (int)blockDim.x) {
        int loaded_column = tile + (int)threadIdx.x;
        if (loaded_column < descriptor.cols) {
            uint64_t k = (uint64_t)(uint32_t)r *
                (uint32_t)descriptor.cols + (uint32_t)loaded_column;
            uint64_t g = k / 64u;
            uint32_t packed = hip_load_u32(resource + payload +
                descriptor.value_offset + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = hip_bf16(resource + payload +
                descriptor.scale_offset + g * 2u);
            float bias = hip_bf16(resource + payload +
                descriptor.bias_offset + g * 2u);
            float value = __fmul_rn((float)q, scale);
            dequant[threadIdx.x] = __fadd_rn(value, bias);
        } else {
            dequant[threadIdx.x] = 0.0f;
        }
        __syncthreads();
        int index = 0;
        for (int token = token_base + warp;
             token < descriptor.batch && token < token_base + token_span;
             token += warps, index++) {
            for (int offset = 0; offset < (int)blockDim.x; offset += 32) {
                int c = tile + offset + lane;
                if (c < descriptor.cols) {
                    float product = __fmul_rn(dequant[offset + lane],
                        x[descriptor.input_offset + (size_t)token *
                            (size_t)descriptor.cols + (size_t)c]);
                    acc[index] = __fadd_rn(acc[index], product);
                }
            }
        }
        __syncthreads();
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    int index = 0;
    for (int token = token_base + warp;
         token < descriptor.batch && token < token_base + token_span;
         token += warps, index++) {
        float lane_sum = 0.0f;
        for (int a = 0; a < 8; a++)
            lane_sum = __fadd_rn(
                lane_sum, __shfl_sync(mask, acc[index], base + 4 * a));
        float lane0 = __shfl_sync(mask, lane_sum, 0);
        float lane1 = __shfl_sync(mask, lane_sum, 1);
        float lane2 = __shfl_sync(mask, lane_sum, 2);
        float lane3 = __shfl_sync(mask, lane_sum, 3);
        float t0 = __fadd_rn(lane0, lane2);
        float t1 = __fadd_rn(lane1, lane3);
        if (lane == 0)
            y[descriptor.output_offset + (size_t)token *
                (size_t)descriptor.rows + (size_t)r] = __fadd_rn(t0, t1);
    }
}

__global__ static void hip_q4_heterogeneous_warp(
    const HipBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y,
    unsigned char *const *selected_resources,
    const int32_t *logical_slots, const uint32_t *payload_offsets) {
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x * 4 + warp;
    float acc = 0.0f;
    HipBatchDesc descriptor = {0};
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    const unsigned char *resource = descriptor.resource;
    uint64_t payload = 0;
    if (descriptor.selected == 1u && !descriptor.resource) {
        int32_t slot = logical_slots[descriptor.logical_resource_id];
        if (slot < 0 || slot != descriptor.resource_slot ||
            !selected_resources[slot])
            return;
        resource = selected_resources[slot];
        payload = payload_offsets[descriptor.logical_resource_id];
    }
    if (descriptor_index < descriptor_count && descriptor.batch == 1 &&
        r < descriptor.rows) {
        const float *input = x + descriptor.input_offset;
        for (int c = lane; c < descriptor.cols; c += 32) {
            uint64_t k = (uint64_t)(uint32_t)r *
                         (uint32_t)descriptor.cols + (uint32_t)c;
            uint64_t g = k / 64u;
            uint32_t packed = hip_load_u32_aligned(resource + payload +
                descriptor.value_offset + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = hip_bf16(resource + payload +
                descriptor.scale_offset + g * 2u);
            float bias = hip_bf16(resource + payload +
                descriptor.bias_offset + g * 2u);
            float dequant = __fmul_rn((float)q, scale);
            dequant = __fadd_rn(dequant, bias);
            float product = __fmul_rn(dequant, input[c]);
            acc = __fadd_rn(acc, product);
        }
    }
    unsigned int mask = 0xffffffffu;
    int base = lane & 3;
    float lane_sum = 0.0f;
    for (int a = 0; a < 8; a++)
        lane_sum = __fadd_rn(
            lane_sum, __shfl_sync(mask, acc, base + 4 * a));
    float lane0 = __shfl_sync(mask, lane_sum, 0);
    float lane1 = __shfl_sync(mask, lane_sum, 1);
    float lane2 = __shfl_sync(mask, lane_sum, 2);
    float lane3 = __shfl_sync(mask, lane_sum, 3);
    float t0 = __fadd_rn(lane0, lane2);
    float t1 = __fadd_rn(lane1, lane3);
    if (lane == 0 && descriptor_index < descriptor_count &&
        descriptor.batch == 1 && r < descriptor.rows)
        y[descriptor.output_offset + (size_t)r] = __fadd_rn(t0, t1);
}

static int hip_selected_warp_geometry(
        int batch, unsigned int *threads, unsigned int *groups) {
    unsigned int warps, span;
    if (batch < 2 || !threads || !groups) return -1;
    warps = (unsigned int)batch;
    if (warps > hip_text_wave_groups) warps = hip_text_wave_groups;
    span = warps * 16u;
    *threads = warps * 32u;
    *groups = ((unsigned int)batch + span - 1u) / span;
    return *groups > 0u ? 0 : -1;
}

static int hip_bf16_warp_geometry(
        uint32_t batch, unsigned int *threads, unsigned int *groups) {
    unsigned int warps, span;
    if (batch < 2u || !threads || !groups) return -1;
    warps = batch;
    if (warps > hip_text_bf16_wave_groups)
        warps = hip_text_bf16_wave_groups;
    span = warps * 8u;
    *threads = warps * 32u;
    *groups = (batch + span - 1u) / span;
    return *groups > 0u ? 0 : -1;
}

static HipResource *hip_resource_record(uint32_t kind, uint32_t id) {
    for (int i = 0; i < HIP_MAX_RESOURCES; i++)
        if (hip_resources[i].used && hip_resources[i].kind == kind &&
            hip_resources[i].id == id)
            return &hip_resources[i];
    return NULL;
}

static HipResource *hip_resource(uint32_t kind, uint32_t id) {
    HipResource *resource = hip_resource_record(kind, id);
    return resource && resource->active && resource->device ? resource : NULL;
}

static HipTensor *hip_tensor(const void *key) {
    for (int i = 0; i < hip_tensor_count; i++)
        if (hip_tensors[i].used && hip_tensors[i].key == key)
            return &hip_tensors[i];
    return NULL;
}

static int hip_tensor_physical(
        HipTensor *tensor, HipResource **resource_out,
        unsigned char **device_out, uint64_t *value_offset,
        uint64_t *scale_offset, uint64_t *bias_offset,
        uint64_t *aux_offset) {
    HipResource *resource;
    if (!tensor || !resource_out || !device_out || !value_offset ||
        !scale_offset || !bias_offset || !aux_offset ||
        !(resource = hip_resource_record(tensor->kind, tensor->resource_id)) ||
        !resource->active)
        return -1;
    if (tensor->resident) {
        if (!hip_residency.permanent_published ||
            !tensor->resident_device)
            return -1;
        *device_out = tensor->resident_device;
        *value_offset = tensor->resident_value_offset;
        *scale_offset = tensor->resident_scale_offset;
        *bias_offset = tensor->resident_bias_offset;
        *aux_offset = tensor->resident_aux_offset;
    } else {
        if (!resource->device) return -1;
        *device_out = resource->device;
        *value_offset = tensor->value_offset;
        *scale_offset = tensor->scale_offset;
        *bias_offset = tensor->bias_offset;
        *aux_offset = tensor->aux_offset;
    }
    *resource_out = resource;
    return 0;
}

static int hip_range(const HipResource *resource, const void *pointer,
                      size_t nbytes, uint64_t *offset) {
    uintptr_t base, value;
    if (!resource || !pointer || !offset) return -1;
    base = (uintptr_t)resource->host;
    value = (uintptr_t)pointer;
    if (value < base || value - base > resource->nbytes ||
        nbytes > resource->nbytes - (value - base))
        return -1;
    *offset = (uint64_t)(value - base);
    return 0;
}

static int hip_tensor_sizes(int bits, int R, int C,
                             size_t *vbytes, size_t *sbytes) {
    uint64_t elements, groups, values;
    if ((bits != 4 && bits != 8) || R < 1 || C < 1 || !vbytes || !sbytes)
        return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    groups = (elements + 63u) / 64u;
    values = bits == 4 ? ((elements + 7u) / 8u) * 4u : elements;
    if (values > SIZE_MAX || groups > SIZE_MAX / 2u) return -1;
    *vbytes = (size_t)values;
    *sbytes = (size_t)groups * 2u;
    return 0;
}

static int hip_component_region_add(HipResource *resource,
        uint64_t offset, size_t nbytes) {
    long page_l;
    uint64_t page, aligned, end, rounded_end, rounded_resource;
    uintptr_t start_address, end_address;
    int resource_index;
    if (!resource || !resource->active || !resource->component_pool ||
        hip_component_pool_finalized || nbytes < 1 ||
        offset > resource->nbytes || nbytes > resource->nbytes - (size_t)offset ||
        (page_l = sysconf(_SC_PAGESIZE)) <= 0)
        return -1;
    page = (uint64_t)page_l;
    aligned = offset - offset % page;
    end = offset + (uint64_t)nbytes;
    if (end > UINT64_MAX - (page - 1u) ||
        resource->nbytes > SIZE_MAX - (size_t)(page - 1u))
        return -1;
    rounded_end = (end + page - 1u) / page * page;
    rounded_resource = ((uint64_t)resource->nbytes + page - 1u) / page * page;
    if (rounded_end > rounded_resource ||
        (uintptr_t)resource->host > UINTPTR_MAX - rounded_end)
        return -1;
    start_address = (uintptr_t)(const void *)resource->host + aligned;
    end_address = (uintptr_t)(const void *)resource->host + rounded_end;
    resource_index = (int)(resource - hip_resources);
    for (int scan = 0; scan < hip_component_region_count;) {
        HipComponentRegion *existing = &hip_component_regions[scan];
        uintptr_t existing_start = (uintptr_t)(const void *)existing->host;
        uintptr_t existing_end = existing_start + existing->nbytes;
        if (existing->resource_index != resource_index ||
            end_address < existing_start || existing_end < start_address) {
            scan++;
            continue;
        }
        if (existing_start < start_address) start_address = existing_start;
        if (existing_end > end_address) end_address = existing_end;
        hip_component_regions[scan] =
            hip_component_regions[--hip_component_region_count];
        memset(&hip_component_regions[hip_component_region_count], 0,
            sizeof hip_component_regions[hip_component_region_count]);
        scan = 0;
    }
    if (hip_component_region_count >= HIP_MAX_COMPONENT_REGIONS ||
        end_address <= start_address || end_address - start_address > SIZE_MAX)
        return -1;
    HipComponentRegion *region =
        &hip_component_regions[hip_component_region_count++];
    region->resource_index = resource_index;
    region->host = (const unsigned char *)(const void *)start_address;
    region->nbytes = (size_t)(end_address - start_address);
    return 0;
}

static int hip_component_region_compare(const void *left, const void *right) {
    const HipComponentRegion *a = (const HipComponentRegion *)left;
    const HipComponentRegion *b = (const HipComponentRegion *)right;
    if (a->resource_index != b->resource_index)
        return a->resource_index < b->resource_index ? -1 : 1;
    if (a->source_offset != b->source_offset)
        return a->source_offset < b->source_offset ? -1 : 1;
    return 0;
}

static int hip_component_pool_build_regions(void) {
    if (hip_component_pool_finalized) return -1;
    if (hip_component_region_count != 0) return 0;
    for (int tensor_index = 0; tensor_index < hip_tensor_count; tensor_index++) {
        HipTensor *tensor = &hip_tensors[tensor_index];
        HipResource *resource;
        size_t vbytes = 0, sbytes = 0;
        uint64_t elements;
        if (!tensor->used) continue;
        resource = hip_resource_record(tensor->kind, tensor->resource_id);
        if (!resource || !resource->component_pool) continue;
        elements = (uint64_t)(uint32_t)tensor->rows *
            (uint64_t)(uint32_t)tensor->cols;
        int region_failed = 0;
        if (tensor->bits == 4 || tensor->bits == 8) {
            region_failed = hip_tensor_sizes(
                    tensor->bits, tensor->rows, tensor->cols,
                    &vbytes, &sbytes) != 0 ||
                hip_component_region_add(resource,
                    tensor->value_offset, vbytes) != 0 ||
                hip_component_region_add(resource,
                    tensor->scale_offset, sbytes) != 0 ||
                hip_component_region_add(resource,
                    tensor->bias_offset, sbytes) != 0;
        } else if (tensor->bits == 16) {
            region_failed = elements > SIZE_MAX / sizeof(uint16_t) ||
                hip_component_region_add(resource, tensor->value_offset,
                    (size_t)elements * sizeof(uint16_t)) != 0;
        } else if (tensor->bits == HIP_FORMAT_NVFP4) {
            region_failed = (tensor->cols & 15) != 0 ||
                elements > SIZE_MAX ||
                hip_component_region_add(resource, tensor->value_offset,
                    (size_t)(elements / 2u)) != 0 ||
                hip_component_region_add(resource, tensor->scale_offset,
                    (size_t)(elements / 16u)) != 0 ||
                hip_component_region_add(resource, tensor->bias_offset,
                    sizeof(float)) != 0 ||
                hip_component_region_add(resource, tensor->aux_offset,
                    sizeof(float)) != 0;
        } else {
            region_failed = 1;
        }
        if (region_failed) {
            memset(hip_component_regions, 0, sizeof hip_component_regions);
            hip_component_region_count = 0;
            return -1;
        }
    }
    for (int index = 0; index < hip_component_region_count; index++) {
        HipComponentRegion *region = &hip_component_regions[index];
        HipResource *resource;
        if (region->resource_index < 0 ||
            region->resource_index >= HIP_MAX_RESOURCES ||
            !(resource = &hip_resources[region->resource_index]) ||
            !resource->used || !region->host ||
            region->host < resource->host ||
            (uintptr_t)region->host - (uintptr_t)resource->host >
                resource->nbytes)
            return -1;
        region->source_offset = (uint64_t)(
            (uintptr_t)region->host - (uintptr_t)resource->host);
    }
    qsort(hip_component_regions, (size_t)hip_component_region_count,
        sizeof hip_component_regions[0], hip_component_region_compare);
    return hip_component_region_count > 0 ? 0 : -1;
}

static int hip_component_physical_offset(
        int resource_index, uint64_t source_offset, uint64_t nbytes,
        uint64_t *device_offset) {
    if (!device_offset || nbytes == 0 ||
        source_offset > UINT64_MAX - nbytes)
        return -1;
    for (int index = 0; index < hip_component_region_count; index++) {
        const HipComponentRegion *region = &hip_component_regions[index];
        if (region->resource_index != resource_index ||
            source_offset < region->source_offset ||
            source_offset - region->source_offset > region->nbytes ||
            nbytes > region->nbytes - (source_offset - region->source_offset))
            continue;
        *device_offset = region->residency_offset +
            (source_offset - region->source_offset);
        return 0;
    }
    return -1;
}

static int hip_tensor_bind_resident(HipTensor *tensor) {
    HipResource *resource;
    int resource_index;
    uint64_t elements;
    size_t vbytes = 0, sbytes = 0;
    if (!tensor || !tensor->used ||
        !(resource = hip_resource_record(tensor->kind, tensor->resource_id)))
        return -1;
    resource_index = (int)(resource - hip_resources);
    elements = (uint64_t)(uint32_t)tensor->rows *
        (uint64_t)(uint32_t)tensor->cols;
    tensor->resident_device = hip_residency.device_arena;
    if (tensor->bits == 4 || tensor->bits == 8) {
        if (hip_tensor_sizes(tensor->bits, tensor->rows, tensor->cols,
                &vbytes, &sbytes) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->value_offset, vbytes,
                &tensor->resident_value_offset) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->scale_offset, sbytes,
                &tensor->resident_scale_offset) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->bias_offset, sbytes,
                &tensor->resident_bias_offset) != 0)
            return -1;
    } else if (tensor->bits == 16) {
        if (elements > SIZE_MAX / sizeof(uint16_t) ||
            hip_component_physical_offset(resource_index,
                tensor->value_offset, elements * sizeof(uint16_t),
                &tensor->resident_value_offset) != 0)
            return -1;
    } else if (tensor->bits == HIP_FORMAT_NVFP4) {
        if ((tensor->cols & 15) != 0 || elements > SIZE_MAX ||
            hip_component_physical_offset(resource_index,
                tensor->value_offset, elements / 2u,
                &tensor->resident_value_offset) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->scale_offset, elements / 16u,
                &tensor->resident_scale_offset) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->bias_offset, sizeof(float),
                &tensor->resident_bias_offset) != 0 ||
            hip_component_physical_offset(resource_index,
                tensor->aux_offset, sizeof(float),
                &tensor->resident_aux_offset) != 0)
            return -1;
    } else {
        return -1;
    }
    tensor->resident = 1;
    return 0;
}

static int hip_component_source_read(
        void *context, uint64_t source_offset,
        void *destination, size_t nbytes) {
    HipComponentRegion *region = (HipComponentRegion *)context;
    uint64_t local;
    if (!region || !destination || nbytes == 0 || !region->host ||
        source_offset < region->source_offset ||
        (local = source_offset - region->source_offset) > region->nbytes ||
        nbytes > region->nbytes - local)
        return -1;
    memcpy(destination, region->host + (size_t)local, nbytes);
    return 0;
}

static int hip_residency_permanent_layout(
        void *context, SaltGpuResidencyPermanentLayout *layout,
        size_t layout_size) {
    const uint64_t alignment = UINT64_C(256);
    uint64_t cursor = 0, maximum = 0;
    (void)context;
    if (!hip_ready || !layout || layout_size != sizeof *layout ||
        hip_component_pool_finalized ||
        hip_component_pool_build_regions() != 0)
        return -1;
    if (hip_residency.permanent_span_count != 0) {
        layout->permanent_bytes = hip_residency.permanent_layout_bytes;
        layout->maximum_span_bytes = hip_residency.permanent_max_span_bytes;
        layout->span_count = hip_residency.permanent_span_count;
        layout->alignment = (uint32_t)alignment;
        return 0;
    }
    for (int index = 0; index < hip_component_region_count; index++) {
        HipComponentRegion *region = &hip_component_regions[index];
        if (cursor > UINT64_MAX - (alignment - 1u)) return -1;
        cursor = (cursor + alignment - 1u) / alignment * alignment;
        if ((uint64_t)region->nbytes > UINT64_MAX - cursor) return -1;
        region->residency_offset = cursor;
        region->populated_bytes = 0;
        region->resident = 0;
        cursor += (uint64_t)region->nbytes;
        if ((uint64_t)region->nbytes > maximum) maximum = region->nbytes;
    }
    if (cursor == 0 || maximum == 0 || hip_component_region_count < 1)
        return -1;
    hip_residency.permanent_layout_bytes = cursor;
    hip_residency.permanent_max_span_bytes = maximum;
    hip_residency.permanent_span_count = (uint32_t)hip_component_region_count;
    layout->permanent_bytes = cursor;
    layout->maximum_span_bytes = maximum;
    layout->span_count = (uint32_t)hip_component_region_count;
    layout->alignment = (uint32_t)alignment;
    return 0;
}

static int hip_residency_permanent_span(
        void *context, uint32_t span_index,
        SaltGpuResidencyPermanentSpan *span, size_t span_size) {
    HipComponentRegion *region;
    HipResource *resource;
    (void)context;
    if (!span || span_size != sizeof *span ||
        span_index >= hip_residency.permanent_span_count ||
        span_index >= (uint32_t)hip_component_region_count)
        return -1;
    region = &hip_component_regions[span_index];
    if (region->resource_index < 0 ||
        region->resource_index >= HIP_MAX_RESOURCES ||
        !(resource = &hip_resources[region->resource_index]) ||
        !resource->used || !resource->active || !resource->component_pool)
        return -1;
    memset(span, 0, sizeof *span);
    span->source.resource.kind = resource->kind;
    span->source.resource.resource_id = resource->id;
    span->source.source_offset = region->source_offset;
    span->source.nbytes = region->nbytes;
    span->source.read = hip_component_source_read;
    span->source.read_context = region;
    span->device_offset = region->residency_offset;
    return 0;
}

static int hip_residency_permanent_copy(
        void *context, uint32_t span_index, uint64_t device_offset,
        uint32_t staging_slot, size_t nbytes, uint64_t ticket) {
    HipComponentRegion *region;
    uint64_t expected;
    hipError_t error;
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned || ticket == 0 ||
        nbytes == 0 || span_index >= hip_residency.permanent_span_count ||
        span_index >= (uint32_t)hip_component_region_count ||
        staging_slot >= hip_residency.plan.staging_count)
        return -1;
    region = &hip_component_regions[span_index];
    expected = region->residency_offset + region->populated_bytes;
    if (device_offset != expected || region->populated_bytes > region->nbytes ||
        nbytes > region->nbytes - region->populated_bytes ||
        device_offset > hip_residency.plan.permanent_bytes ||
        nbytes > hip_residency.plan.permanent_bytes - device_offset)
        return -1;
    error = hipMemcpyAsync(
        hip_residency.device_arena + (size_t)device_offset,
        hip_residency.staging_arena +
            (size_t)staging_slot * (size_t)hip_residency.plan.staging_bytes,
        nbytes, hipMemcpyHostToDevice, hip_residency.stream);
    if (error != hipSuccess)
        return hip_success(error, "hipMemcpyAsync permanent residency");
    if (hip_success(hipEventRecord(
            hip_residency.permanent_event, hip_residency.stream),
            "hipEventRecord permanent residency") != 0 ||
        hip_success(hipEventSynchronize(hip_residency.permanent_event),
            "hipEventSynchronize permanent residency") != 0) {
        hip_residency.poisoned = 1;
        return -1;
    }
    region->populated_bytes += nbytes;
    return 0;
}

static int hip_residency_permanent_publish(void *context) {
    uint64_t total = 0;
    (void)context;
    if (!hip_residency.ready || hip_residency.poisoned ||
        hip_residency.permanent_published ||
        hip_residency.permanent_span_count !=
            (uint32_t)hip_component_region_count)
        return -1;
    for (int index = 0; index < hip_component_region_count; index++) {
        HipComponentRegion *region = &hip_component_regions[index];
        if (region->populated_bytes != region->nbytes ||
            region->residency_offset > hip_residency.plan.permanent_bytes ||
            region->nbytes > hip_residency.plan.permanent_bytes -
                region->residency_offset)
            return -1;
        region->device = hip_residency.device_arena +
            (size_t)region->residency_offset;
        region->resident = 1;
        total += region->nbytes;
    }
    for (int index = 0; index < hip_tensor_count; index++)
        if (hip_tensors[index].used &&
            hip_tensor_bind_resident(&hip_tensors[index]) != 0)
            return -1;
    hip_component_pool_bytes = total;
    if (total > hip_component_pool_peak_bytes)
        hip_component_pool_peak_bytes = total;
    hip_component_pool_finalized = 1;
    hip_residency.permanent_published = 1;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_RESIDENCY_PERMANENT spans=%u payload_bytes=%llu "
            "arena_bytes=%llu\n",
            hip_residency.permanent_span_count,
            (unsigned long long)total,
            (unsigned long long)hip_residency.permanent_layout_bytes);
    return 0;
}

static int hip_component_pool_finalize(void) {
    uint64_t total = 0;
    if (hip_component_pool_finalized) return 0;
    if (hip_component_pool_build_regions() != 0) return -1;
    for (int index = 0; index < hip_component_region_count; index++) {
        HipComponentRegion *region = &hip_component_regions[index];
        void *device = NULL;
        if (region->registered || !region->host || region->nbytes < 1 ||
            region->resource_index < 0 ||
            region->resource_index >= HIP_MAX_RESOURCES ||
            hip_success(hipHostRegister((void *)region->host, region->nbytes,
                hipHostRegisterMapped), "hipHostRegister component pool") != 0)
            goto fail;
        region->registered = 1;
        if (!hip_component_test_failure_fired &&
            getenv("SALT_TEST_HIP_COMPONENT_FAIL_AFTER_REGISTER")) {
            hip_component_test_failure_fired = 1;
            goto fail;
        }
        if (hip_success(hipHostGetDevicePointer(
                &device, (void *)region->host, 0),
                "hipHostGetDevicePointer component pool") != 0 ||
            device != (const void *)region->host ||
            (uint64_t)region->nbytes > UINT64_MAX - total)
            goto fail;
        region->device = (unsigned char *)device;
        total += (uint64_t)region->nbytes;
    }
    for (int index = 0; index < HIP_MAX_RESOURCES; index++)
        if (hip_resources[index].used && hip_resources[index].active &&
            hip_resources[index].component_pool)
            hip_resources[index].device =
                (unsigned char *)(uintptr_t)hip_resources[index].host;
    hip_component_pool_bytes = total;
    if (total > hip_component_pool_peak_bytes)
        hip_component_pool_peak_bytes = total;
    hip_component_pool_finalized = 1;
    return 0;
fail: {
    int cleanup_failed = 0;
    for (int index = 0; index < hip_component_region_count; index++) {
        HipComponentRegion *region = &hip_component_regions[index];
        if (region->registered &&
            hip_success(hipHostUnregister((void *)region->host),
                "hipHostUnregister component rollback") != 0) {
            cleanup_failed = 1;
            continue;
        }
        region->registered = 0;
        region->device = NULL;
    }
    if (!cleanup_failed) {
        memset(hip_component_regions, 0, sizeof hip_component_regions);
        hip_component_region_count = 0;
    }
    return -1;
}
}

static int hip_component_pool_release_resource(HipResource *resource) {
    int resource_index, failed = 0;
    uint64_t total = 0;
    if (!resource) return -1;
    resource_index = (int)(resource - hip_resources);
    for (int index = 0; index < hip_component_region_count;) {
        HipComponentRegion *region = &hip_component_regions[index];
        if (region->resource_index != resource_index) {
            index++;
            continue;
        }
        if (region->registered &&
            ((!hip_component_test_unreg_failure_fired &&
              getenv("SALT_TEST_HIP_COMPONENT_FAIL_UNREGISTER_ONCE") &&
              (hip_component_test_unreg_failure_fired = 1)) ||
             hip_success(hipHostUnregister((void *)region->host),
                "hipHostUnregister component pool") != 0)) {
            failed = 1;
            index++;
            continue;
        }
        hip_component_regions[index] =
            hip_component_regions[--hip_component_region_count];
        memset(&hip_component_regions[hip_component_region_count], 0,
            sizeof hip_component_regions[hip_component_region_count]);
    }
    for (int index = 0; index < hip_component_region_count; index++) {
        if ((uint64_t)hip_component_regions[index].nbytes > UINT64_MAX - total)
            return -1;
        total += (uint64_t)hip_component_regions[index].nbytes;
    }
    hip_component_pool_bytes = total;
    if (failed) return -1;
    if (hip_component_region_count == 0)
        hip_component_pool_finalized = 0;
    resource->device = NULL;
    resource->component_pool = 0;
    return 0;
}

extern "C" int salt_gpu_startup_requirements(
    SaltGpuStartupRequirements *requirements, size_t requirements_size) {
    uint64_t output_bytes, descriptor_bytes;
    if (!requirements || requirements_size != sizeof *requirements)
        return -1;
    memset(requirements, 0, sizeof *requirements);
    output_bytes = (uint64_t)HIP_MAX_OUTPUTS * sizeof(float);
    descriptor_bytes =
        (uint64_t)HIP_MAX_BATCH_JOBS * sizeof(HipBatchDesc);
    requirements->fixed_device_bytes =
        3u * output_bytes + output_bytes / 2u + output_bytes / 16u +
        descriptor_bytes + sizeof(int);
    requirements->fixed_pinned_host_bytes =
        2u * output_bytes + descriptor_bytes;
    requirements->descriptor_bytes = 2u * descriptor_bytes;
    requirements->max_output_floats = HIP_MAX_OUTPUTS;
    requirements->max_batch_jobs = HIP_MAX_BATCH_JOBS;
    requirements->shared_buffer_slots = HIP_SHARED_SLOTS;
    return 0;
}

extern "C" int salt_gpu_init(void) {
    hipDeviceProp_t property;
    const char *pageable = getenv("SALT_HIP_PAGEABLE_MMAP");
    if (hip_ready) return 0;
    if (pageable && strcmp(pageable, "0") && strcmp(pageable, "1"))
        return -1;
    if (hip_env_u32("SALT_HIP_TEXT_WAVE_GROUPS", 1u, 8u, 8u,
            &hip_text_wave_groups) != 0 ||
        hip_env_u32("SALT_HIP_TEXT_BF16_WAVE_GROUPS", 1u, 8u, 8u,
            &hip_text_bf16_wave_groups) != 0 ||
        hip_env_u32("SALT_HIP_TEXT_NROW_MIN_B", 2u, 64u, 8u,
            &hip_text_nrow_min_b) != 0)
        return -1;
    hipError_t flags = hipSetDeviceFlags(hipDeviceMapHost);
    if (flags != hipSuccess && flags != hipErrorSetOnActiveProcess)
        return hip_success(flags, "hipSetDeviceFlags");
    if (hip_success(hipGetDeviceProperties(&property, 0),
                     "hipGetDeviceProperties") != 0 ||
        !property.canMapHostMemory)
        return -1;
    hip_pageable_mmap = pageable && !strcmp(pageable, "1");
    if (hip_pageable_mmap) {
        hip_pageable_mmap = 0;
        return -1;
    }
    {
        const char *telemetry = getenv("SALT_NVFP4_TELEMETRY");
        const char *timing = getenv("SALT_NVFP4_TIMING");
        hip_nvfp4_telemetry_enabled =
            !telemetry || !*telemetry || *telemetry != '0';
        hip_nvfp4_timing_enabled = hip_nvfp4_telemetry_enabled &&
            timing && *timing && *timing != '0';
    }
    if (hip_nvfp4_timing_enabled &&
        (hip_success(hipEventCreate(&hip_nvfp4_event_start),
                      "hipEventCreate NVFP4 start") != 0 ||
         hip_success(hipEventCreate(&hip_nvfp4_event_qdq),
                      "hipEventCreate NVFP4 QDQ") != 0 ||
         hip_success(hipEventCreate(&hip_nvfp4_event_projection),
                      "hipEventCreate NVFP4 projection") != 0 ||
         hip_success(hipEventCreate(&hip_nvfp4_event_d2h),
                      "hipEventCreate NVFP4 D2H") != 0)) {
        if (hip_nvfp4_event_d2h) hipEventDestroy(hip_nvfp4_event_d2h);
        if (hip_nvfp4_event_projection)
            hipEventDestroy(hip_nvfp4_event_projection);
        if (hip_nvfp4_event_qdq) hipEventDestroy(hip_nvfp4_event_qdq);
        if (hip_nvfp4_event_start) hipEventDestroy(hip_nvfp4_event_start);
        hip_nvfp4_event_start = hip_nvfp4_event_qdq =
            hip_nvfp4_event_projection = hip_nvfp4_event_d2h = NULL;
        hip_nvfp4_timing_enabled = 0;
        return -1;
    }
    hip_moe_timing_enabled = getenv("SALT_HIP_TOLL_SUMMARY") != NULL;
    if (hip_moe_timing_enabled &&
        (hip_success(hipEventCreate(&hip_moe_event_start),
            "hipEventCreate MoE start") != 0 ||
         hip_success(hipEventCreate(&hip_moe_event_gate_up),
            "hipEventCreate MoE gate/up") != 0 ||
         hip_success(hipEventCreate(&hip_moe_event_activation),
            "hipEventCreate MoE activation") != 0 ||
         hip_success(hipEventCreate(&hip_moe_event_down),
            "hipEventCreate MoE down") != 0)) {
        if (hip_moe_event_down) hipEventDestroy(hip_moe_event_down);
        if (hip_moe_event_activation) hipEventDestroy(hip_moe_event_activation);
        if (hip_moe_event_gate_up) hipEventDestroy(hip_moe_event_gate_up);
        if (hip_moe_event_start) hipEventDestroy(hip_moe_event_start);
        hip_moe_event_start = hip_moe_event_gate_up =
            hip_moe_event_activation = hip_moe_event_down = NULL;
        hip_moe_timing_enabled = 0;
        return -1;
    }
    if (hip_success(hipMalloc((void **)&hip_x,
                                (size_t)HIP_MAX_OUTPUTS * sizeof(float)),
                     "hipMalloc input") != 0 ||
        hip_success(hipMalloc((void **)&hip_y,
                                (size_t)HIP_MAX_OUTPUTS * sizeof(float)),
                     "hipMalloc output") != 0 ||
        hip_success(hipMalloc((void **)&hip_attention_status, sizeof(int)),
                     "hipMalloc attention status") != 0 ||
        hip_success(hipMalloc((void **)&hip_nvfp4_xdq,
                                (size_t)HIP_MAX_OUTPUTS * sizeof(float)),
                     "hipMalloc NVFP4 input QDQ") != 0 ||
        hip_success(hipMalloc((void **)&hip_nvfp4_packed,
                                (size_t)HIP_MAX_OUTPUTS / 2u),
                     "hipMalloc NVFP4 input packed") != 0 ||
        hip_success(hipMalloc((void **)&hip_nvfp4_scales,
                                (size_t)HIP_MAX_OUTPUTS / 16u),
                     "hipMalloc NVFP4 input scales") != 0 ||
        hip_success(hipMalloc((void **)&hip_device_desc,
                                (size_t)HIP_MAX_BATCH_JOBS *
                                    sizeof(HipBatchDesc)),
                     "hipMalloc descriptors") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_host_x,
                                   (size_t)HIP_MAX_OUTPUTS * sizeof(float),
                                   hipHostMallocDefault),
                     "hipHostMalloc input") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_host_y,
                                   (size_t)HIP_MAX_OUTPUTS * sizeof(float),
                                   hipHostMallocDefault),
                     "hipHostMalloc output") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_host_desc,
                                   (size_t)HIP_MAX_BATCH_JOBS *
                                       sizeof(HipBatchDesc),
                                   hipHostMallocDefault),
                     "hipHostMalloc descriptors") != 0) {
        if (hip_host_desc) hipHostFree(hip_host_desc);
        if (hip_host_y) hipHostFree(hip_host_y);
        if (hip_host_x) hipHostFree(hip_host_x);
        if (hip_device_desc) hipFree(hip_device_desc);
        if (hip_nvfp4_scales) hipFree(hip_nvfp4_scales);
        if (hip_nvfp4_packed) hipFree(hip_nvfp4_packed);
        if (hip_nvfp4_xdq) hipFree(hip_nvfp4_xdq);
        if (hip_y) hipFree(hip_y);
        if (hip_attention_status) hipFree(hip_attention_status);
        if (hip_x) hipFree(hip_x);
        if (hip_moe_event_down) hipEventDestroy(hip_moe_event_down);
        if (hip_moe_event_activation) hipEventDestroy(hip_moe_event_activation);
        if (hip_moe_event_gate_up) hipEventDestroy(hip_moe_event_gate_up);
        if (hip_moe_event_start) hipEventDestroy(hip_moe_event_start);
        if (hip_nvfp4_event_d2h) hipEventDestroy(hip_nvfp4_event_d2h);
        if (hip_nvfp4_event_projection)
            hipEventDestroy(hip_nvfp4_event_projection);
        if (hip_nvfp4_event_qdq) hipEventDestroy(hip_nvfp4_event_qdq);
        if (hip_nvfp4_event_start) hipEventDestroy(hip_nvfp4_event_start);
        hip_host_desc = NULL;
        hip_host_x = hip_host_y = NULL;
        hip_device_desc = NULL;
        hip_nvfp4_xdq = NULL;
        hip_nvfp4_packed = hip_nvfp4_scales = NULL;
        hip_x = hip_y = NULL;
        hip_attention_status = NULL;
        hip_nvfp4_event_start = hip_nvfp4_event_qdq =
            hip_nvfp4_event_projection = hip_nvfp4_event_d2h = NULL;
        hip_moe_event_start = hip_moe_event_gate_up =
            hip_moe_event_activation = hip_moe_event_down = NULL;
        hip_moe_timing_enabled = 0;
        hip_nvfp4_timing_enabled = 0;
        return -1;
    }
    memset(hip_resources, 0, sizeof hip_resources);
    memset(hip_shared_slots, 0, sizeof hip_shared_slots);
    memset(hip_tensors, 0, sizeof hip_tensors);
    memset(hip_component_regions, 0, sizeof hip_component_regions);
    hip_component_region_count = 0;
    hip_component_pool_finalized = 0;
    hip_component_pool_bytes = 0;
    hip_component_pool_peak_bytes = 0;
    hip_component_test_failure_fired = 0;
    hip_component_test_unreg_failure_fired = 0;
    hip_selected_epoch_state_reset();
    memset(&hip_stats, 0, sizeof hip_stats);
    hip_q4_hetero_logical_input_bytes = 0;
    hip_q4_hetero_transfer_input_bytes = 0;
    hip_q4_hetero_output_bytes = 0;
    hip_q4_hetero_descriptor_bytes = 0;
    hip_q4_hetero_reused_inputs = 0;
    hip_q4_hetero_kernel_launches = 0;
    hip_q4_hetero_completion_fences = 0;
    hip_q4_hetero_h2d_ns = 0;
    hip_q4_hetero_launch_ns = 0;
    hip_q4_hetero_completion_ns = 0;
    hip_q4_hetero_scatter_ns = 0;
    hip_selected_bind_calls = 0;
    hip_selected_bind_reuses = 0;
    hip_selected_register_ns = 0;
    hip_selected_pointer_ns = 0;
    hip_selected_unbind_calls = 0;
    hip_selected_unbind_sync_ns = 0;
    hip_selected_unregister_ns = 0;
    hip_selected_shared_calls = 0;
    hip_selected_shared_jobs = 0;
    hip_selected_shared_h2d_ns = 0;
    hip_selected_shared_completion_ns = 0;
    hip_selected_gate_up_gpu_ns = 0;
    hip_selected_activation_gpu_ns = 0;
    hip_selected_down_gpu_ns = 0;
    hip_consumer_sync_calls = 0;
    hip_consumer_sync_ns = 0;
    hip_indexed_calls = 0;
    hip_indexed_jobs = 0;
    hip_indexed_h2d_ns = 0;
    hip_indexed_launch_ns = 0;
    hip_peak_window_bytes = 0;
    hip_path_enabled = getenv("SALT_HIP_PATH_SUMMARY") != NULL;
    memset(hip_path_counters, 0, sizeof hip_path_counters);
    hip_selected_legacy_refusals = 0;
    hip_api_failures = 0;
    hip_text_selected_wave_enabled =
        !getenv("SALT_HIP_TEXT_SELECTED_WAVE") ||
        strcmp(getenv("SALT_HIP_TEXT_SELECTED_WAVE"), "0") != 0;
    hip_text_nrow_enabled = !getenv("SALT_HIP_TEXT_NROW") ||
        strcmp(getenv("SALT_HIP_TEXT_NROW"), "0") != 0;
    hip_text_pair_enabled = !getenv("SALT_HIP_TEXT_PAIR") ||
        strcmp(getenv("SALT_HIP_TEXT_PAIR"), "0") != 0;
    hip_text_system_fence_enabled =
        getenv("SALT_HIP_TEXT_SYSTEM_FENCE") &&
        strcmp(getenv("SALT_HIP_TEXT_SYSTEM_FENCE"), "0") != 0;
    hip_tensor_count = 0;
    hip_ready = 1;
    fprintf(stderr, "gpu: ROCm ready (%s gfx=%d.%d)\n",
            property.name, property.major, property.minor);
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_GEOMETRY wave_groups=%u bf16_wave_groups=%u "
            "nrow_min_b=%u\n",
            hip_text_wave_groups, hip_text_bf16_wave_groups,
            hip_text_nrow_min_b);
    if (hip_pageable_mmap)
        fputs("gpu: HIP pageable mmap enabled\n", stderr);
    return 0;
}

extern "C" int salt_gpu_free(void) {
    int failed = 0;
    if (hip_residency.ready) return -1;
    if (hip_ready &&
        (getenv("SALT_TEST_HIP_FREE_SYNC_FAILURE") ||
         hip_success(hipDeviceSynchronize(),
             "hipDeviceSynchronize free") != 0))
        return -1;
    for (int index = HIP_MAX_RESOURCES - 1; index >= 0; index--)
        if (hip_resources[index].used && hip_resources[index].component_pool &&
            hip_component_pool_release_resource(&hip_resources[index]) != 0)
            failed = 1;
    if (hip_selected_resources) {
        for (int slot = hip_selected_capacity - 1; slot >= 0; slot--) {
            HipSelectedResource *resource = &hip_selected_resources[slot];
            if (resource->active && resource->registered &&
                hip_success(hipHostUnregister((void *)resource->host),
                    "hipHostUnregister selected free") != 0)
                failed = 1;
        }
        free(hip_selected_resources);
        if (hip_selected_resource_slots &&
            hip_success(hipHostFree(hip_selected_resource_slots),
                "hipHostFree selected resource table") != 0)
            failed = 1;
        if (hip_selected_logical_slots &&
            hip_success(hipHostFree(hip_selected_logical_slots),
                "hipHostFree selected logical slot table") != 0)
            failed = 1;
        if (hip_selected_payload_offsets &&
            hip_success(hipHostFree(hip_selected_payload_offsets),
                "hipHostFree selected payload offset table") != 0)
            failed = 1;
        hip_selected_resources = NULL;
        hip_selected_resource_slots = NULL;
        hip_selected_logical_slots = NULL;
        hip_selected_payload_offsets = NULL;
        hip_selected_device_resources = NULL;
        hip_selected_device_logical_slots = NULL;
        hip_selected_device_payload_offsets = NULL;
        hip_selected_capacity = 0;
        hip_selected_logical_capacity = 0;
    }
    for (int i = HIP_MAX_RESOURCES - 1; i >= 0; i--)
        if (hip_resources[i].used) {
            if (hip_resources[i].registered &&
                hip_success(hipHostUnregister(
                    (void *)(hip_resources[i].host +
                        hip_resources[i].window_offset)),
                    "hipHostUnregister free") != 0)
                failed = 1;
            memset(&hip_resources[i], 0, sizeof hip_resources[i]);
        }
    memset(hip_component_regions, 0, sizeof hip_component_regions);
    hip_component_region_count = 0;
    hip_component_pool_finalized = 0;
    hip_component_pool_bytes = 0;
    hip_component_pool_peak_bytes = 0;
    if (hip_x && hip_success(hipFree(hip_x), "hipFree input") != 0)
        failed = 1;
    if (hip_y && hip_success(hipFree(hip_y), "hipFree output") != 0)
        failed = 1;
    if (hip_attention_kv &&
        hip_success(hipFree(hip_attention_kv),
                     "hipFree attention KV") != 0)
        failed = 1;
    if (hip_attention_status &&
        hip_success(hipFree(hip_attention_status),
                     "hipFree attention status") != 0)
        failed = 1;
    if (hip_nvfp4_xdq &&
        hip_success(hipFree(hip_nvfp4_xdq),
                     "hipFree NVFP4 input QDQ") != 0)
        failed = 1;
    if (hip_nvfp4_packed &&
        hip_success(hipFree(hip_nvfp4_packed),
                     "hipFree NVFP4 input packed") != 0)
        failed = 1;
    if (hip_nvfp4_scales &&
        hip_success(hipFree(hip_nvfp4_scales),
                     "hipFree NVFP4 input scales") != 0)
        failed = 1;
    if (hip_device_desc &&
        hip_success(hipFree(hip_device_desc),
                     "hipFree descriptors") != 0)
        failed = 1;
    if (hip_host_x &&
        hip_success(hipHostFree(hip_host_x),
                     "hipHostFree input") != 0)
        failed = 1;
    if (hip_host_y &&
        hip_success(hipHostFree(hip_host_y),
                     "hipHostFree output") != 0)
        failed = 1;
    if (hip_host_desc &&
        hip_success(hipHostFree(hip_host_desc),
                     "hipHostFree descriptors") != 0)
        failed = 1;
    if (hip_moe_event_down &&
        hip_success(hipEventDestroy(hip_moe_event_down),
            "hipEventDestroy MoE down") != 0)
        failed = 1;
    if (hip_moe_event_activation &&
        hip_success(hipEventDestroy(hip_moe_event_activation),
            "hipEventDestroy MoE activation") != 0)
        failed = 1;
    if (hip_moe_event_gate_up &&
        hip_success(hipEventDestroy(hip_moe_event_gate_up),
            "hipEventDestroy MoE gate/up") != 0)
        failed = 1;
    if (hip_moe_event_start &&
        hip_success(hipEventDestroy(hip_moe_event_start),
            "hipEventDestroy MoE start") != 0)
        failed = 1;
    if (hip_nvfp4_event_d2h &&
        hip_success(hipEventDestroy(hip_nvfp4_event_d2h),
                     "hipEventDestroy NVFP4 D2H") != 0)
        failed = 1;
    if (hip_nvfp4_event_projection &&
        hip_success(hipEventDestroy(hip_nvfp4_event_projection),
                     "hipEventDestroy NVFP4 projection") != 0)
        failed = 1;
    if (hip_nvfp4_event_qdq &&
        hip_success(hipEventDestroy(hip_nvfp4_event_qdq),
                     "hipEventDestroy NVFP4 QDQ") != 0)
        failed = 1;
    if (hip_nvfp4_event_start &&
        hip_success(hipEventDestroy(hip_nvfp4_event_start),
                     "hipEventDestroy NVFP4 start") != 0)
        failed = 1;
    hip_x = hip_y = NULL;
    hip_nvfp4_xdq = NULL;
    hip_nvfp4_packed = hip_nvfp4_scales = NULL;
    hip_host_x = hip_host_y = NULL;
    hip_host_desc = hip_device_desc = NULL;
    hip_nvfp4_event_start = hip_nvfp4_event_qdq =
        hip_nvfp4_event_projection = hip_nvfp4_event_d2h = NULL;
    hip_moe_event_start = hip_moe_event_gate_up =
        hip_moe_event_activation = hip_moe_event_down = NULL;
    hip_moe_timing_enabled = 0;
    hip_attention_kv = NULL;
    hip_attention_kv_nbytes = 0;
    hip_attention_kv_host = NULL;
    hip_attention_status = NULL;
    hip_attention_kv_sync_count = 0;
    memset(hip_attention_kv_sync, 0, sizeof hip_attention_kv_sync);
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    if (getenv("SALT_HIP_TRANSFER_SUMMARY"))
        fprintf(stderr,
            "SALT_HIP_Q4_TRANSFER logical_input_bytes=%llu "
            "transfer_input_bytes=%llu reused_inputs=%llu "
            "output_bytes=%llu descriptor_bytes=%llu "
            "kernel_launches=%llu completion_fences=%llu\n",
            (unsigned long long)hip_q4_hetero_logical_input_bytes,
            (unsigned long long)hip_q4_hetero_transfer_input_bytes,
            (unsigned long long)hip_q4_hetero_reused_inputs,
            (unsigned long long)hip_q4_hetero_output_bytes,
            (unsigned long long)hip_q4_hetero_descriptor_bytes,
            (unsigned long long)hip_q4_hetero_kernel_launches,
            (unsigned long long)hip_q4_hetero_completion_fences);
    if (getenv("SALT_HIP_TOLL_SUMMARY"))
        fprintf(stderr,
            "SALT_HIP_TOLL q4_h2d_ns=%llu q4_launch_ns=%llu "
            "q4_completion_ns=%llu q4_scatter_ns=%llu "
            "selected_bind_calls=%llu selected_bind_reuses=%llu "
            "selected_register_ns=%llu selected_pointer_ns=%llu "
            "selected_unbind_calls=%llu selected_unbind_sync_calls=%llu "
            "selected_unbind_sync_skips=%llu selected_unbind_sync_ns=%llu "
            "selected_unregister_ns=%llu selected_shared_calls=%llu "
            "selected_shared_jobs=%llu selected_shared_h2d_ns=%llu "
            "selected_shared_completion_ns=%llu "
            "selected_gate_up_gpu_ns=%llu selected_activation_gpu_ns=%llu "
            "selected_down_gpu_ns=%llu consumer_sync_calls=%llu "
            "consumer_sync_ns=%llu indexed_calls=%llu indexed_jobs=%llu "
            "indexed_h2d_ns=%llu indexed_launch_ns=%llu\n",
            (unsigned long long)hip_q4_hetero_h2d_ns,
            (unsigned long long)hip_q4_hetero_launch_ns,
            (unsigned long long)hip_q4_hetero_completion_ns,
            (unsigned long long)hip_q4_hetero_scatter_ns,
            (unsigned long long)hip_selected_bind_calls,
            (unsigned long long)hip_selected_bind_reuses,
            (unsigned long long)hip_selected_register_ns,
            (unsigned long long)hip_selected_pointer_ns,
            (unsigned long long)hip_selected_unbind_calls,
            (unsigned long long)hip_selected_unbind_sync_calls,
            (unsigned long long)hip_selected_unbind_sync_skips,
            (unsigned long long)hip_selected_unbind_sync_ns,
            (unsigned long long)hip_selected_unregister_ns,
            (unsigned long long)hip_selected_shared_calls,
            (unsigned long long)hip_selected_shared_jobs,
            (unsigned long long)hip_selected_shared_h2d_ns,
            (unsigned long long)hip_selected_shared_completion_ns,
            (unsigned long long)hip_selected_gate_up_gpu_ns,
            (unsigned long long)hip_selected_activation_gpu_ns,
            (unsigned long long)hip_selected_down_gpu_ns,
            (unsigned long long)hip_consumer_sync_calls,
            (unsigned long long)hip_consumer_sync_ns,
            (unsigned long long)hip_indexed_calls,
            (unsigned long long)hip_indexed_jobs,
            (unsigned long long)hip_indexed_h2d_ns,
            (unsigned long long)hip_indexed_launch_ns);
    hip_path_publish();
    if (getenv("SALT_HIP_PATH_SUMMARY"))
        fprintf(stderr,
            "SALT_HIP_FALLBACK selected_legacy_refusals=%llu "
            "mapped_only_misses=%llu hip_api_failures=%llu\n",
            (unsigned long long)hip_selected_legacy_refusals,
            (unsigned long long)hip_stats.mapped_only_misses,
            (unsigned long long)hip_api_failures);
    hip_ready = hip_mapped_only = hip_defer = hip_pageable_mmap = 0;
    hip_nvfp4_telemetry_enabled = 1;
    hip_nvfp4_timing_enabled = 0;
    hip_tensor_count = 0;
    memset(hip_tensors, 0, sizeof hip_tensors);
    memset(&hip_stats, 0, sizeof hip_stats);
    hip_q4_hetero_logical_input_bytes = 0;
    hip_q4_hetero_transfer_input_bytes = 0;
    hip_q4_hetero_output_bytes = 0;
    hip_q4_hetero_descriptor_bytes = 0;
    hip_q4_hetero_reused_inputs = 0;
    hip_q4_hetero_kernel_launches = 0;
    hip_q4_hetero_completion_fences = 0;
    hip_q4_hetero_h2d_ns = 0;
    hip_q4_hetero_launch_ns = 0;
    hip_q4_hetero_completion_ns = 0;
    hip_q4_hetero_scatter_ns = 0;
    hip_selected_bind_calls = 0;
    hip_selected_bind_reuses = 0;
    hip_selected_register_ns = 0;
    hip_selected_pointer_ns = 0;
    hip_selected_unbind_calls = 0;
    hip_selected_unbind_sync_ns = 0;
    hip_selected_unregister_ns = 0;
    hip_selected_shared_calls = 0;
    hip_selected_shared_jobs = 0;
    hip_selected_shared_h2d_ns = 0;
    hip_selected_shared_completion_ns = 0;
    hip_selected_gate_up_gpu_ns = 0;
    hip_selected_activation_gpu_ns = 0;
    hip_selected_down_gpu_ns = 0;
    hip_consumer_sync_calls = 0;
    hip_consumer_sync_ns = 0;
    hip_indexed_calls = 0;
    hip_indexed_jobs = 0;
    hip_indexed_h2d_ns = 0;
    hip_indexed_launch_ns = 0;
    hip_peak_window_bytes = 0;
    memset(hip_path_counters, 0, sizeof hip_path_counters);
    hip_path_enabled = 0;
    hip_selected_legacy_refusals = 0;
    hip_api_failures = 0;
    hip_text_selected_wave_enabled = 0;
    hip_text_nrow_enabled = 0;
    hip_text_pair_enabled = 0;
    hip_text_system_fence_enabled = 0;
    hip_text_wave_groups = 0;
    hip_text_bf16_wave_groups = 0;
    hip_text_nrow_min_b = 0;
    hip_selected_epoch_state_reset();
    return failed ? -1 : 0;
}

extern "C" int salt_gpu_weight_resource_describe(
    uint32_t kind, uint32_t resource_id, const void *base, size_t nbytes) {
    HipResource *slot = NULL;
    if (!hip_ready || !base || nbytes < 1 ||
        hip_resource_record(kind, resource_id))
        return -1;
    for (int i = 0; i < HIP_MAX_RESOURCES; i++)
        if (!hip_resources[i].used) { slot = &hip_resources[i]; break; }
    if (!slot) return -1;
    slot->used = 1;
    slot->active = 0;
    slot->registered = 0;
    slot->kind = kind;
    slot->id = resource_id;
    slot->policy = SALT_GPU_WEIGHT_ADDRESS_AUTO;
    slot->host = (const unsigned char *)base;
    slot->device = NULL;
    slot->nbytes = nbytes;
    return 0;
}

extern "C" int salt_gpu_weight_resource_activate(
    uint32_t kind, uint32_t resource_id, SaltGpuWeightAddressability policy) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    if (!resource || resource->active || resource->device ||
        policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
        return -1;
    resource->policy = policy;
    resource->active = 1;
    return 0;
}

extern "C" int salt_gpu_weight_resource_pool_bind(
    uint32_t kind, uint32_t resource_id) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    if (!resource || !resource->active || resource->device ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        resource->component_pool || hip_component_pool_finalized)
        return -1;
    resource->component_pool = 1;
    return 0;
}

extern "C" int salt_gpu_weight_resource_deactivate(
    uint32_t kind, uint32_t resource_id) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    if (!resource || !resource->active || resource->window_nbytes ||
        hip_success(hipDeviceSynchronize(),
                     "hipDeviceSynchronize deactivate") != 0 ||
        (resource->component_pool &&
         hip_component_pool_release_resource(resource) != 0) ||
        (resource->registered &&
         hip_success(hipHostUnregister((void *)resource->host),
                      "hipHostUnregister deactivate") != 0))
        return -1;
    resource->active = 0;
    resource->registered = 0;
    resource->policy = SALT_GPU_WEIGHT_ADDRESS_AUTO;
    resource->device = NULL;
    resource->window_offset = 0;
    resource->window_nbytes = 0;
    resource->component_pool = 0;
    return 0;
}

extern "C" int salt_gpu_weight_resource_usage(
    SaltGpuWeightResourceUsage *usage, size_t usage_size) {
    if (!usage || usage_size != sizeof *usage) return -1;
    memset(usage, 0, sizeof *usage);
    for (int i = 0; i < HIP_MAX_RESOURCES; i++) {
        const HipResource *resource = &hip_resources[i];
        if (!resource->used) continue;
        usage->described_resources++;
        usage->described_bytes += resource->nbytes;
        if (!resource->active) continue;
        usage->active_resources++;
        if (resource->window_nbytes) {
            usage->active_windows++;
            usage->active_window_bytes += resource->window_nbytes;
        }
        if (resource->component_pool) usage->active_windows++;
    }
    if (hip_residency.permanent_published)
        usage->device_copied_weight_bytes = hip_component_pool_bytes;
    else {
        if (usage->active_window_bytes > UINT64_MAX - hip_component_pool_bytes)
            return -1;
        usage->active_window_bytes += hip_component_pool_bytes;
    }
    usage->peak_window_bytes = hip_peak_window_bytes >
        hip_component_pool_peak_bytes ? hip_peak_window_bytes :
        hip_component_pool_peak_bytes;
    return 0;
}

extern "C" int salt_gpu_weight_resource_bind(
    uint32_t kind, uint32_t resource_id, const void *base, size_t nbytes) {
    (void)kind;
    (void)resource_id;
    (void)base;
    (void)nbytes;
    return -1;
}

extern "C" int salt_gpu_weight_window_bind(
    uint32_t kind, uint32_t resource_id, uint64_t offset, size_t nbytes) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    const unsigned char *host;
    void *device = NULL;
    if (!resource || !resource->active || resource->window_nbytes || nbytes < 1 ||
        offset > resource->nbytes || nbytes > resource->nbytes - (size_t)offset ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        resource->component_pool)
        return -1;
    host = resource->host + offset;
    if (hip_success(hipHostRegister((void *)host, nbytes,
            hipHostRegisterMapped), "hipHostRegister bounded window") != 0 ||
        hip_success(hipHostGetDevicePointer(&device, (void *)host, 0),
            "hipHostGetDevicePointer bounded window") != 0 || !device) {
        (void)hipHostUnregister((void *)host);
        return -1;
    }
    resource->device = (unsigned char *)(uintptr_t)(
        (uintptr_t)device - (uintptr_t)offset);
    resource->registered = 1;
    resource->window_offset = offset;
    resource->window_nbytes = nbytes;
    if ((uint64_t)nbytes > hip_peak_window_bytes)
        hip_peak_window_bytes = nbytes;
    return 0;
}

extern "C" int salt_gpu_weight_window_unbind(
    uint32_t kind, uint32_t resource_id) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    const unsigned char *host;
    if (!resource || !resource->active || !resource->window_nbytes ||
        resource->policy != SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW ||
        hip_success(hipDeviceSynchronize(),
                     "hipDeviceSynchronize window unbind") != 0)
        return -1;
    host = resource->host + resource->window_offset;
    if (!resource->registered ||
        hip_success(hipHostUnregister((void *)host),
            "hipHostUnregister bounded window") != 0)
        return -1;
    resource->device = NULL;
    resource->registered = 0;
    resource->window_offset = 0;
    resource->window_nbytes = 0;
    return 0;
}

extern "C" int salt_gpu_weight_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint32_t *vals, const uint16_t *scales, const uint16_t *biases,
    int bits, int R, int C) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    HipTensor *existing;
    size_t vbytes, sbytes;
    uint64_t voff, soff, boff;
    if (!resource || !resource->active || !key || !vals || !scales || !biases ||
        hip_tensor_sizes(bits, R, C, &vbytes, &sbytes) != 0 ||
        hip_range(resource, vals, vbytes, &voff) != 0 ||
        hip_range(resource, scales, sbytes, &soff) != 0 ||
        hip_range(resource, biases, sbytes, &boff) != 0)
        return -1;
    existing = hip_tensor(key);
    if (existing)
        return existing->kind == kind && existing->resource_id == resource_id &&
               existing->bits == bits && existing->rows == R &&
               existing->cols == C && existing->value_offset == voff &&
               existing->scale_offset == soff &&
               existing->bias_offset == boff ? 0 : -1;
    if (hip_tensor_count >= HIP_MAX_TENSORS) return -1;
    HipTensor *tensor = &hip_tensors[hip_tensor_count++];
    tensor->used = 1;
    tensor->bits = bits;
    tensor->rows = R;
    tensor->cols = C;
    tensor->key = key;
    tensor->kind = kind;
    tensor->resource_id = resource_id;
    tensor->value_offset = voff;
    tensor->scale_offset = soff;
    tensor->bias_offset = boff;
    return 0;
}

extern "C" int salt_gpu_nvfp4_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint8_t *weight_packed, const uint8_t *weight_scales,
    const float *weight_global_scale, const float *input_global_scale,
    int R, int C) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    HipTensor *existing;
    uint64_t elements, value_bytes, scale_bytes;
    uint64_t voff, soff, wgoff, igoff;
    if (!resource || !resource->active || !key || !weight_packed ||
        !weight_scales ||
        !weight_global_scale || !input_global_scale || R < 1 || C < 16 ||
        (C & 15) != 0)
        return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    value_bytes = elements / 2u;
    scale_bytes = elements / 16u;
    if (value_bytes > SIZE_MAX || scale_bytes > SIZE_MAX ||
        hip_range(resource, weight_packed, (size_t)value_bytes, &voff) != 0 ||
        hip_range(resource, weight_scales, (size_t)scale_bytes, &soff) != 0 ||
        hip_range(resource, weight_global_scale, sizeof(float), &wgoff) != 0 ||
        hip_range(resource, input_global_scale, sizeof(float), &igoff) != 0)
        return -1;
    existing = hip_tensor(key);
    if (existing)
        return existing->kind == kind &&
               existing->resource_id == resource_id &&
               existing->bits == HIP_FORMAT_NVFP4 &&
               existing->rows == R && existing->cols == C &&
               existing->value_offset == voff &&
               existing->scale_offset == soff &&
               existing->bias_offset == wgoff &&
               existing->aux_offset == igoff ? 0 : -1;
    if (hip_tensor_count >= HIP_MAX_TENSORS) return -1;
    HipTensor *tensor = &hip_tensors[hip_tensor_count++];
    tensor->used = 1;
    tensor->bits = HIP_FORMAT_NVFP4;
    tensor->rows = R;
    tensor->cols = C;
    tensor->key = key;
    tensor->kind = kind;
    tensor->resource_id = resource_id;
    tensor->value_offset = voff;
    tensor->scale_offset = soff;
    tensor->bias_offset = wgoff;
    tensor->aux_offset = igoff;
    return 0;
}

extern "C" int salt_gpu_bf16_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint16_t *weights, int R, int C) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    HipTensor *existing;
    uint64_t elements, bytes, offset;
    if (!resource || !resource->active || !key || !weights ||
        R < 1 || C < 8 || (C & 7) != 0)
        return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    if (elements > SIZE_MAX / 2u) return -1;
    bytes = elements * 2u;
    if (hip_range(resource, weights, (size_t)bytes, &offset) != 0)
        return -1;
    existing = hip_tensor(key);
    if (existing)
        return existing->kind == kind && existing->resource_id == resource_id &&
               existing->bits == 16 && existing->rows == R &&
               existing->cols == C && existing->value_offset == offset ? 0 : -1;
    if (hip_tensor_count >= HIP_MAX_TENSORS) return -1;
    HipTensor *tensor = &hip_tensors[hip_tensor_count++];
    memset(tensor, 0, sizeof *tensor);
    tensor->used = 1;
    tensor->bits = 16;
    tensor->rows = R;
    tensor->cols = C;
    tensor->key = key;
    tensor->kind = kind;
    tensor->resource_id = resource_id;
    tensor->value_offset = offset;
    return 0;
}

extern "C" int salt_gpu_weight_resource_unbind(uint32_t kind,
                                                uint32_t resource_id) {
    HipResource *resource = hip_resource_record(kind, resource_id);
    if (!resource) return -1;
    if (resource->active &&
        salt_gpu_weight_resource_deactivate(kind, resource_id) != 0)
        return -1;
    if (resource->device || resource->registered || resource->window_nbytes)
        return -1;
    for (int i = 0; i < hip_tensor_count; i++)
        if (hip_tensors[i].used && hip_tensors[i].kind == kind &&
            hip_tensors[i].resource_id == resource_id)
            hip_tensors[i].used = 0;
    memset(resource, 0, sizeof *resource);
    return 0;
}

static int hip_run(const void *key, const uint32_t *vals, int bits,
                    int R, int C, const float *x, float *y) {
    HipTensor *tensor = hip_tensor(key ? key : vals);
    HipResource *resource;
    unsigned char *device;
    uint64_t voff, soff, boff, auxoff;
    int blocks;
    if (!hip_ready || !tensor || tensor->bits != bits || tensor->rows != R ||
        tensor->cols != C || C > HIP_MAX_C || (size_t)R > HIP_MAX_OUTPUTS ||
        !x || !y) {
        if (hip_mapped_only) hip_stats.mapped_only_misses++;
        return -1;
    }
    if (hip_tensor_physical(tensor, &resource, &device,
            &voff, &soff, &boff, &auxoff) != 0)
        return -1;
    if (hip_success(hipMemcpy(hip_x, x, (size_t)C * sizeof(float),
                                hipMemcpyHostToDevice), "hipMemcpy input") != 0)
        return -1;
    blocks = (R + 255) / 256;
    if (bits == 4)
        hip_q4_exact<<<blocks, 256>>>(
            device, voff, soff, boff, hip_x, hip_y, R, C);
    else
        hip_q8_exact<<<blocks, 256>>>(
            device, voff, soff, boff, hip_x, hip_y, R, C);
    if (hip_success(hipGetLastError(), "projection launch") != 0 ||
        hip_success(hipMemcpy(y, hip_y, (size_t)R * sizeof(float),
                                hipMemcpyDeviceToHost), "hipMemcpy output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) hip_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs++;
    hip_path_record(bits == 4 ? HIP_PATH_Q4_SINGLE_EXACT
                              : HIP_PATH_Q8_SINGLE_EXACT,
        UINT64_C(1), (uint64_t)(uint32_t)R, UINT64_C(1), UINT64_C(1));
    return 0;
}

static int hip_run_same_weight_batch(const void *key, const uint32_t *vals,
    int bits, int R, int C, int B, const float *const *xs,
    float *const *ys) {
    HipTensor *tensor = hip_tensor(key ? key : vals);
    HipResource *resource;
    unsigned char *device;
    uint64_t voff, soff, boff, auxoff;
    size_t input_floats, output_floats;
    enum HipPathId path = HIP_PATH_NONE;
    if (!hip_ready || !tensor || tensor->bits != bits || tensor->rows != R ||
        tensor->cols != C || B < 1 || B > 512 || C > HIP_MAX_C ||
        !xs || !ys || !xs[0] || !ys[0] ||
        (size_t)B > SIZE_MAX / (size_t)C ||
        (size_t)B > SIZE_MAX / (size_t)R) {
        if (hip_mapped_only) hip_stats.mapped_only_misses++;
        return -1;
    }
    input_floats = (size_t)B * (size_t)C;
    output_floats = (size_t)B * (size_t)R;
    if (input_floats > HIP_MAX_OUTPUTS || output_floats > HIP_MAX_OUTPUTS)
        return -1;
    for (int b = 1; b < B; b++)
        if (xs[b] != xs[0] + (size_t)b * (size_t)C ||
            ys[b] != ys[0] + (size_t)b * (size_t)R)
            return -1;
    if (hip_tensor_physical(tensor, &resource, &device,
            &voff, &soff, &boff, &auxoff) != 0)
        return -1;
    if (hip_success(hipMemcpy(hip_x, xs[0],
                                input_floats * sizeof(float),
                                hipMemcpyHostToDevice),
                     "hipMemcpy batch input") != 0)
        return -1;
    if (B >= 8) {
        if (bits == 4) {
            path = HIP_PATH_Q4_BATCH_WARP;
            hip_q4_warp_batch<<<(unsigned int)R, 1024>>>(device,
                voff, soff, boff, hip_x, hip_y, R, C, B);
        } else {
            path = HIP_PATH_Q8_BATCH_TILE;
            dim3 grid((unsigned int)R,
                      (unsigned int)((B + 127) / 128));
            hip_q8_shared_tile<<<grid, 128>>>(device,
                voff, soff, boff, hip_x, hip_y, R, C, B);
        }
    } else if (bits == 4) {
        path = HIP_PATH_Q4_BATCH_EXACT;
        hip_q4_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)B), 128>>>(
            device, voff, soff, boff, hip_x, hip_y, R, C);
    } else {
        path = HIP_PATH_Q8_BATCH_EXACT;
        hip_q8_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)B), 128>>>(
            device, voff, soff, boff, hip_x, hip_y, R, C);
    }
    if (hip_success(hipGetLastError(), "batch projection launch") != 0 ||
        hip_success(hipMemcpy(ys[0], hip_y,
                                output_floats * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy batch output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) hip_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += (uint64_t)B;
    hip_path_record(path, (uint64_t)(uint32_t)B,
        (uint64_t)(uint32_t)B * (uint32_t)R, UINT64_C(1), UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_q4_matvec(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C,
    const float *x, float *y) {
    (void)scales; (void)biases;
    return hip_run(vals, vals, 4, R, C, x, y);
}

extern "C" int salt_gpu_q8_matvec(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C,
    const float *x, float *y) {
    (void)scales; (void)biases;
    return hip_run(vals, vals, 8, R, C, x, y);
}

extern "C" int salt_gpu_bf16_matvec(
    const void *key, int R, int C, const float *input, float *output) {
    HipTensor *tensor = hip_tensor(key);
    HipResource *resource;
    unsigned char *device;
    uint64_t voff, soff, boff, auxoff;
    if (!hip_ready || !tensor || tensor->bits != 16 ||
        tensor->rows != R || tensor->cols != C || R < 1 || C < 8 ||
        (C & 7) != 0 || (size_t)R > HIP_MAX_OUTPUTS || !input || !output)
        return -1;
    if (hip_tensor_physical(tensor, &resource, &device,
            &voff, &soff, &boff, &auxoff) != 0 ||
        hip_success(hipMemcpy(hip_x, input, (size_t)C * sizeof(float),
                                hipMemcpyHostToDevice),
                     "hipMemcpy BF16 input") != 0)
        return -1;
    hip_bf16_warp_exact<<<(unsigned int)((R + 3) / 4), 128>>>(
        device, voff, hip_x, hip_y, R, C);
    if (hip_success(hipGetLastError(), "BF16 projection launch") != 0 ||
        hip_success(hipMemcpy(output, hip_y, (size_t)R * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy BF16 output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) hip_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs++;
    hip_path_record(HIP_PATH_BF16_WARP_EXACT, UINT64_C(1),
        (uint64_t)(uint32_t)R, UINT64_C(1), UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_bf16_matvec_batch(
    const void *key, int R, int C, int B,
    const float *inputs, float *outputs) {
    (void)key; (void)R; (void)C; (void)B; (void)inputs; (void)outputs;
    return -1;
}

static int hip_nvfp4_qdq_launch(const float *input, int C, int B,
                                 float input_global_scale,
                                 HipNvfp4Telemetry *telemetry) {
    int groups, blocks;
    size_t elements;
    uint64_t h2d_start = 0;
    if (!hip_ready || !input || C < 16 || C > HIP_MAX_C || B < 1 ||
        (C & 15) != 0 || !(input_global_scale > 0.0f) ||
        (size_t)B > HIP_MAX_OUTPUTS / (size_t)C || !telemetry)
        return -1;
    elements = (size_t)B * (size_t)C;
    if (hip_nvfp4_timing_enabled) h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(hip_x, input, elements * sizeof(float),
                                hipMemcpyHostToDevice),
                     "hipMemcpy NVFP4 input") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled) {
        telemetry->h2d_ns = hip_host_now_ns() - h2d_start;
        if (hip_success(hipEventRecord(hip_nvfp4_event_start),
                         "hipEventRecord NVFP4 start") != 0)
            return -1;
    }
    groups = (int)(elements / 16u);
    blocks = (groups + 255) / 256;
    hip_nvfp4_qdq_exact<<<blocks, 256>>>(
        hip_x, C, B, input_global_scale, hip_nvfp4_packed,
        hip_nvfp4_scales, hip_nvfp4_xdq);
    if (hip_success(hipGetLastError(), "NVFP4 QDQ launch") != 0)
        return -1;
    hip_path_record(HIP_PATH_NVFP4_QDQ, (uint64_t)(uint32_t)B,
        (uint64_t)(uint32_t)B * (uint32_t)C, UINT64_C(1), UINT64_C(0));
    if (hip_nvfp4_timing_enabled &&
        hip_success(hipEventRecord(hip_nvfp4_event_qdq),
                     "hipEventRecord NVFP4 QDQ") != 0)
        return -1;
    return 0;
}

extern "C" int salt_gpu_nvfp4_qdq(
    const float *input, int C, float input_global_scale,
    uint8_t *packed, uint8_t *scales, float *dequant) {
    HipNvfp4Telemetry telemetry;
    if (!packed || !scales || !dequant ||
        hip_nvfp4_telemetry(1, C, 1, &telemetry) != 0 ||
        hip_nvfp4_qdq_launch(input, C, 1, input_global_scale,
                              &telemetry) != 0)
        return -1;
    if (
        hip_success(hipMemcpy(packed, hip_nvfp4_packed,
                                (size_t)C / 2u, hipMemcpyDeviceToHost),
                     "hipMemcpy NVFP4 packed") != 0 ||
        hip_success(hipMemcpy(scales, hip_nvfp4_scales,
                                (size_t)C / 16u, hipMemcpyDeviceToHost),
                     "hipMemcpy NVFP4 scales") != 0 ||
        hip_success(hipMemcpy(dequant, hip_nvfp4_xdq,
                                (size_t)C * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy NVFP4 QDQ") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled) {
        if (hip_event_elapsed_ns(hip_nvfp4_event_start,
                                  hip_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            hip_nvfp4_d2h_timing(hip_nvfp4_event_qdq, &telemetry) != 0)
            return -1;
    }
    telemetry.packed_weight_bytes = 0;
    telemetry.weight_scale_bytes = 0;
    telemetry.projection_output_bytes = 0;
    hip_nvfp4_telemetry_publish(&telemetry, UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_nvfp4_matvec(
    const void *key, int R, int C, const float *input, float *output) {
    return salt_gpu_nvfp4_matvec_batch(key, R, C, 1, input, output);
}

extern "C" int salt_gpu_nvfp4_matvec_batch(
    const void *key, int R, int C, int B,
    const float *inputs, float *outputs) {
    HipTensor *tensor = hip_tensor(key);
    HipResource *resource;
    unsigned char *device;
    uint64_t voff, soff, boff, auxoff;
    HipNvfp4Telemetry telemetry;
    float input_global_scale;
    if (!hip_ready || !tensor || tensor->bits != HIP_FORMAT_NVFP4 ||
        tensor->rows != R || tensor->cols != C || R < 1 || B < 1 ||
        (size_t)B > HIP_MAX_OUTPUTS / (size_t)R || !inputs || !outputs ||
        hip_nvfp4_telemetry(R, C, B, &telemetry) != 0)
        return -1;
    if (hip_tensor_physical(tensor, &resource, &device,
            &voff, &soff, &boff, &auxoff) != 0)
        return -1;
    memcpy(&input_global_scale, resource->host + tensor->aux_offset,
           sizeof input_global_scale);
    if (hip_nvfp4_qdq_launch(inputs, C, B, input_global_scale,
                              &telemetry) != 0)
        return -1;
    hip_nvfp4_warp_exact<<<
        dim3((unsigned int)((R + 3) / 4), (unsigned int)B), 128>>>(
        device, voff, soff, boff, hip_nvfp4_xdq, hip_y, R, C, B);
    if (hip_success(hipGetLastError(), "NVFP4 projection launch") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled &&
        hip_success(hipEventRecord(hip_nvfp4_event_projection),
                     "hipEventRecord NVFP4 projection") != 0)
        return -1;
    if (
        hip_success(hipMemcpy(outputs, hip_y,
                                (size_t)B * (size_t)R * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy NVFP4 output") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled) {
        if (hip_event_elapsed_ns(hip_nvfp4_event_start,
                                  hip_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            hip_event_elapsed_ns(hip_nvfp4_event_qdq,
                                  hip_nvfp4_event_projection,
                                  &telemetry.projection_ns) != 0 ||
            hip_nvfp4_d2h_timing(hip_nvfp4_event_projection,
                                   &telemetry) != 0)
            return -1;
    }
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) hip_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += (uint64_t)B;
    hip_nvfp4_telemetry_publish(&telemetry, UINT64_C(2));
    hip_path_record(HIP_PATH_NVFP4_PROJECT, (uint64_t)(uint32_t)B,
        (uint64_t)(uint32_t)B * (uint32_t)R, UINT64_C(1), UINT64_C(1));
    return 0;
}

static int hip_run_nvfp4_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    const int *rows, int common_rows, int C, int njobs) {
    size_t input_floats = 0, output_floats = 0;
    int max_groups = 0, max_batch = 0, max_rows = 0;
    int saw_trunk = 0, saw_expert = 0;
    uint64_t logical_jobs = 0;
    uint64_t h2d_start = 0;
    HipNvfp4Telemetry telemetry = {0};
    if (!hip_ready || !keys || !inputs || !outputs || !batches ||
        (!rows && common_rows < 1) || C < 16 || C > HIP_MAX_C ||
        (C & 15) != 0 ||
        njobs < 1 || njobs > HIP_MAX_BATCH_JOBS)
        return -1;
    for (int job = 0; job < njobs; job++) {
        HipTensor *tensor = hip_tensor(keys[job]);
        HipResource *resource;
        unsigned char *device;
        uint64_t voff, soff, boff, auxoff;
        HipNvfp4Telemetry delta;
        int batch = batches[job];
        int R = rows ? rows[job] : common_rows;
        if (!tensor || tensor->bits != HIP_FORMAT_NVFP4 ||
            R < 1 || tensor->rows != R || tensor->cols != C || batch < 1 ||
            !inputs[job] || !outputs[job] ||
            (size_t)batch > (HIP_MAX_OUTPUTS - input_floats) / (size_t)C ||
            (size_t)batch > (HIP_MAX_OUTPUTS - output_floats) / (size_t)R)
            goto miss;
        if (hip_tensor_physical(tensor, &resource, &device,
                &voff, &soff, &boff, &auxoff) != 0 ||
            hip_nvfp4_telemetry(R, C, batch, &delta) != 0)
            goto miss;
        /* The heterogeneous Q/DQ kernel writes only FP32 dequant output;
         * unlike the standalone path it does not materialize packed codes or
         * activation-scale planes. */
        delta.activation_qdq_bytes = delta.activation_input_bytes;
        if (hip_nvfp4_telemetry_add(&telemetry, &delta) != 0) goto miss;
        HipBatchDesc *descriptor = &hip_host_desc[job];
        descriptor->resource = device;
        descriptor->value_offset = voff;
        descriptor->scale_offset = soff;
        descriptor->bias_offset = boff;
        descriptor->aux_offset = auxoff;
        descriptor->input_offset = (uint32_t)input_floats;
        descriptor->output_offset = (uint32_t)output_floats;
        descriptor->rows = R;
        descriptor->cols = C;
        descriptor->batch = batch;
        memcpy(hip_host_x + input_floats, inputs[job],
               (size_t)batch * (size_t)C * sizeof(float));
        hip_job_output_offset[job] = (uint32_t)output_floats;
        hip_job_rows[job] = batch * R;
        input_floats += (size_t)batch * (size_t)C;
        output_floats += (size_t)batch * (size_t)R;
        int groups = batch * C / 16;
        if (groups > max_groups) max_groups = groups;
        if (batch > max_batch) max_batch = batch;
        if (R > max_rows) max_rows = R;
        logical_jobs += (uint64_t)batch;
        if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) saw_trunk = 1;
        if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER) saw_expert = 1;
    }
    if (hip_nvfp4_timing_enabled) h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(hip_x, hip_host_x,
                                input_floats * sizeof(float),
                                hipMemcpyHostToDevice),
                     "hipMemcpy NVFP4 heterogeneous input") != 0 ||
        hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
                                (size_t)njobs * sizeof(HipBatchDesc),
                                hipMemcpyHostToDevice),
                     "hipMemcpy NVFP4 heterogeneous descriptors") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled) {
        telemetry.h2d_ns = hip_host_now_ns() - h2d_start;
        if (hip_success(hipEventRecord(hip_nvfp4_event_start),
                         "hipEventRecord NVFP4 heterogeneous start") != 0)
            return -1;
    }
    hip_nvfp4_heterogeneous_qdq<<<
        dim3((unsigned int)((max_groups + 255) / 256), 1u,
             (unsigned int)njobs), 256>>>(
        hip_device_desc, njobs, hip_x, hip_nvfp4_xdq);
    if (hip_success(hipGetLastError(),
                     "NVFP4 heterogeneous QDQ launch") != 0)
        return -1;
    hip_path_record(HIP_PATH_NVFP4_HET_QDQ, logical_jobs,
        (uint64_t)input_floats, UINT64_C(1), UINT64_C(0));
    if (hip_nvfp4_timing_enabled &&
        hip_success(hipEventRecord(hip_nvfp4_event_qdq),
                     "hipEventRecord NVFP4 heterogeneous QDQ") != 0)
        return -1;
    hip_nvfp4_heterogeneous_warp<<<
        dim3((unsigned int)((max_rows + 3) / 4), (unsigned int)max_batch,
             (unsigned int)njobs), 128>>>(
        hip_device_desc, njobs, hip_nvfp4_xdq, hip_y);
    if (hip_success(hipGetLastError(),
                     "NVFP4 heterogeneous projection launch") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled &&
        hip_success(hipEventRecord(hip_nvfp4_event_projection),
                     "hipEventRecord NVFP4 heterogeneous projection") != 0)
        return -1;
    if (hip_success(hipMemcpy(hip_host_y, hip_y,
                                output_floats * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy NVFP4 heterogeneous output") != 0)
        return -1;
    if (hip_nvfp4_timing_enabled) {
        if (hip_event_elapsed_ns(hip_nvfp4_event_start,
                                  hip_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            hip_event_elapsed_ns(hip_nvfp4_event_qdq,
                                  hip_nvfp4_event_projection,
                                  &telemetry.projection_ns) != 0 ||
            hip_nvfp4_d2h_timing(hip_nvfp4_event_projection,
                                   &telemetry) != 0)
            return -1;
    }
    for (int job = 0; job < njobs; job++)
        memcpy(outputs[job], hip_host_y + hip_job_output_offset[job],
               (size_t)hip_job_rows[job] * sizeof(float));
    if (saw_trunk) hip_stats.trunk_batches++;
    if (saw_expert) hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += logical_jobs;
    hip_nvfp4_telemetry_publish(&telemetry, UINT64_C(2));
    hip_path_record(HIP_PATH_NVFP4_HET_PROJECT, logical_jobs,
        (uint64_t)output_floats, UINT64_C(1), UINT64_C(1));
    return 0;

miss:
    if (hip_mapped_only) hip_stats.mapped_only_misses++;
    return -1;
}

extern "C" int salt_gpu_nvfp4_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    int R, int C, int njobs) {
    return hip_run_nvfp4_batch(
        keys, inputs, outputs, batches, NULL, R, C, njobs);
}

extern "C" int salt_gpu_nvfp4_mixed_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    const int *rows, int C, int njobs) {
    if (!rows) return -1;
    return hip_run_nvfp4_batch(
        keys, inputs, outputs, batches, rows, 0, C, njobs);
}

static int hip_run_batch(const uint32_t *const *vals,
    const float *const *xs, float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs, int bits) {
    if (!vals || !xs || !ys || !ids || !Rj || C < 1 || njobs < 1)
        return -1;
    for (int job = 0; job < njobs; job++)
        if (hip_run(ids[job], vals[job], bits, Rj[job], C,
                     xs[job], ys[job]) != 0)
            return -1;
    return 0;
}

static int hip_run_heterogeneous_q4(
    const uint32_t *const *vals, const uint16_t *const *scales,
    const uint16_t *const *biases, const float *const *xs,
    float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs) {
    size_t input_floats = 0, output_floats = 0;
    uint64_t logical_input_floats = 0, reused_inputs = 0;
    const float *last_input = NULL;
    uint32_t last_input_offset = 0;
    int last_input_run = 0;
    int descriptor_count = 0, max_rows = 0, max_groups = 0;
    int max_warp_groups = 0;
    int all_single = 1;
    int saw_trunk = 0, saw_expert = 0;
    enum HipPathId path = HIP_PATH_NONE;
    uint64_t h2d_start, h2d_end, launch_end, completion_end, scatter_end;
    if (!hip_ready || !vals || !scales || !biases || !xs || !ys ||
        !ids || !Rj || C < 1 || C > HIP_MAX_C || njobs < 1 ||
        njobs > HIP_MAX_BATCH_JOBS) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "gpu-hip-hetero: arguments C=%d jobs=%d\n",
                    C, njobs);
        return -1;
    }
    for (int job = 0; job < njobs;) {
        HipTensor *tensor = hip_tensor(ids[job]);
        HipResource *resource;
        unsigned char *device;
        uint64_t voff, soff, boff, auxoff;
        int run = 1;
        int reuse_input;
        if (!tensor || tensor->bits != 4 || tensor->rows != Rj[job] ||
            tensor->cols != C || !vals[job] || !scales[job] ||
            !biases[job] || !xs[job] || !ys[job]) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-hip-hetero: tensor job=%d tensor=%p R=%d C=%d\n",
                    job, (void *)tensor, Rj[job], C);
            goto miss;
        }
        if (hip_tensor_physical(tensor, &resource, &device,
                &voff, &soff, &boff, &auxoff) != 0 ||
            resource->host + tensor->value_offset !=
                (const unsigned char *)(const void *)vals[job] ||
            resource->host + tensor->scale_offset !=
                (const unsigned char *)(const void *)scales[job] ||
            resource->host + tensor->bias_offset !=
                (const unsigned char *)(const void *)biases[job]) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-hip-hetero: resource job=%d kind=%u id=%u\n",
                    job, tensor->kind, tensor->resource_id);
            goto miss;
        }
        while (job + run < njobs && ids[job + run] == ids[job] &&
               vals[job + run] == vals[job] &&
               scales[job + run] == scales[job] &&
               biases[job + run] == biases[job] &&
               Rj[job + run] == Rj[job] &&
               xs[job + run] == xs[job] + (size_t)run * (size_t)C &&
               ys[job + run] == ys[job] +
                                  (size_t)run * (size_t)Rj[job])
            run++;
        reuse_input = last_input == xs[job] && last_input_run == run;
        if ((!reuse_input &&
             (size_t)run > (HIP_MAX_OUTPUTS - input_floats) / (size_t)C) ||
            (size_t)run > (HIP_MAX_OUTPUTS - output_floats) /
                              (size_t)Rj[job] ||
            descriptor_count >= HIP_MAX_BATCH_JOBS) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-hip-hetero: capacity job=%d run=%d in=%zu out=%zu desc=%d\n",
                    job, run, input_floats, output_floats,
                    descriptor_count);
            return -1;
        }
        HipBatchDesc *descriptor = &hip_host_desc[descriptor_count++];
        descriptor->resource = device;
        descriptor->value_offset = voff;
        descriptor->scale_offset = soff;
        descriptor->bias_offset = boff;
        descriptor->aux_offset = 0;
        descriptor->input_offset = reuse_input
            ? last_input_offset : (uint32_t)input_floats;
        descriptor->output_offset = (uint32_t)output_floats;
        descriptor->rows = Rj[job];
        descriptor->cols = C;
        descriptor->batch = run;
        logical_input_floats += (uint64_t)(uint32_t)run * (uint32_t)C;
        if (reuse_input) {
            reused_inputs++;
        } else {
            last_input = xs[job];
            last_input_run = run;
            last_input_offset = (uint32_t)input_floats;
            memcpy(hip_host_x + input_floats, xs[job],
                   (size_t)run * (size_t)C * sizeof(float));
            input_floats += (size_t)run * (size_t)C;
        }
        for (int item = 0; item < run; item++) {
            hip_job_output_offset[job + item] =
                (uint32_t)(output_floats +
                           (size_t)item * (size_t)Rj[job]);
            hip_job_rows[job + item] = Rj[job];
        }
        output_floats += (size_t)run * (size_t)Rj[job];
        if (Rj[job] > max_rows) max_rows = Rj[job];
        int groups = (run + 31) / 32;
        if (groups > max_groups) max_groups = groups;
        int warp_groups = (run + 511) / 512;
        if (warp_groups > max_warp_groups) max_warp_groups = warp_groups;
        if (run != 1) all_single = 0;
        if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) saw_trunk = 1;
        if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER) saw_expert = 1;
        job += run;
    }
    if (descriptor_count < 1 || max_rows < 1 || max_groups < 1)
        return -1;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "gpu-hip-hetero: submit jobs=%d desc=%d C=%d rows=%d groups=%d in=%zu out=%zu\n",
            njobs, descriptor_count, C, max_rows, max_groups,
            input_floats, output_floats);
    h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(hip_x, hip_host_x,
                                input_floats * sizeof(float),
                                hipMemcpyHostToDevice),
                     "hipMemcpy heterogeneous input") != 0 ||
        hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
                                (size_t)descriptor_count *
                                    sizeof(HipBatchDesc),
                                hipMemcpyHostToDevice),
                     "hipMemcpy heterogeneous descriptors") != 0)
        return -1;
    h2d_end = hip_host_now_ns();
    if (all_single) {
        path = HIP_PATH_Q4_HET_WARP;
        hip_q4_heterogeneous_warp<<<
            dim3((unsigned int)((max_rows + 3) / 4), 1u,
                 (unsigned int)descriptor_count), 128>>>(
            hip_device_desc, descriptor_count, hip_x, hip_y,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    } else if (descriptor_count <= 3) {
        path = HIP_PATH_Q4_HET_WARP_BATCH;
        hip_q4_heterogeneous_warp_batch<<<
            dim3((unsigned int)max_rows, (unsigned int)max_warp_groups,
                 (unsigned int)descriptor_count), 1024>>>(
            hip_device_desc, descriptor_count, hip_x, hip_y);
    } else {
        path = HIP_PATH_Q4_HET_TOKEN;
        hip_q4_heterogeneous<<<dim3((unsigned int)max_rows,
                                     (unsigned int)max_groups,
                                     (unsigned int)descriptor_count), 32>>>(
            hip_device_desc, descriptor_count, hip_x, hip_y,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    }
    if (hip_success(hipGetLastError(), "heterogeneous Q4 launch") != 0)
        return -1;
    launch_end = hip_host_now_ns();
    if (hip_success(hipMemcpy(hip_host_y, hip_y,
                                output_floats * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy heterogeneous output") != 0)
        return -1;
    completion_end = hip_host_now_ns();
    for (int job = 0; job < njobs; job++)
        memcpy(ys[job], hip_host_y + hip_job_output_offset[job],
               (size_t)hip_job_rows[job] * sizeof(float));
    scatter_end = hip_host_now_ns();
    if (saw_trunk) hip_stats.trunk_batches++;
    if (saw_expert) hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += (uint64_t)njobs;
    hip_q4_hetero_logical_input_bytes += logical_input_floats * sizeof(float);
    hip_q4_hetero_transfer_input_bytes +=
        (uint64_t)input_floats * sizeof(float);
    hip_q4_hetero_output_bytes += (uint64_t)output_floats * sizeof(float);
    hip_q4_hetero_descriptor_bytes +=
        (uint64_t)(uint32_t)descriptor_count * sizeof(HipBatchDesc);
    hip_q4_hetero_reused_inputs += reused_inputs;
    hip_q4_hetero_kernel_launches++;
    hip_q4_hetero_completion_fences++;
    hip_u64_add_saturating(&hip_q4_hetero_h2d_ns, h2d_end - h2d_start);
    hip_u64_add_saturating(&hip_q4_hetero_launch_ns,
        launch_end - h2d_end);
    hip_u64_add_saturating(&hip_q4_hetero_completion_ns,
        completion_end - launch_end);
    hip_u64_add_saturating(&hip_q4_hetero_scatter_ns,
        scatter_end - completion_end);
    hip_path_record(path, (uint64_t)(uint32_t)njobs,
        (uint64_t)output_floats, UINT64_C(1), UINT64_C(1));
    return 0;

miss:
    if (hip_mapped_only) hip_stats.mapped_only_misses++;
    return -1;
}

extern "C" int salt_gpu_q4_batch(const uint32_t *const *vals,
    const uint16_t *const *scales, const uint16_t *const *biases,
    const float *const *xs, float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs) {
    return hip_run_heterogeneous_q4(vals, scales, biases, xs, ys,
                                     ids, Rj, C, njobs);
}

extern "C" int salt_gpu_q8_batch(const uint32_t *const *vals,
    const uint16_t *const *scales, const uint16_t *const *biases,
    const float *const *xs, float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs) {
    (void)scales; (void)biases;
    if (njobs > 0 && vals && xs && ys && ids && Rj) {
        int same = 1;
        for (int job = 1; job < njobs; job++)
            if (vals[job] != vals[0] || ids[job] != ids[0] ||
                Rj[job] != Rj[0]) {
                same = 0;
                break;
            }
        if (same)
            return hip_run_same_weight_batch(ids[0], vals[0], 8,
                                               Rj[0], C, njobs, xs, ys);
    }
    return hip_run_batch(vals, xs, ys, ids, Rj, C, njobs, 8);
}

extern "C" void salt_gpu_set_mapped_only(int on) {
    hip_mapped_only = on ? 1 : 0;
}
extern "C" int salt_gpu_mapped_only(void) { return hip_mapped_only; }
extern "C" void salt_gpu_batch_stats_get(SaltGpuBatchStats *stats) {
    if (!stats) return;
    stats->arena_batches = hip_stats.arena_batches;
    stats->trunk_batches = hip_stats.trunk_batches;
    stats->expert_layer_batches = hip_stats.expert_layer_batches;
    stats->mapped_only_misses = hip_stats.mapped_only_misses;
    stats->direct_output_batches = hip_stats.direct_output_batches;
    stats->direct_output_jobs = hip_stats.direct_output_jobs;
}
extern "C" int salt_gpu_batch_stats_get_v2(
        SaltGpuBatchStatsV2 *stats, size_t stats_size) {
    if (!stats || stats_size != sizeof *stats) return -1;
    *stats = hip_stats;
    return 0;
}

extern "C" int salt_gpu_resident_trunk_map(const void *base, size_t nbytes) {
    (void)base;
    (void)nbytes;
    return -1;
}
extern "C" int salt_gpu_resident_trunk_unmap(void) {
    return salt_gpu_weight_resource_unbind(SALT_GPU_RESOURCE_TRUNK, 0);
}
extern "C" int salt_gpu_expert_resource_bind(uint32_t id,
    const void *base, size_t nbytes) {
    (void)id;
    (void)base;
    (void)nbytes;
    return -1;
}
extern "C" int salt_gpu_expert_resource_slot(uint32_t id, const void *key,
    const uint32_t *vals, const uint16_t *scales, const uint16_t *biases,
    int R, int C) {
    return salt_gpu_weight_resource_slot(SALT_GPU_RESOURCE_EXPERT_LAYER,
        id, key, vals, scales, biases, 4, R, C);
}
extern "C" int salt_gpu_resident_slot(const void *key,
    const uint32_t *vals, const uint16_t *scales, const uint16_t *biases,
    int R, int C) {
    for (int i = 0; i < HIP_MAX_RESOURCES; i++) {
        uint64_t offset = 0;
        if (hip_resources[i].used &&
            hip_resources[i].kind == SALT_GPU_RESOURCE_EXPERT_LAYER &&
            hip_range(&hip_resources[i], vals, 1, &offset) == 0)
            return salt_gpu_expert_resource_slot(hip_resources[i].id,
                key, vals, scales, biases, R, C);
    }
    return -1;
}
extern "C" int salt_gpu_expert_resource_unbind(uint32_t id) {
    return salt_gpu_weight_resource_unbind(SALT_GPU_RESOURCE_EXPERT_LAYER, id);
}
extern "C" int salt_gpu_resident_pool_map(const void *base, size_t nbytes) {
    (void)base;
    (void)nbytes;
    return -1;
}

extern "C" int salt_gpu_proj_batch(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C, int B,
    const float *const *xs, float *const *ys, const void *id) {
    (void)scales; (void)biases;
    return hip_run_same_weight_batch(id, vals, 4, R, C, B, xs, ys);
}

static int hip_shared_pointer(const void *pointer, size_t nbytes,
        int *slot_out, size_t *offset_out) {
    uintptr_t address;
    int found = -1;
    size_t found_offset = 0;
    if (!pointer || nbytes < 1 || !slot_out || !offset_out) return -1;
    address = (uintptr_t)pointer;
    for (int slot = 0; slot < HIP_SHARED_SLOTS; slot++) {
        HipSharedSlot *entry = &hip_shared_slots[slot];
        uintptr_t base;
        size_t offset;
        if (!entry->contents || !entry->device || entry->nbytes < 1) continue;
        base = (uintptr_t)entry->contents;
        if (address < base || address - base > SIZE_MAX) continue;
        offset = (size_t)(address - base);
        if (offset > entry->nbytes || nbytes > entry->nbytes - offset) continue;
        if (found >= 0) return -1;
        found = slot;
        found_offset = offset;
    }
    if (found < 0) return -1;
    *slot_out = found;
    *offset_out = found_offset;
    return 0;
}

extern "C" int salt_gpu_shared_buffer_alloc(SaltGpuSharedBuffer *buffer,
                                              size_t nbytes) {
    void *host = NULL, *device = NULL;
    int slot = -1;
    if (!buffer || !hip_ready || nbytes < 1) return -1;
    for (int index = 0; index < HIP_SHARED_SLOTS; index++)
        if (!hip_shared_slots[index].contents) {
            slot = index;
            break;
        }
    if (slot < 0) return -1;
    memset(buffer, 0, sizeof *buffer);
    if (
        hip_success(hipHostMalloc(&host, nbytes, hipHostMallocMapped),
                     "hipHostMalloc shared") != 0 ||
        hip_success(hipHostGetDevicePointer(&device, host, 0),
                     "hipHostGetDevicePointer shared") != 0) {
        if (host) hipHostFree(host);
        return -1;
    }
    buffer->contents = host;
    buffer->nbytes = nbytes;
    buffer->backend = device;
    hip_shared_slots[slot].contents = host;
    hip_shared_slots[slot].device = device;
    hip_shared_slots[slot].nbytes = nbytes;
    return 0;
}
static int hip_exact_view(const SaltGpuSharedBuffer *buffer,
                           size_t offset, size_t elements,
                           void **device_out) {
    size_t capacity;
    if (!buffer || !buffer->contents || !buffer->backend || !device_out ||
        buffer->nbytes < sizeof(float))
        return -1;
    capacity = buffer->nbytes / sizeof(float);
    if (offset > capacity || elements > capacity - offset) return -1;
    *device_out = (unsigned char *)buffer->backend +
        offset * sizeof(float);
    return 0;
}

extern "C" int salt_gpu_exact_cells(const SaltGpuExactCell *cells, int count) {
    int host_status = 0;
    if (!hip_ready || !cells || count < 1 || !hip_attention_status ||
        hip_success(hipMemset(hip_attention_status, 0, sizeof(int)),
                     "hipMemset exact-cell status") != 0)
        return -1;
    for (int index = 0; index < count; index++) {
        const SaltGpuExactCell *cell = &cells[index];
        void *views[8] = {0};
        if (cell->n == 0) return -1;
        switch (cell->kind) {
        case SALT_GPU_EXACT_RMSNORM:
            if (cell->aux > 1u ||
                hip_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                hip_exact_view(cell->views[1], cell->offsets[1],
                    cell->aux ? cell->n : 1u, &views[1]) != 0 ||
                hip_exact_view(cell->views[2], cell->offsets[2],
                    cell->n, &views[2]) != 0)
                return -1;
            hip_text_rmsnorm_rows<<<1, 256>>>(
                (const float *)views[0], cell->n, (const float *)views[1],
                (float *)views[2], cell->n, 1u, cell->n, cell->eps,
                (int)cell->aux, hip_attention_status);
            break;
        case SALT_GPU_EXACT_TOPK:
            if (cell->aux == 0 || cell->aux > cell->n ||
                hip_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                hip_exact_view(cell->views[1], cell->offsets[1],
                    cell->n, &views[1]) != 0 ||
                hip_exact_view(cell->views[2], cell->offsets[2],
                    cell->aux, &views[2]) != 0 ||
                hip_exact_view(cell->views[3], cell->offsets[3],
                    cell->aux, &views[3]) != 0)
                return -1;
            hip_text_topk_rows<<<1, 1>>>(
                (const float *)views[0], (const float *)views[1],
                (int32_t *)views[2], (float *)views[3], 1u,
                cell->n, cell->aux, hip_attention_status);
            break;
        case SALT_GPU_EXACT_RESIDUAL_POSTNORM:
            for (int view = 0; view < 4; view++)
                if (hip_exact_view(cell->views[view], cell->offsets[view],
                        cell->n, &views[view]) != 0)
                    return -1;
            hip_text_residual_postnorm_rows<<<1, 256>>>(
                (const float *)views[0], (float *)views[1],
                (const float *)views[2], (float *)views[3],
                1u, cell->n, cell->eps, hip_attention_status);
            break;
        case SALT_GPU_EXACT_PARALLEL_COMBINE:
            for (int view = 0; view < 8; view++) {
                size_t elements = view == 6 ? (size_t)cell->n * 2u : cell->n;
                if (elements < cell->n ||
                    hip_exact_view(cell->views[view], cell->offsets[view],
                        elements, &views[view]) != 0)
                    return -1;
            }
            hip_text_parallel_combine_rows<<<1, 256>>>(
                (const float *)views[0], (const float *)views[1],
                (const float *)views[2], (const float *)views[3],
                (const float *)views[4], (const float *)views[5],
                (float *)views[6], (float *)views[6] + cell->n,
                (float *)views[7], 1u, cell->n, cell->eps,
                cell->scalar, NULL, hip_attention_status);
            break;
        case SALT_GPU_EXACT_SOFTCAP:
            if (!(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
                hip_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0)
                return -1;
            hip_text_softcap<<<(cell->n + 255u) / 256u, 256>>>(
                (float *)views[0], cell->n, cell->scalar,
                hip_attention_status);
            break;
        case SALT_GPU_EXACT_ROUTER_INPUT:
            if (hip_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                hip_exact_view(cell->views[1], cell->offsets[1],
                    cell->n, &views[1]) != 0 ||
                hip_exact_view(cell->views[2], cell->offsets[2],
                    cell->n, &views[2]) != 0)
                return -1;
            hip_text_router_input_rows<<<1, 256>>>(
                (const float *)views[0], (const float *)views[1],
                (float *)views[2], 1u, cell->n, cell->eps,
                cell->scalar, hip_attention_status);
            break;
        default:
            return -1;
        }
        if (hip_success(hipGetLastError(), "exact-cell launch") != 0)
            return -1;
    }
    if (hip_success(hipMemcpy(&host_status, hip_attention_status,
            sizeof host_status, hipMemcpyDeviceToHost),
            "hipMemcpy exact-cell status") != 0)
        return -1;
    return host_status ? -1 : 0;
}
extern "C" int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer) {
    int rc;
    int slot = -1;
    if (!buffer) return -1;
    if (!buffer->contents && !buffer->backend && buffer->nbytes == 0) return 0;
    for (int index = 0; index < HIP_SHARED_SLOTS; index++)
        if (hip_shared_slots[index].contents == buffer->contents &&
            hip_shared_slots[index].device == buffer->backend &&
            hip_shared_slots[index].nbytes == buffer->nbytes) {
            slot = index;
            break;
        }
    if (!buffer->contents || !buffer->backend || buffer->nbytes < 1 ||
        slot < 0 ||
        hip_success(hipDeviceSynchronize(),
                     "hipDeviceSynchronize shared free") != 0)
        return -1;
    if (buffer->contents == hip_attention_kv_host) {
        if (!hip_attention_kv || buffer->nbytes != hip_attention_kv_nbytes ||
            hip_success(hipFree(hip_attention_kv),
                         "hipFree attention KV shared owner") != 0)
            return -1;
        hip_attention_kv = NULL;
        hip_attention_kv_nbytes = 0;
        hip_attention_kv_host = NULL;
        hip_attention_kv_sync_count = 0;
        memset(hip_attention_kv_sync, 0, sizeof hip_attention_kv_sync);
        memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    }
    rc = hip_success(hipHostFree(buffer->contents), "hipHostFree shared");
    if (rc == 0) memset(&hip_shared_slots[slot], 0, sizeof hip_shared_slots[slot]);
    memset(buffer, 0, sizeof *buffer);
    return rc;
}

static int hip_shared_host_float_range(const SaltGpuSharedBuffer *buffer,
                                        size_t offset, size_t count,
                                        float **host_out) {
    size_t capacity;
    if (!buffer || !buffer->contents || !buffer->backend || !host_out ||
        buffer->nbytes < sizeof(float))
        return -1;
    capacity = buffer->nbytes / sizeof(float);
    if (offset > capacity || count > capacity - offset) return -1;
    *host_out = (float *)buffer->contents + offset;
    return 0;
}

extern "C" int salt_gpu_attention_prepare(const SaltGpuSharedBuffer *kv) {
    if (!hip_ready || !kv || !kv->contents || !kv->backend || kv->nbytes < 1)
        return -1;
    if (hip_attention_kv)
        return hip_attention_kv_host == kv->contents &&
               hip_attention_kv_nbytes == kv->nbytes ? 0 : -1;
    if (hip_success(hipMalloc((void **)&hip_attention_kv, kv->nbytes),
                     "hipMalloc attention KV") != 0 ||
        hip_success(hipMemset(hip_attention_kv, 0, kv->nbytes),
                     "hipMemset attention KV") != 0) {
        if (hip_attention_kv) hipFree(hip_attention_kv);
        hip_attention_kv = NULL;
        return -1;
    }
    hip_attention_kv_host = kv->contents;
    hip_attention_kv_nbytes = kv->nbytes;
    hip_attention_kv_sync_count = 0;
    memset(hip_attention_kv_sync, 0, sizeof hip_attention_kv_sync);
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    return 0;
}

static HipAttentionKvSync *hip_attention_sync_slot(
    size_t key_offset, size_t value_offset, int kv_stride) {
    size_t row_capacity;
    if (kv_stride < 1 || value_offset <= key_offset ||
        (value_offset - key_offset) % (size_t)kv_stride != 0)
        return NULL;
    row_capacity = (value_offset - key_offset) / (size_t)kv_stride;
    if (row_capacity == 0 || row_capacity > (size_t)INT_MAX)
        return NULL;
    for (int i = 0; i < hip_attention_kv_sync_count; i++)
        if (hip_attention_kv_sync[i].key_offset == key_offset &&
            hip_attention_kv_sync[i].value_offset == value_offset)
            return hip_attention_kv_sync[i].kv_stride == kv_stride &&
                   hip_attention_kv_sync[i].row_capacity == row_capacity
                ? &hip_attention_kv_sync[i] : NULL;
    if (hip_attention_kv_sync_count >=
            (int)(sizeof hip_attention_kv_sync /
                  sizeof hip_attention_kv_sync[0]))
        return NULL;
    HipAttentionKvSync *sync =
        &hip_attention_kv_sync[hip_attention_kv_sync_count++];
    sync->key_offset = key_offset;
    sync->value_offset = value_offset;
    sync->row_capacity = row_capacity;
    sync->kv_stride = kv_stride;
    sync->device_positions = 0;
    sync->host_positions = 0;
    return sync;
}

extern "C" int salt_gpu_attention_bind_kv(
        size_t key_float_offset, size_t value_float_offset, int kv_stride) {
    return hip_ready && hip_attention_kv &&
        hip_attention_sync_slot(
            key_float_offset, value_float_offset, kv_stride) ? 0 : -1;
}

static int hip_attention_state_range(
        const SaltGpuSharedBuffer *kv, const HipAttentionKvSync *sync,
        int position, size_t *offset_count) {
    size_t capacity, count;
    if (!kv || !sync || !offset_count || kv->contents != hip_attention_kv_host ||
        kv->nbytes != hip_attention_kv_nbytes || position < 0 ||
        sync->kv_stride < 1 || sync->row_capacity == 0)
        return -1;
    capacity = kv->nbytes / sizeof(float);
    count = (size_t)position < sync->row_capacity
        ? (size_t)position : sync->row_capacity;
    if (count > SIZE_MAX / (size_t)sync->kv_stride)
        return -1;
    count *= (size_t)sync->kv_stride;
    if (sync->key_offset > capacity || count > capacity - sync->key_offset ||
        sync->value_offset > capacity || count > capacity - sync->value_offset)
        return -1;
    *offset_count = count;
    return 0;
}

extern "C" int salt_gpu_attention_materialize(
        SaltGpuSharedBuffer *kv, int position, uint64_t *copied_bytes) {
    uint64_t total = 0;
    if (copied_bytes) *copied_bytes = 0;
    if (!hip_ready || !kv || !copied_bytes || position < 0 ||
        hip_attention_kv_sync_count < 1 ||
        hip_success(hipDeviceSynchronize(),
            "hipDeviceSynchronize materialize KV") != 0)
        return -1;
    for (int i = 0; i < hip_attention_kv_sync_count; i++) {
        HipAttentionKvSync *sync = &hip_attention_kv_sync[i];
        size_t device_count, delta, start;
        int device_position = position < sync->device_positions
            ? position : sync->device_positions;
        if (hip_attention_state_range(
                kv, sync, device_position, &device_count) != 0)
            return -1;
        if (sync->host_positions == device_position)
            start = device_count;
        else if ((size_t)device_position > sync->row_capacity ||
                 (size_t)sync->host_positions > sync->row_capacity ||
                 sync->host_positions > device_position)
            start = 0;
        else
            start = (size_t)sync->host_positions * (size_t)sync->kv_stride;
        delta = device_count - start;
        if (delta &&
            (hip_success(hipMemcpy(
                (float *)kv->contents + sync->key_offset + start,
                hip_attention_kv + sync->key_offset + start,
                delta * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy materialize KV keys") != 0 ||
             hip_success(hipMemcpy(
                (float *)kv->contents + sync->value_offset + start,
                hip_attention_kv + sync->value_offset + start,
                delta * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy materialize KV values") != 0))
            return -1;
        if ((uint64_t)delta > (UINT64_MAX - total) / (2u * sizeof(float)))
            return -1;
        total += (uint64_t)delta * 2u * sizeof(float);
        sync->host_positions = device_position;
    }
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    *copied_bytes = total;
    return 0;
}

extern "C" int salt_gpu_attention_import(
        SaltGpuSharedBuffer *kv, int position, uint64_t *copied_bytes) {
    uint64_t total = 0;
    if (copied_bytes) *copied_bytes = 0;
    if (!hip_ready || !kv || !copied_bytes || position < 0 ||
        hip_attention_kv_sync_count < 1)
        return -1;
    for (int i = 0; i < hip_attention_kv_sync_count; i++) {
        HipAttentionKvSync *sync = &hip_attention_kv_sync[i];
        size_t count;
        if (hip_attention_state_range(kv, sync, position, &count) != 0 ||
            (count &&
             (hip_success(hipMemcpy(
                hip_attention_kv + sync->key_offset,
                (float *)kv->contents + sync->key_offset,
                count * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy import KV keys") != 0 ||
              hip_success(hipMemcpy(
                hip_attention_kv + sync->value_offset,
                (float *)kv->contents + sync->value_offset,
                count * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy import KV values") != 0)))
            return -1;
        if ((uint64_t)count > (UINT64_MAX - total) / (2u * sizeof(float)))
            return -1;
        total += (uint64_t)count * 2u * sizeof(float);
        sync->device_positions = position;
        sync->host_positions = position;
    }
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    *copied_bytes = total;
    return 0;
}

extern "C" int salt_gpu_attention_rewind(int position) {
    if (!hip_ready || position < 0 || hip_attention_kv_sync_count < 1)
        return -1;
    for (int i = 0; i < hip_attention_kv_sync_count; i++)
        if (position > hip_attention_kv_sync[i].device_positions)
            return -1;
    if (hip_success(hipDeviceSynchronize(),
            "hipDeviceSynchronize attention rewind") != 0)
        return -1;
    for (int i = 0; i < hip_attention_kv_sync_count; i++) {
        hip_attention_kv_sync[i].device_positions = position;
        if (hip_attention_kv_sync[i].host_positions > position)
            hip_attention_kv_sync[i].host_positions = position;
    }
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    return 0;
}

extern "C" int salt_gpu_attention_transform(
    int n_heads, int n_kv_heads, int head_dim, int rope_dim,
    int start_position, int batch, int query_stride, int kv_stride, float eps,
    SaltGpuSharedBuffer *queries, size_t query_float_offset,
    SaltGpuSharedBuffer *keys, size_t key_float_offset,
    SaltGpuSharedBuffer *values, size_t value_float_offset,
    const float *q_weight, const float *k_weight,
    const float *cosines, const float *sines, int rope_pairs,
    SaltGpuSharedBuffer *kv, size_t key_cache_float_offset,
    size_t value_cache_float_offset) {
    float *host_queries, *host_keys, *host_values;
    float *host_key_cache, *host_value_cache;
    HipAttentionKvSync *sync;
    uint64_t query_count, new_kv_count, cache_count, factor_count;
    uint64_t metadata_offset, metadata_count;
    size_t cache_rows, prefix_count, prefix_start;
    int copy_start, deferred_publish;
    memset(&hip_attention_stage, 0, sizeof hip_attention_stage);
    if (!hip_ready || n_heads < 1 || n_kv_heads < 1 ||
        n_heads % n_kv_heads != 0 || head_dim < 2 || head_dim > 512 ||
        head_dim % 2 != 0 || rope_dim < 2 || rope_dim > head_dim ||
        rope_dim % 2 != 0 || rope_pairs != rope_dim / 2 ||
        start_position < 0 || batch < 1 || start_position > INT_MAX - batch ||
        query_stride != n_heads * head_dim ||
        kv_stride != n_kv_heads * head_dim || !(eps >= 0.0f) ||
        !q_weight || !k_weight || !cosines || !sines)
        return -1;
    query_count = (uint64_t)(uint32_t)batch * (uint32_t)query_stride;
    new_kv_count = (uint64_t)(uint32_t)batch * (uint32_t)kv_stride;
    sync = hip_attention_sync_slot(
        key_cache_float_offset, value_cache_float_offset, kv_stride);
    if (!sync) return -1;
    deferred_publish =
        (size_t)(start_position % (int)sync->row_capacity) +
            (size_t)batch > sync->row_capacity;
    cache_rows = (size_t)(start_position + batch) < sync->row_capacity
        ? (size_t)(start_position + batch) : sync->row_capacity;
    cache_count = (uint64_t)cache_rows * (uint32_t)kv_stride;
    factor_count = (uint64_t)(uint32_t)batch * (uint32_t)rope_pairs;
    metadata_offset = query_count;
    metadata_count = 2u * (uint32_t)head_dim + 2u * factor_count;
    if (query_count > HIP_MAX_OUTPUTS ||
        new_kv_count > HIP_MAX_OUTPUTS / 2u ||
        metadata_offset > HIP_MAX_OUTPUTS ||
        metadata_count > HIP_MAX_OUTPUTS - metadata_offset ||
        cache_count > SIZE_MAX || factor_count > SIZE_MAX ||
        !hip_attention_kv || hip_attention_kv_host != kv->contents ||
        hip_attention_kv_nbytes != kv->nbytes ||
        hip_shared_host_float_range(queries, query_float_offset,
            (size_t)query_count, &host_queries) != 0 ||
        hip_shared_host_float_range(keys, key_float_offset,
            (size_t)new_kv_count, &host_keys) != 0 ||
        hip_shared_host_float_range(values, value_float_offset,
            (size_t)new_kv_count, &host_values) != 0 ||
        hip_shared_host_float_range(kv, key_cache_float_offset,
            (size_t)cache_count, &host_key_cache) != 0 ||
        hip_shared_host_float_range(kv, value_cache_float_offset,
            (size_t)cache_count, &host_value_cache) != 0)
        return -1;
    copy_start = start_position < sync->device_positions
        ? 0 : sync->device_positions;
    prefix_start = ((size_t)start_position > sync->row_capacity ||
                    (size_t)copy_start > sync->row_capacity)
        ? 0u : (size_t)copy_start;
    cache_rows = (size_t)start_position < sync->row_capacity
        ? (size_t)start_position : sync->row_capacity;
    prefix_count = cache_rows - prefix_start;
    if (copy_start < start_position && prefix_count != 0 &&
        (hip_success(hipMemcpy(
            hip_attention_kv + key_cache_float_offset +
                prefix_start * (size_t)kv_stride,
            host_key_cache + prefix_start * (size_t)kv_stride,
            prefix_count * (size_t)kv_stride *
                sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform prefix keys") != 0 ||
         hip_success(hipMemcpy(
            hip_attention_kv + value_cache_float_offset +
                prefix_start * (size_t)kv_stride,
            host_value_cache + prefix_start * (size_t)kv_stride,
            prefix_count * (size_t)kv_stride *
                sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform prefix values") != 0))
        return -1;
    if (copy_start < start_position)
        sync->host_positions = start_position;
    float *device_q_weight = hip_x + metadata_offset;
    float *device_k_weight = device_q_weight + head_dim;
    float *device_cosines = device_k_weight + head_dim;
    float *device_sines = device_cosines + factor_count;
    if (hip_success(hipMemcpy(hip_x, host_queries,
            (size_t)query_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform queries") != 0 ||
        hip_success(hipMemcpy(hip_y, host_keys,
            (size_t)new_kv_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform keys") != 0 ||
        hip_success(hipMemcpy(hip_y + new_kv_count, host_values,
            (size_t)new_kv_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform values") != 0 ||
        hip_success(hipMemcpy(device_q_weight, q_weight,
            (size_t)head_dim * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform q weight") != 0 ||
        hip_success(hipMemcpy(device_k_weight, k_weight,
            (size_t)head_dim * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform k weight") != 0 ||
        hip_success(hipMemcpy(device_cosines, cosines,
            (size_t)factor_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform cosines") != 0 ||
        hip_success(hipMemcpy(device_sines, sines,
            (size_t)factor_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy transform sines") != 0 ||
        hip_success(hipMemset(hip_attention_status, 0, sizeof(int)),
            "hipMemset transform status") != 0)
        return -1;
    unsigned int tasks = (unsigned int)((uint64_t)(uint32_t)batch *
        (uint32_t)(n_heads + 2 * n_kv_heads));
    hip_attention_transform_exact<<<tasks, 256>>>(
        n_heads, n_kv_heads, head_dim, rope_dim, start_position, batch,
        query_stride, kv_stride, (int)sync->row_capacity,
        !deferred_publish, eps, hip_x, hip_y,
        hip_y + new_kv_count, device_q_weight, device_k_weight,
        device_cosines, device_sines, rope_pairs,
        hip_attention_kv + key_cache_float_offset,
        hip_attention_kv + value_cache_float_offset, hip_attention_status);
    if (hip_success(hipGetLastError(), "attention transform launch") != 0)
        return -1;
    if (hip_pageable_mmap) {
        int failure = 0;
        if (hip_success(hipMemcpy(&failure, hip_attention_status,
                sizeof failure, hipMemcpyDeviceToHost),
                "hipMemcpy pageable transform status") != 0 || failure)
            return -1;
    }
    sync->device_positions = start_position + batch;
    hip_attention_stage.query_host = queries->contents;
    hip_attention_stage.query_offset = query_float_offset;
    hip_attention_stage.key_offset = key_cache_float_offset;
    hip_attention_stage.value_offset = value_cache_float_offset;
    hip_attention_stage.start = start_position;
    hip_attention_stage.batch = batch;
    hip_attention_stage.query_stride = query_stride;
    hip_attention_stage.kv_stride = kv_stride;
    hip_attention_stage.deferred_publish = deferred_publish;
    hip_attention_stage.valid = 1;
    hip_path_record(HIP_PATH_ATTN_TRANSFORM, (uint64_t)(uint32_t)batch,
        (uint64_t)tasks, UINT64_C(1), UINT64_C(0));
    return 0;
}

extern "C" int salt_gpu_attention_batch(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const SaltGpuSharedBuffer *queries, size_t query_float_offset,
    const SaltGpuSharedBuffer *kv, size_t key_float_offset,
    size_t value_float_offset, int start_position, int batch,
    int query_stride, int kv_stride, SaltGpuSharedBuffer *outputs,
    size_t output_float_offset) {
    float *host_queries, *host_keys, *host_values, *host_outputs;
    uint64_t query_count, kv_count, output_count, tasks, new_kv_count;
    HipAttentionKvSync *sync = NULL;
    int staged, deferred_publish;
    int failure = 0;
    size_t kv_rows, score_rows, score_bytes;
    if (!hip_ready || (full_attention != 0 && full_attention != 1) ||
        n_heads < 1 || n_kv_heads < 1 || n_heads % n_kv_heads != 0 ||
        head_dim < 1 || head_dim > 512 || (!full_attention && window < 1) ||
        start_position < 0 || batch < 1 ||
        start_position > INT_MAX - batch ||
        query_stride != n_heads * head_dim ||
        kv_stride != n_kv_heads * head_dim)
        return -1;
    tasks = (uint64_t)(uint32_t)batch * (uint32_t)n_heads;
    query_count = (uint64_t)(uint32_t)batch * (uint32_t)query_stride;
    new_kv_count = (uint64_t)(uint32_t)batch * (uint32_t)kv_stride;
    sync = hip_attention_sync_slot(
        key_float_offset, value_float_offset, kv_stride);
    if (!sync) return -1;
    kv_rows = (size_t)(start_position + batch) < sync->row_capacity
        ? (size_t)(start_position + batch) : sync->row_capacity;
    kv_count = (uint64_t)kv_rows * (uint32_t)kv_stride;
    staged = hip_attention_stage.valid &&
        hip_attention_stage.query_host == queries->contents &&
        hip_attention_stage.query_offset == query_float_offset &&
        hip_attention_stage.key_offset == key_float_offset &&
        hip_attention_stage.value_offset == value_float_offset &&
        hip_attention_stage.start == start_position &&
        hip_attention_stage.batch == batch &&
        hip_attention_stage.query_stride == query_stride &&
        hip_attention_stage.kv_stride == kv_stride;
    deferred_publish = staged && hip_attention_stage.deferred_publish;
    hip_attention_stage.valid = 0;
    output_count = query_count;
    if ((!staged &&
         (size_t)(start_position % (int)sync->row_capacity) +
             (size_t)batch > sync->row_capacity) ||
        tasks > UINT32_MAX || query_count > SIZE_MAX || kv_count > SIZE_MAX ||
        new_kv_count > HIP_MAX_OUTPUTS / 2u ||
        output_count > SIZE_MAX ||
        query_count > HIP_MAX_OUTPUTS || output_count > HIP_MAX_OUTPUTS ||
        (size_t)(start_position + batch) > SIZE_MAX / sizeof(float))
        return -1;
    score_rows = (size_t)(start_position + batch);
    if (!full_attention && score_rows > (size_t)window)
        score_rows = (size_t)window;
    score_bytes = score_rows * sizeof(float);
    if (score_bytes > 48u * 1024u || !hip_attention_kv ||
        hip_attention_kv_host != kv->contents ||
        hip_attention_kv_nbytes != kv->nbytes ||
        hip_shared_host_float_range(queries, query_float_offset,
            (size_t)query_count, &host_queries) != 0 ||
        hip_shared_host_float_range(kv, key_float_offset,
            (size_t)kv_count, &host_keys) != 0 ||
        hip_shared_host_float_range(kv, value_float_offset,
            (size_t)kv_count, &host_values) != 0 ||
        hip_shared_host_float_range(outputs, output_float_offset,
            (size_t)output_count, &host_outputs) != 0)
        return -1;
    if (!staged &&
        (hip_success(hipMemcpy(hip_x, host_queries,
            (size_t)query_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy attention queries") != 0 ||
         hip_success(hipMemcpy(
            hip_attention_kv + key_float_offset, host_keys,
            (size_t)kv_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy attention keys") != 0 ||
         hip_success(hipMemcpy(
            hip_attention_kv + value_float_offset, host_values,
            (size_t)kv_count * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy attention values") != 0))
        return -1;
    if (!staged &&
        hip_success(hipMemset(hip_attention_status, 0, sizeof(int)),
                     "hipMemset attention status") != 0)
        return -1;
    if (!staged)
        sync->host_positions = start_position + batch;
    sync->device_positions = start_position + batch;
    hip_attention_exact<<<(unsigned int)tasks, 64, score_bytes>>>(
        full_attention, n_heads, n_kv_heads, head_dim, window,
        hip_x, hip_attention_kv + key_float_offset,
        hip_attention_kv + value_float_offset, start_position, batch,
        query_stride, kv_stride, (int)sync->row_capacity,
        deferred_publish ? hip_y : NULL,
        deferred_publish ? hip_y + new_kv_count : NULL,
        start_position, deferred_publish ? batch : 0,
        deferred_publish ? hip_x : hip_y, hip_attention_status);
    if (hip_success(hipGetLastError(), "attention launch") != 0)
        return -1;
    if (deferred_publish) {
        hip_attention_publish_exact<<<
            (unsigned int)((new_kv_count + 255u) / 256u), 256>>>(
            hip_y, hip_y + new_kv_count,
            hip_attention_kv + key_float_offset,
            hip_attention_kv + value_float_offset,
            start_position, batch, kv_stride, (int)sync->row_capacity);
        if (hip_success(hipGetLastError(),
                "attention ring publish launch") != 0)
            return -1;
    }
    if (hip_success(hipMemcpy(host_outputs,
                                deferred_publish ? hip_x : hip_y,
                                (size_t)output_count * sizeof(float),
                                hipMemcpyDeviceToHost),
                     "hipMemcpy attention output") != 0 ||
        hip_success(hipMemcpy(&failure, hip_attention_status, sizeof failure,
                                hipMemcpyDeviceToHost),
                     "hipMemcpy attention status") != 0)
        return -1;
    if (failure) return -1;
    hip_path_record(HIP_PATH_ATTN_BODY, (uint64_t)(uint32_t)batch, tasks,
        (uint64_t)(deferred_publish ? 2u : 1u), UINT64_C(1));
    return 0;
}

static int hip_moe_descriptor(const uint32_t *vals,
                               const uint16_t *scales,
                               const uint16_t *biases, const void *id,
                               int rows, int cols, int batch, int expected_slot,
                               uint64_t expected_logical_resource_id,
                               int require_selected,
                               uint32_t input_offset, uint32_t output_offset,
                               HipBatchDesc *descriptor) {
    HipTensor *tensor = require_selected ? NULL : hip_tensor(id);
    HipResource *resource = NULL;
    HipSelectedResource *selected = NULL;
    unsigned char *device = NULL;
    size_t vbytes, sbytes;
    uint64_t voff = 0, soff = 0, boff = 0, auxoff = 0;
    if (!vals || !scales || !biases || !id || !descriptor ||
        rows < 1 || cols < 1 || batch < 1 ||
        hip_tensor_sizes(4, rows, cols, &vbytes, &sbytes) != 0)
        return -1;
    if (tensor) {
        if (tensor->bits != 4 || tensor->rows != rows || tensor->cols != cols)
            return -1;
        if (hip_tensor_physical(tensor, &resource, &device,
                &voff, &soff, &boff, &auxoff) != 0 ||
            resource->host + tensor->value_offset !=
                (const unsigned char *)(const void *)vals ||
            resource->host + tensor->scale_offset !=
                (const unsigned char *)(const void *)scales ||
            resource->host + tensor->bias_offset !=
                (const unsigned char *)(const void *)biases)
            return -1;

    } else {
        uintptr_t payload_base, payload_end, v, s, b, identity;
        if (expected_slot < 0 || expected_slot >= hip_selected_capacity)
            return -1;
        selected = &hip_selected_resources[expected_slot];
        if (!selected->active ||
            (!selected->registered && !selected->resident) ||
            !selected->device ||
            selected->logical_resource_id >=
                (uint64_t)(uint32_t)hip_selected_logical_capacity ||
            selected->logical_resource_id != expected_logical_resource_id ||
            hip_selected_logical_slots[selected->logical_resource_id] !=
                expected_slot)
            return -1;
        payload_base = (uintptr_t)(const void *)selected->payload;
        payload_end = (uintptr_t)(const void *)selected->host + selected->nbytes;
        v = (uintptr_t)(const void *)vals;
        s = (uintptr_t)(const void *)scales;
        b = (uintptr_t)(const void *)biases;
        identity = (uintptr_t)id;
        if (v < payload_base || s < payload_base || b < payload_base ||
            identity < payload_base || v > payload_end || s > payload_end ||
            b > payload_end || identity >= payload_end ||
            vbytes > payload_end - v || sbytes > payload_end - s ||
            sbytes > payload_end - b)
            return -1;
        voff = (uint64_t)(v - payload_base);
        soff = (uint64_t)(s - payload_base);
        boff = (uint64_t)(b - payload_base);
    }
    memset(descriptor, 0, sizeof *descriptor);
    descriptor->resource = selected ? NULL : device;
    descriptor->value_offset = voff;
    descriptor->scale_offset = soff;
    descriptor->bias_offset = boff;
    descriptor->input_offset = input_offset;
    descriptor->output_offset = output_offset;
    descriptor->rows = rows;
    descriptor->cols = cols;
    descriptor->batch = batch;
    if (selected) {
        descriptor->resource_slot = expected_slot;
        descriptor->selected = 1u;
        descriptor->logical_resource_id = expected_logical_resource_id;
    }
    return 0;
}

extern "C" int salt_gpu_q4_moe_chain(const SaltGpuMoeExpert *experts,
    int count, int hidden, int routed, const float *inputs, float *outputs,
    float *gate_scratch, float *up_scratch, size_t scratch_float_capacity) {
    HipBatchDesc activation[128];
    uint64_t total_rows = 0, input_floats, gate_up_floats, output_floats;
    uint64_t gate_up_cursor = 0, chain_cursor = 0;
    int max_group = 0, max_gate_groups = 0, max_down_groups = 0;
    if (!hip_ready || !experts || !inputs || !outputs || !gate_scratch ||
        !up_scratch || scratch_float_capacity < 1 || count < 1 ||
        count > 128 || hidden < 1 || routed < 1)
        return -1;
    for (int i = 0; i < count; i++) {
        if (experts[i].group < 1) return -1;
        total_rows += (uint32_t)experts[i].group;
        if (experts[i].group > max_group) max_group = experts[i].group;
    }
    input_floats = total_rows * (uint32_t)hidden;
    gate_up_floats = total_rows * (uint32_t)routed * 2u;
    output_floats = input_floats;
    if (total_rows > UINT32_MAX || input_floats > HIP_MAX_OUTPUTS ||
        gate_up_floats > HIP_MAX_OUTPUTS || output_floats > HIP_MAX_OUTPUTS)
        return -1;
    if (hip_success(hipMemcpy(hip_x, inputs,
            (size_t)input_floats * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy MoE inputs") != 0)
        return -1;
    if (count == 1) {
        HipBatchDesc gate, up, down;
        int group = experts[0].group;
        size_t elements = (size_t)group * (size_t)routed;
        if (hip_moe_descriptor(experts[0].gate_vals,
                experts[0].gate_scales, experts[0].gate_biases,
                experts[0].gate_id, routed, hidden, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                0, 0, 0, &gate) != 0 ||
            hip_moe_descriptor(experts[0].up_vals,
                experts[0].up_scales, experts[0].up_biases,
                experts[0].up_id, routed, hidden, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                0, 0, (uint32_t)elements, &up) != 0 ||
            hip_moe_descriptor(experts[0].down_vals,
                experts[0].down_scales, experts[0].down_biases,
                experts[0].down_id, hidden, routed, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                0, 0, 0, &down) != 0)
            return -1;
        if (group >= 8) {
            hip_q4_warp_batch<<<(unsigned int)routed, 1024>>>(
                gate.resource, gate.value_offset, gate.scale_offset,
                gate.bias_offset, hip_x, hip_y, routed, hidden, group);
            hip_q4_warp_batch<<<(unsigned int)routed, 1024>>>(
                up.resource, up.value_offset, up.scale_offset,
                up.bias_offset, hip_x, hip_y + elements,
                routed, hidden, group);
        } else {
            hip_q4_warp_exact<<<dim3((unsigned int)((routed + 3) / 4),
                                       (unsigned int)group), 128>>>(
                gate.resource, gate.value_offset, gate.scale_offset,
                gate.bias_offset, hip_x, hip_y, routed, hidden);
            hip_q4_warp_exact<<<dim3((unsigned int)((routed + 3) / 4),
                                       (unsigned int)group), 128>>>(
                up.resource, up.value_offset, up.scale_offset,
                up.bias_offset, hip_x, hip_y + elements, routed, hidden);
        }
        hip_moe_activate_contiguous<<<
            (unsigned int)((elements + 255u) / 256u), 256>>>(
                hip_y, hip_y + elements, hip_x, elements);
        if (group >= 8)
            hip_q4_warp_batch<<<(unsigned int)hidden, 1024>>>(
                down.resource, down.value_offset, down.scale_offset,
                down.bias_offset, hip_x, hip_y, hidden, routed, group);
        else
            hip_q4_warp_exact<<<dim3((unsigned int)((hidden + 3) / 4),
                                       (unsigned int)group), 128>>>(
                down.resource, down.value_offset, down.scale_offset,
                down.bias_offset, hip_x, hip_y, hidden, routed);
        if (hip_success(hipGetLastError(), "dense FFN chain launch") != 0 ||
            hip_success(hipMemcpy(outputs, hip_y,
                (size_t)output_floats * sizeof(float), hipMemcpyDeviceToHost),
                "hipMemcpy dense FFN outputs") != 0)
            return -1;
        hip_path_record(group >= 8 ? HIP_PATH_MOE_GATE_UP_WARP_BATCH
                                   : HIP_PATH_MOE_GATE_UP_WARP_EXACT,
            (uint64_t)(uint32_t)(2 * group),
            UINT64_C(2) * (uint32_t)group * (uint32_t)routed,
            UINT64_C(2), UINT64_C(0));
        hip_path_record(HIP_PATH_MOE_ACTIVATION,
            (uint64_t)(uint32_t)group, (uint64_t)elements,
            UINT64_C(1), UINT64_C(0));
        hip_path_record(group >= 8 ? HIP_PATH_MOE_DOWN_WARP_BATCH
                                   : HIP_PATH_MOE_DOWN_WARP_EXACT,
            (uint64_t)(uint32_t)group,
            (uint64_t)(uint32_t)group * (uint32_t)hidden,
            UINT64_C(1), UINT64_C(1));
        return 0;
    }
    uint64_t row_cursor = 0;
    for (int i = 0; i < count; i++) {
        uint64_t elements = (uint64_t)(uint32_t)experts[i].group *
            (uint32_t)routed;
        if (row_cursor > UINT32_MAX / (uint32_t)hidden ||
            gate_up_cursor > UINT32_MAX - 2u * elements ||
            chain_cursor > UINT32_MAX - elements)
            return -1;
        uint32_t input_offset = (uint32_t)(row_cursor * (uint32_t)hidden);
        uint32_t gate_offset = (uint32_t)gate_up_cursor;
        uint32_t up_offset = (uint32_t)(gate_up_cursor + elements);
        if (hip_moe_descriptor(experts[i].gate_vals,
                experts[i].gate_scales, experts[i].gate_biases,
                experts[i].gate_id, routed, hidden, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                0, input_offset, gate_offset,
                &hip_host_desc[2 * i]) != 0 ||
            hip_moe_descriptor(experts[i].up_vals,
                experts[i].up_scales, experts[i].up_biases,
                experts[i].up_id, routed, hidden, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                0, input_offset, up_offset,
                &hip_host_desc[2 * i + 1]) != 0)
            return -1;
        memset(&activation[i], 0, sizeof activation[i]);
        activation[i].input_offset = gate_offset;
        activation[i].output_offset = up_offset;
        activation[i].aux_offset = chain_cursor;
        activation[i].rows = routed;
        activation[i].batch = experts[i].group;
        gate_up_cursor += 2u * elements;
        chain_cursor += elements;
        row_cursor += (uint32_t)experts[i].group;
        int groups = (experts[i].group + 31) / 32;
        if (groups > max_gate_groups) max_gate_groups = groups;
    }
    if (hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
            (size_t)(2 * count) * sizeof(HipBatchDesc),
            hipMemcpyHostToDevice), "hipMemcpy MoE gate/up descriptors") != 0)
        return -1;
    hip_q4_heterogeneous<<<dim3((unsigned int)routed,
        (unsigned int)max_gate_groups, (unsigned int)(2 * count)), 32>>>(
        hip_device_desc, 2 * count, hip_x, hip_y,
        hip_selected_device_resources, hip_selected_device_logical_slots,
        hip_selected_device_payload_offsets);
    if (hip_success(hipGetLastError(), "MoE gate/up launch") != 0 ||
        hip_success(hipMemcpy(hip_device_desc, activation,
            (size_t)count * sizeof(HipBatchDesc), hipMemcpyHostToDevice),
            "hipMemcpy MoE activation descriptors") != 0)
        return -1;
    hip_moe_activate<<<dim3((unsigned int)(((uint64_t)max_group *
        (uint32_t)routed + 255u) / 256u), 1u, (unsigned int)count), 256>>>(
        hip_device_desc, count, hip_y, hip_x);
    if (hip_success(hipGetLastError(), "MoE activation launch") != 0)
        return -1;
    row_cursor = 0;
    chain_cursor = 0;
    for (int i = 0; i < count; i++) {
        uint64_t elements = (uint64_t)(uint32_t)experts[i].group *
            (uint32_t)routed;
        uint32_t output_offset = (uint32_t)(row_cursor * (uint32_t)hidden);
        if (hip_moe_descriptor(experts[i].down_vals,
                experts[i].down_scales, experts[i].down_biases,
                experts[i].down_id, hidden, routed, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                0, (uint32_t)chain_cursor, output_offset,
                &hip_host_desc[i]) != 0)
            return -1;
        chain_cursor += elements;
        row_cursor += (uint32_t)experts[i].group;
        int groups = (experts[i].group + 31) / 32;
        if (groups > max_down_groups) max_down_groups = groups;
    }
    if (hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
            (size_t)count * sizeof(HipBatchDesc), hipMemcpyHostToDevice),
            "hipMemcpy MoE down descriptors") != 0)
        return -1;
    hip_q4_heterogeneous<<<dim3((unsigned int)hidden,
        (unsigned int)max_down_groups, (unsigned int)count), 32>>>(
        hip_device_desc, count, hip_x, hip_y,
        hip_selected_device_resources, hip_selected_device_logical_slots,
        hip_selected_device_payload_offsets);
    if (hip_success(hipGetLastError(), "MoE down launch") != 0 ||
        hip_success(hipMemcpy(outputs, hip_y,
            (size_t)output_floats * sizeof(float), hipMemcpyDeviceToHost),
            "hipMemcpy MoE outputs") != 0)
        return -1;
    hip_path_record(HIP_PATH_MOE_HET_GATE_UP, UINT64_C(2) * total_rows,
        UINT64_C(2) * total_rows * (uint32_t)routed,
        UINT64_C(1), UINT64_C(0));
    hip_path_record(HIP_PATH_MOE_HET_ACTIVATION, total_rows,
        total_rows * (uint32_t)routed, UINT64_C(1), UINT64_C(0));
    hip_path_record(HIP_PATH_MOE_HET_DOWN, total_rows,
        total_rows * (uint32_t)hidden, UINT64_C(1), UINT64_C(1));
    return 0;
}

static int hip_q4_moe_chain_shared(const SaltGpuMoeExpert *experts,
        int count, int hidden, int routed, const float *inputs, float *outputs,
        float *gate_scratch, float *up_scratch,
        size_t scratch_float_capacity) {
    uint64_t total_rows = 0, input_floats, chain_floats;
    size_t input_bytes, output_bytes, scratch_bytes;
    int input_slot, output_slot, gate_slot, up_slot;
    size_t input_byte_offset, output_byte_offset;
    size_t gate_byte_offset, up_byte_offset;
    int max_gate_groups = 0, max_down_groups = 0, max_group = 0;
    int all_single = 1;
    uint64_t row_cursor = 0, chain_cursor = 0;
    uint64_t h2d_start, h2d_end, completion_end;
    uint64_t gate_up_gpu_ns = 0, activation_gpu_ns = 0, down_gpu_ns = 0;
    uint64_t stage_start = 0;
    int stage_sync = getenv("SALT_HIP_MOE_STAGE_SYNC") != NULL;
    if (!experts || count < 1 || count > 128 || hidden < 1 || routed < 1 ||
        !inputs || !outputs || !gate_scratch || !up_scratch)
        return -1;
    for (int index = 0; index < count; index++) {
        if (experts[index].group < 1) return -1;
        total_rows += (uint32_t)experts[index].group;
        if (experts[index].group > max_group) max_group = experts[index].group;
        if (experts[index].group != 1) all_single = 0;
    }
    if (total_rows > SIZE_MAX / (size_t)hidden / sizeof(float) ||
        total_rows > SIZE_MAX / (size_t)routed / sizeof(float))
        return -1;
    chain_floats = total_rows * (uint32_t)routed;
    input_floats = total_rows * (uint32_t)hidden;
    if (chain_floats > scratch_float_capacity) return -1;
    input_bytes = (size_t)total_rows * (size_t)hidden * sizeof(float);
    output_bytes = input_bytes;
    scratch_bytes = (size_t)chain_floats * sizeof(float);
    if (hip_shared_pointer(inputs, input_bytes, &input_slot,
            &input_byte_offset) != 0 &&
        hip_shared_pointer(outputs, output_bytes, &output_slot,
            &output_byte_offset) != 0 &&
        hip_shared_pointer(gate_scratch, scratch_bytes, &gate_slot,
            &gate_byte_offset) != 0 &&
        hip_shared_pointer(up_scratch, scratch_bytes, &up_slot,
            &up_byte_offset) != 0)
        return 1;
    if (hip_shared_pointer(inputs, input_bytes, &input_slot,
            &input_byte_offset) != 0 ||
        hip_shared_pointer(outputs, output_bytes, &output_slot,
            &output_byte_offset) != 0 ||
        hip_shared_pointer(gate_scratch, scratch_bytes, &gate_slot,
            &gate_byte_offset) != 0 ||
        hip_shared_pointer(up_scratch, scratch_bytes, &up_slot,
            &up_byte_offset) != 0 ||
        (input_byte_offset | output_byte_offset |
         gate_byte_offset | up_byte_offset) % sizeof(float) != 0)
        return -1;
    size_t input_base = 0;
    size_t output_base = (size_t)input_floats;
    size_t gate_base = output_base + (size_t)input_floats;
    size_t up_base = gate_base + (size_t)chain_floats;
    if (up_base > HIP_MAX_OUTPUTS || chain_floats > HIP_MAX_OUTPUTS - up_base ||
        (size_t)count > HIP_MAX_BATCH_JOBS / 4u)
        return -1;
    for (int index = 0; index < count; index++) {
        uint64_t elements = (uint64_t)(uint32_t)experts[index].group *
            (uint32_t)routed;
        uint64_t input_offset = input_base +
            row_cursor * (uint32_t)hidden;
        uint64_t output_offset = output_base +
            row_cursor * (uint32_t)hidden;
        uint64_t gate_offset = gate_base + chain_cursor;
        uint64_t up_offset = up_base + chain_cursor;
        if (input_offset > UINT32_MAX || output_offset > UINT32_MAX ||
            gate_offset > UINT32_MAX || up_offset > UINT32_MAX ||
            hip_moe_descriptor(experts[index].gate_vals,
                experts[index].gate_scales, experts[index].gate_biases,
                experts[index].gate_id, routed, hidden,
                experts[index].group, experts[index].resource_slot,
                experts[index].logical_resource_id,
                1,
                (uint32_t)input_offset, (uint32_t)gate_offset,
                &hip_host_desc[2 * index]) != 0 ||
            hip_moe_descriptor(experts[index].up_vals,
                experts[index].up_scales, experts[index].up_biases,
                experts[index].up_id, routed, hidden,
                experts[index].group, experts[index].resource_slot,
                experts[index].logical_resource_id,
                1,
                (uint32_t)input_offset, (uint32_t)up_offset,
                &hip_host_desc[2 * index + 1]) != 0 ||
            hip_moe_descriptor(experts[index].down_vals,
                experts[index].down_scales, experts[index].down_biases,
                experts[index].down_id, hidden, routed,
                experts[index].group, experts[index].resource_slot,
                experts[index].logical_resource_id,
                1,
                (uint32_t)gate_offset, (uint32_t)output_offset,
                &hip_host_desc[3 * count + index]) != 0)
            return -1;
        HipBatchDesc *activation = &hip_host_desc[2 * count + index];
        memset(activation, 0, sizeof *activation);
        activation->input_offset = (uint32_t)gate_offset;
        activation->output_offset = (uint32_t)up_offset;
        activation->aux_offset = (uint32_t)gate_offset;
        activation->rows = routed;
        activation->batch = experts[index].group;
        int groups = (experts[index].group + 31) / 32;
        if (groups > max_gate_groups) max_gate_groups = groups;
        if (groups > max_down_groups) max_down_groups = groups;
        chain_cursor += elements;
        row_cursor += (uint32_t)experts[index].group;
    }
    if (!hip_x) return -1;
    float *arena = hip_x;
    h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(arena + input_base, inputs, input_bytes,
            hipMemcpyHostToDevice), "hipMemcpy shared MoE canonical input") != 0 ||
        hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
            (size_t)(4 * count) * sizeof(HipBatchDesc),
            hipMemcpyHostToDevice), "hipMemcpy shared MoE descriptors") != 0)
        return -1;
    h2d_end = hip_host_now_ns();
    if (hip_moe_timing_enabled &&
        hip_success(hipEventRecord(hip_moe_event_start),
            "hipEventRecord MoE start") != 0)
        return -1;
    if (stage_sync) stage_start = hip_host_now_ns();
    hip_selected_read_submitted();
    if (all_single)
        hip_q4_heterogeneous_warp<<<dim3(
            (unsigned int)((routed + 3) / 4), 1u,
            (unsigned int)(2 * count)), 128>>>(
            hip_device_desc, 2 * count, arena, arena,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    else
        hip_q4_heterogeneous<<<dim3((unsigned int)routed,
            (unsigned int)max_gate_groups, (unsigned int)(2 * count)), 32>>>(
            hip_device_desc, 2 * count, arena, arena,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    if (hip_moe_timing_enabled &&
        hip_success(hipEventRecord(hip_moe_event_gate_up),
            "hipEventRecord MoE gate/up") != 0)
        return -1;
    if (stage_sync) {
        if (hip_success(hipGetLastError(), "profile MoE gate/up launch") != 0 ||
            hip_success(hipDeviceSynchronize(),
                "profile hipDeviceSynchronize MoE gate/up") != 0)
            return -1;
        hip_selected_reads_completed();
        gate_up_gpu_ns = hip_host_now_ns() - stage_start;
        stage_start = hip_host_now_ns();
    }
    hip_moe_activate<<<dim3((unsigned int)(((uint64_t)max_group *
        (uint32_t)routed + 255u) / 256u), 1u, (unsigned int)count), 256>>>(
        hip_device_desc + 2 * count, count, arena, arena);
    if (hip_moe_timing_enabled &&
        hip_success(hipEventRecord(hip_moe_event_activation),
            "hipEventRecord MoE activation") != 0)
        return -1;
    if (stage_sync) {
        if (hip_success(hipGetLastError(), "profile MoE activation launch") != 0 ||
            hip_success(hipDeviceSynchronize(),
                "profile hipDeviceSynchronize MoE activation") != 0)
            return -1;
        activation_gpu_ns = hip_host_now_ns() - stage_start;
        stage_start = hip_host_now_ns();
    }
    hip_selected_read_submitted();
    if (all_single)
        hip_q4_heterogeneous_warp<<<dim3(
            (unsigned int)((hidden + 3) / 4), 1u,
            (unsigned int)count), 128>>>(
            hip_device_desc + 3 * count, count, arena, arena,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    else
        hip_q4_heterogeneous<<<dim3((unsigned int)hidden,
            (unsigned int)max_down_groups, (unsigned int)count), 32>>>(
            hip_device_desc + 3 * count, count, arena, arena,
            hip_selected_device_resources, hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    if (hip_moe_timing_enabled &&
        hip_success(hipEventRecord(hip_moe_event_down),
            "hipEventRecord MoE down") != 0)
        return -1;
    if (stage_sync) {
        if (hip_success(hipGetLastError(), "profile MoE down launch") != 0 ||
            hip_success(hipDeviceSynchronize(),
                "profile hipDeviceSynchronize MoE down") != 0)
            return -1;
        hip_selected_reads_completed();
        down_gpu_ns = hip_host_now_ns() - stage_start;
    }
    if (hip_success(hipGetLastError(), "shared MoE wave launch") != 0 ||
        hip_success(hipMemcpy(outputs, arena + output_base, output_bytes,
            hipMemcpyDeviceToHost), "hipMemcpy shared MoE canonical output") != 0)
        return -1;
    hip_selected_reads_completed();
    completion_end = hip_host_now_ns();
    hip_selected_shared_calls++;
    hip_selected_shared_jobs += total_rows;
    hip_u64_add_saturating(&hip_selected_shared_h2d_ns,
        h2d_end - h2d_start);
    hip_u64_add_saturating(&hip_selected_shared_completion_ns,
        completion_end - h2d_end);
    if (stage_sync) {
        hip_u64_add_saturating(&hip_selected_gate_up_gpu_ns,
            gate_up_gpu_ns);
        hip_u64_add_saturating(&hip_selected_activation_gpu_ns,
            activation_gpu_ns);
        hip_u64_add_saturating(&hip_selected_down_gpu_ns, down_gpu_ns);
    } else if (hip_moe_timing_enabled) {
        if (hip_event_elapsed_ns(hip_moe_event_start,
                hip_moe_event_gate_up, &gate_up_gpu_ns) != 0 ||
            hip_event_elapsed_ns(hip_moe_event_gate_up,
                hip_moe_event_activation, &activation_gpu_ns) != 0 ||
            hip_event_elapsed_ns(hip_moe_event_activation,
                hip_moe_event_down, &down_gpu_ns) != 0)
            return -1;
        hip_u64_add_saturating(&hip_selected_gate_up_gpu_ns,
            gate_up_gpu_ns);
        hip_u64_add_saturating(&hip_selected_activation_gpu_ns,
            activation_gpu_ns);
        hip_u64_add_saturating(&hip_selected_down_gpu_ns, down_gpu_ns);
    }
    hip_path_record(all_single ? HIP_PATH_MOE_SELECTED_GATE_UP_WARP
                               : HIP_PATH_MOE_SELECTED_GATE_UP_TOKEN,
        UINT64_C(2) * total_rows,
        UINT64_C(2) * total_rows * (uint32_t)routed,
        UINT64_C(1), UINT64_C(0));
    hip_path_record(HIP_PATH_MOE_SELECTED_ACTIVATION, total_rows,
        total_rows * (uint32_t)routed, UINT64_C(1), UINT64_C(0));
    hip_path_record(all_single ? HIP_PATH_MOE_SELECTED_DOWN_WARP
                               : HIP_PATH_MOE_SELECTED_DOWN_TOKEN,
        total_rows, total_rows * (uint32_t)hidden,
        UINT64_C(1), UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_q4_moe_chain_selected(
    const SaltGpuMoeExpert *experts, int count, int hidden, int routed,
    const float *inputs, float *outputs, float *gate_scratch,
    float *up_scratch, size_t scratch_float_capacity) {
    int rc = hip_q4_moe_chain_shared(experts, count, hidden, routed,
        inputs, outputs, gate_scratch, up_scratch, scratch_float_capacity);
    if (rc != 1) return rc;
    __atomic_fetch_add(&hip_selected_legacy_refusals, UINT64_C(1),
        __ATOMIC_RELAXED);
    if (getenv("SALT_GPU_DIAG"))
        fputs("gpu-hip: selected MoE refused non-shared canonical buffers\n",
            stderr);
    return -1;
}

extern "C" int salt_gpu_selected_resources_prepare(
    int capacity, int logical_capacity) {
    if (!hip_ready || capacity < 1 || logical_capacity < 1 ||
        capacity > logical_capacity || logical_capacity > INT32_MAX ||
        (size_t)capacity > SIZE_MAX / sizeof *hip_selected_resources ||
        (size_t)capacity > SIZE_MAX / sizeof *hip_selected_resource_slots ||
        (size_t)logical_capacity >
            SIZE_MAX / sizeof *hip_selected_logical_slots ||
        (size_t)logical_capacity >
            SIZE_MAX / sizeof *hip_selected_payload_offsets)
        return -1;
    if (hip_selected_resources)
        return capacity == hip_selected_capacity &&
            logical_capacity == hip_selected_logical_capacity ? 0 : -1;
    hip_selected_resources = (HipSelectedResource *)calloc(
        (size_t)capacity, sizeof *hip_selected_resources);
    if (!hip_selected_resources ||
        hip_success(hipHostMalloc((void **)&hip_selected_resource_slots,
            (size_t)capacity * sizeof *hip_selected_resource_slots,
            hipHostMallocMapped), "hipHostMalloc selected resource table") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_selected_logical_slots,
            (size_t)logical_capacity * sizeof *hip_selected_logical_slots,
            hipHostMallocMapped), "hipHostMalloc selected logical slot table") != 0 ||
        hip_success(hipHostMalloc((void **)&hip_selected_payload_offsets,
            (size_t)logical_capacity * sizeof *hip_selected_payload_offsets,
            hipHostMallocMapped), "hipHostMalloc selected payload offset table") != 0 ||
        hip_success(hipHostGetDevicePointer(
            (void **)&hip_selected_device_resources,
            hip_selected_resource_slots, 0),
            "hipHostGetDevicePointer selected resource table") != 0 ||
        hip_success(hipHostGetDevicePointer(
            (void **)&hip_selected_device_logical_slots,
            hip_selected_logical_slots, 0),
            "hipHostGetDevicePointer selected logical slot table") != 0 ||
        hip_success(hipHostGetDevicePointer(
            (void **)&hip_selected_device_payload_offsets,
            hip_selected_payload_offsets, 0),
            "hipHostGetDevicePointer selected payload offset table") != 0) {
        free(hip_selected_resources);
        if (hip_selected_resource_slots) hipHostFree(hip_selected_resource_slots);
        if (hip_selected_logical_slots) hipHostFree(hip_selected_logical_slots);
        if (hip_selected_payload_offsets) hipHostFree(hip_selected_payload_offsets);
        hip_selected_resources = NULL;
        hip_selected_resource_slots = NULL;
        hip_selected_logical_slots = NULL;
        hip_selected_payload_offsets = NULL;
        hip_selected_device_resources = NULL;
        hip_selected_device_logical_slots = NULL;
        hip_selected_device_payload_offsets = NULL;
        return -1;
    }
    memset(hip_selected_resource_slots, 0,
        (size_t)capacity * sizeof *hip_selected_resource_slots);
    memset(hip_selected_payload_offsets, 0,
        (size_t)logical_capacity * sizeof *hip_selected_payload_offsets);
    for (int logical = 0; logical < logical_capacity; logical++)
        hip_selected_logical_slots[logical] = -1;
    hip_selected_capacity = capacity;
    hip_selected_logical_capacity = logical_capacity;
    return 0;
}

extern "C" int salt_gpu_selected_resource_bind(
    int slot, uint64_t logical_resource_id, const void *base, size_t nbytes,
    const void *payload) {
    HipSelectedResource *resource;
    uintptr_t b, p;
    void *device = NULL;
    uint64_t register_start, register_end, pointer_end;
    if (!hip_ready || !hip_selected_resources ||
        !hip_selected_resource_slots || !hip_selected_logical_slots ||
        !hip_selected_payload_offsets || !hip_selected_device_resources ||
        !hip_selected_device_logical_slots ||
        !hip_selected_device_payload_offsets ||
        slot < 0 || slot >= hip_selected_capacity ||
        logical_resource_id >= (uint64_t)(uint32_t)hip_selected_logical_capacity ||
        !base || !payload || nbytes < 1)
        return -1;
    b = (uintptr_t)base;
    p = (uintptr_t)payload;
    if (p < b || p - b >= nbytes || p - b > UINT32_MAX) return -1;
    resource = &hip_selected_resources[slot];
    if (resource->active) {
        int same = resource->logical_resource_id == logical_resource_id &&
            resource->host == (const unsigned char *)base &&
            resource->payload == (const unsigned char *)payload &&
            resource->nbytes == nbytes &&
            hip_selected_resource_slots[slot] == resource->device &&
            hip_selected_logical_slots[logical_resource_id] == slot &&
            hip_selected_payload_offsets[logical_resource_id] ==
                (uint32_t)(p - b);
        if (same) hip_selected_bind_reuses++;
        return same ? 0 : -1;
    }
    if (hip_selected_logical_slots[logical_resource_id] != -1)
        return -1;
    register_start = hip_host_now_ns();
    if (hip_success(hipHostRegister(
            (void *)base, nbytes, hipHostRegisterMapped),
            "hipHostRegister selected resource") != 0)
        return -1;
    register_end = hip_host_now_ns();
    if (hip_success(hipHostGetDevicePointer(&device, (void *)base, 0),
            "hipHostGetDevicePointer selected resource") != 0 || !device) {
        (void)hipHostUnregister((void *)base);
        return -1;
    }
    pointer_end = hip_host_now_ns();
    resource->active = 1;
    resource->registered = 1;
    resource->logical_resource_id = logical_resource_id;
    resource->host = (const unsigned char *)base;
    resource->payload = (const unsigned char *)payload;
    resource->device = (unsigned char *)device;
    resource->nbytes = nbytes;
    hip_selected_resource_slots[slot] = resource->device;
    hip_selected_logical_slots[logical_resource_id] = slot;
    hip_selected_payload_offsets[logical_resource_id] = (uint32_t)(p - b);
    hip_selected_bind_calls++;
    hip_u64_add_saturating(&hip_selected_register_ns,
        register_end - register_start);
    hip_u64_add_saturating(&hip_selected_pointer_ns,
        pointer_end - register_end);
    if (getenv("SALT_HIP_RESOURCE_TRACE"))
        fprintf(stderr,
            "SALT_HIP_RESOURCE event=bind slot=%d logical=%llu bytes=%zu "
            "register_ns=%llu pointer_ns=%llu\n",
            slot, (unsigned long long)logical_resource_id, nbytes,
            (unsigned long long)(register_end - register_start),
            (unsigned long long)(pointer_end - register_end));
    return 0;
}

extern "C" int salt_gpu_selected_resource_bind_resident(
    int slot, uint64_t logical_resource_id, const void *payload, size_t nbytes,
    uintptr_t device_address, uint64_t generation) {
    HipSelectedResource *resource;
    uint64_t expected_offset;
    unsigned char *device = (unsigned char *)(void *)device_address;
    if (!hip_ready || !hip_residency.ready || !hip_selected_resources ||
        !hip_selected_resource_slots || !hip_selected_logical_slots ||
        !hip_selected_payload_offsets || !hip_selected_device_resources ||
        !hip_selected_device_logical_slots ||
        !hip_selected_device_payload_offsets || slot < 0 ||
        slot >= hip_selected_capacity ||
        (uint32_t)slot >= hip_residency.plan.slot_count ||
        logical_resource_id >=
            (uint64_t)(uint32_t)hip_selected_logical_capacity ||
        !payload || nbytes == 0 ||
        nbytes > hip_residency.plan.slot_bytes || !device || generation == 0)
        return -1;
    expected_offset = hip_residency.plan.slot_base_offset +
        (uint64_t)(uint32_t)slot * hip_residency.plan.slot_bytes;
    if (expected_offset > hip_residency.plan.device_budget_bytes ||
        nbytes > hip_residency.plan.device_budget_bytes - expected_offset ||
        device != hip_residency.device_arena + (size_t)expected_offset)
        return -1;
    resource = &hip_selected_resources[slot];
    if (resource->active)
        return resource->resident && !resource->registered &&
            resource->logical_resource_id == logical_resource_id &&
            resource->generation == generation &&
            resource->host == (const unsigned char *)payload &&
            resource->payload == (const unsigned char *)payload &&
            resource->device == device && resource->nbytes == nbytes &&
            hip_selected_resource_slots[slot] == device &&
            hip_selected_logical_slots[logical_resource_id] == slot &&
            hip_selected_payload_offsets[logical_resource_id] == 0 ? 0 : -1;
    if (hip_selected_logical_slots[logical_resource_id] != -1)
        return -1;
    resource->active = 1;
    resource->resident = 1;
    resource->logical_resource_id = logical_resource_id;
    resource->generation = generation;
    resource->host = (const unsigned char *)payload;
    resource->payload = (const unsigned char *)payload;
    resource->device = device;
    resource->nbytes = nbytes;
    hip_selected_resource_slots[slot] = device;
    hip_selected_logical_slots[logical_resource_id] = slot;
    hip_selected_payload_offsets[logical_resource_id] = 0;
    hip_selected_bind_calls++;
    return 0;
}

extern "C" int salt_gpu_selected_resources_fence(void) {
    return salt_gpu_sync();
}

extern "C" int salt_gpu_selected_resource_unbind(
    int slot, uint64_t logical_resource_id) {
    HipSelectedResource *resource;
    uint64_t sync_start, sync_end, unregister_end;
    int synchronized = 0;
    if (!hip_selected_resources || !hip_selected_resource_slots ||
        !hip_selected_logical_slots ||
        !hip_selected_payload_offsets || !hip_selected_device_resources ||
        !hip_selected_device_logical_slots ||
        !hip_selected_device_payload_offsets || slot < 0 ||
        slot >= hip_selected_capacity ||
        logical_resource_id >= (uint64_t)(uint32_t)hip_selected_logical_capacity)
        return -1;
    resource = &hip_selected_resources[slot];
    if (!resource->active) return 0;
    if (resource->logical_resource_id != logical_resource_id ||
        hip_selected_logical_slots[logical_resource_id] != slot ||
        (!resource->registered && !resource->resident))
        return -1;
    sync_start = hip_host_now_ns();
    if (hip_selected_complete_epoch != hip_selected_read_epoch) {
        if (hip_success(hipDeviceSynchronize(),
                "hipDeviceSynchronize selected unbind") != 0)
            return -1;
        hip_selected_reads_completed();
        hip_selected_unbind_sync_calls++;
        synchronized = 1;
    } else {
        hip_selected_unbind_sync_skips++;
    }
    sync_end = hip_host_now_ns();
    hip_selected_resource_slots[slot] = NULL;
    hip_selected_logical_slots[logical_resource_id] = -1;
    hip_selected_payload_offsets[logical_resource_id] = 0;
    if (resource->registered &&
        hip_success(hipHostUnregister((void *)resource->host),
            "hipHostUnregister selected resource") != 0) {
        hip_selected_resource_slots[slot] = resource->device;
        hip_selected_logical_slots[logical_resource_id] = slot;
        hip_selected_payload_offsets[logical_resource_id] = (uint32_t)(
            (uintptr_t)resource->payload - (uintptr_t)resource->host);
        return -1;
    }
    unregister_end = hip_host_now_ns();
    hip_selected_unbind_calls++;
    hip_u64_add_saturating(&hip_selected_unbind_sync_ns,
        sync_end - sync_start);
    hip_u64_add_saturating(&hip_selected_unregister_ns,
        unregister_end - sync_end);
    if (getenv("SALT_HIP_RESOURCE_TRACE"))
        fprintf(stderr,
            "SALT_HIP_RESOURCE event=unbind slot=%d logical=%llu "
            "sync=%d sync_ns=%llu unregister_ns=%llu\n",
            slot, (unsigned long long)logical_resource_id,
            synchronized,
            (unsigned long long)(sync_end - sync_start),
            (unsigned long long)(unregister_end - sync_end));
    memset(resource, 0, sizeof *resource);
    return 0;
}

extern "C" int salt_gpu_q4_selected_shared(
    const SaltGpuSelectedProjectionJob *jobs, int job_count,
    SaltGpuSharedBuffer *arena) {
    uintptr_t arena_base;
    int max_rows = 0, max_groups = 0, max_batch = 0;
    uint32_t occupancy[6] = {0, 0, 0, 0, 0, 0};
    uint64_t logical_jobs = 0;
    uint64_t logical_rows = 0;
    uint64_t h2d_start, h2d_end, completion_end;
    if (!hip_ready || !jobs || job_count < 1 || job_count > 256 || !arena ||
        !arena->contents || !arena->backend || arena->nbytes < 1)
        return -1;
    arena_base = (uintptr_t)arena->contents;
    for (int job = 0; job < job_count; job++) {
        const SaltGpuSelectedProjectionJob *entry = &jobs[job];
        uintptr_t input, output;
        size_t input_bytes, output_bytes, input_offset, output_offset;
        int groups;
        if (!entry->vals || !entry->scales || !entry->biases || !entry->id ||
            !entry->inputs || !entry->outputs || entry->rows < 1 ||
            entry->cols < 1 || entry->batch < 1 ||
            logical_jobs > UINT64_MAX - (uint32_t)entry->batch ||
            (size_t)entry->cols > SIZE_MAX / (size_t)entry->batch /
                sizeof(float) ||
            (size_t)entry->rows > SIZE_MAX / (size_t)entry->batch /
                sizeof(float))
            return -1;
        input_bytes = (size_t)entry->cols * (size_t)entry->batch *
            sizeof(float);
        output_bytes = (size_t)entry->rows * (size_t)entry->batch *
            sizeof(float);
        input = (uintptr_t)(const void *)entry->inputs;
        output = (uintptr_t)(void *)entry->outputs;
        if (input < arena_base || output < arena_base ||
            input - arena_base > SIZE_MAX || output - arena_base > SIZE_MAX)
            return -1;
        input_offset = (size_t)(input - arena_base);
        output_offset = (size_t)(output - arena_base);
        if (input_offset > arena->nbytes ||
            input_bytes > arena->nbytes - input_offset ||
            output_offset > arena->nbytes ||
            output_bytes > arena->nbytes - output_offset ||
            (input_offset < output_offset + output_bytes &&
             output_offset < input_offset + input_bytes) ||
            (input_offset | output_offset) % sizeof(float) != 0 ||
            input_offset / sizeof(float) > UINT32_MAX ||
            output_offset / sizeof(float) > UINT32_MAX ||
            hip_moe_descriptor(entry->vals, entry->scales, entry->biases,
                entry->id, entry->rows, entry->cols, entry->batch,
                entry->resource_slot, entry->logical_resource_id,
                1,
                (uint32_t)(input_offset / sizeof(float)),
                (uint32_t)(output_offset / sizeof(float)),
                &hip_host_desc[job]) != 0)
            return -1;
        for (int prior = 0; prior < job; prior++) {
            uintptr_t prior_output = (uintptr_t)(void *)jobs[prior].outputs;
            size_t prior_bytes = (size_t)jobs[prior].rows *
                (size_t)jobs[prior].batch * sizeof(float);
            if (output < prior_output + prior_bytes &&
                prior_output < output + output_bytes)
                return -1;
        }
        groups = (entry->batch + 31) / 32;
        if (entry->rows > max_rows) max_rows = entry->rows;
        if (groups > max_groups) max_groups = groups;
        if (entry->batch > max_batch) max_batch = entry->batch;
        if (entry->batch == 1) occupancy[0]++;
        else if (entry->batch == 2) occupancy[1]++;
        else if (entry->batch <= 4) occupancy[2]++;
        else if (entry->batch <= 8) occupancy[3]++;
        else if (entry->batch <= 16) occupancy[4]++;
        else occupancy[5]++;
        logical_jobs += (uint32_t)entry->batch;
        logical_rows += (uint64_t)(uint32_t)entry->batch *
            (uint32_t)entry->rows;
    }
    h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(hip_device_desc, hip_host_desc,
            (size_t)job_count * sizeof(HipBatchDesc), hipMemcpyHostToDevice),
            "hipMemcpy selected shared descriptors") != 0)
        return -1;
    h2d_end = hip_host_now_ns();
    hip_selected_read_submitted();
    if (max_batch == 1)
        hip_q4_heterogeneous<<<dim3((unsigned int)max_rows,
            (unsigned int)max_groups, (unsigned int)job_count), 32>>>(
            hip_device_desc, job_count, (const float *)arena->backend,
            (float *)arena->backend, hip_selected_device_resources,
            hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    else {
        unsigned int threads, groups;
        if (hip_selected_warp_geometry(
                max_batch, &threads, &groups) != 0)
            return -1;
        hip_q4_selected_warp_batch<<<dim3((unsigned int)max_rows,
            groups, (unsigned int)job_count), threads>>>(
            hip_device_desc, job_count, (const float *)arena->backend,
            (float *)arena->backend, hip_selected_device_resources,
            hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    }
    if (hip_success(hipGetLastError(), "selected shared launch") != 0 ||
        hip_success(hipDeviceSynchronize(),
            "hipDeviceSynchronize selected shared") != 0)
        return -1;
    hip_selected_reads_completed();
    completion_end = hip_host_now_ns();
    hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += logical_jobs;
    hip_selected_shared_calls++;
    hip_selected_shared_jobs += logical_jobs;
    hip_u64_add_saturating(&hip_selected_shared_h2d_ns,
        h2d_end - h2d_start);
    hip_u64_add_saturating(&hip_selected_shared_completion_ns,
        completion_end - h2d_end);
    if (getenv("SALT_HIP_OCCUPANCY"))
        fprintf(stderr,
            "SALT_HIP_SELECTED jobs=%d logical=%llu rows=%d max_batch=%d "
            "b1=%u b2=%u b3_4=%u b5_8=%u b9_16=%u b17_plus=%u "
            "descriptor_h2d_ns=%llu completion_ns=%llu\n",
            job_count, (unsigned long long)logical_jobs, max_rows, max_batch,
            occupancy[0], occupancy[1], occupancy[2], occupancy[3],
            occupancy[4], occupancy[5],
            (unsigned long long)(h2d_end - h2d_start),
            (unsigned long long)(completion_end - h2d_end));
    hip_path_record(max_batch == 1 ? HIP_PATH_SELECTED_SHARED_SINGLE
                                   : HIP_PATH_SELECTED_SHARED_WAVE,
        logical_jobs, logical_rows, UINT64_C(1), UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_proj_batch_indexed(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset, const void *id) {
    HipTensor *tensor;
    HipResource *resource;
    unsigned char *device;
    uint64_t voff, soff, boff, auxoff;
    uint64_t output_span, input_floats, selected_output_floats;
    size_t output_end;
    float *device_output;
    uint64_t h2d_start, h2d_end, launch_end;
    if (!hip_ready || !vals || !scales || !biases || !xs || !output ||
        !output->contents || !output->backend || !id ||
        R < 1 || C < 1 || C > HIP_MAX_C || B < 2 || B > 512 ||
        first_job < 0 || job_count < 1 || first_job > B - job_count)
        return -1;
    output_span = (uint64_t)(uint32_t)B * (uint32_t)R;
    input_floats = (uint64_t)(uint32_t)job_count * (uint32_t)C;
    selected_output_floats =
        (uint64_t)(uint32_t)job_count * (uint32_t)R;
    if (output_span > UINT32_MAX ||
        (uint64_t)y_float_offset > (uint64_t)UINT32_MAX - output_span ||
        output_span > (uint64_t)SIZE_MAX ||
        y_float_offset > SIZE_MAX - (size_t)output_span ||
        input_floats > HIP_MAX_OUTPUTS ||
        selected_output_floats > HIP_MAX_OUTPUTS)
        return -1;
    output_end = y_float_offset + (size_t)output_span;
    if (output_end > output->nbytes / sizeof(float)) return -1;
    tensor = hip_tensor(id);
    if (!tensor || tensor->bits != 4 || tensor->rows != R ||
        tensor->cols != C)
        return -1;
    if (hip_tensor_physical(tensor, &resource, &device,
            &voff, &soff, &boff, &auxoff) != 0 ||
        resource->host + tensor->value_offset !=
            (const unsigned char *)(const void *)vals ||
        resource->host + tensor->scale_offset !=
            (const unsigned char *)(const void *)scales ||
        resource->host + tensor->bias_offset !=
            (const unsigned char *)(const void *)biases)
        return -1;
    h2d_start = hip_host_now_ns();
    if (hip_success(hipMemcpy(
            hip_x, xs + (size_t)first_job * (size_t)C,
            (size_t)input_floats * sizeof(float), hipMemcpyHostToDevice),
            "hipMemcpy indexed input") != 0)
        return -1;
    h2d_end = hip_host_now_ns();
    device_output = (float *)output->backend + y_float_offset +
        (size_t)first_job * (size_t)R;
    if (job_count >= 8) {
        hip_q4_warp_batch<<<(unsigned int)R, 1024>>>(
            device, voff, soff, boff,
            hip_x, device_output, R, C, job_count);
    } else {
        hip_q4_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)job_count), 128>>>(
            device, voff, soff, boff, hip_x, device_output, R, C);
    }
    if (hip_success(hipGetLastError(),
                     "indexed projection launch") != 0)
        return -1;
    launch_end = hip_host_now_ns();
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) hip_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        hip_stats.expert_layer_batches++;
    hip_stats.direct_output_batches++;
    hip_stats.direct_output_jobs += (uint64_t)(uint32_t)job_count;
    hip_indexed_calls++;
    hip_indexed_jobs += (uint64_t)(uint32_t)job_count;
    hip_u64_add_saturating(&hip_indexed_h2d_ns, h2d_end - h2d_start);
    hip_u64_add_saturating(&hip_indexed_launch_ns, launch_end - h2d_end);
    hip_path_record(job_count >= 8 ? HIP_PATH_INDEXED_Q4_BATCH
                                   : HIP_PATH_INDEXED_Q4_EXACT,
        (uint64_t)(uint32_t)job_count, selected_output_floats,
        UINT64_C(1), UINT64_C(0));
    return 0;
}

extern "C" int salt_gpu_proj_batch_indexed_exact_views(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset, const void *id) {
    return salt_gpu_proj_batch_indexed(vals, scales, biases, R, C, B,
        first_job, job_count, xs, output, y_float_offset, id);
}

extern "C" int salt_gpu_set_defer(int on) {
    int enabled = on ? 1 : 0;
    int rc = 0;
    if (!enabled && hip_defer) rc = salt_gpu_sync();
    hip_defer = enabled;
    return rc;
}
extern "C" int salt_gpu_defer(void) { return hip_defer; }
extern "C" int salt_gpu_sync(void) {
    uint64_t start, end;
    if (!hip_ready) return 0;
    if (hip_component_pool_finalize() != 0) return -1;
    start = hip_host_now_ns();
    if (hip_success(hipDeviceSynchronize(), "hipDeviceSynchronize") != 0)
        return -1;
    hip_selected_reads_completed();
    end = hip_host_now_ns();
    hip_consumer_sync_calls++;
    hip_u64_add_saturating(&hip_consumer_sync_ns, end - start);
    hip_path_record(HIP_PATH_CONSUMER_SYNC, UINT64_C(0), UINT64_C(0),
        UINT64_C(0), UINT64_C(1));
    return 0;
}

static unsigned char *hip_text_bytes(
        HipTextProgramState *state, size_t offset) {
    return state->device_canonical + offset;
}

static float *hip_text_f32(HipTextProgramState *state, size_t offset) {
    return (float *)(void *)hip_text_bytes(state, offset);
}

static int32_t *hip_text_i32(HipTextProgramState *state, size_t offset) {
    return (int32_t *)(void *)hip_text_bytes(state, offset);
}

static const SaltTensorResourceSpec *hip_text_resource_spec(
        const SaltTextVerifyProgram *program,
        const SaltTensorStorageSpec *storage) {
    return salt_tensor_resource_find(program->descriptor->tensor_resources,
        program->descriptor->tensor_resource_count,
        storage->resource_kind, storage->resource_id);
}

static const unsigned char *hip_text_resource_device(
        const SaltTextVerifyProgram *program,
        const SaltTensorStorageSpec *storage) {
    const SaltTensorResourceSpec *spec =
        hip_text_resource_spec(program, storage);
    void *device = NULL;
    if (!spec || !spec->base || spec->mapped_nbytes < spec->nbytes)
        return NULL;
    for (int index = 0; index < HIP_MAX_RESOURCES; index++)
        if (hip_resources[index].used && hip_resources[index].active &&
            hip_resources[index].device &&
            hip_resources[index].host == spec->base &&
            hip_resources[index].nbytes >= spec->nbytes)
            return hip_resources[index].device;
    if (storage->source_class != SALT_TENSOR_SOURCE_SHARED)
        return NULL;
    if (hipHostGetDevicePointer(&device, (void *)spec->base, 0) != hipSuccess) {
        (void)hipGetLastError();
        return NULL;
    }
    return (const unsigned char *)device;
}

static int hip_text_realize_tensor(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        HipTextTensorRef *ref) {
    const unsigned char *resource;
    HipResource *registered;
    int resource_index;
    if (!program || !tensor || !ref ||
        salt_tensor_desc_validate(tensor) != 0)
        return -1;
    memset(ref, 0, sizeof *ref);
    if (hip_residency.permanent_published &&
        tensor->storage.source_class == SALT_TENSOR_SOURCE_STATIC &&
        tensor->storage.resource_kind == SALT_GPU_RESOURCE_TRUNK) {
        registered = hip_resource_record(
            tensor->storage.resource_kind, tensor->storage.resource_id);
        if (!registered || !registered->active) return -1;
        resource_index = (int)(registered - hip_resources);
        ref->resource = hip_residency.device_arena;
        if (hip_component_physical_offset(resource_index,
                tensor->storage.value_offset, tensor->storage.value_bytes,
                &ref->value_offset) != 0 ||
            (tensor->storage.scale_bytes != 0 &&
             hip_component_physical_offset(resource_index,
                tensor->storage.scale_offset, tensor->storage.scale_bytes,
                &ref->scale_offset) != 0) ||
            (tensor->storage.bias_bytes != 0 &&
             hip_component_physical_offset(resource_index,
                tensor->storage.bias_offset, tensor->storage.bias_bytes,
                &ref->bias_offset) != 0)) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "SALT_HIP_RESIDENCY_TENSOR_MISS resource=%u:%u "
                    "encoding=%u rows=%u cols=%u value=%llu:%llu "
                    "scale=%llu:%llu bias=%llu:%llu\n",
                    tensor->storage.resource_kind,
                    tensor->storage.resource_id,
                    (unsigned int)tensor->storage.encoding,
                    tensor->rows, tensor->cols,
                    (unsigned long long)tensor->storage.value_offset,
                    (unsigned long long)tensor->storage.value_bytes,
                    (unsigned long long)tensor->storage.scale_offset,
                    (unsigned long long)tensor->storage.scale_bytes,
                    (unsigned long long)tensor->storage.bias_offset,
                    (unsigned long long)tensor->storage.bias_bytes);
            return -1;
        }
    } else {
        resource = hip_text_resource_device(program, &tensor->storage);
        if (!resource) return -1;
        ref->resource = resource;
        ref->value_offset = tensor->storage.value_offset;
        ref->scale_offset = tensor->storage.scale_offset;
        ref->bias_offset = tensor->storage.bias_offset;
    }
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)tensor->storage.encoding;
    return 0;
}

static int hip_text_realize_optional(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        HipTextTensorRef *ref) {
    if (salt_tensor_desc_absent(tensor)) {
        memset(ref, 0, sizeof *ref);
        return 0;
    }
    return hip_text_realize_tensor(program, tensor, ref);
}

static int hip_text_defer_selected_tensor(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        HipTextTensorRef *ref) {
    const SaltTensorResourceSpec *spec;
    if (!program || !tensor || !ref ||
        salt_tensor_desc_validate(tensor) != 0 ||
        !(spec = hip_text_resource_spec(program, &tensor->storage)) ||
        spec->source_class != SALT_TENSOR_SOURCE_SELECTED)
        return -1;
    memset(ref, 0, sizeof *ref);
    ref->value_offset = tensor->storage.value_offset;
    ref->scale_offset = tensor->storage.scale_offset;
    ref->bias_offset = tensor->storage.bias_offset;
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)tensor->storage.encoding;
    return 0;
}

static int hip_text_patch_selected_tensor(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        uint64_t payload_base_offset, int slot, HipTextTensorRef *ref) {
    const SaltTensorResourceSpec *spec;
    HipSelectedResource *selected;
    uintptr_t payload_base, selected_base;
    const uint64_t offsets[3] = {
        tensor ? tensor->storage.value_offset : 0,
        tensor ? tensor->storage.scale_offset : 0,
        tensor ? tensor->storage.bias_offset : 0,
    };
    const uint64_t component_bytes[3] = {
        tensor ? tensor->storage.value_bytes : 0,
        tensor ? tensor->storage.scale_bytes : 0,
        tensor ? tensor->storage.bias_bytes : 0,
    };
    uint64_t local[3];
    if (!program || !tensor || !ref || slot < 0 ||
        slot >= hip_selected_capacity ||
        !(spec = hip_text_resource_spec(program, &tensor->storage)) ||
        spec->source_class != SALT_TENSOR_SOURCE_SELECTED || !spec->base ||
        tensor->storage.logical_resource_id >=
            (uint64_t)(uint32_t)hip_selected_logical_capacity)
        return -1;
    selected = &hip_selected_resources[slot];
    if (!selected->active ||
        (!selected->registered && !selected->resident) || !selected->host ||
        !selected->payload || !selected->device ||
        selected->logical_resource_id !=
            tensor->storage.logical_resource_id ||
        hip_selected_logical_slots[tensor->storage.logical_resource_id] != slot)
        return -1;
    payload_base = (uintptr_t)(const void *)selected->payload;
    selected_base = (uintptr_t)(const void *)selected->host;
    for (int index = 0; index < 3; index++) {
        uintptr_t pointer;
        uint64_t delta;
        if (offsets[index] < payload_base_offset ||
            (delta = offsets[index] - payload_base_offset) > UINTPTR_MAX ||
            payload_base > UINTPTR_MAX - (uintptr_t)delta)
            return -1;
        pointer = payload_base + (uintptr_t)delta;
        if (pointer < selected_base ||
            pointer - selected_base > selected->nbytes ||
            component_bytes[index] >
                selected->nbytes - (size_t)(pointer - selected_base))
            return -1;
        local[index] = (uint64_t)(pointer - selected_base);
    }
    ref->resource = selected->device;
    ref->value_offset = local[0];
    ref->scale_offset = local[1];
    ref->bias_offset = local[2];
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)tensor->storage.encoding;
    return 0;
}

static const float *hip_text_tensor_f32(const HipTextTensorRef *ref) {
    return ref && ref->resource &&
        ref->encoding == SALT_TENSOR_ENCODING_F32
        ? (const float *)(const void *)(ref->resource + ref->value_offset)
        : NULL;
}

static int hip_text_direct_q4_descriptor(
        const HipTextProgramState *state, const HipTextTensorRef *ref,
        uint32_t batch, size_t input_byte_offset, size_t output_byte_offset,
        HipBatchDesc *descriptor) {
    uint64_t input_elements, output_elements;
    if (!state || !ref || !descriptor || !ref->resource || batch == 0 ||
        ref->encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        ref->rows == 0 || ref->cols == 0 ||
        (input_byte_offset | output_byte_offset) % sizeof(float) != 0 ||
        batch > UINT64_MAX / ref->cols || batch > UINT64_MAX / ref->rows)
        return -1;
    input_elements = (uint64_t)batch * ref->cols;
    output_elements = (uint64_t)batch * ref->rows;
    if (input_byte_offset > state->canonical_bytes ||
        input_elements >
            (state->canonical_bytes - input_byte_offset) / sizeof(float) ||
        output_byte_offset > state->canonical_bytes ||
        output_elements >
            (state->canonical_bytes - output_byte_offset) / sizeof(float) ||
        input_byte_offset / sizeof(float) > UINT32_MAX ||
        output_byte_offset / sizeof(float) > UINT32_MAX)
        return -1;
    memset(descriptor, 0, sizeof *descriptor);
    descriptor->resource = ref->resource;
    descriptor->value_offset = ref->value_offset;
    descriptor->scale_offset = ref->scale_offset;
    descriptor->bias_offset = ref->bias_offset;
    descriptor->input_offset =
        (uint32_t)(input_byte_offset / sizeof(float));
    descriptor->output_offset =
        (uint32_t)(output_byte_offset / sizeof(float));
    descriptor->rows = (int32_t)ref->rows;
    descriptor->cols = (int32_t)ref->cols;
    descriptor->batch = (int32_t)batch;
    return 0;
}

static int hip_text_launch_q4_pair(HipTextProgramState *state,
        uint32_t descriptor_base, uint32_t batch, uint32_t max_rows,
        uint32_t descriptor_count) {
    if (!state || !state->host_selected_desc ||
        !state->device_selected_desc ||
        (descriptor_count != 2u &&
         (descriptor_count != 3u || batch != 1u)) ||
        descriptor_base > state->selected_job_capacity *
            HIP_TEXT_EXPERT_PROJECTIONS - descriptor_count ||
        batch == 0 || max_rows == 0 ||
        hip_success(hipMemcpyAsync(state->device_selected_desc + descriptor_base,
            state->host_selected_desc + descriptor_base,
            descriptor_count * sizeof(HipBatchDesc),
            hipMemcpyHostToDevice, state->stream),
            "hipMemcpy text Q4 pair descriptors") != 0)
        return -1;
    if (batch == 1u) {
        hip_q4_heterogeneous_warp<<<dim3((max_rows + 3u) / 4u, 1u,
            descriptor_count),
            128, 0, state->stream>>>(
            state->device_selected_desc + descriptor_base,
            (int)descriptor_count,
            (const float *)state->device_canonical,
            (float *)state->device_canonical,
            hip_selected_device_resources,
            hip_selected_device_logical_slots,
            hip_selected_device_payload_offsets);
    } else {
        hip_q4_heterogeneous_warp_batch<<<dim3(max_rows,
            (batch + 511u) / 512u, 2u), 1024, 0, state->stream>>>(
            state->device_selected_desc + descriptor_base, 2,
            (const float *)state->device_canonical,
            (float *)state->device_canonical);
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

static int hip_text_project(
        HipTextProgramState *state, const HipTextTensorRef *ref,
        const float *input, float *output, uint32_t batch) {
    dim3 block(128u, 1u, 1u);
    dim3 grid((ref->rows + 3u) / 4u, batch, 1u);
    if (!state || !ref || !ref->resource || !input || !output || batch == 0 ||
        ref->rows == 0 || ref->cols == 0)
        return -1;
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
        if (hip_text_nrow_enabled && batch >= hip_text_nrow_min_b)
            hip_q4_warp_batch<<<ref->rows, 1024, 0, state->stream>>>(
                ref->resource, ref->value_offset, ref->scale_offset,
                ref->bias_offset, input, output, (int)ref->rows,
                (int)ref->cols, (int)batch);
        else
            hip_q4_warp_exact<<<grid, block, 0, state->stream>>>(
                ref->resource, ref->value_offset, ref->scale_offset,
                ref->bias_offset, input, output, (int)ref->rows,
                (int)ref->cols);
    } else if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8) {
        if (hip_text_nrow_enabled && batch >= hip_text_nrow_min_b) {
            dim3 q8_grid(ref->rows, (batch + 127u) / 128u, 1u);
            hip_q8_shared_tile<<<q8_grid, 128, 0, state->stream>>>(
                ref->resource, ref->value_offset, ref->scale_offset,
                ref->bias_offset, input, output, (int)ref->rows,
                (int)ref->cols, (int)batch);
        } else {
            hip_q8_warp_exact<<<grid, block, 0, state->stream>>>(
                ref->resource, ref->value_offset, ref->scale_offset,
                ref->bias_offset, input, output, (int)ref->rows,
                (int)ref->cols);
        }
    } else if (ref->encoding == SALT_TENSOR_ENCODING_BF16) {
        if (hip_text_nrow_enabled && batch > 1u) {
            unsigned int threads, groups;
            if (hip_bf16_warp_geometry(batch, &threads, &groups) != 0)
                return -1;
            hip_bf16_warp_batch_exact<<<dim3(ref->rows, groups, 1u),
                threads, 0, state->stream>>>(ref->resource,
                ref->value_offset, input, output, (int)ref->rows,
                (int)ref->cols, (int)batch);
        } else {
            dim3 one_grid((ref->rows + 3u) / 4u, 1u, 1u);
            hip_bf16_warp_exact<<<one_grid, block, 0, state->stream>>>(
                ref->resource, ref->value_offset, input, output,
                (int)ref->rows, (int)ref->cols);
        }
    } else {
        return -1;
    }
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

static int hip_text_map_kv(
        const SaltTextKvLayerDesc *layer, uint32_t width,
        HipTextKvRef *output) {
    const SaltTextKvRowsDesc *keys, *values;
    void *mapped_keys = NULL, *mapped_values = NULL;
    uintptr_t host_base, key_address, value_address;
    size_t key_offset, value_offset, capacity;
    HipAttentionKvSync *sync;
    if (!layer || !output || width == 0 || !hip_attention_kv ||
        !hip_attention_kv_host || hip_attention_kv_nbytes < sizeof(float))
        return -1;
    keys = &layer->keys;
    values = &layer->values;
    if ((keys->private_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
         keys->private_mode != SALT_TEXT_KV_PRIVATE_RING) ||
        values->private_mode != keys->private_mode ||
        keys->private_position_base != 0 ||
        values->private_position_base != 0 ||
        keys->private_row_stride != width ||
        values->private_row_stride != width ||
        keys->private_row_capacity == 0 ||
        values->private_row_capacity != keys->private_row_capacity ||
        keys->private_float_capacity <
            (size_t)keys->private_row_capacity * width ||
        values->private_float_capacity <
            (size_t)values->private_row_capacity * width ||
        !keys->private_rows || !values->private_rows ||
        (keys->shared_prefix_state &&
         keys->shared_prefix_state->row_count != 0) ||
        (values->shared_prefix_state &&
         values->shared_prefix_state->row_count != 0))
        return -1;
    host_base = (uintptr_t)hip_attention_kv_host;
    key_address = (uintptr_t)(const void *)keys->private_rows;
    value_address = (uintptr_t)(const void *)values->private_rows;
    if (key_address < host_base || value_address < host_base ||
        key_address - host_base > SIZE_MAX ||
        value_address - host_base > SIZE_MAX ||
        (key_address - host_base) % sizeof(float) != 0 ||
        (value_address - host_base) % sizeof(float) != 0)
        return -1;
    key_offset = (size_t)(key_address - host_base) / sizeof(float);
    value_offset = (size_t)(value_address - host_base) / sizeof(float);
    capacity = hip_attention_kv_nbytes / sizeof(float);
    if ((size_t)keys->private_row_capacity > SIZE_MAX / width ||
        (size_t)keys->private_row_capacity * width > capacity ||
        key_offset > capacity - (size_t)keys->private_row_capacity * width ||
        value_offset > capacity - (size_t)keys->private_row_capacity * width ||
        hipHostGetDevicePointer(&mapped_keys, keys->private_rows, 0) !=
            hipSuccess ||
        hipHostGetDevicePointer(&mapped_values, values->private_rows, 0) !=
            hipSuccess ||
        !(sync = hip_attention_sync_slot(key_offset, value_offset, (int)width))) {
        (void)hipGetLastError();
        return -1;
    }
    memset(output, 0, sizeof *output);
    output->keys = (float *)mapped_keys;
    output->values = (float *)mapped_values;
    output->device_keys = hip_attention_kv + key_offset;
    output->device_values = hip_attention_kv + value_offset;
    output->host_keys = keys->private_rows;
    output->host_values = values->private_rows;
    output->sync = sync;
    output->key_float_offset = key_offset;
    output->value_float_offset = value_offset;
    output->width = width;
    output->private_mode = keys->private_mode;
    output->row_capacity = keys->private_row_capacity;
    return 0;
}

static int hip_text_requirements(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextGpuProgramRequirements *requirements,
        size_t requirements_size) {
    uint64_t selected_jobs;
    if (!program || !program->ready || !program->descriptor ||
        !program->descriptor->model || !policy || !requirements ||
        requirements_size != sizeof *requirements || !hip_ready ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        program->layer_count == 0 || program->hidden == 0 ||
        program->vocabulary == 0 || program->maximum_candidates == 0 ||
        program->dispatch.cell_count == 0 || !program->dispatch.cells ||
        program->layout.maximum_experts == 0 ||
        program->layout.maximum_experts > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS ||
        program->layout.maximum_topk == 0 ||
        program->layout.maximum_topk > program->layout.maximum_experts ||
        hip_u64_mul(program->maximum_candidates,
            program->layout.maximum_topk, &selected_jobs) != 0 ||
        /* Selected row/expert jobs use the existing HIP descriptor capacity;
         * unique expert resources retain their separate bound above. */
        selected_jobs > HIP_MAX_BATCH_JOBS)
        return -1;
    memset(requirements, 0, sizeof *requirements);
    requirements->backend_state_bytes = sizeof(HipTextProgramState);
    requirements->command_bytes = sizeof(HipTextProgramCommand);
    requirements->maximum_commands = program->dispatch.cell_count;
    return 0;
}

static void hip_text_state_release(HipTextProgramState *state) {
    if (!state) return;
    if (state->stream) (void)hipStreamSynchronize(state->stream);
    for (uint32_t index = 0; index < state->cell_event_capacity; index++) {
        if (state->cell_event_start && state->cell_event_start[index])
            (void)hipEventDestroy(state->cell_event_start[index]);
        if (state->cell_event_end && state->cell_event_end[index])
            (void)hipEventDestroy(state->cell_event_end[index]);
    }
    if (state->device_source_position)
        (void)hipFree(state->device_source_position);
    if (state->device_canonical)
        (void)hipFree(state->device_canonical);
    if (state->device_selected_desc)
        (void)hipFree(state->device_selected_desc);
    if (state->host_selected_desc)
        (void)hipHostFree(state->host_selected_desc);
    if (state->stream) (void)hipStreamDestroy(state->stream);
    if (state->device_experts) (void)hipFree(state->device_experts);
    if (state->device_kv) (void)hipFree(state->device_kv);
    if (state->canonical.contents || state->canonical.backend)
        (void)salt_gpu_shared_buffer_free(&state->canonical);
    free(state->templates);
    free(state->cell_event_end);
    free(state->cell_event_start);
    free(state->expert_refs);
    free(state->layers);
    memset(state, 0, sizeof *state);
}

static int hip_text_cell_matrix_rows(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell, uint32_t *rows_out) {
    const SaltTextCompiledLayer *compiled = NULL;
    const SaltTextLayerPlan *plan = NULL;
    uint32_t rows = 0;
    if (!program || !cell || !rows_out) return -1;
    if (cell->layer < program->layer_count) {
        compiled = &program->layers[cell->layer];
        if (!compiled->descriptor || !(plan = compiled->descriptor->plan))
            return -1;
    }
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING:
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
    case SALT_TEXT_CELL_DENSE_NORM:
    case SALT_TEXT_CELL_ROUTER_INPUT:
    case SALT_TEXT_CELL_ROUTED_NORM:
    case SALT_TEXT_CELL_DENSE_DOWN:
    case SALT_TEXT_CELL_EXPERT_DOWN:
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
    case SALT_TEXT_CELL_FFN_COMBINE:
    case SALT_TEXT_CELL_FINAL_NORM:
        rows = program->hidden;
        break;
    case SALT_TEXT_CELL_QUERY_PROJECTION:
    case SALT_TEXT_CELL_ATTENTION_BODY:
        rows = compiled ? compiled->query_width : 0;
        break;
    case SALT_TEXT_CELL_KEY_PROJECTION:
    case SALT_TEXT_CELL_VALUE_PROJECTION:
    case SALT_TEXT_CELL_ATTENTION_TRANSFORM:
        rows = compiled ? compiled->kv_width : 0;
        break;
    case SALT_TEXT_CELL_DENSE_GATE:
    case SALT_TEXT_CELL_DENSE_UP:
    case SALT_TEXT_CELL_DENSE_ACTIVATION:
        rows = plan ? (uint32_t)plan->dense_intermediate : 0;
        break;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        rows = plan ? (uint32_t)plan->n_experts : 0;
        break;
    case SALT_TEXT_CELL_ROUTER_TOPK:
        rows = plan ? (uint32_t)plan->top_k_experts : 0;
        break;
    case SALT_TEXT_CELL_EXPERT_GATE:
    case SALT_TEXT_CELL_EXPERT_UP:
    case SALT_TEXT_CELL_EXPERT_ACTIVATION:
        rows = plan ? (uint32_t)plan->expert_intermediate : 0;
        break;
    case SALT_TEXT_CELL_FINAL_HEAD:
    case SALT_TEXT_CELL_LOGIT_SOFTCAP:
        rows = program->vocabulary;
        break;
    default:
        return -1;
    }
    if (rows == 0) return -1;
    *rows_out = rows;
    return 0;
}

static int hip_text_cell_matrix_validate(HipTextProgramState *state,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell) {
    uint32_t rows;
    uint64_t row_bytes, bytes, end;
    uint64_t hash;
    if (!state || !program || !cell ||
        hip_text_cell_matrix_rows(program, cell, &rows) != 0 ||
        hip_u64_mul(rows, sizeof(float), &row_bytes) != 0 ||
        row_bytes > SIZE_MAX ||
        cell->destination_stride != (size_t)row_bytes ||
        cell->logical_capacity == 0 ||
        cell->logical_capacity > UINT64_MAX / cell->destination_stride)
        return -1;
    bytes = (uint64_t)cell->logical_capacity * cell->destination_stride;
    end = (uint64_t)cell->destination_offset + bytes;
    if (end < cell->destination_offset || end > program->layout.total_bytes)
        return -1;
    hash = state->address_matrix_digest;
#define HIP_MATRIX_HASH(value) do { \
        hash ^= (uint64_t)(value); \
        hash *= UINT64_C(1099511628211); \
    } while (0)
    HIP_MATRIX_HASH((uint32_t)cell->kind);
    HIP_MATRIX_HASH(cell->layer);
    HIP_MATRIX_HASH(cell->logical_capacity);
    HIP_MATRIX_HASH(cell->destination_offset);
    HIP_MATRIX_HASH(cell->destination_stride);
    HIP_MATRIX_HASH(rows);
#undef HIP_MATRIX_HASH
    state->address_matrix_digest = hash;
    state->address_matrix_cells++;
    return 0;
}

static int hip_text_prepare_templates(
        HipTextProgramState *state, const SaltTextVerifyProgram *program) {
    uint32_t selected_capacity = 0;
    if (!state || !program || !state->templates ||
        state->template_capacity < program->dispatch.cell_count)
        return -1;
    memset(state->templates, 0,
        (size_t)state->template_capacity * sizeof *state->templates);
    state->address_matrix_digest = UINT64_C(1469598103934665603);
    state->address_matrix_cells = 0;
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        HipTextCellTemplate *target = &state->templates[index];
        if (hip_text_cell_matrix_validate(state, program, cell) != 0)
            return -1;
        target->cell = cell;
        target->index = index;
        target->kind = (uint32_t)cell->kind;
        target->unit = (uint32_t)cell->unit;
        target->layer = cell->layer;
        target->logical_capacity = cell->logical_capacity;
        target->dependency_epoch = cell->dependency_epoch;
        target->completion_epoch = cell->completion_epoch;
        target->hidden = program->hidden;
        target->destination_offset = cell->destination_offset;
        target->destination_stride = cell->destination_stride;
        target->flags = cell->flags;
        if (cell->layer < program->layer_count) {
            const SaltTextLayerPlan *plan =
                program->layers[cell->layer].descriptor->plan;
            uint64_t jobs;
            if (!plan || plan->dense_intermediate < 1 ||
                plan->expert_intermediate < 1 || plan->n_experts < 1 ||
                plan->top_k_experts < 1)
                return -1;
            target->dense = (uint32_t)plan->dense_intermediate;
            target->routed = (uint32_t)plan->expert_intermediate;
            target->experts = (uint32_t)plan->n_experts;
            target->topk = (uint32_t)plan->top_k_experts;
            jobs = (uint64_t)program->maximum_candidates * target->topk;
            if (jobs > UINT32_MAX) return -1;
            if ((uint32_t)jobs > selected_capacity)
                selected_capacity = (uint32_t)jobs;
        }
    }
    if (selected_capacity == 0u) return -1;
    state->template_count = program->dispatch.cell_count;
    state->selected_job_capacity = selected_capacity;
    if (getenv("SALT_HIP_TEXT_ADDRESS_MATRIX"))
        fprintf(stderr,
            "SALT_HIP_ADDRESS_MATRIX cells=%u digest=%016llx bytes=%zu\n",
            state->address_matrix_cells,
            (unsigned long long)state->address_matrix_digest,
            program->layout.total_bytes);
    return 0;
}

static int hip_text_prepare(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    size_t bytes;
    uint64_t expert_ref_count, expert_ref_bytes, selected_descriptor_bytes;
    HipTextKvRef host_kv[64];
    const char *failure_stage = "validate";
    uint32_t failure_layer = UINT32_MAX, failure_expert = UINT32_MAX;
    if (!program || !plan || plan->program != program ||
        plan->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !canonical_out || !state || backend_state_bytes < sizeof *state ||
        !command || command_bytes < sizeof(HipTextProgramCommand) ||
        program->layout.total_bytes > SIZE_MAX - 64u)
        return -1;
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof(HipTextProgramCommand));
    bytes = (program->layout.total_bytes + 63u) & ~(size_t)63u;
    if (bytes > SIZE_MAX - sizeof(int)) return -1;
    state->status_offset = bytes;
    bytes += sizeof(int);
    failure_stage = "arenas";
    if (salt_gpu_shared_buffer_alloc(&state->canonical, bytes) != 0 ||
        hip_success(hipMalloc((void **)&state->device_canonical, bytes),
            "hipMalloc text canonical arena") != 0 ||
        hip_success(hipStreamCreateWithFlags(
            &state->stream, hipStreamNonBlocking),
            "hipStreamCreate text program") != 0)
        goto fail;
    state->program = program;
    state->canonical_bytes = program->layout.total_bytes;
    state->device_canonical_bytes = bytes;
    if (hip_u64_mul(program->layer_count, program->layout.maximum_experts,
            &expert_ref_count) != 0 ||
        hip_u64_mul(expert_ref_count, HIP_TEXT_EXPERT_PROJECTIONS,
            &expert_ref_count) != 0 ||
        hip_u64_mul(expert_ref_count, sizeof(HipTextTensorRef),
            &expert_ref_bytes) != 0 ||
        expert_ref_count > UINT32_MAX || expert_ref_bytes > SIZE_MAX)
        goto fail;
    state->layer_capacity = program->layer_count;
    state->experts_per_layer = program->layout.maximum_experts;
    state->expert_ref_capacity = (uint32_t)expert_ref_count;
    state->template_capacity = program->dispatch.cell_count;
    failure_stage = "metadata";
    state->layers = (HipTextLayerRefs *)calloc(state->layer_capacity,
        sizeof *state->layers);
    state->expert_refs = (HipTextTensorRef *)calloc(
        state->expert_ref_capacity, sizeof *state->expert_refs);
    state->templates = (HipTextCellTemplate *)calloc(state->template_capacity,
        sizeof *state->templates);
    if (!state->layers || !state->expert_refs || !state->templates)
        goto fail;
    state->cell_timing_enabled = getenv("SALT_HIP_TEXT_CELL_TIMING") != NULL;
    if (state->cell_timing_enabled) {
        state->cell_event_capacity = state->template_capacity;
        state->cell_event_start = (hipEvent_t *)calloc(
            state->cell_event_capacity, sizeof *state->cell_event_start);
        state->cell_event_end = (hipEvent_t *)calloc(
            state->cell_event_capacity, sizeof *state->cell_event_end);
        if (!state->cell_event_start || !state->cell_event_end)
            goto fail;
        for (uint32_t index = 0; index < state->cell_event_capacity; index++)
            if (hip_success(hipEventCreate(&state->cell_event_start[index]),
                    "hipEventCreate text cell start") != 0 ||
                hip_success(hipEventCreate(&state->cell_event_end[index]),
                    "hipEventCreate text cell end") != 0)
                goto fail;
    }
    failure_stage = "fixed-embedding";
    if (hip_text_realize_tensor(program, &program->descriptor->embedding,
            &state->embedding) != 0)
        goto fail;
    failure_stage = "fixed-final-norm";
    if (hip_text_realize_tensor(program, &program->descriptor->final_norm,
            &state->final_norm) != 0)
        goto fail;
    failure_stage = "fixed-output-head";
    if (hip_text_realize_tensor(program, &program->descriptor->output_head,
            &state->output_head) != 0)
        goto fail;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        const SaltTextLayerExecDesc *source = compiled->descriptor;
        HipTextLayerRefs *target = &state->layers[layer];
        failure_layer = layer;
        failure_stage = "layer-fixed";
#define REALIZE(field) \
        if (hip_text_realize_tensor(program, &source->field, \
                &target->field) != 0) goto fail
#define OPTIONAL(field) \
        if (hip_text_realize_optional(program, &source->field, \
                &target->field) != 0) goto fail
        REALIZE(q); REALIZE(k); OPTIONAL(v); REALIZE(o);
        REALIZE(dense_gate); REALIZE(dense_up); REALIZE(dense_down);
        REALIZE(router); REALIZE(pre_attention_norm); REALIZE(q_norm);
        REALIZE(k_norm); REALIZE(post_attention_norm);
        REALIZE(pre_ffn_norm_1); REALIZE(pre_ffn_norm_2);
        REALIZE(post_ffn_norm_1); REALIZE(post_ffn_norm_2);
        REALIZE(post_ffn_norm); REALIZE(router_scale);
        REALIZE(per_expert_scale); OPTIONAL(layer_scalar);
#undef REALIZE
#undef OPTIONAL
        failure_stage = "layer-kv";
        if (!source->kv || layer >= 64u ||
            compiled->tentative_key_offset % sizeof(float) != 0 ||
            compiled->tentative_value_offset % sizeof(float) != 0 ||
            hip_text_map_kv(source->kv, compiled->kv_width,
                &target->kv) != 0 || source->expert_count == 0 ||
            source->expert_count > state->experts_per_layer)
            goto fail;
        target->kv.tentative_key_float_offset =
            compiled->tentative_key_offset / sizeof(float);
        target->kv.tentative_value_float_offset =
            compiled->tentative_value_offset / sizeof(float);
        for (uint32_t expert = 0; expert < source->expert_count; expert++) {
            const SaltTextExpertDesc *entry = &source->experts[expert];
            size_t base = ((size_t)layer * state->experts_per_layer + expert) *
                HIP_TEXT_EXPERT_PROJECTIONS;
            failure_expert = expert;
            failure_stage = "layer-expert";
            if (entry->expert_id != expert ||
                hip_text_defer_selected_tensor(program, &entry->gate,
                    &state->expert_refs[base]) != 0 ||
                hip_text_defer_selected_tensor(program, &entry->up,
                    &state->expert_refs[base + 1u]) != 0 ||
                hip_text_defer_selected_tensor(program, &entry->down,
                    &state->expert_refs[base + 2u]) != 0)
                goto fail;
        }
    }
    failure_layer = UINT32_MAX;
    failure_expert = UINT32_MAX;
    failure_stage = "templates-and-selected";
    if (program->layer_count > 64u)
        goto fail;
    for (uint32_t layer = 0; layer < program->layer_count; layer++)
        host_kv[layer] = state->layers[layer].kv;
    if (hip_text_prepare_templates(state, program) != 0 ||
        hip_u64_mul(state->selected_job_capacity,
            HIP_TEXT_EXPERT_PROJECTIONS * sizeof(HipBatchDesc),
            &selected_descriptor_bytes) != 0 ||
        selected_descriptor_bytes > SIZE_MAX ||
        hip_success(hipHostMalloc((void **)&state->host_selected_desc,
            (size_t)selected_descriptor_bytes,
            hipHostMallocDefault), "hipHostMalloc text selected descriptors") != 0 ||
        hip_success(hipMalloc((void **)&state->device_selected_desc,
            (size_t)selected_descriptor_bytes),
            "hipMalloc text selected descriptors") != 0 ||
        hip_success(hipMalloc((void **)&state->device_experts,
            (size_t)expert_ref_bytes), "hipMalloc text expert refs") != 0 ||
        hip_success(hipMalloc((void **)&state->device_kv,
            (size_t)program->layer_count * sizeof *state->device_kv),
            "hipMalloc text KV refs") != 0 ||
        hip_success(hipMalloc((void **)&state->device_source_position,
            sizeof(uint32_t)), "hipMalloc text source position") != 0 ||
        hip_success(hipMemcpy(state->device_experts, state->expert_refs,
            (size_t)expert_ref_bytes, hipMemcpyHostToDevice),
            "hipMemcpy text expert refs") != 0 ||
        hip_success(hipMemcpy(state->device_kv, host_kv,
            (size_t)program->layer_count * sizeof *state->device_kv,
            hipMemcpyHostToDevice), "hipMemcpy text KV refs") != 0)
        goto fail;
    state->magic = HIP_TEXT_STATE_MAGIC;
    state->ready = 1;
    *canonical_out = &state->canonical;
    return 0;
fail:
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_TEXT_PREPARE_FAIL stage=%s layer=%u expert=%u\n",
            failure_stage, failure_layer, failure_expert);
    hip_text_state_release(state);
    return -1;
}

static int hip_text_begin_depth(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t maximum_depth, uint32_t output_rows) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !state->ready ||
        !cmd || cmd->begun || generation == 0 || !input_token_ids ||
        input_count == 0 || input_count > state->program->maximum_candidates ||
        source_position >= state->program->maximum_context ||
        maximum_depth >= state->program->maximum_context - source_position ||
        output_rows > input_count)
        return -1;
    memset(cmd, 0, sizeof *cmd);
    cmd->ffn_norm_wave_layer = UINT32_MAX;
    cmd->active_resource_layer = UINT32_MAX;
    if (salt_text_attention_score_rows(state->program, source_position,
            maximum_depth, &cmd->attention_score_rows) != 0 ||
        (output_rows == input_count
            ? salt_text_touched_span_plan(state->program, input_count,
                cmd->attention_score_rows,
                cmd->touched_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
                &cmd->touched_span_count, &cmd->touched_span_bytes)
            : salt_text_touched_span_plan_outputs(state->program, input_count,
                cmd->attention_score_rows, output_rows,
                cmd->touched_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
                &cmd->touched_span_count, &cmd->touched_span_bytes)) != 0)
        return -1;
    cmd->stats.canonical_clear_bytes = cmd->touched_span_bytes;
    state->source_position_host = source_position;
    for (uint32_t row = 0; row < input_count; row++) {
        state->parent_rows_host[row] = row == 0u ? UINT32_MAX : row - 1u;
        state->depths_host[row] = row;
    }
    for (uint32_t index = 0; index < cmd->touched_span_count; index++)
        if (hip_success(hipMemsetAsync(
                hip_text_bytes(state, cmd->touched_spans[index].offset), 0,
                cmd->touched_spans[index].bytes, state->stream),
                "hipMemset text touched span") != 0)
            return -1;
    if (hip_success(hipMemsetAsync(hip_text_bytes(state,
            state->status_offset), 0, sizeof(int), state->stream),
            "hipMemset text status") != 0 ||
        hip_success(hipMemcpyAsync(hip_text_i32(state,
            state->program->layout.candidate_token_ids), input_token_ids,
            (size_t)input_count * sizeof(int32_t), hipMemcpyHostToDevice,
            state->stream), "hipMemcpy text token ids") != 0 ||
        hip_success(hipMemcpyAsync(hip_text_bytes(state,
            state->program->layout.target_parent_rows),
            state->parent_rows_host,
            (size_t)input_count * sizeof(uint32_t), hipMemcpyHostToDevice,
            state->stream), "hipMemcpy text parent rows") != 0 ||
        hip_success(hipMemcpyAsync(hip_text_bytes(state,
            state->program->layout.target_depths), state->depths_host,
            (size_t)input_count * sizeof(uint32_t), hipMemcpyHostToDevice,
            state->stream), "hipMemcpy text depths") != 0 ||
        hip_success(hipMemcpyAsync(state->device_source_position,
            &state->source_position_host, sizeof(uint32_t),
            hipMemcpyHostToDevice, state->stream),
            "hipMemcpy text source position") != 0)
        return -1;
    cmd->generation = generation;
    cmd->source_position = source_position;
    cmd->input_count = input_count;
    cmd->maximum_depth = maximum_depth;
    cmd->authoritative_output_rows = output_rows;
    cmd->stats.backend_template_count = state->template_count;
    cmd->stats.backend_dynamic_patches = 4u;
    cmd->stats.backend_selected_job_capacity = state->selected_job_capacity;
    cmd->phase = HIP_TEXT_PHASE_PREFIX;
    cmd->begun = 1;
    return 0;
}

static int hip_text_begin(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    if (input_count == 0) return -1;
    return hip_text_begin_depth(backend_state, command, generation,
        source_position, input_token_ids, input_count, input_count - 1u,
        input_count);
}

static int hip_text_begin_authoritative(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (hip_text_begin(backend_state, command, generation, source_position,
            input_token_ids, input_count) != 0 || !cmd)
        return -1;
    cmd->authoritative = 1u;
    return 0;
}

static int hip_text_begin_authoritative_output(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows) {
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (output_rows > 1u ||
        hip_text_begin_depth(backend_state, command, generation,
            source_position, input_token_ids, input_count, input_count - 1u,
            output_rows) != 0 || !cmd)
        return -1;
    cmd->authoritative = 1u;
    return 0;
}

static int hip_text_begin_frontier(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids,
        const uint32_t *parent_rows, const uint32_t *depths,
        uint32_t input_count) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    uint32_t maximum_depth = 0u;
    if (!state || !state->program || !cmd || !parent_rows || !depths ||
        input_count == 0 || input_count > state->program->maximum_candidates)
        return -1;
    for (uint32_t row = 0; row < input_count; row++)
        if (depths[row] > maximum_depth) maximum_depth = depths[row];
    if (maximum_depth == UINT32_MAX ||
        source_position > state->program->maximum_context -
            (maximum_depth + 1u) ||
        hip_text_begin_depth(backend_state, command, generation,
            source_position, input_token_ids, input_count, maximum_depth,
            input_count) != 0)
        return -1;
    memcpy(state->parent_rows_host, parent_rows,
        (size_t)input_count * sizeof(uint32_t));
    memcpy(state->depths_host, depths,
        (size_t)input_count * sizeof(uint32_t));
    state->source_position_host = source_position;
    cmd->source_position = source_position;
    if (hip_success(hipMemcpyAsync(hip_text_bytes(state,
            state->program->layout.target_parent_rows),
            state->parent_rows_host,
            (size_t)input_count * sizeof(uint32_t), hipMemcpyHostToDevice,
            state->stream), "hipMemcpy frontier parent rows") != 0 ||
        hip_success(hipMemcpyAsync(hip_text_bytes(state,
            state->program->layout.target_depths), state->depths_host,
            (size_t)input_count * sizeof(uint32_t), hipMemcpyHostToDevice,
            state->stream), "hipMemcpy frontier depths") != 0 ||
        hip_success(hipMemcpyAsync(state->device_source_position,
            &state->source_position_host, sizeof(uint32_t),
            hipMemcpyHostToDevice, state->stream),
            "hipMemcpy frontier source position") != 0)
        return -1;
    cmd->frontier = 1u;
    return 0;
}

static int hip_text_launch_status(void) {
    return hipGetLastError() == hipSuccess ? 0 : -1;
}

static void hip_text_record_matrix(
        HipTextProgramCommand *cmd, uint32_t output_rows, uint32_t batch) {
    cmd->stats.area_output_row_tiles += output_rows;
    cmd->stats.area_candidate_output_tiles +=
        (uint64_t)output_rows * batch;
    if (batch > 1u) {
        cmd->stats.area_mn_dispatches++;
        cmd->stats.area_matrix_parallel_dispatches++;
    } else {
        cmd->stats.area_m1_dispatches++;
    }
}

static int hip_text_encode_cell(
        void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    const HipTextCellTemplate *cell_template;
    const SaltTextCanonicalLayout *layout;
    const SaltTextCompiledLayer *compiled = NULL;
    const SaltTextLayerExecDesc *layer = NULL;
    const SaltTextLayerPlan *layer_plan = NULL;
    const SaltAttentionDesc *attention = NULL;
    HipTextLayerRefs *refs = NULL;
    uint32_t count, jobs, hidden, dense, routed, experts, topk;
    uint32_t output_first = 0u;
    uint32_t physical_launches = 1u;
    float *destination;
    int *status;
    int terminal_output;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->submitted || state->program != program ||
        cmd->next_cell >= program->dispatch.cell_count ||
        cmd->next_cell >= state->template_count ||
        cell != &program->dispatch.cells[cmd->next_cell] || !assignment ||
        assignment->cpu.count != 0 || assignment->gpu.first != 0 ||
        input_count != cmd->input_count || assignment->gpu.count == 0)
        return -1;
    cell_template = &state->templates[cmd->next_cell];
    if (cell_template->cell != cell ||
        cell_template->index != cmd->next_cell ||
        cell_template->kind != (uint32_t)cell->kind ||
        cell_template->unit != (uint32_t)cell->unit ||
        cell_template->layer != cell->layer ||
        cell_template->logical_capacity != cell->logical_capacity ||
        cell_template->dependency_epoch != cell->dependency_epoch ||
        cell_template->completion_epoch != cell->completion_epoch ||
        cell_template->destination_offset != cell->destination_offset ||
        cell_template->destination_stride != cell->destination_stride ||
        cell_template->flags != cell->flags)
        return -1;
    terminal_output = cell->kind == SALT_TEXT_CELL_FINAL_NORM ||
        cell->kind == SALT_TEXT_CELL_FINAL_HEAD ||
        cell->kind == SALT_TEXT_CELL_LOGIT_SOFTCAP;
    count = cell->unit == SALT_TEXT_EXECUTION_ROWS
        ? input_count : input_count * cell_template->topk;
    if (terminal_output && cmd->authoritative_output_rows != input_count) {
        count = cmd->authoritative_output_rows;
        output_first = input_count - count;
    }
    if (assignment->gpu.count < count) return -1;
    cmd->stats.backend_template_reuses++;
    layout = &program->layout;
    hidden = cell_template->hidden;
    jobs = input_count * cell_template->topk;
    destination = hip_text_f32(state, cell_template->destination_offset);
    status = (int *)(void *)hip_text_bytes(state, state->status_offset);
    if (cell->layer < program->layer_count) {
        compiled = &program->layers[cell->layer];
        layer = compiled->descriptor;
        layer_plan = layer->plan;
        attention = &layer_plan->attention;
        refs = &state->layers[cell->layer];
    }
    dense = cell_template->dense;
    routed = cell_template->routed;
    experts = cell_template->experts;
    topk = cell_template->topk;
    if (layer_plan &&
        (dense != (uint32_t)layer_plan->dense_intermediate ||
         routed != (uint32_t)layer_plan->expert_intermediate ||
         experts != (uint32_t)layer_plan->n_experts ||
         topk != (uint32_t)layer_plan->top_k_experts))
        return -1;
    if (state->cell_timing_enabled &&
        (cmd->next_cell >= state->cell_event_capacity ||
         hip_success(hipEventRecord(state->cell_event_start[cmd->next_cell],
             state->stream), "hipEventRecord text cell start") != 0))
        return -1;
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING: {
        size_t elements = (size_t)input_count * hidden;
        if (state->embedding.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
            state->embedding.rows != program->vocabulary ||
            state->embedding.cols != hidden)
            return -1;
        hip_text_embedding_q4<<<
            (unsigned int)((elements + 255u) / 256u), 256, 0, state->stream>>>(
            state->embedding.resource, state->embedding.value_offset,
            state->embedding.scale_offset, state->embedding.bias_offset,
            hip_text_i32(state, layout->candidate_token_ids), input_count,
            program->vocabulary, hidden, program->descriptor->embedding_scale,
            destination, status);
        break;
    }
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        hip_text_rmsnorm_rows<<<input_count, 256, 0, state->stream>>>(
            hip_text_f32(state, layout->state_a), hidden,
            hip_text_tensor_f32(&refs->pre_attention_norm), destination,
            hidden, input_count, hidden, program->descriptor->norm_epsilon,
            1, status);
        break;
    case SALT_TEXT_CELL_QUERY_PROJECTION:
        if (hip_text_pair_enabled &&
            refs->q.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
            refs->k.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
            if (hip_text_direct_q4_descriptor(state, &refs->q, input_count,
                    layout->normalized, cell->destination_offset,
                    &state->host_selected_desc[0]) != 0)
                return -1;
            cmd->qk_wave_pending = 1u;
            physical_launches = 0u;
        } else if (hip_text_project(state, &refs->q,
                hip_text_f32(state, layout->normalized), destination,
                input_count) != 0) {
            return -1;
        }
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_KEY_PROJECTION:
        if (cmd->qk_wave_pending) {
            uint32_t max_rows = refs->q.rows > refs->k.rows
                ? refs->q.rows : refs->k.rows;
            uint32_t group = input_count == 1u &&
                !attention->shared_kv_projection &&
                refs->v.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 ? 3u : 2u;
            if (group == 3u) {
                if (cmd->next_cell + 1u >= program->dispatch.cell_count ||
                    program->dispatch.cells[cmd->next_cell + 1u].kind !=
                        SALT_TEXT_CELL_VALUE_PROJECTION ||
                    program->dispatch.cells[cmd->next_cell + 1u].layer !=
                        cell->layer ||
                    hip_text_direct_q4_descriptor(state, &refs->v, input_count,
                        layout->normalized,
                        program->dispatch.cells[cmd->next_cell + 1u].destination_offset,
                        &state->host_selected_desc[2]) != 0)
                    return -1;
                if (refs->v.rows > max_rows) max_rows = refs->v.rows;
            }
            if (hip_text_direct_q4_descriptor(state, &refs->k, input_count,
                    layout->normalized, cell->destination_offset,
                    &state->host_selected_desc[1]) != 0 ||
                hip_text_launch_q4_pair(state, 0u,
                    input_count, max_rows, group) != 0)
                return -1;
            cmd->qk_wave_pending = group == 3u ? 2u : 0u;
        } else if (hip_text_project(state, &refs->k,
                hip_text_f32(state, layout->normalized), destination,
                input_count) != 0) {
            return -1;
        }
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_VALUE_PROJECTION:
        if (cmd->qk_wave_pending == 2u) {
            cmd->qk_wave_pending = 0u;
            physical_launches = 0u;
        } else if (cmd->qk_wave_pending) {
            return -1;
        } else if (attention->shared_kv_projection) {
            physical_launches = 0u;
        } else if (
            hip_text_project(state, &refs->v,
                hip_text_f32(state, layout->normalized), destination,
                input_count) != 0) {
            return -1;
        }
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ATTENTION_TRANSFORM:
        if (attention->shared_kv_projection) {
            size_t elements;
            if (compiled->kv_width == 0u ||
                (size_t)input_count > SIZE_MAX / compiled->kv_width)
                return -1;
            elements = (size_t)input_count * compiled->kv_width;
            if (elements > (size_t)UINT_MAX * 256u) return -1;
            hip_text_copy<<<
                (unsigned int)((elements + 255u) / 256u), 256,
                0, state->stream>>>(
                hip_text_f32(state, layout->keys),
                hip_text_f32(state, layout->values), elements);
            physical_launches++;
        }
        hip_text_attention_transform<<<
            input_count * (attention->n_heads + 2u * attention->n_kv_heads),
            256, 0, state->stream>>>(
            attention->n_heads, attention->n_kv_heads, attention->head_dim,
            attention->rope_dim, attention->rope_base_dim,
            (float)attention->rope_theta, state->device_source_position,
            (const uint32_t *)hip_text_bytes(state, layout->target_depths),
            input_count,
            compiled->query_width, compiled->kv_width,
            program->descriptor->norm_epsilon,
            0,
            hip_text_f32(state, layout->queries),
            hip_text_f32(state, layout->keys),
            hip_text_f32(state, layout->values),
            hip_text_tensor_f32(&refs->q_norm),
            hip_text_tensor_f32(&refs->k_norm),
            hip_text_f32(state, compiled->tentative_key_offset),
            hip_text_f32(state, compiled->tentative_value_offset), status);
        break;
    case SALT_TEXT_CELL_ATTENTION_BODY:
        hip_text_attention_body<<<input_count * attention->n_heads, 256,
            (size_t)cmd->attention_score_rows * sizeof(float), state->stream>>>(
            attention->kind == SALT_ATTN_FULL,
            attention->n_heads, attention->n_kv_heads, attention->head_dim,
            attention->window, attention->score_scale,
            state->device_source_position,
            (const uint32_t *)hip_text_bytes(state,
                layout->target_parent_rows),
            (const uint32_t *)hip_text_bytes(state, layout->target_depths),
            input_count, compiled->query_width,
            compiled->kv_width, refs->kv.private_mode,
            refs->kv.row_capacity,
            hip_text_f32(state, layout->queries),
            refs->kv.device_keys, refs->kv.device_values,
            hip_text_f32(state, compiled->tentative_key_offset),
            hip_text_f32(state, compiled->tentative_value_offset),
            destination, status);
        break;
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
        if (hip_text_project(state, &refs->o,
                hip_text_f32(state, layout->attention_output), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
        hip_text_residual_postnorm_rows<<<input_count, 256, 0,
            state->stream>>>(hip_text_f32(state, layout->state_a),
            hip_text_f32(state, layout->branch),
            hip_text_tensor_f32(&refs->post_attention_norm), destination,
            input_count, hidden, program->descriptor->norm_epsilon, status);
        break;
    case SALT_TEXT_CELL_DENSE_NORM:
        if (cmd->ffn_norm_wave_layer != UINT32_MAX) return -1;
        hip_text_rmsnorm_pair_rows<<<dim3(input_count, 2u, 1u), 256,
            0, state->stream>>>(hip_text_f32(state, layout->state_b), hidden,
            hip_text_tensor_f32(&refs->pre_ffn_norm_1),
            hip_text_tensor_f32(&refs->pre_ffn_norm_2),
            hip_text_f32(state, layout->normalized),
            hip_text_f32(state, layout->combine_a), hidden,
            input_count, hidden, program->descriptor->norm_epsilon, status);
        cmd->ffn_norm_wave_layer = cell->layer;
        break;
    case SALT_TEXT_CELL_DENSE_GATE:
        if (hip_text_pair_enabled &&
            refs->dense_gate.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
            refs->dense_up.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
            if (hip_text_direct_q4_descriptor(state, &refs->dense_gate,
                    input_count, layout->normalized, cell->destination_offset,
                    &state->host_selected_desc[input_count == 1u ? 3u : 2u]) != 0)
                return -1;
            cmd->dense_wave_pending = 1u;
            physical_launches = 0u;
        } else if (hip_text_project(state, &refs->dense_gate,
                hip_text_f32(state, layout->normalized), destination,
                input_count) != 0) {
            return -1;
        }
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_DENSE_UP:
        if (cmd->dense_wave_pending) {
            uint32_t max_rows = refs->dense_gate.rows > refs->dense_up.rows
                ? refs->dense_gate.rows : refs->dense_up.rows;
            if (hip_text_direct_q4_descriptor(state, &refs->dense_up,
                    input_count, layout->normalized, cell->destination_offset,
                    &state->host_selected_desc[input_count == 1u ? 4u : 3u]) != 0 ||
                hip_text_launch_q4_pair(state, input_count == 1u ? 3u : 2u,
                    input_count, max_rows, 2u) != 0)
                return -1;
            cmd->dense_wave_pending = 0u;
        } else if (hip_text_project(state, &refs->dense_up,
                hip_text_f32(state, layout->normalized), destination,
                input_count) != 0) {
            return -1;
        }
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_DENSE_ACTIVATION: {
        if (cmd->dense_wave_pending) return -1;
        size_t elements = (size_t)input_count * dense;
        hip_text_activate<<<(unsigned int)((elements + 255u) / 256u), 256,
            0, state->stream>>>(hip_text_f32(state, layout->dense_gate),
            hip_text_f32(state, layout->dense_up), destination,
            elements, status);
        break;
    }
    case SALT_TEXT_CELL_DENSE_DOWN:
        if (hip_text_project(state, &refs->dense_down,
                hip_text_f32(state, layout->dense_chain), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ROUTER_INPUT:
        hip_text_router_input_rows<<<input_count, 256, 0, state->stream>>>(
            hip_text_f32(state, layout->state_b),
            hip_text_tensor_f32(&refs->router_scale), destination,
            input_count, hidden, program->descriptor->norm_epsilon,
            0.0f, status);
        break;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        if (hip_text_project(state, &refs->router,
                hip_text_f32(state, layout->router_input), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ROUTED_NORM:
        if (cmd->ffn_norm_wave_layer != cell->layer) return -1;
        cmd->ffn_norm_wave_layer = UINT32_MAX;
        physical_launches = 0u;
        break;
    case SALT_TEXT_CELL_ROUTER_TOPK:
        hip_text_topk_rows<<<input_count, 1, 0, state->stream>>>(
            hip_text_f32(state, layout->router_logits),
            hip_text_tensor_f32(&refs->per_expert_scale),
            hip_text_i32(state, layout->selected_experts),
            hip_text_f32(state, layout->selected_weights), input_count,
            experts, topk, status);
        hip_text_group_maps<<<1, 1, 0, state->stream>>>(
            hip_text_i32(state, layout->selected_experts), input_count,
            experts, topk,
            hip_text_i32(state, layout->grouped_to_canonical),
            hip_text_i32(state, layout->canonical_to_grouped),
            hip_text_i32(state, layout->tensor_row), status);
        hip_text_routed_gather<<<
            (unsigned int)(((size_t)jobs * hidden + 255u) / 256u), 256,
            0, state->stream>>>(
            hip_text_i32(state, layout->grouped_to_canonical),
            hip_text_f32(state, layout->combine_a),
            hip_text_f32(state, layout->routed_input), jobs, topk,
            hidden, status);
        physical_launches = 3u;
        break;
    case SALT_TEXT_CELL_EXPERT_GATE:
    case SALT_TEXT_CELL_EXPERT_UP:
    case SALT_TEXT_CELL_EXPERT_DOWN: {
        uint32_t phase = cell->kind == SALT_TEXT_CELL_EXPERT_GATE ? 0u :
            cell->kind == SALT_TEXT_CELL_EXPERT_UP ? 1u : 2u;
        if (jobs == 0u || jobs > state->selected_job_capacity)
            return -1;
        if (jobs > cmd->stats.backend_selected_jobs)
            cmd->stats.backend_selected_jobs = jobs;
        uint32_t rows = phase == 2u ? hidden : routed;
        uint32_t cols = phase == 2u ? routed : hidden;
        const float *input = phase == 2u
            ? hip_text_f32(state, layout->routed_chain)
            : hip_text_f32(state, layout->routed_input);
        if (cmd->selected_descriptor_count == 0 ||
            cmd->selected_descriptor_count != cmd->resource_request_count ||
            cmd->selected_max_occupancy == 0)
            return -1;
        if (input_count == 1u && phase < 2u) {
            if (cmd->next_cell < phase) return -1;
            uint32_t gate_cell = cmd->next_cell - phase;
            if (gate_cell + 1u >= program->dispatch.cell_count ||
                program->dispatch.cells[gate_cell].kind !=
                    SALT_TEXT_CELL_EXPERT_GATE ||
                program->dispatch.cells[gate_cell + 1u].kind !=
                    SALT_TEXT_CELL_EXPERT_UP ||
                program->dispatch.cells[gate_cell].layer != cell->layer ||
                program->dispatch.cells[gate_cell + 1u].layer != cell->layer)
                return -1;
            if (phase == 0u) {
                hip_selected_read_submitted();
                hip_text_expert_q4<<<dim3((rows + 3u) / 4u, jobs, 2u), 128,
                    0, state->stream>>>(state->device_experts, cell->layer, phase,
                    hip_text_i32(state, layout->selected_experts),
                    hip_text_i32(state, layout->grouped_to_canonical), jobs, topk,
                    state->experts_per_layer, input, destination,
                    hip_text_f32(state,
                        program->dispatch.cells[gate_cell + 1u].destination_offset),
                    rows, cols, status);
            } else physical_launches = 0u;
        } else if (!hip_text_selected_wave_enabled ||
            cmd->selected_max_occupancy == 1u) {
            hip_selected_read_submitted();
            hip_text_expert_q4<<<dim3((rows + 3u) / 4u, jobs, 1u), 128,
                0, state->stream>>>(state->device_experts, cell->layer, phase,
                hip_text_i32(state, layout->selected_experts),
                hip_text_i32(state, layout->grouped_to_canonical), jobs, topk,
                state->experts_per_layer,
                input, destination, NULL, rows, cols, status);
        } else {
            hip_selected_read_submitted();
            unsigned int threads, groups;
            if (hip_selected_warp_geometry((int)cmd->selected_max_occupancy,
                    &threads, &groups) != 0)
                return -1;
            hip_q4_selected_warp_batch<<<dim3(rows, groups,
                cmd->selected_descriptor_count), threads, 0, state->stream>>>(
                state->device_selected_desc +
                    (size_t)phase * cmd->selected_descriptor_count,
                (int)cmd->selected_descriptor_count,
                (const float *)state->device_canonical,
                (float *)state->device_canonical,
                hip_selected_device_resources,
                hip_selected_device_logical_slots,
                hip_selected_device_payload_offsets);
        }
        if (phase == 2u) cmd->stats.expert_down_dispatches++;
        else cmd->stats.expert_gate_up_dispatches++;
        break;
    }
    case SALT_TEXT_CELL_EXPERT_ACTIVATION: {
        size_t elements = (size_t)jobs * routed;
        hip_text_activate<<<(unsigned int)((elements + 255u) / 256u), 256,
            0, state->stream>>>(hip_text_f32(state, layout->routed_gate),
            hip_text_f32(state, layout->routed_up), destination,
            elements, status);
        break;
    }
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
        hip_text_expert_reduce<<<input_count, 256, 0, state->stream>>>(
            hip_text_f32(state, layout->selected_weights),
            hip_text_i32(state, layout->canonical_to_grouped),
            hip_text_f32(state, layout->routed_output_jobs), destination,
            input_count, topk, hidden, status);
        break;
    case SALT_TEXT_CELL_FFN_COMBINE:
        hip_text_parallel_combine_rows<<<input_count, 256, 0,
            state->stream>>>(hip_text_f32(state, layout->state_b),
            hip_text_f32(state, layout->dense_output),
            hip_text_f32(state, layout->routed_output),
            hip_text_tensor_f32(&refs->post_ffn_norm_1),
            hip_text_tensor_f32(&refs->post_ffn_norm_2),
            hip_text_tensor_f32(&refs->post_ffn_norm),
            hip_text_f32(state, layout->combine_a),
            hip_text_f32(state, layout->combine_b), destination,
            input_count, hidden, program->descriptor->norm_epsilon, 1.0f,
            layer_plan->final_layer_scale
                ? hip_text_tensor_f32(&refs->layer_scalar) : NULL,
            status);
        break;
    case SALT_TEXT_CELL_FINAL_NORM:
        if (count > 0)
            hip_text_rmsnorm_rows<<<count, 256, 0, state->stream>>>(
                hip_text_f32(state, layout->state_a) +
                    (size_t)output_first * hidden,
                hidden, hip_text_tensor_f32(&state->final_norm),
                destination + (size_t)output_first * hidden,
                hidden, count, hidden, program->descriptor->norm_epsilon,
                1, status);
        else
            physical_launches = 0u;
        break;
    case SALT_TEXT_CELL_FINAL_HEAD:
        if (count > 0) {
            if (hip_text_project(state, &state->output_head,
                    hip_text_f32(state, layout->final_state) +
                        (size_t)output_first * hidden,
                    destination +
                        (size_t)output_first * program->vocabulary,
                    count) != 0)
                return -1;
            cmd->stats.projection_dispatches++;
        } else {
            physical_launches = 0u;
        }
        break;
    case SALT_TEXT_CELL_LOGIT_SOFTCAP: {
        float cap = program->layers[program->layer_count - 1u].descriptor->
            plan->logit_softcap;
        size_t elements = (size_t)count * program->vocabulary;
        if (!(cap > 0.0f) || !isfinite(cap)) return -1;
        if (elements > 0)
            hip_text_softcap<<<(unsigned int)((elements + 255u) / 256u), 256,
                0, state->stream>>>(
                destination + (size_t)output_first * program->vocabulary,
                elements, cap, status);
        else
            physical_launches = 0u;
        break;
    }
    default:
        return -1;
    }
    if (hip_text_system_fence_enabled && physical_launches != 0u) {
        hip_text_system_fence<<<1, 1, 0, state->stream>>>();
        physical_launches++;
    }
    if (state->cell_timing_enabled &&
        hip_success(hipEventRecord(state->cell_event_end[cmd->next_cell],
            state->stream), "hipEventRecord text cell end") != 0)
        return -1;
    if (hip_text_launch_status() != 0) return -1;
    cmd->stats.backend_physical_kernel_nodes += physical_launches;
    cmd->stats.backend_host_kernel_launch_calls += physical_launches;
    switch (cell->kind) {
    case SALT_TEXT_CELL_QUERY_PROJECTION:
    case SALT_TEXT_CELL_KEY_PROJECTION:
    case SALT_TEXT_CELL_VALUE_PROJECTION:
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
    case SALT_TEXT_CELL_DENSE_GATE:
    case SALT_TEXT_CELL_DENSE_UP:
    case SALT_TEXT_CELL_DENSE_DOWN:
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
    case SALT_TEXT_CELL_EXPERT_GATE:
    case SALT_TEXT_CELL_EXPERT_UP:
    case SALT_TEXT_CELL_EXPERT_DOWN:
    case SALT_TEXT_CELL_FINAL_HEAD:
        if (cell->destination_stride == 0 ||
            cell->destination_stride % sizeof(float) != 0 ||
            cell->destination_stride / sizeof(float) > UINT32_MAX)
            return -1;
        hip_text_record_matrix(cmd,
            (uint32_t)(cell->destination_stride / sizeof(float)), count);
        break;
    default:
        break;
    }
    cmd->next_cell++;
    cmd->encoded_cells++;
    cmd->highest_completion = cell->completion_epoch;
    return 0;
}

static int hip_text_complete_phase(HipTextProgramState *state,
                                    const char *label) {
    const int *status;
    if (!state || !label || !state->device_canonical ||
        hip_success(hipMemcpyAsync(
            (unsigned char *)state->canonical.contents + state->status_offset,
            state->device_canonical + state->status_offset, sizeof(int),
            hipMemcpyDeviceToHost, state->stream),
            "hipMemcpy text status") != 0 ||
        hip_success(hipStreamSynchronize(state->stream), label) != 0)
        return -1;
    hip_selected_reads_completed();
    status = (const int *)((const unsigned char *)state->canonical.contents +
        state->status_offset);
    return *status == 0 ? 0 : -1;
}

static int hip_text_publish_resource_metadata(
        HipTextProgramState *state, const HipTextProgramCommand *cmd,
        const SaltTextLayerPlan *plan) {
    uint64_t jobs, bytes;
    const SaltTextCanonicalLayout *layout;
    if (!state || !cmd || !plan || !state->device_canonical ||
        plan->top_k_experts < 1 ||
        cmd->input_count > UINT32_MAX / (uint32_t)plan->top_k_experts)
        return -1;
    jobs = (uint64_t)cmd->input_count * (uint32_t)plan->top_k_experts;
    bytes = jobs * sizeof(int32_t);
    layout = &state->program->layout;
    if (bytes > SIZE_MAX || layout->selected_experts > state->canonical_bytes ||
        bytes > state->canonical_bytes - layout->selected_experts ||
        layout->grouped_to_canonical > state->canonical_bytes ||
        bytes > state->canonical_bytes - layout->grouped_to_canonical ||
        hip_success(hipMemcpyAsync(
            (unsigned char *)state->canonical.contents +
                layout->selected_experts,
            state->device_canonical + layout->selected_experts,
            (size_t)bytes, hipMemcpyDeviceToHost, state->stream),
            "hipMemcpy text selected experts") != 0 ||
        hip_success(hipMemcpyAsync(
            (unsigned char *)state->canonical.contents +
                layout->grouped_to_canonical,
            state->device_canonical + layout->grouped_to_canonical,
            (size_t)bytes, hipMemcpyDeviceToHost, state->stream),
            "hipMemcpy text grouped map") != 0)
        return -1;
    return 0;
}

static int hip_text_publish_touched(HipTextProgramState *state,
        const HipTextProgramCommand *cmd) {
    if (!state || !cmd || !state->device_canonical ||
        cmd->touched_span_count == 0)
        return -1;
    for (uint32_t index = 0; index < cmd->touched_span_count; index++) {
        const SaltTextTouchedSpan *span = &cmd->touched_spans[index];
        if (span->offset > state->canonical_bytes ||
            span->bytes > state->canonical_bytes - span->offset ||
            hip_success(hipMemcpyAsync(
                (unsigned char *)state->canonical.contents + span->offset,
                state->device_canonical + span->offset, span->bytes,
                hipMemcpyDeviceToHost, state->stream),
                "hipMemcpy text final span") != 0)
            return -1;
    }
    return 0;
}

static int hip_text_publish_logits(HipTextProgramState *state,
        const HipTextProgramCommand *cmd) {
    uint32_t rows;
    size_t offset, bytes;
    if (!state || !cmd || !state->device_canonical || !state->program ||
        cmd->input_count == 0 ||
        cmd->authoritative_output_rows > cmd->input_count ||
        state->program->vocabulary > SIZE_MAX / sizeof(float) /
            cmd->input_count)
        return -1;
    rows = cmd->authoritative
        ? cmd->authoritative_output_rows : cmd->input_count;
    offset = state->program->layout.position_logits +
        (size_t)(cmd->input_count - rows) * state->program->vocabulary *
            sizeof(float);
    bytes = (size_t)rows * state->program->vocabulary *
        sizeof(float);
    if (offset > state->canonical_bytes ||
        bytes > state->canonical_bytes - offset ||
        (bytes > 0 && hip_success(hipMemcpyAsync(
            (unsigned char *)state->canonical.contents + offset,
            state->device_canonical + offset, bytes,
            hipMemcpyDeviceToHost, state->stream),
            "hipMemcpy text final logits") != 0))
        return -1;
    return 0;
}

static int hip_text_dependency_barrier(
        void *backend_state, void *command,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    const SaltTextExecutionCell *upcoming;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->submitted || dependency_epoch == 0 ||
        completion_epoch <= dependency_epoch ||
        cmd->next_cell >= state->program->dispatch.cell_count ||
        dependency_epoch > cmd->highest_completion ||
        cmd->phase == HIP_TEXT_PHASE_IDLE ||
        cmd->phase == HIP_TEXT_PHASE_RESOURCE_PENDING ||
        cmd->phase == HIP_TEXT_PHASE_SUBMITTED ||
        cmd->phase == HIP_TEXT_PHASE_FINISHED)
        return -1;
    upcoming = &state->program->dispatch.cells[cmd->next_cell];
    if (upcoming->kind == SALT_TEXT_CELL_EXPERT_GATE) {
        if (cmd->phase == HIP_TEXT_PHASE_EXPERT)
            return SALT_TEXT_GPU_DEPENDENCY_READY;
        if (cmd->phase == HIP_TEXT_PHASE_PREFIX ||
            cmd->phase == HIP_TEXT_PHASE_SUFFIX) {
            const SaltTextLayerPlan *plan;
            const int32_t *selected, *grouped;
            uint32_t jobs, grouped_cursor = 0;
            uint64_t sync_start;
            if (upcoming->layer >= state->program->layer_count ||
                !(plan = state->program->layers[upcoming->layer].descriptor->plan) ||
                plan->n_experts < 1 ||
                (uint32_t)plan->n_experts > state->experts_per_layer ||
                plan->top_k_experts < 1 ||
                cmd->input_count > UINT32_MAX / (uint32_t)plan->top_k_experts ||
                hip_text_publish_resource_metadata(state, cmd, plan) != 0)
                return -1;
            sync_start = hip_host_now_ns();
            if (hip_text_complete_phase(state,
                    "hipStreamSynchronize text prefix") != 0)
                return -1;
            hip_u64_add_saturating(&cmd->prefix_sync_ns,
                hip_host_now_ns() - sync_start);
            jobs = cmd->input_count * (uint32_t)plan->top_k_experts;
            selected = (const int32_t *)((const unsigned char *)
                state->canonical.contents +
                state->program->layout.selected_experts);
            grouped = (const int32_t *)((const unsigned char *)
                state->canonical.contents +
                state->program->layout.grouped_to_canonical);
            memset(cmd->expert_occupancies, 0, sizeof cmd->expert_occupancies);
            for (uint32_t row = 0; row < cmd->input_count; row++)
                for (uint32_t rank = 0;
                     rank < (uint32_t)plan->top_k_experts; rank++) {
                    uint32_t canonical =
                        row * (uint32_t)plan->top_k_experts + rank;
                    int32_t expert = selected[canonical];
                    if (expert < 0 || expert >= plan->n_experts ||
                        (rank > 0 && expert <= selected[canonical - 1u]))
                        return -1;
                    cmd->expert_occupancies[(uint32_t)expert]++;
                }
            cmd->resource_request_count = 0;
            cmd->selected_descriptor_count = 0;
            cmd->selected_max_occupancy = 0;
            for (uint32_t expert = 0;
                 expert < (uint32_t)plan->n_experts; expert++) {
                uint32_t population = cmd->expert_occupancies[expert];
                if (population == 0) continue;
                if (cmd->resource_request_count >=
                    SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
                    return -1;
                cmd->resource_experts[cmd->resource_request_count++] =
                    (int32_t)expert;
                for (uint32_t item = 0; item < population; item++) {
                    uint32_t canonical = (uint32_t)grouped[grouped_cursor++];
                    if (canonical >= jobs ||
                        selected[canonical] != (int32_t)expert)
                        return -1;
                }
            }
            if (grouped_cursor != jobs || cmd->resource_request_count == 0)
                return -1;
            cmd->active_resource_layer = upcoming->layer;
            cmd->phase = HIP_TEXT_PHASE_RESOURCE_PENDING;
        }
        return SALT_TEXT_GPU_NEED_RESOURCE;
    }
    if (upcoming->kind == SALT_TEXT_CELL_EXPERT_REDUCTION &&
        cmd->phase == HIP_TEXT_PHASE_EXPERT) {
        uint64_t sync_start;
        if (cmd->active_resource_layer == UINT32_MAX)
            return -1;
        sync_start = hip_host_now_ns();
        if (hip_text_complete_phase(state,
                "hipStreamSynchronize text expert") != 0)
            return -1;
        hip_u64_add_saturating(&cmd->expert_sync_ns,
            hip_host_now_ns() - sync_start);
        cmd->phase = HIP_TEXT_PHASE_SUFFIX;
        cmd->active_resource_layer = UINT32_MAX;
        cmd->resource_request_count = 0;
        cmd->selected_descriptor_count = 0;
        cmd->selected_max_occupancy = 0;
    }
    cmd->barrier_count++;
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int hip_text_resource_request(
        void *backend_state, void *command, uint32_t *layer,
        int32_t *experts, uint32_t expert_capacity, uint32_t *expert_count) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->phase != HIP_TEXT_PHASE_RESOURCE_PENDING ||
        !layer || !experts ||
        !expert_count || cmd->resource_request_count == 0 ||
        cmd->resource_request_count > expert_capacity)
        return -1;
    *layer = cmd->active_resource_layer;
    *expert_count = cmd->resource_request_count;
    memcpy(experts, cmd->resource_experts,
        (size_t)cmd->resource_request_count * sizeof *experts);
    cmd->resource_wait_started_ns = hip_host_now_ns();
    return 0;
}

static int hip_text_canonical_float_offset(
        const HipTextProgramState *state, size_t byte_offset,
        uint64_t elements, uint32_t *float_offset) {
    uint64_t bytes;
    if (!state || !float_offset || (byte_offset % sizeof(float)) != 0 ||
        elements > UINT64_MAX / sizeof(float))
        return -1;
    bytes = elements * sizeof(float);
    if (byte_offset > state->canonical_bytes ||
        bytes > state->canonical_bytes - byte_offset ||
        byte_offset / sizeof(float) > UINT32_MAX)
        return -1;
    *float_offset = (uint32_t)(byte_offset / sizeof(float));
    return 0;
}

static int hip_text_selected_descriptor(
        const HipTextProgramState *state, const HipTextTensorRef *ref,
        int32_t slot, uint32_t batch, size_t input_byte_offset,
        size_t output_byte_offset, HipBatchDesc *descriptor) {
    const HipSelectedResource *selected;
    uint64_t logical, payload;
    uint64_t input_elements, output_elements;
    uint32_t input_offset, output_offset;
    if (!state || !ref || !descriptor || batch == 0 ||
        ref->encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        ref->rows == 0 || ref->cols == 0 || slot < 0 ||
        slot >= hip_selected_capacity || !hip_selected_resources ||
        !hip_selected_logical_slots || !hip_selected_payload_offsets ||
        !hip_selected_resource_slots)
        return -1;
    selected = &hip_selected_resources[slot];
    logical = selected->logical_resource_id;
    if (!selected->active ||
        (!selected->registered && !selected->resident) ||
        selected->device != ref->resource ||
        logical >= (uint64_t)(uint32_t)hip_selected_logical_capacity ||
        hip_selected_logical_slots[logical] != slot ||
        hip_selected_resource_slots[slot] != selected->device)
        return -1;
    payload = hip_selected_payload_offsets[logical];
    if (ref->value_offset < payload || ref->scale_offset < payload ||
        ref->bias_offset < payload ||
        batch > UINT64_MAX / ref->cols || batch > UINT64_MAX / ref->rows)
        return -1;
    input_elements = (uint64_t)batch * ref->cols;
    output_elements = (uint64_t)batch * ref->rows;
    if (hip_text_canonical_float_offset(state, input_byte_offset,
            input_elements, &input_offset) != 0 ||
        hip_text_canonical_float_offset(state, output_byte_offset,
            output_elements, &output_offset) != 0)
        return -1;
    memset(descriptor, 0, sizeof *descriptor);
    descriptor->value_offset = ref->value_offset - payload;
    descriptor->scale_offset = ref->scale_offset - payload;
    descriptor->bias_offset = ref->bias_offset - payload;
    descriptor->input_offset = input_offset;
    descriptor->output_offset = output_offset;
    descriptor->rows = (int32_t)ref->rows;
    descriptor->cols = (int32_t)ref->cols;
    descriptor->batch = (int32_t)batch;
    descriptor->resource_slot = slot;
    descriptor->selected = 1u;
    descriptor->logical_resource_id = logical;
    return 0;
}

static int hip_text_resource_resume(
        void *backend_state, void *command, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t expert_count) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->phase != HIP_TEXT_PHASE_RESOURCE_PENDING ||
        layer != cmd->active_resource_layer || !experts || !slots ||
        expert_count != cmd->resource_request_count ||
        layer >= state->program->layer_count)
        return -1;
    if (cmd->resource_wait_started_ns) {
        hip_u64_add_saturating(&cmd->resource_wait_ns,
            hip_host_now_ns() - cmd->resource_wait_started_ns);
        cmd->resource_wait_calls++;
        cmd->resource_wait_started_ns = 0;
    }
    const SaltTextLayerExecDesc *source =
        state->program->layers[layer].descriptor;
    const SaltTextCanonicalLayout *layout = &state->program->layout;
    uint32_t grouped_cursor = 0;
    uint32_t max_occupancy = 0;
    if (!state->host_selected_desc || !state->device_selected_desc ||
        expert_count > state->selected_job_capacity)
        return -1;
    for (uint32_t index = 0; index < expert_count; index++) {
        int32_t expert = experts[index];
        int32_t slot = slots[index];
        uint32_t occupancy;
        size_t base;
        if (expert != cmd->resource_experts[index] || expert < 0 ||
            expert >= (int32_t)source->expert_count || slot < 0 ||
            slot >= hip_selected_capacity)
            return -1;
        base = ((size_t)layer * state->experts_per_layer +
            (uint32_t)expert) * HIP_TEXT_EXPERT_PROJECTIONS;
        occupancy = cmd->expert_occupancies[(uint32_t)expert];
        if (occupancy == 0 ||
            grouped_cursor > UINT32_MAX - occupancy)
            return -1;
        if (hip_text_patch_selected_tensor(state->program,
                &source->experts[expert].gate,
                source->experts[expert].gate.storage.value_offset, slot,
                &state->expert_refs[base]) != 0 ||
            hip_text_patch_selected_tensor(state->program,
                &source->experts[expert].up,
                source->experts[expert].gate.storage.value_offset, slot,
                &state->expert_refs[base + 1u]) != 0 ||
            hip_text_patch_selected_tensor(state->program,
                &source->experts[expert].down,
                source->experts[expert].gate.storage.value_offset, slot,
                &state->expert_refs[base + 2u]) != 0 ||
            hip_success(hipMemcpyAsync(state->device_experts + base,
                state->expert_refs + base,
                3u * sizeof(HipTextTensorRef), hipMemcpyHostToDevice,
                state->stream), "hipMemcpy selected text expert refs") != 0)
            return -1;
        if (hip_text_selected_descriptor(state, &state->expert_refs[base], slot,
                occupancy,
                layout->routed_input + (size_t)grouped_cursor *
                    state->program->hidden * sizeof(float),
                layout->routed_gate + (size_t)grouped_cursor *
                    source->plan->expert_intermediate * sizeof(float),
                &state->host_selected_desc[index]) != 0 ||
            hip_text_selected_descriptor(state, &state->expert_refs[base + 1u],
                slot, occupancy,
                layout->routed_input + (size_t)grouped_cursor *
                    state->program->hidden * sizeof(float),
                layout->routed_up + (size_t)grouped_cursor *
                    source->plan->expert_intermediate * sizeof(float),
                &state->host_selected_desc[expert_count + index]) != 0 ||
            hip_text_selected_descriptor(state, &state->expert_refs[base + 2u],
                slot, occupancy,
                layout->routed_chain + (size_t)grouped_cursor *
                    source->plan->expert_intermediate * sizeof(float),
                layout->routed_output_jobs + (size_t)grouped_cursor *
                    state->program->hidden * sizeof(float),
                &state->host_selected_desc[2u * expert_count + index]) != 0)
            return -1;
        cmd->resource_slots[index] = slot;
        grouped_cursor += occupancy;
        if (occupancy > max_occupancy) max_occupancy = occupancy;
    }
    if (grouped_cursor != cmd->input_count * source->plan->top_k_experts ||
        max_occupancy == 0 ||
        hip_success(hipMemcpyAsync(state->device_selected_desc,
            state->host_selected_desc,
            (size_t)expert_count * HIP_TEXT_EXPERT_PROJECTIONS *
                sizeof(HipBatchDesc),
            hipMemcpyHostToDevice, state->stream),
            "hipMemcpy text selected descriptors") != 0)
        return -1;
    cmd->selected_descriptor_count = expert_count;
    cmd->selected_max_occupancy = max_occupancy;
    cmd->phase = HIP_TEXT_PHASE_EXPERT;
    return 0;
}

static int hip_text_encode_extent(void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        uint32_t first_cell, uint32_t cell_count, uint32_t input_count,
        uint32_t *encoded_cells) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd || !program ||
        !plan || !encoded_cells || plan->program != program ||
        plan->assignment_count != program->dispatch.cell_count ||
        first_cell != cmd->next_cell || input_count != cmd->input_count ||
        cell_count == 0 ||
        cell_count > program->dispatch.cell_count - first_cell ||
        first_cell >= state->template_count ||
        cell_count > state->template_count - first_cell)
        return -1;
    *encoded_cells = 0;
    for (uint32_t offset = 0; offset < cell_count; offset++) {
        uint32_t index = first_cell + offset;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        if (offset > 0 && cell->dependency_epoch > 0) {
            int barrier = hip_text_dependency_barrier(backend_state, command,
                cell->dependency_epoch, cell->completion_epoch);
            if (barrier == SALT_TEXT_GPU_NEED_RESOURCE) {
                *encoded_cells = offset;
                return barrier;
            }
            if (barrier != SALT_TEXT_GPU_DEPENDENCY_READY) return -1;
        }
        if (hip_text_encode_cell(backend_state, command, program, cell,
                &plan->assignments[index], input_count) != 0)
            return -1;
        *encoded_cells = offset + 1u;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int hip_text_submit(void *backend_state, void *command) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->submitted ||
        cmd->phase != HIP_TEXT_PHASE_SUFFIX ||
        cmd->qk_wave_pending || cmd->dense_wave_pending ||
        cmd->next_cell != state->program->dispatch.cell_count ||
        cmd->encoded_cells != state->program->dispatch.cell_count)
        return -1;
    cmd->phase = HIP_TEXT_PHASE_SUBMITTED;
    cmd->submitted = 1;
    return 0;
}

static enum HipPathId hip_text_projection_path(
        const HipTextTensorRef *ref, uint64_t *launches) {
    if (!ref || !launches) return HIP_PATH_NONE;
    *launches = UINT64_C(1);
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4)
        return HIP_PATH_TEXT_Q4;
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8)
        return HIP_PATH_TEXT_Q8;
    if (ref->encoding == SALT_TENSOR_ENCODING_BF16)
        return HIP_PATH_TEXT_BF16;
    return HIP_PATH_NONE;
}

static void hip_text_record_completed_paths(
        const HipTextProgramState *state,
        const HipTextProgramCommand *cmd) {
    uint64_t physical_consumers = UINT64_C(1);
    uint64_t system_fences = 0;
    if (!hip_path_enabled || !state || !cmd || !state->program) return;
    for (uint32_t index = 0; index < state->template_count; index++) {
        const HipTextCellTemplate *cell_template = &state->templates[index];
        const SaltTextExecutionCell *cell = cell_template->cell;
        const HipTextLayerRefs *refs = cell->layer < state->program->layer_count
            ? &state->layers[cell->layer] : NULL;
        const HipTextTensorRef *projection = NULL;
        enum HipPathId path = HIP_PATH_NONE;
        uint64_t jobs = cmd->input_count;
        uint64_t rows = jobs * cell_template->hidden;
        uint64_t launches = UINT64_C(1);
        if (cmd->authoritative &&
                (cell->kind == SALT_TEXT_CELL_FINAL_NORM ||
                 cell->kind == SALT_TEXT_CELL_FINAL_HEAD ||
                 cell->kind == SALT_TEXT_CELL_LOGIT_SOFTCAP)) {
            jobs = cmd->authoritative_output_rows;
            rows = jobs * cell_template->hidden;
            if (jobs == 0) launches = UINT64_C(0);
        }
        switch (cell->kind) {
        case SALT_TEXT_CELL_EMBEDDING:
            path = HIP_PATH_TEXT_EMBEDDING;
            break;
        case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        case SALT_TEXT_CELL_DENSE_NORM:
        case SALT_TEXT_CELL_FINAL_NORM:
            path = HIP_PATH_TEXT_NORM;
            break;
        case SALT_TEXT_CELL_QUERY_PROJECTION:
            if (hip_text_pair_enabled && refs &&
                refs->q.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                refs->k.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4)
                launches = UINT64_C(0);
            else
                projection = refs ? &refs->q : NULL;
            break;
        case SALT_TEXT_CELL_KEY_PROJECTION:
            if (hip_text_pair_enabled && refs &&
                refs->q.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                refs->k.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
                path = HIP_PATH_TEXT_Q4;
                uint64_t group = cmd->input_count == 1u &&
                    !state->program->layers[cell->layer].descriptor->plan->
                        attention.shared_kv_projection &&
                    refs->v.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 ? 3u : 2u;
                jobs *= group;
                rows = (uint64_t)cmd->input_count *
                    (refs->q.rows + refs->k.rows +
                     (group == 3u ? refs->v.rows : 0u));
            } else {
                projection = refs ? &refs->k : NULL;
            }
            break;
        case SALT_TEXT_CELL_VALUE_PROJECTION:
            if ((cmd->input_count == 1u && hip_text_pair_enabled && refs &&
                 refs->q.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                 refs->k.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                 refs->v.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) ||
                (cell->layer < state->program->layer_count &&
                state->program->layers[cell->layer].descriptor->plan->attention.
                    shared_kv_projection))
                launches = UINT64_C(0);
            else
                projection = refs ? &refs->v : NULL;
            break;
        case SALT_TEXT_CELL_ATTENTION_TRANSFORM: {
            const SaltAttentionDesc *attention =
                &state->program->layers[cell->layer].descriptor->plan->attention;
            path = HIP_PATH_TEXT_ATTN_TRANSFORM;
            rows = jobs * (attention->n_heads + 2u * attention->n_kv_heads);
            break;
        }
        case SALT_TEXT_CELL_ATTENTION_BODY: {
            const SaltAttentionDesc *attention =
                &state->program->layers[cell->layer].descriptor->plan->attention;
            path = HIP_PATH_TEXT_ATTN_BODY;
            rows = jobs * attention->n_heads;
            break;
        }
        case SALT_TEXT_CELL_OUTPUT_PROJECTION:
            projection = refs ? &refs->o : NULL;
            break;
        case SALT_TEXT_CELL_ATTENTION_COMBINE:
        case SALT_TEXT_CELL_FFN_COMBINE:
            path = HIP_PATH_TEXT_COMBINE;
            break;
        case SALT_TEXT_CELL_DENSE_GATE:
            if (hip_text_pair_enabled && refs &&
                refs->dense_gate.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                refs->dense_up.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4)
                launches = UINT64_C(0);
            else
                projection = refs ? &refs->dense_gate : NULL;
            break;
        case SALT_TEXT_CELL_DENSE_UP:
            if (hip_text_pair_enabled && refs &&
                refs->dense_gate.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
                refs->dense_up.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
                path = HIP_PATH_TEXT_Q4;
                jobs *= UINT64_C(2);
                rows = (uint64_t)cmd->input_count *
                    (refs->dense_gate.rows + refs->dense_up.rows);
            } else {
                projection = refs ? &refs->dense_up : NULL;
            }
            break;
        case SALT_TEXT_CELL_DENSE_ACTIVATION:
            path = HIP_PATH_TEXT_ACTIVATION;
            rows = jobs * cell_template->dense;
            break;
        case SALT_TEXT_CELL_DENSE_DOWN:
            projection = refs ? &refs->dense_down : NULL;
            break;
        case SALT_TEXT_CELL_ROUTER_INPUT:
            path = HIP_PATH_TEXT_ROUTER_INPUT;
            break;
        case SALT_TEXT_CELL_ROUTER_PROJECTION:
            projection = refs ? &refs->router : NULL;
            break;
        case SALT_TEXT_CELL_ROUTED_NORM:
            launches = UINT64_C(0);
            break;
        case SALT_TEXT_CELL_ROUTER_TOPK: {
            uint64_t routed_jobs = jobs * cell_template->topk;
            hip_path_record(HIP_PATH_TEXT_TOPK, jobs,
                jobs * cell_template->experts, UINT64_C(1), UINT64_C(0));
            hip_path_record(HIP_PATH_TEXT_GROUP_MAPS, routed_jobs, routed_jobs,
                UINT64_C(1), UINT64_C(0));
            hip_path_record(HIP_PATH_TEXT_ROUTED_GATHER, routed_jobs,
                routed_jobs * cell_template->hidden, UINT64_C(1), UINT64_C(0));
            path = HIP_PATH_NONE;
            break;
        }
        case SALT_TEXT_CELL_EXPERT_GATE:
        case SALT_TEXT_CELL_EXPERT_UP:
        case SALT_TEXT_CELL_EXPERT_DOWN:
            path = HIP_PATH_TEXT_EXPERT_Q4;
            jobs *= cell_template->topk;
            if (cmd->input_count == 1u) {
                if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) jobs *= 2u;
                if (cell->kind == SALT_TEXT_CELL_EXPERT_UP) launches = 0u;
            }
            rows = jobs * (cell->kind == SALT_TEXT_CELL_EXPERT_DOWN
                ? cell_template->hidden : cell_template->routed);
            break;
        case SALT_TEXT_CELL_EXPERT_ACTIVATION:
            path = HIP_PATH_TEXT_ACTIVATION;
            jobs *= cell_template->topk;
            rows = jobs * cell_template->routed;
            break;
        case SALT_TEXT_CELL_EXPERT_REDUCTION:
            path = HIP_PATH_TEXT_EXPERT_REDUCE;
            physical_consumers++;
            break;
        case SALT_TEXT_CELL_FINAL_HEAD:
            projection = &state->output_head;
            break;
        case SALT_TEXT_CELL_LOGIT_SOFTCAP:
            path = HIP_PATH_TEXT_SOFTCAP;
            rows = jobs * state->program->vocabulary;
            break;
        default:
            break;
        }
        if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE)
            physical_consumers++;
        if (projection && launches != 0) {
            path = hip_text_projection_path(projection, &launches);
            rows = jobs * projection->rows;
        }
        if (path != HIP_PATH_NONE && launches != 0)
            hip_path_record(path, jobs, rows, launches, UINT64_C(0));
        if (hip_text_system_fence_enabled && launches != 0) system_fences++;
    }
    if (system_fences != 0)
        hip_path_record(HIP_PATH_TEXT_SYSTEM_FENCE, system_fences,
            system_fences, system_fences, UINT64_C(0));
    hip_path_record(HIP_PATH_TEXT_SUBMIT, cmd->input_count, cmd->input_count,
        UINT64_C(0), UINT64_C(0));
    hip_path_record(HIP_PATH_TEXT_SYNC, UINT64_C(0), UINT64_C(0),
        UINT64_C(0), physical_consumers);
}

static int hip_text_publish_cell_timing(const HipTextProgramState *state) {
    uint64_t ns[SALT_TEXT_EXECUTION_CELL_KIND_COUNT] = {0};
    uint32_t calls[SALT_TEXT_EXECUTION_CELL_KIND_COUNT] = {0};
    uint64_t total = 0;
    if (!state || !state->cell_timing_enabled ||
        state->cell_event_capacity < state->template_count)
        return state && !state->cell_timing_enabled ? 0 : -1;
    for (uint32_t index = 0; index < state->template_count; index++) {
        float elapsed_ms = 0.0f;
        uint32_t kind = (uint32_t)state->templates[index].cell->kind;
        uint64_t elapsed_ns;
        if (kind >= SALT_TEXT_EXECUTION_CELL_KIND_COUNT ||
            hip_success(hipEventElapsedTime(&elapsed_ms,
                state->cell_event_start[index], state->cell_event_end[index]),
                "hipEventElapsedTime text cell") != 0 ||
            !(elapsed_ms >= 0.0f))
            return -1;
        elapsed_ns = (uint64_t)((double)elapsed_ms * 1000000.0 + 0.5);
        hip_u64_add_saturating(&ns[kind], elapsed_ns);
        hip_u64_add_saturating(&total, elapsed_ns);
        calls[kind]++;
    }
    for (uint32_t kind = 1; kind < SALT_TEXT_EXECUTION_CELL_KIND_COUNT;
         kind++)
        if (calls[kind])
            fprintf(stderr,
                "SALT_HIP_CELL_TIMING kind=%u calls=%u gpu_ns=%llu\n",
                kind, calls[kind], (unsigned long long)ns[kind]);
    fprintf(stderr, "SALT_HIP_CELL_TIMING_TOTAL cells=%u gpu_ns=%llu\n",
        state->template_count, (unsigned long long)total);
    return 0;
}

static int hip_text_finish(
        void *backend_state, void *command,
        SaltTextExecutionView *view, SaltTextVerifyBackendStats *stats) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    const int *status;
    uint64_t sync_start;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->submitted || cmd->finished ||
        cmd->phase != HIP_TEXT_PHASE_SUBMITTED || !view || !stats ||
        (cmd->frontier
            ? hip_text_publish_touched(state, cmd)
            : hip_text_publish_logits(state, cmd)) != 0)
        return -1;
    sync_start = hip_host_now_ns();
    if (hip_text_complete_phase(state,
            "hipStreamSynchronize text program") != 0)
        return -1;
    cmd->final_sync_ns = hip_host_now_ns() - sync_start;
    status = (const int *)((const unsigned char *)state->canonical.contents +
        state->status_offset);
    if (*status != 0 || cmd->resource_wait_started_ns ||
        hip_text_publish_cell_timing(state) != 0)
        return -1;
    if (state->cell_timing_enabled)
        fprintf(stderr,
            "SALT_HIP_RESOURCE_WAIT calls=%u host_ns=%llu\n",
            cmd->resource_wait_calls,
            (unsigned long long)cmd->resource_wait_ns);
    if (state->cell_timing_enabled)
        fprintf(stderr,
            "SALT_HIP_SYNC_WALL prefix_ns=%llu expert_ns=%llu final_ns=%llu\n",
            (unsigned long long)cmd->prefix_sync_ns,
            (unsigned long long)cmd->expert_sync_ns,
            (unsigned long long)cmd->final_sync_ns);
    hip_text_record_completed_paths(state, cmd);
    cmd->stats.final_logits_transfer_bytes =
        (uint64_t)(cmd->authoritative
            ? cmd->authoritative_output_rows : cmd->input_count) *
        state->program->vocabulary * sizeof(float);
    view->canonical_base = (unsigned char *)state->canonical.contents;
    view->canonical_bytes = state->canonical_bytes;
    view->backend_published_kv = 1u;
    view->reserved = 0u;
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_KV_PUBLICATION source=%u rows=%u frontier=%u "
            "authoritative=%u backend_published=%u host_bytes=%llu "
            "touched_bytes=%llu resource_wait_calls=%u "
            "resource_wait_ns=%llu prefix_sync_ns=%llu expert_sync_ns=%llu "
            "final_sync_ns=%llu command_submitted=%u "
            "final_completion_waits=1 "
            "physical_kernels=%llu host_launches=%llu\n",
            cmd->source_position, cmd->input_count, cmd->frontier,
            cmd->authoritative, view->backend_published_kv,
            (unsigned long long)(cmd->stats.final_logits_transfer_bytes +
                sizeof(int)),
            (unsigned long long)cmd->touched_span_bytes,
            cmd->resource_wait_calls,
            (unsigned long long)cmd->resource_wait_ns,
            (unsigned long long)cmd->prefix_sync_ns,
            (unsigned long long)cmd->expert_sync_ns,
            (unsigned long long)cmd->final_sync_ns,
            cmd->submitted,
            (unsigned long long)cmd->stats.backend_physical_kernel_nodes,
            (unsigned long long)cmd->stats.backend_host_kernel_launch_calls);
    *stats = cmd->stats;
    cmd->phase = HIP_TEXT_PHASE_FINISHED;
    cmd->finished = 1;
    return 0;
}

static int hip_text_resolve(
        void *backend_state, void *command,
        uint32_t committed_count, uint32_t input_count,
        uint64_t *scrubbed_bytes) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    uint32_t committed_position, maximum_width = 0u;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !cmd->submitted || !cmd->finished ||
        cmd->phase != HIP_TEXT_PHASE_FINISHED || !scrubbed_bytes ||
        input_count != cmd->input_count || committed_count > input_count)
        return -1;
    if (committed_count != 0u) {
        if (!state->device_kv || state->program->layer_count == 0u ||
            state->program->layer_count > 64u ||
            cmd->source_position > UINT32_MAX - committed_count)
            return -1;
        for (uint32_t layer = 0; layer < state->program->layer_count; layer++)
            if (state->layers[layer].kv.width > maximum_width)
                maximum_width = state->layers[layer].kv.width;
        if (maximum_width == 0u)
            return -1;
        hip_text_publish_committed_kv<<<
            dim3((maximum_width + 255u) / 256u, committed_count,
                 state->program->layer_count), 256, 0, state->stream>>>(
                state->device_kv, state->program->layer_count,
                (const float *)state->device_canonical,
                cmd->source_position, committed_count,
                (const int *)(const void *)(state->device_canonical +
                    state->status_offset));
        if (hip_text_launch_status() != 0)
            return -1;
        committed_position = cmd->source_position + committed_count;
        for (uint32_t layer = 0; layer < state->program->layer_count; layer++) {
            HipAttentionKvSync *sync = state->layers[layer].kv.sync;
            if (!sync || sync->device_positions != (int)cmd->source_position)
                return -1;
            sync->device_positions = (int)committed_position;
        }
    }
    *scrubbed_bytes = 0;
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static int hip_text_scrub(
        void *backend_state, void *command, uint64_t *scrubbed_bytes) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        !scrubbed_bytes ||
        hip_success(hipStreamSynchronize(state->stream),
            "hipStreamSynchronize text scrub") != 0)
        return -1;
    {
        uint64_t cleared = 0;
        if (cmd->touched_span_count == 0 ||
            salt_text_touched_span_clear(
                (unsigned char *)state->canonical.contents,
                state->canonical_bytes, cmd->touched_spans,
                cmd->touched_span_count, &cleared) != 0)
            return -1;
    }
    *scrubbed_bytes = state->program->tentative_kv_bytes;
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static int hip_text_destroy(void *backend_state, void *command) {
    HipTextProgramState *state = (HipTextProgramState *)backend_state;
    HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
    if (!state || state->magic != HIP_TEXT_STATE_MAGIC || !cmd ||
        cmd->begun || cmd->submitted)
        return -1;
    hip_text_state_release(state);
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static void hip_text_diag_failure(const char *stage) {
    if (getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "SALT_HIP_TEXT_FAIL stage=%s\n", stage);
}

static int hip_text_requirements_diag(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextGpuProgramRequirements *requirements,
        size_t requirements_size) {
    int rc = hip_text_requirements(program, policy, requirements,
        requirements_size);
    if (rc < 0) hip_text_diag_failure("requirements");
    return rc;
}

static int hip_text_prepare_diag(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    int rc = hip_text_prepare(program, plan, canonical_out, backend_state,
        backend_state_bytes, command, command_bytes);
    if (rc < 0) hip_text_diag_failure("prepare");
    return rc;
}

static int hip_text_begin_diag(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    int rc = hip_text_begin(backend_state, command, generation,
        source_position, input_token_ids, input_count);
    if (rc < 0) hip_text_diag_failure("begin");
    return rc;
}

static int hip_text_encode_cell_diag(
        void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count) {
    int rc = hip_text_encode_cell(backend_state, command, program, cell,
        assignment, input_count);
    if (rc == 0 && getenv("SALT_HIP_TEXT_CELL_DIAG")) {
        HipTextProgramState *state = (HipTextProgramState *)backend_state;
        const int *status;
        if (!state || !state->device_canonical ||
            hip_success(hipMemcpyAsync(
                (unsigned char *)state->canonical.contents +
                    state->status_offset,
                state->device_canonical + state->status_offset, sizeof(int),
                hipMemcpyDeviceToHost, state->stream),
                "hipMemcpy text cell diag status") != 0 ||
            hip_success(hipStreamSynchronize(state->stream),
                "hipStreamSynchronize text cell diag") != 0)
            rc = -1;
        else {
            status = (const int *)((const unsigned char *)
                state->canonical.contents + state->status_offset);
            if (*status != 0) {
                fprintf(stderr,
                    "SALT_HIP_TEXT_FAIL stage=cell_status kind=%d layer=%u\n",
                    cell ? (int)cell->kind : -1,
                    cell ? cell->layer : UINT32_MAX);
                rc = -1;
            }
        }
    }
    if (rc < 0 && getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "SALT_HIP_TEXT_FAIL stage=encode kind=%d layer=%u\n",
            cell ? (int)cell->kind : -1, cell ? cell->layer : UINT32_MAX);
    return rc;
}

static int hip_text_dependency_barrier_diag(
        void *backend_state, void *command,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    int rc = hip_text_dependency_barrier(backend_state, command,
        dependency_epoch, completion_epoch);
    if (rc < 0 && getenv("SALT_GPU_DIAG")) {
        HipTextProgramState *state = (HipTextProgramState *)backend_state;
        HipTextProgramCommand *cmd = (HipTextProgramCommand *)command;
        const SaltTextExecutionCell *cell = state && cmd && state->program &&
            cmd->next_cell < state->program->dispatch.cell_count
            ? &state->program->dispatch.cells[cmd->next_cell] : NULL;
        fprintf(stderr,
            "SALT_HIP_TEXT_FAIL stage=dependency_barrier phase=%u next=%u "
            "kind=%d layer=%u highest=%u dep=%u completion=%u\n",
            cmd ? cmd->phase : UINT32_MAX,
            cmd ? cmd->next_cell : UINT32_MAX,
            cell ? (int)cell->kind : -1,
            cell ? cell->layer : UINT32_MAX,
            cmd ? cmd->highest_completion : UINT32_MAX,
            dependency_epoch, completion_epoch);
    }
    return rc;
}

static int hip_text_encode_extent_diag(void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        uint32_t first_cell, uint32_t cell_count, uint32_t input_count,
        uint32_t *encoded_cells) {
    int rc = hip_text_encode_extent(backend_state, command, program, plan,
        first_cell, cell_count, input_count, encoded_cells);
    if (rc < 0 && getenv("SALT_GPU_DIAG"))
        fprintf(stderr,
            "SALT_HIP_TEXT_FAIL stage=encode_extent first=%u count=%u\n",
            first_cell, cell_count);
    return rc;
}

static int hip_text_resource_request_diag(
        void *backend_state, void *command, uint32_t *layer,
        int32_t *experts, uint32_t expert_capacity, uint32_t *expert_count) {
    int rc = hip_text_resource_request(backend_state, command, layer,
        experts, expert_capacity, expert_count);
    if (rc < 0) hip_text_diag_failure("resource_request");
    return rc;
}

static int hip_text_resource_resume_diag(
        void *backend_state, void *command, uint32_t layer,
        const int32_t *experts, const int32_t *slots,
        uint32_t expert_count) {
    int rc = hip_text_resource_resume(backend_state, command, layer,
        experts, slots, expert_count);
    if (rc < 0 && getenv("SALT_GPU_DIAG"))
        fprintf(stderr, "SALT_HIP_TEXT_FAIL stage=resource_resume layer=%u\n",
            layer);
    return rc;
}

static int hip_text_submit_diag(void *backend_state, void *command) {
    int rc = hip_text_submit(backend_state, command);
    if (rc < 0) hip_text_diag_failure("submit");
    return rc;
}

static int hip_text_finish_diag(
        void *backend_state, void *command,
        SaltTextExecutionView *view, SaltTextVerifyBackendStats *stats) {
    int rc = hip_text_finish(backend_state, command, view, stats);
    if (rc < 0) hip_text_diag_failure("finish");
    return rc;
}

static const SaltTextGpuProgramOps hip_text_program_ops = {
    hip_text_requirements_diag,
    hip_text_prepare_diag,
    hip_text_begin_diag,
    hip_text_encode_cell_diag,
    hip_text_encode_extent_diag,
    hip_text_dependency_barrier_diag,
    hip_text_resource_request_diag,
    hip_text_resource_resume_diag,
    hip_text_submit_diag,
    hip_text_finish_diag,
    hip_text_resolve,
    hip_text_scrub,
    hip_text_destroy,
    hip_text_begin_frontier,
    hip_text_begin_authoritative,
    hip_text_begin_authoritative_output,
};

extern "C" const struct SaltTextGpuProgramOps *
salt_gpu_tensor_program_ops(void) {
    return hip_ready ? &hip_text_program_ops : NULL;
}
extern "C" const SaltGpuDispatch *salt_dispatch_get(void) { return NULL; }
extern "C" int salt_gpu_hip_present(void) { return 0; }
extern "C" int salt_gpu_rocm_present(void) { return 1; }
#define HIP_COMPAT_SYMBOL_INNER(left, right) left##right
#define HIP_COMPAT_SYMBOL(left, right) HIP_COMPAT_SYMBOL_INNER(left, right)
extern "C" int HIP_COMPAT_SYMBOL(salt_gpu_cu, da_present)(void) { return 0; }
#undef HIP_COMPAT_SYMBOL
#undef HIP_COMPAT_SYMBOL_INNER
extern "C" int salt_gpu_pageable_mmap_active(void) {
    return hip_pageable_mmap;
}
