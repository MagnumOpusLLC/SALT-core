#include "salt/gpu.h"
#include "salt/gpu_residency.h"
#include "salt/gpu_resource.h"
#include "salt/text_verify.h"

#include <cuda_runtime.h>

#include <chrono>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_MAX_RESOURCES 16
#define CUDA_MAX_TENSORS 16384
#define CUDA_MAX_C 8192
#define CUDA_MAX_OUTPUTS (4096u * 2816u)
#define CUDA_MAX_BATCH_JOBS 4096
#define CUDA_FORMAT_NVFP4 40

typedef struct CudaResource {
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
} CudaResource;

typedef struct CudaTensor {
    int used, bits, rows, cols;
    const void *key;
    uint32_t kind, resource_id;
    uint64_t value_offset, scale_offset, bias_offset, aux_offset;
} CudaTensor;

typedef struct CudaSelectedResource {
    int active;
    uint64_t logical_resource_id;
    const unsigned char *base;
    const unsigned char *payload;
    const unsigned char *device_payload;
    size_t nbytes;
} CudaSelectedResource;

typedef struct CudaBatchDesc {
    const unsigned char *resource;
    uint64_t value_offset, scale_offset, bias_offset, aux_offset;
    uint32_t input_offset, output_offset;
    int rows, cols, batch;
} CudaBatchDesc;

/* The target-program realization is deliberately narrow and fail-closed.  It
 * admits only the authenticated Gemma-4 MLX4 text program whose complete
 * immutable graph is 689 cells (all 27 public kinds).  Keeping fixed bounds
 * here makes every descriptor and command byte a startup cost. */
#define CUDA_TEXT_LAYERS 30u
#define CUDA_TEXT_EXPERTS 128u
#define CUDA_TEXT_TOPK 8u
#define CUDA_TEXT_MAX_CANDIDATES 64u
#define CUDA_TEXT_MAX_ROWS 256u
#define CUDA_TEXT_ATTENTION_SCORE_TILE 8192u
#define CUDA_TEXT_ATTENTION_TASK_TILE 256u
#define CUDA_TEXT_CELLS 689u
#define CUDA_TEXT_HIDDEN 2816u
#define CUDA_TEXT_DENSE 2112u
#define CUDA_TEXT_ROUTED 704u
#define CUDA_TEXT_EXPERT_REFS \
    (CUDA_TEXT_LAYERS * CUDA_TEXT_EXPERTS * 3u)
#define CUDA_TEXT_GRAPH_COUNT CUDA_TEXT_MAX_CANDIDATES
#define CUDA_TEXT_GRAPH_MAX_NODES 1024u
#define CUDA_TEXT_MAX_EXTENTS (CUDA_TEXT_LAYERS * 2u + 1u)
#define CUDA_TEXT_STATE_MAGIC UINT64_C(0x4355545854505247)

typedef struct CudaAttentionKvSync CudaAttentionKvSync;

typedef struct CudaTextTensorRef {
    const unsigned char *resource;
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint32_t rows;
    uint32_t cols;
    uint32_t encoding;
    uint32_t reserved;
} CudaTextTensorRef;

typedef struct CudaTextKvRef {
    float *keys;
    float *values;
    CudaAttentionKvSync *sync;
    size_t tentative_key_float_offset;
    size_t tentative_value_float_offset;
    uint32_t width;
    uint32_t private_mode;
    uint32_t row_capacity;
    uint32_t reserved;
} CudaTextKvRef;

typedef struct CudaTextLayerRefs {
    CudaTextTensorRef q, k, v, o;
    CudaTextTensorRef dense_gate, dense_up, dense_down, router;
    CudaTextTensorRef pre_attention_norm, q_norm, k_norm;
    CudaTextTensorRef post_attention_norm;
    CudaTextTensorRef pre_ffn_norm_1, pre_ffn_norm_2;
    CudaTextTensorRef post_ffn_norm_1, post_ffn_norm_2;
    CudaTextTensorRef post_ffn_norm, router_scale, per_expert_scale;
    CudaTextTensorRef layer_scalar;
    CudaTextKvRef kv;
} CudaTextLayerRefs;

typedef struct CudaTextCellTemplate {
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
} CudaTextCellTemplate;

typedef struct CudaTextGraphTemplate {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    uint32_t kernel_nodes;
    cudaGraphNode_t attention_nodes[CUDA_TEXT_LAYERS];
    uint32_t attention_node_count;
    SaltTextVerifyBackendStats stats;
} CudaTextGraphTemplate;

typedef struct CudaTextExtentGraph {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cudaGraphNode_t attention_node;
    uint32_t first_cell;
    uint32_t cell_count;
    uint32_t kernel_nodes;
    uint32_t attention_nodes;
    SaltTextVerifyBackendStats stats;
} CudaTextExtentGraph;

typedef struct CudaTextProgramState {
    uint64_t magic;
    const SaltTextVerifyProgram *program;
    SaltGpuSharedBuffer canonical;
    size_t canonical_bytes;
    size_t status_offset;
    cudaStream_t stream;
    CudaTextTensorRef *device_experts;
    CudaTextKvRef *device_kv;
    CudaTextTensorRef embedding, final_norm, output_head;
    CudaTextLayerRefs layers[CUDA_TEXT_LAYERS];
    CudaTextTensorRef expert_refs[CUDA_TEXT_EXPERT_REFS];
    CudaTextCellTemplate templates[CUDA_TEXT_CELLS];
    CudaTextGraphTemplate graphs[CUDA_TEXT_GRAPH_COUNT];
    CudaTextExtentGraph extent_graphs[CUDA_TEXT_GRAPH_COUNT]
                                           [CUDA_TEXT_MAX_EXTENTS];
    int32_t *graph_tokens;
    uint32_t *graph_parent_rows;
    uint32_t *graph_depths;
    uint32_t *graph_source_positions;
    uint32_t *device_source_position;
    uint32_t template_count;
    uint32_t selected_job_capacity;
    size_t attention_dynamic_limit;
    float *attention_score_workspace;
    size_t attention_score_workspace_floats;
    uint32_t graphs_ready;
    uint32_t extent_graph_count;
    uint32_t extent_graphs_ready;
    uint32_t capture_mode;
    uint32_t dynamic_experts;
    int ready;
} CudaTextProgramState;

typedef struct CudaTextProgramCommand {
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
    uint32_t resource_layer;
    uint32_t resource_pending;
    uint32_t request_count;
    uint32_t expert_chain_mask;
    int32_t request_experts[CUDA_TEXT_EXPERTS];
    int32_t request_slots[CUDA_TEXT_EXPERTS];
    CudaSelectedResource request_bindings[CUDA_TEXT_EXPERTS];
    SaltTextTouchedSpan touched_spans[SALT_TEXT_MAX_TOUCHED_SPANS];
    uint32_t touched_span_count;
    uint64_t touched_span_bytes;
    uint32_t begun;
    uint32_t submitted;
    uint32_t finished;
    uint32_t authoritative;
    uint32_t authoritative_output_rows;
    SaltTextVerifyBackendStats stats;
} CudaTextProgramCommand;

static int cuda_ready, cuda_mapped_only, cuda_defer;
static int cuda_pageable_mmap;
static int cuda_nvfp4_telemetry_enabled = 1;
static int cuda_nvfp4_timing_enabled;
static cudaEvent_t cuda_nvfp4_event_start, cuda_nvfp4_event_qdq;
static cudaEvent_t cuda_nvfp4_event_projection, cuda_nvfp4_event_d2h;
static CudaResource cuda_resources[CUDA_MAX_RESOURCES];
static CudaTensor cuda_tensors[CUDA_MAX_TENSORS];
static int cuda_tensor_count;
static CudaSelectedResource *cuda_selected_resources;
static int *cuda_selected_logical_slots;
static int cuda_selected_capacity;
static int cuda_selected_logical_capacity;
static int cuda_selected_retirement_fenced;
/* One-shot proof that the text-program stream was synchronized immediately
 * before its synchronous SaltCache acquire.  The existing retirement hook
 * consumes this proof; it is not a second readiness state or address map. */
static int cuda_text_resource_fence_credit;
static float *cuda_x, *cuda_y;
static float *cuda_attention_kv;
static size_t cuda_attention_kv_nbytes;
static const void *cuda_attention_kv_host;
static int *cuda_attention_status;
struct CudaAttentionKvSync {
    size_t key_offset, value_offset;
    size_t row_capacity;
    int kv_stride;
    int device_positions;
    int host_positions;
};
static CudaAttentionKvSync cuda_attention_kv_sync[64];
static int cuda_attention_kv_sync_count;
typedef struct CudaAttentionStage {
    const void *query_host;
    size_t query_offset, key_offset, value_offset;
    int start, batch, query_stride, kv_stride;
    int deferred_publish;
    int valid;
} CudaAttentionStage;
static CudaAttentionStage cuda_attention_stage;
static float *cuda_nvfp4_xdq;
static uint8_t *cuda_nvfp4_packed, *cuda_nvfp4_scales;
static float *cuda_host_x, *cuda_host_y;
static CudaBatchDesc *cuda_host_desc, *cuda_device_desc;
static uint32_t cuda_job_output_offset[CUDA_MAX_BATCH_JOBS];
static int cuda_job_rows[CUDA_MAX_BATCH_JOBS];
static SaltGpuBatchStatsV2 cuda_stats;
static uint64_t cuda_q4_hetero_logical_input_bytes;
static uint64_t cuda_q4_hetero_transfer_input_bytes;
static uint64_t cuda_q4_hetero_output_bytes;
static uint64_t cuda_q4_hetero_descriptor_bytes;
static uint64_t cuda_q4_hetero_reused_inputs;
static uint64_t cuda_q4_hetero_kernel_launches;
static uint64_t cuda_q4_hetero_completion_fences;
static uint64_t cuda_peak_window_bytes;

typedef struct CudaNvfp4Telemetry {
    uint64_t packed_weight_bytes;
    uint64_t weight_scale_bytes;
    uint64_t activation_input_bytes;
    uint64_t activation_qdq_bytes;
    uint64_t projection_output_bytes;
    uint64_t h2d_ns;
    uint64_t qdq_ns;
    uint64_t projection_ns;
    uint64_t d2h_ns;
} CudaNvfp4Telemetry;

static int cuda_success(cudaError_t error, const char *where);

static uint64_t cuda_host_now_ns(void) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int cuda_event_elapsed_ns(cudaEvent_t start, cudaEvent_t end,
                                 uint64_t *result) {
    float elapsed_ms = 0.0f;
    if (!result || cuda_success(cudaEventElapsedTime(&elapsed_ms, start, end),
                                "cudaEventElapsedTime NVFP4") != 0 ||
        !(elapsed_ms >= 0.0f) || elapsed_ms > (float)UINT64_MAX / 1000000.0f)
        return -1;
    *result = (uint64_t)((double)elapsed_ms * 1000000.0 + 0.5);
    return 0;
}

static int cuda_nvfp4_d2h_timing(cudaEvent_t producer,
                                 CudaNvfp4Telemetry *telemetry) {
    if (!cuda_nvfp4_timing_enabled) return 0;
    if (!telemetry ||
        cuda_success(cudaEventRecord(cuda_nvfp4_event_d2h),
                     "cudaEventRecord NVFP4 D2H") != 0 ||
        cuda_success(cudaEventSynchronize(cuda_nvfp4_event_d2h),
                     "cudaEventSynchronize NVFP4 D2H") != 0)
        return -1;
    return cuda_event_elapsed_ns(producer, cuda_nvfp4_event_d2h,
                                 &telemetry->d2h_ns);
}

static int cuda_u64_mul(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || (a != 0 && b > UINT64_MAX / a)) return -1;
    *result = a * b;
    return 0;
}

static int cuda_u64_add(uint64_t a, uint64_t b, uint64_t *result) {
    if (!result || b > UINT64_MAX - a) return -1;
    *result = a + b;
    return 0;
}

static void cuda_u64_add_saturating(uint64_t *counter, uint64_t delta) {
    if (!counter) return;
    *counter = delta > UINT64_MAX - *counter ? UINT64_MAX : *counter + delta;
}

static int cuda_nvfp4_telemetry(int R, int C, int B,
                                CudaNvfp4Telemetry *telemetry) {
    uint64_t rows = (uint64_t)(unsigned int)R;
    uint64_t cols = (uint64_t)(unsigned int)C;
    uint64_t batch = (uint64_t)(unsigned int)B;
    uint64_t rc, bc, br, qdq_per_row, value;
    if (!telemetry) return -1;
    memset(telemetry, 0, sizeof *telemetry);
    if (R < 1 || C < 16 || (C & 15) != 0 || B < 1 ||
        cuda_u64_mul(rows, cols, &rc) != 0 ||
        cuda_u64_mul(batch, cols, &bc) != 0 ||
        cuda_u64_mul(batch, rows, &br) != 0 ||
        cuda_u64_mul(batch, rc / UINT64_C(2),
                     &telemetry->packed_weight_bytes) != 0 ||
        cuda_u64_mul(batch, rc / UINT64_C(16),
                     &telemetry->weight_scale_bytes) != 0 ||
        cuda_u64_mul(bc, (uint64_t)sizeof(float),
                     &telemetry->activation_input_bytes) != 0 ||
        cuda_u64_add(cols / UINT64_C(2), cols / UINT64_C(16),
                     &qdq_per_row) != 0 ||
        cuda_u64_mul(cols, (uint64_t)sizeof(float), &value) != 0 ||
        cuda_u64_add(qdq_per_row, value, &qdq_per_row) != 0 ||
        cuda_u64_mul(batch, qdq_per_row,
                     &telemetry->activation_qdq_bytes) != 0 ||
        cuda_u64_mul(br, (uint64_t)sizeof(float),
                     &telemetry->projection_output_bytes) != 0)
        return -1;
    return 0;
}

static int cuda_nvfp4_telemetry_add(CudaNvfp4Telemetry *total,
                                    const CudaNvfp4Telemetry *delta) {
    CudaNvfp4Telemetry next;
    if (!total || !delta) return -1;
    next = *total;
    if (cuda_u64_add(total->packed_weight_bytes, delta->packed_weight_bytes,
                     &next.packed_weight_bytes) != 0 ||
        cuda_u64_add(total->weight_scale_bytes, delta->weight_scale_bytes,
                     &next.weight_scale_bytes) != 0 ||
        cuda_u64_add(total->activation_input_bytes,
                     delta->activation_input_bytes,
                     &next.activation_input_bytes) != 0 ||
        cuda_u64_add(total->activation_qdq_bytes,
                     delta->activation_qdq_bytes,
                     &next.activation_qdq_bytes) != 0 ||
        cuda_u64_add(total->projection_output_bytes,
                     delta->projection_output_bytes,
                     &next.projection_output_bytes) != 0)
        return -1;
    *total = next;
    return 0;
}

static void cuda_nvfp4_telemetry_publish(
        const CudaNvfp4Telemetry *telemetry, uint64_t launches) {
    if (!telemetry || !cuda_nvfp4_telemetry_enabled) return;
    cuda_u64_add_saturating(&cuda_stats.nvfp4_logical_packed_weight_bytes,
                            telemetry->packed_weight_bytes);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_logical_weight_scale_bytes,
                            telemetry->weight_scale_bytes);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_activation_input_bytes,
                            telemetry->activation_input_bytes);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_activation_qdq_bytes,
                            telemetry->activation_qdq_bytes);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_projection_output_bytes,
                            telemetry->projection_output_bytes);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_kernel_launches, launches);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_completion_fences, UINT64_C(1));
    cuda_u64_add_saturating(&cuda_stats.nvfp4_h2d_ns, telemetry->h2d_ns);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_qdq_ns, telemetry->qdq_ns);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_projection_ns,
                            telemetry->projection_ns);
    cuda_u64_add_saturating(&cuda_stats.nvfp4_d2h_ns, telemetry->d2h_ns);
}

static int cuda_success(cudaError_t error, const char *where) {
    if (error == cudaSuccess) return 0;
    fprintf(stderr, "gpu-cuda: %s: %s\n", where, cudaGetErrorString(error));
    return -1;
}

__device__ static uint16_t cuda_load_u16(const unsigned char *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

__device__ static uint32_t cuda_load_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}


__device__ static float cuda_bf16(const unsigned char *p) {
    return __uint_as_float((uint32_t)cuda_load_u16(p) << 16);
}

__device__ static float cuda_f32(const unsigned char *p) {
    return __uint_as_float(cuda_load_u32(p));
}

__device__ static float cuda_nvfp4_e2m1(uint8_t code) {
    const float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                             2.0f, 3.0f, 4.0f, 6.0f};
    float value = values[code & 7u];
    if (value == 0.0f) return 0.0f;
    return (code & 8u) ? -value : value;
}

__device__ static float cuda_nvfp4_e4m3fn(uint8_t code) {
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

__device__ static uint8_t cuda_nvfp4_e4m3fn_encode(float value) {
    uint8_t sign = (__float_as_uint(value) >> 31) ? 0x80u : 0u;
    float magnitude = fabsf(value);
    uint8_t best = 0;
    float best_distance = __uint_as_float(0x7f800000u);
    if (magnitude > 448.0f) magnitude = 448.0f;
    for (unsigned int candidate = 0; candidate <= 0x7eu; candidate++) {
        float decoded = cuda_nvfp4_e4m3fn((uint8_t)candidate);
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

__device__ static uint8_t cuda_nvfp4_e2m1_encode(float value) {
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

__global__ static void cuda_nvfp4_qdq_exact(
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
    uint8_t scale_code = cuda_nvfp4_e4m3fn_encode(raw_scale);
    scales[block] = scale_code;
    float scale = cuda_nvfp4_e4m3fn(scale_code);
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
        uint8_t first_code = cuda_nvfp4_e2m1_encode(first);
        uint8_t second_code = cuda_nvfp4_e2m1_encode(second);
        packed[(base + lane) / 2] =
            (uint8_t)(first_code | (uint8_t)(second_code << 4));
        dequant[base + lane] =
            __fmul_rn(cuda_nvfp4_e2m1(first_code), dequant_scale);
        dequant[base + lane + 1] =
            __fmul_rn(cuda_nvfp4_e2m1(second_code), dequant_scale);
    }
}

__global__ static void cuda_q4_exact(
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
        uint32_t packed = cuda_load_u32(resource + voff + (k >> 3u) * 4u);
        uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
        float scale = cuda_bf16(resource + soff + g * 2u);
        float bias = cuda_bf16(resource + boff + g * 2u);
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

__global__ static void cuda_q4_warp_exact(
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
            uint32_t packed = cuda_load_u32(
                resource + voff + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = cuda_bf16(resource + soff + g * 2u);
            float bias = cuda_bf16(resource + boff + g * 2u);
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

__global__ static void cuda_q8_exact(
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
        float scale = cuda_bf16(resource + soff + g * 2u);
        float bias = cuda_bf16(resource + boff + g * 2u);
        float t = __fmul_rn((float)q, scale);
        t = __fadd_rn(t, bias);
        float u = __fmul_rn(t, x[c]);
        acc = __fadd_rn(acc, u);
    }
    y[r] = acc;
}

__global__ static void cuda_q8_warp_exact(
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
            float scale = cuda_bf16(resource + soff + g * 2u);
            float bias = cuda_bf16(resource + boff + g * 2u);
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

__global__ static void cuda_bf16_warp_exact(
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
            float weight = cuda_bf16(weights + (size_t)col * 2u);
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

/* Preserve the portable scalar's column order for every independent output
 * row. Vision obtains parallelism across (batch,row), never by reassociating
 * one row's reduction. */
__global__ static void cuda_bf16_batch_serial_exact(
    const unsigned char *resource, uint64_t value_offset,
    const float *input, float *output, int R, int C) {
    int row = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int batch = (int)blockIdx.y;
    if (row >= R) return;
    const unsigned char *weights = resource + value_offset +
        (uint64_t)(uint32_t)row * (uint32_t)C * 2u;
    const float *x = input + (size_t)batch * (size_t)C;
    float sum = 0.0f;
    for (int col = 0; col < C; col++) {
        float product = __fmul_rn(cuda_bf16(weights + (size_t)col * 2u),
                                  x[col]);
        sum = __fadd_rn(sum, product);
    }
    output[(size_t)batch * (size_t)R + (size_t)row] = sum;
}

__global__ static void cuda_nvfp4_warp_exact(
    const unsigned char *resource, uint64_t value_offset,
    uint64_t scale_offset, uint64_t weight_global_offset,
    const float *input_dequant, float *output, int R, int C, int B) {
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    float acc = 0.0f;
    if (row < R && batch < B) {
        float stored_weight_global = cuda_f32(resource + weight_global_offset);
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
            float scale = cuda_nvfp4_e4m3fn(
                resource[scale_offset + scale_index]);
            float scaled = __fmul_rn(scale, weight_global);
            float weight = __fmul_rn(cuda_nvfp4_e2m1(code), scaled);
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

__global__ static void cuda_nvfp4_heterogeneous_qdq(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *input, float *dequant) {
    int descriptor_index = (int)blockIdx.z;
    int group = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (descriptor_index >= descriptor_count) return;
    CudaBatchDesc descriptor = descriptors[descriptor_index];
    int group_count = descriptor.batch * descriptor.cols / 16;
    if (group >= group_count) return;
    int base = group * 16;
    const float *source = input + descriptor.input_offset + base;
    float *target = dequant + descriptor.input_offset + base;
    float input_global_scale =
        cuda_f32(descriptor.resource + descriptor.aux_offset);
    float maximum = 0.0f;
    for (int lane = 0; lane < 16; lane++) {
        float magnitude = fabsf(source[lane]);
        if (magnitude > maximum) maximum = magnitude;
    }
    float scaled_maximum = __fmul_rn(maximum, 1.0f / 6.0f);
    float raw_scale = __fmul_rn(input_global_scale, scaled_maximum);
    if (raw_scale > 448.0f) raw_scale = 448.0f;
    uint8_t scale_code = cuda_nvfp4_e4m3fn_encode(raw_scale);
    float scale = cuda_nvfp4_e4m3fn(scale_code);
    float output_scale = scale == 0.0f ? 0.0f :
                         __fdiv_rn(input_global_scale, scale);
    float dequant_scale = __fdiv_rn(scale, input_global_scale);
    for (int lane = 0; lane < 16; lane++) {
        float quantized = __fmul_rn(source[lane], output_scale);
        if (quantized > 6.0f) quantized = 6.0f;
        if (quantized < -6.0f) quantized = -6.0f;
        uint8_t code = cuda_nvfp4_e2m1_encode(quantized);
        target[lane] = __fmul_rn(cuda_nvfp4_e2m1(code), dequant_scale);
    }
}

__global__ static void cuda_nvfp4_heterogeneous_warp(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *input_dequant, float *output) {
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int row = (int)blockIdx.x * 4 + warp;
    int batch = (int)blockIdx.y;
    CudaBatchDesc descriptor = {0};
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    float acc = 0.0f;
    if (descriptor_index < descriptor_count && row < descriptor.rows &&
        batch < descriptor.batch) {
        float stored_weight_global =
            cuda_f32(descriptor.resource + descriptor.bias_offset);
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
            float scale = cuda_nvfp4_e4m3fn(descriptor.resource[
                descriptor.scale_offset + scale_index]);
            float scaled = __fmul_rn(scale, weight_global);
            float weight = __fmul_rn(cuda_nvfp4_e2m1(code), scaled);
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

__global__ static void cuda_q4_warp_batch(
    const unsigned char *resource, uint64_t voff, uint64_t soff,
    uint64_t boff, const float *x, float *y, int R, int C, int B) {
    __shared__ float dequant[32];
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x;
    float acc[16];
    if (r >= R) return;
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int cb = 0; cb < C; cb += 32) {
        if (warp == 0) {
            int c = cb + lane;
            if (c < C) {
                uint64_t k = (uint64_t)(uint32_t)r * (uint32_t)C +
                             (uint32_t)c;
                uint64_t g = k / 64u;
                uint32_t packed = cuda_load_u32(
                    resource + voff + (k >> 3u) * 4u);
                uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
                float scale = cuda_bf16(resource + soff + g * 2u);
                float bias = cuda_bf16(resource + boff + g * 2u);
                float value = __fmul_rn((float)q, scale);
                dequant[lane] = __fadd_rn(value, bias);
            } else {
                dequant[lane] = 0.0f;
            }
        }
        __syncthreads();
        int index = 0;
        for (int token = warp; token < B; token += 32, index++) {
            int c = cb + lane;
            if (c < C) {
                float product = __fmul_rn(
                    dequant[lane], x[(size_t)token * (size_t)C + (size_t)c]);
                acc[index] = __fadd_rn(acc[index], product);
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

__device__ static float cuda_salt_expf(float x) {
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

__device__ static float cuda_round_bf16(float x) {
    uint32_t bits = __float_as_uint(x);
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
        bits &= UINT32_C(0xffff0000);
    }
    return __uint_as_float(bits);
}

__device__ static float cuda_bf16_mul(float a, float b) {
    return cuda_round_bf16(__fmul_rn(cuda_round_bf16(a),
                                     cuda_round_bf16(b)));
}

__device__ static float cuda_bf16_add(float a, float b) {
    return cuda_round_bf16(__fadd_rn(cuda_round_bf16(a),
                                     cuda_round_bf16(b)));
}

__device__ static float cuda_salt_rsqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f) return 1.0f / 0.0f;
    if (x == 1.0f) return 1.0f;
    if (x < 0x1p-126f)
        return __fmul_rn(cuda_salt_rsqrtf(__fmul_rn(x, 0x1p24f)),
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

__global__ static void cuda_attention_transform_exact(
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
        inverse = cuda_salt_rsqrtf(mean);
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
            row[i] = cuda_round_bf16(row[i]);
        __syncthreads();
        int half = head_dim / 2;
        for (int pair = (int)threadIdx.x; pair < rope_dim / 2;
             pair += (int)blockDim.x) {
            float c = cuda_round_bf16(
                cosines[(size_t)token * (size_t)rope_pairs + (size_t)pair]);
            float s = cuda_round_bf16(
                sines[(size_t)token * (size_t)rope_pairs + (size_t)pair]);
            float a = row[pair];
            float b = row[half + pair];
            float ac = cuda_bf16_mul(a, c);
            float bs = cuda_bf16_mul(b, s);
            float bc = cuda_bf16_mul(b, c);
            float as = cuda_bf16_mul(a, s);
            row[pair] = cuda_bf16_add(ac, -bs);
            row[half + pair] = cuda_bf16_add(bc, as);
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

__global__ static void cuda_attention_exact(
    int full_attention, int n_heads, int n_kv_heads, int head_dim, int window,
    const float *queries, const float *keys, const float *values,
    int start_position, int batch, int query_stride, int kv_stride,
    int row_capacity, const float *new_keys, const float *new_values,
    int new_start_position, int new_count, float *outputs, int *failure,
    float *global_scores, size_t score_stride, uint32_t task_first) {
    extern __shared__ float shared_scores[];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ int score_count;
    float *scores = global_scores
        ? global_scores + (size_t)blockIdx.x * score_stride : shared_scores;
    int task = (int)(task_first + blockIdx.x);
    int token = task / n_heads;
    int head = task % n_heads;
    int groups = n_heads / n_kv_heads;
    int kv_head = head / groups;
    int position = start_position + token;
    int first = full_attention ? 0 : position - window + 1;
    if (first < 0) first = 0;
    int count = position - first + 1;
    const float *query = queries + (size_t)token * (size_t)query_stride +
        (size_t)head * (size_t)head_dim;
    /* Independent QK matrix cells are GPU lanes; each dot retains the exact
     * canonical increasing-dimension accumulation. */
    for (int i = (int)threadIdx.x; i < count; i += (int)blockDim.x) {
        int position_k = first + i;
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
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float peak = -1.0f / 0.0f;
        for (int i = 0; i < count; i++) {
            float score = scores[i];
            if (score > peak) peak = score;
        }
        maximum = peak;
    }
    __syncthreads();
    /* Exponentials are independent; denominator order stays increasing-i. */
    for (int i = (int)threadIdx.x; i < count; i += (int)blockDim.x)
        scores[i] = cuda_salt_expf(__fadd_rn(scores[i], -maximum));
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < count; i++)
            sum = __fadd_rn(sum, scores[i]);
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

/* Publish only after every causal query has consumed the retained history. */
__global__ static void cuda_attention_publish_exact(
        const float *keys, const float *values,
        float *key_cache, float *value_cache,
        int start_position, int batch, int kv_stride, int row_capacity,
        const int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = (size_t)batch * (size_t)kv_stride;
    if (index >= count || row_capacity < 1 || *status != 0) return;
    size_t row = index / (size_t)kv_stride;
    if (batch > row_capacity && row < (size_t)(batch - row_capacity)) return;
    size_t column = index - row * (size_t)kv_stride;
    size_t destination = (size_t)(start_position + (int)row) %
        (size_t)row_capacity;
    key_cache[destination * (size_t)kv_stride + column] = keys[index];
    value_cache[destination * (size_t)kv_stride + column] = values[index];
}

__device__ static float cuda_salt_tanhf(float x) {
    if (x != x) return x;
    if (x >= 10.0f) return 1.0f;
    if (x <= -10.0f) return -1.0f;
    float magnitude = x < 0.0f ? -x : x;
    float exponential = cuda_salt_expf(__fmul_rn(2.0f, magnitude));
    float result = __fdiv_rn(__fadd_rn(exponential, -1.0f),
                             __fadd_rn(exponential, 1.0f));
    return x < 0.0f ? -result : result;
}

__device__ static float cuda_gemma4_gelu_tanh(float x) {
    const float k = 0.7978845608028654f;
    float cubic = __fmul_rn(0.044715f, x);
    cubic = __fmul_rn(cubic, x);
    cubic = __fmul_rn(cubic, x);
    float inner = __fadd_rn(x, cubic);
    float tanh_value = cuda_salt_tanhf(__fmul_rn(k, inner));
    float half_x = __fmul_rn(0.5f, x);
    return __fmul_rn(half_x, __fadd_rn(1.0f, tanh_value));
}

__device__ static double cuda_text_invsqrt64(float x) {
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

__device__ static float cuda_text_sqrtf(float x) {
    if (x != x) return x;
    if (x < 0.0f) return 0.0f / 0.0f;
    if (x == 0.0f || x == 1.0f) return x;
    if (x < 0x1p-126f)
        return __fmul_rn(cuda_text_sqrtf(__fmul_rn(x, 0x1p24f)), 0x1p-12f);
    return (float)__dmul_rn((double)x, cuda_text_invsqrt64(x));
}

__device__ static float cuda_text_logf(float x) {
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

__device__ static float cuda_text_powf(float x, float y) {
    if (x == 1.0f || y == 0.0f) return 1.0f;
    if (x > 0.0f && y == -0.5f) return cuda_salt_rsqrtf(x);
    if (x > 0.0f) return cuda_salt_expf(__fmul_rn(y, cuda_text_logf(x)));
    return x;
}

__device__ static void cuda_text_sincos_reduce(
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

__device__ static float cuda_text_sin_poly(float x) {
    float x2 = __fmul_rn(x, x);
    float z = __fadd_rn(__fmul_rn(x2, -1.95152958910081473e-04f),
                          8.33302275008926213e-03f);
    z = __fadd_rn(__fmul_rn(x2, z), -1.66666666641626524e-01f);
    z = __fmul_rn(x2, z);
    return __fadd_rn(x, __fmul_rn(x, z));
}

__device__ static float cuda_text_cos_poly(float x) {
    float x2 = __fmul_rn(x, x);
    float z = __fadd_rn(__fmul_rn(x2, 2.44331571180994839e-05f),
                         -1.38873162549376522e-03f);
    z = __fadd_rn(__fmul_rn(x2, z), 4.16666679023301001e-02f);
    z = __fadd_rn(__fmul_rn(x2, z), -0.5f);
    return __fadd_rn(1.0f, __fmul_rn(x2, z));
}

__device__ static float cuda_text_sinf(float x) {
    if (x != x) return x;
    int quadrant;
    float reduced;
    cuda_text_sincos_reduce(x, &quadrant, &reduced);
    if (quadrant == 0) return cuda_text_sin_poly(reduced);
    if (quadrant == 1) return cuda_text_cos_poly(reduced);
    if (quadrant == 2) return -cuda_text_sin_poly(reduced);
    return -cuda_text_cos_poly(reduced);
}

__device__ static float cuda_text_cosf(float x) {
    if (x != x) return x;
    int quadrant;
    float reduced;
    cuda_text_sincos_reduce(x, &quadrant, &reduced);
    if (quadrant == 0) return cuda_text_cos_poly(reduced);
    if (quadrant == 1) return -cuda_text_sin_poly(reduced);
    if (quadrant == 2) return -cuda_text_cos_poly(reduced);
    return cuda_text_sin_poly(reduced);
}

__device__ static void cuda_text_fail(int *status) {
    atomicExch(status, 1);
}

__device__ static void cuda_text_fail_code(int *status, int code) {
    atomicCAS(status, 0, code);
}

__device__ static void cuda_text_norm_block(
        const float *input, const float *weight, float *output,
        uint32_t width, float epsilon, int with_scale,
        float *inverse, int *status) {
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < width; column++) {
            float value = input[column];
            if (!isfinite(value) || (with_scale && !isfinite(weight[column])))
                cuda_text_fail(status);
            sum = __fadd_rn(sum, __fmul_rn(value, value));
        }
        *inverse = cuda_salt_rsqrtf(
            __fadd_rn(__fdiv_rn(sum, (float)width), epsilon));
        if (!isfinite(*inverse)) cuda_text_fail(status);
    }
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(input[column], *inverse);
        if (with_scale) value = __fmul_rn(value, weight[column]);
        output[column] = value;
        if (!isfinite(value)) cuda_text_fail(status);
    }
    __syncthreads();
}

__global__ static void cuda_text_rmsnorm_rows(
        const float *input, size_t input_stride, const float *weight,
        float *output, size_t output_stride, uint32_t rows, uint32_t width,
        float epsilon, int with_scale, int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    if (row >= rows) return;
    cuda_text_norm_block(input + (size_t)row * input_stride, weight,
        output + (size_t)row * output_stride, width, epsilon,
        with_scale, &inverse, status);
}

__global__ static void cuda_text_rmsnorm_pair_rows(
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
    cuda_text_norm_block(input + (size_t)row * input_stride, weight,
        output + (size_t)row * output_stride, width, epsilon,
        1, &inverse, status);
}

__global__ static void cuda_text_embedding_q4(
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
        cuda_text_fail(status);
        return;
    }
    uint64_t packed_index = (uint64_t)(uint32_t)token * width + column;
    uint64_t group = packed_index / 64u;
    uint32_t packed = cuda_load_u32(
        resource + value_offset + (packed_index >> 3u) * 4u);
    uint32_t quantized =
        (packed >> ((packed_index & 7u) * 4u)) & 15u;
    float value = __fmul_rn((float)quantized,
        cuda_bf16(resource + scale_offset + group * 2u));
    value = __fadd_rn(value,
        cuda_bf16(resource + bias_offset + group * 2u));
    value = __fmul_rn(value, scale);
    output[index] = value;
    if (!isfinite(value)) cuda_text_fail(status);
}

__global__ static void cuda_text_copy(
        const float *input, float *output, size_t elements) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) output[index] = input[index];
}

__global__ static void cuda_text_attention_transform(
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
        cuda_text_fail(status);
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
    cuda_text_norm_block(row, weight, row, head_dim, epsilon,
                         kind != 2u, &inverse, status);
    if (kind != 2u) {
        for (uint32_t column = threadIdx.x; column < head_dim;
             column += blockDim.x)
            row[column] = cuda_round_bf16(row[column]);
        __syncthreads();
        uint32_t half = head_dim / 2u;
        uint32_t position = source_position + depths[row_index];
        for (uint32_t pair = threadIdx.x; pair < rope_dim / 2u;
             pair += blockDim.x) {
            float exponent = __fdiv_rn((float)(2u * pair),
                                       (float)rope_base_dim);
            float angle = __fdiv_rn((float)position,
                cuda_text_powf(rope_theta, exponent));
            float cosine = cuda_round_bf16(cuda_text_cosf(angle));
            float sine = cuda_round_bf16(cuda_text_sinf(angle));
            float left = row[pair];
            float right = row[half + pair];
            float lc = cuda_bf16_mul(left, cosine);
            float rs = cuda_bf16_mul(right, sine);
            float rc = cuda_bf16_mul(right, cosine);
            float ls = cuda_bf16_mul(left, sine);
            row[pair] = cuda_bf16_add(lc, -rs);
            row[half + pair] = cuda_bf16_add(rc, ls);
            if (!isfinite(row[pair]) || !isfinite(row[half + pair]))
                cuda_text_fail(status);
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

__device__ static const float *cuda_text_frontier_row(
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

__device__ static const float *cuda_text_committed_row(
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

__global__ static void cuda_text_publish_authoritative_kv(
        const CudaTextKvRef *layers, uint32_t layer_count,
        const float *canonical, uint32_t source_position,
        uint32_t row_count, const int *status) {
    uint32_t layer = (uint32_t)blockIdx.z;
    uint32_t row = (uint32_t)blockIdx.y;
    if (!layers || !canonical || !status || *status != 0 ||
        layer >= layer_count || row >= row_count)
        return;
    const CudaTextKvRef *kv = &layers[layer];
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
    float *destination_keys = kv->keys +
        (size_t)destination_row * kv->width;
    float *destination_values = kv->values +
        (size_t)destination_row * kv->width;
    for (uint32_t column = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
         column < kv->width; column += gridDim.x * blockDim.x) {
        destination_keys[column] = source_keys[column];
        destination_values[column] = source_values[column];
    }
}

__global__ static void cuda_text_attention_body(
        int full_attention, uint32_t n_heads, uint32_t n_kv_heads,
        uint32_t head_dim, uint32_t window, float score_scale,
        const uint32_t *source_position_pointer, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity,
        const float *queries, const float *committed_keys,
        const float *committed_values, const float *tentative_keys,
        const float *tentative_values, float *output, int *status) {
    extern __shared__ float scores[];
    if (!source_position_pointer || !parents || !depths) {
        cuda_text_fail(status);
        return;
    }
    uint32_t source_position = *source_position_pointer;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ int score_count;
    __shared__ int scores_valid;
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
    if (threadIdx.x == 0) scores_valid = 1;
    __syncthreads();
    for (uint32_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        uint32_t key_position = first + index;
        const float *key = key_position < source_position
                ? cuda_text_committed_row(
                    committed_keys, key_position, source_position,
                    kv_width, kv_mode, kv_capacity)
                : cuda_text_frontier_row(tentative_keys, parents, depths,
                    rows, row, key_position - source_position, kv_width);
        if (!key) {
            scores[index] = 0.0f;
            atomicExch(&scores_valid, 0);
            cuda_text_fail(status);
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
        if (!scores_valid) {
            score_count = -1;
        } else {
            float peak = -1.0f / 0.0f;
            for (uint32_t index = 0; index < count; index++) {
                float score = scores[index];
                if (score > peak) peak = score;
            }
            maximum = peak;
            score_count = (int)count;
        }
        if (score_count < 0) cuda_text_fail(status);
    }
    __syncthreads();
    if (score_count < 0) return;
    for (uint32_t index = threadIdx.x; index < count;
         index += blockDim.x)
        scores[index] = cuda_salt_expf(
            __fadd_rn(scores[index], -maximum));
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t index = 0; index < count; index++)
            sum = __fadd_rn(sum, scores[index]);
        denominator = sum;
        score_count = (!(sum > 0.0f) || !isfinite(sum))
            ? -1 : (int)count;
        if (score_count < 0) cuda_text_fail(status);
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
            const float *value = value_position < source_position
                ? cuda_text_committed_row(
                    committed_values, value_position, source_position,
                    kv_width, kv_mode, kv_capacity)
                : cuda_text_frontier_row(tentative_values, parents, depths,
                    rows, row, value_position - source_position, kv_width);
            if (!value) {
                cuda_text_fail(status);
                return;
            }
            value += (size_t)kv_head * head_dim;
            float probability = __fdiv_rn(scores[index], denominator);
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(probability, value[column]));
        }
        head_output[column] = accumulator;
        if (!isfinite(accumulator)) cuda_text_fail(status);
    }
}

/* Preserve the exact attention operation order beyond the device's dynamic
 * shared-memory limit. Scores are recomputed in fixed tiles: thread zero still
 * walks maximum and denominator in ascending position order, and each output
 * component still divides, multiplies, and accumulates in that same order. */
__global__ static void cuda_text_attention_body_tiled_exact(
        int full_attention, uint32_t n_heads, uint32_t n_kv_heads,
        uint32_t head_dim, uint32_t window, float score_scale,
        const uint32_t *source_position_pointer, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity,
        const float *queries, const float *committed_keys,
        const float *committed_values, const float *tentative_keys,
        const float *tentative_values, float *output, int *status) {
    extern __shared__ float scores[];
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ int scores_valid;
    uint32_t task = (uint32_t)blockIdx.x;
    uint32_t row = task / n_heads;
    uint32_t head = task - row * n_heads;
    uint32_t groups = n_heads / n_kv_heads;
    uint32_t kv_head = head / groups;
    uint32_t source_position;
    uint32_t position;
    uint32_t first = 0;
    uint32_t count;
    const float *query;

    if (!source_position_pointer)
        return cuda_text_fail_code(status, 211);
    if (!parents)
        return cuda_text_fail_code(status, 212);
    if (!depths)
        return cuda_text_fail_code(status, 213);

    source_position = *source_position_pointer;
    if (row >= rows) return;
    position = source_position + depths[row];
    if (!full_attention && position + 1u > window)
        first = position + 1u - window;
    count = position - first + 1u;
    query = queries + (size_t)row * query_width +
        (size_t)head * head_dim;
    if (threadIdx.x == 0) {
        maximum = -1.0f / 0.0f;
        denominator = 0.0f;
        scores_valid = 1;
    }
    __syncthreads();

    for (uint32_t base = 0; base < count;
         base += CUDA_TEXT_ATTENTION_SCORE_TILE) {
        uint32_t tile_count = count - base;
        if (tile_count > CUDA_TEXT_ATTENTION_SCORE_TILE)
            tile_count = CUDA_TEXT_ATTENTION_SCORE_TILE;
        for (uint32_t local = threadIdx.x; local < tile_count;
             local += blockDim.x) {
            uint32_t key_position = first + base + local;
            const float *key = key_position < source_position
                ? cuda_text_committed_row(
                    committed_keys, key_position, source_position,
                    kv_width, kv_mode, kv_capacity)
                : cuda_text_frontier_row(tentative_keys, parents, depths,
                    rows, row, key_position - source_position, kv_width);
            if (!key) {
                scores[local] = 0.0f;
                atomicExch(&scores_valid, 0);
                cuda_text_fail(status);
            } else {
                float score = 0.0f;
                key += (size_t)kv_head * head_dim;
                for (uint32_t column = 0; column < head_dim; column++)
                    score = __fadd_rn(score,
                        __fmul_rn(query[column], key[column]));
                scores[local] = __fmul_rn(score, score_scale);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && scores_valid)
            for (uint32_t local = 0; local < tile_count; local++)
                if (scores[local] > maximum) maximum = scores[local];
        __syncthreads();
    }
    if (!scores_valid) return;

    for (uint32_t base = 0; base < count;
         base += CUDA_TEXT_ATTENTION_SCORE_TILE) {
        uint32_t tile_count = count - base;
        if (tile_count > CUDA_TEXT_ATTENTION_SCORE_TILE)
            tile_count = CUDA_TEXT_ATTENTION_SCORE_TILE;
        for (uint32_t local = threadIdx.x; local < tile_count;
             local += blockDim.x) {
            uint32_t key_position = first + base + local;
            const float *key = key_position < source_position
                ? cuda_text_committed_row(
                    committed_keys, key_position, source_position,
                    kv_width, kv_mode, kv_capacity)
                : cuda_text_frontier_row(tentative_keys, parents, depths,
                    rows, row, key_position - source_position, kv_width);
            if (!key) {
                scores[local] = 0.0f;
                atomicExch(&scores_valid, 0);
                cuda_text_fail(status);
            } else {
                float score = 0.0f;
                key += (size_t)kv_head * head_dim;
                for (uint32_t column = 0; column < head_dim; column++)
                    score = __fadd_rn(score,
                        __fmul_rn(query[column], key[column]));
                scores[local] = __fmul_rn(score, score_scale);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && scores_valid)
            for (uint32_t local = 0; local < tile_count; local++) {
                scores[local] = cuda_salt_expf(
                    __fadd_rn(scores[local], -maximum));
                denominator = __fadd_rn(denominator, scores[local]);
            }
        __syncthreads();
    }
    if (!scores_valid || !(denominator > 0.0f) ||
        !isfinite(denominator)) {
        if (threadIdx.x == 0) cuda_text_fail_code(status, 203);
        return;
    }

    float *head_output = output + (size_t)row * query_width +
        (size_t)head * head_dim;
    uint32_t column_rounds = (head_dim + blockDim.x - 1u) / blockDim.x;
    for (uint32_t column_round = 0; column_round < column_rounds;
         column_round++) {
        uint32_t output_column = column_round * blockDim.x + threadIdx.x;
        int active_column = output_column < head_dim;
        float accumulator = 0.0f;
        for (uint32_t base = 0; base < count;
             base += CUDA_TEXT_ATTENTION_SCORE_TILE) {
            uint32_t tile_count = count - base;
            if (tile_count > CUDA_TEXT_ATTENTION_SCORE_TILE)
                tile_count = CUDA_TEXT_ATTENTION_SCORE_TILE;
            for (uint32_t local = threadIdx.x; local < tile_count;
                 local += blockDim.x) {
                uint32_t key_position = first + base + local;
                const float *key = key_position < source_position
                    ? cuda_text_committed_row(
                        committed_keys, key_position, source_position,
                        kv_width, kv_mode, kv_capacity)
                    : cuda_text_frontier_row(tentative_keys, parents, depths,
                        rows, row, key_position - source_position, kv_width);
                if (!key) {
                    scores[local] = 0.0f;
                    atomicExch(&scores_valid, 0);
                    cuda_text_fail(status);
                } else {
                    float score = 0.0f;
                    key += (size_t)kv_head * head_dim;
                    for (uint32_t column = 0; column < head_dim; column++)
                        score = __fadd_rn(score,
                            __fmul_rn(query[column], key[column]));
                    scores[local] = __fmul_rn(score, score_scale);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && scores_valid)
                for (uint32_t local = 0; local < tile_count; local++)
                    scores[local] = cuda_salt_expf(
                        __fadd_rn(scores[local], -maximum));
            __syncthreads();
            if (active_column && scores_valid) {
                uint32_t value_position = first + base;
                for (uint32_t local = 0; local < tile_count;
                     local++, value_position++) {
                    const float *value = value_position < source_position
                        ? cuda_text_committed_row(
                            committed_values, value_position, source_position,
                            kv_width, kv_mode, kv_capacity)
                        : cuda_text_frontier_row(tentative_values, parents,
                            depths, rows, row,
                            value_position - source_position, kv_width);
                    if (!value) {
                        atomicExch(&scores_valid, 0);
                        cuda_text_fail_code(status, 204);
                        break;
                    }
                    value += (size_t)kv_head * head_dim;
                    float probability = __fdiv_rn(scores[local], denominator);
                    accumulator = __fadd_rn(accumulator,
                        __fmul_rn(probability, value[output_column]));
                }
            }
            __syncthreads();
        }
        if (!scores_valid) return;
        if (active_column) {
            head_output[output_column] = accumulator;
            if (!isfinite(accumulator)) cuda_text_fail_code(status, 205);
        }
        __syncthreads();
    }
}

__global__ static void cuda_text_attention_scores_global_exact(
        int full_attention, uint32_t n_heads, uint32_t n_kv_heads,
        uint32_t head_dim, uint32_t window, float score_scale,
        const uint32_t *source_position_pointer, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity,
        const float *queries, const float *committed_keys,
        const float *tentative_keys, uint32_t task_first,
        uint32_t task_count, uint32_t score_stride,
        float *scores, int *status) {
    uint32_t local_task = (uint32_t)blockIdx.x;
    uint32_t task, source_position, row, head, groups, kv_head;
    uint32_t position, first = 0, count;
    const float *query;
    if (!status) return;
    if (!source_position_pointer || !parents || !depths || !queries ||
        !tentative_keys || !scores) {
        cuda_text_fail_code(status, 231);
        return;
    }
    if (n_heads == 0u || n_kv_heads == 0u ||
        n_heads % n_kv_heads != 0u || head_dim == 0u || rows == 0u ||
        query_width == 0u || kv_width == 0u || score_stride == 0u) {
        cuda_text_fail_code(status, 232);
        return;
    }
    if (local_task >= task_count) return;
    task = task_first + local_task;
    row = task / n_heads;
    head = task - row * n_heads;
    groups = n_heads / n_kv_heads;
    kv_head = head / groups;
    source_position = *source_position_pointer;
    if (row >= rows) return;
    position = source_position + depths[row];
    if (!full_attention && position + 1u > window)
        first = position + 1u - window;
    count = position - first + 1u;
    if (count > score_stride) {
        cuda_text_fail_code(status, 233);
        return;
    }
    query = queries + (size_t)row * query_width +
        (size_t)head * head_dim;
    scores += (size_t)local_task * score_stride;
    for (uint32_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        uint32_t key_position = first + index;
        const float *key = key_position < source_position
            ? cuda_text_committed_row(
                committed_keys, key_position, source_position,
                kv_width, kv_mode, kv_capacity)
            : cuda_text_frontier_row(tentative_keys, parents, depths,
                rows, row, key_position - source_position, kv_width);
        if (!key) {
            scores[index] = 0.0f;
            cuda_text_fail_code(status, 234);
            continue;
        }
        key += (size_t)kv_head * head_dim;
        float score = 0.0f;
        for (uint32_t column = 0; column < head_dim; column++)
            score = __fadd_rn(score,
                __fmul_rn(query[column], key[column]));
        scores[index] = __fmul_rn(score, score_scale);
    }
}

__global__ static void cuda_text_attention_reduce_global_exact(
        int full_attention, uint32_t n_heads, uint32_t n_kv_heads,
        uint32_t head_dim, uint32_t window,
        const uint32_t *source_position_pointer, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity,
        const float *committed_values, const float *tentative_values,
        uint32_t task_first, uint32_t task_count, uint32_t score_stride,
        float *scores, float *output, int *status) {
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ int score_count;
    uint32_t local_task = (uint32_t)blockIdx.x;
    uint32_t task, source_position, row, head, groups, kv_head;
    uint32_t position, first = 0, count;
    if (!status) return;
    if (!source_position_pointer || !parents || !depths || !scores ||
        !output || !committed_values || !tentative_values ||
        n_heads == 0u || n_kv_heads == 0u ||
        n_heads % n_kv_heads != 0u || head_dim == 0u ||
        query_width == 0u || kv_width == 0u || score_stride == 0u) {
        cuda_text_fail_code(status, 241);
        return;
    }
    if (*status != 0 || local_task >= task_count) return;
    task = task_first + local_task;
    row = task / n_heads;
    head = task - row * n_heads;
    if (row >= rows) return;
    groups = n_heads / n_kv_heads;
    kv_head = head / groups;
    source_position = *source_position_pointer;
    position = source_position + depths[row];
    if (!full_attention && position + 1u > window)
        first = position + 1u - window;
    count = position - first + 1u;
    if (count > score_stride) {
        cuda_text_fail_code(status, 242);
        return;
    }
    scores += (size_t)local_task * score_stride;
    if (threadIdx.x == 0) {
        float peak = -1.0f / 0.0f;
        for (uint32_t index = 0; index < count; index++)
            if (scores[index] > peak) peak = scores[index];
        maximum = peak;
    }
    __syncthreads();
    for (uint32_t index = threadIdx.x; index < count;
         index += blockDim.x)
        scores[index] = cuda_salt_expf(
            __fadd_rn(scores[index], -maximum));
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (uint32_t index = 0; index < count; index++)
            sum = __fadd_rn(sum, scores[index]);
        denominator = sum;
        score_count = (!(sum > 0.0f) || !isfinite(sum))
            ? -1 : (int)count;
        if (score_count < 0) cuda_text_fail_code(status, 243);
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
            const float *value = value_position < source_position
                ? cuda_text_committed_row(
                    committed_values, value_position, source_position,
                    kv_width, kv_mode, kv_capacity)
                : cuda_text_frontier_row(tentative_values, parents, depths,
                    rows, row, value_position - source_position, kv_width);
            if (!value) {
                cuda_text_fail_code(status, 244);
                return;
            }
            value += (size_t)kv_head * head_dim;
            float probability = __fdiv_rn(scores[index], denominator);
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(probability, value[column]));
        }
        head_output[column] = accumulator;
        if (!isfinite(accumulator)) cuda_text_fail_code(status, 245);
    }
}

__global__ static void cuda_text_residual_postnorm_rows(
        const float *residual, float *branch, const float *weight,
        float *output, uint32_t rows, uint32_t width, float epsilon,
        int *status) {
    uint32_t row = (uint32_t)blockIdx.x;
    __shared__ float inverse;
    if (row >= rows) return;
    residual += (size_t)row * width;
    branch += (size_t)row * width;
    output += (size_t)row * width;
    cuda_text_norm_block(branch, weight, branch, width, epsilon, 1,
                         &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fadd_rn(residual[column], branch[column]);
        output[column] = value;
        if (!isfinite(value)) cuda_text_fail(status);
    }
}

__global__ static void cuda_text_router_input_rows(
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
                cuda_text_fail(status);
            sum = __fadd_rn(sum,
                __fmul_rn(input[column], input[column]));
        }
        inverse = cuda_salt_rsqrtf(
            __fadd_rn(__fdiv_rn(sum, (float)width), epsilon));
        effective_root = root > 0.0f ? root :
            __fdiv_rn(1.0f, cuda_text_sqrtf((float)width));
        if (!isfinite(inverse) || !isfinite(effective_root))
            cuda_text_fail(status);
    }
    __syncthreads();
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(input[column], inverse);
        value = __fmul_rn(value, weight[column]);
        value = __fmul_rn(value, effective_root);
        output[column] = value;
        if (!isfinite(value)) cuda_text_fail(status);
    }
}

__global__ static void cuda_text_topk_rows(
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
            cuda_text_fail(status);
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
            cuda_salt_expf(__fadd_rn(logits[expert], -maximum)));
    if (!(all_sum > 0.0f) || !isfinite(all_sum)) cuda_text_fail(status);
    for (uint32_t rank = 0; rank < topk; rank++) {
        weights[rank] = __fdiv_rn(
            cuda_salt_expf(__fadd_rn(weights[rank], -maximum)), all_sum);
        selected_sum = __fadd_rn(selected_sum, weights[rank]);
    }
    if (!(selected_sum > 0.0f) || !isfinite(selected_sum))
        cuda_text_fail(status);
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

__global__ static void cuda_text_group_maps(
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
            cuda_text_fail(status);
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
        cuda_text_fail(status);
        return;
    }
    for (uint32_t canonical = 0; canonical < jobs; canonical++) {
        int32_t expert = selected[canonical];
        uint32_t grouped = (uint32_t)scratch[expert]++;
        if (grouped >= jobs) {
            cuda_text_fail(status);
            return;
        }
        grouped_to_canonical[grouped] = (int32_t)canonical;
    }
    for (uint32_t grouped = 0; grouped < jobs; grouped++) {
        uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
        if (canonical >= jobs) {
            cuda_text_fail(status);
            return;
        }
        canonical_to_grouped[canonical] = (int32_t)grouped;
    }
}

__global__ static void cuda_text_routed_gather(
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
        cuda_text_fail(status);
        return;
    }
    uint32_t row = canonical / topk;
    routed_input[index] = rows_input[(size_t)row * hidden + column];
}

__global__ static void cuda_text_activate(
        const float *gate, const float *up, float *output,
        size_t elements, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = __fmul_rn(cuda_gemma4_gelu_tanh(gate[index]), up[index]);
    output[index] = value;
    if (!isfinite(value)) cuda_text_fail(status);
}

__global__ static void cuda_text_expert_q4(
        const CudaTextTensorRef *experts, uint32_t layer, uint32_t phase,
        const int32_t *selected, const int32_t *grouped_to_canonical,
        uint32_t jobs, uint32_t topk, const float *input,
        float *output, float *paired_up,
        uint32_t expected_rows, uint32_t expected_cols,
        int *status) {
    /* Both independent projections use the existing selected-resource table
     * and canonical seats; their accumulators and reduction order are private. */
    if (paired_up) {
        phase = (uint32_t)blockIdx.z;
        if (phase == 1u) output = paired_up;
    }
    uint32_t grouped = (uint32_t)blockIdx.y;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    uint32_t row = (uint32_t)blockIdx.x * 4u + (uint32_t)warp;
    CudaTextTensorRef ref = {0};
    int valid = grouped < jobs;
    if (valid) {
        uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
        if (canonical >= jobs) valid = 0;
        else {
            int32_t expert = selected[canonical];
            if (expert < 0 || (uint32_t)expert >= CUDA_TEXT_EXPERTS)
                valid = 0;
            else
                ref = experts[((size_t)layer * CUDA_TEXT_EXPERTS +
                    (uint32_t)expert) * 3u + phase];
        }
    }
    if (!valid || ref.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        ref.rows != expected_rows || ref.cols != expected_cols ||
        !ref.resource) {
        if (lane == 0 && warp == 0) cuda_text_fail(status);
        return;
    }
    float accumulator = 0.0f;
    if (row < ref.rows) {
        const float *row_input = input + (size_t)grouped * ref.cols;
        for (uint32_t column = (uint32_t)lane; column < ref.cols;
             column += 32u) {
            uint64_t k = (uint64_t)row * ref.cols + column;
            uint64_t group = k / 64u;
            uint32_t packed = cuda_load_u32(
                ref.resource + ref.value_offset + (k >> 3u) * 4u);
            uint32_t quantized = (packed >> ((k & 7u) * 4u)) & 15u;
            float value = __fmul_rn((float)quantized,
                cuda_bf16(ref.resource + ref.scale_offset + group * 2u));
            value = __fadd_rn(value,
                cuda_bf16(ref.resource + ref.bias_offset + group * 2u));
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

__global__ static void cuda_text_expert_reduce(
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
                cuda_text_fail(status);
                return;
            }
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(weights[canonical],
                    job_outputs[(size_t)grouped * hidden + column]));
        }
        output[(size_t)row * hidden + column] = accumulator;
        if (!isfinite(accumulator)) cuda_text_fail(status);
    }
}

__global__ static void cuda_text_parallel_combine_rows(
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
        if (!isfinite(effective_scalar)) cuda_text_fail(status);
    }
    __syncthreads();
    residual += (size_t)row * width;
    dense += (size_t)row * width;
    routed += (size_t)row * width;
    scratch_a += (size_t)row * width;
    scratch_b += (size_t)row * width;
    output += (size_t)row * width;
    cuda_text_norm_block(dense, dense_weight, scratch_a, width, epsilon, 1,
                         &inverse, status);
    cuda_text_norm_block(routed, routed_weight, scratch_b, width, epsilon, 1,
                         &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        scratch_a[column] = __fadd_rn(scratch_a[column], scratch_b[column]);
        if (!isfinite(scratch_a[column])) cuda_text_fail(status);
    }
    __syncthreads();
    cuda_text_norm_block(scratch_a, final_weight, scratch_b,
                         width, epsilon, 1, &inverse, status);
    for (uint32_t column = threadIdx.x; column < width;
         column += blockDim.x) {
        float value = __fmul_rn(
            __fadd_rn(residual[column], scratch_b[column]), effective_scalar);
        output[column] = value;
        if (!isfinite(value)) cuda_text_fail(status);
    }
}

__global__ static void cuda_text_softcap(
        float *values, size_t elements, float cap, int *status) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float value = values[index];
    if (!isfinite(value)) {
        cuda_text_fail(status);
        return;
    }
    values[index] = __fmul_rn(cap,
        cuda_salt_tanhf(__fdiv_rn(value, cap)));
}

__global__ static void cuda_moe_activate(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *gate_up, float *chains) {
    int descriptor_index = (int)blockIdx.z;
    if (descriptor_index >= descriptor_count) return;
    CudaBatchDesc descriptor = descriptors[descriptor_index];
    size_t elements = (size_t)descriptor.batch * (size_t)descriptor.rows;
    size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index >= elements) return;
    float gate = gate_up[descriptor.input_offset + index];
    float up = gate_up[descriptor.output_offset + index];
    chains[descriptor.aux_offset + index] =
        __fmul_rn(cuda_gemma4_gelu_tanh(gate), up);
}

__global__ static void cuda_moe_activate_contiguous(
    const float *gate, const float *up, float *chain, size_t elements) {
    size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (index < elements)
        chain[index] = __fmul_rn(cuda_gemma4_gelu_tanh(gate[index]), up[index]);
}

__global__ static void cuda_q8_shared_tile(
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
                float scale = cuda_bf16(resource + soff + g * 2u);
                float bias = cuda_bf16(resource + boff + g * 2u);
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

__global__ static void cuda_q4_heterogeneous(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y) {
    __shared__ float dequant[32];
    int descriptor_index = (int)blockIdx.z;
    int r = (int)blockIdx.x;
    int token = (int)blockIdx.y * (int)blockDim.x + (int)threadIdx.x;
    if (descriptor_index >= descriptor_count) return;
    CudaBatchDesc descriptor = descriptors[descriptor_index];
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
                uint32_t packed = cuda_load_u32(descriptor.resource +
                    descriptor.value_offset + (k >> 3u) * 4u);
                uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
                float scale = cuda_bf16(descriptor.resource +
                    descriptor.scale_offset + g * 2u);
                float bias = cuda_bf16(descriptor.resource +
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

__global__ static void cuda_q4_heterogeneous_warp_batch(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y) {
    __shared__ float dequant[32];
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x;
    int token_base = (int)blockIdx.y * 512;
    CudaBatchDesc descriptor = {0};
    float acc[16];
    if (descriptor_index < descriptor_count)
        descriptor = descriptors[descriptor_index];
    if (descriptor_index >= descriptor_count || r >= descriptor.rows ||
        token_base >= descriptor.batch)
        return;
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    for (int cb = 0; cb < descriptor.cols; cb += 32) {
        if (warp == 0) {
            int c = cb + lane;
            if (c < descriptor.cols) {
                uint64_t k = (uint64_t)(uint32_t)r *
                             (uint32_t)descriptor.cols + (uint32_t)c;
                uint64_t g = k / 64u;
                uint32_t packed = cuda_load_u32(descriptor.resource +
                    descriptor.value_offset + (k >> 3u) * 4u);
                uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
                float scale = cuda_bf16(descriptor.resource +
                    descriptor.scale_offset + g * 2u);
                float bias = cuda_bf16(descriptor.resource +
                    descriptor.bias_offset + g * 2u);
                float value = __fmul_rn((float)q, scale);
                dequant[lane] = __fadd_rn(value, bias);
            } else {
                dequant[lane] = 0.0f;
            }
        }
        __syncthreads();
        int index = 0;
        for (int token = token_base + warp;
             token < descriptor.batch && token < token_base + 512;
             token += 32, index++) {
            int c = cb + lane;
            if (c < descriptor.cols) {
                float product = __fmul_rn(
                    dequant[lane],
                    x[descriptor.input_offset + (size_t)token *
                        (size_t)descriptor.cols + (size_t)c]);
                acc[index] = __fadd_rn(acc[index], product);
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

__global__ static void cuda_q4_heterogeneous_warp(
    const CudaBatchDesc *descriptors, int descriptor_count,
    const float *x, float *y, CudaBatchDesc first,
    CudaBatchDesc second, CudaBatchDesc third) {
    int descriptor_index = (int)blockIdx.z;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    int r = (int)blockIdx.x * 4 + warp;
    float acc = 0.0f;
    CudaBatchDesc descriptor = {0};
    if (descriptor_index < descriptor_count) {
        if (descriptors) descriptor = descriptors[descriptor_index];
        else if (descriptor_index == 0) descriptor = first;
        else if (descriptor_index == 1) descriptor = second;
        else if (descriptor_index == 2) descriptor = third;
    }
    if (descriptor_index < descriptor_count && descriptor.batch == 1 &&
        r < descriptor.rows) {
        const float *input = x + descriptor.input_offset;
        for (int c = lane; c < descriptor.cols; c += 32) {
            uint64_t k = (uint64_t)(uint32_t)r *
                         (uint32_t)descriptor.cols + (uint32_t)c;
            uint64_t g = k / 64u;
            uint32_t packed = cuda_load_u32(descriptor.resource +
                descriptor.value_offset + (k >> 3u) * 4u);
            uint32_t q = (packed >> ((k & 7u) * 4u)) & 15u;
            float scale = cuda_bf16(descriptor.resource +
                descriptor.scale_offset + g * 2u);
            float bias = cuda_bf16(descriptor.resource +
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

static CudaResource *cuda_resource_record(uint32_t kind, uint32_t id) {
    for (int i = 0; i < CUDA_MAX_RESOURCES; i++)
        if (cuda_resources[i].used && cuda_resources[i].kind == kind &&
            cuda_resources[i].id == id)
            return &cuda_resources[i];
    return NULL;
}

static CudaResource *cuda_resource(uint32_t kind, uint32_t id) {
    CudaResource *resource = cuda_resource_record(kind, id);
    return resource && resource->active && resource->device ? resource : NULL;
}

static CudaTensor *cuda_tensor(const void *key) {
    for (int i = 0; i < cuda_tensor_count; i++)
        if (cuda_tensors[i].used && cuda_tensors[i].key == key)
            return &cuda_tensors[i];
    return NULL;
}

static int cuda_range(const CudaResource *resource, const void *pointer,
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

static int cuda_tensor_sizes(int bits, int R, int C,
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

extern "C" int salt_gpu_startup_requirements(
    SaltGpuStartupRequirements *requirements, size_t requirements_size) {
    uint64_t output_bytes, descriptor_bytes;
    if (!requirements || requirements_size != sizeof *requirements)
        return -1;
    memset(requirements, 0, sizeof *requirements);
    output_bytes = (uint64_t)CUDA_MAX_OUTPUTS * sizeof(float);
    descriptor_bytes =
        (uint64_t)CUDA_MAX_BATCH_JOBS * sizeof(CudaBatchDesc);
    requirements->fixed_device_bytes =
        3u * output_bytes + output_bytes / 2u + output_bytes / 16u +
        descriptor_bytes + sizeof(int);
    requirements->fixed_pinned_host_bytes =
        2u * output_bytes + descriptor_bytes;
    requirements->descriptor_bytes = 2u * descriptor_bytes;
    requirements->max_output_floats = CUDA_MAX_OUTPUTS;
    requirements->max_batch_jobs = CUDA_MAX_BATCH_JOBS;
    requirements->shared_buffer_slots = 0;
    return 0;
}

extern "C" int salt_gpu_init(void) {
    cudaDeviceProp property;
    const char *pageable = getenv("SALT_CUDA_PAGEABLE_MMAP");
    int pageable_access = 0, host_page_tables = 0;
    if (cuda_ready) return 0;
    if (pageable && strcmp(pageable, "0") && strcmp(pageable, "1"))
        return -1;
    cudaError_t flags = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (flags != cudaSuccess && flags != cudaErrorSetOnActiveProcess)
        return cuda_success(flags, "cudaSetDeviceFlags");
    if (cuda_success(cudaGetDeviceProperties(&property, 0),
                     "cudaGetDeviceProperties") != 0 ||
        !property.unifiedAddressing)
        return -1;
    cuda_pageable_mmap = pageable && !strcmp(pageable, "1");
    if (cuda_pageable_mmap &&
        (cuda_success(cudaDeviceGetAttribute(&pageable_access,
                      cudaDevAttrPageableMemoryAccess, 0),
                      "cudaDeviceGetAttribute pageable memory") != 0 ||
         cuda_success(cudaDeviceGetAttribute(&host_page_tables,
                      cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0),
                      "cudaDeviceGetAttribute host page tables") != 0 ||
         !pageable_access || !host_page_tables)) {
        cuda_pageable_mmap = 0;
        return -1;
    }
    {
        const char *telemetry = getenv("SALT_NVFP4_TELEMETRY");
        const char *timing = getenv("SALT_NVFP4_TIMING");
        cuda_nvfp4_telemetry_enabled =
            !telemetry || !*telemetry || *telemetry != '0';
        cuda_nvfp4_timing_enabled = cuda_nvfp4_telemetry_enabled &&
            timing && *timing && *timing != '0';
    }
    if (cuda_nvfp4_timing_enabled &&
        (cuda_success(cudaEventCreate(&cuda_nvfp4_event_start),
                      "cudaEventCreate NVFP4 start") != 0 ||
         cuda_success(cudaEventCreate(&cuda_nvfp4_event_qdq),
                      "cudaEventCreate NVFP4 QDQ") != 0 ||
         cuda_success(cudaEventCreate(&cuda_nvfp4_event_projection),
                      "cudaEventCreate NVFP4 projection") != 0 ||
         cuda_success(cudaEventCreate(&cuda_nvfp4_event_d2h),
                      "cudaEventCreate NVFP4 D2H") != 0)) {
        if (cuda_nvfp4_event_d2h) cudaEventDestroy(cuda_nvfp4_event_d2h);
        if (cuda_nvfp4_event_projection)
            cudaEventDestroy(cuda_nvfp4_event_projection);
        if (cuda_nvfp4_event_qdq) cudaEventDestroy(cuda_nvfp4_event_qdq);
        if (cuda_nvfp4_event_start) cudaEventDestroy(cuda_nvfp4_event_start);
        cuda_nvfp4_event_start = cuda_nvfp4_event_qdq =
            cuda_nvfp4_event_projection = cuda_nvfp4_event_d2h = NULL;
        cuda_nvfp4_timing_enabled = 0;
        return -1;
    }
    if (cuda_success(cudaMalloc((void **)&cuda_x,
                                (size_t)CUDA_MAX_OUTPUTS * sizeof(float)),
                     "cudaMalloc input") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_y,
                                (size_t)CUDA_MAX_OUTPUTS * sizeof(float)),
                     "cudaMalloc output") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_attention_status, sizeof(int)),
                     "cudaMalloc attention status") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_nvfp4_xdq,
                                (size_t)CUDA_MAX_OUTPUTS * sizeof(float)),
                     "cudaMalloc NVFP4 input QDQ") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_nvfp4_packed,
                                (size_t)CUDA_MAX_OUTPUTS / 2u),
                     "cudaMalloc NVFP4 input packed") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_nvfp4_scales,
                                (size_t)CUDA_MAX_OUTPUTS / 16u),
                     "cudaMalloc NVFP4 input scales") != 0 ||
        cuda_success(cudaMalloc((void **)&cuda_device_desc,
                                (size_t)CUDA_MAX_BATCH_JOBS *
                                    sizeof(CudaBatchDesc)),
                     "cudaMalloc descriptors") != 0 ||
        cuda_success(cudaHostAlloc((void **)&cuda_host_x,
                                   (size_t)CUDA_MAX_OUTPUTS * sizeof(float),
                                   cudaHostAllocDefault),
                     "cudaHostAlloc input") != 0 ||
        cuda_success(cudaHostAlloc((void **)&cuda_host_y,
                                   (size_t)CUDA_MAX_OUTPUTS * sizeof(float),
                                   cudaHostAllocDefault),
                     "cudaHostAlloc output") != 0 ||
        cuda_success(cudaHostAlloc((void **)&cuda_host_desc,
                                   (size_t)CUDA_MAX_BATCH_JOBS *
                                       sizeof(CudaBatchDesc),
                                   cudaHostAllocDefault),
                     "cudaHostAlloc descriptors") != 0) {
        if (cuda_host_desc) cudaFreeHost(cuda_host_desc);
        if (cuda_host_y) cudaFreeHost(cuda_host_y);
        if (cuda_host_x) cudaFreeHost(cuda_host_x);
        if (cuda_device_desc) cudaFree(cuda_device_desc);
        if (cuda_nvfp4_scales) cudaFree(cuda_nvfp4_scales);
        if (cuda_nvfp4_packed) cudaFree(cuda_nvfp4_packed);
        if (cuda_nvfp4_xdq) cudaFree(cuda_nvfp4_xdq);
        if (cuda_y) cudaFree(cuda_y);
        if (cuda_attention_status) cudaFree(cuda_attention_status);
        if (cuda_x) cudaFree(cuda_x);
        if (cuda_nvfp4_event_d2h) cudaEventDestroy(cuda_nvfp4_event_d2h);
        if (cuda_nvfp4_event_projection)
            cudaEventDestroy(cuda_nvfp4_event_projection);
        if (cuda_nvfp4_event_qdq) cudaEventDestroy(cuda_nvfp4_event_qdq);
        if (cuda_nvfp4_event_start) cudaEventDestroy(cuda_nvfp4_event_start);
        cuda_host_desc = NULL;
        cuda_host_x = cuda_host_y = NULL;
        cuda_device_desc = NULL;
        cuda_nvfp4_xdq = NULL;
        cuda_nvfp4_packed = cuda_nvfp4_scales = NULL;
        cuda_x = cuda_y = NULL;
        cuda_attention_status = NULL;
        cuda_nvfp4_event_start = cuda_nvfp4_event_qdq =
            cuda_nvfp4_event_projection = cuda_nvfp4_event_d2h = NULL;
        cuda_nvfp4_timing_enabled = 0;
        return -1;
    }
    memset(cuda_resources, 0, sizeof cuda_resources);
    memset(cuda_tensors, 0, sizeof cuda_tensors);
    memset(&cuda_stats, 0, sizeof cuda_stats);
    cuda_q4_hetero_logical_input_bytes = 0;
    cuda_q4_hetero_transfer_input_bytes = 0;
    cuda_q4_hetero_output_bytes = 0;
    cuda_q4_hetero_descriptor_bytes = 0;
    cuda_q4_hetero_reused_inputs = 0;
    cuda_q4_hetero_kernel_launches = 0;
    cuda_q4_hetero_completion_fences = 0;
    cuda_peak_window_bytes = 0;
    cuda_selected_retirement_fenced = 0;
    cuda_text_resource_fence_credit = 0;

    cuda_tensor_count = 0;
    cuda_ready = 1;
    fprintf(stderr, "gpu: CUDA ready (%s cc=%d.%d)\n",
            property.name, property.major, property.minor);
    if (cuda_pageable_mmap)
        fputs("gpu: CUDA pageable mmap enabled\n", stderr);
    return 0;
}

extern "C" int salt_gpu_free(void) {
    int failed = 0;
    if (cuda_ready && cuda_success(cudaDeviceSynchronize(),
                                   "cudaDeviceSynchronize free") != 0)
        failed = 1;
    for (int i = CUDA_MAX_RESOURCES - 1; i >= 0; i--)
        if (cuda_resources[i].used) {
            if (cuda_resources[i].registered &&
                cuda_success(cudaHostUnregister(
                    (void *)cuda_resources[i].host),
                    "cudaHostUnregister free") != 0)
                failed = 1;
            memset(&cuda_resources[i], 0, sizeof cuda_resources[i]);
        }
    if (cuda_x && cuda_success(cudaFree(cuda_x), "cudaFree input") != 0)
        failed = 1;
    if (cuda_y && cuda_success(cudaFree(cuda_y), "cudaFree output") != 0)
        failed = 1;
    /* Attention borrows the initialized shared KV seat; its owner frees it. */
    if (cuda_attention_status &&
        cuda_success(cudaFree(cuda_attention_status),
                     "cudaFree attention status") != 0)
        failed = 1;
    if (cuda_nvfp4_xdq &&
        cuda_success(cudaFree(cuda_nvfp4_xdq),
                     "cudaFree NVFP4 input QDQ") != 0)
        failed = 1;
    if (cuda_nvfp4_packed &&
        cuda_success(cudaFree(cuda_nvfp4_packed),
                     "cudaFree NVFP4 input packed") != 0)
        failed = 1;
    if (cuda_nvfp4_scales &&
        cuda_success(cudaFree(cuda_nvfp4_scales),
                     "cudaFree NVFP4 input scales") != 0)
        failed = 1;
    if (cuda_device_desc &&
        cuda_success(cudaFree(cuda_device_desc),
                     "cudaFree descriptors") != 0)
        failed = 1;
    if (cuda_host_x &&
        cuda_success(cudaFreeHost(cuda_host_x),
                     "cudaFreeHost input") != 0)
        failed = 1;
    if (cuda_host_y &&
        cuda_success(cudaFreeHost(cuda_host_y),
                     "cudaFreeHost output") != 0)
        failed = 1;
    if (cuda_host_desc &&
        cuda_success(cudaFreeHost(cuda_host_desc),
                     "cudaFreeHost descriptors") != 0)
        failed = 1;
    free(cuda_selected_logical_slots);
    free(cuda_selected_resources);
    cuda_selected_logical_slots = NULL;
    cuda_selected_resources = NULL;
    cuda_selected_capacity = 0;
    cuda_selected_logical_capacity = 0;
    cuda_selected_retirement_fenced = 0;
    cuda_text_resource_fence_credit = 0;
    if (cuda_nvfp4_event_d2h &&
        cuda_success(cudaEventDestroy(cuda_nvfp4_event_d2h),
                     "cudaEventDestroy NVFP4 D2H") != 0)
        failed = 1;
    if (cuda_nvfp4_event_projection &&
        cuda_success(cudaEventDestroy(cuda_nvfp4_event_projection),
                     "cudaEventDestroy NVFP4 projection") != 0)
        failed = 1;
    if (cuda_nvfp4_event_qdq &&
        cuda_success(cudaEventDestroy(cuda_nvfp4_event_qdq),
                     "cudaEventDestroy NVFP4 QDQ") != 0)
        failed = 1;
    if (cuda_nvfp4_event_start &&
        cuda_success(cudaEventDestroy(cuda_nvfp4_event_start),
                     "cudaEventDestroy NVFP4 start") != 0)
        failed = 1;
    cuda_x = cuda_y = NULL;
    cuda_nvfp4_xdq = NULL;
    cuda_nvfp4_packed = cuda_nvfp4_scales = NULL;
    cuda_host_x = cuda_host_y = NULL;
    cuda_host_desc = cuda_device_desc = NULL;
    cuda_nvfp4_event_start = cuda_nvfp4_event_qdq =
        cuda_nvfp4_event_projection = cuda_nvfp4_event_d2h = NULL;
    cuda_attention_kv = NULL;
    cuda_attention_kv_nbytes = 0;
    cuda_attention_kv_host = NULL;
    cuda_attention_status = NULL;
    cuda_attention_kv_sync_count = 0;
    memset(cuda_attention_kv_sync, 0, sizeof cuda_attention_kv_sync);
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    if (getenv("SALT_CUDA_TRANSFER_SUMMARY"))
        fprintf(stderr,
            "SALT_CUDA_Q4_TRANSFER logical_input_bytes=%llu "
            "transfer_input_bytes=%llu reused_inputs=%llu "
            "output_bytes=%llu descriptor_bytes=%llu "
            "kernel_launches=%llu completion_fences=%llu\n",
            (unsigned long long)cuda_q4_hetero_logical_input_bytes,
            (unsigned long long)cuda_q4_hetero_transfer_input_bytes,
            (unsigned long long)cuda_q4_hetero_reused_inputs,
            (unsigned long long)cuda_q4_hetero_output_bytes,
            (unsigned long long)cuda_q4_hetero_descriptor_bytes,
            (unsigned long long)cuda_q4_hetero_kernel_launches,
            (unsigned long long)cuda_q4_hetero_completion_fences);
    cuda_ready = cuda_mapped_only = cuda_defer = cuda_pageable_mmap = 0;
    cuda_nvfp4_telemetry_enabled = 1;
    cuda_nvfp4_timing_enabled = 0;
    cuda_tensor_count = 0;
    memset(cuda_tensors, 0, sizeof cuda_tensors);
    memset(&cuda_stats, 0, sizeof cuda_stats);
    cuda_q4_hetero_logical_input_bytes = 0;
    cuda_q4_hetero_transfer_input_bytes = 0;
    cuda_q4_hetero_output_bytes = 0;
    cuda_q4_hetero_descriptor_bytes = 0;
    cuda_q4_hetero_reused_inputs = 0;
    cuda_q4_hetero_kernel_launches = 0;
    cuda_q4_hetero_completion_fences = 0;
    cuda_peak_window_bytes = 0;
    return failed ? -1 : 0;
}

extern "C" int salt_gpu_weight_resource_describe(
    uint32_t kind, uint32_t resource_id, const void *base, size_t nbytes) {
    CudaResource *slot = NULL;
    if (!cuda_ready || !base || nbytes < 1 ||
        cuda_resource_record(kind, resource_id))
        return -1;
    for (int i = 0; i < CUDA_MAX_RESOURCES; i++)
        if (!cuda_resources[i].used) { slot = &cuda_resources[i]; break; }
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
    CudaResource *resource = cuda_resource_record(kind, resource_id);
    void *device = NULL;
    if (!resource || resource->active || resource->device ||
        policy == SALT_GPU_WEIGHT_ADDRESS_AUTO ||
        policy == SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
        return -1;
    if (policy == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE) {
        if (!cuda_pageable_mmap) return -1;
        device = (void *)resource->host;
    } else if (policy == SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT) {
        if (cuda_pageable_mmap ||
            cuda_success(cudaHostRegister((void *)resource->host,
                resource->nbytes, cudaHostRegisterMapped),
                "cudaHostRegister") != 0)
            return -1;
        if (cuda_success(cudaHostGetDevicePointer(
                &device, (void *)resource->host, 0),
                "cudaHostGetDevicePointer") != 0) {
            cudaHostUnregister((void *)resource->host);
            return -1;
        }
        resource->registered = 1;
    } else {
        return -1;
    }
    resource->device = (unsigned char *)device;
    resource->policy = policy;
    resource->active = 1;
    return 0;
}

extern "C" int salt_gpu_weight_resource_pool_bind(
    uint32_t kind, uint32_t resource_id) {
    (void)kind;
    (void)resource_id;
    return -1;
}

extern "C" int salt_gpu_weight_resource_deactivate(
    uint32_t kind, uint32_t resource_id) {
    CudaResource *resource = cuda_resource_record(kind, resource_id);
    if (!resource || !resource->active ||
        cuda_success(cudaDeviceSynchronize(),
                     "cudaDeviceSynchronize deactivate") != 0 ||
        (resource->registered &&
         cuda_success(cudaHostUnregister((void *)resource->host),
                      "cudaHostUnregister deactivate") != 0))
        return -1;
    resource->active = 0;
    resource->registered = 0;
    resource->policy = SALT_GPU_WEIGHT_ADDRESS_AUTO;
    resource->device = NULL;
    resource->window_offset = 0;
    resource->window_nbytes = 0;
    return 0;
}

extern "C" int salt_gpu_weight_resource_usage(
    SaltGpuWeightResourceUsage *usage, size_t usage_size) {
    if (!usage || usage_size != sizeof *usage) return -1;
    memset(usage, 0, sizeof *usage);
    for (int i = 0; i < CUDA_MAX_RESOURCES; i++) {
        const CudaResource *resource = &cuda_resources[i];
        if (!resource->used) continue;
        usage->described_resources++;
        usage->described_bytes += resource->nbytes;
        if (!resource->active) continue;
        usage->active_resources++;
        if (resource->policy == SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT)
            usage->registered_bytes += resource->nbytes;
        else if (resource->policy == SALT_GPU_WEIGHT_ADDRESS_PAGEABLE)
            usage->pageable_bytes += resource->nbytes;
        if (resource->window_nbytes) {
            usage->active_windows++;
            usage->active_window_bytes += resource->window_nbytes;
        }
    }
    usage->peak_window_bytes = cuda_peak_window_bytes;
    return 0;
}

extern "C" int salt_gpu_weight_resource_bind(
    uint32_t kind, uint32_t resource_id, const void *base, size_t nbytes) {
    SaltGpuWeightAddressability policy = cuda_pageable_mmap
        ? SALT_GPU_WEIGHT_ADDRESS_PAGEABLE
        : SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT;
    if (salt_gpu_weight_resource_describe(kind, resource_id, base, nbytes) != 0)
        return -1;
    if (salt_gpu_weight_resource_activate(kind, resource_id, policy) != 0) {
        CudaResource *resource = cuda_resource_record(kind, resource_id);
        if (resource) memset(resource, 0, sizeof *resource);
        return -1;
    }
    return 0;
}

extern "C" int salt_gpu_weight_window_bind(
    uint32_t kind, uint32_t resource_id, uint64_t offset, size_t nbytes) {
    CudaResource *resource = cuda_resource_record(kind, resource_id);
    if (!resource || !resource->active || resource->window_nbytes || nbytes < 1 ||
        offset > resource->nbytes || nbytes > resource->nbytes - (size_t)offset ||
        resource->policy == SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW)
        return -1;
    resource->window_offset = offset;
    resource->window_nbytes = nbytes;
    if ((uint64_t)nbytes > cuda_peak_window_bytes)
        cuda_peak_window_bytes = nbytes;
    return 0;
}

extern "C" int salt_gpu_weight_window_unbind(
    uint32_t kind, uint32_t resource_id) {
    CudaResource *resource = cuda_resource_record(kind, resource_id);
    if (!resource || !resource->active || !resource->window_nbytes ||
        cuda_success(cudaDeviceSynchronize(),
                     "cudaDeviceSynchronize window unbind") != 0)
        return -1;
    resource->window_offset = 0;
    resource->window_nbytes = 0;
    return 0;
}

extern "C" int salt_gpu_weight_resource_slot(
    uint32_t kind, uint32_t resource_id, const void *key,
    const uint32_t *vals, const uint16_t *scales, const uint16_t *biases,
    int bits, int R, int C) {
    CudaResource *resource = cuda_resource(kind, resource_id);
    CudaTensor *existing;
    size_t vbytes, sbytes;
    uint64_t voff, soff, boff;
    if (!resource || !key || !vals || !scales || !biases ||
        cuda_tensor_sizes(bits, R, C, &vbytes, &sbytes) != 0 ||
        cuda_range(resource, vals, vbytes, &voff) != 0 ||
        cuda_range(resource, scales, sbytes, &soff) != 0 ||
        cuda_range(resource, biases, sbytes, &boff) != 0)
        return -1;
    existing = cuda_tensor(key);
    if (existing)
        return existing->kind == kind && existing->resource_id == resource_id &&
               existing->bits == bits && existing->rows == R &&
               existing->cols == C && existing->value_offset == voff &&
               existing->scale_offset == soff &&
               existing->bias_offset == boff ? 0 : -1;
    if (cuda_tensor_count >= CUDA_MAX_TENSORS) return -1;
    CudaTensor *tensor = &cuda_tensors[cuda_tensor_count++];
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
    CudaResource *resource = cuda_resource(kind, resource_id);
    CudaTensor *existing;
    uint64_t elements, value_bytes, scale_bytes;
    uint64_t voff, soff, wgoff, igoff;
    if (!resource || !key || !weight_packed || !weight_scales ||
        !weight_global_scale || !input_global_scale || R < 1 || C < 16 ||
        (C & 15) != 0)
        return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    value_bytes = elements / 2u;
    scale_bytes = elements / 16u;
    if (value_bytes > SIZE_MAX || scale_bytes > SIZE_MAX ||
        cuda_range(resource, weight_packed, (size_t)value_bytes, &voff) != 0 ||
        cuda_range(resource, weight_scales, (size_t)scale_bytes, &soff) != 0 ||
        cuda_range(resource, weight_global_scale, sizeof(float), &wgoff) != 0 ||
        cuda_range(resource, input_global_scale, sizeof(float), &igoff) != 0)
        return -1;
    existing = cuda_tensor(key);
    if (existing)
        return existing->kind == kind &&
               existing->resource_id == resource_id &&
               existing->bits == CUDA_FORMAT_NVFP4 &&
               existing->rows == R && existing->cols == C &&
               existing->value_offset == voff &&
               existing->scale_offset == soff &&
               existing->bias_offset == wgoff &&
               existing->aux_offset == igoff ? 0 : -1;
    if (cuda_tensor_count >= CUDA_MAX_TENSORS) return -1;
    CudaTensor *tensor = &cuda_tensors[cuda_tensor_count++];
    tensor->used = 1;
    tensor->bits = CUDA_FORMAT_NVFP4;
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
    CudaResource *resource = cuda_resource(kind, resource_id);
    CudaTensor *existing;
    uint64_t elements, bytes, offset;
    if (!resource || !key || !weights || R < 1 || C < 8 || (C & 7) != 0)
        return -1;
    elements = (uint64_t)(uint32_t)R * (uint64_t)(uint32_t)C;
    if (elements > SIZE_MAX / 2u) return -1;
    bytes = elements * 2u;
    if (cuda_range(resource, weights, (size_t)bytes, &offset) != 0)
        return -1;
    existing = cuda_tensor(key);
    if (existing)
        return existing->kind == kind && existing->resource_id == resource_id &&
               existing->bits == 16 && existing->rows == R &&
               existing->cols == C && existing->value_offset == offset ? 0 : -1;
    if (cuda_tensor_count >= CUDA_MAX_TENSORS) return -1;
    CudaTensor *tensor = &cuda_tensors[cuda_tensor_count++];
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
    CudaResource *resource = cuda_resource_record(kind, resource_id);
    if (!resource) return -1;
    if (resource->active &&
        salt_gpu_weight_resource_deactivate(kind, resource_id) != 0)
        return -1;
    if (resource->device || resource->registered || resource->window_nbytes)
        return -1;
    for (int i = 0; i < cuda_tensor_count; i++)
        if (cuda_tensors[i].used && cuda_tensors[i].kind == kind &&
            cuda_tensors[i].resource_id == resource_id)
            cuda_tensors[i].used = 0;
    memset(resource, 0, sizeof *resource);
    return 0;
}

static int cuda_run(const void *key, const uint32_t *vals, int bits,
                    int R, int C, const float *x, float *y) {
    CudaTensor *tensor = cuda_tensor(key ? key : vals);
    CudaResource *resource;
    int blocks;
    if (!cuda_ready || !tensor || tensor->bits != bits || tensor->rows != R ||
        tensor->cols != C || C > CUDA_MAX_C || (size_t)R > CUDA_MAX_OUTPUTS ||
        !x || !y) {
        if (cuda_mapped_only) cuda_stats.mapped_only_misses++;
        return -1;
    }
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource) return -1;
    if (cuda_success(cudaMemcpy(cuda_x, x, (size_t)C * sizeof(float),
                                cudaMemcpyHostToDevice), "cudaMemcpy input") != 0)
        return -1;
    blocks = (R + 255) / 256;
    if (bits == 4)
        cuda_q4_exact<<<blocks, 256>>>(resource->device,
            tensor->value_offset, tensor->scale_offset, tensor->bias_offset,
            cuda_x, cuda_y, R, C);
    else
        cuda_q8_exact<<<blocks, 256>>>(resource->device,
            tensor->value_offset, tensor->scale_offset, tensor->bias_offset,
            cuda_x, cuda_y, R, C);
    if (cuda_success(cudaGetLastError(), "projection launch") != 0 ||
        cuda_success(cudaMemcpy(y, cuda_y, (size_t)R * sizeof(float),
                                cudaMemcpyDeviceToHost), "cudaMemcpy output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs++;
    return 0;
}

static int cuda_run_same_weight_batch(const void *key, const uint32_t *vals,
    int bits, int R, int C, int B, const float *const *xs,
    float *const *ys) {
    CudaTensor *tensor = cuda_tensor(key ? key : vals);
    CudaResource *resource;
    size_t input_floats, output_floats;
    if (!cuda_ready || !tensor || tensor->bits != bits || tensor->rows != R ||
        tensor->cols != C || B < 1 || B > 512 || C > CUDA_MAX_C ||
        !xs || !ys || !xs[0] || !ys[0] ||
        (size_t)B > SIZE_MAX / (size_t)C ||
        (size_t)B > SIZE_MAX / (size_t)R) {
        if (cuda_mapped_only) cuda_stats.mapped_only_misses++;
        return -1;
    }
    input_floats = (size_t)B * (size_t)C;
    output_floats = (size_t)B * (size_t)R;
    if (input_floats > CUDA_MAX_OUTPUTS || output_floats > CUDA_MAX_OUTPUTS)
        return -1;
    for (int b = 1; b < B; b++)
        if (xs[b] != xs[0] + (size_t)b * (size_t)C ||
            ys[b] != ys[0] + (size_t)b * (size_t)R)
            return -1;
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource) return -1;
    if (cuda_success(cudaMemcpy(cuda_x, xs[0],
                                input_floats * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy batch input") != 0)
        return -1;
    if (B >= 8) {
        if (bits == 4)
            cuda_q4_warp_batch<<<(unsigned int)R, 1024>>>(resource->device,
                tensor->value_offset, tensor->scale_offset,
                tensor->bias_offset, cuda_x, cuda_y, R, C, B);
        else {
            dim3 grid((unsigned int)R,
                      (unsigned int)((B + 127) / 128));
            cuda_q8_shared_tile<<<grid, 128>>>(resource->device,
                tensor->value_offset, tensor->scale_offset,
                tensor->bias_offset, cuda_x, cuda_y, R, C, B);
        }
    } else if (bits == 4) {
        cuda_q4_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)B), 128>>>(
            resource->device, tensor->value_offset, tensor->scale_offset,
            tensor->bias_offset, cuda_x, cuda_y, R, C);
    } else {
        cuda_q8_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)B), 128>>>(
            resource->device, tensor->value_offset, tensor->scale_offset,
            tensor->bias_offset, cuda_x, cuda_y, R, C);
    }
    if (cuda_success(cudaGetLastError(), "batch projection launch") != 0 ||
        cuda_success(cudaMemcpy(ys[0], cuda_y,
                                output_floats * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy batch output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += (uint64_t)B;
    return 0;
}

extern "C" int salt_gpu_q4_matvec(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C,
    const float *x, float *y) {
    (void)scales; (void)biases;
    return cuda_run(vals, vals, 4, R, C, x, y);
}

extern "C" int salt_gpu_q8_matvec(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C,
    const float *x, float *y) {
    (void)scales; (void)biases;
    return cuda_run(vals, vals, 8, R, C, x, y);
}

extern "C" int salt_gpu_bf16_matvec(
    const void *key, int R, int C, const float *input, float *output) {
    CudaTensor *tensor = cuda_tensor(key);
    CudaResource *resource;
    if (!cuda_ready || !tensor || tensor->bits != 16 ||
        tensor->rows != R || tensor->cols != C || R < 1 || C < 8 ||
        (C & 7) != 0 || (size_t)R > CUDA_MAX_OUTPUTS || !input || !output)
        return -1;
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource ||
        cuda_success(cudaMemcpy(cuda_x, input, (size_t)C * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy BF16 input") != 0)
        return -1;
    cuda_bf16_warp_exact<<<(unsigned int)((R + 3) / 4), 128>>>(
        resource->device, tensor->value_offset, cuda_x, cuda_y, R, C);
    if (cuda_success(cudaGetLastError(), "BF16 projection launch") != 0 ||
        cuda_success(cudaMemcpy(output, cuda_y, (size_t)R * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy BF16 output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs++;
    return 0;
}

extern "C" int salt_gpu_bf16_matvec_batch(
    const void *key, int R, int C, int B,
    const float *inputs, float *outputs) {
    CudaTensor *tensor = cuda_tensor(key);
    CudaResource *resource;
    size_t input_floats, output_floats;
    if (!cuda_ready || !cuda_pageable_mmap || !tensor || tensor->bits != 16 ||
        tensor->rows != R || tensor->cols != C || R < 1 || C < 8 ||
        (C & 7) != 0 || B < 1 || B > CUDA_MAX_BATCH_JOBS ||
        !inputs || !outputs || (size_t)B > SIZE_MAX / (size_t)C ||
        (size_t)B > SIZE_MAX / (size_t)R)
        return -1;
    input_floats = (size_t)B * (size_t)C;
    output_floats = (size_t)B * (size_t)R;
    if (input_floats > CUDA_MAX_OUTPUTS || output_floats > CUDA_MAX_OUTPUTS)
        return -1;
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource || resource->policy != SALT_GPU_WEIGHT_ADDRESS_PAGEABLE ||
        cuda_success(cudaMemcpy(cuda_x, inputs,
                                input_floats * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy BF16 batch input") != 0)
        return -1;
    cuda_bf16_batch_serial_exact<<<
        dim3((unsigned int)((R + 255) / 256), (unsigned int)B), 256>>>(
            resource->device, tensor->value_offset,
            cuda_x, cuda_y, R, C);
    if (cuda_success(cudaGetLastError(), "BF16 batch projection launch") != 0 ||
        cuda_success(cudaMemcpy(outputs, cuda_y,
                                output_floats * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy BF16 batch output") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK)
        cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += (uint64_t)(uint32_t)B;
    return 0;
}

static int cuda_nvfp4_qdq_launch(const float *input, int C, int B,
                                 float input_global_scale,
                                 CudaNvfp4Telemetry *telemetry) {
    int groups, blocks;
    size_t elements;
    uint64_t h2d_start = 0;
    if (!cuda_ready || !input || C < 16 || C > CUDA_MAX_C || B < 1 ||
        (C & 15) != 0 || !(input_global_scale > 0.0f) ||
        (size_t)B > CUDA_MAX_OUTPUTS / (size_t)C || !telemetry)
        return -1;
    elements = (size_t)B * (size_t)C;
    if (cuda_nvfp4_timing_enabled) h2d_start = cuda_host_now_ns();
    if (cuda_success(cudaMemcpy(cuda_x, input, elements * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy NVFP4 input") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled) {
        telemetry->h2d_ns = cuda_host_now_ns() - h2d_start;
        if (cuda_success(cudaEventRecord(cuda_nvfp4_event_start),
                         "cudaEventRecord NVFP4 start") != 0)
            return -1;
    }
    groups = (int)(elements / 16u);
    blocks = (groups + 255) / 256;
    cuda_nvfp4_qdq_exact<<<blocks, 256>>>(
        cuda_x, C, B, input_global_scale, cuda_nvfp4_packed,
        cuda_nvfp4_scales, cuda_nvfp4_xdq);
    if (cuda_success(cudaGetLastError(), "NVFP4 QDQ launch") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled &&
        cuda_success(cudaEventRecord(cuda_nvfp4_event_qdq),
                     "cudaEventRecord NVFP4 QDQ") != 0)
        return -1;
    return 0;
}

extern "C" int salt_gpu_nvfp4_qdq(
    const float *input, int C, float input_global_scale,
    uint8_t *packed, uint8_t *scales, float *dequant) {
    CudaNvfp4Telemetry telemetry;
    if (!packed || !scales || !dequant ||
        cuda_nvfp4_telemetry(1, C, 1, &telemetry) != 0 ||
        cuda_nvfp4_qdq_launch(input, C, 1, input_global_scale,
                              &telemetry) != 0)
        return -1;
    if (
        cuda_success(cudaMemcpy(packed, cuda_nvfp4_packed,
                                (size_t)C / 2u, cudaMemcpyDeviceToHost),
                     "cudaMemcpy NVFP4 packed") != 0 ||
        cuda_success(cudaMemcpy(scales, cuda_nvfp4_scales,
                                (size_t)C / 16u, cudaMemcpyDeviceToHost),
                     "cudaMemcpy NVFP4 scales") != 0 ||
        cuda_success(cudaMemcpy(dequant, cuda_nvfp4_xdq,
                                (size_t)C * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy NVFP4 QDQ") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled) {
        if (cuda_event_elapsed_ns(cuda_nvfp4_event_start,
                                  cuda_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            cuda_nvfp4_d2h_timing(cuda_nvfp4_event_qdq, &telemetry) != 0)
            return -1;
    }
    telemetry.packed_weight_bytes = 0;
    telemetry.weight_scale_bytes = 0;
    telemetry.projection_output_bytes = 0;
    cuda_nvfp4_telemetry_publish(&telemetry, UINT64_C(1));
    return 0;
}

extern "C" int salt_gpu_nvfp4_matvec(
    const void *key, int R, int C, const float *input, float *output) {
    return salt_gpu_nvfp4_matvec_batch(key, R, C, 1, input, output);
}

extern "C" int salt_gpu_nvfp4_matvec_batch(
    const void *key, int R, int C, int B,
    const float *inputs, float *outputs) {
    CudaTensor *tensor = cuda_tensor(key);
    CudaResource *resource;
    CudaNvfp4Telemetry telemetry;
    float input_global_scale;
    if (!cuda_ready || !tensor || tensor->bits != CUDA_FORMAT_NVFP4 ||
        tensor->rows != R || tensor->cols != C || R < 1 || B < 1 ||
        (size_t)B > CUDA_MAX_OUTPUTS / (size_t)R || !inputs || !outputs ||
        cuda_nvfp4_telemetry(R, C, B, &telemetry) != 0)
        return -1;
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource) return -1;
    memcpy(&input_global_scale, resource->host + tensor->aux_offset,
           sizeof input_global_scale);
    if (cuda_nvfp4_qdq_launch(inputs, C, B, input_global_scale,
                              &telemetry) != 0)
        return -1;
    cuda_nvfp4_warp_exact<<<
        dim3((unsigned int)((R + 3) / 4), (unsigned int)B), 128>>>(
        resource->device, tensor->value_offset, tensor->scale_offset,
        tensor->bias_offset, cuda_nvfp4_xdq, cuda_y, R, C, B);
    if (cuda_success(cudaGetLastError(), "NVFP4 projection launch") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled &&
        cuda_success(cudaEventRecord(cuda_nvfp4_event_projection),
                     "cudaEventRecord NVFP4 projection") != 0)
        return -1;
    if (
        cuda_success(cudaMemcpy(outputs, cuda_y,
                                (size_t)B * (size_t)R * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy NVFP4 output") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled) {
        if (cuda_event_elapsed_ns(cuda_nvfp4_event_start,
                                  cuda_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            cuda_event_elapsed_ns(cuda_nvfp4_event_qdq,
                                  cuda_nvfp4_event_projection,
                                  &telemetry.projection_ns) != 0 ||
            cuda_nvfp4_d2h_timing(cuda_nvfp4_event_projection,
                                   &telemetry) != 0)
            return -1;
    }
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += (uint64_t)B;
    cuda_nvfp4_telemetry_publish(&telemetry, UINT64_C(2));
    return 0;
}

static int cuda_run_nvfp4_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    const int *rows, int common_rows, int C, int njobs) {
    size_t input_floats = 0, output_floats = 0;
    int max_groups = 0, max_batch = 0, max_rows = 0;
    int saw_trunk = 0, saw_expert = 0;
    uint64_t logical_jobs = 0;
    uint64_t h2d_start = 0;
    CudaNvfp4Telemetry telemetry = {0};
    if (!cuda_ready || !keys || !inputs || !outputs || !batches ||
        (!rows && common_rows < 1) || C < 16 || C > CUDA_MAX_C ||
        (C & 15) != 0 ||
        njobs < 1 || njobs > CUDA_MAX_BATCH_JOBS)
        return -1;
    for (int job = 0; job < njobs; job++) {
        CudaTensor *tensor = cuda_tensor(keys[job]);
        CudaResource *resource;
        CudaNvfp4Telemetry delta;
        int batch = batches[job];
        int R = rows ? rows[job] : common_rows;
        if (!tensor || tensor->bits != CUDA_FORMAT_NVFP4 ||
            R < 1 || tensor->rows != R || tensor->cols != C || batch < 1 ||
            !inputs[job] || !outputs[job] ||
            (size_t)batch > (CUDA_MAX_OUTPUTS - input_floats) / (size_t)C ||
            (size_t)batch > (CUDA_MAX_OUTPUTS - output_floats) / (size_t)R)
            goto miss;
        resource = cuda_resource(tensor->kind, tensor->resource_id);
        if (!resource || cuda_nvfp4_telemetry(R, C, batch, &delta) != 0)
            goto miss;
        /* The heterogeneous Q/DQ kernel writes only FP32 dequant output;
         * unlike the standalone path it does not materialize packed codes or
         * activation-scale planes. */
        delta.activation_qdq_bytes = delta.activation_input_bytes;
        if (cuda_nvfp4_telemetry_add(&telemetry, &delta) != 0) goto miss;
        CudaBatchDesc *descriptor = &cuda_host_desc[job];
        descriptor->resource = resource->device;
        descriptor->value_offset = tensor->value_offset;
        descriptor->scale_offset = tensor->scale_offset;
        descriptor->bias_offset = tensor->bias_offset;
        descriptor->aux_offset = tensor->aux_offset;
        descriptor->input_offset = (uint32_t)input_floats;
        descriptor->output_offset = (uint32_t)output_floats;
        descriptor->rows = R;
        descriptor->cols = C;
        descriptor->batch = batch;
        memcpy(cuda_host_x + input_floats, inputs[job],
               (size_t)batch * (size_t)C * sizeof(float));
        cuda_job_output_offset[job] = (uint32_t)output_floats;
        cuda_job_rows[job] = batch * R;
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
    if (cuda_nvfp4_timing_enabled) h2d_start = cuda_host_now_ns();
    if (cuda_success(cudaMemcpy(cuda_x, cuda_host_x,
                                input_floats * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy NVFP4 heterogeneous input") != 0 ||
        cuda_success(cudaMemcpy(cuda_device_desc, cuda_host_desc,
                                (size_t)njobs * sizeof(CudaBatchDesc),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy NVFP4 heterogeneous descriptors") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled) {
        telemetry.h2d_ns = cuda_host_now_ns() - h2d_start;
        if (cuda_success(cudaEventRecord(cuda_nvfp4_event_start),
                         "cudaEventRecord NVFP4 heterogeneous start") != 0)
            return -1;
    }
    cuda_nvfp4_heterogeneous_qdq<<<
        dim3((unsigned int)((max_groups + 255) / 256), 1u,
             (unsigned int)njobs), 256>>>(
        cuda_device_desc, njobs, cuda_x, cuda_nvfp4_xdq);
    if (cuda_success(cudaGetLastError(),
                     "NVFP4 heterogeneous QDQ launch") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled &&
        cuda_success(cudaEventRecord(cuda_nvfp4_event_qdq),
                     "cudaEventRecord NVFP4 heterogeneous QDQ") != 0)
        return -1;
    cuda_nvfp4_heterogeneous_warp<<<
        dim3((unsigned int)((max_rows + 3) / 4), (unsigned int)max_batch,
             (unsigned int)njobs), 128>>>(
        cuda_device_desc, njobs, cuda_nvfp4_xdq, cuda_y);
    if (cuda_success(cudaGetLastError(),
                     "NVFP4 heterogeneous projection launch") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled &&
        cuda_success(cudaEventRecord(cuda_nvfp4_event_projection),
                     "cudaEventRecord NVFP4 heterogeneous projection") != 0)
        return -1;
    if (cuda_success(cudaMemcpy(cuda_host_y, cuda_y,
                                output_floats * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy NVFP4 heterogeneous output") != 0)
        return -1;
    if (cuda_nvfp4_timing_enabled) {
        if (cuda_event_elapsed_ns(cuda_nvfp4_event_start,
                                  cuda_nvfp4_event_qdq,
                                  &telemetry.qdq_ns) != 0 ||
            cuda_event_elapsed_ns(cuda_nvfp4_event_qdq,
                                  cuda_nvfp4_event_projection,
                                  &telemetry.projection_ns) != 0 ||
            cuda_nvfp4_d2h_timing(cuda_nvfp4_event_projection,
                                   &telemetry) != 0)
            return -1;
    }
    for (int job = 0; job < njobs; job++)
        memcpy(outputs[job], cuda_host_y + cuda_job_output_offset[job],
               (size_t)cuda_job_rows[job] * sizeof(float));
    if (saw_trunk) cuda_stats.trunk_batches++;
    if (saw_expert) cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += logical_jobs;
    cuda_nvfp4_telemetry_publish(&telemetry, UINT64_C(2));
    return 0;

miss:
    if (cuda_mapped_only) cuda_stats.mapped_only_misses++;
    return -1;
}

extern "C" int salt_gpu_nvfp4_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    int R, int C, int njobs) {
    return cuda_run_nvfp4_batch(
        keys, inputs, outputs, batches, NULL, R, C, njobs);
}

extern "C" int salt_gpu_nvfp4_mixed_batch(
    const void *const *keys, const float *const *inputs,
    float *const *outputs, const int *batches,
    const int *rows, int C, int njobs) {
    if (!rows) return -1;
    return cuda_run_nvfp4_batch(
        keys, inputs, outputs, batches, rows, 0, C, njobs);
}

static int cuda_run_batch(const uint32_t *const *vals,
    const float *const *xs, float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs, int bits) {
    if (!vals || !xs || !ys || !ids || !Rj || C < 1 || njobs < 1)
        return -1;
    for (int job = 0; job < njobs; job++)
        if (cuda_run(ids[job], vals[job], bits, Rj[job], C,
                     xs[job], ys[job]) != 0)
            return -1;
    return 0;
}

static int cuda_run_heterogeneous_q4(
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
    if (!cuda_ready || !vals || !scales || !biases || !xs || !ys ||
        !ids || !Rj || C < 1 || C > CUDA_MAX_C || njobs < 1 ||
        njobs > CUDA_MAX_BATCH_JOBS) {
        if (getenv("SALT_GPU_DIAG"))
            fprintf(stderr, "gpu-cuda-hetero: arguments C=%d jobs=%d\n",
                    C, njobs);
        return -1;
    }
    for (int job = 0; job < njobs;) {
        CudaTensor *tensor = cuda_tensor(ids[job]);
        CudaResource *resource;
        int run = 1;
        int reuse_input;
        if (!tensor || tensor->bits != 4 || tensor->rows != Rj[job] ||
            tensor->cols != C || !vals[job] || !scales[job] ||
            !biases[job] || !xs[job] || !ys[job]) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-cuda-hetero: tensor job=%d tensor=%p R=%d C=%d\n",
                    job, (void *)tensor, Rj[job], C);
            goto miss;
        }
        resource = cuda_resource(tensor->kind, tensor->resource_id);
        if (!resource ||
            resource->host + tensor->value_offset !=
                (const unsigned char *)(const void *)vals[job] ||
            resource->host + tensor->scale_offset !=
                (const unsigned char *)(const void *)scales[job] ||
            resource->host + tensor->bias_offset !=
                (const unsigned char *)(const void *)biases[job]) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-cuda-hetero: resource job=%d kind=%u id=%u\n",
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
             (size_t)run > (CUDA_MAX_OUTPUTS - input_floats) / (size_t)C) ||
            (size_t)run > (CUDA_MAX_OUTPUTS - output_floats) /
                              (size_t)Rj[job] ||
            descriptor_count >= CUDA_MAX_BATCH_JOBS) {
            if (getenv("SALT_GPU_DIAG"))
                fprintf(stderr,
                    "gpu-cuda-hetero: capacity job=%d run=%d in=%zu out=%zu desc=%d\n",
                    job, run, input_floats, output_floats,
                    descriptor_count);
            return -1;
        }
        CudaBatchDesc *descriptor = &cuda_host_desc[descriptor_count++];
        descriptor->resource = resource->device;
        descriptor->value_offset = tensor->value_offset;
        descriptor->scale_offset = tensor->scale_offset;
        descriptor->bias_offset = tensor->bias_offset;
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
            memcpy(cuda_host_x + input_floats, xs[job],
                   (size_t)run * (size_t)C * sizeof(float));
            input_floats += (size_t)run * (size_t)C;
        }
        for (int item = 0; item < run; item++) {
            cuda_job_output_offset[job + item] =
                (uint32_t)(output_floats +
                           (size_t)item * (size_t)Rj[job]);
            cuda_job_rows[job + item] = Rj[job];
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
            "gpu-cuda-hetero: submit jobs=%d desc=%d C=%d rows=%d groups=%d in=%zu out=%zu\n",
            njobs, descriptor_count, C, max_rows, max_groups,
            input_floats, output_floats);
    if (cuda_success(cudaMemcpy(cuda_x, cuda_host_x,
                                input_floats * sizeof(float),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy heterogeneous input") != 0 ||
        cuda_success(cudaMemcpy(cuda_device_desc, cuda_host_desc,
                                (size_t)descriptor_count *
                                    sizeof(CudaBatchDesc),
                                cudaMemcpyHostToDevice),
                     "cudaMemcpy heterogeneous descriptors") != 0)
        return -1;
    if (all_single)
        cuda_q4_heterogeneous_warp<<<
            dim3((unsigned int)((max_rows + 3) / 4), 1u,
                 (unsigned int)descriptor_count), 128>>>(
            cuda_device_desc, descriptor_count, cuda_x, cuda_y,
            CudaBatchDesc{}, CudaBatchDesc{}, CudaBatchDesc{});
    else if (descriptor_count <= 3)
        cuda_q4_heterogeneous_warp_batch<<<
            dim3((unsigned int)max_rows, (unsigned int)max_warp_groups,
                 (unsigned int)descriptor_count), 1024>>>(
            cuda_device_desc, descriptor_count, cuda_x, cuda_y);
    else
        cuda_q4_heterogeneous<<<dim3((unsigned int)max_rows,
                                     (unsigned int)max_groups,
                                     (unsigned int)descriptor_count), 32>>>(
            cuda_device_desc, descriptor_count, cuda_x, cuda_y);
    if (cuda_success(cudaGetLastError(), "heterogeneous Q4 launch") != 0 ||
        cuda_success(cudaMemcpy(cuda_host_y, cuda_y,
                                output_floats * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy heterogeneous output") != 0)
        return -1;
    for (int job = 0; job < njobs; job++)
        memcpy(ys[job], cuda_host_y + cuda_job_output_offset[job],
               (size_t)cuda_job_rows[job] * sizeof(float));
    if (saw_trunk) cuda_stats.trunk_batches++;
    if (saw_expert) cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += (uint64_t)njobs;
    cuda_q4_hetero_logical_input_bytes += logical_input_floats * sizeof(float);
    cuda_q4_hetero_transfer_input_bytes +=
        (uint64_t)input_floats * sizeof(float);
    cuda_q4_hetero_output_bytes += (uint64_t)output_floats * sizeof(float);
    cuda_q4_hetero_descriptor_bytes +=
        (uint64_t)(uint32_t)descriptor_count * sizeof(CudaBatchDesc);
    cuda_q4_hetero_reused_inputs += reused_inputs;
    cuda_q4_hetero_kernel_launches++;
    cuda_q4_hetero_completion_fences++;
    return 0;

miss:
    if (cuda_mapped_only) cuda_stats.mapped_only_misses++;
    return -1;
}

extern "C" int salt_gpu_q4_batch(const uint32_t *const *vals,
    const uint16_t *const *scales, const uint16_t *const *biases,
    const float *const *xs, float *const *ys, const void *const *ids,
    const int *Rj, int C, int njobs) {
    return cuda_run_heterogeneous_q4(vals, scales, biases, xs, ys,
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
            return cuda_run_same_weight_batch(ids[0], vals[0], 8,
                                               Rj[0], C, njobs, xs, ys);
    }
    return cuda_run_batch(vals, xs, ys, ids, Rj, C, njobs, 8);
}

extern "C" void salt_gpu_set_mapped_only(int on) {
    cuda_mapped_only = on ? 1 : 0;
}
extern "C" int salt_gpu_mapped_only(void) { return cuda_mapped_only; }
extern "C" void salt_gpu_batch_stats_get(SaltGpuBatchStats *stats) {
    if (!stats) return;
    stats->arena_batches = cuda_stats.arena_batches;
    stats->trunk_batches = cuda_stats.trunk_batches;
    stats->expert_layer_batches = cuda_stats.expert_layer_batches;
    stats->mapped_only_misses = cuda_stats.mapped_only_misses;
    stats->direct_output_batches = cuda_stats.direct_output_batches;
    stats->direct_output_jobs = cuda_stats.direct_output_jobs;
}
extern "C" int salt_gpu_batch_stats_get_v2(
        SaltGpuBatchStatsV2 *stats, size_t stats_size) {
    if (!stats || stats_size != sizeof *stats) return -1;
    *stats = cuda_stats;
    return 0;
}

extern "C" int salt_gpu_resident_trunk_map(const void *base, size_t nbytes) {
    return salt_gpu_weight_resource_bind(SALT_GPU_RESOURCE_TRUNK, 0,
                                         base, nbytes);
}
extern "C" int salt_gpu_resident_trunk_unmap(void) {
    return salt_gpu_weight_resource_unbind(SALT_GPU_RESOURCE_TRUNK, 0);
}
extern "C" int salt_gpu_expert_resource_bind(uint32_t id,
    const void *base, size_t nbytes) {
    return salt_gpu_weight_resource_bind(SALT_GPU_RESOURCE_EXPERT_LAYER,
                                         id, base, nbytes);
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
    for (int i = 0; i < CUDA_MAX_RESOURCES; i++) {
        uint64_t offset = 0;
        if (cuda_resources[i].used &&
            cuda_resources[i].kind == SALT_GPU_RESOURCE_EXPERT_LAYER &&
            cuda_range(&cuda_resources[i], vals, 1, &offset) == 0)
            return salt_gpu_expert_resource_slot(cuda_resources[i].id,
                key, vals, scales, biases, R, C);
    }
    return -1;
}
extern "C" int salt_gpu_expert_resource_unbind(uint32_t id) {
    return salt_gpu_weight_resource_unbind(SALT_GPU_RESOURCE_EXPERT_LAYER, id);
}
extern "C" int salt_gpu_resident_pool_map(const void *base, size_t nbytes) {
    return salt_gpu_expert_resource_bind(0, base, nbytes);
}

extern "C" int salt_gpu_proj_batch(const uint32_t *vals,
    const uint16_t *scales, const uint16_t *biases, int R, int C, int B,
    const float *const *xs, float *const *ys, const void *id) {
    (void)scales; (void)biases;
    return cuda_run_same_weight_batch(id, vals, 4, R, C, B, xs, ys);
}

extern "C" int salt_gpu_shared_buffer_alloc(SaltGpuSharedBuffer *buffer,
                                              size_t nbytes) {
    void *host = NULL, *device = NULL;
    if (!buffer || !cuda_ready || nbytes < 1) return -1;
    memset(buffer, 0, sizeof *buffer);
    if (
        cuda_success(cudaHostAlloc(&host, nbytes, cudaHostAllocMapped),
                     "cudaHostAlloc shared") != 0 ||
        cuda_success(cudaHostGetDevicePointer(&device, host, 0),
                     "cudaHostGetDevicePointer shared") != 0) {
        if (host) cudaFreeHost(host);
        return -1;
    }
    buffer->contents = host;
    buffer->nbytes = nbytes;
    buffer->backend = device;
    return 0;
}
static int cuda_exact_view(const SaltGpuSharedBuffer *buffer,
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
    if (!cuda_ready || !cells || count < 1 || !cuda_attention_status ||
        cuda_success(cudaMemset(cuda_attention_status, 0, sizeof(int)),
                     "cudaMemset exact-cell status") != 0)
        return -1;
    for (int index = 0; index < count; index++) {
        const SaltGpuExactCell *cell = &cells[index];
        void *views[8] = {0};
        if (cell->n == 0) return -1;
        switch (cell->kind) {
        case SALT_GPU_EXACT_RMSNORM:
            if (cell->aux > 1u ||
                cuda_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                cuda_exact_view(cell->views[1], cell->offsets[1],
                    cell->aux ? cell->n : 1u, &views[1]) != 0 ||
                cuda_exact_view(cell->views[2], cell->offsets[2],
                    cell->n, &views[2]) != 0)
                return -1;
            cuda_text_rmsnorm_rows<<<1, 256>>>(
                (const float *)views[0], cell->n, (const float *)views[1],
                (float *)views[2], cell->n, 1u, cell->n, cell->eps,
                (int)cell->aux, cuda_attention_status);
            break;
        case SALT_GPU_EXACT_TOPK:
            if (cell->aux == 0 || cell->aux > cell->n ||
                cuda_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                cuda_exact_view(cell->views[1], cell->offsets[1],
                    cell->n, &views[1]) != 0 ||
                cuda_exact_view(cell->views[2], cell->offsets[2],
                    cell->aux, &views[2]) != 0 ||
                cuda_exact_view(cell->views[3], cell->offsets[3],
                    cell->aux, &views[3]) != 0)
                return -1;
            cuda_text_topk_rows<<<1, 1>>>(
                (const float *)views[0], (const float *)views[1],
                (int32_t *)views[2], (float *)views[3], 1u,
                cell->n, cell->aux, cuda_attention_status);
            break;
        case SALT_GPU_EXACT_RESIDUAL_POSTNORM:
            for (int view = 0; view < 4; view++)
                if (cuda_exact_view(cell->views[view], cell->offsets[view],
                        cell->n, &views[view]) != 0)
                    return -1;
            cuda_text_residual_postnorm_rows<<<1, 256>>>(
                (const float *)views[0], (float *)views[1],
                (const float *)views[2], (float *)views[3],
                1u, cell->n, cell->eps, cuda_attention_status);
            break;
        case SALT_GPU_EXACT_PARALLEL_COMBINE:
            for (int view = 0; view < 8; view++) {
                size_t elements = view == 6 ? (size_t)cell->n * 2u : cell->n;
                if (elements < cell->n ||
                    cuda_exact_view(cell->views[view], cell->offsets[view],
                        elements, &views[view]) != 0)
                    return -1;
            }
            cuda_text_parallel_combine_rows<<<1, 256>>>(
                (const float *)views[0], (const float *)views[1],
                (const float *)views[2], (const float *)views[3],
                (const float *)views[4], (const float *)views[5],
                (float *)views[6], (float *)views[6] + cell->n,
                (float *)views[7], 1u, cell->n, cell->eps,
                cell->scalar, NULL, cuda_attention_status);
            break;
        case SALT_GPU_EXACT_SOFTCAP:
            if (!(cell->scalar > 0.0f) || !isfinite(cell->scalar) ||
                cuda_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0)
                return -1;
            cuda_text_softcap<<<(cell->n + 255u) / 256u, 256>>>(
                (float *)views[0], cell->n, cell->scalar,
                cuda_attention_status);
            break;
        case SALT_GPU_EXACT_ROUTER_INPUT:
            if (cuda_exact_view(cell->views[0], cell->offsets[0],
                    cell->n, &views[0]) != 0 ||
                cuda_exact_view(cell->views[1], cell->offsets[1],
                    cell->n, &views[1]) != 0 ||
                cuda_exact_view(cell->views[2], cell->offsets[2],
                    cell->n, &views[2]) != 0)
                return -1;
            cuda_text_router_input_rows<<<1, 256>>>(
                (const float *)views[0], (const float *)views[1],
                (float *)views[2], 1u, cell->n, cell->eps,
                cell->scalar, cuda_attention_status);
            break;
        default:
            return -1;
        }
        if (cuda_success(cudaGetLastError(), "exact-cell launch") != 0)
            return -1;
    }
    if (cuda_success(cudaMemcpy(&host_status, cuda_attention_status,
            sizeof host_status, cudaMemcpyDeviceToHost),
            "cudaMemcpy exact-cell status") != 0)
        return -1;
    return host_status ? -1 : 0;
}
extern "C" int salt_gpu_shared_buffer_free(SaltGpuSharedBuffer *buffer) {
    int rc;
    if (!buffer) return -1;
    if (!buffer->contents && !buffer->backend && buffer->nbytes == 0) return 0;
    if (!buffer->contents || !buffer->backend || buffer->nbytes < 1 ||
        cuda_success(cudaDeviceSynchronize(),
                     "cudaDeviceSynchronize shared free") != 0)
        return -1;
    if (buffer->contents == cuda_attention_kv_host) {
        if (buffer->backend != cuda_attention_kv ||
            buffer->nbytes != cuda_attention_kv_nbytes)
            return -1;
        cuda_attention_kv = NULL;
        cuda_attention_kv_nbytes = 0;
        cuda_attention_kv_host = NULL;
        cuda_attention_kv_sync_count = 0;
        memset(cuda_attention_kv_sync, 0, sizeof cuda_attention_kv_sync);
        memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    }
    rc = cuda_success(cudaFreeHost(buffer->contents), "cudaFreeHost shared");
    memset(buffer, 0, sizeof *buffer);
    return rc;
}

static int cuda_shared_host_float_range(const SaltGpuSharedBuffer *buffer,
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
    if (!cuda_ready || !kv || !kv->contents || !kv->backend || kv->nbytes < 1)
        return -1;
    if (cuda_attention_kv)
        return cuda_attention_kv_host == kv->contents &&
               cuda_attention_kv == kv->backend &&
               cuda_attention_kv_nbytes == kv->nbytes ? 0 : -1;
    /* Same canonical storage as CPU and compiled text execution. */
    cuda_attention_kv = (float *)kv->backend;
    cuda_attention_kv_host = kv->contents;
    cuda_attention_kv_nbytes = kv->nbytes;
    cuda_attention_kv_sync_count = 0;
    memset(cuda_attention_kv_sync, 0, sizeof cuda_attention_kv_sync);
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    return 0;
}

static CudaAttentionKvSync *cuda_attention_sync_slot(
    size_t key_offset, size_t value_offset, int kv_stride) {
    size_t row_capacity;
    if (kv_stride < 1 || value_offset <= key_offset ||
        (value_offset - key_offset) % (size_t)kv_stride != 0)
        return NULL;
    row_capacity = (value_offset - key_offset) / (size_t)kv_stride;
    if (row_capacity == 0 || row_capacity > (size_t)INT_MAX ||
        value_offset > cuda_attention_kv_nbytes / sizeof(float) ||
        value_offset - key_offset >
            cuda_attention_kv_nbytes / sizeof(float) - value_offset)
        return NULL;
    for (int i = 0; i < cuda_attention_kv_sync_count; i++)
        if (cuda_attention_kv_sync[i].key_offset == key_offset &&
            cuda_attention_kv_sync[i].value_offset == value_offset)
            return cuda_attention_kv_sync[i].kv_stride == kv_stride &&
                   cuda_attention_kv_sync[i].row_capacity == row_capacity
                ? &cuda_attention_kv_sync[i] : NULL;
    if (cuda_attention_kv_sync_count >=
            (int)(sizeof cuda_attention_kv_sync /
                  sizeof cuda_attention_kv_sync[0]))
        return NULL;
    CudaAttentionKvSync *sync =
        &cuda_attention_kv_sync[cuda_attention_kv_sync_count++];
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
    return cuda_ready && cuda_attention_kv &&
        cuda_attention_sync_slot(
            key_float_offset, value_float_offset, kv_stride) ? 0 : -1;
}

static int cuda_attention_state_range(
        const SaltGpuSharedBuffer *kv, const CudaAttentionKvSync *sync,
        int position, size_t *offset_count) {
    size_t capacity, count;
    if (!kv || !sync || !offset_count || kv->contents != cuda_attention_kv_host ||
        kv->backend != cuda_attention_kv ||
        kv->nbytes != cuda_attention_kv_nbytes || position < 0 ||
        sync->kv_stride < 1 || sync->row_capacity == 0)
        return -1;
    capacity = kv->nbytes / sizeof(float);
    count = (size_t)position < sync->row_capacity
        ? (size_t)position : sync->row_capacity;
    if (count > SIZE_MAX / (size_t)sync->kv_stride) return -1;
    count *= (size_t)sync->kv_stride;
    if (sync->key_offset > capacity || count > capacity - sync->key_offset ||
        sync->value_offset > capacity || count > capacity - sync->value_offset)
        return -1;
    *offset_count = count;
    return 0;
}

extern "C" int salt_gpu_attention_materialize(
        SaltGpuSharedBuffer *kv, int position, uint64_t *copied_bytes) {
    if (copied_bytes) *copied_bytes = 0;
    if (!cuda_ready || !kv || !copied_bytes || position < 0 ||
        cuda_attention_kv_sync_count < 1 ||
        cuda_success(cudaDeviceSynchronize(),
                     "cudaDeviceSynchronize shared KV materialize") != 0)
        return -1;
    for (int i = 0; i < cuda_attention_kv_sync_count; i++) {
        CudaAttentionKvSync *sync = &cuda_attention_kv_sync[i];
        size_t count;
        if (cuda_attention_state_range(kv, sync, position, &count) != 0)
            return -1;
        /* CPU and both GPU executors already share the canonical rows. */
        sync->device_positions = position;
        sync->host_positions = position;
    }
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    return 0;
}

extern "C" int salt_gpu_attention_import(
        SaltGpuSharedBuffer *kv, int position, uint64_t *copied_bytes) {
    if (copied_bytes) *copied_bytes = 0;
    if (!cuda_ready || !kv || !copied_bytes || position < 0 ||
        cuda_attention_kv_sync_count < 1 ||
        cuda_success(cudaDeviceSynchronize(),
                     "cudaDeviceSynchronize shared KV import") != 0)
        return -1;
    for (int i = 0; i < cuda_attention_kv_sync_count; i++) {
        CudaAttentionKvSync *sync = &cuda_attention_kv_sync[i];
        size_t count;
        if (cuda_attention_state_range(kv, sync, position, &count) != 0)
            return -1;
        sync->device_positions = position;
        sync->host_positions = position;
    }
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    return 0;
}

extern "C" int salt_gpu_attention_rewind(int position) {
    if (!cuda_ready || position < 0 || cuda_attention_kv_sync_count < 1)
        return -1;
    for (int i = 0; i < cuda_attention_kv_sync_count; i++)
        if (position > cuda_attention_kv_sync[i].device_positions)
            return -1;
    for (int i = 0; i < cuda_attention_kv_sync_count; i++) {
        cuda_attention_kv_sync[i].device_positions = position;
        if (cuda_attention_kv_sync[i].host_positions > position)
            cuda_attention_kv_sync[i].host_positions = position;
    }
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
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
    CudaAttentionKvSync *sync;
    uint64_t query_count, new_kv_count, cache_count, factor_count;
    uint64_t metadata_offset, metadata_count;
    size_t cache_rows;
    int deferred_publish;
    memset(&cuda_attention_stage, 0, sizeof cuda_attention_stage);
    if (!cuda_ready || n_heads < 1 || n_kv_heads < 1 ||
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
    sync = cuda_attention_sync_slot(
        key_cache_float_offset, value_cache_float_offset, kv_stride);
    if (!sync) return -1;
    /* Any multirow overwrite after ring population can destroy history still
     * needed by an earlier query, even without crossing physical index zero. */
    deferred_publish = batch > 1 &&
        (size_t)(start_position + batch) > sync->row_capacity;
    cache_rows = (size_t)(start_position + batch) < sync->row_capacity
        ? (size_t)(start_position + batch) : sync->row_capacity;
    cache_count = (uint64_t)cache_rows * (uint32_t)kv_stride;
    factor_count = (uint64_t)(uint32_t)batch * (uint32_t)rope_pairs;
    metadata_offset = query_count;
    metadata_count = 2u * (uint32_t)head_dim + 2u * factor_count;
    if (query_count > CUDA_MAX_OUTPUTS ||
        new_kv_count > CUDA_MAX_OUTPUTS / 2u ||
        metadata_offset > CUDA_MAX_OUTPUTS ||
        metadata_count > CUDA_MAX_OUTPUTS - metadata_offset ||
        cache_count > SIZE_MAX || factor_count > SIZE_MAX ||
        !cuda_attention_kv || cuda_attention_kv_host != kv->contents ||
        cuda_attention_kv_nbytes != kv->nbytes ||
        cuda_shared_host_float_range(queries, query_float_offset,
            (size_t)query_count, &host_queries) != 0 ||
        cuda_shared_host_float_range(keys, key_float_offset,
            (size_t)new_kv_count, &host_keys) != 0 ||
        cuda_shared_host_float_range(values, value_float_offset,
            (size_t)new_kv_count, &host_values) != 0 ||
        cuda_shared_host_float_range(kv, key_cache_float_offset,
            (size_t)cache_count, &host_key_cache) != 0 ||
        cuda_shared_host_float_range(kv, value_cache_float_offset,
            (size_t)cache_count, &host_value_cache) != 0)
        return -1;
    /* Prefix rows are already in the shared seat, including CPU-written rows. */
    sync->host_positions = start_position;
    float *device_q_weight = cuda_x + metadata_offset;
    float *device_k_weight = device_q_weight + head_dim;
    float *device_cosines = device_k_weight + head_dim;
    float *device_sines = device_cosines + factor_count;
    if (cuda_success(cudaMemcpy(cuda_x, host_queries,
            (size_t)query_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform queries") != 0 ||
        cuda_success(cudaMemcpy(cuda_y, host_keys,
            (size_t)new_kv_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform keys") != 0 ||
        cuda_success(cudaMemcpy(cuda_y + new_kv_count, host_values,
            (size_t)new_kv_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform values") != 0 ||
        cuda_success(cudaMemcpy(device_q_weight, q_weight,
            (size_t)head_dim * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform q weight") != 0 ||
        cuda_success(cudaMemcpy(device_k_weight, k_weight,
            (size_t)head_dim * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform k weight") != 0 ||
        cuda_success(cudaMemcpy(device_cosines, cosines,
            (size_t)factor_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform cosines") != 0 ||
        cuda_success(cudaMemcpy(device_sines, sines,
            (size_t)factor_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy transform sines") != 0 ||
        cuda_success(cudaMemset(cuda_attention_status, 0, sizeof(int)),
            "cudaMemset transform status") != 0)
        return -1;
    unsigned int tasks = (unsigned int)((uint64_t)(uint32_t)batch *
        (uint32_t)(n_heads + 2 * n_kv_heads));
    cuda_attention_transform_exact<<<tasks, 256>>>(
        n_heads, n_kv_heads, head_dim, rope_dim, start_position, batch,
        query_stride, kv_stride, (int)sync->row_capacity,
        !deferred_publish, eps, cuda_x, cuda_y,
        cuda_y + new_kv_count, device_q_weight, device_k_weight,
        device_cosines, device_sines, rope_pairs,
        cuda_attention_kv + key_cache_float_offset,
        cuda_attention_kv + value_cache_float_offset, cuda_attention_status);
    if (cuda_success(cudaGetLastError(), "attention transform launch") != 0)
        return -1;
    if (cuda_pageable_mmap) {
        int failure = 0;
        if (cuda_success(cudaMemcpy(&failure, cuda_attention_status,
                sizeof failure, cudaMemcpyDeviceToHost),
                "cudaMemcpy pageable transform status") != 0 || failure)
            return -1;
    }
    sync->device_positions = start_position + batch;
    cuda_attention_stage.query_host = queries->contents;
    cuda_attention_stage.query_offset = query_float_offset;
    cuda_attention_stage.key_offset = key_cache_float_offset;
    cuda_attention_stage.value_offset = value_cache_float_offset;
    cuda_attention_stage.start = start_position;
    cuda_attention_stage.batch = batch;
    cuda_attention_stage.query_stride = query_stride;
    cuda_attention_stage.kv_stride = kv_stride;
    cuda_attention_stage.deferred_publish = deferred_publish;
    cuda_attention_stage.valid = 1;
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
    CudaAttentionKvSync *sync = NULL;
    int staged, deferred_publish;
    int failure = 0;
    size_t kv_rows, score_rows, score_bytes;
    if (!cuda_ready || (full_attention != 0 && full_attention != 1) ||
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
    sync = cuda_attention_sync_slot(
        key_float_offset, value_float_offset, kv_stride);
    if (!sync ||
        (full_attention && (size_t)(start_position + batch) > sync->row_capacity) ||
        (!full_attention && (size_t)window > sync->row_capacity))
        return -1;
    kv_rows = (size_t)(start_position + batch) < sync->row_capacity
        ? (size_t)(start_position + batch) : sync->row_capacity;
    kv_count = (uint64_t)kv_rows * (uint32_t)kv_stride;
    staged = cuda_attention_stage.valid &&
        cuda_attention_stage.query_host == queries->contents &&
        cuda_attention_stage.query_offset == query_float_offset &&
        cuda_attention_stage.key_offset == key_float_offset &&
        cuda_attention_stage.value_offset == value_float_offset &&
        cuda_attention_stage.start == start_position &&
        cuda_attention_stage.batch == batch &&
        cuda_attention_stage.query_stride == query_stride &&
        cuda_attention_stage.kv_stride == kv_stride;
    deferred_publish = staged && cuda_attention_stage.deferred_publish;
    cuda_attention_stage.valid = 0;
    output_count = query_count;
    if ((!staged && batch > 1 &&
         (size_t)(start_position + batch) > sync->row_capacity) ||
        tasks > UINT32_MAX || query_count > SIZE_MAX || kv_count > SIZE_MAX ||
        new_kv_count > CUDA_MAX_OUTPUTS / 2u || output_count > SIZE_MAX ||
        query_count > CUDA_MAX_OUTPUTS || output_count > CUDA_MAX_OUTPUTS ||
        (size_t)(start_position + batch) > SIZE_MAX / sizeof(float))
        return -1;
    score_rows = (size_t)(start_position + batch);
    if (!full_attention && score_rows > (size_t)window)
        score_rows = (size_t)window;
    score_bytes = score_rows * sizeof(float);
    if (!cuda_attention_kv ||
        cuda_attention_kv_host != kv->contents ||
        cuda_attention_kv_nbytes != kv->nbytes ||
        cuda_shared_host_float_range(queries, query_float_offset,
            (size_t)query_count, &host_queries) != 0 ||
        cuda_shared_host_float_range(kv, key_float_offset,
            (size_t)kv_count, &host_keys) != 0 ||
        cuda_shared_host_float_range(kv, value_float_offset,
            (size_t)kv_count, &host_values) != 0 ||
        cuda_shared_host_float_range(outputs, output_float_offset,
            (size_t)output_count, &host_outputs) != 0)
        return -1;
    /* The input seat is startup-owned; the query span remains live for every
     * score wave. Reject an unrepresentable row before mutating device state. */
    size_t score_wave = (size_t)tasks;
    float *global_scores = NULL;
    if (score_bytes > 48u * 1024u) {
        if (!full_attention || query_count >= CUDA_MAX_OUTPUTS ||
            score_rows > (CUDA_MAX_OUTPUTS - (size_t)query_count))
            return -1;
        global_scores = cuda_x + query_count;
        score_wave = (CUDA_MAX_OUTPUTS - (size_t)query_count) / score_rows;
    }
    if (!staged &&
        cuda_success(cudaMemcpy(cuda_x, host_queries,
            (size_t)query_count * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy attention queries") != 0)
        return -1;
    if (!staged &&
        cuda_success(cudaMemset(cuda_attention_status, 0, sizeof(int)),
                     "cudaMemset attention status") != 0)
        return -1;
    if (!staged)
        sync->host_positions = start_position + batch;
    sync->device_positions = start_position + batch;
    /* Larger full-attention contexts use the unused tail of the input seat
     * allocated at startup. Each wave owns disjoint score rows and writes the
     * same canonical output as the shared-memory path; no scores are truncated. */
    for (uint64_t first = 0; first < tasks; first += score_wave) {
        unsigned int wave = (unsigned int)((tasks - first) < score_wave
            ? (tasks - first) : score_wave);
        cuda_attention_exact<<<wave, 256, global_scores ? 0 : score_bytes>>>(
            full_attention, n_heads, n_kv_heads, head_dim, window,
            cuda_x, cuda_attention_kv + key_float_offset,
            cuda_attention_kv + value_float_offset, start_position, batch,
            query_stride, kv_stride, (int)sync->row_capacity,
            deferred_publish ? cuda_y : NULL,
            deferred_publish ? cuda_y + new_kv_count : NULL,
            start_position, deferred_publish ? batch : 0,
            deferred_publish ? cuda_x : cuda_y, cuda_attention_status,
            global_scores, score_rows, (uint32_t)first);
        if (cuda_success(cudaGetLastError(), "attention launch") != 0)
            return -1;
    }
    if (deferred_publish) {
        cuda_attention_publish_exact<<<
            (unsigned int)((new_kv_count + 255u) / 256u), 256>>>(
            cuda_y, cuda_y + new_kv_count,
            cuda_attention_kv + key_float_offset,
            cuda_attention_kv + value_float_offset,
            start_position, batch, kv_stride, (int)sync->row_capacity,
            cuda_attention_status);
        if (cuda_success(cudaGetLastError(),
                "attention ring publish launch") != 0)
            return -1;
    }
    if (cuda_success(cudaMemcpy(host_outputs,
                                deferred_publish ? cuda_x : cuda_y,
                                (size_t)output_count * sizeof(float),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy attention output") != 0 ||
        cuda_success(cudaMemcpy(&failure, cuda_attention_status, sizeof failure,
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpy attention status") != 0)
        return -1;
    if (!failure) sync->host_positions = sync->device_positions;
    return failure ? -1 : 0;
}

static int cuda_moe_descriptor(const uint32_t *vals,
                               const uint16_t *scales,
                               const uint16_t *biases, const void *id,
                               int rows, int cols, int batch, int expected_slot,
                               uint64_t expected_logical_resource_id,
                               int require_selected,
                               uint32_t input_offset, uint32_t output_offset,
                               CudaBatchDesc *descriptor) {
    CudaTensor *tensor = require_selected ? NULL : cuda_tensor(id);
    CudaResource *resource = NULL;
    CudaSelectedResource *selected = NULL;
    size_t vbytes, sbytes;
    uint64_t voff = 0, soff = 0, boff = 0;
    if (!vals || !scales || !biases || !id || !descriptor ||
        rows < 1 || cols < 1 || batch < 1 ||
        cuda_tensor_sizes(4, rows, cols, &vbytes, &sbytes) != 0)
        return -1;
    if (tensor) {
        if (tensor->bits != 4 || tensor->rows != rows || tensor->cols != cols)
            return -1;
        resource = cuda_resource(tensor->kind, tensor->resource_id);
        if (!resource ||
            resource->host + tensor->value_offset !=
                (const unsigned char *)(const void *)vals ||
            resource->host + tensor->scale_offset !=
                (const unsigned char *)(const void *)scales ||
            resource->host + tensor->bias_offset !=
                (const unsigned char *)(const void *)biases)
            return -1;
        voff = tensor->value_offset;
        soff = tensor->scale_offset;
        boff = tensor->bias_offset;
    } else {
        uintptr_t payload_base, payload_end, v, s, b, identity;
        if (!cuda_selected_resources ||
            !cuda_selected_logical_slots || expected_slot < 0 ||
            expected_slot >= cuda_selected_capacity)
            return -1;
        selected = &cuda_selected_resources[expected_slot];
        if (!selected->active || !selected->device_payload ||
            selected->logical_resource_id >=
                (uint64_t)(uint32_t)cuda_selected_logical_capacity ||
            selected->logical_resource_id != expected_logical_resource_id ||
            cuda_selected_logical_slots[selected->logical_resource_id] !=
                expected_slot)
            return -1;
        payload_base = (uintptr_t)(const void *)selected->payload;
        payload_end = (uintptr_t)(const void *)selected->base + selected->nbytes;
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
    descriptor->resource = selected ? selected->device_payload : resource->device;
    descriptor->value_offset = voff;
    descriptor->scale_offset = soff;
    descriptor->bias_offset = boff;
    descriptor->input_offset = input_offset;
    descriptor->output_offset = output_offset;
    descriptor->rows = rows;
    descriptor->cols = cols;
    descriptor->batch = batch;
    return 0;
}

static int cuda_q4_moe_chain_impl(const SaltGpuMoeExpert *experts,
    int count, int hidden, int routed, const float *inputs, float *outputs,
    float *gate_scratch, float *up_scratch, size_t scratch_float_capacity,
    int require_selected) {
    CudaBatchDesc activation[128];
    uint64_t total_rows = 0, input_floats, gate_up_floats, output_floats;
    uint64_t gate_up_cursor = 0, chain_cursor = 0;
    int max_group = 0, max_gate_groups = 0, max_down_groups = 0;
    if (!cuda_ready || !experts || !inputs || !outputs || !gate_scratch ||
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
    if (total_rows > UINT32_MAX || input_floats > CUDA_MAX_OUTPUTS ||
        gate_up_floats > CUDA_MAX_OUTPUTS || output_floats > CUDA_MAX_OUTPUTS)
        return -1;
    if (cuda_success(cudaMemcpy(cuda_x, inputs,
            (size_t)input_floats * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy MoE inputs") != 0)
        return -1;
    if (count == 1) {
        CudaBatchDesc gate, up, down;
        int group = experts[0].group;
        size_t elements = (size_t)group * (size_t)routed;
        if (cuda_moe_descriptor(experts[0].gate_vals,
                experts[0].gate_scales, experts[0].gate_biases,
                experts[0].gate_id, routed, hidden, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                require_selected, 0, 0, &gate) != 0 ||
            cuda_moe_descriptor(experts[0].up_vals,
                experts[0].up_scales, experts[0].up_biases,
                experts[0].up_id, routed, hidden, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                require_selected, 0, (uint32_t)elements, &up) != 0 ||
            cuda_moe_descriptor(experts[0].down_vals,
                experts[0].down_scales, experts[0].down_biases,
                experts[0].down_id, hidden, routed, group,
                experts[0].resource_slot, experts[0].logical_resource_id,
                require_selected, 0, 0, &down) != 0)
            return -1;
        if (group >= 8) {
            cuda_q4_warp_batch<<<(unsigned int)routed, 1024>>>(
                gate.resource, gate.value_offset, gate.scale_offset,
                gate.bias_offset, cuda_x, cuda_y, routed, hidden, group);
            cuda_q4_warp_batch<<<(unsigned int)routed, 1024>>>(
                up.resource, up.value_offset, up.scale_offset,
                up.bias_offset, cuda_x, cuda_y + elements,
                routed, hidden, group);
        } else {
            cuda_q4_warp_exact<<<dim3((unsigned int)((routed + 3) / 4),
                                       (unsigned int)group), 128>>>(
                gate.resource, gate.value_offset, gate.scale_offset,
                gate.bias_offset, cuda_x, cuda_y, routed, hidden);
            cuda_q4_warp_exact<<<dim3((unsigned int)((routed + 3) / 4),
                                       (unsigned int)group), 128>>>(
                up.resource, up.value_offset, up.scale_offset,
                up.bias_offset, cuda_x, cuda_y + elements, routed, hidden);
        }
        cuda_moe_activate_contiguous<<<
            (unsigned int)((elements + 255u) / 256u), 256>>>(
                cuda_y, cuda_y + elements, cuda_x, elements);
        if (group >= 8)
            cuda_q4_warp_batch<<<(unsigned int)hidden, 1024>>>(
                down.resource, down.value_offset, down.scale_offset,
                down.bias_offset, cuda_x, cuda_y, hidden, routed, group);
        else
            cuda_q4_warp_exact<<<dim3((unsigned int)((hidden + 3) / 4),
                                       (unsigned int)group), 128>>>(
                down.resource, down.value_offset, down.scale_offset,
                down.bias_offset, cuda_x, cuda_y, hidden, routed);
        if (cuda_success(cudaGetLastError(), "dense FFN chain launch") != 0 ||
            cuda_success(cudaMemcpy(outputs, cuda_y,
                (size_t)output_floats * sizeof(float), cudaMemcpyDeviceToHost),
                "cudaMemcpy dense FFN outputs") != 0)
            return -1;
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
        if (cuda_moe_descriptor(experts[i].gate_vals,
                experts[i].gate_scales, experts[i].gate_biases,
                experts[i].gate_id, routed, hidden, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                require_selected, input_offset, gate_offset,
                &cuda_host_desc[2 * i]) != 0 ||
            cuda_moe_descriptor(experts[i].up_vals,
                experts[i].up_scales, experts[i].up_biases,
                experts[i].up_id, routed, hidden, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                require_selected, input_offset, up_offset,
                &cuda_host_desc[2 * i + 1]) != 0)
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
    if (cuda_success(cudaMemcpy(cuda_device_desc, cuda_host_desc,
            (size_t)(2 * count) * sizeof(CudaBatchDesc),
            cudaMemcpyHostToDevice), "cudaMemcpy MoE gate/up descriptors") != 0)
        return -1;
    cuda_q4_heterogeneous<<<dim3((unsigned int)routed,
        (unsigned int)max_gate_groups, (unsigned int)(2 * count)), 32>>>(
        cuda_device_desc, 2 * count, cuda_x, cuda_y);
    if (cuda_success(cudaGetLastError(), "MoE gate/up launch") != 0 ||
        cuda_success(cudaMemcpy(cuda_device_desc, activation,
            (size_t)count * sizeof(CudaBatchDesc), cudaMemcpyHostToDevice),
            "cudaMemcpy MoE activation descriptors") != 0)
        return -1;
    cuda_moe_activate<<<dim3((unsigned int)(((uint64_t)max_group *
        (uint32_t)routed + 255u) / 256u), 1u, (unsigned int)count), 256>>>(
        cuda_device_desc, count, cuda_y, cuda_x);
    if (cuda_success(cudaGetLastError(), "MoE activation launch") != 0)
        return -1;
    row_cursor = 0;
    chain_cursor = 0;
    for (int i = 0; i < count; i++) {
        uint64_t elements = (uint64_t)(uint32_t)experts[i].group *
            (uint32_t)routed;
        uint32_t output_offset = (uint32_t)(row_cursor * (uint32_t)hidden);
        if (cuda_moe_descriptor(experts[i].down_vals,
                experts[i].down_scales, experts[i].down_biases,
                experts[i].down_id, hidden, routed, experts[i].group,
                experts[i].resource_slot, experts[i].logical_resource_id,
                require_selected, (uint32_t)chain_cursor, output_offset,
                &cuda_host_desc[i]) != 0)
            return -1;
        chain_cursor += elements;
        row_cursor += (uint32_t)experts[i].group;
        int groups = (experts[i].group + 31) / 32;
        if (groups > max_down_groups) max_down_groups = groups;
    }
    if (cuda_success(cudaMemcpy(cuda_device_desc, cuda_host_desc,
            (size_t)count * sizeof(CudaBatchDesc), cudaMemcpyHostToDevice),
            "cudaMemcpy MoE down descriptors") != 0)
        return -1;
    cuda_q4_heterogeneous<<<dim3((unsigned int)hidden,
        (unsigned int)max_down_groups, (unsigned int)count), 32>>>(
        cuda_device_desc, count, cuda_x, cuda_y);
    if (cuda_success(cudaGetLastError(), "MoE down launch") != 0 ||
        cuda_success(cudaMemcpy(outputs, cuda_y,
            (size_t)output_floats * sizeof(float), cudaMemcpyDeviceToHost),
            "cudaMemcpy MoE outputs") != 0)
        return -1;
    return 0;
}

extern "C" int salt_gpu_q4_moe_chain(const SaltGpuMoeExpert *experts,
    int count, int hidden, int routed, const float *inputs, float *outputs,
    float *gate_scratch, float *up_scratch, size_t scratch_float_capacity) {
    return cuda_q4_moe_chain_impl(experts, count, hidden, routed,
        inputs, outputs, gate_scratch, up_scratch, scratch_float_capacity, 0);
}

extern "C" int salt_gpu_q4_moe_chain_selected(
    const SaltGpuMoeExpert *experts, int count, int hidden, int routed,
    const float *inputs, float *outputs, float *gate_scratch,
    float *up_scratch, size_t scratch_float_capacity) {
    return cuda_q4_moe_chain_impl(experts, count, hidden, routed,
        inputs, outputs, gate_scratch, up_scratch, scratch_float_capacity, 1);
}

extern "C" int salt_gpu_selected_resources_prepare(
    int capacity, int logical_capacity) {
    if (!cuda_ready || capacity < 1 ||
        logical_capacity < 1 || logical_capacity > INT_MAX)
        return -1;
    if (cuda_selected_resources)
        return capacity == cuda_selected_capacity &&
            logical_capacity == cuda_selected_logical_capacity ? 0 : -1;
    cuda_selected_resources = (CudaSelectedResource *)calloc(
        (size_t)capacity, sizeof *cuda_selected_resources);
    cuda_selected_logical_slots = (int *)malloc(
        (size_t)logical_capacity * sizeof *cuda_selected_logical_slots);
    if (!cuda_selected_resources || !cuda_selected_logical_slots) {
        free(cuda_selected_logical_slots);
        free(cuda_selected_resources);
        cuda_selected_logical_slots = NULL;
        cuda_selected_resources = NULL;
        return -1;
    }
    for (int logical = 0; logical < logical_capacity; logical++)
        cuda_selected_logical_slots[logical] = -1;
    cuda_selected_capacity = capacity;
    cuda_selected_logical_capacity = logical_capacity;
    return 0;
}

extern "C" int salt_gpu_selected_resource_bind(
    int slot, uint64_t logical_resource_id, const void *base, size_t nbytes,
    const void *payload) {
    CudaSelectedResource *resource;
    const unsigned char *device_payload;
    uintptr_t b, p;
    if (!cuda_ready || !cuda_selected_resources ||
        !cuda_selected_logical_slots || slot < 0 ||
        slot >= cuda_selected_capacity || logical_resource_id >=
            (uint64_t)(uint32_t)cuda_selected_logical_capacity ||
        !base || !payload || nbytes < 1)
        return -1;
    b = (uintptr_t)base;
    p = (uintptr_t)payload;
    if (nbytes > UINTPTR_MAX - b || p < b || p - b >= nbytes) return -1;
    device_payload = (const unsigned char *)payload;
    if (!cuda_pageable_mmap) {
        CudaResource *pool = cuda_resource(SALT_GPU_RESOURCE_EXPERT_LAYER, 15);
        uintptr_t host, device;
        size_t offset;
        if (!pool || !pool->active || !pool->registered || !pool->device ||
            pool->policy != SALT_GPU_WEIGHT_ADDRESS_REGISTERED_PERSISTENT ||
            !pool->host)
            return -1;
        host = (uintptr_t)(const void *)pool->host;
        device = (uintptr_t)(void *)pool->device;
        if (p < host || p - host > pool->nbytes ||
            nbytes > pool->nbytes - (size_t)(p - host) ||
            device > UINTPTR_MAX - (p - host))
            return -1;
        offset = (size_t)(p - host);
        device_payload = (const unsigned char *)(void *)(device + offset);
    }
    resource = &cuda_selected_resources[slot];
    if (resource->active)
        return resource->logical_resource_id == logical_resource_id &&
            resource->base == (const unsigned char *)base &&
            resource->payload == (const unsigned char *)payload &&
            resource->device_payload == device_payload &&
            resource->nbytes == nbytes &&
            cuda_selected_logical_slots[logical_resource_id] == slot ? 0 : -1;
    if (cuda_selected_logical_slots[logical_resource_id] != -1) return -1;
    resource->active = 1;
    resource->logical_resource_id = logical_resource_id;
    resource->base = (const unsigned char *)base;
    resource->payload = (const unsigned char *)payload;
    resource->device_payload = device_payload;
    resource->nbytes = nbytes;
    cuda_selected_logical_slots[logical_resource_id] = slot;
    cuda_selected_retirement_fenced = 0;
    return 0;
}

extern "C" int salt_gpu_selected_resource_bind_resident(
    int slot, uint64_t logical_resource_id, const void *payload, size_t nbytes,
    uintptr_t device_address, uint64_t generation) {
    (void)slot; (void)logical_resource_id; (void)payload; (void)nbytes;
    (void)device_address; (void)generation;
    return -1;
}

extern "C" int salt_gpu_selected_resources_fence(void) {
    if (!cuda_ready || !cuda_selected_resources)
        return -1;
    if (cuda_text_resource_fence_credit) {
        cuda_text_resource_fence_credit = 0;
    } else if (cuda_success(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize selected retirement batch") != 0) {
        return -1;
    }
    cuda_selected_retirement_fenced = 1;
    return 0;
}

extern "C" int salt_gpu_selected_resource_unbind(
    int slot, uint64_t logical_resource_id) {
    CudaSelectedResource *resource;
    if (!cuda_selected_resources || !cuda_selected_logical_slots || slot < 0 ||
        slot >= cuda_selected_capacity || logical_resource_id >=
            (uint64_t)(uint32_t)cuda_selected_logical_capacity)
        return -1;
    resource = &cuda_selected_resources[slot];
    if (!resource->active) return 0;
    if (resource->logical_resource_id != logical_resource_id ||
        cuda_selected_logical_slots[logical_resource_id] != slot ||
        (!cuda_selected_retirement_fenced &&
         cuda_success(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize selected unbind") != 0))
        return -1;
    cuda_selected_logical_slots[logical_resource_id] = -1;
    memset(resource, 0, sizeof *resource);
    return 0;
}

extern "C" int salt_gpu_q4_selected_shared(
    const SaltGpuSelectedProjectionJob *jobs, int job_count,
    SaltGpuSharedBuffer *arena) {
    uintptr_t arena_base;
    int max_rows = 0, max_groups = 0;
    uint64_t logical_jobs = 0;
    if (!cuda_ready || !jobs || job_count < 1 || job_count > 256 || !arena ||
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
            cuda_moe_descriptor(entry->vals, entry->scales, entry->biases,
                entry->id, entry->rows, entry->cols, entry->batch,
                entry->resource_slot, entry->logical_resource_id,
                1,
                (uint32_t)(input_offset / sizeof(float)),
                (uint32_t)(output_offset / sizeof(float)),
                &cuda_host_desc[job]) != 0)
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
        logical_jobs += (uint32_t)entry->batch;
    }
    if (cuda_success(cudaMemcpy(cuda_device_desc, cuda_host_desc,
            (size_t)job_count * sizeof(CudaBatchDesc), cudaMemcpyHostToDevice),
            "cudaMemcpy selected shared descriptors") != 0)
        return -1;
    cuda_q4_heterogeneous<<<dim3((unsigned int)max_rows,
        (unsigned int)max_groups, (unsigned int)job_count), 32>>>(
        cuda_device_desc, job_count, (const float *)arena->backend,
        (float *)arena->backend);
    if (cuda_success(cudaGetLastError(), "selected shared launch") != 0 ||
        cuda_success(cudaDeviceSynchronize(),
            "cudaDeviceSynchronize selected shared") != 0)
        return -1;
    cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += logical_jobs;
    return 0;
}

extern "C" int salt_gpu_proj_batch_indexed(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int first_job, int job_count, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset, const void *id) {
    CudaTensor *tensor;
    CudaResource *resource;
    uint64_t output_span, input_floats, selected_output_floats;
    size_t output_end;
    float *device_output;
    if (!cuda_ready || !vals || !scales || !biases || !xs || !output ||
        !output->contents || !output->backend || !id ||
        R < 1 || C < 1 || C > CUDA_MAX_C || B < 2 || B > 512 ||
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
        input_floats > CUDA_MAX_OUTPUTS ||
        selected_output_floats > CUDA_MAX_OUTPUTS)
        return -1;
    output_end = y_float_offset + (size_t)output_span;
    if (output_end > output->nbytes / sizeof(float)) return -1;
    tensor = cuda_tensor(id);
    if (!tensor || tensor->bits != 4 || tensor->rows != R ||
        tensor->cols != C)
        return -1;
    resource = cuda_resource(tensor->kind, tensor->resource_id);
    if (!resource ||
        resource->host + tensor->value_offset !=
            (const unsigned char *)(const void *)vals ||
        resource->host + tensor->scale_offset !=
            (const unsigned char *)(const void *)scales ||
        resource->host + tensor->bias_offset !=
            (const unsigned char *)(const void *)biases)
        return -1;
    if (cuda_success(cudaMemcpy(
            cuda_x, xs + (size_t)first_job * (size_t)C,
            (size_t)input_floats * sizeof(float), cudaMemcpyHostToDevice),
            "cudaMemcpy indexed input") != 0)
        return -1;
    device_output = (float *)output->backend + y_float_offset +
        (size_t)first_job * (size_t)R;
    if (job_count >= 8) {
        cuda_q4_warp_batch<<<(unsigned int)R, 1024>>>(
            resource->device, tensor->value_offset, tensor->scale_offset,
            tensor->bias_offset, cuda_x, device_output, R, C, job_count);
    } else {
        cuda_q4_warp_exact<<<dim3((unsigned int)((R + 3) / 4),
                                   (unsigned int)job_count), 128>>>(
            resource->device, tensor->value_offset, tensor->scale_offset,
            tensor->bias_offset, cuda_x, device_output, R, C);
    }
    if (cuda_success(cudaGetLastError(),
                     "indexed projection launch") != 0)
        return -1;
    if (tensor->kind == SALT_GPU_RESOURCE_TRUNK) cuda_stats.trunk_batches++;
    else if (tensor->kind == SALT_GPU_RESOURCE_EXPERT_LAYER)
        cuda_stats.expert_layer_batches++;
    cuda_stats.direct_output_batches++;
    cuda_stats.direct_output_jobs += (uint64_t)(uint32_t)job_count;
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
    if (!enabled && cuda_defer) rc = salt_gpu_sync();
    cuda_defer = enabled;
    return rc;
}
extern "C" int salt_gpu_defer(void) { return cuda_defer; }
extern "C" int salt_gpu_sync(void) {
    return cuda_ready ? cuda_success(cudaDeviceSynchronize(),
                                     "cudaDeviceSynchronize") : 0;
}

static unsigned char *cuda_text_bytes(
        CudaTextProgramState *state, size_t offset) {
    return (unsigned char *)state->canonical.backend + offset;
}

static float *cuda_text_f32(CudaTextProgramState *state, size_t offset) {
    return (float *)(void *)cuda_text_bytes(state, offset);
}

static int32_t *cuda_text_i32(CudaTextProgramState *state, size_t offset) {
    return (int32_t *)(void *)cuda_text_bytes(state, offset);
}

static const SaltTensorResourceSpec *cuda_text_resource_spec(
        const SaltTextVerifyProgram *program,
        const SaltTensorStorageSpec *storage) {
    return salt_tensor_resource_find(program->descriptor->tensor_resources,
        program->descriptor->tensor_resource_count,
        storage->resource_kind, storage->resource_id);
}

static const unsigned char *cuda_text_resource_device(
        const SaltTextVerifyProgram *program,
        const SaltTensorStorageSpec *storage) {
    const SaltTensorResourceSpec *spec =
        cuda_text_resource_spec(program, storage);
    void *device = NULL;
    if (!spec || !spec->base || spec->mapped_nbytes < spec->nbytes)
        return NULL;
    for (int index = 0; index < CUDA_MAX_RESOURCES; index++)
        if (cuda_resources[index].used && cuda_resources[index].active &&
            cuda_resources[index].device &&
            cuda_resources[index].host == spec->base &&
            cuda_resources[index].nbytes >= spec->nbytes)
            return cuda_resources[index].device;
    if (storage->source_class != SALT_TENSOR_SOURCE_SHARED)
        return NULL;
    if (cudaHostGetDevicePointer(&device, (void *)spec->base, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return NULL;
    }
    return (const unsigned char *)device;
}

static int cuda_text_realize_tensor(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        CudaTextTensorRef *ref) {
    const unsigned char *resource;
    if (!program || !tensor || !ref ||
        salt_tensor_desc_validate(tensor) != 0 ||
        !(resource = cuda_text_resource_device(program, &tensor->storage)))
        return -1;
    memset(ref, 0, sizeof *ref);
    ref->resource = resource;
    ref->value_offset = tensor->storage.value_offset;
    ref->scale_offset = tensor->storage.scale_offset;
    ref->bias_offset = tensor->storage.bias_offset;
    ref->rows = tensor->rows;
    ref->cols = tensor->cols;
    ref->encoding = (uint32_t)tensor->storage.encoding;
    return 0;
}

static int cuda_text_realize_optional(
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        CudaTextTensorRef *ref) {
    if (salt_tensor_desc_absent(tensor)) {
        memset(ref, 0, sizeof *ref);
        return 0;
    }
    return cuda_text_realize_tensor(program, tensor, ref);
}

typedef struct CudaTextBootSamples {
    uint64_t offset[3], nbytes[3];
    unsigned char first[3], middle[3], last[3];
} CudaTextBootSamples;

/* One startup-only device dereference of the already-realized address book.
 * The counter lives in the program's existing attention workspace; no weight
 * registration, residency action or inference state is involved. */
__global__ static void cuda_text_boot_check_kernel(
        const unsigned char *resource, CudaTextBootSamples samples,
        unsigned int *mismatches) {
    unsigned int part = threadIdx.x;
    if (part >= 3u || !samples.nbytes[part]) return;
    const unsigned char *bytes = resource + samples.offset[part];
    uint64_t last = samples.nbytes[part] - 1u;
    if (bytes[0] != samples.first[part] ||
        bytes[samples.nbytes[part] / 2u] != samples.middle[part] ||
        bytes[last] != samples.last[part])
        atomicAdd(mismatches, 1u);
}

static int cuda_text_boot_check_tensor(CudaTextProgramState *state,
        const SaltTextVerifyProgram *program, const SaltTextTensorDesc *tensor,
        const CudaTextTensorRef *ref, unsigned int *parts) {
    const SaltTensorResourceSpec *spec;
    CudaTextBootSamples samples = {};
    const uint64_t offsets[3] = {tensor->storage.value_offset,
        tensor->storage.scale_offset, tensor->storage.bias_offset};
    const uint64_t sizes[3] = {tensor->storage.value_bytes,
        tensor->storage.scale_bytes, tensor->storage.bias_bytes};
    if (salt_tensor_desc_absent(tensor)) return 0;
    if (tensor->storage.source_class == SALT_TENSOR_SOURCE_SELECTED)
        return 0; /* A selected slot has no payload until the router asks. */
    if (!ref->resource ||
        !(spec = cuda_text_resource_spec(program, &tensor->storage)) ||
        !spec->base || spec->mapped_nbytes < spec->nbytes ||
        ref->resource != cuda_text_resource_device(program, &tensor->storage))
        return -1;
    for (unsigned int part = 0; part < 3u; part++) {
        uint64_t offset = offsets[part], size = sizes[part];
        if (!size) continue;
        if (offset > spec->nbytes || size > spec->nbytes - offset ||
            offset > spec->mapped_nbytes ||
            size > spec->mapped_nbytes - offset)
            return -1;
        const unsigned char *expected =
            (const unsigned char *)spec->base + (size_t)offset;
        samples.offset[part] = offset;
        samples.nbytes[part] = size;
        samples.first[part] = expected[0];
        samples.middle[part] = expected[(size_t)(size / 2u)];
        samples.last[part] = expected[(size_t)(size - 1u)];
        ++*parts;
    }
    cuda_text_boot_check_kernel<<<1, 32, 0, state->stream>>>(
        ref->resource, samples,
        (unsigned int *)(void *)state->attention_score_workspace);
    if (cuda_success(cudaGetLastError(), "CUDA trunk boot check launch") != 0)
        return -1;
    return 1;
}

static int cuda_text_boot_check_program(CudaTextProgramState *state,
        const SaltTextVerifyProgram *program) {
    unsigned int tensors = 0, parts = 0, mismatches = 0;
    if (!cuda_pageable_mmap) return 0;
    if (!state->stream || !state->attention_score_workspace ||
        !state->attention_score_workspace_floats ||
        cuda_success(cudaMemsetAsync(state->attention_score_workspace, 0,
            sizeof(unsigned int), state->stream),
            "CUDA trunk boot check reset") != 0)
        return -1;
#define BOOT_CHECK(tensor, ref) do { \
    int check = cuda_text_boot_check_tensor(state, program, (tensor), (ref), \
        &parts); \
    if (check < 0) return -1; \
    tensors += (unsigned int)check; \
} while (0)
    BOOT_CHECK(&program->descriptor->embedding, &state->embedding);
    BOOT_CHECK(&program->descriptor->final_norm, &state->final_norm);
    BOOT_CHECK(&program->descriptor->output_head, &state->output_head);
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextLayerExecDesc *source = program->layers[layer].descriptor;
        const CudaTextLayerRefs *target = &state->layers[layer];
        if (!source) return -1;
#define CHECK(field) BOOT_CHECK(&source->field, &target->field)
        CHECK(q); CHECK(k); CHECK(v); CHECK(o);
        CHECK(dense_gate); CHECK(dense_up); CHECK(dense_down);
        CHECK(router); CHECK(pre_attention_norm); CHECK(q_norm);
        CHECK(k_norm); CHECK(post_attention_norm);
        CHECK(pre_ffn_norm_1); CHECK(pre_ffn_norm_2);
        CHECK(post_ffn_norm_1); CHECK(post_ffn_norm_2);
        CHECK(post_ffn_norm); CHECK(router_scale);
        CHECK(per_expert_scale); CHECK(layer_scalar);
#undef CHECK
    }
#undef BOOT_CHECK
    if (!tensors || !parts ||
        cuda_success(cudaMemcpyAsync(&mismatches,
            state->attention_score_workspace, sizeof mismatches,
            cudaMemcpyDeviceToHost, state->stream),
            "CUDA trunk boot check readback") != 0 ||
        cuda_success(cudaStreamSynchronize(state->stream),
            "CUDA trunk boot check fence") != 0 || mismatches) {
        fprintf(stderr, "CUDA_TRUNK_BOOT_ADDRESS_CHECK status=FAIL "
            "tensors=%u parts=%u mismatches=%u\n",
            tensors, parts, mismatches);
        return -1;
    }
    fprintf(stderr, "CUDA_TRUNK_BOOT_ADDRESS_CHECK status=PASS "
        "tensors=%u parts=%u mismatches=0\n", tensors, parts);
    return 0;
}

static int cuda_text_expert_spec(const SaltTextVerifyProgram *program,
        uint32_t layer, uint32_t expert) {
    const SaltTextLayerExecDesc *source;
    const SaltTextExpertDesc *entry;
    const SaltTextTensorDesc *tensors[3];
    uint64_t logical;
    if (!program || layer >= program->layer_count ||
        expert >= CUDA_TEXT_EXPERTS)
        return -1;
    source = program->layers[layer].descriptor;
    if (!source || !source->plan || !source->experts ||
        source->expert_count != CUDA_TEXT_EXPERTS ||
        source->plan->n_experts != (int)CUDA_TEXT_EXPERTS ||
        source->plan->top_k_experts != (int)CUDA_TEXT_TOPK)
        return -1;
    entry = &source->experts[expert];
    tensors[0] = &entry->gate;
    tensors[1] = &entry->up;
    tensors[2] = &entry->down;
    logical = (uint64_t)layer * CUDA_TEXT_EXPERTS + expert;
    if (entry->expert_id != expert) return -1;
    for (uint32_t phase = 0; phase < 3u; phase++) {
        const SaltTextTensorDesc *tensor = tensors[phase];
        const SaltTensorStorageSpec *storage = &tensor->storage;
        if (salt_tensor_desc_validate(tensor) != 0 ||
            salt_tensor_storage_resolves(storage,
                program->descriptor->tensor_resources,
                program->descriptor->tensor_resource_count) != 0 ||
            storage->source_class != SALT_TENSOR_SOURCE_SELECTED ||
            storage->encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
            storage->logical_resource_id != logical ||
            storage->resource_kind != entry->gate.storage.resource_kind ||
            storage->resource_id != entry->gate.storage.resource_id ||
            tensor->rows != (phase == 2u ? program->hidden :
                (uint32_t)source->plan->expert_intermediate) ||
            tensor->cols != (phase == 2u ?
                (uint32_t)source->plan->expert_intermediate : program->hidden))
            return -1;
    }
    return 0;
}

static int cuda_text_selected_refs(const SaltTextVerifyProgram *program,
        uint32_t layer, uint32_t expert, int32_t slot,
        CudaTextTensorRef refs[3]) {
    const SaltTextExpertDesc *entry;
    const SaltTextTensorDesc *tensors[3];
    const SaltTensorResourceSpec *resource;
    const CudaSelectedResource *selected;
    uint64_t logical, origin;
    uintptr_t base, payload;
    if (!refs ||
        cuda_text_expert_spec(program, layer, expert) != 0 ||
        !cuda_selected_resources || !cuda_selected_logical_slots ||
        slot < 0 || slot >= cuda_selected_capacity)
        return -1;
    entry = &program->layers[layer].descriptor->experts[expert];
    tensors[0] = &entry->gate;
    tensors[1] = &entry->up;
    tensors[2] = &entry->down;
    resource = cuda_text_resource_spec(program, &entry->gate.storage);
    selected = &cuda_selected_resources[slot];
    logical = entry->gate.storage.logical_resource_id;
    origin = entry->gate.storage.value_offset;
    base = (uintptr_t)(const void *)selected->base;
    payload = (uintptr_t)(const void *)selected->payload;
    if (!resource || !selected->active || !selected->device_payload ||
        !base || !payload ||
        logical >= (uint64_t)(uint32_t)cuda_selected_logical_capacity ||
        selected->logical_resource_id != logical ||
        cuda_selected_logical_slots[logical] != slot ||
        selected->nbytes > UINTPTR_MAX - base || payload < base ||
        payload - base >= selected->nbytes)
        return -1;
    if (resource->mapped_nbytes == resource->nbytes &&
        (!resource->base || origin > UINTPTR_MAX - (uintptr_t)resource->base ||
         payload != (uintptr_t)resource->base + (uintptr_t)origin))
        return -1;
    size_t available = selected->nbytes - (size_t)(payload - base);
    for (uint32_t phase = 0; phase < 3u; phase++) {
        const SaltTensorStorageSpec *storage = &tensors[phase]->storage;
        const uint64_t offsets[3] = {storage->value_offset,
            storage->scale_offset, storage->bias_offset};
        const uint64_t sizes[3] = {storage->value_bytes,
            storage->scale_bytes, storage->bias_bytes};
        for (uint32_t part = 0; part < 3u; part++)
            if (offsets[part] < origin || offsets[part] - origin > available ||
                sizes[part] > available - (size_t)(offsets[part] - origin))
                return -1;
        CudaBatchDesc descriptor;
        const unsigned char *values = selected->payload +
            (size_t)(storage->value_offset - origin);
        if (cuda_moe_descriptor((const uint32_t *)(const void *)values,
                (const uint16_t *)(const void *)(selected->payload +
                    (size_t)(storage->scale_offset - origin)),
                (const uint16_t *)(const void *)(selected->payload +
                    (size_t)(storage->bias_offset - origin)), values,
                (int)tensors[phase]->rows, (int)tensors[phase]->cols, 1,
                slot, logical, 1, 0u, 0u, &descriptor) != 0 ||
            descriptor.resource != selected->device_payload ||
            descriptor.value_offset != storage->value_offset - origin ||
            descriptor.scale_offset != storage->scale_offset - origin ||
            descriptor.bias_offset != storage->bias_offset - origin)
            return -1;
        memset(&refs[phase], 0, sizeof refs[phase]);
        refs[phase].resource = selected->device_payload;
        refs[phase].value_offset = storage->value_offset - origin;
        refs[phase].scale_offset = storage->scale_offset - origin;
        refs[phase].bias_offset = storage->bias_offset - origin;
        refs[phase].rows = tensors[phase]->rows;
        refs[phase].cols = tensors[phase]->cols;
        refs[phase].encoding = (uint32_t)storage->encoding;
    }
    return 0;
}

static int cuda_text_selected_address_book_realize(
        CudaTextProgramState *state,
        const SaltTextVerifyProgram *program) {
    const uint32_t logical_count = CUDA_TEXT_LAYERS * CUDA_TEXT_EXPERTS;
    if (!state || !program ||
        !cuda_selected_resources || !cuda_selected_logical_slots)
        return 0;
    if (cuda_selected_capacity != (int)logical_count ||
        cuda_selected_logical_capacity != (int)logical_count)
        return 0;
    for (uint32_t layer = 0; layer < CUDA_TEXT_LAYERS; layer++)
        for (uint32_t expert = 0; expert < CUDA_TEXT_EXPERTS; expert++) {
            uint32_t logical = layer * CUDA_TEXT_EXPERTS + expert;
            int32_t slot = cuda_selected_logical_slots[logical];
            size_t base = (size_t)logical * 3u;
            if (slot < 0 ||
                cuda_text_selected_refs(program, layer, expert, slot,
                    &state->expert_refs[base]) != 0)
                return -1;
        }
    return 1;
}

static const float *cuda_text_tensor_f32(const CudaTextTensorRef *ref) {
    return ref && ref->resource &&
        ref->encoding == SALT_TENSOR_ENCODING_F32
        ? (const float *)(const void *)(ref->resource + ref->value_offset)
        : NULL;
}

static int cuda_text_project(
        CudaTextProgramState *state, const CudaTextTensorRef *ref,
        const float *input, float *output, uint32_t batch) {
    dim3 block(128u, 1u, 1u);
    dim3 grid((ref->rows + 3u) / 4u, batch, 1u);
    if (!state || !ref || !ref->resource || !input || !output || batch == 0 ||
        ref->rows == 0 || ref->cols == 0)
        return -1;
    if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
        cuda_q4_warp_exact<<<grid, block, 0, state->stream>>>(
            ref->resource, ref->value_offset, ref->scale_offset,
            ref->bias_offset, input, output, (int)ref->rows,
            (int)ref->cols);
    } else if (ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8) {
        cuda_q8_warp_exact<<<grid, block, 0, state->stream>>>(
            ref->resource, ref->value_offset, ref->scale_offset,
            ref->bias_offset, input, output, (int)ref->rows,
            (int)ref->cols);
    } else if (ref->encoding == SALT_TENSOR_ENCODING_BF16) {
        dim3 one_grid((ref->rows + 3u) / 4u, 1u, 1u);
        for (uint32_t row = 0; row < batch; row++)
            cuda_bf16_warp_exact<<<one_grid, block, 0, state->stream>>>(
                ref->resource, ref->value_offset,
                input + (size_t)row * ref->cols,
                output + (size_t)row * ref->rows,
                (int)ref->rows, (int)ref->cols);
    } else {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

/* Existing tensor bindings become by-value command arguments captured by the
 * existing extent graph.  No table, resource, or activation staging is added. */
static uint32_t cuda_text_projection_group_size(
        const CudaTextProgramState *state, uint32_t first_cell) {
    const SaltTextVerifyProgram *program = state->program;
    const SaltTextExecutionCell *cells = program->dispatch.cells;
    const CudaTextTensorRef *refs[3];
    uint32_t kinds[3], count;
    if (first_cell >= program->dispatch.cell_count ||
        cells[first_cell].layer >= program->layer_count) return 0u;
    const CudaTextLayerRefs *layer =
        &state->layers[cells[first_cell].layer];
    if (cells[first_cell].kind == SALT_TEXT_CELL_QUERY_PROJECTION) {
        count = program->layers[cells[first_cell].layer].descriptor->plan->
            attention.shared_kv_projection ? 2u : 3u;
        kinds[0] = SALT_TEXT_CELL_QUERY_PROJECTION;
        kinds[1] = SALT_TEXT_CELL_KEY_PROJECTION;
        kinds[2] = SALT_TEXT_CELL_VALUE_PROJECTION;
        refs[0] = &layer->q; refs[1] = &layer->k; refs[2] = &layer->v;
    } else if (cells[first_cell].kind == SALT_TEXT_CELL_DENSE_GATE) {
        count = 2u;
        kinds[0] = SALT_TEXT_CELL_DENSE_GATE;
        kinds[1] = SALT_TEXT_CELL_DENSE_UP;
        refs[0] = &layer->dense_gate; refs[1] = &layer->dense_up;
    } else return 0u;
    if (count > program->dispatch.cell_count - first_cell) return 0u;
    for (uint32_t i = 0; i < count; i++)
        if ((uint32_t)cells[first_cell + i].kind != kinds[i] ||
            cells[first_cell + i].layer != cells[first_cell].layer ||
            refs[i]->encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
            refs[i]->cols != refs[0]->cols) return 0u;
    return count;
}

static int cuda_text_project_group(
        CudaTextProgramState *state, uint32_t first_cell, uint32_t count) {
    const SaltTextVerifyProgram *program = state->program;
    const SaltTextExecutionCell *cell = &program->dispatch.cells[first_cell];
    const CudaTextLayerRefs *layer = &state->layers[cell->layer];
    const CudaTextTensorRef *refs[3];
    CudaBatchDesc descriptors[3] = {};
    uint32_t max_rows = 0u;
    uint64_t input_offset = program->layout.normalized;
    if (count < 2u || count > 3u ||
        cuda_text_projection_group_size(state, first_cell) != count ||
        input_offset % sizeof(float) ||
        input_offset / sizeof(float) > UINT32_MAX) return -1;
    if (cell->kind == SALT_TEXT_CELL_QUERY_PROJECTION) {
        refs[0] = &layer->q; refs[1] = &layer->k; refs[2] = &layer->v;
    } else {
        refs[0] = &layer->dense_gate; refs[1] = &layer->dense_up;
    }
    for (uint32_t i = 0; i < count; i++) {
        const CudaTextTensorRef *ref = refs[i];
        uint64_t output_offset = cell[i].destination_offset;
        if (!ref->resource || !ref->rows || !ref->cols ||
            ref->rows > INT_MAX || ref->cols > INT_MAX ||
            input_offset > state->canonical_bytes ||
            (uint64_t)ref->cols * sizeof(float) >
                state->canonical_bytes - input_offset ||
            output_offset % sizeof(float) ||
            output_offset / sizeof(float) > UINT32_MAX ||
            output_offset > state->canonical_bytes ||
            (uint64_t)ref->rows * sizeof(float) >
                state->canonical_bytes - output_offset) return -1;
        descriptors[i].resource = ref->resource;
        descriptors[i].value_offset = ref->value_offset;
        descriptors[i].scale_offset = ref->scale_offset;
        descriptors[i].bias_offset = ref->bias_offset;
        descriptors[i].input_offset = (uint32_t)(input_offset / sizeof(float));
        descriptors[i].output_offset = (uint32_t)(output_offset / sizeof(float));
        descriptors[i].rows = (int)ref->rows;
        descriptors[i].cols = (int)ref->cols;
        descriptors[i].batch = 1;
        if (ref->rows > max_rows) max_rows = ref->rows;
    }
    cuda_q4_heterogeneous_warp<<<dim3((max_rows + 3u) / 4u, 1u, count),
        128, 0, state->stream>>>(NULL, (int)count,
            cuda_text_f32(state, 0u), cuda_text_f32(state, 0u),
            descriptors[0], descriptors[1], descriptors[2]);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int cuda_text_map_kv(const SaltTextKvRowsDesc *rows,
                            uint32_t width, float **device_out) {
    void *device = NULL;
    if (!rows || !device_out ||
        (rows->private_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
         rows->private_mode != SALT_TEXT_KV_PRIVATE_RING) ||
        rows->private_position_base != 0 ||
        rows->private_row_stride != width ||
        rows->private_float_capacity <
            (size_t)rows->private_row_capacity * width ||
        !rows->private_rows)
        return -1;
    if (rows->shared_prefix_state &&
        rows->shared_prefix_state->row_count != 0)
        return -1;
    if (cudaHostGetDevicePointer(&device, rows->private_rows, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        return -1;
    }
    *device_out = (float *)device;
    return 0;
}

static int cuda_text_requirements(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextGpuProgramRequirements *requirements,
        size_t requirements_size) {
    if (!program || !program->ready || !program->descriptor ||
        !program->descriptor->model || !policy || !requirements ||
        requirements_size != sizeof *requirements || !cuda_ready ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        program->layer_count != CUDA_TEXT_LAYERS ||
        program->maximum_candidates > CUDA_TEXT_MAX_ROWS ||
        program->dispatch.cell_count != CUDA_TEXT_CELLS ||
        program->hidden != CUDA_TEXT_HIDDEN)
        return -1;
    memset(requirements, 0, sizeof *requirements);
    requirements->backend_state_bytes = sizeof(CudaTextProgramState);
    requirements->command_bytes = sizeof(CudaTextProgramCommand);
    requirements->maximum_commands = program->dispatch.cell_count;
    /* Selected reads remain in flight after expert-down encoding. The next
     * NEED_RESOURCE wait (or final finish) is the existing release fence. */
    requirements->flags = SALT_TEXT_GPU_DEFER_EXPERT_RELEASE;
    return 0;
}

static void cuda_text_state_release(CudaTextProgramState *state) {
    if (!state) return;
    if (state->stream) (void)cudaStreamSynchronize(state->stream);
    for (uint32_t index = 0; index < CUDA_TEXT_GRAPH_COUNT; index++) {
        if (state->graphs[index].executable)
            (void)cudaGraphExecDestroy(state->graphs[index].executable);
        if (state->graphs[index].graph)
            (void)cudaGraphDestroy(state->graphs[index].graph);
        for (uint32_t extent = 0; extent < CUDA_TEXT_MAX_EXTENTS; extent++) {
            CudaTextExtentGraph *entry = &state->extent_graphs[index][extent];
            if (entry->executable)
                (void)cudaGraphExecDestroy(entry->executable);
            if (entry->graph) (void)cudaGraphDestroy(entry->graph);
        }
    }
    if (state->graph_tokens) (void)cudaFreeHost(state->graph_tokens);
    if (state->graph_parent_rows) (void)cudaFreeHost(state->graph_parent_rows);
    if (state->graph_depths) (void)cudaFreeHost(state->graph_depths);
    if (state->graph_source_positions)
        (void)cudaFreeHost(state->graph_source_positions);
    if (state->device_source_position)
        (void)cudaFree(state->device_source_position);
    if (state->device_kv) (void)cudaFree(state->device_kv);
    if (state->stream) (void)cudaStreamDestroy(state->stream);
    if (state->device_experts) (void)cudaFree(state->device_experts);
    if (state->canonical.backend)
        (void)cudaFree(state->canonical.backend);
    if (state->canonical.contents)
        (void)cudaFreeHost(state->canonical.contents);
    memset(state, 0, sizeof *state);
}

static int cuda_text_prepare_templates(
        CudaTextProgramState *state, const SaltTextVerifyProgram *program) {
    uint32_t selected_capacity = 0;
    if (!state || !program || program->dispatch.cell_count != CUDA_TEXT_CELLS)
        return -1;
    memset(state->templates, 0, sizeof state->templates);
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        CudaTextCellTemplate *target = &state->templates[index];
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
    return 0;
}

static int cuda_text_capture_graphs(
    CudaTextProgramState *state, const SaltTextVerifyProgram *program,
    const SaltTextExecutorPlan *plan);
static int cuda_text_capture_extent_graphs(
    CudaTextProgramState *state, const SaltTextVerifyProgram *program,
    const SaltTextExecutorPlan *plan);

static int cuda_text_prepare(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    const SaltTensorResourceSpec *expert_resource;
    cudaDeviceProp property;
    cudaFuncAttributes attention_attributes;
    int device = 0;
    size_t bytes, device_bytes, score_offset, score_floats, score_bytes;
    if (!program || !plan || plan->program != program ||
        plan->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !canonical_out || !state || backend_state_bytes < sizeof *state ||
        !command || command_bytes < sizeof(CudaTextProgramCommand) ||
        program->layout.total_bytes > SIZE_MAX - 64u)
        return -1;
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof(CudaTextProgramCommand));
    if (cuda_success(cudaGetDevice(&device), "cudaGetDevice text program") != 0 ||
        cuda_success(cudaGetDeviceProperties(&property, device),
            "cudaGetDeviceProperties text program") != 0 ||
        cuda_success(cudaFuncGetAttributes(&attention_attributes,
            cuda_text_attention_body),
            "cudaFuncGetAttributes text attention") != 0 ||
        property.sharedMemPerBlockOptin <= attention_attributes.sharedSizeBytes)
        return -1;
    state->attention_dynamic_limit = property.sharedMemPerBlockOptin -
        attention_attributes.sharedSizeBytes;
    if (state->attention_dynamic_limit > (size_t)INT_MAX ||
        cuda_success(cudaFuncSetAttribute(cuda_text_attention_body,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)state->attention_dynamic_limit),
            "cudaFuncSetAttribute text attention") != 0)
        return -1;
    bytes = (program->layout.total_bytes + 63u) & ~(size_t)63u;
    if (bytes > SIZE_MAX - sizeof(int)) return -1;
    state->status_offset = bytes;
    bytes += sizeof(int);
    score_offset = (bytes + 63u) & ~(size_t)63u;
    if ((size_t)program->maximum_context >
            SIZE_MAX / CUDA_TEXT_ATTENTION_TASK_TILE)
        return -1;
    score_floats = (size_t)program->maximum_context *
        CUDA_TEXT_ATTENTION_TASK_TILE;
    if (score_floats > SIZE_MAX / sizeof(float)) return -1;
    score_bytes = score_floats * sizeof(float);
    if (score_offset > SIZE_MAX - score_bytes) return -1;
    device_bytes = score_offset + score_bytes;
    if (cuda_success(cudaHostAlloc(&state->canonical.contents, bytes,
            cudaHostAllocDefault), "cudaHostAlloc text canonical") != 0 ||
        cuda_success(cudaMalloc(&state->canonical.backend, device_bytes),
            "cudaMalloc text canonical") != 0 ||
        cuda_success(cudaMemset(state->canonical.backend, 0, bytes),
            "cudaMemset text canonical") != 0 ||
        cuda_success(cudaStreamCreateWithFlags(
            &state->stream, cudaStreamNonBlocking),
            "cudaStreamCreate text program") != 0)
        goto fail;
    state->canonical.nbytes = bytes;
    state->attention_score_workspace = (float *)(void *)(
        (unsigned char *)state->canonical.backend + score_offset);
    state->attention_score_workspace_floats = score_floats;
    memset(state->canonical.contents, 0, bytes);
    state->program = program;
    expert_resource = salt_tensor_resource_find(
        program->descriptor->tensor_resources,
        program->descriptor->tensor_resource_count,
        SALT_TENSOR_RESOURCE_EXPERT, 0u);
    if (!expert_resource ||
        (expert_resource->source_class != SALT_TENSOR_SOURCE_SELECTED &&
         expert_resource->source_class != SALT_TENSOR_SOURCE_STATIC))
        goto fail;
    if (expert_resource->source_class == SALT_TENSOR_SOURCE_SELECTED) {
        int address_book =
            cuda_text_selected_address_book_realize(state, program);
        if (address_book < 0 || (!cuda_pageable_mmap && address_book != 1))
            goto fail;
        state->dynamic_experts = address_book == 1 ? 0u : 1u;
    } else {
        state->dynamic_experts = 0u;
    }
    state->canonical_bytes = program->layout.total_bytes;
    if (cuda_text_realize_tensor(program, &program->descriptor->embedding,
            &state->embedding) != 0 ||
        cuda_text_realize_tensor(program, &program->descriptor->final_norm,
            &state->final_norm) != 0 ||
        cuda_text_realize_tensor(program, &program->descriptor->output_head,
            &state->output_head) != 0)
        goto fail;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        const SaltTextLayerExecDesc *source = compiled->descriptor;
        CudaTextLayerRefs *target = &state->layers[layer];
#define REALIZE(field) \
        if (cuda_text_realize_tensor(program, &source->field, \
                &target->field) != 0) goto fail
#define OPTIONAL(field) \
        if (cuda_text_realize_optional(program, &source->field, \
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
        target->kv.width = compiled->kv_width;
        if (!source->kv ||
            cuda_text_map_kv(&source->kv->keys, compiled->kv_width,
                &target->kv.keys) != 0 ||
            cuda_text_map_kv(&source->kv->values, compiled->kv_width,
                &target->kv.values) != 0 ||
            source->kv->keys.private_mode != source->kv->values.private_mode ||
            source->kv->keys.private_row_capacity !=
                source->kv->values.private_row_capacity ||
            source->expert_count != CUDA_TEXT_EXPERTS)
            goto fail;
        target->kv.private_mode = source->kv->keys.private_mode;
        target->kv.row_capacity = source->kv->keys.private_row_capacity;
        target->kv.tentative_key_float_offset =
            compiled->tentative_key_offset / sizeof(float);
        target->kv.tentative_value_float_offset =
            compiled->tentative_value_offset / sizeof(float);
        for (uint32_t expert = 0; expert < source->expert_count; expert++) {
            const SaltTextExpertDesc *entry = &source->experts[expert];
            size_t base = ((size_t)layer * CUDA_TEXT_EXPERTS + expert) * 3u;
            if (entry->gate.storage.source_class ==
                    SALT_TENSOR_SOURCE_SELECTED) {
                if (cuda_text_expert_spec(program, layer, expert) != 0 ||
                    (!state->dynamic_experts &&
                     !state->expert_refs[base].resource))
                    goto fail;
                continue;
            }
            if (entry->expert_id != expert ||
                cuda_text_realize_tensor(program, &entry->gate,
                    &state->expert_refs[base]) != 0 ||
                cuda_text_realize_tensor(program, &entry->up,
                    &state->expert_refs[base + 1u]) != 0 ||
                cuda_text_realize_tensor(program, &entry->down,
                    &state->expert_refs[base + 2u]) != 0)
                goto fail;
        }
    }
    if (cuda_text_boot_check_program(state, program) != 0 ||
        cuda_text_prepare_templates(state, program) != 0 ||
        cuda_success(cudaMalloc((void **)&state->device_experts,
            sizeof state->expert_refs), "cudaMalloc text expert refs") != 0 ||
        cuda_success(cudaMalloc((void **)&state->device_kv,
            sizeof(CudaTextKvRef) * CUDA_TEXT_LAYERS),
            "cudaMalloc text KV refs") != 0 ||
        cuda_success(cudaHostAlloc((void **)&state->graph_tokens,
            (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
                sizeof(int32_t), cudaHostAllocDefault),
            "cudaHostAlloc text graph tokens") != 0 ||
        cuda_success(cudaHostAlloc((void **)&state->graph_parent_rows,
            (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
                sizeof(uint32_t), cudaHostAllocDefault),
            "cudaHostAlloc text graph parent rows") != 0 ||
        cuda_success(cudaHostAlloc((void **)&state->graph_depths,
            (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
                sizeof(uint32_t), cudaHostAllocDefault),
            "cudaHostAlloc text graph depths") != 0 ||
        cuda_success(cudaHostAlloc((void **)&state->graph_source_positions,
            (size_t)CUDA_TEXT_MAX_ROWS * sizeof(uint32_t),
            cudaHostAllocDefault),
            "cudaHostAlloc text graph source positions") != 0 ||
        cuda_success(cudaMalloc((void **)&state->device_source_position,
            sizeof(uint32_t)), "cudaMalloc text source position") != 0 ||
        cuda_success(cudaMemcpy(state->device_experts, state->expert_refs,
            sizeof state->expert_refs, cudaMemcpyHostToDevice),
            "cudaMemcpy text expert refs") != 0)
        goto fail;
    for (uint32_t layer = 0; layer < CUDA_TEXT_LAYERS; layer++)
        if (cuda_success(cudaMemcpy(state->device_kv + layer,
                &state->layers[layer].kv, sizeof(CudaTextKvRef),
                cudaMemcpyHostToDevice), "cudaMemcpy text KV ref") != 0)
            goto fail;
    state->magic = CUDA_TEXT_STATE_MAGIC;
    state->ready = 1;
    if (state->dynamic_experts && cuda_pageable_mmap) {
        if (cuda_text_capture_extent_graphs(state, program, plan) != 0)
            goto fail;
    } else if (!state->dynamic_experts &&
               cuda_text_capture_graphs(state, program, plan) != 0) {
        goto fail;
    }
    *canonical_out = &state->canonical;
    return 0;
fail:
    cuda_text_state_release(state);
    return -1;
}

static int cuda_text_full_graph_eligible(
        const CudaTextProgramState *state, uint32_t input_count,
        uint32_t output_rows, uint32_t score_rows) {
    return state && state->graphs_ready && !state->capture_mode &&
        input_count <= CUDA_TEXT_GRAPH_COUNT && output_rows == input_count &&
        (size_t)score_rows <= state->attention_dynamic_limit / sizeof(float);
}

static int cuda_text_begin_depth(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t maximum_depth, uint32_t output_rows, int authoritative) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    uint32_t *parents, *depths;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !state->ready ||
        !cmd || cmd->begun || generation == 0 || !input_token_ids ||
        input_count == 0 || input_count > state->program->maximum_candidates ||
        source_position >= state->program->maximum_context ||
        maximum_depth >= state->program->maximum_context - source_position ||
        output_rows > input_count)
        return -1;
    memset(cmd, 0, sizeof *cmd);
    cuda_text_resource_fence_credit = 0;
    cmd->ffn_norm_wave_layer = UINT32_MAX;
    cmd->resource_layer = UINT32_MAX;
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
    /* The one-row authoritative HMM program overwrites every live operand
     * before consuming it. Keep its startup-owned seats resident; clearing
     * those spans is neither initialization nor KV commitment. Preserve the
     * tentative/multirow path and its failure scrub unchanged. */
    int retain_scratch = authoritative && input_count == 1u &&
        maximum_depth == 0u && output_rows == 1u &&
        state->dynamic_experts && cuda_pageable_mmap;
    cmd->authoritative = authoritative ? 1u : 0u;
    cmd->stats.canonical_clear_bytes =
        retain_scratch ? 0u : cmd->touched_span_bytes;
    parents = state->graph_parent_rows +
        (size_t)(input_count - 1u) * CUDA_TEXT_MAX_ROWS;
    depths = state->graph_depths +
        (size_t)(input_count - 1u) * CUDA_TEXT_MAX_ROWS;
    state->graph_source_positions[input_count - 1u] = source_position;
    for (uint32_t row = 0; row < input_count; row++) {
        parents[row] = row == 0u ? UINT32_MAX : row - 1u;
        depths[row] = row;
    }
    if (cuda_text_full_graph_eligible(state, input_count, output_rows,
            cmd->attention_score_rows)) {
        int32_t *staging = state->graph_tokens +
            (size_t)(input_count - 1u) * CUDA_TEXT_MAX_ROWS;
        memcpy(staging, input_token_ids,
            (size_t)input_count * sizeof(int32_t));
    } else {
        for (uint32_t index = 0;
             !retain_scratch && index < cmd->touched_span_count; index++)
            if (cuda_success(cudaMemsetAsync(
                    cuda_text_bytes(state, cmd->touched_spans[index].offset), 0,
                    cmd->touched_spans[index].bytes, state->stream),
                    "cudaMemset text touched span") != 0)
                return -1;
        if (cuda_success(cudaMemsetAsync(cuda_text_bytes(state,
                state->status_offset), 0, sizeof(int), state->stream),
                "cudaMemset text status") != 0 ||
            cuda_success(cudaMemcpyAsync(cuda_text_i32(state,
                state->program->layout.candidate_token_ids), input_token_ids,
                (size_t)input_count * sizeof(int32_t), cudaMemcpyHostToDevice,
                state->stream), "cudaMemcpy text token ids") != 0 ||
            cuda_success(cudaMemcpyAsync(cuda_text_bytes(state,
                state->program->layout.target_parent_rows), parents,
                (size_t)input_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                state->stream), "cudaMemcpy text parent rows") != 0 ||
            cuda_success(cudaMemcpyAsync(cuda_text_bytes(state,
                state->program->layout.target_depths), depths,
                (size_t)input_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
                state->stream), "cudaMemcpy text depths") != 0 ||
            cuda_success(cudaMemcpyAsync(state->device_source_position,
                &state->graph_source_positions[input_count - 1u],
                sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream),
                "cudaMemcpy text source position") != 0)
            return -1;
    }
    cmd->generation = generation;
    cmd->source_position = source_position;
    cmd->input_count = input_count;
    cmd->maximum_depth = maximum_depth;
    cmd->authoritative_output_rows = output_rows;
    cmd->stats.backend_template_count = state->template_count;
    cmd->stats.backend_dynamic_patches = 4u;
    cmd->stats.backend_selected_job_capacity = state->selected_job_capacity;
    cmd->begun = 1;
    return 0;
}

static int cuda_text_begin(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    if (input_count == 0) return -1;
    return cuda_text_begin_depth(backend_state, command, generation,
        source_position, input_token_ids, input_count, input_count - 1u,
        input_count, 0);
}

static int cuda_text_begin_authoritative(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    if (input_count == 0u) return -1;
    return cuda_text_begin_depth(backend_state, command, generation,
        source_position, input_token_ids, input_count, input_count - 1u,
        input_count, 1);
}

static int cuda_text_begin_authoritative_output(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows) {
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (output_rows > 1u ||
        cuda_text_begin_depth(backend_state, command, generation,
            source_position, input_token_ids, input_count, input_count - 1u,
            output_rows, 1) != 0 || !cmd)
        return -1;
    cmd->authoritative = 1u;
    return 0;
}

static int cuda_text_begin_frontier(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids,
        const uint32_t *parent_rows, const uint32_t *depths,
        uint32_t input_count) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    uint32_t *staged_parents, *staged_depths;
    uint32_t maximum_depth = 0u;
    if (!state || !state->program || !cmd || !parent_rows || !depths ||
        input_count == 0 || input_count > state->program->maximum_candidates)
        return -1;
    for (uint32_t row = 0; row < input_count; row++)
        if (depths[row] > maximum_depth) maximum_depth = depths[row];
    if (maximum_depth == UINT32_MAX ||
        source_position > state->program->maximum_context -
            (maximum_depth + 1u) ||
        cuda_text_begin_depth(backend_state, command, generation,
            source_position, input_token_ids, input_count, maximum_depth,
            input_count, 0) != 0)
        return -1;
    staged_parents = state->graph_parent_rows +
        (size_t)(input_count - 1u) * CUDA_TEXT_MAX_ROWS;
    staged_depths = state->graph_depths +
        (size_t)(input_count - 1u) * CUDA_TEXT_MAX_ROWS;
    memcpy(staged_parents, parent_rows,
        (size_t)input_count * sizeof(uint32_t));
    memcpy(staged_depths, depths, (size_t)input_count * sizeof(uint32_t));
    state->graph_source_positions[input_count - 1u] = source_position;
    cmd->source_position = source_position;
    if (!cuda_text_full_graph_eligible(state, input_count,
            cmd->authoritative_output_rows, cmd->attention_score_rows) &&
        (cuda_success(cudaMemcpyAsync(cuda_text_bytes(state,
            state->program->layout.target_parent_rows), staged_parents,
            (size_t)input_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
            state->stream), "cudaMemcpy frontier parent rows") != 0 ||
         cuda_success(cudaMemcpyAsync(cuda_text_bytes(state,
            state->program->layout.target_depths), staged_depths,
            (size_t)input_count * sizeof(uint32_t), cudaMemcpyHostToDevice,
            state->stream), "cudaMemcpy frontier depths") != 0 ||
         cuda_success(cudaMemcpyAsync(state->device_source_position,
            &state->graph_source_positions[input_count - 1u],
            sizeof(uint32_t), cudaMemcpyHostToDevice, state->stream),
            "cudaMemcpy frontier source position") != 0))
        return -1;
    return 0;
}

static int cuda_text_launch_status(void) {
    return cuda_success(cudaGetLastError(), "text program launch");
}

static int cuda_text_selected_live(const CudaTextProgramState *state,
        const CudaTextProgramCommand *cmd, uint32_t layer) {
    if (!state || !cmd || cmd->resource_pending ||
        cmd->resource_layer != layer || layer >= state->program->layer_count ||
        !cmd->request_count || cmd->request_count > CUDA_TEXT_EXPERTS ||
        !cuda_selected_resources || !cuda_selected_logical_slots)
        return -1;
    for (uint32_t index = 0; index < cmd->request_count; index++) {
        int32_t expert = cmd->request_experts[index];
        int32_t slot = cmd->request_slots[index];
        const CudaSelectedResource *bound = &cmd->request_bindings[index];
        uint64_t logical;
        if (expert < 0 || expert >= (int32_t)CUDA_TEXT_EXPERTS ||
            slot < 0 || slot >= cuda_selected_capacity)
            return -1;
        logical = (uint64_t)layer * CUDA_TEXT_EXPERTS + (uint32_t)expert;
        if (logical >= (uint64_t)(uint32_t)cuda_selected_logical_capacity ||
            cuda_selected_logical_slots[logical] != slot)
            return -1;
        const CudaSelectedResource *live = &cuda_selected_resources[slot];
        if (!live->active || live->logical_resource_id != logical ||
            live->logical_resource_id != bound->logical_resource_id ||
            live->base != bound->base || live->payload != bound->payload ||
            live->nbytes != bound->nbytes ||
            state->expert_refs[(size_t)logical * 3u].resource != live->payload)
            return -1;
    }
    /* Validated selected-resource reads are about to enter the stream. */
    cuda_selected_retirement_fenced = 0;
    return 0;
}

static void cuda_text_record_matrix(
        CudaTextProgramCommand *cmd, uint32_t output_rows, uint32_t batch) {
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

static int cuda_text_encode_cell(
        void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    const CudaTextCellTemplate *cell_template;
    const SaltTextCanonicalLayout *layout;
    const SaltTextCompiledLayer *compiled = NULL;
    const SaltTextLayerExecDesc *layer = NULL;
    const SaltTextLayerPlan *layer_plan = NULL;
    const SaltAttentionDesc *attention = NULL;
    CudaTextLayerRefs *refs = NULL;
    uint32_t count, jobs, hidden, dense, routed, experts, topk;
    uint32_t output_first = 0u;
    uint32_t physical_launches = 1u;
    float *destination;
    int *status;
    int terminal_output;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
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
        ? input_count : input_count * CUDA_TEXT_TOPK;
    if (terminal_output && cmd->authoritative_output_rows != input_count) {
        count = cmd->authoritative_output_rows;
        output_first = input_count - count;
    }
    if (assignment->gpu.count < count) return -1;
    if (cuda_text_full_graph_eligible(state, input_count,
            cmd->authoritative_output_rows, cmd->attention_score_rows)) {
        cmd->next_cell++;
        cmd->encoded_cells++;
        cmd->highest_completion = cell_template->completion_epoch;
        return 0;
    }
    cmd->stats.backend_template_reuses++;
    layout = &program->layout;
    hidden = cell_template->hidden;
    count = cell->unit == SALT_TEXT_EXECUTION_ROWS
        ? input_count : input_count * CUDA_TEXT_TOPK;
    if (terminal_output && cmd->authoritative_output_rows != input_count)
        count = cmd->authoritative_output_rows;
    if (assignment->gpu.count < count) return -1;
    jobs = input_count * CUDA_TEXT_TOPK;
    destination = cuda_text_f32(state, cell_template->destination_offset);
    status = (int *)(void *)cuda_text_bytes(state, state->status_offset);
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
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING: {
        size_t elements = (size_t)input_count * hidden;
        if (state->embedding.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
            state->embedding.rows != program->vocabulary ||
            state->embedding.cols != hidden)
            return -1;
        cuda_text_embedding_q4<<<
            (unsigned int)((elements + 255u) / 256u), 256, 0, state->stream>>>(
            state->embedding.resource, state->embedding.value_offset,
            state->embedding.scale_offset, state->embedding.bias_offset,
            cuda_text_i32(state, layout->candidate_token_ids), input_count,
            program->vocabulary, hidden, program->descriptor->embedding_scale,
            destination, status);
        break;
    }
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        cuda_text_rmsnorm_rows<<<input_count, 256, 0, state->stream>>>(
            cuda_text_f32(state, layout->state_a), hidden,
            cuda_text_tensor_f32(&refs->pre_attention_norm), destination,
            hidden, input_count, hidden, program->descriptor->norm_epsilon,
            1, status);
        break;
    case SALT_TEXT_CELL_QUERY_PROJECTION: {
        uint32_t group = input_count == 1u
            ? cuda_text_projection_group_size(state, cmd->next_cell) : 0u;
        if (group) {
            if (cuda_text_project_group(state, cmd->next_cell, group) != 0)
                return -1;
        } else if (cuda_text_project(state, &refs->q,
                cuda_text_f32(state, layout->normalized), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    }
    case SALT_TEXT_CELL_KEY_PROJECTION:
        if (input_count == 1u && cmd->next_cell > 0u &&
            cuda_text_projection_group_size(state, cmd->next_cell - 1u))
            physical_launches = 0u;
        else if (cuda_text_project(state, &refs->k,
                cuda_text_f32(state, layout->normalized), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_VALUE_PROJECTION:
        if (input_count == 1u && cmd->next_cell > 1u &&
            cuda_text_projection_group_size(state, cmd->next_cell - 2u) == 3u)
            physical_launches = 0u;
        else if (attention->shared_kv_projection ||
            cuda_text_project(state, &refs->v,
                cuda_text_f32(state, layout->normalized), destination,
                input_count) != 0) return -1;
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
            cuda_text_copy<<<
                (unsigned int)((elements + 255u) / 256u), 256,
                0, state->stream>>>(
                cuda_text_f32(state, layout->keys),
                cuda_text_f32(state, layout->values), elements);
            physical_launches++;
        }
        cuda_text_attention_transform<<<
            input_count * (attention->n_heads + 2u * attention->n_kv_heads),
            256, 0, state->stream>>>(
            attention->n_heads, attention->n_kv_heads, attention->head_dim,
            attention->rope_dim, attention->rope_base_dim,
            (float)attention->rope_theta, state->device_source_position,
            (const uint32_t *)cuda_text_bytes(state,
                layout->target_depths),
            input_count,
            compiled->query_width, compiled->kv_width,
            program->descriptor->norm_epsilon,
            0,
            cuda_text_f32(state, layout->queries),
            cuda_text_f32(state, layout->keys),
            cuda_text_f32(state, layout->values),
            cuda_text_tensor_f32(&refs->q_norm),
            cuda_text_tensor_f32(&refs->k_norm),
            cuda_text_f32(state, compiled->tentative_key_offset),
            cuda_text_f32(state, compiled->tentative_value_offset), status);
        break;
    case SALT_TEXT_CELL_ATTENTION_BODY: {
        uint32_t score_rows = cmd->attention_score_rows;
        size_t score_bytes;
        if (attention->kind != SALT_ATTN_FULL &&
            score_rows > attention->window)
            score_rows = attention->window;
        score_bytes = (size_t)score_rows * sizeof(float);
        if (score_bytes <= state->attention_dynamic_limit)
            cuda_text_attention_body<<<input_count * attention->n_heads, 256,
                score_bytes, state->stream>>>(
                attention->kind == SALT_ATTN_FULL,
                attention->n_heads, attention->n_kv_heads,
                attention->head_dim, attention->window,
                attention->score_scale, state->device_source_position,
                (const uint32_t *)cuda_text_bytes(state,
                    layout->target_parent_rows),
                (const uint32_t *)cuda_text_bytes(state,
                    layout->target_depths), input_count,
                compiled->query_width, compiled->kv_width,
                refs->kv.private_mode, refs->kv.row_capacity,
                cuda_text_f32(state, layout->queries),
                refs->kv.keys, refs->kv.values,
                cuda_text_f32(state, compiled->tentative_key_offset),
                cuda_text_f32(state, compiled->tentative_value_offset),
                destination, status);
        else {
            uint64_t total_tasks =
                (uint64_t)input_count * attention->n_heads;
            size_t wave_capacity;
            uint32_t task_first = 0u;
            if (total_tasks == 0u || total_tasks > UINT32_MAX ||
                score_rows == 0u || !state->attention_score_workspace ||
                state->attention_score_workspace_floats < score_rows)
                return -1;
            wave_capacity = state->attention_score_workspace_floats / score_rows;
            if (wave_capacity > UINT32_MAX) wave_capacity = UINT32_MAX;
            if (wave_capacity > total_tasks) wave_capacity = (size_t)total_tasks;
            if (wave_capacity == 0u) return -1;
            physical_launches = 0u;
            while ((uint64_t)task_first < total_tasks) {
                uint32_t wave_tasks = (uint32_t)(total_tasks - task_first);
                if ((size_t)wave_tasks > wave_capacity)
                    wave_tasks = (uint32_t)wave_capacity;
                cuda_text_attention_scores_global_exact<<<wave_tasks, 256, 0,
                    state->stream>>>(
                    attention->kind == SALT_ATTN_FULL,
                    attention->n_heads, attention->n_kv_heads,
                    attention->head_dim, attention->window,
                    attention->score_scale, state->device_source_position,
                    (const uint32_t *)cuda_text_bytes(state,
                        layout->target_parent_rows),
                    (const uint32_t *)cuda_text_bytes(state,
                        layout->target_depths), input_count,
                    compiled->query_width, compiled->kv_width,
                    refs->kv.private_mode, refs->kv.row_capacity,
                    cuda_text_f32(state, layout->queries), refs->kv.keys,
                    cuda_text_f32(state, compiled->tentative_key_offset),
                    task_first, wave_tasks, score_rows,
                    state->attention_score_workspace, status);
                cuda_text_attention_reduce_global_exact<<<wave_tasks, 256, 0,
                    state->stream>>>(
                    attention->kind == SALT_ATTN_FULL,
                    attention->n_heads, attention->n_kv_heads,
                    attention->head_dim, attention->window,
                    state->device_source_position,
                    (const uint32_t *)cuda_text_bytes(state,
                        layout->target_parent_rows),
                    (const uint32_t *)cuda_text_bytes(state,
                        layout->target_depths), input_count,
                    compiled->query_width, compiled->kv_width,
                    refs->kv.private_mode, refs->kv.row_capacity,
                    refs->kv.values,
                    cuda_text_f32(state, compiled->tentative_value_offset),
                    task_first, wave_tasks, score_rows,
                    state->attention_score_workspace, destination, status);
                if (cuda_text_launch_status() != 0 ||
                    physical_launches > UINT32_MAX - 2u)
                    return -1;
                physical_launches += 2u;
                task_first += wave_tasks;
            }
        }
        break;
    }
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
        if (cuda_text_project(state, &refs->o,
                cuda_text_f32(state, layout->attention_output), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
        cuda_text_residual_postnorm_rows<<<input_count, 256, 0,
            state->stream>>>(cuda_text_f32(state, layout->state_a),
            cuda_text_f32(state, layout->branch),
            cuda_text_tensor_f32(&refs->post_attention_norm), destination,
            input_count, hidden, program->descriptor->norm_epsilon, status);
        break;
    case SALT_TEXT_CELL_DENSE_NORM:
        if (cmd->ffn_norm_wave_layer != UINT32_MAX) return -1;
        cuda_text_rmsnorm_pair_rows<<<dim3(input_count, 2u, 1u), 256,
            0, state->stream>>>(cuda_text_f32(state, layout->state_b), hidden,
            cuda_text_tensor_f32(&refs->pre_ffn_norm_1),
            cuda_text_tensor_f32(&refs->pre_ffn_norm_2),
            cuda_text_f32(state, layout->normalized),
            cuda_text_f32(state, layout->combine_a), hidden,
            input_count, hidden, program->descriptor->norm_epsilon, status);
        cmd->ffn_norm_wave_layer = cell->layer;
        break;
    case SALT_TEXT_CELL_DENSE_GATE: {
        uint32_t group = input_count == 1u
            ? cuda_text_projection_group_size(state, cmd->next_cell) : 0u;
        if (group) {
            if (cuda_text_project_group(state, cmd->next_cell, group) != 0)
                return -1;
        } else if (cuda_text_project(state, &refs->dense_gate,
                cuda_text_f32(state, layout->normalized), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    }
    case SALT_TEXT_CELL_DENSE_UP:
        if (input_count == 1u && cmd->next_cell > 0u &&
            cuda_text_projection_group_size(state, cmd->next_cell - 1u) == 2u)
            physical_launches = 0u;
        else if (cuda_text_project(state, &refs->dense_up,
                cuda_text_f32(state, layout->normalized), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_DENSE_ACTIVATION: {
        size_t elements = (size_t)input_count * dense;
        cuda_text_activate<<<(unsigned int)((elements + 255u) / 256u), 256,
            0, state->stream>>>(cuda_text_f32(state, layout->dense_gate),
            cuda_text_f32(state, layout->dense_up), destination,
            elements, status);
        break;
    }
    case SALT_TEXT_CELL_DENSE_DOWN:
        if (cuda_text_project(state, &refs->dense_down,
                cuda_text_f32(state, layout->dense_chain), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ROUTER_INPUT:
        cuda_text_router_input_rows<<<input_count, 256, 0, state->stream>>>(
            cuda_text_f32(state, layout->state_b),
            cuda_text_tensor_f32(&refs->router_scale), destination,
            input_count, hidden, program->descriptor->norm_epsilon,
            0.0f, status);
        break;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        if (cuda_text_project(state, &refs->router,
                cuda_text_f32(state, layout->router_input), destination,
                input_count) != 0) return -1;
        cmd->stats.projection_dispatches++;
        break;
    case SALT_TEXT_CELL_ROUTED_NORM:
        if (cmd->ffn_norm_wave_layer != cell->layer) return -1;
        cmd->ffn_norm_wave_layer = UINT32_MAX;
        break;
    case SALT_TEXT_CELL_ROUTER_TOPK:
        cuda_text_topk_rows<<<input_count, 1, 0, state->stream>>>(
            cuda_text_f32(state, layout->router_logits),
            cuda_text_tensor_f32(&refs->per_expert_scale),
            cuda_text_i32(state, layout->selected_experts),
            cuda_text_f32(state, layout->selected_weights), input_count,
            experts, topk, status);
        cuda_text_group_maps<<<1, 1, 0, state->stream>>>(
            cuda_text_i32(state, layout->selected_experts), input_count,
            experts, topk,
            cuda_text_i32(state, layout->grouped_to_canonical),
            cuda_text_i32(state, layout->canonical_to_grouped),
            cuda_text_i32(state, layout->tensor_row), status);
        cuda_text_routed_gather<<<
            (unsigned int)(((size_t)jobs * hidden + 255u) / 256u), 256,
            0, state->stream>>>(
            cuda_text_i32(state, layout->grouped_to_canonical),
            cuda_text_f32(state, layout->combine_a),
            cuda_text_f32(state, layout->routed_input), jobs, topk,
            hidden, status);
        break;
    case SALT_TEXT_CELL_EXPERT_GATE:
    case SALT_TEXT_CELL_EXPERT_UP:
    case SALT_TEXT_CELL_EXPERT_DOWN: {
        uint32_t phase = cell->kind == SALT_TEXT_CELL_EXPERT_GATE ? 0u :
            cell->kind == SALT_TEXT_CELL_EXPERT_UP ? 1u : 2u;
        if (jobs == 0u || jobs > state->selected_job_capacity)
            return -1;
        if (state->dynamic_experts &&
            ((!state->capture_mode &&
              cuda_text_selected_live(state, cmd, cell->layer) != 0) ||
             cmd->expert_chain_mask != (phase == 0u ? 0u :
                 phase == 1u ? 1u : 7u)))
            return -1;
        if (jobs > cmd->stats.backend_selected_jobs)
            cmd->stats.backend_selected_jobs = jobs;
        uint32_t rows = phase == 2u ? hidden : routed;
        uint32_t cols = phase == 2u ? routed : hidden;
        const float *input = phase == 2u
            ? cuda_text_f32(state, layout->routed_chain)
            : cuda_text_f32(state, layout->routed_input);
        if (state->dynamic_experts)
            cmd->expert_chain_mask |= phase == 2u ? 8u : (1u << phase);
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
                cuda_text_expert_q4<<<dim3((rows + 3u) / 4u, jobs, 2u),
                    128, 0, state->stream>>>(state->device_experts,
                    cell->layer, phase,
                    cuda_text_i32(state, layout->selected_experts),
                    cuda_text_i32(state, layout->grouped_to_canonical),
                    jobs, topk, input, destination,
                    cuda_text_f32(state,
                        program->dispatch.cells[gate_cell + 1u].destination_offset),
                    rows, cols, status);
            } else physical_launches = 0u;
        } else {
            cuda_text_expert_q4<<<dim3((rows + 3u) / 4u, jobs, 1u), 128,
                0, state->stream>>>(state->device_experts, cell->layer, phase,
                cuda_text_i32(state, layout->selected_experts),
                cuda_text_i32(state, layout->grouped_to_canonical), jobs, topk,
                input, destination, NULL, rows, cols, status);
        }
        if (phase == 2u) cmd->stats.expert_down_dispatches++;
        else cmd->stats.expert_gate_up_dispatches++;
        break;
    }
    case SALT_TEXT_CELL_EXPERT_ACTIVATION: {
        size_t elements = (size_t)jobs * routed;
        if (state->dynamic_experts) {
            if ((!state->capture_mode &&
                 cuda_text_selected_live(state, cmd, cell->layer) != 0) ||
                cmd->expert_chain_mask != 3u)
                return -1;
            cmd->expert_chain_mask |= 4u;
        }
        cuda_text_activate<<<(unsigned int)((elements + 255u) / 256u), 256,
            0, state->stream>>>(cuda_text_f32(state, layout->routed_gate),
            cuda_text_f32(state, layout->routed_up), destination,
            elements, status);
        break;
    }
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
        if (state->dynamic_experts &&
            (cmd->resource_layer != UINT32_MAX || cmd->request_count ||
             cmd->expert_chain_mask))
            return -1;
        cuda_text_expert_reduce<<<input_count, 256, 0, state->stream>>>(
            cuda_text_f32(state, layout->selected_weights),
            cuda_text_i32(state, layout->canonical_to_grouped),
            cuda_text_f32(state, layout->routed_output_jobs), destination,
            input_count, topk, hidden, status);
        break;
    case SALT_TEXT_CELL_FFN_COMBINE:
        cuda_text_parallel_combine_rows<<<input_count, 256, 0,
            state->stream>>>(cuda_text_f32(state, layout->state_b),
            cuda_text_f32(state, layout->dense_output),
            cuda_text_f32(state, layout->routed_output),
            cuda_text_tensor_f32(&refs->post_ffn_norm_1),
            cuda_text_tensor_f32(&refs->post_ffn_norm_2),
            cuda_text_tensor_f32(&refs->post_ffn_norm),
            cuda_text_f32(state, layout->combine_a),
            cuda_text_f32(state, layout->combine_b), destination,
            input_count, hidden, program->descriptor->norm_epsilon, 1.0f,
            layer_plan->final_layer_scale
                ? cuda_text_tensor_f32(&refs->layer_scalar) : NULL,
            status);
        break;
    case SALT_TEXT_CELL_FINAL_NORM:
        if (count > 0)
            cuda_text_rmsnorm_rows<<<count, 256, 0, state->stream>>>(
                cuda_text_f32(state, layout->state_a) +
                    (size_t)output_first * hidden,
                hidden, cuda_text_tensor_f32(&state->final_norm),
                destination + (size_t)output_first * hidden,
                hidden, count, hidden, program->descriptor->norm_epsilon,
                1, status);
        else
            physical_launches = 0u;
        break;
    case SALT_TEXT_CELL_FINAL_HEAD:
        if (count > 0) {
            if (cuda_text_project(state, &state->output_head,
                    cuda_text_f32(state, layout->final_state) +
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
            cuda_text_softcap<<<(unsigned int)((elements + 255u) / 256u), 256,
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
    if (cuda_text_launch_status() != 0) return -1;
    if (cell->kind == SALT_TEXT_CELL_ROUTED_NORM) {
        physical_launches = 0u;
    } else if (cell->kind == SALT_TEXT_CELL_ROUTER_TOPK) {
        physical_launches = 3u;
    } else {
        const CudaTextTensorRef *projection = NULL;
        switch (cell->kind) {
        case SALT_TEXT_CELL_QUERY_PROJECTION: projection = &refs->q; break;
        case SALT_TEXT_CELL_KEY_PROJECTION: projection = &refs->k; break;
        case SALT_TEXT_CELL_VALUE_PROJECTION:
            if (!attention->shared_kv_projection) projection = &refs->v;
            else physical_launches = 0u;
            break;
        case SALT_TEXT_CELL_OUTPUT_PROJECTION: projection = &refs->o; break;
        case SALT_TEXT_CELL_DENSE_GATE: projection = &refs->dense_gate; break;
        case SALT_TEXT_CELL_DENSE_UP: projection = &refs->dense_up; break;
        case SALT_TEXT_CELL_DENSE_DOWN: projection = &refs->dense_down; break;
        case SALT_TEXT_CELL_ROUTER_PROJECTION: projection = &refs->router; break;
        case SALT_TEXT_CELL_FINAL_HEAD: projection = &state->output_head; break;
        default: break;
        }
        if (projection && projection->encoding == SALT_TENSOR_ENCODING_BF16)
            physical_launches = input_count;
    }
    if (UINT32_MAX - cmd->stats.backend_physical_kernel_nodes <
            physical_launches ||
        UINT32_MAX - cmd->stats.backend_host_kernel_launch_calls <
            physical_launches)
        return -1;
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
        cuda_text_record_matrix(cmd,
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

static int cuda_text_dependency_barrier(
        void *backend_state, void *command,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->submitted ||
        cmd->next_cell >= state->program->dispatch.cell_count ||
        dependency_epoch == 0 || completion_epoch <= dependency_epoch ||
        dependency_epoch > cmd->highest_completion)
        return -1;
    const SaltTextExecutionCell *upcoming =
        &state->program->dispatch.cells[cmd->next_cell];
    if (upcoming->dependency_epoch != dependency_epoch ||
        upcoming->completion_epoch != completion_epoch)
        return -1;
    if (state->dynamic_experts) {
        if (state->graphs_ready || state->capture_mode) return -1;
        if (cmd->resource_pending) {
            if (upcoming->kind != SALT_TEXT_CELL_EXPERT_GATE ||
                upcoming->layer != cmd->resource_layer)
                return -1;
            return SALT_TEXT_GPU_NEED_RESOURCE;
        }
        if (upcoming->kind == SALT_TEXT_CELL_EXPERT_GATE &&
            cmd->resource_layer == UINT32_MAX) {
            const SaltTextCanonicalLayout *layout = &state->program->layout;
            const SaltTextLayerPlan *plan =
                state->program->layers[upcoming->layer].descriptor->plan;
            unsigned char *host = (unsigned char *)state->canonical.contents;
            uint32_t occupancies[CUDA_TEXT_EXPERTS] = {0};
            uint32_t jobs;
            size_t bytes;
            if (!plan || plan->n_experts != (int)CUDA_TEXT_EXPERTS ||
                plan->top_k_experts != (int)CUDA_TEXT_TOPK ||
                upcoming->layer >= state->program->layer_count ||
                cmd->input_count > UINT32_MAX / CUDA_TEXT_TOPK)
                return -1;
            jobs = cmd->input_count * CUDA_TEXT_TOPK;
            bytes = (size_t)jobs * sizeof(int32_t);
            if (!jobs || jobs > state->selected_job_capacity ||
                layout->selected_experts > state->canonical_bytes ||
                bytes > state->canonical_bytes - layout->selected_experts ||
                cmd->request_count || cmd->expert_chain_mask ||
                cuda_success(cudaMemcpyAsync(
                    host + layout->selected_experts,
                    cuda_text_bytes(state, layout->selected_experts), bytes,
                    cudaMemcpyDeviceToHost, state->stream),
                    "cudaMemcpy text selected IDs") != 0 ||
                cuda_success(cudaStreamSynchronize(state->stream),
                    "cudaStreamSynchronize text resource prefix") != 0)
                return -1;
            const int32_t *selected = (const int32_t *)(const void *)(
                host + layout->selected_experts);
            for (uint32_t row = 0; row < cmd->input_count; row++)
                for (uint32_t rank = 0; rank < CUDA_TEXT_TOPK; rank++) {
                    uint32_t canonical = row * CUDA_TEXT_TOPK + rank;
                    int32_t expert = selected[canonical];
                    if (expert < 0 || expert >= (int32_t)CUDA_TEXT_EXPERTS ||
                        (rank > 0 && expert <= selected[canonical - 1u]))
                        return -1;
                    occupancies[(uint32_t)expert]++;
                }
            cmd->request_count = 0;
            for (uint32_t expert = 0; expert < CUDA_TEXT_EXPERTS; expert++) {
                uint32_t population = occupancies[expert];
                if (!population) continue;
                if (cmd->request_count >= CUDA_TEXT_EXPERTS)
                    return -1;
                cmd->request_experts[cmd->request_count++] = (int32_t)expert;
            }
            if (cmd->request_count == 0)
                return -1;
            /* This stream synchronization already fenced every prior
             * selected-resource consumer.  The immediately following cache
             * acquire may consume that proof instead of issuing a redundant
             * device-wide synchronization before slot retirement. */
            cuda_text_resource_fence_credit = 1;
            cmd->resource_layer = upcoming->layer;
            cmd->resource_pending = 1u;
            return SALT_TEXT_GPU_NEED_RESOURCE;
        }
        if (upcoming->kind == SALT_TEXT_CELL_EXPERT_GATE) {
            if (cmd->resource_layer != upcoming->layer ||
                !cmd->request_count || cmd->expert_chain_mask)
                return -1;
        } else if (upcoming->kind == SALT_TEXT_CELL_EXPERT_REDUCTION) {
            if (cmd->resource_layer != upcoming->layer ||
                !cmd->request_count || cmd->expert_chain_mask != 15u)
                return -1;
            /* The portable executor retains this layer's SaltCache leases
             * through reduction/combine.  The next layer's resource-prefix
             * synchronization therefore proves every queued expert consumer
             * complete before heterogeneous_resume_resource() releases those
             * leases.  On the final layer, cuda_text_finish() supplies the same
             * proof before resolve releases them.  Do not add a second stream
             * synchronization between expert down and reduction. */
            cmd->resource_layer = UINT32_MAX;
            cmd->request_count = 0;
            cmd->expert_chain_mask = 0;
            memset(cmd->request_slots, 0, sizeof cmd->request_slots);
            memset(cmd->request_bindings, 0, sizeof cmd->request_bindings);
        }
    }
    cmd->barrier_count++;
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int cuda_text_resource_request(
        void *backend_state, void *command, uint32_t *layer,
        int32_t *experts, uint32_t expert_capacity, uint32_t *expert_count) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC ||
        !state->dynamic_experts || !cmd || !cmd->begun || cmd->submitted ||
        !cmd->resource_pending || !layer || !experts || !expert_count ||
        !cmd->request_count || cmd->request_count > expert_capacity ||
        cmd->request_count > CUDA_TEXT_EXPERTS)
        return -1;
    *layer = cmd->resource_layer;
    *expert_count = cmd->request_count;
    memcpy(experts, cmd->request_experts,
        (size_t)cmd->request_count * sizeof *experts);
    return 0;
}

static int cuda_text_resource_resume(
        void *backend_state, void *command, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t expert_count) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    CudaTextTensorRef staged[CUDA_TEXT_EXPERTS][3];
    uint32_t changed = 0;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC ||
        !state->dynamic_experts || !cmd || !cmd->begun || cmd->submitted ||
        !cmd->resource_pending || layer != cmd->resource_layer ||
        layer >= state->program->layer_count || !experts || !slots ||
        !expert_count || expert_count != cmd->request_count ||
        expert_count > CUDA_TEXT_EXPERTS || cmd->expert_chain_mask ||
        cmd->next_cell >= state->template_count ||
        state->templates[cmd->next_cell].kind != SALT_TEXT_CELL_EXPERT_GATE ||
        state->templates[cmd->next_cell].layer != layer)
        return -1;
    cuda_text_resource_fence_credit = 0;
    memset(staged, 0, sizeof staged);
    for (uint32_t index = 0; index < expert_count; index++) {
        int32_t expert = experts[index];
        if (expert != cmd->request_experts[index] || expert < 0 ||
            expert >= (int32_t)CUDA_TEXT_EXPERTS ||
            cuda_text_selected_refs(state->program, layer,
                (uint32_t)expert, slots[index], staged[index]) != 0)
            return -1;
    }
    for (uint32_t index = 0; index < expert_count; index++) {
        size_t base = ((size_t)layer * CUDA_TEXT_EXPERTS +
            (uint32_t)experts[index]) * 3u;
        if (memcmp(state->expert_refs + base, staged[index],
                sizeof staged[index]) != 0) {
            memcpy(state->expert_refs + base, staged[index],
                sizeof staged[index]);
            changed++;
        }
        cmd->request_slots[index] = slots[index];
        cmd->request_bindings[index] = cuda_selected_resources[slots[index]];
    }
    if (changed) {
        size_t base = (size_t)layer * CUDA_TEXT_EXPERTS * 3u;
        size_t count = (size_t)CUDA_TEXT_EXPERTS * 3u;
        if (cuda_success(cudaMemcpyAsync(state->device_experts + base,
                state->expert_refs + base,
                count * sizeof *state->expert_refs,
                cudaMemcpyHostToDevice, state->stream),
                "cudaMemcpy text selected layer refs") != 0)
            return -1;
    }
    cmd->stats.backend_dynamic_patches += changed * 3u;
    cmd->resource_pending = 0;
    return 0;
}

static int cuda_text_extent_stats_add(
        CudaTextProgramCommand *cmd, const CudaTextExtentGraph *graph) {
#define ADD_U32(field) do { \
    if (UINT32_MAX - cmd->stats.field < graph->stats.field) return -1; \
    cmd->stats.field += graph->stats.field; \
} while (0)
#define ADD_U64(field) do { \
    if (UINT64_MAX - cmd->stats.field < graph->stats.field) return -1; \
    cmd->stats.field += graph->stats.field; \
} while (0)
    if (!cmd || !graph || !graph->executable || graph->kernel_nodes == 0u)
        return -1;
    ADD_U32(projection_dispatches);
    ADD_U32(expert_gate_up_dispatches);
    ADD_U32(expert_down_dispatches);
    ADD_U32(area_m1_dispatches);
    ADD_U32(area_mk_dispatches);
    ADD_U32(area_mn_dispatches);
    ADD_U32(area_matrix_parallel_dispatches);
    ADD_U64(area_output_row_tiles);
    ADD_U64(area_candidate_output_tiles);
    ADD_U32(backend_template_reuses);
    if (graph->stats.backend_selected_jobs > cmd->stats.backend_selected_jobs)
        cmd->stats.backend_selected_jobs = graph->stats.backend_selected_jobs;
    if (UINT32_MAX - cmd->stats.backend_physical_kernel_nodes <
            graph->kernel_nodes ||
        cmd->stats.backend_graph_launches == UINT32_MAX ||
        UINT32_MAX - cmd->stats.backend_graph_parameter_patches <
            graph->attention_nodes ||
        cmd->stats.backend_host_kernel_launch_calls == UINT32_MAX)
        return -1;
    cmd->stats.backend_physical_kernel_nodes += graph->kernel_nodes;
    cmd->stats.backend_graph_launches++;
    cmd->stats.backend_graph_parameter_patches += graph->attention_nodes;
    cmd->stats.backend_host_kernel_launch_calls++;
#undef ADD_U32
#undef ADD_U64
    return 0;
}

static int cuda_text_encode_extent(void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        uint32_t first_cell, uint32_t cell_count, uint32_t input_count,
        uint32_t *encoded_cells) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd || !program ||
        !plan || !encoded_cells || plan->program != program ||
        plan->assignment_count != program->dispatch.cell_count ||
        first_cell != cmd->next_cell || input_count != cmd->input_count ||
        cell_count == 0 ||
        cell_count > program->dispatch.cell_count - first_cell ||
        first_cell >= state->template_count ||
        cell_count > state->template_count - first_cell)
        return -1;
    *encoded_cells = 0;
    if (state->extent_graphs_ready && !state->capture_mode &&
        input_count <= CUDA_TEXT_GRAPH_COUNT &&
        cmd->authoritative_output_rows == input_count) {
        CudaTextExtentGraph *graph = NULL;
        const SaltTextExecutionCell *first =
            &program->dispatch.cells[first_cell];
        const SaltTextExecutionCell *last =
            &program->dispatch.cells[first_cell + cell_count - 1u];
        uint32_t barriers =
            plan->assignments[first_cell].gpu_traversal_barriers;
        for (uint32_t extent = 0; extent < state->extent_graph_count; extent++) {
            CudaTextExtentGraph *candidate =
                &state->extent_graphs[input_count - 1u][extent];
            if (candidate->first_cell == first_cell &&
                candidate->cell_count == cell_count) {
                graph = candidate;
                break;
            }
        }
        if (!graph || !graph->executable ||
            last->completion_epoch <= cmd->highest_completion)
            return -1;
        if (first->dependency_epoch > 0u) {
            if (barriers == 0u) return -1;
            barriers--;
        }
        if (UINT32_MAX - cmd->barrier_count < barriers)
            return -1;
        if (first->kind == SALT_TEXT_CELL_EXPERT_GATE &&
            (cuda_text_selected_live(state, cmd, first->layer) != 0 ||
             cmd->expert_chain_mask != 0u))
            return -1;
        if (graph->attention_nodes) {
            cudaKernelNodeParams parameters;
            if (graph->attention_nodes != 1u || !graph->attention_node ||
                cuda_success(cudaGraphKernelNodeGetParams(
                    graph->attention_node, &parameters),
                    "cudaGraphKernelNodeGetParams text extent attention") != 0)
                return -1;
            parameters.sharedMemBytes =
                (size_t)cmd->attention_score_rows * sizeof(float);
            if (cuda_success(cudaGraphExecKernelNodeSetParams(
                    graph->executable, graph->attention_node, &parameters),
                    "cudaGraphExecKernelNodeSetParams text extent attention") != 0)
                return -1;
        }
        if (cuda_success(cudaGraphLaunch(graph->executable, state->stream),
                "cudaGraphLaunch text extent") != 0 ||
            cuda_text_extent_stats_add(cmd, graph) != 0)
            return -1;
        if (first->kind == SALT_TEXT_CELL_EXPERT_GATE)
            cmd->expert_chain_mask = 15u;
        cmd->barrier_count += barriers;
        cmd->stats.backend_graph_count = state->extent_graphs_ready;
        cmd->next_cell += cell_count;
        cmd->encoded_cells += cell_count;
        cmd->highest_completion = last->completion_epoch;
        *encoded_cells = cell_count;
        return SALT_TEXT_GPU_DEPENDENCY_READY;
    }
    if (cuda_text_full_graph_eligible(state, input_count,
            cmd->authoritative_output_rows, cmd->attention_score_rows)) {
        const SaltTextExecutionCell *first =
            &program->dispatch.cells[first_cell];
        const SaltTextExecutionCell *last =
            &program->dispatch.cells[first_cell + cell_count - 1u];
        uint32_t barriers =
            plan->assignments[first_cell].gpu_traversal_barriers;
        if (first->dependency_epoch > 0) barriers--;
        if (UINT32_MAX - cmd->barrier_count < barriers ||
            last->completion_epoch <= cmd->highest_completion)
            return -1;
        cmd->barrier_count += barriers;
        cmd->next_cell += cell_count;
        cmd->encoded_cells += cell_count;
        cmd->highest_completion = last->completion_epoch;
        *encoded_cells = cell_count;
        return SALT_TEXT_GPU_DEPENDENCY_READY;
    }
    for (uint32_t offset = 0; offset < cell_count; offset++) {
        uint32_t index = first_cell + offset;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        if (offset > 0 && cell->dependency_epoch > 0) {
            int barrier = cuda_text_dependency_barrier(backend_state, command,
                cell->dependency_epoch, cell->completion_epoch);
            if (barrier == SALT_TEXT_GPU_NEED_RESOURCE) {
                *encoded_cells = offset;
                return barrier;
            }
            if (barrier != SALT_TEXT_GPU_DEPENDENCY_READY) return -1;
        }
        if (cuda_text_encode_cell(backend_state, command, program, cell,
                &plan->assignments[index], input_count) != 0)
            return -1;
        *encoded_cells = offset + 1u;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int cuda_text_submit(void *backend_state, void *command) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        !cmd->begun || cmd->submitted ||
        (state->dynamic_experts &&
         (cmd->resource_pending || cmd->request_count ||
          cmd->expert_chain_mask || cmd->resource_layer != UINT32_MAX)) ||
        cmd->next_cell != state->program->dispatch.cell_count ||
        cmd->encoded_cells != state->program->dispatch.cell_count)
        return -1;
    if (cuda_text_full_graph_eligible(state, cmd->input_count,
            cmd->authoritative_output_rows, cmd->attention_score_rows)) {
        CudaTextGraphTemplate *graph = &state->graphs[cmd->input_count - 1u];
        if (!graph->executable || graph->kernel_nodes == 0u ||
            graph->attention_node_count != state->program->layer_count)
            return -1;
        for (uint32_t layer = 0; layer < graph->attention_node_count; layer++) {
            cudaKernelNodeParams parameters;
            if (cuda_success(cudaGraphKernelNodeGetParams(
                    graph->attention_nodes[layer], &parameters),
                    "cudaGraphKernelNodeGetParams text attention") != 0)
                return -1;
            parameters.sharedMemBytes =
                (unsigned int)((size_t)cmd->attention_score_rows * sizeof(float));
            if (cuda_success(cudaGraphExecKernelNodeSetParams(
                    graph->executable, graph->attention_nodes[layer],
                    &parameters),
                    "cudaGraphExecKernelNodeSetParams text attention") != 0)
                return -1;
        }
        if (cuda_success(cudaGraphLaunch(graph->executable, state->stream),
                "cudaGraphLaunch text") != 0)
            return -1;
        cmd->stats = graph->stats;
        cmd->stats.backend_graph_count = state->graphs_ready;
        cmd->stats.backend_graph_launches = 1u;
        cmd->stats.backend_graph_parameter_patches =
            1u + graph->attention_node_count;
        cmd->stats.backend_physical_kernel_nodes = graph->kernel_nodes;
        cmd->stats.backend_host_kernel_launch_calls = 1u;
    }
    if (cmd->authoritative) {
        uint32_t maximum_width = 0u;
        for (uint32_t layer = 0; layer < state->program->layer_count; layer++)
            if (state->layers[layer].kv.width > maximum_width)
                maximum_width = state->layers[layer].kv.width;
        if (!state->device_kv || maximum_width == 0u ||
            state->program->layer_count == 0u ||
            state->program->layer_count > CUDA_TEXT_LAYERS)
            return -1;
        cuda_text_publish_authoritative_kv<<<
            dim3((maximum_width + 255u) / 256u, cmd->input_count,
                 state->program->layer_count), 256, 0, state->stream>>>(
                state->device_kv, state->program->layer_count,
                (const float *)state->canonical.backend,
                cmd->source_position, cmd->input_count,
                (const int *)cuda_text_bytes(state, state->status_offset));
        if (cuda_text_launch_status() != 0)
            return -1;
        cmd->stats.backend_physical_kernel_nodes++;
        cmd->stats.backend_host_kernel_launch_calls++;
    }
    cmd->submitted = 1;
    return 0;
}

static int cuda_text_capture_nodes(CudaTextGraphTemplate *target) {
    cudaGraphNode_t nodes[CUDA_TEXT_GRAPH_MAX_NODES];
    size_t node_count = CUDA_TEXT_GRAPH_MAX_NODES;
    if (!target || !target->graph ||
        cuda_success(cudaGraphGetNodes(target->graph, nodes, &node_count),
            "cudaGraphGetNodes text") != 0 ||
        node_count == 0 || node_count > CUDA_TEXT_GRAPH_MAX_NODES)
        return -1;
    target->kernel_nodes = 0;
    target->attention_node_count = 0;
    for (size_t index = 0; index < node_count; index++) {
        cudaGraphNodeType type;
        if (cuda_success(cudaGraphNodeGetType(nodes[index], &type),
                "cudaGraphNodeGetType text") != 0)
            return -1;
        if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams parameters;
            target->kernel_nodes++;
            if (cuda_success(cudaGraphKernelNodeGetParams(
                    nodes[index], &parameters),
                    "cudaGraphKernelNodeGetParams text capture") != 0)
                return -1;
            if (parameters.func == (void *)cuda_text_attention_body) {
                if (target->attention_node_count >= CUDA_TEXT_LAYERS)
                    return -1;
                target->attention_nodes[target->attention_node_count++] =
                    nodes[index];
            }
        }
    }
    return target->kernel_nodes > 0u &&
        target->attention_node_count == CUDA_TEXT_LAYERS ? 0 : -1;
}

static int cuda_text_capture_graphs(
        CudaTextProgramState *state, const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan) {
    if (!state || !program || !plan || plan->program != program ||
        plan->assignment_count != program->dispatch.cell_count ||
        !state->graph_tokens || !state->graph_parent_rows ||
        !state->graph_depths || !state->graph_source_positions ||
        !state->device_source_position || !state->stream)
        return -1;
    memset(state->graph_tokens, 0,
        (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
            sizeof(int32_t));
    memset(state->graph_parent_rows, 0,
        (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
            sizeof(uint32_t));
    memset(state->graph_depths, 0,
        (size_t)CUDA_TEXT_MAX_ROWS * CUDA_TEXT_MAX_ROWS *
            sizeof(uint32_t));
    memset(state->graph_source_positions, 0,
        (size_t)CUDA_TEXT_MAX_ROWS * sizeof(uint32_t));
    state->capture_mode = 1u;
    for (uint32_t batch = 1u; batch <= CUDA_TEXT_GRAPH_COUNT; batch++) {
        CudaTextProgramCommand command;
        CudaTextGraphTemplate *target = &state->graphs[batch - 1u];
        const int32_t *tokens = state->graph_tokens +
            (size_t)(batch - 1u) * CUDA_TEXT_MAX_ROWS;
        memset(&command, 0, sizeof command);
        if (cuda_success(cudaStreamBeginCapture(
                state->stream, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture text") != 0)
            goto fail;
        if (cuda_text_begin(state, &command, 1u, 0u, tokens, batch) != 0)
            goto capture_fail;
        for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
            const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
            if (cell->dependency_epoch > 0u &&
                cuda_text_dependency_barrier(state, &command,
                    cell->dependency_epoch, cell->completion_epoch) !=
                        SALT_TEXT_GPU_DEPENDENCY_READY)
                goto capture_fail;
            if (cuda_text_encode_cell(state, &command, program, cell,
                    &plan->assignments[index], batch) != 0)
                goto capture_fail;
        }
        if (cuda_text_submit(state, &command) != 0)
            goto capture_fail;
        if (cuda_success(cudaStreamEndCapture(state->stream, &target->graph),
                "cudaStreamEndCapture text") != 0 || !target->graph)
            goto fail;
        target->stats = command.stats;
        if (cuda_text_capture_nodes(target) != 0 ||
            cuda_success(cudaGraphInstantiate(
                &target->executable, target->graph, NULL, NULL, 0),
                "cudaGraphInstantiate text") != 0 || !target->executable)
            goto fail;
    }
    state->capture_mode = 0u;
    state->graphs_ready = CUDA_TEXT_GRAPH_COUNT;
    return 0;
capture_fail:
    {
        cudaGraph_t abandoned = NULL;
        (void)cudaStreamEndCapture(state->stream, &abandoned);
        if (abandoned) (void)cudaGraphDestroy(abandoned);
    }
fail:
    state->capture_mode = 0u;
    return -1;
}

static int cuda_text_extent_capture_nodes(CudaTextExtentGraph *target) {
    cudaGraphNode_t nodes[CUDA_TEXT_GRAPH_MAX_NODES];
    size_t node_count = CUDA_TEXT_GRAPH_MAX_NODES;
    if (!target || !target->graph ||
        cuda_success(cudaGraphGetNodes(target->graph, nodes, &node_count),
            "cudaGraphGetNodes text extent") != 0 ||
        node_count == 0u || node_count > CUDA_TEXT_GRAPH_MAX_NODES)
        return -1;
    target->kernel_nodes = 0u;
    target->attention_nodes = 0u;
    target->attention_node = NULL;
    for (size_t index = 0; index < node_count; index++) {
        cudaGraphNodeType type;
        if (cuda_success(cudaGraphNodeGetType(nodes[index], &type),
                "cudaGraphNodeGetType text extent") != 0)
            return -1;
        if (type != cudaGraphNodeTypeKernel) continue;
        cudaKernelNodeParams parameters;
        target->kernel_nodes++;
        if (cuda_success(cudaGraphKernelNodeGetParams(nodes[index], &parameters),
                "cudaGraphKernelNodeGetParams text extent capture") != 0)
            return -1;
        if (parameters.func == (void *)cuda_text_attention_body) {
            if (target->attention_nodes != 0u) return -1;
            target->attention_node = nodes[index];
            target->attention_nodes = 1u;
        }
    }
    return target->kernel_nodes > 0u ? 0 : -1;
}

static int cuda_text_capture_extent_graphs(
        CudaTextProgramState *state, const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan) {
    uint32_t extent_first[CUDA_TEXT_MAX_EXTENTS];
    uint32_t extent_count[CUDA_TEXT_MAX_EXTENTS];
    uint32_t extents = 0u, cursor = 0u;
    if (!state || !program || !plan || plan->program != program ||
        !state->dynamic_experts || !cuda_pageable_mmap ||
        plan->assignment_count != program->dispatch.cell_count ||
        !state->stream)
        return -1;
    while (cursor < plan->assignment_count) {
        uint32_t count = plan->assignments[cursor].gpu_traversal_count;
        if (count == 0u || count > plan->assignment_count - cursor ||
            extents >= CUDA_TEXT_MAX_EXTENTS)
            return -1;
        extent_first[extents] = cursor;
        extent_count[extents] = count;
        extents++;
        cursor += count;
    }
    if (cursor != plan->assignment_count || extents == 0u)
        return -1;
    state->capture_mode = 1u;
    for (uint32_t batch = 1u; batch <= CUDA_TEXT_GRAPH_COUNT; batch++) {
        for (uint32_t extent = 0u; extent < extents; extent++) {
            CudaTextExtentGraph *target =
                &state->extent_graphs[batch - 1u][extent];
            CudaTextProgramCommand command;
            uint32_t first = extent_first[extent];
            uint32_t count = extent_count[extent];
            const SaltTextExecutionCell *first_cell =
                &program->dispatch.cells[first];
            memset(&command, 0, sizeof command);
            command.generation = 1u;
            command.input_count = batch;
            command.maximum_depth = batch - 1u;
            command.authoritative_output_rows = batch;
            command.attention_score_rows = program->maximum_context;
            command.next_cell = first;
            command.highest_completion = first_cell->dependency_epoch;
            command.ffn_norm_wave_layer = UINT32_MAX;
            command.resource_layer = UINT32_MAX;
            command.begun = 1u;
            if (first_cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
                command.resource_layer = first_cell->layer;
                command.request_count = 1u;
                command.request_experts[0] = 0;
                command.request_slots[0] = 0;
            }
            if (cuda_success(cudaStreamBeginCapture(
                    state->stream, cudaStreamCaptureModeThreadLocal),
                    "cudaStreamBeginCapture text extent") != 0)
                goto fail;
            for (uint32_t offset = 0u; offset < count; offset++) {
                uint32_t index = first + offset;
                if (cuda_text_encode_cell(state, &command, program,
                        &program->dispatch.cells[index],
                        &plan->assignments[index], batch) != 0)
                    goto capture_fail;
            }
            if (command.next_cell != first + count ||
                command.encoded_cells != count ||
                cuda_success(cudaStreamEndCapture(state->stream, &target->graph),
                    "cudaStreamEndCapture text extent") != 0 ||
                !target->graph)
                goto fail;
            target->first_cell = first;
            target->cell_count = count;
            target->stats = command.stats;
            if (cuda_text_extent_capture_nodes(target) != 0 ||
                cuda_success(cudaGraphInstantiate(
                    &target->executable, target->graph, NULL, NULL, 0),
                    "cudaGraphInstantiate text extent") != 0 ||
                !target->executable)
                goto fail;
        }
    }
    state->capture_mode = 0u;
    state->extent_graph_count = extents;
    if (extents > UINT32_MAX / CUDA_TEXT_GRAPH_COUNT) goto fail;
    state->extent_graphs_ready = extents * CUDA_TEXT_GRAPH_COUNT;
    return 0;
capture_fail:
    {
        cudaGraph_t abandoned = NULL;
        (void)cudaStreamEndCapture(state->stream, &abandoned);
        if (abandoned) (void)cudaGraphDestroy(abandoned);
    }
fail:
    state->capture_mode = 0u;
    return -1;
}

static int cuda_text_finish(
        void *backend_state, void *command,
        SaltTextExecutionView *view, SaltTextVerifyBackendStats *stats) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    const int *status;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        !cmd->submitted || cmd->finished || !view || !stats ||
        !state->canonical.contents || !state->canonical.backend)
        return -1;
    if (cmd->authoritative) {
        uint32_t rows = cmd->authoritative_output_rows;
        size_t offset = state->program->layout.position_logits +
            (size_t)(cmd->input_count - rows) * state->program->vocabulary *
                sizeof(float);
        size_t bytes = (size_t)rows * state->program->vocabulary *
            sizeof(float);
        if (offset > state->canonical_bytes ||
            bytes > state->canonical_bytes - offset ||
            (bytes > 0 && cuda_success(cudaMemcpyAsync(
                (unsigned char *)state->canonical.contents + offset,
                (const unsigned char *)state->canonical.backend + offset,
                bytes, cudaMemcpyDeviceToHost, state->stream),
                "cudaMemcpy text authoritative logits") != 0))
            return -1;
    } else {
        for (uint32_t index = 0; index < cmd->touched_span_count; index++) {
            const SaltTextTouchedSpan *span = &cmd->touched_spans[index];
            if (span->offset > state->canonical_bytes ||
                span->bytes > state->canonical_bytes - span->offset ||
                cuda_success(cudaMemcpyAsync(
                    (unsigned char *)state->canonical.contents + span->offset,
                    (const unsigned char *)state->canonical.backend + span->offset,
                    span->bytes, cudaMemcpyDeviceToHost, state->stream),
                    "cudaMemcpy text touched span") != 0)
                return -1;
        }
    }
    if (cuda_success(cudaMemcpyAsync(
            (unsigned char *)state->canonical.contents + state->status_offset,
            (const unsigned char *)state->canonical.backend +
                state->status_offset,
            sizeof(int), cudaMemcpyDeviceToHost, state->stream),
            "cudaMemcpy text status") != 0 ||
        cuda_success(cudaStreamSynchronize(state->stream),
            "cudaStreamSynchronize text program") != 0)
        return -1;
    status = (const int *)((const unsigned char *)state->canonical.contents +
        state->status_offset);
    if (*status != 0) {
        fprintf(stderr,
            "gpu-cuda: text finish status=%d source=%u input=%u scores=%u\n",
            *status, cmd->source_position, cmd->input_count,
            cmd->attention_score_rows);
        return -1;
    }
    cmd->stats.final_logits_transfer_bytes =
        (uint64_t)(cmd->authoritative
            ? cmd->authoritative_output_rows : cmd->input_count) *
        state->program->vocabulary * sizeof(float);
    view->canonical_base = (unsigned char *)state->canonical.contents;
    view->canonical_bytes = state->canonical_bytes;
    view->backend_published_kv = cmd->authoritative;
    view->reserved = 0u;
    *stats = cmd->stats;
    cmd->finished = 1;
    return 0;
}

static int cuda_text_resolve(
        void *backend_state, void *command,
        uint32_t committed_count, uint32_t input_count,
        uint64_t *scrubbed_bytes) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        !cmd->submitted || !cmd->finished || !scrubbed_bytes ||
        input_count != cmd->input_count || committed_count > input_count)
        return -1;
    *scrubbed_bytes = 0;
    cuda_text_resource_fence_credit = 0;
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static int cuda_text_scrub(
        void *backend_state, void *command, uint64_t *scrubbed_bytes) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        !scrubbed_bytes ||
        cuda_success(cudaStreamSynchronize(state->stream),
            "cudaStreamSynchronize text scrub") != 0)
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
    cuda_text_resource_fence_credit = 0;
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static int cuda_text_destroy(void *backend_state, void *command) {
    CudaTextProgramState *state = (CudaTextProgramState *)backend_state;
    CudaTextProgramCommand *cmd = (CudaTextProgramCommand *)command;
    if (!state || state->magic != CUDA_TEXT_STATE_MAGIC || !cmd ||
        cmd->begun || cmd->submitted)
        return -1;
    cuda_text_state_release(state);
    memset(cmd, 0, sizeof *cmd);
    return 0;
}

static const SaltTextGpuProgramOps cuda_text_program_ops = {
    cuda_text_requirements,
    cuda_text_prepare,
    cuda_text_begin,
    cuda_text_encode_cell,
    cuda_text_encode_extent,
    cuda_text_dependency_barrier,
    cuda_text_resource_request,
    cuda_text_resource_resume,
    cuda_text_submit,
    cuda_text_finish,
    cuda_text_resolve,
    cuda_text_scrub,
    cuda_text_destroy,
    cuda_text_begin_frontier,
    cuda_text_begin_authoritative,
    cuda_text_begin_authoritative_output,
};

extern "C" const struct SaltTextGpuProgramOps *
salt_gpu_tensor_program_ops(void) {
    return cuda_ready ? &cuda_text_program_ops : NULL;
}
extern "C" const SaltGpuDispatch *salt_dispatch_get(void) { return NULL; }
extern "C" const SaltGpuResidencyBackendOps *
salt_gpu_residency_backend_ops(void) { return NULL; }
extern "C" int salt_gpu_cuda_present(void) { return 1; }
extern "C" int salt_gpu_rocm_present(void) { return 0; }
extern "C" int salt_gpu_pageable_mmap_active(void) {
    return cuda_pageable_mmap;
}
