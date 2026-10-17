#include "lfm_attention.h"
#include "lfm.h"

#if defined(LFM_GPU_HIP)
#include "gpu_hip_text_kv.hpp"
#include <hip/hip_runtime.h>
#define cudaStream_t hipStream_t
#define cudaGetLastError hipGetLastError
#define cudaSuccess hipSuccess
#define LFM_ATTENTION_GPU_BACKEND SALT_TEXT_MODEL_GPU_HIP
#else
#include "gpu_cuda_text_kv.cuh"
#include <cuda_runtime.h>
#define LFM_ATTENTION_GPU_BACKEND SALT_TEXT_MODEL_GPU_CUDA
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define SALT_BITMATH_EMBEDDED 1
#define SALT_BITMATH_API __device__ static
#define SALT_BITMATH_LOCAL __device__ static
#define salt_expf lfm_attention_owned_expf
#define salt_logf lfm_attention_owned_logf
#define salt_powf lfm_attention_owned_powf
#define salt_rsqrtf lfm_attention_owned_rsqrtf
#define salt_sqrtf lfm_attention_owned_sqrtf
#define salt_tanhf lfm_attention_owned_tanhf
#define salt_sinf lfm_attention_owned_sinf
#define salt_cosf lfm_attention_owned_cosf
#include "../../src/bitmath.c"
#undef salt_cosf
#undef salt_sinf
#undef salt_tanhf
#undef salt_sqrtf
#undef salt_rsqrtf
#undef salt_powf
#undef salt_logf
#undef salt_expf
#undef SALT_BITMATH_LOCAL
#undef SALT_BITMATH_API
#undef SALT_BITMATH_EMBEDDED

namespace {

enum {
    LFM_ATTN_QUERY_HEADS = LFM_HEADS,
    LFM_ATTN_KV_HEADS = LFM_KV_HEADS,
    LFM_ATTN_HEAD_DIM = LFM_HEAD_DIM,
    LFM_ATTN_HEADS_PER_KV = LFM_HEADS / LFM_KV_HEADS,
    LFM_ATTN_KV_WIDTH = LFM_KV_HEADS * LFM_HEAD_DIM,
    LFM_ATTN_ROWS_PER_WAVE = 8u,
    LFM_ATTN_SCORE_THREADS = 256u,
    LFM_ATTN_DECODE_SCORE_WORKERS = 2u,
    LFM_ATTN_AV_KEY_TILE = 64u,
    LFM_ATTN_VALUE_COLUMN_TILE = 64u
};

__device__ static void lfm_attention_fail(int *status, int code) {
    if (status) atomicCAS(status, 0, code);
}

__device__ static const float *lfm_attention_row(
        const float *committed, const float *tentative,
        const uint32_t *parents, const uint32_t *depths,
        uint32_t rows, uint32_t current, uint32_t position,
        uint32_t source_position, uint32_t width,
        uint32_t mode, uint32_t capacity) {
    if (position < source_position)
        return cuda_text_committed_row(committed, position, source_position,
                                       width, mode, capacity);
    uint32_t target_depth = position - source_position;
    if (!parents) {
        if (target_depth > current || target_depth >= rows)
            return NULL;
        return tentative + (size_t)target_depth * width;
    }
    return cuda_text_frontier_row(tentative, parents, depths, rows, current,
                                  target_depth, width);
}

/* Each immutable SCORE ledger row owns one (local row, query head, worker)
 * range. Decode has two disjoint workers per head; multirow PREFILL already
 * exposes enough row/head blocks and uses one worker per head. */
__global__ static void lfm_attention_scores_exact(
        const LfmAttentionIoLedger ledger,
        const uint32_t *source_position_pointer,
        const uint32_t *parents, const uint32_t *depths,
        const float *queries, const float *committed_keys,
        const float *tentative_keys, float *scores,
        uint32_t rows, uint32_t wave_first, uint32_t wave_rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity, float score_scale,
        uint32_t score_workers, int *status) {
    uint32_t ledger_job = (uint32_t)blockIdx.x;
    uint32_t task = ledger_job / score_workers;
    uint32_t worker = ledger_job - task * score_workers;
    uint32_t local_row = task / ledger.query_heads;
    uint32_t head = task - local_row * ledger.query_heads;
    uint32_t kv_head = head / ledger.heads_per_kv;
    uint32_t source_position;
    uint32_t row;
    uint32_t position;
    const float *query;
    float *row_scores;

    if (!source_position_pointer || !depths || !queries ||
        !tentative_keys || !scores || !status || score_workers == 0u ||
        local_row >= wave_rows || wave_first + local_row >= rows ||
        blockDim.x != ledger.score_threads) {
        lfm_attention_fail(status, 301);
        return;
    }
    if (*status != 0) return;
    source_position = *source_position_pointer;
    row = wave_first + local_row;
    position = source_position + depths[row];
    if (position >= ledger.maximum_context) {
        lfm_attention_fail(status, 302);
        return;
    }
    query = queries + (size_t)row * query_width +
        (size_t)head * ledger.head_dim;
    row_scores = scores + (size_t)task * ledger.maximum_context;
    for (uint32_t key_position = worker * ledger.score_threads + threadIdx.x;
         key_position <= position;
         key_position += score_workers * ledger.score_threads) {
        const float *key = lfm_attention_row(
            committed_keys, tentative_keys, parents, depths, rows, row,
            key_position, source_position, kv_width, kv_mode, kv_capacity);
        if (!key) {
            lfm_attention_fail(status, 303);
            continue;
        }
        key += (size_t)kv_head * ledger.head_dim;
        float accumulator = 0.0f;
        for (uint32_t column = 0; column < ledger.head_dim; column++)
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(query[column], key[column]));
        row_scores[key_position] = __fmul_rn(accumulator, score_scale);
    }
}

/* One NORMALIZE+AV ledger row owns one complete (query row, query head)
 * output. It writes the ordered denominator to the fixed dead-Q seat and then
 * performs the canonical ascending-position AV accumulation in the same block. */
__global__ static void lfm_attention_reduce_exact(
        const LfmAttentionIoLedger ledger,
        const uint32_t *source_position_pointer,
        const uint32_t *parents, const uint32_t *depths,
        const float *committed_values, const float *tentative_values,
        float *scores, float *denominators, float *output,
        uint32_t rows, uint32_t wave_first, uint32_t wave_rows,
        uint32_t query_width, uint32_t kv_width,
        uint32_t kv_mode, uint32_t kv_capacity, int *status) {
    uint32_t task = (uint32_t)blockIdx.x;
    uint32_t local_row = task / ledger.query_heads;
    uint32_t head = task - local_row * ledger.query_heads;
    uint32_t kv_head = head / ledger.heads_per_kv;
    uint32_t row;
    uint32_t source_position;
    uint32_t position;
    float *row_scores;
    __shared__ float maximum;
    __shared__ float denominator;
    __shared__ float probabilities[LFM_ATTN_AV_KEY_TILE];
    __shared__ float products[LFM_ATTN_AV_KEY_TILE * LFM_ATTN_HEAD_DIM];

    if (!source_position_pointer || !depths || !tentative_values ||
        !scores || !denominators || !output || !status ||
        local_row >= wave_rows || wave_first + local_row >= rows) {
        lfm_attention_fail(status, 311);
        return;
    }
    if (*status != 0) return;
    row = wave_first + local_row;
    source_position = *source_position_pointer;
    position = source_position + depths[row];
    if (position >= ledger.maximum_context) {
        lfm_attention_fail(status, 312);
        return;
    }
    row_scores = scores + (size_t)task * ledger.maximum_context;
    if (threadIdx.x == 0u) {
        float peak = -1.0f / 0.0f;
        for (uint32_t key_position = 0; key_position <= position;
             key_position++)
            if (row_scores[key_position] > peak)
                peak = row_scores[key_position];
        maximum = peak;
    }
    __syncthreads();
    for (uint32_t key_position = threadIdx.x; key_position <= position;
         key_position += blockDim.x)
        row_scores[key_position] = lfm_attention_owned_expf(
            __fadd_rn(row_scores[key_position], -maximum));
    __syncthreads();
    if (threadIdx.x == 0u) {
        float sum = 0.0f;
        for (uint32_t key_position = 0; key_position <= position;
             key_position++)
            sum = __fadd_rn(sum, row_scores[key_position]);
        denominator = sum;
        denominators[(size_t)row * ledger.query_heads + head] = sum;
        if (!(sum > 0.0f) || !isfinite(sum))
            lfm_attention_fail(status, 313);
    }
    __syncthreads();
    if (*status != 0) return;
    float accumulator = 0.0f;
    for (uint32_t base = 0u; base <= position;
         base += LFM_ATTN_AV_KEY_TILE) {
        uint32_t tile_count = position - base + 1u;
        if (tile_count > LFM_ATTN_AV_KEY_TILE)
            tile_count = LFM_ATTN_AV_KEY_TILE;
        if (threadIdx.x < tile_count)
            probabilities[threadIdx.x] = __fdiv_rn(
                row_scores[base + threadIdx.x], denominator);
        __syncthreads();
        uint32_t product_count = tile_count * ledger.head_dim;
        for (uint32_t product_index = threadIdx.x;
             product_index < product_count;
             product_index += blockDim.x) {
            uint32_t local_position = product_index / ledger.head_dim;
            uint32_t column = product_index -
                local_position * ledger.head_dim;
            uint32_t key_position = base + local_position;
            const float *value = lfm_attention_row(
                committed_values, tentative_values, parents, depths, rows, row,
                key_position, source_position, kv_width, kv_mode, kv_capacity);
            if (!value) {
                products[product_index] = 0.0f;
                lfm_attention_fail(status, 314);
            } else {
                value += (size_t)kv_head * ledger.head_dim;
                products[product_index] = __fmul_rn(
                    probabilities[local_position], value[column]);
            }
        }
        __syncthreads();
        if (threadIdx.x < ledger.head_dim)
            for (uint32_t local_position = 0; local_position < tile_count;
                 local_position++)
                accumulator = __fadd_rn(accumulator,
                    products[(size_t)local_position * ledger.head_dim +
                        threadIdx.x]);
        __syncthreads();
    }
    if (threadIdx.x < ledger.head_dim) {
        output[(size_t)row * query_width + (size_t)head * ledger.head_dim +
            threadIdx.x] = accumulator;
        if (!isfinite(accumulator)) lfm_attention_fail(status, 315);
    }
}

static int view_ok(const SaltTextModelGpuBufferView *view, uint64_t need) {
    return view && view->resource && view->offset <= UINT64_MAX - need &&
        need <= view->bytes;
}

template <typename T>
static T *view_pointer(const SaltTextModelGpuBufferView *view) {
    return (T *)((unsigned char *)view->resource + view->offset);
}

} /* namespace */

extern "C" int lfm_attention_cuda_launch(
        const LfmAttentionIoLedger *ledger,
        const SaltTextModelGpuAttentionArgs *args) {
    cudaStream_t stream;
    uint64_t score_floats;
    uint64_t score_bytes;
    uint32_t wave_first;

    if (!ledger || ledger->ready != 1u || !args ||
        args->backend != LFM_ATTENTION_GPU_BACKEND || !args->command ||
        !args->physical_launches || !args->full_attention || args->rows == 0u ||
        args->maximum_context != ledger->maximum_context ||
        args->query_width != LFM_HIDDEN || args->kv_width != LFM_ATTN_KV_WIDTH ||
        args->n_heads != ledger->query_heads ||
        args->n_kv_heads != ledger->kv_heads ||
        args->head_dim != ledger->head_dim ||
        args->n_heads / args->n_kv_heads != ledger->heads_per_kv ||
        args->kv_capacity == 0u)
        return -1;

    score_floats = (uint64_t)ledger->score_task_capacity *
        ledger->maximum_context;
    if (score_floats > UINT64_MAX / sizeof(float)) return -1;
    score_bytes = score_floats * sizeof(float);
    if (!view_ok(&args->source_position, sizeof(uint32_t)) ||
        (args->parent_rows.resource &&
         !view_ok(&args->parent_rows,
                  (uint64_t)args->rows * sizeof(uint32_t))) ||
        !view_ok(&args->depths,
                 (uint64_t)args->rows * sizeof(uint32_t)) ||
        !view_ok(&args->queries,
                 (uint64_t)args->rows * args->query_width * sizeof(float)) ||
        !view_ok(&args->committed_keys,
                 (uint64_t)args->kv_capacity * args->kv_width * sizeof(float)) ||
        !view_ok(&args->committed_values,
                 (uint64_t)args->kv_capacity * args->kv_width * sizeof(float)) ||
        !view_ok(&args->tentative_keys,
                 (uint64_t)args->rows * args->kv_width * sizeof(float)) ||
        !view_ok(&args->tentative_values,
                 (uint64_t)args->rows * args->kv_width * sizeof(float)) ||
        !view_ok(&args->score_workspace, score_bytes) ||
        !view_ok(&args->output,
                 (uint64_t)args->rows * args->query_width * sizeof(float)) ||
        !view_ok(&args->status, sizeof(int)))
        return -1;

    stream = (cudaStream_t)args->command;
    *args->physical_launches = 0u;
    for (wave_first = 0u; wave_first < args->rows;
         wave_first += ledger->rows_per_wave) {
        uint32_t wave_rows = args->rows - wave_first;
        uint32_t score_workers;
        uint32_t score_blocks;
        uint32_t reduce_blocks;
        if (wave_rows > ledger->rows_per_wave)
            wave_rows = ledger->rows_per_wave;
        score_workers = wave_rows == 1u
            ? ledger->decode_score_workers : 1u;
        score_blocks = wave_rows * ledger->query_heads * score_workers;
        reduce_blocks = wave_rows * ledger->query_heads;

        lfm_attention_scores_exact<<<score_blocks, ledger->score_threads,
            0u, stream>>>(*ledger,
            view_pointer<const uint32_t>(&args->source_position),
            args->parent_rows.resource
                ? view_pointer<const uint32_t>(&args->parent_rows) : NULL,
            view_pointer<const uint32_t>(&args->depths),
            view_pointer<const float>(&args->queries),
            view_pointer<const float>(&args->committed_keys),
            view_pointer<const float>(&args->tentative_keys),
            view_pointer<float>(&args->score_workspace),
            args->rows, wave_first, wave_rows, args->query_width,
            args->kv_width, args->kv_mode, args->kv_capacity,
            args->score_scale, score_workers,
            view_pointer<int>(&args->status));
        if (cudaGetLastError() != cudaSuccess) return -1;

        lfm_attention_reduce_exact<<<reduce_blocks, 256u, 0u, stream>>>(*ledger,
            view_pointer<const uint32_t>(&args->source_position),
            args->parent_rows.resource
                ? view_pointer<const uint32_t>(&args->parent_rows) : NULL,
            view_pointer<const uint32_t>(&args->depths),
            view_pointer<const float>(&args->committed_values),
            view_pointer<const float>(&args->tentative_values),
            view_pointer<float>(&args->score_workspace),
            view_pointer<float>(&args->queries),
            view_pointer<float>(&args->output), args->rows, wave_first,
            wave_rows, args->query_width, args->kv_width, args->kv_mode,
            args->kv_capacity, view_pointer<int>(&args->status));
        if (cudaGetLastError() != cudaSuccess) return -1;
        if (*args->physical_launches > UINT32_MAX - 2u) return -1;
        *args->physical_launches += 2u;
    }
    return 0;
}
