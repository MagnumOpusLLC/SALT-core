#include "lfm_gpu.h"
#include "lfm.h"
#include "lfm_attention.h"

#if defined(LFM_GPU_HIP)
#include <hip/hip_runtime.h>
#define cudaStream_t hipStream_t
#define cudaGetLastError hipGetLastError
#define cudaSuccess hipSuccess
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaDeviceSynchronize hipDeviceSynchronize
#define LFM_MODEL_GPU_BACKEND SALT_TEXT_MODEL_GPU_HIP
#else
#include <cuda_runtime.h>
#define LFM_MODEL_GPU_BACKEND SALT_TEXT_MODEL_GPU_CUDA
#endif

#include <stdint.h>
#include <string.h>

#define SALT_BITMATH_EMBEDDED 1
#define SALT_BITMATH_API __device__ static
#define SALT_BITMATH_LOCAL __device__ static
#define salt_expf lfm_cuda_owned_expf
#define salt_logf lfm_cuda_owned_logf
#define salt_powf lfm_cuda_owned_powf
#define salt_rsqrtf lfm_cuda_owned_rsqrtf
#define salt_sqrtf lfm_cuda_owned_sqrtf
#define salt_tanhf lfm_cuda_owned_tanhf
#define salt_sinf lfm_cuda_owned_sinf
#define salt_cosf lfm_cuda_owned_cosf
#define salt_log1pf lfm_cuda_owned_log1pf
#include "../../src/bitmath.c"
#undef salt_log1pf
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

typedef struct {
    LfmAttentionIoLedger attention;
    uint32_t ready;
} LfmCudaState;

__device__ static float lfm_cuda_bf16(const unsigned char *p) {
    uint32_t bits = ((uint32_t)p[0] | ((uint32_t)p[1] << 8u)) << 16u;
    return __uint_as_float(bits);
}

__device__ static float lfm_cuda_bf16_aligned(const unsigned char *p) {
    return __uint_as_float((uint32_t)*(const uint16_t *)(const void *)p << 16u);
}

__global__ static void lfm_cuda_project_kernel(
        const unsigned char *weights, uint64_t value_offset,
        uint64_t scale_offset, uint64_t bias_offset, uint32_t encoding,
        const float *input, float *output,
        uint32_t rows, uint32_t cols, uint32_t jobs) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = rows * jobs;
    if (index >= total) return;
    uint32_t job = index / rows;
    uint32_t row = index - job * rows;
    float acc = 0.0f;
    uint64_t row_base = (uint64_t)row * cols;
    for (uint32_t col = 0; col < cols; col++) {
        uint64_t p = row_base + col;
        float weight;
        if (encoding == SALT_TENSOR_ENCODING_F32) {
            const float *values = (const float *)(const void *)(
                weights + value_offset);
            weight = values[p];
        } else if (encoding == SALT_TENSOR_ENCODING_BF16) {
            weight = lfm_cuda_bf16(weights + value_offset + p * 2u);
        } else {
            uint64_t bit = p * 6u;
            uint64_t byte = bit >> 3u;
            uint32_t packed = weights[value_offset + byte];
            if ((bit & 7u) > 2u)
                packed |= (uint32_t)weights[value_offset + byte + 1u] << 8u;
            uint32_t code = (packed >> (uint32_t)(bit & 7u)) & 63u;
            uint64_t group = p / 64u;
            float scale = lfm_cuda_bf16(weights + scale_offset + group * 2u);
            float bias = lfm_cuda_bf16(weights + bias_offset + group * 2u);
            float left = __fmul_rn((float)code, scale);
            left = __fmul_rn(left, input[(size_t)job * cols + col]);
            float right = __fmul_rn(bias,
                input[(size_t)job * cols + col]);
            acc = __fadd_rn(acc, __fadd_rn(left, right));
            continue;
        }
        float product = __fmul_rn(weight, input[(size_t)job * cols + col]);
        acc = __fadd_rn(acc, product);
    }
    output[(size_t)job * rows + row] = acc;
}

__global__ static void lfm_cuda_f32_tile_exact(
        const unsigned char *weights, uint64_t value_offset,
        const float *input, float *output,
        uint32_t rows, uint32_t cols, uint32_t jobs) {
    uint32_t lane = (uint32_t)threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t job = (uint32_t)blockIdx.y;
    extern __shared__ float products[];
    const float *matrix = (const float *)(const void *)(weights + value_offset);
    const float *x = input + (size_t)job * cols;
    for (uint32_t column = lane; column < cols; column += blockDim.x)
        products[column] = __fmul_rn(
            matrix[(size_t)row * cols + column], x[column]);
    __syncthreads();
    if (lane == 0u && row < rows && job < jobs) {
        float accumulator = 0.0f;
        for (uint32_t column = 0u; column < cols; column++)
            accumulator = __fadd_rn(accumulator, products[column]);
        output[(size_t)job * rows + row] = accumulator;
    }
}

__global__ static void lfm_cuda_q4_matrix_exact(
        const unsigned char *weights, uint64_t value_offset,
        uint64_t scale_offset, uint64_t bias_offset,
        const float *input, float *output, uint32_t rows,
        uint32_t cols, uint32_t jobs) {
    enum { TILE_ROWS = 6 };
    uint32_t lane = (uint32_t)threadIdx.x & 31u;
    uint32_t warp = (uint32_t)threadIdx.x >> 5u;
    uint32_t row = (uint32_t)blockIdx.x * 4u + warp;
    uint32_t job_base = (uint32_t)blockIdx.y * TILE_ROWS;
    float accumulators[TILE_ROWS] = {
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f
    };
#if defined(LFM_GPU_HIP)
    unsigned long long mask = ~0ull;
#else
    unsigned int mask = 0xffffffffu;
#endif
    if (row < rows) {
        for (uint32_t column = lane; column < cols; column += 32u) {
            uint64_t p = (uint64_t)row * cols + column;
            uint64_t group = p / 64u;
            uint32_t packed = *(const uint32_t *)(const void *)(
                weights + value_offset + (p >> 3u) * 4u);
            uint32_t code = (packed >> ((uint32_t)(p & 7u) * 4u)) & 15u;
            float value = __fadd_rn(
                __fmul_rn((float)code,
                    lfm_cuda_bf16(weights + scale_offset + group * 2u)),
                lfm_cuda_bf16(weights + bias_offset + group * 2u));
            #pragma unroll
            for (uint32_t item = 0; item < TILE_ROWS; item++) {
                uint32_t job = job_base + item;
                if (job < jobs)
                    accumulators[item] = __fadd_rn(accumulators[item],
                        __fmul_rn(value,
                            input[(size_t)job * cols + column]));
            }
        }
    }
    #pragma unroll
    for (uint32_t item = 0; item < TILE_ROWS; item++) {
        uint32_t job = job_base + item;
        float lane_sum = 0.0f;
        int base = (int)lane & 3;
        #pragma unroll
        for (int source = 0; source < 8; source++)
            lane_sum = __fadd_rn(lane_sum,
                __shfl_sync(mask, accumulators[item], base + 4 * source, 32));
        float lane0 = __shfl_sync(mask, lane_sum, 0, 32);
        float lane1 = __shfl_sync(mask, lane_sum, 1, 32);
        float lane2 = __shfl_sync(mask, lane_sum, 2, 32);
        float lane3 = __shfl_sync(mask, lane_sum, 3, 32);
        float even = __fadd_rn(lane0, lane2);
        float odd = __fadd_rn(lane1, lane3);
        if (lane == 0u && row < rows && job < jobs)
            output[(size_t)job * rows + row] = __fadd_rn(even, odd);
    }
}

__global__ static void lfm_cuda_q6_warp_exact(
        const unsigned char *weights, uint64_t value_offset,
        uint64_t scale_offset, uint64_t bias_offset,
        const float *input, float *output, uint32_t rows,
        uint32_t cols, uint32_t jobs) {
    uint32_t lane = (uint32_t)threadIdx.x;
    uint32_t row = (uint32_t)blockIdx.x;
    uint32_t job = (uint32_t)blockIdx.y;
    __shared__ float products[64];
    __shared__ float metadata[2];
    float acc = 0.0f;
    const float *x = input + (size_t)job * cols;
    for (uint32_t group_base = 0; group_base < cols; group_base += 64u) {
        if (lane == 0u) {
            uint64_t group = ((uint64_t)row * cols + group_base) / 64u;
            metadata[0] = lfm_cuda_bf16(
                weights + scale_offset + group * 2u);
            metadata[1] = lfm_cuda_bf16(
                weights + bias_offset + group * 2u);
        }
        __syncthreads();
        uint32_t column = group_base + lane;
        float product = 0.0f;
        if (column < cols) {
            uint64_t p = (uint64_t)row * cols + column;
            uint64_t bit = p * 6u;
            uint64_t byte = bit >> 3u;
            uint32_t packed = weights[value_offset + byte];
            if ((bit & 7u) > 2u)
                packed |=
                    (uint32_t)weights[value_offset + byte + 1u] << 8u;
            uint32_t code = (packed >> (uint32_t)(bit & 7u)) & 63u;
            float left = __fmul_rn((float)code, metadata[0]);
            left = __fmul_rn(left, x[column]);
            float right = __fmul_rn(metadata[1], x[column]);
            product = __fadd_rn(left, right);
        }
        products[lane] = product;
        __syncthreads();
        if (lane == 0u) {
            uint32_t count = cols - group_base;
            if (count > 64u) count = 64u;
            for (uint32_t source = 0; source < count; source++)
                acc = __fadd_rn(acc, products[source]);
        }
        __syncthreads();
    }
    if (lane == 0u && row < rows && job < jobs)
        output[(size_t)job * rows + row] = acc;
}

__device__ static float lfm_cuda_expert_reduce(float accumulator) {
#if defined(LFM_GPU_HIP)
    unsigned long long mask = ~0ull;
#else
    unsigned int mask = 0xffffffffu;
#endif
    int lane = (int)threadIdx.x & 31;
    int base = lane & 3;
    float lane_sum = 0.0f;
    #pragma unroll
    for (int source = 0; source < 8; source++)
        lane_sum = __fadd_rn(lane_sum,
            __shfl_sync(mask, accumulator, base + 4 * source, 32));
    float lane0 = __shfl_sync(mask, lane_sum, 0, 32);
    float lane1 = __shfl_sync(mask, lane_sum, 1, 32);
    float lane2 = __shfl_sync(mask, lane_sum, 2, 32);
    float lane3 = __shfl_sync(mask, lane_sum, 3, 32);
    return __fadd_rn(__fadd_rn(lane0, lane2),
                     __fadd_rn(lane1, lane3));
}

__device__ static int lfm_cuda_expert_ref_valid(
        const SaltTextModelGpuTensorView *ref, uint32_t rows, uint32_t cols) {
    return ref && ref->resource &&
        ref->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 &&
        ref->rows == rows && ref->cols == cols;
}

__global__ static void lfm_cuda_expert_project(
        const SaltTextModelGpuTensorView *table, uint32_t table_stride,
        uint32_t layer, uint32_t phase, const int32_t *selected,
        const int32_t *grouped_to_canonical, uint32_t jobs,
        const float *input, float *output, float *paired_up,
        uint32_t expected_rows, uint32_t expected_cols,
        uint32_t experts, int *status) {
    if (paired_up) {
        phase = (uint32_t)blockIdx.z;
        if (phase == 1u) output = paired_up;
    }
    uint32_t grouped = (uint32_t)blockIdx.y;
    int lane = (int)threadIdx.x & 31;
    int warp = (int)threadIdx.x >> 5;
    uint32_t row = (uint32_t)blockIdx.x * 8u + (uint32_t)warp;
    uint32_t canonical = 0u;
    int32_t expert = -1;
    const SaltTextModelGpuTensorView *projection = NULL;
    int valid = grouped < jobs && phase < 3u;
    if (valid) {
        canonical = (uint32_t)grouped_to_canonical[grouped];
        if (canonical >= jobs) valid = 0;
        else expert = selected[canonical];
        if (expert < 0 || (uint32_t)expert >= experts) valid = 0;
    }
    if (valid) {
        size_t base = ((size_t)layer * table_stride + (uint32_t)expert) * 3u;
        projection = &table[base + phase];
        valid = lfm_cuda_expert_ref_valid(
            projection, expected_rows, expected_cols);
    }
    if (!valid) {
        if (lane == 0 && warp == 0) atomicCAS(status, 0, 403);
        return;
    }
    float accumulator = 0.0f;
    if (row < expected_rows) {
        const unsigned char *resource =
            (const unsigned char *)projection->resource;
        const float *x = input + (size_t)grouped * expected_cols;
        uint64_t row_base = (uint64_t)row * expected_cols;
        for (uint32_t column = (uint32_t)lane; column < expected_cols;
             column += 64u) {
            uint64_t p0 = row_base + column;
            uint64_t group = p0 / 64u;
            float scale = lfm_cuda_bf16_aligned(
                resource + projection->scale_offset + group * 2u);
            float bias = lfm_cuda_bf16_aligned(
                resource + projection->bias_offset + group * 2u);
            uint32_t packed0 = *(const uint32_t *)(const void *)(
                resource + projection->value_offset + (p0 >> 3u) * 4u);
            uint32_t code0 =
                (packed0 >> ((uint32_t)(p0 & 7u) * 4u)) & 15u;
            float value0 = __fadd_rn(__fmul_rn((float)code0, scale), bias);
            accumulator = __fadd_rn(accumulator,
                __fmul_rn(value0, x[column]));
            uint32_t column1 = column + 32u;
            if (column1 < expected_cols) {
                uint64_t p1 = row_base + column1;
                uint32_t packed1 = *(const uint32_t *)(const void *)(
                    resource + projection->value_offset + (p1 >> 3u) * 4u);
                uint32_t code1 =
                    (packed1 >> ((uint32_t)(p1 & 7u) * 4u)) & 15u;
                float value1 = __fadd_rn(
                    __fmul_rn((float)code1, scale), bias);
                accumulator = __fadd_rn(accumulator,
                    __fmul_rn(value1, x[column1]));
            }
        }
    }
    float value = lfm_cuda_expert_reduce(accumulator);
    if (lane == 0 && row < expected_rows)
        output[(size_t)grouped * expected_rows + row] = value;
}

__global__ static void lfm_cuda_embedding_kernel(
        const unsigned char *weights, uint64_t value_offset,
        uint64_t scale_offset, uint64_t bias_offset,
        const int32_t *ids, float *output, uint32_t vocabulary,
        uint32_t width, uint32_t tokens, float embedding_scale) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t elements = (uint64_t)tokens * width;
    if (index >= elements) return;
    uint32_t row = (uint32_t)(index / width);
    uint32_t col = (uint32_t)(index - (uint64_t)row * width);
    int32_t token = ids[row];
    if (token < 0 || (uint32_t)token >= vocabulary) return;
    uint64_t p = (uint64_t)(uint32_t)token * width + col;
    uint64_t bit = p * 6u;
    uint64_t byte = bit >> 3u;
    uint32_t packed = weights[value_offset + byte];
    if ((bit & 7u) > 2u)
        packed |= (uint32_t)weights[value_offset + byte + 1u] << 8u;
    uint32_t code = (packed >> (uint32_t)(bit & 7u)) & 63u;
    uint64_t group = p / 64u;
    float value = __fadd_rn(
        __fmul_rn((float)code,
                  lfm_cuda_bf16(weights + scale_offset + group * 2u)),
        lfm_cuda_bf16(weights + bias_offset + group * 2u));
    output[index] = __fmul_rn(value, embedding_scale);
}

__global__ static void lfm_cuda_silu(const float *gate, const float *up,
                                     float *output, uint32_t count) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float g = gate[index];
    float active = __fdiv_rn(g,
        __fadd_rn(1.0f, lfm_cuda_owned_expf(-g)));
    output[index] = __fmul_rn(active, up[index]);
}

__global__ static void lfm_cuda_router_scores(
        float *scores, const float *bias, float *raw_weights,
        uint32_t rows, uint32_t experts) {
    uint32_t index = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = rows * experts;
    if (index >= total) return;
    uint32_t expert = index % experts;
    float weight = __fdiv_rn(1.0f,
        __fadd_rn(1.0f, lfm_cuda_owned_expf(-scores[index])));
    raw_weights[index] = weight;
    scores[index] = __fadd_rn(weight, bias[expert]);
}

__global__ static void lfm_cuda_router_select(
        const float *scores, const float *raw_weights,
        int32_t *selected, float *weights,
        uint32_t rows, uint32_t experts, uint32_t topk,
        uint32_t normalize, float scale) {
    uint32_t row = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float *row_scores = scores + (size_t)row * experts;
    const float *row_raw_weights = raw_weights + (size_t)row * experts;
    int32_t *row_selected = selected + (size_t)row * topk;
    float *row_weights = weights + (size_t)row * topk;
    float chosen = 0.0f;
    for (uint32_t rank = 0; rank < topk; rank++) {
        int32_t best = -1;
        float best_score = 0.0f, best_weight = 0.0f;
        for (uint32_t expert = 0; expert < experts; expert++) {
            int taken = 0;
            for (uint32_t prior = 0; prior < rank; prior++)
                if (row_selected[prior] == (int32_t)expert) taken = 1;
            if (taken) continue;
            float score = row_scores[expert];
            if (best < 0 || score > best_score) {
                best = (int32_t)expert;
                best_score = score;
                best_weight = row_raw_weights[expert];
            }
        }
        row_selected[rank] = best;
        row_weights[rank] = best_weight;
        chosen = __fadd_rn(chosen, best_weight);
    }
    if (normalize)
        for (uint32_t rank = 0; rank < topk; rank++)
            row_weights[rank] = __fdiv_rn(row_weights[rank], chosen);
    if (scale != 0.0f)
        for (uint32_t rank = 0; rank < topk; rank++)
            row_weights[rank] = __fmul_rn(row_weights[rank], scale);
    for (uint32_t left = 0; left < topk; left++)
        for (uint32_t right = left + 1u; right < topk; right++)
            if (row_selected[right] < row_selected[left]) {
                int32_t id = row_selected[left];
                float weight = row_weights[left];
                row_selected[left] = row_selected[right];
                row_weights[left] = row_weights[right];
                row_selected[right] = id;
                row_weights[right] = weight;
            }
}

__global__ static void lfm_cuda_route_complete(
        const int32_t *selected, int32_t *grouped, int32_t *inverse,
        int32_t *scratch, uint32_t rows, uint32_t experts, uint32_t topk) {
    if (blockIdx.x || threadIdx.x) return;
    uint32_t jobs = rows * topk;
    for (uint32_t expert = 0; expert < experts; expert++) scratch[expert] = 0;
    for (uint32_t canonical = 0; canonical < jobs; canonical++)
        scratch[(uint32_t)selected[canonical]]++;
    uint32_t next = 0;
    for (uint32_t expert = 0; expert < experts; expert++) {
        uint32_t population = (uint32_t)scratch[expert];
        scratch[expert] = (int32_t)next;
        next += population;
    }
    for (uint32_t canonical = 0; canonical < jobs; canonical++) {
        uint32_t expert = (uint32_t)selected[canonical];
        uint32_t grouped_row = (uint32_t)scratch[expert]++;
        grouped[grouped_row] = (int32_t)canonical;
    }
    for (uint32_t grouped_row = 0; grouped_row < jobs; grouped_row++)
        inverse[(uint32_t)grouped[grouped_row]] = (int32_t)grouped_row;
}

__global__ static void lfm_cuda_routed_gather(
        const int32_t *grouped, const float *source, float *routed,
        uint32_t jobs, uint32_t topk, uint32_t hidden) {
    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t elements = (uint64_t)jobs * hidden;
    if (index >= elements) return;
    uint32_t grouped_row = (uint32_t)(index / hidden);
    uint32_t column = (uint32_t)(index - (uint64_t)grouped_row * hidden);
    uint32_t canonical = (uint32_t)grouped[grouped_row];
    uint32_t row = canonical / topk;
    routed[index] = source[(size_t)row * hidden + column];
}

__global__ static void lfm_cuda_short_conv(float *canonical,
        uint64_t scratch_float, uint64_t state_float,
        const unsigned char *kernel, uint64_t kernel_byte,
        uint32_t rows, uint32_t hidden, uint32_t kernel_size) {
    uint32_t ch = (uint32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= hidden) return;
    float *window = canonical + state_float + (uint64_t)ch * kernel_size;
    for (uint32_t row = 0; row < rows; row++) {
        uint64_t base = scratch_float + (uint64_t)row * 3u * hidden;
        float b = canonical[base + ch];
        float c = canonical[base + hidden + ch];
        float x = canonical[base + 2u * hidden + ch];
        for (uint32_t tap = 0; tap + 1u < kernel_size; tap++)
            window[tap] = window[tap + 1u];
        window[kernel_size - 1u] = __fmul_rn(b, x);
        float acc = 0.0f;
        for (uint32_t tap = 0; tap < kernel_size; tap++) {
            float weight = lfm_cuda_bf16(kernel + kernel_byte +
                ((uint64_t)ch * kernel_size + tap) * 2u);
            acc = __fadd_rn(acc, __fmul_rn(weight, window[tap]));
        }
        canonical[scratch_float + (uint64_t)rows * 3u * hidden +
            (uint64_t)row * hidden + ch] = __fmul_rn(c, acc);
    }
}

static int lfm_cuda_prepare(void *context,
        const SaltTextModelGpuPrepareArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    (void)context;
    if (!args || args->backend != LFM_MODEL_GPU_BACKEND || !state ||
        state_bytes != sizeof *state || args->maximum_rows == 0u ||
        args->maximum_context == 0u ||
        lfm_attention_io_ledger_prepare(&state->attention,
                                        args->maximum_context) != 0)
        return -1;
    state->ready = 1u;
    return 0;
}

static int range_ok(uint64_t offset, uint64_t bytes, size_t capacity) {
    return offset <= (uint64_t)capacity && bytes <= (uint64_t)capacity - offset;
}

static int lfm_cuda_project(void *context,
        const SaltTextModelGpuProjectArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    uint64_t input_bytes, output_bytes, elements;
    (void)context;
    if (!args || args->backend != LFM_MODEL_GPU_BACKEND ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || state->ready != 1u ||
        args->rows == 0u ||
        (args->tensor.encoding != SALT_TENSOR_ENCODING_F32 &&
         args->tensor.encoding != SALT_TENSOR_ENCODING_BF16 &&
         args->tensor.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 &&
         args->tensor.encoding != SALT_TENSOR_ENCODING_AFFINE_Q6) ||
        args->tensor.rows == 0u || args->tensor.cols == 0u ||
        !args->tensor.resource)
        return -1;
    input_bytes = (uint64_t)args->rows * args->tensor.cols * sizeof(float);
    output_bytes = (uint64_t)args->rows * args->tensor.rows * sizeof(float);
    elements = (uint64_t)args->rows * args->tensor.rows;
    if (!range_ok(args->input_offset, input_bytes, args->canonical_bytes) ||
        !range_ok(args->output_offset, output_bytes, args->canonical_bytes) ||
        elements > UINT32_MAX)
        return -1;
    stream = (cudaStream_t)args->command;
    if (args->tensor.encoding == SALT_TENSOR_ENCODING_F32) {
        lfm_cuda_f32_tile_exact<<<
            dim3(args->tensor.rows, args->rows, 1u), 256u,
            (size_t)args->tensor.cols * sizeof(float), stream>>>(
            (const unsigned char *)args->tensor.resource,
            args->tensor.value_offset,
            (const float *)((const unsigned char *)args->canonical +
                args->input_offset),
            (float *)((unsigned char *)args->canonical + args->output_offset),
            args->tensor.rows, args->tensor.cols, args->rows);
    } else if (args->tensor.encoding == SALT_TENSOR_ENCODING_AFFINE_Q4) {
        lfm_cuda_q4_matrix_exact<<<
            dim3((args->tensor.rows + 3u) / 4u,
                 (args->rows + 5u) / 6u, 1u),
            128u, 0u, stream>>>(
            (const unsigned char *)args->tensor.resource,
            args->tensor.value_offset, args->tensor.scale_offset,
            args->tensor.bias_offset,
            (const float *)((const unsigned char *)args->canonical +
                args->input_offset),
            (float *)((unsigned char *)args->canonical + args->output_offset),
            args->tensor.rows, args->tensor.cols, args->rows);
    } else if (args->tensor.encoding == SALT_TENSOR_ENCODING_AFFINE_Q6) {
        lfm_cuda_q6_warp_exact<<<
            dim3(args->tensor.rows, args->rows, 1u),
            64u, 0u, stream>>>(
            (const unsigned char *)args->tensor.resource,
            args->tensor.value_offset, args->tensor.scale_offset,
            args->tensor.bias_offset,
            (const float *)((const unsigned char *)args->canonical +
                args->input_offset),
            (float *)((unsigned char *)args->canonical + args->output_offset),
            args->tensor.rows, args->tensor.cols, args->rows);
    } else {
        lfm_cuda_project_kernel<<<((uint32_t)elements + 255u) / 256u,
            256u, 0u, stream>>>(
            (const unsigned char *)args->tensor.resource,
            args->tensor.value_offset, args->tensor.scale_offset,
            args->tensor.bias_offset, (uint32_t)args->tensor.encoding,
            (const float *)((const unsigned char *)args->canonical +
                args->input_offset),
            (float *)((unsigned char *)args->canonical + args->output_offset),
            args->tensor.rows, args->tensor.cols, args->rows);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int lfm_cuda_embedding(void *context,
        const SaltTextModelGpuEmbeddingArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    uint64_t elements;
    (void)context;
    if (!args) return -101;
    if (args->backend != LFM_MODEL_GPU_BACKEND) return -102;
    if (!args->command) return -103;
    if (!args->canonical) return -104;
    if (!state) return -105;
    if (state_bytes != sizeof *state) return -106;
    if (state->ready != 1u) return -107;
    if (args->token_count == 0u) return -108;
    if (args->tensor.encoding != SALT_TENSOR_ENCODING_AFFINE_Q6)
        return -109;
    if (args->tensor.rows != LFM_VOCAB ||
        args->tensor.cols != LFM_HIDDEN)
        return -110;
    if (!args->tensor.resource) return -111;
    elements = (uint64_t)args->token_count * LFM_HIDDEN;
    if (!range_ok(args->token_ids_offset,
                  (uint64_t)args->token_count * sizeof(int32_t),
                  args->canonical_bytes))
        return -112;
    if (!range_ok(args->output_offset, elements * sizeof(float),
                  args->canonical_bytes))
        return -113;
    if (elements > UINT32_MAX) return -114;
    stream = (cudaStream_t)args->command;
    lfm_cuda_embedding_kernel<<<((uint32_t)elements + 255u) / 256u,
        256u, 0u, stream>>>(
        (const unsigned char *)args->tensor.resource,
        args->tensor.value_offset, args->tensor.scale_offset,
        args->tensor.bias_offset,
        (const int32_t *)((const unsigned char *)args->canonical +
            args->token_ids_offset),
        (float *)((unsigned char *)args->canonical + args->output_offset),
        args->tensor.rows, args->tensor.cols, args->token_count, args->scale);
    {
        int error = (int)cudaGetLastError();
        return error == (int)cudaSuccess ? 0 : -200 - error;
    }
}

static int lfm_cuda_activation(void *context,
        const SaltTextModelGpuActivationArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    (void)context;
    if (!args || args->backend != LFM_MODEL_GPU_BACKEND ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || state->ready != 1u ||
        args->activation != SALT_ACT_SILU || args->elements == 0u ||
        args->elements > UINT32_MAX)
        return -1;
    stream = (cudaStream_t)args->command;
    lfm_cuda_silu<<<((uint32_t)args->elements + 255u) / 256u,
        256u, 0u, stream>>>(
        (const float *)((const unsigned char *)args->canonical +
            args->gate_offset),
        (const float *)((const unsigned char *)args->canonical +
            args->up_offset),
        (float *)((unsigned char *)args->canonical + args->output_offset),
        (uint32_t)args->elements);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int lfm_cuda_expert_chain(void *context,
        const SaltTextModelGpuExpertChainArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    unsigned char *canonical;
    uint64_t selected_bytes, grouped_bytes, input_bytes, chain_bytes;
    uint64_t output_bytes, table_need;
    (void)context;
    if (!args) return -101;
    if (args->backend != LFM_MODEL_GPU_BACKEND) return -102;
    if (!args->command) return -103;
    if (!args->canonical) return -104;
    if (!args->expert_table.resource) return -105;
    if (!args->physical_launches) return -106;
    if (!state) return -107;
    if (state_bytes != sizeof *state) return -108;
    if (state->ready != 1u) return -109;
    if (args->rows == 0u) return -110;
    if (args->jobs == 0u) return -111;
    if (args->topk != LFM_TOP_K) return -112;
    if (args->experts != LFM_EXPERTS) return -113;
    if (args->hidden != LFM_HIDDEN) return -114;
    if (args->routed != LFM_MOE_INTERMEDIATE) return -115;
    if (args->expert_table_stride < LFM_EXPERTS) return -116;
    if (args->activation != SALT_ACT_SILU) return -117;
    if (args->rows > UINT32_MAX / args->topk ||
        args->jobs != args->rows * args->topk)
        return -118;
    selected_bytes = (uint64_t)args->jobs * sizeof(int32_t);
    grouped_bytes = selected_bytes;
    input_bytes = (uint64_t)args->jobs * args->hidden * sizeof(float);
    chain_bytes = (uint64_t)args->jobs * args->routed * sizeof(float);
    output_bytes = (uint64_t)args->jobs * args->hidden * sizeof(float);
    table_need = ((uint64_t)args->layer * args->expert_table_stride +
        args->experts) * 3u * sizeof(SaltTextModelGpuTensorView);
    if (table_need > args->expert_table.bytes) return -119;
    if (!range_ok(args->selected_offset, selected_bytes,
            args->canonical_bytes)) return -120;
    if (!range_ok(args->grouped_offset, grouped_bytes,
            args->canonical_bytes)) return -121;
    if (!range_ok(args->input_offset, input_bytes,
            args->canonical_bytes)) return -122;
    if (!range_ok(args->gate_offset, chain_bytes,
            args->canonical_bytes)) return -128;
    if (!range_ok(args->up_offset, chain_bytes,
            args->canonical_bytes)) return -129;
    if (!range_ok(args->chain_offset, chain_bytes,
            args->canonical_bytes)) return -123;
    if (!range_ok(args->output_jobs_offset, output_bytes,
            args->canonical_bytes)) return -124;
    if (!args->status.resource || args->status.bytes < sizeof(int)) return -125;
    stream = (cudaStream_t)args->command;
    canonical = (unsigned char *)args->canonical;
    lfm_cuda_expert_project<<<
        dim3((args->routed + 7u) / 8u, args->jobs, 2u), 256u, 0u, stream>>>(
        (const SaltTextModelGpuTensorView *)args->expert_table.resource,
        args->expert_table_stride, args->layer, 0u,
        (const int32_t *)(const void *)(canonical + args->selected_offset),
        (const int32_t *)(const void *)(canonical + args->grouped_offset),
        args->jobs,
        (const float *)(const void *)(canonical + args->input_offset),
        (float *)(void *)(canonical + args->gate_offset),
        (float *)(void *)(canonical + args->up_offset),
        args->routed, args->hidden, args->experts,
        (int *)(void *)((unsigned char *)args->status.resource +
            args->status.offset));
    if (cudaGetLastError() != cudaSuccess)
        return -126;
    lfm_cuda_silu<<<
        (args->jobs * args->routed + 255u) / 256u,
        256u, 0u, stream>>>(
        (const float *)(const void *)(canonical + args->gate_offset),
        (const float *)(const void *)(canonical + args->up_offset),
        (float *)(void *)(canonical + args->chain_offset),
        args->jobs * args->routed);
    if (cudaGetLastError() != cudaSuccess)
        return -130;
    lfm_cuda_expert_project<<<
        dim3((args->hidden + 7u) / 8u, args->jobs, 1u), 256u, 0u, stream>>>(
        (const SaltTextModelGpuTensorView *)args->expert_table.resource,
        args->expert_table_stride, args->layer, 2u,
        (const int32_t *)(const void *)(canonical + args->selected_offset),
        (const int32_t *)(const void *)(canonical + args->grouped_offset),
        args->jobs,
        (const float *)(const void *)(canonical + args->chain_offset),
        (float *)(void *)(canonical + args->output_jobs_offset), NULL,
        args->hidden, args->routed, args->experts,
        (int *)(void *)((unsigned char *)args->status.resource +
            args->status.offset));
    if (cudaGetLastError() != cudaSuccess)
        return -127;
    *args->physical_launches = 3u;
    return 0;
}

static int lfm_cuda_router_topk(void *context,
        const SaltTextModelGpuRouterArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    (void)context;
    if (!args || args->backend != LFM_MODEL_GPU_BACKEND ||
        !args->command || !args->canonical || !state ||
        state_bytes != sizeof *state || state->ready != 1u ||
        args->rows == 0u || args->experts != LFM_EXPERTS ||
        args->topk != LFM_TOP_K || args->hidden != LFM_HIDDEN ||
        args->expert_bias.encoding != SALT_TENSOR_ENCODING_F32 ||
        !args->expert_bias.resource)
        return -1;
    if (!range_ok(args->logits_offset,
            (uint64_t)args->rows * args->experts * sizeof(float),
            args->canonical_bytes) ||
        !range_ok(args->scratch_offset,
            (uint64_t)args->rows * args->experts * sizeof(float),
            args->canonical_bytes))
        return -1;
    stream = (cudaStream_t)args->command;
    uint32_t score_count = args->rows * args->experts;
    lfm_cuda_router_scores<<<(score_count + 255u) / 256u,
        256u, 0u, stream>>>(
        (float *)((unsigned char *)args->canonical + args->logits_offset),
        (const float *)((const unsigned char *)args->expert_bias.resource +
            args->expert_bias.value_offset),
        (float *)((unsigned char *)args->canonical + args->scratch_offset),
        args->rows, args->experts);
    if (cudaGetLastError() != cudaSuccess) return -1;
    lfm_cuda_router_select<<<(args->rows + 255u) / 256u,
        256u, 0u, stream>>>(
        (const float *)((const unsigned char *)args->canonical +
            args->logits_offset),
        (const float *)((const unsigned char *)args->canonical +
            args->scratch_offset),
        (int32_t *)((unsigned char *)args->canonical + args->selected_offset),
        (float *)((unsigned char *)args->canonical + args->weights_offset),
        args->rows, args->experts, args->topk,
        args->normalize_topk ? 1u : 0u, args->router_scale);
    if (cudaGetLastError() != cudaSuccess) return -1;
    lfm_cuda_route_complete<<<1u, 1u, 0u, stream>>>(
        (const int32_t *)((const unsigned char *)args->canonical +
            args->selected_offset),
        (int32_t *)((unsigned char *)args->canonical + args->grouped_offset),
        (int32_t *)((unsigned char *)args->canonical + args->inverse_offset),
        (int32_t *)((unsigned char *)args->canonical + args->scratch_offset),
        args->rows, args->experts, args->topk);
    if (cudaGetLastError() != cudaSuccess) return -1;
    uint32_t jobs = args->rows * args->topk;
    uint64_t elements = (uint64_t)jobs * args->hidden;
    lfm_cuda_routed_gather<<<(unsigned int)((elements + 255u) / 256u),
        256u, 0u, stream>>>(
        (const int32_t *)((const unsigned char *)args->canonical +
            args->grouped_offset),
        (const float *)((const unsigned char *)args->canonical +
            args->source_offset),
        (float *)((unsigned char *)args->canonical + args->routed_offset),
        jobs, args->topk, args->hidden);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

static int lfm_cuda_attention(void *context,
        const SaltTextModelGpuAttentionArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    (void)context;
    if (!state || state_bytes != sizeof *state || state->ready != 1u)
        return -1;
    return lfm_attention_cuda_launch(&state->attention, args);
}

static int lfm_cuda_encode(void *context,
        const SaltTextModelGpuEncodeArgs *args,
        void *state_memory, size_t state_bytes) {
    LfmCudaState *state = (LfmCudaState *)state_memory;
    cudaStream_t stream;
    uint64_t scratch_bytes, recurrent_bytes, gated_offset;
    (void)context;
    if (!args || args->backend != LFM_MODEL_GPU_BACKEND ||
        !args->command || !args->canonical || !args->project ||
        !args->backend_context || !state || state_bytes != sizeof *state ||
        state->ready != 1u || args->hidden != LFM_HIDDEN || args->rows == 0u ||
        args->kernel.encoding != SALT_TENSOR_ENCODING_BF16 ||
        args->kernel.rows != LFM_HIDDEN || args->kernel.cols != LFM_CONV_L ||
        args->input_projection.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        args->input_projection.rows != 3u * LFM_HIDDEN ||
        args->input_projection.cols != LFM_HIDDEN ||
        args->output_projection.encoding != SALT_TENSOR_ENCODING_AFFINE_Q4 ||
        args->output_projection.rows != LFM_HIDDEN ||
        args->output_projection.cols != LFM_HIDDEN ||
        args->scratch_float_capacity < (size_t)args->rows * 4u * LFM_HIDDEN ||
        args->state_bytes != (size_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float))
        return -1;
    scratch_bytes = (uint64_t)args->rows * 4u * LFM_HIDDEN * sizeof(float);
    recurrent_bytes = (uint64_t)LFM_HIDDEN * LFM_CONV_L * sizeof(float);
    if (!range_ok(args->normalized_offset,
            (uint64_t)args->rows * LFM_HIDDEN * sizeof(float),
            args->canonical_bytes) ||
        !range_ok(args->scratch_offset, scratch_bytes, args->canonical_bytes) ||
        !range_ok(args->state_offset, recurrent_bytes, args->canonical_bytes) ||
        !range_ok(args->output_offset,
            (uint64_t)args->rows * LFM_HIDDEN * sizeof(float),
            args->canonical_bytes))
        return -1;
    if (args->project(args->backend_context, &args->input_projection,
            args->normalized_offset, args->scratch_offset, args->rows) != 0)
        return -1;
    stream = (cudaStream_t)args->command;
    lfm_cuda_short_conv<<<(LFM_HIDDEN + 255u) / 256u, 256, 0, stream>>>(
        (float *)args->canonical,
        args->scratch_offset / sizeof(float),
        args->state_offset / sizeof(float),
        (const unsigned char *)args->kernel.resource,
        args->kernel.value_offset, args->rows, LFM_HIDDEN, LFM_CONV_L);
    if (cudaGetLastError() != cudaSuccess)
        return -1;
    gated_offset = args->scratch_offset +
        (uint64_t)args->rows * 3u * LFM_HIDDEN * sizeof(float);
    return args->project(args->backend_context, &args->output_projection,
        gated_offset, args->output_offset, args->rows);
}

static void lfm_cuda_destroy(void *context, SaltTextModelGpuBackend backend,
        void *state_memory, size_t state_bytes) {
    (void)context;
    if (backend == LFM_MODEL_GPU_BACKEND && state_memory &&
        state_bytes == sizeof(LfmCudaState))
        memset(state_memory, 0, state_bytes);
}

static const SaltTextModelGpuOps LFM_CUDA_OPS = {
    .state_bytes = sizeof(LfmCudaState),
    .flags = SALT_TEXT_MODEL_GPU_EXTENT_GRAPHS,
    .prepare = lfm_cuda_prepare,
    .project = lfm_cuda_project,
    .embedding = lfm_cuda_embedding,
    .activation = lfm_cuda_activation,
    .router_topk = lfm_cuda_router_topk,
    .expert_chain = lfm_cuda_expert_chain,
    .attention = lfm_cuda_attention,
    .encode = lfm_cuda_encode,
    .destroy = lfm_cuda_destroy,
};

extern "C" const SaltTextModelGpuOps *lfm_gpu_ops(void) {
    return &LFM_CUDA_OPS;
}

extern "C" int lfm_gpu_canonical_readback(
        void *host, const void *backend, size_t offset, size_t bytes) {
    if (!host || !backend || bytes == 0u)
        return -1;
    if (cudaMemcpy((unsigned char *)host + offset,
            (const unsigned char *)backend + offset,
            bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return -1;
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : -1;
}
