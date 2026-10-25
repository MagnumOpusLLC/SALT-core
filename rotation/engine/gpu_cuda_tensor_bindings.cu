/* Linkage-only CUDA compilation of unchanged authoritative C sources.
 * No mathematical bodies, kernels, allocations or controllers are added.
 * Host scheduling remains host-only and is never called by these wrappers. */
#include <cuda_runtime.h>
#include "salt/ledger.h"
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "salt/kernels.h"
#include "salt/gpu.h"
#include "salt/simd.h"
#include "compiler.h"
#include "thread-lifecycle.h"
#include "gpu_cuda_tensor_bindings.cuh"

#ifndef SALT_Q4_CORE_INPUT
#error "Build must select the byte-verified canonical Q4 compiler input"
#endif

/* The host is ARM, but device compilation selects the existing scalar path. */
#ifdef __aarch64__
#undef __aarch64__
#endif

/* Rename included public host symbols to avoid ADL into the original API. */
#define salt_kernels_set_simd source_kernels_set_simd
#define salt_kernels_simd source_kernels_simd
#define salt_kernels_set_in_expert source_kernels_set_in_expert
#define salt_kernels_in_expert source_kernels_in_expert
#define salt_kernels_set_cpu_expert_scope source_kernels_set_cpu_expert_scope
#define salt_kernels_cpu_expert_scope source_kernels_cpu_expert_scope
#define salt_kernels_set_tiled source_kernels_set_tiled
#define salt_e8m0_value source_e8m0_value
#define salt_mx4_decode source_mx4_decode
#define salt_mx4_matvec source_mx4_matvec
#define salt_router_scores source_router_scores
#define salt_f32_matvec source_f32_matvec
#define salt_bf16_matvec source_bf16_matvec
#define salt_q4_decode source_q4_decode
#define salt_q4_matvec_status source_q4_matvec_status
#define salt_q4_matvec source_q4_matvec
#define salt_q8_matvec source_q8_matvec
#define salt_q8_matvec_batch_rows source_q8_matvec_batch_rows
#define salt_rank1 source_rank1
#define salt_f8_matvec source_f8_matvec
#define salt_f8_matvec_bf16 source_f8_matvec_bf16
#define salt_f8_matvec_rows source_f8_matvec_rows
#define salt_f8_decode_row source_f8_decode_row
#define salt_f8_value source_f8_value
#define salt_i8_matvec source_i8_matvec
#define salt_f16_to_f32 source_f16_to_f32
#define salt_f16_matvec source_f16_matvec
#define salt_q4_matvec2 source_q4_matvec2
#define salt_q4_multi_batch source_q4_multi_batch
#define salt_q4_matvec_batch_rows source_q4_matvec_batch_rows
#define salt_q4_matvec_batch_tile source_q4_matvec_batch_tile
#define salt_q4_multi_batch_rows source_q4_multi_batch_rows
#define salt_q4_matvec_batch source_q4_matvec_batch
#define salt_gpu_mixed_proj_batch source_gpu_mixed_proj_batch

namespace salt_tensor_device {
namespace canonical_source {
static __host__ __device__ float bf16_to_f32(uint16_t);
static __host__ __device__ uint16_t load_u16_unaligned(const uint16_t *, size_t);
static __host__ __device__ uint32_t load_u32_unaligned(const uint32_t *, size_t);
static __host__ __device__ float q4_canonical_row(
    const uint32_t *, const uint16_t *, const uint16_t *, int, const float *, int);
__host__ __device__ void source_q4_decode(
    const uint32_t *, const uint16_t *, const uint16_t *, int, float *);
#include SALT_Q4_CORE_INPUT
}

__device__ float q4_row(const uint32_t *vals, const uint16_t *scales,
                        const uint16_t *biases, int columns,
                        const float *input, int row) {
    return canonical_source::q4_canonical_row(
        vals, scales, biases, columns, input, row);
}

__device__ void q4_decode(const uint32_t *vals, const uint16_t *scales,
                          const uint16_t *biases, int count, float *output) {
    canonical_source::source_q4_decode(vals, scales, biases, count, output);
}
}

#undef salt_kernels_set_simd
#undef salt_kernels_simd
#undef salt_kernels_set_in_expert
#undef salt_kernels_in_expert
#undef salt_kernels_set_cpu_expert_scope
#undef salt_kernels_cpu_expert_scope
#undef salt_kernels_set_tiled
#undef salt_e8m0_value
#undef salt_mx4_decode
#undef salt_mx4_matvec
#undef salt_router_scores
#undef salt_f32_matvec
#undef salt_bf16_matvec
#undef salt_q4_decode
#undef salt_q4_matvec_status
#undef salt_q4_matvec
#undef salt_q8_matvec
#undef salt_q8_matvec_batch_rows
#undef salt_rank1
#undef salt_f8_matvec
#undef salt_f8_matvec_bf16
#undef salt_f8_matvec_rows
#undef salt_f8_decode_row
#undef salt_f8_value
#undef salt_i8_matvec
#undef salt_f16_to_f32
#undef salt_f16_matvec
#undef salt_q4_matvec2
#undef salt_q4_multi_batch
#undef salt_q4_matvec_batch_rows
#undef salt_q4_matvec_batch_tile
#undef salt_q4_multi_batch_rows
#undef salt_q4_matvec_batch
#undef salt_gpu_mixed_proj_batch


