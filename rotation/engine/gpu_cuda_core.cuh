/* Device linkage only. Mathematical definitions remain in bitmath.c. */
#ifndef SALT_GPU_CUDA_CORE_CUH
#define SALT_GPU_CUDA_CORE_CUH

namespace salt_core_device {
__host__ __device__ float salt_expf(float);
__host__ __device__ float salt_logf(float);
__host__ __device__ float salt_powf(float, float);
__host__ __device__ float salt_rsqrtf(float);
__host__ __device__ float salt_sqrtf(float);
__host__ __device__ float salt_tanhf(float);
__host__ __device__ float salt_sinf(float);
__host__ __device__ float salt_cosf(float);
__host__ __device__ float salt_log1pf(float);
}

#endif
