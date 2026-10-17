/* Compile the canonical C definitions for device linkage without editing,
 * copying or replacing their bodies. No kernels or mathematical expressions
 * are defined by this binding translation unit. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "salt/bitmath.h"
#include "gpu_cuda_core.cuh"

namespace salt_core_device {
static __host__ __device__ double invsqrt64(float);
static __host__ __device__ void sin_cos_reduce(float, int *, float *);
static __host__ __device__ float sin_poly(float);
static __host__ __device__ float cos_poly(float);
#include "bitmath.c"
}
