/* CUDA linkage declarations only; all arithmetic lives in the original C. */
#ifndef SALT_GPU_CUDA_TENSOR_BINDINGS_CUH
#define SALT_GPU_CUDA_TENSOR_BINDINGS_CUH

#include <stdint.h>


namespace salt_tensor_device {
/* Complete canonical row, with global row index and unmodified fold order.
 * The caller owns bounds, pointer validity and disjoint output-row work. */
__device__ float q4_row(const uint32_t *vals, const uint16_t *scales,
                        const uint16_t *biases, int columns,
                        const float *input, int row);
/* Flat canonical decode from the supplied packed origin; no scheduling. */
__device__ void q4_decode(const uint32_t *vals, const uint16_t *scales,
                          const uint16_t *biases, int count, float *output);

}

#endif
