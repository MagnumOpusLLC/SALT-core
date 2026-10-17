#ifndef SALT_LFM_GPU_H
#define SALT_LFM_GPU_H

#include "salt/text_verify.h"

#include <stddef.h>

/* Selected at build time: CPU builds link lfm_gpu_stub.c, Metal builds link
 * lfm_gpu_metal.m, and CUDA builds link lfm_gpu_cuda.cu. */
#ifdef __cplusplus
extern "C" {
#endif
const SaltTextModelGpuOps *lfm_gpu_ops(void);
int lfm_gpu_canonical_readback(void *host, const void *backend,
                               size_t offset, size_t bytes);
#ifdef __cplusplus
}
#endif

#endif /* SALT_LFM_GPU_H */
