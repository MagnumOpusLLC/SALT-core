#include "lfm_gpu.h"

const SaltTextModelGpuOps *lfm_gpu_ops(void) {
    return (const SaltTextModelGpuOps *)0;
}

int lfm_gpu_canonical_readback(void *host, const void *backend,
                               size_t offset, size_t bytes) {
    (void)host;
    (void)backend;
    (void)offset;
    (void)bytes;
    return 0;
}
