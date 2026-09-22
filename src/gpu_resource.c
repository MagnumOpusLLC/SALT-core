#include "salt/gpu_resource.h"

#include <limits.h>

int salt_gpu_resource_ref_valid(const SaltGpuResourceRef *ref) {
    if (!ref) return 0;
    return ref->kind <= SALT_GPU_RESOURCE_EXPERT_LAYER;
}

int salt_gpu_resource_same(const SaltGpuResourceRef *a,
                           const SaltGpuResourceRef *b) {
    if (!salt_gpu_resource_ref_valid(a) || !salt_gpu_resource_ref_valid(b))
        return 0;
    return a->kind == b->kind && a->resource_id == b->resource_id;
}

int salt_gpu_local_offset_u32(uint64_t byte_offset,
                              uint32_t element_nbytes,
                              uint32_t *out) {
    uint64_t element_offset;
    if (!out || element_nbytes == 0 || byte_offset % element_nbytes != 0)
        return -1;
    element_offset = byte_offset / element_nbytes;
    if (element_offset > UINT32_MAX) return -1;
    *out = (uint32_t)element_offset;
    return 0;
}
