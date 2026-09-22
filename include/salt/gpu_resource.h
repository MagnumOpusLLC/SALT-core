#ifndef SALT_GPU_RESOURCE_H
#define SALT_GPU_RESOURCE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SaltGpuResourceKind {
    SALT_GPU_RESOURCE_ARENA = 0,
    SALT_GPU_RESOURCE_TRUNK = 1,
    SALT_GPU_RESOURCE_EXPERT_LAYER = 2
} SaltGpuResourceKind;

typedef struct SaltGpuResourceRef {
    uint32_t kind;
    uint32_t resource_id;
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
} SaltGpuResourceRef;

/* Resource identity is the pair (kind, resource_id). Local offsets have
 * meaning only within the resource named by that pair. */
int salt_gpu_resource_ref_valid(const SaltGpuResourceRef *ref);
int salt_gpu_resource_same(const SaltGpuResourceRef *a,
                           const SaltGpuResourceRef *b);

/* Convert a byte offset to the 32-bit element offset consumed by current
 * Metal kernels. Reject misalignment and every narrowing above UINT32_MAX. */
int salt_gpu_local_offset_u32(uint64_t byte_offset,
                              uint32_t element_nbytes,
                              uint32_t *out);

#ifdef __cplusplus
}
#endif

#endif /* SALT_GPU_RESOURCE_H */
