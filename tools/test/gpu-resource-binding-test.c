#include "salt/gpu_resource.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "gpu-resource: check failed at line %d: %s\n", \
                __LINE__, #x); \
        return 1; \
    } \
} while (0)

int main(void) {
    SaltGpuResourceRef arena = {
        SALT_GPU_RESOURCE_ARENA, 0, 0, 0, 0
    };
    SaltGpuResourceRef trunk = {
        SALT_GPU_RESOURCE_TRUNK, 0, 16, 32, 48
    };
    SaltGpuResourceRef trunk_same = trunk;
    SaltGpuResourceRef other_trunk = trunk;
    SaltGpuResourceRef layer = {
        SALT_GPU_RESOURCE_EXPERT_LAYER, 17, 24, 40, 56
    };
    SaltGpuResourceRef other_layer = layer;
    SaltGpuResourceRef invalid = layer;
    uint32_t offset = 0;

    other_trunk.resource_id = 1;
    other_layer.resource_id = 18;
    invalid.kind = SALT_GPU_RESOURCE_EXPERT_LAYER + 1;

    CHECK(salt_gpu_resource_ref_valid(&arena));
    CHECK(salt_gpu_resource_ref_valid(&trunk));
    CHECK(salt_gpu_resource_ref_valid(&layer));
    CHECK(!salt_gpu_resource_ref_valid(&invalid));
    CHECK(!salt_gpu_resource_ref_valid(NULL));

    CHECK(salt_gpu_resource_same(&trunk, &trunk_same));
    CHECK(!salt_gpu_resource_same(&trunk, &other_trunk));
    CHECK(!salt_gpu_resource_same(&trunk, &layer));
    CHECK(!salt_gpu_resource_same(&layer, &other_layer));
    CHECK(!salt_gpu_resource_same(&layer, &invalid));

    CHECK(salt_gpu_local_offset_u32((uint64_t)UINT32_MAX * 4u,
                                    4, &offset) == 0);
    CHECK(offset == UINT32_MAX);
    CHECK(salt_gpu_local_offset_u32((uint64_t)UINT32_MAX * 4u + 4u,
                                    4, &offset) != 0);
    CHECK(salt_gpu_local_offset_u32(3, 2, &offset) != 0);
    CHECK(salt_gpu_local_offset_u32(0, 0, &offset) != 0);
    CHECK(salt_gpu_local_offset_u32(0, 4, NULL) != 0);

    puts("gpu-resource: PASS");
    return 0;
}
