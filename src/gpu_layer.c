#include "salt/gpu_pool.h"

#include "salt/gpu.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

static int range_inside(const SaltGpuPoolLayer *lease,
                        uint64_t offset, uint64_t nbytes) {
    uint64_t mapped;
    if (!lease || !lease->base) return 0;
    mapped = (uint64_t)lease->mapped_nbytes;
    return offset <= mapped && nbytes <= mapped - offset;
}

int salt_gpu_pool_device_layer_acquire(const SaltGpuPool *pool,
                                       uint32_t layer,
                                       int writable_private,
                                       SaltGpuPoolDeviceLayer *device_layer) {
    if (!pool || !device_layer) return -1;
    memset(device_layer, 0, sizeof *device_layer);
    device_layer->lease.fd = -1;
    if (salt_gpu_pool_layer_acquire(pool, layer, writable_private,
                                    &device_layer->lease) != 0)
        return -1;
    device_layer->resource_id = device_layer->lease.shard_id;
    if (salt_gpu_expert_resource_bind(device_layer->resource_id,
                                      device_layer->lease.base,
                                      device_layer->lease.mapped_nbytes) != 0) {
        if (salt_gpu_pool_layer_release(&device_layer->lease) != 0)
            return -1;
        return -1;
    }
    device_layer->bound = 1;
    return 0;
}

int salt_gpu_pool_device_layer_register(
    const SaltGpuPool *pool, const SaltGpuPoolDeviceLayer *device_layer,
    uint32_t expert, uint32_t component, const void *key) {
    const SaltGpuDispatchTemplate *dispatch;
    const uint8_t *base;
    if (!pool || !device_layer || !device_layer->bound || !key) return -1;
    dispatch = salt_gpu_pool_dispatch(pool, device_layer->lease.layer,
                                      expert, component);
    if (!dispatch || dispatch->shard_id != device_layer->resource_id ||
        dispatch->R > INT_MAX || dispatch->C > INT_MAX ||
        !range_inside(&device_layer->lease,
                      dispatch->voff, dispatch->v_nbytes) ||
        !range_inside(&device_layer->lease,
                      dispatch->soff, dispatch->s_nbytes) ||
        !range_inside(&device_layer->lease,
                      dispatch->boff, dispatch->b_nbytes))
        return -1;
    base = device_layer->lease.base;
    return salt_gpu_expert_resource_slot(
        device_layer->resource_id, key,
        (const uint32_t *)(const void *)(base + (size_t)dispatch->voff),
        (const uint16_t *)(const void *)(base + (size_t)dispatch->soff),
        (const uint16_t *)(const void *)(base + (size_t)dispatch->boff),
        (int)dispatch->R, (int)dispatch->C);
}

int salt_gpu_pool_device_layer_release(
    SaltGpuPoolDeviceLayer *device_layer) {
    if (!device_layer) return -1;
    if (device_layer->bound) {
        if (salt_gpu_expert_resource_unbind(device_layer->resource_id) != 0)
            return -1;
        device_layer->bound = 0;
    }
    if (!device_layer->lease.base && device_layer->lease.fd < 0) {
        device_layer->resource_id = 0;
        return 0;
    }
    if (salt_gpu_pool_layer_release(&device_layer->lease) != 0)
        return -1;
    device_layer->resource_id = 0;
    return 0;
}
