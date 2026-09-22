#ifndef SALT_GPU_POOL_H
#define SALT_GPU_POOL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct SaltExpertPool;
struct SaltPoolLayout;

typedef struct SaltGpuPoolRef {
    uint32_t layer;
    uint32_t expert;
    uint32_t shard_id;
    uint32_t flags;
    uint64_t local_offset;
    uint64_t nbytes;
    uint64_t source_offset;
} SaltGpuPoolRef;

typedef struct SaltGpuDispatchTemplate {
    uint32_t layer;
    uint32_t expert;
    uint32_t component;       /* 0=gate, 1=up, 2=down */
    uint32_t shard_id;
    uint64_t voff;
    uint64_t soff;
    uint64_t boff;
    uint64_t v_nbytes;
    uint64_t s_nbytes;
    uint64_t b_nbytes;
    uint32_t R;
    uint32_t C;
    uint32_t fmt;
    uint32_t bits;
    uint32_t block_size;
    uint32_t pipeline_id;
    uint32_t threadgroup_id;
    uint32_t flags;
} SaltGpuDispatchTemplate;

/* One bounded payload mapping. The caller owns the lease and must retain it
 * until all CPU/GPU users of base have completed. */
typedef struct SaltGpuPoolLayer {
    uint32_t       layer;
    uint32_t       shard_id;
    int            fd;
    uint8_t       *base;
    uint64_t       file_offset;
    size_t         mapped_nbytes;
    uint64_t       payload_offset;
    uint64_t       payload_nbytes;
} SaltGpuPoolLayer;

typedef struct SaltGpuPool {
    uint32_t        alignment;
    uint32_t        n_layers;
    uint32_t        n_experts;
    uint32_t        n_shards;
    uint64_t        expert_nbytes;
    uint64_t        source_nbytes;
    SaltGpuPoolRef *ref;          /* [layer*n_experts + expert] */
    uint64_t        n_dispatch;
    SaltGpuDispatchTemplate *dispatch; /* [layer][expert][gate,up,down] */
    void           *impl;         /* address book only; no payload maps */
} SaltGpuPool;

/* A bounded host lease joined to one process-local backend resource.
 * It contains no serialized or backend-specific handle. */
typedef struct SaltGpuPoolDeviceLayer {
    SaltGpuPoolLayer lease;
    uint32_t         resource_id;
    int              bound;
} SaltGpuPoolDeviceLayer;

/* Derive the process-local GPU sidecar from the already parsed authoritative
 * CPU index and already opened canonical pool. This performs no GPU metadata
 * file reads. The source path is retained only for bounded lazy mappings. */
int salt_gpu_pool_from_layout(SaltGpuPool *pool,
                              const struct SaltPoolLayout *layout,
                              const struct SaltExpertPool *source,
                              const char *source_path);

/* Migration validation only: open a complete salt-gpu-ledger-v1 directory.
 * READY, index geometry, source extent/header, and segment ranges are validated
 * before success. Startup retains metadata only and no source payload fd/map.
 * expected_* are the active model's authoritative geometry. */
int salt_gpu_pool_open(SaltGpuPool *pool, const char *directory,
                       const char *source_pool,
                       int expected_layers, int expected_experts,
                       uint64_t expected_expert_nbytes);

/* Indexed logical and immutable dispatch lookup. */
const SaltGpuPoolRef *salt_gpu_pool_ref(const SaltGpuPool *pool,
                                        uint32_t layer, uint32_t expert);
const SaltGpuDispatchTemplate *salt_gpu_pool_dispatch(
    const SaltGpuPool *pool, uint32_t layer, uint32_t expert,
    uint32_t component);

/* Map exactly one layer. writable_private requests PROT_READ|PROT_WRITE with
 * MAP_PRIVATE for the controlled Metal no-copy qualification; file bytes are
 * never modified. Tensor offsets are relative to lease->base. */
int salt_gpu_pool_layer_acquire(const SaltGpuPool *pool, uint32_t layer,
                                int writable_private,
                                SaltGpuPoolLayer *lease);
const uint8_t *salt_gpu_pool_layer_expert(const SaltGpuPool *pool,
                                          const SaltGpuPoolLayer *lease,
                                          uint32_t expert);
int salt_gpu_pool_layer_release(SaltGpuPoolLayer *lease);

/* Pure-C lease/resource orchestration. release synchronizes and unbinds the
 * backend before dropping the canonical source mapping. */
int salt_gpu_pool_device_layer_acquire(const SaltGpuPool *pool,
                                       uint32_t layer,
                                       int writable_private,
                                       SaltGpuPoolDeviceLayer *device_layer);
int salt_gpu_pool_device_layer_register(
    const SaltGpuPool *pool, const SaltGpuPoolDeviceLayer *device_layer,
    uint32_t expert, uint32_t component, const void *key);
int salt_gpu_pool_device_layer_release(
    SaltGpuPoolDeviceLayer *device_layer);

void salt_gpu_pool_close(SaltGpuPool *pool);

#ifdef __cplusplus
}
#endif

#endif /* SALT_GPU_POOL_H */
