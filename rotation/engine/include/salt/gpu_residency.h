#ifndef SALT_GPU_RESIDENCY_H
#define SALT_GPU_RESIDENCY_H

#include <stddef.h>
#include <stdint.h>

#include "salt/gpu_resource.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SALT_GPU_RESIDENCY_ABI_VERSION 1u

typedef enum SaltGpuResidencyPolicy {
    SALT_GPU_RESIDENCY_DISABLED = 0,
    SALT_GPU_RESIDENCY_DEVICE_LOCAL = 1
} SaltGpuResidencyPolicy;

typedef enum SaltGpuResidencyState {
    SALT_GPU_RESIDENCY_EMPTY = 0,
    SALT_GPU_RESIDENCY_POPULATING = 1,
    SALT_GPU_RESIDENCY_READY = 2
} SaltGpuResidencyState;

typedef struct SaltGpuResidencyPlan {
    uint64_t device_budget_bytes;
    uint64_t permanent_bytes;
    uint64_t slot_base_offset;
    uint64_t slot_bytes;
    uint64_t staging_bytes;
    uint32_t slot_count;
    uint32_t staging_count;
    uint32_t device_alignment;
    uint32_t flags;
} SaltGpuResidencyPlan;

typedef int (*SaltGpuResidencyReadFn)(
    void *context, uint64_t source_offset,
    void *destination, size_t nbytes);

typedef struct SaltGpuResidencySource {
    SaltGpuResourceRef resource;
    uint64_t logical_resource_id;
    uint64_t source_offset;
    uint64_t nbytes;
    SaltGpuResidencyReadFn read;
    void *read_context;
} SaltGpuResidencySource;

typedef struct SaltGpuResidencyPermanentLayout {
    uint64_t permanent_bytes;
    uint64_t maximum_span_bytes;
    uint32_t span_count;
    uint32_t alignment;
} SaltGpuResidencyPermanentLayout;

typedef struct SaltGpuResidencyPermanentSpan {
    SaltGpuResidencySource source;
    uint64_t device_offset;
} SaltGpuResidencyPermanentSpan;

typedef struct SaltGpuResidencyBackendOps {
    uint32_t abi_version;
    size_t ops_size;
    int (*create)(void *context, const SaltGpuResidencyPlan *plan);
    int (*staging)(void *context, uint32_t staging_slot,
                   void **address, size_t *capacity);
    int (*populate)(void *context, uint32_t physical_slot,
                    uint64_t device_offset, uint32_t staging_slot,
                    size_t nbytes, uint64_t ticket);
    int (*poll)(void *context, uint32_t physical_slot,
                uint64_t ticket, int *complete);
    int (*resolve)(void *context, uint64_t device_offset,
                   size_t nbytes, uintptr_t *address);
    int (*retire)(void *context, uint32_t physical_slot,
                  uint64_t generation);
    int (*destroy)(void *context);
    int (*permanent_layout)(
        void *context, SaltGpuResidencyPermanentLayout *layout,
        size_t layout_size);
    int (*permanent_span)(
        void *context, uint32_t span_index,
        SaltGpuResidencyPermanentSpan *span, size_t span_size);
    int (*permanent_copy)(
        void *context, uint32_t span_index, uint64_t device_offset,
        uint32_t staging_slot, size_t nbytes, uint64_t ticket);
    int (*permanent_publish)(void *context);
} SaltGpuResidencyBackendOps;

typedef struct SaltGpuResidencySlot {
    SaltGpuResourceRef resource;
    uint64_t logical_resource_id;
    uint64_t generation;
    uint64_t source_offset;
    uint64_t nbytes;
    uint64_t device_offset;
    uint64_t ticket;
    uint32_t state;
    uint32_t lease_count;
    uint32_t staging_slot;
    uint32_t reserved;
} SaltGpuResidencySlot;

typedef struct SaltGpuResidencyBinding {
    uint64_t logical_resource_id;
    uint64_t generation;
    uint64_t device_offset;
    uint64_t nbytes;
    uint32_t physical_slot;
    uint32_t reserved;
} SaltGpuResidencyBinding;

typedef struct SaltGpuResidencyStats {
    uint64_t planned_device_bytes;
    uint64_t permanent_device_bytes;
    uint64_t slot_capacity_bytes;
    uint64_t staging_capacity_bytes;
    uint64_t population_submissions;
    uint64_t population_completions;
    uint64_t population_aborts;
    uint64_t population_bytes_submitted;
    uint64_t population_bytes_completed;
    uint64_t source_read_failures;
    uint64_t backend_failures;
    uint64_t retirements;
    uint64_t permanent_spans;
    uint64_t permanent_chunks;
    uint64_t permanent_bytes_populated;
    uint32_t ready_slots;
    uint32_t peak_ready_slots;
    uint32_t active_leases;
    uint32_t peak_active_leases;
} SaltGpuResidencyStats;

typedef struct SaltGpuResidency {
    SaltGpuResidencyPlan plan;
    SaltGpuResidencySlot *slots;
    const SaltGpuResidencyBackendOps *backend;
    void *backend_context;
    SaltGpuResidencyStats stats;
    uint64_t next_ticket;
    uint32_t ready;
    uint32_t permanent_ready;
} SaltGpuResidency;

/* Optional physical-residency realization. Returning NULL leaves all existing
 * host/unified source-addressability behavior unchanged. */
const SaltGpuResidencyBackendOps *salt_gpu_residency_backend_ops(void);

int salt_gpu_residency_plan_validate(const SaltGpuResidencyPlan *plan);
size_t salt_gpu_residency_arena_requirement(
    const SaltGpuResidencyPlan *plan);
int salt_gpu_residency_init(
    SaltGpuResidency *residency, const SaltGpuResidencyPlan *plan,
    const SaltGpuResidencyBackendOps *backend, void *backend_context,
    void *arena, size_t arena_bytes);
int salt_gpu_residency_population_begin(
    SaltGpuResidency *residency, uint32_t physical_slot,
    uint64_t generation, const SaltGpuResidencySource *source,
    uint32_t staging_slot, uint64_t *ticket);
/* Returns 1 when this call publishes READY, 0 while still POPULATING, and -1
 * for an invalid/stale transition or backend failure. */
int salt_gpu_residency_population_poll(
    SaltGpuResidency *residency, uint32_t physical_slot,
    uint64_t logical_resource_id, uint64_t generation, uint64_t ticket);
int salt_gpu_residency_acquire(
    SaltGpuResidency *residency, uint32_t physical_slot,
    uint64_t logical_resource_id, uint64_t generation,
    SaltGpuResidencyBinding *binding);
int salt_gpu_residency_release(
    SaltGpuResidency *residency,
    const SaltGpuResidencyBinding *binding);
int salt_gpu_residency_resolve(
    const SaltGpuResidency *residency,
    const SaltGpuResidencyBinding *binding,
    uintptr_t *address);
int salt_gpu_residency_permanent_layout(
    const SaltGpuResidencyBackendOps *backend, void *backend_context,
    SaltGpuResidencyPermanentLayout *layout, size_t layout_size);
int salt_gpu_residency_populate_permanent(SaltGpuResidency *residency);
int salt_gpu_residency_population_abort(
    SaltGpuResidency *residency, uint32_t physical_slot,
    uint64_t logical_resource_id, uint64_t generation, uint64_t ticket);
int salt_gpu_residency_retire(
    SaltGpuResidency *residency, uint32_t physical_slot,
    uint64_t logical_resource_id, uint64_t generation);
int salt_gpu_residency_stats(
    const SaltGpuResidency *residency,
    SaltGpuResidencyStats *stats, size_t stats_size);
int salt_gpu_residency_destroy(SaltGpuResidency *residency);

#ifdef __cplusplus
}
#endif

#endif /* SALT_GPU_RESIDENCY_H */
