#include "salt/gpu_residency.h"

#include <limits.h>
#include <stdint.h>
#include <string.h>

struct SaltGpuResidencySlotAlignment {
    char byte;
    SaltGpuResidencySlot slot;
};

static int u64_add(uint64_t left, uint64_t right, uint64_t *result) {
    if (!result || right > UINT64_MAX - left) return -1;
    *result = left + right;
    return 0;
}

static int u64_mul(uint64_t left, uint64_t right, uint64_t *result) {
    if (!result || (left != 0 && right > UINT64_MAX / left)) return -1;
    *result = left * right;
    return 0;
}

static void counter_add(uint64_t *counter, uint64_t value) {
    if (!counter) return;
    *counter = value > UINT64_MAX - *counter
        ? UINT64_MAX : *counter + value;
}

static size_t slot_alignment(void) {
    return offsetof(struct SaltGpuResidencySlotAlignment, slot);
}

static int power_of_two(uint32_t value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

int salt_gpu_residency_plan_validate(const SaltGpuResidencyPlan *plan) {
    uint64_t slots_bytes, slots_end, staging_bytes;
    if (!plan || plan->device_budget_bytes == 0 || plan->slot_bytes == 0 ||
        plan->staging_bytes == 0 || plan->slot_count == 0 ||
        plan->staging_count == 0 || !power_of_two(plan->device_alignment) ||
        plan->flags != 0u || plan->permanent_bytes > plan->slot_base_offset ||
        plan->slot_base_offset % plan->device_alignment != 0 ||
        plan->slot_bytes % plan->device_alignment != 0 ||
        plan->staging_bytes < plan->slot_bytes ||
        u64_mul(plan->slot_count, plan->slot_bytes, &slots_bytes) != 0 ||
        u64_mul(plan->staging_count, plan->staging_bytes,
            &staging_bytes) != 0 ||
        u64_add(plan->slot_base_offset, slots_bytes, &slots_end) != 0 ||
        slots_end > plan->device_budget_bytes)
        return -1;
    return 0;
}

size_t salt_gpu_residency_arena_requirement(
        const SaltGpuResidencyPlan *plan) {
    size_t alignment = slot_alignment();
    size_t slots_bytes;
    uint64_t slots_bytes_u64;
    if (salt_gpu_residency_plan_validate(plan) != 0 || alignment == 0 ||
        u64_mul(plan->slot_count, sizeof(SaltGpuResidencySlot),
            &slots_bytes_u64) != 0 || slots_bytes_u64 > (uint64_t)SIZE_MAX)
        return 0;
    slots_bytes = (size_t)slots_bytes_u64;
    if (alignment - 1u > SIZE_MAX - slots_bytes) return 0;
    return slots_bytes + alignment - 1u;
}

static int backend_valid(const SaltGpuResidencyBackendOps *backend) {
    return backend &&
        backend->abi_version == SALT_GPU_RESIDENCY_ABI_VERSION &&
        backend->ops_size == sizeof *backend && backend->create &&
        backend->staging && backend->populate && backend->poll &&
        backend->resolve && backend->retire && backend->destroy;
}

static uint64_t slot_device_offset(
        const SaltGpuResidency *residency, uint32_t physical_slot) {
    return residency->plan.slot_base_offset +
        (uint64_t)physical_slot * residency->plan.slot_bytes;
}

static void slot_reset(
        SaltGpuResidency *residency, uint32_t physical_slot) {
    SaltGpuResidencySlot *slot = &residency->slots[physical_slot];
    uint64_t device_offset = slot_device_offset(residency, physical_slot);
    memset(slot, 0, sizeof *slot);
    slot->device_offset = device_offset;
    slot->state = SALT_GPU_RESIDENCY_EMPTY;
}

int salt_gpu_residency_init(
        SaltGpuResidency *residency, const SaltGpuResidencyPlan *plan,
        const SaltGpuResidencyBackendOps *backend, void *backend_context,
        void *arena, size_t arena_bytes) {
    size_t required, alignment, adjustment, slots_bytes;
    uintptr_t raw, aligned;
    if (!residency || !plan || !arena ||
        (required = salt_gpu_residency_arena_requirement(plan)) == 0 ||
        arena_bytes < required || !backend_valid(backend))
        return -1;
    alignment = slot_alignment();
    raw = (uintptr_t)arena;
    if (raw > UINTPTR_MAX - (alignment - 1u)) return -1;
    aligned = (raw + alignment - 1u) /
        (uintptr_t)alignment * (uintptr_t)alignment;
    adjustment = (size_t)(aligned - raw);
    slots_bytes = (size_t)plan->slot_count * sizeof(SaltGpuResidencySlot);
    if (adjustment > arena_bytes || slots_bytes > arena_bytes - adjustment)
        return -1;
    memset(residency, 0, sizeof *residency);
    residency->plan = *plan;
    residency->slots = (SaltGpuResidencySlot *)(void *)aligned;
    memset(residency->slots, 0, slots_bytes);
    residency->backend = backend;
    residency->backend_context = backend_context;
    residency->stats.planned_device_bytes = plan->device_budget_bytes;
    residency->stats.permanent_device_bytes = plan->permanent_bytes;
    residency->stats.slot_capacity_bytes =
        (uint64_t)plan->slot_count * plan->slot_bytes;
    residency->stats.staging_capacity_bytes =
        (uint64_t)plan->staging_count * plan->staging_bytes;
    for (uint32_t slot = 0; slot < plan->slot_count; slot++)
        slot_reset(residency, slot);
    if (backend->create(backend_context, plan) != 0) {
        memset(residency->slots, 0, slots_bytes);
        memset(residency, 0, sizeof *residency);
        return -1;
    }
    residency->ready = 1u;
    return 0;
}

static int source_valid(
        const SaltGpuResidency *residency,
        const SaltGpuResidencySource *source) {
    return residency && source && source->read && source->nbytes != 0 &&
        source->nbytes <= residency->plan.slot_bytes &&
        source->nbytes <= SIZE_MAX &&
        source->source_offset <= UINT64_MAX - source->nbytes &&
        salt_gpu_resource_ref_valid(&source->resource);
}

static int slot_transition_matches(
        const SaltGpuResidencySlot *slot, uint64_t logical_resource_id,
        uint64_t generation, uint64_t ticket, uint32_t state) {
    return slot && slot->state == state &&
        slot->logical_resource_id == logical_resource_id &&
        slot->generation == generation && slot->ticket == ticket;
}

int salt_gpu_residency_population_begin(
        SaltGpuResidency *residency, uint32_t physical_slot,
        uint64_t generation, const SaltGpuResidencySource *source,
        uint32_t staging_slot, uint64_t *ticket) {
    SaltGpuResidencySlot *slot;
    void *staging_address = NULL;
    size_t staging_capacity = 0;
    uint64_t next_ticket;
    if (!residency || !residency->ready || !source || !ticket ||
        physical_slot >= residency->plan.slot_count || generation == 0 ||
        staging_slot >= residency->plan.staging_count ||
        !source_valid(residency, source))
        return -1;
    slot = &residency->slots[physical_slot];
    if (slot->state != SALT_GPU_RESIDENCY_EMPTY || slot->lease_count != 0 ||
        residency->next_ticket == UINT64_MAX)
        return -1;
    for (uint32_t index = 0; index < residency->plan.slot_count; index++)
        if (residency->slots[index].state == SALT_GPU_RESIDENCY_POPULATING &&
            residency->slots[index].staging_slot == staging_slot)
            return -1;
    if (residency->backend->staging(
            residency->backend_context, staging_slot,
            &staging_address, &staging_capacity) != 0 || !staging_address ||
        staging_capacity < source->nbytes) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    if (source->read(source->read_context, source->source_offset,
            staging_address, (size_t)source->nbytes) != 0) {
        counter_add(&residency->stats.source_read_failures, 1u);
        return -1;
    }
    next_ticket = residency->next_ticket + 1u;
    if (residency->backend->populate(
            residency->backend_context, physical_slot, slot->device_offset,
            staging_slot, (size_t)source->nbytes, next_ticket) != 0) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    slot->resource = source->resource;
    slot->logical_resource_id = source->logical_resource_id;
    slot->generation = generation;
    slot->source_offset = source->source_offset;
    slot->nbytes = source->nbytes;
    slot->ticket = next_ticket;
    slot->staging_slot = staging_slot;
    slot->state = SALT_GPU_RESIDENCY_POPULATING;
    residency->next_ticket = next_ticket;
    counter_add(&residency->stats.population_submissions, 1u);
    counter_add(&residency->stats.population_bytes_submitted, source->nbytes);
    *ticket = next_ticket;
    return 0;
}

int salt_gpu_residency_population_poll(
        SaltGpuResidency *residency, uint32_t physical_slot,
        uint64_t logical_resource_id, uint64_t generation, uint64_t ticket) {
    SaltGpuResidencySlot *slot;
    int complete = 0;
    if (!residency || !residency->ready ||
        physical_slot >= residency->plan.slot_count || generation == 0 ||
        ticket == 0)
        return -1;
    slot = &residency->slots[physical_slot];
    if (!slot_transition_matches(slot, logical_resource_id, generation,
            ticket, SALT_GPU_RESIDENCY_POPULATING))
        return -1;
    if (residency->backend->poll(
            residency->backend_context, physical_slot, ticket, &complete) != 0 ||
        (complete != 0 && complete != 1)) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    if (!complete) return 0;
    if (residency->stats.ready_slots >= residency->plan.slot_count)
        return -1;
    slot->state = SALT_GPU_RESIDENCY_READY;
    counter_add(&residency->stats.population_completions, 1u);
    counter_add(&residency->stats.population_bytes_completed, slot->nbytes);
    if (residency->stats.ready_slots != UINT32_MAX)
        residency->stats.ready_slots++;
    if (residency->stats.ready_slots > residency->stats.peak_ready_slots)
        residency->stats.peak_ready_slots = residency->stats.ready_slots;
    return 1;
}

int salt_gpu_residency_acquire(
        SaltGpuResidency *residency, uint32_t physical_slot,
        uint64_t logical_resource_id, uint64_t generation,
        SaltGpuResidencyBinding *binding) {
    SaltGpuResidencySlot *slot;
    if (!residency || !residency->ready || !binding || generation == 0 ||
        physical_slot >= residency->plan.slot_count)
        return -1;
    slot = &residency->slots[physical_slot];
    if (slot->state == SALT_GPU_RESIDENCY_EMPTY) return 0;
    if (slot->logical_resource_id != logical_resource_id ||
        slot->generation != generation)
        return -1;
    if (slot->state == SALT_GPU_RESIDENCY_POPULATING) return 0;
    if (slot->state != SALT_GPU_RESIDENCY_READY ||
        slot->lease_count == UINT32_MAX ||
        residency->stats.active_leases == UINT32_MAX)
        return -1;
    memset(binding, 0, sizeof *binding);
    binding->logical_resource_id = logical_resource_id;
    binding->generation = generation;
    binding->device_offset = slot->device_offset;
    binding->nbytes = slot->nbytes;
    binding->physical_slot = physical_slot;
    slot->lease_count++;
    residency->stats.active_leases++;
    if (residency->stats.active_leases > residency->stats.peak_active_leases)
        residency->stats.peak_active_leases = residency->stats.active_leases;
    return 1;
}

int salt_gpu_residency_release(
        SaltGpuResidency *residency,
        const SaltGpuResidencyBinding *binding) {
    SaltGpuResidencySlot *slot;
    if (!residency || !residency->ready || !binding || binding->reserved != 0 ||
        binding->physical_slot >= residency->plan.slot_count)
        return -1;
    slot = &residency->slots[binding->physical_slot];
    if (slot->state != SALT_GPU_RESIDENCY_READY ||
        slot->logical_resource_id != binding->logical_resource_id ||
        slot->generation != binding->generation ||
        slot->device_offset != binding->device_offset ||
        slot->nbytes != binding->nbytes || slot->lease_count == 0 ||
        residency->stats.active_leases == 0)
        return -1;
    slot->lease_count--;
    residency->stats.active_leases--;
    return 0;
}

int salt_gpu_residency_resolve(
        const SaltGpuResidency *residency,
        const SaltGpuResidencyBinding *binding, uintptr_t *address) {
    const SaltGpuResidencySlot *slot;
    if (!residency || !residency->ready || !binding || !address ||
        !residency->backend || binding->reserved != 0 ||
        binding->physical_slot >= residency->plan.slot_count)
        return -1;
    slot = &residency->slots[binding->physical_slot];
    if (slot->state != SALT_GPU_RESIDENCY_READY || slot->lease_count == 0 ||
        slot->logical_resource_id != binding->logical_resource_id ||
        slot->generation != binding->generation ||
        slot->device_offset != binding->device_offset ||
        slot->nbytes != binding->nbytes)
        return -1;
    *address = 0;
    if (residency->backend->resolve(residency->backend_context,
            binding->device_offset, (size_t)binding->nbytes, address) != 0 ||
        *address == 0) {
        *address = 0;
        return -1;
    }
    return 0;
}

int salt_gpu_residency_permanent_layout(
        const SaltGpuResidencyBackendOps *backend, void *backend_context,
        SaltGpuResidencyPermanentLayout *layout, size_t layout_size) {
    if (!backend_valid(backend) || !layout || layout_size != sizeof *layout ||
        !backend->permanent_layout || !backend->permanent_span ||
        !backend->permanent_copy || !backend->permanent_publish)
        return -1;
    memset(layout, 0, sizeof *layout);
    if (backend->permanent_layout(
            backend_context, layout, sizeof *layout) != 0 ||
        layout->permanent_bytes == 0 || layout->maximum_span_bytes == 0 ||
        layout->span_count == 0 || layout->alignment == 0 ||
        (layout->alignment & (layout->alignment - 1u)) != 0 ||
        layout->maximum_span_bytes > layout->permanent_bytes)
        return -1;
    return 0;
}

int salt_gpu_residency_populate_permanent(SaltGpuResidency *residency) {
    SaltGpuResidencyPermanentLayout layout;
    uint64_t previous_end = 0;
    if (!residency || !residency->ready || residency->permanent_ready ||
        salt_gpu_residency_permanent_layout(
            residency->backend, residency->backend_context,
            &layout, sizeof layout) != 0 ||
        layout.permanent_bytes != residency->plan.permanent_bytes)
        return -1;
    for (uint32_t index = 0; index < layout.span_count; index++) {
        SaltGpuResidencyPermanentSpan span;
        uint64_t done = 0;
        memset(&span, 0, sizeof span);
        if (residency->backend->permanent_span(
                residency->backend_context, index, &span, sizeof span) != 0 ||
            !span.source.read || span.source.nbytes == 0 ||
            span.source.nbytes > SIZE_MAX ||
            !salt_gpu_resource_ref_valid(&span.source.resource) ||
            span.source.source_offset > UINT64_MAX - span.source.nbytes ||
            span.device_offset < previous_end ||
            span.device_offset % layout.alignment != 0 ||
            span.device_offset > layout.permanent_bytes ||
            span.source.nbytes > layout.permanent_bytes - span.device_offset)
            return -1;
        previous_end = span.device_offset + span.source.nbytes;
        while (done < span.source.nbytes) {
            uint32_t staging_slot =
                (uint32_t)(residency->stats.permanent_chunks %
                    residency->plan.staging_count);
            void *staging_address = NULL;
            size_t staging_capacity = 0, chunk;
            uint64_t ticket;
            if (residency->backend->staging(
                    residency->backend_context, staging_slot,
                    &staging_address, &staging_capacity) != 0 ||
                !staging_address || staging_capacity == 0)
                return -1;
            chunk = staging_capacity;
            if ((uint64_t)chunk > span.source.nbytes - done)
                chunk = (size_t)(span.source.nbytes - done);
            if (span.source.read(span.source.read_context,
                    span.source.source_offset + done,
                    staging_address, chunk) != 0) {
                counter_add(&residency->stats.source_read_failures, 1u);
                return -1;
            }
            if (residency->next_ticket == UINT64_MAX) return -1;
            ticket = residency->next_ticket + 1u;
            if (residency->backend->permanent_copy(
                    residency->backend_context, index,
                    span.device_offset + done, staging_slot,
                    chunk, ticket) != 0) {
                counter_add(&residency->stats.backend_failures, 1u);
                return -1;
            }
            residency->next_ticket = ticket;
            counter_add(&residency->stats.permanent_chunks, 1u);
            counter_add(&residency->stats.permanent_bytes_populated, chunk);
            done += chunk;
        }
        counter_add(&residency->stats.permanent_spans, 1u);
    }
    if (previous_end > layout.permanent_bytes ||
        residency->stats.permanent_spans != layout.span_count ||
        residency->stats.permanent_bytes_populated > layout.permanent_bytes ||
        residency->backend->permanent_publish(
            residency->backend_context) != 0) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    residency->permanent_ready = 1u;
    return 0;
}

int salt_gpu_residency_population_abort(
        SaltGpuResidency *residency, uint32_t physical_slot,
        uint64_t logical_resource_id, uint64_t generation, uint64_t ticket) {
    SaltGpuResidencySlot *slot;
    if (!residency || !residency->ready ||
        physical_slot >= residency->plan.slot_count)
        return -1;
    slot = &residency->slots[physical_slot];
    if (!slot_transition_matches(slot, logical_resource_id, generation,
            ticket, SALT_GPU_RESIDENCY_POPULATING) || slot->lease_count != 0)
        return -1;
    if (residency->backend->retire(
            residency->backend_context, physical_slot, generation) != 0) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    counter_add(&residency->stats.population_aborts, 1u);
    slot_reset(residency, physical_slot);
    return 0;
}

int salt_gpu_residency_retire(
        SaltGpuResidency *residency, uint32_t physical_slot,
        uint64_t logical_resource_id, uint64_t generation) {
    SaltGpuResidencySlot *slot;
    if (!residency || !residency->ready || generation == 0 ||
        physical_slot >= residency->plan.slot_count)
        return -1;
    slot = &residency->slots[physical_slot];
    if (slot->state != SALT_GPU_RESIDENCY_READY ||
        slot->logical_resource_id != logical_resource_id ||
        slot->generation != generation || slot->lease_count != 0 ||
        residency->stats.ready_slots == 0)
        return -1;
    if (residency->backend->retire(
            residency->backend_context, physical_slot, generation) != 0) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    residency->stats.ready_slots--;
    counter_add(&residency->stats.retirements, 1u);
    slot_reset(residency, physical_slot);
    return 0;
}

int salt_gpu_residency_stats(
        const SaltGpuResidency *residency,
        SaltGpuResidencyStats *stats, size_t stats_size) {
    if (!residency || !residency->ready || !stats ||
        stats_size != sizeof *stats)
        return -1;
    *stats = residency->stats;
    return 0;
}

int salt_gpu_residency_destroy(SaltGpuResidency *residency) {
    size_t slots_bytes;
    if (!residency || !residency->ready || !residency->slots ||
        !residency->backend)
        return -1;
    for (uint32_t slot = 0; slot < residency->plan.slot_count; slot++)
        if (residency->slots[slot].state == SALT_GPU_RESIDENCY_POPULATING ||
            residency->slots[slot].lease_count != 0)
            return -1;
    if (residency->backend->destroy(residency->backend_context) != 0) {
        counter_add(&residency->stats.backend_failures, 1u);
        return -1;
    }
    slots_bytes = (size_t)residency->plan.slot_count *
        sizeof(SaltGpuResidencySlot);
    memset(residency->slots, 0, slots_bytes);
    memset(residency, 0, sizeof *residency);
    return 0;
}
