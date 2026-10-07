#include "salt/tensorops.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

static int multiply_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int multiply_size(size_t a, size_t b, size_t *out) {
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int range_valid(uint64_t offset, uint64_t bytes) {
    return bytes > 0 && offset <= UINT64_MAX - bytes;
}

int salt_tensor_storage_validate(const SaltTensorStorageSpec *storage,
                                 uint32_t rows, uint32_t cols) {
    uint64_t elements, expected_values, groups, expected_group_bytes;
    if (!storage || rows == 0 || cols == 0 ||
        storage->source_class < SALT_TENSOR_SOURCE_STATIC ||
        storage->source_class > SALT_TENSOR_SOURCE_SELECTED ||
        storage->resource_kind > SALT_TENSOR_RESOURCE_SHARED ||
        storage->encoding < SALT_TENSOR_ENCODING_F32 ||
        storage->encoding > SALT_TENSOR_ENCODING_ROW_INT2 ||
        (storage->flags & ~(uint32_t)(SALT_TENSOR_STORAGE_OPTIONAL_BIAS |
                                     SALT_TENSOR_STORAGE_DECODED_F32)) != 0 ||
        multiply_u64(rows, cols, &elements) != 0)
        return -1;
    if (storage->source_class == SALT_TENSOR_SOURCE_SELECTED) {
        if (storage->resource_kind != SALT_TENSOR_RESOURCE_EXPERT ||
            storage->logical_resource_id == UINT64_MAX)
            return -1;
    } else if (storage->logical_resource_id != 0) {
        return -1;
    }
    switch (storage->encoding) {
    case SALT_TENSOR_ENCODING_F32:
        if (multiply_u64(elements, 4u, &expected_values) != 0 ||
            storage->value_bytes != expected_values ||
            !range_valid(storage->value_offset, storage->value_bytes) ||
            storage->scale_bytes != 0 || storage->bias_bytes != 0 ||
            storage->auxiliary_bytes != 0 ||
            storage->auxiliary_2_bytes != 0 || storage->group_size != 0)
            return -1;
        break;
    case SALT_TENSOR_ENCODING_BF16:
        if (multiply_u64(elements, 2u, &expected_values) != 0 ||
            storage->value_bytes != expected_values ||
            !range_valid(storage->value_offset, storage->value_bytes) ||
            storage->scale_bytes != 0 || storage->bias_bytes != 0 ||
            storage->auxiliary_bytes != 0 ||
            storage->auxiliary_2_bytes != 0 || storage->group_size != 0)
            return -1;
        break;
    case SALT_TENSOR_ENCODING_AFFINE_Q4:
    case SALT_TENSOR_ENCODING_AFFINE_Q8: {
        uint64_t bits = storage->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4
            ? 4u : 8u;
        if (storage->group_size == 0 || cols % storage->group_size != 0 ||
            multiply_u64(elements, bits, &expected_values) != 0 ||
            expected_values % 8u != 0)
            return -1;
        expected_values /= 8u;
        groups = elements / storage->group_size;
        if (multiply_u64(groups, 2u, &expected_group_bytes) != 0 ||
            storage->value_bytes != expected_values ||
            storage->scale_bytes != expected_group_bytes ||
            !range_valid(storage->value_offset, storage->value_bytes) ||
            !range_valid(storage->scale_offset, storage->scale_bytes) ||
            (storage->bias_bytes != expected_group_bytes &&
             !((storage->flags & SALT_TENSOR_STORAGE_OPTIONAL_BIAS) != 0 &&
               storage->bias_bytes == 0)) ||
            (storage->bias_bytes != 0 &&
             !range_valid(storage->bias_offset, storage->bias_bytes)) ||
            storage->auxiliary_bytes != 0 ||
            storage->auxiliary_2_bytes != 0)
            return -1;
        break;
    }
    case SALT_TENSOR_ENCODING_ROW_INT2:
        if (multiply_u64(rows, ((uint64_t)cols + 3u) / 4u,
                         &expected_values) != 0 ||
            storage->value_bytes != expected_values ||
            storage->scale_bytes != (uint64_t)rows * 4u ||
            storage->group_size != cols || storage->flags != 0 ||
            !range_valid(storage->value_offset, storage->value_bytes) ||
            !range_valid(storage->scale_offset, storage->scale_bytes) ||
            storage->bias_bytes || storage->auxiliary_bytes ||
            storage->auxiliary_2_bytes) return -1;
        break;
    case SALT_TENSOR_ENCODING_NVFP4:
        if (cols % 16u != 0 || storage->group_size != 16u ||
            elements % 16u != 0 ||
            storage->value_bytes != elements / 2u ||
            storage->scale_bytes != elements / 16u ||
            !range_valid(storage->value_offset, storage->value_bytes) ||
            !range_valid(storage->scale_offset, storage->scale_bytes) ||
            storage->bias_bytes != 0 ||
            storage->auxiliary_bytes != 4u ||
            storage->auxiliary_2_bytes != 4u ||
            !range_valid(storage->auxiliary_offset,
                         storage->auxiliary_bytes) ||
            !range_valid(storage->auxiliary_2_offset,
                         storage->auxiliary_2_bytes))
            return -1;
        break;
    default:
        return -1;
    }
    return 0;
}

int salt_tensor_resource_validate(const SaltTensorResourceSpec *resource) {
    if (!resource || resource->kind > SALT_TENSOR_RESOURCE_SHARED ||
        resource->source_class < SALT_TENSOR_SOURCE_STATIC ||
        resource->source_class > SALT_TENSOR_SOURCE_SELECTED ||
        resource->flags != 0 || resource->nbytes == 0 ||
        resource->mapped_nbytes > resource->nbytes ||
        (resource->mapped_nbytes > 0 && !resource->base))
        return -1;
    if (resource->source_class != SALT_TENSOR_SOURCE_SELECTED &&
        (!resource->base || resource->mapped_nbytes != resource->nbytes))
        return -1;
    if (resource->source_class == SALT_TENSOR_SOURCE_SHARED &&
        resource->kind != SALT_TENSOR_RESOURCE_SHARED)
        return -1;
    if (resource->source_class == SALT_TENSOR_SOURCE_SELECTED &&
        resource->kind != SALT_TENSOR_RESOURCE_EXPERT)
        return -1;
    return 0;
}

const SaltTensorResourceSpec *salt_tensor_resource_find(
        const SaltTensorResourceSpec *resources, uint32_t resource_count,
        uint32_t kind, uint32_t resource_id) {
    if (!resources || resource_count == 0) return NULL;
    for (uint32_t index = 0; index < resource_count; index++)
        if (resources[index].kind == kind &&
            resources[index].resource_id == resource_id)
            return &resources[index];
    return NULL;
}

int salt_tensor_storage_resolves(const SaltTensorStorageSpec *storage,
                                 const SaltTensorResourceSpec *resources,
                                 uint32_t resource_count) {
    const SaltTensorResourceSpec *resource;
    const uint64_t offsets[5] = {
        storage ? storage->value_offset : 0,
        storage ? storage->scale_offset : 0,
        storage ? storage->bias_offset : 0,
        storage ? storage->auxiliary_offset : 0,
        storage ? storage->auxiliary_2_offset : 0,
    };
    const uint64_t sizes[5] = {
        storage ? storage->value_bytes : 0,
        storage ? storage->scale_bytes : 0,
        storage ? storage->bias_bytes : 0,
        storage ? storage->auxiliary_bytes : 0,
        storage ? storage->auxiliary_2_bytes : 0,
    };
    if (!storage || !(resource = salt_tensor_resource_find(resources,
            resource_count, storage->resource_kind, storage->resource_id)) ||
        salt_tensor_resource_validate(resource) != 0 ||
        resource->source_class != storage->source_class)
        return -1;
    for (uint32_t part = 0; part < 5; part++) {
        if (sizes[part] == 0) continue;
        if (offsets[part] > resource->nbytes ||
            sizes[part] > resource->nbytes - (size_t)offsets[part])
            return -1;
    }
    return 0;
}

int salt_tensor_desc_validate(const SaltTensorDesc *tensor) {
    return tensor && tensor->rows > 0 && tensor->cols > 0 &&
        tensor->stable_handle != 0 && tensor->binding && tensor->host_ops &&
        tensor->host_ops->cpu &&
        salt_tensor_storage_validate(&tensor->storage,
                                     tensor->rows, tensor->cols) == 0
        ? 0 : -1;
}

int salt_tensor_desc_absent(const SaltTensorDesc *tensor) {
    if (!tensor) return 0;
    return tensor->rows == 0 && tensor->cols == 0 &&
        tensor->stable_handle == 0 && !tensor->binding && !tensor->host_ops &&
        tensor->storage.encoding == SALT_TENSOR_ENCODING_NONE &&
        tensor->storage.source_class == SALT_TENSOR_SOURCE_NONE &&
        tensor->storage.resource_kind == 0 && tensor->storage.resource_id == 0 &&
        tensor->storage.logical_resource_id == 0 &&
        tensor->storage.group_size == 0 && tensor->storage.flags == 0 &&
        tensor->storage.value_offset == 0 && tensor->storage.scale_offset == 0 &&
        tensor->storage.bias_offset == 0 && tensor->storage.auxiliary_offset == 0 &&
        tensor->storage.auxiliary_2_offset == 0 &&
        tensor->storage.value_bytes == 0 && tensor->storage.scale_bytes == 0 &&
        tensor->storage.bias_bytes == 0 && tensor->storage.auxiliary_bytes == 0 &&
        tensor->storage.auxiliary_2_bytes == 0;
}

int salt_tensor_embedding_gather(
        const SaltTensorDesc *embedding,
        const int32_t *token_ids, uint32_t token_count,
        float embedding_scale, float *output,
        void *scratch, size_t scratch_bytes) {
    size_t elements;
    if (!embedding || !token_ids || token_count == 0 || !output ||
        !isfinite(embedding_scale) ||
        salt_tensor_desc_validate(embedding) != 0 ||
        !embedding->host_ops->cpu->gather_rows ||
        token_count > SIZE_MAX / embedding->cols)
        return -1;
    elements = (size_t)token_count * embedding->cols;
    if (embedding->host_ops->cpu->gather_rows(
            embedding, token_ids, token_count, output,
            scratch, scratch_bytes) != 0)
        return -1;
    for (size_t index = 0; index < elements; index++) {
        output[index] *= embedding_scale;
        if (!isfinite(output[index])) return -1;
    }
    return 0;
}

int salt_tensor_selected_route_group(
        const int32_t *selected_experts,
        uint32_t row_count, uint32_t topk, uint32_t expert_count,
        const float *row_inputs, uint32_t hidden,
        uint32_t *expert_offsets, uint32_t expert_offset_count,
        int32_t *grouped_to_canonical,
        int32_t *canonical_to_grouped,
        float *routed_inputs) {
    uint32_t jobs, next = 0;
    size_t row_bytes;
    if (!selected_experts || !row_inputs || !expert_offsets ||
        !grouped_to_canonical || !canonical_to_grouped || !routed_inputs ||
        row_count == 0 || topk == 0 || expert_count == 0 || hidden == 0 ||
        topk > expert_count || expert_offset_count < expert_count ||
        row_count > UINT32_MAX / topk ||
        multiply_size((size_t)hidden, sizeof(float), &row_bytes) != 0)
        return -1;
    jobs = row_count * topk;
    if (jobs > INT32_MAX) return -1;
    for (uint32_t canonical = 0; canonical < jobs; canonical++)
        if (selected_experts[canonical] < 0 ||
            (uint32_t)selected_experts[canonical] >= expert_count)
            return -1;
    memset(expert_offsets, 0,
           (size_t)expert_count * sizeof *expert_offsets);
    for (uint32_t canonical = 0; canonical < jobs; canonical++)
        expert_offsets[(uint32_t)selected_experts[canonical]]++;
    for (uint32_t expert = 0; expert < expert_count; expert++) {
        uint32_t population = expert_offsets[expert];
        expert_offsets[expert] = next;
        if (population > jobs - next) return -1;
        next += population;
    }
    if (next != jobs) return -1;
    for (uint32_t canonical = 0; canonical < jobs; canonical++) {
        uint32_t expert = (uint32_t)selected_experts[canonical];
        uint32_t grouped = expert_offsets[expert]++;
        if (grouped >= jobs) return -1;
        grouped_to_canonical[grouped] = (int32_t)canonical;
    }
    for (uint32_t grouped = 0; grouped < jobs; grouped++) {
        uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
        uint32_t row;
        if (canonical >= jobs) return -1;
        row = canonical / topk;
        canonical_to_grouped[canonical] = (int32_t)grouped;
        memcpy(routed_inputs + (size_t)grouped * hidden,
               row_inputs + (size_t)row * hidden, row_bytes);
    }
    return 0;
}

int salt_tensor_selected_weighted_reduce(
        const float *selected_weights,
        const int32_t *canonical_to_grouped,
        const float *grouped_outputs,
        uint32_t row_count, uint32_t topk, uint32_t width,
        float *outputs) {
    uint32_t jobs;
    size_t output_bytes;
    if (!selected_weights || !canonical_to_grouped || !grouped_outputs ||
        !outputs || row_count == 0 || topk == 0 || width == 0 ||
        row_count > UINT32_MAX / topk ||
        multiply_size((size_t)width, sizeof(float), &output_bytes) != 0)
        return -1;
    jobs = row_count * topk;
    for (uint32_t canonical = 0; canonical < jobs; canonical++)
        if (canonical_to_grouped[canonical] < 0 ||
            (uint32_t)canonical_to_grouped[canonical] >= jobs ||
            !isfinite(selected_weights[canonical]))
            return -1;
    for (uint32_t row = 0; row < row_count; row++) {
        float *output = outputs + (size_t)row * width;
        memset(output, 0, output_bytes);
        for (uint32_t rank = 0; rank < topk; rank++) {
            uint32_t canonical = row * topk + rank;
            uint32_t grouped =
                (uint32_t)canonical_to_grouped[canonical];
            const float *job_output =
                grouped_outputs + (size_t)grouped * width;
            for (uint32_t column = 0; column < width; column++)
                output[column] +=
                    selected_weights[canonical] * job_output[column];
        }
    }
    return 0;
}

static int tensor_graph_descriptor_validate(
        const SaltTensorGraphProgramDesc *descriptor,
        uint32_t *root_count_out, size_t *bytes_out) {
    uint32_t expected_operation = 0u;
    uint32_t expected_dependency = 0u;
    uint32_t roots = 0u;
    size_t node_bytes, dependency_bytes;
    if (root_count_out) *root_count_out = 0u;
    if (bytes_out) *bytes_out = 0u;
    if (!descriptor || !root_count_out || !bytes_out || !descriptor->nodes ||
        descriptor->node_count == 0u ||
        descriptor->logical_operation_count == 0u ||
        (descriptor->dependency_count != 0u && !descriptor->dependencies) ||
        multiply_size((size_t)descriptor->node_count,
            sizeof(SaltTensorGraphNode), &node_bytes) != 0 ||
        multiply_size((size_t)descriptor->dependency_count,
            sizeof(uint32_t), &dependency_bytes) != 0 ||
        node_bytes > SIZE_MAX - dependency_bytes)
        return -1;
    for (uint32_t node_index = 0u;
         node_index < descriptor->node_count; node_index++) {
        const SaltTensorGraphNode *node = &descriptor->nodes[node_index];
        if (node->operation_count == 0u ||
            node->operation_first != expected_operation ||
            expected_operation > UINT32_MAX - node->operation_count ||
            node->dependency_first != expected_dependency ||
            node->dependency_count >
                descriptor->dependency_count - expected_dependency ||
            (node->flags & ~SALT_TENSOR_GRAPH_NODE_RESOURCE_CHECK) != 0u)
            return -1;
        if (node->dependency_count == 0u) roots++;
        for (uint32_t dependency = 0u;
             dependency < node->dependency_count; dependency++) {
            uint32_t predecessor = descriptor->dependencies[
                node->dependency_first + dependency];
            if (predecessor >= node_index) return -1;
            for (uint32_t prior = 0u; prior < dependency; prior++)
                if (descriptor->dependencies[
                        node->dependency_first + prior] == predecessor)
                    return -1;
        }
        expected_operation += node->operation_count;
        expected_dependency += node->dependency_count;
    }
    if (roots == 0u ||
        expected_operation != descriptor->logical_operation_count ||
        expected_dependency != descriptor->dependency_count)
        return -1;
    *root_count_out = roots;
    *bytes_out = node_bytes + dependency_bytes;
    return 0;
}

int salt_tensor_graph_program_arena_requirement(
        const SaltTensorGraphProgramDesc *descriptor, size_t *bytes_out) {
    uint32_t roots;
    size_t bytes;
    if (!bytes_out || tensor_graph_descriptor_validate(
            descriptor, &roots, &bytes) != 0)
        return -1;
    (void)roots;
    *bytes_out = bytes;
    return 0;
}

int salt_tensor_graph_program_compile(
        SaltTensorGraphProgram *program,
        const SaltTensorGraphProgramDesc *descriptor,
        void *startup_arena, size_t startup_arena_bytes) {
    SaltTensorGraphNode *nodes;
    uint32_t *dependencies;
    uint32_t roots;
    size_t required, node_bytes;
    if (program) memset(program, 0, sizeof *program);
    if (!program || !startup_arena ||
        ((uintptr_t)startup_arena % sizeof(uint32_t)) != 0u ||
        tensor_graph_descriptor_validate(
            descriptor, &roots, &required) != 0 ||
        startup_arena_bytes < required ||
        multiply_size((size_t)descriptor->node_count,
            sizeof(SaltTensorGraphNode), &node_bytes) != 0)
        return -1;
    nodes = (SaltTensorGraphNode *)startup_arena;
    memcpy(nodes, descriptor->nodes, node_bytes);
    dependencies = descriptor->dependency_count == 0u ? NULL :
        (uint32_t *)((unsigned char *)startup_arena + node_bytes);
    if (descriptor->dependency_count != 0u)
        memcpy(dependencies, descriptor->dependencies,
            (size_t)descriptor->dependency_count * sizeof(uint32_t));
    program->nodes = nodes;
    program->node_count = descriptor->node_count;
    program->dependencies = dependencies;
    program->dependency_count = descriptor->dependency_count;
    program->logical_operation_count = descriptor->logical_operation_count;
    program->root_count = roots;
    program->ready = 1;
    return 0;
}

int salt_tensor_graph_runtime_bind(
        SaltTensorGraphRuntime *runtime, SaltAreaFrontier *frontier) {
    if (!runtime || !frontier || !frontier->tasks ||
        frontier->capacity == 0u || frontier->count != 0u ||
        frontier->generation != 0u)
        return -1;
    memset(runtime, 0, sizeof *runtime);
    runtime->frontier = frontier;
    return 0;
}

static int tensor_graph_node_dependencies_done(
        const SaltTensorGraphRuntime *runtime, uint32_t node_index) {
    const SaltTensorGraphNode *node;
    if (!runtime || !runtime->program || !runtime->frontier ||
        node_index >= runtime->program->node_count)
        return -1;
    node = &runtime->program->nodes[node_index];
    for (uint32_t dependency = 0u;
         dependency < node->dependency_count; dependency++) {
        uint32_t predecessor = runtime->program->dependencies[
            node->dependency_first + dependency];
        if (predecessor >= runtime->frontier->count) return -1;
        if (runtime->frontier->tasks[predecessor].state !=
                SALT_AREA_TASK_DONE)
            return 0;
    }
    return 1;
}

static void tensor_graph_terminal_cancel(
        SaltTensorGraphRuntime *runtime, SaltTensorGraphStatus status) {
    if (!runtime || !runtime->frontier) return;
    for (uint32_t node = 0u; node < runtime->frontier->count; node++)
        if (runtime->frontier->tasks[node].state != SALT_AREA_TASK_DONE)
            runtime->frontier->tasks[node].state = SALT_AREA_TASK_CANCELLED;
    runtime->running_nodes = 0u;
    runtime->waiting_nodes = 0u;
    runtime->status = status;
    runtime->active = 0;
    runtime->last_generation = runtime->generation;
}

int salt_tensor_graph_start(
        SaltTensorGraphRuntime *runtime,
        const SaltTensorGraphProgram *program,
        uint64_t generation, uint32_t phase) {
    if (!runtime || !runtime->frontier || !runtime->frontier->tasks ||
        !program || !program->ready || !program->nodes ||
        program->node_count == 0u || program->root_count == 0u ||
        (program->dependency_count != 0u && !program->dependencies) ||
        runtime->active || generation == 0u ||
        generation <= runtime->last_generation ||
        runtime->frontier->capacity < program->node_count ||
        (runtime->last_generation == 0u &&
         (runtime->frontier->generation != 0u ||
          runtime->frontier->count != 0u)) ||
        (runtime->last_generation != 0u &&
         runtime->frontier->generation != runtime->last_generation))
        return -1;
    runtime->program = program;
    runtime->generation = generation;
    runtime->phase = phase;
    runtime->completed_nodes = 0u;
    runtime->running_nodes = 0u;
    runtime->waiting_nodes = 0u;
    runtime->status = SALT_TENSOR_GRAPH_ACTIVE;
    runtime->active = 1;
    runtime->frontier->generation = generation;
    runtime->frontier->count = program->node_count;
    for (uint32_t node_index = 0u;
         node_index < program->node_count; node_index++) {
        const SaltTensorGraphNode *node = &program->nodes[node_index];
        SaltAreaTask *task = &runtime->frontier->tasks[node_index];
        *task = (SaltAreaTask) {
            .generation = generation,
            .node_id = node_index,
            .parent_id = node->dependency_count == 1u
                ? program->dependencies[node->dependency_first] : UINT32_MAX,
            .phase = phase,
            .expert_id = -1,
            .job_index = node->operation_first,
            .candidate_first = 0u,
            .candidate_count = 1u,
            .output_first = node->operation_first,
            .output_count = node->operation_count,
            .flags = 0u,
            .state = node->dependency_count == 0u
                ? SALT_AREA_TASK_QUEUED : SALT_AREA_TASK_BLOCKED,
        };
    }
    return 0;
}

int salt_tensor_graph_take_ready(
        SaltTensorGraphRuntime *runtime, uint64_t generation,
        uint32_t *node_indices, uint32_t node_capacity,
        uint32_t *node_count) {
    uint32_t taken = 0u;
    if (node_count) *node_count = 0u;
    if (!runtime || !runtime->active || !runtime->program ||
        !runtime->frontier || runtime->generation != generation ||
        runtime->frontier->generation != generation ||
        runtime->frontier->count != runtime->program->node_count ||
        (runtime->status != SALT_TENSOR_GRAPH_ACTIVE &&
         runtime->status != SALT_TENSOR_GRAPH_NEED_RESOURCE) ||
        runtime->running_nodes != 0u || !node_indices ||
        node_capacity == 0u || !node_count)
        return -1;
    for (uint32_t node = 0u;
         node < runtime->frontier->count && taken < node_capacity; node++) {
        SaltAreaTask *task = &runtime->frontier->tasks[node];
        if (task->state != SALT_AREA_TASK_QUEUED) continue;
        task->state = SALT_AREA_TASK_RUNNING;
        node_indices[taken++] = node;
    }
    runtime->running_nodes = taken;
    *node_count = taken;
    if (taken != 0u) {
        runtime->status = SALT_TENSOR_GRAPH_ACTIVE;
        return 0;
    }
    if (runtime->waiting_nodes != 0u) {
        runtime->status = SALT_TENSOR_GRAPH_NEED_RESOURCE;
        return 0;
    }
    tensor_graph_terminal_cancel(runtime, SALT_TENSOR_GRAPH_FAILED);
    return -1;
}

int salt_tensor_graph_complete_ready(
        SaltTensorGraphRuntime *runtime, uint64_t generation,
        const uint32_t *node_indices,
        const SaltTensorGraphNodeOutcome *outcomes, uint32_t node_count) {
    int failed = 0;
    uint32_t previous = UINT32_MAX;
    if (!runtime || !runtime->active || !runtime->program ||
        !runtime->frontier || runtime->generation != generation ||
        runtime->frontier->generation != generation ||
        runtime->frontier->count != runtime->program->node_count ||
        runtime->status != SALT_TENSOR_GRAPH_ACTIVE || !node_indices ||
        !outcomes || node_count == 0u ||
        node_count != runtime->running_nodes)
        return -1;
    for (uint32_t index = 0u; index < node_count; index++) {
        uint32_t node = node_indices[index];
        SaltTensorGraphNodeOutcome outcome = outcomes[index];
        if (node >= runtime->frontier->count ||
            (index != 0u && node <= previous) ||
            runtime->frontier->tasks[node].state != SALT_AREA_TASK_RUNNING ||
            (outcome != SALT_TENSOR_GRAPH_NODE_DONE &&
             outcome != SALT_TENSOR_GRAPH_NODE_NEED_RESOURCE &&
             outcome != SALT_TENSOR_GRAPH_NODE_FAILED) ||
            (outcome == SALT_TENSOR_GRAPH_NODE_NEED_RESOURCE &&
             (runtime->program->nodes[node].flags &
              SALT_TENSOR_GRAPH_NODE_RESOURCE_CHECK) == 0u))
            return -1;
        previous = node;
    }
    for (uint32_t node = 0u; node < runtime->frontier->count; node++) {
        if (runtime->frontier->tasks[node].state != SALT_AREA_TASK_RUNNING)
            continue;
        int present = 0;
        for (uint32_t index = 0u; index < node_count; index++)
            if (node_indices[index] == node) {
                present = 1;
                break;
            }
        if (!present) return -1;
    }
    for (uint32_t index = 0u; index < node_count; index++) {
        uint32_t node = node_indices[index];
        SaltAreaTask *task = &runtime->frontier->tasks[node];
        if (outcomes[index] == SALT_TENSOR_GRAPH_NODE_DONE) {
            task->state = SALT_AREA_TASK_DONE;
            runtime->completed_nodes++;
        } else if (outcomes[index] ==
                SALT_TENSOR_GRAPH_NODE_NEED_RESOURCE) {
            task->state = SALT_AREA_TASK_WAITING;
            runtime->waiting_nodes++;
        } else {
            task->state = SALT_AREA_TASK_CANCELLED;
            failed = 1;
        }
    }
    runtime->running_nodes = 0u;
    if (failed) {
        tensor_graph_terminal_cancel(runtime, SALT_TENSOR_GRAPH_FAILED);
        return -1;
    }
    for (uint32_t node = 0u; node < runtime->frontier->count; node++) {
        int ready;
        if (runtime->frontier->tasks[node].state != SALT_AREA_TASK_BLOCKED)
            continue;
        ready = tensor_graph_node_dependencies_done(runtime, node);
        if (ready < 0) {
            tensor_graph_terminal_cancel(runtime, SALT_TENSOR_GRAPH_FAILED);
            return -1;
        }
        if (ready > 0)
            runtime->frontier->tasks[node].state = SALT_AREA_TASK_QUEUED;
    }
    if (runtime->completed_nodes == runtime->program->node_count) {
        runtime->status = SALT_TENSOR_GRAPH_COMPLETE;
        runtime->active = 0;
        runtime->last_generation = generation;
        return 0;
    }
    {
        uint32_t ready_nodes = 0u;
        for (uint32_t node = 0u; node < runtime->frontier->count; node++)
            if (runtime->frontier->tasks[node].state == SALT_AREA_TASK_QUEUED)
                ready_nodes++;
        if (ready_nodes != 0u)
            runtime->status = SALT_TENSOR_GRAPH_ACTIVE;
        else if (runtime->waiting_nodes != 0u)
            runtime->status = SALT_TENSOR_GRAPH_NEED_RESOURCE;
        else {
            tensor_graph_terminal_cancel(runtime, SALT_TENSOR_GRAPH_FAILED);
            return -1;
        }
    }
    return 0;
}

int salt_tensor_graph_refire(
        SaltTensorGraphRuntime *runtime, uint64_t generation,
        const uint32_t *node_indices, uint32_t node_count) {
    uint32_t previous = UINT32_MAX;
    if (!runtime || !runtime->active || !runtime->program ||
        !runtime->frontier || runtime->generation != generation ||
        runtime->frontier->generation != generation ||
        runtime->frontier->count != runtime->program->node_count ||
        runtime->running_nodes != 0u || !node_indices || node_count == 0u ||
        (runtime->status != SALT_TENSOR_GRAPH_ACTIVE &&
         runtime->status != SALT_TENSOR_GRAPH_NEED_RESOURCE))
        return -1;
    for (uint32_t index = 0u; index < node_count; index++) {
        uint32_t node = node_indices[index];
        if (node >= runtime->frontier->count ||
            (index != 0u && node <= previous) ||
            runtime->frontier->tasks[node].state != SALT_AREA_TASK_WAITING ||
            (runtime->program->nodes[node].flags &
             SALT_TENSOR_GRAPH_NODE_RESOURCE_CHECK) == 0u)
            return -1;
        previous = node;
    }
    for (uint32_t index = 0u; index < node_count; index++) {
        runtime->frontier->tasks[node_indices[index]].state =
            SALT_AREA_TASK_QUEUED;
        runtime->waiting_nodes--;
    }
    runtime->status = SALT_TENSOR_GRAPH_ACTIVE;
    return 0;
}

int salt_tensor_graph_cancel(
        SaltTensorGraphRuntime *runtime, uint64_t generation) {
    if (!runtime || !runtime->active || !runtime->program ||
        !runtime->frontier || runtime->generation != generation ||
        runtime->frontier->generation != generation ||
        runtime->frontier->count != runtime->program->node_count ||
        runtime->running_nodes != 0u)
        return -1;
    tensor_graph_terminal_cancel(runtime, SALT_TENSOR_GRAPH_CANCELLED);
    return 0;
}

int salt_tensor_graph_progress(
        const SaltTensorGraphRuntime *runtime,
        SaltTensorGraphProgress *progress) {
    if (progress) memset(progress, 0, sizeof *progress);
    if (!runtime || !progress || !runtime->program || !runtime->frontier ||
        runtime->frontier->generation != runtime->generation ||
        runtime->frontier->count != runtime->program->node_count)
        return -1;
    progress->generation = runtime->generation;
    progress->total_nodes = runtime->program->node_count;
    progress->running_nodes = runtime->running_nodes;
    progress->waiting_nodes = runtime->waiting_nodes;
    progress->completed_nodes = runtime->completed_nodes;
    progress->status = runtime->status;
    for (uint32_t node = 0u; node < runtime->frontier->count; node++) {
        SaltAreaTaskState state = runtime->frontier->tasks[node].state;
        if (state == SALT_AREA_TASK_BLOCKED) progress->blocked_nodes++;
        else if (state == SALT_AREA_TASK_QUEUED) progress->ready_nodes++;
        else if (state == SALT_AREA_TASK_CANCELLED)
            progress->cancelled_nodes++;
    }
    return 0;
}

typedef SaltTensorHostGraphState SaltTensorHostGraphRuntime;

static void tensor_host_graph_node_worker(int worker, void *opaque) {
    SaltTensorHostGraphRuntime *runtime =
        (SaltTensorHostGraphRuntime *)opaque;
    if (!runtime || worker < 0 ||
        (uint32_t)worker >= runtime->active_workers || !runtime->node_ops ||
        !runtime->node_ops->worker || !runtime->node_seat ||
        runtime->node_ops->worker(
            runtime->node_seat, (uint32_t)worker) != 0)
        __atomic_store_n(&runtime->failed, 1u, __ATOMIC_RELEASE);
}

int salt_tensor_host_graph_execute(
        SaltTensorHostGraphState *state,
        const SaltTensorHostGraphCallbacks *callbacks, void *context,
        uint32_t worker_count, SaltTensorHostGraphParallelRun parallel_run,
        void *parallel_context, SaltTensorHostGraphResult *result) {
    uint32_t executed = 0, parallel_nodes = 0;
    if (result) memset(result, 0, sizeof *result);
    if (!state || !callbacks || !result || callbacks->node_count == 0 ||
        callbacks->logical_node_count == 0 ||
        !callbacks->is_parallel || !callbacks->prepare_parallel ||
        !callbacks->execute_serial || !context || worker_count == 0 ||
        worker_count > 32u || !parallel_run || !parallel_context || !result)
        return -1;
    memset(state, 0, sizeof *state);
    state->callbacks = callbacks;
    state->context = context;
    state->workers = worker_count;
    for (uint32_t node = 0; node < callbacks->node_count; node++) {
        int parallel = callbacks->is_parallel(context, node);
        if (parallel < 0) {
            state->failed = 1u;
            break;
        }
        if (getenv("SALT_TENSOROPS_DIAG"))
            fprintf(stderr, "SALT_TENSOR_GRAPH_NODE index=%u parallel=%u\n",
                    node, parallel > 0 ? 1u : 0u);
        if (parallel > 0) {
            int run_rc, worker_failed, finish_rc;
            state->active_workers = 0u;
            state->node_ops = NULL;
            state->node_seat = NULL;
            if (callbacks->prepare_parallel(
                    context, node, &state->node_ops, &state->node_seat,
                    &state->active_workers) != 0 ||
                !state->node_ops || !state->node_ops->worker ||
                !state->node_ops->finish || !state->node_seat ||
                state->active_workers == 0 ||
                state->active_workers > worker_count) {
                state->failed = 1u;
                break;
            }
            run_rc = parallel_run(
                parallel_context, (int)state->active_workers,
                tensor_host_graph_node_worker, state);
            worker_failed =
                __atomic_load_n(&state->failed, __ATOMIC_ACQUIRE) != 0u;
            finish_rc = state->node_ops->finish(state->node_seat);
            if (run_rc != 0 || worker_failed || finish_rc != 0) {
                state->failed = 1u;
                break;
            }
            parallel_nodes++;
        } else if (callbacks->execute_serial(context, node) != 0) {
            state->failed = 1u;
            break;
        }
        executed++;
        if (getenv("SALT_TENSOROPS_DIAG"))
            fprintf(stderr, "SALT_TENSOR_GRAPH_DONE index=%u\n", node);
    }
    result->nodes_executed = state->failed
        ? 0u : callbacks->logical_node_count;
    result->spans_executed = state->failed ? executed : callbacks->node_count;
    result->parallel_spans = state->failed ? 0u : parallel_nodes;
    result->serial_spans = state->failed
        ? 0u : callbacks->node_count - parallel_nodes;
    result->internal_barriers = state->failed ? 0u : parallel_nodes;
    return state->failed ? -1 : 0;
}

void salt_tensor_host_graph_barrier(SaltTensorHostGraphState *state) {
    uint32_t epoch = __atomic_load_n(&state->barrier_epoch, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(&state->barrier_arrived, 1u, __ATOMIC_ACQ_REL)
            == state->workers) {
        __atomic_store_n(&state->barrier_arrived, 0u, __ATOMIC_RELAXED);
        __atomic_store_n(&state->barrier_epoch, epoch + 1u, __ATOMIC_RELEASE);
    } else {
        unsigned spins = 0;
        while (__atomic_load_n(&state->barrier_epoch, __ATOMIC_ACQUIRE) == epoch)
            if (++spins == 4096u) { sched_yield(); spins = 0; }
    }
}

static void tensor_host_graph_collective_worker(int worker, void *opaque) {
    SaltTensorHostGraphState *state = opaque;
    /* Each physical worker owns this loop, not a function-pointer mailbox. */
    for (uint32_t node = 0; node < state->collective_nodes; node++) {
        int rc = state->collective_cell(state->context, node, (uint32_t)worker);
        if (rc != 0) __atomic_store_n(&state->failed, 1u, __ATOMIC_RELEASE);
        salt_tensor_host_graph_barrier(state);
        rc = (int)__atomic_load_n(&state->failed, __ATOMIC_ACQUIRE);
        /* All members observe this node's outcome before the next node can
         * publish another failure. */
        salt_tensor_host_graph_barrier(state);
        if (rc) break;
    }
}

int salt_tensor_host_graph_collective_execute(
        SaltTensorHostGraphState *state, uint32_t node_count,
        int (*cell)(void *, uint32_t, uint32_t), void *context,
        uint32_t worker_count, SaltTensorHostGraphParallelRun parallel_run,
        void *parallel_context, SaltTensorHostGraphResult *result) {
    if (result) memset(result, 0, sizeof *result);
    if (!state || !node_count || !cell || !context || !worker_count ||
        worker_count > 32u || !parallel_run || !parallel_context || !result)
        return -1;
    memset(state, 0, sizeof *state);
    state->context = context;
    state->workers = worker_count;
    state->collective_nodes = node_count;
    state->collective_cell = cell;
    if (parallel_run(parallel_context, (int)worker_count,
            tensor_host_graph_collective_worker, state) != 0 || state->failed)
        return -1;
    result->nodes_executed = node_count;
    result->spans_executed = node_count;
    result->internal_barriers = state->barrier_epoch;
    return 0;
}

int salt_tensor_program_execute(
        const SaltTensorProgramCallbacks *callbacks,
        void *context, uint32_t *executed_nodes) {
    if (executed_nodes) *executed_nodes = 0u;
    if (!callbacks || callbacks->node_count == 0 ||
        !callbacks->execute_node || !context || !executed_nodes)
        return -1;
    for (uint32_t node = 0; node < callbacks->node_count; node++) {
        if (callbacks->execute_node(context, node) != 0) return -1;
        (*executed_nodes)++;
    }
    return 0;
}
