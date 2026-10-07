#include "salt/text_verify.h"
#include "salt/head.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int salt_text_target_frontier_validate(
        const SaltTextTargetFrontier *frontier,
        uint32_t state_slot_capacity) {
    if (!frontier || !frontier->nodes || frontier->node_count == 0 ||
        frontier->generation == 0 || state_slot_capacity == 0)
        return -1;
    for (uint32_t index = 0; index < frontier->node_count; index++) {
        const SaltTextTargetNode *node = &frontier->nodes[index];
        if (node->token_id < 0 || node->node_id == UINT32_MAX ||
            node->tentative_state_slot >= state_slot_capacity)
            return -1;
        if (node->parent_index == SALT_TEXT_TARGET_NO_PARENT) {
            if (node->depth != 0) return -1;
        } else {
            if (node->parent_index >= index ||
                node->depth != frontier->nodes[node->parent_index].depth + 1u)
                return -1;
        }
        for (uint32_t prior = 0; prior < index; prior++) {
            if (frontier->nodes[prior].node_id == node->node_id ||
                frontier->nodes[prior].tentative_state_slot ==
                    node->tentative_state_slot)
                return -1;
            if (frontier->nodes[prior].parent_index == node->parent_index &&
                frontier->nodes[prior].token_id == node->token_id)
                return -1;
        }
    }
    return 0;
}

int salt_text_target_frontier_winning_path(
        const SaltTextTargetFrontier *frontier,
        uint32_t winning_node_index,
        uint32_t *path_indices, uint32_t path_capacity,
        uint32_t *path_count) {
    uint32_t count = 0, index;
    if (path_count) *path_count = 0;
    if (!frontier || !frontier->nodes || !path_indices || !path_count ||
        winning_node_index >= frontier->node_count || path_capacity == 0)
        return -1;
    index = winning_node_index;
    for (;;) {
        if (count >= path_capacity) return -1;
        path_indices[count++] = index;
        if (frontier->nodes[index].parent_index == SALT_TEXT_TARGET_NO_PARENT)
            break;
        index = frontier->nodes[index].parent_index;
    }
    for (uint32_t left = 0; left < count / 2u; left++) {
        uint32_t right = count - 1u - left;
        uint32_t value = path_indices[left];
        path_indices[left] = path_indices[right];
        path_indices[right] = value;
    }
    *path_count = count;
    return 0;
}

int salt_text_target_frontier_build_area_tasks(
        const SaltTextTargetFrontier *target,
        SaltAreaFrontier *area, uint32_t phase, int32_t expert_id,
        uint32_t output_rows, uint32_t requested_workers) {
    uint32_t tiles_per_node, output_chunk;
    if (!target || !area || !area->tasks || target->node_count == 0 ||
        output_rows == 0 || requested_workers == 0 ||
        salt_text_target_frontier_validate(target, target->node_count) != 0 ||
        salt_area_frontier_reset(area, target->generation) != 0)
        return -1;
    tiles_per_node = requested_workers / target->node_count +
        (requested_workers % target->node_count != 0u);
    if (tiles_per_node < 1u) tiles_per_node = 1u;
    if (tiles_per_node > output_rows) tiles_per_node = output_rows;
    output_chunk = output_rows / tiles_per_node +
        (output_rows % tiles_per_node != 0u);
    for (uint32_t node_index = 0; node_index < target->node_count;
         node_index++) {
        const SaltTextTargetNode *node = &target->nodes[node_index];
        for (uint32_t output_first = 0; output_first < output_rows;
             output_first += output_chunk) {
            uint32_t output_end = output_first + output_chunk;
            SaltAreaTask task;
            if (output_end > output_rows) output_end = output_rows;
            memset(&task, 0, sizeof task);
            task.generation = target->generation;
            task.node_id = node->node_id;
            task.parent_id = node->parent_index == SALT_TEXT_TARGET_NO_PARENT
                ? UINT32_MAX : target->nodes[node->parent_index].node_id;
            task.phase = phase;
            task.expert_id = expert_id;
            task.job_index = node_index;
            task.candidate_first = node->tentative_state_slot;
            task.candidate_count = 1u;
            task.output_first = output_first;
            task.output_count = output_end - output_first;
            task.state = SALT_AREA_TASK_QUEUED;
            if (salt_area_frontier_append(area, &task) != 0)
                return -1;
        }
    }
    return area->count >= requested_workers ||
        (uint64_t)target->node_count * output_rows < requested_workers
        ? 0 : -1;
}

#define TEXT_ALIGNMENT ((size_t)16)
#define TEXT_GLOBAL_LAYER UINT32_MAX
#define TEXT_MAX_AREA_WORKERS 32u

static int text_target_policy_u32(
        const char *name, uint32_t minimum, uint32_t maximum,
        uint32_t *value) {
    const char *text = getenv(name);
    char *end = NULL;
    unsigned long parsed;
    if (!name || !value || !text || !*text || *text < '0' || *text > '9')
        return -1;
    errno = 0;
    parsed = strtoul(text, &end, 10);
    if (errno || !end || *end || parsed < minimum || parsed > maximum)
        return -1;
    *value = (uint32_t)parsed;
    return 0;
}

int salt_text_target_policy_compile_environment(
        SaltTextTargetPolicy *policy,
        SaltTextExecutionClass requested_execution_class,
        uint32_t maximum_candidates) {
    SaltTextTargetPolicy built;
    uint32_t gpu_program;
    uint64_t route_width, target_width;
    if (!policy || maximum_candidates == 0 ||
        (requested_execution_class != SALT_TEXT_EXECUTION_CPU_ONLY &&
         requested_execution_class != SALT_TEXT_EXECUTION_GPU_ONLY &&
         requested_execution_class != SALT_TEXT_EXECUTION_MIXED))
        return -1;
    memset(&built, 0, sizeof built);
    if (text_target_policy_u32(
            "SALT_TARGET_AREA_WORKERS", 8u, 20u,
            &built.worker_budget) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_ROUTE_N", 1u, maximum_candidates,
            &built.sequence_tiles) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_X", 1u, maximum_candidates,
            &built.target_rows) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_ROUTE_F", 1u, 8u,
            &built.route_count) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_ROUTE_Q", 1u, SALT_TEXT_NFQ_MAX_QUEUE,
            &built.queue_length) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_N_PARALLEL", 0u, 1u,
            &built.n_parallel) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_MATRIX_FLOW", 0u, 1u,
            &built.matrix_flow) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_CPU_GRAPH", 0u, 1u,
            &built.cpu_graph) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_KV_WARMUP_ROWS", 1u, UINT32_MAX,
            &built.kv_warmup_rows) != 0 ||
        text_target_policy_u32(
            "SALT_TARGET_WARM_X", 1u, maximum_candidates,
            &built.warm_target_rows) != 0 ||
        (getenv("SALT_DPR_INDEPENDENT_DRAFT") &&
         text_target_policy_u32(
             "SALT_DPR_INDEPENDENT_DRAFT", 0u, 1u,
             &built.dpr_independent_draft) != 0) ||
        text_target_policy_u32(
            "SALT_TARGET_GPU_PROGRAM", 0u, 1u,
            &gpu_program) != 0)
        return -1;
    route_width = (uint64_t)built.sequence_tiles * built.route_count;
    target_width = (uint64_t)built.target_rows * built.route_count;
    if (route_width == 0 || route_width > maximum_candidates ||
        target_width == 0 || target_width > maximum_candidates ||
        route_width < built.worker_budget ||
        route_width % built.worker_budget != 0u ||
        (gpu_program &&
         requested_execution_class != SALT_TEXT_EXECUTION_GPU_ONLY) ||
        (gpu_program && built.cpu_graph))
        return -1;
    built.candidate_count = (uint32_t)route_width;
    built.execution_class = gpu_program
        ? SALT_TEXT_EXECUTION_GPU_ONLY : SALT_TEXT_EXECUTION_CPU_ONLY;
    built.ready = 1;
    *policy = built;
    return 0;
}

int salt_text_target_policy_effective_x(
        const SaltTextTargetPolicy *policy, uint32_t prior_kv_rows,
        uint32_t available_rows, uint32_t *effective_x) {
    uint32_t width;
    if (!policy || !policy->ready || !effective_x || available_rows == 0u ||
        policy->target_rows == 0u || policy->kv_warmup_rows == 0u ||
        policy->warm_target_rows == 0u)
        return -1;
    width = prior_kv_rows < policy->kv_warmup_rows
        ? 1u : policy->warm_target_rows;
    if (width > policy->target_rows) width = policy->target_rows;
    if (width > available_rows) width = available_rows;
    if (width == 0u) return -1;
    *effective_x = width;
    return 0;
}

int salt_text_target_policy_active_shape(
        const SaltTextTargetPolicy *policy, uint32_t candidate_count,
        uint32_t *active_sequence_tiles, uint32_t *active_route_count) {
    uint32_t sequence_tiles, route_count;
    if (!policy || !policy->ready || !active_sequence_tiles ||
        !active_route_count || policy->sequence_tiles == 0 ||
        policy->route_count == 0 || policy->candidate_count == 0 ||
        candidate_count == 0 ||
        (uint64_t)candidate_count >
            (uint64_t)policy->target_rows * policy->route_count)
        return -1;
    /* Proposal selection reduces any configured F-lane surface to one exact
     * winning causal trajectory before TARGET.  A selected trajectory therefore
     * enters as active F1 even when the startup policy admits more lanes. */
    if (candidate_count <= policy->target_rows) {
        sequence_tiles = candidate_count;
        route_count = 1u;
    } else {
        if (candidate_count % policy->route_count != 0u) return -1;
        sequence_tiles = candidate_count / policy->route_count;
        route_count = policy->route_count;
    }
    if (sequence_tiles == 0 || sequence_tiles > policy->target_rows)
        return -1;
    *active_sequence_tiles = sequence_tiles;
    *active_route_count = route_count;
    return 0;
}

int salt_text_verify_program_bind_target_policy(
        SaltTextVerifyProgram *program, const SaltTextTargetPolicy *policy) {
    uint64_t route_width;
    if (!program || !program->ready || program->target_policy || !policy ||
        !policy->ready || policy->candidate_count == 0 ||
        policy->sequence_tiles == 0 || policy->target_rows == 0 ||
        policy->route_count == 0 ||
        policy->queue_length == 0 || policy->kv_warmup_rows == 0 ||
        policy->warm_target_rows == 0 || policy->dpr_independent_draft > 1u ||
        (policy->execution_class != SALT_TEXT_EXECUTION_CPU_ONLY &&
         policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY))
        return -1;
    route_width = (uint64_t)policy->sequence_tiles * policy->route_count;
    if (route_width > program->maximum_candidates ||
        (uint64_t)policy->target_rows * policy->route_count >
            program->maximum_candidates ||
        route_width != policy->candidate_count)
        return -1;
    program->target_policy = policy;
    return 0;
}

static int checked_add(size_t a, size_t b, size_t *out) {
    if (!out || a > SIZE_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static int checked_mul(size_t a, size_t b, size_t *out) {
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int align_size(size_t value, size_t alignment, size_t *out) {
    size_t remainder;
    if (!out || alignment == 0) return -1;
    remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return 0;
    }
    return checked_add(value, alignment - remainder, out);
}

static int reserve_bytes(size_t *cursor, size_t count, size_t element_size,
                         size_t *offset_out) {
    size_t bytes, aligned;
    if (!cursor || !offset_out ||
        align_size(*cursor, TEXT_ALIGNMENT, &aligned) != 0 ||
        checked_mul(count, element_size, &bytes) != 0 ||
        checked_add(aligned, bytes, cursor) != 0)
        return -1;
    *offset_out = aligned;
    return 0;
}

static int touched_span_add(
        const SaltTextVerifyProgram *program,
        SaltTextTouchedSpan *spans, uint32_t capacity, uint32_t *count,
        size_t offset, size_t bytes) {
    size_t start = offset, end;
    if (!program || !spans || !count || bytes == 0 ||
        offset > program->layout.total_bytes ||
        bytes > program->layout.total_bytes - offset)
        return -1;
    end = offset + bytes;
    for (uint32_t index = 0; index < *count;) {
        size_t other_start = spans[index].offset;
        size_t other_end = other_start + spans[index].bytes;
        if (end < other_start || other_end < start) {
            index++;
            continue;
        }
        if (other_start < start) start = other_start;
        if (other_end > end) end = other_end;
        spans[index] = spans[--(*count)];
    }
    if (*count >= capacity) return -1;
    spans[*count].offset = start;
    spans[*count].bytes = end - start;
    (*count)++;
    return 0;
}

static int touched_count_bytes(size_t count, size_t width, size_t element,
                               size_t *bytes) {
    size_t values;
    return checked_mul(count, width, &values) == 0
        ? checked_mul(values, element, bytes) : -1;
}

int salt_text_attention_score_rows(
        const SaltTextVerifyProgram *program, uint32_t source_position,
        uint32_t maximum_depth, uint32_t *score_rows) {
    uint32_t live_rows, maximum = 0;
    if (!program || !program->ready || !score_rows ||
        source_position >= program->maximum_context ||
        maximum_depth >= program->maximum_context - source_position)
        return -1;
    live_rows = source_position + maximum_depth + 1u;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextLayerExecDesc *descriptor =
            program->layers[layer].descriptor;
        const SaltAttentionDesc *attention;
        uint32_t rows = live_rows;
        if (!descriptor || !descriptor->plan) return -1;
        attention = &descriptor->plan->attention;
        if (attention->kind != SALT_ATTN_FULL) {
            if (attention->window < 1) return -1;
            if (rows > (uint32_t)attention->window)
                rows = (uint32_t)attention->window;
        }
        if (rows > maximum) maximum = rows;
    }
    if (maximum == 0 || maximum > program->maximum_context) return -1;
    *score_rows = maximum;
    return 0;
}

static int text_touched_span_plan_outputs(
        const SaltTextVerifyProgram *program, uint32_t candidate_count,
        uint32_t attention_score_rows, uint32_t output_rows,
        SaltTextTouchedSpan *spans, uint32_t span_capacity,
        uint32_t *span_count, uint64_t *total_bytes) {
    uint32_t count = 0;
    size_t bytes;
    if (!program || !program->ready || !spans || span_capacity == 0 ||
        !span_count || !total_bytes || candidate_count == 0 ||
        candidate_count > program->maximum_candidates ||
        output_rows > candidate_count ||
        attention_score_rows == 0 ||
        attention_score_rows > program->maximum_context)
        return -1;
    memset(spans, 0, (size_t)span_capacity * sizeof *spans);
#define ADD_COUNT(off, n, width, type) \
    do { \
        if (touched_count_bytes((n), (width), sizeof(type), &bytes) != 0 || \
            touched_span_add(program, spans, span_capacity, &count, \
                (off), bytes) != 0) return -1; \
    } while (0)
    ADD_COUNT(program->layout.candidate_token_ids, candidate_count, 1u, int32_t);
    ADD_COUNT(program->layout.target_parent_rows, candidate_count, 1u, uint32_t);
    ADD_COUNT(program->layout.target_depths, candidate_count, 1u, uint32_t);
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        uint32_t actual = candidate_count;
        size_t offset = cell->destination_offset;
        if (cell->kind == SALT_TEXT_CELL_FINAL_NORM ||
                cell->kind == SALT_TEXT_CELL_FINAL_HEAD ||
                cell->kind == SALT_TEXT_CELL_LOGIT_SOFTCAP) {
            actual = output_rows;
            if (actual == 0) continue;
            if (checked_mul((size_t)(candidate_count - actual),
                    cell->destination_stride, &bytes) != 0 ||
                checked_add(offset, bytes, &offset) != 0)
                return -1;
        }
        if (cell->unit == SALT_TEXT_EXECUTION_JOBS) {
            if (cell->layer >= program->layer_count ||
                candidate_count > UINT32_MAX /
                    (uint32_t)program->layers[cell->layer].descriptor->
                        plan->top_k_experts)
                return -1;
            actual = candidate_count *
                (uint32_t)program->layers[cell->layer].descriptor->
                    plan->top_k_experts;
        } else if (cell->unit != SALT_TEXT_EXECUTION_ROWS) {
            return -1;
        }
        if (checked_mul((size_t)actual, cell->destination_stride, &bytes) != 0 ||
            touched_span_add(program, spans, span_capacity, &count,
                offset, bytes) != 0)
            return -1;
    }
    ADD_COUNT(program->layout.selected_weights, candidate_count,
        program->layout.maximum_topk, float);
    ADD_COUNT(program->layout.grouped_to_canonical, candidate_count,
        program->layout.maximum_topk, int32_t);
    ADD_COUNT(program->layout.canonical_to_grouped, candidate_count,
        program->layout.maximum_topk, int32_t);
    ADD_COUNT(program->layout.routed_input, candidate_count,
        (size_t)program->layout.maximum_topk * program->hidden, float);
    ADD_COUNT(program->layout.combine_b, candidate_count, program->hidden, float);
    ADD_COUNT(program->layout.attention_scores, 1u,
        attention_score_rows, float);
    ADD_COUNT(program->layout.tensor_row, 1u,
        program->layout.maximum_tensor_row, float);
    for (uint32_t layer = 0; layer < program->layer_count; layer++)
        ADD_COUNT(program->layers[layer].tentative_value_offset,
            candidate_count, program->layers[layer].kv_width, float);
#undef ADD_COUNT
    *total_bytes = 0;
    for (uint32_t index = 0; index < count; index++) {
        if (*total_bytes > UINT64_MAX - spans[index].bytes) return -1;
        *total_bytes += spans[index].bytes;
    }
    *span_count = count;
    return 0;
}

int salt_text_touched_span_plan(
        const SaltTextVerifyProgram *program, uint32_t candidate_count,
        uint32_t attention_score_rows,
        SaltTextTouchedSpan *spans, uint32_t span_capacity,
        uint32_t *span_count, uint64_t *total_bytes) {
    return text_touched_span_plan_outputs(
        program, candidate_count, attention_score_rows, candidate_count,
        spans, span_capacity, span_count, total_bytes);
}

int salt_text_touched_span_plan_outputs(
        const SaltTextVerifyProgram *program, uint32_t candidate_count,
        uint32_t attention_score_rows, uint32_t output_rows,
        SaltTextTouchedSpan *spans, uint32_t span_capacity,
        uint32_t *span_count, uint64_t *total_bytes) {
    if (output_rows > 1u) return -1;
    return text_touched_span_plan_outputs(
        program, candidate_count, attention_score_rows, output_rows,
        spans, span_capacity, span_count, total_bytes);
}

int salt_text_touched_span_clear(
        unsigned char *base, size_t capacity,
        const SaltTextTouchedSpan *spans, uint32_t span_count,
        uint64_t *cleared_bytes) {
    uint64_t total = 0;
    if (!base || !spans || span_count == 0 || !cleared_bytes)
        return -1;
    for (uint32_t index = 0; index < span_count; index++) {
        if (spans[index].bytes == 0 || spans[index].offset > capacity ||
            spans[index].bytes > capacity - spans[index].offset ||
            total > UINT64_MAX - spans[index].bytes)
            return -1;
        memset(base + spans[index].offset, 0, spans[index].bytes);
        total += spans[index].bytes;
    }
    *cleared_bytes = total;
    return 0;
}

static int tensor_shape(const SaltTextTensorDesc *tensor,
                        uint32_t rows, uint32_t cols) {
    int valid = tensor && tensor->rows == rows && tensor->cols == cols &&
        salt_tensor_desc_validate(tensor) == 0;
    if (!valid && getenv("SALT_TENSOROPS_DIAG"))
        fprintf(stderr,
            "SALT_TENSOR_SHAPE_FAIL handle=%llu actual=%ux%u expected=%ux%u "
            "encoding=%d source=%d resource=%u:%u\n",
            (unsigned long long)(tensor ? tensor->stable_handle : 0),
            tensor ? tensor->rows : 0, tensor ? tensor->cols : 0,
            rows, cols, tensor ? (int)tensor->storage.encoding : -1,
            tensor ? (int)tensor->storage.source_class : -1,
            tensor ? tensor->storage.resource_kind : 0,
            tensor ? tensor->storage.resource_id : 0);
    return valid;
}

static int tensor_absent(const SaltTextTensorDesc *tensor) {
    return salt_tensor_desc_absent(tensor);
}

static int tensor_resolves(const SaltTextModelExecDesc *descriptor,
                           const SaltTextTensorDesc *tensor) {
    /* Required tensors are checked by shape; an optional absent cell has no
     * resource to resolve. Absence must be the complete zero descriptor. */
    if (descriptor && tensor_absent(tensor)) return 1;
    int valid = descriptor && tensor &&
        salt_tensor_storage_resolves(&tensor->storage,
            descriptor->tensor_resources,
            descriptor->tensor_resource_count) == 0;
    if (!valid && getenv("SALT_TENSOROPS_DIAG")) {
        const SaltTensorResourceSpec *resource = descriptor && tensor
            ? salt_tensor_resource_find(descriptor->tensor_resources,
                descriptor->tensor_resource_count,
                tensor->storage.resource_kind,
                tensor->storage.resource_id)
            : NULL;
        fprintf(stderr,
            "SALT_TENSOR_RESOURCE_FAIL handle=%llu source=%d resource=%u:%u "
            "logical=%llu values=%llu+%llu scales=%llu+%llu "
            "bias=%llu+%llu resources=%u found_source=%d found_bytes=%llu\n",
            (unsigned long long)(tensor ? tensor->stable_handle : 0),
            tensor ? (int)tensor->storage.source_class : -1,
            tensor ? tensor->storage.resource_kind : 0,
            tensor ? tensor->storage.resource_id : 0,
            (unsigned long long)(tensor ?
                tensor->storage.logical_resource_id : 0),
            (unsigned long long)(tensor ? tensor->storage.value_offset : 0),
            (unsigned long long)(tensor ? tensor->storage.value_bytes : 0),
            (unsigned long long)(tensor ? tensor->storage.scale_offset : 0),
            (unsigned long long)(tensor ? tensor->storage.scale_bytes : 0),
            (unsigned long long)(tensor ? tensor->storage.bias_offset : 0),
            (unsigned long long)(tensor ? tensor->storage.bias_bytes : 0),
            descriptor ? descriptor->tensor_resource_count : 0,
            resource ? (int)resource->source_class : -1,
            (unsigned long long)(resource ? resource->nbytes : 0));
    }
    return valid;
}

static int resource_table_valid(const SaltTextModelExecDesc *descriptor) {
    if (!descriptor || !descriptor->tensor_resources ||
        descriptor->tensor_resource_count == 0)
        return 0;
    for (uint32_t index = 0; index < descriptor->tensor_resource_count;
         index++) {
        const SaltTensorResourceSpec *resource =
            &descriptor->tensor_resources[index];
        if (salt_tensor_resource_validate(resource) != 0) return 0;
        for (uint32_t prior = 0; prior < index; prior++)
            if (descriptor->tensor_resources[prior].kind == resource->kind &&
                descriptor->tensor_resources[prior].resource_id ==
                    resource->resource_id)
                return 0;
    }
    return 1;
}

static int width_product(int a, int b, uint32_t *out) {
    uint64_t product;
    if (!out || a < 1 || b < 1) return -1;
    product = (uint64_t)(uint32_t)a * (uint64_t)(uint32_t)b;
    if (product > UINT32_MAX) return -1;
    *out = (uint32_t)product;
    return 0;
}

static int readiness_bit(int value) {
    return value == 0 || value == 1;
}

static int kv_shared_prefix(
        const SaltTextKvRowsDesc *rows, const float **prefix,
        size_t *capacity, size_t *stride, uint32_t *row_count) {
    if (!rows || !prefix || !capacity || !stride || !row_count) return -1;
    if (rows->shared_prefix_state) {
        const SaltTextKvSharedPrefixState *state = rows->shared_prefix_state;
        if (rows->shared_prefix || rows->shared_prefix_float_capacity != 0 ||
            rows->shared_prefix_row_stride != 0 || rows->shared_prefix_rows != 0)
            return -1;
        *prefix = state->rows;
        *capacity = state->float_capacity;
        *stride = state->row_stride;
        *row_count = state->row_count;
    } else {
        *prefix = rows->shared_prefix;
        *capacity = rows->shared_prefix_float_capacity;
        *stride = rows->shared_prefix_row_stride;
        *row_count = rows->shared_prefix_rows;
    }
    return 0;
}

static int kv_rows_valid(const SaltTextKvRowsDesc *rows,
                         uint32_t width) {
    const float *shared;
    size_t shared_capacity, shared_stride;
    uint32_t shared_rows;
    size_t required;
    if (!rows || width == 0 ||
        (rows->private_mode != SALT_TEXT_KV_PRIVATE_ABSOLUTE &&
         rows->private_mode != SALT_TEXT_KV_PRIVATE_RING) ||
        !rows->private_rows || rows->private_row_capacity == 0 ||
        rows->private_row_stride < width ||
        kv_shared_prefix(rows, &shared, &shared_capacity,
                         &shared_stride, &shared_rows) != 0 ||
        rows->private_position_base > shared_rows ||
        (rows->private_mode == SALT_TEXT_KV_PRIVATE_RING &&
         rows->private_position_base != shared_rows) ||
        checked_mul((size_t)rows->private_row_capacity,
                    rows->private_row_stride, &required) != 0 ||
        rows->private_float_capacity < required)
        return 0;
    if (shared_rows == 0)
        return shared == NULL && shared_capacity == 0 && shared_stride == 0;
    if (!shared || shared_stride < width ||
        checked_mul((size_t)shared_rows, shared_stride, &required) != 0 ||
        shared_capacity < required)
        return 0;
    return 1;
}

static const float *kv_committed_row(
        const SaltTextKvRowsDesc *rows,
        uint32_t source_position,
        uint32_t position) {
    const float *shared;
    size_t shared_capacity, shared_stride;
    uint32_t shared_rows;
    uint32_t logical, index, oldest;
    if (!rows || position >= source_position ||
        kv_shared_prefix(rows, &shared, &shared_capacity,
                         &shared_stride, &shared_rows) != 0)
        return NULL;
    (void)shared_capacity;
    if (position < shared_rows)
        return shared + (size_t)position * shared_stride;
    if (position < rows->private_position_base) return NULL;
    logical = position - rows->private_position_base;
    if (rows->private_mode == SALT_TEXT_KV_PRIVATE_ABSOLUTE) {
        if (logical >= rows->private_row_capacity) return NULL;
        index = logical;
    } else {
        oldest = source_position > rows->private_row_capacity
            ? source_position - rows->private_row_capacity : 0u;
        if (position < oldest) return NULL;
        index = logical % rows->private_row_capacity;
    }
    return rows->private_rows + (size_t)index * rows->private_row_stride;
}

int salt_text_kv_read_view_init(
        SaltTextKvReadView *view,
        const SaltTextKvRowsDesc *committed_rows,
        const float *tentative_rows,
        size_t tentative_float_capacity,
        size_t tentative_row_stride,
        uint32_t tentative_row_count,
        uint32_t source_position,
        const uint32_t *parent_rows,
        const uint32_t *depths,
        uint32_t width) {
    SaltTextKvReadView built;
    size_t required;
    if (!view || !committed_rows || !tentative_rows ||
        tentative_row_count == 0 || tentative_row_stride < width ||
        !kv_rows_valid(committed_rows, width) ||
        checked_mul((size_t)tentative_row_count,
                    tentative_row_stride, &required) != 0 ||
        tentative_float_capacity < required ||
        ((parent_rows == NULL) != (depths == NULL)))
        return -1;
    if (parent_rows) {
        for (uint32_t row = 0; row < tentative_row_count; row++) {
            if (depths[row] == 0u) {
                if (parent_rows[row] != UINT32_MAX) return -1;
            } else {
                uint32_t parent = parent_rows[row];
                if (parent >= tentative_row_count ||
                    depths[parent] + 1u != depths[row])
                    return -1;
            }
        }
    }
    memset(&built, 0, sizeof built);
    built.committed_rows = committed_rows;
    built.tentative_rows = tentative_rows;
    built.tentative_float_capacity = tentative_float_capacity;
    built.tentative_row_stride = tentative_row_stride;
    built.tentative_row_count = tentative_row_count;
    built.source_position = source_position;
    built.parent_rows = parent_rows;
    built.depths = depths;
    *view = built;
    return 0;
}

const float *salt_text_kv_read_view_row(
        const SaltTextKvReadView *view,
        uint32_t current_row,
        uint32_t position) {
    uint32_t row, target_depth, steps = 0;
    if (!view || !view->committed_rows || !view->tentative_rows ||
        current_row >= view->tentative_row_count)
        return NULL;
    if (position < view->source_position)
        return kv_committed_row(
            view->committed_rows, view->source_position, position);
    target_depth = position - view->source_position;
    if (!view->parent_rows) {
        row = target_depth;
        if (row > current_row || row >= view->tentative_row_count)
            return NULL;
    } else {
        row = current_row;
        while (view->depths[row] > target_depth) {
            if (++steps > view->tentative_row_count) return NULL;
            row = view->parent_rows[row];
            if (row == UINT32_MAX || row >= view->tentative_row_count)
                return NULL;
        }
        if (view->depths[row] != target_depth) return NULL;
    }
    return view->tentative_rows +
        (size_t)row * view->tentative_row_stride;
}

static int descriptor_summary(const SaltTextModelExecDesc *descriptor,
                              uint32_t maximum_candidates,
                              SaltTextCanonicalLayout *layout,
                              uint32_t *cell_count_out,
                              size_t *tentative_kv_bytes_out) {
    const SaltModelDesc *model;
    SaltTextCanonicalLayout result;
    size_t cursor = 0, candidates, jobs, count, tentative_bytes = 0;
    uint32_t cell_count = 4;
    if (!descriptor || !layout || !cell_count_out || !tentative_kv_bytes_out ||
        !(model = descriptor->model) || !descriptor->layers ||
        !resource_table_valid(descriptor) ||
        descriptor->layer_count == 0 || model->n_layers < 1 ||
        descriptor->layer_count != (uint32_t)model->n_layers ||
        model->hidden < 1 || !readiness_bit(model->runtime_ready) ||
        descriptor->vocabulary == 0 ||
        descriptor->vocabulary > (uint32_t)INT32_MAX ||
        descriptor->maximum_context == 0 || maximum_candidates == 0 ||
        maximum_candidates > descriptor->maximum_context ||
        !(descriptor->norm_epsilon >= 0.0f) ||
        !isfinite(descriptor->norm_epsilon) ||
        !isfinite(descriptor->embedding_scale) ||
        !tensor_shape(&descriptor->embedding, descriptor->vocabulary,
                      (uint32_t)model->hidden) ||
        !tensor_shape(&descriptor->final_norm, 1u,
                      (uint32_t)model->hidden) ||
        !tensor_shape(&descriptor->output_head, descriptor->vocabulary,
                      (uint32_t)model->hidden) ||
        !tensor_resolves(descriptor, &descriptor->embedding) ||
        !tensor_resolves(descriptor, &descriptor->final_norm) ||
        !tensor_resolves(descriptor, &descriptor->output_head))
        return -1;
    memset(&result, 0, sizeof result);
    candidates = maximum_candidates;
    result.maximum_tensor_row = (uint32_t)model->hidden;
    for (uint32_t layer_index = 0; layer_index < descriptor->layer_count;
         layer_index++) {
        const SaltTextLayerExecDesc *layer = &descriptor->layers[layer_index];
        const SaltTextLayerPlan *plan = layer->plan;
        const SaltAttentionDesc *attention;
        uint32_t query_width, kv_width;
        size_t layer_tentative;
        if (!plan || !layer->kv || plan->layer != (int)layer_index ||
            !readiness_bit(plan->runtime_ready) ||
            plan->runtime_ready != model->runtime_ready ||
            plan->hidden != model->hidden ||
            plan->attention.kind == SALT_ATTN_NONE ||
            (plan->attention.kind != SALT_ATTN_FULL &&
             plan->attention.kind != SALT_ATTN_SLIDING) ||
            !plan->attention.causal || plan->attention.n_heads < 1 ||
            plan->attention.n_kv_heads < 1 ||
            plan->attention.n_heads % plan->attention.n_kv_heads != 0 ||
            plan->attention.head_dim < 2 ||
            plan->attention.head_dim % 2 != 0 ||
            plan->attention.rope_dim < 0 ||
            plan->attention.rope_dim > plan->attention.head_dim ||
            plan->attention.rope_dim % 2 != 0 ||
            plan->attention.rope_base_dim < plan->attention.rope_dim ||
            plan->attention.rope_base_dim > plan->attention.head_dim ||
            (plan->attention.rope_kind == SALT_ROPE_NONE &&
             plan->attention.rope_dim != 0) ||
            (plan->attention.rope_kind != SALT_ROPE_NONE &&
             plan->attention.rope_dim < 2) ||
            plan->attention.rope_kind > SALT_ROPE_PARTIAL_F32 ||
            !(plan->attention.rope_theta > 0.0) ||
            !isfinite(plan->attention.rope_theta) ||
            !isfinite(plan->attention.score_scale) ||
            (plan->attention.kind == SALT_ATTN_SLIDING &&
             plan->attention.window < 1) ||
            plan->dense_intermediate < 0 || plan->n_experts < 1 ||
            plan->top_k_experts < 1 ||
            plan->top_k_experts > plan->n_experts ||
            plan->expert_intermediate < 1 ||
            !readiness_bit(plan->router_rmsnorm) ||
            !readiness_bit(plan->residual_postnorm) ||
            !readiness_bit(plan->router_rank_order) ||
            !readiness_bit(plan->attention.raw_values) ||
            plan->parallel_dense_routed != (plan->dense_intermediate > 0) ||
            (plan->activation == SALT_ACT_SILU_CLAMPED &&
             (!(plan->activation_limit > 0.0f) || !isfinite(plan->activation_limit))) ||
            (plan->activation != SALT_ACT_GELU_TANH &&
             plan->activation != SALT_ACT_SILU &&
             plan->activation != SALT_ACT_SILU_CLAMPED) ||
            !(plan->logit_softcap >= 0.0f) || !isfinite(plan->logit_softcap) ||
            layer->expert_count != (uint32_t)plan->n_experts ||
            !layer->experts ||
            width_product(plan->attention.n_heads,
                          plan->attention.head_dim, &query_width) != 0 ||
            width_product(plan->attention.n_kv_heads,
                          plan->attention.head_dim, &kv_width) != 0)
            return -1;
        attention = &plan->attention;
        if (!tensor_shape(&layer->q, query_width,
                          (uint32_t)model->hidden) ||
            !tensor_shape(&layer->k, kv_width,
                          (uint32_t)model->hidden) ||
            !(attention->shared_kv_projection
                  ? tensor_absent(&layer->v)
                  : tensor_shape(&layer->v, kv_width,
                                 (uint32_t)model->hidden)) ||
            !tensor_shape(&layer->o, (uint32_t)model->hidden,
                          query_width) ||
            !(plan->dense_intermediate == 0 ? tensor_absent(&layer->dense_gate) : tensor_shape(&layer->dense_gate,
                          (uint32_t)plan->dense_intermediate,
                          (uint32_t)model->hidden)) ||
            !(plan->dense_intermediate == 0 ? tensor_absent(&layer->dense_up) : tensor_shape(&layer->dense_up,
                          (uint32_t)plan->dense_intermediate,
                          (uint32_t)model->hidden)) ||
            !(plan->dense_intermediate == 0 ? tensor_absent(&layer->dense_down) : tensor_shape(&layer->dense_down,
                          (uint32_t)model->hidden,
                          (uint32_t)plan->dense_intermediate)) ||
            !tensor_shape(&layer->router, (uint32_t)plan->n_experts,
                          (uint32_t)model->hidden) ||
            !tensor_shape(&layer->pre_attention_norm, 1u,
                          (uint32_t)model->hidden) ||
            !tensor_shape(&layer->q_norm, 1u,
                          (uint32_t)attention->head_dim) ||
            !tensor_shape(&layer->k_norm, 1u,
                          (uint32_t)attention->head_dim) ||
            !(plan->residual_postnorm ? tensor_shape(&layer->post_attention_norm, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->post_attention_norm)) ||
            !(plan->dense_intermediate ? tensor_shape(&layer->pre_ffn_norm_1, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->pre_ffn_norm_1)) ||
            !tensor_shape(&layer->pre_ffn_norm_2, 1u,
                          (uint32_t)model->hidden) ||
            !(plan->residual_postnorm ? tensor_shape(&layer->post_ffn_norm_1, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->post_ffn_norm_1)) ||
            !(plan->residual_postnorm ? tensor_shape(&layer->post_ffn_norm_2, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->post_ffn_norm_2)) ||
            !(plan->residual_postnorm ? tensor_shape(&layer->post_ffn_norm, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->post_ffn_norm)) ||
            !(plan->router_rmsnorm ? tensor_shape(&layer->router_scale, 1u,
                          (uint32_t)model->hidden) : tensor_absent(&layer->router_scale)) ||
            !(plan->router_rank_order ? tensor_absent(&layer->per_expert_scale) :
              tensor_shape(&layer->per_expert_scale, 1u,
                          (uint32_t)plan->n_experts)) ||
            !(plan->final_layer_scale
                  ? tensor_shape(&layer->layer_scalar, 1u, 1u)
                  : tensor_absent(&layer->layer_scalar)) ||
            !kv_rows_valid(&layer->kv->keys, kv_width) ||
            !kv_rows_valid(&layer->kv->values, kv_width) ||
            layer->kv->keys.private_mode !=
                layer->kv->values.private_mode ||
            layer->kv->keys.private_row_capacity !=
                layer->kv->values.private_row_capacity ||
            layer->kv->keys.private_position_base !=
                layer->kv->values.private_position_base ||
            layer->kv->keys.shared_prefix_rows !=
                layer->kv->values.shared_prefix_rows ||
            !tensor_resolves(descriptor, &layer->q) ||
            !tensor_resolves(descriptor, &layer->k) ||
            (!tensor_absent(&layer->v) &&
             !tensor_resolves(descriptor, &layer->v)) ||
            !tensor_resolves(descriptor, &layer->o) ||
            !tensor_resolves(descriptor, &layer->dense_gate) ||
            !tensor_resolves(descriptor, &layer->dense_up) ||
            !tensor_resolves(descriptor, &layer->dense_down) ||
            !tensor_resolves(descriptor, &layer->router) ||
            !tensor_resolves(descriptor, &layer->pre_attention_norm) ||
            !tensor_resolves(descriptor, &layer->q_norm) ||
            !tensor_resolves(descriptor, &layer->k_norm) ||
            !tensor_resolves(descriptor, &layer->post_attention_norm) ||
            !tensor_resolves(descriptor, &layer->pre_ffn_norm_1) ||
            !tensor_resolves(descriptor, &layer->pre_ffn_norm_2) ||
            !tensor_resolves(descriptor, &layer->post_ffn_norm_1) ||
            !tensor_resolves(descriptor, &layer->post_ffn_norm_2) ||
            !tensor_resolves(descriptor, &layer->post_ffn_norm) ||
            !tensor_resolves(descriptor, &layer->router_scale) ||
            !tensor_resolves(descriptor, &layer->per_expert_scale) ||
            (!tensor_absent(&layer->layer_scalar) &&
             !tensor_resolves(descriptor, &layer->layer_scalar)))
            return -1;
        for (uint32_t expert = 0; expert < layer->expert_count; expert++) {
            const SaltTextExpertDesc *entry = &layer->experts[expert];
            if (entry->expert_id != expert ||
                !tensor_shape(&entry->gate,
                              (uint32_t)plan->expert_intermediate,
                              (uint32_t)model->hidden) ||
                !tensor_shape(&entry->up,
                              (uint32_t)plan->expert_intermediate,
                              (uint32_t)model->hidden) ||
                !tensor_shape(&entry->down, (uint32_t)model->hidden,
                              (uint32_t)plan->expert_intermediate) ||
                !tensor_resolves(descriptor, &entry->gate) ||
                !tensor_resolves(descriptor, &entry->up) ||
                !tensor_resolves(descriptor, &entry->down))
                return -1;
        }
        if (query_width > result.maximum_query_width)
            result.maximum_query_width = query_width;
        if (kv_width > result.maximum_kv_width)
            result.maximum_kv_width = kv_width;
        if ((uint32_t)plan->dense_intermediate > result.maximum_dense_width)
            result.maximum_dense_width = (uint32_t)plan->dense_intermediate;
        if ((uint32_t)plan->n_experts > result.maximum_experts)
            result.maximum_experts = (uint32_t)plan->n_experts;
        if ((uint32_t)plan->top_k_experts > result.maximum_topk)
            result.maximum_topk = (uint32_t)plan->top_k_experts;
        if ((uint32_t)plan->expert_intermediate >
                result.maximum_expert_width)
            result.maximum_expert_width =
                (uint32_t)plan->expert_intermediate;
        if ((uint32_t)attention->head_dim > result.maximum_tensor_row)
            result.maximum_tensor_row = (uint32_t)attention->head_dim;
        if ((uint32_t)plan->n_experts > result.maximum_tensor_row)
            result.maximum_tensor_row = (uint32_t)plan->n_experts;
        if (cell_count > UINT32_MAX -
                (attention->shared_kv_projection ? 22u : 23u))
            return -1;
        cell_count += attention->shared_kv_projection ? 22u : 23u;
        if (plan->dense_intermediate == 0) cell_count -= 5u;
        if (checked_mul(candidates, (size_t)kv_width, &layer_tentative) != 0 ||
            checked_mul(layer_tentative, 2u * sizeof(float),
                        &layer_tentative) != 0 ||
            checked_add(tentative_bytes, layer_tentative,
                        &tentative_bytes) != 0)
            return -1;
    }
    if (candidates > UINT32_MAX / result.maximum_topk ||
        checked_mul(candidates, result.maximum_topk, &jobs) != 0)
        return -1;
#define RESERVE_FLOAT(field, rows, cols) do { \
    if (checked_mul((rows), (cols), &count) != 0 || \
        reserve_bytes(&cursor, count, sizeof(float), &result.field) != 0) \
        return -1; \
} while (0)
#define RESERVE_I32(field, elements) do { \
    if (reserve_bytes(&cursor, (elements), sizeof(int32_t), \
                      &result.field) != 0) return -1; \
} while (0)
    RESERVE_I32(candidate_token_ids, candidates);
    RESERVE_I32(target_parent_rows, candidates);
    RESERVE_I32(target_depths, candidates);
    RESERVE_FLOAT(state_a, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(state_b, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(normalized, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(queries, candidates, result.maximum_query_width);
    RESERVE_FLOAT(keys, candidates, result.maximum_kv_width);
    RESERVE_FLOAT(values, candidates, result.maximum_kv_width);
    RESERVE_FLOAT(attention_output, candidates, result.maximum_query_width);
    RESERVE_FLOAT(branch, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(dense_gate, candidates, result.maximum_dense_width);
    RESERVE_FLOAT(dense_up, candidates, result.maximum_dense_width);
    RESERVE_FLOAT(dense_chain, candidates, result.maximum_dense_width);
    RESERVE_FLOAT(dense_output, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(router_input, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(router_logits, candidates, result.maximum_experts);
    RESERVE_I32(selected_experts, jobs);
    RESERVE_FLOAT(selected_weights, 1u, jobs);
    RESERVE_I32(grouped_to_canonical, jobs);
    RESERVE_I32(canonical_to_grouped, jobs);
    RESERVE_FLOAT(routed_input, jobs, (size_t)model->hidden);
    RESERVE_FLOAT(routed_gate, jobs, result.maximum_expert_width);
    RESERVE_FLOAT(routed_up, jobs, result.maximum_expert_width);
    RESERVE_FLOAT(routed_chain, jobs, result.maximum_expert_width);
    RESERVE_FLOAT(routed_output_jobs, jobs, (size_t)model->hidden);
    RESERVE_FLOAT(routed_output, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(combine_a, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(combine_b, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(final_state, candidates, (size_t)model->hidden);
    RESERVE_FLOAT(position_logits, candidates, descriptor->vocabulary);
    RESERVE_FLOAT(attention_scores, 1u, descriptor->maximum_context);
    RESERVE_FLOAT(tensor_row, 1u, result.maximum_tensor_row);
#undef RESERVE_FLOAT
#undef RESERVE_I32
    if (align_size(cursor, TEXT_ALIGNMENT, &cursor) != 0)
        return -1;
    result.tentative_kv = cursor;
    if (checked_add(cursor, tentative_bytes, &cursor) != 0 ||
        align_size(cursor, TEXT_ALIGNMENT, &cursor) != 0)
        return -1;
    result.total_bytes = cursor;
    *layout = result;
    *cell_count_out = cell_count;
    *tentative_kv_bytes_out = tentative_bytes;
    return 0;
}

int salt_text_verify_program_arena_requirement(
        const SaltTextModelExecDesc *descriptor, uint32_t maximum_candidates,
        size_t *bytes_out) {
    SaltTextCanonicalLayout layout;
    uint32_t cells;
    size_t tentative, layers_bytes, cells_bytes, total;
    if (!bytes_out || descriptor_summary(descriptor, maximum_candidates,
            &layout, &cells, &tentative) != 0 ||
        checked_mul(descriptor->layer_count,
                    sizeof(SaltTextCompiledLayer), &layers_bytes) != 0 ||
        checked_mul(cells, sizeof(SaltTextExecutionCell), &cells_bytes) != 0 ||
        checked_add(TEXT_ALIGNMENT - 1u, layers_bytes, &total) != 0 ||
        align_size(total, TEXT_ALIGNMENT, &total) != 0 ||
        checked_add(total, cells_bytes, &total) != 0)
        return -1;
    *bytes_out = total;
    return 0;
}

static int append_cell(SaltTextExecutionCell *cells, uint32_t capacity,
                       uint32_t *count, SaltTextExecutionCellKind kind,
                       SaltTextExecutionUnit unit, uint32_t logical_capacity,
                       uint32_t layer, uint32_t dependency,
                       uint32_t completion, size_t destination,
                       size_t stride, uint32_t flags) {
    SaltTextExecutionCell *cell;
    if (!cells || !count || *count >= capacity || completion <= dependency)
        return -1;
    cell = &cells[(*count)++];
    memset(cell, 0, sizeof *cell);
    cell->kind = kind;
    cell->unit = unit;
    cell->layer = layer;
    cell->dependency_epoch = dependency;
    cell->completion_epoch = completion;
    cell->logical_capacity = logical_capacity;
    cell->destination_offset = destination;
    cell->destination_stride = stride;
    cell->flags = flags;
    return 0;
}

#define APPEND(kind, unit, logical, layer, dep, complete, destination, stride, flags) \
    do { if (append_cell(cells, cell_capacity, &cell_count, (kind), (unit), \
             (logical), (layer), (dep), (complete), (destination), (stride), \
             (flags)) != 0) return -1; } while (0)

int salt_text_verify_program_compile(
        SaltTextVerifyProgram *program,
        const SaltTextModelExecDesc *descriptor, SaltTextKvState *kv_state,
        uint32_t maximum_candidates, void *startup_arena,
        size_t startup_arena_bytes) {
    SaltTextVerifyProgram built;
    SaltTextCanonicalLayout layout;
    SaltTextCompiledLayer *layers;
    SaltTextExecutionCell *cells;
    uintptr_t address, aligned_address;
    size_t need, padding, layers_bytes, metadata_cursor, tentative_cursor;
    size_t tentative_bytes;
    uint32_t cell_capacity, cell_count = 0, epoch = 0;
    if (!program || !kv_state || !startup_arena ||
        salt_text_verify_program_arena_requirement(
            descriptor, maximum_candidates, &need) != 0 ||
        startup_arena_bytes < need ||
        descriptor_summary(descriptor, maximum_candidates, &layout,
            &cell_capacity, &tentative_bytes) != 0)
        return -1;
    address = (uintptr_t)startup_arena;
    aligned_address = (address + (uintptr_t)(TEXT_ALIGNMENT - 1u)) &
        ~(uintptr_t)(TEXT_ALIGNMENT - 1u);
    padding = (size_t)(aligned_address - address);
    if (padding > startup_arena_bytes ||
        checked_mul(descriptor->layer_count,
                    sizeof(SaltTextCompiledLayer), &layers_bytes) != 0 ||
        checked_add(padding, layers_bytes, &metadata_cursor) != 0 ||
        align_size(metadata_cursor, TEXT_ALIGNMENT, &metadata_cursor) != 0)
        return -1;
    layers = (SaltTextCompiledLayer *)
        (void *)((unsigned char *)startup_arena + padding);
    cells = (SaltTextExecutionCell *)
        (void *)((unsigned char *)startup_arena + metadata_cursor);
    memset(layers, 0, layers_bytes);
    memset(cells, 0,
           (size_t)cell_capacity * sizeof(SaltTextExecutionCell));
    APPEND(SALT_TEXT_CELL_EMBEDDING, SALT_TEXT_EXECUTION_ROWS,
           maximum_candidates, TEXT_GLOBAL_LAYER, epoch, epoch + 1u,
           layout.state_a, (size_t)descriptor->model->hidden * sizeof(float), 0);
    epoch++;
    tentative_cursor = layout.tentative_kv;
    for (uint32_t layer_index = 0; layer_index < descriptor->layer_count;
         layer_index++) {
        const SaltTextLayerExecDesc *layer = &descriptor->layers[layer_index];
        const SaltTextLayerPlan *plan = layer->plan;
        SaltTextCompiledLayer *compiled = &layers[layer_index];
        uint32_t base, expert_ready;
        size_t layer_kv_elements, layer_kv_bytes;
        compiled->descriptor = layer;
        (void)width_product(plan->attention.n_heads,
                            plan->attention.head_dim,
                            &compiled->query_width);
        (void)width_product(plan->attention.n_kv_heads,
                            plan->attention.head_dim,
                            &compiled->kv_width);
        compiled->first_cell = cell_count;
        if (checked_mul((size_t)maximum_candidates,
                        compiled->kv_width, &layer_kv_elements) != 0 ||
            checked_mul(layer_kv_elements, sizeof(float),
                        &layer_kv_bytes) != 0 ||
            checked_add(tentative_cursor, layer_kv_bytes,
                        &tentative_cursor) != 0)
            return -1;
        compiled->tentative_key_offset =
            tentative_cursor - layer_kv_bytes;
        if (checked_add(tentative_cursor, layer_kv_bytes,
                        &tentative_cursor) != 0)
            return -1;
        compiled->tentative_value_offset =
            tentative_cursor - layer_kv_bytes;
        base = epoch;
        APPEND(SALT_TEXT_CELL_PRE_ATTENTION_NORM,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base, base + 1u, layout.normalized,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_QUERY_PROJECTION,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 1u, base + 2u, layout.queries,
               (size_t)compiled->query_width * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_KEY_PROJECTION,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 1u, base + 2u, layout.keys,
               (size_t)compiled->kv_width * sizeof(float), 0);
        if (!plan->attention.shared_kv_projection)
            APPEND(SALT_TEXT_CELL_VALUE_PROJECTION,
                   SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
                   base + 1u, base + 2u, layout.values,
                   (size_t)compiled->kv_width * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ATTENTION_TRANSFORM,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 2u, base + 3u, compiled->tentative_key_offset,
               (size_t)compiled->kv_width * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ATTENTION_BODY,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 3u, base + 4u, layout.attention_output,
               (size_t)compiled->query_width * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_OUTPUT_PROJECTION,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 4u, base + 5u, layout.branch,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ATTENTION_COMBINE,
               SALT_TEXT_EXECUTION_ROWS, maximum_candidates, layer_index,
               base + 5u, base + 6u, layout.state_b,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        base += 6u;
        if (plan->dense_intermediate > 0)
        APPEND(SALT_TEXT_CELL_DENSE_NORM, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base, base + 1u,
               layout.normalized,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ROUTER_INPUT, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base, base + 1u,
               layout.router_input,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ROUTED_NORM, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base, base + 1u,
               layout.combine_a,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        if (plan->dense_intermediate > 0) {
        APPEND(SALT_TEXT_CELL_DENSE_GATE, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 1u, base + 2u,
               layout.dense_gate,
               (size_t)plan->dense_intermediate * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_DENSE_UP, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 1u, base + 2u,
               layout.dense_up,
               (size_t)plan->dense_intermediate * sizeof(float), 0);
        }
        APPEND(SALT_TEXT_CELL_ROUTER_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 1u, base + 2u,
               layout.router_logits,
               (size_t)plan->n_experts * sizeof(float), 0);
        if (plan->dense_intermediate > 0)
        APPEND(SALT_TEXT_CELL_DENSE_ACTIVATION, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 2u, base + 3u,
               layout.dense_chain,
               (size_t)plan->dense_intermediate * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_ROUTER_TOPK, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 2u, base + 3u,
               layout.selected_experts,
               (size_t)plan->top_k_experts * sizeof(int32_t), 0);
        if (plan->dense_intermediate > 0)
        APPEND(SALT_TEXT_CELL_DENSE_DOWN, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 3u, base + 4u,
               layout.dense_output,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_EXPERT_GATE, SALT_TEXT_EXECUTION_JOBS,
               maximum_candidates * (uint32_t)plan->top_k_experts,
               layer_index, base + 3u, base + 4u, layout.routed_gate,
               (size_t)plan->expert_intermediate * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_EXPERT_UP, SALT_TEXT_EXECUTION_JOBS,
               maximum_candidates * (uint32_t)plan->top_k_experts,
               layer_index, base + 3u, base + 4u, layout.routed_up,
               (size_t)plan->expert_intermediate * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_EXPERT_ACTIVATION,
               SALT_TEXT_EXECUTION_JOBS,
               maximum_candidates * (uint32_t)plan->top_k_experts,
               layer_index, base + 4u, base + 5u, layout.routed_chain,
               (size_t)plan->expert_intermediate * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_EXPERT_DOWN, SALT_TEXT_EXECUTION_JOBS,
               maximum_candidates * (uint32_t)plan->top_k_experts,
               layer_index, base + 5u, base + 6u,
               layout.routed_output_jobs,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        APPEND(SALT_TEXT_CELL_EXPERT_REDUCTION, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, base + 6u, base + 7u,
               layout.routed_output,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        expert_ready = base + 7u;
        APPEND(SALT_TEXT_CELL_FFN_COMBINE, SALT_TEXT_EXECUTION_ROWS,
               maximum_candidates, layer_index, expert_ready,
               expert_ready + 1u, layout.state_a,
               (size_t)descriptor->model->hidden * sizeof(float), 0);
        epoch = expert_ready + 1u;
        compiled->cell_count = cell_count - compiled->first_cell;
    }
    APPEND(SALT_TEXT_CELL_FINAL_NORM, SALT_TEXT_EXECUTION_ROWS,
           maximum_candidates, TEXT_GLOBAL_LAYER, epoch, epoch + 1u,
           layout.final_state,
           (size_t)descriptor->model->hidden * sizeof(float), 0);
    APPEND(SALT_TEXT_CELL_FINAL_HEAD, SALT_TEXT_EXECUTION_ROWS,
           maximum_candidates, TEXT_GLOBAL_LAYER, epoch + 1u, epoch + 2u,
           layout.position_logits,
           (size_t)descriptor->vocabulary * sizeof(float), 0);
    APPEND(SALT_TEXT_CELL_LOGIT_SOFTCAP, SALT_TEXT_EXECUTION_ROWS,
           maximum_candidates, TEXT_GLOBAL_LAYER, epoch + 2u, epoch + 3u,
           layout.position_logits,
           (size_t)descriptor->vocabulary * sizeof(float),
           SALT_TEXT_EXECUTION_CELL_FINAL_PUBLICATION);
    if (cell_count != cell_capacity ||
        tentative_cursor != layout.tentative_kv + tentative_bytes)
        return -1;
    memset(&built, 0, sizeof built);
    built.descriptor = descriptor;
    built.kv_state = kv_state;
    built.layers = layers;
    built.layer_count = descriptor->layer_count;
    built.hidden = (uint32_t)descriptor->model->hidden;
    built.vocabulary = descriptor->vocabulary;
    built.maximum_candidates = maximum_candidates;
    built.maximum_context = descriptor->maximum_context;
    built.layout = layout;
    built.dispatch.cells = cells;
    built.dispatch.cell_count = cell_count;
    built.dispatch.engine_command_count = 1u;
    built.dispatch.final_dependency_epoch = epoch + 3u;

    built.tentative_kv_bytes = tentative_bytes;
    built.ready = 1;
    *program = built;
    return 0;
}

#undef APPEND

int salt_text_executor_plan_compile(
        SaltTextExecutorPlan *plan, const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextExecutionAssignment *assignments, size_t assignment_count) {
    SaltTextExecutorPlan built;
    if (!plan || !program || !program->ready || !policy || !assignments ||
        assignment_count < program->dispatch.cell_count ||
        (policy->execution_class != SALT_TEXT_EXECUTION_CPU_ONLY &&
         policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY &&
         policy->execution_class != SALT_TEXT_EXECUTION_MIXED))
        return -1;
    memset(assignments, 0,
           (size_t)program->dispatch.cell_count * sizeof *assignments);
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        SaltTextExecutionAssignment *assignment = &assignments[index];
        uint32_t split = 0;
        if (policy->execution_class == SALT_TEXT_EXECUTION_CPU_ONLY) {
            split = cell->logical_capacity;
        } else if (policy->execution_class == SALT_TEXT_EXECUTION_MIXED) {
            split = cell->unit == SALT_TEXT_EXECUTION_ROWS
                ? policy->cpu_rows : policy->cpu_jobs;
            if (split > cell->logical_capacity)
                split = cell->logical_capacity;
        }
        assignment->cpu.count = split;
        assignment->gpu.first = split;
        assignment->gpu.count = cell->logical_capacity - split;
    }
    if (policy->execution_class == SALT_TEXT_EXECUTION_GPU_ONLY) {
        uint32_t first = 0;
        uint32_t barriers = 0;
        for (uint32_t index = 0; index <= program->dispatch.cell_count;
             index++) {
            const int boundary = index == program->dispatch.cell_count ||
                (index > first &&
                 (program->dispatch.cells[index].kind ==
                      SALT_TEXT_CELL_EXPERT_GATE ||
                  program->dispatch.cells[index].kind ==
                      SALT_TEXT_CELL_EXPERT_REDUCTION));
            if (boundary) {
                assignments[first].gpu_traversal_count = index - first;
                assignments[first].gpu_traversal_barriers = barriers;
                first = index;
                barriers = 0;
            }
            if (index < program->dispatch.cell_count &&
                program->dispatch.cells[index].dependency_epoch > 0)
                barriers++;
        }
    }
    memset(&built, 0, sizeof built);
    built.program = program;
    built.assignments = assignments;
    built.assignment_count = program->dispatch.cell_count;
    built.execution_class = policy->execution_class;
    *plan = built;
    return 0;
}

static int executor_plan_valid(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan) {
    if (!program || !plan || plan->program != program || !plan->assignments ||
        plan->assignment_count != program->dispatch.cell_count ||
        (plan->execution_class != SALT_TEXT_EXECUTION_CPU_ONLY &&
         plan->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY &&
         plan->execution_class != SALT_TEXT_EXECUTION_MIXED))
        return 0;
    for (uint32_t index = 0; index < plan->assignment_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        const SaltTextExecutionAssignment *assignment =
            &plan->assignments[index];
        if (assignment->cpu.first != 0 ||
            assignment->gpu.first != assignment->cpu.count ||
            assignment->cpu.count > cell->logical_capacity ||
            assignment->gpu.count !=
                cell->logical_capacity - assignment->cpu.count)
            return 0;
        if ((plan->execution_class == SALT_TEXT_EXECUTION_CPU_ONLY &&
             assignment->cpu.count != cell->logical_capacity) ||
            (plan->execution_class == SALT_TEXT_EXECUTION_GPU_ONLY &&
             assignment->cpu.count != 0))
            return 0;
    }
    if (plan->execution_class == SALT_TEXT_EXECUTION_GPU_ONLY) {
        uint32_t first = 0;
        while (first < plan->assignment_count) {
            const SaltTextExecutionAssignment *assignment =
                &plan->assignments[first];
            uint32_t barriers = 0;
            if (assignment->gpu_traversal_count == 0 ||
                assignment->gpu_traversal_count >
                    plan->assignment_count - first)
                return 0;
            for (uint32_t offset = 0;
                 offset < assignment->gpu_traversal_count; offset++) {
                uint32_t index = first + offset;
                const SaltTextExecutionCell *cell =
                    &program->dispatch.cells[index];
                if (offset > 0 &&
                    (cell->kind == SALT_TEXT_CELL_EXPERT_GATE ||
                     cell->kind == SALT_TEXT_CELL_EXPERT_REDUCTION))
                    return 0;
                if (offset > 0 &&
                    plan->assignments[index].gpu_traversal_count != 0)
                    return 0;
                if (cell->dependency_epoch > 0) barriers++;
            }
            if (assignment->gpu_traversal_barriers != barriers)
                return 0;
            first += assignment->gpu_traversal_count;
        }
    }
    return 1;
}

static int verify_greedy_token(const float *logits, uint32_t vocabulary,
                               int32_t *token_out) {
    float best;
    uint32_t token = 0;
    if (!logits || !token_out || vocabulary == 0 || isnan(logits[0]))
        return -1;
    best = logits[0];
    for (uint32_t i = 1; i < vocabulary; i++) {
        if (isnan(logits[i])) return -1;
        if (logits[i] > best) {
            best = logits[i];
            token = i;
        }
    }
    if (token > (uint32_t)INT32_MAX) return -1;
    *token_out = (int32_t)token;
    return 0;
}

static float *private_kv_row(const SaltTextKvRowsDesc *rows,
                             uint32_t position) {
    uint32_t logical, index;
    if (!rows || position < rows->private_position_base) return NULL;
    logical = position - rows->private_position_base;
    if (rows->private_mode == SALT_TEXT_KV_PRIVATE_ABSOLUTE) {
        if (logical >= rows->private_row_capacity) return NULL;
        index = logical;
    } else {
        index = logical % rows->private_row_capacity;
    }
    return rows->private_rows + (size_t)index * rows->private_row_stride;
}

static const float *committed_kv_row(const SaltTextKvRowsDesc *rows,
                                     uint32_t source_position,
                                     uint32_t position) {
    const float *shared;
    size_t shared_capacity, shared_stride;
    uint32_t shared_rows;
    uint32_t oldest;
    if (!rows || position >= source_position ||
        kv_shared_prefix(rows, &shared, &shared_capacity,
                         &shared_stride, &shared_rows) != 0)
        return NULL;
    (void)shared_capacity;
    if (position < shared_rows)
        return shared + (size_t)position * shared_stride;
    if (rows->private_mode == SALT_TEXT_KV_PRIVATE_RING) {
        oldest = source_position > rows->private_row_capacity
            ? source_position - rows->private_row_capacity : 0u;
        if (position < oldest) return NULL;
    }
    return private_kv_row(rows, position);
}

static int execution_kv_range_valid(const SaltTextVerifyProgram *program,
                                    uint32_t source, uint32_t count) {
    uint32_t end = source + count;
    for (uint32_t layer_index = 0; layer_index < program->layer_count;
         layer_index++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer_index];
        const SaltTextLayerExecDesc *layer = compiled->descriptor;
        const SaltTextKvLayerDesc *kv = layer->kv;
        const float *key_shared, *value_shared;
        size_t key_capacity, value_capacity, key_stride, value_stride;
        uint32_t key_rows, value_rows;
        if (kv_shared_prefix(&kv->keys, &key_shared, &key_capacity,
                &key_stride, &key_rows) != 0 ||
            kv_shared_prefix(&kv->values, &value_shared, &value_capacity,
                &value_stride, &value_rows) != 0 ||
            source < key_rows || source < value_rows || key_rows != value_rows)
            return -1;
        (void)key_shared;
        (void)value_shared;
        (void)key_capacity;
        (void)value_capacity;
        (void)key_stride;
        (void)value_stride;
        for (uint32_t position = source; position < end; position++)
            if (!private_kv_row(&kv->keys, position) ||
                !private_kv_row(&kv->values, position))
                return -1;
        for (uint32_t row = 0; row < count; row++) {
            uint32_t position = source + row;
            uint32_t first = 0;
            if (layer->plan->attention.kind == SALT_ATTN_SLIDING &&
                position + 1u > (uint32_t)layer->plan->attention.window)
                first = position + 1u -
                    (uint32_t)layer->plan->attention.window;
            for (uint32_t prior = first; prior < source; prior++)
                if (!committed_kv_row(&kv->keys, source, prior) ||
                    !committed_kv_row(&kv->values, source, prior))
                    return -1;
        }
    }
    return 0;
}

static uint64_t tentative_suffix_scrub(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionView *view, uint32_t first_row,
        uint32_t candidate_count) {
    uint64_t bytes = 0;
    if (!program || !view || !view->canonical_base ||
        first_row > candidate_count)
        return 0;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        size_t row_bytes = (size_t)compiled->kv_width * sizeof(float);
        size_t count = candidate_count - first_row;
        size_t span = count * row_bytes;
        memset(view->canonical_base + compiled->tentative_key_offset +
                   (size_t)first_row * row_bytes,
               0, span);
        memset(view->canonical_base + compiled->tentative_value_offset +
                   (size_t)first_row * row_bytes,
               0, span);
        bytes += (uint64_t)span * 2u;
    }
    return bytes;
}

static int commit_tentative_prefix(const SaltTextVerifyProgram *program,
                                   const SaltTextExecutionView *view,
                                   uint32_t source, uint32_t accepted,
                                   uint64_t *published_bytes) {
    uint64_t bytes = 0;
    if (!program || !view || !published_bytes || view->reserved != 0u ||
        view->backend_published_kv > 1u)
        return -1;
    /* Device/live-seat publication is authoritative. Host materialization is
     * an explicit consumer operation, not a per-commit mirror or row walk. */
    *published_bytes = 0;
    if (view->backend_published_kv) {
        /* Logical publication is not host-copy traffic. Count the accepted
         * extent from compiled geometry without visiting or copying KV rows. */
        for (uint32_t layer = 0; layer < program->layer_count; layer++) {
            uint64_t row_bytes =
                (uint64_t)program->layers[layer].kv_width * sizeof(float) * 2u;
            if (row_bytes && accepted > (UINT64_MAX - bytes) / row_bytes)
                return -1;
            bytes += row_bytes * accepted;
        }
        *published_bytes = bytes;
        return 0;
    }
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        const SaltTextKvLayerDesc *kv = compiled->descriptor->kv;
        size_t row_bytes = (size_t)compiled->kv_width * sizeof(float);
        const unsigned char *keys = view->canonical_base +
            compiled->tentative_key_offset;
        const unsigned char *values = view->canonical_base +
            compiled->tentative_value_offset;
        for (uint32_t row = 0; row < accepted; row++) {
            if (!view->backend_published_kv) {
                float *key_destination =
                    private_kv_row(&kv->keys, source + row);
                float *value_destination =
                    private_kv_row(&kv->values, source + row);
                if (!key_destination || !value_destination) return -1;
                memcpy(key_destination, keys + (size_t)row * row_bytes,
                       row_bytes);
                memcpy(value_destination, values + (size_t)row * row_bytes,
                       row_bytes);
            }
            bytes += (uint64_t)row_bytes * 2u;
        }
    }
    *published_bytes = bytes;
    return 0;
}

static int frontier_ancestor_at_depth(
        const SaltTextTargetFrontier *frontier, uint32_t node_index,
        uint32_t depth, uint32_t *ancestor_out) {
    if (!frontier || !frontier->nodes || !ancestor_out ||
        node_index >= frontier->node_count)
        return -1;
    while (frontier->nodes[node_index].depth > depth) {
        node_index = frontier->nodes[node_index].parent_index;
        if (node_index == SALT_TEXT_TARGET_NO_PARENT ||
            node_index >= frontier->node_count)
            return -1;
    }
    if (frontier->nodes[node_index].depth != depth) return -1;
    *ancestor_out = node_index;
    return 0;
}

static int frontier_node_on_path(
        const SaltTextTargetFrontier *frontier,
        uint32_t node_index, uint32_t winning_node_index) {
    if (!frontier || !frontier->nodes ||
        node_index >= frontier->node_count ||
        winning_node_index >= frontier->node_count)
        return 0;
    for (;;) {
        if (winning_node_index == node_index) return 1;
        winning_node_index = frontier->nodes[winning_node_index].parent_index;
        if (winning_node_index == SALT_TEXT_TARGET_NO_PARENT) return 0;
    }
}

static int projection_ranked_ids(const float *logits, uint32_t vocabulary,
                                 uint32_t count, int32_t *ids) {
    float values[SALT_TEXT_NFQ_MAX_CHECKS];
    uint32_t token, used = 0u;
    if (!logits || !ids || count == 0u ||
        count > SALT_TEXT_NFQ_MAX_CHECKS ||
        count > vocabulary) return -1;
    for (token = 0u; token < vocabulary; ++token) {
        uint32_t at = used;
        float value = logits[token];
        if (!isfinite(value)) return -1;
        if (at == count) {
            if (value <= values[count - 1u]) continue;
            at--;
        } else {
            used++;
        }
        while (at > 0u && value > values[at - 1u]) {
            values[at] = values[at - 1u];
            ids[at] = ids[at - 1u];
            at--;
        }
        values[at] = value;
        ids[at] = (int32_t)token;
    }
    return used == count ? 0 : -1;
}

int salt_text_projection_refill_candidates(
        const SaltTextVerifyProgram *program,
        const SaltTextProjectionWindow *previous,
        const float *parent_logits, uint32_t sequence_tiles,
        uint32_t route_count, int32_t *candidate_ids, uint32_t capacity,
        uint32_t *reused_rows) {
    int32_t ranked[SALT_TEXT_NFQ_MAX_CHECKS];
    uint32_t count, route, tile, reused = 0u;
    if (reused_rows) *reused_rows = 0u;
    if (!program || !program->ready || !program->kv_state ||
        !parent_logits || !candidate_ids || !reused_rows || sequence_tiles == 0u ||
        route_count == 0u || route_count > program->maximum_candidates ||
        sequence_tiles > program->maximum_candidates / route_count)
        return -1;
    count = sequence_tiles * route_count;
    if (count > capacity || count > program->maximum_candidates)
        return -1;
    if (!previous || !previous->logits) return 1;
    if (previous->generation != program->kv_state->transition_generation ||
         previous->position != program->kv_state->position ||
         previous->sequence_tiles == 0u || previous->route_count == 0u ||
         previous->route_count > program->maximum_candidates ||
         previous->sequence_tiles >
             program->maximum_candidates / previous->route_count ||
         previous->sequence_tiles * previous->route_count >
             program->maximum_candidates ||
         previous->accepted_count == 0u ||
         previous->accepted_count > previous->sequence_tiles ||
         previous->winning_route >= previous->route_count)
        return -1;
    /* Target-block local memory. The previous block computed one exact row per
     * candidate position; rows past the accepted prefix are speculative (their
     * context contains the rejected token) but already paid for. Row
     * accepted_count-1 produced the parent logits and selects the root; old
     * row accepted_count+t-1 supplies the guess for new tile t. A parent
     * distribution supplies alternatives at one position, never a
     * continuation: uncovered future seats are refused, not filled with
     * lower ranks. The caller shortens sequence_tiles to what is covered. */
    if (sequence_tiles < 2u || sequence_tiles - 1u >
            previous->sequence_tiles - previous->accepted_count)
        return 1;
    if (projection_ranked_ids(parent_logits, program->vocabulary,
                              route_count, ranked) != 0)
        return -1;
    for (route = 0u; route < route_count; ++route) {
        candidate_ids[route * sequence_tiles] = ranked[route];
        for (tile = 1u; tile < sequence_tiles; ++tile) {
            int32_t prediction[SALT_TEXT_NFQ_MAX_CHECKS];
            uint32_t destination = route * sequence_tiles + tile;
            uint32_t old_tile = previous->accepted_count + tile - 1u;
            uint32_t old_route = route == 0u ||
                route >= previous->route_count ?
                previous->winning_route : route;
            size_t row = (size_t)old_route * previous->sequence_tiles + old_tile;
            if (projection_ranked_ids(previous->logits +
                    row * program->vocabulary,
                    program->vocabulary, route_count, prediction) != 0)
                return -1;
            candidate_ids[destination] = prediction[route];
            reused++;
        }
    }
    *reused_rows = reused;
    return 0;
}

static int32_t history_token_at(const int32_t *history_ids,
                                uint32_t history_count,
                                const int32_t *output_ids, uint32_t index) {
    return index < history_count ? history_ids[index]
                                 : output_ids[index - history_count];
}

int salt_text_history_refill_candidates(
        const int32_t *history_ids, uint32_t history_count,
        const int32_t *output_ids, uint32_t output_count,
        int32_t root, uint32_t vocabulary, uint32_t sequence_tiles,
        int32_t *candidate_ids, uint32_t capacity, uint32_t *matched_length) {
    uint32_t total, best_end = 0u, best_length = 0u, proposed = 1u;
    if (matched_length) *matched_length = 0u;
    if (!candidate_ids || !matched_length || sequence_tiles == 0u ||
        sequence_tiles > capacity || root < 0 || (uint32_t)root >= vocabulary ||
        (history_count && !history_ids) || (output_count && !output_ids) ||
        history_count > UINT32_MAX - output_count)
        return -1;
    candidate_ids[0] = root;
    total = history_count + output_count;
    if (sequence_tiles < 2u || total < 2u) return 1;
    /* Committed sequence C[0..total) followed by the root at index total.
     * For every earlier occurrence C[e] == root, e < total - 1, count the
     * backward match between C[..e] and C[..total-1]; keep the longest, most
     * recent match. Tokens after e are the parallel proposal. */
    for (uint32_t end = total - 1u; end-- > 0u;) {
        uint32_t length = 1u;
        if (history_token_at(history_ids, history_count, output_ids, end) != root)
            continue;
        while (length < SALT_TEXT_HISTORY_MATCH_MAX && length <= end &&
               history_token_at(history_ids, history_count, output_ids,
                                end - length) ==
               history_token_at(history_ids, history_count, output_ids,
                                total - length))
            length++;
        if (length > best_length) {
            best_length = length;
            best_end = end;
            if (length == SALT_TEXT_HISTORY_MATCH_MAX) break;
        }
    }
    if (best_length == 0u) return 1;
    for (uint32_t follow = best_end + 1u;
         follow < total && proposed < sequence_tiles; follow++) {
        int32_t token = history_token_at(history_ids, history_count,
                                         output_ids, follow);
        if (token < 0 || (uint32_t)token >= vocabulary) break;
        candidate_ids[proposed++] = token;
    }
    *matched_length = best_length;
    return proposed < 2u ? 1 : (int)proposed;
}

static uint64_t tentative_frontier_scrub(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionView *view,
        const SaltTextTargetFrontier *frontier,
        uint32_t winning_node_index) {
    uint64_t bytes = 0;
    if (!program || !view || !view->canonical_base || !frontier)
        return 0;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        const SaltTextCompiledLayer *compiled = &program->layers[layer];
        size_t row_bytes = (size_t)compiled->kv_width * sizeof(float);
        for (uint32_t node_index = 0; node_index < frontier->node_count;
             node_index++) {
            const SaltTextTargetNode *node = &frontier->nodes[node_index];
            if (frontier_node_on_path(
                    frontier, node_index, winning_node_index))
                continue;
            memset(view->canonical_base + compiled->tentative_key_offset +
                       (size_t)node->tentative_state_slot * row_bytes,
                   0, row_bytes);
            memset(view->canonical_base + compiled->tentative_value_offset +
                       (size_t)node->tentative_state_slot * row_bytes,
                   0, row_bytes);
            bytes += (uint64_t)row_bytes * 2u;
        }
    }
    return bytes;
}

static int commit_tentative_frontier_path(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionView *view,
        const SaltTextTargetFrontier *frontier,
        uint32_t source, uint32_t winning_node_index,
        uint64_t *published_bytes) {
    uint64_t bytes = 0;
    uint32_t accepted;
    if (!program || !view || !published_bytes || !frontier ||
        winning_node_index >= frontier->node_count)
        return -1;
    accepted = frontier->nodes[winning_node_index].depth + 1u;
    for (uint32_t depth = 0; depth < accepted; depth++) {
        uint32_t node_index;
        if (frontier_ancestor_at_depth(
                frontier, winning_node_index, depth, &node_index) != 0)
            return -1;
        for (uint32_t layer = 0; layer < program->layer_count; layer++) {
            const SaltTextCompiledLayer *compiled = &program->layers[layer];
            const SaltTextKvLayerDesc *kv = compiled->descriptor->kv;
            size_t row_bytes = (size_t)compiled->kv_width * sizeof(float);
            size_t row = frontier->nodes[node_index].tentative_state_slot;
            const unsigned char *keys = view->canonical_base +
                compiled->tentative_key_offset + row * row_bytes;
            const unsigned char *values = view->canonical_base +
                compiled->tentative_value_offset + row * row_bytes;
            float *key_destination = private_kv_row(&kv->keys, source + depth);
            float *value_destination =
                private_kv_row(&kv->values, source + depth);
            if (!key_destination || !value_destination) return -1;
            memcpy(key_destination, keys, row_bytes);
            memcpy(value_destination, values, row_bytes);
            bytes += (uint64_t)row_bytes * 2u;
        }
    }
    *published_bytes = bytes;
    return 0;
}

static int executor_scrub(SaltTextVerifyExecutor *executor,
                          const SaltTextVerifyProgram *program,
                          uint64_t generation, SaltTextVerifyResult *result,
                          SaltTextVerifyStatus status) {
    uint64_t scrubbed = 0;
    int rc = executor->ops->scrub(
        executor->context, program, generation, &scrubbed);
    result->backend.tentative_scrub_bytes += scrubbed;
    result->status = rc == 0 ? status : SALT_TEXT_VERIFY_FATAL;
    result->result_position = program->kv_state->position;
    result->transition_generation = generation;
    return -1;
}

int salt_text_verify_execute(const SaltTextVerifyProgram *program,
                             SaltTextVerifyExecutor *executor,
                             const SaltTextTargetBlock *block,
                             SaltTextVerifyResult *result) {
    SaltTextExecutionView view;
    uint32_t accepted = 0;
    uint64_t published = 0;
    int32_t pending = -1;
    float *position_logits;
    const float *pending_distribution = NULL;
    if (result) {
        memset(result, 0, sizeof *result);
        result->winning_node_index = UINT32_MAX;
        result->winning_node_id = UINT32_MAX;
    }
    if (!program || !program->ready || !program->descriptor ||
        !program->kv_state || !executor || !executor->ops ||
        !executor->context || !executor->ops->submit ||
        !executor->ops->finish || !executor->ops->resolve ||
        !executor->ops->scrub ||
        executor->program != program ||
        !executor_plan_valid(program, executor->plan) ||
        !block || !result || block->transition_generation == 0 ||
        block->transition_generation <=
            program->kv_state->transition_generation ||
        !block->candidate_token_ids || !block->parent_logits ||
        block->candidate_count == 0 ||
        block->candidate_count > program->maximum_candidates ||
        block->source_position != program->kv_state->position ||
        block->source_position >
            program->maximum_context - block->candidate_count ||
        program->dispatch.engine_command_count != 1u ||
        execution_kv_range_valid(program, block->source_position,
                                 block->candidate_count) != 0) {
        if (result) result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    for (uint32_t row = 0; row < block->candidate_count; row++)
        if (block->candidate_token_ids[row] < 0 ||
            (uint32_t)block->candidate_token_ids[row] >= program->vocabulary) {
            result->status = SALT_TEXT_VERIFY_INVALID;
            return -1;
        }
    if (verify_greedy_token(
            block->parent_logits, program->vocabulary, &pending) != 0) {
        program->kv_state->transition_generation = block->transition_generation;
        result->status = SALT_TEXT_VERIFY_EXECUTOR_FAILURE;
        result->result_position = program->kv_state->position;
        result->transition_generation = block->transition_generation;
        return -1;
    }
    pending_distribution = block->parent_logits;
    if (pending != block->candidate_token_ids[0]) {
        program->kv_state->transition_generation = block->transition_generation;
        result->status = SALT_TEXT_VERIFY_COMMITTED;
        result->accepted_count = 0u;
        result->produced_count = 1u;
        result->committed_count = 0u;
        result->pending_token_id = pending;
        result->result_position = program->kv_state->position;
        result->transition_generation = block->transition_generation;
        result->pending_logits = pending_distribution;
        result->pending_logits_count = program->vocabulary;
        return 0;
    }
    accepted = 1u;
    memset(&view, 0, sizeof view);
    if (executor->ops->submit(
            executor->context, program, block->transition_generation,
            block->source_position, block->candidate_token_ids,
            block->candidate_count) != 0) {
        program->kv_state->transition_generation =
            block->transition_generation;
        return executor_scrub(executor, program,
            block->transition_generation, result,
            SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
    }
    program->kv_state->transition_generation = block->transition_generation;
    if (executor->ops->finish(executor->context, program,
            &view, &result->backend) != 0 ||
        !view.canonical_base ||
        view.canonical_bytes < program->layout.total_bytes ||
        result->backend.engine_submissions != 1u ||
        result->backend.completion_fences != 1u ||
        result->backend.intermediate_host_publications != 0u) {
        return executor_scrub(executor, program,
            block->transition_generation, result,
            SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
    }
    position_logits = (float *)(void *)(
        view.canonical_base + program->layout.position_logits);
    while (accepted < block->candidate_count) {
        const float *target = position_logits +
            (size_t)(accepted - 1u) * program->vocabulary;
        pending_distribution = target;
        if (verify_greedy_token(target, program->vocabulary, &pending) != 0)
            return executor_scrub(executor, program,
                block->transition_generation, result,
                SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
        if (pending != block->candidate_token_ids[accepted]) break;
        accepted++;
    }
    if (accepted == block->candidate_count) {
        const float *target = position_logits +
            (size_t)(block->candidate_count - 1u) * program->vocabulary;
        pending_distribution = target;
        if (verify_greedy_token(target, program->vocabulary, &pending) != 0)
            return executor_scrub(executor, program,
                block->transition_generation, result,
                SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
    }
    {
        uint64_t backend_scrubbed = 0;
        if (executor->ops->resolve(
                executor->context, program, block->transition_generation,
                block->source_position, accepted, block->candidate_count,
                &backend_scrubbed) != 0)
            return executor_scrub(executor, program,
                block->transition_generation, result,
                SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
        result->backend.tentative_scrub_bytes += backend_scrubbed;
    }
    result->backend.tentative_scrub_bytes += tentative_suffix_scrub(
        program, &view, accepted, block->candidate_count);
    if (commit_tentative_prefix(program, &view, block->source_position,
            accepted, &published) != 0)
        return executor_scrub(executor, program,
            block->transition_generation, result, SALT_TEXT_VERIFY_FATAL);
    program->kv_state->position = block->source_position + accepted;
    result->backend.final_kv_publish_bytes = published;
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = accepted;
    result->produced_count = accepted + 1u;
    result->committed_count = accepted;
    result->pending_token_id = pending;
    result->result_position = program->kv_state->position;
    result->transition_generation = block->transition_generation;
    result->pending_logits = pending_distribution;
    result->projection_logits = (const float *)(view.canonical_base +
        program->layout.position_logits);
    result->projection_rows = block->candidate_count;
    result->pending_logits_count = program->vocabulary;
    return 0;
}

int salt_text_verify_frontier_execute(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        const SaltTextTargetFrontierBlock *block,
        SaltTextVerifyResult *result) {
    SaltTextExecutionView view;
    const SaltTextTargetFrontier *frontier;
    float *position_logits;
    const float *pending_distribution;
    uint32_t root = UINT32_MAX, current = UINT32_MAX;
    uint32_t accepted = 0, maximum_depth = 0;
    uint64_t published = 0;
    int32_t pending = -1;
    int root_only = 1;
    if (result) {
        memset(result, 0, sizeof *result);
        result->winning_node_index = UINT32_MAX;
        result->winning_node_id = UINT32_MAX;
    }
    if (!program || !program->ready || !program->descriptor ||
        !program->kv_state || !executor || !executor->ops ||
        !executor->context || !executor->ops->submit_frontier ||
        !executor->ops->finish || !executor->ops->resolve ||
        !executor->ops->scrub || executor->program != program ||
        !executor_plan_valid(program, executor->plan) ||
        !block || !result || !block->parent_logits ||
        block->force_full_frontier > 1u ||
        !(frontier = &block->frontier)->nodes ||
        frontier->generation == 0 ||
        frontier->generation <= program->kv_state->transition_generation ||
        frontier->node_count == 0 ||
        frontier->node_count > program->maximum_candidates ||
        block->source_position != program->kv_state->position ||
        salt_text_target_frontier_validate(
            frontier, frontier->node_count) != 0 ||
        (block->area_frontier &&
         (!block->area_frontier->tasks ||
          block->area_frontier->generation != frontier->generation))) {
        if (result) result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    for (uint32_t index = 0; index < frontier->node_count; index++) {
        const SaltTextTargetNode *node = &frontier->nodes[index];
        if ((uint32_t)node->token_id >= program->vocabulary) {
            result->status = SALT_TEXT_VERIFY_INVALID;
            return -1;
        }
        if (node->parent_index != SALT_TEXT_TARGET_NO_PARENT || node->depth != 0u)
            root_only = 0;
        if (node->depth > maximum_depth) maximum_depth = node->depth;
    }
    if (maximum_depth == UINT32_MAX ||
        maximum_depth + 1u > program->maximum_context ||
        block->source_position >
            program->maximum_context - (maximum_depth + 1u) ||
        program->dispatch.engine_command_count != 1u ||
        execution_kv_range_valid(program, block->source_position,
                                 maximum_depth + 1u) != 0) {
        result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    if (verify_greedy_token(
            block->parent_logits, program->vocabulary, &pending) != 0) {
        program->kv_state->transition_generation = frontier->generation;
        result->status = SALT_TEXT_VERIFY_EXECUTOR_FAILURE;
        result->result_position = program->kv_state->position;
        result->transition_generation = frontier->generation;
        return -1;
    }
    pending_distribution = block->parent_logits;
    for (uint32_t index = 0; index < frontier->node_count; index++)
        if (frontier->nodes[index].parent_index ==
                SALT_TEXT_TARGET_NO_PARENT &&
            frontier->nodes[index].token_id == pending) {
            root = index;
            break;
        }
    if (root == UINT32_MAX) {
        if (block->area_frontier && salt_area_frontier_cancel_except(
                block->area_frontier, frontier->generation,
                UINT32_MAX) != 0) {
            result->status = SALT_TEXT_VERIFY_FATAL;
            return -1;
        }
        program->kv_state->transition_generation = frontier->generation;
        result->status = SALT_TEXT_VERIFY_COMMITTED;
        result->accepted_count = 0u;
        result->produced_count = 1u;
        result->committed_count = 0u;
        result->pending_token_id = pending;
        result->result_position = program->kv_state->position;
        result->transition_generation = frontier->generation;
        result->pending_logits = pending_distribution;
        result->pending_logits_count = program->vocabulary;
        return 0;
    }
    if (root_only && !block->force_full_frontier) {
        SaltTextTargetBlock selected;
        SaltTextVerifyResult committed;
        if (!executor->ops->submit ||
            (block->area_frontier && salt_area_frontier_cancel_except(
                block->area_frontier, frontier->generation,
                frontier->nodes[root].node_id) != 0)) {
            result->status = SALT_TEXT_VERIFY_FATAL;
            return -1;
        }
        memset(&selected, 0, sizeof selected);
        selected.transition_generation = frontier->generation;
        selected.source_position = block->source_position;
        selected.candidate_token_ids = &frontier->nodes[root].token_id;
        selected.candidate_count = 1u;
        selected.parent_logits = block->parent_logits;
        memset(&committed, 0, sizeof committed);
        if (salt_text_verify_execute(
                program, executor, &selected, &committed) != 0) {
            *result = committed;
            return -1;
        }
        if (committed.status != SALT_TEXT_VERIFY_COMMITTED ||
            committed.accepted_count != 1u ||
            committed.committed_count != 1u ||
            committed.produced_count != 2u) {
            committed.status = SALT_TEXT_VERIFY_FATAL;
            *result = committed;
            return -1;
        }
        committed.winning_node_index = root;
        committed.winning_node_id = frontier->nodes[root].node_id;
        *result = committed;
        return 0;
    }
    memset(&view, 0, sizeof view);
    if (executor->ops->submit_frontier(
            executor->context, program, block->source_position,
            frontier) != 0) {
        program->kv_state->transition_generation = frontier->generation;
        return executor_scrub(executor, program, frontier->generation,
            result, SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
    }
    program->kv_state->transition_generation = frontier->generation;
    if (executor->ops->finish(executor->context, program,
            &view, &result->backend) != 0 ||
        !view.canonical_base ||
        view.canonical_bytes < program->layout.total_bytes ||
        result->backend.engine_submissions != 1u ||
        result->backend.completion_fences != 1u ||
        result->backend.intermediate_host_publications != 0u)
        return executor_scrub(executor, program, frontier->generation,
            result, SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
    position_logits = (float *)(void *)(
        view.canonical_base + program->layout.position_logits);
    current = root;
    accepted = 1u;
    for (;;) {
        uint32_t child = UINT32_MAX;
        const SaltTextTargetNode *node = &frontier->nodes[current];
        const float *target = position_logits +
            (size_t)node->tentative_state_slot * program->vocabulary;
        pending_distribution = target;
        if (verify_greedy_token(target, program->vocabulary, &pending) != 0)
            return executor_scrub(executor, program, frontier->generation,
                result, SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
        for (uint32_t index = current + 1u;
             index < frontier->node_count; index++)
            if (frontier->nodes[index].parent_index == current &&
                frontier->nodes[index].token_id == pending) {
                child = index;
                break;
            }
        if (child == UINT32_MAX) break;
        current = child;
        accepted++;
    }
    if (accepted != frontier->nodes[current].depth + 1u)
        return executor_scrub(executor, program, frontier->generation,
            result, SALT_TEXT_VERIFY_FATAL);
    {
        uint64_t backend_scrubbed = 0;
        if (executor->ops->resolve(executor->context, program,
                frontier->generation, block->source_position,
                accepted, frontier->node_count, &backend_scrubbed) != 0)
            return executor_scrub(executor, program, frontier->generation,
                result, SALT_TEXT_VERIFY_EXECUTOR_FAILURE);
        result->backend.tentative_scrub_bytes += backend_scrubbed;
    }
    if (block->area_frontier && salt_area_frontier_cancel_except(
            block->area_frontier, frontier->generation,
            frontier->nodes[current].node_id) != 0)
        return executor_scrub(executor, program, frontier->generation,
            result, SALT_TEXT_VERIFY_FATAL);
    result->backend.tentative_scrub_bytes += tentative_frontier_scrub(
        program, &view, frontier, current);
    if (commit_tentative_frontier_path(program, &view, frontier,
            block->source_position, current, &published) != 0)
        return executor_scrub(executor, program, frontier->generation,
            result, SALT_TEXT_VERIFY_FATAL);
    program->kv_state->position = block->source_position + accepted;
    result->backend.final_kv_publish_bytes = published;
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = accepted;
    result->produced_count = accepted + 1u;
    result->committed_count = accepted;
    result->pending_token_id = pending;
    result->result_position = program->kv_state->position;
    result->transition_generation = frontier->generation;
    result->winning_node_index = current;
    result->winning_node_id = frontier->nodes[current].node_id;
    result->pending_logits = pending_distribution;
    result->projection_logits = (const float *)(view.canonical_base +
        program->layout.position_logits);
    result->projection_rows = frontier->node_count;
    result->pending_logits_count = program->vocabulary;
    return 0;
}

static int add_backend_stats(SaltTextVerifyBackendStats *total,
                             const SaltTextVerifyBackendStats *value) {
#define ADD_U32(field) \
    do { \
        if (UINT32_MAX - total->field < value->field) return -1; \
        total->field += value->field; \
    } while (0)
#define ADD_U64(field) \
    do { \
        if (UINT64_MAX - total->field < value->field) return -1; \
        total->field += value->field; \
    } while (0)
    if (!total || !value) return -1;
    ADD_U32(engine_submissions);
    ADD_U32(completion_fences);
    ADD_U32(intermediate_host_publications);
    ADD_U32(internal_dependency_barriers);
    ADD_U32(projection_dispatches);
    ADD_U32(expert_gate_up_dispatches);
    ADD_U32(expert_down_dispatches);
    ADD_U32(area_m1_dispatches);
    ADD_U32(area_mk_dispatches);
    ADD_U32(area_mn_dispatches);
    if (value->area_peak_active_workers > total->area_peak_active_workers)
        total->area_peak_active_workers = value->area_peak_active_workers;
    ADD_U64(area_output_row_tiles);
    ADD_U64(area_candidate_output_tiles);
    ADD_U32(area_matrix_parallel_dispatches);
    ADD_U64(initial_transfer_bytes);
    ADD_U64(final_logits_transfer_bytes);
    ADD_U64(final_kv_publish_bytes);
    ADD_U64(tentative_scrub_bytes);
    ADD_U64(canonical_clear_bytes);
    ADD_U32(production_frontier_sessions);
    ADD_U32(production_frontier_tasks_queued);
    ADD_U32(production_frontier_tasks_executed);
    ADD_U32(production_frontier_helper_executions);
    ADD_U32(production_frontier_worker_reassignments);
    ADD_U32(production_frontier_queued_cancellations);
    ADD_U32(production_frontier_cancel_requested_completions);
    ADD_U32(production_frontier_stale_rejections);
    if (value->production_frontier_peak_ready_depth >
            total->production_frontier_peak_ready_depth)
        total->production_frontier_peak_ready_depth =
            value->production_frontier_peak_ready_depth;
    ADD_U32(cpu_matrix_pool_phases);
    ADD_U32(cpu_qkv_waves);
    ADD_U32(cpu_dense_gate_up_waves);
    ADD_U32(cpu_expert_gate_up_waves);
    ADD_U32(cpu_expert_down_waves);
    ADD_U32(backend_template_count);
    ADD_U32(backend_template_reuses);
    ADD_U32(backend_dynamic_patches);
    ADD_U32(backend_selected_job_capacity);
    ADD_U32(backend_selected_jobs);
    ADD_U32(backend_graph_count);
    ADD_U32(backend_graph_launches);
    ADD_U32(backend_graph_parameter_patches);
    ADD_U32(backend_physical_kernel_nodes);
    ADD_U32(backend_host_kernel_launch_calls);
#undef ADD_U32
#undef ADD_U64
    return 0;
}

int salt_text_verify_cascade_execute(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        const SaltTextTargetCascadeBlock *block,
        SaltTextVerifyResult *result) {
    SaltTextVerifyBackendStats backend;
    SaltTextVerifyResult aggregate;
    const float *parent;
    if (result) {
        memset(result, 0, sizeof *result);
        result->winning_node_index = UINT32_MAX;
        result->winning_node_id = UINT32_MAX;
    }
    if (!program || !program->ready || !program->kv_state || !executor ||
        !block || !result || block->first_transition_generation == 0 ||
        block->first_transition_generation <=
            program->kv_state->transition_generation ||
        !block->seed_token_ids || block->tile_count == 0 ||
        block->frontier_width == 0 ||
        block->frontier_width > program->maximum_candidates ||
        !block->parent_logits || !block->node_scratch ||
        block->node_scratch_capacity < block->frontier_width ||
        block->source_position != program->kv_state->position ||
        block->tile_count > program->maximum_context - block->source_position ||
        block->tile_count - 1u >
            UINT64_MAX - block->first_transition_generation) {
        if (result) result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    for (uint32_t tile = 0; tile < block->tile_count; tile++)
        for (uint32_t branch = 0; branch < block->frontier_width; branch++) {
            int32_t token = block->seed_token_ids[
                (size_t)tile * block->frontier_width + branch];
            if (token < 0 || (uint32_t)token >= program->vocabulary) {
                result->status = SALT_TEXT_VERIFY_INVALID;
                return -1;
            }
            for (uint32_t prior = 0; prior < branch; prior++)
                if (block->seed_token_ids[
                        (size_t)tile * block->frontier_width + prior] == token) {
                    result->status = SALT_TEXT_VERIFY_INVALID;
                    return -1;
                }
        }
    memset(&backend, 0, sizeof backend);
    memset(&aggregate, 0, sizeof aggregate);
    aggregate.winning_node_index = UINT32_MAX;
    aggregate.winning_node_id = UINT32_MAX;
    parent = block->parent_logits;
    for (uint32_t tile = 0; tile < block->tile_count; tile++) {
        SaltTextTargetFrontierBlock frontier_block;
        SaltTextVerifyResult step;
        for (uint32_t branch = 0;
             branch < block->frontier_width; branch++)
            block->node_scratch[branch] = (SaltTextTargetNode) {
                block->seed_token_ids[
                    (size_t)tile * block->frontier_width + branch],
                branch, SALT_TEXT_TARGET_NO_PARENT, 0u, branch, 0u,
            };
        memset(&frontier_block, 0, sizeof frontier_block);
        frontier_block.source_position = block->source_position + tile;
        frontier_block.frontier.nodes = block->node_scratch;
        frontier_block.frontier.node_count = block->frontier_width;
        frontier_block.frontier.generation =
            block->first_transition_generation + tile;
        frontier_block.parent_logits = parent;
        memset(&step, 0, sizeof step);
        if (salt_text_verify_frontier_execute(
                program, executor, &frontier_block, &step) != 0 ||
            step.status != SALT_TEXT_VERIFY_COMMITTED ||
            add_backend_stats(&backend, &step.backend) != 0) {
            step.backend = backend;
            *result = step;
            return -1;
        }
        aggregate = step;
        aggregate.backend = backend;
        if (step.accepted_count == 0u) {
            aggregate.accepted_count = tile;
            aggregate.committed_count = tile;
            aggregate.produced_count = tile + 1u;
            *result = aggregate;
            return 0;
        }
        if (step.accepted_count != 1u || step.committed_count != 1u ||
            step.produced_count != 2u || !step.pending_logits ||
            step.pending_logits_count != program->vocabulary ||
            step.result_position != block->source_position + tile + 1u) {
            aggregate.status = SALT_TEXT_VERIFY_FATAL;
            *result = aggregate;
            return -1;
        }
        aggregate.accepted_count = tile + 1u;
        aggregate.committed_count = tile + 1u;
        aggregate.produced_count = tile + 2u;
        parent = step.pending_logits;
    }
    *result = aggregate;
    return 0;
}

typedef struct SaltTextRouteWfqContext {
    const SaltTextTargetRouteBlock *block;
    int32_t target_token;
} SaltTextRouteWfqContext;

static int text_route_wfq_item_execute(
        void *opaque, const SaltAreaWfqItem *item,
        SaltAreaWfqItemResult *result) {
    SaltTextRouteWfqContext *context = (SaltTextRouteWfqContext *)opaque;
    uint32_t route;
    if (!context || !context->block || !item || !item->task || !result ||
        item->queue_index >= 2u ||
        (route = item->task->job_index) >= context->block->route_count ||
        item->task->output_first != item->queue_index ||
        item->task->candidate_count != 1u)
        return -1;
    memset(result, 0, sizeof *result);
    if (context->block->route_token_ids[
            (size_t)route * context->block->sequence_tiles] ==
            context->target_token) {
        if (item->queue_index == 0u) {
            result->outcome = SALT_AREA_WFQ_CONTINUE;
        } else {
            result->outcome = SALT_AREA_WFQ_TARGET_WIN;
            result->target_token = context->target_token;
            result->target_state_slot = route;
        }
    } else {
        result->outcome = SALT_AREA_WFQ_BRANCH_FAIL;
    }
    return 0;
}

int salt_text_verify_route_execute(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        const SaltTextTargetRouteBlock *block,
        SaltTextVerifyResult *result) {
    SaltTextTargetBlock selected;
    SaltTextVerifyResult committed;
    SaltAreaWfqResult wfq_result;
    SaltTextRouteWfqContext wfq_context;
    uint32_t winning_route = UINT32_MAX;
    int32_t target_token = -1;
    if (result) {
        memset(result, 0, sizeof *result);
        result->winning_node_index = UINT32_MAX;
        result->winning_node_id = UINT32_MAX;
    }
    if (!program || !program->ready || !program->kv_state || !executor ||
        !block || !result || block->transition_generation == 0 ||
        block->transition_generation <=
            program->kv_state->transition_generation ||
        !block->route_token_ids || block->sequence_tiles == 0 ||
        block->sequence_tiles > program->maximum_candidates ||
        block->route_count == 0 ||
        block->route_count > program->maximum_candidates ||
        !block->parent_logits ||
        ((block->wfq == NULL) != (block->wfq_execute == NULL)) ||
        ((block->wfq == NULL) != (block->wfq_context == NULL)) ||
        block->source_position != program->kv_state->position ||
        block->sequence_tiles >
            program->maximum_context - block->source_position) {
        if (result) result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    for (uint32_t route = 0; route < block->route_count; route++)
        for (uint32_t tile = 0; tile < block->sequence_tiles; tile++) {
            int32_t token = block->route_token_ids[
                (size_t)route * block->sequence_tiles + tile];
            if (token < 0 || (uint32_t)token >= program->vocabulary) {
                result->status = SALT_TEXT_VERIFY_INVALID;
                return -1;
            }
            if (tile == 0)
                for (uint32_t prior = 0; prior < route; prior++)
                    if (block->route_token_ids[
                            (size_t)prior * block->sequence_tiles] == token) {
                        result->status = SALT_TEXT_VERIFY_INVALID;
                        return -1;
                    }
        }
    if (verify_greedy_token(
            block->parent_logits, program->vocabulary,
            &target_token) != 0) {
        result->status = SALT_TEXT_VERIFY_EXECUTOR_FAILURE;
        return -1;
    }
    for (uint32_t route = 0; route < block->route_count; route++)
        if (block->route_token_ids[
                (size_t)route * block->sequence_tiles] == target_token) {
            winning_route = route;
            break;
        }
    memset(&wfq_result, 0, sizeof wfq_result);
    if (block->wfq) {
        wfq_context = (SaltTextRouteWfqContext) { block, target_token };
        if (block->wfq->generation != block->transition_generation ||
            block->wfq->plan.frontier_width != block->route_count ||
            block->wfq->plan.queue_length != 2u ||
            block->wfq_execute(block->wfq_context, block->wfq,
                text_route_wfq_item_execute, &wfq_context, &wfq_result) != 0 ||
            (winning_route == UINT32_MAX
                ? wfq_result.status != SALT_AREA_WFQ_EXHAUSTED
                : (wfq_result.status != SALT_AREA_WFQ_WIN ||
                   wfq_result.ledger.winning_branch !=
                       block->route_count - 1u - winning_route ||
                   wfq_result.ledger.winning_queue != 1u ||
                   wfq_result.ledger.target_token != target_token ||
                   wfq_result.ledger.target_state_slot != winning_route))) {
            result->status = SALT_TEXT_VERIFY_FATAL;
            return -1;
        }
    }
    memset(&selected, 0, sizeof selected);
    selected.transition_generation = block->transition_generation;
    selected.source_position = block->source_position;
    selected.candidate_token_ids = block->route_token_ids +
        (size_t)(winning_route == UINT32_MAX ? 0u : winning_route) *
            block->sequence_tiles;
    selected.candidate_count = block->sequence_tiles;
    selected.parent_logits = block->parent_logits;
    memset(&committed, 0, sizeof committed);
    if (salt_text_verify_execute(
            program, executor, &selected, &committed) != 0) {
        *result = committed;
        return -1;
    }
    if (block->wfq) {
        committed.backend.production_frontier_sessions = 1u;
        committed.backend.production_frontier_tasks_queued =
            block->route_count * block->wfq->plan.queue_length;
        committed.backend.production_frontier_tasks_executed =
            wfq_result.executed_items;
        committed.backend.production_frontier_helper_executions =
            wfq_result.executed_items;
        committed.backend.production_frontier_peak_ready_depth =
            block->route_count;
        for (uint32_t index = 0;
             index < block->route_count * block->wfq->plan.queue_length;
             index++) {
            const SaltAreaTask *task = block->wfq->items[index].task;
            if (task && task->state == SALT_AREA_TASK_CANCELLED)
                committed.backend.production_frontier_queued_cancellations++;
            if (task &&
                (task->flags & SALT_AREA_TASK_CANCEL_REQUESTED) != 0u)
                committed.backend
                    .production_frontier_cancel_requested_completions++;
            if (task && task->state == SALT_AREA_TASK_DONE &&
                block->wfq->items[index].owner_worker !=
                    block->wfq->items[index].branch_index)
                committed.backend.production_frontier_worker_reassignments++;
        }
    }
    if (winning_route == UINT32_MAX) {
        if (committed.accepted_count != 0u ||
            committed.committed_count != 0u ||
            committed.pending_token_id != target_token ||
            committed.backend.engine_submissions != 0u) {
            committed.status = SALT_TEXT_VERIFY_FATAL;
            *result = committed;
            return -1;
        }
    } else {
        uint32_t accepted_depth;
        if (committed.accepted_count == 0u ||
            committed.accepted_count > block->sequence_tiles)
            return -1;
        accepted_depth = committed.accepted_count - 1u;
        committed.winning_node_index =
            winning_route * block->sequence_tiles + accepted_depth;
        committed.winning_node_id = committed.winning_node_index;
    }
    *result = committed;
    return 0;
}

int salt_text_verify_nfq_execute(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        const SaltTextTargetNfqBlock *block,
        SaltTextVerifyResult *result) {
    const SaltTextTargetRouteBlock *route;
    uint64_t width64, checks64, generation, parent_generation;
    uint32_t checks, executed = 0u, retired = 0u;
    uint32_t winner_item = UINT32_MAX;
    int32_t target_token = -1;
    if (result) memset(result, 0, sizeof *result);
    if (!program || !program->ready || !program->kv_state || !executor ||
        !block || !result || !block->matrix ||
        !block->route_ids || !block->ready_items)
        return -1;
    route = &block->route;
    width64 = (uint64_t)route->sequence_tiles * route->route_count;
    checks64 = width64 * block->plan.queue_length;
    generation = route->transition_generation;
    parent_generation = program->kv_state->transition_generation;
    if (!route->route_token_ids || !route->parent_logits ||
        route->wfq || route->wfq_execute || route->wfq_context ||
        generation == 0 ||
        generation <= parent_generation ||
        route->source_position != program->kv_state->position ||
        route->source_position >= program->maximum_context ||
        route->sequence_tiles == 0 || route->route_count == 0 ||
        width64 == 0 || width64 > program->maximum_candidates ||
        width64 > UINT32_MAX || checks64 == 0 || checks64 > UINT32_MAX ||
        route->route_count > block->route_capacity ||
        block->plan.sequence_tiles != route->sequence_tiles ||
        block->plan.route_count != route->route_count ||
        block->plan.workers == 0 || block->plan.workers > 32u ||
        block->ready_capacity < block->plan.workers ||
        block->plan.queue_length == 0 ||
        block->plan.queue_length > SALT_TEXT_NFQ_MAX_QUEUE ||
        width64 < block->plan.workers ||
        width64 % block->plan.workers != 0u ||
        block->matrix->ready || !block->matrix->routes ||
        !block->matrix->items ||
        route->route_count > block->matrix->route_capacity ||
        checks64 > block->matrix->item_capacity ||
        (program->target_policy &&
         program->target_policy->candidate_count != width64))
        return -1;
    checks = (uint32_t)checks64;
    for (uint32_t index = 0; index < checks; index++)
        if (route->route_token_ids[index] < 0 ||
            (uint32_t)route->route_token_ids[index] >= program->vocabulary)
            return -1;
    if (verify_greedy_token(
            route->parent_logits, program->vocabulary, &target_token) != 0)
        return -1;
    for (uint32_t branch = 0; branch < route->route_count; branch++)
        block->route_ids[branch] = branch;
    if (salt_area_nfq_start(block->matrix, &block->plan,
            generation, parent_generation, route->source_position,
            block->route_ids, route->route_count) != 0)
        return -1;
    while (executed < checks && winner_item == UINT32_MAX) {
        uint32_t ready_count = 0u;
        if (salt_area_nfq_take_ready(block->matrix,
                generation, parent_generation, block->ready_items,
                block->ready_capacity, &ready_count) != 0)
            return -1;
        if (ready_count == 0u) break;
        for (uint32_t ready = 0u; ready < ready_count; ready++) {
            uint32_t item_index = block->ready_items[ready];
            SaltAreaNfqItem *item;
            uint32_t retired_now = 0u;
            if (item_index >= checks || item_index >= block->matrix->item_count)
                return -1;
            item = &block->matrix->items[item_index];
            if (route->route_token_ids[item_index] == target_token) {
                uint32_t cancelled = 0u;
                if (salt_area_nfq_complete(block->matrix,
                        generation, parent_generation, item_index,
                        SALT_AREA_WFQ_TARGET_WIN, &retired_now) != 0 ||
                    salt_area_nfq_resolve(block->matrix,
                        generation, parent_generation, item->route_slot,
                        item->sequence_index, item->queue_index,
                        &cancelled) != 0 ||
                    UINT32_MAX - retired < retired_now ||
                    UINT32_MAX - retired - retired_now < cancelled)
                    return -1;
                retired += retired_now + cancelled;
                executed++;
                winner_item = item_index;
                break;
            }
            {
                SaltAreaWfqOutcome outcome =
                    item->queue_index + 1u < block->plan.queue_length
                        ? SALT_AREA_WFQ_CONTINUE : SALT_AREA_WFQ_STALE;
            if (salt_area_nfq_complete(block->matrix,
                    generation, parent_generation, item_index,
                    outcome, &retired_now) != 0 ||
                UINT32_MAX - retired < retired_now)
                return -1;
            retired += retired_now;
            executed++;
            }
        }
    }
    if (winner_item == UINT32_MAX) {
        uint32_t cancelled = 0u;
        if (executed != checks || salt_area_nfq_cancel(block->matrix,
                generation, parent_generation, &cancelled) != 0 ||
            UINT32_MAX - retired < cancelled)
            return -1;
        retired += cancelled;
        program->kv_state->transition_generation = generation;
        memset(result, 0, sizeof *result);
        result->status = SALT_TEXT_VERIFY_COMMITTED;
        result->accepted_count = 0u;
        result->produced_count = 1u;
        result->committed_count = 0u;
        result->pending_token_id = target_token;
        result->result_position = program->kv_state->position;
        result->transition_generation = generation;
        result->winning_node_index = UINT32_MAX;
        result->winning_node_id = UINT32_MAX;
        result->pending_logits = route->parent_logits;
        result->pending_logits_count = program->vocabulary;
    } else {
        memset(result, 0, sizeof *result);
        result->status = SALT_TEXT_VERIFY_COMMITTED;
        result->accepted_count = 0u;
        result->produced_count = 1u;
        result->committed_count = 0u;
        result->pending_token_id = route->route_token_ids[winner_item];
        result->result_position = program->kv_state->position;
        result->transition_generation = generation;
        result->winning_node_index = winner_item;
        result->winning_node_id = winner_item;
        result->pending_logits = route->parent_logits;
        result->pending_logits_count = program->vocabulary;
    }
    result->backend.production_frontier_sessions = 1u;
    result->backend.production_frontier_tasks_queued = checks;
    result->backend.production_frontier_tasks_executed = executed;
    result->backend.production_frontier_helper_executions = executed;
    result->backend.production_frontier_queued_cancellations = retired;
    result->backend.production_frontier_peak_ready_depth = block->plan.workers;
    return 0;
}

int salt_text_token_epoch_init(SaltTextTokenEpochController *controller) {
    if (!controller) return -1;
    memset(controller, 0, sizeof *controller);
    controller->phase = SALT_TEXT_TOKEN_EPOCH_BOUNDARY;
    controller->ready = 1;
    return 0;
}

static int token_epoch_result_valid(
        const SaltTextVerifyResult *target,
        uint64_t epoch_generation, uint32_t parent_position,
        int exact_hit) {
    uint64_t position;
    if (!target || target->status != SALT_TEXT_VERIFY_COMMITTED ||
        target->transition_generation != epoch_generation ||
        target->accepted_count != target->committed_count ||
        target->committed_count == UINT32_MAX ||
        target->produced_count != target->committed_count + 1u ||
        (exact_hit && target->committed_count == 0u) ||
        (exact_hit &&
         (target->backend.engine_submissions != 0u ||
          target->backend.completion_fences != 0u)))
        return -1;
    position = (uint64_t)parent_position + target->committed_count;
    if (position > UINT32_MAX || target->result_position != (uint32_t)position)
        return -1;
    return 0;
}

static int token_epoch_fail(
        SaltTextTokenEpochController *controller,
        uint64_t epoch_generation) {
    if (!controller) return -1;
    controller->active = 0;
    controller->ready = 0;
    controller->phase = SALT_TEXT_TOKEN_EPOCH_FAILED;
    controller->last_epoch_generation = epoch_generation;
    return -1;
}

int salt_text_token_epoch_execute(
        SaltTextTokenEpochController *controller,
        const SaltTextTokenEpochRequest *request,
        SaltTextTokenEpochResult *result) {
    SaltTextVerifyResult target;
    int lookup;
    if (result) memset(result, 0, sizeof *result);
    if (!controller || !request || !result || !controller->ready ||
        !request->route || !request->exact_lookup_commit ||
        !request->cold_search || request->epoch_generation == 0 ||
        request->epoch_generation <= request->parent_generation ||
        request->epoch_generation <= controller->last_epoch_generation ||
        request->route->transition_generation != request->epoch_generation ||
        request->route->source_position != request->parent_position)
        return -1;
    if (controller->active) {
        if (controller->duplicate_admission_count != UINT64_MAX)
            controller->duplicate_admission_count++;
        return 1;
    }
    if (controller->lookup_count == UINT64_MAX) return -1;
    controller->active = 1;
    controller->phase = SALT_TEXT_TOKEN_EPOCH_CACHE_SCAN;
    controller->active_epoch_generation = request->epoch_generation;
    controller->parent_generation = request->parent_generation;
    controller->parent_position = request->parent_position;
    controller->lookup_count++;
    memset(&target, 0, sizeof target);
    lookup = request->exact_lookup_commit(
        request->exact_context, request->epoch_generation,
        request->parent_generation, request->parent_position, &target);
    if (lookup < 0 || lookup > 1)
        return token_epoch_fail(controller, request->epoch_generation);
    if (lookup == 1) {
        if (controller->exact_hit_count == UINT64_MAX ||
            token_epoch_result_valid(
                &target, request->epoch_generation,
                request->parent_position, 1) != 0)
            return token_epoch_fail(controller, request->epoch_generation);
        controller->exact_hit_count++;
        result->path = SALT_TEXT_TOKEN_PATH_EXACT_HIT;
    } else {
        if (controller->exact_miss_count == UINT64_MAX ||
            controller->cold_search_count == UINT64_MAX)
            return token_epoch_fail(controller, request->epoch_generation);
        controller->exact_miss_count++;
        controller->phase = SALT_TEXT_TOKEN_EPOCH_COLD_SEARCH;
        controller->cold_search_count++;
        memset(&target, 0, sizeof target);
        if (request->cold_search(
                request->cold_context, request->route, &target) != 0 ||
            token_epoch_result_valid(
                &target, request->epoch_generation,
                request->parent_position, 0) != 0)
            return token_epoch_fail(controller, request->epoch_generation);
        result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    }
    result->target = target;
    controller->active = 0;
    controller->phase = SALT_TEXT_TOKEN_EPOCH_RESOLVED;
    controller->last_epoch_generation = request->epoch_generation;
    return 0;
}

static uint32_t target_sublane_active_workers(
        uint32_t candidate_count, uint32_t worker_budget) {
    uint32_t workers = worker_budget;
    if (workers > candidate_count) workers = candidate_count;
    while (workers > 1u && candidate_count % workers != 0u) workers--;
    return workers;
}

/* Proof chooses only the authoritative payload extent B; it never performs
 * neural work or mutates state.  The chosen known-token prefix is then executed
 * once through the model's existing authoritative materializer.  B=1 and B=j
 * are therefore the same mechanism with different extents. */
static int target_proof_materialize_execute(
        const SaltTextGenerationBinding *binding,
        const int32_t *candidate_token_ids, uint32_t candidate_count,
        SaltTextGenerated *generated) {
    const SaltTextVerifyProgram *program;
    SaltTextVerifyResult materialized;
    uint64_t parent_generation;
    uint32_t parent_position, checked = 0u, proven = 0u;
    if (!binding || !(program = binding->program) || !program->kv_state ||
        !binding->prove_prefix || !binding->materialize_known ||
        !candidate_token_ids || candidate_count == 0u || !generated)
        return -1;
    parent_generation = program->kv_state->transition_generation;
    parent_position = program->kv_state->position;
    if (binding->prove_prefix(binding->context, candidate_token_ids,
            candidate_count, parent_generation, parent_position,
            &checked, &proven) != 0 ||
        program->kv_state->transition_generation != parent_generation ||
        program->kv_state->position != parent_position || checked == 0u ||
        checked > candidate_count || proven == 0u || proven > checked ||
        proven > candidate_count ||
        (proven < candidate_count && checked <= proven))
        return -1;
    memset(&materialized, 0, sizeof materialized);
    if (binding->materialize_known(binding->context, candidate_token_ids,
            proven, &materialized) != 0 ||
        materialized.status != SALT_TEXT_VERIFY_COMMITTED ||
        materialized.accepted_count != proven ||
        materialized.committed_count != proven ||
        materialized.produced_count != proven + 1u ||
        materialized.result_position != parent_position + proven ||
        materialized.transition_generation != parent_generation + 1u ||
        program->kv_state->position != parent_position + proven ||
        program->kv_state->transition_generation != parent_generation + 1u ||
        materialized.winning_node_index + 1u != proven ||
        materialized.winning_node_id + 1u != proven ||
        !materialized.pending_logits ||
        materialized.pending_logits_count != program->vocabulary ||
        materialized.backend.engine_submissions != 1u ||
        materialized.backend.completion_fences != 1u ||
        materialized.backend.intermediate_host_publications != 0u)
        return -1;
    generated->target = materialized;
    generated->epoch_path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    generated->proof_checked_rows = checked;
    generated->proven_prefix_rows = proven;
    generated->first_unproven = proven < candidate_count
        ? proven : candidate_count;
    generated->materialized_rows = proven;
    generated->target_model_rows = proven;
    generated->target_sublane_cancelled_rows = candidate_count - proven;
    return 0;
}

/* Reuse the resolved horizontal NFQ storage as a vertical Wa x Y lifetime
 * board.  Round q submits the existing contiguous causal slab
 * [q*Wa,(q+1)*Wa).  A first partial slab globally terminates every later q
 * seat; target completion order never selects or commits. */
static int target_sublane_execute(
        SaltTextTokenEpochController *controller,
        const SaltTextGenerationBinding *binding,
        const int32_t *candidate_token_ids, uint32_t candidate_count,
        const float *parent_logits, SaltTextGenerated *generated) {
    SaltAreaNfqMatrix *matrix;
    SaltAreaNfqPlan plan;
    SaltTextVerifyBackendStats backend;
    SaltTextTokenEpochResult final;
    const SaltTextVerifyProgram *program;
    const float *parent = parent_logits;
    uint32_t route_id = 0u, accepted = 0u, model_rows = 0u;
    uint32_t workers, depth, rounds = 0u;
    uint64_t epoch_generation, parent_generation;
    int resolved = 0;
    if (!controller || !binding || !(program = binding->program) ||
        !program->kv_state || !binding->policy || !binding->target ||
        !candidate_token_ids || candidate_count == 0u || !parent_logits ||
        !generated || !(matrix = binding->target_sublane_matrix) ||
        !binding->target_sublane_ready_items ||
        binding->target_sublane_ready_capacity == 0u ||
        binding->policy->worker_budget == 0u ||
        binding->policy->worker_budget > 32u || matrix->ready ||
        !matrix->routes || !matrix->items)
        return -1;
    workers = target_sublane_active_workers(
        candidate_count, binding->policy->worker_budget);
    if (workers == 0u || workers > binding->target_sublane_ready_capacity ||
        candidate_count % workers != 0u)
        return -1;
    depth = candidate_count / workers;
    if (depth == 0u || depth > SALT_TEXT_NFQ_MAX_QUEUE)
        return -1;
    generated->target_sublane_workers = workers;
    generated->target_sublane_depth = depth;
    parent_generation = program->kv_state->transition_generation;
    epoch_generation = parent_generation;
    if (epoch_generation < matrix->epoch_generation)
        epoch_generation = matrix->epoch_generation;
    if (epoch_generation == UINT64_MAX) return -1;
    epoch_generation++;
    memset(&plan, 0, sizeof plan);
    plan.workers = workers;
    plan.sequence_tiles = workers;
    plan.route_count = 1u;
    plan.queue_length = depth;
    if (salt_area_nfq_start(matrix, &plan, epoch_generation,
            parent_generation, program->kv_state->position,
            &route_id, 1u) != 0)
        return -1;
    memset(&backend, 0, sizeof backend);
    memset(&final, 0, sizeof final);
    for (uint32_t round = 0u; round < depth; round++) {
        SaltTextTokenEpochResult step;
        uint32_t ready_count = 0u, seen = 0u;
        uint32_t source_position = program->kv_state->position;
        if (salt_area_nfq_take_ready(matrix,
                epoch_generation, parent_generation,
                binding->target_sublane_ready_items,
                binding->target_sublane_ready_capacity, &ready_count) != 0 ||
            ready_count != workers)
            goto fail;
        for (uint32_t ready = 0u; ready < ready_count; ready++) {
            uint32_t index = binding->target_sublane_ready_items[ready];
            const SaltAreaNfqItem *item;
            if (index >= matrix->item_count ||
                !(item = &matrix->items[index]) ||
                item->route_slot != 0u || item->sequence_index >= workers ||
                item->queue_index != round || item->owner_worker >= workers ||
                item->state != SALT_AREA_WFQ_ITEM_RUNNING ||
                (seen & ((uint32_t)1u << item->sequence_index)) != 0u)
                goto fail;
            seen |= (uint32_t)1u << item->sequence_index;
        }
        if (seen != (workers == 32u ? UINT32_MAX :
                (((uint32_t)1u << workers) - 1u)))
            goto fail;
        memset(&step, 0, sizeof step);
        if (binding->target(binding->context, controller,
                candidate_token_ids + (size_t)round * workers,
                workers, parent, &step) != 0 ||
            step.path != SALT_TEXT_TOKEN_PATH_COLD_SEARCH ||
            step.target.status != SALT_TEXT_VERIFY_COMMITTED ||
            step.target.accepted_count > workers ||
            step.target.committed_count != step.target.accepted_count ||
            step.target.produced_count != step.target.accepted_count + 1u ||
            step.target.result_position !=
                source_position + step.target.accepted_count ||
            !step.target.pending_logits ||
            step.target.pending_logits_count != program->vocabulary ||
            step.target.backend.intermediate_host_publications != 0u)
            goto fail;
        if (step.target.accepted_count == 0u) {
            if (step.target.backend.engine_submissions != 0u ||
                step.target.backend.completion_fences != 0u ||
                step.target.projection_rows != 0u)
                goto fail;
        } else if (step.target.backend.engine_submissions != 1u ||
                   step.target.backend.completion_fences != 1u ||
                   !step.target.projection_logits ||
                   step.target.projection_rows != workers ||
                   step.target.winning_node_index + 1u !=
                       step.target.accepted_count) {
            goto fail;
        }
        if (add_backend_stats(&backend, &step.target.backend) != 0 ||
            UINT32_MAX - model_rows < step.target.projection_rows ||
            UINT32_MAX - accepted < step.target.accepted_count)
            goto fail;
        model_rows += step.target.projection_rows;
        accepted += step.target.accepted_count;
        rounds++;
        generated->target_sublane_rounds = rounds;
        final = step;
        if (step.target.accepted_count < workers) {
            uint32_t retired = 0u;
            if (step.target.accepted_count == 0u) {
                if (salt_area_nfq_cancel(matrix, epoch_generation,
                        parent_generation, &retired) != 0)
                    goto fail;
            } else if (salt_area_nfq_resolve(matrix, epoch_generation,
                    parent_generation, 0u,
                    step.target.accepted_count - 1u, round, &retired) != 0) {
                goto fail;
            }
            resolved = 1;
            break;
        }
        for (uint32_t ready = 0u; ready < ready_count; ready++) {
            uint32_t retired = 0u;
            if (salt_area_nfq_complete(matrix, epoch_generation,
                    parent_generation,
                    binding->target_sublane_ready_items[ready],
                    SALT_AREA_WFQ_CONTINUE, &retired) != 0 || retired != 0u)
                goto fail;
        }
        if (round + 1u == depth) {
            uint32_t retired = 0u;
            if (salt_area_nfq_resolve(matrix, epoch_generation,
                    parent_generation, 0u, workers - 1u, round,
                    &retired) != 0)
                goto fail;
            resolved = 1;
            break;
        }
        parent = step.target.pending_logits;
    }
    if (!resolved || matrix->ready || !matrix->resolved || accepted == 0u ||
        final.target.status != SALT_TEXT_VERIFY_COMMITTED)
        return -1;
    final.target.accepted_count = accepted;
    final.target.committed_count = accepted;
    final.target.produced_count = accepted + 1u;
    final.target.result_position =
        program->kv_state->position;
    final.target.winning_node_index = accepted - 1u;
    final.target.winning_node_id = accepted - 1u;
    final.target.backend = backend;
    generated->target = final.target;
    generated->epoch_path = (uint32_t)final.path;
    generated->target_model_rows = model_rows;
    generated->target_sublane_rounds = rounds;
    generated->target_sublane_cancelled_rows =
        candidate_count - model_rows;
    return 0;
fail:
    if (matrix->ready && !matrix->resolved) {
        uint32_t ignored = 0u;
        (void)salt_area_nfq_cancel(
            matrix, epoch_generation, parent_generation, &ignored);
    }
    return -1;
}

int salt_text_generate_candidates(
        SaltTextTokenEpochController *controller,
        const SaltTextGenerationBinding *binding, float *logits,
        uint32_t maximum_tokens, float *scratch_logits,
        uint64_t (*now_ns)(void *), void *clock_context,
        SaltTextGenerated *result) {
    SaltTextTokenEpochController selection_controller;
    SaltTextTokenEpochResult selection, epoch;
    const SaltTextTargetPolicy *policy;
    const SaltTextVerifyProgram *program;
    const SaltTextProjectionWindow *previous;
    int32_t nfq_ids[SALT_TEXT_NFQ_MAX_CHECKS];
    int32_t root = -1;
    uint32_t x, f, n = 1u, checks, position;
    uint32_t reused = 0u, matched = 0u, winning_trajectory = UINT32_MAX;
    uint64_t started, ended, checks64, parent_generation;
    int rc, selected_root;
    if (result) memset(result, 0, sizeof *result);
    if (!controller || !binding || !(program = binding->program) ||
        !program->ready || !program->kv_state ||
        !(policy = binding->policy) || !policy->ready ||
        !(previous = binding->projection) || !binding->select_proposal ||
        !binding->is_stop || !logits || !scratch_logits || !result ||
        !now_ns || maximum_tokens == 0u)
        return -1;
    position = program->kv_state->position;
    if (position >= program->maximum_context) return -1;
    /* X is the causal X-TARGET depth of one attempt. N/F/Q never enter here:
     * this generator builds one trajectory per F lane, not NFQ checks. */
    x = binding->target ? policy->target_rows : 1u;
    f = policy->route_count;
    if (x == 0u || f == 0u || x > UINT32_MAX / f ||
        x * f > program->maximum_candidates ||
        x * f > SALT_TEXT_NFQ_MAX_CHECKS)
        return -1;
    checks64 = (uint64_t)policy->candidate_count * policy->queue_length;
    if (policy->candidate_count == 0u || policy->queue_length == 0u ||
        checks64 == 0u || checks64 > SALT_TEXT_NFQ_MAX_CHECKS ||
        checks64 > program->vocabulary)
        return -1;
    checks = (uint32_t)checks64;
    if (x > maximum_tokens) x = maximum_tokens;
    if (x > program->maximum_context - position)
        x = program->maximum_context - position;
    started = now_ns(clock_context);
    if (verify_greedy_token(logits, program->vocabulary, &root) != 0)
        return -1;
    /* NFQ is an independent cheap parent-root selection. Execute it before
     * X-only exits, including X disabled, X1, stop and no causal trajectory.
     * The existing selector must leave authoritative KV/state untouched. */
    if (projection_ranked_ids(logits, program->vocabulary,
            checks, nfq_ids) != 0)
        return -1;
    memcpy(scratch_logits, logits,
           (size_t)program->vocabulary * sizeof(float));
    parent_generation = program->kv_state->transition_generation;
    memset(&selection_controller, 0, sizeof selection_controller);
    memset(&selection, 0, sizeof selection);
    if (salt_text_token_epoch_init(&selection_controller) != 0 ||
        binding->select_proposal(binding->context, &selection_controller,
            nfq_ids, checks, scratch_logits, &selection) != 0 ||
        selection.path != SALT_TEXT_TOKEN_PATH_COLD_SEARCH ||
        selection.target.status != SALT_TEXT_VERIFY_COMMITTED ||
        selection.target.accepted_count != 0u ||
        selection.target.committed_count != 0u ||
        selection.target.produced_count != 1u ||
        selection.target.result_position != position ||
        selection.target.transition_generation != parent_generation + 1u ||
        selection.target.winning_node_index >= checks ||
        selection.target.pending_token_id < 0 ||
        (uint32_t)selection.target.pending_token_id >= program->vocabulary ||
        selection.target.pending_logits != scratch_logits ||
        selection.target.pending_logits_count != program->vocabulary ||
        selection.target.backend.engine_submissions != 0u ||
        selection.target.backend.completion_fences != 0u ||
        selection.target.backend.intermediate_host_publications != 0u ||
        program->kv_state->position != position ||
        program->kv_state->transition_generation != parent_generation)
        return -1;
    selected_root = selection.target.pending_token_id;
    if (nfq_ids[selection.target.winning_node_index] != selected_root)
        return -1;
    result->candidate_count = checks;
    if (!binding->target || x < 2u || binding->is_stop(binding->context, root)) {
        ended = now_ns(clock_context);
        if (!started || ended < started || selected_root != root) return -1;
        result->proposal_ns = ended - started;
        return 1;
    }
    /* Source 1, parent cache route: committed history. One lane only; a
     * greedy target can accept no other root, so extra lanes are waste. */
    if (f == 1u && (controller->schedule_history_count ||
                    controller->schedule_output_count)) {
        rc = salt_text_history_refill_candidates(
            controller->schedule_history_ids,
            controller->schedule_history_count,
            controller->schedule_output_ids,
            controller->schedule_output_count,
            root, program->vocabulary, x, result->candidate_token_ids,
            SALT_TEXT_NFQ_MAX_CHECKS, &matched);
        if (rc < 0) return -1;
        if (rc >= 2) {
            n = (uint32_t)rc;
            result->proposal_source = SALT_TEXT_PROPOSAL_SOURCE_HISTORY;
            result->history_matched_length = matched;
        }
    }
    /* Source 2, target-block local memory: rows the previous X-TARGET block
     * already computed past its accepted prefix. */
    if (n < 2u && previous->logits) {
        uint32_t covered;
        if (previous->generation != program->kv_state->transition_generation ||
            previous->position != position ||
            previous->accepted_count == 0u ||
            previous->accepted_count > previous->sequence_tiles)
            return -1;
        covered = previous->sequence_tiles - previous->accepted_count + 1u;
        if (covered > x) covered = x;
        if (covered >= 2u) {
            rc = salt_text_projection_refill_candidates(
                program, previous, logits, covered, f,
                result->candidate_token_ids, SALT_TEXT_NFQ_MAX_CHECKS,
                &reused);
            if (rc < 0) return -1;
            if (rc == 0) {
                n = covered;
                result->proposal_source = SALT_TEXT_PROPOSAL_SOURCE_PROJECTION;
                result->projection_reused_rows = reused;
            }
        }
    }
    /* Source 3, true cold: root only. The caller performs one ordinary
     * transition; this generator never manufactures ranks or model steps. */
    if (n < 2u) {
        ended = now_ns(clock_context);
        if (!started || ended < started || selected_root != root) return -1;
        result->proposal_ns = ended - started;
        return 1;
    }
    if (result->candidate_token_ids[0] != root) return -1;
    /* A stop token ends the trajectory; nothing may be proposed after it. */
    for (uint32_t lane = 0u; lane < f; lane++)
        for (uint32_t tile = 1u; tile < n; tile++)
            if (binding->is_stop(binding->context,
                    result->candidate_token_ids[lane * n + tile])) {
                if (f != 1u) return 1;
                n = tile + 1u;
                break;
            }
    if (n < 2u) return 1;
    result->proposal_count = n;
    /* Map the NFQ-selected root to one causal trajectory. Losing roots never
     * enter model execution; X remains an independently admitted consumer. */
    for (uint32_t trajectory = 0u; trajectory < f; trajectory++) {
        if (result->candidate_token_ids[trajectory * n] != selected_root)
            continue;
        if (winning_trajectory != UINT32_MAX) return -1;
        winning_trajectory = trajectory;
    }
    if (winning_trajectory == UINT32_MAX) {
        ended = now_ns(clock_context);
        if (!started || ended < started) return -1;
        result->proposal_ns = ended - started;
        return 1;
    }
    if (winning_trajectory != 0u)
        memmove(result->candidate_token_ids,
            result->candidate_token_ids + winning_trajectory * n,
            (size_t)n * sizeof *result->candidate_token_ids);
    /* A vertical board uses every allocated lane in each submitted slab.  A
     * stop or short proposal can leave a ragged suffix; defer that suffix to
     * the next authoritative boundary instead of creating phantom seats or a
     * one-lane board whose Q lifetime exceeds the fixed NFQ matrix. */
    if (binding->target_sublane_matrix && policy->worker_budget != 0u &&
        n > policy->worker_budget) {
        uint32_t remainder = n % policy->worker_budget;
        if (remainder != 0u) n -= remainder;
        if (n < 2u) return 1;
        result->proposal_count = n;
        if (result->projection_reused_rows >= n)
            result->projection_reused_rows = n - 1u;
    }
    result->route_token_count = n;
    result->candidate_count = checks;
    ended = now_ns(clock_context);
    if (!started || ended < started) return -1;
    result->proposal_ns = ended - started;
    /* Exact authority chooses B, then one common materializer executes B known
     * rows.  The preserved model-backed sublane scheduler remains available as
     * a control when no proof/materializer pair is bound. */
    if (binding->prove_prefix || binding->materialize_known) {
        if (!binding->prove_prefix || !binding->materialize_known ||
            target_proof_materialize_execute(binding,
                result->candidate_token_ids, n, result) != 0)
            return -1;
    } else if (binding->target_sublane_matrix) {
        if (target_sublane_execute(controller, binding,
                result->candidate_token_ids, n, scratch_logits, result) != 0)
            return -1;
    } else {
        memset(&epoch, 0, sizeof epoch);
        if (binding->target(binding->context, controller,
                result->candidate_token_ids, n, scratch_logits, &epoch) != 0 ||
            epoch.path != SALT_TEXT_TOKEN_PATH_COLD_SEARCH ||
            epoch.target.status != SALT_TEXT_VERIFY_COMMITTED ||
            epoch.target.accepted_count == 0u ||
            epoch.target.accepted_count > n ||
            epoch.target.committed_count != epoch.target.accepted_count ||
            epoch.target.produced_count != epoch.target.accepted_count + 1u ||
            epoch.target.backend.engine_submissions != 1u ||
            epoch.target.backend.completion_fences != 1u ||
            epoch.target.backend.intermediate_host_publications != 0u ||
            !epoch.target.projection_logits ||
            epoch.target.projection_rows != n ||
            !epoch.target.pending_logits ||
            epoch.target.pending_logits_count != program->vocabulary ||
            epoch.target.result_position != position + epoch.target.accepted_count)
            return -1;
        if (epoch.target.winning_node_index >= n ||
            epoch.target.winning_node_index + 1u !=
                epoch.target.accepted_count)
            return -1;
        result->epoch_path = (uint32_t)epoch.path;
        result->target = epoch.target;
        result->target_model_rows = epoch.target.projection_rows;
        result->target_sublane_workers = n;
        result->target_sublane_depth = 1u;
        result->target_sublane_rounds = 1u;
    }
    /* The retained boundary: target logits at j-1 select the next root. */
    memmove(logits, result->target.pending_logits,
           (size_t)program->vocabulary * sizeof(float));
    return 0;
}

static uint32_t schedule_position(const SaltTextSchedulerBindings *b) {
    return b->generation.program->kv_state->position;
}

static int schedule_stop(const SaltTextSchedulerBindings *b, int32_t token) {
    return b->generation.is_stop(b->generation.context, token);
}

static int schedule_emit(const SaltTextSchedulerBindings *b,
                         const SaltTextScheduleRequest *q,
                         SaltTextScheduleResult *r, int32_t token) {
    int stop;
    if (r->output_count >= q->output_limit) return -1;
    q->output_ids[r->output_count++] = token;
    stop = schedule_stop(b, token);
    if (b->emit_token(b->context, r->output_count, token, stop) != 0) return -1;
    if (stop) r->stop_token = token;
    return 0;
}

static int schedule_sample(const SaltTextSchedulerBindings *b,
                           const SaltTextScheduleRequest *q,
                           SaltTextScheduleResult *r, int *token) {
    const float *logits = q->logits;
    if (q->repetition_penalty > 1.0f && r->output_count > 0u) {
        uint32_t count = r->output_count;
        if (count > SALT_REPETITION_MAX_WINDOW) count = SALT_REPETITION_MAX_WINDOW;
        memcpy(q->scratch_logits, q->logits,
            (size_t)b->generation.program->vocabulary * sizeof(float));
        if (salt_apply_repetition_penalty(q->scratch_logits,
                (int)b->generation.program->vocabulary,
                q->output_ids + r->output_count - count, count,
                q->repetition_penalty) != 0) return -1;
        logits = q->scratch_logits;
    }
    int rc = q->top_p > 0.0f && q->top_p < 1.0f
        ? salt_sampler_select_top_p(&q->sampler, logits,
            (int)b->generation.program->vocabulary, q->top_p, schedule_position(b), token)
        : salt_sampler_select(&q->sampler, logits,
            (int)b->generation.program->vocabulary, schedule_position(b), token);
    if (rc == 0 && q->sampler.abi != SALT_SAMPLER_GREEDY_V1)
        r->sampler_draws++;
    return rc;
}

static int schedule_native(SaltTextTokenEpochController *controller,
                           const SaltTextScheduleRequest *q,
                           SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b = controller->scheduler;
    uint32_t source = schedule_position(b);
    uint64_t begin, selected, step_begin = 0, end = 0, elapsed;
    int token, committed = 0;
    begin = q->sampler.abi == SALT_SAMPLER_GREEDY_V1 ? 0 : b->now_ns(b->context);
    if (schedule_sample(b, q, r, &token) != 0) return -1;
    selected = q->sampler.abi == SALT_SAMPLER_GREEDY_V1 ? 0 : b->now_ns(b->context);
    if (schedule_emit(b, q, r, (int32_t)token) != 0) return -1;
    if (r->stop_token >= 0 && q->sampler.abi == SALT_SAMPLER_GREEDY_V1)
        return 0;
    if (r->stop_token < 0 && r->output_count < q->output_limit) {
        step_begin = b->now_ns(b->context);
        if (!step_begin || b->step_known(b->context, token, q->logits) != 0)
            return -1;
        end = b->now_ns(b->context);
        if (end < step_begin || schedule_position(b) != source + 1u) return -1;
        committed = 1;
    }
    elapsed = committed ? end - step_begin : 0;
    if (q->sampler.abi != SALT_SAMPLER_GREEDY_V1) {
        if (!begin || selected < begin) return -1;
        elapsed += selected - begin;
    }
    if (committed && q->sampler.abi == SALT_SAMPLER_GREEDY_V1) {
        if (controller->schedule_bootstraps == UINT32_MAX) return -1;
        controller->schedule_bootstraps++;
    }
    b->stats->decode_ordinary_ns += elapsed;
    return b->observe_native
        ? b->observe_native(b->context, source, token, committed, elapsed,
                            r->sampler_draws) : 0;
}

static int schedule_normal(SaltTextTokenEpochController *controller,
                           const SaltTextScheduleRequest *q,
                           SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b = controller->scheduler;
    SaltTextGenerated generated;
    uint32_t remaining = q->output_limit - r->output_count;
    uint32_t effective_x;
    uint64_t begin, end;
    int rc;
    if (!b->generation.select_proposal || q->sampler.abi != SALT_SAMPLER_GREEDY_V1)
        return schedule_native(controller, q, r);
    /* Parent cache route views: committed prompt/session tokens, then the
     * outputs this run has already committed. Borrowed, never copied. */
    controller->schedule_history_ids = q->history_ids;
    controller->schedule_history_count = q->history_ids ? q->history_count : 0u;
    controller->schedule_output_ids = q->output_ids;
    controller->schedule_output_count = r->output_count;
    effective_x = 1u;
    if (b->generation.target && salt_text_target_policy_effective_x(
            b->generation.policy, controller->schedule_prior_kv_rows, remaining,
            &effective_x) != 0)
        return -1;
    begin = b->now_ns(b->context);
    rc = salt_text_generate_candidates(controller, &b->generation,
        q->logits, effective_x, q->scratch_logits,
        b->now_ns, b->context, &generated);
    if (rc == 1) {
        /* True cold or one remaining seat: one ordinary transition. */
        b->stats->decode_ordinary_ns += generated.proposal_ns;
        return schedule_native(controller, q, r);
    }
    if (rc != 0) return -1;
    end = b->now_ns(b->context);
    if (!begin || end < begin || generated.proposal_count < 2u ||
        generated.proposal_count > remaining ||
        generated.target.accepted_count == 0u ||
        generated.target.accepted_count > generated.proposal_count ||
        generated.target.committed_count != generated.target.accepted_count ||
        generated.target.produced_count != generated.target.accepted_count + 1u ||
        b->stats->proposed_tokens > UINT32_MAX - generated.proposal_count ||
        b->stats->accepted_tokens > UINT32_MAX - generated.target.accepted_count)
        return -1;
    b->stats->proposed_tokens += generated.proposal_count;
    b->stats->accepted_tokens += generated.target.accepted_count;
    if (generated.target.accepted_count < generated.proposal_count)
        b->stats->rejection_count++;
    b->stats->decode_ordinary_ns += generated.proposal_ns;
    if (end - begin >= generated.proposal_ns)
        b->stats->decode_verify_ns += end - begin - generated.proposal_ns;
    if (b->observe_target && b->observe_target(b->context, &generated, end - begin) != 0)
        return -1;
    /* Only the target-accepted prefix is emitted; the correction at j stays
     * pending in q->logits as the next episode's root. */
    for (uint32_t index = 0; index < generated.target.accepted_count; index++) {
        if (schedule_emit(b, q, r, generated.candidate_token_ids[index]) != 0)
            return -1;
        if (r->stop_token >= 0 && index + 1u != generated.target.accepted_count)
            return -1;
    }
    return 0;
}

static int schedule_normal_dpr(SaltTextTokenEpochController *controller,
                               const SaltTextScheduleRequest *q,
                               SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b = controller->scheduler;
    uint32_t source = schedule_position(b), first = r->output_count, committed;
    if (schedule_normal(controller, q, r) != 0 || schedule_position(b) < source)
        return -1;
    committed = schedule_position(b) - source;
    if (committed > r->output_count - first ||
        r->output_count - first - committed > 1u)
        return -1;
    return b->advance_chain(b->context, q->output_ids + first, committed);
}

static int schedule_pending(const SaltTextSchedulerBindings *b,
                            const SaltTextScheduleRequest *q,
                            const SaltTextScheduleResult *r, float *logits) {
    int32_t token;
    uint64_t position;
    int pending;
    if (!b->dpr || b->dpr->mode == SALT_DPR_OFF) return 0;
    pending = salt_dpr_pending_read(b->dpr->runtime, &token, &position);
    if (pending <= 0) return pending;
    if (r->output_count == 0u || q->output_ids[r->output_count - 1u] != token ||
        position != schedule_position(b) ||
        b->step_known(b->context, token, logits) != 0 ||
        b->advance_chain(b->context, &token, 1u) != 0 ||
        schedule_position(b) != position + 1u ||
        salt_dpr_pending_commit(b->dpr->runtime, token, position + 1u) != 0) {
        (void)salt_dpr_runtime_fail(b->dpr->runtime);
        return -1;
    }
    return 1;
}

static int schedule_nomogram(const SaltTextSchedulerBindings *b,
                             const SaltTextScheduleRequest *q,
                             const SaltTextScheduleResult *r,
                             int32_t target, uint32_t maximum_tokens,
                             SaltTextScheduleSelection *s) {
    SaltTextScheduleDpr *d = b->dpr;
    uint32_t limit = q->output_limit - r->output_count - 1u, attempts = 0;
    size_t candidates = 0;
    int selected = 0;
    if (maximum_tokens == 0u) return -1;
    if (limit > maximum_tokens) limit = maximum_tokens;
    if (limit > d->draft_n) limit = d->draft_n;
    if (d->mentor_enabled) {
        if (!d->families || !d->bindings || !d->mentor_candidates ||
            !d->mentor_capacity || !d->effective_charts) return -1;
        for (size_t edge = 0; edge < d->store->edge_count; edge++) {
            d->effective_charts[edge] = d->store->edges[edge].edge.chart;
            if (d->stats[edge].round_count >= SALT_DPR_STATS_MIN_SAMPLES &&
                salt_dpr_stats_chart(&d->stats[edge], &d->effective_charts[edge]) != 0)
                return -1;
        }
        if (salt_dpr_mentor_candidates_build(d->bindings, d->families,
                d->store, d->retained, d->effective_charts, d->mode, s->node,
                d->serial_ns, SALT_DPR_STATS_MIN_SAMPLES, d->mentor_candidates,
                d->mentor_capacity, &candidates) != 0) return -1;
        selected = salt_dpr_selector_choose(d->families->families,
            d->families->family_count, d->mentor_candidates, candidates,
            s->node, d->compatibility, d->mentor_policy, &s->mentor);
        if (selected < 0) return -1;
        if (selected > 0) {
            selected = salt_dpr_mentor_walk_nomogram(d->bindings, d->families,
                d->store, d->retained, d->effective_charts, s->mentor.family_index,
                d->mode, s->node, d->serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
                (int32_t)b->generation.program->vocabulary,
                b->generation.program->maximum_context, b->generation.is_stop,
                b->generation.context, limit, s->tokens, s->indices,
                s->binding_indices, &s->count);
            if (selected < 0) return -1;
            if (selected > 0 && (!s->count || s->tokens[0] != target)) {
                selected = 0;
                s->count = 0;
            }
            if (selected > 0) {
                if (memcmp(d->store->edges[s->indices[0]].edge_sha256,
                        s->mentor.edge_sha256, 32u) != 0 ||
                    memcmp(d->bindings->bindings[s->binding_indices[0]].binding_sha256,
                        s->mentor.binding_sha256, 32u) != 0) return -1;
                s->mentor_selected = 1;
                s->coverage_roots = 1u;
            }
        }
    }
    if (!s->mentor_selected) {
        selected = salt_dpr_stats_walk_nomogram_coverage(d->store, d->stats,
            d->retained, s->node, d->mode, d->serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
            (int32_t)b->generation.program->vocabulary,
            b->generation.program->maximum_context, b->generation.is_stop,
            b->generation.context, target, limit, s->tokens, s->indices,
            &s->count, &attempts);
        s->coverage_roots += attempts;
    }
    if (!s->mentor_selected && selected == 0 && r->output_count == 0u) {
        s->count = 0;
        attempts = 0;
        selected = salt_dpr_stats_walk_nomogram_coverage(d->store, d->stats,
            d->retained, d->qa_key, d->mode, d->serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
            (int32_t)b->generation.program->vocabulary,
            b->generation.program->maximum_context, b->generation.is_stop,
            b->generation.context, target, limit, s->tokens, s->indices,
            &s->count, &attempts);
        s->coverage_roots += attempts;
    }
    return selected;
}

static int schedule_record_stats(const SaltTextSchedulerBindings *b,
                                 const SaltTextScheduleSelection *s,
                                 const SaltTextVerifyResult *target, int fallback) {
    uint32_t accepted = target ? target->accepted_count : 0u;
    uint32_t attempted = fallback ? 0u :
        (accepted < s->count ? accepted + 1u : s->count);
    uint32_t count = s->kind == SALT_DPR_EDGE_PARENT_EXACT ? 1u : s->count;
    if (!count || accepted > s->count) return -1;
    for (uint32_t edge = 0; edge < count; edge++) {
        SaltDprEdgeStats *stats = &b->dpr->stats[s->indices[edge]];
        uint64_t lookup = s->lookup_ns / count + (edge < s->lookup_ns % count);
        uint64_t verify = attempted && edge < attempted
            ? s->verify_ns / attempted + (edge < s->verify_ns % attempted) : 0;
        uint32_t committed = !fallback && edge < accepted ? 1u : 0u;
        uint32_t hit_accepted = committed;
        if (s->kind == SALT_DPR_EDGE_PARENT_EXACT) {
            if (!target) return -1;
            hit_accepted = target->accepted_count;
            committed = target->committed_count;
        }
        if (stats->last_used_epoch == UINT64_MAX || salt_dpr_stats_observe(stats,
                lookup, s->recover_ns, verify, hit_accepted, committed,
                !fallback, fallback, stats->last_used_epoch + 1u) != 0) return -1;
    }
    return 0;
}

static int schedule_attention_prepare(const SaltTextSchedulerBindings *b,
                                      const SaltTextScheduleSelection *s,
                                      int has_transition) {
    SaltTextScheduleDpr *d = b->dpr;
    SaltDprComputeIntent intent;
    size_t selected = SIZE_MAX;
    uint64_t start, end, elapsed = 0;
    int rc;
    memset(&intent, 0, sizeof intent);
    intent.schema_version = SALT_DPR_COMPUTE_INTENT_VERSION;
    intent.kind = has_transition ? SALT_DPR_OPERATION_DRAFT_VERIFY :
                                  SALT_DPR_OPERATION_NATIVE_DECODE;
    intent.position = s->source_position;
    intent.horizon = has_transition ? s->count : 1u;
    intent.candidate_count = has_transition ? s->count : 0u;
    memcpy(intent.target_node_sha256, s->node, 32);
    memcpy(intent.state_compatibility_sha256, d->compatibility, 32);
    if (d->attention_enabled) memcpy(intent.active_policy_sha256, d->attention_policy, 32);
    else memset(intent.active_policy_sha256, 0xff, 32);
    if (has_transition && salt_dpr_candidate_sha256(s->tokens, s->count,
            intent.candidate_sha256) != 0) return -1;
    if (!d->attention_enabled) return 0;
    if (!d->attention_store) return -1;
    start = b->now_ns(b->context);
    if (d->hot_ready && *d->hot_ready) {
        SaltDprAttentionPlan runtime_plan;
        uint8_t sha[32];
        if (!d->hot_candidate || !d->hot_mindset || !d->hot_provenance) return -1;
        runtime_plan = *d->hot_candidate;
        if (salt_dpr_attention_plan_finalize_runtime(&runtime_plan, &intent,
                d->hot_mindset, d->hot_provenance) != 0 ||
            salt_dpr_attention_plan_store_upsert_runtime(d->attention_store,
                &runtime_plan, sha, &selected) != 0) return -1;
    }
    rc = salt_dpr_attention_plan_store_select(d->attention_store, &intent,
        d->attention_layers, d->attention_experts,
        b->generation.program->maximum_context, &selected);
    end = b->now_ns(b->context);
    if (start && end >= start) elapsed = end - start;
    if (rc <= 0) return rc;
    rc = b->prepare_attention(b->context, &intent,
                              &d->attention_store->plans[selected], elapsed);
    return rc < 0 ? -1 : (rc == 0 ? 1 : 0);
}

static int schedule_dpr_cycle(SaltTextTokenEpochController *controller,
                              const SaltTextScheduleRequest *q,
                              SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b = controller->scheduler;
    SaltTextScheduleDpr *d = b->dpr;
    SaltTextScheduleSelection s;
    SaltDprNodeMaterial material;
    SaltDprMemoryRequest memory;
    SaltDprMemoryDecision decision;
    SaltTextTokenEpochResult epoch;
    SaltTextVerifyResult target_result;
    uint32_t source = schedule_position(b), first = r->output_count;
    uint32_t remaining = q->output_limit - first, effective_x;
    uint64_t begin, end;
    size_t selected = SIZE_MAX;
    int rc, token, prepared;
    if (remaining == 1u) return schedule_normal_dpr(controller, q, r);
    memset(&s, 0, sizeof s);
    memset(&target_result, 0, sizeof target_result);
    s.bonus = -1;
    s.source_position = source;
    s.target_token = -1;
    s.output_base = first;
    b->stats->decode_cycles++;
    begin = b->now_ns(b->context);
    if (b->node_identity(b->context, &material, s.node) != 0) return -1;
    end = b->now_ns(b->context);
    if (begin && end >= begin) b->stats->decode_node_ns += end - begin;
    begin = b->now_ns(b->context);
    rc = salt_dpr_stats_store_select(d->store, d->stats, d->retained,
        s.node, d->mode, d->serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
        (int32_t)b->generation.program->vocabulary,
        b->generation.program->maximum_context, b->generation.is_stop,
        b->generation.context, &selected);
    if (rc < 0) return -1;
    if (rc > 0) {
        s.stored = &d->store->edges[selected];
        s.kind = s.stored->edge.kind;
        s.count = s.stored->edge.horizon;
        s.indices[0] = selected;
        end = b->now_ns(b->context);
        if (begin && end >= begin) s.lookup_ns = end - begin;
        b->stats->decode_lookup_ns += s.lookup_ns;
        if (s.kind != SALT_DPR_EDGE_PARENT_EXACT || !s.count ||
            s.count > SALT_DPR_MAX_HORIZON ||
            s.count >= b->generation.program->maximum_context ||
            source > b->generation.program->maximum_context - s.count - 1u ||
            s.stored->edge.reference_position != (uint64_t)source + s.count)
            return -1;
        if (s.count + 1u > remaining) {
            b->stats->decode_misses++;
            return schedule_normal_dpr(controller, q, r);
        }
        b->stats->parent_hits++;
        memcpy(s.tokens, s.stored->edge.candidate_token_ids,
               (size_t)s.count * sizeof(int32_t));
        s.bonus = s.stored->edge.bonus_token_id;
        begin = b->now_ns(b->context);
        if (b->load_exact(b->context, s.stored, source, s.count) != 0 ||
            schedule_position(b) != source + s.count) return -1;
        end = b->now_ns(b->context);
        if (begin && end >= begin) s.recover_ns = end - begin;
        b->stats->decode_parent_restore_ns += s.recover_ns;
        if (b->advance_chain(b->context, s.tokens, s.count) != 0) return -1;
        memcpy(q->output_ids + first, s.tokens, (size_t)s.count * sizeof(int32_t));
        q->output_ids[first + s.count] = s.bonus;
        r->output_count = first + s.count + 1u;
        r->stop_token = schedule_stop(b, s.bonus) ? s.bonus : -1;
        target_result.accepted_count = s.count;
        target_result.committed_count = s.count;
        target_result.produced_count = s.count + 1u;
        target_result.pending_token_id = s.bonus;
        if (r->stop_token < 0) {
            if (b->step_known(b->context, s.bonus, q->logits) != 0 ||
                b->advance_chain(b->context, &s.bonus, 1u) != 0) return -1;
            target_result.committed_count++;
        }
        target_result.status = SALT_TEXT_VERIFY_COMMITTED;
        target_result.result_position = schedule_position(b);
        if (b->observe && b->observe(b->context, &s, &target_result, 0) != 0)
            return -1;
        if (schedule_record_stats(b, &s, &target_result, 0) != 0) return -1;
        for (uint32_t index = first; index < r->output_count; index++) {
            int stop = schedule_stop(b, q->output_ids[index]);
            if (b->emit_token(b->context, index + 1u, q->output_ids[index], stop) != 0 ||
                (stop && index + 1u != r->output_count)) return -1;
        }
        return 0;
    }
    if (b->generation.policy->dpr_independent_draft) {
        effective_x = b->generation.policy->target_rows;
        if (effective_x > remaining) effective_x = remaining;
    } else if (salt_text_target_policy_effective_x(
            b->generation.policy, controller->schedule_prior_kv_rows, remaining,
            &effective_x) != 0)
        return -1;
    if (schedule_sample(b, q, r, &token) != 0) return -1;
    s.target_token = token;
    rc = schedule_nomogram(b, q, r, token, effective_x, &s);
    end = b->now_ns(b->context);
    if (begin && end >= begin) s.lookup_ns = end - begin;
    b->stats->decode_lookup_ns += s.lookup_ns;
    if (rc < 0) return -1;
    s.kind = SALT_DPR_EDGE_NOMOGRAM_DRAFT;
    if (b->observe && b->observe(b->context, &s, NULL,
            SALT_TEXT_SCHEDULE_OBSERVE_SOURCE) != 0) return -1;
    if (rc > 0) {
        if (!s.count || s.count > SALT_DPR_MAX_WALK_HORIZON ||
            s.count >= b->generation.program->maximum_context ||
            s.count + 1u > remaining ||
            source > b->generation.program->maximum_context - s.count - 1u)
            return -1;
        s.stored = &d->store->edges[s.indices[0]];
        b->stats->nomogram_hits++;
        b->stats->proposed_tokens += s.count;
        memset(&memory, 0, sizeof memory);
        memory.mode = d->mode;
        memory.current_kv_budget_bytes = d->current_kv_bytes;
        memory.additive_dpr_budget_bytes = d->additive_bytes;
        memory.target_required_bytes = b->kv_required(b->context, source + s.count + 1u);
        if (salt_dpr_memory_admit(&memory, &decision) != 0) return -1;
        if (!decision.admitted) {
            if (b->observe && b->observe(b->context, &s, NULL, 1) != 0) return -1;
            if (schedule_record_stats(b, &s, NULL, 1) != 0) return -1;
            rc = 0;
        }
    } else {
        b->stats->decode_misses++;
        if (b->observe && b->observe(b->context, &s, NULL, 0) != 0) return -1;
    }
    prepared = schedule_attention_prepare(b, &s, rc > 0);
    if (prepared < 0) return -1;
    if (rc == 0) {
        rc = schedule_normal_dpr(controller, q, r);
        if (b->release_attention(b->context) != 0) return -1;
        if (rc == 0 && prepared > 0 && b->observe &&
            b->observe(b->context, &s, NULL,
                SALT_TEXT_SCHEDULE_OBSERVE_ATTENTION_READY) != 0) return -1;
        return rc;
    }
    if (salt_dpr_verify_begin(d->runtime) != 0) {
        (void)b->release_attention(b->context);
        return -1;
    }
    begin = b->now_ns(b->context);
    memset(&epoch, 0, sizeof epoch);
    rc = b->generation.target(b->generation.context, controller,
        s.tokens, s.count, q->logits, &epoch);
    end = b->now_ns(b->context);
    if (begin && end >= begin) s.verify_ns = end - begin;
    target_result = epoch.target;
    if (rc != 0 || epoch.path != SALT_TEXT_TOKEN_PATH_COLD_SEARCH ||
        target_result.status != SALT_TEXT_VERIFY_COMMITTED ||
        target_result.accepted_count > s.count ||
        target_result.committed_count != target_result.accepted_count ||
        target_result.produced_count != target_result.accepted_count + 1u ||
        target_result.result_position != source + target_result.accepted_count ||
        target_result.backend.engine_submissions != (target_result.accepted_count != 0u) ||
        target_result.backend.completion_fences != (target_result.accepted_count != 0u) ||
        target_result.backend.intermediate_host_publications != 0u ||
        !target_result.pending_logits ||
        target_result.pending_logits_count != b->generation.program->vocabulary) {
        (void)b->release_attention(b->context);
        return -1;
    }
    memcpy(q->output_ids + first, s.tokens,
           (size_t)target_result.accepted_count * sizeof(int32_t));
    q->output_ids[first + target_result.accepted_count] = target_result.pending_token_id;
    r->output_count = first + target_result.produced_count;
    r->stop_token = schedule_stop(b, target_result.pending_token_id)
        ? target_result.pending_token_id : -1;
    memmove(q->logits, target_result.pending_logits,
            (size_t)b->generation.program->vocabulary * sizeof(float));
    if (salt_dpr_verify_finish_pending(d->runtime, target_result.pending_token_id,
            schedule_position(b)) != 0) {
        (void)b->release_attention(b->context);
        return -1;
    }
    if (b->observe && b->observe(b->context, &s, &target_result,
            SALT_TEXT_SCHEDULE_OBSERVE_PENDING) != 0) {
        (void)b->release_attention(b->context);
        return -1;
    }
    for (uint32_t index = first; index < r->output_count; index++) {
        int stop = schedule_stop(b, q->output_ids[index]);
        if (b->emit_token(b->context, index + 1u, q->output_ids[index], stop) != 0 ||
            (stop && index + 1u != r->output_count)) {
            (void)b->release_attention(b->context);
            return -1;
        }
    }
    if (b->release_attention(b->context) != 0) return -1;
    if (prepared > 0 && b->observe && b->observe(b->context, &s, &target_result,
            SALT_TEXT_SCHEDULE_OBSERVE_ATTENTION_READY) != 0) return -1;
    if (b->advance_chain(b->context, q->output_ids + first,
                         target_result.committed_count) != 0) return -1;
    b->stats->decode_verify_ns += s.verify_ns;
    b->stats->accepted_tokens += target_result.accepted_count;
    if (target_result.accepted_count < s.count) b->stats->rejection_count++;
    if (b->observe && b->observe(b->context, &s, &target_result, 0) != 0) return -1;
    return schedule_record_stats(b, &s, &target_result, 0);
}

int salt_text_scheduler_bind(SaltTextTokenEpochController *controller,
                             const SaltTextSchedulerBindings *b) {
    if (!controller || !b || !b->generation.program ||
        !b->generation.program->ready || !b->generation.program->kv_state ||
        b->generation.program->vocabulary > INT_MAX || !b->generation.policy ||
        !b->generation.projection ||
        (b->generation.target && !b->generation.select_proposal) ||
        !b->generation.is_stop ||
        !b->stats || !b->now_ns || !b->step_known || !b->emit_token || !b->publish)
        return -1;
    if (b->dpr && b->dpr->mode != SALT_DPR_OFF &&
        (!b->generation.target || !b->dpr->store || !b->dpr->stats ||
         !b->dpr->retained || !b->dpr->runtime ||
         !b->dpr->qa_key || !b->node_identity || !b->kv_required || !b->load_exact ||
         !b->advance_chain || !b->prepare_attention || !b->release_attention))
        return -1;
    if (salt_text_token_epoch_init(controller) != 0) return -1;
    controller->scheduler = b;
    return 0;
}

void salt_text_scheduler_abort(SaltTextTokenEpochController *controller) {
    if (!controller) return;
    if (controller->scheduler && controller->scheduler->dpr &&
        controller->scheduler->dpr->mode != SALT_DPR_OFF)
        (void)salt_dpr_runtime_fail(controller->scheduler->dpr->runtime);
    controller->schedule_active = 0;
    controller->schedule_closed = 0;
    controller->phase = SALT_TEXT_TOKEN_EPOCH_FAILED;
}

int salt_text_scheduler_run(SaltTextTokenEpochController *controller,
                            const SaltTextScheduleRequest *q,
                            SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b;
    int dpr;
    if (!controller || !(b = controller->scheduler) || controller->schedule_active ||
        controller->active || !q || !r || !q->logits || !q->scratch_logits ||
        !q->output_ids || !q->output_limit || q->output_limit > INT_MAX ||
        q->close_token < 0 || (uint32_t)q->close_token >= b->generation.program->vocabulary)
        return -1;
    dpr = b->dpr && b->dpr->mode != SALT_DPR_OFF;
    if (dpr && q->sampler.abi != SALT_SAMPLER_GREEDY_V1) return -1;
    if (!isfinite(q->top_p) || q->top_p < 0.0f || q->top_p > 1.0f ||
        (dpr && q->top_p > 0.0f && q->top_p < 1.0f)) return -1;
    if (!isfinite(q->repetition_penalty) || q->repetition_penalty < 0.0f ||
        (q->repetition_penalty > 0.0f && q->repetition_penalty < 1.0f) ||
        q->repetition_penalty > 2.0f) return -1;
    if (q->repetition_penalty > 1.0f) {
        uintptr_t raw = (uintptr_t)q->logits, scratch = (uintptr_t)q->scratch_logits;
        size_t bytes;
        if (dpr || q->sampler.abi != SALT_SAMPLER_TEMPERATURE_COUNTER_V1) return -1;
        bytes = (size_t)b->generation.program->vocabulary * sizeof(float);
        if (bytes / sizeof(float) != b->generation.program->vocabulary) return -1;
        if ((raw >= scratch ? raw - scratch : scratch - raw) < bytes) return -1;
    }
    if (salt_text_token_epoch_init(controller) != 0) return -1;
    controller->scheduler = b;
    controller->schedule_active = 1;
    memset(r, 0, sizeof *r);
    r->source_position = schedule_position(b);
    if ((q->history_ids == NULL && q->history_count != 0u) ||
        q->history_count > r->source_position)
        goto fail;
    controller->schedule_prior_kv_rows =
        r->source_position - (q->history_ids ? q->history_count : 0u);
    r->stop_token = -1;
    r->path = dpr ? 2u : (q->sampler.abi == SALT_SAMPLER_GREEDY_V1 ? 0u : 1u);
    while (r->output_count < q->output_limit && r->stop_token < 0) {
        uint32_t before = r->output_count;
        int rc;
        if (dpr) {
            if (schedule_pending(b, q, r, q->logits) < 0) goto fail;
            rc = schedule_dpr_cycle(controller, q, r);
        } else {
            rc = schedule_normal(controller, q, r);
        }
        if (rc != 0 || r->output_count <= before) goto fail;
        if (!dpr) b->stats->decode_cycles++;
    }
    controller->schedule_source = r->source_position;
    controller->schedule_count = r->output_count;
    controller->schedule_stop = r->stop_token;
    r->result_position = schedule_position(b);
    return 0;
fail:
    salt_text_scheduler_abort(controller);
    return -1;
}

int salt_text_scheduler_finish(SaltTextTokenEpochController *controller,
                               const SaltTextScheduleRequest *q,
                               SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b;
    uint32_t expected, position;
    int pending = 0, dpr;
    if (!controller || !(b = controller->scheduler) || !controller->schedule_active ||
        controller->schedule_closed || !q || !r || !r->output_count ||
        r->source_position != controller->schedule_source ||
        r->output_count != controller->schedule_count || r->stop_token != controller->schedule_stop ||
        r->output_count > q->output_limit ||
        (r->stop_token < 0 && r->output_count != q->output_limit) ||
        (r->stop_token >= 0 && q->output_ids[r->output_count - 1u] != r->stop_token) ||
        r->source_position > UINT32_MAX - r->output_count)
        return -1;
    expected = r->source_position + r->output_count;
    position = schedule_position(b);
    dpr = b->dpr && b->dpr->mode != SALT_DPR_OFF;
    if (position == expected - 1u && dpr)
        pending = schedule_pending(b, q, r, q->scratch_logits);
    if (pending < 0) goto fail;
    if (position == expected - 1u && pending == 0) {
        int32_t token = q->output_ids[r->output_count - 1u];
        if (b->step_known(b->context, token, q->scratch_logits) != 0 ||
            (dpr && b->advance_chain(b->context, &token, 1u) != 0)) goto fail;
    }
    if (schedule_position(b) != expected) goto fail;
    if (r->stop_token != q->close_token) {
        if (b->step_known(b->context, q->close_token, q->scratch_logits) != 0 ||
            (dpr && b->advance_chain(b->context, &q->close_token, 1u) != 0)) {
            r->close_failed = 1;
            goto fail;
        }
        r->synthetic_close = 1;
    }
    r->result_position = schedule_position(b);
    controller->schedule_closed = 1;
    return 0;
fail:
    salt_text_scheduler_abort(controller);
    return -1;
}

int salt_text_scheduler_publish(SaltTextTokenEpochController *controller,
                                const SaltTextScheduleResult *r) {
    const SaltTextSchedulerBindings *b;
    if (!controller || !(b = controller->scheduler) || !r ||
        !controller->schedule_active || !controller->schedule_closed ||
        controller->schedule_published || r->source_position != controller->schedule_source ||
        r->output_count != controller->schedule_count || r->stop_token != controller->schedule_stop ||
        r->result_position != schedule_position(b)) return -1;
    controller->schedule_published = 1;
    if (b->publish(b->context, r) != 0) {
        salt_text_scheduler_abort(controller);
        return -1;
    }
    controller->schedule_active = 0;
    return 0;
}

int salt_text_verify_progressive_execute(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        uint64_t transition_generation,
        uint32_t source_position,
        const int32_t *candidate_token_ids,
        uint32_t candidate_count,
        const float *parent_logits,
        const SaltTextProgressivePlan *plan,
        SaltTextProgressiveResult *result) {
    SaltTextProgressiveResult built;
    SaltTextVerifyBackendStats backend;
    const float *parent = parent_logits;
    uint32_t offset = 0, tile;
    uint64_t generation = transition_generation;
    if (result) memset(result, 0, sizeof *result);
    if (!program || !program->ready || !program->kv_state || !executor ||
        !candidate_token_ids || candidate_count == 0 ||
        candidate_count > program->maximum_candidates || !parent_logits ||
        !plan || plan->seed_candidates == 0 ||
        plan->maximum_tile_candidates < plan->seed_candidates ||
        plan->maximum_tile_candidates > program->maximum_candidates ||
        !result || transition_generation == 0 ||
        transition_generation <= program->kv_state->transition_generation ||
        source_position != program->kv_state->position ||
        source_position > program->maximum_context - candidate_count) {
        if (result) result->target.status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    memset(&built, 0, sizeof built);
    memset(&backend, 0, sizeof backend);
    for (uint32_t row = 0; row < candidate_count; row++)
        if (candidate_token_ids[row] < 0 ||
            (uint32_t)candidate_token_ids[row] >= program->vocabulary) {
            result->target.status = SALT_TEXT_VERIFY_INVALID;
            return -1;
        }
    tile = plan->seed_candidates;
    while (offset < candidate_count) {
        SaltTextTargetBlock block;
        SaltTextVerifyResult step;
        uint32_t active = candidate_count - offset;
        if (active > tile) active = tile;
        memset(&block, 0, sizeof block);
        memset(&step, 0, sizeof step);
        block.transition_generation = generation;
        block.source_position = source_position + offset;
        block.candidate_token_ids = candidate_token_ids + offset;
        block.candidate_count = active;
        block.parent_logits = parent;
        built.tile_count++;
        if (active > built.largest_tile) built.largest_tile = active;
        if (salt_text_verify_execute(
                program, executor, &block, &step) != 0 ||
            step.status != SALT_TEXT_VERIFY_COMMITTED ||
            add_backend_stats(&backend, &step.backend) != 0) {
            built.target = step;
            built.target.backend = backend;
            *result = built;
            return -1;
        }
        if (step.backend.engine_submissions != 0)
            built.submitted_tile_count++;
        built.target = step;
        offset += step.accepted_count;
        if (step.accepted_count < active || offset == candidate_count) break;
        if (!step.pending_logits ||
            step.pending_logits_count != program->vocabulary ||
            generation == UINT64_MAX) {
            built.target.status = SALT_TEXT_VERIFY_FATAL;
            built.target.backend = backend;
            *result = built;
            return -1;
        }
        parent = step.pending_logits;
        generation++;
        if (tile < plan->maximum_tile_candidates) {
            if (tile > plan->maximum_tile_candidates / 2u)
                tile = plan->maximum_tile_candidates;
            else
                tile *= 2u;
        }
    }
    built.target.accepted_count = offset;
    built.target.committed_count = offset;
    built.target.produced_count = offset + 1u;
    built.target.backend = backend;
    *result = built;
    return 0;
}

static int text_execute_authoritative(
        const SaltTextVerifyProgram *program,
        SaltTextVerifyExecutor *executor,
        uint64_t transition_generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows, SaltTextExecuteAllResult *result) {
    SaltTextExecutionView view;
    uint64_t backend_scrubbed = 0, published = 0;
    if (result) memset(result, 0, sizeof *result);
    if (!program || !program->ready || !program->descriptor ||
        !program->kv_state || !executor || !executor->ops ||
        !executor->context || !executor->ops->submit_authoritative ||
        !executor->ops->finish || !executor->ops->resolve ||
        !executor->ops->scrub || executor->program != program ||
        !executor_plan_valid(program, executor->plan) || !result ||
        transition_generation == 0 ||
        transition_generation <= program->kv_state->transition_generation ||
        !input_token_ids || input_count == 0 || output_rows > input_count ||
        input_count > program->maximum_candidates ||
        input_count > program->maximum_context ||
        (output_rows != input_count &&
         !executor->ops->submit_authoritative_output) ||
        source_position != program->kv_state->position ||
        source_position > program->maximum_context - input_count ||
        program->dispatch.engine_command_count != 1u ||
        (program->proof_state &&
         execution_kv_range_valid(program, source_position, input_count) != 0)) {
        if (result) result->status = SALT_TEXT_VERIFY_INVALID;
        return -1;
    }
    for (uint32_t row = 0; row < input_count; row++)
        if (input_token_ids[row] < 0 ||
            (uint32_t)input_token_ids[row] >= program->vocabulary) {
            result->status = SALT_TEXT_VERIFY_INVALID;
            return -1;
        }
    memset(&view, 0, sizeof view);
    if ((output_rows == input_count
            ? executor->ops->submit_authoritative(
                executor->context, program, transition_generation,
                source_position, input_token_ids, input_count)
            : executor->ops->submit_authoritative_output(
                executor->context, program, transition_generation,
                source_position, input_token_ids, input_count,
                output_rows)) != 0)
        goto fail;
    program->kv_state->transition_generation = transition_generation;
    if (executor->ops->finish(executor->context, program,
            &view, &result->backend) != 0 || !view.canonical_base ||
        view.canonical_bytes < program->layout.total_bytes ||
        result->backend.engine_submissions != 1u ||
        result->backend.completion_fences != 1u ||
        result->backend.intermediate_host_publications != 0u)
        goto fail;
    if (executor->ops->resolve(executor->context, program,
            transition_generation, source_position, input_count, input_count,
            &backend_scrubbed) != 0)
        goto fail;
    result->backend.tentative_scrub_bytes += backend_scrubbed;
    if (commit_tentative_prefix(program, &view, source_position,
            input_count, &published) != 0) {
        result->status = SALT_TEXT_VERIFY_FATAL;
        goto fail;
    }
    program->kv_state->position = source_position + input_count;
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->committed_count = input_count;
    result->result_position = program->kv_state->position;
    result->transition_generation = transition_generation;
    result->view = view;
    result->backend.final_kv_publish_bytes = published;
    return 0;
fail:
    program->kv_state->transition_generation = transition_generation;
    if (executor->ops->scrub(executor->context, program,
            transition_generation, &backend_scrubbed) != 0)
        result->status = SALT_TEXT_VERIFY_FATAL;
    else if (result->status != SALT_TEXT_VERIFY_FATAL)
        result->status = SALT_TEXT_VERIFY_EXECUTOR_FAILURE;
    result->backend.tentative_scrub_bytes += backend_scrubbed;
    result->result_position = program->kv_state->position;
    result->transition_generation = transition_generation;
    return -1;
}

int salt_text_execute_all(const SaltTextVerifyProgram *program,
                          SaltTextVerifyExecutor *executor,
                          uint64_t transition_generation,
                          uint32_t source_position,
                          const int32_t *input_token_ids,
                          uint32_t input_count,
                          SaltTextExecuteAllResult *result) {
    return text_execute_authoritative(
        program, executor, transition_generation, source_position,
        input_token_ids, input_count, input_count, result);
}

int salt_text_execute_prefill(const SaltTextVerifyProgram *program,
                              SaltTextVerifyExecutor *executor,
                              uint64_t transition_generation,
                              uint32_t source_position,
                              const int32_t *input_token_ids,
                              uint32_t input_count, uint32_t output_rows,
                              SaltTextExecuteAllResult *result) {
    if (output_rows > 1u) {
        if (result) {
            memset(result, 0, sizeof *result);
            result->status = SALT_TEXT_VERIFY_INVALID;
        }
        return -1;
    }
    return text_execute_authoritative(
        program, executor, transition_generation, source_position,
        input_token_ids, input_count, output_rows, result);
}

static int heterogeneous_ops_valid(const SaltTextGpuProgramOps *ops) {
    return ops && ops->requirements && ops->prepare && ops->begin &&
        ops->encode_cell && ops->encode_extent && ops->dependency_barrier &&
        ops->resource_request && ops->resource_resume && ops->submit &&
        ops->finish && ops->resolve && ops->scrub && ops->destroy;
}

static int heterogeneous_release_experts(
        SaltTextVerifyHeterogeneousContext *context) {
    if (!context) return -1;
    if (!context->expert_lease_active) return 0;
    if (!context->expert_resource_ops ||
        !context->expert_resource_ops->release ||
        context->expert_resource_count == 0 ||
        context->expert_resource_ops->release(
            context->expert_resource_opaque,
            context->active_expert_lease_layer,
            context->expert_resource_ids, context->expert_resource_slots,
            context->expert_resource_count) != 0)
        return -1;
    context->expert_lease_active = 0;
    context->expert_resource_count = 0;
    context->active_expert_lease_layer = UINT32_MAX;
    return 0;
}

int salt_text_verify_heterogeneous_backend_requirements(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        const SaltTextGpuProgramOps *backend_ops,
        SaltTextGpuProgramRequirements *requirements,
        size_t requirements_size) {
    SaltTextGpuProgramRequirements built;
    if (!program || !program->ready || !policy ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !heterogeneous_ops_valid(backend_ops) || !requirements ||
        requirements_size != sizeof *requirements)
        return -1;
    memset(&built, 0, sizeof built);
    if (backend_ops->requirements(program, policy, &built, sizeof built) != 0 ||
        built.backend_state_bytes == 0 || built.command_bytes == 0 ||
        built.maximum_commands < program->dispatch.cell_count ||
        (built.flags & ~SALT_TEXT_GPU_DEFER_EXPERT_RELEASE) != 0)
        return -1;
    *requirements = built;
    return 0;
}

int salt_text_verify_heterogeneous_arena_requirement(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        const SaltTextGpuProgramOps *backend_ops,
        size_t *bytes_out) {
    SaltTextGpuProgramRequirements requirement;
    size_t assignments, tasks, total = TEXT_ALIGNMENT - 1u;
    uint32_t task_capacity;
    if (!bytes_out ||
        salt_text_verify_heterogeneous_backend_requirements(program, policy,
            backend_ops, &requirement, sizeof requirement) != 0 ||
        checked_mul(program->dispatch.cell_count,
            sizeof(SaltTextExecutionAssignment), &assignments) != 0 ||
        program->maximum_candidates > UINT32_MAX -
            (TEXT_MAX_AREA_WORKERS - 1u))
        return -1;
    task_capacity = program->maximum_candidates + TEXT_MAX_AREA_WORKERS - 1u;
    if (checked_mul(task_capacity, sizeof(SaltAreaTask), &tasks) != 0 ||
        align_size(total, TEXT_ALIGNMENT, &total) != 0 ||
        checked_add(total, assignments, &total) != 0 ||
        align_size(total, TEXT_ALIGNMENT, &total) != 0 ||
        checked_add(total, tasks, &total) != 0 ||
        align_size(total, TEXT_ALIGNMENT, &total) != 0 ||
        checked_add(total, requirement.backend_state_bytes, &total) != 0 ||
        align_size(total, TEXT_ALIGNMENT, &total) != 0 ||
        checked_add(total, requirement.command_bytes, &total) != 0)
        return -1;
    *bytes_out = total;
    return 0;
}

int salt_text_verify_heterogeneous_compile(
        SaltTextVerifyHeterogeneousContext *context,
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        const SaltTextGpuProgramOps *backend_ops,
        void *startup_arena, size_t startup_arena_bytes) {
    SaltTextVerifyHeterogeneousContext built;
    SaltTextGpuProgramRequirements requirement;
    uintptr_t base, aligned;
    size_t need, cursor, assignment_bytes, task_bytes;
    uint32_t task_capacity;
    if (!context || !program || !program->ready || !policy ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !startup_arena ||
        salt_text_verify_heterogeneous_arena_requirement(program, policy,
            backend_ops, &need) != 0 || startup_arena_bytes < need ||
        salt_text_verify_heterogeneous_backend_requirements(program, policy,
            backend_ops, &requirement, sizeof requirement) != 0 ||
        checked_mul(program->dispatch.cell_count,
            sizeof(SaltTextExecutionAssignment), &assignment_bytes) != 0 ||
        program->maximum_candidates > UINT32_MAX -
            (TEXT_MAX_AREA_WORKERS - 1u))
        return -1;
    task_capacity = program->maximum_candidates + TEXT_MAX_AREA_WORKERS - 1u;
    if (checked_mul(task_capacity, sizeof(SaltAreaTask), &task_bytes) != 0)
        return -1;
    memset(&built, 0, sizeof built);
    memset(startup_arena, 0, startup_arena_bytes);
    base = (uintptr_t)startup_arena;
    aligned = (base + TEXT_ALIGNMENT - 1u) &
        ~(uintptr_t)(TEXT_ALIGNMENT - 1u);
    cursor = (size_t)(aligned - base);
    built.assignments = (SaltTextExecutionAssignment *)
        (void *)((unsigned char *)startup_arena + cursor);
    built.assignment_count = program->dispatch.cell_count;
    cursor += assignment_bytes;
    if (align_size(cursor, TEXT_ALIGNMENT, &cursor) != 0 ||
        cursor > startup_arena_bytes ||
        task_bytes > startup_arena_bytes - cursor)
        return -1;
    built.area_tasks = (SaltAreaTask *)(void *)(
        (unsigned char *)startup_arena + cursor);
    built.area_task_capacity = task_capacity;
    if (salt_area_frontier_bind(&built.area_frontier,
            built.area_tasks, built.area_task_capacity) != 0)
        return -1;
    cursor += task_bytes;
    if (align_size(cursor, TEXT_ALIGNMENT, &cursor) != 0 ||
        cursor > startup_arena_bytes ||
        requirement.backend_state_bytes > startup_arena_bytes - cursor)
        return -1;
    built.backend_state = (unsigned char *)startup_arena + cursor;
    cursor += requirement.backend_state_bytes;
    if (align_size(cursor, TEXT_ALIGNMENT, &cursor) != 0 ||
        cursor > startup_arena_bytes ||
        requirement.command_bytes > startup_arena_bytes - cursor)
        return -1;
    built.command = (unsigned char *)startup_arena + cursor;
    if (salt_text_executor_plan_compile(&built.assignment_plan, program, policy,
            built.assignments, built.assignment_count) != 0 ||
        backend_ops->prepare(program, &built.assignment_plan, &built.canonical,
            built.backend_state, requirement.backend_state_bytes,
            built.command, requirement.command_bytes) != 0 || !built.canonical)
        return -1;
    built.program = program;
    built.backend_ops = backend_ops;
    built.backend_flags = requirement.flags;
    built.startup_arena = (unsigned char *)startup_arena;
    built.startup_arena_bytes = startup_arena_bytes;
    built.ready = 1;
    built.active_expert_lease_layer = UINT32_MAX;
    *context = built;
    return 0;
}

int salt_text_verify_heterogeneous_resource_bind(
        SaltTextVerifyHeterogeneousContext *context,
        const SaltTextExpertResourceOps *ops, void *opaque) {
    if (!context || !context->ready || context->submitted ||
        !ops || !ops->acquire || !ops->release || !opaque)
        return -1;
    context->expert_resource_ops = ops;
    context->expert_resource_opaque = opaque;
    return 0;
}

static int heterogeneous_resume_resource(
        SaltTextVerifyHeterogeneousContext *context,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell) {
    uint32_t layer = UINT32_MAX;
    uint32_t resource_count = 0;
    if (!context || !program || !cell ||
        !context->expert_resource_ops)
        return -1;
    /* The NEED_RESOURCE barrier has completed all prior consumers. Release
     * before resource_request overwrites the engine-owned ID/slot arrays with
     * the next layer's selection. */
    if (heterogeneous_release_experts(context) != 0 ||
        context->backend_ops->resource_request(
            context->backend_state, context->command, &layer,
            context->expert_resource_ids,
            SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS, &resource_count) != 0 ||
        layer >= program->layer_count || resource_count == 0 ||
        resource_count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    /* NEED_RESOURCE is emitted only after the backend has completed every
     * prior consumer. Keep the previous layer's leases through expert
     * reduction/combine and release them here, immediately before their slots
     * may be reused for the next layer. */
    if (context->expert_resource_ops->acquire(
            context->expert_resource_opaque, layer,
            context->expert_resource_ids, resource_count,
            context->expert_resource_slots) != 0)
        return -1;
    context->active_expert_lease_layer = layer;
    context->expert_resource_count = resource_count;
    context->expert_lease_active = 1;
    if (context->backend_ops->resource_resume(
            context->backend_state, context->command, layer,
            context->expert_resource_ids,
            context->expert_resource_slots, resource_count) != 0 ||
        context->backend_ops->dependency_barrier(
            context->backend_state, context->command,
            cell->dependency_epoch, cell->completion_epoch) !=
                SALT_TEXT_GPU_DEPENDENCY_READY)
        return -1;
    return 0;
}

static int heterogeneous_execute_program(
        SaltTextVerifyHeterogeneousContext *context,
        const SaltTextVerifyProgram *program, uint32_t input_count,
        uint32_t *executed_cells, uint32_t *final_epoch) {
    uint32_t index = 0;
    uint32_t epoch = 0;
    if (!context || !program || !executed_cells || !final_epoch ||
        context->assignment_plan.program != program ||
        context->assignment_plan.execution_class !=
            SALT_TEXT_EXECUTION_GPU_ONLY)
        return -1;
    while (index < program->dispatch.cell_count) {
        const SaltTextExecutionAssignment *assignment =
            &context->assignments[index];
        uint32_t count = assignment->gpu_traversal_count;
        uint32_t cursor = index;
        uint32_t remaining = count;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[cursor];
        if (count == 0 || count > program->dispatch.cell_count - index ||
            cell->dependency_epoch > epoch ||
            cell->completion_epoch <= cell->dependency_epoch)
            return -1;
        if (cell->dependency_epoch > 0) {
            int barrier = context->backend_ops->dependency_barrier(
                context->backend_state, context->command,
                cell->dependency_epoch, cell->completion_epoch);
            if ((barrier == SALT_TEXT_GPU_NEED_RESOURCE &&
                 heterogeneous_resume_resource(context, program, cell) != 0) ||
                (barrier != SALT_TEXT_GPU_NEED_RESOURCE &&
                 barrier != SALT_TEXT_GPU_DEPENDENCY_READY))
                return -1;
        }
        if (cell->kind == SALT_TEXT_CELL_EXPERT_REDUCTION &&
            context->expert_lease_active &&
            (cell->layer != context->active_expert_lease_layer ||
             (!(context->backend_flags &
                    SALT_TEXT_GPU_DEFER_EXPERT_RELEASE) &&
              heterogeneous_release_experts(context) != 0)))
            return -1;
        while (remaining > 0) {
            uint32_t encoded = 0;
            int status = context->backend_ops->encode_extent(
                context->backend_state, context->command, program,
                &context->assignment_plan, cursor, remaining, input_count,
                &encoded);
            if (encoded > remaining ||
                (status == SALT_TEXT_GPU_DEPENDENCY_READY &&
                 encoded != remaining) ||
                (status == SALT_TEXT_GPU_NEED_RESOURCE &&
                 (encoded == 0 || encoded == remaining)) ||
                (status != SALT_TEXT_GPU_DEPENDENCY_READY &&
                 status != SALT_TEXT_GPU_NEED_RESOURCE))
                return -1;
            if (encoded > 0) {
                uint32_t last = cursor + encoded - 1u;
                if (program->dispatch.cells[last].completion_epoch <= epoch)
                    return -1;
                epoch = program->dispatch.cells[last].completion_epoch;
                cursor += encoded;
                remaining -= encoded;
            }
            if (status == SALT_TEXT_GPU_NEED_RESOURCE) {
                cell = &program->dispatch.cells[cursor];
                if (cell->dependency_epoch > epoch ||
                    heterogeneous_resume_resource(context, program, cell) != 0)
                    return -1;
            }
        }
        context->stats.internal_dependency_barriers +=
            assignment->gpu_traversal_barriers;
        index += count;
    }
    *executed_cells = index;
    *final_epoch = epoch;
    return 0;
}

static int heterogeneous_submit_mode(void *opaque,
        const SaltTextVerifyProgram *program, uint64_t generation,
        uint32_t source_position, const int32_t *input_token_ids,
        uint32_t input_count, int authoritative, uint32_t output_rows) {
    SaltTextVerifyHeterogeneousContext *context =
        (SaltTextVerifyHeterogeneousContext *)opaque;
    uint32_t executed_cells = 0;
    uint32_t final_epoch = 0;
    if (!context || !context->ready || context->submitted ||
        context->program != program || !input_token_ids || input_count == 0 ||
        input_count > program->maximum_candidates ||
        output_rows > input_count ||
        context->assignment_plan.execution_class !=
            SALT_TEXT_EXECUTION_GPU_ONLY ||
        (output_rows != input_count &&
         (!authoritative ||
          !context->backend_ops->begin_authoritative_output)))
        return -1;
    if ((output_rows != input_count
            ? context->backend_ops->begin_authoritative_output(
                context->backend_state, context->command, generation,
                source_position, input_token_ids, input_count, output_rows)
            : (authoritative && context->backend_ops->begin_authoritative)
                ? context->backend_ops->begin_authoritative(
                    context->backend_state, context->command, generation,
                    source_position, input_token_ids, input_count)
                : context->backend_ops->begin(
                    context->backend_state, context->command, generation,
                    source_position, input_token_ids, input_count)) != 0)
        return -1;
    memset(&context->stats, 0, sizeof context->stats);
    {
        int execute_status = heterogeneous_execute_program(
            context, program, input_count, &executed_cells, &final_epoch);
        int submit_status = -1;
        if (execute_status == 0 &&
            executed_cells == program->dispatch.cell_count &&
            final_epoch == program->dispatch.final_dependency_epoch)
            submit_status = context->backend_ops->submit(
                context->backend_state, context->command);
        if (execute_status != 0 ||
            executed_cells != program->dispatch.cell_count ||
            final_epoch != program->dispatch.final_dependency_epoch ||
            submit_status != 0) {
            const char *diag = getenv("SALT_GPU_DIAG");
            if (diag && strcmp(diag, "1") == 0)
                fprintf(stderr,
                    "text-gpu-traverse: execute=%d cells=%u/%u "
                    "epoch=%u/%u submit=%d\n",
                    execute_status, executed_cells,
                    program->dispatch.cell_count, final_epoch,
                    program->dispatch.final_dependency_epoch, submit_status);
            goto fail;
        }
    }
    context->submitted_generation = generation;
    context->submitted_source_position = source_position;
    context->submitted_candidates = input_count;
    context->stats.engine_submissions = 1u;
    context->stats.initial_transfer_bytes =
        (uint64_t)input_count * sizeof(int32_t);
    context->submitted = 1;
    context->finished = 0;
    context->resolved = 0;
    return 0;
fail:
    {
        uint64_t ignored = 0;
        (void)context->backend_ops->scrub(
            context->backend_state, context->command, &ignored);
        (void)heterogeneous_release_experts(context);
    }
    return -1;
}

static int heterogeneous_submit(void *opaque,
        const SaltTextVerifyProgram *program, uint64_t generation,
        uint32_t source_position, const int32_t *input_token_ids,
        uint32_t input_count) {
    return heterogeneous_submit_mode(opaque, program, generation,
        source_position, input_token_ids, input_count, 0, input_count);
}

static int heterogeneous_submit_authoritative(void *opaque,
        const SaltTextVerifyProgram *program, uint64_t generation,
        uint32_t source_position, const int32_t *input_token_ids,
        uint32_t input_count) {
    return heterogeneous_submit_mode(opaque, program, generation,
        source_position, input_token_ids, input_count, 1, input_count);
}

static int heterogeneous_submit_authoritative_output(
        void *opaque, const SaltTextVerifyProgram *program,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows) {
    return heterogeneous_submit_mode(opaque, program, generation,
        source_position, input_token_ids, input_count, 1, output_rows);
}

static int heterogeneous_submit_frontier(
        void *opaque, const SaltTextVerifyProgram *program,
        uint32_t source_position, const SaltTextTargetFrontier *frontier) {
    SaltTextVerifyHeterogeneousContext *context =
        (SaltTextVerifyHeterogeneousContext *)opaque;
    uint32_t executed_cells = 0;
    uint32_t final_epoch = 0;
    if (!context || !context->ready || context->submitted ||
        context->program != program || !frontier || !frontier->nodes ||
        frontier->node_count == 0 ||
        frontier->node_count > program->maximum_candidates ||
        frontier->node_count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS ||
        !context->backend_ops->begin_frontier ||
        context->assignment_plan.execution_class !=
            SALT_TEXT_EXECUTION_GPU_ONLY ||
        salt_text_target_frontier_validate(
            frontier, frontier->node_count) != 0)
        return -1;
    for (uint32_t index = 0; index < frontier->node_count; index++) {
        const SaltTextTargetNode *node = &frontier->nodes[index];
        uint32_t row = node->tentative_state_slot;
        if (row >= frontier->node_count) return -1;
        context->frontier_token_ids[row] = node->token_id;
        context->frontier_depths[row] = node->depth;
        context->frontier_parent_rows[row] =
            node->parent_index == SALT_TEXT_TARGET_NO_PARENT
                ? UINT32_MAX
                : frontier->nodes[node->parent_index].tentative_state_slot;
    }
    if (context->backend_ops->begin_frontier(
            context->backend_state, context->command,
            frontier->generation, source_position,
            context->frontier_token_ids, context->frontier_parent_rows,
            context->frontier_depths, frontier->node_count) != 0)
        return -1;
    memset(&context->stats, 0, sizeof context->stats);
    if (heterogeneous_execute_program(context, program, frontier->node_count,
            &executed_cells, &final_epoch) != 0 ||
        executed_cells != program->dispatch.cell_count ||
        final_epoch != program->dispatch.final_dependency_epoch ||
        context->backend_ops->submit(
            context->backend_state, context->command) != 0)
        goto fail;
    context->submitted_generation = frontier->generation;
    context->submitted_source_position = source_position;
    context->submitted_candidates = frontier->node_count;
    context->stats.engine_submissions = 1u;
    context->stats.initial_transfer_bytes =
        (uint64_t)frontier->node_count *
        (sizeof(int32_t) + 2u * sizeof(uint32_t));
    context->submitted = 1;
    context->finished = 0;
    context->resolved = 0;
    return 0;
fail:
    {
        uint64_t ignored = 0;
        (void)context->backend_ops->scrub(
            context->backend_state, context->command, &ignored);
        (void)heterogeneous_release_experts(context);
    }
    return -1;
}

static int heterogeneous_finish(void *opaque,
        const SaltTextVerifyProgram *program, SaltTextExecutionView *view,
        SaltTextVerifyBackendStats *stats) {
    SaltTextVerifyHeterogeneousContext *context =
        (SaltTextVerifyHeterogeneousContext *)opaque;
    SaltTextVerifyBackendStats backend;
    if (!context || !program || !view || !stats || !context->submitted ||
        context->finished || context->program != program)
        return -1;
    memset(&backend, 0, sizeof backend);
    if (context->backend_ops->finish(context->backend_state, context->command,
            view, &backend) != 0 || !view->canonical_base ||
        view->canonical_bytes < program->layout.total_bytes)
        return -1;
    backend.engine_submissions = 1u;
    backend.completion_fences = 1u;
    backend.intermediate_host_publications = 0u;
    backend.internal_dependency_barriers =
        context->stats.internal_dependency_barriers;
    backend.initial_transfer_bytes = context->stats.initial_transfer_bytes;
    context->stats = backend;
    *stats = backend;
    context->finished = 1;
    return 0;
}

static int heterogeneous_resolve(void *opaque,
        const SaltTextVerifyProgram *program, uint64_t generation,
        uint32_t source_position, uint32_t committed_count,
        uint32_t input_count, uint64_t *scrubbed_bytes) {
    SaltTextVerifyHeterogeneousContext *context =
        (SaltTextVerifyHeterogeneousContext *)opaque;
    if (!context || !program || !scrubbed_bytes || !context->submitted ||
        !context->finished || context->resolved || context->program != program ||
        generation != context->submitted_generation ||
        source_position != context->submitted_source_position ||
        input_count != context->submitted_candidates ||
        committed_count > input_count ||
        context->backend_ops->resolve(context->backend_state, context->command,
            committed_count, input_count, scrubbed_bytes) != 0)
        return -1;
    if (heterogeneous_release_experts(context) != 0) return -1;
    context->resolved = 1;
    context->submitted = 0;
    context->finished = 0;
    return 0;
}

static int heterogeneous_scrub(void *opaque,
        const SaltTextVerifyProgram *program, uint64_t generation,
        uint64_t *scrubbed_bytes) {
    SaltTextVerifyHeterogeneousContext *context =
        (SaltTextVerifyHeterogeneousContext *)opaque;
    if (!context || !program || !scrubbed_bytes || context->program != program ||
        generation == 0 || context->backend_ops->scrub(
            context->backend_state, context->command, scrubbed_bytes) != 0)
        return -1;
    if (heterogeneous_release_experts(context) != 0) return -1;
    context->submitted = 0;
    context->finished = 0;
    context->resolved = 0;
    return 0;
}

static const SaltTextVerifyExecutorOps heterogeneous_executor_ops = {
    heterogeneous_submit,
    heterogeneous_submit_frontier,
    heterogeneous_finish,
    heterogeneous_resolve,
    heterogeneous_scrub,
    heterogeneous_submit_authoritative,
    heterogeneous_submit_authoritative_output
};

int salt_text_verify_heterogeneous_executor_init(
        SaltTextVerifyExecutor *executor,
        SaltTextVerifyHeterogeneousContext *context) {
    if (!executor || !context || !context->ready || !context->program ||
        !context->canonical || !heterogeneous_ops_valid(context->backend_ops) ||
        context->assignment_plan.program != context->program ||
        context->assignment_plan.execution_class !=
            SALT_TEXT_EXECUTION_GPU_ONLY)
        return -1;
    memset(executor, 0, sizeof *executor);
    executor->program = context->program;
    executor->plan = &context->assignment_plan;
    executor->ops = &heterogeneous_executor_ops;
    executor->context = context;
    return 0;
}

int salt_text_verify_heterogeneous_destroy(
        SaltTextVerifyHeterogeneousContext *context) {
    if (!context || !context->ready || context->submitted ||
        context->expert_lease_active ||
        !heterogeneous_ops_valid(context->backend_ops) ||
        context->backend_ops->destroy(
            context->backend_state, context->command) != 0)
        return -1;
    memset(context, 0, sizeof *context);
    return 0;
}
