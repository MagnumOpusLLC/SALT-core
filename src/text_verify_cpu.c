#include "salt/text_verify.h"
#include "salt/attn.h"
#include "salt/bitmath.h"
#include "text_verify_internal.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CPU_ALIGNMENT ((size_t)16)
#define CPU_MAX_AREA_WORKERS 32u
#define CPU_MAX_FFN_FRONTIER_EXPERTS 64u

static uint64_t cpu_now_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
           (uint64_t)value.tv_nsec;
}

static void cpu_add_elapsed(uint64_t *total, uint64_t started) {
    uint64_t ended;
    if (!total || started == 0 || (ended = cpu_now_ns()) < started) return;
    if (*total <= UINT64_MAX - (ended - started))
        *total += ended - started;
}

static int cpu_checked_add(size_t a, size_t b, size_t *out) {
    if (!out || a > SIZE_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static int cpu_checked_mul(size_t a, size_t b, size_t *out) {
    if (!out || (a != 0 && b > SIZE_MAX / a)) return -1;
    *out = a * b;
    return 0;
}

static int cpu_align_size(size_t value, size_t alignment, size_t *out) {
    size_t remainder;
    if (!out || alignment == 0) return -1;
    remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return 0;
    }
    return cpu_checked_add(value, alignment - remainder, out);
}

static int cpu_graph_projection_kind(SaltTextExecutionCellKind kind) {
    switch (kind) {
    case SALT_TEXT_CELL_QUERY_PROJECTION:
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
    case SALT_TEXT_CELL_DENSE_GATE:
    case SALT_TEXT_CELL_DENSE_DOWN:
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
    case SALT_TEXT_CELL_FINAL_HEAD:
        return 1;
    default:
        return 0;
    }
}

static int cpu_graph_retained_ffn_kind(SaltTextExecutionCellKind kind) {
    return kind == SALT_TEXT_CELL_EXPERT_GATE ||
           kind == SALT_TEXT_CELL_EXPERT_UP ||
           kind == SALT_TEXT_CELL_EXPERT_DOWN;
}

static int cpu_graph_alias_kind(SaltTextExecutionCellKind kind) {
    return kind == SALT_TEXT_CELL_KEY_PROJECTION ||
           kind == SALT_TEXT_CELL_VALUE_PROJECTION ||
           kind == SALT_TEXT_CELL_DENSE_UP ||
           kind == SALT_TEXT_CELL_EXPERT_UP;
}

static int cpu_graph_node_job_capacity(
        const SaltTextVerifyProgram *program, uint32_t *capacity_out) {
    uint64_t capacity = 3u;
    if (!program || !program->ready || !capacity_out) return -1;
    for (uint32_t layer = 0; layer < program->layer_count; layer++) {
        uint64_t experts = program->layers[layer].descriptor->expert_count;
        uint64_t wave = 1u + 2u * experts;
        if (wave > capacity) capacity = wave;
    }
    if (capacity == 0 || capacity > UINT32_MAX) return -1;
    *capacity_out = (uint32_t)capacity;
    return 0;
}

static int cpu_ffn_frontier_requirement(
        const SaltTextVerifyProgram *program, size_t *seat_stride_out,
        uint32_t *seat_capacity_out, uint32_t *task_capacity_out) {
    size_t maximum = 0u;
    uint32_t topk;
    if (!program || !program->ready || !seat_stride_out ||
        !seat_capacity_out || !task_capacity_out)
        return -1;
    *seat_stride_out = 0u;
    *seat_capacity_out = 0u;
    *task_capacity_out = 0u;
    topk = program->layout.maximum_topk;
    if (topk == 0u || topk > CPU_MAX_FFN_FRONTIER_EXPERTS)
        return 0;
    for (uint32_t layer_index = 0u;
         layer_index < program->layer_count; layer_index++) {
        const SaltTextLayerExecDesc *layer =
            program->layers[layer_index].descriptor;
        if (!layer || !layer->plan)
            return -1;
        if (layer->expert_count == 0u) continue;
        if (layer->plan->top_k_experts < 1 ||
            (uint32_t)layer->plan->top_k_experts > topk)
            return -1;
        for (uint32_t expert = 0u; expert < layer->expert_count; expert++) {
            const SaltTextExpertDesc *entry = &layer->experts[expert];
            const SaltTextTensorDesc *tensors[3] = {
                &entry->gate, &entry->up, &entry->down,
            };
            for (uint32_t projection = 0u; projection < 3u; projection++) {
                const SaltTextTensorDesc *tensor = tensors[projection];
                const SaltTensorHostNodeOps *ops;
                size_t required;
                if (!tensor->host_ops || !tensor->host_ops->cpu ||
                    !(ops = tensor->host_ops->cpu->retained_node) ||
                    !ops->fixed_requirement || !ops->prepare ||
                    !ops->worker || !ops->finish)
                    return 0;
                required = ops->fixed_requirement(tensor);
                if (required == 0u || required == SIZE_MAX) return -1;
                if (required > maximum) maximum = required;
            }
        }
    }
    if (maximum == 0u ||
        cpu_align_size(maximum, CPU_ALIGNMENT, &maximum) != 0 ||
        topk > UINT32_MAX / 2u ||
        2u * topk > UINT32_MAX / CPU_MAX_AREA_WORKERS)
        return -1;
    *seat_stride_out = maximum;
    *seat_capacity_out = 2u * topk;
    *task_capacity_out = 2u * topk * CPU_MAX_AREA_WORKERS;
    return 0;
}

static int cpu_graph_compile_spans(SaltTextVerifyCpuContext *context) {
    const SaltTextVerifyProgram *program;
    uint32_t first = 0, span_count = 0;
    if (!context || !(program = context->program) || !program->ready ||
        !context->graph_spans ||
        context->graph_span_capacity < program->dispatch.cell_count)
        return -1;
    while (first < program->dispatch.cell_count) {
        const SaltTextExecutionCell *head = &program->dispatch.cells[first];
        uint32_t end = first + 1u;
        int parallel = 0, serial = 0;
        int retained_ffn = cpu_graph_retained_ffn_kind(head->kind);
        while (end < program->dispatch.cell_count &&
               program->dispatch.cells[end].dependency_epoch ==
                   head->dependency_epoch &&
               program->dispatch.cells[end].completion_epoch ==
                   head->completion_epoch &&
               cpu_graph_retained_ffn_kind(
                   program->dispatch.cells[end].kind) == retained_ffn)
            end++;
        for (uint32_t node = first; node < end; node++) {
            SaltTextExecutionCellKind kind =
                program->dispatch.cells[node].kind;
            if (cpu_graph_projection_kind(kind)) parallel = 1;
            else if (!cpu_graph_alias_kind(kind)) serial = 1;
        }
        if (parallel && serial) return -1;
        if (!parallel) {
            uint32_t scan = end;
            while (scan < program->dispatch.cell_count) {
                const SaltTextExecutionCell *next =
                    &program->dispatch.cells[scan];
                uint32_t next_end = scan + 1u;
                int next_parallel = 0, next_serial = 0;
                int next_retained_ffn =
                    cpu_graph_retained_ffn_kind(next->kind);
                while (next_end < program->dispatch.cell_count &&
                       program->dispatch.cells[next_end].dependency_epoch ==
                           next->dependency_epoch &&
                       program->dispatch.cells[next_end].completion_epoch ==
                           next->completion_epoch &&
                       cpu_graph_retained_ffn_kind(
                           program->dispatch.cells[next_end].kind) ==
                           next_retained_ffn)
                    next_end++;
                for (uint32_t node = scan; node < next_end; node++) {
                    SaltTextExecutionCellKind kind =
                        program->dispatch.cells[node].kind;
                    if (cpu_graph_projection_kind(kind)) next_parallel = 1;
                    else if (!cpu_graph_alias_kind(kind)) next_serial = 1;
                }
                if (next_parallel && next_serial) return -1;
                if (next_parallel) break;
                end = next_end;
                scan = next_end;
            }
        }
        context->graph_spans[span_count++] = (SaltTensorHostGraphSpan) {
            first, end - first, head->dependency_epoch,
            head->completion_epoch, parallel ? 1u : 0u,
        };
        first = end;
    }
    context->graph_span_count = span_count;
    return span_count > 0 ? 0 : -1;
}

static int cpu_tensor_requirement(const SaltTextTensorDesc *tensor,
                                  size_t *maximum) {
    const SaltTextTensorBackendOps *ops;
    size_t required;
    if (maximum && salt_tensor_desc_absent(tensor)) return 0;
    if (!tensor || !maximum || !tensor->host_ops ||
        !(ops = tensor->host_ops->cpu) || !ops->fixed_scratch_requirement ||
        !ops->gather_rows || !ops->matvec)
        return -1;
    required = ops->fixed_scratch_requirement(tensor);
    if (required == SIZE_MAX) return -1;
    if (required > *maximum) *maximum = required;
    return 0;
}

int salt_text_cpu_tensor_scratch_requirement(
        const SaltTextVerifyProgram *program, size_t *maximum) {
    const SaltTextModelExecDesc *model;
    if (!program || !program->ready || !maximum ||
        !(model = program->descriptor) ||
        cpu_tensor_requirement(&model->embedding, maximum) != 0 ||
        cpu_tensor_requirement(&model->final_norm, maximum) != 0 ||
        cpu_tensor_requirement(&model->output_head, maximum) != 0)
        return -1;
    for (uint32_t layer_index = 0; layer_index < program->layer_count;
         layer_index++) {
        const SaltTextLayerExecDesc *layer =
            program->layers[layer_index].descriptor;
#define REQUIRE_TENSOR(field) \
        do { if (cpu_tensor_requirement(&layer->field, maximum) != 0) \
            return -1; } while (0)
        REQUIRE_TENSOR(q);
        REQUIRE_TENSOR(k);
        if (!layer->plan->attention.shared_kv_projection)
            REQUIRE_TENSOR(v);
        REQUIRE_TENSOR(o);
        REQUIRE_TENSOR(dense_gate);
        REQUIRE_TENSOR(dense_up);
        REQUIRE_TENSOR(dense_down);
        REQUIRE_TENSOR(router);
        REQUIRE_TENSOR(pre_attention_norm);
        REQUIRE_TENSOR(q_norm);
        REQUIRE_TENSOR(k_norm);
        REQUIRE_TENSOR(post_attention_norm);
        REQUIRE_TENSOR(pre_ffn_norm_1);
        REQUIRE_TENSOR(pre_ffn_norm_2);
        REQUIRE_TENSOR(post_ffn_norm_1);
        REQUIRE_TENSOR(post_ffn_norm_2);
        REQUIRE_TENSOR(post_ffn_norm);
        REQUIRE_TENSOR(router_scale);
        REQUIRE_TENSOR(per_expert_scale);
        if (layer->plan->final_layer_scale)
            REQUIRE_TENSOR(layer_scalar);
#undef REQUIRE_TENSOR
        for (uint32_t expert = 0; expert < layer->expert_count; expert++) {
            const SaltTextExpertDesc *entry = &layer->experts[expert];
            if (cpu_tensor_requirement(&entry->gate, maximum) != 0 ||
                cpu_tensor_requirement(&entry->up, maximum) != 0 ||
                cpu_tensor_requirement(&entry->down, maximum) != 0)
                return -1;
        }
    }
    return 0;
}

int salt_text_verify_cpu_arena_requirement(
        const SaltTextVerifyProgram *program, size_t *bytes_out) {
    size_t scratch = 0, ffn_seat_stride = 0, ffn_seat_bytes = 0;
    size_t assignment_bytes, task_bytes, node_job_bytes, span_bytes, total;
    uint32_t task_capacity, node_job_capacity, ffn_seat_capacity = 0;
    uint32_t ffn_task_capacity = 0;
    if (!bytes_out || !program || !program->ready ||
        salt_text_cpu_tensor_scratch_requirement(program, &scratch) != 0 ||
        program->maximum_candidates > UINT32_MAX -
            (CPU_MAX_AREA_WORKERS - 1u) ||
        cpu_checked_mul((size_t)program->dispatch.cell_count,
            sizeof(SaltTextExecutionAssignment), &assignment_bytes) != 0 ||
        cpu_graph_node_job_capacity(program, &node_job_capacity) != 0 ||
        cpu_ffn_frontier_requirement(program, &ffn_seat_stride,
            &ffn_seat_capacity, &ffn_task_capacity) != 0 ||
        cpu_checked_mul((size_t)ffn_seat_capacity, ffn_seat_stride,
            &ffn_seat_bytes) != 0)
        return -1;
    task_capacity = program->maximum_candidates + CPU_MAX_AREA_WORKERS - 1u;
    if (ffn_task_capacity > task_capacity) task_capacity = ffn_task_capacity;
    if (cpu_checked_mul((size_t)task_capacity,
            sizeof(SaltAreaTask), &task_bytes) != 0)
        return -1;
    if (cpu_checked_mul((size_t)node_job_capacity,
            sizeof(SaltTextTensorHostBatch), &node_job_bytes) != 0 ||
        cpu_checked_mul((size_t)program->dispatch.cell_count,
            sizeof(SaltTensorHostGraphSpan), &span_bytes) != 0)
        return -1;
    if (cpu_checked_add(CPU_ALIGNMENT - 1u,
                        program->layout.total_bytes, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, scratch, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, ffn_seat_bytes, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, assignment_bytes, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, task_bytes, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, node_job_bytes, &total) != 0 ||
        cpu_align_size(total, CPU_ALIGNMENT, &total) != 0 ||
        cpu_checked_add(total, span_bytes, &total) != 0)
        return -1;
    *bytes_out = total;
    return 0;
}

int salt_text_verify_cpu_compile(SaltTextVerifyCpuContext *context,
                                 const SaltTextVerifyProgram *program,
                                 void *arena, size_t arena_bytes) {
    SaltTextVerifyCpuContext built;
    SaltTextDispatchPolicy policy;
    uintptr_t address, aligned_address;
    size_t need, padding, scratch = 0, scratch_offset;
    size_t ffn_seat_offset, ffn_seat_stride = 0, ffn_seat_bytes = 0;
    size_t assignment_offset, assignment_bytes, task_offset, task_bytes;
    size_t node_job_offset, node_job_bytes, span_offset, span_bytes;
    uint32_t task_capacity, node_job_capacity, ffn_seat_capacity = 0;
    uint32_t ffn_task_capacity = 0;
    if (!context || !arena ||
        salt_text_verify_cpu_arena_requirement(program, &need) != 0 ||
        arena_bytes < need ||
        salt_text_cpu_tensor_scratch_requirement(program, &scratch) != 0 ||
        cpu_graph_node_job_capacity(program, &node_job_capacity) != 0 ||
        cpu_ffn_frontier_requirement(program, &ffn_seat_stride,
            &ffn_seat_capacity, &ffn_task_capacity) != 0 ||
        cpu_checked_mul((size_t)ffn_seat_capacity, ffn_seat_stride,
            &ffn_seat_bytes) != 0)
        return -1;
    address = (uintptr_t)arena;
    aligned_address = (address + (uintptr_t)(CPU_ALIGNMENT - 1u)) &
        ~(uintptr_t)(CPU_ALIGNMENT - 1u);
    padding = (size_t)(aligned_address - address);
    if (padding > arena_bytes ||
        cpu_align_size(program->layout.total_bytes,
                       CPU_ALIGNMENT, &scratch_offset) != 0 ||
        cpu_checked_add(scratch_offset, scratch, &ffn_seat_offset) != 0 ||
        cpu_align_size(ffn_seat_offset,
                       CPU_ALIGNMENT, &ffn_seat_offset) != 0 ||
        cpu_checked_add(ffn_seat_offset, ffn_seat_bytes,
                        &assignment_offset) != 0 ||
        cpu_align_size(assignment_offset,
                       CPU_ALIGNMENT, &assignment_offset) != 0 ||
        cpu_checked_mul((size_t)program->dispatch.cell_count,
            sizeof(SaltTextExecutionAssignment), &assignment_bytes) != 0 ||
        program->maximum_candidates > UINT32_MAX -
            (CPU_MAX_AREA_WORKERS - 1u))
        return -1;
    task_capacity = program->maximum_candidates + CPU_MAX_AREA_WORKERS - 1u;
    if (ffn_task_capacity > task_capacity) task_capacity = ffn_task_capacity;
    if (cpu_checked_add(assignment_offset, assignment_bytes,
            &task_offset) != 0 ||
        cpu_align_size(task_offset, CPU_ALIGNMENT, &task_offset) != 0 ||
        cpu_checked_mul((size_t)task_capacity,
            sizeof(SaltAreaTask), &task_bytes) != 0 ||
        cpu_checked_add(task_offset, task_bytes, &node_job_offset) != 0 ||
        cpu_align_size(node_job_offset, CPU_ALIGNMENT, &node_job_offset) != 0 ||
        cpu_checked_mul((size_t)node_job_capacity,
            sizeof(SaltTextTensorHostBatch), &node_job_bytes) != 0 ||
        cpu_checked_add(node_job_offset, node_job_bytes, &span_offset) != 0 ||
        cpu_align_size(span_offset, CPU_ALIGNMENT, &span_offset) != 0 ||
        cpu_checked_mul((size_t)program->dispatch.cell_count,
            sizeof(SaltTensorHostGraphSpan), &span_bytes) != 0)
        return -1;
    if (ffn_seat_offset > arena_bytes - padding ||
        ffn_seat_bytes > arena_bytes - padding - ffn_seat_offset ||
        assignment_offset > arena_bytes - padding ||
        assignment_bytes > arena_bytes - padding - assignment_offset ||
        task_offset > arena_bytes - padding ||
        task_bytes > arena_bytes - padding - task_offset ||
        node_job_offset > arena_bytes - padding ||
        node_job_bytes > arena_bytes - padding - node_job_offset ||
        span_offset > arena_bytes - padding ||
        span_bytes > arena_bytes - padding - span_offset)
        return -1;
    memset(&built, 0, sizeof built);
    built.program = program;
    built.arena = (unsigned char *)(void *)aligned_address;
    built.arena_bytes = arena_bytes - padding;
    built.tensor_scratch_offset = scratch_offset;
    built.tensor_scratch = built.arena + scratch_offset;
    built.tensor_scratch_bytes = scratch;
    built.ffn_frontier_seats = built.arena + ffn_seat_offset;
    built.ffn_frontier_seat_stride = ffn_seat_stride;
    built.ffn_frontier_seat_capacity = ffn_seat_capacity;
    built.assignments = (SaltTextExecutionAssignment *)(void *)(
        built.arena + assignment_offset);
    built.assignment_count = program->dispatch.cell_count;
    built.area_tasks = (SaltAreaTask *)(void *)(built.arena + task_offset);
    built.area_task_capacity = task_capacity;
    built.node_jobs = (SaltTextTensorHostBatch *)(void *)(
        built.arena + node_job_offset);
    built.node_job_capacity = node_job_capacity;
    built.graph_spans = (SaltTensorHostGraphSpan *)(void *)(
        built.arena + span_offset);
    built.graph_span_capacity = program->dispatch.cell_count;
    built.frontier_expert_layer = UINT32_MAX;
    if (salt_area_frontier_bind(&built.area_frontier,
            built.area_tasks, built.area_task_capacity) != 0)
        return -1;
    memset(&policy, 0, sizeof policy);
    policy.execution_class = SALT_TEXT_EXECUTION_CPU_ONLY;
    if (salt_text_executor_plan_compile(
            &built.assignment_plan, program, &policy,
            built.assignments, built.assignment_count) != 0)
        return -1;
    if (cpu_graph_compile_spans(&built) != 0) return -1;
    built.ready = 1;
    /* Reserve canonical execution storage at startup without touching the
     * request-sized pages. cpu_submit() zeroes the complete canonical layout
     * before every operation, while the immutable assignment plan above is
     * already fully initialized in its own tail range. */
    *context = built;
    return 0;
}

int salt_text_verify_cpu_parallel_bind(
        SaltTextVerifyCpuContext *context,
        int (*parallel_run)(void *parallel_context, int active_workers,
                            void (*worker)(int worker, void *task), void *task),
        void *parallel_context, float *parallel_scores,
        size_t parallel_score_stride, uint32_t parallel_workers) {
    if (!context || !context->ready || !context->program || !parallel_run ||
        !parallel_context || !parallel_scores || parallel_workers == 0 ||
        parallel_workers > 32u ||
        parallel_score_stride < context->program->maximum_context ||
        context->submitted || context->parallel_run ||
        context->parallel_context || context->parallel_scores ||
        context->parallel_score_stride != 0 || context->parallel_workers != 0)
        return -1;
    context->parallel_run = parallel_run;
    context->parallel_context = parallel_context;
    context->parallel_scores = parallel_scores;
    context->parallel_score_stride = parallel_score_stride;
    context->parallel_workers = parallel_workers;
    return 0;
}

int salt_text_verify_cpu_scope_bind(
        SaltTextVerifyCpuContext *context,
        int (*parallel_scope)(void *, int (*)(void *), void *)) {
    if (!context || !context->ready || !context->parallel_run ||
        !context->parallel_context || !parallel_scope || context->parallel_scope ||
        context->collective_enabled || context->submitted || context->submitted_generation != 0u)
        return -1;
    context->parallel_scope = parallel_scope;
    return 0;
}

int salt_text_verify_cpu_collective_bind(SaltTextVerifyCpuContext *context) {
    if (!context || !context->ready || !context->parallel_run ||
        !context->parallel_context || !context->parallel_workers ||
        context->collective_enabled || context->parallel_scope ||
        context->submitted || context->submitted_generation != 0u ||
        !context->program || !context->program->target_policy ||
        context->program->target_policy->cpu_graph)
        return -1;
    context->collective_enabled = 1;
    return 0;
}

static float *cpu_floats(SaltTextVerifyCpuContext *context, size_t offset) {
    return (float *)(void *)(context->arena + offset);
}

static int32_t *cpu_i32(SaltTextVerifyCpuContext *context, size_t offset) {
    return (int32_t *)(void *)(context->arena + offset);
}

static void *cpu_tensor_scratch(SaltTextVerifyCpuContext *context) {
    return context->tensor_scratch;
}

static int cpu_record_area_plan(SaltTextVerifyCpuContext *context,
                                const SaltAreaScanPlan *plan) {
    if (!context || !plan || plan->active_workers == 0 ||
        plan->output_row_tiles == 0 || plan->matrix_parallel > 1u ||
        plan->output_row_owned > 1u || plan->candidate_row_split > 1u ||
        (plan->output_row_owned && plan->candidate_row_split) ||
        (plan->candidate_row_split && plan->active_rows < 2u) ||
        (plan->matrix_parallel != (plan->active_rows > 1u ? 1u : 0u)) ||
        plan->candidate_output_tiles < plan->output_row_tiles ||
        (plan->matrix_parallel == 0u &&
         plan->candidate_output_tiles != plan->output_row_tiles))
        return -1;
    if (plan->mode == SALT_AREA_SCAN_M1)
        context->stats.area_m1_dispatches++;
    else if (plan->mode == SALT_AREA_SCAN_MK)
        context->stats.area_mk_dispatches++;
    else if (plan->mode == SALT_AREA_SCAN_MN)
        context->stats.area_mn_dispatches++;
    else
        return -1;
    if (plan->active_workers > context->stats.area_peak_active_workers)
        context->stats.area_peak_active_workers = plan->active_workers;
    if (UINT64_MAX - context->stats.area_output_row_tiles <
            plan->output_row_tiles)
        return -1;
    context->stats.area_output_row_tiles += plan->output_row_tiles;
    if (plan->matrix_parallel) {
        if (context->stats.area_matrix_parallel_dispatches == UINT32_MAX ||
            UINT64_MAX - context->stats.area_candidate_output_tiles <
                plan->candidate_output_tiles)
            return -1;
        context->stats.area_matrix_parallel_dispatches++;
        context->stats.area_candidate_output_tiles +=
            plan->candidate_output_tiles;
    }
    return 0;
}

static int cpu_gather(SaltTextVerifyCpuContext *context,
                      const SaltTextTensorDesc *tensor,
                      const int32_t *rows, uint32_t row_count, float *output) {
    return tensor->host_ops->cpu->gather_rows(
        tensor, rows, row_count, output, cpu_tensor_scratch(context),
        context->tensor_scratch_bytes);
}

static int cpu_gather_row_zero(SaltTextVerifyCpuContext *context,
                               const SaltTextTensorDesc *tensor,
                               float *output) {
    const int32_t row = 0;
    return cpu_gather(context, tensor, &row, 1u, output);
}

static int cpu_project_ordinary(SaltTextVerifyCpuContext *context,
                                const SaltTextTensorDesc *tensor,
                                const float *inputs, uint32_t row_count,
                                float *outputs) {
    const SaltTextTensorBackendOps *ops = tensor->host_ops->cpu;
    if (ops->matvec_batch)
        return ops->matvec_batch(tensor, inputs, row_count, outputs,
            cpu_tensor_scratch(context), context->tensor_scratch_bytes);
    if (!ops->matvec) return -1;
    for (uint32_t row = 0; row < row_count; row++)
        if (ops->matvec(tensor,
                inputs + (size_t)row * tensor->cols,
                outputs + (size_t)row * tensor->rows,
                cpu_tensor_scratch(context),
                context->tensor_scratch_bytes) != 0)
            return -1;
    return 0;
}

static int cpu_project(SaltTextVerifyCpuContext *context,
                       const SaltTextTensorDesc *tensor,
                       const float *inputs, uint32_t row_count,
                       float *outputs) {
    const SaltTextTensorBackendOps *ops = tensor->host_ops->cpu;
    int rc = 0;
    if (ops->matvec_area_batch) {
        SaltTextTensorHostBatch job = {
            tensor, inputs, row_count, outputs,
        };
        SaltAreaScanPlan plan;
        memset(&plan, 0, sizeof plan);
        rc = ops->matvec_area_batch(&job, 1u,
            context->submitted_candidates, &plan,
            cpu_tensor_scratch(context), context->tensor_scratch_bytes);
        if (rc == 0) {
            if (cpu_record_area_plan(context, &plan) != 0) rc = -1;
        } else if (rc > 0) {
            rc = cpu_project_ordinary(
                context, tensor, inputs, row_count, outputs);
        }
    } else {
        rc = cpu_project_ordinary(
            context, tensor, inputs, row_count, outputs);
    }
    if (rc == 0) {
        context->stats.projection_dispatches++;
        context->stats.cpu_matrix_pool_phases++;
    }
    return rc;
}

/* Return 1 when one backend epoch executed all jobs, 0 when the backend does
 * not expose multi-projection execution, and -1 on a backend failure. */
static int cpu_project_multi(SaltTextVerifyCpuContext *context,
                             const SaltTextTensorHostBatch *jobs,
                             uint32_t job_count) {
    const SaltTextTensorBackendOps *ops;
    if (!context || !jobs || job_count < 2 || !jobs[0].tensor ||
        !jobs[0].tensor->host_ops ||
        !(ops = jobs[0].tensor->host_ops->cpu) ||
        (!ops->matvec_multi_batch && !ops->matvec_area_batch))
        return 0;
    for (uint32_t job = 0; job < job_count; job++)
        if (!jobs[job].tensor || !jobs[job].inputs || !jobs[job].outputs ||
            jobs[job].row_count == 0 || !jobs[job].tensor->host_ops ||
            jobs[job].tensor->host_ops->cpu != ops)
            return -1;
    if (ops->matvec_area_batch) {
        SaltAreaScanPlan plan;
        int area_rc;
        memset(&plan, 0, sizeof plan);
        area_rc = ops->matvec_area_batch(jobs, job_count,
                context->submitted_candidates, &plan,
                cpu_tensor_scratch(context), context->tensor_scratch_bytes);
        if (area_rc < 0) return -1;
        if (area_rc == 0) {
            if (cpu_record_area_plan(context, &plan) != 0) return -1;
            context->stats.projection_dispatches++;
            context->stats.cpu_matrix_pool_phases++;
            return 1;
        }
    }
    if (!ops->matvec_multi_batch) return 0;
    if (ops->matvec_multi_batch(jobs, job_count,
            cpu_tensor_scratch(context), context->tensor_scratch_bytes) != 0)
        return -1;
    context->stats.projection_dispatches++;
    context->stats.cpu_matrix_pool_phases++;
    return 1;
}

static int cpu_rmsnorm(float *output, const float *input,
                       const float *weight, uint32_t width,
                       float epsilon, int with_scale) {
    float sum = 0.0f, inverse;
    if (!output || !input || width == 0 ||
        (with_scale && !weight))
        return -1;
    for (uint32_t column = 0; column < width; column++) {
        if (!isfinite(input[column]) ||
            (with_scale && !isfinite(weight[column])))
            return -1;
        sum += input[column] * input[column];
    }
    inverse = salt_powf(sum / (float)width + epsilon, -0.5f);
    if (!isfinite(inverse)) return -1;
    for (uint32_t column = 0; column < width; column++) {
        float value = input[column] * inverse;
        if (with_scale) value = value * weight[column];
        if (!isfinite(value)) return -1;
        output[column] = value;
    }
    return 0;
}

static int cpu_rmsnorm_rows(float *output, const float *input,
                            const float *weight, uint32_t rows,
                            uint32_t width, float epsilon) {
    for (uint32_t row = 0; row < rows; row++)
        if (cpu_rmsnorm(output + (size_t)row * width,
                input + (size_t)row * width, weight, width,
                epsilon, 1) != 0)
            return -1;
    return 0;
}

static float cpu_round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
        bits &= UINT32_C(0xffff0000);
    }
    memcpy(&value, &bits, sizeof value);
    return value;
}

static float cpu_bf16_add(float left, float right) {
    return cpu_round_bf16(cpu_round_bf16(left) + cpu_round_bf16(right));
}

static float cpu_bf16_mul(float left, float right) {
    return cpu_round_bf16(cpu_round_bf16(left) * cpu_round_bf16(right));
}


static int cpu_rope(float *head, const SaltAttentionDesc *attention,
                    uint32_t position) {
    uint32_t half, pairs;
    if (head && attention && attention->rope_kind == SALT_ROPE_NONE)
        return attention->rope_dim == 0 ? 0 : -1;
    if (!head || !attention || attention->head_dim < 2 ||
        attention->rope_dim < 2 || attention->rope_base_dim < 2 ||
        !(attention->rope_theta > 0.0))
        return -1;
    half = (uint32_t)attention->head_dim / 2u;
    pairs = (uint32_t)attention->rope_dim / 2u;
    if (attention->rope_kind == SALT_ROPE_PARTIAL_F32) {
        /* Split only the rotary prefix; the unrotated suffix is untouched. */
        for (uint32_t pair = 0; pair < pairs; pair++) {
            float exponent = (float)(2u * pair) / (float)attention->rope_base_dim;
            float angle = (float)position / salt_powf((float)attention->rope_theta, exponent);
            float cosine = salt_cosf(angle), sine = salt_sinf(angle);
            float left = head[pair], right = head[pairs + pair];
            head[pair] = left * cosine - right * sine;
            head[pairs + pair] = right * cosine + left * sine;
            if (!isfinite(head[pair]) || !isfinite(head[pairs + pair])) return -1;
        }
        return 0;
    }
    for (uint32_t column = 0; column < (uint32_t)attention->head_dim;
         column++) {
        if (!isfinite(head[column])) return -1;
        head[column] = cpu_round_bf16(head[column]);
    }
    for (uint32_t pair = 0; pair < pairs; pair++) {
        float exponent = (float)(2u * pair) /
            (float)attention->rope_base_dim;
        float angle = (float)position /
            salt_powf((float)attention->rope_theta, exponent);
        float cosine = cpu_round_bf16(salt_cosf(angle));
        float sine = cpu_round_bf16(salt_sinf(angle));
        float left = head[pair];
        float right = head[half + pair];
        float lc = cpu_bf16_mul(left, cosine);
        float rs = cpu_bf16_mul(right, sine);
        float rc = cpu_bf16_mul(right, cosine);
        float ls = cpu_bf16_mul(left, sine);
        head[pair] = cpu_bf16_add(lc, -rs);
        head[half + pair] = cpu_bf16_add(rc, ls);
        if (!isfinite(head[pair]) || !isfinite(head[half + pair]))
            return -1;
    }
    return 0;
}

static float cpu_activation(float value, SaltActivationKind activation) {
    if (activation == SALT_ACT_GELU_TANH) {
        const float coefficient = 0.7978845608028654f;
        return 0.5f * value * (1.0f + salt_tanhf(
            coefficient * (value + 0.044715f * value * value * value)));
    }
    return value / (1.0f + salt_expf(-value));
}

static float cpu_gate_up(float gate, float up,
                         SaltActivationKind activation, float limit) {
    if (activation == SALT_ACT_SILU_CLAMPED) {
        if (gate > limit) gate = limit;
        if (up > limit) up = limit;
        if (up < -limit) up = -limit;
    }
    return cpu_activation(gate, activation) * up;
}


static int cpu_row_position(const SaltTextVerifyCpuContext *context,
                            uint32_t source, uint32_t row,
                            uint32_t *position_out) {
    uint32_t delta = row;
    if (!context || !position_out || row >= context->submitted_candidates)
        return -1;
    if (context->submitted_frontier) {
        const uint32_t *depths = (const uint32_t *)(const void *)(
            context->arena + context->program->layout.target_depths);
        delta = depths[row];
    }
    if (delta > UINT32_MAX - source) return -1;
    *position_out = source + delta;
    return 0;
}

typedef struct CpuAttentionRowResolver {
    const SaltTextKvReadView *key_view;
    const SaltTextKvReadView *value_view;
    uint32_t current_row;
    uint32_t first_position;
} CpuAttentionRowResolver;

static const float *cpu_attention_resolve_row(
        const void *opaque, int value_row, int position,
        int kv_head, int head_dim) {
    const CpuAttentionRowResolver *resolver =
        (const CpuAttentionRowResolver *)opaque;
    const SaltTextKvReadView *view;
    const float *row;
    uint32_t absolute;
    if (!resolver || position < 0 || kv_head < 0 || head_dim < 1 ||
        (uint32_t)position > UINT32_MAX - resolver->first_position)
        return NULL;
    absolute = resolver->first_position + (uint32_t)position;
    view = value_row ? resolver->value_view : resolver->key_view;
    row = salt_text_kv_read_view_row(view, resolver->current_row, absolute);
    return row ? row + (size_t)(uint32_t)kv_head * (uint32_t)head_dim : NULL;
}

static int cpu_attention_linear_spans(
        const SaltTextKvReadView *key_view,
        const SaltTextKvReadView *value_view,
        uint32_t current_row, uint32_t first_position, uint32_t row_count,
        uint32_t width, SaltAttnKvSpan *spans, int *span_count) {
    const float *prior_key = NULL, *prior_value = NULL;
    int count = 0;
    if (!key_view || !value_view || !spans || !span_count ||
        row_count == 0 || width == 0 ||
        first_position > UINT32_MAX - (row_count - 1u))
        return -1;
    for (uint32_t row = 0; row < row_count; row++) {
        uint32_t position = first_position + row;
        const float *key = salt_text_kv_read_view_row(
            key_view, current_row, position);
        const float *value = salt_text_kv_read_view_row(
            value_view, current_row, position);
        if (!key || !value) return -1;
        if (count > 0 && key == prior_key + width &&
            value == prior_value + width) {
            spans[count - 1].count++;
        } else {
            if (count >= SALT_ATTN_HEAD_MAX_SPANS) return 1;
            spans[count++] = (SaltAttnKvSpan) {
                .keys = key, .values = value, .count = 1,
            };
        }
        prior_key = key;
        prior_value = value;
    }
    *span_count = count;
    return 0;
}

static int cpu_attention_head(
        SaltTextVerifyCpuContext *context,
        const SaltTextCompiledLayer *compiled,
        const SaltTextKvReadView *key_view,
        const SaltTextKvReadView *value_view,
        const SaltAttnKvSpan *spans, int span_count,
        uint32_t source, uint32_t row, uint32_t head,
        float *scores) {
    const SaltTextLayerExecDesc *layer = compiled->descriptor;
    const SaltAttentionDesc *attention = &layer->plan->attention;
    const SaltTextCanonicalLayout *layout = &context->program->layout;
    float *queries = cpu_floats(context, layout->queries);
    float *destination = cpu_floats(context, layout->attention_output);

    uint32_t groups = (uint32_t)attention->n_heads /
        (uint32_t)attention->n_kv_heads;
    uint32_t position, first = 0;
    uint32_t kv_head = head / groups;
    const float *query;
    float *head_output;
    CpuAttentionRowResolver resolver;
    SaltAttnHeadFold fold;
    if (!scores || head >= (uint32_t)attention->n_heads ||
        cpu_row_position(context, source, row, &position) != 0)
        return -1;
    if (attention->kind == SALT_ATTN_SLIDING &&
        position + 1u > (uint32_t)attention->window)
        first = position + 1u - (uint32_t)attention->window;
    query = queries + (size_t)row * compiled->query_width +
        (size_t)head * (uint32_t)attention->head_dim;
    head_output = destination + (size_t)row * compiled->query_width +
        (size_t)head * (uint32_t)attention->head_dim;
    memset(&resolver, 0, sizeof resolver);
    resolver.key_view = key_view;
    resolver.value_view = value_view;
    resolver.current_row = row;
    resolver.first_position = first;
    memset(&fold, 0, sizeof fold);
    fold.query = query;
    fold.output = head_output;
    fold.scores = scores;
    fold.row_count = (int)(position - first + 1u);
    fold.head_dim = attention->head_dim;
    fold.kv_head = (int)kv_head;
    fold.score_scale = attention->score_scale;
    if (spans && span_count > 0) {
        fold.kv_row_stride = (int)compiled->kv_width;
        fold.spans = spans;
        fold.span_count = span_count;
        fold.span_accumulate = salt_attn_weighted_value_accumulate;
    } else {
        fold.row_resolver = cpu_attention_resolve_row;
        fold.row_context = &resolver;
    }
    return salt_attn_head_fold(&fold);
}

typedef struct CpuAttentionBodyTask {
    SaltTextVerifyCpuContext *context;
    const SaltTextCompiledLayer *compiled;
    SaltTextKvReadView key_view;
    SaltTextKvReadView value_view;
    uint32_t source;
    uint32_t first_row;
    uint32_t work_count;
    uint32_t workers;
    SaltAttnKvSpan spans[SALT_ATTN_HEAD_MAX_SPANS];
    int span_count;
    int failed[32];
} CpuAttentionBodyTask;

static void cpu_attention_body_worker(int worker, void *opaque) {
    CpuAttentionBodyTask *task = (CpuAttentionBodyTask *)opaque;
    const SaltAttentionDesc *attention =
        &task->compiled->descriptor->plan->attention;
    uint32_t first = (uint32_t)((uint64_t)task->work_count *
        (uint32_t)worker / task->workers);
    uint32_t end = (uint32_t)((uint64_t)task->work_count *
        ((uint32_t)worker + 1u) / task->workers);
    float *scores = task->context->parallel_scores +
        (size_t)(uint32_t)worker * task->context->parallel_score_stride;
    for (uint32_t item = first; item < end; item++) {
        uint32_t row = task->first_row +
            item / (uint32_t)attention->n_heads;
        uint32_t head = item % (uint32_t)attention->n_heads;
        if (cpu_attention_head(task->context, task->compiled,
                &task->key_view, &task->value_view,
                task->span_count ? task->spans : NULL, task->span_count,
                task->source, row, head, scores) != 0) {
            task->failed[worker] = 1;
            return;
        }
    }
}

static int cpu_attention_cell(
        SaltTextVerifyCpuContext *context,
        const SaltTextCompiledLayer *compiled,
        const SaltTextExecutionCell *cell,
        uint32_t source, uint32_t first, uint32_t count) {
    const SaltTextLayerExecDesc *layer = compiled->descriptor;
    const SaltAttentionDesc *attention = &layer->plan->attention;
    const SaltTextCanonicalLayout *layout = &context->program->layout;
    float *state_a = cpu_floats(context, layout->state_a);
    float *normalized = cpu_floats(context, layout->normalized);
    float *queries = cpu_floats(context, layout->queries);
    float *keys = cpu_floats(context, layout->keys);
    float *values = cpu_floats(context, layout->values);
    float *attention_output = cpu_floats(context, layout->attention_output);
    float *branch = cpu_floats(context, layout->branch);
    float *scores = cpu_floats(context, layout->attention_scores);
    float *tensor_row = cpu_floats(context, layout->tensor_row);
    float *tentative_keys = cpu_floats(context,
        compiled->tentative_key_offset);
    float *tentative_values = cpu_floats(context,
        compiled->tentative_value_offset);
    float *destination = cpu_floats(context, cell->destination_offset);
    switch (cell->kind) {
    case SALT_TEXT_CELL_PRE_ATTENTION_NORM:
        if (cpu_gather_row_zero(context, &layer->pre_attention_norm,
                tensor_row) != 0)
            return -1;
        return cpu_rmsnorm_rows(
            destination + (size_t)first * context->program->hidden,
            state_a + (size_t)first * context->program->hidden,
            tensor_row, count,
            context->program->hidden,
            context->program->descriptor->norm_epsilon);
    case SALT_TEXT_CELL_QUERY_PROJECTION: {
        SaltTextTensorHostBatch jobs[3];
        uint32_t job_count = attention->shared_kv_projection ? 2u : 3u;
        int multi;
        jobs[0] = (SaltTextTensorHostBatch) {
            &layer->q, normalized + (size_t)first * layer->q.cols,
            count, queries + (size_t)first * layer->q.rows,
        };
        jobs[1] = (SaltTextTensorHostBatch) {
            &layer->k, normalized + (size_t)first * layer->k.cols,
            count, keys + (size_t)first * layer->k.rows,
        };
        if (!attention->shared_kv_projection)
            jobs[2] = (SaltTextTensorHostBatch) {
                &layer->v, normalized + (size_t)first * layer->v.cols,
                count, values + (size_t)first * layer->v.rows,
            };
        multi = context->coalescing_enabled
            ? cpu_project_multi(context, jobs, job_count) : 0;
        if (multi < 0) return -1;
        if (multi > 0) {
            context->coalesced_attention_layer = cell->layer;
            context->stats.cpu_qkv_waves++;
            return 0;
        }
        return cpu_project(context, &layer->q,
            normalized + (size_t)first * layer->q.cols, count,
            destination + (size_t)first * layer->q.rows);
    }
    case SALT_TEXT_CELL_KEY_PROJECTION:
        if (context->coalescing_enabled &&
            context->coalesced_attention_layer == cell->layer) {
            if (attention->shared_kv_projection)
                context->coalesced_attention_layer = UINT32_MAX;
            return 0;
        }
        return cpu_project(context, &layer->k,
            normalized + (size_t)first * layer->k.cols, count,
            destination + (size_t)first * layer->k.rows);
    case SALT_TEXT_CELL_VALUE_PROJECTION:
        if (attention->shared_kv_projection) return -1;
        if (context->coalescing_enabled &&
            context->coalesced_attention_layer == cell->layer) {
            context->coalesced_attention_layer = UINT32_MAX;
            return 0;
        }
        return cpu_project(context, &layer->v,
            normalized + (size_t)first * layer->v.cols, count,
            destination + (size_t)first * layer->v.rows);
    case SALT_TEXT_CELL_ATTENTION_TRANSFORM:
        if (destination != tentative_keys) return -1;
        if (attention->shared_kv_projection)
            memcpy(values + (size_t)first * compiled->kv_width,
                   keys + (size_t)first * compiled->kv_width,
                   (size_t)count * compiled->kv_width * sizeof(float));
        if (cpu_gather_row_zero(context, &layer->q_norm, tensor_row) != 0)
            return -1;
        for (uint32_t row = first; row < first + count; row++)
            for (uint32_t head = 0; head < (uint32_t)attention->n_heads;
                 head++) {
                float *query = queries +
                    (size_t)row * compiled->query_width +
                    (size_t)head * (uint32_t)attention->head_dim;
                uint32_t position;
                if (cpu_row_position(context, source, row, &position) != 0 ||
                    cpu_rmsnorm(query, query, tensor_row,
                        (uint32_t)attention->head_dim,
                        context->program->descriptor->norm_epsilon, 1) != 0 ||
                    cpu_rope(query, attention, position) != 0)
                    return -1;
            }
        if (cpu_gather_row_zero(context, &layer->k_norm, tensor_row) != 0)
            return -1;
        for (uint32_t row = first; row < first + count; row++)
            for (uint32_t head = 0;
                 head < (uint32_t)attention->n_kv_heads; head++) {
                float *key = keys + (size_t)row * compiled->kv_width +
                    (size_t)head * (uint32_t)attention->head_dim;
                float *value = values + (size_t)row * compiled->kv_width +
                    (size_t)head * (uint32_t)attention->head_dim;
                uint32_t position;
                if (cpu_row_position(context, source, row, &position) != 0 ||
                    cpu_rmsnorm(key, key, tensor_row,
                        (uint32_t)attention->head_dim,
                        context->program->descriptor->norm_epsilon, 1) != 0 ||
                    cpu_rope(key, attention, position) != 0 ||
                    (!attention->raw_values && cpu_rmsnorm(value, value, NULL,
                        (uint32_t)attention->head_dim,
                        context->program->descriptor->norm_epsilon, 0) != 0))
                    return -1;
            }
        memcpy(tentative_keys + (size_t)first * compiled->kv_width,
               keys + (size_t)first * compiled->kv_width,
               (size_t)count * compiled->kv_width * sizeof(float));
        memcpy(tentative_values + (size_t)first * compiled->kv_width,
               values + (size_t)first * compiled->kv_width,
               (size_t)count * compiled->kv_width * sizeof(float));
        return 0;
    case SALT_TEXT_CELL_ATTENTION_BODY: {
        SaltTextKvReadView key_view, value_view;
        SaltAttnKvSpan linear_spans[SALT_ATTN_HEAD_MAX_SPANS];
        const uint32_t *parents = NULL, *depths = NULL;
        size_t tentative_capacity;
        uint32_t work;
        int linear_span_count = 0;
        if (count > UINT32_MAX / (uint32_t)attention->n_heads ||
            context->program->maximum_candidates >
                SIZE_MAX / compiled->kv_width)
            return -1;
        tentative_capacity =
            (size_t)context->program->maximum_candidates * compiled->kv_width;
        if (context->submitted_frontier) {
            parents = (const uint32_t *)(const void *)(
                context->arena + context->program->layout.target_parent_rows);
            depths = (const uint32_t *)(const void *)(
                context->arena + context->program->layout.target_depths);
        }
        if (salt_text_kv_read_view_init(&key_view, &layer->kv->keys,
                tentative_keys, tentative_capacity, compiled->kv_width,
                context->submitted_candidates, source,
                parents, depths, compiled->kv_width) != 0 ||
            salt_text_kv_read_view_init(&value_view, &layer->kv->values,
                tentative_values, tentative_capacity, compiled->kv_width,
                context->submitted_candidates, source,
                parents, depths, compiled->kv_width) != 0)
            return -1;
        if (!context->submitted_frontier &&
            context->submitted_candidates == 1u && first == 0u && count == 1u) {
            uint32_t position, first_position = 0u;
            int span_rc;
            if (cpu_row_position(context, source, 0u, &position) != 0)
                return -1;
            if (attention->kind == SALT_ATTN_SLIDING &&
                position + 1u > (uint32_t)attention->window)
                first_position = position + 1u - (uint32_t)attention->window;
            memset(linear_spans, 0, sizeof linear_spans);
            span_rc = cpu_attention_linear_spans(
                &key_view, &value_view, 0u, first_position,
                position - first_position + 1u, compiled->kv_width,
                linear_spans, &linear_span_count);
            if (span_rc < 0) return -1;
            if (span_rc > 0) linear_span_count = 0;
        }
        work = count * (uint32_t)attention->n_heads;
        /* Graph serial spans run on the caller only after the preceding
         * projection phase has completed. graph_workers is the retained
         * projection width, not an active nested pool call; preserve the
         * existing disjoint row/head attention block in both execution modes. */
        if (context->parallel_run &&
            context->parallel_workers > 1u && work >= context->parallel_workers) {
            CpuAttentionBodyTask task;
            uint32_t active = context->parallel_workers;
            if (active > work) active = work;
            memset(&task, 0, sizeof task);
            task.context = context;
            task.compiled = compiled;
            task.key_view = key_view;
            task.value_view = value_view;
            task.source = source;
            task.first_row = first;
            task.work_count = work;
            task.workers = active;
            task.span_count = linear_span_count;
            if (linear_span_count > 0)
                memcpy(task.spans, linear_spans,
                    (size_t)linear_span_count * sizeof(*linear_spans));
            if (context->parallel_run(context->parallel_context, (int)active,
                    cpu_attention_body_worker, &task) != 0)
                return -1;
            for (uint32_t worker = 0; worker < active; worker++)
                if (task.failed[worker]) return -1;
            return 0;
        }
        for (uint32_t row = first; row < first + count; row++)
            for (uint32_t head = 0;
                 head < (uint32_t)attention->n_heads; head++)
                if (cpu_attention_head(
                        context, compiled, &key_view, &value_view,
                        linear_span_count ? linear_spans : NULL,
                        linear_span_count,
                        source, row, head, scores) != 0)
                    return -1;
        return 0;
    }
    case SALT_TEXT_CELL_OUTPUT_PROJECTION:
        return cpu_project(context, &layer->o,
            attention_output + (size_t)first * layer->o.cols, count,
            destination + (size_t)first * layer->o.rows);
    case SALT_TEXT_CELL_ATTENTION_COMBINE:
        if (layer->plan->residual_postnorm && cpu_gather_row_zero(context,
                &layer->post_attention_norm, tensor_row) != 0)
            return -1;
        for (uint32_t row = first; row < first + count; row++) {
            float *output = destination +
                (size_t)row * context->program->hidden;
            const float *residual = state_a +
                (size_t)row * context->program->hidden;
            float *branch_row = branch +
                (size_t)row * context->program->hidden;
            if (layer->plan->residual_postnorm && cpu_rmsnorm(branch_row, branch_row, tensor_row,
                    context->program->hidden,
                    context->program->descriptor->norm_epsilon, 1) != 0)
                return -1;
            for (uint32_t column = 0;
                 column < context->program->hidden; column++) {
                output[column] = residual[column] + branch_row[column];
                if (!isfinite(output[column])) return -1;
            }
        }
        return 0;
    default:
        return -1;
    }
}

static int cpu_router_input(float *output, const float *input,
                            const float *scale, uint32_t hidden,
                            float epsilon) {
    float sum = 0.0f, inverse, root;
    for (uint32_t column = 0; column < hidden; column++) {
        if (!isfinite(input[column]) || !isfinite(scale[column])) return -1;
        sum += input[column] * input[column];
    }
    inverse = salt_powf(sum / (float)hidden + epsilon, -0.5f);
    root = 1.0f / salt_sqrtf((float)hidden);
    if (!isfinite(inverse) || !isfinite(root)) return -1;
    for (uint32_t column = 0; column < hidden; column++) {
        output[column] = ((input[column] * inverse) * scale[column]) * root;
        if (!isfinite(output[column])) return -1;
    }
    return 0;
}

static int cpu_router_topk(const float *logits, const float *scales,
                           uint32_t experts, uint32_t topk,
                           int32_t *selected, float *weights, int rank_order) {
    float maximum = -INFINITY, all_sum = 0.0f, selected_sum = 0.0f;
    for (uint32_t expert = 0; expert < experts; expert++) {
        if (!isfinite(logits[expert]) || (scales && !isfinite(scales[expert]))) return -1;
        if (logits[expert] > maximum) maximum = logits[expert];
    }
    for (uint32_t rank = 0; rank < topk; rank++) {
        selected[rank] = -1;
        weights[rank] = -INFINITY;
    }
    for (uint32_t expert = 0; expert < experts; expert++) {
        int rank = (int)topk - 1;
        float score = logits[expert];
        while (rank >= 0 &&
               (selected[rank] < 0 || score > weights[rank]))
            rank--;
        if (rank < (int)topk - 1) {
            for (int move = (int)topk - 2; move > rank; move--) {
                selected[move + 1] = selected[move];
                weights[move + 1] = weights[move];
            }
            selected[rank + 1] = (int32_t)expert;
            weights[rank + 1] = score;
        }
    }
    for (uint32_t expert = 0; expert < experts; expert++)
        all_sum += salt_expf(logits[expert] - maximum);
    if (!(all_sum > 0.0f) || !isfinite(all_sum)) return -1;
    for (uint32_t rank = 0; rank < topk; rank++) {
        weights[rank] = salt_expf(weights[rank] - maximum) / all_sum;
        selected_sum += weights[rank];
    }
    if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) return -1;
    for (uint32_t rank = 0; rank < topk; rank++)
        weights[rank] = (weights[rank] / selected_sum) *
            (scales ? scales[(uint32_t)selected[rank]] : 1.0f);
    if (rank_order) return 0;
    for (uint32_t left = 0; left < topk; left++)
        for (uint32_t right = left + 1u; right < topk; right++)
            if (selected[right] < selected[left]) {
                int32_t id = selected[left];
                float weight = weights[left];
                selected[left] = selected[right];
                weights[left] = weights[right];
                selected[right] = id;
                weights[right] = weight;
            }
    return 0;
}

static int cpu_graph_build_jobs(
    SaltTextVerifyCpuContext *context,
    const SaltTextExecutionCell *cell, uint32_t rows,
    uint32_t first, uint32_t count, uint32_t *job_count);
static int cpu_ffn_frontier_execute(
    SaltTextVerifyCpuContext *context,
    const SaltTextExecutionCell *gate_cell, uint32_t first, uint32_t count);

static int cpu_ffn_cell(
        SaltTextVerifyCpuContext *context,
        const SaltTextCompiledLayer *compiled,
        const SaltTextExecutionCell *cell, uint32_t candidate_count,
        uint32_t first, uint32_t count) {
    const SaltTextLayerExecDesc *layer = compiled->descriptor;
    const SaltTextLayerPlan *plan = layer->plan;
    const SaltTextCanonicalLayout *layout = &context->program->layout;
    const uint32_t hidden = context->program->hidden;
    const uint32_t dense_width = (uint32_t)plan->dense_intermediate;
    const uint32_t experts = (uint32_t)plan->n_experts;
    const uint32_t topk = (uint32_t)plan->top_k_experts;
    const uint32_t expert_width = (uint32_t)plan->expert_intermediate;
    if ((cell->unit == SALT_TEXT_EXECUTION_ROWS &&
         (first > candidate_count || count > candidate_count - first)) ||
        (cell->unit == SALT_TEXT_EXECUTION_JOBS &&
         (candidate_count > UINT32_MAX / topk ||
          first > candidate_count * topk ||
          count > candidate_count * topk - first)))
        return -1;
    float *state_b = cpu_floats(context, layout->state_b);
    float *normalized = cpu_floats(context, layout->normalized);
    float *dense_gate = cpu_floats(context, layout->dense_gate);
    float *dense_up = cpu_floats(context, layout->dense_up);
    float *dense_chain = cpu_floats(context, layout->dense_chain);
    float *dense_output = cpu_floats(context, layout->dense_output);
    float *router_input = cpu_floats(context, layout->router_input);
    float *router_logits = cpu_floats(context, layout->router_logits);
    int32_t *selected = cpu_i32(context, layout->selected_experts);
    float *weights = cpu_floats(context, layout->selected_weights);
    int32_t *grouped_to_canonical = cpu_i32(context,
        layout->grouped_to_canonical);
    int32_t *canonical_to_grouped = cpu_i32(context,
        layout->canonical_to_grouped);
    float *routed_input = cpu_floats(context, layout->routed_input);
    float *routed_gate = cpu_floats(context, layout->routed_gate);
    float *routed_up = cpu_floats(context, layout->routed_up);
    float *routed_chain = cpu_floats(context, layout->routed_chain);
    float *routed_jobs = cpu_floats(context, layout->routed_output_jobs);
    float *routed_output = cpu_floats(context, layout->routed_output);
    float *combine_a = cpu_floats(context, layout->combine_a);
    float *combine_b = cpu_floats(context, layout->combine_b);
    float *tensor_row = cpu_floats(context, layout->tensor_row);
    float *destination = cpu_floats(context, cell->destination_offset);
    switch (cell->kind) {
    case SALT_TEXT_CELL_DENSE_NORM:
        if (cpu_gather_row_zero(context, &layer->pre_ffn_norm_1,
                                tensor_row) != 0)
            return -1;
        return cpu_rmsnorm_rows(
            destination + (size_t)first * hidden,
            state_b + (size_t)first * hidden, tensor_row,
            count, hidden, context->program->descriptor->norm_epsilon);
    case SALT_TEXT_CELL_DENSE_GATE: {
        SaltTextTensorHostBatch jobs[2] = {
            {
                &layer->dense_gate,
                normalized + (size_t)first * hidden,
                count,
                dense_gate + (size_t)first * dense_width,
            },
            {
                &layer->dense_up,
                normalized + (size_t)first * hidden,
                count,
                dense_up + (size_t)first * dense_width,
            },
        };
        int multi = context->coalescing_enabled
            ? cpu_project_multi(context, jobs, 2u) : 0;
        if (multi < 0) return -1;
        if (multi > 0) {
            context->coalesced_dense_layer = cell->layer;
            context->stats.cpu_dense_gate_up_waves++;
            return 0;
        }
        return cpu_project(context, &layer->dense_gate,
            normalized + (size_t)first * hidden, count,
            destination + (size_t)first * dense_width);
    }
    case SALT_TEXT_CELL_DENSE_UP:
        if (context->coalescing_enabled &&
            context->coalesced_dense_layer == cell->layer) {
            context->coalesced_dense_layer = UINT32_MAX;
            return 0;
        }
        return cpu_project(context, &layer->dense_up,
            normalized + (size_t)first * hidden, count,
            destination + (size_t)first * dense_width);
    case SALT_TEXT_CELL_DENSE_ACTIVATION:
        for (uint32_t index = first * dense_width;
             index < (first + count) * dense_width; index++) {
            destination[index] = cpu_gate_up(dense_gate[index], dense_up[index],
                plan->activation, plan->activation_limit);
            if (!isfinite(destination[index])) return -1;
        }
        return 0;
    case SALT_TEXT_CELL_DENSE_DOWN:
        return cpu_project(context, &layer->dense_down,
            dense_chain + (size_t)first * dense_width, count,
            destination + (size_t)first * hidden);
    case SALT_TEXT_CELL_ROUTER_INPUT:
        if (!plan->router_rmsnorm) {
            if (cpu_gather_row_zero(context, &layer->pre_ffn_norm_2, tensor_row) != 0)
                return -1;
            return cpu_rmsnorm_rows(destination + (size_t)first * hidden,
                state_b + (size_t)first * hidden, tensor_row, count, hidden,
                context->program->descriptor->norm_epsilon);
        }
        if (cpu_gather_row_zero(context, &layer->router_scale,
                                tensor_row) != 0)
            return -1;
        for (uint32_t row = first; row < first + count; row++)
            if (cpu_router_input(destination + (size_t)row * hidden,
                    state_b + (size_t)row * hidden, tensor_row, hidden,
                    context->program->descriptor->norm_epsilon) != 0)
                return -1;
        return 0;
    case SALT_TEXT_CELL_ROUTER_PROJECTION:
        return cpu_project(context, &layer->router,
            router_input + (size_t)first * hidden, count,
            destination + (size_t)first * experts);
    case SALT_TEXT_CELL_ROUTED_NORM:
        if (cpu_gather_row_zero(context, &layer->pre_ffn_norm_2,
                                tensor_row) != 0)
            return -1;
        return cpu_rmsnorm_rows(
            destination + (size_t)first * hidden,
            state_b + (size_t)first * hidden, tensor_row,
            count, hidden, context->program->descriptor->norm_epsilon);
    case SALT_TEXT_CELL_ROUTER_TOPK: {
        int32_t *expert_next = (int32_t *)(void *)tensor_row;
        if (destination != (float *)(void *)selected ||
            (!plan->router_rank_order && cpu_gather_row_zero(context,
                &layer->per_expert_scale, tensor_row) != 0))
            return -1;
        for (uint32_t row = first; row < first + count; row++)
            if (cpu_router_topk(router_logits + (size_t)row * experts,
                    plan->router_rank_order ? NULL : tensor_row, experts, topk,
                    selected + (size_t)row * topk,
                    weights + (size_t)row * topk, plan->router_rank_order) != 0)
                return -1;
        if (first == 0 && count == candidate_count) {
            uint32_t jobs = candidate_count * topk;
            uint32_t next = 0;
            memset(expert_next, 0, (size_t)experts * sizeof(*expert_next));
            for (uint32_t canonical = 0; canonical < jobs; canonical++) {
                int32_t expert = selected[canonical];
                if (expert < 0 || (uint32_t)expert >= experts ||
                    expert_next[expert] == INT32_MAX)
                    return -1;
                expert_next[expert]++;
            }
            for (uint32_t expert = 0; expert < experts; expert++) {
                uint32_t population = (uint32_t)expert_next[expert];
                expert_next[expert] = (int32_t)next;
                next += population;
            }
            if (next != jobs) return -1;
            for (uint32_t canonical = 0; canonical < jobs; canonical++) {
                int32_t expert = selected[canonical];
                uint32_t grouped = (uint32_t)expert_next[expert]++;
                if (grouped >= jobs) return -1;
                grouped_to_canonical[grouped] = (int32_t)canonical;
            }
            for (uint32_t grouped = 0; grouped < jobs; grouped++) {
                uint32_t canonical = (uint32_t)grouped_to_canonical[grouped];
                uint32_t row = canonical / topk;
                if (canonical >= jobs || row >= candidate_count)
                    return -1;
                canonical_to_grouped[canonical] = (int32_t)grouped;
                memcpy(routed_input + (size_t)grouped * hidden,
                       combine_a + (size_t)row * hidden,
                       (size_t)hidden * sizeof(float));
            }
        } else {
            for (uint32_t row = first; row < first + count; row++)
            for (uint32_t rank = 0; rank < topk; rank++) {
                uint32_t canonical = row * topk + rank;
                grouped_to_canonical[canonical] = (int32_t)canonical;
                canonical_to_grouped[canonical] = (int32_t)canonical;
                memcpy(routed_input + (size_t)canonical * hidden,
                       combine_a + (size_t)row * hidden,
                       (size_t)hidden * sizeof(float));
            }
        }
        return 0;
    }
    case SALT_TEXT_CELL_EXPERT_GATE:
    case SALT_TEXT_CELL_EXPERT_UP:
    case SALT_TEXT_CELL_EXPERT_DOWN: {
        uint32_t job = first;
        int gate_up_multi = 0;
        if (context->ffn_frontier_seat_capacity == 0u &&
            cell->kind == SALT_TEXT_CELL_EXPERT_GATE &&
            candidate_count == 1u && first == 0u && count == topk) {
            int frontier = cpu_ffn_frontier_execute(
                context, cell, first, count);
            if (frontier <= 0) return frontier;
        }
        if (context->frontier_expert_layer == cell->layer) {
            if (cell->kind == SALT_TEXT_CELL_EXPERT_UP) return 0;
            if (cell->kind == SALT_TEXT_CELL_EXPERT_DOWN) {
                context->frontier_expert_layer = UINT32_MAX;
                return 0;
            }
        }
        if (getenv("SALT_GPU_DIAG") &&
            cell->kind == SALT_TEXT_CELL_EXPERT_UP)
            fprintf(stderr,
                "SALT_TEXT_EXPERT_COALESCE phase=up layer=%u marker=%u enabled=%d\n",
                cell->layer, context->coalesced_expert_layer,
                context->coalescing_enabled);
        if (context->coalescing_enabled &&
            cell->kind == SALT_TEXT_CELL_EXPERT_UP &&
            context->coalesced_expert_layer == cell->layer) {
            context->coalesced_expert_layer = UINT32_MAX;
            return 0;
        }
        if (context->coalescing_enabled &&
            (cell->kind == SALT_TEXT_CELL_EXPERT_GATE ||
             cell->kind == SALT_TEXT_CELL_EXPERT_DOWN)) {
            SaltTextTensorHostBatch multi_jobs[256];
            uint32_t scan = first, multi_count = 0;
            while (scan < first + count) {
                uint32_t canonical =
                    (uint32_t)grouped_to_canonical[scan];
                int32_t expert;
                uint32_t last = scan + 1u;
                if (canonical >= candidate_count * topk) return -1;
                expert = selected[canonical];
                if (expert < 0 || (uint32_t)expert >= experts) return -1;
                while (last < first + count) {
                    uint32_t next_canonical =
                        (uint32_t)grouped_to_canonical[last];
                    if (next_canonical >= candidate_count * topk ||
                        selected[next_canonical] != expert)
                        break;
                    last++;
                }
                if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
                    if (multi_count > 254u) return -1;
                    multi_jobs[multi_count++] = (SaltTextTensorHostBatch) {
                        &layer->experts[expert].gate,
                        routed_input + (size_t)scan * hidden,
                        last - scan,
                        routed_gate + (size_t)scan * expert_width,
                    };
                    multi_jobs[multi_count++] = (SaltTextTensorHostBatch) {
                        &layer->experts[expert].up,
                        routed_input + (size_t)scan * hidden,
                        last - scan,
                        routed_up + (size_t)scan * expert_width,
                    };
                } else {
                    if (multi_count >= 256u) return -1;
                    multi_jobs[multi_count++] = (SaltTextTensorHostBatch) {
                        &layer->experts[expert].down,
                        routed_chain + (size_t)scan * expert_width,
                        last - scan,
                        routed_jobs + (size_t)scan * hidden,
                    };
                }
                scan = last;
            }
            if (multi_count >= 2u) {
                int multi = cpu_project_multi(
                    context, multi_jobs, multi_count);
                if (multi < 0) return -1;
                if (multi > 0) {
                    if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
                        context->coalesced_expert_layer = cell->layer;
                        if (getenv("SALT_GPU_DIAG"))
                            fprintf(stderr,
                                "SALT_TEXT_EXPERT_COALESCE phase=gate layer=%u "
                                "marker=%u jobs=%u\n",
                                cell->layer, context->coalesced_expert_layer,
                                multi_count);
                        context->stats.expert_gate_up_dispatches++;
                        context->stats.cpu_expert_gate_up_waves++;
                    } else {
                        context->stats.expert_down_dispatches++;
                        context->stats.cpu_expert_down_waves++;
                    }
                    return 0;
                }
            }
        }
        while (job < first + count) {
            uint32_t canonical = (uint32_t)grouped_to_canonical[job];
            int32_t expert;
            uint32_t last = job + 1u;
            const SaltTextTensorDesc *tensor;
            const float *inputs;
            uint32_t input_width, output_width;
            if (canonical >= candidate_count * topk)
                return -1;
            expert = selected[canonical];
            if (expert < 0 || (uint32_t)expert >= experts)
                return -1;
            while (last < first + count) {
                uint32_t next_canonical =
                    (uint32_t)grouped_to_canonical[last];
                if (next_canonical >= candidate_count * topk ||
                    selected[next_canonical] != expert)
                    break;
                last++;
            }
            if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
                SaltTextTensorHostBatch jobs[2] = {
                    {
                        &layer->experts[expert].gate,
                        routed_input + (size_t)job * hidden,
                        last - job,
                        routed_gate + (size_t)job * expert_width,
                    },
                    {
                        &layer->experts[expert].up,
                        routed_input + (size_t)job * hidden,
                        last - job,
                        routed_up + (size_t)job * expert_width,
                    },
                };
                if (gate_up_multi != 0) {
                    int multi = cpu_project_multi(context, jobs, 2u);
                    if (multi < 0) return -1;
                    if (gate_up_multi < 0) gate_up_multi = multi > 0 ? 1 : 0;
                    if ((multi > 0) != (gate_up_multi > 0)) return -1;
                    if (multi > 0) {
                        context->stats.expert_gate_up_dispatches++;
                        job = last;
                        continue;
                    }
                }
                tensor = &layer->experts[expert].gate;
                inputs = routed_input + (size_t)job * hidden;
                input_width = hidden;
                output_width = expert_width;
            } else if (cell->kind == SALT_TEXT_CELL_EXPERT_UP) {
                tensor = &layer->experts[expert].up;
                inputs = routed_input + (size_t)job * hidden;
                input_width = hidden;
                output_width = expert_width;
            } else {
                tensor = &layer->experts[expert].down;
                inputs = routed_chain + (size_t)job * expert_width;
                input_width = expert_width;
                output_width = hidden;
            }
            if (tensor->cols != input_width || tensor->rows != output_width ||
                cpu_project(context, tensor, inputs, last - job,
                    destination + (size_t)job * output_width) != 0)
                return -1;
            if (cell->kind == SALT_TEXT_CELL_EXPERT_DOWN)
                context->stats.expert_down_dispatches++;
            else
                context->stats.expert_gate_up_dispatches++;
            job = last;
        }
        if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE && gate_up_multi > 0)
            context->coalesced_expert_layer = cell->layer;
        return 0;
    }
    case SALT_TEXT_CELL_EXPERT_ACTIVATION:
        if (context->frontier_expert_layer == cell->layer) return 0;
        for (uint32_t index = first * expert_width;
             index < (first + count) * expert_width; index++) {
            destination[index] = cpu_gate_up(routed_gate[index], routed_up[index],
                plan->activation, plan->activation_limit);
            if (!isfinite(destination[index])) return -1;
        }
        return 0;
    case SALT_TEXT_CELL_EXPERT_REDUCTION:
        memset(destination + (size_t)first * hidden, 0,
               (size_t)count * hidden * sizeof(float));
        for (uint32_t token = first; token < first + count; token++)
            for (uint32_t rank = 0; rank < topk; rank++) {
                uint32_t canonical = token * topk + rank;
                uint32_t grouped =
                    (uint32_t)canonical_to_grouped[canonical];
                const float *job_output;
                if (grouped >= candidate_count * topk) return -1;
                job_output = routed_jobs + (size_t)grouped * hidden;
                for (uint32_t column = 0; column < hidden; column++)
                    destination[(size_t)token * hidden + column] +=
                        weights[canonical] * job_output[column];
            }
        return 0;
    case SALT_TEXT_CELL_FFN_COMBINE: {
        float scalar = 1.0f;
        if (!plan->residual_postnorm) {
            if (plan->dense_intermediate || plan->final_layer_scale) return -1;
            for (size_t i = (size_t)first * hidden; i < (size_t)(first + count) * hidden; i++) {
                destination[i] = state_b[i] + routed_output[i];
                if (!isfinite(destination[i])) return -1;
            }
            return 0;
        }
        if (cpu_gather_row_zero(context, &layer->post_ffn_norm_1,
                                tensor_row) != 0 ||
            cpu_rmsnorm_rows(combine_a + (size_t)first * hidden,
                dense_output + (size_t)first * hidden, tensor_row,
                count, hidden,
                context->program->descriptor->norm_epsilon) != 0 ||
            cpu_gather_row_zero(context, &layer->post_ffn_norm_2,
                                tensor_row) != 0 ||
            cpu_rmsnorm_rows(combine_b + (size_t)first * hidden,
                routed_output + (size_t)first * hidden, tensor_row,
                count, hidden,
                context->program->descriptor->norm_epsilon) != 0)
            return -1;
        for (uint32_t index = first * hidden;
             index < (first + count) * hidden; index++) {
            combine_a[index] += combine_b[index];
            if (!isfinite(combine_a[index])) return -1;
        }
        if (cpu_gather_row_zero(context, &layer->post_ffn_norm,
                                tensor_row) != 0 ||
            cpu_rmsnorm_rows(combine_b + (size_t)first * hidden,
                combine_a + (size_t)first * hidden, tensor_row,
                count, hidden,
                context->program->descriptor->norm_epsilon) != 0)
            return -1;
        if (plan->final_layer_scale) {
            if (cpu_gather_row_zero(context, &layer->layer_scalar,
                                    tensor_row) != 0)
                return -1;
            scalar = tensor_row[0];
        }
        if (!isfinite(scalar)) return -1;
        for (uint32_t index = first * hidden;
             index < (first + count) * hidden; index++) {
            destination[index] = (state_b[index] + combine_b[index]) * scalar;
            if (!isfinite(destination[index])) return -1;
        }
        return 0;
    }
    default:
        return -1;
    }
}

static int cpu_global_cell(
        SaltTextVerifyCpuContext *context,
        const SaltTextExecutionCell *cell,
        const int32_t *tokens, uint32_t first, uint32_t count) {
    const SaltTextVerifyProgram *program = context->program;
    float *state_a = cpu_floats(context, program->layout.state_a);
    float *final_state = cpu_floats(context, program->layout.final_state);
    float *position_logits = cpu_floats(context,
        program->layout.position_logits);
    float *tensor_row = cpu_floats(context, program->layout.tensor_row);
    float *destination = cpu_floats(context, cell->destination_offset);
    switch (cell->kind) {
    case SALT_TEXT_CELL_EMBEDDING:
        if (cpu_gather(context, &program->descriptor->embedding,
                tokens + first, count,
                destination + (size_t)first * program->hidden) != 0)
            return -1;
        for (uint32_t index = first * program->hidden;
             index < (first + count) * program->hidden; index++) {
            destination[index] *= program->descriptor->embedding_scale;
            if (!isfinite(destination[index])) return -1;
        }
        return 0;
    case SALT_TEXT_CELL_FINAL_NORM:
        if (cpu_gather_row_zero(context, &program->descriptor->final_norm,
                                tensor_row) != 0)
            return -1;
        return cpu_rmsnorm_rows(
            destination + (size_t)first * program->hidden,
            state_a + (size_t)first * program->hidden, tensor_row,
            count, program->hidden, program->descriptor->norm_epsilon);
    case SALT_TEXT_CELL_FINAL_HEAD:
        return cpu_project(context, &program->descriptor->output_head,
            final_state + (size_t)first * program->hidden, count,
            destination + (size_t)first * program->vocabulary);
    case SALT_TEXT_CELL_LOGIT_SOFTCAP: {
        float cap = program->layers[program->layer_count - 1u].descriptor->
            plan->logit_softcap;
        if (destination != position_logits || !(cap >= 0.0f) || !isfinite(cap))
            return -1;
        for (uint32_t index = first * program->vocabulary;
             index < (first + count) * program->vocabulary; index++) {
            if (!isfinite(destination[index])) return -1;
            if (cap > 0.0f) destination[index] = cap * salt_tanhf(destination[index] / cap);
        }
        return 0;
    }
    default:
        return -1;
    }
}

static int cpu_dispatch_cell(
        SaltTextVerifyCpuContext *context,
        const SaltTextExecutionCell *cell,
        uint32_t source_position,
        const int32_t *candidate_token_ids,
        uint32_t candidate_count,
        SaltTextExecutionSlice assigned_complete) {
    const SaltTextVerifyProgram *program = context->program;
    if (cell->layer == UINT32_MAX)
        return cpu_global_cell(context, cell, candidate_token_ids,
            assigned_complete.first, assigned_complete.count);
    if (cell->layer < program->layer_count &&
        cell->kind >= SALT_TEXT_CELL_PRE_ATTENTION_NORM &&
        cell->kind <= SALT_TEXT_CELL_ATTENTION_COMBINE)
        return cpu_attention_cell(context, &program->layers[cell->layer], cell,
            source_position, assigned_complete.first,
            assigned_complete.count);
    if (cell->layer < program->layer_count)
        return cpu_ffn_cell(context, &program->layers[cell->layer], cell,
            candidate_count, assigned_complete.first,
            assigned_complete.count);
    return -1;
}

int salt_text_cpu_execute_cell_range(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        uint32_t source_position,
        const int32_t *candidate_token_ids,
        uint32_t candidate_count,
        SaltTextExecutionSlice assigned_complete,
        unsigned char *canonical_host_arena,
        size_t canonical_host_bytes,
        void *tensor_scratch,
        size_t tensor_scratch_bytes,
        SaltTextVerifyBackendStats *stats) {
    SaltTextVerifyCpuContext local;
    uint32_t actual;
    int rc;
    if (!program || !program->ready || !cell || !candidate_token_ids ||
        candidate_count == 0 || candidate_count > program->maximum_candidates ||
        !canonical_host_arena || canonical_host_bytes < program->layout.total_bytes ||
        (tensor_scratch_bytes != 0 && !tensor_scratch) || !stats ||
        cell->destination_stride == 0 ||
        cell->destination_offset >= program->layout.total_bytes)
        return -1;
    actual = candidate_count;
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
    if (actual > cell->logical_capacity ||
        assigned_complete.first > actual ||
        assigned_complete.count > actual - assigned_complete.first ||
        assigned_complete.first + assigned_complete.count >
            (program->layout.total_bytes - cell->destination_offset) /
                cell->destination_stride)
        return -1;
    if (assigned_complete.count == 0) return 0;
    memset(&local, 0, sizeof local);
    local.program = program;
    local.arena = canonical_host_arena;
    local.arena_bytes = canonical_host_bytes;
    local.tensor_scratch = tensor_scratch;
    local.tensor_scratch_bytes = tensor_scratch_bytes;
    rc = cpu_dispatch_cell(&local, cell, source_position,
        candidate_token_ids, candidate_count, assigned_complete);
    if (rc != 0) return -1;
    stats->projection_dispatches += local.stats.projection_dispatches;
    stats->expert_gate_up_dispatches +=
        local.stats.expert_gate_up_dispatches;
    stats->expert_down_dispatches += local.stats.expert_down_dispatches;
    return 0;
}

static int cpu_graph_build_jobs(
        SaltTextVerifyCpuContext *context,
        const SaltTextExecutionCell *cell, uint32_t rows,
        uint32_t first, uint32_t count, uint32_t *job_count) {
    const SaltTextVerifyProgram *program = context->program;
    SaltTextTensorHostBatch *jobs = context->node_jobs;
    const SaltTextCanonicalLayout *layout = &program->layout;
    uint32_t initial;
    if (!jobs || context->node_job_capacity == 0 || !job_count)
        return -1;
    initial = *job_count;
#define ADD_JOB(tensor_, input_, rows_, output_) do { \
    if (*job_count >= context->node_job_capacity) return -1; \
    jobs[*job_count] = (SaltTextTensorHostBatch) { \
        (tensor_), (input_), (rows_), (output_), \
    }; \
    (*job_count)++; \
} while (0)
    if (cell->kind == SALT_TEXT_CELL_FINAL_HEAD) {
        ADD_JOB(&program->descriptor->output_head,
            cpu_floats(context, layout->final_state) +
                (size_t)first * program->hidden,
            count, cpu_floats(context, cell->destination_offset) +
                (size_t)first * program->vocabulary);
    } else {
        const SaltTextLayerExecDesc *layer;
        const SaltTextLayerPlan *plan;
        uint32_t hidden = program->hidden;
        if (cell->layer >= program->layer_count) return -1;
        layer = program->layers[cell->layer].descriptor;
        plan = layer->plan;
        switch (cell->kind) {
        case SALT_TEXT_CELL_QUERY_PROJECTION:
            ADD_JOB(&layer->q, cpu_floats(context, layout->normalized) +
                (size_t)first * layer->q.cols, count,
                cpu_floats(context, layout->queries) +
                (size_t)first * layer->q.rows);
            ADD_JOB(&layer->k, cpu_floats(context, layout->normalized) +
                (size_t)first * layer->k.cols, count,
                cpu_floats(context, layout->keys) +
                (size_t)first * layer->k.rows);
            if (!plan->attention.shared_kv_projection)
                ADD_JOB(&layer->v, cpu_floats(context, layout->normalized) +
                    (size_t)first * layer->v.cols, count,
                    cpu_floats(context, layout->values) +
                    (size_t)first * layer->v.rows);
            break;
        case SALT_TEXT_CELL_OUTPUT_PROJECTION:
            ADD_JOB(&layer->o, cpu_floats(context, layout->attention_output) +
                (size_t)first * layer->o.cols, count,
                cpu_floats(context, cell->destination_offset) +
                (size_t)first * layer->o.rows);
            break;
        case SALT_TEXT_CELL_DENSE_GATE:
            ADD_JOB(&layer->dense_gate,
                cpu_floats(context, layout->normalized) +
                (size_t)first * hidden, count,
                cpu_floats(context, layout->dense_gate) +
                (size_t)first * (uint32_t)plan->dense_intermediate);
            ADD_JOB(&layer->dense_up,
                cpu_floats(context, layout->normalized) +
                (size_t)first * hidden, count,
                cpu_floats(context, layout->dense_up) +
                (size_t)first * (uint32_t)plan->dense_intermediate);
            break;
        case SALT_TEXT_CELL_DENSE_DOWN:
            ADD_JOB(&layer->dense_down,
                cpu_floats(context, layout->dense_chain) +
                (size_t)first * (uint32_t)plan->dense_intermediate, count,
                cpu_floats(context, cell->destination_offset) +
                (size_t)first * hidden);
            break;
        case SALT_TEXT_CELL_ROUTER_PROJECTION:
            ADD_JOB(&layer->router, cpu_floats(context, layout->router_input) +
                (size_t)first * hidden, count,
                cpu_floats(context, cell->destination_offset) +
                (size_t)first * (uint32_t)plan->n_experts);
            break;
        case SALT_TEXT_CELL_EXPERT_GATE:
        case SALT_TEXT_CELL_EXPERT_DOWN: {
            int32_t *selected = cpu_i32(context, layout->selected_experts);
            int32_t *grouped = cpu_i32(context, layout->grouped_to_canonical);
            float *routed_input = cpu_floats(context, layout->routed_input);
            float *routed_chain = cpu_floats(context, layout->routed_chain);
            float *destination = cpu_floats(context, cell->destination_offset);
            uint32_t topk = (uint32_t)plan->top_k_experts;
            uint32_t expert_width = (uint32_t)plan->expert_intermediate;
            uint32_t scan = first;
            while (scan < first + count) {
                uint32_t canonical = (uint32_t)grouped[scan];
                int32_t expert;
                uint32_t last = scan + 1u, output_width;
                const SaltTextTensorDesc *tensor;
                const float *inputs;
                if (canonical >= rows * topk) return -1;
                expert = selected[canonical];
                if (expert < 0 || expert >= plan->n_experts) return -1;
                while (last < first + count) {
                    uint32_t next = (uint32_t)grouped[last];
                    if (next >= rows * topk || selected[next] != expert) break;
                    last++;
                }
                if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) {
                    inputs = routed_input + (size_t)scan * hidden;
                    ADD_JOB(&layer->experts[expert].gate,
                        inputs, last - scan,
                        cpu_floats(context, layout->routed_gate) +
                            (size_t)scan * expert_width);
                    ADD_JOB(&layer->experts[expert].up,
                        inputs, last - scan,
                        cpu_floats(context, layout->routed_up) +
                            (size_t)scan * expert_width);
                    scan = last;
                    continue;
                } else {
                    tensor = &layer->experts[expert].down;
                    inputs = routed_chain + (size_t)scan * expert_width;
                    output_width = hidden;
                }
                ADD_JOB(tensor, inputs, last - scan,
                    destination + (size_t)scan * output_width);
                scan = last;
            }
            break;
        }
        default:
            return -1;
        }
    }
#undef ADD_JOB
    return *job_count > initial ? 0 : -1;
}

typedef struct CpuFfnFrontierExpert {
    const SaltTensorHostNodeOps *gate_ops;
    const SaltTensorHostNodeOps *down_ops;
    void *gate_seat;
    void *down_seat;
    float *gate;
    float *up;
    float *chain;
    uint32_t value_count;
    uint32_t gate_remaining;
    int32_t expert_id;
} CpuFfnFrontierExpert;

typedef struct CpuFfnFrontierCall {
    SaltAreaFrontier *frontier;
    CpuFfnFrontierExpert experts[CPU_MAX_FFN_FRONTIER_EXPERTS];
    SaltActivationKind activation;
    float activation_limit;
    uint32_t expert_count;
    uint32_t workers;
    uint32_t gate_tasks;
    uint32_t total_tasks;
    uint32_t completed_tasks;
    uint32_t failed;
} CpuFfnFrontierCall;

static int cpu_ffn_frontier_activate(CpuFfnFrontierCall *call,
                                     uint32_t expert) {
    CpuFfnFrontierExpert *entry;
    if (!call || expert >= call->expert_count ||
        !(entry = &call->experts[expert])->gate || !entry->up ||
        !entry->chain || entry->value_count == 0u)
        return -1;
    for (uint32_t index = 0u; index < entry->value_count; index++) {
        float value = cpu_gate_up(entry->gate[index], entry->up[index],
            call->activation, call->activation_limit);
        if (!isfinite(value)) return -1;
        entry->chain[index] = value;
    }
    for (uint32_t slice = 0u; slice < call->workers; slice++) {
        uint32_t task_index = call->gate_tasks +
            slice * call->expert_count + expert;
        if (task_index >= call->total_tasks ||
            call->frontier->tasks[task_index].state !=
                SALT_AREA_TASK_BLOCKED)
            return -1;
        __atomic_store_n(&call->frontier->tasks[task_index].state,
                         SALT_AREA_TASK_QUEUED, __ATOMIC_RELEASE);
    }
    return 0;
}

static void cpu_ffn_frontier_worker(int worker, void *opaque) {
    CpuFfnFrontierCall *call = (CpuFfnFrontierCall *)opaque;
    uint32_t cursor;
    if (!call || worker < 0 || (uint32_t)worker >= call->workers) return;
    cursor = (uint32_t)worker;
    while (__atomic_load_n(&call->completed_tasks, __ATOMIC_ACQUIRE) <
               call->total_tasks &&
           __atomic_load_n(&call->failed, __ATOMIC_ACQUIRE) == 0u) {
        int claimed = 0;
        for (uint32_t scan = 0u; scan < call->total_tasks; scan++) {
            uint32_t task_index = (cursor + scan) % call->total_tasks;
            SaltAreaTask *task = &call->frontier->tasks[task_index];
            SaltAreaTaskState expected = SALT_AREA_TASK_QUEUED;
            CpuFfnFrontierExpert *entry;
            int rc;
            if (!__atomic_compare_exchange_n(
                    &task->state, &expected, SALT_AREA_TASK_RUNNING, 0,
                    __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
                continue;
            claimed = 1;
            cursor = task_index + 1u;
            if (task->candidate_first >= call->expert_count ||
                task->job_index >= call->workers) {
                __atomic_store_n(&call->failed, 1u, __ATOMIC_RELEASE);
                __atomic_store_n(&task->state, SALT_AREA_TASK_CANCELLED,
                                 __ATOMIC_RELEASE);
                return;
            }
            entry = &call->experts[task->candidate_first];
            if (task->phase == (uint32_t)SALT_TEXT_CELL_EXPERT_GATE)
                rc = entry->gate_ops->worker(
                    entry->gate_seat, task->job_index);
            else if (task->phase == (uint32_t)SALT_TEXT_CELL_EXPERT_DOWN)
                rc = entry->down_ops->worker(
                    entry->down_seat, task->job_index);
            else
                rc = -1;
            if (rc != 0) {
                __atomic_store_n(&call->failed, 1u, __ATOMIC_RELEASE);
                __atomic_store_n(&task->state, SALT_AREA_TASK_CANCELLED,
                                 __ATOMIC_RELEASE);
                return;
            }
            __atomic_store_n(&task->state, SALT_AREA_TASK_DONE,
                             __ATOMIC_RELEASE);
            if (task->phase == (uint32_t)SALT_TEXT_CELL_EXPERT_GATE) {
                uint32_t prior = __atomic_fetch_sub(
                    &entry->gate_remaining, 1u, __ATOMIC_ACQ_REL);
                if (prior == 0u ||
                    (prior == 1u && cpu_ffn_frontier_activate(
                        call, task->candidate_first) != 0)) {
                    __atomic_store_n(&call->failed, 1u, __ATOMIC_RELEASE);
                    return;
                }
            }
            __atomic_fetch_add(
                &call->completed_tasks, 1u, __ATOMIC_RELEASE);
            break;
        }
        if (!claimed) {
            if (__atomic_load_n(&call->completed_tasks, __ATOMIC_ACQUIRE) >=
                    call->total_tasks ||
                __atomic_load_n(&call->failed, __ATOMIC_ACQUIRE) != 0u)
                return;
            sched_yield();
        }
    }
}

static int cpu_ffn_frontier_finish(CpuFfnFrontierCall *call,
                                   uint32_t gate_prepared,
                                   uint32_t down_prepared, int result) {
    if (!call) return -1;
    for (uint32_t expert = down_prepared; expert > 0u; expert--)
        if (call->experts[expert - 1u].down_ops->finish(
                call->experts[expert - 1u].down_seat) != 0)
            result = -1;
    for (uint32_t expert = gate_prepared; expert > 0u; expert--)
        if (call->experts[expert - 1u].gate_ops->finish(
                call->experts[expert - 1u].gate_seat) != 0)
            result = -1;
    return result;
}

/* One-row routed FFN realization over the existing persistent pool. Every
 * retained node still owns its exact row slices and arithmetic. The engine
 * frontier changes only readiness: a free physical worker claims any ready
 * logical slice, and the last gate/up slice for one expert exposes that
 * expert's already prepared down slices without waiting for unrelated experts. */
static int cpu_ffn_frontier_execute(
        SaltTextVerifyCpuContext *context,
        const SaltTextExecutionCell *gate_cell,
        uint32_t first, uint32_t count) {
    CpuFfnFrontierCall call;
    const SaltTextVerifyProgram *program;
    const SaltTextCompiledLayer *compiled;
    const SaltTextLayerExecDesc *layer;
    const SaltTextExecutionCell *down_cell = NULL;
    SaltTextTensorHostBatch *jobs;
    int32_t *selected, *grouped;
    float *routed_gate, *routed_up, *routed_chain;
    uint32_t job_count = 0u, gate_prepared = 0u, down_prepared = 0u;
    uint32_t topk, workers;
    int result = -1;
    if (!context || !(program = context->program) || !gate_cell ||
        gate_cell->kind != SALT_TEXT_CELL_EXPERT_GATE ||
        gate_cell->layer >= program->layer_count || first != 0u ||
        !context->parallel_run || !context->parallel_context ||
        context->parallel_workers == 0u ||
        !context->ffn_frontier_seats ||
        context->ffn_frontier_seat_stride == 0u)
        return 1;
    compiled = &program->layers[gate_cell->layer];
    layer = compiled->descriptor;
    topk = (uint32_t)layer->plan->top_k_experts;
    workers = context->parallel_workers;
    if (count != topk || topk == 0u ||
        topk > CPU_MAX_FFN_FRONTIER_EXPERTS ||
        2u * topk > context->ffn_frontier_seat_capacity ||
        topk > UINT32_MAX / workers ||
        2u * topk * workers > context->area_task_capacity)
        return 1;
    for (uint32_t index = 0u;
         index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *candidate = &program->dispatch.cells[index];
        if (candidate->layer == gate_cell->layer &&
            candidate->kind == SALT_TEXT_CELL_EXPERT_DOWN) {
            down_cell = candidate;
            break;
        }
    }
    if (!down_cell) return -1;
    memset(&call, 0, sizeof call);
    call.frontier = &context->area_frontier;
    call.activation = layer->plan->activation;
    call.activation_limit = layer->plan->activation_limit;
    call.expert_count = topk;
    call.workers = workers;
    call.gate_tasks = topk * workers;
    call.total_tasks = 2u * call.gate_tasks;
    jobs = context->node_jobs;
    selected = cpu_i32(context, program->layout.selected_experts);
    grouped = cpu_i32(context, program->layout.grouped_to_canonical);
    routed_gate = cpu_floats(context, program->layout.routed_gate);
    routed_up = cpu_floats(context, program->layout.routed_up);
    routed_chain = cpu_floats(context, program->layout.routed_chain);
    if (cpu_graph_build_jobs(
            context, gate_cell, 1u, 0u, topk, &job_count) != 0 ||
        job_count != 2u * topk)
        return -1;
    for (uint32_t expert = 0u; expert < topk; expert++) {
        SaltTextTensorHostBatch pair[2] = {
            jobs[2u * expert], jobs[2u * expert + 1u],
        };
        CpuFfnFrontierExpert *entry = &call.experts[expert];
        SaltTensorHostNodePlan plan;
        const SaltTensorHostNodeOps *ops;
        ptrdiff_t gate_offset, up_offset;
        uint32_t canonical = (uint32_t)grouped[expert];
        if (canonical >= topk || selected[canonical] < 0 ||
            pair[0].row_count == 0u ||
            pair[0].row_count != pair[1].row_count ||
            !pair[0].tensor || !pair[1].tensor ||
            pair[0].tensor->rows != (uint32_t)layer->plan->expert_intermediate ||
            pair[1].tensor->rows != (uint32_t)layer->plan->expert_intermediate ||
            !pair[0].tensor->host_ops || !pair[0].tensor->host_ops->cpu ||
            !(ops = pair[0].tensor->host_ops->cpu->retained_node) ||
            !pair[1].tensor->host_ops || !pair[1].tensor->host_ops->cpu ||
            pair[1].tensor->host_ops->cpu->retained_node != ops)
            goto done;
        gate_offset = pair[0].outputs - routed_gate;
        up_offset = pair[1].outputs - routed_up;
        if (gate_offset < 0 || up_offset != gate_offset ||
            (size_t)gate_offset > SIZE_MAX -
                (size_t)pair[0].row_count * pair[0].tensor->rows)
            goto done;
        entry->gate_ops = ops;
        entry->gate_seat = context->ffn_frontier_seats +
            (size_t)expert * context->ffn_frontier_seat_stride;
        entry->gate = pair[0].outputs;
        entry->up = pair[1].outputs;
        entry->chain = routed_chain + (size_t)gate_offset;
        entry->value_count = pair[0].row_count * pair[0].tensor->rows;
        entry->gate_remaining = workers;
        entry->expert_id = selected[canonical];
        memset(&plan, 0, sizeof plan);
        if (ops->prepare(pair, 2u, pair[0].row_count, workers,
                entry->gate_seat, context->ffn_frontier_seat_stride,
                &plan) != 0 || plan.active_workers != workers)
            goto done;
        gate_prepared++;
    }
    job_count = 0u;
    if (cpu_graph_build_jobs(
            context, down_cell, 1u, 0u, topk, &job_count) != 0 ||
        job_count != topk)
        goto done;
    for (uint32_t expert = 0u; expert < topk; expert++) {
        CpuFfnFrontierExpert *entry = &call.experts[expert];
        SaltTensorHostNodePlan plan;
        const SaltTensorHostNodeOps *ops;
        if (!jobs[expert].tensor || !jobs[expert].tensor->host_ops ||
            !jobs[expert].tensor->host_ops->cpu ||
            !(ops = jobs[expert].tensor->host_ops->cpu->retained_node) ||
            jobs[expert].row_count == 0u)
            goto done;
        entry->down_ops = ops;
        entry->down_seat = context->ffn_frontier_seats +
            (size_t)(topk + expert) * context->ffn_frontier_seat_stride;
        memset(&plan, 0, sizeof plan);
        if (ops->prepare(&jobs[expert], 1u, jobs[expert].row_count, workers,
                entry->down_seat, context->ffn_frontier_seat_stride,
                &plan) != 0 || plan.active_workers != workers)
            goto done;
        down_prepared++;
    }
    if (salt_area_frontier_reset(
            call.frontier, context->submitted_generation) != 0)
        goto done;
    for (uint32_t phase = 0u; phase < 2u; phase++)
    for (uint32_t slice = 0u; slice < workers; slice++)
    for (uint32_t expert = 0u; expert < topk; expert++) {
        uint32_t index = phase * call.gate_tasks + slice * topk + expert;
        SaltAreaTask task = {
            context->submitted_generation,
            gate_cell->layer,
            phase == 0u ? UINT32_MAX : slice * topk + expert,
            phase == 0u ? (uint32_t)SALT_TEXT_CELL_EXPERT_GATE
                        : (uint32_t)SALT_TEXT_CELL_EXPERT_DOWN,
            call.experts[expert].expert_id,
            slice,
            expert, 1u, slice, 1u, 0u,
            SALT_AREA_TASK_QUEUED,
        };
        if (salt_area_frontier_append(call.frontier, &task) != 0)
            goto done;
        if (phase != 0u)
            call.frontier->tasks[index].state = SALT_AREA_TASK_BLOCKED;
    }
    if (call.frontier->count != call.total_tasks ||
        context->parallel_run(context->parallel_context, (int)workers,
            cpu_ffn_frontier_worker, &call) != 0 ||
        __atomic_load_n(&call.failed, __ATOMIC_ACQUIRE) != 0u ||
        __atomic_load_n(&call.completed_tasks, __ATOMIC_ACQUIRE) !=
            call.total_tasks)
        goto done;
    result = 0;
done:
    result = cpu_ffn_frontier_finish(
        &call, gate_prepared, down_prepared, result);
    if (result == 0) {
        context->frontier_expert_layer = gate_cell->layer;
        context->stats.projection_dispatches += 2u;
        context->stats.expert_gate_up_dispatches++;
        context->stats.expert_down_dispatches++;
        context->stats.cpu_matrix_pool_phases++;
        context->stats.cpu_expert_gate_up_waves++;
        context->stats.cpu_expert_down_waves++;
    }
    return result;
}

static int cpu_graph_is_parallel(void *opaque, uint32_t node) {
    SaltTextVerifyCpuContext *context = (SaltTextVerifyCpuContext *)opaque;
    return context && context->program && context->graph_spans &&
        node < context->graph_span_count
        ? (int)context->graph_spans[node].parallel : -1;
}

static int cpu_graph_prepare_parallel(
        void *opaque, uint32_t node,
        const SaltTensorHostNodeOps **ops_out,
        void **seat_out, uint32_t *active_workers_out) {
    SaltTextVerifyCpuContext *context = (SaltTextVerifyCpuContext *)opaque;
    const SaltTextVerifyProgram *program = context->program;
    const SaltTensorHostGraphSpan *span;
    uint32_t rows = context->submitted_candidates;
    uint32_t job_count = 0;
    int qkv = 0, dense = 0, expert_gate = 0, expert_down = 0;
    if (!context || !program || node >= context->graph_span_count ||
        !ops_out || !seat_out || !active_workers_out)
        return -1;
    span = &context->graph_spans[node];
    if (!span->parallel || span->node_count == 0 ||
        span->first_node >= program->dispatch.cell_count ||
        span->node_count > program->dispatch.cell_count - span->first_node)
        return -1;
    for (uint32_t offset = 0; offset < span->node_count; offset++) {
        uint32_t cell_index = span->first_node + offset;
        const SaltTextExecutionCell *cell =
            &program->dispatch.cells[cell_index];
        const SaltTextExecutionAssignment *assignment =
            &context->assignment_plan.assignments[cell_index];
        uint32_t actual = rows;
        if (cpu_graph_alias_kind(cell->kind)) continue;
        if (!cpu_graph_projection_kind(cell->kind)) return -1;
        if (cell->unit == SALT_TEXT_EXECUTION_JOBS)
            actual = rows * (uint32_t)program->layers[cell->layer].descriptor->
                plan->top_k_experts;
        if (assignment->cpu.first != 0 || assignment->cpu.count < actual ||
            assignment->gpu.count != 0 ||
            cpu_graph_build_jobs(context, cell, rows, 0u, actual,
                &job_count) != 0)
            return -1;
        if (cell->kind == SALT_TEXT_CELL_QUERY_PROJECTION) qkv = 1;
        else if (cell->kind == SALT_TEXT_CELL_DENSE_GATE) dense = 1;
        else if (cell->kind == SALT_TEXT_CELL_EXPERT_GATE) expert_gate = 1;
        else if (cell->kind == SALT_TEXT_CELL_EXPERT_DOWN) expert_down = 1;
    }
    if (job_count == 0 ||
        !context->node_jobs[0].tensor->host_ops ||
        !context->node_jobs[0].tensor->host_ops->cpu ||
        !(*ops_out = context->node_jobs[0].tensor->host_ops->cpu->retained_node) ||
        !(*ops_out)->prepare || !(*ops_out)->worker || !(*ops_out)->finish)
        return -1;
    for (uint32_t job = 1; job < job_count; job++)
        if (!context->node_jobs[job].tensor->host_ops ||
            !context->node_jobs[job].tensor->host_ops->cpu ||
            context->node_jobs[job].tensor->host_ops->cpu->retained_node !=
                *ops_out)
            return -1;
    if (
        (*ops_out)->prepare(context->node_jobs, job_count, rows,
            context->graph_workers, context->tensor_scratch,
            context->tensor_scratch_bytes, &context->graph_node_plan) != 0)
        return -1;
    *seat_out = context->tensor_scratch;
    *active_workers_out = context->graph_node_plan.active_workers;
    context->stats.projection_dispatches++;
    context->stats.cpu_matrix_pool_phases++;
    if (qkv) context->stats.cpu_qkv_waves++;
    if (dense) context->stats.cpu_dense_gate_up_waves++;
    if (expert_gate) {
        context->stats.expert_gate_up_dispatches++;
        context->stats.cpu_expert_gate_up_waves++;
    }
    if (expert_down) {
        context->stats.expert_down_dispatches++;
        context->stats.cpu_expert_down_waves++;
    }
    return 0;
}

static int cpu_finish_host_node(SaltTextVerifyCpuContext *context);

static int cpu_graph_execute_serial(void *opaque, uint32_t node) {
    SaltTextVerifyCpuContext *context = (SaltTextVerifyCpuContext *)opaque;
    const SaltTextVerifyProgram *program = context->program;
    const SaltTensorHostGraphSpan *span;
    const int32_t *tokens;
    if (!context || !program || node >= context->graph_span_count) return -1;
    span = &context->graph_spans[node];
    if (span->parallel || span->node_count == 0 ||
        span->first_node >= program->dispatch.cell_count ||
        span->node_count > program->dispatch.cell_count - span->first_node)
        return -1;
    context->graph_state.node_ops = NULL;
    context->graph_state.node_seat = NULL;
    context->graph_state.active_workers = 0u;
    tokens = (const int32_t *)(const void *)(context->arena +
        program->layout.candidate_token_ids);
    for (uint32_t offset = 0; offset < span->node_count; offset++) {
        uint32_t cell_index = span->first_node + offset;
        const SaltTextExecutionCell *cell =
            &program->dispatch.cells[cell_index];
        const SaltTextExecutionAssignment *assignment =
            &context->assignment_plan.assignments[cell_index];
        uint32_t actual = context->submitted_candidates;
        SaltTextExecutionSlice active;
        uint64_t cell_started = context->profile_enabled ? cpu_now_ns() : 0;
        if (cell->unit == SALT_TEXT_EXECUTION_JOBS)
            actual *= (uint32_t)program->layers[cell->layer].descriptor->
                plan->top_k_experts;
        if (cpu_graph_alias_kind(cell->kind)) {
            if (cell->kind == SALT_TEXT_CELL_EXPERT_UP) continue;
            goto fail;
        }
        if (assignment->cpu.first != 0 || assignment->cpu.count < actual ||
            assignment->gpu.count != 0)
            goto fail;
        active = (SaltTextExecutionSlice) { 0u, actual };
        context->coalescing_enabled =
            cell->kind == SALT_TEXT_CELL_EXPERT_GATE ||
            cell->kind == SALT_TEXT_CELL_EXPERT_DOWN;
        if (cpu_dispatch_cell(context, cell,
                context->submitted_source_position, tokens,
                context->submitted_candidates, active) != 0) {
            context->coalescing_enabled = 0;
            goto fail;
        }
        context->coalescing_enabled = 0;
        if (cell->kind == SALT_TEXT_CELL_EXPERT_DOWN)
            context->coalesced_expert_layer = UINT32_MAX;
        if (context->profile_enabled &&
            (uint32_t)cell->kind < SALT_TEXT_EXECUTION_CELL_KIND_COUNT) {
            cpu_add_elapsed(&context->stats.cpu_target_cell_ns[cell->kind],
                            cell_started);
            context->stats.cpu_target_cell_calls[cell->kind]++;
        }
    }
    if (context->graph_state.node_ops || context->graph_state.node_seat)
        goto fail;
    return 0;
fail:
    context->coalescing_enabled = 0;
    (void)cpu_finish_host_node(context);
    return -1;
}

/* A host realization may retain its existing scratch-seat resource leases
 * across producer/consumer cells. The existing node finish callback is the
 * cleanup authority; no arithmetic or graph scheduling is performed here. */
static int cpu_finish_host_node(SaltTextVerifyCpuContext *context) {
    const SaltTensorHostNodeOps *ops = context->graph_state.node_ops;
    void *seat = context->graph_state.node_seat;
    context->graph_state.node_ops = NULL;
    context->graph_state.node_seat = NULL;
    if (!ops) return seat ? -1 : 0;
    return ops->finish && seat ? ops->finish(seat) : -1;
}

static int cpu_interpret_nodes(SaltTextVerifyCpuContext *context,
                         uint32_t source, const int32_t *tokens,
                         uint32_t rows, uint32_t output_rows) {
    const SaltTextVerifyProgram *program = context->program;
    int graph_enabled;
    uint32_t epoch = 0;
    uint64_t interpret_started = context->profile_enabled ? cpu_now_ns() : 0;
    if (!program || !program->target_policy ||
        !program->target_policy->ready ||
        !context->assignment_plan.assignments ||
        context->assignment_plan.program != program ||
        context->assignment_plan.execution_class != SALT_TEXT_EXECUTION_CPU_ONLY ||
        context->assignment_plan.assignment_count != program->dispatch.cell_count)
        return -1;
    graph_enabled = program->target_policy->cpu_graph != 0u;
    if (graph_enabled) {
        SaltTensorHostGraphCallbacks callbacks = {
            context->graph_span_count,
            program->dispatch.cell_count,
            cpu_graph_is_parallel,
            cpu_graph_prepare_parallel,
            cpu_graph_execute_serial,
        };
        SaltTensorHostGraphResult graph_result;
        int graph_rc;
        if (!context->parallel_run || !context->parallel_context ||
            context->parallel_workers == 0 || context->parallel_workers > 32u)
            return -1;
        context->coalescing_enabled = 0;
        context->graph_workers = context->parallel_workers;
        context->stats.cpu_graph_sessions = 1u;
        graph_rc = salt_tensor_host_graph_execute(
            &context->graph_state, &callbacks, context, context->graph_workers,
            context->parallel_run, context->parallel_context, &graph_result);
        context->graph_workers = 0u;
        context->stats.cpu_graph_nodes = graph_result.nodes_executed;
        context->stats.cpu_graph_spans = graph_result.spans_executed;
        context->stats.cpu_graph_serial_spans = graph_result.serial_spans;
        context->stats.cpu_graph_parallel_spans = graph_result.parallel_spans;
        context->stats.cpu_graph_barriers = graph_result.internal_barriers;
        if (context->profile_enabled)
            cpu_add_elapsed(&context->stats.cpu_target_interpret_ns,
                            interpret_started);
        if (graph_rc != 0 ||
            graph_result.nodes_executed != program->dispatch.cell_count ||
            context->graph_state.node_ops || context->graph_state.node_seat) {
            (void)cpu_finish_host_node(context);
            return -1;
        }
        return 0;
    }
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        const SaltTextExecutionAssignment *assignment =
            &context->assignment_plan.assignments[index];
        SaltTextExecutionSlice active_cpu;
        uint32_t actual = rows;
        if (cell->unit == SALT_TEXT_EXECUTION_JOBS) {
            if (cell->layer >= program->layer_count) goto fail;
            actual = rows * (uint32_t)program->layers[cell->layer].descriptor->
                plan->top_k_experts;
        }
        if (actual > cell->logical_capacity || assignment->cpu.first != 0 ||
            assignment->cpu.count < actual || assignment->gpu.count != 0 ||
            cell->dependency_epoch > epoch ||
            cell->completion_epoch <= cell->dependency_epoch ||
            cell->destination_offset >= program->layout.total_bytes ||
            cell->destination_stride == 0 ||
            actual > (program->layout.total_bytes -
                cell->destination_offset) / cell->destination_stride)
            goto fail;
        active_cpu.first = 0;
        active_cpu.count = actual;
        if (cell->layer == UINT32_MAX &&
            cell->kind >= SALT_TEXT_CELL_FINAL_NORM) {
            active_cpu.first = rows - output_rows;
            active_cpu.count = output_rows;
            if (output_rows == 0u) {
                if (cell->completion_epoch > epoch) epoch = cell->completion_epoch;
                continue;
            }
        }
        uint64_t cell_started = context->profile_enabled ? cpu_now_ns() : 0;
        if (cpu_dispatch_cell(context, cell, source, tokens, rows,
                active_cpu) != 0) {
            fprintf(stderr,
                "SALT_TEXT_CPU_CELL_FAIL index=%u kind=%d layer=%u "
                "unit=%d first=%u count=%u dependency=%u completion=%u\n",
                index, (int)cell->kind, cell->layer, (int)cell->unit,
                active_cpu.first, active_cpu.count,
                cell->dependency_epoch, cell->completion_epoch);
            goto fail;
        }
        if (context->profile_enabled &&
            (uint32_t)cell->kind < SALT_TEXT_EXECUTION_CELL_KIND_COUNT) {
            cpu_add_elapsed(&context->stats.cpu_target_cell_ns[cell->kind],
                            cell_started);
            context->stats.cpu_target_cell_calls[cell->kind]++;
        }
        if (cell->completion_epoch > epoch) epoch = cell->completion_epoch;
    }
    if (epoch != program->dispatch.final_dependency_epoch ||
        context->graph_state.node_ops || context->graph_state.node_seat)
        goto fail;
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_interpret_ns,
                        interpret_started);
    return 0;
fail:
    (void)cpu_finish_host_node(context);
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_interpret_ns,
                        interpret_started);
    return -1;
}

/* Collective attention consumes the identical head-fold and KV-view helpers.
 * Only row/head ownership changes; no nested pool dispatch from a graph cell. */
static int cpu_collective_attention(SaltTextVerifyCpuContext *c,
        const SaltTextExecutionCell *cell, uint32_t worker) {
    const SaltTextCompiledLayer *compiled = &c->program->layers[cell->layer];
    const SaltTextLayerExecDesc *layer = compiled->descriptor;
    const SaltAttentionDesc *attention = &layer->plan->attention;
    SaltTextKvReadView keys, values;
    SaltAttnKvSpan spans[SALT_ATTN_HEAD_MAX_SPANS];
    const uint32_t *parents = NULL, *depths = NULL;
    uint32_t rows = c->submitted_candidates, source = c->submitted_source_position;
    size_t capacity = (size_t)c->program->maximum_candidates * compiled->kv_width;
    int nspan = 0;
    if (c->submitted_frontier) {
        parents = (const uint32_t *)(const void *)(c->arena + c->program->layout.target_parent_rows);
        depths = (const uint32_t *)(const void *)(c->arena + c->program->layout.target_depths);
    }
    if (salt_text_kv_read_view_init(&keys, &layer->kv->keys,
            cpu_floats(c, compiled->tentative_key_offset), capacity, compiled->kv_width,
            rows, source, parents, depths, compiled->kv_width) ||
        salt_text_kv_read_view_init(&values, &layer->kv->values,
            cpu_floats(c, compiled->tentative_value_offset), capacity, compiled->kv_width,
            rows, source, parents, depths, compiled->kv_width)) return -1;
    if (!c->submitted_frontier && rows == 1u) {
        uint32_t first = 0u;
        int rc;
        if (attention->kind == SALT_ATTN_SLIDING && source + 1u > (uint32_t)attention->window)
            first = source + 1u - (uint32_t)attention->window;
        memset(spans, 0, sizeof spans);
        rc = cpu_attention_linear_spans(&keys, &values, 0u, first,
            source - first + 1u, compiled->kv_width, spans, &nspan);
        if (rc < 0) return -1;
        if (rc > 0) nspan = 0;
    }
    uint32_t work = rows * (uint32_t)attention->n_heads;
    uint32_t active = c->parallel_workers > 1u && work >= c->parallel_workers
        ? c->parallel_workers : 1u;
    if (worker >= active) return 0;
    uint32_t first = (uint32_t)((uint64_t)work * worker / active);
    uint32_t end = (uint32_t)((uint64_t)work * (worker + 1u) / active);
    float *scores = active == 1u
        ? cpu_floats(c, c->program->layout.attention_scores)
        : c->parallel_scores + (size_t)worker * c->parallel_score_stride;
    for (uint32_t i = first; i < end; i++)
        if (cpu_attention_head(c, compiled, &keys, &values, nspan ? spans : NULL,
                nspan, source, i / (uint32_t)attention->n_heads,
                i % (uint32_t)attention->n_heads, scores)) return -1;
    return 0;
}

static int cpu_collective_cell(void *opaque, uint32_t node, uint32_t worker) {
    SaltTextVerifyCpuContext *c = opaque;
    const SaltTextVerifyProgram *p = c->program;
    const SaltTextExecutionCell *cell = &p->dispatch.cells[node];
    const SaltTextExecutionAssignment *a = &c->assignment_plan.assignments[node];
    SaltTensorHostGraphState *g = &c->graph_state;
    uint32_t rows = c->submitted_candidates, actual = rows, first = 0u;
    uint32_t owner = node % c->parallel_workers;
    int projection = cpu_graph_projection_kind(cell->kind) ||
        cell->kind == SALT_TEXT_CELL_EXPERT_GATE || cell->kind == SALT_TEXT_CELL_EXPERT_DOWN;
    if (cell->unit == SALT_TEXT_EXECUTION_JOBS) {
        if (cell->layer >= p->layer_count) return -1;
        actual = rows * (uint32_t)p->layers[cell->layer].descriptor->plan->top_k_experts;
    }
    if (actual > cell->logical_capacity || a->cpu.first != 0u || a->cpu.count < actual ||
        a->gpu.count != 0u || !cell->destination_stride ||
        cell->destination_offset >= p->layout.total_bytes ||
        actual > (p->layout.total_bytes - cell->destination_offset) / cell->destination_stride)
        return -1;
    if (cell->layer == UINT32_MAX && cell->kind >= SALT_TEXT_CELL_FINAL_NORM) {
        first = rows - c->collective_output_rows;
        actual = c->collective_output_rows;
        if (!actual) return 0;
    }
    if (cpu_graph_alias_kind(cell->kind)) return 0;
    if (cell->kind == SALT_TEXT_CELL_ATTENTION_BODY)
        return cpu_collective_attention(c, cell, worker);
    if (!projection) {
        /* Canonical serial folds retain one owner, distributed across nodes;
         * no arithmetic is recreated and there is no coordinator mailbox. */
        return worker == owner ? cpu_dispatch_cell(c, cell,
            c->submitted_source_position,
            (const int32_t *)(const void *)(c->arena + p->layout.candidate_token_ids),
            rows, (SaltTextExecutionSlice){first, actual}) : 0;
    }
    if (worker == owner) {
        c->collective_job_count = 0u;
        c->collective_error = cpu_graph_build_jobs(c, cell, rows, first, actual,
                                                  &c->collective_job_count);
    }
    salt_tensor_host_graph_barrier(g);
    int prepare_error = c->collective_error;
    uint32_t total = c->collective_job_count;
    salt_tensor_host_graph_barrier(g);
    if (prepare_error) return -1;
    for (uint32_t done = 0u; done < total;) {
        const SaltTensorHostNodeOps *ops;
        uint32_t take;
        if (worker == owner) {
            SaltTensorHostNodePlan *plan = &c->graph_node_plan;
            const SaltTensorHostBatch *job = c->node_jobs + done;
            g->node_ops = NULL; g->node_seat = NULL;
            c->collective_job_take = 0u;
            memset(plan, 0, sizeof *plan);
            ops = job->tensor->host_ops->cpu->retained_node;
            c->collective_error = !ops || !ops->prepare || !ops->worker || !ops->finish;
            if (!c->collective_error) {
                c->collective_error = ops->prepare(job, total-done, rows,
                    c->parallel_workers, c->tensor_scratch, c->tensor_scratch_bytes, plan);
                if (!c->collective_error) {
                    g->node_ops = ops; g->node_seat = c->tensor_scratch;
                    c->collective_job_take = plan->consumed_jobs ? plan->consumed_jobs : total-done;
                    if (!plan->active_workers || plan->active_workers > c->parallel_workers ||
                        c->collective_job_take > total-done) c->collective_error = -1;
                }
            }
        }
        salt_tensor_host_graph_barrier(g);
        take = c->collective_job_take;
        ops = g->node_ops;
        if (!c->collective_error && worker < c->graph_node_plan.active_workers &&
            ops->worker(g->node_seat, worker) != 0)
            __atomic_store_n(&g->failed, 1u, __ATOMIC_RELEASE);
        salt_tensor_host_graph_barrier(g);
        if (worker == owner) {
            if (ops && ops->finish(g->node_seat) != 0) c->collective_error = -1;
            g->node_ops = NULL; g->node_seat = NULL;
        }
        salt_tensor_host_graph_barrier(g);
        int failed = c->collective_error || __atomic_load_n(&g->failed, __ATOMIC_ACQUIRE);
        salt_tensor_host_graph_barrier(g);
        if (failed) return -1;
        done += take;
    }
    if (worker == owner) c->stats.projection_dispatches++;
    return 0;
}

typedef struct CpuInterpretCall {
    SaltTextVerifyCpuContext *context;
    uint32_t source;
    const int32_t *tokens;
    uint32_t rows, output_rows;
} CpuInterpretCall;

static int cpu_interpret_scope(void *opaque) {
    CpuInterpretCall *call = opaque;
    return cpu_interpret_nodes(call->context, call->source, call->tokens,
                               call->rows, call->output_rows);
}

static int cpu_interpret(SaltTextVerifyCpuContext *context,
                         uint32_t source, const int32_t *tokens,
                         uint32_t rows, uint32_t output_rows) {
    CpuInterpretCall call = {context, source, tokens, rows, output_rows};
    if (context->collective_enabled) {
        SaltTensorHostGraphResult result;
        context->collective_output_rows = output_rows;
        context->coalescing_enabled = 0;
        int rc = salt_tensor_host_graph_collective_execute(&context->graph_state,
            context->program->dispatch.cell_count, cpu_collective_cell, context,
            context->parallel_workers, context->parallel_run, context->parallel_context, &result);
        context->stats.cpu_graph_sessions = 1u;
        context->stats.cpu_graph_nodes = result.nodes_executed;
        context->stats.cpu_graph_barriers = result.internal_barriers;
        return rc;
    }
    return context->parallel_scope
        ? context->parallel_scope(context->parallel_context,
                                   cpu_interpret_scope, &call)
        : cpu_interpret_scope(&call);
}

static int cpu_submit_output(void *opaque, const SaltTextVerifyProgram *program,
                      uint64_t generation, uint32_t source_position,
                      const int32_t *candidate_token_ids,
                      uint32_t candidate_count, uint32_t output_rows) {
    SaltTextVerifyCpuContext *context =
        (SaltTextVerifyCpuContext *)opaque;
    uint64_t clear_started, setup_started;
    uint32_t score_rows;
    if (!context || !context->ready || context->program != program ||
        context->assignment_plan.program != program ||
        context->assignment_plan.execution_class != SALT_TEXT_EXECUTION_CPU_ONLY ||
        generation == 0 || !candidate_token_ids || candidate_count == 0 ||
        candidate_count > program->maximum_candidates || output_rows > candidate_count ||
        (output_rows != candidate_count && program->target_policy &&
         program->target_policy->cpu_graph))
        return -1;
    if (salt_text_attention_score_rows(program, source_position,
            candidate_count - 1u, &score_rows) != 0)
        return -1;
    memset(&context->stats, 0, sizeof context->stats);
    clear_started = context->profile_enabled ? cpu_now_ns() : 0;
    if (salt_text_touched_span_plan(program, candidate_count, score_rows,
            context->touched_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
            &context->touched_span_count, &context->touched_span_bytes) != 0 ||
        salt_text_touched_span_clear(context->arena, context->arena_bytes,
            context->touched_spans, context->touched_span_count,
            &context->stats.canonical_clear_bytes) != 0)
        return -1;
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_clear_ns, clear_started);
    setup_started = context->profile_enabled ? cpu_now_ns() : 0;
    memcpy(context->arena + program->layout.candidate_token_ids,
           candidate_token_ids,
           (size_t)candidate_count * sizeof *candidate_token_ids);
    context->submitted_generation = generation;
    context->submitted_source_position = source_position;
    context->submitted_candidates = candidate_count;
    context->submitted_frontier = 0;
    context->coalesced_attention_layer = UINT32_MAX;
    context->coalesced_dense_layer = UINT32_MAX;
    context->coalesced_expert_layer = UINT32_MAX;
    context->frontier_expert_layer = UINT32_MAX;
    context->coalescing_enabled = 1;
    context->submitted = 1;
    context->finished = 0;
    context->stats.engine_submissions = 1u;
    context->stats.initial_transfer_bytes =
        (uint64_t)candidate_count * sizeof(int32_t);
    context->stats.final_logits_transfer_bytes =
        (uint64_t)output_rows * program->vocabulary * sizeof(float);
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_setup_ns, setup_started);
    if (cpu_interpret(context, source_position,
            (const int32_t *)(const void *)(context->arena +
                program->layout.candidate_token_ids), candidate_count, output_rows) != 0)
        return -1;
    return 0;
}

static int cpu_submit(void *opaque, const SaltTextVerifyProgram *program,
                      uint64_t generation, uint32_t source_position,
                      const int32_t *candidate_token_ids,
                      uint32_t candidate_count) {
    return cpu_submit_output(opaque, program, generation, source_position,
        candidate_token_ids, candidate_count, candidate_count);
}

static int cpu_submit_frontier(
        void *opaque, const SaltTextVerifyProgram *program,
        uint32_t source_position,
        const SaltTextTargetFrontier *frontier) {
    SaltTextVerifyCpuContext *context =
        (SaltTextVerifyCpuContext *)opaque;
    int32_t *tokens;
    uint32_t *parents, *depths;
    uint64_t clear_started, setup_started;
    uint32_t maximum_depth = 0, score_rows;
    if (!context || !context->ready || context->program != program ||
        context->assignment_plan.program != program ||
        context->assignment_plan.execution_class != SALT_TEXT_EXECUTION_CPU_ONLY ||
        !frontier || frontier->node_count == 0 ||
        frontier->node_count > program->maximum_candidates ||
        salt_text_target_frontier_validate(
            frontier, frontier->node_count) != 0)
        return -1;
    for (uint32_t index = 0; index < frontier->node_count; index++)
        if (frontier->nodes[index].depth > maximum_depth)
            maximum_depth = frontier->nodes[index].depth;
    if (salt_text_attention_score_rows(program, source_position,
            maximum_depth, &score_rows) != 0)
        return -1;
    memset(&context->stats, 0, sizeof context->stats);
    clear_started = context->profile_enabled ? cpu_now_ns() : 0;
    if (salt_text_touched_span_plan(program, frontier->node_count, score_rows,
            context->touched_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
            &context->touched_span_count, &context->touched_span_bytes) != 0 ||
        salt_text_touched_span_clear(context->arena, context->arena_bytes,
            context->touched_spans, context->touched_span_count,
            &context->stats.canonical_clear_bytes) != 0)
        return -1;
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_clear_ns, clear_started);
    setup_started = context->profile_enabled ? cpu_now_ns() : 0;
    tokens = cpu_i32(context, program->layout.candidate_token_ids);
    parents = (uint32_t *)(void *)(
        context->arena + program->layout.target_parent_rows);
    depths = (uint32_t *)(void *)(
        context->arena + program->layout.target_depths);
    for (uint32_t index = 0; index < frontier->node_count; index++) {
        const SaltTextTargetNode *node = &frontier->nodes[index];
        uint32_t row = node->tentative_state_slot;
        tokens[row] = node->token_id;
        depths[row] = node->depth;
        parents[row] = node->parent_index == SALT_TEXT_TARGET_NO_PARENT
            ? UINT32_MAX
            : frontier->nodes[node->parent_index].tentative_state_slot;
    }
    context->submitted_generation = frontier->generation;
    context->submitted_source_position = source_position;
    context->submitted_candidates = frontier->node_count;
    context->submitted_frontier = 1;
    context->coalesced_attention_layer = UINT32_MAX;
    context->coalesced_dense_layer = UINT32_MAX;
    context->coalesced_expert_layer = UINT32_MAX;
    context->frontier_expert_layer = UINT32_MAX;
    context->coalescing_enabled = 1;
    context->submitted = 1;
    context->finished = 0;
    context->stats.engine_submissions = 1u;
    context->stats.initial_transfer_bytes =
        (uint64_t)frontier->node_count *
        (sizeof(int32_t) + 2u * sizeof(uint32_t));
    context->stats.final_logits_transfer_bytes =
        (uint64_t)frontier->node_count * program->vocabulary * sizeof(float);
    if (context->profile_enabled)
        cpu_add_elapsed(&context->stats.cpu_target_setup_ns, setup_started);
    return cpu_interpret(
        context, source_position, tokens, frontier->node_count, frontier->node_count);
}

static int cpu_finish(void *opaque, const SaltTextVerifyProgram *program,
                      SaltTextExecutionView *view,
                      SaltTextVerifyBackendStats *stats) {
    SaltTextVerifyCpuContext *context =
        (SaltTextVerifyCpuContext *)opaque;
    if (!context || !program || !view || !stats || !context->submitted ||
        context->finished || context->program != program)
        return -1;
    context->stats.completion_fences = 1u;
    context->stats.intermediate_host_publications = 0u;
    view->canonical_base = context->arena;
    view->canonical_bytes = program->layout.total_bytes;
    *stats = context->stats;
    context->finished = 1;
    return 0;
}

static int cpu_resolve(void *opaque, const SaltTextVerifyProgram *program,
                       uint64_t generation, uint32_t source_position,
                       uint32_t accepted_count, uint32_t candidate_count,
                       uint64_t *scrubbed_bytes) {
    SaltTextVerifyCpuContext *context =
        (SaltTextVerifyCpuContext *)opaque;
    if (!context || !program || !scrubbed_bytes ||
        context->program != program || !context->submitted ||
        !context->finished || generation != context->submitted_generation ||
        source_position != context->submitted_source_position ||
        candidate_count != context->submitted_candidates ||
        accepted_count > candidate_count)
        return -1;
    *scrubbed_bytes = 0;
    return 0;
}

static int cpu_scrub(void *opaque, const SaltTextVerifyProgram *program,
                     uint64_t generation, uint64_t *scrubbed_bytes) {
    SaltTextVerifyCpuContext *context =
        (SaltTextVerifyCpuContext *)opaque;
    uint64_t cleared = 0;
    if (!context || !program || !scrubbed_bytes ||
        context->program != program || generation == 0)
        return -1;
    if (cpu_finish_host_node(context) != 0) return -1;
    if (context->touched_span_count == 0 ||
        salt_text_touched_span_clear(context->arena, context->arena_bytes,
            context->touched_spans, context->touched_span_count,
            &cleared) != 0)
        return -1;
    context->submitted = 0;
    context->finished = 0;
    context->submitted_candidates = 0;
    context->submitted_frontier = 0;
    *scrubbed_bytes = program->tentative_kv_bytes;
    return 0;
}

int salt_text_verify_cpu_profile_set(
        SaltTextVerifyCpuContext *context, int enabled) {
    if (!context || !context->ready || (enabled != 0 && enabled != 1) ||
        context->submitted)
        return -1;
    context->profile_enabled = enabled;
    return 0;
}

static const SaltTextVerifyExecutorOps cpu_executor_ops = {
    cpu_submit,
    cpu_submit_frontier,
    cpu_finish,
    cpu_resolve,
    cpu_scrub,
    cpu_submit,
    cpu_submit_output
};

int salt_text_verify_cpu_executor_init(SaltTextVerifyExecutor *executor,
                                       SaltTextVerifyCpuContext *context) {
    SaltTextVerifyExecutor built;
    if (!executor || !context || !context->ready || !context->program ||
        context->assignment_plan.program != context->program ||
        context->assignment_plan.execution_class !=
            SALT_TEXT_EXECUTION_CPU_ONLY)
        return -1;
    memset(&built, 0, sizeof built);
    built.program = context->program;
    built.plan = &context->assignment_plan;
    built.ops = &cpu_executor_ops;
    built.context = context;
    *executor = built;
    return 0;
}
