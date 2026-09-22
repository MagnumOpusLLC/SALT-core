#include "salt/text_token.h"
#include "salt/tensorops.h"

#include <limits.h>
#include <string.h>

static int token_program_validate(
        const SaltTextTokenProgramDesc *descriptor,
        uint64_t *write_bytes_out, uint32_t *barrier_count_out) {
    uint64_t write_bytes = 0;
    uint32_t barriers = 0, previous_completion = 0;
    if (!descriptor || !write_bytes_out || !barrier_count_out ||
        !descriptor->cells || descriptor->cell_count == 0 ||
        descriptor->layer_count == 0)
        return -1;
    for (uint32_t index = 0; index < descriptor->cell_count; index++) {
        const SaltTextTokenCell *cell = &descriptor->cells[index];
        if (cell->kind < 0 || cell->kind >= SALT_TEXT_TOKEN_CELL_KIND_COUNT ||
            cell->layer < -1 ||
            (cell->layer >= 0 &&
             (uint32_t)cell->layer >= descriptor->layer_count) ||
            cell->dependency_cell < -1 ||
            (cell->dependency_cell >= 0 &&
             (uint32_t)cell->dependency_cell >= index) ||
            cell->completion_id == 0 ||
            cell->completion_id <= previous_completion ||
            cell->flags & ~(SALT_TEXT_TOKEN_CELL_BARRIER_AFTER |
                            SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE |
                            SALT_TEXT_TOKEN_CELL_RESOURCE_CHECK) ||
            cell->destination_offset > UINT64_MAX - cell->destination_bytes ||
            write_bytes > UINT64_MAX - cell->destination_bytes)
            return -1;
        previous_completion = cell->completion_id;
        write_bytes += cell->destination_bytes;
        if (cell->flags & SALT_TEXT_TOKEN_CELL_BARRIER_AFTER) barriers++;
    }
    if (barriers > descriptor->maximum_internal_barriers) return -1;
    *write_bytes_out = write_bytes;
    *barrier_count_out = barriers;
    return 0;
}

int salt_text_token_program_arena_requirement(
        const SaltTextTokenProgramDesc *descriptor, size_t *bytes_out) {
    uint64_t write_bytes;
    uint32_t barriers;
    if (!bytes_out || token_program_validate(
            descriptor, &write_bytes, &barriers) != 0 ||
        sizeof(SaltTextTokenCell) >
            SIZE_MAX / (size_t)descriptor->cell_count)
        return -1;
    (void)write_bytes;
    (void)barriers;
    *bytes_out = (size_t)descriptor->cell_count * sizeof(SaltTextTokenCell);
    return 0;
}

int salt_text_token_program_compile(
        SaltTextTokenProgram *program,
        const SaltTextTokenProgramDesc *descriptor,
        void *startup_arena, size_t startup_arena_bytes) {
    uint64_t write_bytes;
    uint32_t barriers;
    size_t required;
    if (!program || !startup_arena || token_program_validate(
            descriptor, &write_bytes, &barriers) != 0 ||
        salt_text_token_program_arena_requirement(descriptor, &required) != 0 ||
        startup_arena_bytes < required)
        return -1;
    memset(program, 0, sizeof *program);
    memcpy(startup_arena, descriptor->cells, required);
    program->cells = (const SaltTextTokenCell *)startup_arena;
    program->cell_count = descriptor->cell_count;
    program->layer_count = descriptor->layer_count;
    program->maximum_internal_barriers =
        descriptor->maximum_internal_barriers;
    program->canonical_write_bytes = write_bytes;
    program->ready = 1;
    return 0;
}

static int token_abort(
        SaltTextTokenExecutor *executor,
        const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        SaltTextTokenStatus status, uint32_t completed,
        SaltTextTokenResult *result) {
    if (executor->ops->abort(executor->context, program, execution,
            status, completed) != 0) {
        result->status = SALT_TEXT_TOKEN_FATAL;
        return -1;
    }
    result->status = status;
    return status == SALT_TEXT_TOKEN_NEED_RESOURCE ? 0 : -1;
}

typedef struct SaltTextTokenProgramRun {
    const SaltTextTokenProgram *program;
    SaltTextTokenExecutor *executor;
    const SaltTextTokenExecution *execution;
    SaltTextTokenResult *result;
    uint32_t completed;
    SaltTextTokenStatus failure_status;
} SaltTextTokenProgramRun;

static int token_execute_node(void *opaque, uint32_t index) {
    SaltTextTokenProgramRun *run = (SaltTextTokenProgramRun *)opaque;
    const SaltTextTokenCell *cell;
    SaltTextTokenCellResult rc;
    if (!run || !run->program || !run->executor || !run->execution ||
        !run->result || index >= run->program->cell_count)
        return -1;
    cell = &run->program->cells[index];
    if (cell->dependency_cell >= 0 &&
        (uint32_t)cell->dependency_cell >= run->completed) {
        run->failure_status = SALT_TEXT_TOKEN_FATAL;
        return -1;
    }
    if (run->executor->layer_lookahead &&
            cell->kind == SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT &&
            cell->layer >= 0 &&
            (uint32_t)cell->layer + 1u < run->program->layer_count) {
        int prepared;
        if (!run->executor->ops->prepare_next_layer) {
            run->failure_status = SALT_TEXT_TOKEN_FATAL;
            return -1;
        }
        prepared = run->executor->ops->prepare_next_layer(
            run->executor->context, run->program, run->execution,
            (uint32_t)cell->layer, (uint32_t)cell->layer + 1u);
        if (prepared < 0) {
            run->failure_status = SALT_TEXT_TOKEN_EXECUTOR_FAILURE;
            return -1;
        }
    }
    rc = run->executor->ops->execute_cell(
        run->executor->context, run->program, run->execution, cell, index);
    if (rc == SALT_TEXT_TOKEN_CELL_NEED_RESOURCE) {
        if (!(cell->flags & SALT_TEXT_TOKEN_CELL_RESOURCE_CHECK)) {
            run->failure_status = SALT_TEXT_TOKEN_FATAL;
            return -1;
        }
        run->result->backend.resource_retries = 1;
        run->result->resource_cell = (int32_t)index;
        run->failure_status = SALT_TEXT_TOKEN_NEED_RESOURCE;
        return -1;
    }
    if (rc != SALT_TEXT_TOKEN_CELL_OK) {
        run->failure_status = SALT_TEXT_TOKEN_EXECUTOR_FAILURE;
        return -1;
    }
    run->completed++;
    run->result->backend.cells_completed = run->completed;
    if (cell->flags & SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE) {
        if (run->result->backend.tentative_write_bytes >
                UINT64_MAX - cell->destination_bytes) {
            run->failure_status = SALT_TEXT_TOKEN_FATAL;
            return -1;
        }
        run->result->backend.tentative_write_bytes += cell->destination_bytes;
    }
    if (cell->flags & SALT_TEXT_TOKEN_CELL_BARRIER_AFTER) {
        if (run->executor->ops->dependency_barrier(
                run->executor->context, run->program, run->execution,
                cell, index) != 0) {
            run->failure_status = SALT_TEXT_TOKEN_EXECUTOR_FAILURE;
            return -1;
        }
        run->result->backend.internal_dependency_barriers++;
    }
    return 0;
}

int salt_text_token_execute(
        const SaltTextTokenProgram *program,
        SaltTextTokenExecutor *executor,
        const SaltTextTokenExecution *execution,
        SaltTextTokenResult *result) {
    SaltTextTokenProgramRun run;
    SaltTensorProgramCallbacks callbacks;
    uint32_t executed_nodes = 0;
    if (!result) return -1;
    memset(result, 0, sizeof *result);
    result->resource_cell = -1;
    if (!program || !program->ready || !program->cells ||
        program->cell_count == 0 || !executor || !executor->ops ||
        !execution || execution->source_position == UINT32_MAX ||
        !executor->ops->begin || !executor->ops->execute_cell ||
        !executor->ops->dependency_barrier || !executor->ops->finish ||
        !executor->ops->abort || !executor->ops->publish)
        return -1;
    if (executor->layer_lookahead && !executor->ops->prepare_next_layer)
        return -1;
    result->source_position = execution->source_position;
    result->result_position = execution->source_position;
    result->input_token_id = execution->input_token_id;
    result->transition_generation = execution->transition_generation;
    if (executor->ops->begin(
            executor->context, program, execution) != 0) {
        result->status = SALT_TEXT_TOKEN_EXECUTOR_FAILURE;
        return -1;
    }
    result->backend.caller_submissions = 1;
    run = (SaltTextTokenProgramRun) {
        program, executor, execution, result, 0u, SALT_TEXT_TOKEN_FATAL,
    };
    callbacks = (SaltTensorProgramCallbacks) {
        program->cell_count, token_execute_node,
    };
    if (salt_tensor_program_execute(
            &callbacks, &run, &executed_nodes) != 0)
        return token_abort(executor, program, execution,
            run.failure_status, run.completed, result);
    if (executor->ops->finish(executor->context, program, execution) != 0)
        return token_abort(executor, program, execution,
            SALT_TEXT_TOKEN_EXECUTOR_FAILURE, run.completed, result);
    result->backend.completion_fences = 1;
    if (run.completed != program->cell_count ||
        executed_nodes != program->cell_count ||
        result->backend.caller_submissions != 1 ||
        result->backend.intermediate_host_publications != 0 ||
        result->backend.completion_fences != 1)
        return token_abort(executor, program, execution,
            SALT_TEXT_TOKEN_FATAL, run.completed, result);
    if (executor->ops->publish(executor->context, program, execution,
            execution->source_position + 1u) != 0)
        return token_abort(executor, program, execution,
            SALT_TEXT_TOKEN_FATAL, run.completed, result);
    result->backend.final_publications = 1;
    result->result_position = execution->source_position + 1u;
    result->status = SALT_TEXT_TOKEN_COMMITTED;
    return 0;
}
