#include "salt/text_token.h"

#include <stdio.h>
#include <string.h>

typedef struct FixtureContext {
    int begin_calls;
    int cell_calls;
    int barrier_calls;
    int finish_calls;
    int abort_calls;
    int publish_calls;
    int fail_cell;
    int need_cell;
    int fail_publish;
    int prepare_result;
    int prepare_calls;
    uint32_t prepare_current;
    uint32_t prepare_next;
    SaltTextTokenStatus abort_status;
    uint32_t abort_completed;
    uint32_t published_position;
} FixtureContext;

static int fixture_begin(void *opaque, const SaltTextTokenProgram *program,
                         const SaltTextTokenExecution *execution) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution) return -1;
    context->begin_calls++;
    return 0;
}

static SaltTextTokenCellResult fixture_cell(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution || !cell) return SALT_TEXT_TOKEN_CELL_FAILED;
    context->cell_calls++;
    if ((int)cell_index == context->need_cell)
        return SALT_TEXT_TOKEN_CELL_NEED_RESOURCE;
    if ((int)cell_index == context->fail_cell)
        return SALT_TEXT_TOKEN_CELL_FAILED;
    return SALT_TEXT_TOKEN_CELL_OK;
}

static int fixture_barrier(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index) {
    FixtureContext *context = (FixtureContext *)opaque;
    (void)cell_index;
    if (!context || !program || !execution || !cell) return -1;
    context->barrier_calls++;
    return 0;
}

static int fixture_finish(void *opaque, const SaltTextTokenProgram *program,
                          const SaltTextTokenExecution *execution) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution) return -1;
    context->finish_calls++;
    return 0;
}

static int fixture_abort(void *opaque, const SaltTextTokenProgram *program,
                         const SaltTextTokenExecution *execution,
                         SaltTextTokenStatus status, uint32_t completed) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution) return -1;
    context->abort_calls++;
    context->abort_status = status;
    context->abort_completed = completed;
    return 0;
}

static int fixture_publish(void *opaque, const SaltTextTokenProgram *program,
                           const SaltTextTokenExecution *execution,
                           uint32_t result_position) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution) return -1;
    context->publish_calls++;
    context->published_position = result_position;
    return context->fail_publish ? -1 : 0;
}

static int fixture_prepare_next_layer(
        void *opaque, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        uint32_t current_layer, uint32_t next_layer) {
    FixtureContext *context = (FixtureContext *)opaque;
    if (!context || !program || !execution) return -1;
    context->prepare_calls++;
    context->prepare_current = current_layer;
    context->prepare_next = next_layer;
    return context->prepare_result;
}

static const SaltTextTokenExecutorOps fixture_ops = {
    fixture_begin,
    fixture_cell,
    fixture_barrier,
    fixture_finish,
    fixture_abort,
    fixture_publish,
    fixture_prepare_next_layer,
};

static const SaltTextTokenCell valid_cells[] = {
    {SALT_TEXT_TOKEN_EMBED, -1, 0, -1, 1,
     SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE, 0, 0, 16},
    {SALT_TEXT_TOKEN_QKV, 0, 0, 0, 2,
     SALT_TEXT_TOKEN_CELL_BARRIER_AFTER |
         SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
     16, 16, 32},
    {SALT_TEXT_TOKEN_EXPERT_RESOURCE_CHECK, 0, 0, 1, 3,
     SALT_TEXT_TOKEN_CELL_RESOURCE_CHECK, 48, 48, 0},
    {SALT_TEXT_TOKEN_EXPERT_GATE_UP, 0, 0, 2, 4,
     SALT_TEXT_TOKEN_CELL_BARRIER_AFTER |
         SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE,
     48, 48, 64},
    {SALT_TEXT_TOKEN_FINALIZE, -1, 0, 3, 5,
     SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE, 112, 112, 8},
};

static void fixture_reset(FixtureContext *context) {
    memset(context, 0, sizeof *context);
    context->fail_cell = -1;
    context->need_cell = -1;
}

int main(void) {
    SaltTextTokenProgramDesc descriptor = {
        valid_cells,
        (uint32_t)(sizeof valid_cells / sizeof valid_cells[0]),
        1,
        2,
    };
    SaltTextTokenProgram program;
    SaltTextTokenCell source_cells[sizeof valid_cells / sizeof valid_cells[0]];
    unsigned char arena[sizeof valid_cells];
    SaltTextTokenExecution execution = {7, 200, 818};
    SaltTextTokenResult result;
    FixtureContext context;
    SaltTextTokenExecutor executor = {&fixture_ops, &context, 0};
    size_t required = 0;

    memcpy(source_cells, valid_cells, sizeof source_cells);
    descriptor.cells = source_cells;
    if (salt_text_token_program_arena_requirement(&descriptor, &required) != 0 ||
        required != sizeof valid_cells ||
        salt_text_token_program_compile(
            &program, &descriptor, arena, sizeof arena) != 0)
        return 1;
    source_cells[0].kind = SALT_TEXT_TOKEN_SOFTCAP;
    if (program.cells[0].kind != SALT_TEXT_TOKEN_EMBED ||
        program.canonical_write_bytes != 120)
        return 1;

    fixture_reset(&context);
    if (salt_text_token_execute(&program, &executor, &execution, &result) != 0 ||
        result.status != SALT_TEXT_TOKEN_COMMITTED ||
        result.source_position != 200 || result.result_position != 201 ||
        result.input_token_id != 818 || result.transition_generation != 7 ||
        result.backend.caller_submissions != 1 ||
        result.backend.cells_completed != 5 ||
        result.backend.internal_dependency_barriers != 2 ||
        result.backend.completion_fences != 1 ||
        result.backend.intermediate_host_publications != 0 ||
        result.backend.final_publications != 1 ||
        result.backend.tentative_write_bytes != 120 ||
        context.begin_calls != 1 || context.cell_calls != 5 ||
        context.barrier_calls != 2 || context.finish_calls != 1 ||
        context.abort_calls != 0 || context.publish_calls != 1 ||
        context.published_position != 201)
        return 1;
    puts("token program committed transaction: PASS");

    fixture_reset(&context);
    context.need_cell = 2;
    if (salt_text_token_execute(&program, &executor, &execution, &result) != 0 ||
        result.status != SALT_TEXT_TOKEN_NEED_RESOURCE ||
        result.resource_cell != 2 || result.result_position != 200 ||
        result.backend.cells_completed != 2 ||
        result.backend.resource_retries != 1 ||
        result.backend.final_publications != 0 ||
        context.abort_calls != 1 ||
        context.abort_status != SALT_TEXT_TOKEN_NEED_RESOURCE ||
        context.abort_completed != 2 || context.finish_calls != 0 ||
        context.publish_calls != 0)
        return 1;
    puts("token program resource retry without publication: PASS");

    fixture_reset(&context);
    context.fail_cell = 3;
    if (salt_text_token_execute(&program, &executor, &execution, &result) == 0 ||
        result.status != SALT_TEXT_TOKEN_EXECUTOR_FAILURE ||
        context.abort_calls != 1 || context.abort_completed != 3 ||
        context.publish_calls != 0)
        return 1;
    puts("token program backend failure abort: PASS");

    fixture_reset(&context);
    context.need_cell = 1;
    if (salt_text_token_execute(&program, &executor, &execution, &result) == 0 ||
        result.status != SALT_TEXT_TOKEN_FATAL || context.abort_calls != 1 ||
        context.publish_calls != 0)
        return 1;
    puts("token program illegal resource retry: PASS");

    fixture_reset(&context);
    context.fail_publish = 1;
    if (salt_text_token_execute(&program, &executor, &execution, &result) == 0 ||
        result.status != SALT_TEXT_TOKEN_FATAL ||
        result.backend.completion_fences != 1 ||
        result.backend.final_publications != 0 ||
        context.finish_calls != 1 || context.publish_calls != 1 ||
        context.abort_calls != 1)
        return 1;
    puts("token program final publication failure: PASS");

    {
        const SaltTextTokenCell lookahead_cells[] = {
            {SALT_TEXT_TOKEN_EMBED, -1, 0, -1, 1, 0, 0, 0, 0},
            {SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT, 0, 0, 0, 2,
             0, 0, 0, 0},
            {SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT, 0, 0, 1, 3,
             0, 0, 0, 0},
            {SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT, 1, 0, 2, 4,
             0, 0, 0, 0},
            {SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT, 1, 0, 3, 5,
             0, 0, 0, 0},
            {SALT_TEXT_TOKEN_FINALIZE, -1, 0, 4, 6, 0, 0, 0, 0},
        };
        SaltTextTokenProgramDesc lookahead_descriptor = {
            lookahead_cells,
            (uint32_t)(sizeof lookahead_cells / sizeof lookahead_cells[0]),
            2, 0,
        };
        SaltTextTokenProgram lookahead_program;
        unsigned char lookahead_arena[sizeof lookahead_cells];
        SaltTextTokenExecutor lookahead_executor = {
            &fixture_ops, &context, 1,
        };
        fixture_reset(&context);
        if (salt_text_token_program_compile(
                &lookahead_program, &lookahead_descriptor,
                lookahead_arena, sizeof lookahead_arena) != 0 ||
            salt_text_token_execute(
                &lookahead_program, &lookahead_executor,
                &execution, &result) != 0 ||
            result.status != SALT_TEXT_TOKEN_COMMITTED ||
            context.prepare_calls != 1 || context.prepare_current != 0 ||
            context.prepare_next != 1)
            return 1;
        fixture_reset(&context);
        context.prepare_result = 1;
        if (salt_text_token_execute(
                &lookahead_program, &lookahead_executor,
                &execution, &result) != 0 ||
            result.status != SALT_TEXT_TOKEN_COMMITTED ||
            context.prepare_calls != 1)
            return 1;
        fixture_reset(&context);
        context.prepare_result = -1;
        if (salt_text_token_execute(
                &lookahead_program, &lookahead_executor,
                &execution, &result) == 0 ||
            result.status != SALT_TEXT_TOKEN_EXECUTOR_FAILURE ||
            context.prepare_calls != 1 || context.abort_calls != 1 ||
            context.abort_completed != 1)
            return 1;
        puts("token program rolling layer lookahead: PASS");
    }

    {
        SaltTextTokenCell illegal[2] = {
            valid_cells[0], valid_cells[1],
        };
        SaltTextTokenProgramDesc bad = {illegal, 2, 1, 1};
        illegal[1].dependency_cell = 1;
        if (salt_text_token_program_arena_requirement(&bad, &required) == 0 ||
            salt_text_token_program_compile(
                &program, &bad, arena, sizeof arena) == 0)
            return 1;
    }
    if (salt_text_token_program_compile(
            &program, &descriptor, arena, sizeof arena - 1) == 0)
        return 1;
    puts("token program compile guards: PASS");
    return 0;
}
