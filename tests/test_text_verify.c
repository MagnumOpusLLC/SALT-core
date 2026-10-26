#include "salt/text_verify.h"
#include "salt/gpu.h"
#include "salt/moe.h"

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "text verify: %s\n", message); \
        return 1; \
    } \
} while (0)

int salt_hc_params(const SaltTrunkLayout *layout, int fn_index,
        int base_index, int scale_index, const uint8_t *trunk, int hidden,
        const float *state, int *count, float *a, float *c, float *b) {
    (void)layout;
    (void)fn_index;
    (void)base_index;
    (void)scale_index;
    (void)trunk;
    (void)hidden;
    (void)state;
    (void)count;
    (void)a;
    (void)c;
    (void)b;
    return 0;
}

void salt_hc_combine(int count, int hidden, const float *a,
        const float *state, float *output) {
    (void)count;
    (void)hidden;
    (void)a;
    (void)state;
    (void)output;
}

enum {
    LAYERS = 2,
    HIDDEN = 4,
    HEADS = 2,
    KV_HEADS = 1,
    HEAD_DIM = 2,
    QUERY_WIDTH = HEADS * HEAD_DIM,
    KV_WIDTH = KV_HEADS * HEAD_DIM,
    EXPERTS = 3,
    TOPK = 2,
    EXPERT_WIDTH = 3,
    DENSE_WIDTH = 5,
    VOCABULARY = 7,
    CANDIDATES = 4,
    MAXIMUM_CONTEXT = 12,
    SHARED_ROWS = 2,
    PRIVATE_ROWS = MAXIMUM_CONTEXT - SHARED_ROWS,
    RING_ROWS = 5,
    MAX_TENSOR_VALUES = 32,
    MAX_BINDINGS = 64,
    MAX_EXECUTION_CELLS = 64,
    PROGRAM_ARENA_BYTES = 16384,
    CPU_ARENA_BYTES = 65536
};

typedef struct SyntheticPackage SyntheticPackage;

typedef struct F32Binding {
    SyntheticPackage *owner;
    const float *values;
    uint32_t rows;
    uint32_t cols;
    size_t scratch_bytes;
} F32Binding;

struct SyntheticPackage {
    SaltModelDesc model;
    SaltTextLayerPlan base_plans[LAYERS];
    SaltTextLayerExecDesc weight_layers[LAYERS];
    SaltTextExpertDesc experts[LAYERS][EXPERTS];
    SaltTextTensorDesc embedding;
    SaltTextTensorDesc final_norm;
    F32Binding bindings[MAX_BINDINGS];
    float values[MAX_BINDINGS][MAX_TENSOR_VALUES];
    SaltTensorResourceSpec resources[1];
    uint32_t binding_count;
    uint64_t fail_handle;
    uint32_t operation_calls;
    uint32_t scratch_queries;
};

typedef struct DescriptorSnapshot {
    SaltModelDesc salt_model;
    SaltTextTensorDesc package_embedding;
    SaltTextTensorDesc package_final_norm;
    SaltTextModelExecDesc model;
    SaltTextLayerPlan plans[LAYERS];
    SaltTextLayerExecDesc layers[LAYERS];
    SaltTextKvLayerDesc kv[LAYERS];
    SaltTextExpertDesc experts[LAYERS][EXPERTS];
} DescriptorSnapshot;

typedef struct KvSnapshot {
    float keys[LAYERS][PRIVATE_ROWS * KV_WIDTH];
    float values[LAYERS][PRIVATE_ROWS * KV_WIDTH];
    float shared_keys[LAYERS][SHARED_ROWS * KV_WIDTH];
    float shared_values[LAYERS][SHARED_ROWS * KV_WIDTH];
} KvSnapshot;

typedef struct ExecutionFixture {
    SaltTextLayerPlan plans[LAYERS];
    SaltTextLayerExecDesc layers[LAYERS];
    SaltTextKvLayerDesc kv[LAYERS];
    SaltTextModelExecDesc model;
    SaltTextKvState state;
    SaltTextTargetPolicy target_policy;
    SaltTextVerifyProgram program;
    SaltTextVerifyCpuContext cpu;
    SaltTextVerifyExecutor executor;
    float private_keys[LAYERS][PRIVATE_ROWS * KV_WIDTH];
    float private_values[LAYERS][PRIVATE_ROWS * KV_WIDTH];
    float shared_keys[LAYERS][SHARED_ROWS * KV_WIDTH];
    float shared_values[LAYERS][SHARED_ROWS * KV_WIDTH];
    unsigned char program_arena[PROGRAM_ARENA_BYTES];
    unsigned char cpu_arena[CPU_ARENA_BYTES];
    size_t program_need;
    size_t cpu_need;
} ExecutionFixture;

typedef enum FixtureTensorKind {
    FIXTURE_MATRIX = 1,
    FIXTURE_NORM = 2,
    FIXTURE_SCALE = 3,
    FIXTURE_SCALAR = 4,
    FIXTURE_EMBEDDING = 5
} FixtureTensorKind;

static SyntheticPackage package_fixture;
static ExecutionFixture serial_fixture;
static ExecutionFixture all_fixture;
static ExecutionFixture block_fixture;
static ExecutionFixture deterministic_fixture;
static ExecutionFixture progressive_fixture;
static ExecutionFixture progressive_reject_fixture;
static ExecutionFixture frontier_fixture;
static ExecutionFixture frontier_root_fixture;
static ExecutionFixture frontier_cascade_fixture;
static ExecutionFixture frontier_cascade_reject_fixture;
static ExecutionFixture route_fixture;
static ExecutionFixture route_reject_fixture;
static ExecutionFixture route_root_miss_fixture;
static ExecutionFixture reject_fixture;
static ExecutionFixture failure_fixture;
static ExecutionFixture probe_fixture;
static ExecutionFixture heterogeneous_fixture;
static ExecutionFixture heterogeneous_shadow_fixture;
static ExecutionFixture operator_contract_fixture;

static int attention_worker_calls;
static float attention_worker_scores[2u * MAXIMUM_CONTEXT];
static SaltTextVerifyCpuContext *cleanup_test_context;
static uint64_t cleanup_test_handle;
static int cleanup_test_opens, cleanup_test_finishes, cleanup_test_fail;

static int cleanup_test_finish(void *seat) {
    if (seat != &cleanup_test_finishes) return -1;
    cleanup_test_finishes++;
    return cleanup_test_fail ? -1 : 0;
}

static const SaltTensorHostNodeOps cleanup_test_ops = {
    NULL, NULL, NULL, cleanup_test_finish
};

static int attention_test_workers(void *opaque, int workers,
        void (*worker)(int worker, void *task), void *task) {
    int *calls = (int *)opaque;
    if (!calls || !worker || !task || workers != 2) return -1;
    (*calls)++;
    /* Reverse ownership order to expose accidental cross-worker dependence. */
    for (int index = workers; index > 0; index--) worker(index - 1, task);
    return 0;
}

typedef enum TextOperatorContractClass {
    TEXT_OPERATOR_SHARED_TENSOROPS = 1,
    TEXT_OPERATOR_SHARED_ENCODE_PENDING = 2,
    TEXT_OPERATOR_REALIZATION_PENDING = 3,
    TEXT_OPERATOR_KV_VIEW_PENDING = 4
} TextOperatorContractClass;

typedef struct TextOperatorContract {
    SaltTextExecutionCellKind kind;
    SaltTextExecutionUnit unit;
    TextOperatorContractClass contract_class;
} TextOperatorContract;

static const TextOperatorContract text_operator_contract[] = {
    { SALT_TEXT_CELL_EMBEDDING, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_PRE_ATTENTION_NORM, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_QUERY_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_KEY_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_VALUE_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_ATTENTION_TRANSFORM, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_ATTENTION_BODY, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_OUTPUT_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_ATTENTION_COMBINE, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_DENSE_NORM, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_DENSE_GATE, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_DENSE_UP, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_DENSE_ACTIVATION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_DENSE_DOWN, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_ROUTER_INPUT, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_ROUTER_PROJECTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_ROUTER_TOPK, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_ROUTED_NORM, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_EXPERT_GATE, SALT_TEXT_EXECUTION_JOBS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_EXPERT_UP, SALT_TEXT_EXECUTION_JOBS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_EXPERT_ACTIVATION, SALT_TEXT_EXECUTION_JOBS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_EXPERT_DOWN, SALT_TEXT_EXECUTION_JOBS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_EXPERT_REDUCTION, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_FFN_COMBINE, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_FINAL_NORM, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
    { SALT_TEXT_CELL_FINAL_HEAD, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_TENSOROPS },
    { SALT_TEXT_CELL_LOGIT_SOFTCAP, SALT_TEXT_EXECUTION_ROWS,
      TEXT_OPERATOR_SHARED_ENCODE_PENDING },
};

typedef struct MockGpuState {
    SaltGpuSharedBuffer canonical;
    const SaltTextVerifyProgram *program;
    SaltTextVerifyExecutor *shadow;
    uint64_t generation;
    uint32_t source_position;
    uint32_t input_count;
    uint32_t encoded_cells;
    uint32_t barriers;
    int need_resource;
    int resource_resumed;
} MockGpuState;

typedef struct MockGpuCommand {
    uint32_t next_cell;
    int begun;
    int submitted;
} MockGpuCommand;

static unsigned char mock_gpu_canonical[CPU_ARENA_BYTES];
static unsigned char mock_gpu_startup_arena[CPU_ARENA_BYTES];
static SaltTextVerifyExecutor *mock_shadow_executor;
static int mock_need_resource;
static int mock_resource_acquires;
static int mock_resource_releases;
static int mock_defer_resources;
static int mock_consumers_completed;
static const SaltTextVerifyExecutorOps *publication_test_ops;

static int publication_test_finish(void *context,
        const SaltTextVerifyProgram *program, SaltTextExecutionView *view,
        SaltTextVerifyBackendStats *stats) {
    int rc = publication_test_ops->finish(context, program, view, stats);
    if (rc == 0) view->backend_published_kv = 1u;
    return rc;
}

static const F32Binding *f32_binding(const SaltTextTensorDesc *tensor) {
    const F32Binding *binding;
    if (!tensor || !tensor->binding) return NULL;
    binding = (const F32Binding *)tensor->binding;
    if (!binding->owner || !binding->values ||
        binding->rows != tensor->rows || binding->cols != tensor->cols)
        return NULL;
    return binding;
}

static size_t f32_scratch_requirement(const SaltTextTensorDesc *tensor) {
    const F32Binding *binding = f32_binding(tensor);
    if (!binding) return SIZE_MAX;
    binding->owner->scratch_queries++;
    return binding->scratch_bytes;
}

static int f32_operation_begin(const SaltTextTensorDesc *tensor,
                               void *scratch, size_t scratch_bytes,
                               const F32Binding **binding_out) {
    const F32Binding *binding = f32_binding(tensor);
    if (!binding || !binding_out || binding->scratch_bytes > scratch_bytes ||
        (binding->scratch_bytes > 0 && !scratch))
        return -1;
    binding->owner->operation_calls++;
    if (cleanup_test_context && tensor->stable_handle == cleanup_test_handle) {
        if (cleanup_test_context->graph_state.node_ops ||
            cleanup_test_context->graph_state.node_seat)
            return -1;
        cleanup_test_context->graph_state.node_ops = &cleanup_test_ops;
        cleanup_test_context->graph_state.node_seat = &cleanup_test_finishes;
        cleanup_test_opens++;
    }
    if (binding->owner->fail_handle == tensor->stable_handle)
        return -1;
    if (binding->scratch_bytes > 0)
        memset(scratch, (int)(tensor->stable_handle & 0xffu),
               binding->scratch_bytes);
    *binding_out = binding;
    return 0;
}

static int f32_gather_rows(const SaltTextTensorDesc *tensor,
                           const int32_t *row_ids, uint32_t row_count,
                           float *output, void *scratch,
                           size_t scratch_bytes) {
    const F32Binding *binding;
    if (!row_ids || !output || row_count == 0 ||
        f32_operation_begin(tensor, scratch, scratch_bytes, &binding) != 0)
        return -1;
    for (uint32_t row = 0; row < row_count; row++) {
        if (row_ids[row] < 0 || (uint32_t)row_ids[row] >= binding->rows)
            return -1;
        memcpy(output + (size_t)row * binding->cols,
               binding->values + (size_t)(uint32_t)row_ids[row] *
                   binding->cols,
               (size_t)binding->cols * sizeof(float));
    }
    return 0;
}

static void f32_matvec_body(const F32Binding *binding,
                            const float *input, float *output) {
    for (uint32_t row = 0; row < binding->rows; row++) {
        float sum = 0.0f;
        for (uint32_t column = 0; column < binding->cols; column++)
            sum += binding->values[(size_t)row * binding->cols + column] *
                input[column];
        output[row] = sum;
    }
}

static int f32_matvec(const SaltTextTensorDesc *tensor,
                      const float *input, float *output,
                      void *scratch, size_t scratch_bytes) {
    const F32Binding *binding;
    if (!input || !output ||
        f32_operation_begin(tensor, scratch, scratch_bytes, &binding) != 0)
        return -1;
    f32_matvec_body(binding, input, output);
    return 0;
}

static int f32_matvec_batch(const SaltTextTensorDesc *tensor,
                            const float *inputs, uint32_t row_count,
                            float *outputs, void *scratch,
                            size_t scratch_bytes) {
    const F32Binding *binding;
    if (!inputs || !outputs || row_count == 0 ||
        f32_operation_begin(tensor, scratch, scratch_bytes, &binding) != 0)
        return -1;
    for (uint32_t row = 0; row < row_count; row++)
        f32_matvec_body(binding,
            inputs + (size_t)row * binding->cols,
            outputs + (size_t)row * binding->rows);
    return 0;
}

static int f32_area_decline(const SaltTextTensorHostBatch *jobs,
                            uint32_t job_count, uint32_t active_rows,
                            SaltAreaScanPlan *plan, void *scratch,
                            size_t scratch_bytes) {
    (void)jobs;
    (void)job_count;
    (void)active_rows;
    (void)plan;
    (void)scratch;
    (void)scratch_bytes;
    return 1;
}

static const SaltTextTensorBackendOps f32_backend_ops = {
    f32_scratch_requirement,
    f32_gather_rows,
    f32_matvec,
    f32_matvec_batch,
    NULL,
    NULL,
    NULL
};

static const SaltTextTensorBackendOps f32_decline_backend_ops = {
    f32_scratch_requirement,
    f32_gather_rows,
    f32_matvec,
    NULL,
    NULL,
    f32_area_decline,
    NULL
};

static SaltTextTensorOps f32_tensor_ops = {
    &f32_backend_ops,
    &f32_backend_ops
};


static int mock_gpu_requirements(
        const SaltTextVerifyProgram *program,
        const SaltTextDispatchPolicy *policy,
        SaltTextGpuProgramRequirements *requirement,
        size_t requirement_size) {
    if (!program || !program->ready || !policy ||
        policy->execution_class != SALT_TEXT_EXECUTION_GPU_ONLY ||
        !requirement || requirement_size != sizeof *requirement)
        return -1;
    memset(requirement, 0, sizeof *requirement);
    requirement->backend_state_bytes = sizeof(MockGpuState);
    requirement->command_bytes = sizeof(MockGpuCommand);
    requirement->maximum_commands = program->dispatch.cell_count;
    requirement->flags = mock_defer_resources
        ? SALT_TEXT_GPU_DEFER_EXPERT_RELEASE : 0u;
    return 0;
}

static int mock_gpu_prepare(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltGpuSharedBuffer **canonical_out,
        void *backend_state, size_t backend_state_bytes,
        void *command, size_t command_bytes) {
    MockGpuState *state = (MockGpuState *)backend_state;
    if (!program || !plan || plan->program != program || !canonical_out ||
        !state || backend_state_bytes < sizeof *state || !command ||
        command_bytes < sizeof(MockGpuCommand) || !mock_shadow_executor ||
        mock_shadow_executor->program->layout.total_bytes !=
            program->layout.total_bytes ||
        program->layout.total_bytes > sizeof mock_gpu_canonical)
        return -1;
    memset(state, 0, sizeof *state);
    memset(command, 0, sizeof(MockGpuCommand));
    memset(mock_gpu_canonical, 0, sizeof mock_gpu_canonical);
    state->program = program;
    state->shadow = mock_shadow_executor;
    state->canonical.contents = mock_gpu_canonical;
    state->canonical.nbytes = program->layout.total_bytes;
    state->canonical.backend = state;
    *canonical_out = &state->canonical;
    return 0;
}

static int mock_gpu_begin(void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !state->program || !state->shadow || !cmd || cmd->begun ||
        generation == 0 || !input_token_ids || input_count == 0 ||
        state->shadow->ops->submit(state->shadow->context,
            state->shadow->program, generation, source_position,
            input_token_ids, input_count) != 0)
        return -1;
    state->generation = generation;
    state->source_position = source_position;
    state->input_count = input_count;
    state->encoded_cells = 0;
    state->barriers = 0;
    state->need_resource = mock_need_resource;
    state->resource_resumed = 0;
    cmd->next_cell = 0;
    cmd->begun = 1;
    cmd->submitted = 0;
    return 0;
}

static uint32_t mock_authoritative_output_rows = UINT32_MAX;

static int mock_gpu_begin_authoritative_output(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows) {
    if (output_rows > 1u ||
        mock_gpu_begin(backend_state, command, generation, source_position,
            input_token_ids, input_count) != 0)
        return -1;
    mock_authoritative_output_rows = output_rows;
    return 0;
}

static int mock_gpu_encode_cell(void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutionCell *cell,
        const SaltTextExecutionAssignment *assignment,
        uint32_t input_count) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !cmd || !cmd->begun || cmd->submitted ||
        state->program != program || input_count != state->input_count ||
        cmd->next_cell >= program->dispatch.cell_count ||
        cell != &program->dispatch.cells[cmd->next_cell] || !assignment ||
        assignment->cpu.count != 0 || assignment->gpu.first != 0 ||
        assignment->gpu.count == 0)
        return -1;
    cmd->next_cell++;
    state->encoded_cells++;
    return 0;
}

static int mock_gpu_barrier(void *backend_state, void *command,
        uint32_t dependency_epoch, uint32_t completion_epoch) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !cmd || !cmd->begun || cmd->submitted ||
        dependency_epoch == 0 || completion_epoch <= dependency_epoch)
        return -1;
    state->barriers++;
    if (state->need_resource &&
        cmd->next_cell < state->program->dispatch.cell_count &&
        state->program->dispatch.cells[cmd->next_cell].kind ==
            SALT_TEXT_CELL_EXPERT_GATE &&
        (uint32_t)state->resource_resumed !=
            state->program->dispatch.cells[cmd->next_cell].layer + 1u) {
        mock_consumers_completed = 1;
        return SALT_TEXT_GPU_NEED_RESOURCE;
    }
    return 0;
}

static uint32_t mock_gpu_extent_calls;

static int mock_gpu_encode_extent(void *backend_state, void *command,
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        uint32_t first_cell, uint32_t cell_count, uint32_t input_count,
        uint32_t *encoded_cells) {
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!cmd || !plan || !encoded_cells || plan->program != program ||
        first_cell != cmd->next_cell || cell_count == 0 ||
        cell_count > program->dispatch.cell_count - first_cell)
        return -1;
    *encoded_cells = 0;
    mock_gpu_extent_calls++;
    for (uint32_t offset = 0; offset < cell_count; offset++) {
        uint32_t index = first_cell + offset;
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        if (offset > 0 && cell->dependency_epoch > 0) {
            int barrier = mock_gpu_barrier(backend_state, command,
                cell->dependency_epoch, cell->completion_epoch);
            if (barrier == SALT_TEXT_GPU_NEED_RESOURCE) {
                *encoded_cells = offset;
                return barrier;
            }
            if (barrier != SALT_TEXT_GPU_DEPENDENCY_READY) return -1;
        }
        if (mock_gpu_encode_cell(backend_state, command, program, cell,
                &plan->assignments[index], input_count) != 0)
            return -1;
        *encoded_cells = offset + 1u;
    }
    return SALT_TEXT_GPU_DEPENDENCY_READY;
}

static int mock_gpu_resource_request(
        void *backend_state, void *command, uint32_t *layer,
        int32_t *experts, uint32_t expert_capacity, uint32_t *expert_count) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !cmd || !layer || !experts || !expert_count ||
        expert_capacity < 1 || !state->need_resource ||
        cmd->next_cell >= state->program->dispatch.cell_count ||
        state->program->dispatch.cells[cmd->next_cell].kind !=
            SALT_TEXT_CELL_EXPERT_GATE)
        return -1;
    *layer = state->program->dispatch.cells[cmd->next_cell].layer;
    experts[0] = 3;
    *expert_count = 1;
    return 0;
}

static int mock_gpu_resource_resume(
        void *backend_state, void *command, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t expert_count) {
    MockGpuState *state = (MockGpuState *)backend_state;
    (void)command;
    if (!state || layer >= state->program->layer_count ||
        !experts || experts[0] != 3 || !slots ||
        slots[0] != 42 || expert_count != 1)
        return -1;
    state->resource_resumed = (int)layer + 1;
    mock_consumers_completed = 0;
    return 0;
}

static int mock_resource_acquire(void *opaque, uint32_t layer,
        const int32_t *experts, uint32_t count, int32_t *slots) {
    (void)opaque;
    if (layer >= mock_shadow_executor->program->layer_count ||
        !experts || experts[0] != 3 || count != 1 || !slots)
        return -1;
    slots[0] = 42;
    mock_resource_acquires++;
    return 0;
}

static int mock_resource_release(void *opaque, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t count) {
    (void)opaque;
    if (layer >= mock_shadow_executor->program->layer_count ||
        !experts || experts[0] != 3 || !slots ||
        slots[0] != 42 || count != 1 ||
        (mock_defer_resources && !mock_consumers_completed))
        return -1;
    mock_resource_releases++;
    return 0;
}

static const SaltTextExpertResourceOps mock_resource_ops = {
    mock_resource_acquire,
    mock_resource_release,
};

static int mock_gpu_submit(void *backend_state, void *command) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !cmd || !cmd->begun || cmd->submitted ||
        cmd->next_cell != state->program->dispatch.cell_count ||
        state->encoded_cells != state->program->dispatch.cell_count)
        return -1;
    cmd->submitted = 1;
    return 0;
}

static int mock_gpu_finish(void *backend_state, void *command,
        SaltTextExecutionView *view, SaltTextVerifyBackendStats *stats) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    SaltTextExecutionView shadow_view;
    SaltTextVerifyBackendStats shadow_stats;
    if (!state || !cmd || !cmd->submitted || !view || !stats ||
        (mock_defer_resources && state->need_resource &&
         mock_resource_acquires != mock_resource_releases + 1) ||
        state->shadow->ops->finish(state->shadow->context,
            state->shadow->program, &shadow_view, &shadow_stats) != 0 ||
        !shadow_view.canonical_base ||
        shadow_view.canonical_bytes < state->program->layout.total_bytes)
        return -1;
    memcpy(mock_gpu_canonical, shadow_view.canonical_base,
           state->program->layout.total_bytes);
    view->canonical_base = mock_gpu_canonical;
    view->canonical_bytes = state->program->layout.total_bytes;
    *stats = shadow_stats;
    mock_consumers_completed = 1;
    return 0;
}

static int mock_gpu_resolve(void *backend_state, void *command,
        uint32_t committed_count, uint32_t input_count,
        uint64_t *scrubbed_bytes) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    int rc;
    if (!state || !cmd || !cmd->submitted || !scrubbed_bytes ||
        input_count != state->input_count)
        return -1;
    rc = state->shadow->ops->resolve(state->shadow->context,
        state->shadow->program, state->generation, state->source_position,
        committed_count, input_count, scrubbed_bytes);
    memset(cmd, 0, sizeof *cmd);
    return rc;
}

static int mock_gpu_scrub(void *backend_state, void *command,
        uint64_t *scrubbed_bytes) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    int rc;
    if (!state || !cmd || !scrubbed_bytes) return -1;
    rc = state->shadow->ops->scrub(state->shadow->context,
        state->shadow->program, state->generation, scrubbed_bytes);
    memset(mock_gpu_canonical, 0, state->program->layout.total_bytes);
    memset(cmd, 0, sizeof *cmd);
    return rc;
}

static int mock_gpu_destroy(void *backend_state, void *command) {
    MockGpuState *state = (MockGpuState *)backend_state;
    MockGpuCommand *cmd = (MockGpuCommand *)command;
    if (!state || !cmd || cmd->begun || !state->canonical.contents)
        return -1;
    memset(mock_gpu_canonical, 0, state->program->layout.total_bytes);
    memset(state, 0, sizeof *state);
    memset(cmd, 0, sizeof *cmd);
    mock_shadow_executor = NULL;
    return 0;
}

static const SaltTextGpuProgramOps mock_gpu_ops = {
    mock_gpu_requirements,
    mock_gpu_prepare,
    mock_gpu_begin,
    mock_gpu_encode_cell,
    mock_gpu_encode_extent,
    mock_gpu_barrier,
    mock_gpu_resource_request,
    mock_gpu_resource_resume,
    mock_gpu_submit,
    mock_gpu_finish,
    mock_gpu_resolve,
    mock_gpu_scrub,
    mock_gpu_destroy,
    NULL,
    NULL,
    mock_gpu_begin_authoritative_output
};

static int package_tensor(SyntheticPackage *package,
                          uint32_t rows, uint32_t cols,
                          FixtureTensorKind kind,
                          SaltTextTensorDesc *tensor_out) {
    SaltTextTensorDesc tensor;
    F32Binding *binding;
    float *values;
    uint32_t index, count;
    if (!package || !tensor_out || rows == 0 || cols == 0 ||
        rows > UINT32_MAX / cols ||
        (count = rows * cols) > MAX_TENSOR_VALUES ||
        package->binding_count >= MAX_BINDINGS)
        return -1;
    index = package->binding_count++;
    binding = &package->bindings[index];
    values = package->values[index];
    memset(binding, 0, sizeof *binding);
    memset(values, 0, sizeof package->values[index]);
    binding->owner = package;
    binding->values = values;
    binding->rows = rows;
    binding->cols = cols;
    binding->scratch_bytes = (size_t)(index % 3u) * 8u;
    for (uint32_t item = 0; item < count; item++) {
        int raw = (int)((item * 7u + (index + 1u) * 11u) % 19u) - 9;
        if (kind == FIXTURE_NORM)
            values[item] = 0.92f + 0.015f * (float)(item % 5u);
        else if (kind == FIXTURE_SCALE)
            values[item] = 0.80f + 0.07f * (float)(item % 4u);
        else if (kind == FIXTURE_SCALAR)
            values[item] = 0.85f + 0.01f * (float)(index % 4u);
        else if (kind == FIXTURE_EMBEDDING)
            values[item] = (float)raw * 0.075f +
                0.01f * (float)(item / cols);
        else
            values[item] = (float)raw * 0.035f;
    }
    memset(&tensor, 0, sizeof tensor);
    tensor.rows = rows;
    tensor.cols = cols;
    tensor.stable_handle = (uint64_t)index + 1u;
    tensor.binding = binding;
    tensor.host_ops = &f32_tensor_ops;
    tensor.storage.encoding = SALT_TENSOR_ENCODING_F32;
    tensor.storage.source_class = SALT_TENSOR_SOURCE_STATIC;
    tensor.storage.resource_kind = SALT_TENSOR_RESOURCE_TRUNK;
    tensor.storage.resource_id = 0u;
    tensor.storage.value_offset =
        (uint64_t)((unsigned char *)(void *)values -
                   (unsigned char *)(void *)package->values);
    tensor.storage.value_bytes = (uint64_t)count * sizeof(float);
    *tensor_out = tensor;
    return 0;
}

static int initialize_package(SyntheticPackage *package) {
    if (!package) return -1;
    memset(package, 0, sizeof *package);
    package->resources[0] = (SaltTensorResourceSpec) {
        SALT_TENSOR_RESOURCE_TRUNK, 0u, SALT_TENSOR_SOURCE_STATIC, 0u,
        package->values, sizeof package->values, sizeof package->values,
    };
    memset(&package->model, 0, sizeof package->model);
    package->model.name = "synthetic-text-package";
    package->model.n_layers = LAYERS;
    package->model.n_experts = EXPERTS;
    package->model.topk = TOPK;
    package->model.n_shared = 1;
    package->model.hidden = HIDDEN;
    package->model.moe_inter = EXPERT_WIDTH;
    package->model.n_heads = HEADS;
    package->model.n_kv_heads = KV_HEADS;
    package->model.runtime_ready = 1;
    if (package_tensor(package, VOCABULARY, HIDDEN,
            FIXTURE_EMBEDDING, &package->embedding) != 0 ||
        package_tensor(package, 1, HIDDEN,
            FIXTURE_NORM, &package->final_norm) != 0)
        return -1;
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        SaltTextLayerPlan *plan = &package->base_plans[layer];
        SaltTextLayerExecDesc *weights = &package->weight_layers[layer];
        memset(plan, 0, sizeof *plan);
        plan->layer = (int)layer;
        plan->runtime_ready = 1;
        plan->hidden = HIDDEN;
        plan->attention.kind = layer == 0 ? SALT_ATTN_FULL : SALT_ATTN_SLIDING;
        plan->attention.causal = 1;
        plan->attention.n_heads = HEADS;
        plan->attention.n_kv_heads = KV_HEADS;
        plan->attention.head_dim = HEAD_DIM;
        plan->attention.rope_dim = HEAD_DIM;
        plan->attention.rope_base_dim = HEAD_DIM;
        plan->attention.rope_kind = SALT_ROPE_DEFAULT;
        plan->attention.window = layer == 0 ? 0 : 3;
        plan->attention.shared_kv_projection = layer == 1;
        plan->attention.rope_theta = 10000.0;
        plan->attention.score_scale = 0.75f;
        plan->n_experts = EXPERTS;
        plan->top_k_experts = TOPK;
        plan->expert_intermediate = EXPERT_WIDTH;
        plan->dense_intermediate = DENSE_WIDTH;
        plan->activation = SALT_ACT_GELU_TANH;
        plan->router_rmsnorm = 1;
        plan->parallel_dense_routed = 1;
        plan->residual_postnorm = 1;
        plan->final_layer_scale = 1;
        plan->tied_embeddings = 1;
        plan->logit_softcap = 5.0f;
        memset(weights, 0, sizeof *weights);
#define MATRIX(field, rows, cols) \
        do { if (package_tensor(package, (rows), (cols), FIXTURE_MATRIX, \
                 &weights->field) != 0) return -1; } while (0)
#define NORM(field, cols) \
        do { if (package_tensor(package, 1u, (cols), FIXTURE_NORM, \
                 &weights->field) != 0) return -1; } while (0)
        MATRIX(q, QUERY_WIDTH, HIDDEN);
        MATRIX(k, KV_WIDTH, HIDDEN);
        if (!plan->attention.shared_kv_projection)
            MATRIX(v, KV_WIDTH, HIDDEN);
        MATRIX(o, HIDDEN, QUERY_WIDTH);
        MATRIX(dense_gate, DENSE_WIDTH, HIDDEN);
        MATRIX(dense_up, DENSE_WIDTH, HIDDEN);
        MATRIX(dense_down, HIDDEN, DENSE_WIDTH);
        MATRIX(router, EXPERTS, HIDDEN);
        NORM(pre_attention_norm, HIDDEN);
        NORM(q_norm, HEAD_DIM);
        NORM(k_norm, HEAD_DIM);
        NORM(post_attention_norm, HIDDEN);
        NORM(pre_ffn_norm_1, HIDDEN);
        NORM(pre_ffn_norm_2, HIDDEN);
        NORM(post_ffn_norm_1, HIDDEN);
        NORM(post_ffn_norm_2, HIDDEN);
        NORM(post_ffn_norm, HIDDEN);
        if (package_tensor(package, 1, HIDDEN, FIXTURE_SCALE,
                &weights->router_scale) != 0 ||
            package_tensor(package, 1, EXPERTS, FIXTURE_SCALE,
                &weights->per_expert_scale) != 0 ||
            package_tensor(package, 1, 1, FIXTURE_SCALAR,
                &weights->layer_scalar) != 0)
            return -1;
#undef MATRIX
#undef NORM
        weights->experts = package->experts[layer];
        weights->expert_count = EXPERTS;
        for (uint32_t expert = 0; expert < EXPERTS; expert++) {
            SaltTextExpertDesc *entry = &package->experts[layer][expert];
            memset(entry, 0, sizeof *entry);
            entry->expert_id = expert;
            if (package_tensor(package, EXPERT_WIDTH, HIDDEN,
                    FIXTURE_MATRIX, &entry->gate) != 0 ||
                package_tensor(package, EXPERT_WIDTH, HIDDEN,
                    FIXTURE_MATRIX, &entry->up) != 0 ||
                package_tensor(package, HIDDEN, EXPERT_WIDTH,
                    FIXTURE_MATRIX, &entry->down) != 0)
                return -1;
        }
    }
    package->weight_layers[0].plan = &package->base_plans[0];
    package->weight_layers[1].plan = &package->base_plans[1];
    /* The tied head is the exact same stable tensor descriptor. */
    package->weight_layers[0].kv = NULL;
    package->weight_layers[1].kv = NULL;
    package->fail_handle = 0;
    package->operation_calls = 0;
    package->scratch_queries = 0;
    return 0;
}

static SaltTextTensorDesc package_model_tensor(
        const SyntheticPackage *package, int embedding) {
    return embedding ? package->embedding : package->final_norm;
}

static float *fixture_private_row(ExecutionFixture *fixture,
                                  uint32_t layer, int values,
                                  uint32_t position) {
    SaltTextKvRowsDesc *rows = values
        ? &fixture->kv[layer].values : &fixture->kv[layer].keys;
    float *base = values
        ? fixture->private_values[layer] : fixture->private_keys[layer];
    uint32_t logical = position - rows->private_position_base;
    uint32_t index = rows->private_mode == SALT_TEXT_KV_PRIVATE_RING
        ? logical % rows->private_row_capacity : logical;
    return base + (size_t)index * rows->private_row_stride;
}

static int prepare_execution(ExecutionFixture *fixture,
                             SyntheticPackage *package,
                             SaltTextKvPrivateMode mode,
                             uint32_t source_position) {
    uint32_t private_capacity = mode == SALT_TEXT_KV_PRIVATE_RING
        ? RING_ROWS : PRIVATE_ROWS;
    uint32_t shared = mode == SALT_TEXT_KV_PRIVATE_RING ? 0u : SHARED_ROWS;
    if (!fixture || !package) return -1;
    memset(fixture, 0, sizeof *fixture);
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        fixture->plans[layer] = package->base_plans[layer];
        if (mode == SALT_TEXT_KV_PRIVATE_RING) {
            fixture->plans[layer].attention.kind = SALT_ATTN_SLIDING;
            fixture->plans[layer].attention.window = 4;
        }
        fixture->layers[layer] = package->weight_layers[layer];
        fixture->layers[layer].plan = &fixture->plans[layer];
        fixture->layers[layer].kv = &fixture->kv[layer];
        fixture->layers[layer].experts = package->experts[layer];
        memset(fixture->private_keys[layer], 0,
               sizeof fixture->private_keys[layer]);
        memset(fixture->private_values[layer], 0,
               sizeof fixture->private_values[layer]);
        memset(fixture->shared_keys[layer], 0,
               sizeof fixture->shared_keys[layer]);
        memset(fixture->shared_values[layer], 0,
               sizeof fixture->shared_values[layer]);
        fixture->kv[layer].keys.private_mode = mode;
        fixture->kv[layer].keys.private_rows = fixture->private_keys[layer];
        fixture->kv[layer].keys.private_float_capacity =
            (size_t)private_capacity * KV_WIDTH;
        fixture->kv[layer].keys.private_row_stride = KV_WIDTH;
        fixture->kv[layer].keys.private_row_capacity = private_capacity;
        fixture->kv[layer].keys.private_position_base = shared;
        fixture->kv[layer].values.private_mode = mode;
        fixture->kv[layer].values.private_rows = fixture->private_values[layer];
        fixture->kv[layer].values.private_float_capacity =
            (size_t)private_capacity * KV_WIDTH;
        fixture->kv[layer].values.private_row_stride = KV_WIDTH;
        fixture->kv[layer].values.private_row_capacity = private_capacity;
        fixture->kv[layer].values.private_position_base = shared;
        if (shared > 0) {
            fixture->kv[layer].keys.shared_prefix =
                fixture->shared_keys[layer];
            fixture->kv[layer].keys.shared_prefix_float_capacity =
                SHARED_ROWS * KV_WIDTH;
            fixture->kv[layer].keys.shared_prefix_row_stride = KV_WIDTH;
            fixture->kv[layer].keys.shared_prefix_rows = SHARED_ROWS;
            fixture->kv[layer].values.shared_prefix =
                fixture->shared_values[layer];
            fixture->kv[layer].values.shared_prefix_float_capacity =
                SHARED_ROWS * KV_WIDTH;
            fixture->kv[layer].values.shared_prefix_row_stride = KV_WIDTH;
            fixture->kv[layer].values.shared_prefix_rows = SHARED_ROWS;
            for (uint32_t position = 0; position < shared; position++)
                for (uint32_t column = 0; column < KV_WIDTH; column++) {
                    fixture->shared_keys[layer][position * KV_WIDTH + column] =
                        0.03f * (float)(1u + layer * 11u +
                                       position * 3u + column);
                    fixture->shared_values[layer][position * KV_WIDTH + column] =
                        -0.025f * (float)(1u + layer * 7u +
                                         position * 2u + column);
                }
        }
        for (uint32_t position = shared; position < source_position;
             position++) {
            float *key = fixture_private_row(
                fixture, layer, 0, position);
            float *value = fixture_private_row(
                fixture, layer, 1, position);
            for (uint32_t column = 0; column < KV_WIDTH; column++) {
                key[column] = 0.02f * (float)(1u + layer * 13u +
                    position * 5u + column);
                value[column] = -0.018f * (float)(1u + layer * 9u +
                    position * 4u + column);
            }
        }
    }
    memset(&fixture->model, 0, sizeof fixture->model);
    fixture->model.model = &package->model;
    fixture->model.layers = fixture->layers;
    fixture->model.layer_count = LAYERS;
    fixture->model.tensor_resources = package->resources;
    fixture->model.tensor_resource_count = 1u;
    fixture->model.vocabulary = VOCABULARY;
    fixture->model.maximum_context = MAXIMUM_CONTEXT;
    fixture->model.norm_epsilon = 1.0e-5f;
    fixture->model.embedding_scale = 1.25f;
    fixture->model.embedding = package_model_tensor(package, 1);
    fixture->model.final_norm = package_model_tensor(package, 0);
    fixture->model.output_head = fixture->model.embedding;
    fixture->state.position = source_position;
    fixture->state.transition_generation = 0;
    return 0;
}

static int compile_execution(ExecutionFixture *fixture) {
    fixture->target_policy = (SaltTextTargetPolicy) {
        .execution_class = SALT_TEXT_EXECUTION_CPU_ONLY,
        .worker_budget = 8u,
        .sequence_tiles = CANDIDATES,
        .target_rows = CANDIDATES,
        .route_count = 1u,
        .queue_length = 4u,
        .candidate_count = CANDIDATES,
        .kv_warmup_rows = 1u,
        .warm_target_rows = CANDIDATES,
        .ready = 1,
    };
    if (salt_text_verify_program_arena_requirement(
            &fixture->model, CANDIDATES, &fixture->program_need) != 0 ||
        fixture->program_need > sizeof fixture->program_arena ||
        salt_text_verify_program_compile(&fixture->program,
            &fixture->model, &fixture->state, CANDIDATES,
            fixture->program_arena, fixture->program_need) != 0 ||
        salt_text_verify_program_bind_target_policy(
            &fixture->program, &fixture->target_policy) != 0 ||
        salt_text_verify_cpu_arena_requirement(
            &fixture->program, &fixture->cpu_need) != 0 ||
        fixture->cpu_need > sizeof fixture->cpu_arena ||
        salt_text_verify_cpu_compile(&fixture->cpu, &fixture->program,
            fixture->cpu_arena, fixture->cpu_need) != 0 ||
        salt_text_verify_cpu_executor_init(
            &fixture->executor, &fixture->cpu) != 0)
        return -1;
    return 0;
}

static void snapshot_descriptors(const ExecutionFixture *fixture,
                                 const SyntheticPackage *package,
                                 DescriptorSnapshot *snapshot) {
    snapshot->salt_model = package->model;
    snapshot->package_embedding = package->embedding;
    snapshot->package_final_norm = package->final_norm;
    snapshot->model = fixture->model;
    memcpy(snapshot->plans, fixture->plans, sizeof snapshot->plans);
    memcpy(snapshot->layers, fixture->layers, sizeof snapshot->layers);
    memcpy(snapshot->kv, fixture->kv, sizeof snapshot->kv);
    memcpy(snapshot->experts, package->experts, sizeof snapshot->experts);
}

static int descriptors_equal(const ExecutionFixture *fixture,
                             const SyntheticPackage *package,
                             const DescriptorSnapshot *snapshot) {
    return memcmp(&snapshot->salt_model, &package->model,
                  sizeof snapshot->salt_model) == 0 &&
        memcmp(&snapshot->package_embedding, &package->embedding,
               sizeof snapshot->package_embedding) == 0 &&
        memcmp(&snapshot->package_final_norm, &package->final_norm,
               sizeof snapshot->package_final_norm) == 0 &&
        memcmp(&snapshot->model, &fixture->model,
                  sizeof snapshot->model) == 0 &&
        memcmp(snapshot->plans, fixture->plans,
               sizeof snapshot->plans) == 0 &&
        memcmp(snapshot->layers, fixture->layers,
               sizeof snapshot->layers) == 0 &&
        memcmp(snapshot->kv, fixture->kv,
               sizeof snapshot->kv) == 0 &&
        memcmp(snapshot->experts, package->experts,
               sizeof snapshot->experts) == 0;
}

static void snapshot_kv(const ExecutionFixture *fixture,
                        KvSnapshot *snapshot) {
    memcpy(snapshot->keys, fixture->private_keys, sizeof snapshot->keys);
    memcpy(snapshot->values, fixture->private_values,
           sizeof snapshot->values);
    memcpy(snapshot->shared_keys, fixture->shared_keys,
           sizeof snapshot->shared_keys);
    memcpy(snapshot->shared_values, fixture->shared_values,
           sizeof snapshot->shared_values);
}

static int kv_matches(const ExecutionFixture *fixture,
                      const KvSnapshot *snapshot) {
    return memcmp(snapshot->keys, fixture->private_keys,
                  sizeof snapshot->keys) == 0 &&
        memcmp(snapshot->values, fixture->private_values,
               sizeof snapshot->values) == 0 &&
        memcmp(snapshot->shared_keys, fixture->shared_keys,
               sizeof snapshot->shared_keys) == 0 &&
        memcmp(snapshot->shared_values, fixture->shared_values,
               sizeof snapshot->shared_values) == 0;
}

static void select_parent_token(float *logits, uint32_t token) {
    for (uint32_t item = 0; item < VOCABULARY; item++)
        logits[item] = -0.4f * (float)item;
    logits[token] = 6.0f;
}

static float *fixture_logits(ExecutionFixture *fixture) {
    return (float *)(void *)(fixture->cpu.arena +
        fixture->program.layout.position_logits);
}

static int tentative_suffix_is_zero(const ExecutionFixture *fixture,
                                    uint32_t first) {
    for (uint32_t layer = 0; layer < LAYERS; layer++) {
        const SaltTextCompiledLayer *compiled = &fixture->program.layers[layer];
        const float *keys = (const float *)(const void *)(fixture->cpu.arena +
            compiled->tentative_key_offset);
        const float *values = (const float *)(const void *)(fixture->cpu.arena +
            compiled->tentative_value_offset);
        for (uint32_t row = first; row < CANDIDATES; row++)
            for (uint32_t column = 0; column < KV_WIDTH; column++)
                if (keys[row * KV_WIDTH + column] != 0.0f ||
                    values[row * KV_WIDTH + column] != 0.0f)
                    return 0;
    }
    return 1;
}

static int canonical_is_zero(const ExecutionFixture *fixture) {
    for (size_t byte = 0; byte < fixture->program.layout.total_bytes; byte++)
        if (fixture->cpu.arena[byte] != 0)
            return 0;
    return 1;
}

static int program_plan_valid(const SaltTextVerifyProgram *program) {
    uint32_t final_publications = 0;
    if (!program || program->dispatch.engine_command_count != 1u ||
        program->dispatch.cell_count == 0)
        return 0;
    for (uint32_t index = 0; index < program->dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        if (cell->completion_epoch <= cell->dependency_epoch ||
            cell->logical_capacity == 0 ||
            cell->destination_offset >= program->layout.total_bytes)
            return 0;
        if (cell->flags & SALT_TEXT_EXECUTION_CELL_FINAL_PUBLICATION)
            final_publications++;
        else if (cell->flags != 0)
            return 0;
    }
    return final_publications == 1u;
}

static const TextOperatorContract *text_operator_contract_find(
        SaltTextExecutionCellKind kind) {
    for (size_t index = 0;
         index < sizeof text_operator_contract / sizeof text_operator_contract[0];
         index++)
        if (text_operator_contract[index].kind == kind)
            return &text_operator_contract[index];
    return NULL;
}

static int text_operator_contract_fixture(SyntheticPackage *package) {
    uint32_t seen = 0;
    uint32_t runtime_seen = 0;
    uint32_t class_counts[5] = { 0, 0, 0, 0, 0 };
    const size_t contract_count =
        sizeof text_operator_contract / sizeof text_operator_contract[0];
    CHECK(contract_count == 27u,
          "text operator contract does not cover 27 cell kinds");
    for (size_t index = 0; index < contract_count; index++) {
        const TextOperatorContract *contract = &text_operator_contract[index];
        CHECK(contract->kind >= SALT_TEXT_CELL_EMBEDDING &&
                  contract->kind <= SALT_TEXT_CELL_LOGIT_SOFTCAP &&
                  contract->unit >= SALT_TEXT_EXECUTION_ROWS &&
                  contract->unit <= SALT_TEXT_EXECUTION_JOBS &&
                  contract->contract_class >= TEXT_OPERATOR_SHARED_TENSOROPS &&
                  contract->contract_class <= TEXT_OPERATOR_KV_VIEW_PENDING &&
                  (seen & (1u << (uint32_t)contract->kind)) == 0,
              "text operator contract contains an invalid or duplicate cell");
        seen |= 1u << (uint32_t)contract->kind;
        class_counts[contract->contract_class]++;
    }
    CHECK(seen == ((UINT32_C(1) << 28u) - 2u) &&
              class_counts[TEXT_OPERATOR_SHARED_TENSOROPS] == 12u &&
              class_counts[TEXT_OPERATOR_SHARED_ENCODE_PENDING] == 15u &&
              class_counts[TEXT_OPERATOR_REALIZATION_PENDING] == 0u &&
              class_counts[TEXT_OPERATOR_KV_VIEW_PENDING] == 0u,
          "text operator contract class census changed");
    CHECK(prepare_execution(&operator_contract_fixture, package,
              SALT_TEXT_KV_PRIVATE_ABSOLUTE, 4u) == 0 &&
              compile_execution(&operator_contract_fixture) == 0,
          "text operator contract program compilation failed");
    for (uint32_t index = 0;
         index < operator_contract_fixture.program.dispatch.cell_count; index++) {
        const SaltTextExecutionCell *cell =
            &operator_contract_fixture.program.dispatch.cells[index];
        const TextOperatorContract *contract =
            text_operator_contract_find(cell->kind);
        CHECK(contract && contract->unit == cell->unit,
              "compiled text cell violates the frozen operator contract");
        runtime_seen |= 1u << (uint32_t)cell->kind;
    }
    if (runtime_seen != seen) {
        fprintf(stderr,
                "text operator contract: compiled program missing mask=0x%08x\n",
                seen & ~runtime_seen);
        return -1;
    }
    printf("text operator contract: PASS kinds=%zu shared_tensor=%u "
           "encode_pending=%u realization_pending=%u kv_view_pending=%u\n",
           contract_count,
           class_counts[TEXT_OPERATOR_SHARED_TENSOROPS],
           class_counts[TEXT_OPERATOR_SHARED_ENCODE_PENDING],
           class_counts[TEXT_OPERATOR_REALIZATION_PENDING],
           class_counts[TEXT_OPERATOR_KV_VIEW_PENDING]);
    return 0;
}

static int shared_operator_realization_contract(SyntheticPackage *package) {
    const int32_t token_ids[2] = { 1, 3 };
    unsigned char embedding_scratch[32];
    float embedding_output[2 * HIDDEN], embedding_reference[2 * HIDDEN];
    const int32_t selected[6] = { 2, 0, 2, 1, 0, 2 };
    const int32_t expected_grouped_to_canonical[6] = { 1, 4, 3, 0, 2, 5 };
    const int32_t expected_canonical_to_grouped[6] = { 3, 0, 4, 2, 1, 5 };
    const float row_inputs[9] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const float expected_routed_inputs[18] = {
        1.0f, 2.0f, 3.0f,
        7.0f, 8.0f, 9.0f,
        4.0f, 5.0f, 6.0f,
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
    };
    const float weights[6] = { 0.6f, 0.4f, 0.25f, 0.75f, 0.3f, 0.7f };
    const float grouped_outputs[18] = {
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
        40.0f, 41.0f, 42.0f,
        50.0f, 51.0f, 52.0f,
        60.0f, 61.0f, 62.0f,
    };
    uint32_t expert_offsets[4];
    int32_t grouped_to_canonical[6], canonical_to_grouped[6];
    float routed_inputs[18], reduced[9], reduced_reference[9];
    const F32Binding *embedding =
        (const F32Binding *)package->embedding.binding;
    CHECK(embedding && embedding->values && embedding->cols == HIDDEN,
          "shared embedding fixture binding is invalid");
    for (uint32_t row = 0; row < 2u; row++)
        for (uint32_t column = 0; column < HIDDEN; column++)
            embedding_reference[row * HIDDEN + column] =
                embedding->values[(uint32_t)token_ids[row] * HIDDEN + column] *
                1.25f;
    CHECK(salt_tensor_embedding_gather(&package->embedding,
              token_ids, 2u, 1.25f, embedding_output,
              embedding_scratch, sizeof embedding_scratch) == 0 &&
              memcmp(embedding_output, embedding_reference,
                     sizeof embedding_output) == 0,
          "shared embedding gather changed canonical bytes");
    CHECK(salt_tensor_selected_route_group(
              selected, 3u, 2u, 4u, row_inputs, 3u,
              expert_offsets, 4u, grouped_to_canonical,
              canonical_to_grouped, routed_inputs) == 0 &&
              memcmp(grouped_to_canonical,
                     expected_grouped_to_canonical,
                     sizeof grouped_to_canonical) == 0 &&
              memcmp(canonical_to_grouped,
                     expected_canonical_to_grouped,
                     sizeof canonical_to_grouped) == 0 &&
              memcmp(routed_inputs, expected_routed_inputs,
                     sizeof routed_inputs) == 0,
          "shared selected-expert grouping changed canonical rank");
    memset(reduced_reference, 0, sizeof reduced_reference);
    for (uint32_t row = 0; row < 3u; row++)
        for (uint32_t rank = 0; rank < 2u; rank++) {
            uint32_t canonical = row * 2u + rank;
            uint32_t grouped =
                (uint32_t)expected_canonical_to_grouped[canonical];
            for (uint32_t column = 0; column < 3u; column++)
                reduced_reference[row * 3u + column] +=
                    weights[canonical] *
                    grouped_outputs[grouped * 3u + column];
        }
    CHECK(salt_tensor_selected_weighted_reduce(
              weights, canonical_to_grouped, grouped_outputs,
              3u, 2u, 3u, reduced) == 0 &&
              memcmp(reduced, reduced_reference, sizeof reduced) == 0,
          "shared weighted reduction changed canonical selected-rank order");
    puts("shared embedding/group/reduction realizations: PASS");
    return 0;
}

static int kv_read_view_contract(void) {
    float shared[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    float private_rows[8] = {
        10.0f, 11.0f, 12.0f, 13.0f,
        14.0f, 15.0f, 16.0f, 17.0f,
    };
    float tentative[8] = {
        20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f,
    };
    uint32_t parents[4] = { UINT32_MAX, 0u, 1u, 0u };
    uint32_t depths[4] = { 0u, 1u, 2u, 1u };
    SaltTextKvRowsDesc rows, ring_rows;
    SaltTextKvReadView linear, frontier, ring, unchanged;
    memset(&rows, 0, sizeof rows);
    rows.private_mode = SALT_TEXT_KV_PRIVATE_ABSOLUTE;
    rows.private_rows = private_rows;
    rows.private_float_capacity = 8u;
    rows.private_row_stride = 2u;
    rows.private_row_capacity = 4u;
    rows.private_position_base = 2u;
    rows.shared_prefix = shared;
    rows.shared_prefix_float_capacity = 4u;
    rows.shared_prefix_row_stride = 2u;
    rows.shared_prefix_rows = 2u;
    CHECK(salt_text_kv_read_view_init(&linear, &rows,
              tentative, 8u, 2u, 4u, 4u, NULL, NULL, 2u) == 0 &&
              salt_text_kv_read_view_row(&linear, 2u, 0u) == shared &&
              salt_text_kv_read_view_row(&linear, 2u, 2u) == private_rows &&
              salt_text_kv_read_view_row(&linear, 2u, 4u) == tentative &&
              salt_text_kv_read_view_row(&linear, 2u, 6u) == tentative + 4u &&
              salt_text_kv_read_view_row(&linear, 1u, 6u) == NULL,
          "linear committed/tentative KV view changed row identity");
    CHECK(salt_text_kv_read_view_init(&frontier, &rows,
              tentative, 8u, 2u, 4u, 4u,
              parents, depths, 2u) == 0 &&
              salt_text_kv_read_view_row(&frontier, 2u, 4u) == tentative &&
              salt_text_kv_read_view_row(&frontier, 2u, 5u) == tentative + 2u &&
              salt_text_kv_read_view_row(&frontier, 2u, 6u) == tentative + 4u &&
              salt_text_kv_read_view_row(&frontier, 3u, 5u) == tentative + 6u &&
              salt_text_kv_read_view_row(&frontier, 3u, 6u) == NULL,
          "frontier KV view crossed a tentative branch");
    memset(&ring_rows, 0, sizeof ring_rows);
    ring_rows.private_mode = SALT_TEXT_KV_PRIVATE_RING;
    ring_rows.private_rows = private_rows;
    ring_rows.private_float_capacity = 8u;
    ring_rows.private_row_stride = 2u;
    ring_rows.private_row_capacity = 4u;
    CHECK(salt_text_kv_read_view_init(&ring, &ring_rows,
              tentative, 8u, 2u, 4u, 4u, NULL, NULL, 2u) == 0 &&
              salt_text_kv_read_view_row(&ring, 0u, 1u) == private_rows + 2u &&
              salt_text_kv_read_view_row(&ring, 0u, 3u) == private_rows + 6u &&
              salt_text_kv_read_view_row(&ring, 0u, 4u) == tentative &&
              salt_text_kv_read_view_row(&ring, 1u, 5u) == tentative + 2u,
          "ring wrap reinterpreted committed rows as tentative rows");
    CHECK(salt_text_kv_read_view_init(&ring, &ring_rows,
              tentative, 8u, 2u, 4u, 131070u, NULL, NULL, 2u) == 0 &&
              salt_text_kv_read_view_row(&ring, 0u, 131065u) == NULL &&
              salt_text_kv_read_view_row(&ring, 0u, 131066u) == private_rows + 4u &&
              salt_text_kv_read_view_row(&ring, 0u, 131069u) == private_rows + 2u &&
              salt_text_kv_read_view_row(&ring, 0u, 131070u) == tentative &&
              salt_text_kv_read_view_row(&ring, 2u, 131072u) == tentative + 4u &&
              ring_rows.private_float_capacity == 8u,
          "large absolute position expanded or misaddressed bounded ring");
    unchanged = frontier;
    parents[2] = 2u;
    CHECK(salt_text_kv_read_view_init(&frontier, &rows,
              tentative, 8u, 2u, 4u, 4u,
              parents, depths, 2u) != 0 &&
              memcmp(&frontier, &unchanged, sizeof frontier) == 0,
          "invalid frontier topology mutated the KV view");
    tentative[0] = -999.0f;
    CHECK(salt_text_kv_read_view_row(&linear, 2u, 0u) == shared &&
              salt_text_kv_read_view_row(&linear, 2u, 2u) == private_rows &&
              shared[0] == 1.0f && private_rows[0] == 10.0f,
          "tentative poison escaped into committed KV rows");
    puts("committed/tentative KV read view: PASS");
    return 0;
}

static int executor_plan_valid(
        const SaltTextVerifyProgram *program,
        const SaltTextExecutorPlan *plan,
        SaltTextExecutionClass execution_class,
        uint32_t cpu_rows, uint32_t cpu_jobs) {
    if (!program || !plan || plan->program != program ||
        plan->execution_class != execution_class || !plan->assignments ||
        plan->assignment_count != program->dispatch.cell_count)
        return 0;
    for (uint32_t index = 0; index < plan->assignment_count; index++) {
        const SaltTextExecutionCell *cell = &program->dispatch.cells[index];
        const SaltTextExecutionAssignment *assignment =
            &plan->assignments[index];
        uint32_t expected_cpu = 0;
        if (execution_class == SALT_TEXT_EXECUTION_CPU_ONLY)
            expected_cpu = cell->logical_capacity;
        else if (execution_class == SALT_TEXT_EXECUTION_MIXED) {
            expected_cpu = cell->unit == SALT_TEXT_EXECUTION_ROWS
                ? cpu_rows : cpu_jobs;
            if (expected_cpu > cell->logical_capacity)
                expected_cpu = cell->logical_capacity;
        }
        if (assignment->cpu.first != 0 ||
            assignment->cpu.count != expected_cpu ||
            assignment->gpu.first != expected_cpu ||
            assignment->gpu.count != cell->logical_capacity - expected_cpu)
            return 0;
    }
    return 1;
}

static int exercise_arena_and_dispatch_contracts(SyntheticPackage *package) {
    SaltTextVerifyProgram unchanged_program, private_program, program_snapshot;
    SaltTextVerifyCpuContext unchanged_cpu;
    SaltTextKvState private_state;
    SaltTextModelExecDesc overflow;
    SaltTextExecutionAssignment mixed_assignments[MAX_EXECUTION_CELLS];
    SaltTextExecutionAssignment gpu_assignments[MAX_EXECUTION_CELLS];
    SaltTextExecutorPlan mixed_plan, gpu_plan;
    SaltTextDispatchPolicy mixed_policy, gpu_policy;
    unsigned char private_arena[PROGRAM_ARENA_BYTES];
    size_t program_need, cpu_need, private_need, ignored;
    {
        unsigned char header[24];
        SaltTensorResourceSpec selected_resource = {
            SALT_TENSOR_RESOURCE_EXPERT, 7u,
            SALT_TENSOR_SOURCE_SELECTED, 0u,
            header, 4096u, sizeof header,
        };
        SaltTensorStorageSpec selected_storage;
        memset(&selected_storage, 0, sizeof selected_storage);
        selected_storage.encoding = SALT_TENSOR_ENCODING_AFFINE_Q4;
        selected_storage.source_class = SALT_TENSOR_SOURCE_SELECTED;
        selected_storage.resource_kind = SALT_TENSOR_RESOURCE_EXPERT;
        selected_storage.resource_id = 7u;
        selected_storage.logical_resource_id = 11u;
        selected_storage.group_size = 64u;
        selected_storage.value_offset = 128u;
        selected_storage.scale_offset = 2176u;
        selected_storage.bias_offset = 2304u;
        selected_storage.value_bytes = 2048u;
        selected_storage.scale_bytes = 128u;
        selected_storage.bias_bytes = 128u;
        CHECK(salt_tensor_storage_validate(
                  &selected_storage, 64u, 64u) == 0 &&
              salt_tensor_resource_validate(&selected_resource) == 0 &&
              salt_tensor_storage_resolves(&selected_storage,
                  &selected_resource, 1u) == 0,
              "selected canonical extent was confused with mapped bytes");
        selected_resource.source_class = SALT_TENSOR_SOURCE_STATIC;
        CHECK(salt_tensor_resource_validate(&selected_resource) != 0,
              "incomplete static mapping was accepted");
    }
    CHECK(prepare_execution(&probe_fixture, package,
              SALT_TEXT_KV_PRIVATE_ABSOLUTE, 4u) == 0,
          "could not prepare arena probe");
    CHECK(salt_tensor_desc_validate(&probe_fixture.model.embedding) == 0 &&
          salt_tensor_storage_resolves(
              &probe_fixture.model.embedding.storage,
              probe_fixture.model.tensor_resources,
              probe_fixture.model.tensor_resource_count) == 0 &&
          probe_fixture.model.embedding.storage.encoding ==
              SALT_TENSOR_ENCODING_F32 &&
          probe_fixture.model.embedding.storage.source_class ==
              SALT_TENSOR_SOURCE_STATIC,
          "engine TensorOps static specification is incomplete");
    CHECK(salt_text_verify_program_arena_requirement(
              &probe_fixture.model, CANDIDATES, &program_need) == 0 &&
              program_need <= sizeof probe_fixture.program_arena,
          "program arena requirement failed");
    memset(&unchanged_program, 0xa5, sizeof unchanged_program);
    {
        SaltTextVerifyProgram sentinel = unchanged_program;
        CHECK(salt_text_verify_program_compile(&unchanged_program,
                  &probe_fixture.model, &probe_fixture.state, CANDIDATES,
                  probe_fixture.program_arena, program_need - 1u) != 0 &&
                  memcmp(&unchanged_program, &sentinel,
                         sizeof sentinel) == 0,
              "undersized program arena partially published");
    }
    CHECK(salt_text_verify_program_compile(&probe_fixture.program,
              &probe_fixture.model, &probe_fixture.state, CANDIDATES,
              probe_fixture.program_arena, program_need) == 0 &&
              program_plan_valid(&probe_fixture.program),
          "mode-neutral architecture program invalid");
    CHECK(salt_text_verify_cpu_arena_requirement(
              &probe_fixture.program, &cpu_need) == 0 &&
              cpu_need <= sizeof probe_fixture.cpu_arena,
          "CPU arena requirement failed");
    memset(&unchanged_cpu, 0xa5, sizeof unchanged_cpu);
    {
        SaltTextVerifyCpuContext sentinel = unchanged_cpu;
        CHECK(salt_text_verify_cpu_compile(&unchanged_cpu,
                  &probe_fixture.program, probe_fixture.cpu_arena,
                  cpu_need - 1u) != 0 &&
                  memcmp(&unchanged_cpu, &sentinel, sizeof sentinel) == 0,
              "undersized CPU arena partially published");
    }
    CHECK(salt_text_verify_cpu_compile(&probe_fixture.cpu,
              &probe_fixture.program, probe_fixture.cpu_arena, cpu_need) == 0 &&
          salt_text_verify_cpu_executor_init(
              &probe_fixture.executor, &probe_fixture.cpu) == 0 &&
          executor_plan_valid(&probe_fixture.program,
              probe_fixture.executor.plan,
              SALT_TEXT_EXECUTION_CPU_ONLY, 0u, 0u),
          "CPU executor did not own its complete assignment overlay");
    {
        SaltTextTargetNode nodes[CANDIDATES];
        SaltTextTargetFrontier frontier;
        unsigned char *task_begin =
            (unsigned char *)(void *)probe_fixture.cpu.area_tasks;
        unsigned char *task_end;
        memset(nodes, 0, sizeof nodes);
        for (uint32_t node = 0; node < CANDIDATES; node++)
            nodes[node] = (SaltTextTargetNode) {
                (int32_t)node, 700u + node, SALT_TEXT_TARGET_NO_PARENT,
                0u, node, 0u,
            };
        frontier = (SaltTextTargetFrontier) {
            nodes, CANDIDATES, 77u,
        };
        task_end = task_begin +
            (size_t)probe_fixture.cpu.area_task_capacity * sizeof(SaltAreaTask);
        CHECK(probe_fixture.cpu.area_task_capacity == CANDIDATES + 31u &&
                  probe_fixture.cpu.area_frontier.tasks ==
                      probe_fixture.cpu.area_tasks &&
                  probe_fixture.cpu.area_frontier.capacity ==
                      probe_fixture.cpu.area_task_capacity &&
                  task_begin >= probe_fixture.cpu.arena &&
                  task_end <= probe_fixture.cpu.arena +
                      probe_fixture.cpu.arena_bytes &&
                  salt_text_target_frontier_build_area_tasks(
                      &frontier, &probe_fixture.cpu.area_frontier,
                      9u, -1, 32u, 32u) == 0 &&
                  probe_fixture.cpu.area_frontier.count == 32u,
              "fixed startup area-task capacity is incomplete");
        for (uint32_t task = 0;
             task < probe_fixture.cpu.area_frontier.count; task++)
            CHECK(probe_fixture.cpu.area_tasks[task].generation == 77u &&
                      probe_fixture.cpu.area_tasks[task].phase == 9u &&
                      probe_fixture.cpu.area_tasks[task].candidate_count == 1u &&
                      probe_fixture.cpu.area_tasks[task].output_count > 0u &&
                      probe_fixture.cpu.area_tasks[task].state ==
                          SALT_AREA_TASK_QUEUED,
                  "fixed startup area task changed immutable fields");
    }

    CHECK(probe_fixture.program.dispatch.cell_count <= MAX_EXECUTION_CELLS,
          "synthetic assignment capacity too small");
    program_snapshot = probe_fixture.program;
    memset(&mixed_policy, 0, sizeof mixed_policy);
    mixed_policy.execution_class = SALT_TEXT_EXECUTION_MIXED;
    mixed_policy.cpu_rows = 2u;
    mixed_policy.cpu_jobs = 3u;
    memset(&gpu_policy, 0, sizeof gpu_policy);
    gpu_policy.execution_class = SALT_TEXT_EXECUTION_GPU_ONLY;
    CHECK(salt_text_executor_plan_compile(&mixed_plan,
              &probe_fixture.program, &mixed_policy, mixed_assignments,
              MAX_EXECUTION_CELLS) == 0 &&
          executor_plan_valid(&probe_fixture.program, &mixed_plan,
              SALT_TEXT_EXECUTION_MIXED, 2u, 3u) &&
          salt_text_executor_plan_compile(&gpu_plan,
              &probe_fixture.program, &gpu_policy, gpu_assignments,
              MAX_EXECUTION_CELLS) == 0 &&
          executor_plan_valid(&probe_fixture.program, &gpu_plan,
              SALT_TEXT_EXECUTION_GPU_ONLY, 0u, 0u) &&
          mixed_plan.program == gpu_plan.program &&
          mixed_plan.program == &probe_fixture.program &&
          memcmp(&program_snapshot, &probe_fixture.program,
                 sizeof program_snapshot) == 0,
          "executor overlays recompiled or mutated the model program");

    overflow = probe_fixture.model;
    overflow.maximum_context = UINT32_MAX;
    CHECK(salt_text_verify_program_arena_requirement(
              &overflow, UINT32_MAX, &ignored) != 0,
          "overflowing arena geometry was accepted");

    package->model.runtime_ready = 0;
    for (uint32_t layer = 0; layer < LAYERS; layer++)
        probe_fixture.plans[layer].runtime_ready = 0;
    private_state = probe_fixture.state;
    CHECK(salt_text_verify_program_arena_requirement(
              &probe_fixture.model, CANDIDATES, &private_need) == 0 &&
              private_need <= sizeof private_arena &&
              salt_text_verify_program_compile(&private_program,
                  &probe_fixture.model, &private_state, CANDIDATES,
                  private_arena, private_need) == 0 && private_program.ready,
          "private runtime_ready=false qualification was rejected");
    probe_fixture.plans[0].runtime_ready = 1;
    CHECK(salt_text_verify_program_arena_requirement(
              &probe_fixture.model, CANDIDATES, &ignored) != 0,
          "model/plan readiness mismatch was accepted");
    probe_fixture.plans[0].runtime_ready = 0;
    package->model.runtime_ready = 2;
    CHECK(salt_text_verify_program_arena_requirement(
              &probe_fixture.model, CANDIDATES, &ignored) != 0,
          "non-boolean model readiness was accepted");
    package->model.runtime_ready = 1;
    for (uint32_t layer = 0; layer < LAYERS; layer++)
        probe_fixture.plans[layer].runtime_ready = 1;
    puts("mode-neutral program + executor-owned assignments: PASS");
    return 0;
}

static int run_scenario(SyntheticPackage *package,
                        SaltTextKvPrivateMode mode,
                        uint32_t source, const char *label) {
    float parent_logits[VOCABULARY];
    float serial_logits[CANDIDATES][VOCABULARY];
    int32_t candidates[CANDIDATES];
    KvSnapshot serial_kv[CANDIDATES + 1u];
    KvSnapshot initial;
    SaltTextVerifyResult serial_result, block_result, deterministic_result;
    SaltTextProgressiveResult progressive_result;
    SaltTextProgressivePlan progressive_plan;
    SaltTextExecuteAllResult all_result;
    SaltTextExecuteAllResult heterogeneous_result;
    SaltTextVerifyHeterogeneousContext heterogeneous_context;
    SaltTextVerifyExecutor heterogeneous_executor;
    SaltTextDispatchPolicy gpu_policy;
    DescriptorSnapshot block_descriptors;
    KvSnapshot deterministic_kv;
    uint32_t scratch_queries_before;
    SaltTextTouchedSpan poison_spans[SALT_TEXT_MAX_TOUCHED_SPANS];
    uint32_t poison_span_count = 0;
    uint64_t poison_span_bytes = 0;
    uint64_t all_output_span_bytes = 0;
    uint64_t one_output_span_bytes = 0;
    uint64_t zero_output_span_bytes = 0;
    size_t poison_offset = SIZE_MAX;
    uint64_t expected_all_kv =
        (uint64_t)LAYERS * CANDIDATES * 2u * KV_WIDTH * sizeof(float);
    select_parent_token(parent_logits, 1u);
    candidates[0] = 1;
    CHECK(prepare_execution(&serial_fixture, package, mode, source) == 0 &&
              compile_execution(&serial_fixture) == 0,
          "serial fixture compilation failed");
    {
        uint32_t score_rows = 0;
        CHECK(salt_text_attention_score_rows(&serial_fixture.program,
                  source, 0u, &score_rows) == 0 &&
                  score_rows > 0u && score_rows <= source + 1u &&
                  score_rows < serial_fixture.program.maximum_context,
              "active attention score span followed capacity instead of position");
        CHECK(salt_text_touched_span_plan(&serial_fixture.program, 1u,
                  score_rows, poison_spans, SALT_TEXT_MAX_TOUCHED_SPANS,
                  &poison_span_count, &poison_span_bytes) == 0 &&
                  poison_span_bytes < serial_fixture.program.layout.total_bytes,
              "B1 touched-span plan did not leave a poisonable span");
    }
    memset(serial_fixture.cpu_arena, 0xa5,
        serial_fixture.program.layout.total_bytes);
    for (size_t offset = 0;
            offset < serial_fixture.program.layout.total_bytes; offset++) {
        int touched = 0;
        for (uint32_t span = 0; span < poison_span_count; span++)
            if (offset >= poison_spans[span].offset &&
                offset - poison_spans[span].offset < poison_spans[span].bytes) {
                touched = 1;
                break;
            }
        if (!touched) {
            poison_offset = offset;
            break;
        }
    }
    CHECK(poison_offset != SIZE_MAX,
          "B1 touched-span plan covered the full canonical arena");
    snapshot_kv(&serial_fixture, &initial);
    serial_kv[0] = initial;
    for (uint32_t row = 0; row < CANDIDATES; row++) {
        SaltTextTargetBlock block;
        memset(&block, 0, sizeof block);
        block.transition_generation = (uint64_t)row + 1u;
        block.source_position = source + row;
        block.candidate_token_ids = &candidates[row];
        block.candidate_count = 1;
        block.parent_logits = row == 0
            ? parent_logits : serial_logits[row - 1u];
        CHECK(salt_text_verify_execute(&serial_fixture.program,
                  &serial_fixture.executor, &block, &serial_result) == 0 &&
                  serial_result.status == SALT_TEXT_VERIFY_COMMITTED &&
                  serial_result.accepted_count == 1u &&
                  serial_result.produced_count == 2u &&
                  serial_result.committed_count == 1u,
              "ordinary serial target step failed");
        memcpy(serial_logits[row], fixture_logits(&serial_fixture),
               sizeof serial_logits[row]);
        snapshot_kv(&serial_fixture, &serial_kv[row + 1u]);
        if (row + 1u < CANDIDATES)
            candidates[row + 1u] = serial_result.pending_token_id;
    }
    CHECK(serial_fixture.cpu_arena[poison_offset] == 0xa5,
          "B1 submit cleared an untouched canonical byte");

    {
        SaltTextProjectionWindow previous;
        float current_logits[VOCABULARY];
        int32_t proposed[CANDIDATES];
        uint32_t total = 0u, reused_total = 0u;
        CHECK(prepare_execution(&block_fixture, package, mode, source) == 0 &&
              compile_execution(&block_fixture) == 0,
              "projection cascade fixture compilation failed");
        memset(&previous, 0, sizeof previous);
        memcpy(current_logits, parent_logits, sizeof current_logits);
        while (total < CANDIDATES) {
            SaltTextTargetBlock block;
            SaltTextVerifyResult resolved;
            uint32_t count = CANDIDATES - total, reused = 0u;
            int proposal_rc = salt_text_projection_refill_candidates(
                      &block_fixture.program, &previous, current_logits,
                      count, 1u, proposed, CANDIDATES, &reused);
            CHECK(proposal_rc >= 0,
                  "projection cascade candidate extraction failed");
            if (proposal_rc == 1) {
                /* Seed this component test from its existing serial oracle;
                 * runtime absence must not manufacture a candidate block. */
                if (total != 0u) count = 1u;
                memcpy(proposed, candidates + total,
                       (size_t)count * sizeof proposed[0]);
            }
            reused_total += reused;
            if (total == 0u && count > 1u)
                proposed[1] = (candidates[1] + 1) % VOCABULARY;
            memset(&block, 0, sizeof block);
            block.transition_generation =
                block_fixture.program.kv_state->transition_generation + 1u;
            block.source_position = source + total;
            block.candidate_token_ids = proposed;
            block.candidate_count = count;
            block.parent_logits = current_logits;
            CHECK(salt_text_verify_execute(&block_fixture.program,
                      &block_fixture.executor, &block, &resolved) == 0 &&
                  resolved.accepted_count > 0u &&
                  resolved.accepted_count <= count &&
                  resolved.committed_count == resolved.accepted_count &&
                  resolved.produced_count == resolved.accepted_count + 1u &&
                  resolved.projection_rows == count &&
                  resolved.projection_logits != NULL,
                  "projection cascade target transition failed");
            for (uint32_t row = 0u; row < resolved.accepted_count; ++row)
                CHECK(proposed[row] == candidates[total + row],
                      "projection guesses overrode target token authority");
            total += resolved.accepted_count;
            CHECK(kv_matches(&block_fixture, &serial_kv[total]) &&
                  memcmp(resolved.pending_logits, serial_logits[total - 1u],
                         sizeof current_logits) == 0,
                  "projection cascade KV/pending logits differ from serial");
            previous.logits = resolved.projection_logits;
            previous.generation = resolved.transition_generation;
            previous.position = resolved.result_position;
            previous.sequence_tiles = count;
            previous.route_count = 1u;
            previous.accepted_count = resolved.accepted_count;
            previous.winning_route = 0u;
            memcpy(current_logits, resolved.pending_logits,
                   sizeof current_logits);
        }
        CHECK(reused_total > 0u,
              "projection continuation did not consume prior TARGET rows");
        printf("projection continuation: PASS %s reused=%u "
               "tokens/KV/pending-logits=byte-identical\n", label, reused_total);
    }

    {
        SaltTextTargetNode nodes[CANDIDATES];
        SaltTextTargetFrontierBlock frontier_block;
        SaltTextVerifyResult frontier_result;
        SaltAreaTask area_tasks[16];
        SaltAreaFrontier area;
        int kept = 0, cancelled = 0;
        CHECK(prepare_execution(
                  &frontier_root_fixture, package, mode, source) == 0 &&
              compile_execution(&frontier_root_fixture) == 0,
              "target root-frontier fixture compilation failed");
        memset(nodes, 0, sizeof nodes);
        for (uint32_t branch = 0; branch < CANDIDATES; branch++)
            nodes[branch] = (SaltTextTargetNode) {
                (candidates[0] + (int32_t)branch) % VOCABULARY,
                300u + branch, SALT_TEXT_TARGET_NO_PARENT,
                0u, branch, 0u,
            };
        memset(area_tasks, 0, sizeof area_tasks);
        memset(&area, 0, sizeof area);
        memset(&frontier_block, 0, sizeof frontier_block);
        frontier_block.source_position = source;
        frontier_block.frontier.nodes = nodes;
        frontier_block.frontier.node_count = CANDIDATES;
        frontier_block.frontier.generation = 90u;
        frontier_block.parent_logits = parent_logits;
        CHECK(salt_area_frontier_bind(&area, area_tasks, 16u) == 0 &&
              salt_text_target_frontier_build_area_tasks(
                  &frontier_block.frontier, &area,
                  1u, -1, 4u, 4u) == 0,
              "target root-frontier task materialization failed");
        frontier_block.area_frontier = &area;
        memset(&frontier_result, 0, sizeof frontier_result);
        CHECK(salt_text_verify_frontier_execute(
                  &frontier_root_fixture.program,
                  &frontier_root_fixture.executor,
                  &frontier_block, &frontier_result) == 0 &&
              frontier_result.accepted_count == 1u &&
              frontier_result.committed_count == 1u &&
              frontier_result.produced_count == 2u &&
              frontier_result.pending_token_id == candidates[1] &&
              frontier_result.winning_node_index == 0u &&
              frontier_result.winning_node_id == 300u &&
              frontier_result.backend.engine_submissions == 1u &&
              frontier_result.backend.completion_fences == 1u &&
              frontier_result.backend.initial_transfer_bytes ==
                  sizeof(int32_t) &&
              frontier_result.backend.final_logits_transfer_bytes ==
                  (uint64_t)VOCABULARY * sizeof(float) &&
              frontier_result.backend.tentative_scrub_bytes == 0u,
              "target root-frontier did not collapse to one B1 transition");
        CHECK(kv_matches(&frontier_root_fixture, &serial_kv[1]),
              "target root-frontier KV differs from one serial step");
        for (uint32_t index = 0; index < area.count; index++) {
            if (area.tasks[index].node_id == 300u &&
                area.tasks[index].state == SALT_AREA_TASK_QUEUED)
                kept++;
            else if (area.tasks[index].state == SALT_AREA_TASK_CANCELLED)
                cancelled++;
        }
        CHECK(kept == 1 && cancelled == (int)area.count - 1,
              "target root-frontier losers were not retired before submit");
    }

    {
        int32_t seeds[CANDIDATES * CANDIDATES];
        SaltTextTargetNode node_scratch[CANDIDATES];
        SaltTextTargetCascadeBlock cascade;
        SaltTextVerifyResult cascade_result;
        for (uint32_t tile = 0; tile < CANDIDATES; tile++)
            for (uint32_t branch = 0; branch < CANDIDATES; branch++)
                seeds[tile * CANDIDATES + branch] =
                    (candidates[tile] + (int32_t)branch) % VOCABULARY;
        CHECK(prepare_execution(
                  &frontier_cascade_fixture, package, mode, source) == 0 &&
              compile_execution(&frontier_cascade_fixture) == 0,
              "target cascade fixture compilation failed");
        memset(&cascade, 0, sizeof cascade);
        cascade.first_transition_generation = 91u;
        cascade.source_position = source;
        cascade.seed_token_ids = seeds;
        cascade.tile_count = CANDIDATES;
        cascade.frontier_width = CANDIDATES;
        cascade.parent_logits = parent_logits;
        cascade.node_scratch = node_scratch;
        cascade.node_scratch_capacity = CANDIDATES;
        memset(&cascade_result, 0, sizeof cascade_result);
        CHECK(salt_text_verify_cascade_execute(
                  &frontier_cascade_fixture.program,
                  &frontier_cascade_fixture.executor,
                  &cascade, &cascade_result) == 0 &&
              cascade_result.accepted_count == CANDIDATES &&
              cascade_result.committed_count == CANDIDATES &&
              cascade_result.produced_count == CANDIDATES + 1u &&
              cascade_result.pending_token_id == serial_result.pending_token_id &&
              cascade_result.result_position == source + CANDIDATES &&
              cascade_result.backend.engine_submissions == CANDIDATES &&
              cascade_result.backend.completion_fences == CANDIDATES &&
              cascade_result.backend.initial_transfer_bytes ==
                  CANDIDATES * sizeof(int32_t) &&
              cascade_result.backend.final_logits_transfer_bytes ==
                  (uint64_t)CANDIDATES * VOCABULARY * sizeof(float) &&
              cascade_result.backend.tentative_scrub_bytes == 0u,
              "target N-tile cascade changed");
        CHECK(kv_matches(&frontier_cascade_fixture,
                  &serial_kv[CANDIDATES]),
              "target N-tile cascade KV differs from serial steps");

        CHECK(prepare_execution(&frontier_cascade_reject_fixture,
                  package, mode, source) == 0 &&
              compile_execution(&frontier_cascade_reject_fixture) == 0,
              "target cascade rejection fixture compilation failed");
        seeds[2u * CANDIDATES] =
            (candidates[2] + CANDIDATES + 1) % VOCABULARY;
        cascade.first_transition_generation = 101u;
        memset(&cascade_result, 0, sizeof cascade_result);
        CHECK(salt_text_verify_cascade_execute(
                  &frontier_cascade_reject_fixture.program,
                  &frontier_cascade_reject_fixture.executor,
                  &cascade, &cascade_result) == 0 &&
              cascade_result.accepted_count == 2u &&
              cascade_result.committed_count == 2u &&
              cascade_result.produced_count == 3u &&
              cascade_result.pending_token_id == candidates[2] &&
              cascade_result.result_position == source + 2u &&
              cascade_result.backend.engine_submissions == 2u &&
              cascade_result.backend.completion_fences == 2u &&
              cascade_result.backend.initial_transfer_bytes ==
                  2u * sizeof(int32_t) &&
              cascade_result.backend.tentative_scrub_bytes == 0u,
              "target N-tile cascade failed to stop at first miss");
        CHECK(kv_matches(&frontier_cascade_reject_fixture, &serial_kv[2]),
              "target N-tile cascade replayed or changed committed prefix");
    }

    {
        int32_t routes[2][CANDIDATES];
        SaltTextTargetRouteBlock route;
        SaltTextVerifyResult route_result;
        for (uint32_t tile = 0; tile < CANDIDATES; tile++) {
            routes[0][tile] = candidates[tile];
            routes[1][tile] = (candidates[tile] + 1) % VOCABULARY;
        }
        CHECK(prepare_execution(&route_fixture, package, mode, source) == 0 &&
              compile_execution(&route_fixture) == 0,
              "target route fixture compilation failed");
        memset(&route, 0, sizeof route);
        route.transition_generation = 110u;
        route.source_position = source;
        route.route_token_ids = &routes[0][0];
        route.sequence_tiles = CANDIDATES;
        route.route_count = 2u;
        route.parent_logits = parent_logits;
        memset(&route_result, 0, sizeof route_result);
        CHECK(salt_text_verify_route_execute(
                  &route_fixture.program, &route_fixture.executor,
                  &route, &route_result) == 0 &&
              route_result.accepted_count == CANDIDATES &&
              route_result.committed_count == CANDIDATES &&
              route_result.produced_count == CANDIDATES + 1u &&
              route_result.pending_token_id == serial_result.pending_token_id &&
              route_result.winning_node_index == CANDIDATES - 1u &&
              route_result.projection_rows == CANDIDATES &&
              route_result.backend.engine_submissions == 1u &&
              route_result.backend.completion_fences == 1u &&
              route_result.backend.initial_transfer_bytes ==
                  CANDIDATES * sizeof(int32_t),
              "N4-F2 route selection changed");
        CHECK(kv_matches(&route_fixture, &serial_kv[CANDIDATES]),
              "N4-F2 selected-route KV differs from serial steps");

        CHECK(prepare_execution(
                  &route_reject_fixture, package, mode, source) == 0 &&
              compile_execution(&route_reject_fixture) == 0,
              "target route rejection fixture compilation failed");
        routes[0][2] = (candidates[2] + 3) % VOCABULARY;
        route.transition_generation = 111u;
        memset(&route_result, 0, sizeof route_result);
        CHECK(salt_text_verify_route_execute(
                  &route_reject_fixture.program,
                  &route_reject_fixture.executor,
                  &route, &route_result) == 0 &&
              route_result.accepted_count == 2u &&
              route_result.committed_count == 2u &&
              route_result.produced_count == 3u &&
              route_result.pending_token_id == candidates[2] &&
              route_result.result_position == source + 2u &&
              route_result.winning_node_index == 1u &&
              route_result.backend.engine_submissions == 1u &&
              route_result.backend.completion_fences == 1u &&
              route_result.projection_rows == CANDIDATES &&
              route_result.backend.initial_transfer_bytes ==
                  CANDIDATES * sizeof(int32_t),
              "N4-F2 depth miss changed accepted prefix");
        CHECK(kv_matches(&route_reject_fixture, &serial_kv[2]),
              "N4-F2 depth miss changed committed KV");

        /* Resume only the corrected suffix through the same engine route used
         * by a selected production trajectory; never replay accepted rows. */
        route.source_position = source + 2u;
        route.transition_generation = 112u;
        route.route_token_ids = candidates + 2u;
        route.sequence_tiles = CANDIDATES - 2u;
        route.route_count = 1u;
        route.parent_logits = route_result.pending_logits;
        CHECK(salt_text_verify_route_execute(
                  &route_reject_fixture.program,
                  &route_reject_fixture.executor,
                  &route, &route_result) == 0 &&
              route_result.accepted_count == CANDIDATES - 2u &&
              route_result.result_position == source + CANDIDATES &&
              route_result.pending_token_id == serial_result.pending_token_id &&
              route_result.projection_rows == CANDIDATES - 2u &&
              route_result.backend.engine_submissions == 1u &&
              route_result.backend.completion_fences == 1u &&
              route_result.backend.initial_transfer_bytes ==
                  (CANDIDATES - 2u) * sizeof(int32_t),
              "X suffix was split or replayed accepted prefix");
        CHECK(kv_matches(&route_reject_fixture, &serial_kv[CANDIDATES]),
              "X suffix continuation changed complete KV");
        route.source_position = source;
        route.route_token_ids = &routes[0][0];
        route.sequence_tiles = CANDIDATES;
        route.route_count = 2u;
        route.parent_logits = parent_logits;

        CHECK(prepare_execution(&route_root_miss_fixture,
                  package, mode, source) == 0 &&
              compile_execution(&route_root_miss_fixture) == 0,
              "target route root-miss fixture compilation failed");
        routes[0][0] = (candidates[0] + 3) % VOCABULARY;
        route.transition_generation = 112u;
        memset(&route_result, 0, sizeof route_result);
        CHECK(salt_text_verify_route_execute(
                  &route_root_miss_fixture.program,
                  &route_root_miss_fixture.executor,
                  &route, &route_result) == 0 &&
              route_result.accepted_count == 0u &&
              route_result.committed_count == 0u &&
              route_result.produced_count == 1u &&
              route_result.pending_token_id == candidates[0] &&
              route_result.backend.engine_submissions == 0u &&
              route_result.backend.completion_fences == 0u,
              "N4-F2 root miss submitted model work");
        CHECK(kv_matches(&route_root_miss_fixture, &initial),
              "N4-F2 root miss changed KV");
    }

    {
        SaltAreaNfqMatrix matrix;
        SaltAreaNfqRoute routes[1];
        SaltAreaNfqItem items[CANDIDATES * 4u];
        SaltTextTargetNode nodes[CANDIDATES];
        uint32_t route_ids[1], ready_items[32];
        int32_t tokens[CANDIDATES * 4u];
        SaltTextTargetNfqBlock nfq;
        SaltTextVerifyResult target;
        const uint32_t checks = CANDIDATES * 4u;
        const uint32_t late_winner = 1u * 4u + 2u;
        /* One late local-Q win and one complete NFQ exhaustion. These supplied
         * IDs test matrix/check ownership, not a production proposal source. */
        for (uint32_t cell = 0; cell < 2u; cell++) {
            uint32_t accepted = 0u;
            int has_winner = cell == 0u;
            int32_t miss = (candidates[0] + 1) % VOCABULARY;
            CHECK(prepare_execution(&route_fixture, package, mode, source) == 0 &&
                  compile_execution(&route_fixture) == 0 &&
                  salt_area_nfq_bind(&matrix, routes, 1u,
                      items, checks) == 0,
                  "core NFQ fixture binding failed");
            for (uint32_t index = 0; index < checks; index++)
                tokens[index] = (miss + (int32_t)(index / 4u)) % VOCABULARY;
            if (cell == 0u) tokens[late_winner] = candidates[0];
            memset(&nfq, 0, sizeof nfq);
            nfq.route.transition_generation = 120u + cell;
            nfq.route.source_position = source;
            nfq.route.route_token_ids = tokens;
            nfq.route.sequence_tiles = CANDIDATES;
            nfq.route.route_count = 1u;
            nfq.route.parent_logits = parent_logits;
            nfq.plan = (SaltAreaNfqPlan) { 2u, CANDIDATES, 1u, 4u };
            nfq.matrix = &matrix;
            nfq.nodes = nodes;
            nfq.node_capacity = CANDIDATES;
            nfq.route_ids = route_ids;
            nfq.route_capacity = 1u;
            nfq.ready_items = ready_items;
            nfq.ready_capacity = 32u;
            {
                int nfq_rc = salt_text_verify_nfq_execute(
                    &route_fixture.program, &route_fixture.executor,
                    &nfq, &target);
                if (nfq_rc != 0 || target.accepted_count != accepted ||
                    target.winning_node_id !=
                        (has_winner ? late_winner : UINT32_MAX))
                    fprintf(stderr,
                        "NFQ_FIXTURE rc=%d cell=%u accepted=%u committed=%u "
                        "produced=%u pending=%d position=%u generation=%llu "
                        "winner_row=%u winner_item=%u rows=%u queued=%u "
                        "executed=%u cancelled=%u submissions=%u fences=%u\n",
                        nfq_rc, cell, target.accepted_count,
                        target.committed_count, target.produced_count,
                        target.pending_token_id, target.result_position,
                        (unsigned long long)target.transition_generation,
                        target.winning_node_index, target.winning_node_id,
                        target.projection_rows,
                        target.backend.production_frontier_tasks_queued,
                        target.backend.production_frontier_tasks_executed,
                        target.backend.production_frontier_queued_cancellations,
                        target.backend.engine_submissions,
                        target.backend.completion_fences);
                CHECK(nfq_rc == 0 &&
                  target.status == SALT_TEXT_VERIFY_COMMITTED &&
                  target.accepted_count == accepted &&
                  target.committed_count == accepted &&
                  target.produced_count == accepted + 1u &&
                  target.pending_token_id == (accepted
                      ? candidates[1] : candidates[0]) &&
                  target.result_position == source + accepted &&
                  target.transition_generation == 120u + cell &&
                  target.backend.engine_submissions == (accepted != 0u) &&
                  target.backend.completion_fences == (accepted != 0u) &&
                  target.backend.intermediate_host_publications == 0u &&
                  target.backend.production_frontier_tasks_queued ==
                      checks &&
                  (has_winner
                    ? target.backend.production_frontier_tasks_executed > 0u &&
                      target.backend.production_frontier_tasks_executed < checks
                    : target.backend.production_frontier_tasks_executed == checks) &&
                  target.projection_rows == 0u &&
                  matrix.resolved && !matrix.ready,
                  "core NFQ target/queue result changed");
            }
            CHECK(kv_matches(&route_fixture, &serial_kv[accepted]) &&
                  target.pending_logits_count == VOCABULARY &&
                  memcmp(target.pending_logits, accepted == 0u
                      ? parent_logits : serial_logits[accepted - 1u],
                      sizeof parent_logits) == 0,
                  "core NFQ KV or pending logits differ from serial target");
            if (has_winner)
                CHECK(target.winning_node_index == late_winner &&
                      target.winning_node_id == late_winner,
                      "core NFQ winner did not retain (n,q) identity");
            CHECK(salt_text_verify_nfq_execute(
                      &route_fixture.program, &route_fixture.executor,
                      &nfq, &target) != 0 &&
                  kv_matches(&route_fixture, &serial_kv[accepted]),
                  "core NFQ readmitted an old epoch");
        }
        puts("core NFQ ownership: local Q, exhaustion, winner, replay PASS");

    }

    {
        SaltTextTargetNode nodes[CANDIDATES];
        SaltTextTargetFrontierBlock frontier_block;
        SaltTextVerifyResult frontier_result;
        SaltAreaTask area_tasks[16];
        SaltAreaFrontier area;
        const float *frontier_logits;
        int saw_winning_root = 0, saw_winning_child = 0;
        int saw_cancelled_root = 0, saw_cancelled_child = 0;
        CHECK(prepare_execution(&frontier_fixture, package, mode, source) == 0 &&
                  compile_execution(&frontier_fixture) == 0,
              "target branch fixture compilation failed");
        CHECK(salt_text_verify_cpu_parallel_bind(&frontier_fixture.cpu,
                  attention_test_workers, &attention_worker_calls,
                  attention_worker_scores, 2u * MAXIMUM_CONTEXT, 2u) == 0,
              "frontier attention worker binding failed");
        memset(nodes, 0, sizeof nodes);
        nodes[0] = (SaltTextTargetNode) {
            candidates[0], 100u, SALT_TEXT_TARGET_NO_PARENT, 0u, 2u, 0u,
        };
        nodes[1] = (SaltTextTargetNode) {
            (candidates[0] + 1) % VOCABULARY, 101u,
            SALT_TEXT_TARGET_NO_PARENT, 0u, 0u, 0u,
        };
        nodes[2] = (SaltTextTargetNode) {
            candidates[1], 200u, 0u, 1u, 3u, 0u,
        };
        nodes[3] = (SaltTextTargetNode) {
            (candidates[1] + 1) % VOCABULARY, 201u, 0u, 1u, 1u, 0u,
        };
        memset(area_tasks, 0, sizeof area_tasks);
        memset(&area, 0, sizeof area);
        memset(&frontier_block, 0, sizeof frontier_block);
        frontier_block.source_position = source;
        frontier_block.frontier.nodes = nodes;
        frontier_block.frontier.node_count = CANDIDATES;
        frontier_block.frontier.generation = 100u;
        frontier_block.parent_logits = parent_logits;
        CHECK(salt_area_frontier_bind(&area, area_tasks, 16u) == 0 &&
                  salt_text_target_frontier_build_area_tasks(
                      &frontier_block.frontier, &area, 1u, -1, 8u, 8u) == 0,
              "target branch task materialization failed");
        frontier_block.area_frontier = &area;
        memset(&frontier_result, 0, sizeof frontier_result);
        CHECK(salt_text_verify_frontier_execute(
                  &frontier_fixture.program, &frontier_fixture.executor,
                  &frontier_block, &frontier_result) == 0 &&
                  frontier_result.status == SALT_TEXT_VERIFY_COMMITTED &&
                  frontier_result.accepted_count == 2u &&
                  frontier_result.committed_count == 2u &&
                  frontier_result.produced_count == 3u &&
                  frontier_result.pending_token_id == candidates[2] &&
                  frontier_result.result_position == source + 2u &&
                  frontier_result.winning_node_index == 2u &&
                  frontier_result.winning_node_id == 200u &&
                  frontier_result.backend.engine_submissions == 1u &&
                  frontier_result.backend.completion_fences == 1u &&
                  frontier_result.backend.intermediate_host_publications == 0u,
              "target branch transaction changed");
        frontier_logits = fixture_logits(&frontier_fixture);
        CHECK(memcmp(frontier_logits +
                         (size_t)nodes[0].tentative_state_slot * VOCABULARY,
                     serial_logits[0], sizeof serial_logits[0]) == 0 &&
                  memcmp(frontier_logits +
                         (size_t)nodes[2].tentative_state_slot * VOCABULARY,
                     serial_logits[1], sizeof serial_logits[1]) == 0,
              "target branch winning logits differ from serial path");
        CHECK(kv_matches(&frontier_fixture, &serial_kv[2]),
              "target branch committed KV differs from serial path");
        for (uint32_t index = 0; index < area.count; index++) {
            if (area.tasks[index].node_id == 100u &&
                area.tasks[index].state == SALT_AREA_TASK_QUEUED)
                saw_winning_root = 1;
            else if (area.tasks[index].node_id == 200u &&
                area.tasks[index].state == SALT_AREA_TASK_QUEUED)
                saw_winning_child = 1;
            else if (area.tasks[index].node_id == 101u &&
                area.tasks[index].state == SALT_AREA_TASK_CANCELLED)
                saw_cancelled_root = 1;
            else if (area.tasks[index].node_id == 201u &&
                area.tasks[index].state == SALT_AREA_TASK_CANCELLED)
                saw_cancelled_child = 1;
        }
        CHECK(saw_winning_root && saw_winning_child &&
                  saw_cancelled_root && saw_cancelled_child,
              "target branch cancellation did not preserve only winning path");
    }

    for (int proof = 0; proof <= 1; proof++) {
    CHECK(prepare_execution(&all_fixture, package, mode, source) == 0 &&
              compile_execution(&all_fixture) == 0,
          "ordinary all-row fixture compilation failed");
    CHECK(all_fixture.program.proof_state == 0,
          "compiled program enabled proof by default");
    all_fixture.program.proof_state = proof;
    {
        SaltTextTouchedSpan spans[SALT_TEXT_MAX_TOUCHED_SPANS];
        uint32_t span_count = 0, score_rows = 0;
        CHECK(salt_text_attention_score_rows(&all_fixture.program,
                  source, CANDIDATES - 1u, &score_rows) == 0 &&
              salt_text_touched_span_plan(&all_fixture.program, CANDIDATES,
                  score_rows, spans, SALT_TEXT_MAX_TOUCHED_SPANS,
                  &span_count, &all_output_span_bytes) == 0 &&
              salt_text_touched_span_plan_outputs(&all_fixture.program,
                  CANDIDATES, score_rows, 1u, spans,
                  SALT_TEXT_MAX_TOUCHED_SPANS, &span_count,
                  &one_output_span_bytes) == 0 &&
              salt_text_touched_span_plan_outputs(&all_fixture.program,
                  CANDIDATES, score_rows, 0u, spans,
                  SALT_TEXT_MAX_TOUCHED_SPANS, &span_count,
                  &zero_output_span_bytes) == 0 &&
              zero_output_span_bytes < one_output_span_bytes &&
              one_output_span_bytes < all_output_span_bytes,
              "ordinary PREFILL output spans were not reduced 0 < 1 < all");
    }
    CHECK(salt_text_execute_all(&all_fixture.program,
              &all_fixture.executor, 100u, source,
              candidates, CANDIDATES, &all_result) == 0 &&
              all_result.status == SALT_TEXT_VERIFY_COMMITTED &&
              all_result.committed_count == CANDIDATES &&
              all_result.result_position == source + CANDIDATES &&
              all_result.transition_generation == 100u &&
              all_result.backend.engine_submissions == 1u &&
              all_result.backend.completion_fences == 1u &&
              all_result.backend.final_kv_publish_bytes == expected_all_kv &&
              all_result.backend.intermediate_host_publications == 0u &&
              all_result.view.canonical_base != NULL &&
              all_result.view.canonical_bytes >=
                  all_fixture.program.layout.total_bytes,
          "ordinary all-row transaction failed");
    CHECK(memcmp(fixture_logits(&all_fixture), serial_logits,
                 sizeof serial_logits) == 0,
          "ordinary all-row logits differ from serial target steps");
    CHECK(kv_matches(&all_fixture, &serial_kv[CANDIDATES]),
          "ordinary all-row KV differs from serial target steps");

    {
        SaltTextVerifyExecutorOps published_ops;
        KvSnapshot before;
        CHECK(prepare_execution(&all_fixture, package, mode, source) == 0 &&
                  compile_execution(&all_fixture) == 0,
              "backend publication fixture compilation failed");
        all_fixture.program.proof_state = proof;
        snapshot_kv(&all_fixture, &before);
        publication_test_ops = all_fixture.executor.ops;
        published_ops = *publication_test_ops;
        published_ops.finish = publication_test_finish;
        all_fixture.executor.ops = &published_ops;
        /* Mock only the publication claim: untouched destination sentinels
         * prove the core does not copy tentative rows on this branch. */
        CHECK(salt_text_execute_all(&all_fixture.program,
                  &all_fixture.executor, 100u, source, candidates,
                  CANDIDATES, &all_result) == 0 &&
                  all_result.status == SALT_TEXT_VERIFY_COMMITTED &&
                  all_result.backend.final_kv_publish_bytes == expected_all_kv &&
                  all_result.view.backend_published_kv == 1u &&
                  kv_matches(&all_fixture, &before),
              "backend publication lost logical bytes or copied host KV");
        all_fixture.executor.ops = publication_test_ops;
    }
    }

    for (int defer_release = 0; defer_release < 2; defer_release++) {
    mock_defer_resources = defer_release;
    mock_consumers_completed = 0;
    CHECK(prepare_execution(&heterogeneous_fixture, package, mode, source) == 0 &&
              compile_execution(&heterogeneous_fixture) == 0 &&
              prepare_execution(&heterogeneous_shadow_fixture, package,
                  mode, source) == 0 &&
              compile_execution(&heterogeneous_shadow_fixture) == 0,
          "heterogeneous fixture compilation failed");
    memset(&gpu_policy, 0, sizeof gpu_policy);
    gpu_policy.execution_class = SALT_TEXT_EXECUTION_GPU_ONLY;
    mock_shadow_executor = &heterogeneous_shadow_fixture.executor;
    mock_need_resource = 1;
    mock_resource_acquires = 0;
    mock_resource_releases = 0;
    mock_gpu_extent_calls = 0;
    mock_authoritative_output_rows = UINT32_MAX;
    memset(&heterogeneous_context, 0, sizeof heterogeneous_context);
    memset(&heterogeneous_executor, 0, sizeof heterogeneous_executor);
    CHECK(salt_text_verify_heterogeneous_compile(&heterogeneous_context,
              &heterogeneous_fixture.program, &gpu_policy, &mock_gpu_ops,
              mock_gpu_startup_arena, sizeof mock_gpu_startup_arena) == 0 &&
              salt_text_verify_heterogeneous_resource_bind(
                  &heterogeneous_context, &mock_resource_ops,
                  &mock_resource_acquires) == 0 &&
              salt_text_verify_heterogeneous_executor_init(
                  &heterogeneous_executor, &heterogeneous_context) == 0,
          "heterogeneous GPU-only executor initialization failed");
    CHECK(salt_text_execute_prefill(&heterogeneous_fixture.program,
              &heterogeneous_executor, 100u, source,
              candidates, CANDIDATES, 1u, &heterogeneous_result) == 0 &&
              heterogeneous_result.status == SALT_TEXT_VERIFY_COMMITTED &&
              heterogeneous_result.committed_count == CANDIDATES &&
              heterogeneous_result.result_position == source + CANDIDATES &&
              heterogeneous_result.backend.engine_submissions == 1u &&
              heterogeneous_result.backend.completion_fences == 1u &&
              heterogeneous_result.backend.intermediate_host_publications == 0u &&
              heterogeneous_result.backend.internal_dependency_barriers + 1u ==
                  heterogeneous_fixture.program.dispatch.cell_count &&
              mock_resource_acquires ==
                  (int)heterogeneous_fixture.program.layer_count &&
              mock_resource_releases == mock_resource_acquires &&
              mock_authoritative_output_rows == 1u &&
              mock_gpu_extent_calls > 0 &&
              mock_gpu_extent_calls <
                  heterogeneous_fixture.program.dispatch.cell_count,
          "heterogeneous GPU-only transaction contract failed");
    CHECK(memcmp(heterogeneous_result.view.canonical_base +
                     heterogeneous_fixture.program.layout.position_logits,
                 serial_logits, sizeof serial_logits) == 0 &&
              kv_matches(&heterogeneous_fixture,
                  &serial_kv[CANDIDATES]),
          "heterogeneous GPU-only logits/KV differ from serial target steps");
    CHECK(salt_text_verify_heterogeneous_destroy(
              &heterogeneous_context) == 0,
          "heterogeneous GPU-only executor teardown failed");
    printf("heterogeneous GPU-only C99 executor (%s, deferred=%d): PASS\n",
           label, defer_release);
    }
    mock_defer_resources = 0;

    CHECK(prepare_execution(&block_fixture, package, mode, source) == 0 &&
              compile_execution(&block_fixture) == 0,
          "target-block fixture compilation failed");
    attention_worker_calls = 0;
    CHECK(salt_text_verify_cpu_parallel_bind(&block_fixture.cpu,
              attention_test_workers, &attention_worker_calls,
              attention_worker_scores, 2u * MAXIMUM_CONTEXT, 2u) == 0,
          "linear attention worker binding failed");
    snapshot_descriptors(&block_fixture, package, &block_descriptors);
    scratch_queries_before = package->scratch_queries;
    {
        SaltTextTargetBlock block;
        memset(&block, 0, sizeof block);
        block.transition_generation = 100u;
        block.source_position = source;
        block.candidate_token_ids = candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&block_fixture.program,
                  &block_fixture.executor, &block, &block_result) == 0,
              "all-accepted target block failed");
    }
    CHECK(block_result.status == SALT_TEXT_VERIFY_COMMITTED &&
              block_result.accepted_count == CANDIDATES &&
              block_result.produced_count == CANDIDATES + 1u &&
              block_result.committed_count == CANDIDATES &&
              block_result.pending_token_id == serial_result.pending_token_id &&
              block_result.result_position == source + CANDIDATES,
          "all-accepted pending bonus boundary changed");
    CHECK(attention_worker_calls > 0,
          "linear TARGET attention bypassed the supplied worker pool");
    CHECK(memcmp(fixture_logits(&block_fixture), serial_logits,
                 sizeof serial_logits) == 0,
          "target-block logits differ from ordinary serial logits");
    CHECK(kv_matches(&block_fixture, &serial_kv[CANDIDATES]),
          "target-block KV bytes differ from ordinary serial KV bytes");
    CHECK(block_result.backend.engine_submissions == 1u &&
              block_result.backend.completion_fences == 1u &&
              block_result.backend.intermediate_host_publications == 0u &&
              block_result.backend.final_kv_publish_bytes == expected_all_kv &&
              block_result.backend.tentative_scrub_bytes == 0u,
          "one-command completion/publication contract changed");
    CHECK(package->scratch_queries == scratch_queries_before,
          "fixed tensor scratch was queried during execution");
    CHECK(descriptors_equal(&block_fixture, package, &block_descriptors),
          "compiled descriptors mutated during execution");

    snapshot_kv(&block_fixture, &deterministic_kv);
    CHECK(prepare_execution(&deterministic_fixture, package, mode, source) == 0 &&
              compile_execution(&deterministic_fixture) == 0,
          "deterministic replay fixture compilation failed");
    {
        SaltTextTargetBlock block;
        memset(&block, 0, sizeof block);
        block.transition_generation = 100u;
        block.source_position = source;
        block.candidate_token_ids = candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&deterministic_fixture.program,
                  &deterministic_fixture.executor, &block,
                  &deterministic_result) == 0,
              "identical target-block replay failed");
    }
    CHECK(deterministic_result.status == block_result.status &&
              deterministic_result.accepted_count == block_result.accepted_count &&
              deterministic_result.produced_count == block_result.produced_count &&
              deterministic_result.committed_count == block_result.committed_count &&
              deterministic_result.pending_token_id ==
                  block_result.pending_token_id &&
              deterministic_result.result_position == block_result.result_position &&
              deterministic_result.transition_generation ==
                  block_result.transition_generation &&
              memcmp(fixture_logits(&deterministic_fixture),
                     fixture_logits(&block_fixture),
                     sizeof serial_logits) == 0 &&
              kv_matches(&deterministic_fixture, &deterministic_kv),
          "identical artifacts/input/sampler/seed produced nondeterministic logits/state/KV");

    CHECK(prepare_execution(&progressive_fixture, package, mode, source) == 0 &&
              compile_execution(&progressive_fixture) == 0,
          "progressive target fixture compilation failed");
    memset(&progressive_plan, 0, sizeof progressive_plan);
    progressive_plan.seed_candidates = 1u;
    progressive_plan.maximum_tile_candidates = CANDIDATES;
    memset(&progressive_result, 0, sizeof progressive_result);
    CHECK(salt_text_verify_progressive_execute(
              &progressive_fixture.program, &progressive_fixture.executor,
              100u, source, candidates, CANDIDATES, parent_logits,
              &progressive_plan, &progressive_result) == 0 &&
              progressive_result.target.status == SALT_TEXT_VERIFY_COMMITTED &&
              progressive_result.target.accepted_count == CANDIDATES &&
              progressive_result.target.produced_count == CANDIDATES + 1u &&
              progressive_result.target.committed_count == CANDIDATES &&
              progressive_result.target.pending_token_id ==
                  serial_result.pending_token_id &&
              progressive_result.target.result_position == source + CANDIDATES &&
              progressive_result.target.transition_generation == 102u &&
              progressive_result.target.pending_logits != NULL &&
              progressive_result.target.pending_logits_count == VOCABULARY &&
              progressive_result.tile_count == 3u &&
              progressive_result.submitted_tile_count == 3u &&
              progressive_result.largest_tile == 2u &&
              progressive_result.target.backend.engine_submissions == 3u &&
              progressive_result.target.backend.completion_fences == 3u &&
              progressive_result.target.backend.intermediate_host_publications == 0u &&
              progressive_result.target.backend.final_kv_publish_bytes ==
                  expected_all_kv &&
              kv_matches(&progressive_fixture, &serial_kv[CANDIDATES]),
          "progressive seed/grow target execution changed canonical result");

    {
        int32_t rejected_candidates[CANDIDATES];
        memcpy(rejected_candidates, candidates, sizeof rejected_candidates);
        rejected_candidates[0] = (rejected_candidates[0] + 1) % VOCABULARY;
        CHECK(prepare_execution(&progressive_reject_fixture,
                  package, mode, source) == 0 &&
                  compile_execution(&progressive_reject_fixture) == 0,
              "progressive rejection fixture compilation failed");
        memset(&progressive_result, 0, sizeof progressive_result);
        CHECK(salt_text_verify_progressive_execute(
                  &progressive_reject_fixture.program,
                  &progressive_reject_fixture.executor,
                  200u, source, rejected_candidates, CANDIDATES, parent_logits,
                  &progressive_plan, &progressive_result) == 0 &&
                  progressive_result.target.accepted_count == 0u &&
                  progressive_result.target.produced_count == 1u &&
                  progressive_result.target.committed_count == 0u &&
                  progressive_result.target.pending_token_id == candidates[0] &&
                  progressive_result.target.result_position == source &&
                  progressive_result.target.pending_logits == parent_logits &&
                  progressive_result.target.pending_logits_count == VOCABULARY &&
                  progressive_result.tile_count == 1u &&
                  progressive_result.submitted_tile_count == 0u &&
                  progressive_result.target.backend.engine_submissions == 0u &&
                  progressive_result.target.backend.completion_fences == 0u &&
                  kv_matches(&progressive_reject_fixture, &initial),
              "progressive seed miss performed target work or changed state");
    }

    {
        int32_t invalid_candidates[CANDIDATES];
        memcpy(invalid_candidates, candidates, sizeof invalid_candidates);
        invalid_candidates[CANDIDATES - 1u] = VOCABULARY;
        CHECK(prepare_execution(&progressive_reject_fixture,
                  package, mode, source) == 0 &&
                  compile_execution(&progressive_reject_fixture) == 0,
              "progressive invalid-input fixture compilation failed");
        memset(&progressive_result, 0, sizeof progressive_result);
        CHECK(salt_text_verify_progressive_execute(
                  &progressive_reject_fixture.program,
                  &progressive_reject_fixture.executor,
                  300u, source, invalid_candidates, CANDIDATES, parent_logits,
                  &progressive_plan, &progressive_result) != 0 &&
                  progressive_result.target.status == SALT_TEXT_VERIFY_INVALID &&
                  progressive_result.tile_count == 0u &&
                  progressive_reject_fixture.state.position == source &&
                  kv_matches(&progressive_reject_fixture, &initial),
              "progressive invalid suffix committed an earlier tile");
    }

    {
        int32_t rejected_candidates[CANDIDATES];
        int32_t repaired_candidates[2];
        const float *repair_parent;
        uint64_t repair_generation;
        memcpy(rejected_candidates, candidates, sizeof rejected_candidates);
        rejected_candidates[2] = (rejected_candidates[2] + 1) % VOCABULARY;
        CHECK(prepare_execution(&progressive_reject_fixture,
                  package, mode, source) == 0 &&
                  compile_execution(&progressive_reject_fixture) == 0,
              "progressive partial-prefix fixture compilation failed");
        memset(&progressive_result, 0, sizeof progressive_result);
        CHECK(salt_text_verify_progressive_execute(
                  &progressive_reject_fixture.program,
                  &progressive_reject_fixture.executor,
                  400u, source, rejected_candidates, CANDIDATES, parent_logits,
                  &progressive_plan, &progressive_result) == 0 &&
                  progressive_result.target.accepted_count == 2u &&
                  progressive_result.target.committed_count == 2u &&
                  progressive_result.target.produced_count == 3u &&
                  progressive_result.target.pending_token_id == candidates[2] &&
                  progressive_result.target.result_position == source + 2u &&
                  progressive_result.tile_count == 2u &&
                  progressive_result.submitted_tile_count == 2u &&
                  progressive_result.target.backend.engine_submissions == 2u &&
                  progressive_result.target.backend.completion_fences == 2u &&
                  kv_matches(&progressive_reject_fixture, &serial_kv[2]),
              "progressive partial-prefix rejection changed canonical state");
        repair_parent = progressive_result.target.pending_logits;
        repair_generation = progressive_result.target.transition_generation + 1u;
        repaired_candidates[0] = candidates[2];
        repaired_candidates[1] = candidates[3];
        memset(&progressive_result, 0, sizeof progressive_result);
        CHECK(salt_text_verify_progressive_execute(
                  &progressive_reject_fixture.program,
                  &progressive_reject_fixture.executor,
                  repair_generation, source + 2u,
                  repaired_candidates, 2u, repair_parent,
                  &progressive_plan, &progressive_result) == 0 &&
                  progressive_result.target.accepted_count == 2u &&
                  progressive_result.target.committed_count == 2u &&
                  progressive_result.target.produced_count == 3u &&
                  progressive_result.target.pending_token_id ==
                      serial_result.pending_token_id &&
                  progressive_result.target.result_position ==
                      source + CANDIDATES &&
                  progressive_result.tile_count == 2u &&
                  progressive_result.submitted_tile_count == 2u &&
                  kv_matches(&progressive_reject_fixture,
                      &serial_kv[CANDIDATES]),
              "progressive correction restarted or changed the accepted prefix");
    }
    {
        float continuation_parent[VOCABULARY];
        int32_t pending = block_result.pending_token_id;
        SaltTextTargetBlock block;
        SaltTextVerifyResult continuation_a, continuation_b;
        KvSnapshot continuation_kv;
        memcpy(continuation_parent,
               fixture_logits(&block_fixture) +
                   (size_t)(CANDIDATES - 1u) * VOCABULARY,
               sizeof continuation_parent);
        memset(&block, 0, sizeof block);
        block.transition_generation = 101u;
        block.source_position = source + CANDIDATES;
        block.candidate_token_ids = &pending;
        block.candidate_count = 1;
        block.parent_logits = continuation_parent;
        CHECK(salt_text_verify_execute(&block_fixture.program,
                  &block_fixture.executor, &block, &continuation_a) == 0 &&
              salt_text_verify_execute(&deterministic_fixture.program,
                  &deterministic_fixture.executor, &block, &continuation_b) == 0,
              "pending-token continuation replay failed");
        snapshot_kv(&block_fixture, &continuation_kv);
        CHECK(continuation_a.accepted_count == 1u &&
                  continuation_a.pending_token_id ==
                      continuation_b.pending_token_id &&
                  continuation_a.result_position == continuation_b.result_position &&
                  continuation_a.transition_generation ==
                      continuation_b.transition_generation &&
                  block_fixture.state.position == deterministic_fixture.state.position &&
                  block_fixture.state.transition_generation ==
                      deterministic_fixture.state.transition_generation &&
                  memcmp(fixture_logits(&block_fixture),
                         fixture_logits(&deterministic_fixture),
                         VOCABULARY * sizeof(float)) == 0 &&
                  kv_matches(&deterministic_fixture, &continuation_kv),
              "pending continuation token/logits/complete committed state drifted");
    }

    for (uint32_t reject = 0; reject < CANDIDATES; reject++) {
        int32_t rejected_candidates[CANDIDATES];
        SaltTextTargetBlock block;
        DescriptorSnapshot snapshot;
        uint64_t expected_scrub =
            (uint64_t)LAYERS * (CANDIDATES - reject) *
            2u * KV_WIDTH * sizeof(float);
        memcpy(rejected_candidates, candidates, sizeof rejected_candidates);
        rejected_candidates[reject] =
            (rejected_candidates[reject] + 1) % VOCABULARY;
        CHECK(prepare_execution(&reject_fixture, package, mode, source) == 0 &&
                  compile_execution(&reject_fixture) == 0,
              "rejection fixture compilation failed");
        snapshot_descriptors(&reject_fixture, package, &snapshot);
        memset(&block, 0, sizeof block);
        block.transition_generation = 200u + reject;
        block.source_position = source;
        block.candidate_token_ids = rejected_candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&reject_fixture.program,
                  &reject_fixture.executor, &block, &block_result) == 0 &&
                  block_result.accepted_count == reject &&
                  block_result.produced_count == reject + 1u &&
                  block_result.committed_count == reject &&
                  block_result.pending_token_id == candidates[reject] &&
                  block_result.result_position == source + reject,
              "accepted-prefix rollback/commit result changed");
        CHECK(kv_matches(&reject_fixture, &serial_kv[reject]),
              "rejection KV bytes differ from serial accepted prefix");
        CHECK(tentative_suffix_is_zero(&reject_fixture, reject) &&
                  block_result.backend.tentative_scrub_bytes ==
                      (reject == 0u ? 0u : expected_scrub),
              "rejected tentative suffix was not completely scrubbed");
        CHECK(block_result.backend.engine_submissions ==
                      (reject == 0u ? 0u : 1u) &&
                  block_result.backend.completion_fences ==
                      (reject == 0u ? 0u : 1u) &&
                  block_result.backend.intermediate_host_publications == 0u,
              "rejection used unexpected work or an intermediate publication");
        CHECK(descriptors_equal(&reject_fixture, package, &snapshot),
              "rejection mutated immutable descriptors");
        if (reject == 0) {
            CHECK(salt_text_verify_execute(&reject_fixture.program,
                      &reject_fixture.executor, &block, &block_result) != 0 &&
                      block_result.status == SALT_TEXT_VERIFY_INVALID,
                  "duplicate transition generation was accepted");
        }
    }

    CHECK(prepare_execution(&failure_fixture, package, mode, source) == 0 &&
              compile_execution(&failure_fixture) == 0,
          "failure fixture compilation failed");
    snapshot_descriptors(&failure_fixture, package, &block_descriptors);
    package->fail_handle = failure_fixture.layers[1].q.stable_handle;
    cleanup_test_context = &failure_fixture.cpu;
    cleanup_test_handle = failure_fixture.layers[0].q.stable_handle;
    cleanup_test_opens = cleanup_test_finishes = cleanup_test_fail = 0;
    {
        SaltTextTargetBlock block;
        memset(&block, 0, sizeof block);
        block.transition_generation = 400u;
        block.source_position = source;
        block.candidate_token_ids = candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&failure_fixture.program,
                  &failure_fixture.executor, &block, &block_result) != 0 &&
                  block_result.status == SALT_TEXT_VERIFY_EXECUTOR_FAILURE &&
                  failure_fixture.state.position == source &&
                  failure_fixture.state.transition_generation == 400u,
              "post-tentative tensor failure did not preserve parent state");
    }
    cleanup_test_context = NULL;
    cleanup_test_handle = 0;
    CHECK(cleanup_test_opens == 1 && cleanup_test_finishes == 1 &&
              !failure_fixture.cpu.graph_state.node_ops &&
              !failure_fixture.cpu.graph_state.node_seat,
          "intervening failure did not finish retained resources exactly once");
    package->fail_handle = 0;
    CHECK(kv_matches(&failure_fixture, &initial) &&
              canonical_is_zero(&failure_fixture) &&
              block_result.backend.tentative_scrub_bytes ==
                  failure_fixture.program.tentative_kv_bytes,
          "post-tentative failure did not scrub the complete fixed arena");
    CHECK(descriptors_equal(&failure_fixture, package, &block_descriptors),
          "failure path mutated immutable descriptors");

    {
        uint64_t scrubbed = 0;
        failure_fixture.cpu.graph_state.node_ops = &cleanup_test_ops;
        failure_fixture.cpu.graph_state.node_seat = &cleanup_test_finishes;
        CHECK(failure_fixture.executor.ops->scrub(&failure_fixture.cpu,
                  &failure_fixture.program, 400u, &scrubbed) == 0 &&
                  cleanup_test_finishes == 2 &&
                  !failure_fixture.cpu.graph_state.node_ops,
              "explicit abort did not finish the retained resource seat");
        failure_fixture.cpu.graph_state.node_ops = &cleanup_test_ops;
        failure_fixture.cpu.graph_state.node_seat = &cleanup_test_finishes;
        cleanup_test_fail = 1;
        CHECK(failure_fixture.executor.ops->scrub(&failure_fixture.cpu,
                  &failure_fixture.program, 400u, &scrubbed) != 0 &&
                  cleanup_test_finishes == 3 &&
                  !failure_fixture.cpu.graph_state.node_ops &&
                  !failure_fixture.cpu.graph_state.node_seat,
              "resource finish failure was hidden or retained for replay");
        cleanup_test_fail = 0;
    }

    CHECK(prepare_execution(&failure_fixture, package, mode, source) == 0 &&
              compile_execution(&failure_fixture) == 0,
          "pre-mutation failure fixture compilation failed");
    package->fail_handle = failure_fixture.model.embedding.stable_handle;
    {
        SaltTextTargetBlock block;
        memset(&block, 0, sizeof block);
        block.transition_generation = 500u;
        block.source_position = source;
        block.candidate_token_ids = candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&failure_fixture.program,
                  &failure_fixture.executor, &block, &block_result) != 0 &&
                  block_result.status == SALT_TEXT_VERIFY_EXECUTOR_FAILURE &&
                  kv_matches(&failure_fixture, &initial) &&
                  canonical_is_zero(&failure_fixture),
              "pre-mutation tensor failure changed target state");
    }
    package->fail_handle = 0;

    CHECK(prepare_execution(&failure_fixture, package, mode, source) == 0 &&
              compile_execution(&failure_fixture) == 0,
          "validation fixture compilation failed");
    {
        SaltTextTargetBlock block;
        SaltTextVerifyResult invalid_result;
        int32_t invalid_candidates[CANDIDATES];
        uint32_t operation_calls = package->operation_calls;
        memcpy(invalid_candidates, candidates, sizeof invalid_candidates);
        invalid_candidates[1] = VOCABULARY;
        memset(&block, 0, sizeof block);
        block.transition_generation = 600u;
        block.source_position = source;
        block.candidate_token_ids = invalid_candidates;
        block.candidate_count = CANDIDATES;
        block.parent_logits = parent_logits;
        CHECK(salt_text_verify_execute(&failure_fixture.program,
                  &failure_fixture.executor, &block, &invalid_result) != 0 &&
                  invalid_result.status == SALT_TEXT_VERIFY_INVALID &&
                  failure_fixture.state.position == source &&
                  failure_fixture.state.transition_generation == 0u &&
                  package->operation_calls == operation_calls,
              "invalid token reached the executor");
        block.candidate_token_ids = candidates;
        block.source_position = source + 1u;
        CHECK(salt_text_verify_execute(&failure_fixture.program,
                  &failure_fixture.executor, &block, &invalid_result) != 0 &&
                  invalid_result.status == SALT_TEXT_VERIFY_INVALID &&
                  package->operation_calls == operation_calls,
              "stale source position reached the executor");
        block.source_position = source;
        block.candidate_count = 0;
        CHECK(salt_text_verify_execute(&failure_fixture.program,
                  &failure_fixture.executor, &block, &invalid_result) != 0 &&
                  invalid_result.status == SALT_TEXT_VERIFY_INVALID &&
                  package->operation_calls == operation_calls,
              "empty target block reached the executor");
    }
    printf("descriptor CPU target block (%s): PASS\n", label);
    return 0;
}

static int area_scan_planner_contract(void) {
    SaltAreaScanRequest request;
    SaltAreaScanPlan plan;
    SaltAreaTask tasks[4], task;
    SaltAreaFrontier frontier;
    memset(&request, 0, sizeof request);
    request.requested_rows = 1u;
    request.admitted_rows = 1u;
    request.resident_workers = 20u;
    request.active_worker_limit = 16u;
    request.narrow_workers = 8u;
    request.wide_allowed = 1u;
    request.output_row_tiles = 704u;
    request.candidate_output_tiles = 704u;
    CHECK(salt_area_scan_plan(&request, &plan) == 0 &&
              plan.mode == SALT_AREA_SCAN_M1 &&
              plan.rows_per_wave == 1u && plan.active_workers == 8u &&
              plan.output_row_owned == 1u &&
              plan.candidate_row_split == 0u &&
              plan.matrix_parallel == 0u &&
              plan.candidate_output_tiles == 704u,
          "M1 area-scan plan changed");
    request.requested_rows = 4u;
    request.admitted_rows = 4u;
    request.output_row_tiles = 2816u;
    request.candidate_output_tiles = 11264u;
    CHECK(salt_area_scan_plan(&request, &plan) == 0 &&
              plan.mode == SALT_AREA_SCAN_MN &&
              plan.rows_per_wave == 4u && plan.active_workers == 16u &&
              plan.output_row_owned == 1u &&
              plan.candidate_row_split == 0u &&
              plan.matrix_parallel == 1u &&
              plan.candidate_output_tiles == 11264u,
          "MN area-scan plan changed");
    request.admitted_rows = 2u;
    request.candidate_output_tiles = 5632u;
    CHECK(salt_area_scan_plan(&request, &plan) == 0 &&
              plan.mode == SALT_AREA_SCAN_MK &&
              plan.rows_per_wave == 2u && plan.active_workers == 16u &&
              plan.output_row_owned == 1u &&
              plan.candidate_row_split == 0u &&
              plan.matrix_parallel == 1u &&
              plan.candidate_output_tiles == 5632u,
          "MK area-scan plan changed");
    request.output_row_tiles = 4u;
    request.candidate_output_tiles = 8u;
    CHECK(salt_area_scan_plan(&request, &plan) == 0 &&
              plan.active_workers == 8u &&
              plan.output_row_owned == 0u &&
              plan.candidate_row_split == 1u &&
              plan.candidate_output_tiles == 8u,
          "narrow output rows did not admit bounded candidate splitting");
    request.output_row_tiles = 2816u;
    request.candidate_output_tiles = 11264u;
    request.admitted_rows = 4u;
    request.wide_allowed = 0u;
    CHECK(salt_area_scan_plan(&request, &plan) == 0 &&
              plan.mode == SALT_AREA_SCAN_M1 &&
              plan.rows_per_wave == 1u && plan.active_workers == 8u,
          "guarded M1 fallback changed");
    request.active_worker_limit = 21u;
    CHECK(salt_area_scan_plan(&request, &plan) != 0,
          "area-scan worker guard admitted over-capacity plan");
    memset(tasks, 0, sizeof tasks);
    memset(&frontier, 0, sizeof frontier);
    CHECK(salt_area_frontier_bind(&frontier, tasks, 4u) == 0 &&
              salt_area_frontier_reset(&frontier, 42u) == 0,
          "area frontier startup/reset failed");
    memset(&task, 0, sizeof task);
    task.generation = 42u;
    task.node_id = 1u;
    task.candidate_count = 2u;
    task.output_count = 16u;
    task.state = SALT_AREA_TASK_QUEUED;
    CHECK(salt_area_frontier_append(&frontier, &task) == 0,
          "area frontier first append failed");
    task.node_id = 2u;
    task.candidate_first = 2u;
    CHECK(salt_area_frontier_append(&frontier, &task) == 0 &&
              salt_area_frontier_cancel_except(&frontier, 42u, 2u) == 0 &&
              frontier.tasks[0].state == SALT_AREA_TASK_CANCELLED &&
              frontier.tasks[1].state == SALT_AREA_TASK_QUEUED,
          "area frontier target cancellation changed");
    puts("area-scan dispatcher plans: PASS");
    return 0;
}

typedef struct WfqTestContext {
    SaltAreaWfqOutcome outcomes[8][4];
    int32_t tokens[8][4];
    uint32_t slots[8][4];
    uint32_t calls[8][4];
    uint32_t workers[8][4];
    uint32_t phase_workers[4];
    uint32_t phases;
    uint32_t cancel_on_call[8][4];
    int reverse;
} WfqTestContext;

static int wfq_test_pool_run(
        void *opaque, int active_workers,
        void (*worker)(int worker, void *task), void *task) {
    WfqTestContext *context = (WfqTestContext *)opaque;
    if (!context || !worker || !task || active_workers < 1 ||
        active_workers > 8 || context->phases >= 4u)
        return -1;
    context->phase_workers[context->phases++] = (uint32_t)active_workers;
    if (context->reverse)
        for (int index = active_workers; index > 0; index--)
            worker(index - 1, task);
    else
        for (int index = 0; index < active_workers; index++)
            worker(index, task);
    return 0;
}

static int wfq_test_item_execute(
        void *opaque, const SaltAreaWfqItem *item,
        SaltAreaWfqItemResult *result) {
    WfqTestContext *context = (WfqTestContext *)opaque;
    if (!context || !item || !result || item->branch_index >= 8u ||
        item->queue_index >= 4u)
        return -1;
    context->calls[item->branch_index][item->queue_index]++;
    context->workers[item->branch_index][item->queue_index] =
        item->owner_worker;
    if (context->cancel_on_call[item->branch_index][item->queue_index])
        item->task->flags |= SALT_AREA_TASK_CANCEL_REQUESTED;
    result->outcome =
        context->outcomes[item->branch_index][item->queue_index];
    result->target_token =
        context->tokens[item->branch_index][item->queue_index];
    result->target_state_slot =
        context->slots[item->branch_index][item->queue_index];
    return 0;
}

static void wfq_test_continue(WfqTestContext *context) {
    memset(context, 0, sizeof *context);
    for (uint32_t branch = 0; branch < 8u; branch++)
        for (uint32_t queue = 0; queue < 4u; queue++)
            context->outcomes[branch][queue] = SALT_AREA_WFQ_CONTINUE;
}

static void wfq_test_tasks(
        SaltAreaTask *tasks, uint32_t count, uint64_t generation) {
    memset(tasks, 0, (size_t)count * sizeof *tasks);
    for (uint32_t index = 0; index < count; index++)
        tasks[index] = (SaltAreaTask) {
            generation, 900u + index, UINT32_MAX, 1u, -1,
            index, index, 1u, index, 1u, 0u, SALT_AREA_TASK_QUEUED,
        };
}

static int wfq_scheduler_contract(void) {
    SaltAreaWfqLedger first_ledger;
    uint32_t first_retired = 0;
    memset(&first_ledger, 0, sizeof first_ledger);
    for (int reverse = 0; reverse < 2; reverse++) {
        SaltAreaWfqRuntime runtime;
        SaltAreaWfqBranch branches[8];
        SaltAreaWfqItem items[32];
        SaltAreaWfqItemResult item_results[8];
        SaltAreaTask tasks[32];
        SaltAreaWfqPlan plan = { 8u, 8u, 4u };
        SaltAreaWfqResult result;
        WfqTestContext context;
        uint32_t seeds[8], next_seeds[8];
        int execute_rc;
        wfq_test_continue(&context);
        context.reverse = reverse;
        context.outcomes[1][0] = SALT_AREA_WFQ_BRANCH_FAIL;
        context.outcomes[5][1] = SALT_AREA_WFQ_TARGET_WIN;
        context.tokens[5][1] = 77;
        context.slots[5][1] = 9u;
        for (uint32_t index = 0; index < 8u; index++) {
            seeds[index] = index + 10u;
            next_seeds[index] = index + 100u;
        }
        wfq_test_tasks(tasks, 32u, 10u);
        memset(&runtime, 0, sizeof runtime);
        memset(&result, 0, sizeof result);
        CHECK(salt_area_wfq_bind(&runtime, branches, 8u,
                  items, 32u, item_results, 8u, tasks, 32u) == 0 &&
              salt_area_wfq_start(
                  &runtime, &plan, 10u, seeds, 8u) == 0,
              "W8-F8-Q4 setup failed");
        execute_rc = salt_area_wfq_execute(&runtime,
            wfq_test_pool_run, &context,
            wfq_test_item_execute, &context, &result);
        CHECK(execute_rc == 0 && context.phases == 2u &&
                  context.phase_workers[0] == 8u &&
                  context.phase_workers[1] == 7u,
              "W8-F8-Q4 execution failed");
        CHECK(result.status == SALT_AREA_WFQ_WIN &&
                  result.ledger.generation == 11u &&
                  result.ledger.winning_branch == 5u &&
                  result.ledger.winning_queue == 1u &&
                  result.ledger.target_token == 77 &&
                  result.ledger.target_state_slot == 9u &&
                  result.executed_items == 15u &&
                  result.retired_items == 18u && !runtime.ready,
              "W8-F8-Q4 ledger/retirement result changed");
        for (uint32_t branch = 0; branch < 8u; branch++) {
            uint32_t expected_worker = branch > 1u ? branch - 1u : branch;
            CHECK(branches[branch].owner_worker == expected_worker,
                  "W8-F8-Q4 returned seat was not reassigned");
            CHECK(context.workers[branch][0] == branch,
                  "W8-F8-Q4 first phase worker identity changed");
            if (branch != 1u)
                CHECK(context.workers[branch][1] == expected_worker,
                      "W8-F8-Q4 second phase did not execute reassigned work");
            for (uint32_t queue = 0; queue < 4u; queue++) {
                SaltAreaWfqItemState state =
                    items[branch * 4u + queue].state;
                if (branch == 1u)
                    CHECK(state == SALT_AREA_WFQ_ITEM_RETIRED,
                          "failed W8-F8-Q4 branch was not fully retired");
                else if (queue < 2u)
                    CHECK(state == SALT_AREA_WFQ_ITEM_DONE,
                          "completed W8-F8-Q4 prefix changed");
                else
                    CHECK(state == SALT_AREA_WFQ_ITEM_RETIRED,
                          "old-generation W8-F8-Q4 suffix survived win");
            }
        }
        if (!reverse) {
            first_ledger = result.ledger;
            first_retired = result.retired_items;
        } else {
            CHECK(memcmp(&first_ledger, &result.ledger,
                         sizeof first_ledger) == 0 &&
                      first_retired == result.retired_items,
                  "worker completion order changed W8-F8-Q4 winner");
        }
        wfq_test_tasks(tasks, 32u, result.ledger.generation);
        CHECK(salt_area_wfq_start(&runtime, &plan,
                  result.ledger.generation, next_seeds, 8u) == 0,
              "W8-F8-Q4 ledger refanout failed");
        for (uint32_t branch = 0; branch < 8u; branch++)
            CHECK(branches[branch].active &&
                      branches[branch].owner_worker == branch &&
                      branches[branch].next_queue == 0u &&
                      items[branch * 4u].generation == 11u &&
                      items[branch * 4u].state == SALT_AREA_WFQ_ITEM_QUEUED,
                  "W8-F8-Q4 refanout did not restart clean queues");
    }
    {
        SaltAreaWfqRuntime runtime;
        SaltAreaWfqBranch branches[8];
        SaltAreaWfqItem items[32];
        SaltAreaWfqItemResult item_results[8];
        SaltAreaTask tasks[32];
        SaltAreaWfqPlan plan = { 8u, 8u, 4u };
        SaltAreaWfqResult result;
        WfqTestContext context;
        uint32_t seeds[8];
        wfq_test_continue(&context);
        context.outcomes[2][0] = SALT_AREA_WFQ_RESOURCE_WAIT;
        context.cancel_on_call[6][0] = 1u;
        for (uint32_t index = 0; index < 8u; index++) seeds[index] = index;
        wfq_test_tasks(tasks, 32u, 20u);
        memset(&runtime, 0, sizeof runtime);
        CHECK(salt_area_wfq_bind(&runtime, branches, 8u,
                  items, 32u, item_results, 8u, tasks, 32u) == 0 &&
              salt_area_wfq_start(
                  &runtime, &plan, 20u, seeds, 8u) == 0 &&
              salt_area_wfq_execute(&runtime,
                  wfq_test_pool_run, &context,
                  wfq_test_item_execute, &context, &result) == 0 &&
              result.status == SALT_AREA_WFQ_NEED_RESOURCE &&
              branches[2].active && branches[2].next_queue == 0u &&
              items[8u].state == SALT_AREA_WFQ_ITEM_WAITING &&
              tasks[8u].state == SALT_AREA_TASK_QUEUED &&
              !branches[6].active &&
              items[24u].state == SALT_AREA_WFQ_ITEM_RETIRED &&
              tasks[24u].state == SALT_AREA_TASK_CANCELLED &&
              tasks[25u].state == SALT_AREA_TASK_CANCELLED &&
              branches[0].next_queue == 1u,
              "resource wait incorrectly retired a W8-F8-Q4 branch");
        context.outcomes[2][0] = SALT_AREA_WFQ_CONTINUE;
        context.outcomes[4][1] = SALT_AREA_WFQ_TARGET_WIN;
        context.tokens[4][1] = 88;
        context.slots[4][1] = 12u;
        memset(&result, 0, sizeof result);
        CHECK(salt_area_wfq_execute(&runtime,
                  wfq_test_pool_run, &context,
                  wfq_test_item_execute, &context, &result) == 0 &&
              result.status == SALT_AREA_WFQ_WIN &&
              result.ledger.winning_branch == 4u &&
              result.ledger.winning_queue == 1u &&
              result.ledger.target_token == 88,
              "resource-ready W8-F8-Q4 refire failed");
    }
    puts("W8-F8-Q4 ledger-preemptive scheduler: PASS");
    return 0;
}

static int nfq_matrix_contract(void) {
    SaltAreaNfqMatrix matrix;
    SaltAreaNfqRoute routes[2];
    SaltAreaNfqItem items[32];
    SaltAreaNfqPlan plan = { 8u, 4u, 2u, 4u };
    uint32_t route_ids[2] = { 100u, 200u };
    uint32_t next_route_ids[2] = { 400u, 500u };
    uint32_t retired = 0;
    memset(&matrix, 0, sizeof matrix);
    CHECK(salt_area_nfq_bind(
              &matrix, routes, 2u, items, 32u) == 0 &&
          salt_area_nfq_start(
              &matrix, &plan, 50u, 49u, 10u,
              route_ids, 2u) == 0 &&
          matrix.item_count == 32u && matrix.ready && !matrix.resolved &&
          routes[0].worker_mask == 0u &&
          routes[1].worker_mask == 0u,
          "N4-F2-Q4 matrix startup failed");
    for (uint32_t route = 0; route < 2u; route++)
        for (uint32_t sequence = 0; sequence < 4u; sequence++)
            for (uint32_t queue = 0; queue < 4u; queue++) {
                uint32_t index = ((route * 4u + sequence) * 4u) + queue;
                CHECK(items[index].route_slot == route &&
                          items[index].route_id == route_ids[route] &&
                          items[index].route_version == 1u &&
                          items[index].sequence_index == sequence &&
                          items[index].queue_index == queue &&
                          items[index].owner_worker == UINT32_MAX &&
                          items[index].state == (queue == 0u
                              ? SALT_AREA_WFQ_ITEM_QUEUED
                              : SALT_AREA_WFQ_ITEM_WAITING),
                      "N4-F2-Q4 matrix ownership changed");
            }
    items[0].state = SALT_AREA_WFQ_ITEM_RUNNING;
    CHECK(salt_area_nfq_publishable(&matrix, 0u, 50u, 49u) == 1 &&
              salt_area_nfq_publishable(&matrix, 0u, 51u, 49u) == 0,
          "N4-F2-Q4 generation publication guard changed");
    CHECK(salt_area_nfq_retire_route(
              &matrix, 50u, 1u, &retired) == 0 && retired == 16u &&
              !routes[1].active,
          "N4-F2-Q4 route retirement changed");
    for (uint32_t index = 16u; index < 32u; index++)
        CHECK(items[index].owner_worker == UINT32_MAX,
              "retired N4-F2-Q4 route kept a worker seat");
    CHECK(routes[0].worker_mask == 0u && routes[1].worker_mask == 0u,
          "retired N4-F2-Q4 retained an active worker claim");
    CHECK(salt_area_nfq_reschedule_route(
              &matrix, 50u, 49u, 1u, 300u) == 0 &&
          routes[1].active && routes[1].route_id == 300u &&
          routes[1].version == 2u &&
          routes[0].worker_mask == 0u &&
          routes[1].worker_mask == 0u,
          "N4-F2-Q4 same-epoch hot reschedule failed");
    for (uint32_t index = 16u; index < 32u; index++)
        CHECK(items[index].route_id == 300u &&
                  items[index].route_version == 2u &&
                  items[index].state == (items[index].queue_index == 0u
                      ? SALT_AREA_WFQ_ITEM_QUEUED
                      : SALT_AREA_WFQ_ITEM_WAITING),
              "N4-F2-Q4 hot route did not replace retired items");
    items[16].state = SALT_AREA_WFQ_ITEM_DONE;
    CHECK(salt_area_nfq_publishable(&matrix, 16u, 50u, 49u) == 1,
          "N4-F2-Q4 rescheduled item was not publishable");
    CHECK(salt_area_nfq_resolve(
              &matrix, 50u, 49u, 1u, 0u, 0u, &retired) == 0 &&
          retired == 31u && matrix.resolved && !matrix.ready &&
          items[16].state == SALT_AREA_WFQ_ITEM_DONE &&
          items[16].owner_worker == UINT32_MAX &&
          routes[0].worker_mask == 0u && routes[1].worker_mask == 0u &&
          salt_area_nfq_publishable(&matrix, 16u, 50u, 49u) == 0,
          "N4-F2-Q4 resolution did not stale the old epoch");
    for (uint32_t index = 0; index < 32u; index++)
        if (index != 16u)
            CHECK(items[index].owner_worker == UINT32_MAX,
                  "resolved N4-F2-Q4 loser kept a worker seat");
    CHECK(salt_area_nfq_reschedule_route(
              &matrix, 50u, 49u, 0u, 600u) != 0,
          "resolved N4-F2-Q4 epoch admitted immediate refanout");
    CHECK(salt_area_nfq_start(
              &matrix, &plan, 51u, 50u, 11u,
              next_route_ids, 2u) == 0 && matrix.ready && !matrix.resolved &&
          matrix.parent_position == 11u,
          "explicit next-token N4-F2-Q4 start failed");
    puts("N4-F2-Q4 fixed matrix and hot reschedule: PASS");
    return 0;
}

static int nfq_ready_shape_contract(
        uint32_t sequence_tiles, uint32_t route_count,
        uint32_t queue_length) {
    SaltAreaNfqMatrix matrix;
    SaltAreaNfqRoute routes[8];
    SaltAreaNfqItem items[32];
    SaltAreaNfqPlan plan = {
        8u, sequence_tiles, route_count, queue_length,
    };
    uint32_t route_ids[8], ready[8], original[8];
    uint32_t ready_count = 0, retired = 0;
    uint64_t epoch = 100u + sequence_tiles;
    if (sequence_tiles * route_count != 8u ||
        sequence_tiles * route_count * queue_length > 32u)
        return 1;
    for (uint32_t route = 0; route < route_count; route++)
        route_ids[route] = 1000u + route;
    memset(&matrix, 0, sizeof matrix);
    CHECK(salt_area_nfq_bind(&matrix, routes, 8u, items, 32u) == 0 &&
          salt_area_nfq_start(&matrix, &plan, epoch, epoch - 1u, 20u,
                              route_ids, route_count) == 0 &&
          salt_area_nfq_take_ready(&matrix, epoch, epoch - 1u,
                                   ready, 8u, &ready_count) == 0 &&
          ready_count == 8u,
          "NFQ initial ready width changed");
    memcpy(original, ready, sizeof original);
    for (uint32_t slot = 0; slot < ready_count; slot++)
        CHECK(items[ready[slot]].queue_index == 0u &&
              items[ready[slot]].state == SALT_AREA_WFQ_ITEM_RUNNING,
              "NFQ initial item ownership changed");
    CHECK(salt_area_nfq_complete(&matrix, epoch, epoch - 1u, original[0],
              SALT_AREA_WFQ_RESOURCE_WAIT, &retired) == 0 && retired == 0u &&
          salt_area_nfq_refire(
              &matrix, epoch, epoch - 1u, original[0]) == 0 &&
          salt_area_nfq_take_ready(&matrix, epoch, epoch - 1u,
              ready, 1u, &ready_count) == 0 && ready_count == 1u &&
          ready[0] == original[0] &&
          salt_area_nfq_complete(&matrix, epoch, epoch - 1u, ready[0],
              SALT_AREA_WFQ_CONTINUE, &retired) == 0,
          "NFQ resource refire changed");
    for (uint32_t slot = 1; slot < 8u; slot++)
        CHECK(salt_area_nfq_complete(&matrix, epoch, epoch - 1u, original[slot],
                  SALT_AREA_WFQ_CONTINUE, &retired) == 0,
              "NFQ Q0 completion changed");
    for (uint32_t queue = 1u; queue < queue_length; queue++) {
        CHECK(salt_area_nfq_take_ready(&matrix, epoch, epoch - 1u,
                  ready, 8u, &ready_count) == 0 && ready_count == 8u,
              "NFQ refill width changed");
        for (uint32_t slot = 0; slot < ready_count; slot++)
            CHECK(items[ready[slot]].queue_index == queue &&
                  salt_area_nfq_complete(&matrix, epoch, epoch - 1u,
                      ready[slot], SALT_AREA_WFQ_CONTINUE, &retired) == 0,
                  "NFQ refill completion changed");
    }
    CHECK(salt_area_nfq_take_ready(&matrix, epoch, epoch - 1u,
              ready, 8u, &ready_count) == 0 && ready_count == 0u &&
          salt_area_nfq_resolve(&matrix, epoch, epoch - 1u,
              0u, 0u, queue_length - 1u, &retired) == 0 &&
          retired == sequence_tiles * route_count * queue_length - 1u,
          "NFQ terminal resolution changed");
    return 0;
}

static int nfq_worker_multiple_contract(void) {
    SaltAreaNfqMatrix matrix;
    SaltAreaNfqRoute routes[1];
    SaltAreaNfqItem items[32];
    SaltAreaNfqPlan plan = { 8u, 16u, 1u, 2u };
    uint32_t route_id = 77u, ready[8], ready_count = 0, retired = 0;
    uint32_t ownership[2][8] = {{0}};
    memset(&matrix, 0, sizeof matrix);
    CHECK(salt_area_nfq_bind(&matrix, routes, 1u, items, 32u) == 0 &&
          salt_area_nfq_start(&matrix, &plan, 200u, 199u, 30u,
                              &route_id, 1u) == 0 &&
          routes[0].worker_mask == 0u,
          "NFQ B=2W startup failed");
    for (uint32_t queue = 0; queue < 2u; queue++)
        for (uint32_t wave = 0; wave < 2u; wave++) {
            CHECK(salt_area_nfq_take_ready(&matrix, 200u, 199u,
                      ready, 8u, &ready_count) == 0 && ready_count == 8u,
                  "NFQ B=2W ready wave changed");
            for (uint32_t slot = 0; slot < ready_count; slot++) {
                SaltAreaNfqItem *item = &items[ready[slot]];
                CHECK(item->queue_index == queue &&
                      item->owner_worker < 8u,
                      "NFQ B=2W queue priority changed");
                ownership[queue][item->owner_worker]++;
                CHECK(salt_area_nfq_complete(&matrix, 200u, 199u,
                          ready[slot], SALT_AREA_WFQ_CONTINUE,
                          &retired) == 0,
                      "NFQ B=2W completion changed");
            }
        }
    for (uint32_t queue = 0; queue < 2u; queue++)
        for (uint32_t worker = 0; worker < 8u; worker++)
            CHECK(ownership[queue][worker] == 2u,
                  "NFQ B=2W worker ownership is not uniform");
    CHECK(salt_area_nfq_resolve(&matrix, 200u, 199u,
              0u, 0u, 1u, &retired) == 0 && retired == 31u,
          "NFQ B=2W resolve changed");
    return 0;
}

static int nfq_cpu_single_owner_contract(void) {
    SaltAreaNfqMatrix matrix;
    SaltAreaNfqRoute routes[1];
    SaltAreaNfqItem items[4];
    SaltAreaNfqPlan plan = { 1u, 1u, 1u, 4u };
    uint32_t route_id = 9u, ready = UINT32_MAX;
    uint32_t ready_count = 0, retired = 0;
    memset(&matrix, 0, sizeof matrix);
    CHECK(salt_area_nfq_bind(&matrix, routes, 1u, items, 4u) == 0 &&
          salt_area_nfq_start(&matrix, &plan, 300u, 299u, 40u,
                              &route_id, 1u) == 0 &&
          routes[0].worker_mask == 0u && matrix.item_count == 4u,
          "CPU N1-F1-Q4 startup failed");
    CHECK(salt_area_nfq_take_ready(&matrix, 300u, 299u,
              &ready, 1u, &ready_count) == 0 && ready_count == 1u &&
          ready == 0u && items[ready].queue_index == 0u &&
          items[ready].owner_worker == 0u && routes[0].worker_mask == 1u,
          "CPU N1-F1-Q4 owner runway changed");
    items[ready].state = SALT_AREA_WFQ_ITEM_DONE;
    CHECK(salt_area_nfq_resolve(&matrix, 300u, 299u,
              0u, 0u, 0u, &retired) == 0 && retired == 3u &&
          items[0].state == SALT_AREA_WFQ_ITEM_DONE,
          "CPU N1-F1-Q4 resolution changed");
    return 0;
}

typedef struct TokenEpochFixture {
    SaltTextTokenEpochController *controller;
    const SaltTextTokenEpochRequest *request;
    int exact_hit;
    int nested;
    int nested_rc;
    uint32_t exact_calls;
    uint32_t cold_calls;
    uint32_t committed;
} TokenEpochFixture;

static int token_epoch_exact_lookup(
        void *opaque, uint64_t epoch_generation,
        uint64_t parent_generation, uint32_t parent_position,
        SaltTextVerifyResult *result) {
    TokenEpochFixture *fixture = (TokenEpochFixture *)opaque;
    SaltTextTokenEpochResult nested_result;
    (void)parent_generation;
    if (!fixture || !result) return -1;
    fixture->exact_calls++;
    if (fixture->nested) {
        fixture->nested = 0;
        memset(&nested_result, 0, sizeof nested_result);
        fixture->nested_rc = salt_text_token_epoch_execute(
            fixture->controller, fixture->request, &nested_result);
    }
    if (!fixture->exact_hit) return 0;
    memset(result, 0, sizeof *result);
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = fixture->committed;
    result->committed_count = fixture->committed;
    result->produced_count = fixture->committed + 1u;
    result->result_position = parent_position + fixture->committed;
    result->transition_generation = epoch_generation;
    return 1;
}

static int token_epoch_cold_search(
        void *opaque, const SaltTextTargetRouteBlock *route,
        SaltTextVerifyResult *result) {
    TokenEpochFixture *fixture = (TokenEpochFixture *)opaque;
    if (!fixture || !route || !result) return -1;
    fixture->cold_calls++;
    memset(result, 0, sizeof *result);
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = fixture->committed;
    result->committed_count = fixture->committed;
    result->produced_count = fixture->committed + 1u;
    result->result_position = route->source_position + fixture->committed;
    result->transition_generation = route->transition_generation;
    result->backend.engine_submissions = fixture->committed ? 1u : 0u;
    result->backend.completion_fences = fixture->committed ? 1u : 0u;
    return 0;
}

static int token_epoch_contract(void) {
    SaltTextTokenEpochController controller;
    SaltTextTokenEpochRequest request;
    SaltTextTokenEpochResult result;
    SaltTextTargetRouteBlock route;
    TokenEpochFixture fixture;
    int32_t route_tokens[2] = { 10, 11 };
    float parent_logits[2] = { 0.0f, 1.0f };
    memset(&controller, 0, sizeof controller);
    memset(&request, 0, sizeof request);
    memset(&route, 0, sizeof route);
    memset(&fixture, 0, sizeof fixture);
    route.transition_generation = 2u;
    route.source_position = 10u;
    route.route_token_ids = route_tokens;
    route.sequence_tiles = 1u;
    route.route_count = 2u;
    route.parent_logits = parent_logits;
    request.epoch_generation = 2u;
    request.parent_generation = 1u;
    request.parent_position = 10u;
    request.route = &route;
    request.exact_lookup_commit = token_epoch_exact_lookup;
    request.exact_context = &fixture;
    request.cold_search = token_epoch_cold_search;
    request.cold_context = &fixture;
    fixture.controller = &controller;
    fixture.request = &request;
    fixture.exact_hit = 1;
    fixture.nested = 1;
    fixture.committed = 1u;
    CHECK(salt_text_token_epoch_init(&controller) == 0 &&
          salt_text_token_epoch_execute(
              &controller, &request, &result) == 0 &&
          result.path == SALT_TEXT_TOKEN_PATH_EXACT_HIT &&
          result.target.committed_count == 1u &&
          fixture.exact_calls == 1u && fixture.cold_calls == 0u &&
          fixture.nested_rc == 1 &&
          controller.duplicate_admission_count == 1u &&
          controller.phase == SALT_TEXT_TOKEN_EPOCH_RESOLVED &&
          !controller.active,
          "exact-hit-first token epoch changed");

    route.transition_generation = 3u;
    route.source_position = 11u;
    request.epoch_generation = 3u;
    request.parent_generation = 2u;
    request.parent_position = 11u;
    fixture.exact_hit = 0;
    fixture.committed = 2u;
    memset(&result, 0, sizeof result);
    CHECK(salt_text_token_epoch_execute(
              &controller, &request, &result) == 0 &&
          result.path == SALT_TEXT_TOKEN_PATH_COLD_SEARCH &&
          result.target.committed_count == 2u &&
          result.target.result_position == 13u &&
          fixture.exact_calls == 2u && fixture.cold_calls == 1u &&
          controller.lookup_count == 2u &&
          controller.exact_hit_count == 1u &&
          controller.exact_miss_count == 1u &&
          controller.cold_search_count == 1u &&
          controller.phase == SALT_TEXT_TOKEN_EPOCH_RESOLVED,
          "true-miss cold-search token epoch changed");
    CHECK(salt_text_token_epoch_execute(
              &controller, &request, &result) != 0 &&
          fixture.exact_calls == 2u && fixture.cold_calls == 1u,
          "resolved token epoch refanned without a new boundary");
    puts("cache-first one-owner token epoch: PASS");
    return 0;
}

static int target_frontier_contract(void) {
    SaltTextTargetNode nodes[5] = {
        { 10, 100u, SALT_TEXT_TARGET_NO_PARENT, 0u, 0u, 0u },
        { 11, 101u, SALT_TEXT_TARGET_NO_PARENT, 0u, 1u, 0u },
        { 20, 200u, 0u, 1u, 2u, 0u },
        { 30, 300u, 2u, 2u, 3u, 0u },
        { 21, 201u, 1u, 1u, 4u, 0u },
    };
    SaltTextTargetFrontier frontier = { nodes, 5u, 7u };
    SaltAreaTask area_tasks[16];
    SaltAreaFrontier area;
    uint32_t path[5], path_count = 0;
    CHECK(salt_text_target_frontier_validate(&frontier, 5u) == 0,
          "target branch frontier validation failed");
    CHECK(salt_text_target_frontier_winning_path(
              &frontier, 3u, path, 5u, &path_count) == 0 &&
              path_count == 3u && path[0] == 0u &&
              path[1] == 2u && path[2] == 3u,
          "target branch winning path changed");
    memset(area_tasks, 0, sizeof area_tasks);
    memset(&area, 0, sizeof area);
    CHECK(salt_area_frontier_bind(&area, area_tasks, 16u) == 0 &&
              salt_text_target_frontier_build_area_tasks(
                  &frontier, &area, 3u, 7, 16u, 8u) == 0 &&
              area.count >= 8u &&
              area.tasks[0].node_id == nodes[0].node_id &&
              area.tasks[0].candidate_first ==
                  nodes[0].tentative_state_slot &&
              area.tasks[0].phase == 3u && area.tasks[0].expert_id == 7,
          "target branch area materialization changed");
    nodes[4].tentative_state_slot = 3u;
    CHECK(salt_text_target_frontier_validate(&frontier, 5u) != 0,
          "duplicate target state seat admitted");
    nodes[4].tentative_state_slot = 4u;
    nodes[3].depth = 1u;
    CHECK(salt_text_target_frontier_validate(&frontier, 5u) != 0,
          "invalid target parent depth admitted");
    puts("target branch frontier contract: PASS");
    return 0;
}

typedef struct HostGraphFixture {
    SaltTensorHostNodeOps node_ops;
    uint32_t serial_calls;
    uint32_t prepare_calls;
    uint32_t worker_calls;
    uint32_t finish_calls;
    uint32_t parallel_runs;
    uint32_t last_workers;
    int fail_worker;
} HostGraphFixture;

static int host_graph_is_parallel(void *opaque, uint32_t node) {
    (void)opaque;
    return node == 1u ? 1 : 0;
}

static int host_graph_prepare(
        void *opaque, uint32_t node, const SaltTensorHostNodeOps **ops_out,
        void **seat_out, uint32_t *active_workers_out) {
    HostGraphFixture *fixture = (HostGraphFixture *)opaque;
    if (!fixture || node != 1u || !ops_out || !seat_out ||
        !active_workers_out)
        return -1;
    fixture->prepare_calls++;
    *ops_out = &fixture->node_ops;
    *seat_out = fixture;
    *active_workers_out = 3u;
    return 0;
}

static int host_graph_serial(void *opaque, uint32_t node) {
    HostGraphFixture *fixture = (HostGraphFixture *)opaque;
    if (!fixture || node == 1u) return -1;
    fixture->serial_calls++;
    return 0;
}

static int host_graph_worker(void *opaque, uint32_t worker) {
    HostGraphFixture *fixture = (HostGraphFixture *)opaque;
    if (!fixture || worker >= 3u) return -1;
    fixture->worker_calls++;
    if (fixture->fail_worker && worker == 1u) return -1;
    return 0;
}

static int host_graph_finish(void *opaque) {
    HostGraphFixture *fixture = (HostGraphFixture *)opaque;
    if (!fixture) return -1;
    fixture->finish_calls++;
    return 0;
}

static int host_graph_parallel_run(
        void *opaque, int active_workers,
        void (*worker)(int worker, void *task), void *task) {
    HostGraphFixture *fixture = (HostGraphFixture *)opaque;
    if (!fixture || active_workers != 3 || !worker || !task) return -1;
    fixture->parallel_runs++;
    fixture->last_workers = (uint32_t)active_workers;
    for (int index = 0; index < active_workers; index++)
        worker(index, task);
    return 0;
}

static int host_graph_retained_flow_contract(void) {
    HostGraphFixture fixture;
    SaltTensorHostGraphCallbacks callbacks;
    SaltTensorHostGraphState state;
    SaltTensorHostGraphResult result;
    memset(&fixture, 0, sizeof fixture);
    memset(&callbacks, 0, sizeof callbacks);
    fixture.node_ops.worker = host_graph_worker;
    fixture.node_ops.finish = host_graph_finish;
    callbacks.node_count = 3u;
    callbacks.logical_node_count = 5u;
    callbacks.is_parallel = host_graph_is_parallel;
    callbacks.prepare_parallel = host_graph_prepare;
    callbacks.execute_serial = host_graph_serial;
    CHECK(salt_tensor_host_graph_execute(
              &state, &callbacks, &fixture, 8u, host_graph_parallel_run,
              &fixture, &result) == 0 &&
          fixture.serial_calls == 2u && fixture.prepare_calls == 1u &&
          fixture.worker_calls == 3u && fixture.finish_calls == 1u &&
          fixture.parallel_runs == 1u && fixture.last_workers == 3u &&
          result.nodes_executed == 5u && result.spans_executed == 3u &&
          result.serial_spans == 2u && result.parallel_spans == 1u &&
          result.internal_barriers == 1u,
          "retained graph coordinator/parallel boundary changed");
    memset(&fixture, 0, sizeof fixture);
    fixture.node_ops.worker = host_graph_worker;
    fixture.node_ops.finish = host_graph_finish;
    fixture.fail_worker = 1;
    CHECK(salt_tensor_host_graph_execute(
              &state, &callbacks, &fixture, 8u, host_graph_parallel_run,
              &fixture, &result) != 0 && fixture.prepare_calls == 1u &&
          fixture.finish_calls == 1u,
          "retained graph worker failure skipped lease cleanup");
    puts("retained graph coordinator flow: PASS");
    return 0;
}

static int tensor_engine_ready_graph_contract(void) {
    const SaltTensorGraphNode nodes[5] = {
        { 0u, 1u, 0u, 0u, 0u },
        { 1u, 2u, 0u, 1u, 0u },
        { 3u, 1u, 1u, 1u, SALT_TENSOR_GRAPH_NODE_RESOURCE_CHECK },
        { 4u, 2u, 2u, 2u, 0u },
        { 6u, 1u, 4u, 1u, 0u },
    };
    const uint32_t dependencies[5] = { 0u, 0u, 1u, 2u, 3u };
    SaltTensorGraphProgramDesc descriptor = {
        nodes, 5u, dependencies, 5u, 7u,
    };
    SaltTensorGraphProgram program;
    union {
        uint64_t alignment;
        unsigned char bytes[
            5u * sizeof(SaltTensorGraphNode) + 5u * sizeof(uint32_t)];
    } arena;
    SaltAreaTask tasks[5];
    SaltAreaFrontier frontier;
    SaltTensorGraphRuntime runtime;
    SaltTensorGraphProgress progress;
    uint32_t ready[5], ready_count = 0u;
    SaltTensorGraphNodeOutcome outcomes[5];
    size_t arena_bytes = 0u;

    memset(&program, 0, sizeof program);
    memset(&frontier, 0, sizeof frontier);
    memset(&runtime, 0, sizeof runtime);
    CHECK(salt_tensor_graph_program_arena_requirement(
              &descriptor, &arena_bytes) == 0 &&
          arena_bytes == sizeof arena.bytes,
          "engine TensorOps graph arena requirement changed");
    CHECK(salt_tensor_graph_program_compile(
              &program, &descriptor, arena.bytes, sizeof arena.bytes) == 0 &&
          program.ready && program.root_count == 1u &&
          program.logical_operation_count == 7u,
          "engine TensorOps graph compile failed");
    CHECK(salt_area_frontier_bind(&frontier, tasks, 5u) == 0 &&
          salt_tensor_graph_runtime_bind(&runtime, &frontier) == 0,
          "engine TensorOps graph did not bind the existing area frontier");
    CHECK(salt_tensor_graph_start(&runtime, &program, 7u, 2u) == 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.status == SALT_TENSOR_GRAPH_ACTIVE &&
          progress.ready_nodes == 1u && progress.completed_nodes == 0u,
          "engine TensorOps graph root admission changed");
    CHECK(salt_tensor_graph_take_ready(
              &runtime, 7u, ready, 5u, &ready_count) == 0 &&
          ready_count == 1u && ready[0] == 0u &&
          tasks[0].phase == 2u && tasks[0].output_first == 0u &&
          tasks[0].output_count == 1u,
          "engine TensorOps graph did not take the root deterministically");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 7u, ready, outcomes, ready_count) == 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.ready_nodes == 2u && progress.completed_nodes == 1u,
          "engine TensorOps graph did not expose independent successors");
    CHECK(salt_tensor_graph_take_ready(
              &runtime, 7u, ready, 5u, &ready_count) == 0 &&
          ready_count == 2u && ready[0] == 1u && ready[1] == 2u,
          "engine TensorOps ready set lost canonical node order");
    {
        const uint32_t reversed[2] = { 2u, 1u };
        const SaltTensorGraphNodeOutcome reversed_outcomes[2] = {
            SALT_TENSOR_GRAPH_NODE_DONE,
            SALT_TENSOR_GRAPH_NODE_DONE,
        };
        CHECK(salt_tensor_graph_complete_ready(
                  &runtime, 7u, reversed, reversed_outcomes, 2u) != 0 &&
              runtime.running_nodes == 2u,
              "engine TensorOps accepted completion-order authority");
    }
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    outcomes[1] = SALT_TENSOR_GRAPH_NODE_NEED_RESOURCE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 7u, ready, outcomes, ready_count) == 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.status == SALT_TENSOR_GRAPH_NEED_RESOURCE &&
          progress.ready_nodes == 0u && progress.waiting_nodes == 1u &&
          progress.completed_nodes == 2u,
          "engine TensorOps resource wait crossed a true merge");
    {
        const uint32_t refire[1] = { 2u };
        CHECK(salt_tensor_graph_refire(
                  &runtime, 7u, refire, 1u) == 0 &&
              salt_tensor_graph_progress(&runtime, &progress) == 0 &&
              progress.status == SALT_TENSOR_GRAPH_ACTIVE &&
              progress.ready_nodes == 1u && progress.waiting_nodes == 0u,
              "engine TensorOps resource refire changed generation state");
    }
    CHECK(salt_tensor_graph_take_ready(
              &runtime, 7u, ready, 5u, &ready_count) == 0 &&
          ready_count == 1u && ready[0] == 2u,
          "engine TensorOps did not refire the same resource node");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 7u, ready, outcomes, 1u) == 0,
          "engine TensorOps resource node completion failed");
    CHECK(salt_tensor_graph_take_ready(
              &runtime, 7u, ready, 5u, &ready_count) == 0 &&
          ready_count == 1u && ready[0] == 3u,
          "engine TensorOps merge became ready before both predecessors");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 7u, ready, outcomes, 1u) == 0 &&
          salt_tensor_graph_take_ready(
              &runtime, 7u, ready, 5u, &ready_count) == 0 &&
          ready_count == 1u && ready[0] == 4u,
          "engine TensorOps final successor readiness changed");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 7u, ready, outcomes, 1u) == 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.status == SALT_TENSOR_GRAPH_COMPLETE &&
          progress.completed_nodes == 5u && progress.ready_nodes == 0u &&
          runtime.active == 0,
          "engine TensorOps graph did not close exactly once");
    CHECK(salt_tensor_graph_start(&runtime, &program, 7u, 2u) != 0,
          "engine TensorOps admitted a stale graph generation");

    CHECK(salt_tensor_graph_start(&runtime, &program, 8u, 1u) == 0 &&
          salt_tensor_graph_take_ready(
              &runtime, 8u, ready, 5u, &ready_count) == 0,
          "engine TensorOps failure generation did not start");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 8u, ready, outcomes, 1u) == 0 &&
          salt_tensor_graph_take_ready(
              &runtime, 8u, ready, 5u, &ready_count) == 0 &&
          ready_count == 2u,
          "engine TensorOps failure ready set changed");
    outcomes[0] = SALT_TENSOR_GRAPH_NODE_DONE;
    outcomes[1] = SALT_TENSOR_GRAPH_NODE_FAILED;
    CHECK(salt_tensor_graph_complete_ready(
              &runtime, 8u, ready, outcomes, 2u) != 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.status == SALT_TENSOR_GRAPH_FAILED &&
          progress.completed_nodes == 2u && progress.cancelled_nodes == 3u &&
          runtime.active == 0,
          "engine TensorOps graph failure did not cancel unpublished work");
    CHECK(salt_tensor_graph_start(&runtime, &program, 9u, 0u) == 0 &&
          salt_tensor_graph_cancel(&runtime, 9u) == 0 &&
          salt_tensor_graph_progress(&runtime, &progress) == 0 &&
          progress.status == SALT_TENSOR_GRAPH_CANCELLED &&
          progress.cancelled_nodes == 5u && runtime.active == 0,
          "engine TensorOps explicit cancellation changed");
    {
        uint32_t invalid_dependencies[5] = { 0u, 0u, 1u, 1u, 3u };
        SaltTensorGraphProgramDesc invalid = descriptor;
        invalid.dependencies = invalid_dependencies;
        CHECK(salt_tensor_graph_program_arena_requirement(
                  &invalid, &arena_bytes) != 0,
              "engine TensorOps admitted a duplicate predecessor");
    }
    puts("portable-engine dependency-ready TensorOps graph: PASS");
    return 0;
}

static int target_policy_fixed_shape_contract(void) {
    SaltTextTargetPolicy compiled = {
        .execution_class = SALT_TEXT_EXECUTION_CPU_ONLY,
        .worker_budget = 8u,
        .sequence_tiles = 32u,
        .target_rows = 16u,
        .route_count = 1u,
        .queue_length = 4u,
        .candidate_count = 32u,
        .kv_warmup_rows = 512u,
        .warm_target_rows = 4u,
        .ready = 1,
    };
    SaltTextTargetPolicy invalid = compiled;
    SaltTextVerifyProgram program;
    uint32_t effective_x = 0u;
    CHECK(SALT_TEXT_NFQ_MAX_CHECKS == 512u &&
              SALT_DPR_MAX_WALK_HORIZON == 356u,
          "DPR walk expansion changed NFQ checking capacity");
    memset(&program, 0, sizeof program);
    program.ready = 1;
    program.maximum_candidates = 64u;
    CHECK(salt_text_verify_program_bind_target_policy(
              &program, &compiled) == 0 &&
          program.target_policy == &compiled &&
          compiled.sequence_tiles == 32u && compiled.target_rows == 16u &&
          compiled.route_count == 1u &&
          compiled.queue_length == 4u && compiled.candidate_count == 32u,
          "configured target policy did not retain fixed N/F/Q");
    CHECK(salt_text_target_policy_effective_x(
              &compiled, 511u, 16u, &effective_x) == 0 && effective_x == 1u &&
          salt_text_target_policy_effective_x(
              &compiled, 512u, 16u, &effective_x) == 0 && effective_x == 4u &&
          salt_text_target_policy_effective_x(
              &compiled, 512u, 2u, &effective_x) == 0 && effective_x == 2u,
          "KV warmup did not enforce historical X1/warm-X/remaining bounds");
    compiled.dpr_independent_draft = 1u;
    CHECK(salt_text_target_policy_effective_x(
              &compiled, 0u, 16u, &effective_x) == 0 && effective_x == 1u,
          "DPR draft opt-in changed ordinary cold admission");
    program.target_policy = NULL;
    invalid.candidate_count = 128u;
    CHECK(salt_text_verify_program_bind_target_policy(
              &program, &invalid) != 0,
          "configured target policy admitted N*F*Q candidate count");
    invalid = compiled;
    invalid.dpr_independent_draft = 2u;
    CHECK(salt_text_verify_program_bind_target_policy(
              &program, &invalid) != 0,
          "target policy admitted invalid DPR draft mode");
    printf("configured NFQ N/F/Q and X-TARGET policy: PASS\n");
    return 0;
}

static int projection_refill_contract(void) {
    SaltTextVerifyProgram program;
    SaltTextKvState state;
    SaltTextProjectionWindow previous;
    const float parent[8] = { 2, 7, 7, 3, 6, 0, 5, 4 };

    float rows[8][8], before[8][8], invalid_parent[8];
    int32_t ids[8], repeat[8];
    uint32_t reused = UINT32_MAX;
    memset(&program, 0, sizeof program);
    memset(&state, 0, sizeof state);
    memset(&previous, 0, sizeof previous);
    memset(rows, 0, sizeof rows);
    for (uint32_t row = 0; row < 8u; ++row)
        rows[row][row] = 10.0f;
    memcpy(before, rows, sizeof rows);
    state.position = 4u;
    state.transition_generation = 7u;
    program.ready = 1;
    program.vocabulary = 8u;
    program.maximum_candidates = 8u;
    program.kv_state = &state;

    memset(ids, 0xff, sizeof ids);
    memcpy(repeat, ids, sizeof ids);
    CHECK(salt_text_projection_refill_candidates(&program, NULL, parent,
              4u, 2u, ids, 8u, &reused) == 1 && reused == 0u &&
          memcmp(ids, repeat, sizeof ids) == 0,
          "missing projections fabricated a future trajectory");
    previous.logits = &rows[0][0];
    previous.generation = 7u;
    previous.position = 4u;
    previous.sequence_tiles = 4u;
    previous.route_count = 2u;
    previous.accepted_count = 2u;
    previous.winning_route = 0u;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              4u, 2u, ids, 8u, &reused) == 1 && reused == 0u,
          "refill fabricated seats the previous block never computed");
    /* Tile t of the new episode reads old row accepted+t-1: the row that
     * predicted exactly that position. Route 0 follows the winning route. */
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              2u, 2u, ids, 8u, &reused) == 0 && reused == 2u &&
          ids[0] == 1 && ids[1] == 2 && ids[2] == 2 && ids[3] == 0,
          "refill did not map new tiles onto already-computed rows");
    memcpy(repeat, ids, sizeof ids);
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              2u, 2u, repeat, 8u, &reused) == 0 &&
          memcmp(ids, repeat, sizeof ids) == 0 &&
          memcmp(rows, before, sizeof rows) == 0 &&
          state.position == 4u && state.transition_generation == 7u,
          "projection selection mutated logits/state or replay changed");
    previous.generation--;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              4u, 2u, ids, 8u, &reused) != 0,
          "projection refill admitted a stale generation");
    previous.generation++;
    previous.position--;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              4u, 2u, ids, 8u, &reused) != 0,
          "projection refill admitted a stale parent position");
    previous.position++;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              4u, 2u, ids, 7u, &reused) != 0,
          "projection refill ignored fixed output capacity");
    /* A full hit leaves no speculative rows: local memory is empty, not
     * refilled from parent ranks. */
    previous.accepted_count = 4u;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              2u, 2u, ids, 8u, &reused) == 1 && reused == 0u,
          "full hit fabricated local memory from parent ranks");
    previous.accepted_count = 1u;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              3u, 1u, ids, 8u, &reused) == 0 && reused == 2u &&
          ids[0] == 1 && ids[1] == 1 && ids[2] == 2,
          "rejection at 1 did not continue from rows 1 and 2");
    rows[1][0] = NAN;
    CHECK(salt_text_projection_refill_candidates(&program, &previous, parent,
              3u, 1u, ids, 8u, &reused) != 0,
          "refill admitted a nonfinite speculative row");
    rows[1][0] = 0.0f;
    memcpy(invalid_parent, parent, sizeof invalid_parent);
    invalid_parent[0] = NAN;
    CHECK(salt_text_projection_refill_candidates(
              &program, &previous, invalid_parent,
              4u, 1u, ids, 8u, &reused) != 0,
          "refill admitted nonfinite authoritative logits");
    puts("native projection local memory/stale/capacity/replay: PASS");
    return 0;
}

static int history_refill_contract(void) {
    const int32_t history[8] = { 1, 2, 3, 4, 1, 2, 3, 4 };
    const int32_t outputs[3] = { 1, 2, 3 };
    int32_t ids[6];
    uint32_t matched = 99u;
    memset(ids, 0xff, sizeof ids);
    /* Root 1 after ...3,4: the longest earlier match ends at index 4 (length
     * 5 including the root); the tokens that followed there are 2,3,4. */
    CHECK(salt_text_history_refill_candidates(history, 8u, NULL, 0u, 1, 10u,
              6u, ids, 6u, &matched) == 4 && matched == 5u &&
          ids[0] == 1 && ids[1] == 2 && ids[2] == 3 && ids[3] == 4,
          "history refill did not follow the longest earlier occurrence");
    /* Outputs extend the committed sequence; the proposal may reach into
     * them and stops at the end of committed tokens. */
    CHECK(salt_text_history_refill_candidates(history, 8u, outputs, 3u, 4, 10u,
              6u, ids, 6u, &matched) == 4 && matched == 8u &&
          ids[0] == 4 && ids[1] == 1 && ids[2] == 2 && ids[3] == 3,
          "history refill ignored committed outputs");
    /* Depth is bounded by the requested X. */
    CHECK(salt_text_history_refill_candidates(history, 8u, NULL, 0u, 1, 10u,
              2u, ids, 6u, &matched) == 2 && ids[0] == 1 && ids[1] == 2,
          "history refill exceeded the requested depth");
    /* No earlier occurrence: root only, no fabricated seats. */
    CHECK(salt_text_history_refill_candidates(history, 8u, NULL, 0u, 7, 10u,
              6u, ids, 6u, &matched) == 1 && matched == 0u && ids[0] == 7,
          "history refill fabricated a trajectory without a match");
    /* An occurrence at the very end has nothing after it; the earlier one
     * (a one-token match) still supplies its followers. */
    CHECK(salt_text_history_refill_candidates(history, 7u, NULL, 0u, 3, 10u,
              6u, ids, 6u, &matched) == 5 && matched == 1u &&
          ids[1] == 4 && ids[2] == 1 && ids[3] == 2 && ids[4] == 3,
          "history refill did not prefer the earlier occurrence with a suffix");
    CHECK(salt_text_history_refill_candidates(history, 8u, NULL, 0u, 10, 10u,
              6u, ids, 6u, &matched) == -1 &&
          salt_text_history_refill_candidates(history, 8u, NULL, 0u, 1, 10u,
              6u, ids, 4u, &matched) == -1,
          "history refill admitted an invalid root or capacity");
    puts("parent history route: longest/most-recent/bounded/miss PASS");
    return 0;
}

/* End-to-end scheduler fixture: an oracle truth sequence stands in for the
 * target model. TARGET accepts the longest proposed prefix that matches
 * truth, returns the correction logits at j-1, and publishes per-row
 * projections. It never proposes and never runs for a root-only boundary. */
typedef struct HistoryTargetContract {
    SaltTextKvState state;
    SaltTextVerifyProgram program;
    SaltTextTargetPolicy policy;
    SaltTextProjectionWindow projection;
    SaltTextScheduleStats stats;
    const int32_t *truth;
    uint32_t truth_count, position0;
    uint64_t clock;
    uint32_t steps, selections, targets, emissions, publications, rows_submitted;
    uint32_t nfq_checks;
    uint32_t selection_rank;
    int32_t selected_root;
    uint32_t proposed_total, accepted_total, history_hits, projection_hits;
    uint32_t invalid_target_shape;
    uint32_t strict_parent_root;
    uint32_t proof_limit, proof_calls, proof_checks;
    uint32_t materializations, materialized_rows;
#ifdef SALT_TEXT_COMMITTED_LOGIT_OBSERVER
    uint32_t logit_rounds, logit_rows;
    uint32_t logit_parent_base, logit_parent_ready;
    int32_t logit_parent_token;
#endif
    SaltAreaNfqMatrix sublane_matrix;
    SaltAreaNfqRoute sublane_routes[64];
    SaltAreaNfqItem sublane_items[64];
    uint32_t sublane_ready[8];
    float rows[64][10];
} HistoryTargetContract;

static uint64_t ht_clock(void *context) {
    return ++((HistoryTargetContract *)context)->clock;
}

static int ht_stop(void *context, int32_t token) {
    (void)context;
    return token == 9;
}

static void ht_select(float *logits, int32_t token) {
    for (uint32_t i = 0; i < 10u; i++) logits[i] = i == (uint32_t)token ? 2.0f : 0.0f;
}

static int ht_truth(const HistoryTargetContract *f, uint32_t position,
                    int32_t *token) {
    if (position < f->position0 || position - f->position0 >= f->truth_count)
        return -1;
    *token = f->truth[position - f->position0];
    return 0;
}

static int ht_step(void *context, int32_t token, float *logits) {
    HistoryTargetContract *f = context;
    int32_t expected, next;
    /* Ordinary transitions must follow truth; the synthetic close (8) is the
     * scheduler's own terminal step. */
    if (token != 8 &&
        (ht_truth(f, f->state.position, &expected) != 0 || token != expected))
        return -1;
    memset(&f->projection, 0, sizeof f->projection);
    f->steps++;
    f->state.position++;
    f->state.transition_generation++;
    if (ht_truth(f, f->state.position, &next) != 0) next = 9;
    ht_select(logits, next);
    return 0;
}

static int ht_emit(void *context, uint32_t count, int32_t token, int stop) {
    HistoryTargetContract *f = context;
    (void)count; (void)token; (void)stop;
    f->emissions++;
    return 0;
}

static int ht_publish(void *context, const SaltTextScheduleResult *r) {
    HistoryTargetContract *f = context;
    (void)r;
    f->publications++;
    return 0;
}

static int ht_observe_target(void *context, const SaltTextGenerated *g,
                             uint64_t elapsed) {
    HistoryTargetContract *f = context;
    (void)elapsed;
    f->proposed_total += g->proposal_count;
    f->accepted_total += g->target.accepted_count;
    if (g->proposal_source == SALT_TEXT_PROPOSAL_SOURCE_HISTORY) f->history_hits++;
    if (g->proposal_source == SALT_TEXT_PROPOSAL_SOURCE_PROJECTION) f->projection_hits++;
    return 0;
}

#ifdef SALT_TEXT_COMMITTED_LOGIT_OBSERVER
static int ht_observe_parent_logits(
        void *context, uint32_t accepted_base, int32_t token,
        const float *parent_logits, uint32_t vocabulary) {
    HistoryTargetContract *f = context;
    int32_t selected = 0;
    if (!f || !parent_logits || vocabulary != 10u ||
        accepted_base != f->logit_rows || f->logit_parent_ready)
        return -1;
    for (uint32_t index = 1u; index < vocabulary; index++)
        if (parent_logits[index] > parent_logits[selected])
            selected = (int32_t)index;
    if (selected != token) return -1;
    f->logit_parent_base = accepted_base;
    f->logit_parent_token = token;
    f->logit_parent_ready = 1u;
    return 0;
}

static int ht_observe_committed_logits(
        void *context, uint32_t accepted_base,
        const int32_t *tokens, uint32_t accepted_count,
        const float *projection_logits, uint32_t projection_rows,
        uint32_t vocabulary) {
    HistoryTargetContract *f = context;
    if (!f || vocabulary != 10u || !f->logit_parent_ready ||
        accepted_base != f->logit_parent_base || accepted_base != f->logit_rows)
        return -1;
    f->logit_parent_ready = 0u;
    if (accepted_count == 0u) return 0;
    if (!tokens || !projection_logits || projection_rows < accepted_count ||
        tokens[0] != f->logit_parent_token)
        return -1;
    for (uint32_t index = 1u; index < accepted_count; index++) {
        const float *source = projection_logits +
            (size_t)(index - 1u) * vocabulary;
        int32_t selected = 0;
        for (uint32_t token = 1u; token < vocabulary; token++)
            if (source[token] > source[selected]) selected = (int32_t)token;
        if (selected != tokens[index]) return -1;
    }
    f->logit_rounds++;
    f->logit_rows += accepted_count;
    return 0;
}
#endif

static int ht_select_proposal(void *context,
        SaltTextTokenEpochController *controller,
        const int32_t *ids, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    HistoryTargetContract *f = context;
    int32_t selected;
    uint32_t winner = UINT32_MAX;
    if (!f || !controller || !controller->ready || !ids || !parent || !result ||
        count != f->policy.candidate_count * f->policy.queue_length ||
        f->selection_rank >= count)
        return -1;
    selected = ids[f->selection_rank];
    for (uint32_t index = 0u; index < count; index++)
        if (ids[index] == selected) { winner = index; break; }
    if (winner == UINT32_MAX) return -1;
    f->selected_root = selected;
    f->selections++;
    f->nfq_checks += count;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.produced_count = 1u;
    result->target.pending_token_id = selected;
    result->target.winning_node_index = winner;
    result->target.winning_node_id = winner;
    result->target.result_position = f->state.position;
    result->target.transition_generation = f->state.transition_generation + 1u;
    result->target.pending_logits = parent;
    result->target.pending_logits_count = 10u;
    return 0;
}

static int ht_target(void *context, SaltTextTokenEpochController *controller,
        const int32_t *ids, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    HistoryTargetContract *f = context;
    int32_t expected, pending, parent_root = 0;
    uint32_t accepted = 0u;
    if (!f || !controller || !ids || count < 2u || count > 64u ||
        !parent || !result)
        return -1;
    for (int32_t token = 1; token < 10; token++)
        if (parent[token] > parent[parent_root]) parent_root = token;
    if (ids[0] != (f->strict_parent_root ? parent_root : f->selected_root))
        return -1;
    /* One batched forward: every row is "computed", then the validity vector
     * is reduced to its first false. */
    f->targets++;
    f->rows_submitted += count;
    for (uint32_t row = 0; row < count; row++) {
        int32_t guess;
        if (ht_truth(f, f->state.position + row + 1u, &guess) != 0) guess = 9;
        ht_select(f->rows[row], guess);
    }
    while (accepted < count &&
           ht_truth(f, f->state.position + accepted, &expected) == 0 &&
           ids[accepted] == expected)
        accepted++;
    if (accepted == 0u) return -1;
    if (ht_truth(f, f->state.position + accepted, &pending) != 0) pending = 9;
    f->state.position += accepted;
    f->state.transition_generation++;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.accepted_count = accepted;
    result->target.committed_count = accepted;
    result->target.produced_count = accepted + 1u;
    result->target.pending_token_id = pending;
    result->target.winning_node_index = accepted - 1u;
    result->target.winning_node_id = accepted - 1u;
    result->target.result_position = f->state.position;
    result->target.transition_generation = f->state.transition_generation;
    result->target.pending_logits = f->rows[accepted - 1u];
    result->target.pending_logits_count = 10u;
    result->target.projection_logits = &f->rows[0][0];
    result->target.projection_rows = count;
    result->target.backend.engine_submissions =
        f->invalid_target_shape == 1u ? count : 1u;
    result->target.backend.completion_fences = 1u;
    if (f->invalid_target_shape == 2u)
        result->target.projection_rows = 1u;
    f->projection.logits = &f->rows[0][0];
    f->projection.generation = f->state.transition_generation;
    f->projection.position = f->state.position;
    f->projection.sequence_tiles = count;
    f->projection.route_count = 1u;
    f->projection.accepted_count = accepted;
    f->projection.winning_route = 0u;
    return 0;
}

static int ht_prove_prefix(void *context, const int32_t *ids, uint32_t count,
        uint64_t parent_generation, uint32_t parent_position,
        uint32_t *checked_count, uint32_t *proven_count) {
    HistoryTargetContract *f = context;
    if (checked_count) *checked_count = 0u;
    if (proven_count) *proven_count = 0u;
    if (!f || !ids || count == 0u || !checked_count || !proven_count ||
        parent_generation != f->state.transition_generation ||
        parent_position != f->state.position || f->proof_limit == 0u ||
        f->proof_limit > count)
        return -1;
    for (uint32_t row = 0u; row < f->proof_limit; row++) {
        int32_t expected;
        if (ht_truth(f, parent_position + row, &expected) != 0 ||
            ids[row] != expected)
            return -1;
    }
    f->proof_calls++;
    *proven_count = f->proof_limit;
    *checked_count = f->proof_limit < count ? f->proof_limit + 1u : count;
    f->proof_checks += *checked_count;
    return 0;
}

static int ht_materialize_known(void *context, const int32_t *ids,
        uint32_t count, SaltTextVerifyResult *result) {
    HistoryTargetContract *f = context;
    int32_t expected, pending;
    if (!f || !ids || count == 0u || count > 64u || !result)
        return -1;
    for (uint32_t row = 0u; row < count; row++)
        if (ht_truth(f, f->state.position + row, &expected) != 0 ||
            ids[row] != expected)
            return -1;
    f->materializations++;
    f->materialized_rows += count;
    f->state.position += count;
    f->state.transition_generation++;
    if (ht_truth(f, f->state.position, &pending) != 0) pending = 9;
    ht_select(f->rows[count - 1u], pending);
    memset(result, 0, sizeof *result);
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = count;
    result->committed_count = count;
    result->produced_count = count + 1u;
    result->pending_token_id = pending;
    result->winning_node_index = count - 1u;
    result->winning_node_id = count - 1u;
    result->result_position = f->state.position;
    result->transition_generation = f->state.transition_generation;
    result->pending_logits = f->rows[count - 1u];
    result->pending_logits_count = 10u;
    result->backend.engine_submissions = 1u;
    result->backend.completion_fences = 1u;
    return 0;
}

static int unified_extent_materializer_contract(void) {
    static const uint32_t extents[] = { 1u, 19u, 40u };
    int32_t truth[40];
    for (uint32_t cell = 0u; cell < 3u; cell++) {
        HistoryTargetContract f;
        SaltTextTokenEpochController controller = {0};
        SaltTextGenerationBinding b;
        SaltTextGenerated generated;
        float logits[10] = {0}, scratch[10];
        uint32_t extent = extents[cell];
        memset(&f, 0, sizeof f);
        memset(&b, 0, sizeof b);
        for (uint32_t index = 0u; index < 40u; index++)
            truth[index] = (int32_t)(index % 8u) + 1;
        f.truth = truth;
        f.truth_count = 40u;
        f.position0 = 80u;
        f.state.position = f.position0;
        f.state.transition_generation = 1u;
        f.proof_limit = extent;
        f.program.ready = 1;
        f.program.kv_state = &f.state;
        f.program.vocabulary = 10u;
        f.program.maximum_context = 256u;
        f.program.maximum_candidates = 64u;
        f.policy.ready = 1;
        f.policy.worker_budget = 8u;
        f.policy.sequence_tiles = 2u;
        f.policy.target_rows = 40u;
        f.policy.route_count = 1u;
        f.policy.queue_length = 4u;
        f.policy.candidate_count = 2u;
        f.policy.kv_warmup_rows = 1u;
        f.policy.warm_target_rows = 40u;
        f.strict_parent_root = 1u;
        f.projection.logits = &f.rows[0][0];
        f.projection.generation = f.state.transition_generation;
        f.projection.position = f.state.position;
        f.projection.sequence_tiles = 41u;
        f.projection.route_count = 1u;
        f.projection.accepted_count = 1u;
        f.projection.winning_route = 0u;
        for (uint32_t row = 1u; row < 40u; row++)
            ht_select(f.rows[row], truth[row]);
        b.program = &f.program;
        b.policy = &f.policy;
        b.projection = &f.projection;
        b.context = &f;
        b.is_stop = ht_stop;
        b.select_proposal = ht_select_proposal;
        b.target = ht_target;
        b.prove_prefix = ht_prove_prefix;
        b.materialize_known = ht_materialize_known;
        CHECK(salt_text_token_epoch_init(&controller) == 0,
              "unified materializer token epoch init failed");
        controller.schedule_prior_kv_rows = 80u;
        ht_select(logits, 1);
        CHECK(salt_text_generate_candidates(&controller, &b, logits, 40u,
                  scratch, ht_clock, &f, &generated) == 0,
              "unified B extent generation failed");
        CHECK(generated.proposal_count == 40u &&
                  generated.proof_checked_rows ==
                      (extent < 40u ? extent + 1u : 40u) &&
                  generated.proven_prefix_rows == extent &&
                  generated.first_unproven == extent &&
                  generated.materialized_rows == extent &&
                  generated.state_reuse_rows == 0u &&
                  generated.target_model_rows == extent &&
                  generated.target_sublane_rounds == 0u &&
                  generated.target_sublane_cancelled_rows == 40u - extent &&
                  generated.target.accepted_count == extent &&
                  generated.target.committed_count == extent &&
                  generated.target.result_position == 80u + extent &&
                  generated.target.backend.engine_submissions == 1u &&
                  generated.target.backend.completion_fences == 1u &&
                  generated.target.backend.intermediate_host_publications == 0u &&
                  f.proof_calls == 1u &&
                  f.proof_checks == (extent < 40u ? extent + 1u : 40u) &&
                  f.materializations == 1u &&
                  f.materialized_rows == extent && f.targets == 0u &&
                  f.rows_submitted == 0u && f.state.position == 80u + extent,
              "B=1/B=j did not use one proof-selected materializer");
        CHECK(memcmp(generated.candidate_token_ids, truth,
                  (size_t)extent * sizeof truth[0]) == 0,
              "unified materializer changed the proven prefix");
        CHECK(logits[extent < 40u ? truth[extent] : 9] > logits[0],
              "row B-1 did not publish the authoritative boundary logits");
    }
    puts("unified materializer: B=1/B=j same mechanism, one submission PASS");
    return 0;
}

static int two_lane_target_sublane_contract(void) {
    int32_t truth[40];
    for (uint32_t cell = 0u; cell < 2u; cell++) {
        HistoryTargetContract f;
        SaltTextTokenEpochController controller = {0};
        SaltTextGenerationBinding b;
        SaltTextGenerated generated;
        float logits[10] = {0}, scratch[10];
        uint32_t expected_accepted = cell == 0u ? 40u : 10u;
        uint32_t expected_rounds = cell == 0u ? 5u : 2u;
        uint32_t expected_rows = cell == 0u ? 40u : 16u;
        uint32_t expected_cancelled = cell == 0u ? 0u : 24u;
        memset(&f, 0, sizeof f);
        memset(&b, 0, sizeof b);
        for (uint32_t index = 0u; index < 40u; index++)
            truth[index] = (int32_t)(index % 8u) + 1;
        f.truth = truth;
        f.truth_count = 40u;
        f.position0 = 80u;
        f.state.position = f.position0;
        f.state.transition_generation = 1u;
        f.program.ready = 1;
        f.program.kv_state = &f.state;
        f.program.vocabulary = 10u;
        f.program.maximum_context = 256u;
        f.program.maximum_candidates = 64u;
        f.policy.ready = 1;
        f.policy.worker_budget = 8u;
        f.policy.sequence_tiles = 2u;
        f.policy.target_rows = 40u;
        f.policy.route_count = 1u;
        f.policy.queue_length = 4u;
        f.policy.candidate_count = 2u;
        f.policy.kv_warmup_rows = 1u;
        f.policy.warm_target_rows = 40u;
        f.strict_parent_root = 1u;
        f.projection.logits = &f.rows[0][0];
        f.projection.generation = f.state.transition_generation;
        f.projection.position = f.state.position;
        f.projection.sequence_tiles = 41u;
        f.projection.route_count = 1u;
        f.projection.accepted_count = 1u;
        f.projection.winning_route = 0u;
        for (uint32_t row = 1u; row < 40u; row++)
            ht_select(f.rows[row], truth[row]);
        if (cell == 1u) truth[10] = 9;
        CHECK(salt_area_nfq_bind(&f.sublane_matrix,
                  f.sublane_routes, 64u, f.sublane_items, 64u) == 0,
              "two-lane startup-owned matrix bind failed");
        b.program = &f.program;
        b.policy = &f.policy;
        b.projection = &f.projection;
        b.context = &f;
        b.is_stop = ht_stop;
        b.select_proposal = ht_select_proposal;
        b.target = ht_target;
#ifdef SALT_TEXT_COMMITTED_LOGIT_OBSERVER
        b.observe_parent_logits = ht_observe_parent_logits;
        b.observe_committed_logits = ht_observe_committed_logits;
        b.observe_committed_logits_context = &f;
#endif
        b.target_sublane_matrix = &f.sublane_matrix;
        b.target_sublane_ready_items = f.sublane_ready;
        b.target_sublane_ready_capacity = 8u;
        CHECK(salt_text_token_epoch_init(&controller) == 0,
              "two-lane token epoch init failed");
        controller.schedule_prior_kv_rows = 80u;
        ht_select(logits, 1);
        {
            int generate_rc = salt_text_generate_candidates(
                &controller, &b, logits, 40u,
                scratch, ht_clock, &f, &generated);
            if (generate_rc != 0)
                fprintf(stderr, "two-lane diagnostic rc=%d selections=%u "
                    "targets=%u rows=%u position=%u generation=%llu "
                    "proposal=%u route=%u checks=%u source=%u "
                    "workers=%u depth=%u rounds=%u "
                    "matrix_ready=%d matrix_resolved=%d item_count=%u\n",
                    generate_rc, f.selections, f.targets, f.rows_submitted,
                    f.state.position,
                    (unsigned long long)f.state.transition_generation,
                    generated.proposal_count, generated.route_token_count,
                    generated.candidate_count, generated.proposal_source,
                    generated.target_sublane_workers,
                    generated.target_sublane_depth,
                    generated.target_sublane_rounds,
                    f.sublane_matrix.ready, f.sublane_matrix.resolved,
                    f.sublane_matrix.item_count);
            CHECK(generate_rc == 0,
              "two-lane X40/W8 target generation failed");
        }
        CHECK(generated.proposal_count == 40u &&
                  generated.route_token_count == 40u &&
                  generated.target_sublane_workers == 8u &&
                  generated.target_sublane_depth == 5u &&
                  generated.target_sublane_rounds == expected_rounds &&
                  generated.target_sublane_cancelled_rows == expected_cancelled &&
                  generated.target_model_rows == expected_rows &&
                  generated.target.accepted_count == expected_accepted &&
                  generated.target.committed_count == expected_accepted &&
                  generated.target.result_position == 80u + expected_accepted &&
                  generated.target.backend.engine_submissions == expected_rounds &&
                  generated.target.backend.completion_fences == expected_rounds &&
                  generated.target.backend.intermediate_host_publications == 0u &&
                  generated.target.projection_rows == 8u &&
                  f.targets == expected_rounds &&
                  f.rows_submitted == expected_rows &&
                  f.state.position == 80u + expected_accepted &&
                  f.sublane_matrix.resolved && !f.sublane_matrix.ready,
              "two-lane X40/W8/Y5 ownership or first-j termination changed");
#ifdef SALT_TEXT_COMMITTED_LOGIT_OBSERVER
        CHECK(f.logit_rounds == expected_rounds &&
                  f.logit_rows == expected_accepted &&
                  f.logit_parent_ready == 0u,
              "two-lane target logit observation changed");
#endif
        CHECK(memcmp(generated.candidate_token_ids, truth,
                  (size_t)expected_accepted * sizeof truth[0]) == 0,
              "two-lane target changed the accepted global prefix");
        if (cell == 1u)
            CHECK(logits[9] > logits[0],
                  "two-lane first-j correction was not grounded once");
    }
    puts("two-lane verdict: NFQ horizontal -> X40/W8/Y5 vertical PASS");
    return 0;
}

static int history_target_schedule_contract(void) {
    /* Cell 0: repetitive truth, every episode comes from history; zero B1.
     * Cell 1: truth deviates at output 6, forcing a rejection; the next
     * episode comes from target-block local memory at the retained j-1. */
    static const int32_t prompt[8] = { 1, 2, 3, 4, 1, 2, 3, 4 };
    static const int32_t truth_repeat[9] = { 1, 2, 3, 4, 1, 2, 3, 4, 1 };
    static const int32_t truth_deviate[9] = { 1, 2, 3, 4, 1, 2, 7, 4, 1 };
    for (int cell = 0; cell < 8; cell++) {
        HistoryTargetContract f;
        SaltTextTokenEpochController controller = {0};
        SaltTextSchedulerBindings b;
        SaltTextScheduleRequest q;
        SaltTextScheduleResult r = {0};
        float logits[10], scratch[10];
        int32_t ids[8] = {0};
        memset(&f, 0, sizeof f);
        memset(&b, 0, sizeof b);
        memset(&q, 0, sizeof q);
        f.truth = cell == 1 ? truth_deviate : truth_repeat;
        f.truth_count = 9u;
        f.invalid_target_shape = cell == 3 || cell == 4
            ? (uint32_t)(cell - 2) : 0u;
        f.position0 = cell == 5 ? 519u : 520u;
        f.state.position = f.position0;
        f.state.transition_generation = 1u;
        f.program.ready = 1;
        f.program.kv_state = &f.state;
        f.program.vocabulary = 10u;
        f.program.maximum_context = 1024u;
        f.program.maximum_candidates = 8u;
        f.policy.ready = 1;
        f.policy.sequence_tiles = 2u;   /* NFQ N: independent of X */
        f.policy.target_rows = cell == 5 ? 1u : 4u; /* X, independent of NFQ */
        f.policy.route_count = 1u;
        f.policy.queue_length = 4u;
        f.policy.candidate_count = 2u;
        f.policy.kv_warmup_rows = 512u;
        f.policy.warm_target_rows = 4u;
        b.generation.program = &f.program;
        b.generation.policy = &f.policy;
        b.generation.projection = &f.projection;
        b.generation.context = &f;
        b.generation.is_stop = ht_stop;
        b.generation.select_proposal = ht_select_proposal;
        b.generation.target = ht_target;
        if (cell == 2 || cell == 6)
            b.generation.target = NULL;
        if (cell == 7) {
            b.generation.select_proposal = NULL;
            b.generation.target = NULL;
        }
        b.context = &f;
        b.stats = &f.stats;
        b.now_ns = ht_clock;
        b.step_known = ht_step;
        b.emit_token = ht_emit;
        b.publish = ht_publish;
        b.observe_target = ht_observe_target;
        ht_select(logits, f.truth[0]);
        q.sampler.abi = SALT_SAMPLER_GREEDY_V1;
        q.logits = logits;
        q.scratch_logits = scratch;
        q.output_ids = ids;
        q.output_limit = 8u;
        q.close_token = 8;
        q.history_ids = prompt;
        q.history_count = 8u;
        if (cell == 6) {
            q.history_ids = NULL;
            q.history_count = 0u;
        }
        CHECK(salt_text_scheduler_bind(&controller, &b) == 0,
              "history-target scheduler bind failed");
        if (cell == 3 || cell == 4) {
            CHECK(salt_text_scheduler_run(&controller, &q, &r) != 0 &&
                  f.targets == 1u && f.publications == 0u,
                  "X-TARGET accepted split or truncated target execution");
            continue;
        }
        {
            int run_rc = salt_text_scheduler_run(&controller, &q, &r);
            if (run_rc != 0 || r.output_count != 8u ||
                memcmp(ids, f.truth, 8u * sizeof ids[0]) != 0 ||
                f.state.position != f.position0 + (cell >= 2 ? 7u : 8u))
                fprintf(stderr, "history-target cell=%d rc=%d outputs=%u "
                    "position=%u source=%u selections=%u targets=%u steps=%u\n",
                    cell, run_rc, r.output_count, f.state.position, f.position0,
                    f.selections, f.targets, f.steps);
            CHECK(run_rc == 0 && r.output_count == 8u &&
                  memcmp(ids, f.truth, 8u * sizeof ids[0]) == 0 &&
                  f.state.position == f.position0 + (cell >= 2 ? 7u : 8u),
                  "history-target output sequence diverged from truth");
        }
        if (cell >= 2) {
            CHECK(f.selections == (cell == 7 ? 0u : 8u) &&
                  f.nfq_checks == (cell == 7 ? 0u : 64u) &&
                  f.targets == 0u && f.steps == 7u &&
                  f.proposed_total == 0u && f.accepted_total == 0u &&
                  f.history_hits == 0u && f.projection_hits == 0u,
                  "ordinary NFQ depended on X or entered TARGET execution");
        } else if (cell == 0) {
            /* Two episodes of X=4, 8/8 accepted, no ordinary transition. */
            CHECK(f.selections == 2u && f.nfq_checks == 16u &&
                  f.targets == 2u && f.steps == 0u && f.rows_submitted == 8u &&
                  f.proposed_total == 8u && f.accepted_total == 8u &&
                  f.history_hits == 2u && f.projection_hits == 0u &&
                  f.stats.rejection_count == 0u &&
                  controller.schedule_bootstraps == 0u,
                  "repetitive truth did not amortize through history");
        } else {
            /* Episode 1: 4/4 from history. Episode 2: 4 proposed, 2 accepted
             * (truth 7 at output 6). Episode 3 resumes at j from the retained
             * rows: root 7 plus one local-memory row (O_remaining=2). */
            CHECK(f.selections == 3u && f.nfq_checks == 24u &&
                  f.targets == 3u && f.steps == 0u && f.rows_submitted == 10u &&
                  f.proposed_total == 10u && f.accepted_total == 8u &&
                  f.history_hits == 2u && f.projection_hits == 1u &&
                  f.stats.rejection_count == 1u &&
                  controller.schedule_bootstraps == 0u,
                  "rejection did not resume at j from local memory");
        }
        CHECK(salt_text_scheduler_finish(&controller, &q, &r) == 0 &&
              f.steps == (cell >= 2 ? 9u : 1u) &&
              f.state.position == f.position0 + 9u,
              "history-target close transition failed");
        CHECK(salt_text_scheduler_publish(&controller, &r) == 0 &&
              f.publications == 1u,
              "history-target publication failed");
    }
    puts("core scheduler: NFQ independent of X-disabled / X1 / history; X-TARGET preserved PASS");
    return 0;
}

static int multi_trajectory_root_mapping_contract(void) {
    static const int32_t truth[4] = { 2, 5, 6, 7 };
    HistoryTargetContract f;
    SaltTextTokenEpochController controller = {0};
    SaltTextGenerationBinding b;
    SaltTextGenerated generated;
    float logits[10] = {0}, scratch[10];
    uint32_t active_n = 0u, active_f = 0u;
    memset(&f, 0, sizeof f);
    memset(&b, 0, sizeof b);
    f.truth = truth;
    f.truth_count = 4u;
    f.position0 = 520u;
    f.state.position = f.position0;
    f.state.transition_generation = 7u;
    f.program.ready = 1;
    f.program.kv_state = &f.state;
    f.program.vocabulary = 10u;
    f.program.maximum_context = 1024u;
    f.program.maximum_candidates = 8u;
    f.policy.ready = 1;
    f.policy.sequence_tiles = 1u;
    f.policy.target_rows = 4u;
    f.policy.route_count = 2u;
    f.policy.queue_length = 4u;
    f.policy.candidate_count = 2u;
    f.policy.kv_warmup_rows = 512u;
    f.policy.warm_target_rows = 4u;
    f.projection.logits = &f.rows[0][0];
    f.projection.generation = f.state.transition_generation;
    f.projection.position = f.state.position;
    f.projection.sequence_tiles = 4u;
    f.projection.route_count = 2u;
    f.projection.accepted_count = 1u;
    f.projection.winning_route = 0u;
    /* Route zero is rooted at greedy token 1. Route one is rooted at token 2
     * and carries the causal suffix selected by the synthetic sampler. */
    logits[1] = 3.0f;
    logits[2] = 2.0f;
    for (uint32_t row = 1u; row < 4u; row++) {
        f.rows[row][3] = 3.0f;
        f.rows[row][8] = 2.0f;
    }
    for (uint32_t row = 5u; row < 8u; row++) {
        f.rows[row][4] = 3.0f;
        f.rows[row][truth[row - 4u]] = 2.0f;
    }
    f.selection_rank = 1u;
    b.program = &f.program;
    b.policy = &f.policy;
    b.projection = &f.projection;
    b.context = &f;
    b.is_stop = ht_stop;
    b.select_proposal = ht_select_proposal;
    b.target = ht_target;
    CHECK(salt_text_token_epoch_init(&controller) == 0,
          "multi-trajectory epoch init failed");
    controller.schedule_prior_kv_rows = 520u;
    CHECK(salt_text_generate_candidates(&controller, &b, logits, 4u, scratch,
              ht_clock, &f, &generated) == 0,
          "multi-trajectory proposal generation failed");
    CHECK(f.selections == 1u && f.nfq_checks == 8u &&
              f.targets == 1u && f.rows_submitted == 4u &&
              generated.proposal_count == 4u &&
              generated.route_token_count == 4u &&
              generated.candidate_count == 8u &&
              generated.proposal_source == SALT_TEXT_PROPOSAL_SOURCE_PROJECTION &&
              generated.projection_reused_rows == 6u &&
              generated.target.accepted_count == 4u &&
              memcmp(generated.candidate_token_ids, truth, sizeof truth) == 0,
          "selected root did not map to exactly one complete causal trajectory");
    CHECK(salt_text_target_policy_active_shape(&f.policy,
              generated.route_token_count, &active_n, &active_f) == 0 &&
              active_n == 4u && active_f == 1u,
          "selected trajectory did not reduce TARGET to active F1");
    puts("multi-trajectory NFQ root mapping -> one causal X-TARGET: PASS");
    return 0;
}

typedef struct ScheduleContract {
    SaltTextKvState state;
    SaltTextVerifyProgram program;
    SaltTextTargetPolicy policy;
    SaltTextProjectionWindow projection;
    SaltTextScheduleStats stats;
    uint64_t clock;
    uint32_t steps, targets, selections, nfq_checks, emissions, publications;
    int allow_target, step_token, target_next_token;
    float target_logits[2][7];
    int fail_emit, fail_publish, fail_step;
} ScheduleContract;

static uint64_t schedule_test_clock(void *context) {
    return ++((ScheduleContract *)context)->clock;
}

static int schedule_test_stop(void *context, int32_t token) {
    (void)context;
    return token == 6;
}


static int schedule_test_step(void *context, int32_t token, float *logits) {
    ScheduleContract *f = context;
    (void)token;
    if (f->fail_step) return -1;
    memset(&f->projection, 0, sizeof f->projection);
    f->steps++;
    f->state.position++;
    f->state.transition_generation++;
    for (uint32_t i = 0; i < 7u; i++)
        logits[i] = i == (uint32_t)(f->allow_target && f->steps > 1u
            ? f->target_next_token : (f->step_token ? f->step_token : 6))
            ? 2.0f : 0.0f;
    if (f->allow_target && f->steps > 1u && f->target_next_token < 6)
        f->target_next_token++;
    return 0;
}

static int schedule_test_emit(void *context, uint32_t count, int32_t token, int stop) {
    ScheduleContract *f = context;
    (void)count; (void)token; (void)stop;
    f->emissions++;
    return f->fail_emit ? -1 : 0;
}

static int schedule_test_publish(void *context, const SaltTextScheduleResult *r) {
    ScheduleContract *f = context;
    (void)r;
    f->publications++;
    return f->fail_publish ? -1 : 0;
}

static int schedule_test_select_proposal(void *context,
        SaltTextTokenEpochController *controller,
        const int32_t *ids, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    ScheduleContract *f = context;
    int32_t selected = 0;
    uint32_t winner = UINT32_MAX;
    if (!f || !controller || !controller->ready || !ids || !parent || !result ||
        count != f->policy.candidate_count * f->policy.queue_length)
        return -1;
    for (int32_t token = 1; token < 7; token++)
        if (parent[token] > parent[selected]) selected = token;
    for (uint32_t index = 0; index < count; index++)
        if (ids[index] == selected) { winner = index; break; }
    if (winner == UINT32_MAX) return -1;
    f->selections++;
    f->nfq_checks += count;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.produced_count = 1u;
    result->target.pending_token_id = selected;
    result->target.winning_node_index = winner;
    result->target.winning_node_id = winner;
    result->target.result_position = f->state.position;
    result->target.transition_generation = f->state.transition_generation + 1u;
    result->target.pending_logits = parent;
    result->target.pending_logits_count = 7u;
    return 0;
}

static int schedule_test_target(void *context, SaltTextTokenEpochController *controller,
        const int32_t *ids, uint32_t count, const float *parent,
        SaltTextTokenEpochResult *result) {
    ScheduleContract *f = context;
    int32_t selected = 0;
    uint32_t winner = UINT32_MAX;
    if (!f || !f->allow_target || !controller || !ids ||
        count == 0u || count > f->policy.candidate_count ||
        !parent || !result || f->target_next_token < 0 ||
        f->target_next_token >= 7)
        return -1;
    for (int32_t token = 1; token < 7; token++)
        if (parent[token] > parent[selected]) selected = token;
    for (uint32_t index = 0; index < count; index++)
        if (ids[index] == selected) {
            winner = index;
            break;
        }
    if (winner == UINT32_MAX) return -1;
    f->targets++;
    f->state.position++;
    f->state.transition_generation++;
    for (uint32_t row = 0; row < 2u; row++)
        for (uint32_t token = 0; token < 7u; token++)
            f->target_logits[row][token] =
                token == (uint32_t)(f->target_next_token + (int)row)
                    ? 2.0f : 0.0f;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.accepted_count = 1u;
    result->target.committed_count = 1u;
    result->target.produced_count = 2u;
    result->target.pending_token_id = f->target_next_token;
    result->target.winning_node_index = winner;
    result->target.winning_node_id = winner;
    result->target.result_position = f->state.position;
    result->target.transition_generation = f->state.transition_generation;
    result->target.pending_logits = f->target_logits[0];
    result->target.pending_logits_count = 7u;
    f->projection.logits = &f->target_logits[0][0];
    f->projection.generation = f->state.transition_generation;
    f->projection.position = f->state.position;
    f->projection.sequence_tiles = count;
    f->projection.route_count = 1u;
    f->projection.accepted_count = 1u;
    f->projection.winning_route = 0u;
    if (f->target_next_token < 6) f->target_next_token++;
    return 0;
}

static int schedule_bootstrap_target_contract(void) {
    ScheduleContract f;
    SaltTextTokenEpochController controller = {0};
    SaltTextSchedulerBindings b;
    SaltTextScheduleRequest q;
    SaltTextScheduleResult r = {0};
    float logits[7] = {0, 0, 2, 0, 0, 0, 0}, scratch[7];
    int32_t ids[4] = {0};
    memset(&f, 0, sizeof f);
    memset(&b, 0, sizeof b);
    memset(&q, 0, sizeof q);
    f.state.position = 1u;
    f.state.transition_generation = 1u;
    f.program.ready = 1;
    f.program.kv_state = &f.state;
    f.program.vocabulary = 7u;
    f.program.maximum_context = 16u;
    f.program.maximum_candidates = 4u;
    f.policy.ready = 1;
    f.policy.sequence_tiles = 2u;
    f.policy.target_rows = 2u;
    f.policy.route_count = 1u;
    f.policy.queue_length = 2u;
    f.policy.candidate_count = 2u;
    f.policy.kv_warmup_rows = 1u;
    f.policy.warm_target_rows = 2u;
    f.allow_target = 1;
    f.step_token = 3;
    f.target_next_token = 4;
    b.generation.program = &f.program;
    b.generation.policy = &f.policy;
    b.generation.projection = &f.projection;
    b.generation.context = &f;
    b.generation.is_stop = schedule_test_stop;
    b.generation.select_proposal = schedule_test_select_proposal;
    b.generation.target = schedule_test_target;
    b.context = &f;
    b.stats = &f.stats;
    b.now_ns = schedule_test_clock;
    b.step_known = schedule_test_step;
    b.emit_token = schedule_test_emit;
    b.publish = schedule_test_publish;
    q.sampler.abi = SALT_SAMPLER_GREEDY_V1;
    q.logits = logits;
    q.scratch_logits = scratch;
    q.output_ids = ids;
    q.output_limit = 4u;
    q.close_token = 5;
    CHECK(salt_text_scheduler_bind(&controller, &b) == 0,
          "bootstrap-target scheduler bind failed");
    CHECK(salt_text_scheduler_run(&controller, &q, &r) == 0 &&
          r.output_count == 4u && ids[0] == 2 && ids[1] == 3 &&
          ids[2] == 4 && ids[3] == 5,
          "bootstrap-target output sequence failed");
    CHECK(controller.schedule_bootstraps == 3u && f.steps == 3u &&
          f.selections == 4u && f.nfq_checks == 16u &&
          f.targets == 0u && f.state.position == 4u,
          "scheduler manufactured TARGET work without a causal trajectory");
    CHECK(salt_text_scheduler_finish(&controller, &q, &r) == 0 &&
          f.steps == 5u && f.targets == 0u && f.state.position == 6u,
          "native close transition failed");
    CHECK(salt_text_scheduler_publish(&controller, &r) == 0 &&
          f.publications == 1u,
          "bootstrap-target publication failed");
    puts("core scheduler: no-candidate native causal decode PASS");
    return 0;
}

static int schedule_request_contract(void) {
    for (int cell = 0; cell < 6; cell++) {
        ScheduleContract f;
        SaltTextTokenEpochController controller = {0};
        SaltTextSchedulerBindings b;
        SaltTextScheduleRequest q;
        SaltTextScheduleResult r = {0};
        float logits[7] = {0, 0, 2, 0, 0, 0, 0}, scratch[7];
        int32_t ids[4] = {0};
        memset(&f, 0, sizeof f);
        memset(&b, 0, sizeof b);
        memset(&q, 0, sizeof q);
        f.state.position = 1u;
        f.program.ready = 1;
        f.program.kv_state = &f.state;
        f.program.vocabulary = 7u;
        f.program.maximum_context = 16u;
        f.program.maximum_candidates = 1u;
        f.policy.ready = 1;
        f.policy.sequence_tiles = f.policy.route_count = 1u;
        f.policy.target_rows = 1u;
        f.policy.queue_length = 1u;
        f.policy.candidate_count = 1u;
        f.policy.kv_warmup_rows = 1u;
        f.policy.warm_target_rows = 1u;
        b.generation.program = &f.program;
        b.generation.policy = &f.policy;
        b.generation.projection = &f.projection;
        b.generation.context = &f;
        b.generation.is_stop = schedule_test_stop;
        b.generation.select_proposal = schedule_test_select_proposal;
        b.generation.target = schedule_test_target;
        b.context = &f;
        b.stats = &f.stats;
        b.now_ns = schedule_test_clock;
        b.step_known = schedule_test_step;
        b.emit_token = schedule_test_emit;
        b.publish = schedule_test_publish;
        q.sampler.abi = SALT_SAMPLER_GREEDY_V1;
        q.logits = logits;
        q.scratch_logits = scratch;
        q.output_ids = ids;
        q.output_limit = cell == 1 ? 1u : 4u;
        q.close_token = cell == 2 ? 6 : 5;
        f.fail_emit = cell == 3;
        f.fail_step = cell == 4;
        f.fail_publish = cell == 5;
        CHECK(salt_text_scheduler_bind(&controller, &b) == 0, "scheduler bind failed");
        CHECK(salt_text_scheduler_publish(&controller, &r) != 0,
              "scheduler published before execution");
        if (cell == 3 || cell == 4) {
            CHECK(salt_text_scheduler_run(&controller, &q, &r) != 0 &&
                  !controller.schedule_active && !f.publications,
                  "scheduler failed open after operation/presentation failure");
            continue;
        }
        CHECK(salt_text_scheduler_run(&controller, &q, &r) == 0,
              "scheduler ordinary run failed");
        CHECK(f.selections == r.output_count && f.nfq_checks == r.output_count &&
              f.targets == 0u,
              "X1/cap/stop skipped cheap NFQ or submitted target compute");
        CHECK(salt_text_scheduler_run(&controller, &q, &r) != 0,
              "scheduler admitted a duplicate active request");
        CHECK(salt_text_scheduler_publish(&controller, &r) != 0 && !f.publications,
              "scheduler published before pending/close completion");
        CHECK(salt_text_scheduler_finish(&controller, &q, &r) == 0,
              "scheduler pending/close failed");
        CHECK(f.state.position == r.source_position + r.output_count +
                  (q.close_token != r.stop_token) &&
              f.steps == r.output_count + (q.close_token != r.stop_token),
              "scheduler consumed pending/close more than once");
        CHECK(salt_text_scheduler_finish(&controller, &q, &r) != 0,
              "scheduler repeated finalization");
        CHECK((salt_text_scheduler_publish(&controller, &r) == 0) == !f.fail_publish,
              "scheduler publication failure result changed");
        CHECK(salt_text_scheduler_publish(&controller, &r) != 0 && f.publications == 1u,
              "scheduler repeated publication");
    }
    puts("core request scheduler: terminal/cap/pending/close/failure/publication PASS");
    return 0;
}

/* The native token-tape API was removed at b23dff9. Current scheduler,
 * generation, NFQ and TARGET coverage above remains authoritative. */
#if 0
/* Scheduler admission must name a real token tape, not a ready-only verifier
 * stand-in. These cells exercise the real compiler/interpreter/fence/publish
 * contract with deterministic fixture payloads, not model arithmetic. */
typedef struct NativeScheduleContract {
    SaltTextTokenProgram program;
    SaltTextTokenCell arena[3];
    SaltTextTokenExecutor executor;
    SaltTextNativeBinding native;
    SaltTextScheduleStats stats;
    SaltTextProjectionWindow projection;
    SaltTextKvState *state;
    SaltTextTokenResult last;
    float next_logits[VOCABULARY], staged_logits[VOCABULARY];
    float *output_logits;
    int32_t committed_ids[MAXIMUM_CONTEXT], staged_token, stop_token;
    uint64_t clock;
    uint32_t calls, submissions, cells, barriers, fences, token_publications;
    uint32_t emissions, observations, publications, aborts, next_cell;
    int failure, fail_token;
} NativeScheduleContract;

enum {
    NATIVE_FAIL_BEGIN = 1, NATIVE_FAIL_CELL, NATIVE_FAIL_BARRIER,
    NATIVE_FAIL_FINISH, NATIVE_FAIL_TOKEN_PUBLISH, NATIVE_FAIL_ABORT,
    NATIVE_FAIL_EMIT, NATIVE_FAIL_PUBLISH, NATIVE_FAIL_POSITION
};

static int native_schedule_begin(void *context, const SaltTextTokenProgram *p,
        const SaltTextTokenExecution *e) {
    NativeScheduleContract *f = context;
    if (f->failure == NATIVE_FAIL_BEGIN || e->input_token_id == f->fail_token ||
        p != &f->program || e->source_position != f->state->position ||
        e->transition_generation != f->state->transition_generation + 1u ||
        e->source_position >= f->native.maximum_context ||
        e->input_token_id < 0 || (uint32_t)e->input_token_id >= f->native.vocabulary)
        return -1;
    f->submissions++;
    f->next_cell = 0u;
    f->staged_token = -1;
    return 0;
}

static SaltTextTokenCellResult native_schedule_cell(void *context,
        const SaltTextTokenProgram *p, const SaltTextTokenExecution *e,
        const SaltTextTokenCell *cell, uint32_t index) {
    NativeScheduleContract *f = context;
    if (f->failure == NATIVE_FAIL_CELL || f->failure == NATIVE_FAIL_ABORT ||
        p != &f->program || cell != &p->cells[index] || index != f->next_cell)
        return SALT_TEXT_TOKEN_CELL_FAILED;
    if (cell->kind == SALT_TEXT_TOKEN_EMBED) f->staged_token = e->input_token_id;
    if (cell->kind == SALT_TEXT_TOKEN_HEAD)
        memcpy(f->staged_logits, f->next_logits, sizeof f->staged_logits);
    f->next_cell++;
    f->cells++;
    return SALT_TEXT_TOKEN_CELL_OK;
}

static int native_schedule_barrier(void *context, const SaltTextTokenProgram *p,
        const SaltTextTokenExecution *e, const SaltTextTokenCell *cell, uint32_t index) {
    NativeScheduleContract *f = context;
    (void)p; (void)e; (void)cell; (void)index;
    if (f->failure == NATIVE_FAIL_BARRIER) return -1;
    f->barriers++;
    return 0;
}

static int native_schedule_fence(void *context, const SaltTextTokenProgram *p,
        const SaltTextTokenExecution *e) {
    NativeScheduleContract *f = context;
    (void)e;
    if (f->failure == NATIVE_FAIL_FINISH || f->next_cell != p->cell_count) return -1;
    f->fences++;
    return 0;
}

static int native_schedule_abort(void *context, const SaltTextTokenProgram *p,
        const SaltTextTokenExecution *e, SaltTextTokenStatus status, uint32_t completed) {
    NativeScheduleContract *f = context;
    (void)p; (void)e; (void)status; (void)completed;
    f->aborts++;
    f->staged_token = -1;
    memset(f->staged_logits, 0, sizeof f->staged_logits);
    return f->failure == NATIVE_FAIL_ABORT ? -1 : 0;
}

static int native_schedule_token_publish(void *context, const SaltTextTokenProgram *p,
        const SaltTextTokenExecution *e, uint32_t position) {
    NativeScheduleContract *f = context;
    if (f->failure == NATIVE_FAIL_TOKEN_PUBLISH || f->next_cell != p->cell_count ||
        f->fences != f->token_publications + 1u ||
        f->staged_token != e->input_token_id || position != f->state->position + 1u)
        return -1;
    f->committed_ids[f->token_publications++] = f->staged_token;
    memcpy(f->output_logits, f->staged_logits, sizeof f->staged_logits);
    f->state->position = f->failure == NATIVE_FAIL_POSITION ? position - 1u : position;
    f->state->transition_generation = e->transition_generation;
    return 0;
}

static const SaltTextTokenExecutorOps native_schedule_ops = {
    .begin = native_schedule_begin,
    .execute_cell = native_schedule_cell,
    .dependency_barrier = native_schedule_barrier,
    .finish = native_schedule_fence,
    .abort = native_schedule_abort,
    .publish = native_schedule_token_publish,
};

static int native_schedule_step(void *context, int32_t token, float *logits) {
    NativeScheduleContract *f = context;
    SaltTextTokenExecution e;
    f->calls++;
    if (f->state->transition_generation == UINT64_MAX) return -1;
    e = (SaltTextTokenExecution) {
        f->state->transition_generation + 1u, f->state->position, token,
    };
    f->output_logits = logits;
    return salt_text_token_execute(f->native.program, f->native.executor, &e, &f->last) == 0 &&
        f->last.status == SALT_TEXT_TOKEN_COMMITTED ? 0 : -1;
}

static uint64_t native_schedule_clock(void *context) {
    return ++((NativeScheduleContract *)context)->clock;
}

static int native_schedule_stop(void *context, int32_t token) {
    return token == *(const int32_t *)context;
}

static int native_schedule_emit(void *context, uint32_t count, int32_t token, int stop) {
    NativeScheduleContract *f = context;
    (void)token; (void)stop;
    if (count != ++f->emissions || f->failure == NATIVE_FAIL_EMIT) return -1;
    return 0;
}

static int native_schedule_observe(void *context, uint32_t source, int32_t token,
        int committed, uint64_t elapsed, uint64_t draws) {
    NativeScheduleContract *f = context;
    (void)token; (void)elapsed; (void)draws;
    if (f->state->position != source + (uint32_t)committed) return -1;
    f->observations++;
    return 0;
}

static int native_schedule_publish(void *context, const SaltTextScheduleResult *r) {
    NativeScheduleContract *f = context;
    f->publications++;
    return f->failure == NATIVE_FAIL_PUBLISH || r->result_position != f->state->position
        ? -1 : 0;
}

static int native_schedule_forbidden_release(void *context) {
    (void)context;
    return -1;
}

static int native_schedule_init(NativeScheduleContract *f, SaltTextKvState *state,
        SaltTextSchedulerBindings *b) {
    static const SaltTextTokenCell cells[] = {
        {SALT_TEXT_TOKEN_EMBED, -1, 0, -1, 1,
         SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE, 0, 0, sizeof(int32_t)},
        {SALT_TEXT_TOKEN_HEAD, 0, 0, 0, 2,
         SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE | SALT_TEXT_TOKEN_CELL_BARRIER_AFTER,
         0, sizeof(int32_t), VOCABULARY * sizeof(float)},
        {SALT_TEXT_TOKEN_FINALIZE, -1, 0, 1, 3, 0, 0, 0, 0},
    };
    const SaltTextTokenProgramDesc descriptor = {cells, 3u, 1u, 1u};
    memset(f, 0, sizeof *f);
    memset(b, 0, sizeof *b);
    f->state = state;
    f->stop_token = -1;
    f->fail_token = -1;
    f->next_logits[3] = 2.0f;
    if (salt_text_token_program_compile(&f->program, &descriptor,
            f->arena, sizeof f->arena) != 0) return -1;
    f->executor = (SaltTextTokenExecutor) {&native_schedule_ops, f, 0};
    f->native = (SaltTextNativeBinding) {
        &f->program, &f->executor, state, VOCABULARY, MAXIMUM_CONTEXT,
    };
    b->native = &f->native;
    b->generation.context = &f->stop_token;
    b->generation.is_stop = native_schedule_stop;
    b->stats = &f->stats;
    b->context = f;
    b->now_ns = native_schedule_clock;
    b->step_known = native_schedule_step;
    b->emit_token = native_schedule_emit;
    b->observe_native = native_schedule_observe;
    b->publish = native_schedule_publish;
    return 0;
}

static int native_schedule_binding_contract(void) {
    NativeScheduleContract f;
    SaltTextKvState state = {1u, 1u};
    SaltTextSchedulerBindings binding;
    SaltTextTokenEpochController controller = {0};
    SaltTextScheduleDpr dpr = {0};
    SaltAreaNfqMatrix matrix;
    uint32_t ready_item = 0u;
    CHECK(native_schedule_init(&f, &state, &binding) == 0,
          "native scheduler tape compile failed");
    CHECK(f.program.cells == f.arena && f.program.cell_count == 3u &&
          salt_text_scheduler_bind(&controller, &binding) == 0,
          "native scheduler rejected compiled tape");
    binding.dpr = &dpr;
    CHECK(salt_text_scheduler_bind(&controller, &binding) == 0,
          "native scheduler rejected explicit DPR off");
#define NATIVE_REFUSE(mutation, message) do { \
    SaltTextSchedulerBindings b = binding; \
    SaltTextNativeBinding n = f.native; \
    SaltTextTokenProgram p = f.program; \
    SaltTextTokenExecutor executor = f.executor; \
    SaltTextTokenExecutorOps ops = native_schedule_ops; \
    SaltTextKvState kv = state; \
    SaltTextTokenCell cells[3]; \
    SaltTextTokenEpochController before; \
    memcpy(&before, &controller, sizeof before); \
    memcpy(cells, f.arena, sizeof cells); \
    p.cells = cells; n.program = &p; n.executor = &executor; n.kv_state = &kv; \
    executor.ops = &ops; b.native = &n; \
    mutation; \
    CHECK(salt_text_scheduler_bind(&controller, &b) != 0 && \
          memcmp(&before, &controller, sizeof before) == 0, message); \
} while (0)
    NATIVE_REFUSE(b.native = NULL, "missing native and verifier program admitted");
    NATIVE_REFUSE(n.program = NULL, "missing native program admitted");
    NATIVE_REFUSE(p = ((SaltTextTokenProgram){.ready = 1}), "ready-only tape admitted");
    NATIVE_REFUSE(p.ready = 0, "unready native program admitted");
    NATIVE_REFUSE(p.cells = NULL, "missing native cells admitted");
    NATIVE_REFUSE(p.cell_count = 0u, "empty native tape admitted");
    NATIVE_REFUSE(p.layer_count = 0u, "zero native layers admitted");
    NATIVE_REFUSE(cells[1].dependency_cell = 1, "native forward dependency admitted");
    NATIVE_REFUSE(cells[1].completion_id = 0u, "native incomplete cell admitted");
    NATIVE_REFUSE(p.maximum_internal_barriers = 0u, "native barrier overflow admitted");
    NATIVE_REFUSE(n.kv_state = NULL, "missing native state admitted");
    NATIVE_REFUSE(n.vocabulary = 0u, "zero native vocabulary admitted");
    NATIVE_REFUSE(n.vocabulary = (uint32_t)INT_MAX + 1u, "oversized native vocabulary admitted");
    NATIVE_REFUSE(n.maximum_context = 0u, "zero native context admitted");
    NATIVE_REFUSE(kv.position = MAXIMUM_CONTEXT + 1u, "out-of-context native state admitted");
    NATIVE_REFUSE(n.executor = NULL, "missing native executor admitted");
    NATIVE_REFUSE(executor.ops = NULL, "missing native ops admitted");
    NATIVE_REFUSE(ops.begin = NULL, "missing native begin admitted");
    NATIVE_REFUSE(ops.execute_cell = NULL, "missing native cell executor admitted");
    NATIVE_REFUSE(ops.dependency_barrier = NULL, "missing native barrier admitted");
    NATIVE_REFUSE(ops.finish = NULL, "missing native fence admitted");
    NATIVE_REFUSE(ops.abort = NULL, "missing native abort admitted");
    NATIVE_REFUSE(ops.publish = NULL, "missing native publisher admitted");
    NATIVE_REFUSE(executor.layer_lookahead = 1, "unbound native lookahead admitted");
    NATIVE_REFUSE(b.generation.program = &serial_fixture.program, "ambiguous native/verifier admitted");
    NATIVE_REFUSE(b.generation.policy = &serial_fixture.target_policy, "native target policy admitted");
    NATIVE_REFUSE(b.generation.projection = &f.projection, "native projection admitted");
    NATIVE_REFUSE(b.generation.select_proposal = schedule_test_select_proposal, "native proposal admitted");
    NATIVE_REFUSE(b.generation.target = schedule_test_target, "native TARGET admitted");
#ifdef SALT_TEXT_COMMITTED_LOGIT_OBSERVER
    NATIVE_REFUSE(b.generation.observe_parent_logits = ht_observe_parent_logits,
                  "native parent-logit observer admitted");
    NATIVE_REFUSE(b.generation.observe_committed_logits = ht_observe_committed_logits,
                  "native committed-logit observer admitted");
    NATIVE_REFUSE(b.generation.observe_committed_logits_context = &f,
                  "native logit observer context admitted");
#endif
    NATIVE_REFUSE(b.generation.prove_prefix = ht_prove_prefix, "native prefix proof admitted");
    NATIVE_REFUSE(b.generation.materialize_known = ht_materialize_known, "native materializer admitted");
    NATIVE_REFUSE(b.generation.target_sublane_matrix = &matrix, "native target matrix admitted");
    NATIVE_REFUSE(b.generation.target_sublane_ready_items = &ready_item, "native target ready seats admitted");
    NATIVE_REFUSE(b.generation.target_sublane_ready_capacity = 1u, "native target capacity admitted");
    NATIVE_REFUSE(b.release_attention = native_schedule_forbidden_release, "native DPR callback admitted");
    NATIVE_REFUSE(b.observe_target = ht_observe_target, "native target observer admitted");
    NATIVE_REFUSE(b.step_known = NULL, "native missing step callback admitted");
    NATIVE_REFUSE(b.generation.is_stop = NULL, "native missing stop callback admitted");
    dpr.mode = SALT_DPR_PERSIST;
    NATIVE_REFUSE((void)0, "native enabled DPR admitted");
    dpr.mode = SALT_DPR_OFF;
    dpr.attention_enabled = 1;
    NATIVE_REFUSE((void)0, "native attention DPR admitted");
#undef NATIVE_REFUSE
    CHECK(!f.calls && !f.submissions && !f.publications &&
          state.position == 1u && state.transition_generation == 1u,
          "native binding performed token work or changed canonical state");
    return 0;
}

static int native_schedule_parity_contract(SyntheticPackage *package) {
    static ExecutionFixture verifier;
    for (int scenario = 0; scenario < 8; scenario++) {
        NativeScheduleContract f[2];
        SaltTextKvState native_state = {1u, 0u};
        SaltTextSchedulerBindings b[2];
        SaltTextScheduleResult r[2] = {{0}};
        float logits[2][VOCABULARY] = {{0}}, scratch[2][VOCABULARY] = {{0}};
        int32_t ids[2][3] = {{0}};
        CHECK(prepare_execution(&verifier, package, SALT_TEXT_KV_PRIVATE_ABSOLUTE, 1u) == 0 &&
              compile_execution(&verifier) == 0,
              "native parity verifier compile failed");
        for (int route = 0; route < 2; route++) {
            SaltTextTokenEpochController controller = {0};
            SaltTextScheduleRequest q = {0};
            CHECK(native_schedule_init(&f[route], route ? &native_state : &verifier.state,
                      &b[route]) == 0, "native parity tape compile failed");
            if (!route) {
                b[route].native = NULL;
                b[route].generation.program = &verifier.program;
                b[route].generation.policy = &verifier.target_policy;
                b[route].generation.projection = &f[route].projection;
            }
            {
                SaltTextSchedulerBindings invalid = b[route];
                if (route) invalid.generation.program = &verifier.program;
                else invalid.generation.program = NULL;
                CHECK(salt_text_scheduler_bind(&controller, &invalid) != 0 && !controller.ready,
                      "missing verifier or ambiguous real native/verifier binding admitted");
                if (!route) {
                    invalid = b[route]; invalid.generation.policy = NULL;
                    CHECK(salt_text_scheduler_bind(&controller, &invalid) != 0,
                          "legacy verifier no longer requires its policy");
                    invalid = b[route]; invalid.generation.projection = NULL;
                    CHECK(salt_text_scheduler_bind(&controller, &invalid) != 0,
                          "legacy verifier no longer requires its projection");
                }
            }
            f[route].stop_token = scenario == 1 || scenario == 2 || scenario == 4 ? 6 : -1;
            if (scenario == 3) f[route].stop_token = 2;
            if (f[route].stop_token == 6) {
                f[route].next_logits[3] = 0.0f;
                f[route].next_logits[6] = 2.0f;
            }
            q.sampler = (SaltSamplerConfig) {scenario >= 4 ?
                SALT_SAMPLER_TEMPERATURE_COUNTER_V1 : SALT_SAMPLER_GREEDY_V1,
                0.75f, UINT64_C(12345), scenario >= 5 ? 3u : 1u};
            q.logits = logits[route]; q.scratch_logits = scratch[route];
            q.output_ids = ids[route]; q.output_limit = scenario == 7 ? 1u : 3u;
            q.close_token = scenario == 2 || scenario == 4 ? 6 : 5;
            logits[route][2] = 2.0f;
            if (scenario >= 5) {
                logits[route][1] = 1.5f;
                logits[route][3] = 1.0f;
                f[route].next_logits[1] = 1.5f;
                f[route].next_logits[2] = 1.0f;
                q.repetition_penalty = 1.5f;
                q.top_p = scenario == 6 ? 0.75f : 1.0f;
            }
            CHECK(salt_text_scheduler_bind(&controller, &b[route]) == 0 &&
                  salt_text_scheduler_publish(&controller, &r[route]) != 0,
                  "native parity binding or premature publication failed");
            CHECK(salt_text_scheduler_run(&controller, &q, &r[route]) == 0,
                  "native parity run failed");
            CHECK(f[route].token_publications + 1u == r[route].output_count &&
                  f[route].state->position == r[route].source_position + r[route].output_count - 1u &&
                  !f[route].publications &&
                  r[route].sampler_draws == (scenario >= 4 ? r[route].output_count : 0u),
                  "native pending token or sampler draw boundary changed");
            CHECK(salt_text_scheduler_run(&controller, &q, &r[route]) != 0 &&
                  salt_text_scheduler_publish(&controller, &r[route]) != 0,
                  "native duplicate run or early publication admitted");
            CHECK(salt_text_scheduler_finish(&controller, &q, &r[route]) == 0 &&
                  f[route].token_publications == r[route].output_count + (uint32_t)r[route].synthetic_close &&
                  f[route].cells == 3u * f[route].token_publications &&
                  f[route].barriers == f[route].token_publications &&
                  f[route].fences == f[route].token_publications &&
                  f[route].last.backend.final_publications == 1u &&
                  f[route].last.backend.intermediate_host_publications == 0u &&
                  memcmp(f[route].committed_ids, ids[route],
                      r[route].output_count * sizeof ids[route][0]) == 0,
                  "native pending/close bypassed token tape or publication");
            CHECK(r[route].synthetic_close == (q.close_token != r[route].stop_token) &&
                  (!r[route].synthetic_close ||
                   f[route].committed_ids[r[route].output_count] == q.close_token) &&
                  salt_text_scheduler_finish(&controller, &q, &r[route]) != 0 &&
                  salt_text_scheduler_publish(&controller, &r[route]) == 0 &&
                  salt_text_scheduler_publish(&controller, &r[route]) != 0 &&
                  f[route].publications == 1u,
                  "native finalization/publication not exactly once");
        }
        CHECK(memcmp(ids[0], ids[1], sizeof ids[0]) == 0 &&
              memcmp(logits[0], logits[1], sizeof logits[0]) == 0 &&
              memcmp(scratch[0], scratch[1], sizeof scratch[0]) == 0 &&
              memcmp(&r[0], &r[1], sizeof r[0]) == 0 &&
              memcmp(&f[0].stats, &f[1].stats, sizeof f[0].stats) == 0 &&
              f[0].state->position == f[1].state->position &&
              f[0].state->transition_generation == f[1].state->transition_generation &&
              f[0].observations == f[1].observations,
              "verifier/native scheduler or sampler semantics diverged");
    }
    return 0;
}

static int native_schedule_failure_contract(void) {
    for (int failure = NATIVE_FAIL_BEGIN; failure <= NATIVE_FAIL_POSITION; failure++) {
        NativeScheduleContract f;
        SaltTextKvState state = {1u, 0u};
        SaltTextSchedulerBindings b;
        SaltTextTokenEpochController controller = {0};
        SaltTextScheduleRequest q = {0};
        SaltTextScheduleResult r = {0};
        float logits[VOCABULARY] = {0, 0, 2}, scratch[VOCABULARY] = {0};
        int32_t ids[3] = {0};
        CHECK(native_schedule_init(&f, &state, &b) == 0 &&
              salt_text_scheduler_bind(&controller, &b) == 0, "native failure fixture failed");
        q.sampler.abi = SALT_SAMPLER_GREEDY_V1;
        q.logits = logits; q.scratch_logits = scratch; q.output_ids = ids;
        q.output_limit = 3u; q.close_token = 5;
        if (failure == NATIVE_FAIL_PUBLISH) {
            CHECK(salt_text_scheduler_run(&controller, &q, &r) == 0 &&
                  salt_text_scheduler_finish(&controller, &q, &r) == 0,
                  "native publish failure setup failed");
            f.failure = failure;
            CHECK(salt_text_scheduler_publish(&controller, &r) != 0 && f.publications == 1u &&
                  salt_text_scheduler_publish(&controller, &r) != 0 && f.publications == 1u,
                  "native failed publication retried");
        } else {
            f.failure = failure;
            CHECK(salt_text_scheduler_run(&controller, &q, &r) != 0 && !f.publications,
                  "native executor/presentation failure accepted");
            if (failure != NATIVE_FAIL_POSITION)
                CHECK(state.position == 1u && state.transition_generation == 0u &&
                      !f.token_publications, "failed token published canonical state");
            if (failure == NATIVE_FAIL_TOKEN_PUBLISH || failure == NATIVE_FAIL_ABORT)
                CHECK(f.last.status == SALT_TEXT_TOKEN_FATAL, "native fatal result lost");
        }
        CHECK(!controller.schedule_active && controller.phase == SALT_TEXT_TOKEN_EPOCH_FAILED &&
              salt_text_scheduler_publish(&controller, &r) != 0,
              "native scheduler did not fail closed");
    }
    for (int close_failure = 0; close_failure < 2; close_failure++) {
        NativeScheduleContract f;
        SaltTextKvState state = {1u, 0u};
        SaltTextSchedulerBindings b;
        SaltTextTokenEpochController controller = {0};
        SaltTextScheduleRequest q = {0};
        SaltTextScheduleResult r = {0};
        float logits[VOCABULARY] = {0, 0, 2}, scratch[VOCABULARY] = {0};
        int32_t ids[1] = {0};
        CHECK(native_schedule_init(&f, &state, &b) == 0 &&
              salt_text_scheduler_bind(&controller, &b) == 0, "native finish failure setup failed");
        q.sampler.abi = SALT_SAMPLER_GREEDY_V1;
        q.logits = logits; q.scratch_logits = scratch; q.output_ids = ids;
        q.output_limit = 1u; q.close_token = 5;
        CHECK(salt_text_scheduler_run(&controller, &q, &r) == 0 && !f.calls,
              "native cap-one request consumed pending token early");
        f.fail_token = close_failure ? 5 : 2;
        CHECK(salt_text_scheduler_finish(&controller, &q, &r) != 0 &&
              r.close_failed == close_failure && f.token_publications == (uint32_t)close_failure &&
              state.position == 1u + (uint32_t)close_failure && !controller.schedule_active &&
              salt_text_scheduler_publish(&controller, &r) != 0 && !f.publications,
              "native failed pending/close published request");
    }
    return 0;
}

static int native_schedule_request_contract(void) {
    for (int invalid = 0; invalid < 9; invalid++) {
        NativeScheduleContract f;
        SaltTextKvState state = {1u, 0u};
        SaltTextSchedulerBindings b;
        SaltTextTokenEpochController controller = {0};
        SaltTextScheduleRequest q = {0};
        SaltTextScheduleResult r = {0};
        float logits[VOCABULARY] = {0, 0, 2}, scratch[VOCABULARY] = {0};
        int32_t ids[3] = {0};
        CHECK(native_schedule_init(&f, &state, &b) == 0 &&
              salt_text_scheduler_bind(&controller, &b) == 0, "native request fixture failed");
        q.sampler = (SaltSamplerConfig) {SALT_SAMPLER_TEMPERATURE_COUNTER_V1, 1.0f, 1u, 1u};
        q.logits = logits; q.scratch_logits = scratch; q.output_ids = ids;
        q.output_limit = 3u; q.close_token = 5;
        switch (invalid) {
            case 0: q.close_token = VOCABULARY; break;
            case 1: q.repetition_penalty = 1.5f; q.scratch_logits = logits; break;
            case 2: q.repetition_penalty = 0.5f; break;
            case 3: q.top_p = NAN; break;
            case 4: q.history_count = 1u; break;
            case 5: q.history_ids = ids; q.history_count = 2u; break;
            case 6: q.sampler.abi = 0u; break;
            case 7: q.scratch_logits = NULL; break;
            case 8: q.output_limit = 0u; break;
        }
        CHECK(salt_text_scheduler_run(&controller, &q, &r) != 0 &&
              !controller.schedule_active && !f.calls && !f.emissions && !f.publications &&
              state.position == 1u && state.transition_generation == 0u,
              "native malformed request reached executor or publication");
    }
    return 0;
}
#endif

int main(void) {
    CHECK(schedule_request_contract() == 0, "request scheduler contract failed");
    CHECK(schedule_bootstrap_target_contract() == 0,
          "bootstrap-to-target scheduler contract failed");
    CHECK(history_refill_contract() == 0,
          "parent history refill contract failed");
    CHECK(history_target_schedule_contract() == 0,
          "history/local-memory X-TARGET scheduler contract failed");
    CHECK(multi_trajectory_root_mapping_contract() == 0,
          "multi-trajectory NFQ root mapping contract failed");
    CHECK(two_lane_target_sublane_contract() == 0,
          "two-lane X40/W8/Y5 target sublane contract failed");
    CHECK(unified_extent_materializer_contract() == 0,
          "unified B=1/B=j proof-selected materializer contract failed");
    CHECK(projection_refill_contract() == 0,
          "native projection refill contract failed");
    CHECK(area_scan_planner_contract() == 0,
          "area-scan planner contract failed");
    CHECK(wfq_scheduler_contract() == 0,
          "W8-F8-Q4 scheduler contract failed");
    CHECK(nfq_matrix_contract() == 0,
          "N4-F2-Q4 matrix contract failed");
    CHECK(nfq_ready_shape_contract(1u, 8u, 4u) == 0,
          "N1-F8-Q4 ready frontier failed");
    CHECK(nfq_ready_shape_contract(2u, 4u, 4u) == 0,
          "N2-F4-Q4 ready frontier failed");
    CHECK(nfq_ready_shape_contract(4u, 2u, 4u) == 0,
          "N4-F2-Q4 ready frontier failed");
    CHECK(nfq_worker_multiple_contract() == 0,
          "N16-F1-Q2 B=2W ready frontier failed");
    CHECK(nfq_cpu_single_owner_contract() == 0,
          "CPU N1-F1-Q4 owner runway failed");
    CHECK(token_epoch_contract() == 0,
          "cache-first token epoch contract failed");
    CHECK(target_frontier_contract() == 0,
          "target branch frontier contract failed");
    CHECK(host_graph_retained_flow_contract() == 0,
          "retained graph coordinator flow failed");
    CHECK(tensor_engine_ready_graph_contract() == 0,
          "portable-engine dependency-ready TensorOps graph failed");
    CHECK(target_policy_fixed_shape_contract() == 0,
          "configured target policy fixed-shape contract failed");
    CHECK(initialize_package(&package_fixture) == 0,
          "synthetic F32 model package initialization failed");
    CHECK(text_operator_contract_fixture(&package_fixture) == 0,
          "text operator contract fixture failed");
    CHECK(shared_operator_realization_contract(&package_fixture) == 0,
          "shared operator realization fixture failed");
    CHECK(kv_read_view_contract() == 0,
          "committed/tentative KV view fixture failed");
    CHECK(package_fixture.binding_count < MAX_BINDINGS,
          "synthetic binding table overflowed");
    CHECK(exercise_arena_and_dispatch_contracts(&package_fixture) == 0,
          "arena/dispatch contracts failed");
    CHECK(run_scenario(&package_fixture, SALT_TEXT_KV_PRIVATE_ABSOLUTE,
              4u, "shared-prefix + absolute KV") == 0,
          "shared-prefix absolute scenario failed");
    CHECK(run_scenario(&package_fixture, SALT_TEXT_KV_PRIVATE_RING,
              5u, "ring KV") == 0,
          "ring scenario failed");
    f32_tensor_ops.cpu = &f32_decline_backend_ops;
    f32_tensor_ops.compatibility_gpu = &f32_decline_backend_ops;
    CHECK(run_scenario(&package_fixture, SALT_TEXT_KV_PRIVATE_ABSOLUTE,
              4u, "declined area + ordinary row fallback") == 0,
          "declined area fallback scenario failed");
    f32_tensor_ops.cpu = &f32_backend_ops;
    f32_tensor_ops.compatibility_gpu = &f32_backend_ops;
    puts("portable descriptor-driven target-block transaction: PASS");
    return 0;
}
