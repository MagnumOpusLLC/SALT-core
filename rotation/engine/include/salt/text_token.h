#ifndef SALT_TEXT_TOKEN_H
#define SALT_TEXT_TOKEN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SaltTextTokenCellKind {
    SALT_TEXT_TOKEN_EMBED = 0,
    SALT_TEXT_TOKEN_LAYER_ATTENTION_PARENT,
    SALT_TEXT_TOKEN_LAYER_FEED_FORWARD_PARENT,
    SALT_TEXT_TOKEN_INPUT_NORM,
    SALT_TEXT_TOKEN_QKV,
    SALT_TEXT_TOKEN_QK_TRANSFORM_KV_STAGE,
    SALT_TEXT_TOKEN_ATTENTION,
    SALT_TEXT_TOKEN_O_PROJECTION,
    SALT_TEXT_TOKEN_ATTENTION_RESIDUAL,
    SALT_TEXT_TOKEN_POST_ATTN_NORM,
    SALT_TEXT_TOKEN_DENSE_GATE_UP,
    SALT_TEXT_TOKEN_DENSE_ACTIVATION,
    SALT_TEXT_TOKEN_DENSE_DOWN,
    SALT_TEXT_TOKEN_ROUTER,
    SALT_TEXT_TOKEN_TOPK,
    SALT_TEXT_TOKEN_EXPERT_RESOURCE_CHECK,
    SALT_TEXT_TOKEN_EXPERT_GATE_UP,
    SALT_TEXT_TOKEN_EXPERT_ACTIVATION,
    SALT_TEXT_TOKEN_EXPERT_DOWN,
    SALT_TEXT_TOKEN_EXPERT_COMBINE,
    SALT_TEXT_TOKEN_LAYER_RESIDUAL,
    SALT_TEXT_TOKEN_FINAL_NORM,
    SALT_TEXT_TOKEN_HEAD,
    SALT_TEXT_TOKEN_SOFTCAP,
    SALT_TEXT_TOKEN_FINALIZE,
    SALT_TEXT_TOKEN_CELL_KIND_COUNT
} SaltTextTokenCellKind;

typedef enum SaltTextTokenStatus {
    SALT_TEXT_TOKEN_INVALID = 0,
    SALT_TEXT_TOKEN_COMMITTED = 1,
    SALT_TEXT_TOKEN_NEED_RESOURCE = 2,
    SALT_TEXT_TOKEN_EXECUTOR_FAILURE = 3,
    SALT_TEXT_TOKEN_FATAL = 4
} SaltTextTokenStatus;

typedef enum SaltTextTokenCellResult {
    SALT_TEXT_TOKEN_CELL_OK = 0,
    SALT_TEXT_TOKEN_CELL_NEED_RESOURCE = 1,
    SALT_TEXT_TOKEN_CELL_FAILED = -1
} SaltTextTokenCellResult;

enum {
    SALT_TEXT_TOKEN_CELL_BARRIER_AFTER = 1u << 0,
    SALT_TEXT_TOKEN_CELL_TENTATIVE_WRITE = 1u << 1,
    SALT_TEXT_TOKEN_CELL_RESOURCE_CHECK = 1u << 2
};

typedef struct SaltTextTokenCell {
    SaltTextTokenCellKind kind;
    int32_t layer;
    int32_t unit;
    int32_t dependency_cell;
    uint32_t completion_id;
    uint32_t flags;
    uint64_t source_offset;
    uint64_t destination_offset;
    uint64_t destination_bytes;
} SaltTextTokenCell;

typedef struct SaltTextTokenProgramDesc {
    const SaltTextTokenCell *cells;
    uint32_t cell_count;
    uint32_t layer_count;
    uint32_t maximum_internal_barriers;
} SaltTextTokenProgramDesc;

typedef struct SaltTextTokenProgram {
    const SaltTextTokenCell *cells;
    uint32_t cell_count;
    uint32_t layer_count;
    uint32_t maximum_internal_barriers;
    uint64_t canonical_write_bytes;
    int ready;
} SaltTextTokenProgram;

typedef struct SaltTextTokenExecution {
    uint64_t transition_generation;
    uint32_t source_position;
    int32_t input_token_id;
} SaltTextTokenExecution;

typedef struct SaltTextTokenBackendStats {
    uint32_t caller_submissions;
    uint32_t cells_completed;
    uint32_t internal_dependency_barriers;
    uint32_t completion_fences;
    uint32_t intermediate_host_publications;
    uint32_t final_publications;
    uint32_t resource_retries;
    uint64_t tentative_write_bytes;
} SaltTextTokenBackendStats;

typedef struct SaltTextTokenResult {
    SaltTextTokenStatus status;
    uint32_t source_position;
    uint32_t result_position;
    int32_t input_token_id;
    uint64_t transition_generation;
    int32_t resource_cell;
    SaltTextTokenBackendStats backend;
} SaltTextTokenResult;

typedef struct SaltTextTokenExecutorOps {
    int (*begin)(void *context, const SaltTextTokenProgram *program,
                 const SaltTextTokenExecution *execution);
    SaltTextTokenCellResult (*execute_cell)(
        void *context, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index);
    int (*dependency_barrier)(
        void *context, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        const SaltTextTokenCell *cell, uint32_t cell_index);
    int (*finish)(void *context, const SaltTextTokenProgram *program,
                  const SaltTextTokenExecution *execution);
    int (*abort)(void *context, const SaltTextTokenProgram *program,
                 const SaltTextTokenExecution *execution,
                 SaltTextTokenStatus status, uint32_t completed_cells);
    int (*publish)(void *context, const SaltTextTokenProgram *program,
                   const SaltTextTokenExecution *execution,
                   uint32_t result_position);
    /* Optional engine-scheduled resource lookahead. Called before the current
     * layer begins so a model adapter may advise only the next declared layer.
     * Returns 0 prepared/no-op, 1 refused, or -1 on lifecycle failure. */
    int (*prepare_next_layer)(
        void *context, const SaltTextTokenProgram *program,
        const SaltTextTokenExecution *execution,
        uint32_t current_layer, uint32_t next_layer);
} SaltTextTokenExecutorOps;

typedef struct SaltTextTokenExecutor {
    const SaltTextTokenExecutorOps *ops;
    void *context;
    int layer_lookahead;
} SaltTextTokenExecutor;

int salt_text_token_program_arena_requirement(
    const SaltTextTokenProgramDesc *descriptor, size_t *bytes_out);
int salt_text_token_program_compile(
    SaltTextTokenProgram *program,
    const SaltTextTokenProgramDesc *descriptor,
    void *startup_arena, size_t startup_arena_bytes);
int salt_text_token_execute(
    const SaltTextTokenProgram *program,
    SaltTextTokenExecutor *executor,
    const SaltTextTokenExecution *execution,
    SaltTextTokenResult *result);

#ifdef __cplusplus
}
#endif

#endif /* SALT_TEXT_TOKEN_H */
