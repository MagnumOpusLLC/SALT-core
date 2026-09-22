#ifndef SALT_TEXT_VERIFY_H
#define SALT_TEXT_VERIFY_H

#include "salt/area_scan.h"
#include "salt/model.h"
#include "salt/tensorops.h"
#include "salt/sampling.h"
#include "salt/dpr_stats.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SaltTextVerifyStatus {
    SALT_TEXT_VERIFY_INVALID = 0,
    SALT_TEXT_VERIFY_COMMITTED = 1,
    SALT_TEXT_VERIFY_EXECUTOR_FAILURE = 2,
    SALT_TEXT_VERIFY_FATAL = 3
} SaltTextVerifyStatus;

struct SaltTextGpuProgramOps;
struct SaltTextGpuProgramRequirements;
struct SaltGpuSharedBuffer;
struct SaltTextExecutionView;
struct SaltTextTargetFrontier;

/* Compatibility names for the text program. TensorOps itself is engine-wide;
 * model packages provide immutable SaltTensorDesc specs, while executors select
 * CPU/Metal/CUDA realization without mutating the program. */
typedef SaltTensorHostOps SaltTextTensorBackendOps;
typedef SaltTensorHostBatch SaltTextTensorHostBatch;
typedef SaltTensorHostRealizations SaltTextTensorOps;
typedef SaltTensorDesc SaltTextTensorDesc;

typedef struct SaltTextExpertDesc {
    uint32_t expert_id;
    SaltTextTensorDesc gate;
    SaltTextTensorDesc up;
    SaltTextTensorDesc down;
} SaltTextExpertDesc;

/* tensor_table addresses one fixed tensor (tensor_count == 1, no selector),
 * or one member across an expert descriptor table. A selector is an int32
 * tensor-table index stored in the canonical device arena. Packages inspect
 * only their opaque tensor bindings; the engine never switches on format. */
typedef SaltTensorDeviceBatch SaltTextTensorDeviceBatch;
#define SALT_TEXT_DEVICE_NO_OFFSET SALT_TENSOR_NO_OFFSET

typedef enum SaltTextKvPrivateMode {
    SALT_TEXT_KV_PRIVATE_ABSOLUTE = 1,
    SALT_TEXT_KV_PRIVATE_RING = 2
} SaltTextKvPrivateMode;

/* Mutable KV state referenced by an immutable row descriptor. Model packages
 * may attach or replace an authenticated shared prefix without rebuilding the
 * compiled target program or changing the descriptor itself. */
typedef struct SaltTextKvSharedPrefixState {
    const float *rows;
    size_t float_capacity;
    size_t row_stride;
    uint32_t row_count;
} SaltTextKvSharedPrefixState;

/* A logical row below shared_prefix_rows is read from shared_prefix. All other
 * rows use private storage. Absolute private rows use position -
 * private_position_base; ring rows use that logical index modulo capacity.
 * shared_prefix_state, when non-NULL, is the live state authority and the
 * direct shared_prefix fields must remain zero. */
typedef struct SaltTextKvRowsDesc {
    SaltTextKvPrivateMode private_mode;
    float *private_rows;
    size_t private_float_capacity;
    size_t private_row_stride;
    uint32_t private_row_capacity;
    uint32_t private_position_base;
    const float *shared_prefix;
    size_t shared_prefix_float_capacity;
    size_t shared_prefix_row_stride;
    uint32_t shared_prefix_rows;
    const SaltTextKvSharedPrefixState *shared_prefix_state;
} SaltTextKvRowsDesc;

/* One ordered read view over the session-owned committed rows and fixed
 * tentative future seats. parent_rows/depths are both NULL for a linear block;
 * when present they describe the already validated target frontier. The view
 * owns no storage and grants no commit/publication authority. */
typedef struct SaltTextKvReadView {
    const SaltTextKvRowsDesc *committed_rows;
    const float *tentative_rows;
    size_t tentative_float_capacity;
    size_t tentative_row_stride;
    uint32_t tentative_row_count;
    uint32_t source_position;
    const uint32_t *parent_rows;
    const uint32_t *depths;
} SaltTextKvReadView;

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
    uint32_t width);

const float *salt_text_kv_read_view_row(
    const SaltTextKvReadView *view,
    uint32_t current_row,
    uint32_t position);

typedef struct SaltTextKvLayerDesc {
    SaltTextKvRowsDesc keys;
    SaltTextKvRowsDesc values;
} SaltTextKvLayerDesc;

typedef struct SaltTextKvState {
    uint32_t position;
    uint64_t transition_generation;
} SaltTextKvState;

typedef struct SaltTextLayerExecDesc {
    const SaltTextLayerPlan *plan;
    const SaltTextKvLayerDesc *kv;

    SaltTextTensorDesc q;
    SaltTextTensorDesc k;
    SaltTextTensorDesc v;
    SaltTextTensorDesc o;

    SaltTextTensorDesc dense_gate;
    SaltTextTensorDesc dense_up;
    SaltTextTensorDesc dense_down;
    SaltTextTensorDesc router;

    SaltTextTensorDesc pre_attention_norm;
    SaltTextTensorDesc q_norm;
    SaltTextTensorDesc k_norm;
    SaltTextTensorDesc post_attention_norm;
    SaltTextTensorDesc pre_ffn_norm_1;
    SaltTextTensorDesc pre_ffn_norm_2;
    SaltTextTensorDesc post_ffn_norm_1;
    SaltTextTensorDesc post_ffn_norm_2;
    SaltTextTensorDesc post_ffn_norm;
    SaltTextTensorDesc router_scale;
    SaltTextTensorDesc per_expert_scale;
    SaltTextTensorDesc layer_scalar;

    const SaltTextExpertDesc *experts;
    uint32_t expert_count;
} SaltTextLayerExecDesc;

typedef enum SaltTextExecutionClass {
    SALT_TEXT_EXECUTION_CPU_ONLY = 1,
    SALT_TEXT_EXECUTION_GPU_ONLY = 2,
    SALT_TEXT_EXECUTION_MIXED = 3
} SaltTextExecutionClass;

enum {
    /* Existing startup-owned NFQ storage admits at most eight local Q checks
     * for every target row. This is an ABI capacity, not an N*F*Q policy cap. */
    SALT_TEXT_NFQ_MAX_QUEUE = 8u,
    SALT_TEXT_NFQ_MAX_CHECKS = SALT_DPR_MAX_HORIZON * SALT_TEXT_NFQ_MAX_QUEUE
};

/* One immutable startup policy for TARGET execution. Recipe values remain
 * model/platform data, while portable C owns their validation and execution
 * meaning. Callers provide candidate IDs only; they cannot redefine NFQ N/F/Q,
 * X-TARGET width, or the admitted executor after model startup. */
typedef struct SaltTextTargetPolicy {
    SaltTextExecutionClass execution_class;
    uint32_t worker_budget;
    uint32_t sequence_tiles; /* NFQ N */
    uint32_t target_rows;    /* X-TARGET */
    uint32_t route_count;
    uint32_t queue_length;
    /* Complete bounded NFQ checking surface: N x F x Q. X-TARGET rows are
     * independently bounded by target_rows; Q never creates model rows. */
    uint32_t candidate_count;
    uint32_t n_parallel;
    uint32_t matrix_flow;
    uint32_t cpu_graph;
    /* Runtime causal-width gate. A request with fewer pre-existing live-KV
     * rows than kv_warmup_rows executes X=1. Warm requests are capped at
     * min(target_rows, warm_target_rows) until a later policy promotes them. */
    uint32_t kv_warmup_rows;
    uint32_t warm_target_rows;
    int ready;
} SaltTextTargetPolicy;

typedef struct SaltTextDispatchPolicy {
    SaltTextExecutionClass execution_class;
    /* For MIXED cells CPU owns [0, cpu_*), GPU owns the remaining complete
     * rows/jobs. Runtime candidate counts clamp both slices without overlap. */
    uint32_t cpu_rows;
    uint32_t cpu_jobs;
} SaltTextDispatchPolicy;

typedef struct SaltTextModelExecDesc {
    const SaltModelDesc *model;
    const SaltTextLayerExecDesc *layers;
    uint32_t layer_count;
    const SaltTensorResourceSpec *tensor_resources;
    uint32_t tensor_resource_count;
    uint32_t vocabulary;
    uint32_t maximum_context;
    float norm_epsilon;
    float embedding_scale;
    SaltTextTensorDesc embedding;
    SaltTextTensorDesc final_norm;
    SaltTextTensorDesc output_head;
} SaltTextModelExecDesc;

typedef enum SaltTextExecutionUnit {
    SALT_TEXT_EXECUTION_ROWS = 1,
    SALT_TEXT_EXECUTION_JOBS = 2
} SaltTextExecutionUnit;

typedef enum SaltTextExecutionCellKind {
    SALT_TEXT_CELL_EMBEDDING = 1,
    SALT_TEXT_CELL_PRE_ATTENTION_NORM,
    SALT_TEXT_CELL_QUERY_PROJECTION,
    SALT_TEXT_CELL_KEY_PROJECTION,
    SALT_TEXT_CELL_VALUE_PROJECTION,
    SALT_TEXT_CELL_ATTENTION_TRANSFORM,
    SALT_TEXT_CELL_ATTENTION_BODY,
    SALT_TEXT_CELL_OUTPUT_PROJECTION,
    SALT_TEXT_CELL_ATTENTION_COMBINE,
    SALT_TEXT_CELL_DENSE_NORM,
    SALT_TEXT_CELL_DENSE_GATE,
    SALT_TEXT_CELL_DENSE_UP,
    SALT_TEXT_CELL_DENSE_ACTIVATION,
    SALT_TEXT_CELL_DENSE_DOWN,
    SALT_TEXT_CELL_ROUTER_INPUT,
    SALT_TEXT_CELL_ROUTER_PROJECTION,
    SALT_TEXT_CELL_ROUTER_TOPK,
    SALT_TEXT_CELL_ROUTED_NORM,
    SALT_TEXT_CELL_EXPERT_GATE,
    SALT_TEXT_CELL_EXPERT_UP,
    SALT_TEXT_CELL_EXPERT_ACTIVATION,
    SALT_TEXT_CELL_EXPERT_DOWN,
    SALT_TEXT_CELL_EXPERT_REDUCTION,
    SALT_TEXT_CELL_FFN_COMBINE,
    SALT_TEXT_CELL_FINAL_NORM,
    SALT_TEXT_CELL_FINAL_HEAD,
    SALT_TEXT_CELL_LOGIT_SOFTCAP
} SaltTextExecutionCellKind;

#define SALT_TEXT_EXECUTION_CELL_KIND_COUNT \
    ((uint32_t)SALT_TEXT_CELL_LOGIT_SOFTCAP + 1u)

typedef struct SaltTextExecutionSlice {
    uint32_t first;
    uint32_t count;
} SaltTextExecutionSlice;

enum {
    SALT_TEXT_EXECUTION_CELL_FINAL_PUBLICATION = 1u
};

typedef struct SaltTextExecutionCell {
    SaltTextExecutionCellKind kind;
    SaltTextExecutionUnit unit;
    uint32_t layer;
    uint32_t dependency_epoch;
    uint32_t completion_epoch;
    uint32_t logical_capacity;
    size_t destination_offset;
    size_t destination_stride;
    uint32_t flags;
} SaltTextExecutionCell;

typedef struct SaltTextDispatchPlan {
    const SaltTextExecutionCell *cells;
    uint32_t cell_count;
    uint32_t engine_command_count;
    uint32_t final_dependency_epoch;
} SaltTextDispatchPlan;

typedef struct SaltTextExecutionAssignment {
    SaltTextExecutionSlice cpu;
    SaltTextExecutionSlice gpu;
    /* GPU-only traversal metadata lives in this existing assignment overlay.
     * A nonzero count marks the first cell of a contiguous extent; extents
     * split only at engine-owned expert acquire/release boundaries. */
    uint32_t gpu_traversal_count;
    uint32_t gpu_traversal_barriers;
} SaltTextExecutionAssignment;

/* Architecture-owned startup overlay for one immutable, mode-neutral program.
 * Different executors may assign the same cells differently without compiling
 * a second model program. */
typedef struct SaltTextExecutorPlan {
    const struct SaltTextVerifyProgram *program;
    const SaltTextExecutionAssignment *assignments;
    uint32_t assignment_count;
    SaltTextExecutionClass execution_class;
} SaltTextExecutorPlan;

/* Every destination is a backend-neutral byte offset. An executor maps these
 * offsets into its own fixed arena; the portable program holds no CPU/GPU
 * workspace pointers or executor context. */
typedef struct SaltTextCanonicalLayout {
    size_t total_bytes;
    size_t candidate_token_ids;
    size_t target_parent_rows;
    size_t target_depths;
    size_t state_a;
    size_t state_b;
    size_t normalized;
    size_t queries;
    size_t keys;
    size_t values;
    size_t attention_output;
    size_t branch;
    size_t dense_gate;
    size_t dense_up;
    size_t dense_chain;
    size_t dense_output;
    size_t router_input;
    size_t router_logits;
    size_t selected_experts;
    size_t selected_weights;
    size_t grouped_to_canonical;
    size_t canonical_to_grouped;
    size_t routed_input;
    size_t routed_gate;
    size_t routed_up;
    size_t routed_chain;
    size_t routed_output_jobs;
    size_t routed_output;
    size_t combine_a;
    size_t combine_b;
    size_t final_state;
    size_t position_logits;
    size_t attention_scores;
    size_t tensor_row;
    size_t tentative_kv;
    uint32_t maximum_query_width;
    uint32_t maximum_kv_width;
    uint32_t maximum_dense_width;
    uint32_t maximum_experts;
    uint32_t maximum_topk;
    uint32_t maximum_expert_width;
    uint32_t maximum_tensor_row;
} SaltTextCanonicalLayout;

typedef struct SaltTextCompiledLayer {
    const SaltTextLayerExecDesc *descriptor;
    uint32_t query_width;
    uint32_t kv_width;
    size_t tentative_key_offset;
    size_t tentative_value_offset;
    uint32_t first_cell;
    uint32_t cell_count;
} SaltTextCompiledLayer;

typedef struct SaltTextVerifyProgram {
    const SaltTextModelExecDesc *descriptor;
    SaltTextKvState *kv_state;
    const SaltTextTargetPolicy *target_policy;
    const SaltTextCompiledLayer *layers;
    uint32_t layer_count;
    uint32_t hidden;
    uint32_t vocabulary;
    uint32_t maximum_candidates;
    uint32_t maximum_context;
    SaltTextCanonicalLayout layout;
    SaltTextDispatchPlan dispatch;
    size_t tentative_kv_bytes;
    int ready;
} SaltTextVerifyProgram;

int salt_text_target_policy_compile_environment(
    SaltTextTargetPolicy *policy,
    SaltTextExecutionClass requested_execution_class,
    uint32_t maximum_candidates);
int salt_text_target_policy_active_shape(
    const SaltTextTargetPolicy *policy, uint32_t candidate_count,
    uint32_t *active_sequence_tiles, uint32_t *active_route_count);
int salt_text_target_policy_effective_x(
    const SaltTextTargetPolicy *policy, uint32_t prior_kv_rows,
    uint32_t available_rows, uint32_t *effective_x);
int salt_text_verify_program_bind_target_policy(
    SaltTextVerifyProgram *program, const SaltTextTargetPolicy *policy);

typedef struct SaltTextVerifyBackendStats {
    uint32_t engine_submissions;
    uint32_t completion_fences;
    uint32_t intermediate_host_publications;
    uint32_t internal_dependency_barriers;
    uint32_t projection_dispatches;
    uint32_t expert_gate_up_dispatches;
    uint32_t expert_down_dispatches;
    uint32_t area_m1_dispatches;
    uint32_t area_mk_dispatches;
    uint32_t area_mn_dispatches;
    uint32_t area_peak_active_workers;
    uint64_t area_output_row_tiles;
    uint64_t area_candidate_output_tiles;
    uint32_t area_matrix_parallel_dispatches;
    uint64_t initial_transfer_bytes;
    uint64_t final_logits_transfer_bytes;
    uint64_t final_kv_publish_bytes;
    uint64_t tentative_scrub_bytes;
    uint64_t canonical_clear_bytes;
    uint32_t production_frontier_sessions;
    uint32_t production_frontier_tasks_queued;
    uint32_t production_frontier_tasks_executed;
    uint32_t production_frontier_helper_executions;
    uint32_t production_frontier_worker_reassignments;
    uint32_t production_frontier_queued_cancellations;
    uint32_t production_frontier_cancel_requested_completions;
    uint32_t production_frontier_stale_rejections;
    uint32_t production_frontier_peak_ready_depth;
    uint32_t cpu_matrix_pool_phases;
    uint32_t cpu_qkv_waves;
    uint32_t cpu_dense_gate_up_waves;
    uint32_t cpu_expert_gate_up_waves;
    uint32_t cpu_expert_down_waves;
    uint32_t cpu_graph_sessions;
    uint32_t cpu_graph_nodes;
    uint32_t cpu_graph_spans;
    uint32_t cpu_graph_serial_spans;
    uint32_t cpu_graph_parallel_spans;
    uint32_t cpu_graph_barriers;
    uint64_t cpu_target_clear_ns;
    uint64_t cpu_target_setup_ns;
    uint64_t cpu_target_interpret_ns;
    uint64_t cpu_target_cell_ns[SALT_TEXT_EXECUTION_CELL_KIND_COUNT];
    uint32_t cpu_target_cell_calls[SALT_TEXT_EXECUTION_CELL_KIND_COUNT];
    uint32_t backend_template_count;
    uint32_t backend_template_reuses;
    uint32_t backend_dynamic_patches;
    uint32_t backend_selected_job_capacity;
    uint32_t backend_selected_jobs;
    uint32_t backend_graph_count;
    uint32_t backend_graph_launches;
    uint32_t backend_graph_parameter_patches;
    uint32_t backend_physical_kernel_nodes;
    uint32_t backend_host_kernel_launch_calls;
} SaltTextVerifyBackendStats;

#define SALT_TEXT_MAX_TOUCHED_SPANS 128u

typedef struct SaltTextTouchedSpan {
    size_t offset;
    size_t bytes;
} SaltTextTouchedSpan;

int salt_text_attention_score_rows(
    const SaltTextVerifyProgram *program, uint32_t source_position,
    uint32_t maximum_depth, uint32_t *score_rows);
int salt_text_touched_span_plan(
    const SaltTextVerifyProgram *program, uint32_t candidate_count,
    uint32_t attention_score_rows,
    SaltTextTouchedSpan *spans, uint32_t span_capacity,
    uint32_t *span_count, uint64_t *total_bytes);
int salt_text_touched_span_plan_outputs(
    const SaltTextVerifyProgram *program, uint32_t candidate_count,
    uint32_t attention_score_rows, uint32_t output_rows,
    SaltTextTouchedSpan *spans, uint32_t span_capacity,
    uint32_t *span_count, uint64_t *total_bytes);
int salt_text_touched_span_clear(
    unsigned char *base, size_t capacity,
    const SaltTextTouchedSpan *spans, uint32_t span_count,
    uint64_t *cleared_bytes);

typedef struct SaltTextGpuProgramRequirements {
    size_t backend_state_bytes;
    size_t command_bytes;
    uint32_t maximum_commands;
    uint32_t flags;
} SaltTextGpuProgramRequirements;

enum {
    /* The backend proves expert consumption at the next NEED_RESOURCE or final
     * finish boundary, allowing the engine to retain one layer's leases across
     * reduction/combine without creating another completion. */
    SALT_TEXT_GPU_DEFER_EXPERT_RELEASE = 1u << 0
};

enum {
    SALT_TEXT_GPU_DEPENDENCY_READY = 0,
    SALT_TEXT_GPU_NEED_RESOURCE = 1,
    SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS = 1024
};

typedef struct SaltTextExpertResourceOps {
    int (*acquire)(void *opaque, uint32_t layer,
                   const int32_t *experts, uint32_t count, int32_t *slots);
    int (*release)(void *opaque, uint32_t layer,
                   const int32_t *experts, const int32_t *slots,
                   uint32_t count);
} SaltTextExpertResourceOps;

/* Backend realization of the immutable C99 program. The engine owns traversal
 * extents, dependency legality, resource/lease boundaries, completion
 * accounting, resolve, and publication. A backend may directly traverse only
 * the exact contiguous extent and assignments admitted by the engine. */
typedef struct SaltTextGpuProgramOps {
    int (*requirements)(const SaltTextVerifyProgram *program,
                        const SaltTextDispatchPolicy *policy,
                        SaltTextGpuProgramRequirements *requirements,
                        size_t requirements_size);
    int (*prepare)(const SaltTextVerifyProgram *program,
                   const SaltTextExecutorPlan *plan,
                   struct SaltGpuSharedBuffer **canonical_out,
                   void *backend_state, size_t backend_state_bytes,
                   void *command, size_t command_bytes);
    int (*begin)(void *backend_state, void *command,
                 uint64_t generation, uint32_t source_position,
                 const int32_t *input_token_ids, uint32_t input_count);
    int (*encode_cell)(void *backend_state, void *command,
                       const SaltTextVerifyProgram *program,
                       const SaltTextExecutionCell *cell,
                       const SaltTextExecutionAssignment *assignment,
                       uint32_t input_count);
    int (*encode_extent)(void *backend_state, void *command,
                         const SaltTextVerifyProgram *program,
                         const SaltTextExecutorPlan *plan,
                         uint32_t first_cell, uint32_t cell_count,
                         uint32_t input_count,
                         uint32_t *encoded_cells);
    int (*dependency_barrier)(void *backend_state, void *command,
                              uint32_t dependency_epoch,
                              uint32_t completion_epoch);
    int (*resource_request)(void *backend_state, void *command,
                            uint32_t *layer, int32_t *experts,
                            uint32_t expert_capacity, uint32_t *expert_count);
    int (*resource_resume)(void *backend_state, void *command,
                           uint32_t layer, const int32_t *experts,
                           const int32_t *slots, uint32_t expert_count);
    int (*submit)(void *backend_state, void *command);
    int (*finish)(void *backend_state, void *command,
                  struct SaltTextExecutionView *view,
                  SaltTextVerifyBackendStats *stats);
    int (*resolve)(void *backend_state, void *command,
                   uint32_t committed_count, uint32_t input_count,
                   uint64_t *scrubbed_bytes);
    int (*scrub)(void *backend_state, void *command,
                 uint64_t *scrubbed_bytes);
    int (*destroy)(void *backend_state, void *command);
    int (*begin_frontier)(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids,
        const uint32_t *parent_rows, const uint32_t *depths,
        uint32_t input_count);
    int (*begin_authoritative)(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count);
    int (*begin_authoritative_output)(
        void *backend_state, void *command,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows);
} SaltTextGpuProgramOps;

typedef struct SaltTextExecutionView {
    unsigned char *canonical_base;
    size_t canonical_bytes;
    uint32_t backend_published_kv;
    uint32_t reserved;
} SaltTextExecutionView;

typedef struct SaltTextVerifyExecutorOps {
    int (*submit)(void *context, const SaltTextVerifyProgram *program,
                  uint64_t generation, uint32_t source_position,
                  const int32_t *candidate_token_ids,
                  uint32_t candidate_count);
    int (*submit_frontier)(void *context,
                           const SaltTextVerifyProgram *program,
                           uint32_t source_position,
                           const struct SaltTextTargetFrontier *frontier);
    int (*finish)(void *context, const SaltTextVerifyProgram *program,
                  SaltTextExecutionView *view,
                  SaltTextVerifyBackendStats *stats);
    int (*resolve)(void *context, const SaltTextVerifyProgram *program,
                   uint64_t generation, uint32_t source_position,
                   uint32_t accepted_count, uint32_t candidate_count,
                   uint64_t *scrubbed_bytes);
    int (*scrub)(void *context, const SaltTextVerifyProgram *program,
                 uint64_t generation, uint64_t *scrubbed_bytes);
    int (*submit_authoritative)(
        void *context, const SaltTextVerifyProgram *program,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count);
    int (*submit_authoritative_output)(
        void *context, const SaltTextVerifyProgram *program,
        uint64_t generation, uint32_t source_position,
        const int32_t *input_token_ids, uint32_t input_count,
        uint32_t output_rows);
} SaltTextVerifyExecutorOps;

typedef struct SaltTextVerifyExecutor {
    const SaltTextVerifyProgram *program;
    const SaltTextExecutorPlan *plan;
    const SaltTextVerifyExecutorOps *ops;
    void *context;
} SaltTextVerifyExecutor;

typedef struct SaltTextTargetBlock {
    uint64_t transition_generation;
    uint32_t source_position;
    const int32_t *candidate_token_ids;
    uint32_t candidate_count;
    const float *parent_logits;
} SaltTextTargetBlock;

#define SALT_TEXT_TARGET_NO_PARENT UINT32_MAX

typedef struct SaltTextTargetNode {
    int32_t token_id;
    uint32_t node_id;
    uint32_t parent_index;
    uint32_t depth;
    uint32_t tentative_state_slot;
    uint32_t flags;
} SaltTextTargetNode;

typedef struct SaltTextTargetFrontier {
    const SaltTextTargetNode *nodes;
    uint32_t node_count;
    uint64_t generation;
} SaltTextTargetFrontier;

typedef struct SaltTextTargetFrontierBlock {
    uint32_t source_position;
    SaltTextTargetFrontier frontier;
    const float *parent_logits;
    SaltAreaFrontier *area_frontier;
    uint32_t force_full_frontier;
} SaltTextTargetFrontierBlock;

typedef struct SaltTextTargetCascadeBlock {
    uint64_t first_transition_generation;
    uint32_t source_position;
    const int32_t *seed_token_ids;
    uint32_t tile_count;
    uint32_t frontier_width;
    const float *parent_logits;
    SaltTextTargetNode *node_scratch;
    uint32_t node_scratch_capacity;
} SaltTextTargetCascadeBlock;

typedef struct SaltTextTargetRouteBlock {
    uint64_t transition_generation;
    uint32_t source_position;
    const int32_t *route_token_ids;
    uint32_t sequence_tiles;
    uint32_t route_count;
    const float *parent_logits;
    SaltAreaWfqRuntime *wfq;
    int (*wfq_execute)(
        void *context, SaltAreaWfqRuntime *runtime,
        SaltAreaWfqItemExecute item_execute, void *item_context,
        SaltAreaWfqResult *result);
    void *wfq_context;
} SaltTextTargetRouteBlock;

/* Borrowed views of the existing startup-owned NFQ storage. Binding these
 * pointers creates no arena, pool, scheduler, or candidate source. */
typedef struct SaltTextTargetNfqBlock {
    SaltTextTargetRouteBlock route;
    SaltAreaNfqPlan plan;
    SaltAreaNfqMatrix *matrix;
    SaltTextTargetNode *nodes;
    uint32_t node_capacity;
    uint32_t *route_ids;
    uint32_t route_capacity;
    uint32_t *ready_items;
    uint32_t ready_capacity;
} SaltTextTargetNfqBlock;

int salt_text_target_frontier_validate(
    const SaltTextTargetFrontier *frontier, uint32_t state_slot_capacity);
int salt_text_target_frontier_winning_path(
    const SaltTextTargetFrontier *frontier, uint32_t winning_node_index,
    uint32_t *path_indices, uint32_t path_capacity,
    uint32_t *path_count);
int salt_text_target_frontier_build_area_tasks(
    const SaltTextTargetFrontier *target,
    SaltAreaFrontier *area, uint32_t phase, int32_t expert_id,
    uint32_t output_rows, uint32_t requested_workers);

typedef struct SaltTextVerifyResult {
    SaltTextVerifyStatus status;
    uint32_t accepted_count;
    uint32_t produced_count;
    uint32_t committed_count;
    int32_t pending_token_id;
    uint32_t result_position;
    uint64_t transition_generation;
    uint32_t winning_node_index;
    uint32_t winning_node_id;
    /* Borrowed target distribution that produced pending_token_id. It remains
     * valid until the next submit on this executor. A zero-work seed rejection
     * may alias the caller-owned parent_logits. */
    const float *pending_logits;
    /* Borrowed proposals, never state authority; valid until the next submit. */
    const float *projection_logits;
    uint32_t projection_rows;
    uint32_t pending_logits_count;
    SaltTextVerifyBackendStats backend;
} SaltTextVerifyResult;

/* Metadata only over existing completed TARGET logits; owns no allocation. */
typedef struct SaltTextProjectionWindow {
    const float *logits;
    uint64_t generation;
    uint32_t position;
    uint32_t sequence_tiles;
    uint32_t route_count;
    uint32_t accepted_count;
    uint32_t winning_route;
} SaltTextProjectionWindow;

/* Extract candidate IDs before any executor submission overwrites the window.
 * The parent selects route zero's root. Prior per-row predictions supply only
 * guesses for remaining seats; uncovered future seats are never fabricated.
 * No model execution, sampler counter, KV, or committed-state mutation occurs.
 * Returns 0 with proposals, 1 when no sufficient completed window is available,
 * and -1 for stale/malformed input. A miss permits ordinary native B1; errors
 * fail closed. Never silently read stale projections.
 * The bounded limit matches the existing target candidate ABI. */
int salt_text_projection_refill_candidates(
    const SaltTextVerifyProgram *program,
    const SaltTextProjectionWindow *previous,
    const float *parent_logits, uint32_t sequence_tiles, uint32_t route_count,
    int32_t *candidate_ids, uint32_t capacity, uint32_t *reused_rows);

/* Parent cache route: the committed token history (prompt/session tokens,
 * then this run's emitted outputs) is the parent-owned trajectory memory.
 * The root is the parent's own greedy selection. The longest, most recent
 * earlier occurrence of the suffix ending in that root supplies the tokens
 * that followed it as one parallel proposal for tiles 1..sequence_tiles-1.
 * Zero model work, no allocation, no state mutation. Returns the proposed
 * tile count including the root (1 means no usable match), or -1 on invalid
 * input. TARGET alone validates and commits. */
enum {
    SALT_TEXT_HISTORY_MATCH_MAX = 16u
};
int salt_text_history_refill_candidates(
    const int32_t *history_ids, uint32_t history_count,
    const int32_t *output_ids, uint32_t output_count,
    int32_t root, uint32_t vocabulary, uint32_t sequence_tiles,
    int32_t *candidate_ids, uint32_t capacity, uint32_t *matched_length);

enum {
    SALT_TEXT_PROPOSAL_SOURCE_NONE = 0,
    SALT_TEXT_PROPOSAL_SOURCE_HISTORY = 1,
    SALT_TEXT_PROPOSAL_SOURCE_PROJECTION = 2
};

typedef enum SaltTextTokenEpochPhase {
    SALT_TEXT_TOKEN_EPOCH_INVALID = 0,
    SALT_TEXT_TOKEN_EPOCH_BOUNDARY = 1,
    SALT_TEXT_TOKEN_EPOCH_CACHE_SCAN = 2,
    SALT_TEXT_TOKEN_EPOCH_COLD_SEARCH = 3,
    SALT_TEXT_TOKEN_EPOCH_RESOLVED = 4,
    SALT_TEXT_TOKEN_EPOCH_FAILED = 5
} SaltTextTokenEpochPhase;

typedef enum SaltTextTokenEpochPath {
    SALT_TEXT_TOKEN_PATH_INVALID = 0,
    SALT_TEXT_TOKEN_PATH_EXACT_HIT = 1,
    SALT_TEXT_TOKEN_PATH_COLD_SEARCH = 2
} SaltTextTokenEpochPath;

typedef struct SaltTextTokenEpochController {
    SaltTextTokenEpochPhase phase;
    uint64_t last_epoch_generation;
    uint64_t active_epoch_generation;
    uint64_t parent_generation;
    uint32_t parent_position;
    uint64_t lookup_count;
    uint64_t exact_hit_count;
    uint64_t exact_miss_count;
    uint64_t cold_search_count;
    uint64_t duplicate_admission_count;
    int active;
    int ready;
    const struct SaltTextSchedulerBindings *scheduler;
    int schedule_active;
    int schedule_closed;
    int schedule_published;
    /* One committed ordinary transition may bootstrap a request when no
     * conditioned candidate state exists. Subsequent greedy misses admit only
     * the current parent-logit root through TARGET; they never repeat B1 or
     * fabricate uncovered future seats. Reset with each scheduler run. */
    uint32_t schedule_bootstraps;
    uint32_t schedule_source;
    uint32_t schedule_count;
    int32_t schedule_stop;
    /* Borrowed per-run committed history views for the parent cache route.
     * Set by the scheduler; never owned, copied, or mutated here. */
    const int32_t *schedule_history_ids;
    uint32_t schedule_history_count;
    const int32_t *schedule_output_ids;
    uint32_t schedule_output_count;
    uint32_t schedule_prior_kv_rows;
} SaltTextTokenEpochController;

typedef int (*SaltTextTokenExactLookupCommit)(
    void *context, uint64_t epoch_generation,
    uint64_t parent_generation, uint32_t parent_position,
    SaltTextVerifyResult *result);
typedef int (*SaltTextTokenColdSearch)(
    void *context, const SaltTextTargetRouteBlock *route,
    SaltTextVerifyResult *result);

typedef struct SaltTextTokenEpochRequest {
    uint64_t epoch_generation;
    uint64_t parent_generation;
    uint32_t parent_position;
    const SaltTextTargetRouteBlock *route;
    SaltTextTokenExactLookupCommit exact_lookup_commit;
    void *exact_context;
    SaltTextTokenColdSearch cold_search;
    void *cold_context;
} SaltTextTokenEpochRequest;

typedef struct SaltTextTokenEpochResult {
    SaltTextTokenEpochPath path;
    SaltTextVerifyResult target;
} SaltTextTokenEpochResult;

/* Borrowed model views and primitive bindings. These callbacks must not select
 * a serving path, construct proposals, or run an output-generation loop. */
typedef struct SaltTextGenerationBinding {
    const SaltTextVerifyProgram *program;
    const SaltTextTargetPolicy *policy;
    SaltTextProjectionWindow *projection;
    void *context;
    int (*is_stop)(void *context, int32_t token);

    /* Cheap proposal selection is separate from causal batch verification. */
    int (*select_proposal)(void *context, SaltTextTokenEpochController *controller,
                          const int32_t *tokens, uint32_t count,
                          const float *parent, SaltTextTokenEpochResult *result);
    int (*target)(void *context, SaltTextTokenEpochController *controller,
                  const int32_t *tokens, uint32_t count,
                  const float *parent, SaltTextTokenEpochResult *result);
} SaltTextGenerationBinding;

typedef struct SaltTextGenerated {
    /* Proposal checks in canonical NFQ order. After selection, entry zero
     * carries the winner; these IDs are not a causal target-block sequence. */
    int32_t candidate_token_ids[SALT_TEXT_NFQ_MAX_CHECKS];
    uint32_t proposal_count;
    uint32_t route_token_count;
    /* Proposal-check count N*F*Q; route_token_count remains N*F. */
    uint32_t candidate_count;
    uint32_t projection_reused_rows;
    /* SALT_TEXT_PROPOSAL_SOURCE_*; history match length when HISTORY. */
    uint32_t proposal_source;
    uint32_t history_matched_length;

    uint64_t proposal_ns;
    uint32_t epoch_path;
    SaltTextVerifyResult target;
} SaltTextGenerated;

typedef struct SaltTextScheduleStats {
    uint64_t prefill_node_ns, prefill_lookup_ns, prefill_restore_ns;
    uint64_t prefill_ordinary_ns;
    uint64_t decode_node_ns, decode_lookup_ns, decode_parent_restore_ns;
    uint64_t decode_verify_ns, decode_ordinary_ns;
    uint32_t prefill_hits, prefill_cached_tokens, decode_cycles;
    uint32_t parent_hits, nomogram_hits, decode_misses;
    uint32_t proposed_tokens, accepted_tokens, rejection_count;
} SaltTextScheduleStats;

typedef struct SaltTextScheduleDpr {
    SaltDprMode mode;
    SaltDprStore *store;
    SaltDprEdgeStats *stats;
    uint8_t *retained;
    SaltDprRuntime *runtime;
    const SaltDprMentorFamilyStore *families;
    const SaltDprMentorBindingStore *bindings;
    SaltDprSelectorCandidate *mentor_candidates;
    size_t mentor_capacity;
    SaltDprChart *effective_charts;
    uint32_t draft_n;
    uint64_t serial_ns, current_kv_bytes, additive_bytes;
    int mentor_enabled;
    uint8_t compatibility[32], mentor_policy[32];
    const uint8_t *qa_key;
    int attention_enabled;
    SaltDprAttentionPlanStore *attention_store;
    const int *hot_ready;
    const SaltDprAttentionPlan *hot_candidate;
    const uint8_t *hot_mindset, *hot_provenance;
    uint8_t attention_policy[32];
    uint32_t attention_layers, attention_experts;
} SaltTextScheduleDpr;

typedef struct SaltTextScheduleSelection {
    SaltDprEdgeKind kind;
    const SaltDprStoredEdge *stored;
    int32_t tokens[SALT_DPR_MAX_HORIZON];
    size_t indices[SALT_DPR_MAX_HORIZON];
    size_t binding_indices[SALT_DPR_MAX_HORIZON];
    uint32_t count, coverage_roots;
    uint32_t source_position;
    int32_t target_token;
    uint32_t output_base;
    int32_t bonus;
    int mentor_selected;
    SaltDprSelectorResult mentor;
    uint8_t node[32];
    uint64_t lookup_ns, recover_ns, verify_ns;
} SaltTextScheduleSelection;

typedef struct SaltTextScheduleRequest {
    SaltSamplerConfig sampler;
    float *logits;
    float *scratch_logits;
    int32_t *output_ids;
    uint32_t output_limit;
    int32_t close_token;
    /* Optional committed prompt/session tokens preceding output_ids. */
    const int32_t *history_ids;
    uint32_t history_count;
} SaltTextScheduleRequest;

typedef struct SaltTextScheduleResult {
    uint32_t source_position, output_count, result_position;
    int32_t stop_token;
    int synthetic_close;
    uint64_t sampler_draws;
    uint32_t path;
    int close_failed;
} SaltTextScheduleResult;

enum {
    SALT_TEXT_SCHEDULE_OBSERVE_RESULT = 0,
    SALT_TEXT_SCHEDULE_OBSERVE_FALLBACK = 1,
    SALT_TEXT_SCHEDULE_OBSERVE_PENDING = 2,
    SALT_TEXT_SCHEDULE_OBSERVE_SOURCE = 3,
    SALT_TEXT_SCHEDULE_OBSERVE_ATTENTION_READY = 4
};

typedef struct SaltTextSchedulerBindings {
    SaltTextGenerationBinding generation;
    SaltTextScheduleDpr *dpr;
    SaltTextScheduleStats *stats;
    void *context;
    uint64_t (*now_ns)(void *context);
    int (*step_known)(void *context, int32_t token, float *logits);
    int (*emit_token)(void *context, uint32_t count, int32_t token, int stop);
    int (*node_identity)(void *context, SaltDprNodeMaterial *material,
                         uint8_t node[32]);
    uint64_t (*kv_required)(void *context, uint32_t position);
    int (*load_exact)(void *context, const SaltDprStoredEdge *stored,
                      uint32_t position, uint32_t count);
    int (*advance_chain)(void *context, const int32_t *tokens, uint32_t count);
    int (*prepare_attention)(void *context, const SaltDprComputeIntent *intent,
                             const SaltDprStoredAttentionPlan *plan, uint64_t lookup_ns);
    int (*release_attention)(void *context);
    int (*observe)(void *context, const SaltTextScheduleSelection *selection,
                   const SaltTextVerifyResult *target, int fallback);
    int (*observe_native)(void *context, uint32_t source, int32_t token,
                          int committed, uint64_t elapsed_ns, uint64_t draws);
    int (*observe_target)(void *context, const SaltTextGenerated *generated,
                          uint64_t elapsed_ns);
    int (*publish)(void *context, const SaltTextScheduleResult *result);
} SaltTextSchedulerBindings;

int salt_text_generate_candidates(
    SaltTextTokenEpochController *controller,
    const SaltTextGenerationBinding *binding, float *logits,
    uint32_t maximum_tokens, float *scratch_logits,
    uint64_t (*now_ns)(void *), void *clock_context,
    SaltTextGenerated *result);
int salt_text_scheduler_bind(SaltTextTokenEpochController *controller,
                            const SaltTextSchedulerBindings *bindings);
int salt_text_scheduler_run(SaltTextTokenEpochController *controller,
                           const SaltTextScheduleRequest *request,
                           SaltTextScheduleResult *result);
int salt_text_scheduler_finish(SaltTextTokenEpochController *controller,
                              const SaltTextScheduleRequest *request,
                              SaltTextScheduleResult *result);
int salt_text_scheduler_publish(SaltTextTokenEpochController *controller,
                               const SaltTextScheduleResult *result);
void salt_text_scheduler_abort(SaltTextTokenEpochController *controller);

typedef struct SaltTextProgressivePlan {
    uint32_t seed_candidates;
    uint32_t maximum_tile_candidates;
} SaltTextProgressivePlan;

typedef struct SaltTextProgressiveResult {
    SaltTextVerifyResult target;
    uint32_t tile_count;
    uint32_t submitted_tile_count;
    uint32_t largest_tile;
} SaltTextProgressiveResult;

/* Ordinary all-row refinement of the same immutable execution program. Every
 * supplied input token is target-computed and committed; there is no draft
 * validation, accepted-prefix decision, or pending token. */
typedef struct SaltTextExecuteAllResult {
    SaltTextVerifyStatus status;
    uint32_t committed_count;
    uint32_t result_position;
    uint64_t transition_generation;
    SaltTextExecutionView view;
    SaltTextVerifyBackendStats backend;
} SaltTextExecuteAllResult;

/* Program compilation writes only backend-neutral layer/cell metadata into the
 * caller's startup arena. The descriptor and its nested descriptors remain
 * immutable and must outlive the program. */
int salt_text_verify_program_arena_requirement(
    const SaltTextModelExecDesc *descriptor, uint32_t maximum_candidates,
    size_t *bytes_out);
int salt_text_verify_program_compile(
    SaltTextVerifyProgram *program,
    const SaltTextModelExecDesc *descriptor, SaltTextKvState *kv_state,
    uint32_t maximum_candidates, void *startup_arena,
    size_t startup_arena_bytes);
int salt_text_executor_plan_compile(
    SaltTextExecutorPlan *plan, const SaltTextVerifyProgram *program,
    const SaltTextDispatchPolicy *policy,
    SaltTextExecutionAssignment *assignments, size_t assignment_count);

/* Execute one portable target-block transaction. The executor may mutate only
 * its preassigned canonical arena. Portable C validates, commits target-made
 * rows, scrubs rejected/failing tentative rows, and publishes the pending
 * correction/bonus and position exactly once. */
int salt_text_verify_execute(const SaltTextVerifyProgram *program,
                             SaltTextVerifyExecutor *executor,
                             const SaltTextTargetBlock *block,
                             SaltTextVerifyResult *result);
int salt_text_verify_frontier_execute(
    const SaltTextVerifyProgram *program,
    SaltTextVerifyExecutor *executor,
    const SaltTextTargetFrontierBlock *block,
    SaltTextVerifyResult *result);
int salt_text_verify_cascade_execute(
    const SaltTextVerifyProgram *program,
    SaltTextVerifyExecutor *executor,
    const SaltTextTargetCascadeBlock *block,
    SaltTextVerifyResult *result);
int salt_text_verify_route_execute(
    const SaltTextVerifyProgram *program,
    SaltTextVerifyExecutor *executor,
    const SaltTextTargetRouteBlock *block,
    SaltTextVerifyResult *result);
/* Proposal-only NFQ selection using existing parent logits. No model work or
 * KV publication occurs here; downstream compute consumes only the winner. */
int salt_text_verify_nfq_execute(
    const SaltTextVerifyProgram *program,
    SaltTextVerifyExecutor *executor,
    const SaltTextTargetNfqBlock *block,
    SaltTextVerifyResult *result);
int salt_text_token_epoch_init(SaltTextTokenEpochController *controller);
/* Return 0 after an exact-hit or cold-search resolution, 1 when a nested
 * admission observes the sole coordinator already active, and -1 on malformed
 * input or fail-closed lookup/search failure. This function never starts the
 * next token epoch; the caller must invoke it again at the next boundary. */
int salt_text_token_epoch_execute(
    SaltTextTokenEpochController *controller,
    const SaltTextTokenEpochRequest *request,
    SaltTextTokenEpochResult *result);
/* Verify one known candidate span through seed-and-grow complete target tiles.
 * Each full-hit tile commits before the next tile. The next tile consumes only
 * the prior target-owned pending logits; a first miss stops the sequence. */
int salt_text_verify_progressive_execute(
    const SaltTextVerifyProgram *program,
    SaltTextVerifyExecutor *executor,
    uint64_t transition_generation,
    uint32_t source_position,
    const int32_t *candidate_token_ids,
    uint32_t candidate_count,
    const float *parent_logits,
    const SaltTextProgressivePlan *plan,
    SaltTextProgressiveResult *result);
int salt_text_execute_all(const SaltTextVerifyProgram *program,
                          SaltTextVerifyExecutor *executor,
                          uint64_t transition_generation,
                          uint32_t source_position,
                          const int32_t *input_token_ids,
                          uint32_t input_count,
                          SaltTextExecuteAllResult *result);
/* Ordinary PREFILL computes and commits every input row, while zero intermediate
 * chunks or one final row pass through final norm/head/logit publication. */
int salt_text_execute_prefill(const SaltTextVerifyProgram *program,
                              SaltTextVerifyExecutor *executor,
                              uint64_t transition_generation,
                              uint32_t source_position,
                              const int32_t *input_token_ids,
                              uint32_t input_count, uint32_t output_rows,
                              SaltTextExecuteAllResult *result);

/* Phase-1 CPU executor. Its arena/context are deliberately separate from the
 * architecture-neutral program. No operation-time allocation is permitted. */
typedef struct SaltTextVerifyCpuContext {
    const SaltTextVerifyProgram *program;
    unsigned char *arena;
    size_t arena_bytes;
    size_t tensor_scratch_offset;
    void *tensor_scratch;
    size_t tensor_scratch_bytes;
    SaltTextExecutionAssignment *assignments;
    uint32_t assignment_count;
    SaltAreaTask *area_tasks;
    uint32_t area_task_capacity;
    SaltAreaFrontier area_frontier;
    unsigned char *ffn_frontier_seats;
    size_t ffn_frontier_seat_stride;
    uint32_t ffn_frontier_seat_capacity;
    SaltTextTensorHostBatch *node_jobs;
    uint32_t node_job_capacity;
    SaltTextExecutorPlan assignment_plan;
    int (*parallel_run)(void *parallel_context, int active_workers,
                        void (*worker)(int worker, void *task), void *task);
    void *parallel_context;
    float *parallel_scores;
    size_t parallel_score_stride;
    uint32_t parallel_workers;
    uint64_t submitted_generation;
    uint32_t submitted_source_position;
    uint32_t submitted_candidates;
    int submitted_frontier;
    uint32_t coalesced_attention_layer;
    uint32_t coalesced_dense_layer;
    uint32_t coalesced_expert_layer;
    uint32_t frontier_expert_layer;
    SaltTensorHostGraphState graph_state;
    SaltTensorHostNodePlan graph_node_plan;
    SaltTensorHostGraphSpan *graph_spans;
    uint32_t graph_span_capacity;
    uint32_t graph_span_count;
    uint32_t graph_workers;
    SaltTextTouchedSpan touched_spans[SALT_TEXT_MAX_TOUCHED_SPANS];
    uint32_t touched_span_count;
    uint64_t touched_span_bytes;
    SaltTextVerifyBackendStats stats;
    int profile_enabled;
    int coalescing_enabled;
    int ready;
    int submitted;
    int finished;
} SaltTextVerifyCpuContext;

int salt_text_verify_cpu_arena_requirement(
    const SaltTextVerifyProgram *program, size_t *bytes_out);
int salt_text_verify_cpu_compile(SaltTextVerifyCpuContext *context,
                                 const SaltTextVerifyProgram *program,
                                 void *arena, size_t arena_bytes);
int salt_text_verify_cpu_parallel_bind(
    SaltTextVerifyCpuContext *context,
    int (*parallel_run)(void *parallel_context, int active_workers,
                        void (*worker)(int worker, void *task), void *task),
    void *parallel_context, float *parallel_scores,
    size_t parallel_score_stride, uint32_t parallel_workers);
int salt_text_verify_cpu_profile_set(
    SaltTextVerifyCpuContext *context, int enabled);
int salt_text_verify_cpu_executor_init(SaltTextVerifyExecutor *executor,
                                       SaltTextVerifyCpuContext *context);

/* Common fixed-startup executor for GPU_ONLY and MIXED. The package device
 * ABI and backend program ABI are both admitted before context/arena mutation.
 * The backend owns the shared buffer and opaque program/command state. */
typedef struct SaltTextVerifyHeterogeneousContext {
    const SaltTextVerifyProgram *program;
    const struct SaltTextGpuProgramOps *backend_ops;
    void *backend_state;
    void *command;
    struct SaltGpuSharedBuffer *canonical;
    unsigned char *startup_arena;
    size_t startup_arena_bytes;
    void *cpu_tensor_scratch;
    size_t cpu_tensor_scratch_bytes;
    SaltTextExecutionAssignment *assignments;
    uint32_t assignment_count;
    SaltAreaTask *area_tasks;
    uint32_t area_task_capacity;
    SaltAreaFrontier area_frontier;
    SaltTextExecutorPlan assignment_plan;
    uint64_t submitted_generation;
    uint32_t submitted_source_position;
    uint32_t submitted_candidates;
    SaltTextVerifyBackendStats stats;
    uint32_t backend_flags;
    int ready;
    int submitted;
    int finished;
    int resolved;
    uint32_t active_expert_lease_layer;
    int expert_lease_active;
    const SaltTextExpertResourceOps *expert_resource_ops;
    void *expert_resource_opaque;
    int32_t expert_resource_ids[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    int32_t expert_resource_slots[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t expert_resource_count;
    int32_t frontier_token_ids[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t frontier_parent_rows[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    uint32_t frontier_depths[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
} SaltTextVerifyHeterogeneousContext;

int salt_text_verify_heterogeneous_arena_requirement(
    const SaltTextVerifyProgram *program,
    const SaltTextDispatchPolicy *policy,
    const struct SaltTextGpuProgramOps *backend_ops,
    size_t *bytes_out);
int salt_text_verify_heterogeneous_backend_requirements(
    const SaltTextVerifyProgram *program,
    const SaltTextDispatchPolicy *policy,
    const struct SaltTextGpuProgramOps *backend_ops,
    struct SaltTextGpuProgramRequirements *requirements,
    size_t requirements_size);
int salt_text_verify_heterogeneous_compile(
    SaltTextVerifyHeterogeneousContext *context,
    const SaltTextVerifyProgram *program,
    const SaltTextDispatchPolicy *policy,
    const struct SaltTextGpuProgramOps *backend_ops,
    void *startup_arena, size_t startup_arena_bytes);
int salt_text_verify_heterogeneous_resource_bind(
    SaltTextVerifyHeterogeneousContext *context,
    const SaltTextExpertResourceOps *ops, void *opaque);
int salt_text_verify_heterogeneous_executor_init(
    SaltTextVerifyExecutor *executor,
    SaltTextVerifyHeterogeneousContext *context);
int salt_text_verify_heterogeneous_destroy(
    SaltTextVerifyHeterogeneousContext *context);

#ifdef __cplusplus
}
#endif

#endif
