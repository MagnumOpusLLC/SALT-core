#ifndef SALT_TENSOROPS_H
#define SALT_TENSOROPS_H

#include <stddef.h>
#include <stdint.h>

#include "salt/area_scan.h"

#ifdef __cplusplus
extern "C" {
#endif

struct SaltTensorDesc;
struct SaltTensorHostBatch;
typedef struct SaltTensorHostNodeOps SaltTensorHostNodeOps;

/* Engine-wide tensor representation. Models select one of these encodings in
 * immutable configuration; CPU/Metal/CUDA executors privately realize it. */
typedef enum SaltTensorEncoding {
    SALT_TENSOR_ENCODING_NONE = 0,
    SALT_TENSOR_ENCODING_F32 = 1,
    SALT_TENSOR_ENCODING_BF16 = 2,
    SALT_TENSOR_ENCODING_AFFINE_Q4 = 3,
    SALT_TENSOR_ENCODING_AFFINE_Q8 = 4,
    SALT_TENSOR_ENCODING_NVFP4 = 5,
    SALT_TENSOR_ENCODING_ROW_INT2 = 6
} SaltTensorEncoding;

/* Source ownership is independent from execution backend. STATIC names one
 * canonical immutable payload resource. SHARED names startup-owned decoded
 * constants. SELECTED names a model-stable expert payload whose bounded runtime
 * slot must become READY before a backend command may read it. */
typedef enum SaltTensorSourceClass {
    SALT_TENSOR_SOURCE_NONE = 0,
    SALT_TENSOR_SOURCE_STATIC = 1,
    SALT_TENSOR_SOURCE_SHARED = 2,
    SALT_TENSOR_SOURCE_SELECTED = 3
} SaltTensorSourceClass;

typedef enum SaltTensorResourceKind {
    SALT_TENSOR_RESOURCE_ARENA = 0,
    SALT_TENSOR_RESOURCE_TRUNK = 1,
    SALT_TENSOR_RESOURCE_EXPERT = 2,
    SALT_TENSOR_RESOURCE_SHARED = 3
} SaltTensorResourceKind;

enum {
    SALT_TENSOR_STORAGE_OPTIONAL_BIAS = 1u,
    SALT_TENSOR_STORAGE_DECODED_F32 = 2u
};

/* All offsets are byte offsets inside the named canonical source. The model
 * adapter derives them once while authenticated bindings are built. Backends
 * never infer offsets from host pointer containment during execution. */
typedef struct SaltTensorStorageSpec {
    SaltTensorEncoding encoding;
    SaltTensorSourceClass source_class;
    uint32_t resource_kind;
    uint32_t resource_id;
    uint64_t logical_resource_id;
    uint32_t group_size;
    uint32_t flags;
    uint64_t value_offset;
    uint64_t scale_offset;
    uint64_t bias_offset;
    uint64_t auxiliary_offset;
    uint64_t auxiliary_2_offset;
    uint64_t value_bytes;
    uint64_t scale_bytes;
    uint64_t bias_bytes;
    uint64_t auxiliary_bytes;
    uint64_t auxiliary_2_bytes;
} SaltTensorStorageSpec;

typedef struct SaltTensorResourceSpec {
    uint32_t kind;
    uint32_t resource_id;
    SaltTensorSourceClass source_class;
    uint32_t flags;
    const void *base;
    size_t nbytes;        /* complete canonical source extent */
    size_t mapped_nbytes; /* bytes currently addressable at base */
} SaltTensorResourceSpec;

/* Portable-engine operation graph. The model adapter supplies only immutable
 * operation ranges and explicit predecessor indices; it does not supply a
 * scheduler. Dependencies must point backward, making the descriptor a
 * canonical topological order while still allowing several nodes to be READY
 * together. Physical CPU/GPU completion order never changes these indices. */
enum {
    SALT_TENSOR_GRAPH_NODE_RESOURCE_CHECK = 1u
};

typedef struct SaltTensorGraphNode {
    uint32_t operation_first;
    uint32_t operation_count;
    uint32_t dependency_first;
    uint32_t dependency_count;
    uint32_t flags;
} SaltTensorGraphNode;

typedef struct SaltTensorGraphProgramDesc {
    const SaltTensorGraphNode *nodes;
    uint32_t node_count;
    const uint32_t *dependencies;
    uint32_t dependency_count;
    uint32_t logical_operation_count;
} SaltTensorGraphProgramDesc;

typedef struct SaltTensorGraphProgram {
    const SaltTensorGraphNode *nodes;
    uint32_t node_count;
    const uint32_t *dependencies;
    uint32_t dependency_count;
    uint32_t logical_operation_count;
    uint32_t root_count;
    int ready;
} SaltTensorGraphProgram;

typedef enum SaltTensorGraphNodeOutcome {
    SALT_TENSOR_GRAPH_NODE_DONE = 1,
    SALT_TENSOR_GRAPH_NODE_NEED_RESOURCE = 2,
    SALT_TENSOR_GRAPH_NODE_FAILED = 3
} SaltTensorGraphNodeOutcome;

typedef enum SaltTensorGraphStatus {
    SALT_TENSOR_GRAPH_INVALID = 0,
    SALT_TENSOR_GRAPH_ACTIVE = 1,
    SALT_TENSOR_GRAPH_NEED_RESOURCE = 2,
    SALT_TENSOR_GRAPH_COMPLETE = 3,
    SALT_TENSOR_GRAPH_FAILED = 4,
    SALT_TENSOR_GRAPH_CANCELLED = 5
} SaltTensorGraphStatus;

/* Runtime state is bound to the engine's existing fixed-capacity area
 * frontier. One frontier task corresponds to one immutable graph node and
 * retains its generation, operation range, and state across resource yield.
 * Backends receive stable node indices from take_ready(), execute through the
 * existing TensorOps/resource machinery, and return one canonical outcome
 * batch. They never choose graph readiness or publication. */
typedef struct SaltTensorGraphRuntime {
    const SaltTensorGraphProgram *program;
    SaltAreaFrontier *frontier;
    uint64_t generation;
    uint64_t last_generation;
    uint32_t phase;
    uint32_t completed_nodes;
    uint32_t running_nodes;
    uint32_t waiting_nodes;
    SaltTensorGraphStatus status;
    int active;
} SaltTensorGraphRuntime;

typedef struct SaltTensorGraphProgress {
    uint64_t generation;
    uint32_t total_nodes;
    uint32_t blocked_nodes;
    uint32_t ready_nodes;
    uint32_t running_nodes;
    uint32_t waiting_nodes;
    uint32_t completed_nodes;
    uint32_t cancelled_nodes;
    SaltTensorGraphStatus status;
} SaltTensorGraphProgress;

int salt_tensor_graph_program_arena_requirement(
    const SaltTensorGraphProgramDesc *descriptor, size_t *bytes_out);
int salt_tensor_graph_program_compile(
    SaltTensorGraphProgram *program,
    const SaltTensorGraphProgramDesc *descriptor,
    void *startup_arena, size_t startup_arena_bytes);
int salt_tensor_graph_runtime_bind(
    SaltTensorGraphRuntime *runtime, SaltAreaFrontier *frontier);
int salt_tensor_graph_start(
    SaltTensorGraphRuntime *runtime, const SaltTensorGraphProgram *program,
    uint64_t generation, uint32_t phase);
/* Return READY node indices in canonical graph order and mark them RUNNING.
 * Only one physical batch may be outstanding. */
int salt_tensor_graph_take_ready(
    SaltTensorGraphRuntime *runtime, uint64_t generation,
    uint32_t *node_indices, uint32_t node_capacity,
    uint32_t *node_count);
/* Apply the complete outstanding batch in the same canonical index order in
 * which take_ready() returned it. Backends may finish physically in any order;
 * readiness is recomputed only after all outcomes are present. */
int salt_tensor_graph_complete_ready(
    SaltTensorGraphRuntime *runtime, uint64_t generation,
    const uint32_t *node_indices,
    const SaltTensorGraphNodeOutcome *outcomes, uint32_t node_count);
/* Resource acquisition remains outside TensorOps. Refire only names WAITING
 * resource-check nodes from the same generation after their existing resource
 * owner has made them READY. */
int salt_tensor_graph_refire(
    SaltTensorGraphRuntime *runtime, uint64_t generation,
    const uint32_t *node_indices, uint32_t node_count);
/* External cancellation is legal only after the physical executor has drained
 * its outstanding batch. No graph node publishes state through this API. */
int salt_tensor_graph_cancel(
    SaltTensorGraphRuntime *runtime, uint64_t generation);
int salt_tensor_graph_progress(
    const SaltTensorGraphRuntime *runtime, SaltTensorGraphProgress *progress);

/* Host-pointer operations are the portable CPU realization. They remain
 * separate from backend command encoding and may use an opaque model binding
 * for authenticated source/cache access. */
typedef struct SaltTensorHostOps {
    size_t (*fixed_scratch_requirement)(const struct SaltTensorDesc *tensor);
    int (*gather_rows)(const struct SaltTensorDesc *tensor,
                       const int32_t *row_ids, uint32_t row_count,
                       float *output, void *scratch, size_t scratch_bytes);
    int (*matvec)(const struct SaltTensorDesc *tensor,
                  const float *input, float *output,
                  void *scratch, size_t scratch_bytes);
    int (*matvec_batch)(const struct SaltTensorDesc *tensor,
                        const float *inputs, uint32_t row_count,
                        float *outputs, void *scratch, size_t scratch_bytes);
    int (*matvec_multi_batch)(const struct SaltTensorHostBatch *jobs,
                              uint32_t job_count,
                              void *scratch, size_t scratch_bytes);
    /* Return 0 after exact area execution, 1 to decline before mutation so the
     * ordinary host realization may run, and -1 on an execution failure. */
    int (*matvec_area_batch)(const struct SaltTensorHostBatch *jobs,
                             uint32_t job_count, uint32_t active_rows,
                             SaltAreaScanPlan *plan,
                             void *scratch, size_t scratch_bytes);
    const SaltTensorHostNodeOps *retained_node;
} SaltTensorHostOps;

typedef struct SaltTensorHostBatch {
    const struct SaltTensorDesc *tensor;
    const float *inputs;
    uint32_t row_count;
    float *outputs;
} SaltTensorHostBatch;

/* Retained CPU-node realization for an already admitted tensor operation. The
 * executor supplies one fixed startup-owned scratch seat. prepare() resolves
 * the immutable job descriptors and acquires existing resource leases into
 * that seat; worker() executes one deterministic worker slice; finish()
 * validates all slices and releases the exact acquired leases. No callback may
 * allocate, create a resource identity, or change arithmetic ownership. */
typedef struct SaltTensorHostNodePlan {
    uint32_t active_workers;
    SaltAreaScanPlan area;
    /* Optional bounded resource prefix; zero keeps the complete-job contract.
     * The engine, not the binding, advances the remaining canonical jobs. */
    uint32_t consumed_jobs;
} SaltTensorHostNodePlan;

struct SaltTensorHostNodeOps {
    size_t (*fixed_requirement)(const struct SaltTensorDesc *tensor);
    int (*prepare)(const SaltTensorHostBatch *jobs, uint32_t job_count,
                   uint32_t active_rows, uint32_t worker_limit,
                   void *seat, size_t seat_bytes,
                   SaltTensorHostNodePlan *plan);
    int (*worker)(void *seat, uint32_t worker_index);
    int (*finish)(void *seat);
};

typedef struct SaltTensorHostGraphCallbacks {
    uint32_t node_count;
    uint32_t logical_node_count;
    int (*is_parallel)(void *context, uint32_t node_index);
    int (*prepare_parallel)(
        void *context, uint32_t node_index,
        const SaltTensorHostNodeOps **ops_out,
        void **seat_out, uint32_t *active_workers_out);
    int (*execute_serial)(void *context, uint32_t node_index);
} SaltTensorHostGraphCallbacks;

typedef struct SaltTensorHostGraphResult {
    uint32_t nodes_executed;
    uint32_t spans_executed;
    uint32_t serial_spans;
    uint32_t parallel_spans;
    uint32_t internal_barriers;
} SaltTensorHostGraphResult;

/* Startup-compiled level span over one immutable operation program. Every node
 * in a span has the same dependency/completion epoch. Independent tensor jobs
 * may therefore share one retained worker generation without changing any
 * canonical operation or reduction order. */
typedef struct SaltTensorHostGraphSpan {
    uint32_t first_node;
    uint32_t node_count;
    uint32_t dependency_epoch;
    uint32_t completion_epoch;
    uint32_t parallel;
} SaltTensorHostGraphSpan;

typedef int (*SaltTensorHostGraphParallelRun)(
    void *parallel_context, int active_workers,
    void (*worker)(int worker, void *task), void *task);

typedef struct SaltTensorHostGraphState {
    const SaltTensorHostGraphCallbacks *callbacks;
    void *context;
    uint32_t workers;
    uint32_t failed;
    uint32_t active_workers;
    const SaltTensorHostNodeOps *node_ops;
    void *node_seat;
    uint32_t barrier_arrived;
    uint32_t barrier_epoch;
    uint32_t collective_nodes;
    int (*collective_cell)(void *context, uint32_t node, uint32_t worker);
} SaltTensorHostGraphState;

/* Execute one already-compiled span program through the caller's retained
 * worker flow. Serial spans stay on the coordinator; parallel spans use one
 * existing pool phase and one true completion boundary. This creates no plan,
 * resource, worker, allocation, or all-worker spin barrier around serial work. */
int salt_tensor_host_graph_execute(
    SaltTensorHostGraphState *state,
    const SaltTensorHostGraphCallbacks *callbacks, void *context,
    uint32_t worker_count, SaltTensorHostGraphParallelRun parallel_run,
    void *parallel_context, SaltTensorHostGraphResult *result);

/* All members of the existing pool traverse every immutable node themselves.
 * A cell may use the shared graph barrier for preparation/consumer/retirement;
 * it must drain those barriers on failure. No per-node pool submission. */
void salt_tensor_host_graph_barrier(SaltTensorHostGraphState *state);
int salt_tensor_host_graph_collective_execute(
    SaltTensorHostGraphState *state, uint32_t node_count,
    int (*cell)(void *context, uint32_t node, uint32_t worker), void *context,
    uint32_t worker_count, SaltTensorHostGraphParallelRun parallel_run,
    void *parallel_context, SaltTensorHostGraphResult *result);

typedef struct SaltTensorProgramCallbacks {
    uint32_t node_count;
    int (*execute_node)(void *context, uint32_t node_index);
} SaltTensorProgramCallbacks;

/* Execute one immutable node program without embedding verification policy in
 * the loop. The adapter owns node materialization and existing resource calls;
 * this function owns only exact node order and fail-closed progression. */
int salt_tensor_program_execute(
    const SaltTensorProgramCallbacks *callbacks,
    void *context, uint32_t *executed_nodes);

typedef struct SaltTensorHostRealizations {
    const SaltTensorHostOps *cpu;
    /* Compatibility host wrapper only. Device-resident execution is selected
     * by the executor's backend TensorOps, never by this pointer. */
    const SaltTensorHostOps *compatibility_gpu;
} SaltTensorHostRealizations;

typedef struct SaltTensorDesc {
    uint32_t rows;
    uint32_t cols;
    uint64_t stable_handle;
    const void *binding;
    const SaltTensorHostRealizations *host_ops;
    SaltTensorStorageSpec storage;
} SaltTensorDesc;

/* Portable realization of the embedding cell. The tensor's existing host
 * gather owns storage decoding; this helper owns the model-supplied embedding
 * scale and finite-output boundary. Scratch is caller/startup owned. */
int salt_tensor_embedding_gather(
    const SaltTensorDesc *embedding,
    const int32_t *token_ids, uint32_t token_count,
    float embedding_scale, float *output,
    void *scratch, size_t scratch_bytes);

/* Build deterministic expert-grouped execution order from the already
 * authoritative canonical selected-expert IDs. This operation never selects,
 * sorts within an expert, renormalizes, or changes canonical rank. The caller
 * supplies expert_offsets[expert_count] as mutable scratch. */
int salt_tensor_selected_route_group(
    const int32_t *selected_experts,
    uint32_t row_count, uint32_t topk, uint32_t expert_count,
    const float *row_inputs, uint32_t hidden,
    uint32_t *expert_offsets, uint32_t expert_offset_count,
    int32_t *grouped_to_canonical,
    int32_t *canonical_to_grouped,
    float *routed_inputs);

/* Reduce grouped expert outputs in canonical row/rank order. Grouped compute
 * completion order is irrelevant; canonical_to_grouped resolves only the
 * preassigned source row for each selected rank. */
int salt_tensor_selected_weighted_reduce(
    const float *selected_weights,
    const int32_t *canonical_to_grouped,
    const float *grouped_outputs,
    uint32_t row_count, uint32_t topk, uint32_t width,
    float *outputs);

/* Complete-row/job device operation descriptor. It contains no backend pointer
 * and no model-specific enum; an executor maps canonical byte offsets into its
 * fixed arena and resolves tensor storage through the immutable spec. */
typedef struct SaltTensorDeviceBatch {
    const SaltTensorDesc *tensor_table;
    size_t tensor_stride;
    uint32_t tensor_count;
    uint64_t selector_offset;
    size_t selector_stride;
    uint64_t input_offset;
    size_t input_stride;
    uint64_t destination_offset;
    size_t destination_stride;
    uint32_t first_complete;
    uint32_t complete_count;
} SaltTensorDeviceBatch;

#define SALT_TENSOR_NO_OFFSET UINT64_MAX

int salt_tensor_storage_validate(const SaltTensorStorageSpec *storage,
                                 uint32_t rows, uint32_t cols);
int salt_tensor_resource_validate(const SaltTensorResourceSpec *resource);
const SaltTensorResourceSpec *salt_tensor_resource_find(
    const SaltTensorResourceSpec *resources, uint32_t resource_count,
    uint32_t kind, uint32_t resource_id);
int salt_tensor_storage_resolves(const SaltTensorStorageSpec *storage,
                                 const SaltTensorResourceSpec *resources,
                                 uint32_t resource_count);
int salt_tensor_desc_validate(const SaltTensorDesc *tensor);
int salt_tensor_desc_absent(const SaltTensorDesc *tensor);

#ifdef __cplusplus
}
#endif

#endif /* SALT_TENSOROPS_H */
