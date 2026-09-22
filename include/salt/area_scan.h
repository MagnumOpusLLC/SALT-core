#ifndef SALT_AREA_SCAN_H
#define SALT_AREA_SCAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SaltAreaScanMode {
    SALT_AREA_SCAN_INVALID = 0,
    SALT_AREA_SCAN_M1 = 1,
    SALT_AREA_SCAN_MK = 2,
    SALT_AREA_SCAN_MN = 3
} SaltAreaScanMode;

/* Backend-neutral request for one exact row-area dispatch. output_row_tiles is
 * the complete count of independently owned weight/output-row strips across
 * all jobs; each strip spans its job's candidate rows. Candidate width is
 * tracked separately by requested_rows/admitted_rows. */
typedef struct SaltAreaScanRequest {
    uint32_t requested_rows;
    uint32_t admitted_rows;
    uint32_t resident_workers;
    uint32_t active_worker_limit;
    uint32_t narrow_workers;
    uint32_t wide_allowed;
    uint64_t output_row_tiles;
    uint64_t candidate_output_tiles;
} SaltAreaScanRequest;

typedef struct SaltAreaScanPlan {
    SaltAreaScanMode mode;
    uint32_t requested_rows;
    uint32_t active_rows;
    uint32_t rows_per_wave;
    uint32_t active_workers;
    uint64_t output_row_tiles;
    uint64_t candidate_output_tiles;
    uint32_t matrix_parallel;
    /* True ownership shape selected before execution. Output-row ownership
     * keeps every admitted candidate row under one retained weight tile.
     * Candidate splitting is legal only when output rows cannot fill W. */
    uint32_t output_row_owned;
    uint32_t candidate_row_split;
} SaltAreaScanPlan;

typedef enum SaltAreaTaskState {
    SALT_AREA_TASK_EMPTY = 0,
    SALT_AREA_TASK_QUEUED = 1,
    SALT_AREA_TASK_RUNNING = 2,
    SALT_AREA_TASK_DONE = 3,
    SALT_AREA_TASK_CANCELLED = 4,
    /* A graph task that yielded before mutation because an existing bounded
     * resource was not READY. The engine may refire it in the same generation
     * after the authoritative resource owner resolves the wait. */
    SALT_AREA_TASK_WAITING = 5,
    /* A populated operation-graph task whose declared predecessors have not
     * all completed. EMPTY remains reserved for unused frontier capacity. */
    SALT_AREA_TASK_BLOCKED = 6
} SaltAreaTaskState;

enum { SALT_AREA_TASK_CANCEL_REQUESTED = 1u };

typedef struct SaltAreaTask {
    uint64_t generation;
    uint32_t node_id;
    uint32_t parent_id;
    uint32_t phase;
    int32_t expert_id;
    uint32_t job_index;
    uint32_t candidate_first;
    uint32_t candidate_count;
    uint32_t output_first;
    uint32_t output_count;
    uint32_t flags;
    SaltAreaTaskState state;
} SaltAreaTask;

typedef struct SaltAreaFrontier {
    SaltAreaTask *tasks;
    uint32_t capacity;
    uint32_t count;
    uint64_t generation;
} SaltAreaFrontier;

typedef enum SaltAreaWfqItemState {
    SALT_AREA_WFQ_ITEM_EMPTY = 0,
    SALT_AREA_WFQ_ITEM_QUEUED = 1,
    SALT_AREA_WFQ_ITEM_RUNNING = 2,
    SALT_AREA_WFQ_ITEM_DONE = 3,
    SALT_AREA_WFQ_ITEM_WAITING = 4,
    SALT_AREA_WFQ_ITEM_RETIRED = 5
} SaltAreaWfqItemState;

typedef enum SaltAreaWfqOutcome {
    SALT_AREA_WFQ_CONTINUE = 1,
    SALT_AREA_WFQ_BRANCH_FAIL = 2,
    SALT_AREA_WFQ_TARGET_WIN = 3,
    SALT_AREA_WFQ_RESOURCE_WAIT = 4,
    SALT_AREA_WFQ_FATAL = 5,
    SALT_AREA_WFQ_STALE = 6
} SaltAreaWfqOutcome;

typedef enum SaltAreaWfqStatus {
    SALT_AREA_WFQ_INVALID = 0,
    SALT_AREA_WFQ_WIN = 1,
    SALT_AREA_WFQ_EXHAUSTED = 2,
    SALT_AREA_WFQ_NEED_RESOURCE = 3,
    SALT_AREA_WFQ_FAILURE = 4
} SaltAreaWfqStatus;

typedef struct SaltAreaWfqPlan {
    uint32_t workers;
    uint32_t frontier_width;
    uint32_t queue_length;
} SaltAreaWfqPlan;

typedef struct SaltAreaWfqItem {
    uint64_t generation;
    uint32_t branch_index;
    uint32_t queue_index;
    uint32_t owner_worker;
    SaltAreaWfqItemState state;
    SaltAreaTask *task;
} SaltAreaWfqItem;

typedef struct SaltAreaWfqBranch {
    uint32_t seed_id;
    uint32_t owner_worker;
    uint32_t next_queue;
    uint32_t active;
} SaltAreaWfqBranch;

typedef struct SaltAreaWfqItemResult {
    SaltAreaWfqOutcome outcome;
    int32_t target_token;
    uint32_t target_state_slot;
} SaltAreaWfqItemResult;

typedef struct SaltAreaWfqLedger {
    uint64_t generation;
    uint32_t winning_branch;
    uint32_t winning_queue;
    int32_t target_token;
    uint32_t target_state_slot;
} SaltAreaWfqLedger;

typedef struct SaltAreaWfqResult {
    SaltAreaWfqStatus status;
    SaltAreaWfqLedger ledger;
    uint32_t executed_items;
    uint32_t retired_items;
    uint32_t active_branches;
} SaltAreaWfqResult;

typedef struct SaltAreaWfqRuntime {
    SaltAreaWfqPlan plan;
    SaltAreaWfqLedger ledger;
    SaltAreaWfqBranch *branches;
    SaltAreaWfqItem *items;
    SaltAreaWfqItemResult *results;
    SaltAreaTask *tasks;
    uint32_t branch_capacity;
    uint32_t item_capacity;
    uint32_t result_capacity;
    uint32_t task_capacity;
    uint64_t generation;
    int ready;
} SaltAreaWfqRuntime;

typedef struct SaltAreaNfqPlan {
    uint32_t workers;
    uint32_t sequence_tiles;
    uint32_t route_count;
    uint32_t queue_length;
} SaltAreaNfqPlan;

typedef struct SaltAreaNfqRoute {
    uint32_t route_id;
    uint32_t version;
    uint32_t active;
    uint32_t worker_mask;
} SaltAreaNfqRoute;

typedef struct SaltAreaNfqItem {
    uint64_t epoch_generation;
    uint64_t parent_generation;
    uint32_t route_slot;
    uint32_t route_id;
    uint32_t route_version;
    uint32_t sequence_index;
    uint32_t queue_index;
    uint32_t owner_worker;
    SaltAreaWfqItemState state;
} SaltAreaNfqItem;

typedef struct SaltAreaNfqMatrix {
    SaltAreaNfqPlan plan;
    SaltAreaNfqRoute *routes;
    SaltAreaNfqItem *items;
    uint32_t route_capacity;
    uint32_t item_capacity;
    uint32_t item_count;
    uint64_t epoch_generation;
    uint64_t parent_generation;
    uint32_t parent_position;
    int ready;
    int resolved;
} SaltAreaNfqMatrix;

typedef int (*SaltAreaWfqPoolRun)(
    void *pool_context, int active_workers,
    void (*worker)(int worker, void *task), void *task);
typedef int (*SaltAreaWfqItemExecute)(
    void *item_context, const SaltAreaWfqItem *item,
    SaltAreaWfqItemResult *result);

/* Pure deterministic planner. Host/backend resource guards precompute the
 * resident and active limits. The planner chooses ownership width only;
 * arithmetic, selected work, canonical destinations, and commit are unchanged. */
int salt_area_scan_plan(const SaltAreaScanRequest *request,
                        SaltAreaScanPlan *plan);
int salt_area_frontier_bind(SaltAreaFrontier *frontier,
                            SaltAreaTask *tasks, uint32_t capacity);
int salt_area_frontier_reset(SaltAreaFrontier *frontier,
                             uint64_t generation);
int salt_area_frontier_append(SaltAreaFrontier *frontier,
                              const SaltAreaTask *task);
int salt_area_frontier_cancel_except(SaltAreaFrontier *frontier,
                                     uint64_t generation,
                                     uint32_t survivor_node);
int salt_area_wfq_bind(
    SaltAreaWfqRuntime *runtime,
    SaltAreaWfqBranch *branches, uint32_t branch_capacity,
    SaltAreaWfqItem *items, uint32_t item_capacity,
    SaltAreaWfqItemResult *results, uint32_t result_capacity,
    SaltAreaTask *tasks, uint32_t task_capacity);
int salt_area_wfq_start(
    SaltAreaWfqRuntime *runtime, const SaltAreaWfqPlan *plan,
    uint64_t generation, const uint32_t *seed_ids,
    uint32_t seed_count);
int salt_area_wfq_execute(
    SaltAreaWfqRuntime *runtime,
    SaltAreaWfqPoolRun pool_run, void *pool_context,
    SaltAreaWfqItemExecute item_execute, void *item_context,
    SaltAreaWfqResult *result);
int salt_area_nfq_bind(
    SaltAreaNfqMatrix *matrix,
    SaltAreaNfqRoute *routes, uint32_t route_capacity,
    SaltAreaNfqItem *items, uint32_t item_capacity);
int salt_area_nfq_start(
    SaltAreaNfqMatrix *matrix, const SaltAreaNfqPlan *plan,
    uint64_t epoch_generation, uint64_t parent_generation,
    uint32_t parent_position, const uint32_t *route_ids,
    uint32_t route_id_count);
int salt_area_nfq_retire_route(
    SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
    uint32_t route_slot, uint32_t *retired_items);
int salt_area_nfq_reschedule_route(
    SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
    uint64_t parent_generation, uint32_t route_slot,
    uint32_t new_route_id);
int salt_area_nfq_resolve(
    SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
    uint64_t parent_generation, uint32_t winning_route_slot,
    uint32_t winning_sequence_index, uint32_t winning_queue_index,
    uint32_t *retired_items);
/* Return 1 only while the item belongs to the current unresolved epoch and
 * parent and remains eligible to publish; return 0 for stale/retired work and
 * -1 for malformed input. */
int salt_area_nfq_publishable(
    const SaltAreaNfqMatrix *matrix, uint32_t item_index,
    uint64_t epoch_generation, uint64_t parent_generation);
/* Take up to one current-epoch READY item per persistent worker. Items move
 * QUEUED→RUNNING in deterministic owner-worker order. */
int salt_area_nfq_take_ready(
    SaltAreaNfqMatrix *matrix,
    uint64_t epoch_generation, uint64_t parent_generation,
    uint32_t *item_indices, uint32_t item_capacity,
    uint32_t *item_count);
/* Complete one RUNNING item. CONTINUE unlocks its next Q seat; RESOURCE_WAIT
 * preserves the same item for explicit refire; BRANCH_FAIL retires N×Q for the
 * route. TARGET_WIN leaves resolution to salt_area_nfq_resolve(). */
int salt_area_nfq_complete(
    SaltAreaNfqMatrix *matrix,
    uint64_t epoch_generation, uint64_t parent_generation,
    uint32_t item_index, SaltAreaWfqOutcome outcome,
    uint32_t *retired_items);
int salt_area_nfq_refire(
    SaltAreaNfqMatrix *matrix,
    uint64_t epoch_generation, uint64_t parent_generation,
    uint32_t item_index);
int salt_area_nfq_cancel(
    SaltAreaNfqMatrix *matrix,
    uint64_t epoch_generation, uint64_t parent_generation,
    uint32_t *retired_items);

#ifdef __cplusplus
}
#endif

#endif /* SALT_AREA_SCAN_H */
