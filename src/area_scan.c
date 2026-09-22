#include "salt/area_scan.h"

#include <string.h>

int salt_area_scan_plan(const SaltAreaScanRequest *request,
                        SaltAreaScanPlan *plan) {
    SaltAreaScanPlan built;
    uint64_t workers;
    if (plan) memset(plan, 0, sizeof *plan);
    if (!request || !plan || request->requested_rows == 0 ||
        request->admitted_rows == 0 ||
        request->admitted_rows > request->requested_rows ||
        request->resident_workers == 0 ||
        request->active_worker_limit == 0 ||
        request->active_worker_limit > request->resident_workers ||
        request->narrow_workers == 0 ||
        request->narrow_workers > request->resident_workers ||
        request->wide_allowed > 1u || request->output_row_tiles == 0 ||
        request->candidate_output_tiles < request->output_row_tiles)
        return -1;
    memset(&built, 0, sizeof built);
    built.mode = request->admitted_rows == 1u || !request->wide_allowed
        ? SALT_AREA_SCAN_M1
        : request->admitted_rows < request->requested_rows
            ? SALT_AREA_SCAN_MK : SALT_AREA_SCAN_MN;
    built.requested_rows = request->requested_rows;
    built.active_rows = request->admitted_rows;
    built.rows_per_wave = built.mode == SALT_AREA_SCAN_M1
        ? 1u : request->admitted_rows;
    built.output_row_tiles = request->output_row_tiles;
    workers = built.mode == SALT_AREA_SCAN_M1
        ? request->narrow_workers : request->active_worker_limit;
    if (workers == 0 || workers > UINT32_MAX) return -1;
    built.output_row_owned =
        request->output_row_tiles >= workers ? 1u : 0u;
    built.candidate_row_split =
        !built.output_row_owned && built.active_rows > 1u ? 1u : 0u;
    if (built.candidate_row_split) {
        if (workers > request->candidate_output_tiles)
            workers = request->candidate_output_tiles;
    } else if (workers > request->output_row_tiles) {
        workers = request->output_row_tiles;
    }
    if (workers == 0 || workers > UINT32_MAX) return -1;
    built.active_workers = (uint32_t)workers;
    built.matrix_parallel = built.active_rows > 1u ? 1u : 0u;
    built.candidate_output_tiles = request->candidate_output_tiles;
    *plan = built;
    return 0;
}

int salt_area_frontier_bind(SaltAreaFrontier *frontier,
                            SaltAreaTask *tasks, uint32_t capacity) {
    if (!frontier || !tasks || capacity == 0) return -1;
    memset(tasks, 0, (size_t)capacity * sizeof(*tasks));
    memset(frontier, 0, sizeof *frontier);
    frontier->tasks = tasks;
    frontier->capacity = capacity;
    return 0;
}

int salt_area_frontier_reset(SaltAreaFrontier *frontier,
                             uint64_t generation) {
    if (!frontier || !frontier->tasks || frontier->capacity == 0 ||
        generation == 0)
        return -1;
    memset(frontier->tasks, 0,
        (size_t)frontier->capacity * sizeof(*frontier->tasks));
    frontier->count = 0;
    frontier->generation = generation;
    return 0;
}

int salt_area_frontier_append(SaltAreaFrontier *frontier,
                              const SaltAreaTask *task) {
    if (!frontier || !frontier->tasks || !task ||
        frontier->generation == 0 || task->generation != frontier->generation ||
        frontier->count >= frontier->capacity || task->candidate_count == 0 ||
        task->output_count == 0 || task->state != SALT_AREA_TASK_QUEUED)
        return -1;
    frontier->tasks[frontier->count++] = *task;
    return 0;
}

int salt_area_frontier_cancel_except(SaltAreaFrontier *frontier,
                                     uint64_t generation,
                                     uint32_t survivor_node) {
    if (!frontier || !frontier->tasks || generation == 0 ||
        frontier->generation != generation)
        return -1;
    for (uint32_t index = 0; index < frontier->count; index++) {
        SaltAreaTask *task = &frontier->tasks[index];
        uint32_t cursor = survivor_node;
        int survives = 0;
        if (task->generation != generation) continue;
        for (uint32_t step = 0; step <= frontier->count; step++) {
            uint32_t parent = UINT32_MAX;
            if (task->node_id == cursor) {
                survives = 1;
                break;
            }
            if (cursor == UINT32_MAX) break;
            for (uint32_t scan = 0; scan < frontier->count; scan++)
                if (frontier->tasks[scan].generation == generation &&
                    frontier->tasks[scan].node_id == cursor) {
                    parent = frontier->tasks[scan].parent_id;
                    break;
                }
            if (parent == cursor) return -1;
            cursor = parent;
        }
        if (survives) continue;
        if (task->state == SALT_AREA_TASK_QUEUED)
            task->state = SALT_AREA_TASK_CANCELLED;
        else if (task->state == SALT_AREA_TASK_RUNNING)
            task->flags |= SALT_AREA_TASK_CANCEL_REQUESTED;
    }
    return 0;
}

typedef struct SaltAreaWfqPhaseTask {
    SaltAreaWfqRuntime *runtime;
    SaltAreaWfqItemExecute item_execute;
    void *item_context;
    uint32_t active_workers;
    uint32_t branch_for_worker[32];
    int failed[32];
} SaltAreaWfqPhaseTask;

static void area_wfq_worker(int worker, void *opaque) {
    SaltAreaWfqPhaseTask *task = (SaltAreaWfqPhaseTask *)opaque;
    SaltAreaWfqRuntime *runtime;
    SaltAreaWfqBranch *branch;
    SaltAreaWfqItem *item;
    SaltAreaTask *compute_task;
    SaltAreaWfqItemResult *result;
    uint32_t branch_index, index;
    if (!task || !(runtime = task->runtime) || worker < 0 ||
        (uint32_t)worker >= task->active_workers)
        return;
    branch_index = task->branch_for_worker[(uint32_t)worker];
    if (branch_index >= runtime->plan.frontier_width) {
        task->failed[worker] = 1;
        return;
    }
    branch = &runtime->branches[branch_index];
    if (!branch->active || branch->owner_worker != (uint32_t)worker ||
        branch->next_queue >= runtime->plan.queue_length)
        return;
    index = branch_index * runtime->plan.queue_length +
        branch->next_queue;
    if (index >= runtime->item_capacity) {
        task->failed[worker] = 1;
        return;
    }
    item = &runtime->items[index];
    result = &runtime->results[branch_index];
    compute_task = item->task;
    if (!compute_task || item->generation != runtime->generation ||
        item->branch_index != branch_index ||
        item->queue_index != branch->next_queue ||
        item->owner_worker != (uint32_t)worker ||
        (item->state != SALT_AREA_WFQ_ITEM_QUEUED &&
         item->state != SALT_AREA_WFQ_ITEM_WAITING) ||
        compute_task->generation != runtime->generation ||
        compute_task->candidate_count == 0 || compute_task->output_count == 0 ||
        compute_task->state != SALT_AREA_TASK_QUEUED) {
        task->failed[worker] = 1;
        return;
    }
    memset(result, 0, sizeof *result);
    item->state = SALT_AREA_WFQ_ITEM_RUNNING;
    compute_task->state = SALT_AREA_TASK_RUNNING;
    if (task->item_execute(task->item_context, item, result) != 0 ||
        result->outcome < SALT_AREA_WFQ_CONTINUE ||
        result->outcome > SALT_AREA_WFQ_STALE) {
        task->failed[worker] = 1;
        return;
    }
    if (compute_task->generation != runtime->generation ||
        compute_task->state == SALT_AREA_TASK_CANCELLED ||
        (compute_task->flags & SALT_AREA_TASK_CANCEL_REQUESTED) != 0u)
        result->outcome = SALT_AREA_WFQ_STALE;
}

static uint32_t area_wfq_retire_branch(
        SaltAreaWfqRuntime *runtime, uint32_t branch_index,
        uint32_t first_queue) {
    uint32_t retired = 0;
    if (!runtime || branch_index >= runtime->plan.frontier_width ||
        first_queue > runtime->plan.queue_length)
        return 0;
    for (uint32_t queue = first_queue;
         queue < runtime->plan.queue_length; queue++) {
        SaltAreaWfqItem *item = &runtime->items[
            branch_index * runtime->plan.queue_length + queue];
        if (item->generation == runtime->generation &&
            (item->state == SALT_AREA_WFQ_ITEM_QUEUED ||
             item->state == SALT_AREA_WFQ_ITEM_WAITING ||
             item->state == SALT_AREA_WFQ_ITEM_RUNNING)) {
            if (item->task &&
                item->task->generation == runtime->generation) {
                if (item->task->state == SALT_AREA_TASK_QUEUED)
                    item->task->state = SALT_AREA_TASK_CANCELLED;
                else if (item->task->state == SALT_AREA_TASK_RUNNING)
                    item->task->flags |= SALT_AREA_TASK_CANCEL_REQUESTED;
            }
            item->state = SALT_AREA_WFQ_ITEM_RETIRED;
            retired++;
        }
    }
    runtime->branches[branch_index].active = 0;
    return retired;
}

int salt_area_wfq_bind(
        SaltAreaWfqRuntime *runtime,
        SaltAreaWfqBranch *branches, uint32_t branch_capacity,
        SaltAreaWfqItem *items, uint32_t item_capacity,
        SaltAreaWfqItemResult *results, uint32_t result_capacity,
        SaltAreaTask *tasks, uint32_t task_capacity) {
    if (!runtime || !branches || !items || !results || !tasks ||
        branch_capacity == 0 || item_capacity == 0 || result_capacity == 0 ||
        task_capacity == 0)
        return -1;
    memset(runtime, 0, sizeof *runtime);
    memset(branches, 0, (size_t)branch_capacity * sizeof *branches);
    memset(items, 0, (size_t)item_capacity * sizeof *items);
    memset(results, 0, (size_t)result_capacity * sizeof *results);
    runtime->branches = branches;
    runtime->items = items;
    runtime->results = results;
    runtime->tasks = tasks;
    runtime->branch_capacity = branch_capacity;
    runtime->item_capacity = item_capacity;
    runtime->result_capacity = result_capacity;
    runtime->task_capacity = task_capacity;
    runtime->ledger.winning_branch = UINT32_MAX;
    runtime->ledger.winning_queue = UINT32_MAX;
    runtime->ledger.target_token = -1;
    runtime->ledger.target_state_slot = UINT32_MAX;
    return 0;
}

int salt_area_wfq_start(
        SaltAreaWfqRuntime *runtime, const SaltAreaWfqPlan *plan,
        uint64_t generation, const uint32_t *seed_ids,
        uint32_t seed_count) {
    uint64_t required_items;
    if (!runtime || !runtime->branches || !runtime->items ||
        !runtime->results || !runtime->tasks || !plan || !seed_ids || generation == 0 ||
        runtime->ready || plan->workers == 0 || plan->workers > 32u ||
        plan->frontier_width == 0 ||
        plan->frontier_width > plan->workers ||
        plan->queue_length == 0 || seed_count != plan->frontier_width ||
        plan->frontier_width > runtime->branch_capacity ||
        plan->frontier_width > runtime->result_capacity ||
        (runtime->ledger.generation != 0 &&
         generation != runtime->ledger.generation))
        return -1;
    required_items = (uint64_t)plan->frontier_width * plan->queue_length;
    if (required_items > runtime->item_capacity ||
        required_items > runtime->task_capacity)
        return -1;
    for (uint32_t branch = 0; branch < seed_count; branch++)
        for (uint32_t prior = 0; prior < branch; prior++)
            if (seed_ids[prior] == seed_ids[branch]) return -1;
    for (uint32_t index = 0; index < (uint32_t)required_items; index++)
        if (runtime->tasks[index].generation != generation ||
            runtime->tasks[index].candidate_count == 0 ||
            runtime->tasks[index].output_count == 0 ||
            runtime->tasks[index].state != SALT_AREA_TASK_QUEUED ||
            runtime->tasks[index].flags != 0u)
            return -1;
    memset(runtime->branches, 0,
        (size_t)runtime->branch_capacity * sizeof *runtime->branches);
    memset(runtime->items, 0,
        (size_t)runtime->item_capacity * sizeof *runtime->items);
    memset(runtime->results, 0,
        (size_t)runtime->result_capacity * sizeof *runtime->results);
    runtime->plan = *plan;
    runtime->generation = generation;
    for (uint32_t branch = 0; branch < plan->frontier_width; branch++) {
        runtime->branches[branch].seed_id = seed_ids[branch];
        runtime->branches[branch].owner_worker = branch;
        runtime->branches[branch].next_queue = 0;
        runtime->branches[branch].active = 1;
        for (uint32_t queue = 0; queue < plan->queue_length; queue++) {
            SaltAreaWfqItem *item = &runtime->items[
                branch * plan->queue_length + queue];
            item->generation = generation;
            item->branch_index = branch;
            item->queue_index = queue;
            item->owner_worker = branch;
            item->state = SALT_AREA_WFQ_ITEM_QUEUED;
            item->task = &runtime->tasks[
                branch * plan->queue_length + queue];
        }
    }
    runtime->ready = 1;
    return 0;
}

int salt_area_wfq_execute(
        SaltAreaWfqRuntime *runtime,
        SaltAreaWfqPoolRun pool_run, void *pool_context,
        SaltAreaWfqItemExecute item_execute, void *item_context,
        SaltAreaWfqResult *result) {
    SaltAreaWfqResult built;
    if (result) memset(result, 0, sizeof *result);
    if (!runtime || !runtime->ready || !pool_run || !pool_context ||
        !item_execute || !result)
        return -1;
    memset(&built, 0, sizeof built);
    built.status = SALT_AREA_WFQ_INVALID;
    for (;;) {
        SaltAreaWfqPhaseTask task;
        uint32_t active = 0, winner = UINT32_MAX;
        uint32_t winner_queue = UINT32_MAX, active_workers;
        int32_t target_token = -1;
        uint32_t target_state_slot = UINT32_MAX;
        int waiting = 0;
        for (uint32_t branch = 0;
             branch < runtime->plan.frontier_width; branch++)
            if (runtime->branches[branch].active &&
                runtime->branches[branch].next_queue <
                    runtime->plan.queue_length)
                active++;
        if (active == 0) {
            built.status = SALT_AREA_WFQ_EXHAUSTED;
            built.ledger = runtime->ledger;
            runtime->ready = 0;
            *result = built;
            return 0;
        }
        memset(&task, 0, sizeof task);
        memset(runtime->results, 0,
            (size_t)runtime->result_capacity * sizeof *runtime->results);
        task.runtime = runtime;
        task.item_execute = item_execute;
        task.item_context = item_context;
        active_workers = 0;
        for (uint32_t branch = 0;
             branch < runtime->plan.frontier_width; branch++) {
            SaltAreaWfqBranch *branch_state = &runtime->branches[branch];
            if (!branch_state->active ||
                branch_state->next_queue >= runtime->plan.queue_length)
                continue;
            if (active_workers >= 32u) {
                built.status = SALT_AREA_WFQ_FAILURE;
                runtime->ready = 0;
                *result = built;
                return -1;
            }
            task.branch_for_worker[active_workers] = branch;
            branch_state->owner_worker = active_workers;
            for (uint32_t queue = branch_state->next_queue;
                 queue < runtime->plan.queue_length; queue++) {
                SaltAreaWfqItem *item = &runtime->items[
                    branch * runtime->plan.queue_length + queue];
                if (item->generation == runtime->generation &&
                    (item->state == SALT_AREA_WFQ_ITEM_QUEUED ||
                     item->state == SALT_AREA_WFQ_ITEM_WAITING))
                    item->owner_worker = active_workers;
            }
            active_workers++;
        }
        task.active_workers = active_workers;
        if (pool_run(pool_context, (int)active_workers,
                area_wfq_worker, &task) != 0) {
            built.status = SALT_AREA_WFQ_FAILURE;
            runtime->ready = 0;
            *result = built;
            return -1;
        }
        for (uint32_t worker = 0; worker < active_workers; worker++)
            if (task.failed[worker]) {
                built.status = SALT_AREA_WFQ_FAILURE;
                runtime->ready = 0;
                *result = built;
                return -1;
            }
        for (uint32_t branch = 0;
             branch < runtime->plan.frontier_width; branch++) {
            SaltAreaWfqBranch *branch_state = &runtime->branches[branch];
            SaltAreaWfqItemResult *item_result = &runtime->results[branch];
            SaltAreaWfqItem *item;
            uint32_t queue;
            if (!branch_state->active ||
                branch_state->next_queue >= runtime->plan.queue_length)
                continue;
            queue = branch_state->next_queue;
            item = &runtime->items[
                branch * runtime->plan.queue_length + queue];
            built.executed_items++;
            if (item_result->outcome == SALT_AREA_WFQ_FATAL) {
                built.status = SALT_AREA_WFQ_FAILURE;
                runtime->ready = 0;
                *result = built;
                return -1;
            }
            if (item_result->outcome == SALT_AREA_WFQ_TARGET_WIN) {
                item->state = SALT_AREA_WFQ_ITEM_DONE;
                item->task->state = SALT_AREA_TASK_DONE;
                if (winner == UINT32_MAX) {
                    winner = branch;
                    winner_queue = queue;
                    target_token = item_result->target_token;
                    target_state_slot = item_result->target_state_slot;
                } else if (target_token != item_result->target_token ||
                           target_state_slot != item_result->target_state_slot) {
                    built.status = SALT_AREA_WFQ_FAILURE;
                    runtime->ready = 0;
                    *result = built;
                    return -1;
                }
            } else if (item_result->outcome ==
                    SALT_AREA_WFQ_BRANCH_FAIL) {
                item->task->state = SALT_AREA_TASK_DONE;
                item->state = SALT_AREA_WFQ_ITEM_RETIRED;
                built.retired_items++;
                built.retired_items += area_wfq_retire_branch(
                    runtime, branch, queue + 1u);
            } else if (item_result->outcome ==
                    SALT_AREA_WFQ_RESOURCE_WAIT) {
                item->state = SALT_AREA_WFQ_ITEM_WAITING;
                item->task->state = SALT_AREA_TASK_QUEUED;
                waiting = 1;
            } else if (item_result->outcome == SALT_AREA_WFQ_STALE) {
                item->state = SALT_AREA_WFQ_ITEM_RETIRED;
                item->task->state = SALT_AREA_TASK_CANCELLED;
                built.retired_items++;
                built.retired_items += area_wfq_retire_branch(
                    runtime, branch, queue + 1u);
            } else {
                item->state = SALT_AREA_WFQ_ITEM_DONE;
                item->task->state = SALT_AREA_TASK_DONE;
                branch_state->next_queue++;
                if (branch_state->next_queue >=
                        runtime->plan.queue_length)
                    branch_state->active = 0;
            }
        }
        if (winner != UINT32_MAX) {
            if (runtime->generation == UINT64_MAX) {
                built.status = SALT_AREA_WFQ_FAILURE;
                runtime->ready = 0;
                *result = built;
                return -1;
            }
            for (uint32_t branch = 0;
                 branch < runtime->plan.frontier_width; branch++)
                built.retired_items += area_wfq_retire_branch(
                    runtime, branch, 0u);
            runtime->ledger.generation = runtime->generation + 1u;
            runtime->ledger.winning_branch = winner;
            runtime->ledger.winning_queue = winner_queue;
            runtime->ledger.target_token = target_token;
            runtime->ledger.target_state_slot = target_state_slot;
            built.status = SALT_AREA_WFQ_WIN;
            built.ledger = runtime->ledger;
            runtime->ready = 0;
            *result = built;
            return 0;
        }
        if (waiting) {
            built.status = SALT_AREA_WFQ_NEED_RESOURCE;
            built.ledger = runtime->ledger;
            for (uint32_t branch = 0;
                 branch < runtime->plan.frontier_width; branch++)
                if (runtime->branches[branch].active)
                    built.active_branches++;
            *result = built;
            return 0;
        }
    }
}

static uint32_t area_nfq_item_index(
        const SaltAreaNfqPlan *plan, uint32_t route_slot,
        uint32_t sequence_index, uint32_t queue_index) {
    return ((route_slot * plan->sequence_tiles + sequence_index) *
        plan->queue_length) + queue_index;
}

static int area_nfq_rebuild_worker_masks(SaltAreaNfqMatrix *matrix) {
    if (!matrix || !matrix->routes || !matrix->items ||
        matrix->plan.workers == 0 || matrix->plan.workers > 32u)
        return -1;
    for (uint32_t route = 0; route < matrix->plan.route_count; route++)
        matrix->routes[route].worker_mask = 0u;
    for (uint32_t index = 0; index < matrix->item_count; index++) {
        const SaltAreaNfqItem *item = &matrix->items[index];
        if (item->state != SALT_AREA_WFQ_ITEM_RUNNING ||
            item->owner_worker >= matrix->plan.workers ||
            item->route_slot >= matrix->plan.route_count ||
            !matrix->routes[item->route_slot].active)
            continue;
        matrix->routes[item->route_slot].worker_mask |=
            (uint32_t)1u << item->owner_worker;
    }
    return 0;
}

int salt_area_nfq_bind(
        SaltAreaNfqMatrix *matrix,
        SaltAreaNfqRoute *routes, uint32_t route_capacity,
        SaltAreaNfqItem *items, uint32_t item_capacity) {
    if (!matrix || !routes || route_capacity == 0 ||
        !items || item_capacity == 0)
        return -1;
    memset(matrix, 0, sizeof *matrix);
    memset(routes, 0, (size_t)route_capacity * sizeof *routes);
    memset(items, 0, (size_t)item_capacity * sizeof *items);
    matrix->routes = routes;
    matrix->items = items;
    matrix->route_capacity = route_capacity;
    matrix->item_capacity = item_capacity;
    return 0;
}

int salt_area_nfq_start(
        SaltAreaNfqMatrix *matrix, const SaltAreaNfqPlan *plan,
        uint64_t epoch_generation, uint64_t parent_generation,
        uint32_t parent_position, const uint32_t *route_ids,
        uint32_t route_id_count) {
    uint64_t width, required;
    if (!matrix || !matrix->routes || !matrix->items || !plan ||
        !route_ids || epoch_generation == 0 ||
        epoch_generation <= parent_generation || matrix->ready ||
        plan->workers == 0 || plan->workers > 32u ||
        plan->sequence_tiles == 0 || plan->route_count == 0 ||
        plan->queue_length == 0 || route_id_count != plan->route_count ||
        plan->route_count > matrix->route_capacity ||
        (matrix->epoch_generation != 0 &&
         epoch_generation <= matrix->epoch_generation))
        return -1;
    width = (uint64_t)plan->sequence_tiles * plan->route_count;
    if (width < plan->workers || width % plan->workers != 0u ||
        width > UINT64_MAX / plan->queue_length)
        return -1;
    required = width * plan->queue_length;
    if (
        required == 0 || required > matrix->item_capacity ||
        required > UINT32_MAX)
        return -1;
    for (uint32_t route = 0; route < route_id_count; route++)
        for (uint32_t prior = 0; prior < route; prior++)
            if (route_ids[prior] == route_ids[route]) return -1;
    memset(matrix->routes, 0,
        (size_t)matrix->route_capacity * sizeof *matrix->routes);
    memset(matrix->items, 0,
        (size_t)matrix->item_capacity * sizeof *matrix->items);
    matrix->plan = *plan;
    matrix->epoch_generation = epoch_generation;
    matrix->parent_generation = parent_generation;
    matrix->parent_position = parent_position;
    matrix->item_count = (uint32_t)required;
    matrix->resolved = 0;
    for (uint32_t route = 0; route < plan->route_count; route++) {
        matrix->routes[route].route_id = route_ids[route];
        matrix->routes[route].version = 1u;
        matrix->routes[route].active = 1u;
        for (uint32_t sequence = 0;
             sequence < plan->sequence_tiles; sequence++)
            for (uint32_t queue = 0;
                 queue < plan->queue_length; queue++) {
                uint32_t index = area_nfq_item_index(
                    plan, route, sequence, queue);
                SaltAreaNfqItem *item = &matrix->items[index];
                item->epoch_generation = epoch_generation;
                item->parent_generation = parent_generation;
                item->route_slot = route;
                item->route_id = route_ids[route];
                item->route_version = 1u;
                item->sequence_index = sequence;
                item->queue_index = queue;
                item->owner_worker = UINT32_MAX;
                item->state = queue == 0u
                    ? SALT_AREA_WFQ_ITEM_QUEUED
                    : SALT_AREA_WFQ_ITEM_WAITING;
            }
    }
    if (area_nfq_rebuild_worker_masks(matrix) != 0) return -1;
    matrix->ready = 1;
    return 0;
}

int salt_area_nfq_retire_route(
        SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
        uint32_t route_slot, uint32_t *retired_items) {
    uint32_t retired = 0;
    if (retired_items) *retired_items = 0;
    if (!matrix || !matrix->ready || matrix->resolved ||
        epoch_generation == 0 ||
        epoch_generation != matrix->epoch_generation ||
        route_slot >= matrix->plan.route_count ||
        !matrix->routes[route_slot].active)
        return -1;
    for (uint32_t sequence = 0;
         sequence < matrix->plan.sequence_tiles; sequence++)
        for (uint32_t queue = 0;
             queue < matrix->plan.queue_length; queue++) {
            SaltAreaNfqItem *item = &matrix->items[area_nfq_item_index(
                &matrix->plan, route_slot, sequence, queue)];
            if (item->state == SALT_AREA_WFQ_ITEM_QUEUED ||
                item->state == SALT_AREA_WFQ_ITEM_RUNNING ||
                item->state == SALT_AREA_WFQ_ITEM_WAITING ||
                item->state == SALT_AREA_WFQ_ITEM_DONE) {
                item->state = SALT_AREA_WFQ_ITEM_RETIRED;
                item->owner_worker = UINT32_MAX;
                retired++;
            }
        }
    matrix->routes[route_slot].active = 0u;
    if (area_nfq_rebuild_worker_masks(matrix) != 0) return -1;
    if (retired_items) *retired_items = retired;
    return 0;
}

int salt_area_nfq_reschedule_route(
        SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
        uint64_t parent_generation, uint32_t route_slot,
        uint32_t new_route_id) {
    SaltAreaNfqRoute *route;
    if (!matrix || !matrix->ready || matrix->resolved ||
        epoch_generation == 0 ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation ||
        route_slot >= matrix->plan.route_count ||
        matrix->routes[route_slot].active)
        return -1;
    for (uint32_t slot = 0; slot < matrix->plan.route_count; slot++)
        if (matrix->routes[slot].active &&
            matrix->routes[slot].route_id == new_route_id)
            return -1;
    route = &matrix->routes[route_slot];
    if (route->version == UINT32_MAX) return -1;
    route->route_id = new_route_id;
    route->version++;
    route->active = 1u;
    for (uint32_t sequence = 0;
         sequence < matrix->plan.sequence_tiles; sequence++)
        for (uint32_t queue = 0;
             queue < matrix->plan.queue_length; queue++) {
            SaltAreaNfqItem *item = &matrix->items[area_nfq_item_index(
                &matrix->plan, route_slot, sequence, queue)];
            item->epoch_generation = epoch_generation;
            item->parent_generation = parent_generation;
            item->route_slot = route_slot;
            item->route_id = new_route_id;
            item->route_version = route->version;
            item->sequence_index = sequence;
            item->queue_index = queue;
            item->owner_worker = UINT32_MAX;
            item->state = queue == 0u
                ? SALT_AREA_WFQ_ITEM_QUEUED
                : SALT_AREA_WFQ_ITEM_WAITING;
        }
    return area_nfq_rebuild_worker_masks(matrix);
}

int salt_area_nfq_resolve(
        SaltAreaNfqMatrix *matrix, uint64_t epoch_generation,
        uint64_t parent_generation, uint32_t winning_route_slot,
        uint32_t winning_sequence_index, uint32_t winning_queue_index,
        uint32_t *retired_items) {
    uint32_t winner, retired = 0;
    if (retired_items) *retired_items = 0;
    if (!matrix || !matrix->ready || matrix->resolved ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation ||
        winning_route_slot >= matrix->plan.route_count ||
        winning_sequence_index >= matrix->plan.sequence_tiles ||
        winning_queue_index >= matrix->plan.queue_length ||
        !matrix->routes[winning_route_slot].active)
        return -1;
    winner = area_nfq_item_index(
        &matrix->plan, winning_route_slot,
        winning_sequence_index, winning_queue_index);
    if (winner >= matrix->item_count ||
        matrix->items[winner].state == SALT_AREA_WFQ_ITEM_EMPTY ||
        matrix->items[winner].state == SALT_AREA_WFQ_ITEM_RETIRED)
        return -1;
    for (uint32_t index = 0; index < matrix->item_count; index++) {
        SaltAreaNfqItem *item = &matrix->items[index];
        if (index == winner) {
            item->state = SALT_AREA_WFQ_ITEM_DONE;
        } else if (item->state != SALT_AREA_WFQ_ITEM_EMPTY &&
                   item->state != SALT_AREA_WFQ_ITEM_RETIRED) {
            item->state = SALT_AREA_WFQ_ITEM_RETIRED;
            item->owner_worker = UINT32_MAX;
            retired++;
        }
    }
    for (uint32_t route = 0;
         route < matrix->plan.route_count; route++) {
        matrix->routes[route].active = 0u;
        matrix->routes[route].worker_mask = 0u;
    }
    matrix->items[winner].owner_worker = UINT32_MAX;
    matrix->resolved = 1;
    matrix->ready = 0;
    if (retired_items) *retired_items = retired;
    return 0;
}

int salt_area_nfq_publishable(
        const SaltAreaNfqMatrix *matrix, uint32_t item_index,
        uint64_t epoch_generation, uint64_t parent_generation) {
    const SaltAreaNfqItem *item;
    if (!matrix || !matrix->items || item_index >= matrix->item_count ||
        epoch_generation == 0)
        return -1;
    item = &matrix->items[item_index];
    if (!matrix->ready || matrix->resolved ||
        matrix->epoch_generation != epoch_generation ||
        matrix->parent_generation != parent_generation ||
        item->epoch_generation != epoch_generation ||
        item->parent_generation != parent_generation ||
        item->route_slot >= matrix->plan.route_count ||
        !matrix->routes[item->route_slot].active ||
        item->route_version != matrix->routes[item->route_slot].version ||
        item->route_id != matrix->routes[item->route_slot].route_id ||
        (item->state != SALT_AREA_WFQ_ITEM_RUNNING &&
         item->state != SALT_AREA_WFQ_ITEM_DONE))
        return 0;
    return 1;
}

int salt_area_nfq_take_ready(
        SaltAreaNfqMatrix *matrix,
        uint64_t epoch_generation, uint64_t parent_generation,
        uint32_t *item_indices, uint32_t item_capacity,
        uint32_t *item_count) {
    uint32_t count = 0;
    if (item_count) *item_count = 0u;
    if (!matrix || !matrix->ready || matrix->resolved || !matrix->items ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation ||
        !item_indices || item_capacity == 0 || !item_count)
        return -1;
    for (uint32_t worker = 0;
         worker < matrix->plan.workers && count < item_capacity; worker++)
        for (uint32_t queue = 0;
             queue < matrix->plan.queue_length; queue++) {
            int found = 0;
            for (uint32_t index = 0; index < matrix->item_count; index++) {
                SaltAreaNfqItem *item = &matrix->items[index];
                if (item->state != SALT_AREA_WFQ_ITEM_QUEUED ||
                    item->queue_index != queue ||
                    item->owner_worker != UINT32_MAX ||
                    item->epoch_generation != epoch_generation ||
                    item->parent_generation != parent_generation ||
                    item->route_slot >= matrix->plan.route_count ||
                    !matrix->routes[item->route_slot].active ||
                    item->route_version !=
                        matrix->routes[item->route_slot].version)
                    continue;
                item->state = SALT_AREA_WFQ_ITEM_RUNNING;
                item->owner_worker = worker;
                matrix->routes[item->route_slot].worker_mask |=
                    (uint32_t)1u << worker;
                item_indices[count++] = index;
                found = 1;
                break;
            }
            if (found) break;
        }
    *item_count = count;
    return 0;
}

int salt_area_nfq_complete(
        SaltAreaNfqMatrix *matrix,
        uint64_t epoch_generation, uint64_t parent_generation,
        uint32_t item_index, SaltAreaWfqOutcome outcome,
        uint32_t *retired_items) {
    SaltAreaNfqItem *item;
    if (retired_items) *retired_items = 0u;
    if (!matrix || !matrix->ready || matrix->resolved ||
        item_index >= matrix->item_count ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation ||
        outcome < SALT_AREA_WFQ_CONTINUE || outcome > SALT_AREA_WFQ_STALE)
        return -1;
    item = &matrix->items[item_index];
    if (item->state != SALT_AREA_WFQ_ITEM_RUNNING ||
        item->epoch_generation != epoch_generation ||
        item->parent_generation != parent_generation ||
        item->route_slot >= matrix->plan.route_count ||
        !matrix->routes[item->route_slot].active ||
        item->route_version != matrix->routes[item->route_slot].version)
        return -1;
    if (outcome == SALT_AREA_WFQ_BRANCH_FAIL)
        return salt_area_nfq_retire_route(
            matrix, epoch_generation, item->route_slot, retired_items);
    if (outcome == SALT_AREA_WFQ_RESOURCE_WAIT) {
        item->state = SALT_AREA_WFQ_ITEM_WAITING;
        item->owner_worker = UINT32_MAX;
        return area_nfq_rebuild_worker_masks(matrix);
    }
    if (outcome == SALT_AREA_WFQ_STALE) {
        item->state = SALT_AREA_WFQ_ITEM_RETIRED;
        item->owner_worker = UINT32_MAX;
        if (retired_items) *retired_items = 1u;
        return area_nfq_rebuild_worker_masks(matrix);
    }
    if (outcome == SALT_AREA_WFQ_FATAL) return -1;
    item->state = SALT_AREA_WFQ_ITEM_DONE;
    item->owner_worker = UINT32_MAX;
    if (outcome == SALT_AREA_WFQ_CONTINUE &&
        item->queue_index + 1u < matrix->plan.queue_length) {
        uint32_t next = area_nfq_item_index(
            &matrix->plan, item->route_slot,
            item->sequence_index, item->queue_index + 1u);
        if (next >= matrix->item_count ||
            matrix->items[next].state != SALT_AREA_WFQ_ITEM_WAITING)
            return -1;
        matrix->items[next].state = SALT_AREA_WFQ_ITEM_QUEUED;
        matrix->items[next].owner_worker = UINT32_MAX;
    }
    return area_nfq_rebuild_worker_masks(matrix);
}

int salt_area_nfq_refire(
        SaltAreaNfqMatrix *matrix,
        uint64_t epoch_generation, uint64_t parent_generation,
        uint32_t item_index) {
    SaltAreaNfqItem *item;
    if (!matrix || !matrix->ready || matrix->resolved ||
        item_index >= matrix->item_count ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation)
        return -1;
    item = &matrix->items[item_index];
    if (item->state != SALT_AREA_WFQ_ITEM_WAITING ||
        item->route_slot >= matrix->plan.route_count ||
        !matrix->routes[item->route_slot].active ||
        item->route_version != matrix->routes[item->route_slot].version)
        return -1;
    item->state = SALT_AREA_WFQ_ITEM_QUEUED;
    item->owner_worker = UINT32_MAX;
    return area_nfq_rebuild_worker_masks(matrix);
}

int salt_area_nfq_cancel(
        SaltAreaNfqMatrix *matrix,
        uint64_t epoch_generation, uint64_t parent_generation,
        uint32_t *retired_items) {
    uint32_t retired = 0;
    if (retired_items) *retired_items = 0u;
    if (!matrix || !matrix->ready || matrix->resolved ||
        epoch_generation != matrix->epoch_generation ||
        parent_generation != matrix->parent_generation)
        return -1;
    for (uint32_t index = 0; index < matrix->item_count; index++) {
        SaltAreaNfqItem *item = &matrix->items[index];
        if (item->state != SALT_AREA_WFQ_ITEM_EMPTY &&
            item->state != SALT_AREA_WFQ_ITEM_RETIRED) {
            item->state = SALT_AREA_WFQ_ITEM_RETIRED;
            item->owner_worker = UINT32_MAX;
            retired++;
        }
    }
    for (uint32_t route = 0; route < matrix->plan.route_count; route++) {
        matrix->routes[route].active = 0u;
        matrix->routes[route].worker_mask = 0u;
    }
    matrix->resolved = 1;
    matrix->ready = 0;
    if (retired_items) *retired_items = retired;
    return 0;
}
