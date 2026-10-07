#define _POSIX_C_SOURCE 200809L

#include "salt/text_exec.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

/* Same compiler TLS adapter as the C99 persistent pool; no allocation. */
static __thread const SaltTextPrefillTeam *text_prefill_team;

const SaltTextPrefillTeam *salt_text_prefill_team_current(void) {
    return text_prefill_team;
}

int salt_text_prefill_team_owner(void) {
    return !text_prefill_team || text_prefill_team->worker == 0u;
}

int salt_text_prefill_team_sync(int status) {
    const SaltTextPrefillTeam *team = text_prefill_team;
    int failed;
    if (!team) return status;
    if (status) __atomic_store_n(&team->graph->failed, 1u, __ATOMIC_RELEASE);
    salt_tensor_host_graph_barrier(team->graph);
    failed = (int)__atomic_load_n(&team->graph->failed, __ATOMIC_ACQUIRE);
    salt_tensor_host_graph_barrier(team->graph);
    return failed ? -1 : 0;
}

int salt_text_prefill_team_once(int (*operation)(void *), void *context) {
    const SaltTextPrefillTeam *team = text_prefill_team;
    int rc = 0;
    if (!operation || !context) return salt_text_prefill_team_sync(-1);
    if (!team) return operation(context);
    if (team->worker == 0u) {
        /* Inside this complete physical operation, existing single-owner
         * preparation/retirement runs normally, not as nested team barriers. */
        text_prefill_team = NULL;
        rc = operation(context);
        text_prefill_team = team;
    }
    return salt_text_prefill_team_sync(rc);
}

int salt_text_prefill_team_slice(int workers,
        void (*operation)(int, void *), void *context) {
    const SaltTextPrefillTeam *team = text_prefill_team;
    if (!team || !operation || !context || workers < 1 ||
        (uint32_t)workers > team->workers)
        return salt_text_prefill_team_sync(-1);
    if (team->worker < (uint32_t)workers)
        operation((int)team->worker, context);
    return salt_text_prefill_team_sync(0);
}

static double text_now_s(void) {
    struct timespec time_value;
    if (clock_gettime(CLOCK_MONOTONIC, &time_value) != 0) return 0.0;
    return (double)time_value.tv_sec + (double)time_value.tv_nsec * 1e-9;
}

int salt_text_phase_scope_run(
        const SaltTextPhaseScope *scope, SaltTextRuntimePhase phase,
        int (*operation)(void *operation_context), void *operation_context) {
    int operation_status, end_status;
    if (!scope || !scope->ops || !scope->ops->begin || !scope->ops->end ||
        !operation || phase < SALT_TEXT_PHASE_PREFILL ||
        phase > SALT_TEXT_PHASE_TARGET_VERIFY)
        return -1;
    if (scope->ops->begin(scope->context, phase) != 0) return -1;
    operation_status = operation(operation_context);
    end_status = scope->ops->end(scope->context, phase, operation_status);
    return operation_status == 0 && end_status == 0 ? 0 : -1;
}

static int text_phase_lookahead_active(
        const SaltTextPhaseLookahead *lookahead) {
    return lookahead &&
        (lookahead->status == SALT_TEXT_LOOKAHEAD_ELIGIBLE ||
         lookahead->status == SALT_TEXT_LOOKAHEAD_PREPARED ||
         lookahead->status == SALT_TEXT_LOOKAHEAD_READY);
}

int salt_text_phase_lookahead_cancel(SaltTextPhaseLookahead *lookahead) {
    int rc = 0;
    if (!lookahead) return -1;
    if (text_phase_lookahead_active(lookahead)) {
        if (!lookahead->ops || !lookahead->ops->cancel ||
                !lookahead->context)
            rc = -1;
        else
            rc = lookahead->ops->cancel(
                lookahead->context, &lookahead->request);
    }
    lookahead->status = SALT_TEXT_LOOKAHEAD_CANCELLED;
    lookahead->ops = NULL;
    lookahead->context = NULL;
    return rc == 0 ? 0 : -1;
}

int salt_text_phase_lookahead_claim(
        SaltTextPhaseLookahead *lookahead, SaltTextRuntimePhase phase,
        uint64_t generation, uint32_t position) {
    if (!lookahead || phase < SALT_TEXT_PHASE_PREFILL ||
            phase > SALT_TEXT_PHASE_TARGET_VERIFY)
        return -1;
    if (lookahead->status != SALT_TEXT_LOOKAHEAD_READY) {
        if (lookahead->status == SALT_TEXT_LOOKAHEAD_ELIGIBLE ||
                lookahead->status == SALT_TEXT_LOOKAHEAD_PREPARED) {
            if (salt_text_phase_lookahead_cancel(lookahead) != 0) return -1;
            return -1;
        }
        return 0;
    }
    if (lookahead->request.target_phase != phase ||
            lookahead->request.result_generation != generation ||
            lookahead->request.result_position != position) {
        if (salt_text_phase_lookahead_cancel(lookahead) != 0) return -1;
        return 0;
    }
    if (!lookahead->ops || !lookahead->ops->claim ||
            lookahead->ops->claim(
                lookahead->context, &lookahead->request) != 0) {
        (void)salt_text_phase_lookahead_cancel(lookahead);
        return -1;
    }
    lookahead->status = SALT_TEXT_LOOKAHEAD_CLAIMED;
    lookahead->ops = NULL;
    lookahead->context = NULL;
    return 1;
}

typedef enum {
    TEXT_PREFILL_CHUNK_BEGIN = 0,
    TEXT_PREFILL_EMBED,
    TEXT_PREFILL_ATTENTION,
    TEXT_PREFILL_FEED_FORWARD,
    TEXT_PREFILL_RELEASE,
    TEXT_PREFILL_CHUNK_FINISH,
    TEXT_PREFILL_NEXT_PREPARE,
    TEXT_PREFILL_FINAL_FINISH,
    TEXT_PREFILL_PUBLISH,
    TEXT_PREFILL_DONE,
    TEXT_PREFILL_FAILED,
} SaltTextPrefillPhase;

typedef struct {
    const SaltTextPrefillPlan *plan;
    const SaltTextPrefillOps *ops;
    void *context;
    int start_position;
    const int *tokens;
    int token_count;
    float *logits;
    float *states;
    float *branches;
    SaltTextPrefillProfile *profile;
    int completed;
    int batch;
    int chunk_start;
    int layer;
    int layer_rc;
    SaltTextPrefillPhase phase;
} SaltTextPrefillExecution;

static int text_prefill_prepare(SaltTextPrefillExecution *execution) {
    const SaltTextPrefillPlan *plan = execution->plan;
    const SaltTextPrefillOps *ops = execution->ops;
    size_t elements;
    if (!plan || !ops || !execution->context || !execution->tokens ||
        !execution->logits || !ops->embed || !ops->attention ||
        !ops->feed_forward || !ops->finish || !ops->publish_position ||
        !plan->scratch || !plan->scratch->states ||
        !plan->scratch->branches || plan->n_layers < 1 || plan->hidden < 1 ||
        plan->max_batch < 1 ||
        (plan->resource_policy != SALT_TEXT_RELEASE_PER_LAYER &&
         plan->resource_policy != SALT_TEXT_RETAIN_OPERATION) ||
        execution->token_count < 1 || execution->start_position < 0 ||
        execution->start_position > INT_MAX - execution->token_count ||
        (size_t)plan->max_batch > SIZE_MAX / (size_t)plan->hidden)
        return -1;
    elements = (size_t)plan->max_batch * (size_t)plan->hidden;
    if (elements > SIZE_MAX / sizeof(float) ||
        plan->scratch->element_capacity < elements)
        return -1;
    if ((plan->position_logits != NULL) != (ops->finish_batch != NULL) ||
        (plan->position_logits &&
         (plan->position_logits_width == 0 ||
          plan->position_logits_stride < plan->position_logits_width ||
          (size_t)execution->token_count >
              SIZE_MAX / plan->position_logits_stride)))
        return -1;
    if ((plan->lookahead != NULL) != (plan->lookahead_ops != NULL) ||
            (plan->lookahead &&
             (!plan->lookahead_context || !plan->lookahead_ops->prepare ||
              !plan->lookahead_ops->claim || !plan->lookahead_ops->cancel ||
              plan->transition_generation == UINT64_MAX ||
              plan->next_phase <= SALT_TEXT_PHASE_PREFILL ||
              plan->next_phase > SALT_TEXT_PHASE_TARGET_VERIFY)))
        return -1;
    execution->states = plan->scratch->states;
    execution->branches = plan->scratch->branches;
    return 0;
}

static int text_prefill_prepare_next_step(
        SaltTextPrefillExecution *execution) {
    SaltTextPhaseLookahead *lookahead = execution->plan->lookahead;
    int rc;
    if (!lookahead) return 0;
    if (text_phase_lookahead_active(lookahead) &&
            salt_text_phase_lookahead_cancel(lookahead) != 0)
        return -1;
    if (execution->plan->transition_generation == UINT64_MAX ||
            execution->start_position < 0 || execution->token_count < 1 ||
            execution->start_position > INT_MAX - execution->token_count)
        return -1;
    memset(&lookahead->request, 0, sizeof lookahead->request);
    lookahead->request.source_phase = SALT_TEXT_PHASE_PREFILL;
    lookahead->request.target_phase = execution->plan->next_phase;
    lookahead->request.source_generation =
        execution->plan->transition_generation;
    lookahead->request.result_generation =
        execution->plan->transition_generation + 1u;
    lookahead->request.source_position =
        (uint32_t)execution->start_position;
    lookahead->request.result_position =
        (uint32_t)(execution->start_position + execution->token_count);
    lookahead->ops = execution->plan->lookahead_ops;
    lookahead->context = execution->plan->lookahead_context;
    lookahead->status = SALT_TEXT_LOOKAHEAD_ELIGIBLE;
    rc = lookahead->ops->prepare(
        lookahead->context, &lookahead->request);
    if (rc == 1) {
        lookahead->status = SALT_TEXT_LOOKAHEAD_REFUSED;
        lookahead->ops = NULL;
        lookahead->context = NULL;
        return 0;
    }
    if (rc != 0) {
        (void)salt_text_phase_lookahead_cancel(lookahead);
        return -1;
    }
    lookahead->status = SALT_TEXT_LOOKAHEAD_PREPARED;
    return 0;
}

static int text_prefill_embed_step(
        SaltTextPrefillExecution *execution, int batch) {
    double start = execution->profile ? text_now_s() : 0.0;
    int rc = 0;
    if (salt_text_prefill_team_owner()) rc = execution->ops->embed(execution->context,
            execution->tokens + execution->completed, batch,
            execution->states);
    if (salt_text_prefill_team_sync(rc) != 0)
        return -1;
    if (execution->profile)
        execution->profile->embed_s += text_now_s() - start;
    return 0;
}

static int text_prefill_attention_step(
        SaltTextPrefillExecution *execution, int layer,
        int chunk_start, int batch) {
    double start = execution->profile ? text_now_s() : 0.0;
    int rc = execution->ops->attention(execution->context, layer,
        chunk_start, batch, execution->states, execution->branches);
    if (execution->profile)
        execution->profile->attention_s += text_now_s() - start;
    return rc;
}

static int text_prefill_feed_forward_step(
        SaltTextPrefillExecution *execution, int layer,
        int chunk_start, int batch) {
    double start = execution->profile ? text_now_s() : 0.0;
    int rc = execution->ops->feed_forward(execution->context, layer,
        chunk_start, batch, execution->branches, execution->states);
    if (execution->profile)
        execution->profile->feed_forward_s += text_now_s() - start;
    return rc;
}

static void text_prefill_release_step(
        SaltTextPrefillExecution *execution, int layer, int layer_rc) {
    double start = execution->profile ? text_now_s() : 0.0;
    if (salt_text_prefill_team_owner() && execution->ops->release_layer &&
            (layer_rc != 0 || execution->plan->resource_policy ==
                SALT_TEXT_RELEASE_PER_LAYER)) {
        execution->ops->release_layer(execution->context, layer);
        if (execution->profile) execution->profile->released_layers++;
    } else if (layer_rc == 0 && execution->plan->resource_policy ==
            SALT_TEXT_RETAIN_OPERATION) {
        if (execution->profile) execution->profile->retained_layers++;
    }
    if (execution->profile)
        execution->profile->release_s += text_now_s() - start;
    (void)salt_text_prefill_team_sync(0);
}

static int text_prefill_finish_batch_step(
        SaltTextPrefillExecution *execution, int batch) {
    double start;
    if (!execution->plan->position_logits) return 0;
    start = execution->profile ? text_now_s() : 0.0;
    if (execution->ops->finish_batch(execution->context,
            execution->states, batch,
            execution->plan->position_logits +
                (size_t)execution->completed *
                    execution->plan->position_logits_stride,
            execution->plan->position_logits_stride) != 0)
        return -1;
    if (execution->profile)
        execution->profile->finish_s += text_now_s() - start;
    return 0;
}

static int text_prefill_finish_step(SaltTextPrefillExecution *execution) {
    double start = execution->profile ? text_now_s() : 0.0;
    if (execution->plan->position_logits) {
        memcpy(execution->logits,
               execution->plan->position_logits +
                   (size_t)(execution->token_count - 1) *
                       execution->plan->position_logits_stride,
               execution->plan->position_logits_width *
                   sizeof *execution->logits);
    } else if (execution->ops->finish(execution->context,
                   execution->states +
                       (size_t)(execution->completed > 0
                           ? (execution->completed - 1) %
                               execution->plan->max_batch : 0) *
                           execution->plan->hidden,
                   execution->logits) != 0) {
        return -1;
    }
    if (execution->profile)
        execution->profile->finish_s += text_now_s() - start;
    return 0;
}

static int text_prefill_publish_step(SaltTextPrefillExecution *execution) {
    double start = execution->profile ? text_now_s() : 0.0;
    int position = execution->start_position + execution->token_count;
    int rc = 0;
    if (salt_text_prefill_team_owner()) rc = execution->ops->publish_position(
            execution->context, position);
    if (salt_text_prefill_team_sync(rc) != 0)
        return -1;
    if (salt_text_prefill_team_owner() && execution->plan->lookahead &&
            execution->plan->lookahead->status ==
                SALT_TEXT_LOOKAHEAD_PREPARED) {
        if (execution->plan->lookahead->request.result_position !=
                (uint32_t)position)
            return -1;
        execution->plan->lookahead->status = SALT_TEXT_LOOKAHEAD_READY;
    }
    if (execution->profile)
        execution->profile->publish_s += text_now_s() - start;
    return 0;
}

static void text_prefill_advance(SaltTextPrefillExecution *execution) {
    switch (execution->phase) {
    case TEXT_PREFILL_CHUNK_BEGIN:
        if (execution->completed >= execution->token_count) {
            execution->phase = TEXT_PREFILL_FINAL_FINISH;
            return;
        }
        execution->batch = execution->token_count - execution->completed;
        if (execution->batch > execution->plan->max_batch)
            execution->batch = execution->plan->max_batch;
        execution->chunk_start =
            execution->start_position + execution->completed;
        execution->layer = 0;
        execution->layer_rc = 0;
        if (execution->profile) {
            execution->profile->chunks++;
            execution->profile->grouped_chunks++;
        }
        execution->phase = TEXT_PREFILL_EMBED;
        return;
    case TEXT_PREFILL_EMBED:
        if (text_prefill_embed_step(execution, execution->batch) != 0) {
            execution->phase = TEXT_PREFILL_FAILED;
            return;
        }
        execution->phase = TEXT_PREFILL_ATTENTION;
        return;
    case TEXT_PREFILL_ATTENTION:
        execution->layer_rc = text_prefill_attention_step(execution,
            execution->layer, execution->chunk_start, execution->batch);
        execution->phase = execution->layer_rc == 0
            ? TEXT_PREFILL_FEED_FORWARD : TEXT_PREFILL_RELEASE;
        return;
    case TEXT_PREFILL_FEED_FORWARD:
        execution->layer_rc = text_prefill_feed_forward_step(execution,
            execution->layer, execution->chunk_start, execution->batch);
        execution->phase = TEXT_PREFILL_RELEASE;
        return;
    case TEXT_PREFILL_RELEASE:
        text_prefill_release_step(
            execution, execution->layer, execution->layer_rc);
        if (execution->layer_rc != 0) {
            execution->phase = TEXT_PREFILL_FAILED;
            return;
        }
        execution->layer++;
        execution->phase = execution->layer < execution->plan->n_layers
            ? TEXT_PREFILL_ATTENTION : TEXT_PREFILL_CHUNK_FINISH;
        return;
    case TEXT_PREFILL_CHUNK_FINISH:
        if (text_prefill_finish_batch_step(
                execution, execution->batch) != 0) {
            execution->phase = TEXT_PREFILL_FAILED;
            return;
        }
        execution->completed += execution->batch;
        execution->phase = TEXT_PREFILL_CHUNK_BEGIN;
        return;
    case TEXT_PREFILL_NEXT_PREPARE:
        {
        int rc = salt_text_prefill_team_owner()
            ? text_prefill_prepare_next_step(execution) : 0;
        execution->phase = salt_text_prefill_team_sync(rc) == 0
            ? TEXT_PREFILL_PUBLISH : TEXT_PREFILL_FAILED;
        return;
        }
    case TEXT_PREFILL_FINAL_FINISH:
        execution->phase = text_prefill_finish_step(execution) == 0
            ? TEXT_PREFILL_NEXT_PREPARE : TEXT_PREFILL_FAILED;
        return;
    case TEXT_PREFILL_PUBLISH:
        execution->phase = text_prefill_publish_step(execution) == 0
            ? TEXT_PREFILL_DONE : TEXT_PREFILL_FAILED;
        return;
    case TEXT_PREFILL_DONE:
    case TEXT_PREFILL_FAILED:
        return;
    }
    execution->phase = TEXT_PREFILL_FAILED;
}

static int text_prefill_run(void *opaque) {
    SaltTextPrefillExecution *execution = (SaltTextPrefillExecution *)opaque;
    if (!execution) return -1;
    execution->phase = TEXT_PREFILL_CHUNK_BEGIN;
    while (execution->phase != TEXT_PREFILL_DONE &&
           execution->phase != TEXT_PREFILL_FAILED)
        text_prefill_advance(execution);
    if (execution->phase == TEXT_PREFILL_DONE) return 0;
    if (salt_text_prefill_team_owner() && execution->plan->lookahead &&
            text_phase_lookahead_active(execution->plan->lookahead) &&
            salt_text_phase_lookahead_cancel(
                execution->plan->lookahead) != 0)
        return -1;
    return -1;
}

static int text_prefill_collective(void *opaque, uint32_t node, uint32_t worker) {
    SaltTextPrefillExecution execution = *(SaltTextPrefillExecution *)opaque;
    SaltTextPrefillTeam lane = {
        execution.plan->team_state, worker, execution.plan->team_workers,
    };
    int rc;
    (void)node;
    if (text_prefill_team) return -1;
    if (worker != 0u) execution.profile = NULL;
    text_prefill_team = &lane;
    rc = text_prefill_run(&execution);
    text_prefill_team = NULL;
    return rc;
}

static int text_prefill_collective_run(void *opaque) {
    SaltTextPrefillExecution *execution = opaque;
    const SaltTextPrefillPlan *plan = execution->plan;
    SaltTensorHostGraphResult result;
    return salt_tensor_host_graph_collective_execute(plan->team_state, 1u,
        text_prefill_collective, execution, plan->team_workers,
        plan->team_run, plan->team_context, &result);
}

int salt_text_prefill_execute(const SaltTextPrefillPlan *plan,
                              const SaltTextPrefillOps *ops,
                              void *context, int start_position,
                              const int *tokens, int token_count,
                              float *logits) {
    SaltTextPrefillExecution execution;
    SaltTextPrefillProfile *profile = plan ? plan->profile : NULL;
    int rc = -1;
    double wall_start = profile ? text_now_s() : 0.0;
    if (profile) memset(profile, 0, sizeof *profile);
    memset(&execution, 0, sizeof execution);
    execution.plan = plan;
    execution.ops = ops;
    execution.context = context;
    execution.start_position = start_position;
    execution.tokens = tokens;
    execution.token_count = token_count;
    execution.logits = logits;
    execution.profile = profile;
    if (text_prefill_prepare(&execution) != 0)
        return -1;
    if (profile) profile->setup_s += text_now_s() - wall_start;
    if (plan->team_state) {
        if (plan->position_logits || plan->lookahead ||
            !plan->team_run || !plan->team_context || !plan->team_workers ||
            plan->team_workers > 32u)
            return -1;
        rc = plan->transaction_run
            ? plan->transaction_run(plan->transaction_context,
                text_prefill_collective_run, &execution)
            : text_prefill_collective_run(&execution);
    } else rc = plan->transaction_run
        ? plan->transaction_run(plan->transaction_context,
              text_prefill_run, &execution)
        : text_prefill_run(&execution);
    if (rc != 0 && plan->lookahead &&
            text_phase_lookahead_active(plan->lookahead) &&
            salt_text_phase_lookahead_cancel(plan->lookahead) != 0)
        rc = -1;
    if (profile) {
        profile->total_s = text_now_s() - wall_start;
    }
    return rc;
}
