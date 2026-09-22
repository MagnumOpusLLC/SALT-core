#include "gemma4_text.h"
#include "gemma4_operation.h"
#include "gemma4_vision.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_VOCAB 900

struct SaltGemma4Text {
    int position;
    int max_context;
    int prefill_capacity;
    const SaltModelDesc *model_desc;
    SaltStateControl state_control;
    SaltGemma4RouteObserver *route_observer;
    SaltGemma4AttentionPlanLease *attention_plan_lease;
    float target_pending_logits[FIXTURE_VOCAB];
    SaltTextKvState scheduler_state;
    SaltTextVerifyProgram scheduler_program;
    SaltTextTargetPolicy scheduler_policy;
    SaltTextProjectionWindow scheduler_projection;
    SaltTextTokenEpochController scheduler_controller;
    int proof_state;
};

struct SaltGemma4Vision {
    int unused;
};

SaltGemma4Text *salt_gemma4_text_load(FILE *binding, int max_context,
                                      int expert_workers,
                                      char *error, size_t error_size) {
    SaltGemma4Text *model;
    const char *prefill_text = getenv("SALT_PREFILL_B");
    char *prefill_end = NULL;
    long prefill = prefill_text && *prefill_text
        ? strtol(prefill_text, &prefill_end, 10) : 512;
    (void)binding; (void)expert_workers;
    (void)error; (void)error_size;
    model = (SaltGemma4Text *)calloc(1, sizeof *model);
    if (!model) return NULL;
    model->max_context = max_context;
    if ((prefill_text && *prefill_text && (!prefill_end || *prefill_end)) ||
        prefill < 1 || prefill > 4096) {
        free(model);
        return NULL;
    }
    model->prefill_capacity = max_context < prefill ? max_context : (int)prefill;
    model->model_desc = salt_model_get("gemma4-26b-a4b");
    if (!model->model_desc ||
        salt_state_control_init(&model->state_control,
            salt_model_state(model->model_desc), NULL, NULL, model) != 0) {
        free(model);
        return NULL;
    }
    return model;
}

void salt_gemma4_text_free(SaltGemma4Text *model) {
    free(model);
}

int salt_gemma4_text_set_proof_state(SaltGemma4Text *model, int enabled) {
    if (!model || (enabled != 0 && enabled != 1)) return -1;
    model->proof_state = enabled;
    return 0;
}

const char *salt_gemma4_text_expert_cache_mode(
        const SaltGemma4Text *model) {
    return model ? "bounded-mmap-zero-copy" : NULL;
}

int salt_gemma4_text_encode(const SaltGemma4Text *model, const char *text,
                            int *tokens, int token_capacity) {
    (void)model;
    if (!text || !*text || !tokens || token_capacity < 3) return -1;
    tokens[0] = 10; tokens[1] = 11; tokens[2] = 12;
    return 3;
}

int salt_gemma4_text_decode(const SaltGemma4Text *model, const int *tokens,
                            int token_count, char *text, int text_capacity) {
    static const char value[] = "ok";
    (void)model; (void)tokens;
    if (!text || text_capacity < 3 || token_count < 0) return -1;
    if (token_count == 0) return 0;
    memcpy(text, value, sizeof value - 1u);
    return (int)(sizeof value - 1u);
}

static void observe_decode_route(SaltGemma4Text *model, int token) {
    SaltGemma4RouteObserver *observer;
    SaltGemma4RouteObservation *item;
    if (!model || !(observer = model->route_observer)) return;
    if (observer->route_kind != 0 &&
        observer->route_kind != SALT_GEMMA4_ROUTE_DECODE)
        return;
    if (observer->count >= observer->capacity) {
        observer->overflow = 1;
        return;
    }
    item = &observer->items[observer->count++];
    item->kind = SALT_GEMMA4_ROUTE_DECODE;
    item->position = (uint64_t)(uint32_t)model->position;
    item->layer = 0;
    item->expert = (uint32_t)token % 128u;
    if (model->attention_plan_lease && model->attention_plan_lease->active) {
        SaltGemma4AttentionPlanLease *lease = model->attention_plan_lease;
        if (lease->exact_routes != UINT32_MAX) lease->exact_routes++;
        for (uint32_t i = 0; i < lease->count; i++)
            if (lease->entries[i].layer == item->layer &&
                lease->entries[i].expert == item->expert) {
                uint64_t bit = UINT64_C(1) << i;
                if (!(lease->used_mask & bit)) {
                    lease->used_mask |= bit;
                    lease->ready_matches++;
                }
                break;
            }
    }
}

int salt_gemma4_text_step(SaltGemma4Text *model, int token, float *logits) {
    if (!model) return -1;
    observe_decode_route(model, token);
    model->position++;
    model->scheduler_state.position = (uint32_t)model->position;
    if (logits) {
        for (int i = 0; i < FIXTURE_VOCAB; i++) logits[i] = -1000.0f;
        logits[token == 818 ? 106 : 818] = 1.0f;
    }
    return 0;
}

int salt_gemma4_text_is_stop_token(int token) {
    return token == 1 || token == 50 || token == 106;
}

int salt_gemma4_text_consume_known(
        SaltGemma4Text *model, int token, float *logits) {
    return salt_gemma4_text_step(model, token, logits);
}

int salt_gemma4_text_target_generate(
        SaltGemma4Text *model, float *logits, int maximum_tokens,
        float *proposal_logits, SaltGemma4TargetGenerateResult *result) {
    int token, count = 0;
    if (!model || !logits || maximum_tokens < 1 || !proposal_logits || !result)
        return -1;
    memset(result, 0, sizeof *result);
    token = logits[818] > logits[106] ? 818 : 106;
    while (count < maximum_tokens) {
        result->candidate_token_ids[count++] = token;
        observe_decode_route(model, token);
        model->position++;
        model->scheduler_state.position = (uint32_t)model->position;
        if (salt_gemma4_text_is_stop_token(token)) break;
        token = token == 818 ? 106 : 818;
    }
    for (int index = 0; index < FIXTURE_VOCAB; index++)
        model->target_pending_logits[index] = -1000.0f;
    model->target_pending_logits[token == 818 ? 106 : 818] = 1.0f;
    memcpy(logits, model->target_pending_logits,
           sizeof model->target_pending_logits);
    result->proposal_count = (uint32_t)count;
    result->proposal_ns = 1u;
    result->epoch_path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.accepted_count = (uint32_t)count;
    result->target.committed_count = (uint32_t)count;
    result->target.produced_count = (uint32_t)count + 1u;
    result->target.pending_token_id = token == 818 ? 106 : 818;
    result->target.result_position = (uint32_t)model->position;
    result->target.pending_logits = model->target_pending_logits;
    result->target.pending_logits_count = FIXTURE_VOCAB;
    result->target.backend.engine_submissions = 1u;
    result->target.backend.completion_fences = 1u;
    return 0;
}

int salt_gemma4_text_target_policy_get(
        const SaltGemma4Text *model, SaltTextTargetPolicy *policy) {
    if (!model || !policy) return -1;
    *policy = (SaltTextTargetPolicy) {
        .execution_class = SALT_TEXT_EXECUTION_CPU_ONLY,
        .worker_budget = 1u,
        .sequence_tiles = 32u,
        .target_rows = 32u,
        .route_count = 1u,
        .queue_length = 4u,
        .candidate_count = 32u,
        .kv_warmup_rows = 512u,
        .warm_target_rows = 4u,
        .ready = 1,
    };
    return 0;
}

int salt_gemma4_text_prefill(SaltGemma4Text *model,
                             const int *tokens, int token_count,
                             float *logits) {
    if (!model || !tokens || token_count < 1 || !logits) return -1;
    for (int i = 0; i < token_count; i++)
        if (tokens[i] < 0 || tokens[i] >= FIXTURE_VOCAB) return -1;
    model->position += token_count;
    for (int i = 0; i < FIXTURE_VOCAB; i++) logits[i] = -1000.0f;
    logits[tokens[token_count - 1] == 818 ? 106 : 818] = 1.0f;
    model->scheduler_state.position = (uint32_t)model->position;
    return 0;
}

int salt_gemma4_text_prefill_image(SaltGemma4Text *model,
                                   const int *tokens,
                                   const unsigned char *mm_token_type,
                                   int token_count,
                                   const float *image_features,
                                   int image_feature_tokens,
                                   float *logits) {
    (void)tokens; (void)mm_token_type; (void)image_features;
    (void)image_feature_tokens;
    if (!model || token_count < 1 || !logits) return -1;
    model->position += token_count;
    for (int i = 0; i < FIXTURE_VOCAB; i++) logits[i] = -1000.0f;
    logits[818] = 1.0f;
    model->scheduler_state.position = (uint32_t)model->position;
    return 0;
}

int salt_gemma4_text_prefill_verify(SaltGemma4Text *model,
                                    const int *tokens, int token_count,
                                    float *position_logits,
                                    size_t logits_stride,
                                    float *final_logits) {
    if (!model || !tokens || token_count < 1 || !position_logits ||
        logits_stride != FIXTURE_VOCAB || !final_logits)
        return -1;
    for (int token = 0; token < token_count; token++) {
        float *row = position_logits + (size_t)token * logits_stride;
        for (int i = 0; i < FIXTURE_VOCAB; i++) row[i] = -1000.0f;
        row[tokens[token] == 818 ? 106 : 818] = 1.0f;
        model->position++;
        model->scheduler_state.position = (uint32_t)model->position;
    }
    memcpy(final_logits,
           position_logits + (size_t)(token_count - 1) * logits_stride,
           FIXTURE_VOCAB * sizeof(float));
    return 0;
}

int salt_gemma4_text_target_block(
        SaltGemma4Text *model, const int *candidate_tokens,
        int candidate_count, const float *initial_logits,
        SaltTextVerifyResult *result) {
    int source_position, accepted = 0, target, winner = -1;
    if (!model || !candidate_tokens || candidate_count < 1 ||
        candidate_count > SALT_TEXT_NFQ_MAX_CHECKS ||
        !initial_logits || !result)
        return -1;
    source_position = model->position;
    target = 0;
    for (int token = 1; token < FIXTURE_VOCAB; token++)
        if (initial_logits[token] > initial_logits[target]) target = token;
    for (int index = 0; index < candidate_count; index++)
        if (candidate_tokens[index] == target) {
            winner = index;
            break;
        }
    if (winner >= 0) {
        accepted = 1;
        target = target == 818 ? 106 : 818;
    }
    memset(result, 0, sizeof *result);
    model->position = source_position + accepted;
    model->scheduler_state.position = (uint32_t)model->position;
    result->status = SALT_TEXT_VERIFY_COMMITTED;
    result->accepted_count = (uint32_t)accepted;
    result->produced_count = (uint32_t)accepted + 1u;
    result->committed_count = (uint32_t)accepted;
    result->pending_token_id = target;
    result->winning_node_index = winner >= 0 ? (uint32_t)winner : UINT32_MAX;
    result->winning_node_id = result->winning_node_index;
    for (int token = 0; token < FIXTURE_VOCAB; token++)
        model->target_pending_logits[token] = -1000.0f;
    model->target_pending_logits[target] = 1.0f;
    result->pending_logits = model->target_pending_logits;
    result->pending_logits_count = FIXTURE_VOCAB;
    result->result_position = (uint32_t)model->position;
    result->backend.engine_submissions = accepted ? 1u : 0u;
    result->backend.completion_fences = accepted ? 1u : 0u;
    return 0;
}

int salt_gemma4_text_target_epoch(
        SaltGemma4Text *model, SaltTextTokenEpochController *controller,
        const int *route_token_ids, int candidate_count,
        const float *parent_logits,
        SaltTextTokenExactLookupCommit exact_lookup,
        void *exact_context, SaltTextTokenEpochResult *result) {
    (void)exact_lookup;
    (void)exact_context;
    if (!model || !controller || !controller->ready || !result) return -1;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    return salt_gemma4_text_target_block(
        model, route_token_ids, candidate_count, parent_logits,
        &result->target);
}

int salt_gemma4_text_rollback_position(SaltGemma4Text *model,
                                       int new_position) {
    if (!model || new_position < 0 || new_position > model->position) return -1;
    model->position = new_position;
    model->scheduler_state.position = (uint32_t)model->position;
    return 0;
}

static int fixture_scheduler_stop(void *opaque, int32_t token) {
    (void)opaque;
    return salt_gemma4_text_is_stop_token(token);
}

static int fixture_scheduler_target(void *opaque,
        SaltTextTokenEpochController *controller, const int32_t *tokens,
        uint32_t count, const float *parent, SaltTextTokenEpochResult *result) {
    return salt_gemma4_text_target_epoch((SaltGemma4Text *)opaque,
        controller, (const int *)tokens, (int)count, parent, NULL, NULL, result);
}

static int fixture_scheduler_select_proposal(void *opaque,
        SaltTextTokenEpochController *controller, const int32_t *tokens,
        uint32_t count, const float *parent, SaltTextTokenEpochResult *result) {
    SaltGemma4Text *model = opaque;
    uint32_t best = 0u, winner = UINT32_MAX;
    if (!model || !controller || !tokens || !count || !parent || !result)
        return -1;
    for (uint32_t token = 1u; token < FIXTURE_VOCAB; token++)
        if (parent[token] > parent[best]) best = token;
    for (uint32_t index = 0u; index < count; index++)
        if ((uint32_t)tokens[index] == best) { winner = index; break; }
    if (winner == UINT32_MAX) return -1;
    memset(result, 0, sizeof *result);
    result->path = SALT_TEXT_TOKEN_PATH_COLD_SEARCH;
    result->target.status = SALT_TEXT_VERIFY_COMMITTED;
    result->target.produced_count = 1u;
    result->target.pending_token_id = (int32_t)best;
    result->target.winning_node_index = winner;
    result->target.winning_node_id = winner;
    result->target.result_position = model->scheduler_state.position;
    result->target.transition_generation =
        model->scheduler_state.transition_generation + 1u;
    result->target.pending_logits = parent;
    result->target.pending_logits_count = FIXTURE_VOCAB;
    return 0;
}

int salt_gemma4_text_scheduler_binding(SaltGemma4Text *model,
        SaltTextGenerationBinding *binding, SaltTextTokenEpochController **controller) {
    if (!model || !binding || !controller ||
        salt_gemma4_text_target_policy_get(model, &model->scheduler_policy) != 0)
        return -1;
    model->scheduler_program.kv_state = &model->scheduler_state;
    model->scheduler_program.maximum_context = (uint32_t)model->max_context;
    model->scheduler_program.maximum_candidates = SALT_DPR_MAX_HORIZON;
    model->scheduler_program.vocabulary = FIXTURE_VOCAB;
    model->scheduler_program.ready = 1;
    memset(binding, 0, sizeof *binding);
    binding->program = &model->scheduler_program;
    binding->policy = &model->scheduler_policy;
    binding->projection = &model->scheduler_projection;
    binding->context = model;
    binding->is_stop = fixture_scheduler_stop;
    binding->select_proposal = fixture_scheduler_select_proposal;
    binding->target = fixture_scheduler_target;
    *controller = &model->scheduler_controller;
    return 0;
}

int salt_gemma4_text_position(const SaltGemma4Text *model) {
    return model ? model->position : -1;
}

int salt_gemma4_text_context_capacity(const SaltGemma4Text *model) {
    return model ? model->max_context : -1;
}

int salt_gemma4_text_prefill_chunk_capacity(const SaltGemma4Text *model) {
    return model ? model->prefill_capacity : -1;
}

size_t salt_gemma4_text_kv_capacity_bytes(const SaltGemma4Text *model) {
    size_t sliding;
    if (!model || model->max_context < 1) return 0;
    sliding = model->max_context < 1024 ? (size_t)model->max_context : 1024u;
    return sliding * 25u * 16384u +
        (size_t)model->max_context * 5u * 8192u;
}

size_t salt_gemma4_text_kv_required_bytes(const SaltGemma4Text *model,
                                          int position) {
    size_t sliding;
    if (!model || position < 0 || position > model->max_context) return 0;
    sliding = position < 1024 ? (size_t)position : 1024u;
    return sliding * 25u * 16384u + (size_t)position * 5u * 8192u;
}

int salt_gemma4_text_vocab_size(const SaltGemma4Text *model) {
    return model ? FIXTURE_VOCAB : -1;
}

int salt_gemma4_text_state_view(const SaltGemma4Text *model,
                                SaltStateView *view) {
    const SaltStateModelDesc *state_model;
    if (!model || !view || !model->model_desc || model->position < 0 ||
        !(state_model = salt_model_state(model->model_desc)))
        return -1;
    memset(view, 0, sizeof *view);
    view->schema_version = SALT_STATE_SCHEMA_VERSION;
    view->session_epoch = 1u;
    view->lifetime_turn = (uint64_t)(uint32_t)model->position;
    view->position = (uint64_t)(uint32_t)model->position;
    view->mindset_end = state_model->default_mindset_end;
    view->bytes_per_row = 450560u;
    view->mindset_kind = state_model->mindset_kind;
    view->facts_kind = state_model->facts_kind;
    view->facts_optional = state_model->facts_optional;
    return salt_state_view_validate(state_model, view);
}

int salt_gemma4_text_state_transaction_begin(
        SaltGemma4Text *model, SaltStateTransactionKind kind,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
        SaltStateTransaction *transaction) {
    SaltStateView before;
    if (!model || salt_gemma4_text_state_view(model, &before) != 0)
        return -1;
    return salt_state_transaction_begin(&model->state_control, &before,
        kind, artifact, reason, transaction);
}

int salt_gemma4_text_state_transaction_finish(
        SaltGemma4Text *model, SaltStateTransaction *transaction) {
    SaltStateView after;
    if (!model || !transaction ||
        salt_gemma4_text_state_view(model, &after) != 0)
        return -1;
    return salt_state_transaction_finish(transaction, &after);
}

int salt_gemma4_text_state_sha256(const SaltGemma4Text *model,
                                  uint8_t out[32]) {
    if (!model || !out) return -1;
    for (int i = 0; i < 32; i++)
        out[i] = (uint8_t)((model->position + i) & 0xff);
    return 0;
}

int salt_gemma4_text_transition_sha256(const SaltGemma4Text *model,
                                       uint8_t out[32]) {
    return salt_gemma4_text_state_sha256(model, out);
}

int salt_gemma4_text_build_identity_sha256(const SaltGemma4Text *model,
                                           uint8_t out[32]) {
    if (!model || !out) return -1;
    memset(out, 0xff, 32);
    return 0;
}

int salt_gemma4_text_facts_sha256(const SaltGemma4Text *model,
                                  int mindset_end, uint8_t out[32]) {
    if (!model || !out || mindset_end < 0 || mindset_end > model->position)
        return -1;
    for (int i = 0; i < 32; i++)
        out[i] = (uint8_t)((model->position - mindset_end + i + 1) & 0xff);
    return 0;
}

int salt_gemma4_text_kv_attach_shared(SaltGemma4Text *model,
                                      const void *buffer, size_t buffer_size) {
    (void)model; (void)buffer; (void)buffer_size;
    return -1;
}

int salt_gemma4_text_shared_position(const SaltGemma4Text *model) {
    return model ? 0 : -1;
}

size_t salt_gemma4_text_kv_export_size(const SaltGemma4Text *model) {
    if (!model || model->position < 0) return 0;
    return 256u + (size_t)model->position * 450560u;
}

int salt_gemma4_text_kv_export(const SaltGemma4Text *model,
                               void *buffer, size_t buffer_size) {
    uint32_t position;
    size_t expected = salt_gemma4_text_kv_export_size(model);
    if (!model || !buffer || expected == 0 || buffer_size != expected)
        return -1;
    memset(buffer, 0, buffer_size);
    position = (uint32_t)model->position;
    memcpy(buffer, &position, sizeof position);
    return 0;
}

int salt_gemma4_text_kv_import(SaltGemma4Text *model,
                               const void *buffer, size_t buffer_size) {
    uint32_t position;
    size_t expected;
    if (!model || !buffer || buffer_size < 256u) return -1;
    memcpy(&position, buffer, sizeof position);
    if (position > 512u) return -1;
    expected = 256u + (size_t)position * 450560u;
    if (buffer_size != expected) return -1;
    model->position = (int)position;
    model->scheduler_state.position = position;
    return 0;
}

int salt_gemma4_text_kv_export_stream(const SaltGemma4Text *model,
                                      FILE *stream) {
    uint32_t position;
    if (!model || !stream || model->position < 0) return -1;
    position = (uint32_t)model->position;
    return fwrite(&position, sizeof position, 1, stream) == 1 ? 0 : -1;
}

int salt_gemma4_text_kv_import_stream(SaltGemma4Text *model, FILE *stream) {
    uint32_t position;
    if (!model || !stream || fread(&position, sizeof position, 1, stream) != 1 ||
        position > 512u)
        return -1;
    model->position = (int)position;
    model->scheduler_state.position = position;
    return 0;
}

int salt_gemma4_text_kv_import_stream_sha256(
        SaltGemma4Text *model, FILE *stream, uint8_t file_sha256[32]) {
    (void)stream; (void)file_sha256;
    if (model) {
        model->position = 0;
        model->scheduler_state.position = 0;
    }
    return -1;
}

int salt_gemma4_text_kv_import_delta_stream(
        SaltGemma4Text *model, FILE *stream,
        int source_position, int token_count) {
    (void)stream;
    if (!model || source_position < 0 || token_count < 1 ||
        model->position != source_position || source_position > 512 - token_count)
        return -1;
    model->position = source_position + token_count;
    model->scheduler_state.position = (uint32_t)model->position;
    return 0;
}

int salt_gemma4_text_prefix_sha256(const SaltGemma4Text *model,
                                   int position, uint8_t out[32]) {
    if (!model || !out || position < 0 || position > model->position) return -1;
    for (int i = 0; i < 32; i++)
        out[i] = (uint8_t)((position + i) & 0xff);
    return 0;
}

int salt_gemma4_text_clear_facts(SaltGemma4Text *model, int mindset_end) {
    if (!model || mindset_end < 0 || mindset_end > model->position) return -1;
    model->position = mindset_end;
    model->scheduler_state.position = (uint32_t)model->position;
    return 0;
}

int salt_gemma4_text_memory_stats(const SaltGemma4Text *model,
                                  SaltGemma4MemoryStats *stats) {
    const char *text = getenv("SALT_EXPERT_BUDGET_GB");
    char *end = NULL;
    long gb;
    if (!model || !stats || !text || !*text) return -1;
    gb = strtol(text, &end, 10);
    if (*end || gb < 1 || gb > 12) return -1;
    memset(stats, 0, sizeof *stats);
    stats->expert_budget_bytes = (uint64_t)gb * UINT64_C(1000000000);
    return 0;
}

int salt_gemma4_text_prepare_attention_plan_leased(
        SaltGemma4Text *model, const SaltDprComputeIntent *intent,
        const SaltDprAttentionPlan *plan,
        SaltGemma4AttentionPlanLease *lease,
        SaltGemma4AttentionPrepareStats *stats) {
    int position;
    if (!model || !intent || !plan || !lease || !stats ||
        model->attention_plan_lease || lease->active ||
        salt_dpr_attention_plan_validate(
            plan, intent, 30, 128, 512) != 1)
        return -1;
    position = model->position;
    memset(lease, 0, sizeof *lease);
    memset(stats, 0, sizeof *stats);
    stats->requested_items = plan->item_count;
    for (uint32_t i = 0; i < plan->item_count; i++) {
        if (plan->items[i].kind != SALT_DPR_RELEVANCE_EXPERT_PREFETCH)
            return 1;
        lease->entries[lease->count++] = (SaltGemma4AttentionLeaseEntry) {
            plan->items[i].layer, plan->items[i].index, (int32_t)i
        };
        stats->prepared_items++;
        stats->fetch_jobs++;
    }
    if (model->position != position) return -1;
    lease->active = 1;
    model->attention_plan_lease = lease;
    return 0;
}

int salt_gemma4_text_release_attention_plan_lease(
        SaltGemma4Text *model, SaltGemma4AttentionPlanLease *lease) {
    if (!model || !lease || !lease->active ||
        model->attention_plan_lease != lease)
        return -1;
    model->attention_plan_lease = NULL;
    lease->active = 0;
    return 0;
}

int salt_gemma4_text_prepare_attention_plan(
        SaltGemma4Text *model, const SaltDprComputeIntent *intent,
        const SaltDprAttentionPlan *plan,
        SaltGemma4AttentionPrepareStats *stats) {
    SaltGemma4AttentionPlanLease lease = {0};
    int rc = salt_gemma4_text_prepare_attention_plan_leased(
        model, intent, plan, &lease, stats);
    if (rc != 0) return rc;
    return salt_gemma4_text_release_attention_plan_lease(model, &lease);
}

int salt_gemma4_text_route_observer_begin(
        SaltGemma4Text *model, SaltGemma4RouteObserver *observer) {
    if (!model || !observer || !observer->items || observer->capacity == 0 ||
        model->route_observer)
        return -1;
    observer->count = 0;
    observer->overflow = 0;
    model->route_observer = observer;
    return 0;
}

int salt_gemma4_text_route_observer_end(SaltGemma4Text *model) {
    if (!model || !model->route_observer) return -1;
    model->route_observer = NULL;
    return 0;
}

int salt_gemma4_text_route_observer_update_coverage_ledger(
        const SaltGemma4RouteObserver *observer, uint32_t route_kind,
        SaltDprExpertCoverageLedger *ledger) {
    uint64_t counts[30][128] = {{0}};
    int matched = 0;
    if (!observer || !observer->items || observer->overflow ||
        observer->count > observer->capacity || !ledger ||
        ledger->layer_count != 30 || ledger->expert_count != 128 ||
        (route_kind != SALT_GEMMA4_ROUTE_DECODE &&
         route_kind != SALT_GEMMA4_ROUTE_PREFILL))
        return -1;
    for (size_t index = 0; index < observer->count; index++) {
        const SaltGemma4RouteObservation *item = &observer->items[index];
        if (item->kind != route_kind) continue;
        if (item->layer >= 30 || item->expert >= 128 ||
            counts[item->layer][item->expert] == UINT64_MAX)
            return -1;
        counts[item->layer][item->expert]++;
        matched = 1;
    }
    return matched ? salt_dpr_expert_coverage_ledger_merge(
        ledger, &counts[0][0]) : 1;
}

SaltGemma4Vision *salt_gemma4_vision_load(FILE *binding, int workers,
                                          char *error, size_t error_size) {
    (void)binding; (void)workers; (void)error; (void)error_size;
    return (SaltGemma4Vision *)calloc(1, sizeof(SaltGemma4Vision));
}

void salt_gemma4_vision_free(SaltGemma4Vision *model) {
    free(model);
}

int salt_gemma4_vision_close(SaltGemma4Vision **model) {
    if (!model) return -1;
    free(*model);
    *model = NULL;
    return 0;
}

int salt_gemma4_vision_worker_stats(const SaltGemma4Vision *model,
                                    int *workers, uint64_t *submissions) {
    if (!model || !workers || !submissions) return -1;
    *workers = 1;
    *submissions = 0;
    return 0;
}

int salt_gemma4_vision_gpu_stats(const SaltGemma4Vision *model,
                                 int *pageable_hmm,
                                 uint64_t *submissions) {
    if (!model || !pageable_hmm || !submissions) return -1;
    *pageable_hmm = 0;
    *submissions = 0;
    return 0;
}

int salt_gemma4_vision_forward(SaltGemma4Vision *model,
                               const float *patches,
                               const int *positions_xy,
                               const unsigned char *padding,
                               int n_patches,
                               float *features,
                               int feature_token_capacity,
                               char *error, size_t error_size) {
    (void)model; (void)patches; (void)positions_xy; (void)padding;
    (void)n_patches; (void)features; (void)feature_token_capacity;
    (void)error; (void)error_size;
    return -1;
}

int salt_gemma4_ppm_to_patches(const char *path, int max_soft_tokens,
                               float **patches,
                               int **positions_xy,
                               unsigned char **padding,
                               int *n_patches,
                               int *valid_soft_tokens,
                               int *width, int *height,
                               char *error, size_t error_size) {
    (void)path; (void)max_soft_tokens; (void)patches;
    (void)positions_xy; (void)padding; (void)n_patches;
    (void)valid_soft_tokens; (void)width; (void)height;
    (void)error; (void)error_size;
    return -1;
}

void salt_gemma4_free_patches(float *patches, int *positions_xy,
                              unsigned char *padding) {
    free(padding); free(positions_xy); free(patches);
}
