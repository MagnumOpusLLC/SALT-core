/* Persistent single-session Gemma 4 native server runner, protocol V2.
 *
 * The authenticated Python wrapper writes the existing text+vision binding to
 * stdin, then relays strict TURN_V2 frames. One text model and one live KV remain
 * resident until DRAIN. Machine protocol uses stdout only; stderr is diagnostic.
 * runtime_ready remains false.
 */

#include "gemma4_text.h"
#include "gemma4_vision.h"
#include "salt/dpr.h"
#include "salt/dpr_stats.h"
#include "salt/dpr_store.h"
#include "salt/model.h"
#include "salt/salt.h"
#include "salt/state.h"
#include "gemma4_kv_file.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define G4_DEFAULT_CONTEXT 512
#define G4_DEFAULT_OUTPUT_LIMIT 128
#define G4_MODEL_MAX_CONTEXT 262144
#define G4_HIDDEN 2816
#define G4_IMAGE_TOKEN 258880
#define G4_MAX_PROMPT_BYTES (4u * 1024u * 1024u)
#define G4_MAX_IMAGE_PATH 4096
#define G4_MAX_SOFT_TOKENS 70
#define G4_KV_BYTES_PER_TOKEN 450560

#define G4_PROTOCOL 3
#define G4_SHA256_HEX 64
#define G4_JOURNAL_CAPACITY 256
#define G4_RESPONSE_TOKEN_BYTES 64u
#define G4_RESPONSE_OVERHEAD 1024u
#define G4_DPR_MAX_HORIZON 64
#define G4_DPR_CANDIDATE_BYTES (G4_DPR_MAX_HORIZON * 12)
#define G4_DPR_EDGE_CAPACITY 4096u
#define G4_DPR_BUCKET_COUNT 8192u
#define G4_DPR_MENTOR_FAMILY_CAPACITY 256u
#define G4_DPR_MENTOR_FAMILY_BUCKETS 512u
#define G4_DPR_MENTOR_BINDING_CAPACITY G4_DPR_EDGE_CAPACITY
#define G4_DPR_MENTOR_BINDING_BUCKETS G4_DPR_BUCKET_COUNT
#define G4_DPR_ATTENTION_PLAN_CAPACITY G4_DPR_EDGE_CAPACITY
#define G4_DPR_ATTENTION_RUNTIME_CAPACITY \
    (G4_DPR_ATTENTION_PLAN_CAPACITY + 1u)
#define G4_DPR_ATTENTION_PLAN_BUCKETS G4_DPR_BUCKET_COUNT
#define G4_ROUTE_LAYERS 30u
#define G4_ROUTE_EXPERTS 128u
#define G4_ROUTE_TOPK 8u
#define G4_RUNTIME_READY_TEXT "true"
#define G4_ROUTE_OBSERVATION_CAPACITY \
    ((size_t)g_context * G4_ROUTE_LAYERS * G4_ROUTE_TOPK)


#ifndef SALT_GEMMA4_KV_COMPAT_SHA256
#error "SALT_GEMMA4_KV_COMPAT_SHA256 is required"
#endif

static unsigned long long g_session_epoch = 1;
static int g_context = G4_DEFAULT_CONTEXT;
static int g_output_limit = G4_DEFAULT_OUTPUT_LIMIT;
static int g_active_output_limit = G4_DEFAULT_OUTPUT_LIMIT;
static const SaltStateModelDesc *g_state_model = NULL;
static int g_mindset_end = -1;
static const char *g_mindset_mode = NULL;
static char g_build_identity_sha256[G4_SHA256_HEX + 1];
static int g_proof_state = 0;
static int g_process_proof_state = 0;
static SaltDprMode g_dpr_mode = SALT_DPR_OFF;
static const char *g_dpr_root = NULL;
static uint64_t g_dpr_current_kv_bytes = 0;
static uint64_t g_dpr_additive_bytes = 0;
static uint64_t g_dpr_serial_ns = 0;
static uint64_t g_dpr_retention_bytes = 0;
static uint32_t g_dpr_draft_n = 0;
static uint32_t g_dpr_prefill_n = 0;
static uint32_t g_dpr_qa_bucket_bits = 0;
static int g_dpr_mentor_enabled = 0;
static uint8_t g_dpr_mentor_policy_sha256[32];
static int g_dpr_attention_enabled = 0;
static uint8_t g_dpr_attention_policy_sha256[32];
static const SaltDprAttentionPlan *g_hot_route_candidate = NULL;
static uint8_t g_hot_route_provenance_sha256[32];
static uint8_t g_hot_route_mindset_sha256[32];
static int g_hot_route_ready = 0;
static uint8_t g_dpr_transition_chain[32];
static uint8_t g_dpr_sampler_config_sha256[32];
static uint64_t g_dpr_transition_position = 0;

typedef struct G4DprReply {
    int hit;
    int dynamic;
    SaltDprEdgeKind kind;
    int horizon;
    int bonus_token_id;
    int candidate_ids[G4_DPR_MAX_HORIZON];
    char edge_id[G4_SHA256_HEX + 1];
    char reference_sha256[G4_SHA256_HEX + 1];
    char reference_path[G4_MAX_IMAGE_PATH + 1];
    size_t reference_bytes;
} G4DprReply;

typedef struct G4DprObservation {
    uint32_t accepted;
    uint32_t committed;
    uint64_t verify_ns;
} G4DprObservation;

typedef struct G4DprAttentionObservation {
    int applied;
    SaltDprOperationCell cell;
    uint64_t lookup_ns;
    uint64_t prepare_ns;
    SaltGemma4AttentionPrepareStats prepare;
    SaltGemma4AttentionPlanLease lease;
    uint8_t plan_sha256[32];
    uint8_t dpr_sha256[32];
    uint8_t mindset_sha256[32];
} G4DprAttentionObservation;

typedef struct G4DprWaterfall {
    uint64_t prefill_node_ns;
    uint64_t prefill_lookup_ns;
    uint64_t prefill_restore_ns;
    uint64_t prefill_ordinary_ns;
    uint64_t decode_node_ns;
    uint64_t decode_lookup_ns;
    uint64_t decode_parent_restore_ns;
    uint64_t decode_verify_ns;
    uint64_t decode_ordinary_ns;
    uint32_t prefill_hits;
    uint32_t prefill_cached_tokens;
    uint32_t decode_cycles;
    uint32_t parent_hits;
    uint32_t nomogram_hits;
    uint32_t decode_misses;
    uint32_t proposed_tokens;
    uint32_t accepted_tokens;
    uint32_t rejection_count;
} G4DprWaterfall;

static G4DprWaterfall g_dpr_waterfall;

static int fill_state_digests_required(
        const SaltGemma4Text *model,
        unsigned char state_sha256[32],
        unsigned char facts_sha256[32]) {
    if (!model || !state_sha256 || !facts_sha256) return -1;
    return salt_gemma4_text_state_sha256(model, state_sha256) == 0 &&
           salt_gemma4_text_facts_sha256(
               model, g_mindset_end, facts_sha256) == 0 ? 0 : -1;
}

static int fill_state_digests(const SaltGemma4Text *model,
                              unsigned char state_sha256[32],
                              unsigned char facts_sha256[32]) {
    if (!model || !state_sha256 || !facts_sha256) return -1;
    memset(state_sha256, 0, 32u);
    memset(facts_sha256, 0, 32u);
    if (!g_proof_state) return 0;
    return fill_state_digests_required(
        model, state_sha256, facts_sha256);
}

typedef struct G4CommitRecord {
    int used;
    unsigned long long original_request_id;
    unsigned long long turn_id;
    unsigned long long history_turns;
    char client_request_sha256[G4_SHA256_HEX + 1];
    char request_sha256[G4_SHA256_HEX + 1];
    int prompt_tokens;
    int image_tokens;
    int output_limit;
    int output_steps;
    int stop_token;
    int synthetic_close;
    int position_before;
    int position_after;
    int mindset_end;
    int response_bytes;
    size_t prompt_offset;
    size_t output_offset;
    size_t response_offset;
    unsigned char state_sha256[32];
    unsigned char mindset_sha256[32];
    unsigned char facts_sha256[32];
} G4CommitRecord;

typedef struct G4JournalArena {
    int *prompt_ids;
    int *output_ids;
    char *responses;
    size_t prompt_capacity;
    size_t output_capacity;
    size_t response_capacity;
    size_t prompt_used;
    size_t output_used;
    size_t response_used;
} G4JournalArena;

static G4JournalArena g_journal_arena;


static int fill_state_view(const SaltGemma4Text *model,
                           unsigned long long lifetime_turn,
                           unsigned long long history_turns,
                           const unsigned char mindset_sha256[32],
                           SaltStateView *view) {
    int position;
    if (!model || !mindset_sha256 || !view || !g_state_model ||
        g_mindset_end < 0 ||
        (position = salt_gemma4_text_position(model)) < g_mindset_end)
        return -1;
    memset(view, 0, sizeof *view);
    view->schema_version = SALT_STATE_SCHEMA_VERSION;
    view->session_epoch = g_session_epoch;
    view->lifetime_turn = lifetime_turn;
    view->history_turns = history_turns;
    view->position = (uint64_t)position;
    view->mindset_end = (uint64_t)g_mindset_end;
    view->bytes_per_row = G4_KV_BYTES_PER_TOKEN;
    view->mindset_kind = g_state_model->mindset_kind;
    view->facts_kind = g_state_model->facts_kind;
    view->facts_optional = g_state_model->facts_optional;
    view->proof_computed = g_proof_state;
    memcpy(view->mindset_sha256, mindset_sha256, 32u);
    if (fill_state_digests(
            model, view->state_sha256, view->facts_sha256) != 0)
        return -1;
    return salt_state_view_validate(g_state_model, view);
}

static int is_sha256_hex(const char *text) {
    if (!text || strlen(text) != G4_SHA256_HEX) return 0;
    for (int i = 0; i < G4_SHA256_HEX; i++)
        if (!((text[i] >= '0' && text[i] <= '9') ||
              (text[i] >= 'a' && text[i] <= 'f')))
            return 0;
    return 1;
}

static int is_sha256_zero_hex(const char *text) {
    if (!text || strlen(text) != G4_SHA256_HEX) return 0;
    for (int i = 0; i < G4_SHA256_HEX; i++)
        if (text[i] != '0') return 0;
    return 1;
}

static G4CommitRecord *find_record(G4CommitRecord *journal,
                                    int journal_count,
                                    const char *client_request_sha256) {
    for (int i = 0; i < journal_count; i++)
        if (journal[i].used && !strcmp(
                journal[i].client_request_sha256, client_request_sha256))
            return &journal[i];
    return NULL;
}

static int parse_positive(const char *value, int *out) {
    char *end = NULL;
    long parsed;
    if (!value || !*value || !out) return -1;
    parsed = strtol(value, &end, 10);
    if (!end || *end || parsed < 1 || parsed > INT_MAX) return -1;
    *out = (int)parsed;
    return 0;
}

static int parse_epoch(const char *value, unsigned long long *out) {
    unsigned long long parsed = 0;
    size_t length;
    if (!value || !*value || !out) return -1;
    length = strlen(value);
    if ((length > 1u && value[0] == '0') || value[0] == '0') return -1;
    for (size_t i = 0; i < length; i++) {
        unsigned digit;
        if (value[i] < '0' || value[i] > '9') return -1;
        digit = (unsigned)(value[i] - '0');
        if (parsed > (ULLONG_MAX - digit) / 10u) return -1;
        parsed = parsed * 10u + digit;
    }
    if (parsed < 1) return -1;
    *out = parsed;
    return 0;
}

static int parse_u64_positive(const char *value, uint64_t *out) {
    unsigned long long parsed = 0;
    if (!out || parse_epoch(value, &parsed) != 0 || parsed > UINT64_MAX)
        return -1;
    *out = (uint64_t)parsed;
    return 0;
}

static int parse_dpr_horizon_env(const char *name, uint32_t *out) {
    uint64_t parsed = 0;
    const char *value;
    if (!name || !out || !(value = getenv(name)) ||
        parse_u64_positive(value, &parsed) != 0 ||
        parsed > SALT_DPR_MAX_HORIZON)
        return -1;
    *out = (uint32_t)parsed;
    return 0;
}

static int parse_dpr_qa_bits_env(uint32_t *out) {
    uint64_t parsed = 0;
    const char *value = getenv("SALT_DPR_QA_BUCKET_BITS");
    if (!out || !value || parse_u64_positive(value, &parsed) != 0 ||
        parsed > 32)
        return -1;
    *out = (uint32_t)parsed;
    return 0;
}

static uint64_t monotonic_ns(void) {
    struct timespec value;
    uint64_t seconds;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0 || value.tv_sec < 0 ||
        value.tv_nsec < 0 || value.tv_nsec >= 1000000000L ||
        (uint64_t)value.tv_sec > UINT64_MAX / UINT64_C(1000000000))
        return 0;
    seconds = (uint64_t)value.tv_sec * UINT64_C(1000000000);
    return seconds > UINT64_MAX - (uint64_t)value.tv_nsec
        ? 0 : seconds + (uint64_t)value.tv_nsec;
}

static int greedy_token(const float *logits, int count) {
    int best = -1;
    float best_value = -INFINITY;
    for (int i = 0; i < count; i++) {
        if (!isfinite(logits[i])) return -1;
        if (best < 0 || logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    return best;
}

static void digest_hex(const unsigned char digest[32],
                       char text[G4_SHA256_HEX + 1]);

static int is_stop_token(int token) {
    return token == 1 || token == 50 || token == 106;
}

static int dpr_stop_token(void *opaque, int32_t token) {
    (void)opaque;
    return is_stop_token((int)token);
}

static int build_dpr_node(const SaltGemma4Text *model,
                          SaltDprNodeMaterial *material,
                          uint8_t node_sha256[32]) {
    if (!model || !material || !node_sha256) return -1;
    memset(material, 0, sizeof *material);
    if (salt_sha256_hex_parse(
            SALT_GEMMA4_KV_COMPAT_SHA256,
            material->state_compatibility_sha256) != 0 ||
        g_dpr_transition_position !=
            (uint64_t)salt_gemma4_text_position(model))
        return -1;
    memcpy(material->state_sha256, g_dpr_transition_chain, 32);
    memcpy(material->sampler_config_sha256,
       g_dpr_sampler_config_sha256, 32);
    material->sampler_abi = SALT_DPR_SAMPLER_GREEDY_V1;
    material->position = (uint64_t)salt_gemma4_text_position(model);
    return salt_dpr_node_sha256(material, node_sha256);
}

static int dpr_transition_reset(uint64_t position) {
    static const unsigned char sampler_config[] =
        "salt-greedy-v1;temperature=0;target-rng=none";
    uint8_t compatibility[32];
    if (salt_sha256_hex_parse(
            SALT_GEMMA4_KV_COMPAT_SHA256, compatibility) != 0 ||
        salt_sha256_bytes(sampler_config, sizeof sampler_config - 1u,
                          g_dpr_sampler_config_sha256) != 0 ||
        salt_dpr_transition_chain_init(
            compatibility, g_dpr_sampler_config_sha256,
            SALT_DPR_SAMPLER_GREEDY_V1, position,
            g_dpr_transition_chain) != 0)
        return -1;
    g_dpr_transition_position = position;
    return 0;
}

static int dpr_transition_advance(int token_id) {
    uint8_t next[32];
    uint64_t position;
    if (token_id < 0 || g_dpr_transition_position == UINT64_MAX)
        return -1;
    position = g_dpr_transition_position + 1u;
    if (salt_dpr_transition_chain_advance(
            g_dpr_transition_chain, (int32_t)token_id,
            position, next) != 0)
        return -1;
    memcpy(g_dpr_transition_chain, next, 32);
    g_dpr_transition_position = position;
    return 0;
}

static int dpr_transition_advance_many(const int *tokens, int token_count) {
    if (!tokens || token_count < 0) return -1;
    for (int token = 0; token < token_count; token++)
        if (dpr_transition_advance(tokens[token]) != 0) return -1;
    return 0;
}

static int commit_pending_token(SaltGemma4Text *model, int token, float *logits) {
    if (!model || token < 0 || salt_gemma4_text_step(model, token, logits) != 0)
        return -1;
    if (g_dpr_mode != SALT_DPR_OFF &&
            (dpr_transition_advance(token) != 0 ||
             g_dpr_transition_position !=
                (uint64_t)salt_gemma4_text_position(model)))
        return -1;
    return 0;
}

static int dpr_compute_intent(
        SaltDprOperationKind kind, const uint8_t node_sha256[32],
        uint64_t position, const int32_t *candidate_ids,
        uint32_t candidate_count, SaltDprComputeIntent *intent) {
    if (!node_sha256 || !intent ||
        (candidate_count != 0 && !candidate_ids))
        return -1;
    memset(intent, 0, sizeof *intent);
    intent->schema_version = SALT_DPR_COMPUTE_INTENT_VERSION;
    intent->kind = kind;
    intent->position = position;
    intent->horizon = candidate_count ? candidate_count : 1u;
    intent->candidate_count = candidate_count;
    memcpy(intent->target_node_sha256, node_sha256, 32);
    if (salt_sha256_hex_parse(
            SALT_GEMMA4_KV_COMPAT_SHA256,
            intent->state_compatibility_sha256) != 0)
        return -1;
    if (g_dpr_attention_enabled)
        memcpy(intent->active_policy_sha256,
               g_dpr_attention_policy_sha256, 32);
    else
        memset(intent->active_policy_sha256, 0xff, 32);
    if (candidate_count && salt_dpr_candidate_sha256(
            candidate_ids, candidate_count, intent->candidate_sha256) != 0)
        return -1;
    return 0;
}

static int dpr_attention_prepare(
        SaltGemma4Text *model, SaltDprAttentionPlanStore *store,
        const SaltDprComputeIntent *intent, int has_transition,
        G4DprAttentionObservation *observation) {
    size_t selected = SIZE_MAX;
    int selected_rc, prepare_rc;
    uint64_t start, end;
    if (!model || !intent || !observation) return -1;
    memset(observation, 0, sizeof *observation);
    if (!g_dpr_attention_enabled) return 0;
    if (!store) return -1;
    start = monotonic_ns();
    if (g_hot_route_ready) {
        SaltDprAttentionPlan runtime_plan;
        uint8_t plan_sha256[32];
        if (!g_hot_route_candidate) return -1;
        runtime_plan = *g_hot_route_candidate;
        if (salt_dpr_attention_plan_finalize_runtime(
                &runtime_plan, intent, g_hot_route_mindset_sha256,
                g_hot_route_provenance_sha256) != 0 ||
            salt_dpr_attention_plan_store_upsert_runtime(
                store, &runtime_plan, plan_sha256, &selected) != 0)
            return -1;
    }
    selected_rc = salt_dpr_attention_plan_store_select(
        store, intent, 30, 128, g_context, &selected);
    end = monotonic_ns();
    if (start && end >= start) observation->lookup_ns = end - start;
    if (selected_rc < 0) return -1;
    if (selected_rc == 0) return 0;
    start = monotonic_ns();
    prepare_rc = salt_gemma4_text_prepare_attention_plan_leased(
        model, intent, &store->plans[selected].plan,
        &observation->lease, &observation->prepare);
    end = monotonic_ns();
    if (start && end >= start) observation->prepare_ns = end - start;
    if (prepare_rc < 0) return -1;
    if (prepare_rc > 0) return 0;
    observation->applied = 1;
    observation->cell = has_transition ? SALT_DPR_CELL_ND_NM
                                       : SALT_DPR_CELL_NM_ONLY;
    memcpy(observation->plan_sha256,
           store->plans[selected].plan_sha256, 32);
    memcpy(observation->dpr_sha256,
           store->plans[selected].plan.dpr_sha256, 32);
    memcpy(observation->mindset_sha256,
           store->plans[selected].plan.mindset_sha256, 32);
    return 1;
}

static int dpr_attention_release(
        SaltGemma4Text *model, G4DprAttentionObservation *observation) {
    if (!model || !observation) return -1;
    if (!observation->applied) return 0;
    return salt_gemma4_text_release_attention_plan_lease(
        model, &observation->lease);
}

static int build_dpr_qa_key(const int *prompt_ids, int prompt_count,
                            uint8_t out[32]) {
    uint8_t compatibility[32];
    if (!prompt_ids || prompt_count < 1 || !out ||
        salt_sha256_hex_parse(
            SALT_GEMMA4_KV_COMPAT_SHA256, compatibility) != 0)
        return -1;
    return salt_dpr_nomogram_qa_key(
        compatibility, (const int32_t *)prompt_ids,
        (size_t)prompt_count, g_dpr_qa_bucket_bits, out);
}

static int dpr_reply_from_stored(const SaltDprStoredEdge *stored,
                                 G4DprReply *reply) {
    if (!stored || !reply || stored->edge.horizon < 1 ||
        stored->edge.horizon > G4_DPR_MAX_HORIZON)
        return -1;
    memset(reply, 0, sizeof *reply);
    reply->hit = 1;
    reply->dynamic = stored->edge.mode == SALT_DPR_DYNAMIC;
    reply->kind = stored->edge.kind;
    reply->horizon = (int)stored->edge.horizon;
    reply->bonus_token_id = stored->edge.bonus_token_id;
    reply->reference_bytes = (size_t)stored->edge.reference_bytes;
    memcpy(reply->candidate_ids, stored->edge.candidate_token_ids,
           (size_t)reply->horizon * sizeof reply->candidate_ids[0]);
    digest_hex(stored->edge_sha256, reply->edge_id);
    digest_hex(stored->edge.reference_sha256, reply->reference_sha256);
    return 0;
}

static int print_hex(FILE *stream, const unsigned char digest[32]) {
    for (int i = 0; i < 32; i++)
        if (fprintf(stream, "%02x", digest[i]) < 0) return -1;
    return 0;
}

static void digest_hex(const unsigned char digest[32],
                       char text[G4_SHA256_HEX + 1]) {
    static const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        text[2 * i] = digits[digest[i] >> 4];
        text[2 * i + 1] = digits[digest[i] & 15u];
    }
    text[G4_SHA256_HEX] = '\0';
}

static int write_payload(const void *payload, size_t bytes) {
    if (bytes && fwrite(payload, 1, bytes, stdout) != bytes) return -1;
    if (fputc('\n', stdout) == EOF || fflush(stdout) != 0) return -1;
    return 0;
}

static int emit_reject(unsigned long long request_id, const char *code,
                       const char *message) {
    size_t bytes = strlen(message);
    if (fprintf(stdout,
            "GEMMA4_SERVER_REJECT_V2 request=%llu mutated=0 code=%s bytes=%zu "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, code, bytes) < 0)
        return -1;
    return write_payload(message, bytes);
}

static void emit_fatal(unsigned long long request_id, const char *code,
                       const char *message) {
    size_t bytes = strlen(message);
    if (fprintf(stdout,
            "GEMMA4_SERVER_FATAL_V2 request=%llu mutated=1 code=%s bytes=%zu "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, code, bytes) >= 0)
        (void)write_payload(message, bytes);
}

static int emit_start(unsigned long long request_id,
                      unsigned long long turn_id, int position,
                      int prompt_tokens, int image_tokens,
                      int output_limit) {
    if (fprintf(stdout,
            "GEMMA4_SERVER_START_V3 request=%llu session_epoch=%llu "
            "turn_id=%llu position=%d prompt_tokens=%d image_tokens=%d "
            "output_limit=%d runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, g_session_epoch, turn_id, position,
            prompt_tokens, image_tokens, output_limit) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_memory(unsigned long long request_id, const char *phase,
                       const SaltMemSnapshot *memory) {
    if (!phase || !memory || memory->application_b < 0 ||
        memory->application_os_peak_b < 0 || memory->resident_b < 0 ||
        memory->resident_os_peak_b < 0 || memory->anonymous_b < 0 ||
        memory->compressed_b < 0 || memory->file_backed_b < 0 ||
        memory->shared_b < 0 || memory->page_table_b < 0 ||
        memory->swap_b < 0 ||
        fprintf(stdout,
            "GEMMA4_SERVER_MEMORY_V1 request=%llu phase=%s "
            "application_bytes=%lld application_os_peak_bytes=%lld "
            "resident_bytes=%lld resident_os_peak_bytes=%lld "
            "internal_bytes=%lld compressed_bytes=%lld external_bytes=%lld "
            "shared_bytes=%lld page_table_bytes=%lld swap_bytes=%lld "
            "approximate=%d runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, phase,
            (long long)memory->application_b,
            (long long)memory->application_os_peak_b,
            (long long)memory->resident_b,
            (long long)memory->resident_os_peak_b,
            (long long)memory->anonymous_b,
            (long long)memory->compressed_b,
            (long long)memory->file_backed_b,
            (long long)memory->shared_b,
            (long long)memory->page_table_b,
            (long long)memory->swap_b,
            memory->application_is_approximate) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_waterfall(unsigned long long request_id) {
    if (fprintf(stdout,
            "GEMMA4_SERVER_DPR_WATERFALL_V1 request=%llu mode=%s "
            "prefill_node_ns=%llu prefill_lookup_ns=%llu "
            "prefill_restore_ns=%llu prefill_ordinary_ns=%llu "
            "decode_node_ns=%llu decode_lookup_ns=%llu "
            "decode_parent_restore_ns=%llu decode_verify_ns=%llu "
            "decode_ordinary_ns=%llu prefill_hits=%u "
            "prefill_cached_tokens=%u decode_cycles=%u parent_hits=%u "
            "nomogram_hits=%u decode_misses=%u proposed_tokens=%u "
            "accepted_tokens=%u rejection_count=%u runtime_ready="
            G4_RUNTIME_READY_TEXT "\n",
            request_id,
            g_dpr_mode == SALT_DPR_OFF ? "off" :
                (g_dpr_mode == SALT_DPR_DYNAMIC ? "dynamic" : "persist"),
            (unsigned long long)g_dpr_waterfall.prefill_node_ns,
            (unsigned long long)g_dpr_waterfall.prefill_lookup_ns,
            (unsigned long long)g_dpr_waterfall.prefill_restore_ns,
            (unsigned long long)g_dpr_waterfall.prefill_ordinary_ns,
            (unsigned long long)g_dpr_waterfall.decode_node_ns,
            (unsigned long long)g_dpr_waterfall.decode_lookup_ns,
            (unsigned long long)g_dpr_waterfall.decode_parent_restore_ns,
            (unsigned long long)g_dpr_waterfall.decode_verify_ns,
            (unsigned long long)g_dpr_waterfall.decode_ordinary_ns,
            g_dpr_waterfall.prefill_hits,
            g_dpr_waterfall.prefill_cached_tokens,
            g_dpr_waterfall.decode_cycles,
            g_dpr_waterfall.parent_hits,
            g_dpr_waterfall.nomogram_hits,
            g_dpr_waterfall.decode_misses,
            g_dpr_waterfall.proposed_tokens,
            g_dpr_waterfall.accepted_tokens,
            g_dpr_waterfall.rejection_count) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_prefill(unsigned long long request_id, int prompt_tokens,
                            int hits, int cached_tokens,
                            const uint8_t source_node_sha256[32],
                            const uint8_t qa_key_sha256[32]) {
    char source_node[G4_SHA256_HEX + 1], qa_key[G4_SHA256_HEX + 1];
    if (!source_node_sha256 || !qa_key_sha256) return -1;
    digest_hex(source_node_sha256, source_node);
    digest_hex(qa_key_sha256, qa_key);
    if (prompt_tokens < 1 || hits < 0 || cached_tokens < 0 ||
        cached_tokens >= prompt_tokens ||
        fprintf(stdout,
            "GEMMA4_SERVER_DPR_PREFILL_V1 request=%llu mode=%s "
            "hits=%d cached_tokens=%d prompt_tokens=%d "
            "source_node_sha256=%s qa_key_sha256=%s runtime_ready="
            G4_RUNTIME_READY_TEXT "\n",
            request_id, g_dpr_mode == SALT_DPR_DYNAMIC ? "dynamic" : "persist",
            hits, cached_tokens, prompt_tokens, source_node, qa_key) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_token(SaltGemma4Text *model, const int *output_ids,
                      unsigned long long request_id,
                      unsigned long long turn_id, int output_count,
                      int token, int stop, char *buffer, int buffer_size) {
    int visible_count = stop ? output_count - 1 : output_count;
    int response_bytes = salt_gemma4_text_decode(
        model, output_ids, visible_count, buffer, buffer_size);
    if (response_bytes < 0 || response_bytes >= buffer_size ||
        memchr(buffer, '\0', (size_t)response_bytes) != NULL)
        return -1;
    if (fprintf(stdout,
            "GEMMA4_SERVER_TOKEN_V2 request=%llu session_epoch=%llu "
            "turn_id=%llu seq=%d token_id=%d stop=%d bytes=%d\n",
            request_id, g_session_epoch, turn_id, output_count,
            token, stop, response_bytes) < 0)
        return -1;
    return write_payload(buffer, (size_t)response_bytes);
}

static int prefill_ordinary_measured(SaltGemma4Text *model,
                                     const int *tokens, int token_count,
                                     float *logits) {
    uint64_t start = monotonic_ns(), end;
    int rc = salt_gemma4_text_prefill(model, tokens, token_count, logits);
    end = monotonic_ns();
    if (start && end >= start) g_dpr_waterfall.prefill_ordinary_ns += end - start;
    return rc;
}

static int prefill_parent_dpr(
        SaltGemma4Text *model, const int *tokens, int token_count,
        float *logits, const SaltDprStore *store,
        SaltDprEdgeStats *stats, const uint8_t *retained,
        int *hit_count, int *cached_tokens,
        uint8_t source_node_sha256[32]) {
    int consumed = 0;
    if (!model || !tokens || token_count < 1 || !logits || !store ||
        !stats || !retained || !hit_count || !cached_tokens ||
        !source_node_sha256 ||
        g_dpr_prefill_n == 0)
        return -1;
    *hit_count = 0;
    *cached_tokens = 0;
    memset(source_node_sha256, 0, 32);
    while (token_count - consumed > 1) {
        SaltDprNodeMaterial material;
        SaltDprReferenceLease lease;
        uint8_t node_sha256[32];
        size_t selected = SIZE_MAX;
        uint32_t remaining = (uint32_t)(token_count - consumed - 1);
        uint32_t maximum = remaining < g_dpr_prefill_n
            ? remaining : g_dpr_prefill_n;
        int source_position = salt_gemma4_text_position(model);
        int loaded_position = 0, selection;
        char error[256] = {0};
        uint64_t node_start = monotonic_ns(), node_end;
        uint64_t lookup_start, lookup_end, recover_start, recover_end;
        if (build_dpr_node(model, &material, node_sha256) != 0)
            return -1;
        node_end = monotonic_ns();
        if (node_start && node_end >= node_start)
            g_dpr_waterfall.prefill_node_ns += node_end - node_start;
        if (consumed == 0) memcpy(source_node_sha256, node_sha256, 32);
        lookup_start = monotonic_ns();
        selection = salt_dpr_stats_select_prefill(
            store, stats, retained, node_sha256, g_dpr_mode,
            g_dpr_serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
            salt_gemma4_text_vocab_size(model), g_context,
            tokens + consumed, maximum, &selected);
        lookup_end = monotonic_ns();
        if (lookup_start && lookup_end >= lookup_start)
            g_dpr_waterfall.prefill_lookup_ns += lookup_end - lookup_start;
        if (selection < 0) return -1;
        if (selection == 0) break;
        if (store->edges[selected].edge.reference_position !=
            (uint64_t)source_position +
            (uint64_t)store->edges[selected].edge.horizon)
            return -1;
        memset(&lease, 0, sizeof lease);
        lease.fd = -1;
        if (salt_dpr_reference_open(g_dpr_root, &store->edges[selected], &lease) != 0)
            return -1;
        recover_start = monotonic_ns();
        if (g4_kv_fd_load_delta(
                model, lease.fd, g_context,
                (size_t)store->edges[selected].edge.reference_bytes,
                source_position,
                (int)store->edges[selected].edge.horizon,
                &loaded_position,
                error, sizeof error) != 0 ||
            loaded_position != source_position +
                (int)store->edges[selected].edge.horizon) {
            salt_dpr_reference_close(&lease);
            return -1;
        }
        recover_end = monotonic_ns();
        if (recover_start && recover_end >= recover_start)
            g_dpr_waterfall.prefill_restore_ns += recover_end - recover_start;
        salt_dpr_reference_close(&lease);
        if (stats[selected].last_used_epoch == UINT64_MAX ||
            salt_dpr_stats_observe(
                &stats[selected],
                lookup_start && lookup_end >= lookup_start
                    ? lookup_end - lookup_start : 0,
                recover_start && recover_end >= recover_start
                    ? recover_end - recover_start : 0,
                0, store->edges[selected].edge.horizon,
                store->edges[selected].edge.horizon, 1, 0,
                stats[selected].last_used_epoch + 1u) != 0)
            return -1;
        if (dpr_transition_advance_many(
                tokens + consumed,
                (int)store->edges[selected].edge.horizon) != 0 ||
            g_dpr_transition_position !=
                (uint64_t)salt_gemma4_text_position(model))
            return -1;
        consumed += (int)store->edges[selected].edge.horizon;
        (*hit_count)++;
        *cached_tokens = consumed;
        g_dpr_waterfall.prefill_hits++;
        g_dpr_waterfall.prefill_cached_tokens = (uint32_t)consumed;
    }
    {
        uint64_t ordinary_start = monotonic_ns(), ordinary_end;
        int rc = salt_gemma4_text_prefill(
            model, tokens + consumed, token_count - consumed, logits);
        ordinary_end = monotonic_ns();
        if (ordinary_start && ordinary_end >= ordinary_start)
            g_dpr_waterfall.prefill_ordinary_ns += ordinary_end - ordinary_start;
        if (rc == 0 &&
            (dpr_transition_advance_many(
                tokens + consumed, token_count - consumed) != 0 ||
             g_dpr_transition_position !=
                (uint64_t)salt_gemma4_text_position(model)))
            rc = -1;
        return rc;
    }
}

static int generate_continue(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size, int next,
        int *output_count, int *stop_token) {
    if (!model || !logits || !output_ids || !stream_buffer ||
        stream_buffer_size < 1 || !output_count || !stop_token || next < 0 ||
        *output_count < 0 || *output_count >= g_active_output_limit)
        return -1;
    while (*output_count < g_active_output_limit) {
        int stop;
        output_ids[(*output_count)++] = next;
        stop = is_stop_token(next);
        if (emit_token(model, output_ids, request_id, turn_id,
                       *output_count, next, stop,
                       stream_buffer, stream_buffer_size) != 0)
            return -1;
        if (stop) {
            *stop_token = next;
            break;
        }
        if (*output_count >= g_active_output_limit) break;
        if (salt_gemma4_text_step(model, next, logits) != 0) return -1;
        next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
        if (next < 0) return -1;
    }
    return 0;
}

static int generate(SaltGemma4Text *model, float *logits, int *output_ids,
                    unsigned long long request_id,
                    unsigned long long turn_id,
                    char *stream_buffer, int stream_buffer_size,
                    int *output_count, int *stop_token) {
    int next;
    *output_count = 0;
    *stop_token = -1;
    next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
    if (next < 0) return -1;
    return generate_continue(
        model, logits, output_ids, request_id, turn_id,
        stream_buffer, stream_buffer_size, next, output_count, stop_token);
}

static int generate_measured(SaltGemma4Text *model, float *logits,
                             int *output_ids,
                             unsigned long long request_id,
                             unsigned long long turn_id,
                             char *stream_buffer, int stream_buffer_size,
                             int *output_count, int *stop_token) {
    uint64_t start = monotonic_ns(), end;
    int rc = generate(model, logits, output_ids, request_id, turn_id,
                      stream_buffer, stream_buffer_size,
                      output_count, stop_token);
    end = monotonic_ns();
    if (start && end >= start) g_dpr_waterfall.decode_ordinary_ns += end - start;
    return rc;
}

static int generate_one(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        int *output_count, int *stop_token) {
    int token, stop;
    if (!model || !logits || !output_ids || !stream_buffer ||
        !output_count || !stop_token || *output_count < 0 ||
        *output_count >= g_active_output_limit)
        return -1;
    token = greedy_token(logits, salt_gemma4_text_vocab_size(model));
    if (token < 0) return -1;
    output_ids[(*output_count)++] = token;
    stop = is_stop_token(token);
    if (emit_token(model, output_ids, request_id, turn_id, *output_count,
                   token, stop, stream_buffer, stream_buffer_size) != 0)
        return -1;
    if (stop) {
        *stop_token = token;
        return 0;
    }
    if (*output_count < g_active_output_limit &&
        salt_gemma4_text_step(model, token, logits) != 0)
        return -1;
    return 0;
}

static int generate_one_measured(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        int *output_count, int *stop_token) {
    uint64_t start = monotonic_ns(), end;
    int source_position = salt_gemma4_text_position(model);
    int output_base = output_count ? *output_count : -1;
    int rc = generate_one(
        model, logits, output_ids, request_id, turn_id,
        stream_buffer, stream_buffer_size, output_count, stop_token);
    end = monotonic_ns();
    if (start && end >= start) g_dpr_waterfall.decode_ordinary_ns += end - start;
    if (rc == 0) {
        int result_position = salt_gemma4_text_position(model);
        if (result_position == source_position + 1) {
            if (output_base < 0 || dpr_transition_advance(output_ids[output_base]) != 0)
                return -1;
        } else if (result_position != source_position) {
            return -1;
        }
        if (g_dpr_transition_position != (uint64_t)result_position)
            return -1;
    }
    return rc;
}

static int read_exact(FILE *stream, void *buffer, size_t bytes) {
    return bytes == 0 || fread(buffer, 1, bytes, stream) == bytes ? 0 : -1;
}

static int dpr_verify(
        SaltGemma4Text *model, float *initial_logits,
        const G4DprReply *reply, int source_position,
        int *output_ids, int output_base,
        int *output_count, int *stop_token, int *accepted_out,
        int *committed_out, int *pending_final_out,
        int *pending_token_out) {
    SaltTextVerifyResult transition;
    int accepted;
    int token;
    if (!model || !initial_logits || !reply || !reply->hit ||
        source_position < 1 || !output_ids || output_base < 0 ||
        output_base >= g_active_output_limit ||
        !output_count || !stop_token || !accepted_out || !committed_out ||
        !pending_final_out || !pending_token_out ||
        reply->horizon < 1 ||
        reply->horizon > G4_DPR_MAX_HORIZON ||
        output_base > g_active_output_limit - reply->horizon - 1 ||
        source_position + reply->horizon > g_context)
        return -1;
    memset(&transition, 0, sizeof transition);
    if (salt_gemma4_text_target_block(
            model, reply->candidate_ids, reply->horizon,
            initial_logits, &transition) != 0 ||
        transition.status != SALT_TEXT_VERIFY_COMMITTED ||
        transition.accepted_count > (uint32_t)reply->horizon ||
        transition.committed_count != transition.accepted_count ||
        transition.produced_count != transition.accepted_count + 1u ||
        transition.result_position !=
            (uint32_t)source_position + transition.accepted_count ||
        transition.backend.engine_submissions !=
            (transition.accepted_count ? 1u : 0u) ||
        transition.backend.completion_fences !=
            (transition.accepted_count ? 1u : 0u) ||
        transition.backend.intermediate_host_publications != 0u)
        return -1;
    accepted = (int)transition.accepted_count;
    token = transition.pending_token_id;
    memcpy(output_ids + output_base, reply->candidate_ids,
           (size_t)accepted * sizeof output_ids[0]);
    output_ids[output_base + accepted] = token;
    *output_count = output_base + accepted + 1;
    *stop_token = is_stop_token(token) ? token : -1;
    *accepted_out = accepted;
    *committed_out = accepted;
    *pending_final_out = 1;
    *pending_token_out = token;
    return 0;
}

static int emit_dpr_result(
        unsigned long long request_id, const G4DprReply *reply,
        int source_position, int accepted, int produced, int committed,
        int pending_final, int switched) {
    if (!reply || !reply->hit || accepted < 0 || accepted > reply->horizon ||
        produced != accepted + 1 || committed < accepted ||
        committed > produced || (pending_final != 0 && pending_final != 1) ||
        committed + pending_final != produced ||
        (switched != 0 && switched != 1) ||
        fprintf(stdout,
            "GEMMA4_SERVER_DPR_RESULT_V1 request=%llu mode=%s edge_id=%s "
            "kind=%s source_position=%d horizon=%d accepted=%d produced=%d "
            "committed=%d pending_final=%d switched=%d runtime_ready="
            G4_RUNTIME_READY_TEXT "\n",
            request_id, reply->dynamic ? "dynamic" : "persist",
            reply->edge_id,
            reply->kind == SALT_DPR_EDGE_PARENT_EXACT ? "parent" : "nomogram",
            source_position, reply->horizon, accepted,
            produced, committed, pending_final, switched) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_mentor(
        unsigned long long request_id, SaltDprMode mode,
        int source_position, uint32_t horizon,
        const SaltDprSelectorResult *result) {
    char source[65], family[65], mindset[65], binding[65], edge[65];
    if (!result || result->kind != SALT_DPR_SELECTOR_MENTOR_NOMOGRAM ||
        source_position < 1 || horizon == 0 ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    digest_hex(result->source_node_sha256, source);
    digest_hex(result->family_sha256, family);
    digest_hex(result->mindset_sha256, mindset);
    digest_hex(result->binding_sha256, binding);
    digest_hex(result->edge_sha256, edge);
    if (fprintf(stdout,
            "GEMMA4_SERVER_DPR_MENTOR_V1 request=%llu mode=%s "
            "source_position=%d horizon=%u source_node_sha256=%s "
            "family_sha256=%s mindset_sha256=%s binding_sha256=%s "
            "edge_id=%s expected_saved_numerator=%llu "
            "expected_saved_denominator=%llu runtime_ready="
            G4_RUNTIME_READY_TEXT "\n",
            request_id, mode == SALT_DPR_DYNAMIC ? "dynamic" : "persist",
            source_position, horizon, source, family, mindset, binding, edge,
            (unsigned long long)result->expected_saved_numerator,
            (unsigned long long)result->expected_saved_denominator) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_attention(
        unsigned long long request_id, SaltDprMode mode,
        int source_position, const G4DprAttentionObservation *observation) {
    char plan[65], dpr[65], mindset[65];
    const char *cell;
    if (!observation || !observation->applied || source_position < 0 ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    cell = observation->cell == SALT_DPR_CELL_ND_NM ? "ND_NM" :
        (observation->cell == SALT_DPR_CELL_NM_ONLY ? "NM_ONLY" : NULL);
    if (!cell) return -1;
    digest_hex(observation->plan_sha256, plan);
    digest_hex(observation->dpr_sha256, dpr);
    digest_hex(observation->mindset_sha256, mindset);
    if (fprintf(stdout,
            "GEMMA4_SERVER_DPR_ATTENTION_V1 request=%llu mode=%s cell=%s "
            "source_position=%d plan_sha256=%s dpr_sha256=%s "
            "mindset_sha256=%s lookup_ns=%llu prepare_ns=%llu "
            "requested=%u prepared=%u resident_hits=%u fetch_jobs=%u "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, mode == SALT_DPR_DYNAMIC ? "dynamic" : "persist",
            cell, source_position, plan, dpr, mindset,
            (unsigned long long)observation->lookup_ns,
            (unsigned long long)observation->prepare_ns,
            observation->prepare.requested_items,
            observation->prepare.prepared_items,
            observation->prepare.resident_hits,
            observation->prepare.fetch_jobs) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_attention_ready(
        unsigned long long request_id, int source_position,
        const G4DprAttentionObservation *observation) {
    if (!observation || !observation->applied || source_position < 0 ||
        observation->lease.ready_matches > observation->lease.count ||
        fprintf(stdout,
            "GEMMA4_SERVER_DPR_ATTENTION_READY_V1 request=%llu "
            "source_position=%d exact_routes=%u ready_matches=%u unused=%u "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, source_position, observation->lease.exact_routes,
            observation->lease.ready_matches,
            observation->lease.count - observation->lease.ready_matches) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_miss(
        unsigned long long request_id, SaltDprMode mode,
        const uint8_t source_node_sha256[32], int source_position,
        size_t rejected_edges) {
    char node_hex[G4_SHA256_HEX + 1];
    if (!source_node_sha256 || source_position < 1 ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    digest_hex(source_node_sha256, node_hex);
    if (fprintf(stdout,
            "GEMMA4_SERVER_DPR_MISS_V1 request=%llu mode=%s "
            "source_node_sha256=%s source_position=%d rejected_edges=%zu "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            request_id, mode == SALT_DPR_DYNAMIC ? "dynamic" : "persist",
            node_hex, source_position, rejected_edges) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_dpr_fallback(
        unsigned long long request_id, const G4DprReply *reply,
        int source_position, const char *code) {
    if (!reply || !reply->hit || !code ||
        fprintf(stdout,
            "GEMMA4_SERVER_DPR_FALLBACK_V1 request=%llu mode=%s edge_id=%s "
            "source_position=%d code=%s mutated=0 runtime_ready="
            G4_RUNTIME_READY_TEXT "\n",
            request_id, reply->dynamic ? "dynamic" : "persist",
            reply->edge_id, source_position, code) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int record_dpr_stats(
        const char *root, unsigned long long request_id,
        const G4DprReply *reply, SaltDprEdgeStats *stats, int retained,
        uint64_t lookup_ns, uint64_t recover_ns, uint64_t verify_ns,
        uint32_t accepted, uint32_t committed, int hit, int fallback) {
    uint64_t epoch;
    if (!root || !reply || !stats || stats->last_used_epoch == UINT64_MAX)
        return -1;
    epoch = stats->last_used_epoch + 1u;
    if (salt_dpr_stats_observe(
            stats, lookup_ns, recover_ns, verify_ns,
            accepted, committed, hit, fallback, epoch) != 0)
        return -1;
    (void)root;
    (void)request_id;
    (void)reply;
    (void)retained;
    return 0;
}

static int record_nomogram_chain_stats(
        unsigned long long request_id, const G4DprReply *reply,
        SaltDprEdgeStats *stats, const uint8_t *retained,
        const size_t *edge_indices, uint32_t edge_count,
        uint64_t lookup_ns, uint64_t verify_ns,
        uint32_t accepted, int memory_fallback) {
    uint32_t attempted;
    if (!reply || !stats || !retained || !edge_indices || edge_count == 0 ||
        accepted > edge_count || (memory_fallback != 0 && memory_fallback != 1))
        return -1;
    attempted = memory_fallback ? 0u :
        (accepted < edge_count ? accepted + 1u : edge_count);
    for (uint32_t edge = 0; edge < edge_count; edge++) {
        size_t index = edge_indices[edge];
        uint64_t edge_lookup = lookup_ns / edge_count +
            (edge < lookup_ns % edge_count ? 1u : 0u);
        uint64_t edge_verify = attempted && edge < attempted
            ? verify_ns / attempted + (edge < verify_ns % attempted ? 1u : 0u)
            : 0;
        uint32_t edge_accepted = !memory_fallback && edge < accepted ? 1u : 0u;
        uint32_t edge_committed =
            !memory_fallback && edge < accepted ? 1u : 0u;
        if (record_dpr_stats(
                g_dpr_root, request_id, reply, &stats[index], retained[index],
                edge_lookup, 0, edge_verify, edge_accepted, edge_committed,
                memory_fallback ? 0 : 1, memory_fallback) != 0)
            return -1;
    }
    return 0;
}

static int generate_dpr(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        const G4DprReply *reply, int source_position,
        int emit_cycle, int switched, SaltDprRuntime *runtime,
        G4DprObservation *observation,
        int *output_count, int *stop_token) {
    int accepted = 0;
    int committed = 0;
    int pending_final = 0;
    int pending_token = -1;
    int output_base;
    uint64_t verify_start, verify_end;
    if (!runtime || !observation || !output_count ||
        salt_dpr_verify_begin(runtime) != 0)
        return -1;
    output_base = *output_count;
    memset(observation, 0, sizeof *observation);
    verify_start = monotonic_ns();
    if (dpr_verify(
            model, logits, reply, source_position, output_ids, output_base,
            output_count, stop_token, &accepted, &committed,
            &pending_final, &pending_token) != 0) {
        (void)salt_dpr_runtime_fail(runtime);
        return -1;
    }
    verify_end = monotonic_ns();
    observation->accepted = (uint32_t)accepted;
    observation->committed = (uint32_t)committed;
    observation->verify_ns = verify_start && verify_end >= verify_start
        ? verify_end - verify_start : 0;
    if (salt_dpr_verify_finish_pending(
            runtime, pending_token,
            (uint64_t)salt_gemma4_text_position(model)) != 0) {
        (void)salt_dpr_runtime_fail(runtime);
        return -1;
    }
    if (emit_cycle && emit_dpr_result(
            request_id, reply, source_position,
            accepted, *output_count - output_base, committed,
            pending_final, switched) != 0) {
        (void)salt_dpr_runtime_fail(runtime);
        return -1;
    }
    for (int i = output_base; i < *output_count; i++) {
        int stop = is_stop_token(output_ids[i]);
        if (emit_token(model, output_ids, request_id, turn_id,
                       i + 1, output_ids[i], stop,
                       stream_buffer, stream_buffer_size) != 0) {
            (void)salt_dpr_runtime_fail(runtime);
            return -1;
        }
        if (stop) return i + 1 == *output_count ? 0 : -1;
    }
    return 0;
}

static int dpr_consume_pending(
        SaltGemma4Text *model, float *logits, int token,
        SaltDprRuntime *runtime) {
    int32_t pending_token;
    uint64_t pending_position;
    int pending;
    if (!model || !runtime) return -1;
    pending = salt_dpr_pending_read(
        runtime, &pending_token, &pending_position);
    if (pending <= 0) return pending;
    if (token != pending_token ||
        (uint64_t)salt_gemma4_text_position(model) != pending_position ||
        salt_gemma4_text_step(model, token, logits) != 0 ||
        dpr_transition_advance(token) != 0 ||
        (uint64_t)salt_gemma4_text_position(model) != pending_position + 1u ||
        g_dpr_transition_position != pending_position + 1u ||
        salt_dpr_pending_commit(
            runtime, pending_token, pending_position + 1u) != 0) {
        (void)salt_dpr_runtime_fail(runtime);
        return -1;
    }
    return 1;
}

static int generate_parent_exact(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        const SaltDprStoredEdge *stored, const G4DprReply *reply,
        SaltDprEdgeStats *stats, int retained,
        int emit_cycle, int source_position, uint64_t lookup_ns,
        int *output_count, int *stop_token) {
    SaltDprReferenceLease lease;
    char error[256] = {0};
    int loaded_position = 0, committed, pending_final, output_base;
    uint64_t recover_start, recover_end, recover_ns = 0;
    if (!model || !logits || !output_ids || !stream_buffer || !stored ||
        !reply || reply->kind != SALT_DPR_EDGE_PARENT_EXACT || !stats ||
        !output_count || !stop_token || source_position < 1 ||
        reply->horizon < 1 || reply->horizon >= g_active_output_limit)
        return -1;
    output_base = *output_count;
    if (output_base < 0 ||
        output_base > g_active_output_limit - reply->horizon - 1)
        return 1;
    memset(&lease, 0, sizeof lease);
    lease.fd = -1;
    if (salt_dpr_reference_open(g_dpr_root, stored, &lease) != 0)
        return -1;
    recover_start = monotonic_ns();
    if (g4_kv_fd_load_delta(
            model, lease.fd, g_context, reply->reference_bytes,
            source_position, reply->horizon, &loaded_position,
            error, sizeof error) != 0 ||
        loaded_position != source_position + reply->horizon) {
        salt_dpr_reference_close(&lease);
        return -1;
    }
    recover_end = monotonic_ns();
    if (recover_start && recover_end >= recover_start) {
        recover_ns = recover_end - recover_start;
        g_dpr_waterfall.decode_parent_restore_ns += recover_ns;
    }
    salt_dpr_reference_close(&lease);
    if (dpr_transition_advance_many(
            reply->candidate_ids, reply->horizon) != 0 ||
        g_dpr_transition_position !=
            (uint64_t)salt_gemma4_text_position(model))
        return -1;
    memcpy(output_ids + output_base, reply->candidate_ids,
           (size_t)reply->horizon * sizeof output_ids[0]);
    output_ids[output_base + reply->horizon] = reply->bonus_token_id;
    *output_count = output_base + reply->horizon + 1;
    *stop_token = is_stop_token(reply->bonus_token_id)
        ? reply->bonus_token_id : -1;
    committed = reply->horizon;
    pending_final = *stop_token == -1 ? 0 : 1;
    if (*stop_token == -1) {
        if (salt_gemma4_text_step(model, reply->bonus_token_id, logits) != 0)
            return -1;
        if (dpr_transition_advance(reply->bonus_token_id) != 0 ||
            g_dpr_transition_position !=
                (uint64_t)salt_gemma4_text_position(model))
            return -1;
        committed++;
    }
    if ((emit_cycle && emit_dpr_result(
            request_id, reply, source_position, reply->horizon,
            reply->horizon + 1, committed, pending_final,
            reply->dynamic ? 1 : 0) != 0) ||
        record_dpr_stats(
            g_dpr_root, request_id, reply, stats, retained,
            lookup_ns, recover_ns, 0, (uint32_t)reply->horizon,
            (uint32_t)committed, 1, 0) != 0)
        return -1;
    for (int i = output_base; i < *output_count; i++) {
        int stop = is_stop_token(output_ids[i]);
        if (emit_token(model, output_ids, request_id, turn_id,
                       i + 1, output_ids[i], stop,
                       stream_buffer, stream_buffer_size) != 0)
            return -1;
        if (stop) return i + 1 == *output_count ? 0 : -1;
    }
    return 0;
}

static int dpr_decode_cycle(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        SaltDprStore *store,
        SaltDprAttentionPlanStore *attention_store,
        const SaltDprMentorFamilyStore *family_store,
        const SaltDprMentorBindingStore *binding_store,
        SaltDprSelectorCandidate *mentor_candidates,
        size_t mentor_candidate_capacity,
        SaltDprChart *mentor_effective_charts,
        SaltDprEdgeStats *stats,
        uint8_t *retained, SaltDprRuntime *runtime,
        int emit_cycle,
        const uint8_t qa_key_sha256[32],
        int remaining_output,
        int *output_count, int *stop_token) {
    SaltDprNodeMaterial node_material;
    SaltDprMemoryRequest memory_request;
    SaltDprMemoryDecision memory_decision;
    G4DprReply reply;
    G4DprObservation observation;
    G4DprAttentionObservation attention_observation;
    SaltDprComputeIntent compute_intent;
    uint8_t node_sha256[32];
    int32_t nomogram_ids[G4_DPR_MAX_HORIZON];
    size_t nomogram_indices[G4_DPR_MAX_HORIZON];
    size_t mentor_binding_indices[G4_DPR_MAX_HORIZON];
    SaltDprSelectorResult mentor_result;
    size_t selected = SIZE_MAX;
    size_t mentor_candidate_count = 0;
    uint32_t nomogram_count = 0;
    int source_position = salt_gemma4_text_position(model);
    int output_base = output_count ? *output_count : -1;
    int selection, result, mentor_selected = 0;
    uint64_t node_start, node_end, lookup_start, lookup_end, lookup_ns = 0;
    if (!model || !logits || !output_ids || !stream_buffer || !store ||
        !stats || !retained || !runtime ||
        !qa_key_sha256 || remaining_output < 1 || !output_count ||
        !stop_token || source_position < 1)
        return -1;
    if (remaining_output == 1)
        return generate_one_measured(
            model, logits, output_ids, request_id, turn_id,
            stream_buffer, stream_buffer_size, output_count, stop_token);
    if (g_dpr_mentor_enabled &&
        (!family_store || !binding_store || !mentor_candidates ||
         mentor_candidate_capacity == 0 || !mentor_effective_charts))
        return -1;
    if (g_dpr_attention_enabled && !attention_store) return -1;

    g_dpr_waterfall.decode_cycles++;
    node_start = monotonic_ns();
    if (build_dpr_node(model, &node_material, node_sha256) != 0)
        return -1;
    node_end = monotonic_ns();
    if (node_start && node_end >= node_start)
        g_dpr_waterfall.decode_node_ns += node_end - node_start;
    lookup_start = monotonic_ns();

    /* Level 1: canonical parent transition cache always has priority. */
    selection = salt_dpr_stats_store_select(
        store, stats, retained, node_sha256,
        g_dpr_mode, g_dpr_serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
        salt_gemma4_text_vocab_size(model), g_context,
        dpr_stop_token, NULL, &selected);
    if (selection < 0) return -1;
    if (selection > 0) {
        lookup_end = monotonic_ns();
        if (lookup_start && lookup_end >= lookup_start) {
            lookup_ns = lookup_end - lookup_start;
            g_dpr_waterfall.decode_lookup_ns += lookup_ns;
        }
        if (dpr_reply_from_stored(&store->edges[selected], &reply) != 0 ||
            reply.kind != SALT_DPR_EDGE_PARENT_EXACT ||
            source_position > g_context - reply.horizon - 1 ||
            store->edges[selected].edge.reference_position !=
                (uint64_t)source_position + (uint64_t)reply.horizon)
            return -1;
        if (reply.horizon + 1 > remaining_output) {
            g_dpr_waterfall.decode_misses++;
            return generate_one_measured(
                model, logits, output_ids, request_id, turn_id,
                stream_buffer, stream_buffer_size, output_count, stop_token);
        }
        g_dpr_waterfall.parent_hits++;
        result = generate_parent_exact(
            model, logits, output_ids, request_id, turn_id,
            stream_buffer, stream_buffer_size, &store->edges[selected],
            &reply, &stats[selected], retained[selected], emit_cycle, source_position,
            lookup_ns, output_count, stop_token);
        if (result < 0) return -1;
        if (result > 0) {
            g_dpr_waterfall.decode_misses++;
            return generate_one_measured(
                model, logits, output_ids, request_id, turn_id,
                stream_buffer, stream_buffer_size, output_count, stop_token);
        }
        return 0;
    }

    /* Level 2a: optional family-bound Mentor selector. */
    memset(&mentor_result, 0, sizeof mentor_result);
    if (g_dpr_mentor_enabled) {
        uint8_t compatibility[32];
        for (size_t edge = 0; edge < store->edge_count; edge++) {
            mentor_effective_charts[edge] = store->edges[edge].edge.chart;
            if (stats[edge].round_count >= SALT_DPR_STATS_MIN_SAMPLES &&
                salt_dpr_stats_chart(
                    &stats[edge], &mentor_effective_charts[edge]) != 0)
                return -1;
        }
        if (salt_sha256_hex_parse(
                SALT_GEMMA4_KV_COMPAT_SHA256, compatibility) != 0 ||
            salt_dpr_mentor_candidates_build(
                binding_store, family_store, store, retained,
                mentor_effective_charts, g_dpr_mode,
                node_sha256, g_dpr_serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
                mentor_candidates, mentor_candidate_capacity,
                &mentor_candidate_count) != 0)
            return -1;
        selection = salt_dpr_selector_choose(
            family_store->families, family_store->family_count,
            mentor_candidates, mentor_candidate_count, node_sha256,
            compatibility, g_dpr_mentor_policy_sha256, &mentor_result);
        if (selection < 0) return -1;
        if (selection > 0) {
            selection = salt_dpr_mentor_walk_nomogram(
                binding_store, family_store, store, retained,
                mentor_effective_charts,
                mentor_result.family_index, g_dpr_mode, node_sha256,
                g_dpr_serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
                salt_gemma4_text_vocab_size(model), g_context,
                dpr_stop_token, NULL,
                g_dpr_draft_n < (uint32_t)(remaining_output - 1)
                    ? g_dpr_draft_n : (uint32_t)(remaining_output - 1),
                nomogram_ids, nomogram_indices, mentor_binding_indices,
                &nomogram_count);
            if (selection < 0) return -1;
            if (selection > 0) {
                if (memcmp(store->edges[nomogram_indices[0]].edge_sha256,
                           mentor_result.edge_sha256, 32) != 0 ||
                    memcmp(binding_store->bindings[mentor_binding_indices[0]].
                               binding_sha256,
                           mentor_result.binding_sha256, 32) != 0)
                    return -1;
                mentor_selected = 1;
            }
        }
    }

    /* Level 2b: retained legacy flat Nomogram and Q&A-key fallback. */
    if (!mentor_selected) {
        selection = salt_dpr_stats_walk_nomogram(
            store, stats, retained, node_sha256, g_dpr_mode, g_dpr_serial_ns,
            SALT_DPR_STATS_MIN_SAMPLES, salt_gemma4_text_vocab_size(model),
            g_context, dpr_stop_token, NULL,
            g_dpr_draft_n < (uint32_t)(remaining_output - 1)
                ? g_dpr_draft_n : (uint32_t)(remaining_output - 1),
            nomogram_ids, nomogram_indices, &nomogram_count);
    }
    if (!mentor_selected && selection == 0 && *output_count == 0) {
        selected = SIZE_MAX;
        nomogram_count = 0;
        selection = salt_dpr_stats_walk_nomogram(
            store, stats, retained, qa_key_sha256,
            g_dpr_mode, g_dpr_serial_ns, SALT_DPR_STATS_MIN_SAMPLES,
            salt_gemma4_text_vocab_size(model), g_context,
            dpr_stop_token, NULL,
            g_dpr_draft_n < (uint32_t)(remaining_output - 1)
                ? g_dpr_draft_n : (uint32_t)(remaining_output - 1),
            nomogram_ids, nomogram_indices, &nomogram_count);
    }
    lookup_end = monotonic_ns();
    if (lookup_start && lookup_end >= lookup_start) {
        lookup_ns = lookup_end - lookup_start;
        g_dpr_waterfall.decode_lookup_ns += lookup_ns;
    }
    if (selection < 0) return -1;
    if (selection == 0) {
        int attention_rc;
        g_dpr_waterfall.decode_misses++;
        if (emit_cycle && emit_dpr_miss(
                request_id, g_dpr_mode, node_sha256, source_position,
                store->rejected_edges) != 0)
            return -1;
        if (dpr_compute_intent(
                SALT_DPR_OPERATION_NATIVE_DECODE, node_sha256,
                (uint64_t)source_position, NULL, 0, &compute_intent) != 0)
            return -1;
        attention_rc = dpr_attention_prepare(
            model, attention_store, &compute_intent, 0,
            &attention_observation);
        if (attention_rc < 0) return -1;
        if (attention_rc > 0 && emit_cycle && emit_dpr_attention(
                request_id, g_dpr_mode, source_position,
                &attention_observation) != 0) {
            (void)dpr_attention_release(model, &attention_observation);
            return -1;
        }
        result = generate_one_measured(
            model, logits, output_ids, request_id, turn_id,
            stream_buffer, stream_buffer_size, output_count, stop_token);
        if (dpr_attention_release(model, &attention_observation) != 0)
            return -1;
        if (result != 0) return result;
        if (attention_rc > 0 && emit_cycle && emit_dpr_attention_ready(
                request_id, source_position, &attention_observation) != 0)
            return -1;
        return 0;
    }

    g_dpr_waterfall.nomogram_hits++;
    g_dpr_waterfall.proposed_tokens += nomogram_count;
    selected = nomogram_indices[0];
    memset(&reply, 0, sizeof reply);
    reply.hit = 1;
    reply.dynamic = g_dpr_mode == SALT_DPR_DYNAMIC;
    reply.kind = SALT_DPR_EDGE_NOMOGRAM_DRAFT;
    reply.horizon = (int)nomogram_count;
    reply.bonus_token_id = -1;
    memcpy(reply.candidate_ids, nomogram_ids,
           (size_t)nomogram_count * sizeof nomogram_ids[0]);
    digest_hex(store->edges[selected].edge_sha256, reply.edge_id);
    if ((mentor_selected && emit_cycle &&
         emit_dpr_mentor(
            request_id, g_dpr_mode, source_position,
            nomogram_count, &mentor_result) != 0) ||
        reply.horizon + 1 > remaining_output ||
        source_position > g_context - reply.horizon - 1)
        return -1;

    memset(&memory_request, 0, sizeof memory_request);
    memory_request.mode = g_dpr_mode;
    memory_request.current_kv_budget_bytes = g_dpr_current_kv_bytes;
    memory_request.additive_dpr_budget_bytes = g_dpr_additive_bytes;
    memory_request.target_required_bytes =
        salt_gemma4_text_kv_required_bytes(
            model, source_position + reply.horizon + 1);
    memory_request.reference_required_bytes = 0;
    memory_request.verification_scratch_bytes = 0;
    memory_request.checkpoint_max_bytes = 0;
    if (salt_dpr_memory_admit(&memory_request, &memory_decision) != 0)
        return -1;
    if (!memory_decision.admitted) {
        if ((emit_cycle && emit_dpr_fallback(
                request_id, &reply, source_position, "memory_reject") != 0) ||
            record_nomogram_chain_stats(
                request_id, &reply, stats, retained,
                nomogram_indices, nomogram_count,
                lookup_ns, 0, 0, 1) != 0)
            return -1;
        if (dpr_compute_intent(
                SALT_DPR_OPERATION_NATIVE_DECODE, node_sha256,
                (uint64_t)source_position, NULL, 0, &compute_intent) != 0)
            return -1;
        result = dpr_attention_prepare(
            model, attention_store, &compute_intent, 0,
            &attention_observation);
        if (result < 0) return -1;
        if (result > 0 && emit_cycle && emit_dpr_attention(
                request_id, g_dpr_mode, source_position,
                &attention_observation) != 0) {
            (void)dpr_attention_release(model, &attention_observation);
            return -1;
        }
        {
            int attention_rc = result;
            result = generate_one_measured(
            model, logits, output_ids, request_id, turn_id,
            stream_buffer, stream_buffer_size, output_count, stop_token);
            if (dpr_attention_release(model, &attention_observation) != 0)
                return -1;
            if (result != 0) return result;
            if (attention_rc > 0 && emit_cycle && emit_dpr_attention_ready(
                    request_id, source_position, &attention_observation) != 0)
                return -1;
            return 0;
        }
    }
    if (dpr_compute_intent(
            SALT_DPR_OPERATION_DRAFT_VERIFY, node_sha256,
            (uint64_t)source_position, nomogram_ids, nomogram_count,
            &compute_intent) != 0)
        return -1;
    result = dpr_attention_prepare(
        model, attention_store, &compute_intent, 1,
        &attention_observation);
    if (result < 0) return -1;
    if (result > 0 && emit_cycle && emit_dpr_attention(
            request_id, g_dpr_mode, source_position,
            &attention_observation) != 0) {
        (void)dpr_attention_release(model, &attention_observation);
        return -1;
    }
    {
        int attention_rc = result;
    result = generate_dpr(
        model, logits, output_ids, request_id, turn_id,
        stream_buffer, stream_buffer_size, &reply,
        source_position, emit_cycle, 0, runtime, &observation,
        output_count, stop_token);
        if (dpr_attention_release(model, &attention_observation) != 0)
            return -1;
        if (result == 0 && attention_rc > 0 && emit_cycle &&
                emit_dpr_attention_ready(request_id, source_position,
                                         &attention_observation) != 0)
            return -1;
    }
    if (result != 0) return result;
    if (output_base < 0 ||
        dpr_transition_advance_many(
            output_ids + output_base, (int)observation.committed) != 0 ||
        g_dpr_transition_position !=
            (uint64_t)salt_gemma4_text_position(model))
        return -1;
    g_dpr_waterfall.decode_verify_ns += observation.verify_ns;
    g_dpr_waterfall.accepted_tokens += observation.accepted;
    if (observation.accepted < nomogram_count)
        g_dpr_waterfall.rejection_count++;
    return record_nomogram_chain_stats(
        request_id, &reply, stats, retained,
        nomogram_indices, nomogram_count,
        lookup_ns, observation.verify_ns, observation.accepted, 0);
}

static int generate_native_dpr(
        SaltGemma4Text *model, float *logits, int *output_ids,
        unsigned long long request_id, unsigned long long turn_id,
        char *stream_buffer, int stream_buffer_size,
        SaltDprStore *store,
        SaltDprAttentionPlanStore *attention_store,
        const SaltDprMentorFamilyStore *family_store,
        const SaltDprMentorBindingStore *binding_store,
        SaltDprSelectorCandidate *mentor_candidates,
        size_t mentor_candidate_capacity,
        SaltDprChart *mentor_effective_charts,
        SaltDprEdgeStats *stats,
        uint8_t *retained, SaltDprRuntime *runtime,
        const uint8_t qa_key_sha256[32],
        int *output_count, int *stop_token) {
    if (!output_count || !stop_token) return -1;
    *output_count = 0;
    *stop_token = -1;
    while (*output_count < g_active_output_limit && *stop_token == -1) {
        int pending = dpr_consume_pending(
            model, logits,
            *output_count > 0 ? output_ids[*output_count - 1] : -1,
            runtime);
        int before = *output_count;
        if (pending < 0 || dpr_decode_cycle(
                model, logits, output_ids, request_id, turn_id,
                stream_buffer, stream_buffer_size, store,
                attention_store,
                family_store, binding_store,
                mentor_candidates, mentor_candidate_capacity,
                mentor_effective_charts,
                stats, retained,
                runtime,
                before == 0, qa_key_sha256,
                g_active_output_limit - *output_count,
                output_count, stop_token) != 0 || *output_count <= before)
            return -1;
    }
    return 0;
}

static int parse_turn_header(
        const char *line,
        unsigned long long *request_id,
        unsigned long long *session_epoch,
        unsigned long long *turn_id,
        unsigned long long *history_turns,
        int *position,
        char client_request_sha256[G4_SHA256_HEX + 1],
        char request_sha256[G4_SHA256_HEX + 1],
        size_t *prompt_bytes,
        size_t *image_path_bytes,
        int *output_limit,
        int *proof_state) {
    unsigned long long raw_position, raw_prompt, raw_image;
    unsigned long long raw_output, raw_proof;
    char extra;
    char canonical[512];
    int n;
    if (!line || !request_id || !session_epoch || !turn_id ||
        !history_turns || !position || !client_request_sha256 ||
        !request_sha256 || !prompt_bytes || !image_path_bytes ||
        !output_limit || !proof_state)
        return -1;
    if (sscanf(line,
            "SALT_GEMMA4_TURN_V3 request=%llu session_epoch=%llu "
            "turn_id=%llu history_turns=%llu position=%llu "
            "prompt_bytes=%llu image_path_bytes=%llu "
            "output_limit=%llu proof_state=%llu "
            "client_request_sha256=%64s request_sha256=%64s %c",
            request_id, session_epoch, turn_id, history_turns,
            &raw_position, &raw_prompt, &raw_image, &raw_output, &raw_proof,
            client_request_sha256, request_sha256, &extra) != 11)
        return -1;
    n = snprintf(canonical, sizeof canonical,
            "SALT_GEMMA4_TURN_V3 request=%llu session_epoch=%llu "
            "turn_id=%llu history_turns=%llu position=%llu "
            "prompt_bytes=%llu image_path_bytes=%llu "
            "output_limit=%llu proof_state=%llu "
            "client_request_sha256=%s request_sha256=%s\n",
            *request_id, *session_epoch, *turn_id, *history_turns,
            raw_position, raw_prompt, raw_image, raw_output, raw_proof,
            client_request_sha256, request_sha256);
    if (n < 0 || (size_t)n >= sizeof canonical || strcmp(line, canonical) ||
        raw_position > INT_MAX || raw_prompt > G4_MAX_PROMPT_BYTES ||
        raw_image > G4_MAX_IMAGE_PATH || raw_output < 1 ||
        raw_output > (unsigned long long)g_output_limit || raw_proof > 1 ||
        !is_sha256_hex(client_request_sha256) ||
        !is_sha256_hex(request_sha256))
        return -1;
    *position = (int)raw_position;
    *prompt_bytes = (size_t)raw_prompt;
    *image_path_bytes = (size_t)raw_image;
    *output_limit = (int)raw_output;
    *proof_state = (int)raw_proof;
    return 0;
}

static int parse_replay_header(
        const char *line,
        unsigned long long *request_id,
        unsigned long long *session_epoch,
        char client_request_sha256[G4_SHA256_HEX + 1],
        char request_sha256[G4_SHA256_HEX + 1]) {
    char canonical[320];
    char extra;
    int n;
    if (!line || !request_id || !session_epoch ||
        !client_request_sha256 || !request_sha256)
        return -1;
    if (sscanf(line,
            "SALT_GEMMA4_REPLAY_V2 request=%llu session_epoch=%llu "
            "client_request_sha256=%64s request_sha256=%64s %c",
            request_id, session_epoch, client_request_sha256,
            request_sha256, &extra) != 4)
        return -1;
    n = snprintf(canonical, sizeof canonical,
            "SALT_GEMMA4_REPLAY_V2 request=%llu session_epoch=%llu "
            "client_request_sha256=%s request_sha256=%s\n",
            *request_id, *session_epoch, client_request_sha256,
            request_sha256);
    if (n < 0 || (size_t)n >= sizeof canonical || strcmp(line, canonical) ||
        !is_sha256_hex(client_request_sha256) ||
        !is_sha256_hex(request_sha256))
        return -1;
    return 0;
}

static int parse_clear_header(
        const char *line,
        unsigned long long *request_id,
        unsigned long long *session_epoch,
        unsigned long long *turn_id,
        unsigned long long *history_turns,
        int *position,
        char state_sha256[G4_SHA256_HEX + 1]) {
    unsigned long long raw_position;
    char canonical[352], extra;
    int n;
    if (!line || !request_id || !session_epoch || !turn_id ||
        !history_turns || !position || !state_sha256)
        return -1;
    if (sscanf(line,
            "SALT_GEMMA4_CLEAR_FACTS_V2 request=%llu session_epoch=%llu "
            "turn_id=%llu history_turns=%llu position=%llu "
            "state_sha256=%64s %c",
            request_id, session_epoch, turn_id, history_turns,
            &raw_position, state_sha256, &extra) != 6)
        return -1;
    n = snprintf(canonical, sizeof canonical,
            "SALT_GEMMA4_CLEAR_FACTS_V2 request=%llu session_epoch=%llu "
            "turn_id=%llu history_turns=%llu position=%llu "
            "state_sha256=%s\n",
            *request_id, *session_epoch, *turn_id, *history_turns,
            raw_position, state_sha256);
    if (n < 0 || (size_t)n >= sizeof canonical || strcmp(line, canonical) ||
        raw_position > INT_MAX || !is_sha256_hex(state_sha256))
        return -1;
    *position = (int)raw_position;
    return 0;
}

static int emit_clear_result(const SaltGemma4Text *model,
                             unsigned long long request_id,
                             unsigned long long turn_id,
                             unsigned long long history_turns,
                             const unsigned char mindset_sha256[32]) {
    unsigned char state_sha256[32];
    unsigned char facts_sha256[32];
    int position = salt_gemma4_text_position(model);
    if (!mindset_sha256 || position < 0 ||
        fill_state_digests(model, state_sha256, facts_sha256) != 0 ||
        fprintf(stdout,
            "GEMMA4_SERVER_CLEAR_RESULT_V2 request=%llu session_epoch=%llu "
            "turn_id=%llu history_turns=%llu position=%d mindset_mode=%s "
            "mindset_end=%d facts_rows=%d journal_entries=0 state_sha256=",
            request_id, g_session_epoch, turn_id, history_turns, position,
            g_mindset_mode, g_mindset_end,
            position - g_mindset_end) < 0 ||
        print_hex(stdout, state_sha256) != 0 ||
        fputs(" mindset_sha256=", stdout) == EOF ||
        print_hex(stdout, mindset_sha256) != 0 ||
        fputs(" facts_sha256=", stdout) == EOF ||
        print_hex(stdout, facts_sha256) != 0 ||
        fprintf(stdout, " build_identity_sha256=%s runtime_ready="
                G4_RUNTIME_READY_TEXT "\n",
                g_build_identity_sha256) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int emit_ready(const SaltGemma4Text *model,
                      unsigned long long turn_id,
                      unsigned long long history_turns,
                      const unsigned char mindset_sha256[32]) {
    unsigned char digest[32];
    unsigned char facts_sha256[32];
    SaltGemma4MemoryStats memory_stats;
    int position = salt_gemma4_text_position(model);
    if (position < 0 || !mindset_sha256 ||
        salt_gemma4_text_memory_stats(model, &memory_stats) != 0 ||
        fill_state_digests(model, digest, facts_sha256) != 0)
        return -1;
    if (fprintf(stdout,
        "GEMMA4_SERVER_READY_V3 protocol=%d session_epoch=%llu "
        "turn_id=%llu history_turns=%llu model_context_limit=%d "
        "context=%d output_limit=%d response_limit_bytes=%llu "
        "journal_capacity=%d "
        "prefill_chunk_tokens=%d "
        "position=%d shared_position=%d kv_capacity_bytes=%llu "
        "kv_sliding_window=1024 kv_sliding_layers=25 kv_full_layers=5 "
        "expert_budget_bytes=%llu "
        "compatibility_sha256=%s build_identity_sha256=%s state_sha256=",
        G4_PROTOCOL, g_session_epoch, turn_id, history_turns,
        G4_MODEL_MAX_CONTEXT, g_context, g_output_limit,
        (unsigned long long)((size_t)g_output_limit *
            G4_RESPONSE_TOKEN_BYTES + G4_RESPONSE_OVERHEAD),
        G4_JOURNAL_CAPACITY,
        salt_gemma4_text_prefill_chunk_capacity(model),
        position, salt_gemma4_text_shared_position(model),
        (unsigned long long)salt_gemma4_text_kv_capacity_bytes(model),
        (unsigned long long)memory_stats.expert_budget_bytes,
        SALT_GEMMA4_KV_COMPAT_SHA256,
        g_build_identity_sha256) < 0 ||
        print_hex(stdout, digest) != 0 ||
        fprintf(stdout, " mindset_mode=%s mindset_end=%d mindset_sha256=",
                g_mindset_mode, g_mindset_end) < 0 ||
        print_hex(stdout, mindset_sha256) != 0 ||
        fputs(" facts_sha256=", stdout) == EOF ||
        print_hex(stdout, facts_sha256) != 0 ||
        fprintf(stdout, " facts_rows=%d runtime_ready="
                G4_RUNTIME_READY_TEXT "\n",
                position - g_mindset_end) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int fill_record(SaltGemma4Text *model, G4CommitRecord *record,
                       unsigned long long request_id,
                       unsigned long long turn_id,
                       unsigned long long history_turns,
                       const char *client_request_sha256,
                       const char *request_sha256,
                       const unsigned char mindset_sha256[32],
                       const int *prompt_ids, int prompt_tokens, int image_tokens,
                       const int *output_ids, int output_limit, int output_count,
                       int stop_token, int synthetic_close,
                       int position_before, int position_after,
                       const char *response, int response_bytes) {
    if (!model || !record || !client_request_sha256 || !request_sha256 ||
        !mindset_sha256 || !prompt_ids || prompt_tokens < 1 ||
        prompt_tokens > g_context ||
        !output_ids || !response || output_limit < 1 ||
        output_limit > g_output_limit || output_count < 1 ||
        output_count > output_limit ||
        output_count > g_output_limit || response_bytes < 0 ||
        (stop_token == -1 && output_count != output_limit) ||
        (stop_token != -1 && output_ids[output_count - 1] != stop_token) ||
        (size_t)prompt_tokens >
            g_journal_arena.prompt_capacity - g_journal_arena.prompt_used ||
        (size_t)output_count >
            g_journal_arena.output_capacity - g_journal_arena.output_used ||
        (size_t)response_bytes >
            g_journal_arena.response_capacity - g_journal_arena.response_used ||
        !is_sha256_hex(client_request_sha256) ||
        !is_sha256_hex(request_sha256))
        return -1;
    memset(record, 0, sizeof *record);
    record->original_request_id = request_id;
    record->turn_id = turn_id;
    record->history_turns = history_turns;
    memcpy(record->client_request_sha256, client_request_sha256,
           G4_SHA256_HEX + 1u);
    memcpy(record->request_sha256, request_sha256, G4_SHA256_HEX + 1u);
    record->prompt_tokens = prompt_tokens;
    record->image_tokens = image_tokens;
    record->output_limit = output_limit;
    record->output_steps = output_count;
    record->stop_token = stop_token;
    record->synthetic_close = synthetic_close;
    record->position_before = position_before;
    record->position_after = position_after;
    record->mindset_end = g_mindset_end;
    record->response_bytes = response_bytes;
    record->prompt_offset = g_journal_arena.prompt_used;
    record->output_offset = g_journal_arena.output_used;
    record->response_offset = g_journal_arena.response_used;
    memcpy(g_journal_arena.prompt_ids + record->prompt_offset, prompt_ids,
           (size_t)prompt_tokens * sizeof *prompt_ids);
    memcpy(g_journal_arena.output_ids + record->output_offset, output_ids,
           (size_t)output_count * sizeof *output_ids);
    memcpy(g_journal_arena.responses + record->response_offset,
           response, (size_t)response_bytes);
    if (fill_state_digests(
            model, record->state_sha256, record->facts_sha256) != 0)
        return -1;
    memcpy(record->mindset_sha256, mindset_sha256,
           sizeof record->mindset_sha256);
    g_journal_arena.prompt_used += (size_t)prompt_tokens;
    g_journal_arena.output_used += (size_t)output_count;
    g_journal_arena.response_used += (size_t)response_bytes;
    record->used = 1;
    return 0;
}

static int emit_record(unsigned long long request_id,
                       const G4CommitRecord *record, int replayed) {
    const char *marker = replayed
        ? "GEMMA4_SERVER_REPLAY_V3" : "GEMMA4_SERVER_COMMIT_V3";
    const char *finish_reason;
    if (!record || !record->used || (replayed != 0 && replayed != 1))
        return -1;
    finish_reason = record->stop_token == -1 ? "length" : "stop";
    if (fprintf(stdout,
        "%s request=%llu original_request=%llu session_epoch=%llu "
        "turn_id=%llu history_turns=%llu finish_reason=%s "
        "prompt_tokens=%d prompt_ids=",
        marker, request_id, record->original_request_id, g_session_epoch,
        record->turn_id, record->history_turns, finish_reason,
        record->prompt_tokens) < 0)
        return -1;
    for (int i = 0; i < record->prompt_tokens; i++)
        if (fprintf(stdout, "%s%d", i ? "," : "",
                    g_journal_arena.prompt_ids[record->prompt_offset +
                                               (size_t)i]) < 0)
            return -1;
    if (fprintf(stdout,
        " image_tokens=%d output_limit=%d output_steps=%d "
        "stop_token=%d synthetic_close=%d "
        "position_before=%d position_after=%d session_continuable=1 "
        "response_bytes=%d output_ids=",
        record->image_tokens, record->output_limit,
        record->output_steps, record->stop_token,
        record->synthetic_close, record->position_before,
        record->position_after, record->response_bytes) < 0)
        return -1;
    for (int i = 0; i < record->output_steps; i++)
        if (fprintf(stdout, "%s%d", i ? "," : "",
                    g_journal_arena.output_ids[record->output_offset +
                                               (size_t)i]) < 0)
            return -1;
    if (fputs(" state_sha256=", stdout) == EOF ||
        print_hex(stdout, record->state_sha256) != 0 ||
        fprintf(stdout, " mindset_mode=%s mindset_end=%d mindset_sha256=",
                g_mindset_mode, record->mindset_end) < 0 ||
        print_hex(stdout, record->mindset_sha256) != 0 ||
        fputs(" facts_sha256=", stdout) == EOF ||
        print_hex(stdout, record->facts_sha256) != 0 ||
        fprintf(stdout,
            " facts_rows=%d build_identity_sha256=%s "
            "client_request_sha256=%s request_sha256=%s "
            "replayed=%d "
            "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
            record->position_after - record->mindset_end,
            g_build_identity_sha256,
            record->client_request_sha256, record->request_sha256,
            replayed) < 0)
        return -1;
    return write_payload(g_journal_arena.responses + record->response_offset,
                         (size_t)record->response_bytes);
}

static int emit_bye(const SaltGemma4Text *model,
                    unsigned long long turn_id,
                    unsigned long long history_turns,
                    const unsigned char mindset_sha256[32]) {
    unsigned char digest[32];
    unsigned char facts_sha256[32];
    int position = salt_gemma4_text_position(model);
    if (!mindset_sha256 || position < 0 ||
        fill_state_digests(model, digest, facts_sha256) != 0 ||
        fprintf(stdout,
            "GEMMA4_SERVER_BYE_V2 session_epoch=%llu turn_id=%llu "
            "history_turns=%llu position=%d state_sha256=",
            g_session_epoch, turn_id, history_turns, position) < 0 ||
        print_hex(stdout, digest) != 0 ||
        fprintf(stdout, " mindset_mode=%s mindset_end=%d mindset_sha256=",
                g_mindset_mode, g_mindset_end) < 0 ||
        print_hex(stdout, mindset_sha256) != 0 ||
        fputs(" facts_sha256=", stdout) == EOF ||
        print_hex(stdout, facts_sha256) != 0 ||
        fprintf(stdout, " facts_rows=%d build_identity_sha256=%s "
                        "runtime_ready=" G4_RUNTIME_READY_TEXT "\n",
                position - g_mindset_end, g_build_identity_sha256) < 0)
        return -1;
    return fflush(stdout) == 0 ? 0 : -1;
}

static int hot_route_candidate_update(
        const SaltGemma4RouteObserver *observer,
        SaltDprExpertCoverageLedger *ledger,
        SaltDprAttentionPlan *candidate) {
    int updated;
    if (!observer || !ledger || !candidate) return -1;
    updated = salt_gemma4_text_route_observer_update_coverage_ledger(
        observer, SALT_GEMMA4_ROUTE_DECODE, ledger);
    if (updated != 0) return updated;
    return salt_dpr_expert_coverage_ledger_compile(
        ledger, G4_ROUTE_TOPK, SALT_DPR_MAX_RELEVANCE_ITEMS, candidate);
}

static int serve_loop(SaltGemma4Text *text_model,
                      SaltGemma4Vision *vision_model) {
    char header[640];
    char *prompt = NULL, *image_path = NULL;
    int *prompt_ids = NULL, *output_ids = NULL, *positions = NULL;
    unsigned char *mm_types = NULL, *padding = NULL;
    unsigned char mindset_sha256[32];
    unsigned char build_identity_sha256[32];
    float *logits = NULL, *patches = NULL, *features = NULL;
    SaltDprStoredEdge *dpr_edge_slots = NULL;
    SaltDprEdgeStats *dpr_stats_slots = NULL;
    SaltDprRetentionItem *dpr_retention_items = NULL;
    uint8_t *dpr_retained = NULL;
    size_t *dpr_retention_order = NULL;
    uint32_t *dpr_buckets = NULL;
    SaltDprStoredAttentionPlan *dpr_attention_slots = NULL;
    uint32_t *dpr_attention_buckets = NULL;
    SaltGemma4RouteObservation *hot_route_items = NULL;
    uint64_t *hot_route_counts = NULL;
    SaltGemma4RouteObserver hot_route_observer = {0};
    SaltDprExpertCoverageLedger hot_route_ledger = {0};
    SaltDprAttentionPlan hot_route_candidate;
    int hot_route_observer_active = 0;
    SaltDprMentorFamily *dpr_mentor_family_slots = NULL;
    uint32_t *dpr_mentor_family_next = NULL;
    uint32_t *dpr_mentor_family_buckets = NULL;
    SaltDprMentorBinding *dpr_mentor_binding_slots = NULL;
    uint32_t *dpr_mentor_binding_next = NULL;
    uint32_t *dpr_mentor_binding_buckets = NULL;
    SaltDprSelectorCandidate *dpr_mentor_candidates = NULL;
    SaltDprChart *dpr_mentor_effective_charts = NULL;
    SaltDprStore dpr_store;
    SaltDprAttentionPlanStore dpr_attention_store;
    SaltDprMentorFamilyStore dpr_mentor_family_store;
    SaltDprMentorBindingStore dpr_mentor_binding_store;
    SaltDprRuntime dpr_runtime;
    char *response = NULL;
    G4CommitRecord *journal = NULL;
    int journal_count = 0;
    unsigned long long expected_request = 1;
    unsigned long long turn_id = 0;
    unsigned long long history_turns = 0;
    SaltStateView ready_view;
    int rc = 1;
    const int response_capacity =
        g_output_limit * (int)G4_RESPONSE_TOKEN_BYTES +
        (int)G4_RESPONSE_OVERHEAD;
    const size_t journal_response_capacity =
        (size_t)g_context * G4_RESPONSE_TOKEN_BYTES +
        (size_t)G4_JOURNAL_CAPACITY * G4_RESPONSE_OVERHEAD;

    if (salt_gemma4_text_build_identity_sha256(
            text_model, build_identity_sha256) != 0)
        goto done;
    digest_hex(build_identity_sha256, g_build_identity_sha256);
    prompt = (char *)malloc(G4_MAX_PROMPT_BYTES + 1u);
    image_path = (char *)malloc(G4_MAX_IMAGE_PATH + 1u);
    prompt_ids = (int *)malloc((size_t)g_context * sizeof *prompt_ids);
    output_ids = (int *)malloc((size_t)g_output_limit * sizeof *output_ids);
    mm_types = (unsigned char *)calloc((size_t)g_context, 1);
    logits = (float *)malloc(
        (size_t)salt_gemma4_text_vocab_size(text_model) * sizeof *logits);
    response = (char *)malloc((size_t)response_capacity);
    journal = (G4CommitRecord *)calloc(
        G4_JOURNAL_CAPACITY, sizeof *journal);
    g_journal_arena.prompt_ids = (int *)malloc(
        (size_t)g_context * sizeof *g_journal_arena.prompt_ids);
    g_journal_arena.output_ids = (int *)malloc(
        ((size_t)g_context + G4_JOURNAL_CAPACITY) *
        sizeof *g_journal_arena.output_ids);
    g_journal_arena.responses = (char *)malloc(journal_response_capacity);
    g_journal_arena.prompt_capacity = (size_t)g_context;
    g_journal_arena.output_capacity =
        (size_t)g_context + G4_JOURNAL_CAPACITY;
    g_journal_arena.response_capacity = journal_response_capacity;
    g_journal_arena.prompt_used = 0;
    g_journal_arena.output_used = 0;
    g_journal_arena.response_used = 0;
    if (!prompt || !image_path || !prompt_ids || !output_ids || !mm_types ||
        !logits || !response || !journal || !g_journal_arena.prompt_ids ||
        !g_journal_arena.output_ids || !g_journal_arena.responses) {
        fputs("gemma4 persistent server: allocation failed\n", stderr);
        goto done;
    }
    memset(&dpr_store, 0, sizeof dpr_store);
    memset(&dpr_attention_store, 0, sizeof dpr_attention_store);
    memset(&hot_route_candidate, 0, sizeof hot_route_candidate);
    g_hot_route_candidate = &hot_route_candidate;
    memset(g_hot_route_provenance_sha256, 0,
           sizeof g_hot_route_provenance_sha256);
    memset(g_hot_route_mindset_sha256, 0,
           sizeof g_hot_route_mindset_sha256);
    g_hot_route_ready = 0;
    memset(&dpr_mentor_family_store, 0, sizeof dpr_mentor_family_store);
    memset(&dpr_mentor_binding_store, 0, sizeof dpr_mentor_binding_store);
    memset(&dpr_runtime, 0, sizeof dpr_runtime);
    if (g_dpr_mode != SALT_DPR_OFF) {
        dpr_edge_slots = (SaltDprStoredEdge *)calloc(
            G4_DPR_EDGE_CAPACITY, sizeof *dpr_edge_slots);
        dpr_stats_slots = (SaltDprEdgeStats *)calloc(
            G4_DPR_EDGE_CAPACITY, sizeof *dpr_stats_slots);
        dpr_retention_items = (SaltDprRetentionItem *)calloc(
            G4_DPR_EDGE_CAPACITY, sizeof *dpr_retention_items);
        dpr_retained = (uint8_t *)calloc(
            G4_DPR_EDGE_CAPACITY, sizeof *dpr_retained);
        dpr_retention_order = (size_t *)malloc(
            G4_DPR_EDGE_CAPACITY * sizeof *dpr_retention_order);
        dpr_buckets = (uint32_t *)malloc(
            G4_DPR_BUCKET_COUNT * sizeof *dpr_buckets);
        if (!dpr_edge_slots || !dpr_stats_slots ||
            !dpr_retention_items || !dpr_retained ||
            !dpr_retention_order || !dpr_buckets ||
            salt_dpr_store_init(
                &dpr_store, dpr_edge_slots, G4_DPR_EDGE_CAPACITY,
                dpr_buckets, G4_DPR_BUCKET_COUNT) != 0 ||
            salt_dpr_store_load(&dpr_store, g_dpr_root) != 0 ||
            salt_dpr_runtime_init(&dpr_runtime, g_dpr_mode) != 0) {
            fputs("gemma4 persistent server: DPR startup allocation failed\n",
                  stderr);
            goto done;
        }
        if (g_dpr_attention_enabled) {
            dpr_attention_slots = (SaltDprStoredAttentionPlan *)calloc(
                G4_DPR_ATTENTION_RUNTIME_CAPACITY,
                sizeof *dpr_attention_slots);
            dpr_attention_buckets = (uint32_t *)malloc(
                G4_DPR_ATTENTION_PLAN_BUCKETS *
                sizeof *dpr_attention_buckets);
            if (!dpr_attention_slots || !dpr_attention_buckets ||
                salt_dpr_attention_plan_store_init(
                    &dpr_attention_store, dpr_attention_slots,
                    G4_DPR_ATTENTION_RUNTIME_CAPACITY,
                    dpr_attention_buckets,
                    G4_DPR_ATTENTION_PLAN_BUCKETS) != 0 ||
                salt_dpr_attention_plan_store_load(
                    &dpr_attention_store, g_dpr_root) != 0 ||
                dpr_attention_store.plan_count >
                    G4_DPR_ATTENTION_PLAN_CAPACITY) {
                fputs("gemma4 persistent server: NM startup failed\n", stderr);
                goto done;
            }
            hot_route_items = (SaltGemma4RouteObservation *)calloc(
                G4_ROUTE_OBSERVATION_CAPACITY, sizeof *hot_route_items);
            hot_route_counts = (uint64_t *)calloc(
                G4_ROUTE_LAYERS * G4_ROUTE_EXPERTS,
                sizeof *hot_route_counts);
            if (!hot_route_items || !hot_route_counts ||
                salt_dpr_expert_coverage_ledger_init(
                    &hot_route_ledger, hot_route_counts,
                    G4_ROUTE_LAYERS, G4_ROUTE_EXPERTS) != 0) {
                fputs("gemma4 persistent server: hot-route allocation failed\n",
                      stderr);
                goto done;
            }
            hot_route_observer.items = hot_route_items;
            hot_route_observer.capacity = G4_ROUTE_OBSERVATION_CAPACITY;
            hot_route_observer.route_kind = SALT_GEMMA4_ROUTE_DECODE;
            fprintf(stderr,
                    "[dpr-attention] plans=%zu rejected=%zu\n",
                    dpr_attention_store.plan_count,
                    dpr_attention_store.rejected_plans);
        }
        if (g_dpr_mentor_enabled) {
            dpr_mentor_family_slots = (SaltDprMentorFamily *)calloc(
                G4_DPR_MENTOR_FAMILY_CAPACITY,
                sizeof *dpr_mentor_family_slots);
            dpr_mentor_family_next = (uint32_t *)malloc(
                G4_DPR_MENTOR_FAMILY_CAPACITY *
                sizeof *dpr_mentor_family_next);
            dpr_mentor_family_buckets = (uint32_t *)malloc(
                G4_DPR_MENTOR_FAMILY_BUCKETS *
                sizeof *dpr_mentor_family_buckets);
            dpr_mentor_binding_slots = (SaltDprMentorBinding *)calloc(
                G4_DPR_MENTOR_BINDING_CAPACITY,
                sizeof *dpr_mentor_binding_slots);
            dpr_mentor_binding_next = (uint32_t *)malloc(
                G4_DPR_MENTOR_BINDING_CAPACITY *
                sizeof *dpr_mentor_binding_next);
            dpr_mentor_binding_buckets = (uint32_t *)malloc(
                G4_DPR_MENTOR_BINDING_BUCKETS *
                sizeof *dpr_mentor_binding_buckets);
            dpr_mentor_candidates = (SaltDprSelectorCandidate *)calloc(
                G4_DPR_MENTOR_BINDING_CAPACITY,
                sizeof *dpr_mentor_candidates);
            dpr_mentor_effective_charts = (SaltDprChart *)calloc(
                G4_DPR_EDGE_CAPACITY, sizeof *dpr_mentor_effective_charts);
            if (!dpr_mentor_family_slots || !dpr_mentor_family_next ||
                !dpr_mentor_family_buckets || !dpr_mentor_binding_slots ||
                !dpr_mentor_binding_next || !dpr_mentor_binding_buckets ||
                !dpr_mentor_candidates || !dpr_mentor_effective_charts ||
                salt_dpr_mentor_family_store_init(
                    &dpr_mentor_family_store,
                    dpr_mentor_family_slots, dpr_mentor_family_next,
                    G4_DPR_MENTOR_FAMILY_CAPACITY,
                    dpr_mentor_family_buckets,
                    G4_DPR_MENTOR_FAMILY_BUCKETS) != 0 ||
                salt_dpr_mentor_family_store_load(
                    &dpr_mentor_family_store, g_dpr_root) != 0 ||
                salt_dpr_mentor_binding_store_init(
                    &dpr_mentor_binding_store,
                    dpr_mentor_binding_slots, dpr_mentor_binding_next,
                    G4_DPR_MENTOR_BINDING_CAPACITY,
                    dpr_mentor_binding_buckets,
                    G4_DPR_MENTOR_BINDING_BUCKETS) != 0 ||
                salt_dpr_mentor_binding_store_load(
                    &dpr_mentor_binding_store, g_dpr_root,
                    &dpr_mentor_family_store, &dpr_store) != 0) {
                fputs("gemma4 persistent server: Mentor startup failed\n",
                      stderr);
                goto done;
            }
            fprintf(stderr,
                    "[dpr-mentor] families=%zu rejected_families=%zu "
                    "bindings=%zu rejected_bindings=%zu\n",
                    dpr_mentor_family_store.family_count,
                    dpr_mentor_family_store.rejected_families,
                    dpr_mentor_binding_store.binding_count,
                    dpr_mentor_binding_store.rejected_bindings);
        }
        if (salt_dpr_stats_prepare(g_dpr_root) != 0) goto done;
        {
            size_t stats_rejected = 0, retained_count = 0;
            for (size_t edge = 0; edge < dpr_store.edge_count; edge++) {
                int loaded = salt_dpr_stats_load(
                    g_dpr_root, dpr_store.edges[edge].edge_sha256,
                    &dpr_stats_slots[edge]);
                if (loaded != 0 ||
                    dpr_stats_slots[edge].mode !=
                        dpr_store.edges[edge].edge.mode ||
                    dpr_stats_slots[edge].horizon !=
                        dpr_store.edges[edge].edge.horizon) {
                    if (loaded < 0) stats_rejected++;
                    if (salt_dpr_stats_init(
                            &dpr_stats_slots[edge],
                            dpr_store.edges[edge].edge_sha256,
                            dpr_store.edges[edge].edge.mode,
                            dpr_store.edges[edge].edge.horizon) != 0)
                        goto done;
                }
                dpr_retention_items[edge].edge_index = edge;
                memcpy(dpr_retention_items[edge].edge_sha256,
                       dpr_store.edges[edge].edge_sha256, 32);
                dpr_retention_items[edge].reference_bytes =
                    dpr_store.edges[edge].edge.kind ==
                        SALT_DPR_EDGE_NOMOGRAM_DRAFT ||
                    dpr_store.edges[edge].edge.kind ==
                        SALT_DPR_EDGE_NOMOGRAM_PREFILL
                    ? SALT_DPR_EDGE_FILE_BYTES
                    : dpr_store.edges[edge].edge.reference_bytes;
                dpr_retention_items[edge].static_chart =
                    dpr_store.edges[edge].edge.chart;
                dpr_retention_items[edge].observed = &dpr_stats_slots[edge];
            }
            if (salt_dpr_retention_plan(
                    dpr_retention_items, dpr_store.edge_count,
                    g_dpr_retention_bytes, g_dpr_serial_ns,
                    dpr_retained, dpr_retention_order,
                    G4_DPR_EDGE_CAPACITY) != 0)
                goto done;
            for (size_t edge = 0; edge < dpr_store.edge_count; edge++)
                if (dpr_retained[edge]) retained_count++;
            fprintf(stderr,
                    "[dpr-stats] edges=%zu retained=%zu rejected=%zu "
                    "retention_bytes=%llu\n",
                    dpr_store.edge_count, retained_count, stats_rejected,
                    (unsigned long long)g_dpr_retention_bytes);
        }
    }
    if (g_dpr_mode != SALT_DPR_OFF &&
        (salt_gemma4_text_position(text_model) != 0 ||
         dpr_transition_reset(0) != 0))
        goto done;
    memset(mindset_sha256, 0, sizeof mindset_sha256);
    if (salt_gemma4_text_position(text_model) < g_mindset_end ||
        (g_proof_state && salt_gemma4_text_prefix_sha256(
            text_model, g_mindset_end, mindset_sha256) != 0) ||
        fill_state_view(text_model, turn_id, history_turns,
                        mindset_sha256, &ready_view) != 0 ||
        emit_ready(text_model, turn_id, history_turns, mindset_sha256) != 0)
        goto done;

    while (fgets(header, sizeof header, stdin)) {
        SaltMemSnapshot memory_snapshot;
        unsigned long long request_id = 0, expected_epoch = 0;
        unsigned long long expected_turn = 0;
        unsigned long long expected_history_turns = 0;
        char client_request_sha256[G4_SHA256_HEX + 1] = {0};
        char request_sha256[G4_SHA256_HEX + 1] = {0};
        size_t prompt_bytes = 0, image_path_bytes = 0;
        int expected_position = -1;
        int prompt_count, position_before, output_count = 0, stop_token = -1;
        int decode_count, response_bytes, image_tokens = 0;
        int position_after, synthetic_close = 0;
        int request_output_limit = g_output_limit;
        int request_proof_state = 0;
        int started = 0;
        int dpr_prefill_hits = 0, dpr_prefill_cached = 0;
        uint8_t dpr_prefill_source[32] = {0};
        uint8_t dpr_qa_key[32] = {0};
        memset(&g_dpr_waterfall, 0, sizeof g_dpr_waterfall);
        g_proof_state = g_process_proof_state;

        if (!strcmp(header, "SALT_GEMMA4_DRAIN_V2\n")) {
            if (emit_bye(text_model, turn_id, history_turns,
                         mindset_sha256) != 0)
                goto done;
            rc = 0;
            goto done;
        }
        if (!strncmp(header, "SALT_GEMMA4_EXPORT_KV_V2 ",
                     sizeof "SALT_GEMMA4_EXPORT_KV_V2 " - 1u)) {
            unsigned long long export_request = 0, raw_path = 0;
            char canonical[192], extra;
            char export_error[256] = {0};
            int canonical_bytes, saved_position = 0;
            unsigned char state_digest[32], facts_digest[32];
            char state_hex[65], facts_hex[65];
            if (sscanf(header,
                    "SALT_GEMMA4_EXPORT_KV_V2 request=%llu path_bytes=%llu %c",
                    &export_request, &raw_path, &extra) != 2 ||
                export_request != expected_request || raw_path < 1 ||
                raw_path > G4_MAX_IMAGE_PATH) {
                fputs("gemma4 persistent server: malformed EXPORT_KV_V2\n",
                      stderr);
                goto done;
            }
            canonical_bytes = snprintf(canonical, sizeof canonical,
                "SALT_GEMMA4_EXPORT_KV_V2 request=%llu path_bytes=%llu\n",
                export_request, raw_path);
            if (canonical_bytes < 0 ||
                (size_t)canonical_bytes >= sizeof canonical ||
                strcmp(header, canonical) ||
                read_exact(stdin, image_path, (size_t)raw_path) != 0 ||
                fgetc(stdin) != '\n')
                goto done;
            image_path[raw_path] = 0;
            if (memchr(image_path, 0, (size_t)raw_path) ||
                g4_kv_file_save(text_model, image_path, &saved_position,
                                export_error, sizeof export_error) != 0 ||
                saved_position != salt_gemma4_text_position(text_model) ||
                salt_gemma4_text_state_sha256(text_model, state_digest) != 0 ||
                salt_gemma4_text_facts_sha256(
                    text_model, g_mindset_end, facts_digest) != 0)
                goto done;
            digest_hex(state_digest, state_hex);
            digest_hex(facts_digest, facts_hex);
            if (fprintf(stdout,
                    "GEMMA4_SERVER_EXPORT_RESULT_V2 request=%llu position=%d "
                    "state_sha256=%s facts_sha256=%s "
                    "build_identity_sha256=%s runtime_ready="
                    G4_RUNTIME_READY_TEXT "\n",
                    export_request, saved_position, state_hex, facts_hex,
                    g_build_identity_sha256) < 0)
                goto done;
            expected_request++;
            continue;
        }
        if (!strncmp(header, "SALT_GEMMA4_IMPORT_SESSION_V1 ",
                     sizeof "SALT_GEMMA4_IMPORT_SESSION_V1 " - 1u)) {
            unsigned long long import_request = 0, import_epoch = 0;
            unsigned long long import_turn = 0, import_history_turns = 0;
            unsigned long long raw_path = 0;
            int import_position = -1, import_mindset_end = -1;
            int canonical_bytes, loaded_position = 0;
            char canonical[320], extra;
            char import_error[256] = {0};
            unsigned char before_prefix[32], after_prefix[32];
            unsigned char state_digest[32], facts_digest[32];
            char state_hex[65], facts_hex[65], mindset_hex[65];
            if (sscanf(header,
                    "SALT_GEMMA4_IMPORT_SESSION_V1 request=%llu "
                    "expected_epoch=%llu expected_position=%d turn=%llu "
                    "history_turns=%llu mindset_end=%d path_bytes=%llu %c",
                    &import_request, &import_epoch, &import_position,
                    &import_turn, &import_history_turns,
                    &import_mindset_end, &raw_path, &extra) != 7 ||
                import_request != expected_request || raw_path < 1 ||
                raw_path > G4_MAX_IMAGE_PATH ||
                import_history_turns > import_turn) {
                fputs("gemma4 persistent server: malformed IMPORT_SESSION_V1\n",
                      stderr);
                goto done;
            }
            canonical_bytes = snprintf(canonical, sizeof canonical,
                "SALT_GEMMA4_IMPORT_SESSION_V1 request=%llu "
                "expected_epoch=%llu expected_position=%d turn=%llu "
                "history_turns=%llu mindset_end=%d path_bytes=%llu\n",
                import_request, import_epoch, import_position,
                import_turn, import_history_turns,
                import_mindset_end, raw_path);
            if (canonical_bytes < 0 ||
                (size_t)canonical_bytes >= sizeof canonical ||
                strcmp(header, canonical) ||
                read_exact(stdin, image_path, (size_t)raw_path) != 0 ||
                fgetc(stdin) != '\n')
                goto done;
            image_path[raw_path] = 0;
            expected_request++;
            if (memchr(image_path, 0, (size_t)raw_path) ||
                import_epoch != g_session_epoch ||
                import_position != salt_gemma4_text_position(text_model) ||
                import_position != g_mindset_end ||
                import_mindset_end != g_mindset_end) {
                if (emit_reject(import_request, "state_conflict",
                        "session import epoch/position/mindset mismatch") != 0)
                    goto done;
                continue;
            }
            if (salt_gemma4_text_prefix_sha256(
                    text_model, g_mindset_end, before_prefix) != 0 ||
                g4_kv_file_load(text_model, image_path, g_context,
                                &loaded_position,
                                import_error, sizeof import_error) != 0 ||
                loaded_position < g_mindset_end ||
                salt_gemma4_text_prefix_sha256(
                    text_model, g_mindset_end, after_prefix) != 0 ||
                memcmp(before_prefix, after_prefix, sizeof before_prefix) != 0 ||
                (g_dpr_mode != SALT_DPR_OFF &&
                    dpr_transition_reset((uint64_t)loaded_position) != 0) ||
                salt_gemma4_text_state_sha256(
                    text_model, state_digest) != 0 ||
                salt_gemma4_text_facts_sha256(
                    text_model, g_mindset_end, facts_digest) != 0) {
                fputs("gemma4 persistent server: session import failed\n", stderr);
                goto done;
            }
            memset(journal, 0,
                   (size_t)G4_JOURNAL_CAPACITY * sizeof *journal);
            g_journal_arena.prompt_used = 0;
            g_journal_arena.output_used = 0;
            g_journal_arena.response_used = 0;
            journal_count = 0;
            turn_id = import_turn;
            history_turns = import_history_turns;
            if (g_dpr_attention_enabled &&
                (salt_dpr_expert_coverage_ledger_reset(
                    &hot_route_ledger) != 0 ||
                 salt_dpr_attention_plan_store_clear_runtime(
                    &dpr_attention_store) != 0)) {
                fputs("gemma4 persistent server: session import DPR reset failed\n",
                      stderr);
                goto done;
            }
            g_hot_route_ready = 0;
            digest_hex(state_digest, state_hex);
            digest_hex(facts_digest, facts_hex);
            digest_hex(mindset_sha256, mindset_hex);
            if (fprintf(stdout,
                    "GEMMA4_SERVER_IMPORT_RESULT_V1 request=%llu "
                    "session_epoch=%llu turn_id=%llu history_turns=%llu "
                    "position=%d mindset_end=%d state_sha256=%s "
                    "mindset_sha256=%s facts_sha256=%s facts_rows=%d "
                    "build_identity_sha256=%s runtime_ready="
                    G4_RUNTIME_READY_TEXT "\n",
                    import_request, g_session_epoch, turn_id, history_turns,
                    loaded_position, g_mindset_end, state_hex, mindset_hex,
                    facts_hex, loaded_position - g_mindset_end,
                    g_build_identity_sha256) < 0)
                goto done;
            continue;
        }
        if (!strncmp(header, "SALT_GEMMA4_CLEAR_FACTS_V2 ",
                     sizeof "SALT_GEMMA4_CLEAR_FACTS_V2 " - 1u)) {
            unsigned long long clear_request = 0, clear_epoch = 0;
            unsigned long long clear_turn = 0;
            unsigned long long clear_history_turns = 0;
            int clear_position = -1;
            char expected_state_sha256[G4_SHA256_HEX + 1] = {0};
            char actual_state_sha256[G4_SHA256_HEX + 1];
            unsigned char state_digest[32], facts_digest[32], prefix_digest[32];
            SaltStateView before_view, after_view;
            if (parse_clear_header(
                    header, &clear_request, &clear_epoch, &clear_turn,
                    &clear_history_turns,
                    &clear_position, expected_state_sha256) != 0 ||
                clear_request != expected_request) {
                fputs("gemma4 persistent server: malformed CLEAR_FACTS_V2 header\n",
                      stderr);
                goto done;
            }
            expected_request++;
            if (fill_state_digests_required(
                    text_model, state_digest, facts_digest) != 0)
                goto done;
            digest_hex(state_digest, actual_state_sha256);
            if (clear_epoch != g_session_epoch || clear_turn != turn_id ||
                clear_history_turns != history_turns ||
                clear_position != salt_gemma4_text_position(text_model) ||
                (!is_sha256_zero_hex(expected_state_sha256) &&
                 strcmp(expected_state_sha256, actual_state_sha256))) {
                if (emit_reject(clear_request, "state_conflict",
                        "clear-facts epoch/turn/position/state mismatch") != 0)
                    goto done;
                continue;
            }
            if (salt_gemma4_text_prefix_sha256(
                    text_model, g_mindset_end, prefix_digest) != 0 ||
                (g_proof_state &&
                 memcmp(prefix_digest, mindset_sha256, 32u) != 0) ||
                fill_state_view(text_model, turn_id, history_turns,
                                mindset_sha256, &before_view) != 0) {
                emit_fatal(clear_request, "mindset_drift",
                           "mindset prefix changed before factual clear");
                goto done;
            }
            if (salt_gemma4_text_clear_facts(
                    text_model, g_mindset_end) != 0 ||
                (g_dpr_mode != SALT_DPR_OFF &&
                    (dpr_transition_reset((uint64_t)g_mindset_end) != 0 ||
                     g_dpr_transition_position !=
                        (uint64_t)salt_gemma4_text_position(text_model))) ||
                salt_gemma4_text_prefix_sha256(
                    text_model, g_mindset_end, facts_digest) != 0 ||
                memcmp(facts_digest, prefix_digest, 32u) != 0 ||
                (g_proof_state &&
                 memcmp(facts_digest, mindset_sha256, 32u) != 0) ||
                salt_gemma4_text_state_sha256(
                    text_model, state_digest) != 0 ||
                memcmp(state_digest, prefix_digest, 32u) != 0 ||
                fill_state_view(text_model, turn_id, 0,
                                mindset_sha256, &after_view) != 0 ||
                salt_state_clear_transition_validate(
                    g_state_model, &before_view, &after_view) != 0) {
                emit_fatal(clear_request, "clear_facts",
                           "factual clear failed mandatory-state verification");
                goto done;
            }
            memset(journal, 0,
                   (size_t)G4_JOURNAL_CAPACITY * sizeof *journal);
            g_journal_arena.prompt_used = 0;
            g_journal_arena.output_used = 0;
            g_journal_arena.response_used = 0;
            journal_count = 0;
            history_turns = 0;
            if (g_dpr_attention_enabled &&
                salt_dpr_expert_coverage_ledger_reset(
                    &hot_route_ledger) != 0) {
                emit_fatal(clear_request, "hot_route",
                           "hot expert history reset failed");
                goto done;
            }
            if (g_dpr_attention_enabled &&
                salt_dpr_attention_plan_store_clear_runtime(
                    &dpr_attention_store) != 0) {
                emit_fatal(clear_request, "hot_route",
                           "hot expert runtime plan reset failed");
                goto done;
            }
            g_hot_route_ready = 0;
            if (emit_clear_result(
                    text_model, clear_request, turn_id, history_turns,
                    mindset_sha256) != 0)
                goto done;
            continue;
        }
        if (!strncmp(header, "SALT_GEMMA4_REPLAY_V2 ",
                     sizeof "SALT_GEMMA4_REPLAY_V2 " - 1u)) {
            G4CommitRecord *record;
            if (parse_replay_header(
                    header, &request_id, &expected_epoch,
                    client_request_sha256, request_sha256) != 0 ||
                request_id != expected_request) {
                fputs("gemma4 persistent server: malformed REPLAY_V2 header\n",
                      stderr);
                goto done;
            }
            expected_request++;
            if (expected_epoch != g_session_epoch) {
                if (emit_reject(request_id, "epoch_conflict",
                        "persistent session epoch mismatch") != 0)
                    goto done;
                continue;
            }
            record = find_record(
                journal, journal_count, client_request_sha256);
            if (!record) {
                if (emit_reject(request_id, "idempotency_unknown",
                        "committed request is not in the native journal") != 0)
                    goto done;
                continue;
            }
            if (strcmp(record->request_sha256, request_sha256)) {
                if (emit_reject(request_id, "idempotency_conflict",
                        "idempotency key payload mismatch") != 0)
                    goto done;
                continue;
            }
            if (emit_record(request_id, record, 1) != 0) goto done;
            continue;
        }
        if (parse_turn_header(header, &request_id, &expected_epoch,
                &expected_turn, &expected_history_turns, &expected_position,
                client_request_sha256, request_sha256,
                &prompt_bytes, &image_path_bytes, &request_output_limit,
                &request_proof_state) != 0 ||
            request_id != expected_request || prompt_bytes < 1) {
            fputs("gemma4 persistent server: malformed TURN_V3 header\n", stderr);
            goto done;
        }
        g_proof_state = g_process_proof_state || request_proof_state;
        g_active_output_limit = request_output_limit;
        if (read_exact(stdin, prompt, prompt_bytes) != 0 ||
            read_exact(stdin, image_path, image_path_bytes) != 0 ||
            fgetc(stdin) != '\n') {
            fputs("gemma4 persistent server: truncated TURN_V2 frame\n", stderr);
            goto done;
        }
        prompt[prompt_bytes] = '\0';
        image_path[image_path_bytes] = '\0';
        expected_request++;
        {
            G4CommitRecord *record = find_record(
                journal, journal_count, client_request_sha256);
            if (record) {
                if (strcmp(record->request_sha256, request_sha256)) {
                    if (emit_reject(request_id, "idempotency_conflict",
                            "idempotency key payload mismatch") != 0)
                        goto done;
                } else if (emit_record(request_id, record, 1) != 0) {
                    goto done;
                }
                continue;
            }
        }
        if (journal_count >= G4_JOURNAL_CAPACITY) {
            if (emit_reject(request_id, "journal_full",
                    "native result journal is full") != 0)
                goto done;
            continue;
        }
        position_before = salt_gemma4_text_position(text_model);
        if (expected_epoch != g_session_epoch || expected_turn != turn_id ||
            expected_history_turns != history_turns ||
            expected_position != position_before) {
            if (emit_reject(request_id, "state_conflict",
                    "persistent session epoch/turn/position mismatch") != 0)
                goto done;
            continue;
        }
        if (image_path_bytes && position_before != 0) {
            if (emit_reject(request_id, "image_position",
                    "images are supported only on the first persistent turn") != 0)
                goto done;
            continue;
        }
        prompt_count = salt_gemma4_text_encode(
            text_model, prompt, prompt_ids, g_context);
        if (prompt_count < 1 ||
            position_before + prompt_count + request_output_limit + 1 >
                g_context) {
            if (emit_reject(request_id, "context_limit",
                    "persistent session configured context/output limit exceeded; "
                    "factual clear or reset required") != 0)
                goto done;
            continue;
        }
        if (image_path_bytes && prompt_count >
                salt_gemma4_text_prefill_chunk_capacity(text_model)) {
            if (emit_reject(request_id, "image_context_limit",
                    "image prompt exceeds the configured image prefill bound") != 0)
                goto done;
            continue;
        }
        if (g_dpr_mode != SALT_DPR_OFF &&
            build_dpr_qa_key(prompt_ids, prompt_count, dpr_qa_key) != 0) {
            emit_fatal(request_id, "dpr_qa_key",
                       "native DPR Q&A key generation failed");
            goto done;
        }

        if (image_path_bytes) {
            int n_patches = 0, expected_features = 0, width = 0, height = 0;
            int actual_features, placeholder_count = 0;
            char error[256] = {0};
            memset(mm_types, 0, (size_t)prompt_count);
            for (int i = 0; i < prompt_count; i++) {
                if (prompt_ids[i] == G4_IMAGE_TOKEN) {
                    mm_types[i] = 1;
                    placeholder_count++;
                }
            }
            if (salt_gemma4_ppm_to_patches(
                    image_path, G4_MAX_SOFT_TOKENS,
                    &patches, &positions, &padding, &n_patches,
                    &expected_features, &width, &height,
                    error, sizeof error) != 0) {
                salt_gemma4_free_patches(patches, positions, padding);
                patches = NULL; positions = NULL; padding = NULL;
                if (emit_reject(request_id, "invalid_image",
                        error[0] ? error : "image processing failed") != 0)
                    goto done;
                continue;
            }
            features = (float *)calloc(
                (size_t)expected_features * G4_HIDDEN, sizeof *features);
            if (!features) {
                if (emit_reject(request_id, "allocation",
                        "image feature allocation failed") != 0)
                    goto done;
                salt_gemma4_free_patches(patches, positions, padding);
                patches = NULL; positions = NULL; padding = NULL;
                continue;
            }
            actual_features = salt_gemma4_vision_forward(
                vision_model, patches, positions, padding, n_patches,
                features, expected_features, error, sizeof error);
            if (actual_features != expected_features ||
                actual_features != placeholder_count) {
                free(features); features = NULL;
                salt_gemma4_free_patches(patches, positions, padding);
                patches = NULL; positions = NULL; padding = NULL;
                if (emit_reject(request_id, "image_features",
                        "image feature/placeholder mismatch") != 0)
                    goto done;
                continue;
            }
            image_tokens = actual_features;
            salt_gemma4_free_patches(patches, positions, padding);
            patches = NULL; positions = NULL; padding = NULL;
            if (emit_start(request_id, turn_id, position_before,
                           prompt_count, image_tokens,
                           request_output_limit) != 0)
                goto done;
            started = 1;
            if (g_dpr_attention_enabled &&
                salt_gemma4_text_route_observer_begin(
                    text_model, &hot_route_observer) != 0) {
                emit_fatal(request_id, "hot_route",
                           "route observer admission failed");
                goto done;
            }
            hot_route_observer_active = g_dpr_attention_enabled;
            if (salt_gemma4_text_prefill_image(
                    text_model, prompt_ids, mm_types, prompt_count,
                    features, actual_features, logits) != 0) {
                emit_fatal(request_id, "image_prefill",
                           "image text-state prefill failed");
                goto done;
            }
            free(features); features = NULL;
        } else {
            if (emit_start(request_id, turn_id, position_before,
                           prompt_count, 0, request_output_limit) != 0)
                goto done;
            started = 1;
            if (g_dpr_attention_enabled &&
                salt_gemma4_text_route_observer_begin(
                    text_model, &hot_route_observer) != 0) {
                emit_fatal(request_id, "hot_route",
                           "route observer admission failed");
                goto done;
            }
            hot_route_observer_active = g_dpr_attention_enabled;
            if ((g_dpr_mode == SALT_DPR_OFF &&
                 prefill_ordinary_measured(
                    text_model, prompt_ids, prompt_count, logits) != 0) ||
                (g_dpr_mode != SALT_DPR_OFF &&
                 prefill_parent_dpr(
                    text_model, prompt_ids, prompt_count, logits,
                    &dpr_store, dpr_stats_slots, dpr_retained,
                    &dpr_prefill_hits, &dpr_prefill_cached,
                    dpr_prefill_source) != 0)) {
                emit_fatal(request_id, "text_prefill",
                           "engine text prefill failed");
                goto done;
            }
            if (g_dpr_mode != SALT_DPR_OFF &&
                emit_dpr_prefill(
                    request_id, prompt_count,
                    dpr_prefill_hits, dpr_prefill_cached,
                    dpr_prefill_source, dpr_qa_key) != 0)
                goto done;
        }
        if (salt_mem_snapshot(&memory_snapshot) != 0 ||
            emit_memory(request_id, "after_prefill", &memory_snapshot) != 0) {
            emit_fatal(request_id, "memory", "memory snapshot failed");
            goto done;
        }

        if (g_dpr_mode != SALT_DPR_OFF) {
            if (generate_native_dpr(
                    text_model, logits, output_ids, request_id, turn_id,
                    response, response_capacity, &dpr_store,
                    g_dpr_attention_enabled ? &dpr_attention_store : NULL,
                    g_dpr_mentor_enabled ? &dpr_mentor_family_store : NULL,
                    g_dpr_mentor_enabled ? &dpr_mentor_binding_store : NULL,
                    dpr_mentor_candidates,
                    g_dpr_mentor_enabled
                        ? G4_DPR_MENTOR_BINDING_CAPACITY : 0,
                    dpr_mentor_effective_charts,
                    dpr_stats_slots, dpr_retained, &dpr_runtime,
                    dpr_qa_key,
                    &output_count, &stop_token) != 0) {
                emit_fatal(request_id, "dpr_engine",
                           "native DPR execution failed");
                goto done;
            }
        } else if (generate_measured(
                     text_model, logits, output_ids, request_id, turn_id,
                     response, response_capacity,
                     &output_count, &stop_token) != 0) {
            if (started) emit_fatal(request_id, "decode", "decode failed");
            goto done;
        }
        decode_count = output_count;
        if (emit_dpr_waterfall(request_id) != 0) {
            if (started)
                emit_fatal(request_id, "dpr_waterfall",
                           "request waterfall emission failed");
            goto done;
        }
        if (decode_count > 0 && is_stop_token(output_ids[decode_count - 1]))
            decode_count--;
        response_bytes = salt_gemma4_text_decode(
            text_model, output_ids, decode_count,
            response, response_capacity);
        if (response_bytes < 0 || response_bytes >= response_capacity ||
            memchr(response, '\0', (size_t)response_bytes) != NULL) {
            emit_fatal(request_id, "final_decode", "final decode failed");
            goto done;
        }
        {
            int pending;
            if (output_count < 1) {
                emit_fatal(request_id, "final_commit",
                           "final token commit failed");
                goto done;
            }
            pending = g_dpr_mode == SALT_DPR_OFF ? 0 :
                dpr_consume_pending(
                    text_model, NULL, output_ids[output_count - 1],
                    &dpr_runtime);
            if (pending < 0 ||
                (pending == 0 && commit_pending_token(
                    text_model, output_ids[output_count - 1], NULL) != 0)) {
                emit_fatal(request_id, "final_commit",
                           "final token commit failed");
                goto done;
            }
        }
        if (stop_token != 106) {
            if (commit_pending_token(text_model, 106, NULL) != 0) {
                emit_fatal(request_id, "synthetic_close",
                           "synthetic turn close failed");
                goto done;
            }
            synthetic_close = 1;
        }
        if (hot_route_observer_active) {
            int plan_rc;
            if (salt_gemma4_text_route_observer_end(text_model) != 0) {
                emit_fatal(request_id, "hot_route",
                           "route observer close failed");
                goto done;
            }
            hot_route_observer_active = 0;
            plan_rc = hot_route_candidate_update(
                &hot_route_observer, &hot_route_ledger,
                &hot_route_candidate);
            if (plan_rc < 0) {
                emit_fatal(request_id, "hot_route",
                           "hot expert candidate compilation failed");
                goto done;
            }
            if (plan_rc == 0) {
                if (salt_dpr_expert_coverage_ledger_sha256(
                        &hot_route_ledger,
                        g_hot_route_provenance_sha256) != 0) {
                    emit_fatal(request_id, "hot_route",
                               "hot expert provenance failed");
                    goto done;
                }
                memcpy(g_hot_route_mindset_sha256, mindset_sha256, 32);
                g_hot_route_ready = 1;
                fprintf(stderr,
                    "[hot-route] request=%llu request_observations=%zu "
                    "history_observations=%llu items=%u covered=%llu "
                    "total=%llu promoted=1\n",
                    request_id, hot_route_observer.count,
                    (unsigned long long)hot_route_ledger.observation_count,
                    hot_route_candidate.item_count,
                    (unsigned long long)
                        hot_route_candidate.expected_saved_numerator,
                    (unsigned long long)
                        hot_route_candidate.expected_saved_denominator);
            }
        }
        position_after = salt_gemma4_text_position(text_model);
        turn_id++;
        history_turns++;
        if (fill_record(text_model, &journal[journal_count],
                request_id, turn_id, history_turns,
                client_request_sha256, request_sha256,
                mindset_sha256,
                prompt_ids, prompt_count, image_tokens, output_ids,
                request_output_limit, output_count,
                stop_token, synthetic_close, position_before, position_after,
                response, response_bytes) != 0) {
            emit_fatal(request_id, "journal_commit",
                       "native result journal commit failed");
            goto done;
        }
        if (salt_mem_snapshot(&memory_snapshot) != 0 ||
            emit_memory(request_id, "pre_commit", &memory_snapshot) != 0) {
            emit_fatal(request_id, "memory", "memory snapshot failed");
            goto done;
        }
        journal_count++;
        if (emit_record(request_id, &journal[journal_count - 1], 0) != 0)
            goto done;
    }
    if (feof(stdin)) rc = 0;

done:
    g_hot_route_ready = 0;
    g_hot_route_candidate = NULL;
    if (hot_route_observer_active)
        (void)salt_gemma4_text_route_observer_end(text_model);
    if (g_dpr_mode != SALT_DPR_OFF && dpr_stats_slots) {
        for (size_t edge = 0; edge < dpr_store.edge_count; edge++) {
            if (dpr_stats_slots[edge].lookup_count != 0 &&
                salt_dpr_stats_save(g_dpr_root, &dpr_stats_slots[edge]) != 0 &&
                dpr_stats_slots[edge].persist_failures != UINT64_MAX)
                dpr_stats_slots[edge].persist_failures++;
        }
    }
    salt_gemma4_free_patches(patches, positions, padding);

    free(dpr_mentor_effective_charts);
    free(dpr_mentor_candidates);
    free(dpr_mentor_binding_buckets);
    free(dpr_mentor_binding_next);
    free(dpr_mentor_binding_slots);
    free(dpr_mentor_family_buckets);
    free(dpr_mentor_family_next);
    free(dpr_mentor_family_slots);
    free(dpr_attention_buckets);
    free(dpr_attention_slots);
    free(hot_route_counts);
    free(hot_route_items);
    free(dpr_buckets);
    free(dpr_retention_order);
    free(dpr_retained);
    free(dpr_retention_items);
    free(dpr_stats_slots);
    free(dpr_edge_slots);
    free(journal);
    free(g_journal_arena.responses);
    free(g_journal_arena.output_ids);
    free(g_journal_arena.prompt_ids);
    memset(&g_journal_arena, 0, sizeof g_journal_arena);
    free(features);
    free(response);
    free(logits);
    free(mm_types);
    free(output_ids);
    free(prompt_ids);
    free(image_path);
    free(prompt);
    return rc;
}

static void usage(const char *program) {
    fprintf(stderr,
            "usage: %s --serve-v2 --stream-events --ctx 512 --gen 128 "
            "[--workers 3] [--session-epoch 1] [--shared-kv FILE] "
            "[--proof-state] [--dpr-root DIR "
            "--dpr-mode persist|dynamic --dpr-current-kv-bytes N "
            "[--dpr-additive-bytes N] --dpr-serial-ns N "
            "--dpr-retention-bytes N "
            "[--dpr-mentor-policy-sha256 HEX] "
            "[--dpr-attention-policy-sha256 HEX]]\n",
            program);
}

int main(int argc, char **argv) {
    int serve_v2 = 0, context = G4_DEFAULT_CONTEXT;
    int generation = G4_DEFAULT_OUTPUT_LIMIT;
    int workers = 3, stream_events = 0, exit_code = 1;
    unsigned long long session_epoch = 1;
    const char *shared_kv = NULL;
    G4KvSharedLease shared_lease = {-1, MAP_FAILED, 0};
    SaltGemma4Text *text_model = NULL;
    SaltGemma4Vision *vision_model = NULL;
    const SaltModelDesc *model_desc = NULL;
    char error[256] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--serve-v2")) {
            serve_v2 = 1;
        } else if (!strcmp(argv[i], "--stream-events")) {
            stream_events = 1;
        } else if (!strcmp(argv[i], "--proof-state")) {
            g_proof_state = 1;
        } else if (!strcmp(argv[i], "--dpr-root") && i + 1 < argc) {
            g_dpr_root = argv[++i];
            if (!*g_dpr_root) goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-mode") && i + 1 < argc) {
            const char *mode = argv[++i];
            if (!strcmp(mode, "persist")) g_dpr_mode = SALT_DPR_PERSIST;
            else if (!strcmp(mode, "dynamic")) g_dpr_mode = SALT_DPR_DYNAMIC;
            else goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-current-kv-bytes") &&
                   i + 1 < argc) {
            if (parse_u64_positive(
                    argv[++i], &g_dpr_current_kv_bytes) != 0)
                goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-additive-bytes") &&
                   i + 1 < argc) {
            if (parse_u64_positive(argv[++i], &g_dpr_additive_bytes) != 0)
                goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-serial-ns") && i + 1 < argc) {
            if (parse_u64_positive(argv[++i], &g_dpr_serial_ns) != 0)
                goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-retention-bytes") &&
                   i + 1 < argc) {
            if (parse_u64_positive(argv[++i], &g_dpr_retention_bytes) != 0)
                goto bad_args;
        } else if (!strcmp(argv[i], "--dpr-mentor-policy-sha256") &&
                   i + 1 < argc) {
            if (salt_sha256_hex_parse(
                    argv[++i], g_dpr_mentor_policy_sha256) != 0)
                goto bad_args;
            g_dpr_mentor_enabled = 1;
        } else if (!strcmp(argv[i], "--dpr-attention-policy-sha256") &&
                   i + 1 < argc) {
            if (salt_sha256_hex_parse(
                    argv[++i], g_dpr_attention_policy_sha256) != 0)
                goto bad_args;
            g_dpr_attention_enabled = 1;
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            if (parse_positive(argv[++i], &context) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--gen") && i + 1 < argc) {
            if (parse_positive(argv[++i], &generation) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--workers") && i + 1 < argc) {
            if (parse_positive(argv[++i], &workers) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--session-epoch") && i + 1 < argc) {
            if (parse_epoch(argv[++i], &session_epoch) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--shared-kv") && i + 1 < argc) {
            shared_kv = argv[++i];
            if (!*shared_kv) goto bad_args;
        } else {
            goto bad_args;
        }
    }
    if (g_dpr_mode != SALT_DPR_OFF &&
        (parse_dpr_horizon_env("SALT_DPR_DRAFT_N", &g_dpr_draft_n) != 0 ||
         parse_dpr_horizon_env("SALT_DPR_PREFILL_N", &g_dpr_prefill_n) != 0 ||
         parse_dpr_qa_bits_env(&g_dpr_qa_bucket_bits) != 0))
        goto bad_args;
    if (!serve_v2 || !stream_events || context < 3 ||
        context > G4_MODEL_MAX_CONTEXT || generation < 1 ||
        generation > context - 2 || workers < 1 || workers > 8 ||
        (g_dpr_mode == SALT_DPR_OFF &&
         (g_dpr_root || g_dpr_current_kv_bytes || g_dpr_additive_bytes ||
          g_dpr_serial_ns || g_dpr_retention_bytes ||
          g_dpr_mentor_enabled || g_dpr_attention_enabled)) ||
        (g_dpr_mode != SALT_DPR_OFF &&
         (!g_dpr_root || !g_dpr_current_kv_bytes || !g_dpr_serial_ns ||
          !g_dpr_retention_bytes || shared_kv)) ||
        (g_dpr_mode == SALT_DPR_PERSIST && !g_dpr_additive_bytes) ||
        (g_dpr_mode == SALT_DPR_DYNAMIC && g_dpr_additive_bytes))
        goto bad_args;
    g_context = context;
    g_output_limit = generation;
    g_process_proof_state = g_proof_state;

    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    g_session_epoch = session_epoch;
    model_desc = salt_model_get("gemma4-26b-a4b");
    g_state_model = salt_model_state(model_desc);
    if (!model_desc || model_desc->runtime_ready != 1 || !g_state_model ||
        g_state_model->schema_version != SALT_STATE_SCHEMA_VERSION ||
        g_state_model->facts_kind != SALT_FACTS_KV_ROWS ||
        !g_state_model->facts_optional ||
        !g_state_model->clear_facts_supported ||
        g_state_model->default_mindset_end > INT_MAX ||
        !(g_mindset_mode = salt_state_mindset_name(
              g_state_model->mindset_kind))) {
        fputs("gemma4 persistent server: invalid model state descriptor\n",
              stderr);
        goto done;
    }
    g_mindset_end = (int)g_state_model->default_mindset_end;
    text_model = salt_gemma4_text_load(
        stdin, context, workers, error, sizeof error);
    if (!text_model) {
        fprintf(stderr, "gemma4 persistent server: text load failed: %s\n", error);
        goto done;
    }
    vision_model = salt_gemma4_vision_load(
        stdin, workers, error, sizeof error);
    if (!vision_model) {
        fprintf(stderr, "gemma4 persistent server: vision load failed: %s\n", error);
        goto done;
    }
    if (shared_kv) {
        int shared_position = 0;
        if (g4_kv_file_attach_shared(
                text_model, shared_kv, context, &shared_lease,
                &shared_position, error, sizeof error) != 0) {
            fprintf(stderr, "gemma4 persistent server: shared KV failed: %s\n",
                    error);
            goto done;
        }
        if (shared_position < 1) {
            fputs("gemma4 persistent server: empty shared mindset\n", stderr);
            goto done;
        }
        g_mindset_end = shared_position;
    }
    exit_code = serve_loop(text_model, vision_model);
    goto done;

bad_args:
    usage(argv[0]);
    return 2;

done:
    if (salt_gemma4_vision_close(&vision_model) != 0) exit_code = 1;
    salt_gemma4_text_free(text_model);
    g4_kv_shared_release(&shared_lease);
    return exit_code;
}
