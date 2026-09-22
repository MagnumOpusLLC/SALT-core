#include "salt/dpr.h"
#include "sha256.h"

#include <limits.h>
#include <string.h>

static int digest_is_zero(const uint8_t value[32]) {
    uint8_t total = 0;
    if (!value) return 1;
    for (size_t i = 0; i < 32; i++) total |= value[i];
    return total == 0;
}

static void le32_store(uint8_t *out, uint32_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static void le64_store(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) out[i] = (uint8_t)(value >> (8u * i));
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (!out || left > UINT64_MAX - right) return -1;
    *out = left + right;
    return 0;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (!out || (left != 0 && right > UINT64_MAX / left)) return -1;
    *out = left * right;
    return 0;
}

static int chart_round_cost(const SaltDprChart *chart, uint64_t *out) {
    uint64_t total = 0;
    if (!chart || !out ||
        add_u64(total, chart->lookup_ns, &total) != 0 ||
        add_u64(total, chart->recover_ns, &total) != 0 ||
        add_u64(total, chart->draft_ns, &total) != 0 ||
        add_u64(total, chart->verify_ns, &total) != 0 ||
        add_u64(total, chart->commit_ns, &total) != 0)
        return -1;
    *out = total;
    return 0;
}

/* Compare left_num/left_den to right_num/right_den without multiplication.
 * Return -1, 0, or 1. */
static int fraction_compare(uint64_t left_num, uint64_t left_den,
                            uint64_t right_num, uint64_t right_den) {
    int reversed = 0;
    if (left_den == 0 || right_den == 0) return 0;
    for (;;) {
        uint64_t left_q = left_num / left_den;
        uint64_t right_q = right_num / right_den;
        uint64_t left_r, right_r;
        if (left_q != right_q) {
            int result = left_q < right_q ? -1 : 1;
            return reversed ? -result : result;
        }
        left_r = left_num % left_den;
        right_r = right_num % right_den;
        if (left_r == 0 || right_r == 0) {
            int result;
            if (left_r == 0 && right_r == 0) return 0;
            result = left_r == 0 ? -1 : 1;
            return reversed ? -result : result;
        }
        left_num = left_den;
        left_den = left_r;
        right_num = right_den;
        right_den = right_r;
        reversed = !reversed;
    }
}

int salt_dpr_node_sha256(const SaltDprNodeMaterial *material,
                         uint8_t out[32]) {
    uint8_t bytes[152];
    static const uint8_t magic[8] = {'S','A','L','T','D','P','R','N'};
    if (!material || !out ||
        digest_is_zero(material->state_compatibility_sha256) ||
        digest_is_zero(material->state_sha256) ||
        digest_is_zero(material->sampler_config_sha256) ||
        (material->sampler_abi != SALT_DPR_SAMPLER_GREEDY_V1 &&
         material->sampler_abi != SALT_DPR_SAMPLER_COUNTER_V1) ||
        (material->sampler_abi == SALT_DPR_SAMPLER_GREEDY_V1 &&
         !digest_is_zero(material->target_rng_sha256)) ||
        (material->sampler_abi == SALT_DPR_SAMPLER_COUNTER_V1 &&
         digest_is_zero(material->target_rng_sha256)))
        return -1;
    memset(bytes, 0, sizeof bytes);
    memcpy(bytes, magic, sizeof magic);
    le32_store(bytes + 8, 1u);
    le32_store(bytes + 12, material->sampler_abi);
    le64_store(bytes + 16, material->position);
    memcpy(bytes + 24, material->state_compatibility_sha256, 32);
    memcpy(bytes + 56, material->state_sha256, 32);
    memcpy(bytes + 88, material->sampler_config_sha256, 32);
    memcpy(bytes + 120, material->target_rng_sha256, 32);
    return salt_sha256_bytes(bytes, sizeof bytes, out);
}

static uint64_t qa_mix64(uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

int salt_dpr_transition_chain_init(
        const uint8_t state_compatibility_sha256[32],
        const uint8_t sampler_config_sha256[32], uint32_t sampler_abi,
        uint64_t position, uint8_t out[32]) {
    static const uint8_t domain[8] = {'S','A','L','T','T','R','I','1'};
    uint8_t bytes[84];
    if (!state_compatibility_sha256 ||
        digest_is_zero(state_compatibility_sha256) ||
        !sampler_config_sha256 || digest_is_zero(sampler_config_sha256) ||
        sampler_abi == 0 || !out)
        return -1;
    memcpy(bytes, domain, 8);
    memcpy(bytes + 8, state_compatibility_sha256, 32);
    memcpy(bytes + 40, sampler_config_sha256, 32);
    le32_store(bytes + 72, sampler_abi);
    le64_store(bytes + 76, position);
    return salt_sha256_bytes(bytes, sizeof bytes, out);
}

int salt_dpr_transition_chain_advance(
        const uint8_t previous[32], int32_t token_id,
        uint64_t resulting_position, uint8_t out[32]) {
    static const uint8_t domain[8] = {'S','A','L','T','T','R','A','1'};
    uint8_t bytes[52];
    if (!previous || digest_is_zero(previous) || token_id < 0 ||
        resulting_position == 0 || !out)
        return -1;
    memcpy(bytes, domain, 8);
    memcpy(bytes + 8, previous, 32);
    le32_store(bytes + 40, (uint32_t)token_id);
    le64_store(bytes + 44, resulting_position);
    return salt_sha256_bytes(bytes, sizeof bytes, out);
}

int salt_dpr_nomogram_qa_key(const uint8_t state_compatibility_sha256[32],
                             const int32_t *prompt_token_ids,
                             size_t prompt_token_count,
                             uint32_t bucket_bits,
                             uint8_t out[32]) {
    static const uint8_t domain[8] = {'S','A','L','T','Q','A','K','1'};
    int32_t scores[64] = {0};
    uint8_t bytes[56];
    uint64_t signature = 0, mask, bucket;
    if (!state_compatibility_sha256 ||
        digest_is_zero(state_compatibility_sha256) ||
        !prompt_token_ids || prompt_token_count == 0 ||
        prompt_token_count > INT32_MAX || bucket_bits == 0 ||
        bucket_bits > 32 || !out)
        return -1;
    for (size_t token = 0; token < prompt_token_count; token++) {
        uint64_t hash;
        if (prompt_token_ids[token] < 0) return -1;
        hash = qa_mix64((uint64_t)(uint32_t)prompt_token_ids[token]);
        for (uint32_t bit = 0; bit < 64; bit++)
            scores[bit] += (hash >> bit) & 1u ? 1 : -1;
    }
    for (uint32_t bit = 0; bit < 64; bit++)
        if (scores[bit] >= 0) signature |= UINT64_C(1) << bit;
    mask = UINT64_MAX << (64u - bucket_bits);
    bucket = signature & mask;
    memset(bytes, 0, sizeof bytes);
    memcpy(bytes, domain, sizeof domain);
    memcpy(bytes + 8, state_compatibility_sha256, 32);
    le32_store(bytes + 40, bucket_bits);
    le64_store(bytes + 48, bucket);
    return salt_sha256_bytes(bytes, sizeof bytes, out);
}

int salt_dpr_chart_validate(const SaltDprChart *chart) {
    uint64_t maximum_accepted, ignored;
    if (!chart || chart->horizon == 0 ||
        chart->horizon > SALT_DPR_MAX_HORIZON ||
        chart->rounds > chart->sample_count ||
        mul_u64(chart->rounds, chart->horizon, &maximum_accepted) != 0 ||
        chart->accepted_tokens_total > maximum_accepted ||
        (chart->rounds == 0 && chart->accepted_tokens_total != 0) ||
        chart_round_cost(chart, &ignored) != 0)
        return -1;
    return 0;
}

int salt_dpr_chart_fraction(const SaltDprChart *chart,
                            uint64_t *numerator, uint64_t *denominator) {
    uint64_t round_cost;
    if (!numerator || !denominator || salt_dpr_chart_validate(chart) != 0 ||
        chart->rounds == 0 || chart_round_cost(chart, &round_cost) != 0 ||
        mul_u64(round_cost, chart->rounds, numerator) != 0 ||
        add_u64(chart->accepted_tokens_total, chart->rounds, denominator) != 0 ||
        *denominator == 0)
        return -1;
    return 0;
}

int salt_dpr_chart_expected_saved(const SaltDprChart *chart,
                                  uint64_t serial_ns_per_token,
                                  uint64_t *numerator,
                                  uint64_t *denominator) {
    uint64_t cost, committed, serial;
    if (!numerator || !denominator || serial_ns_per_token == 0 ||
        salt_dpr_chart_fraction(chart, &cost, &committed) != 0 ||
        mul_u64(serial_ns_per_token, committed, &serial) != 0)
        return -1;
    *numerator = serial > cost ? serial - cost : 0;
    *denominator = committed;
    return 0;
}

int salt_dpr_chart_is_green(const SaltDprChart *chart,
                            uint64_t serial_ns_per_token,
                            uint64_t minimum_samples) {
    uint64_t numerator, denominator;
    if (serial_ns_per_token == 0 || minimum_samples == 0 || !chart ||
        chart->sample_count < minimum_samples ||
        salt_dpr_chart_fraction(chart, &numerator, &denominator) != 0)
        return 0;
    return fraction_compare(numerator, denominator,
                            serial_ns_per_token, 1) < 0;
}

int salt_dpr_chart_compare(const SaltDprChart *left,
                           const SaltDprChart *right,
                           int *comparison) {
    uint64_t left_num, left_den, right_num, right_den;
    if (!comparison ||
        salt_dpr_chart_fraction(left, &left_num, &left_den) != 0 ||
        salt_dpr_chart_fraction(right, &right_num, &right_den) != 0)
        return -1;
    *comparison = fraction_compare(left_num, left_den, right_num, right_den);
    return 0;
}

int salt_dpr_edge_validate(const SaltDprEdge *edge,
                           int32_t vocab_size,
                           uint64_t max_context,
                           SaltDprStopTokenFn stop_token,
                           void *stop_opaque) {
    int reference_absent, next_absent;
    if (!edge) return -1;
    reference_absent = digest_is_zero(edge->reference_sha256);
    next_absent = digest_is_zero(edge->next_node_sha256);
    if (edge->schema_version != SALT_DPR_EDGE_VERSION ||
        (edge->mode != SALT_DPR_PERSIST && edge->mode != SALT_DPR_DYNAMIC) ||
        edge->horizon == 0 || edge->horizon > SALT_DPR_MAX_HORIZON ||
        edge->candidate_count != edge->horizon ||
        digest_is_zero(edge->source_node_sha256) ||
        vocab_size <= 0 || edge->chart.horizon != edge->horizon ||
        salt_dpr_chart_validate(&edge->chart) != 0)
        return -1;
    if (edge->kind == SALT_DPR_EDGE_PARENT_EXACT ||
        edge->kind == SALT_DPR_EDGE_PARENT_PREFILL) {
        if (reference_absent || edge->reference_bytes == 0 ||
            edge->reference_position == 0 ||
            edge->reference_position > max_context || !next_absent)
            return -1;
        if ((edge->kind == SALT_DPR_EDGE_PARENT_EXACT &&
             (edge->bonus_token_id < 0 || edge->bonus_token_id >= vocab_size)) ||
            (edge->kind == SALT_DPR_EDGE_PARENT_PREFILL &&
             edge->bonus_token_id != -1))
            return -1;
    } else if (edge->kind == SALT_DPR_EDGE_NOMOGRAM_DRAFT ||
               edge->kind == SALT_DPR_EDGE_NOMOGRAM_PREFILL) {
        if (!reference_absent || edge->reference_bytes != 0 ||
            edge->reference_position != 0 || next_absent ||
            edge->horizon != 1 || edge->candidate_count != 1 ||
            edge->bonus_token_id != -1)
            return -1;
    } else {
        return -1;
    }
    for (uint32_t i = 0; i < edge->candidate_count; i++) {
        int32_t token = edge->candidate_token_ids[i];
        if (token < 0 || token >= vocab_size ||
            (stop_token && stop_token(stop_opaque, token)))
            return -1;
    }
    return 0;
}

int salt_dpr_select_edge(const SaltDprEdge *edges, size_t edge_count,
                         const uint8_t source_node_sha256[32],
                         SaltDprMode mode,
                         uint64_t serial_ns_per_token,
                         uint64_t minimum_samples,
                         int32_t vocab_size,
                         uint64_t max_context,
                         SaltDprStopTokenFn stop_token,
                         void *stop_opaque,
                         size_t *selected_index) {
    size_t best = SIZE_MAX;
    uint64_t best_num = 0, best_den = 1;
    if (!selected_index || !source_node_sha256 ||
        (edge_count != 0 && !edges) ||
        (mode != SALT_DPR_OFF && mode != SALT_DPR_PERSIST &&
         mode != SALT_DPR_DYNAMIC))
        return -1;
    *selected_index = SIZE_MAX;
    if (mode == SALT_DPR_OFF) return 0;
    for (size_t i = 0; i < edge_count; i++) {
        uint64_t numerator, denominator;
        if (salt_dpr_edge_validate(
                &edges[i], vocab_size, max_context,
                stop_token, stop_opaque) != 0 ||
            edges[i].mode != mode ||
            memcmp(edges[i].source_node_sha256, source_node_sha256, 32) != 0 ||
            !salt_dpr_chart_is_green(
                &edges[i].chart, serial_ns_per_token, minimum_samples) ||
            salt_dpr_chart_fraction(
                &edges[i].chart, &numerator, &denominator) != 0)
            continue;
        if (best == SIZE_MAX ||
            fraction_compare(numerator, denominator, best_num, best_den) < 0) {
            best = i;
            best_num = numerator;
            best_den = denominator;
        }
    }
    if (best == SIZE_MAX) return 0;
    *selected_index = best;
    return 1;
}

int salt_dpr_mentor_family_validate(const SaltDprMentorFamily *family) {
    if (!family ||
        family->schema_version != SALT_DPR_MENTOR_FAMILY_VERSION ||
        digest_is_zero(family->family_sha256) ||
        digest_is_zero(family->mindset_sha256) ||
        digest_is_zero(family->state_compatibility_sha256) ||
        digest_is_zero(family->policy_sha256) ||
        digest_is_zero(family->nomogram_root_sha256) ||
        digest_is_zero(family->provenance_sha256))
        return -1;
    return 0;
}

static int strict_bool(int value) {
    return value == 0 || value == 1;
}

static int selector_candidate_validate(
        const SaltDprMentorFamily *families, size_t family_count,
        const SaltDprSelectorCandidate *candidate) {
    if (!candidate ||
        candidate->schema_version != SALT_DPR_SELECTOR_CANDIDATE_VERSION ||
        digest_is_zero(candidate->edge_sha256) ||
        digest_is_zero(candidate->source_node_sha256) ||
        digest_is_zero(candidate->state_compatibility_sha256) ||
        digest_is_zero(candidate->policy_sha256) ||
        candidate->expected_saved_denominator == 0 ||
        !strict_bool(candidate->provenance_authenticated) ||
        !strict_bool(candidate->policy_allowed))
        return -1;
    if (candidate->kind == SALT_DPR_SELECTOR_PARENT_EXACT)
        return candidate->family_index == SIZE_MAX &&
               digest_is_zero(candidate->binding_sha256) ? 0 : -1;
    if (candidate->kind != SALT_DPR_SELECTOR_MENTOR_NOMOGRAM ||
        digest_is_zero(candidate->binding_sha256) ||
        candidate->family_index >= family_count || !families ||
        salt_dpr_mentor_family_validate(
            &families[candidate->family_index]) != 0 ||
        memcmp(candidate->state_compatibility_sha256,
               families[candidate->family_index].state_compatibility_sha256,
               32) != 0 ||
        memcmp(candidate->policy_sha256,
               families[candidate->family_index].policy_sha256, 32) != 0)
        return -1;
    return 0;
}

static int selector_candidate_eligible(
        const SaltDprSelectorCandidate *candidate,
        const uint8_t target_node_sha256[32],
        const uint8_t target_state_compatibility_sha256[32],
        const uint8_t active_policy_sha256[32]) {
    return candidate->provenance_authenticated &&
           candidate->policy_allowed &&
           candidate->expected_saved_numerator > 0 &&
           memcmp(candidate->source_node_sha256,
                  target_node_sha256, 32) == 0 &&
           memcmp(candidate->state_compatibility_sha256,
                  target_state_compatibility_sha256, 32) == 0 &&
           memcmp(candidate->policy_sha256, active_policy_sha256, 32) == 0;
}

static int selector_candidate_better(
        const SaltDprMentorFamily *families,
        const SaltDprSelectorCandidate *left,
        const SaltDprSelectorCandidate *right) {
    int comparison = fraction_compare(
        left->expected_saved_numerator, left->expected_saved_denominator,
        right->expected_saved_numerator, right->expected_saved_denominator);
    if (comparison != 0) return comparison > 0;
    if (left->kind == SALT_DPR_SELECTOR_MENTOR_NOMOGRAM) {
        comparison = memcmp(families[left->family_index].family_sha256,
                            families[right->family_index].family_sha256, 32);
        if (comparison != 0) return comparison < 0;
    }
    return memcmp(left->edge_sha256, right->edge_sha256, 32) < 0;
}

static void selector_result_reset(SaltDprSelectorResult *result) {
    memset(result, 0, sizeof *result);
    result->kind = SALT_DPR_SELECTOR_NATIVE_FALLBACK;
    result->candidate_index = SIZE_MAX;
    result->family_index = SIZE_MAX;
}

int salt_dpr_selector_choose(
        const SaltDprMentorFamily *families, size_t family_count,
        const SaltDprSelectorCandidate *candidates, size_t candidate_count,
        const uint8_t target_node_sha256[32],
        const uint8_t target_state_compatibility_sha256[32],
        const uint8_t active_policy_sha256[32],
        SaltDprSelectorResult *result) {
    size_t best_parent = SIZE_MAX, best_mentor = SIZE_MAX, best;
    if (!result || !target_node_sha256 ||
        !target_state_compatibility_sha256 ||
        !active_policy_sha256 ||
        digest_is_zero(target_node_sha256) ||
        digest_is_zero(target_state_compatibility_sha256) ||
        digest_is_zero(active_policy_sha256) ||
        (family_count != 0 && !families) ||
        (candidate_count != 0 && !candidates))
        return -1;
    selector_result_reset(result);
    for (size_t i = 0; i < family_count; i++) {
        if (salt_dpr_mentor_family_validate(&families[i]) != 0)
            return -1;
        for (size_t j = 0; j < i; j++)
            if (memcmp(families[i].family_sha256,
                       families[j].family_sha256, 32) == 0)
                return -1;
    }
    for (size_t i = 0; i < candidate_count; i++) {
        const SaltDprSelectorCandidate *candidate = &candidates[i];
        size_t *slot;
        if (selector_candidate_validate(
                families, family_count, candidate) != 0)
            return -1;
        if (!selector_candidate_eligible(
                candidate, target_node_sha256,
                target_state_compatibility_sha256,
                active_policy_sha256))
            continue;
        slot = candidate->kind == SALT_DPR_SELECTOR_PARENT_EXACT
            ? &best_parent : &best_mentor;
        if (*slot == SIZE_MAX || selector_candidate_better(
                families, candidate, &candidates[*slot]))
            *slot = i;
    }
    best = best_parent != SIZE_MAX ? best_parent : best_mentor;
    if (best == SIZE_MAX) return 0;
    result->kind = candidates[best].kind;
    result->candidate_index = best;
    result->family_index = candidates[best].family_index;
    result->expected_saved_numerator =
        candidates[best].expected_saved_numerator;
    result->expected_saved_denominator =
        candidates[best].expected_saved_denominator;
    memcpy(result->source_node_sha256,
           candidates[best].source_node_sha256, 32);
    memcpy(result->state_compatibility_sha256,
           candidates[best].state_compatibility_sha256, 32);
    memcpy(result->policy_sha256, candidates[best].policy_sha256, 32);
    memcpy(result->edge_sha256, candidates[best].edge_sha256, 32);
    memcpy(result->binding_sha256, candidates[best].binding_sha256, 32);
    if (result->kind == SALT_DPR_SELECTOR_MENTOR_NOMOGRAM) {
        const SaltDprMentorFamily *family = &families[result->family_index];
        memcpy(result->family_sha256, family->family_sha256, 32);
        memcpy(result->mindset_sha256, family->mindset_sha256, 32);
        memcpy(result->provenance_sha256, family->provenance_sha256, 32);
        memcpy(result->nomogram_root_sha256,
               family->nomogram_root_sha256, 32);
    }
    return 1;
}

int salt_dpr_candidate_sha256(const int32_t *candidate_token_ids,
                              uint32_t candidate_count, uint8_t out[32]) {
    static const uint8_t domain[8] = {'S','A','L','T','D','P','D','1'};
    SaltSha256 hasher;
    uint8_t count[4], token[4];
    if (!candidate_token_ids || candidate_count == 0 ||
        candidate_count > SALT_DPR_MAX_HORIZON || !out)
        return -1;
    le32_store(count, candidate_count);
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, domain, sizeof domain);
    salt_sha256_update(&hasher, count, sizeof count);
    for (uint32_t i = 0; i < candidate_count; i++) {
        if (candidate_token_ids[i] < 0) return -1;
        le32_store(token, (uint32_t)candidate_token_ids[i]);
        salt_sha256_update(&hasher, token, sizeof token);
    }
    salt_sha256_final(&hasher, out);
    return 0;
}

static int compute_intent_validate(const SaltDprComputeIntent *intent) {
    int draft;
    if (!intent ||
        intent->schema_version != SALT_DPR_COMPUTE_INTENT_VERSION ||
        intent->kind < SALT_DPR_OPERATION_PREFILL ||
        intent->kind > SALT_DPR_OPERATION_REPAIR ||
        intent->horizon == 0 || intent->horizon > SALT_DPR_MAX_HORIZON ||
        intent->candidate_count > intent->horizon ||
        digest_is_zero(intent->target_node_sha256) ||
        digest_is_zero(intent->state_compatibility_sha256) ||
        digest_is_zero(intent->active_policy_sha256))
        return -1;
    draft = intent->kind == SALT_DPR_OPERATION_DRAFT_VERIFY;
    if ((draft && (intent->candidate_count == 0 ||
                   intent->candidate_count != intent->horizon ||
                   digest_is_zero(intent->candidate_sha256))) ||
        (!draft && (intent->candidate_count != 0 ||
                    !digest_is_zero(intent->candidate_sha256))))
        return -1;
    return 0;
}

int salt_dpr_compute_intent_sha256(const SaltDprComputeIntent *intent,
                                   uint8_t out[32]) {
    static const uint8_t magic[8] = {'S','A','L','T','D','P','C','I'};
    uint8_t bytes[160];
    if (!out || compute_intent_validate(intent) != 0) return -1;
    memset(bytes, 0, sizeof bytes);
    memcpy(bytes, magic, sizeof magic);
    le32_store(bytes + 8, intent->schema_version);
    le32_store(bytes + 12, (uint32_t)intent->kind);
    le64_store(bytes + 16, intent->position);
    le32_store(bytes + 24, intent->horizon);
    le32_store(bytes + 28, intent->candidate_count);
    memcpy(bytes + 32, intent->target_node_sha256, 32);
    memcpy(bytes + 64, intent->state_compatibility_sha256, 32);
    memcpy(bytes + 96, intent->active_policy_sha256, 32);
    memcpy(bytes + 128, intent->candidate_sha256, 32);
    return salt_sha256_bytes(bytes, sizeof bytes, out);
}

int salt_dpr_transition_proposal_validate(
        const SaltDprTransitionProposal *proposal,
        const SaltDprComputeIntent *intent, int32_t vocab_size) {
    uint8_t candidate_sha256[32];
    if (!proposal || compute_intent_validate(intent) != 0 || vocab_size <= 0 ||
        proposal->schema_version != SALT_DPR_TRANSITION_PROPOSAL_VERSION ||
        intent->kind != SALT_DPR_OPERATION_DRAFT_VERIFY ||
        proposal->candidate_count == 0 ||
        proposal->candidate_count != intent->candidate_count ||
        digest_is_zero(proposal->proposal_sha256) ||
        digest_is_zero(proposal->policy_sha256) ||
        proposal->expected_saved_denominator == 0 ||
        (proposal->provenance_authenticated != 0 &&
         proposal->provenance_authenticated != 1) ||
        (proposal->policy_allowed != 0 && proposal->policy_allowed != 1))
        return -1;
    for (uint32_t i = 0; i < proposal->candidate_count; i++)
        if (proposal->candidate_token_ids[i] < 0 ||
            proposal->candidate_token_ids[i] >= vocab_size)
            return -1;
    if (salt_dpr_candidate_sha256(
            proposal->candidate_token_ids, proposal->candidate_count,
            candidate_sha256) != 0)
        return -1;
    if (memcmp(proposal->source_node_sha256,
               intent->target_node_sha256, 32) != 0 ||
        memcmp(proposal->state_compatibility_sha256,
               intent->state_compatibility_sha256, 32) != 0 ||
        memcmp(proposal->policy_sha256,
               intent->active_policy_sha256, 32) != 0 ||
        memcmp(candidate_sha256, intent->candidate_sha256, 32) != 0 ||
        !proposal->provenance_authenticated || !proposal->policy_allowed ||
        proposal->expected_saved_numerator == 0)
        return 0;
    return 1;
}

static int relevance_item_same(const SaltDprRelevanceItem *left,
                               const SaltDprRelevanceItem *right) {
    return left->kind == right->kind && left->layer == right->layer &&
        left->index == right->index && left->start == right->start &&
        left->length == right->length;
}

int salt_dpr_attention_plan_validate(
        const SaltDprAttentionPlan *plan, const SaltDprComputeIntent *intent,
        uint32_t max_layers, uint32_t max_experts, uint64_t max_context) {
    uint8_t intent_sha256[32];
    if (!plan || compute_intent_validate(intent) != 0 ||
        max_layers == 0 || max_experts == 0 || max_context == 0 ||
        plan->schema_version != SALT_DPR_ATTENTION_PLAN_VERSION ||
        plan->exactness < SALT_DPR_EXACTNESS_A0 ||
        plan->exactness > SALT_DPR_EXACTNESS_A2 ||
        digest_is_zero(plan->dpr_sha256) ||
        digest_is_zero(plan->mindset_sha256) ||
        digest_is_zero(plan->provenance_sha256) ||
        plan->item_count == 0 ||
        plan->item_count > SALT_DPR_MAX_RELEVANCE_ITEMS ||
        plan->expected_saved_denominator == 0 ||
        (plan->provenance_authenticated != 0 &&
         plan->provenance_authenticated != 1) ||
        (plan->policy_allowed != 0 && plan->policy_allowed != 1) ||
        salt_dpr_compute_intent_sha256(intent, intent_sha256) != 0)
        return -1;
    for (uint32_t i = 0; i < plan->item_count; i++) {
        const SaltDprRelevanceItem *item = &plan->items[i];
        if (item->priority == 0 || item->layer >= max_layers ||
            (i > 0 && item->priority > plan->items[i - 1].priority))
            return -1;
        if (item->kind == SALT_DPR_RELEVANCE_EXPERT_PREFETCH) {
            if (item->index >= max_experts || item->start != 0 ||
                item->length != 0)
                return -1;
        } else if (item->kind == SALT_DPR_RELEVANCE_KV_PREFETCH) {
            if (item->index != 0 || item->length == 0 ||
                item->start >= max_context ||
                item->length > max_context - item->start)
                return -1;
        } else {
            return -1;
        }
        for (uint32_t prior = 0; prior < i; prior++)
            if (relevance_item_same(item, &plan->items[prior])) return -1;
    }
    if (memcmp(plan->intent_sha256, intent_sha256, 32) != 0 ||
        memcmp(plan->state_compatibility_sha256,
               intent->state_compatibility_sha256, 32) != 0 ||
        memcmp(plan->policy_sha256, intent->active_policy_sha256, 32) != 0 ||
        !plan->provenance_authenticated || !plan->policy_allowed ||
        plan->expected_saved_numerator == 0 ||
        plan->exactness != SALT_DPR_EXACTNESS_A0)
        return 0;
    return 1;
}

static int coverage_item_before(uint64_t left_count, uint32_t left_layer,
        uint32_t left_expert, uint64_t right_count, uint32_t right_layer,
        uint32_t right_expert) {
    if (left_count != right_count) return left_count > right_count;
    if (left_layer != right_layer) return left_layer < right_layer;
    return left_expert < right_expert;
}

int salt_dpr_attention_plan_compile_expert_coverage(
        SaltDprAttentionPlan *plan, const uint64_t *route_counts,
        uint32_t layer_count, uint32_t expert_count,
        uint32_t maximum_per_layer, uint32_t maximum_items) {
    uint64_t selected_counts[SALT_DPR_MAX_RELEVANCE_ITEMS];
    uint64_t total = 0, covered = 0;
    uint32_t item_count = 0;
    if (!plan || !route_counts || layer_count == 0 || expert_count == 0 ||
        maximum_per_layer == 0 || maximum_per_layer > expert_count ||
        maximum_per_layer > SALT_DPR_MAX_RELEVANCE_ITEMS ||
        maximum_items == 0 || maximum_items > SALT_DPR_MAX_RELEVANCE_ITEMS ||
        layer_count > SIZE_MAX / expert_count)
        return -1;
    memset(plan->items, 0, sizeof plan->items);
    memset(selected_counts, 0, sizeof selected_counts);
    plan->item_count = 0;
    plan->expected_saved_numerator = 0;
    plan->expected_saved_denominator = 0;
    for (uint32_t layer = 0; layer < layer_count; layer++) {
        uint32_t chosen[SALT_DPR_MAX_RELEVANCE_ITEMS];
        uint32_t chosen_count = 0;
        for (uint32_t expert = 0; expert < expert_count; expert++) {
            uint64_t count = route_counts[(size_t)layer * expert_count + expert];
            if (count > UINT64_MAX - total) return -1;
            total += count;
        }
        for (uint32_t rank = 0; rank < maximum_per_layer; rank++) {
            uint64_t best_count = 0;
            uint32_t best_expert = UINT32_MAX;
            for (uint32_t expert = 0; expert < expert_count; expert++) {
                uint64_t count = route_counts[
                    (size_t)layer * expert_count + expert];
                int already = 0;
                for (uint32_t prior = 0; prior < chosen_count; prior++)
                    if (chosen[prior] == expert) { already = 1; break; }
                if (already || count == 0) continue;
                if (best_expert == UINT32_MAX || count > best_count ||
                    (count == best_count && expert < best_expert)) {
                    best_count = count;
                    best_expert = expert;
                }
            }
            if (best_expert == UINT32_MAX) break;
            chosen[chosen_count++] = best_expert;
            uint32_t insert = 0;
            while (insert < item_count && !coverage_item_before(
                    best_count, layer, best_expert, selected_counts[insert],
                    plan->items[insert].layer, plan->items[insert].index))
                insert++;
            if (insert < maximum_items) {
                uint32_t limit = item_count < maximum_items
                    ? item_count : maximum_items - 1u;
                for (uint32_t move = limit; move > insert; move--) {
                    plan->items[move] = plan->items[move - 1u];
                    selected_counts[move] = selected_counts[move - 1u];
                }
                plan->items[insert] = (SaltDprRelevanceItem) {
                    SALT_DPR_RELEVANCE_EXPERT_PREFETCH, layer, best_expert,
                    0u, 0u,
                    best_count > UINT32_MAX ? UINT32_MAX : (uint32_t)best_count,
                };
                selected_counts[insert] = best_count;
                if (item_count < maximum_items) item_count++;
            }
        }
    }
    if (total == 0) return 1;
    for (uint32_t index = 0; index < item_count; index++) {
        if (selected_counts[index] > UINT64_MAX - covered) return -1;
        covered += selected_counts[index];
    }
    if (item_count == 0 || covered == 0) return -1;
    plan->item_count = item_count;
    plan->expected_saved_numerator = covered;
    plan->expected_saved_denominator = total;
    return 0;
}

static int expert_coverage_ledger_valid(
        const SaltDprExpertCoverageLedger *ledger) {
    return ledger && ledger->route_counts && ledger->layer_count != 0 &&
        ledger->expert_count != 0 &&
        ledger->layer_count <= SIZE_MAX / ledger->expert_count &&
        (size_t)ledger->layer_count * ledger->expert_count <=
            SIZE_MAX / sizeof *ledger->route_counts;
}

int salt_dpr_expert_coverage_ledger_init(
        SaltDprExpertCoverageLedger *ledger, uint64_t *route_counts,
        uint32_t layer_count, uint32_t expert_count) {
    size_t count;
    if (!ledger || !route_counts || layer_count == 0 || expert_count == 0 ||
        layer_count > SIZE_MAX / expert_count ||
        (size_t)layer_count * expert_count > SIZE_MAX / sizeof *route_counts)
        return -1;
    count = (size_t)layer_count * expert_count;
    memset(route_counts, 0, count * sizeof *route_counts);
    memset(ledger, 0, sizeof *ledger);
    ledger->route_counts = route_counts;
    ledger->layer_count = layer_count;
    ledger->expert_count = expert_count;
    return 0;
}

int salt_dpr_expert_coverage_ledger_reset(
        SaltDprExpertCoverageLedger *ledger) {
    size_t count;
    if (!expert_coverage_ledger_valid(ledger)) return -1;
    count = (size_t)ledger->layer_count * ledger->expert_count;
    memset(ledger->route_counts, 0,
           count * sizeof *ledger->route_counts);
    ledger->observation_count = 0;
    return 0;
}

int salt_dpr_expert_coverage_ledger_merge(
        SaltDprExpertCoverageLedger *ledger, const uint64_t *route_counts) {
    uint64_t added = 0;
    size_t count;
    if (!expert_coverage_ledger_valid(ledger) || !route_counts) return -1;
    count = (size_t)ledger->layer_count * ledger->expert_count;
    for (size_t index = 0; index < count; index++) {
        if (route_counts[index] > UINT64_MAX - added ||
            route_counts[index] > UINT64_MAX - ledger->route_counts[index])
            return -1;
        added += route_counts[index];
    }
    if (added > UINT64_MAX - ledger->observation_count) return -1;
    for (size_t index = 0; index < count; index++)
        ledger->route_counts[index] += route_counts[index];
    ledger->observation_count += added;
    return added == 0 ? 1 : 0;
}

int salt_dpr_expert_coverage_ledger_compile(
        const SaltDprExpertCoverageLedger *ledger,
        uint32_t maximum_per_layer, uint32_t maximum_items,
        SaltDprAttentionPlan *plan) {
    if (!expert_coverage_ledger_valid(ledger) || !plan) return -1;
    return salt_dpr_attention_plan_compile_expert_coverage(
        plan, ledger->route_counts, ledger->layer_count, ledger->expert_count,
        maximum_per_layer, maximum_items);
}

int salt_dpr_expert_coverage_ledger_sha256(
        const SaltDprExpertCoverageLedger *ledger, uint8_t out[32]) {
    static const uint8_t domain[] = "salt-exact-route-ledger-v1";
    SaltSha256 hasher;
    uint8_t scalar[8];
    size_t count;
    if (!expert_coverage_ledger_valid(ledger) || !out ||
        ledger->layer_count > SIZE_MAX / ledger->expert_count)
        return -1;
    count = (size_t)ledger->layer_count * ledger->expert_count;
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, domain, sizeof domain - 1u);
    le32_store(scalar, ledger->layer_count);
    le32_store(scalar + 4, ledger->expert_count);
    salt_sha256_update(&hasher, scalar, sizeof scalar);
    le64_store(scalar, ledger->observation_count);
    salt_sha256_update(&hasher, scalar, sizeof scalar);
    for (size_t i = 0; i < count; i++) {
        le64_store(scalar, ledger->route_counts[i]);
        salt_sha256_update(&hasher, scalar, sizeof scalar);
    }
    salt_sha256_final(&hasher, out);
    return 0;
}

int salt_dpr_attention_plan_finalize_runtime(
        SaltDprAttentionPlan *plan, const SaltDprComputeIntent *intent,
        const uint8_t mindset_sha256[32],
        const uint8_t provenance_sha256[32]) {
    static const uint8_t domain[] = "salt-runtime-expert-coverage-v1";
    SaltSha256 hasher;
    if (!plan || compute_intent_validate(intent) != 0 || !mindset_sha256 ||
        !provenance_sha256 || digest_is_zero(mindset_sha256) ||
        digest_is_zero(provenance_sha256) || plan->item_count == 0 ||
        plan->expected_saved_numerator == 0 ||
        plan->expected_saved_denominator == 0)
        return -1;
    plan->schema_version = SALT_DPR_ATTENTION_PLAN_VERSION;
    plan->exactness = SALT_DPR_EXACTNESS_A0;
    memcpy(plan->mindset_sha256, mindset_sha256, 32);
    memcpy(plan->provenance_sha256, provenance_sha256, 32);
    if (salt_dpr_compute_intent_sha256(intent, plan->intent_sha256) != 0)
        return -1;
    memcpy(plan->state_compatibility_sha256,
           intent->state_compatibility_sha256, 32);
    memcpy(plan->policy_sha256, intent->active_policy_sha256, 32);
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, domain, sizeof domain - 1u);
    salt_sha256_update(&hasher, provenance_sha256, 32);
    salt_sha256_update(&hasher, plan->intent_sha256, 32);
    salt_sha256_final(&hasher, plan->dpr_sha256);
    plan->provenance_authenticated = 1;
    plan->policy_allowed = 1;
    return salt_dpr_attention_plan_validate(
        plan, intent, UINT32_MAX, UINT32_MAX, UINT64_MAX) == 1 ? 0 : -1;
}

int salt_dpr_operation_plan_compose(
        const SaltDprComputeIntent *intent,
        const SaltDprTransitionProposal *transition,
        const SaltDprAttentionPlan *attention,
        uint32_t max_layers, uint32_t max_experts, uint64_t max_context,
        int32_t vocab_size, SaltDprOperationPlan *plan) {
    int nd = 0, nm = 0;
    if (!plan || compute_intent_validate(intent) != 0) return -1;
    memset(plan, 0, sizeof *plan);
    if (transition) {
        nd = salt_dpr_transition_proposal_validate(
            transition, intent, vocab_size);
        if (nd < 0) return -1;
    }
    if (attention) {
        nm = salt_dpr_attention_plan_validate(
            attention, intent, max_layers, max_experts, max_context);
        if (nm < 0) return -1;
    }
    plan->has_transition = nd == 1;
    plan->has_attention = nm == 1;
    plan->cell = plan->has_transition
        ? (plan->has_attention ? SALT_DPR_CELL_ND_NM : SALT_DPR_CELL_ND_ONLY)
        : (plan->has_attention ? SALT_DPR_CELL_NM_ONLY : SALT_DPR_CELL_NATIVE);
    if (plan->has_transition)
        memcpy(plan->transition_sha256, transition->proposal_sha256, 32);
    if (plan->has_attention) {
        memcpy(plan->attention_dpr_sha256, attention->dpr_sha256, 32);
        memcpy(plan->mindset_sha256, attention->mindset_sha256, 32);
    }
    return 0;
}

int salt_dpr_memory_admit(const SaltDprMemoryRequest *request,
                          SaltDprMemoryDecision *decision) {
    uint64_t current_required, additive_required, owned;
    if (!request || !decision ||
        (request->mode != SALT_DPR_OFF &&
         request->mode != SALT_DPR_PERSIST &&
         request->mode != SALT_DPR_DYNAMIC))
        return -1;
    memset(decision, 0, sizeof *decision);
    decision->verification_scratch_bytes = request->verification_scratch_bytes;
    decision->checkpoint_max_bytes = request->checkpoint_max_bytes;
    if (request->mode == SALT_DPR_OFF) {
        decision->reason = SALT_DPR_MEMORY_DISABLED;
        return 0;
    }
    if (request->current_kv_budget_bytes == 0 ||
        request->target_required_bytes == 0)
        return -1;
    if (request->mode == SALT_DPR_PERSIST) {
        current_required = request->target_required_bytes;
        additive_required = request->reference_required_bytes;
    } else {
        if (request->additive_dpr_budget_bytes != 0) return -1;
        current_required = request->target_required_bytes >
                           request->reference_required_bytes
            ? request->target_required_bytes : request->reference_required_bytes;
        additive_required = 0;
    }
    decision->current_kv_required_bytes = current_required;
    decision->additive_dpr_required_bytes = additive_required;
    if (add_u64(current_required, additive_required, &owned) != 0 ||
        add_u64(owned, request->verification_scratch_bytes, &owned) != 0) {
        decision->reason = SALT_DPR_MEMORY_OVERFLOW;
        return 0;
    }
    decision->dpr_owned_bytes = owned;
    if (current_required > request->current_kv_budget_bytes) {
        decision->reason = SALT_DPR_MEMORY_CURRENT_BUDGET;
        return 0;
    }
    if (request->mode == SALT_DPR_PERSIST &&
        additive_required > request->additive_dpr_budget_bytes) {
        decision->reason = SALT_DPR_MEMORY_ADDITIVE_BUDGET;
        return 0;
    }
    decision->admitted = 1;
    decision->reason = SALT_DPR_MEMORY_NONE;
    return 0;
}

static int dpr_pending_empty(const SaltDprRuntime *runtime) {
    return runtime && !runtime->pending_valid &&
           runtime->pending_token_id == -1 && runtime->pending_position == 0;
}

int salt_dpr_runtime_invariant(const SaltDprRuntime *runtime) {
    if (!runtime || runtime->epoch == 0) return -1;
    switch (runtime->phase) {
    case SALT_DPR_PHASE_OFF:
        return runtime->mode == SALT_DPR_OFF && !runtime->target_live &&
               !runtime->reference_live && !runtime->checkpoint_set &&
               dpr_pending_empty(runtime) ? 0 : -1;
    case SALT_DPR_PHASE_TARGET:
        return runtime->mode != SALT_DPR_OFF && runtime->target_live &&
               !runtime->reference_live && !runtime->checkpoint_set &&
               dpr_pending_empty(runtime) ? 0 : -1;
    case SALT_DPR_PHASE_EMPTY:
        return runtime->mode == SALT_DPR_DYNAMIC && !runtime->target_live &&
               !runtime->reference_live && runtime->checkpoint_set &&
               dpr_pending_empty(runtime) ? 0 : -1;
    case SALT_DPR_PHASE_REFERENCE:
        if (runtime->mode == SALT_DPR_PERSIST)
            return runtime->target_live && runtime->reference_live &&
                   !runtime->checkpoint_set && dpr_pending_empty(runtime)
                ? 0 : -1;
        if (runtime->mode == SALT_DPR_DYNAMIC)
            return !runtime->target_live && runtime->reference_live &&
                   runtime->checkpoint_set && dpr_pending_empty(runtime)
                ? 0 : -1;
        return -1;
    case SALT_DPR_PHASE_VERIFYING:
        return runtime->mode != SALT_DPR_OFF && runtime->target_live &&
               !runtime->reference_live && !runtime->checkpoint_set &&
               dpr_pending_empty(runtime) ? 0 : -1;
    case SALT_DPR_PHASE_PENDING:
        return runtime->mode != SALT_DPR_OFF && runtime->target_live &&
               !runtime->reference_live && !runtime->checkpoint_set &&
               runtime->pending_valid && runtime->pending_token_id >= 0 &&
               runtime->pending_position < UINT64_MAX ? 0 : -1;
    case SALT_DPR_PHASE_DEAD:
        return runtime->mode != SALT_DPR_OFF && !runtime->target_live &&
               !runtime->reference_live && !runtime->checkpoint_set &&
               dpr_pending_empty(runtime) ? 0 : -1;
    default:
        return -1;
    }
}

int salt_dpr_runtime_init(SaltDprRuntime *runtime, SaltDprMode mode) {
    if (!runtime || (mode != SALT_DPR_OFF &&
                     mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    memset(runtime, 0, sizeof *runtime);
    runtime->mode = mode;
    runtime->phase = mode == SALT_DPR_OFF
        ? SALT_DPR_PHASE_OFF : SALT_DPR_PHASE_TARGET;
    runtime->target_live = mode == SALT_DPR_OFF ? 0 : 1;
    runtime->pending_token_id = -1;
    runtime->epoch = 1;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_target_unload(SaltDprRuntime *runtime,
                           const uint8_t checkpoint_sha256[32]) {
    if (!runtime || !checkpoint_sha256 ||
        runtime->mode != SALT_DPR_DYNAMIC ||
        runtime->phase != SALT_DPR_PHASE_TARGET ||
        runtime->epoch == UINT64_MAX ||
        digest_is_zero(checkpoint_sha256))
        return -1;
    runtime->target_live = 0;
    runtime->checkpoint_set = 1;
    memcpy(runtime->checkpoint_sha256, checkpoint_sha256, 32);
    runtime->phase = SALT_DPR_PHASE_EMPTY;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_reference_acquire(SaltDprRuntime *runtime) {
    if (!runtime || runtime->epoch == UINT64_MAX) return -1;
    if (runtime->mode == SALT_DPR_PERSIST &&
        runtime->phase == SALT_DPR_PHASE_TARGET) {
        runtime->reference_live = 1;
        runtime->phase = SALT_DPR_PHASE_REFERENCE;
    } else if (runtime->mode == SALT_DPR_DYNAMIC &&
               runtime->phase == SALT_DPR_PHASE_EMPTY) {
        runtime->reference_live = 1;
        runtime->phase = SALT_DPR_PHASE_REFERENCE;
    } else {
        return -1;
    }
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_reference_release(SaltDprRuntime *runtime) {
    if (!runtime || runtime->phase != SALT_DPR_PHASE_REFERENCE ||
        runtime->epoch == UINT64_MAX)
        return -1;
    runtime->reference_live = 0;
    runtime->phase = runtime->mode == SALT_DPR_PERSIST
        ? SALT_DPR_PHASE_TARGET : SALT_DPR_PHASE_EMPTY;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_target_restore(SaltDprRuntime *runtime,
                            const uint8_t checkpoint_sha256[32]) {
    if (!runtime || !checkpoint_sha256 ||
        runtime->mode != SALT_DPR_DYNAMIC ||
        runtime->phase != SALT_DPR_PHASE_EMPTY ||
        runtime->epoch == UINT64_MAX ||
        !runtime->checkpoint_set ||
        memcmp(runtime->checkpoint_sha256, checkpoint_sha256, 32) != 0)
        return -1;
    runtime->target_live = 1;
    runtime->checkpoint_set = 0;
    memset(runtime->checkpoint_sha256, 0, 32);
    runtime->phase = SALT_DPR_PHASE_TARGET;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_verify_begin(SaltDprRuntime *runtime) {
    if (!runtime || runtime->phase != SALT_DPR_PHASE_TARGET ||
        !runtime->target_live || runtime->reference_live ||
        runtime->epoch == UINT64_MAX)
        return -1;
    runtime->phase = SALT_DPR_PHASE_VERIFYING;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_verify_finish_pending(SaltDprRuntime *runtime,
                                   int32_t token_id, uint64_t position) {
    if (!runtime || runtime->phase != SALT_DPR_PHASE_VERIFYING ||
        runtime->epoch == UINT64_MAX || token_id < 0 || position == UINT64_MAX ||
        !dpr_pending_empty(runtime))
        return -1;
    runtime->pending_valid = 1;
    runtime->pending_token_id = token_id;
    runtime->pending_position = position;
    runtime->phase = SALT_DPR_PHASE_PENDING;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_pending_read(const SaltDprRuntime *runtime,
                          int32_t *token_id, uint64_t *position) {
    if (!runtime || !token_id || !position ||
        salt_dpr_runtime_invariant(runtime) != 0)
        return -1;
    if (runtime->phase != SALT_DPR_PHASE_PENDING) return 0;
    *token_id = runtime->pending_token_id;
    *position = runtime->pending_position;
    return 1;
}

int salt_dpr_pending_commit(SaltDprRuntime *runtime,
                            int32_t token_id, uint64_t resulting_position) {
    if (!runtime || runtime->phase != SALT_DPR_PHASE_PENDING ||
        runtime->epoch == UINT64_MAX || !runtime->pending_valid ||
        token_id != runtime->pending_token_id ||
        runtime->pending_position == UINT64_MAX ||
        resulting_position != runtime->pending_position + 1u)
        return -1;
    runtime->pending_valid = 0;
    runtime->pending_token_id = -1;
    runtime->pending_position = 0;
    runtime->phase = SALT_DPR_PHASE_TARGET;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}

int salt_dpr_runtime_fail(SaltDprRuntime *runtime) {
    if (!runtime || runtime->mode == SALT_DPR_OFF ||
        runtime->phase == SALT_DPR_PHASE_DEAD || runtime->epoch == UINT64_MAX)
        return -1;
    runtime->phase = SALT_DPR_PHASE_DEAD;
    runtime->target_live = 0;
    runtime->reference_live = 0;
    runtime->checkpoint_set = 0;
    memset(runtime->checkpoint_sha256, 0, 32);
    runtime->pending_valid = 0;
    runtime->pending_token_id = -1;
    runtime->pending_position = 0;
    runtime->epoch++;
    return salt_dpr_runtime_invariant(runtime);
}
