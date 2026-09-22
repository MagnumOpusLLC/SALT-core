#include "salt/dpr.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static void fill(uint8_t out[32], uint8_t seed) {
    for (size_t i = 0; i < 32; i++) out[i] = (uint8_t)(seed + i);
}

static int stop_token(void *opaque, int32_t token) {
    (void)opaque;
    return token == 1 || token == 50 || token == 106;
}

static SaltDprChart chart(uint32_t horizon, uint64_t verify_ns,
                          uint64_t accepted, uint64_t rounds) {
    SaltDprChart value;
    memset(&value, 0, sizeof value);
    value.horizon = horizon;
    value.verify_ns = verify_ns;
    value.accepted_tokens_total = accepted;
    value.rounds = rounds;
    value.sample_count = rounds;
    return value;
}

static SaltDprEdge edge(const uint8_t node[32], SaltDprMode mode,
                        uint32_t horizon, uint64_t verify_ns,
                        uint64_t accepted) {
    SaltDprEdge value;
    memset(&value, 0, sizeof value);
    value.schema_version = SALT_DPR_EDGE_VERSION;
    value.mode = mode;
    value.horizon = horizon;
    value.candidate_count = horizon;
    memcpy(value.source_node_sha256, node, 32);
    fill(value.reference_sha256, 0x80);
    value.reference_bytes = 4096;
    value.reference_position = 40 + horizon;
    value.kind = SALT_DPR_EDGE_PARENT_EXACT;
    value.bonus_token_id = 300;
    for (uint32_t i = 0; i < horizon; i++)
        value.candidate_token_ids[i] = (int32_t)(200 + i);
    value.chart = chart(horizon, verify_ns, accepted, 3);
    return value;
}

static SaltDprMentorFamily mentor_family(
        uint8_t seed, const uint8_t compatibility[32],
        const uint8_t policy[32]) {
    SaltDprMentorFamily value;
    memset(&value, 0, sizeof value);
    value.schema_version = SALT_DPR_MENTOR_FAMILY_VERSION;
    fill(value.family_sha256, seed);
    fill(value.mindset_sha256, (uint8_t)(seed + 1u));
    memcpy(value.state_compatibility_sha256, compatibility, 32);
    memcpy(value.policy_sha256, policy, 32);
    fill(value.nomogram_root_sha256, (uint8_t)(seed + 2u));
    fill(value.provenance_sha256, (uint8_t)(seed + 3u));
    return value;
}

static SaltDprSelectorCandidate selector_candidate(
        SaltDprSelectorKind kind, size_t family_index, uint8_t edge_seed,
        const uint8_t source_node[32],
        const uint8_t compatibility[32], const uint8_t policy[32],
        uint64_t saved_numerator, uint64_t saved_denominator) {
    SaltDprSelectorCandidate value;
    memset(&value, 0, sizeof value);
    value.schema_version = SALT_DPR_SELECTOR_CANDIDATE_VERSION;
    value.kind = kind;
    value.family_index = family_index;
    fill(value.edge_sha256, edge_seed);
    if (kind == SALT_DPR_SELECTOR_MENTOR_NOMOGRAM)
        fill(value.binding_sha256, (uint8_t)(edge_seed + 1u));
    memcpy(value.source_node_sha256, source_node, 32);
    memcpy(value.state_compatibility_sha256, compatibility, 32);
    memcpy(value.policy_sha256, policy, 32);
    value.expected_saved_numerator = saved_numerator;
    value.expected_saved_denominator = saved_denominator;
    value.provenance_authenticated = 1;
    value.policy_allowed = 1;
    return value;
}

static void test_node(void) {
    SaltDprNodeMaterial material;
    uint8_t a[32], b[32];
    memset(&material, 0, sizeof material);
    fill(material.state_compatibility_sha256, 1);
    fill(material.state_sha256, 2);
    fill(material.sampler_config_sha256, 3);
    material.sampler_abi = SALT_DPR_SAMPLER_GREEDY_V1;
    material.position = 0;
    assert(salt_dpr_node_sha256(&material, a) == 0);
    assert(salt_dpr_node_sha256(&material, b) == 0);
    assert(memcmp(a, b, 32) == 0);
    assert(memcmp(a, (uint8_t[32]){0}, 32) != 0);
    material.position = 40;
    assert(salt_dpr_node_sha256(&material, b) == 0);
    assert(memcmp(a, b, 32) != 0);
    material.position = 0;
    material.sampler_abi = SALT_DPR_SAMPLER_COUNTER_V1;
    fill(material.target_rng_sha256, 4);
    assert(salt_dpr_node_sha256(&material, b) == 0);
    assert(memcmp(a, b, 32) != 0);
    memset(material.target_rng_sha256, 0, 32);
    assert(salt_dpr_node_sha256(&material, b) != 0);
}

static void test_chart(void) {
    SaltDprChart value = chart(4, 400, 12, 3);
    uint64_t numerator = 0, denominator = 0;
    uint64_t saved_numerator = 0, saved_denominator = 0;
    assert(salt_dpr_chart_validate(&value) == 0);
    assert(salt_dpr_chart_fraction(&value, &numerator, &denominator) == 0);
    assert(numerator == 1200 && denominator == 15);
    assert(salt_dpr_chart_expected_saved(
        &value, 100, &saved_numerator, &saved_denominator) == 0);
    assert(saved_numerator == 300 && saved_denominator == 15);
    assert(salt_dpr_chart_expected_saved(
        &value, 80, &saved_numerator, &saved_denominator) == 0);
    assert(saved_numerator == 0 && saved_denominator == 15);
    assert(salt_dpr_chart_expected_saved(
        &value, UINT64_MAX, &saved_numerator, &saved_denominator) != 0);
    assert(salt_dpr_chart_is_green(&value, 100, 3) == 1);
    assert(salt_dpr_chart_is_green(&value, 80, 3) == 0);
    value.accepted_tokens_total = 13;
    assert(salt_dpr_chart_validate(&value) != 0);
    value = chart(4, 1, 0, 3);
    assert(salt_dpr_chart_validate(&value) == 0);
    value = chart(4, UINT64_MAX, 12, 3);
    value.lookup_ns = 1;
    assert(salt_dpr_chart_validate(&value) != 0);
}

static void test_edge_selection(void) {
    SaltDprNodeMaterial material;
    SaltDprEdge edges[4];
    uint8_t node[32], other[32];
    size_t selected = SIZE_MAX;
    memset(&material, 0, sizeof material);
    fill(material.state_compatibility_sha256, 1);
    fill(material.state_sha256, 2);
    fill(material.sampler_config_sha256, 3);
    material.sampler_abi = SALT_DPR_SAMPLER_GREEDY_V1;
    material.position = 40;
    assert(salt_dpr_node_sha256(&material, node) == 0);
    material.position = 41;
    assert(salt_dpr_node_sha256(&material, other) == 0);

    edges[0] = edge(node, SALT_DPR_PERSIST, 4, 450, 12);
    edges[1] = edge(node, SALT_DPR_PERSIST, 2, 120, 6);
    edges[2] = edge(other, SALT_DPR_PERSIST, 4, 1, 12);
    edges[3] = edge(node, SALT_DPR_DYNAMIC, 4, 1, 12);
    assert(salt_dpr_edge_validate(
        &edges[0], 262144, 512, stop_token, NULL) == 0);
    assert(salt_dpr_select_edge(
        edges, 4, node, SALT_DPR_PERSIST, 100, 3,
        262144, 512, stop_token, NULL, &selected) == 1);
    assert(selected == 1);
    assert(salt_dpr_select_edge(
        edges, 4, node, SALT_DPR_OFF, 100, 3,
        262144, 512, stop_token, NULL, &selected) == 0);
    edges[1].candidate_token_ids[0] = 106;
    assert(salt_dpr_edge_validate(
        &edges[1], 262144, 512, stop_token, NULL) != 0);
    edges[1].candidate_token_ids[0] = 262144;
    assert(salt_dpr_edge_validate(
        &edges[1], 262144, 512, stop_token, NULL) != 0);
}

static void test_mentor_selector(void) {
    SaltDprMentorFamily families[2], invalid_family;
    SaltDprSelectorCandidate candidates[4], reversed[3], rejected[1];
    SaltDprSelectorCandidate parents[2];
    SaltDprSelectorResult result, repeated;
    uint8_t target_node[32], other_node[32];
    uint8_t compatibility[32], other_compatibility[32], policy[32];

    fill(target_node, 7);
    memcpy(other_node, target_node, 32);
    other_node[0] ^= 1u;
    fill(compatibility, 11);
    memcpy(other_compatibility, compatibility, 32);
    other_compatibility[0] ^= 1u;
    fill(policy, 91);
    families[0] = mentor_family(0x40, compatibility, policy);
    families[1] = mentor_family(0x20, compatibility, policy);
    assert(salt_dpr_mentor_family_validate(&families[0]) == 0);
    assert(salt_dpr_mentor_family_validate(&families[1]) == 0);

    candidates[0] = selector_candidate(
        SALT_DPR_SELECTOR_MENTOR_NOMOGRAM, 0, 0x60,
        target_node, compatibility, policy, 100, 1);
    candidates[1] = selector_candidate(
        SALT_DPR_SELECTOR_PARENT_EXACT, SIZE_MAX, 0x70,
        target_node, compatibility, policy, 1, 1);
    candidates[2] = selector_candidate(
        SALT_DPR_SELECTOR_MENTOR_NOMOGRAM, 1, 0x50,
        target_node, compatibility, policy, 5, 1);
    candidates[3] = selector_candidate(
        SALT_DPR_SELECTOR_MENTOR_NOMOGRAM, 1, 0x30,
        target_node, compatibility, policy, 5, 1);

    /* Parent authority is lexicographic, not a flat cost race. */
    assert(salt_dpr_selector_choose(
        families, 2, candidates, 4, target_node,
        compatibility, policy, &result) == 1);
    assert(result.kind == SALT_DPR_SELECTOR_PARENT_EXACT);
    assert(result.candidate_index == 1 && result.family_index == SIZE_MAX);
    assert(result.expected_saved_numerator == 1 &&
           result.expected_saved_denominator == 1);
    assert(memcmp(result.edge_sha256, candidates[1].edge_sha256, 32) == 0);
    assert(memcmp(result.source_node_sha256, target_node, 32) == 0);
    assert(memcmp(result.state_compatibility_sha256, compatibility, 32) == 0);
    assert(memcmp(result.policy_sha256, policy, 32) == 0);
    assert(memcmp(result.binding_sha256, (uint8_t[32]){0}, 32) == 0);
    assert(memcmp(result.family_sha256, (uint8_t[32]){0}, 32) == 0);

    /* A zero-saving parent is rejected; family then edge digest break ties. */
    candidates[1].expected_saved_numerator = 0;
    candidates[0].expected_saved_numerator = 5;
    assert(salt_dpr_selector_choose(
        families, 2, candidates, 4, target_node,
        compatibility, policy, &result) == 1);
    assert(result.kind == SALT_DPR_SELECTOR_MENTOR_NOMOGRAM);
    assert(result.candidate_index == 3 && result.family_index == 1);
    assert(result.expected_saved_numerator == 5 &&
           result.expected_saved_denominator == 1);
    assert(memcmp(result.family_sha256, families[1].family_sha256, 32) == 0);
    assert(memcmp(result.mindset_sha256, families[1].mindset_sha256, 32) == 0);
    assert(memcmp(result.provenance_sha256,
                  families[1].provenance_sha256, 32) == 0);
    assert(memcmp(result.nomogram_root_sha256,
                  families[1].nomogram_root_sha256, 32) == 0);
    assert(memcmp(result.edge_sha256, candidates[3].edge_sha256, 32) == 0);
    assert(memcmp(result.binding_sha256,
                  candidates[3].binding_sha256, 32) == 0);

    reversed[0] = candidates[2];
    reversed[1] = candidates[0];
    reversed[2] = candidates[3];
    assert(salt_dpr_selector_choose(
        families, 2, reversed, 3, target_node,
        compatibility, policy, &repeated) == 1);
    assert(repeated.kind == SALT_DPR_SELECTOR_MENTOR_NOMOGRAM);
    assert(memcmp(repeated.family_sha256, result.family_sha256, 32) == 0);
    assert(memcmp(repeated.edge_sha256, result.edge_sha256, 32) == 0);

    /* Equal parent savings resolve by edge digest, independent of order. */
    parents[0] = selector_candidate(
        SALT_DPR_SELECTOR_PARENT_EXACT, SIZE_MAX, 0x70,
        target_node, compatibility, policy, 4, 1);
    parents[1] = selector_candidate(
        SALT_DPR_SELECTOR_PARENT_EXACT, SIZE_MAX, 0x60,
        target_node, compatibility, policy, 4, 1);
    assert(salt_dpr_selector_choose(
        NULL, 0, parents, 2, target_node,
        compatibility, policy, &result) == 1);
    assert(result.candidate_index == 1);
    assert(memcmp(result.edge_sha256, parents[1].edge_sha256, 32) == 0);

    /* Compatibility, authentication, policy, and positive saving all gate. */
    rejected[0] = selector_candidate(
        SALT_DPR_SELECTOR_MENTOR_NOMOGRAM, 0, 1,
        target_node, compatibility, policy, 1, 1);
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, other_node,
        compatibility, policy, &result) == 0);
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        other_compatibility, policy, &result) == 0);
    rejected[0].provenance_authenticated = 0;
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        compatibility, policy, &result) == 0);
    rejected[0].provenance_authenticated = 1;
    rejected[0].policy_allowed = 0;
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        compatibility, policy, &result) == 0);
    rejected[0].policy_allowed = 1;
    rejected[0].expected_saved_numerator = 0;
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        compatibility, policy, &result) == 0);
    assert(result.kind == SALT_DPR_SELECTOR_NATIVE_FALLBACK);
    assert(result.candidate_index == SIZE_MAX && result.family_index == SIZE_MAX);

    rejected[0] = selector_candidate(
        SALT_DPR_SELECTOR_MENTOR_NOMOGRAM, 0, 1,
        target_node, compatibility, policy, 1, 0);
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        compatibility, policy, &result) != 0);
    rejected[0].expected_saved_denominator = 1;
    rejected[0].provenance_authenticated = 2;
    assert(salt_dpr_selector_choose(
        families, 2, rejected, 1, target_node,
        compatibility, policy, &result) != 0);

    invalid_family = families[0];
    memset(invalid_family.mindset_sha256, 0, 32);
    assert(salt_dpr_mentor_family_validate(&invalid_family) != 0);
}

static void test_memory(void) {
    SaltDprMemoryRequest request;
    SaltDprMemoryDecision decision;
    memset(&request, 0, sizeof request);
    request.mode = SALT_DPR_PERSIST;
    request.current_kv_budget_bytes = 1000;
    request.additive_dpr_budget_bytes = 500;
    request.target_required_bytes = 900;
    request.reference_required_bytes = 400;
    request.verification_scratch_bytes = 100;
    request.checkpoint_max_bytes = 0;
    assert(salt_dpr_memory_admit(&request, &decision) == 0);
    assert(decision.admitted == 1);
    assert(decision.current_kv_required_bytes == 900);
    assert(decision.additive_dpr_required_bytes == 400);
    assert(decision.dpr_owned_bytes == 1400);
    request.additive_dpr_budget_bytes = 399;
    assert(salt_dpr_memory_admit(&request, &decision) == 0);
    assert(decision.admitted == 0);
    assert(decision.reason == SALT_DPR_MEMORY_ADDITIVE_BUDGET);

    memset(&request, 0, sizeof request);
    request.mode = SALT_DPR_DYNAMIC;
    request.current_kv_budget_bytes = 1000;
    request.target_required_bytes = 900;
    request.reference_required_bytes = 950;
    request.verification_scratch_bytes = 100;
    request.checkpoint_max_bytes = 1000;
    assert(salt_dpr_memory_admit(&request, &decision) == 0);
    assert(decision.admitted == 1);
    assert(decision.current_kv_required_bytes == 950);
    assert(decision.additive_dpr_required_bytes == 0);
    request.reference_required_bytes = 1001;
    assert(salt_dpr_memory_admit(&request, &decision) == 0);
    assert(decision.admitted == 0);
    assert(decision.reason == SALT_DPR_MEMORY_CURRENT_BUDGET);
    request.reference_required_bytes = 900;
    request.additive_dpr_budget_bytes = 1;
    assert(salt_dpr_memory_admit(&request, &decision) != 0);
}

static void test_lifecycle(void) {
    SaltDprRuntime runtime;
    uint8_t checkpoint[32], wrong[32];
    fill(checkpoint, 9);
    memcpy(wrong, checkpoint, 32);
    wrong[0] ^= 1u;

    assert(salt_dpr_runtime_init(&runtime, SALT_DPR_DYNAMIC) == 0);
    assert(salt_dpr_runtime_invariant(&runtime) == 0);
    assert(runtime.phase == SALT_DPR_PHASE_TARGET);
    assert(salt_dpr_reference_acquire(&runtime) != 0);
    assert(salt_dpr_target_unload(&runtime, checkpoint) == 0);
    assert(runtime.phase == SALT_DPR_PHASE_EMPTY);
    assert(salt_dpr_reference_acquire(&runtime) == 0);
    assert(runtime.phase == SALT_DPR_PHASE_REFERENCE);
    assert(salt_dpr_target_restore(&runtime, checkpoint) != 0);
    assert(salt_dpr_reference_release(&runtime) == 0);
    assert(salt_dpr_target_restore(&runtime, wrong) != 0);
    assert(salt_dpr_target_restore(&runtime, checkpoint) == 0);
    assert(salt_dpr_verify_begin(&runtime) == 0);
    assert(salt_dpr_verify_finish_pending(&runtime, 42, 17) == 0);
    assert(runtime.phase == SALT_DPR_PHASE_PENDING);
    assert(runtime.epoch == 7);
    {
        int32_t token = -1;
        uint64_t position = 0;
        assert(salt_dpr_pending_read(&runtime, &token, &position) == 1);
        assert(token == 42 && position == 17);
        assert(salt_dpr_verify_begin(&runtime) != 0);
        assert(salt_dpr_pending_commit(&runtime, 43, 18) != 0);
        assert(salt_dpr_pending_commit(&runtime, 42, 17) != 0);
        assert(salt_dpr_pending_commit(&runtime, 42, 18) == 0);
        assert(runtime.phase == SALT_DPR_PHASE_TARGET);
        assert(runtime.epoch == 8);
        assert(salt_dpr_pending_read(&runtime, &token, &position) == 0);
    }

    assert(salt_dpr_runtime_init(&runtime, SALT_DPR_PERSIST) == 0);
    assert(salt_dpr_reference_acquire(&runtime) == 0);
    assert(runtime.target_live == 1 && runtime.reference_live == 1);
    assert(salt_dpr_verify_begin(&runtime) != 0);
    assert(salt_dpr_reference_release(&runtime) == 0);
    assert(salt_dpr_verify_begin(&runtime) == 0);
    assert(salt_dpr_runtime_fail(&runtime) == 0);
    assert(runtime.phase == SALT_DPR_PHASE_DEAD);
    assert(salt_dpr_verify_finish_pending(&runtime, 42, 17) != 0);
    assert(salt_dpr_runtime_invariant(&runtime) == 0);

    assert(salt_dpr_runtime_init(&runtime, SALT_DPR_PERSIST) == 0);
    runtime.epoch = UINT64_MAX;
    assert(salt_dpr_reference_acquire(&runtime) != 0);
    assert(runtime.phase == SALT_DPR_PHASE_TARGET);
}

static void test_nomogram_qa_key(void) {
    uint8_t compatibility[32], other_compatibility[32];
    uint8_t known[32], fresh[32], other[32], incompatible[32];
    int32_t known_tokens[64], fresh_tokens[64], other_tokens[64];
    fill(compatibility, 11);
    memcpy(other_compatibility, compatibility, 32);
    other_compatibility[0] ^= 1u;
    for (int i = 0; i < 64; i++) {
        known_tokens[i] = 100 + (i % 13);
        fresh_tokens[i] = known_tokens[i];
        other_tokens[i] = 1000 + i;
    }
    fresh_tokens[63] = 777;
    assert(salt_dpr_nomogram_qa_key(
        compatibility, known_tokens, 64, 12, known) == 0);
    assert(salt_dpr_nomogram_qa_key(
        compatibility, fresh_tokens, 64, 12, fresh) == 0);
    assert(memcmp(known, fresh, 32) == 0);
    assert(salt_dpr_nomogram_qa_key(
        compatibility, other_tokens, 64, 12, other) == 0);
    assert(memcmp(known, other, 32) != 0);
    assert(salt_dpr_nomogram_qa_key(
        other_compatibility, known_tokens, 64, 12, incompatible) == 0);
    assert(memcmp(known, incompatible, 32) != 0);
    assert(salt_dpr_nomogram_qa_key(
        compatibility, known_tokens, 64, 0, other) != 0);
    assert(salt_dpr_nomogram_qa_key(
        compatibility, known_tokens, 64, 33, other) != 0);
}

static void test_transition_chain(void) {
    uint8_t compatibility[32], sampler[32], other_compatibility[32];
    uint8_t root_a[32], root_b[32], next_a[32], next_b[32], different[32];
    fill(compatibility, 31);
    fill(sampler, 63);
    memcpy(other_compatibility, compatibility, 32);
    other_compatibility[0] ^= 1u;
    assert(salt_dpr_transition_chain_init(
        compatibility, sampler, SALT_DPR_SAMPLER_GREEDY_V1, 0, root_a) == 0);
    assert(salt_dpr_transition_chain_init(
        compatibility, sampler, SALT_DPR_SAMPLER_GREEDY_V1, 0, root_b) == 0);
    assert(memcmp(root_a, root_b, 32) == 0);
    assert(salt_dpr_transition_chain_advance(root_a, 42, 1, next_a) == 0);
    assert(salt_dpr_transition_chain_advance(root_b, 42, 1, next_b) == 0);
    assert(memcmp(next_a, next_b, 32) == 0);
    assert(salt_dpr_transition_chain_advance(root_b, 43, 1, different) == 0);
    assert(memcmp(next_a, different, 32) != 0);
    assert(salt_dpr_transition_chain_init(
        other_compatibility, sampler, SALT_DPR_SAMPLER_GREEDY_V1,
        0, different) == 0);
    assert(memcmp(root_a, different, 32) != 0);
    assert(salt_dpr_transition_chain_advance(root_a, 42, 0, different) != 0);
}

int main(void) {
    test_node();
    test_chart();
    test_edge_selection();
    test_mentor_selector();
    test_memory();
    test_lifecycle();
    test_nomogram_qa_key();
    test_transition_chain();
    puts("dpr: PASS");
    return 0;
}
