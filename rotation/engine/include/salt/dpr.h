#ifndef SALT_DPR_H
#define SALT_DPR_H

#include <stddef.h>
#include <stdint.h>

#define SALT_DPR_EDGE_VERSION 2u
#define SALT_DPR_MAX_HORIZON 64u
#define SALT_DPR_MAX_WALK_HORIZON 356u
#define SALT_DPR_MENTOR_FAMILY_VERSION 1u
#define SALT_DPR_SELECTOR_CANDIDATE_VERSION 1u
#define SALT_DPR_COMPUTE_INTENT_VERSION 1u
#define SALT_DPR_TRANSITION_PROPOSAL_VERSION 4u
#define SALT_DPR_ATTENTION_PLAN_VERSION 1u
#define SALT_DPR_MAX_RELEVANCE_ITEMS 64u

typedef enum SaltDprMode {
    SALT_DPR_OFF = 0,
    SALT_DPR_PERSIST = 1,
    SALT_DPR_DYNAMIC = 2
} SaltDprMode;

typedef enum SaltDprEdgeKind {
    SALT_DPR_EDGE_INVALID = 0,
    SALT_DPR_EDGE_PARENT_EXACT = 1,
    SALT_DPR_EDGE_NOMOGRAM_DRAFT = 2,
    SALT_DPR_EDGE_PARENT_PREFILL = 3,
    SALT_DPR_EDGE_NOMOGRAM_PREFILL = 4
} SaltDprEdgeKind;

typedef enum SaltDprSamplerAbi {
    SALT_DPR_SAMPLER_INVALID = 0,
    SALT_DPR_SAMPLER_GREEDY_V1 = 1,
    SALT_DPR_SAMPLER_COUNTER_V1 = 2
} SaltDprSamplerAbi;

typedef struct SaltDprNodeMaterial {
    uint8_t state_compatibility_sha256[32];
    uint8_t state_sha256[32];
    uint8_t sampler_config_sha256[32];
    uint8_t target_rng_sha256[32];
    uint32_t sampler_abi;
    uint64_t position;
} SaltDprNodeMaterial;

typedef struct SaltDprChart {
    uint32_t horizon;
    uint64_t lookup_ns;
    uint64_t recover_ns;
    uint64_t draft_ns;
    uint64_t verify_ns;
    uint64_t commit_ns;
    uint64_t accepted_tokens_total;
    uint64_t rounds;
    uint64_t sample_count;
} SaltDprChart;

typedef struct SaltDprEdge {
    uint32_t schema_version;
    SaltDprMode mode;
    uint32_t horizon;
    uint32_t candidate_count;
    uint8_t source_node_sha256[32];
    uint8_t reference_sha256[32];
    uint64_t reference_bytes;
    uint64_t reference_position;
    int32_t candidate_token_ids[SALT_DPR_MAX_HORIZON];
    SaltDprChart chart;
    SaltDprEdgeKind kind;
    int32_t bonus_token_id;
    uint8_t next_node_sha256[32];
} SaltDprEdge;

/* Mentor is an immutable compiled DeltaNet mindset/reference-family identity.
 * The descriptor contains identities only: no Mentor KV, recurrent payload,
 * logits, token IDs, or state that can be imported into the live target. */
typedef struct SaltDprMentorFamily {
    uint32_t schema_version;
    uint8_t family_sha256[32];
    uint8_t mindset_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint8_t nomogram_root_sha256[32];
    uint8_t provenance_sha256[32];
} SaltDprMentorFamily;

typedef enum SaltDprSelectorKind {
    SALT_DPR_SELECTOR_INVALID = 0,
    SALT_DPR_SELECTOR_PARENT_EXACT = 1,
    SALT_DPR_SELECTOR_MENTOR_NOMOGRAM = 2,
    SALT_DPR_SELECTOR_NATIVE_FALLBACK = 3
} SaltDprSelectorKind;

/* Runtime-only candidate view over already parsed/authenticated artifacts.
 * expected_saved_numerator/denominator is a positive-saving ratio. A zero
 * numerator is a valid cost rejection; a zero denominator is malformed. */
typedef struct SaltDprSelectorCandidate {
    uint32_t schema_version;
    SaltDprSelectorKind kind;
    size_t family_index;
    uint8_t edge_sha256[32];
    uint8_t binding_sha256[32];
    uint8_t source_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint64_t expected_saved_numerator;
    uint64_t expected_saved_denominator;
    int provenance_authenticated;
    int policy_allowed;
} SaltDprSelectorCandidate;

/* Fixed provenance copied from the selected candidate/family. Parent exact
 * results leave all family/mindset/Nomogram digests zero. */
typedef struct SaltDprSelectorResult {
    SaltDprSelectorKind kind;
    size_t candidate_index;
    size_t family_index;
    uint64_t expected_saved_numerator;
    uint64_t expected_saved_denominator;
    uint8_t source_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint8_t family_sha256[32];
    uint8_t mindset_sha256[32];
    uint8_t provenance_sha256[32];
    uint8_t nomogram_root_sha256[32];
    uint8_t edge_sha256[32];
    uint8_t binding_sha256[32];
} SaltDprSelectorResult;

/* Orthogonal operation planning. ND proposes token trajectories; NM proposes
 * A0 compute preparation. Either pointer may be absent/refused independently. */
typedef enum SaltDprOperationKind {
    SALT_DPR_OPERATION_INVALID = 0,
    SALT_DPR_OPERATION_PREFILL = 1,
    SALT_DPR_OPERATION_NATIVE_DECODE = 2,
    SALT_DPR_OPERATION_DRAFT_VERIFY = 3,
    SALT_DPR_OPERATION_REPAIR = 4
} SaltDprOperationKind;

typedef enum SaltDprExactnessLevel {
    SALT_DPR_EXACTNESS_INVALID = 0,
    SALT_DPR_EXACTNESS_A0 = 1,
    SALT_DPR_EXACTNESS_A1 = 2,
    SALT_DPR_EXACTNESS_A2 = 3
} SaltDprExactnessLevel;

typedef enum SaltDprRelevanceKind {
    SALT_DPR_RELEVANCE_INVALID = 0,
    SALT_DPR_RELEVANCE_EXPERT_PREFETCH = 1,
    SALT_DPR_RELEVANCE_KV_PREFETCH = 2
} SaltDprRelevanceKind;

typedef enum SaltDprOperationCell {
    SALT_DPR_CELL_INVALID = 0,
    SALT_DPR_CELL_NATIVE = 1,
    SALT_DPR_CELL_ND_ONLY = 2,
    SALT_DPR_CELL_NM_ONLY = 3,
    SALT_DPR_CELL_ND_NM = 4
} SaltDprOperationCell;

typedef struct SaltDprComputeIntent {
    uint32_t schema_version;
    SaltDprOperationKind kind;
    uint64_t position;
    uint32_t horizon;
    uint32_t candidate_count;
    uint8_t target_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t active_policy_sha256[32];
    uint8_t candidate_sha256[32];
} SaltDprComputeIntent;

typedef struct SaltDprTransitionProposal {
    uint32_t schema_version;
    uint32_t candidate_count;
    int32_t candidate_token_ids[SALT_DPR_MAX_WALK_HORIZON];
    uint8_t source_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t proposal_sha256[32];
    uint8_t policy_sha256[32];
    uint64_t expected_saved_numerator;
    uint64_t expected_saved_denominator;
    int provenance_authenticated;
    int policy_allowed;
} SaltDprTransitionProposal;

typedef struct SaltDprRelevanceItem {
    SaltDprRelevanceKind kind;
    uint32_t layer;
    uint32_t index;
    uint64_t start;
    uint64_t length;
    uint32_t priority;
} SaltDprRelevanceItem;

typedef struct SaltDprAttentionPlan {
    uint32_t schema_version;
    SaltDprExactnessLevel exactness;
    uint8_t dpr_sha256[32];
    uint8_t mindset_sha256[32];
    uint8_t provenance_sha256[32];
    uint8_t intent_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint32_t item_count;
    SaltDprRelevanceItem items[SALT_DPR_MAX_RELEVANCE_ITEMS];
    uint64_t expected_saved_numerator;
    uint64_t expected_saved_denominator;
    int provenance_authenticated;
    int policy_allowed;
} SaltDprAttentionPlan;

/* Caller-owned, fixed-capacity route history for admission-time expert
 * coverage. The ledger holds counts only: it never selects route arithmetic,
 * owns resources, or allocates memory. */
typedef struct SaltDprExpertCoverageLedger {
    uint64_t *route_counts;
    uint32_t layer_count;
    uint32_t expert_count;
    uint64_t observation_count;
} SaltDprExpertCoverageLedger;

typedef struct SaltDprOperationPlan {
    SaltDprOperationCell cell;
    int has_transition;
    int has_attention;
    uint8_t transition_sha256[32];
    uint8_t attention_dpr_sha256[32];
    uint8_t mindset_sha256[32];
} SaltDprOperationPlan;

typedef int (*SaltDprStopTokenFn)(void *opaque, int32_t token);

typedef enum SaltDprMemoryReason {
    SALT_DPR_MEMORY_NONE = 0,
    SALT_DPR_MEMORY_DISABLED = 1,
    SALT_DPR_MEMORY_CURRENT_BUDGET = 2,
    SALT_DPR_MEMORY_ADDITIVE_BUDGET = 3,
    SALT_DPR_MEMORY_OVERFLOW = 4
} SaltDprMemoryReason;

typedef struct SaltDprMemoryRequest {
    SaltDprMode mode;
    uint64_t current_kv_budget_bytes;
    uint64_t additive_dpr_budget_bytes;
    uint64_t target_required_bytes;
    uint64_t reference_required_bytes;
    uint64_t verification_scratch_bytes;
    uint64_t checkpoint_max_bytes;
} SaltDprMemoryRequest;

typedef struct SaltDprMemoryDecision {
    int admitted;
    SaltDprMemoryReason reason;
    uint64_t current_kv_required_bytes;
    uint64_t additive_dpr_required_bytes;
    uint64_t verification_scratch_bytes;
    uint64_t checkpoint_max_bytes;
    uint64_t dpr_owned_bytes;
} SaltDprMemoryDecision;

typedef enum SaltDprPhase {
    SALT_DPR_PHASE_OFF = 0,
    SALT_DPR_PHASE_TARGET = 1,
    SALT_DPR_PHASE_EMPTY = 2,
    SALT_DPR_PHASE_REFERENCE = 3,
    SALT_DPR_PHASE_VERIFYING = 4,
    SALT_DPR_PHASE_PENDING = 5,
    SALT_DPR_PHASE_DEAD = 6
} SaltDprPhase;

typedef struct SaltDprRuntime {
    SaltDprMode mode;
    SaltDprPhase phase;
    uint64_t epoch;
    int target_live;
    int reference_live;
    int checkpoint_set;
    uint8_t checkpoint_sha256[32];
    int pending_valid;
    int32_t pending_token_id;
    uint64_t pending_position;
} SaltDprRuntime;

int salt_dpr_node_sha256(const SaltDprNodeMaterial *material,
                         uint8_t out[32]);
int salt_dpr_transition_chain_init(
    const uint8_t state_compatibility_sha256[32],
    const uint8_t sampler_config_sha256[32], uint32_t sampler_abi,
    uint64_t position, uint8_t out[32]);
int salt_dpr_transition_chain_advance(
    const uint8_t previous[32], int32_t token_id,
    uint64_t resulting_position, uint8_t out[32]);
int salt_dpr_nomogram_qa_key(const uint8_t state_compatibility_sha256[32],
                             const int32_t *prompt_token_ids,
                             size_t prompt_token_count,
                             uint32_t bucket_bits,
                             uint8_t out[32]);
int salt_dpr_chart_validate(const SaltDprChart *chart);
int salt_dpr_chart_fraction(const SaltDprChart *chart,
                            uint64_t *numerator, uint64_t *denominator);
int salt_dpr_chart_expected_saved(const SaltDprChart *chart,
                                  uint64_t serial_ns_per_token,
                                  uint64_t *numerator,
                                  uint64_t *denominator);
int salt_dpr_chart_is_green(const SaltDprChart *chart,
                            uint64_t serial_ns_per_token,
                            uint64_t minimum_samples);
int salt_dpr_chart_compare(const SaltDprChart *left,
                           const SaltDprChart *right,
                           int *comparison);
int salt_dpr_edge_validate(const SaltDprEdge *edge,
                           int32_t vocab_size,
                           uint64_t max_context,
                           SaltDprStopTokenFn stop_token,
                           void *stop_opaque);
int salt_dpr_select_edge(const SaltDprEdge *edges, size_t edge_count,
                         const uint8_t source_node_sha256[32],
                         SaltDprMode mode,
                         uint64_t serial_ns_per_token,
                         uint64_t minimum_samples,
                         int32_t vocab_size,
                         uint64_t max_context,
                         SaltDprStopTokenFn stop_token,
                         void *stop_opaque,
                         size_t *selected_index);
int salt_dpr_mentor_family_validate(const SaltDprMentorFamily *family);
/* Return 1 with a selected reference, 0 with native fallback, or -1 for a
 * malformed caller-owned view. Parent exact candidates have lexicographic
 * authority over Mentor/Nomogram candidates regardless of advertised saving.
 * Equal savings use family digest then edge digest ordering. */
int salt_dpr_selector_choose(
    const SaltDprMentorFamily *families, size_t family_count,
    const SaltDprSelectorCandidate *candidates, size_t candidate_count,
    const uint8_t target_node_sha256[32],
    const uint8_t target_state_compatibility_sha256[32],
    const uint8_t active_policy_sha256[32],
    SaltDprSelectorResult *result);
int salt_dpr_compute_intent_sha256(const SaltDprComputeIntent *intent,
                                   uint8_t out[32]);
int salt_dpr_candidate_sha256(const int32_t *candidate_token_ids,
                              uint32_t candidate_count, uint8_t out[32]);
/* Validation returns 1 for admitted, 0 for a well-formed refused plan, and -1
 * for malformed caller-owned input. */
int salt_dpr_transition_proposal_validate(
    const SaltDprTransitionProposal *proposal,
    const SaltDprComputeIntent *intent, int32_t vocab_size);
int salt_dpr_attention_plan_validate(
    const SaltDprAttentionPlan *plan, const SaltDprComputeIntent *intent,
    uint32_t max_layers, uint32_t max_experts, uint64_t max_context);
/* Compile bounded per-layer route counts into this plan's existing
 * EXPERT_PREFETCH items. Metadata/digests/policy fields remain caller-owned.
 * Returns 0 on a nonempty plan, 1 when all counts are zero, and -1 on invalid
 * input/overflow. No allocation, state mutation, or router authority. */
int salt_dpr_attention_plan_compile_expert_coverage(
    SaltDprAttentionPlan *plan, const uint64_t *route_counts,
    uint32_t layer_count, uint32_t expert_count,
    uint32_t maximum_per_layer, uint32_t maximum_items);
int salt_dpr_expert_coverage_ledger_init(
    SaltDprExpertCoverageLedger *ledger, uint64_t *route_counts,
    uint32_t layer_count, uint32_t expert_count);
int salt_dpr_expert_coverage_ledger_reset(
    SaltDprExpertCoverageLedger *ledger);
/* Atomically merge one fixed layer_count x expert_count count matrix. */
int salt_dpr_expert_coverage_ledger_merge(
    SaltDprExpertCoverageLedger *ledger, const uint64_t *route_counts);
int salt_dpr_expert_coverage_ledger_compile(
    const SaltDprExpertCoverageLedger *ledger,
    uint32_t maximum_per_layer, uint32_t maximum_items,
    SaltDprAttentionPlan *plan);
int salt_dpr_expert_coverage_ledger_sha256(
    const SaltDprExpertCoverageLedger *ledger, uint8_t out[32]);
int salt_dpr_attention_plan_finalize_runtime(
    SaltDprAttentionPlan *plan, const SaltDprComputeIntent *intent,
    const uint8_t mindset_sha256[32],
    const uint8_t provenance_sha256[32]);
int salt_dpr_operation_plan_compose(
    const SaltDprComputeIntent *intent,
    const SaltDprTransitionProposal *transition,
    const SaltDprAttentionPlan *attention,
    uint32_t max_layers, uint32_t max_experts, uint64_t max_context,
    int32_t vocab_size, SaltDprOperationPlan *plan);
int salt_dpr_memory_admit(const SaltDprMemoryRequest *request,
                          SaltDprMemoryDecision *decision);

int salt_dpr_runtime_init(SaltDprRuntime *runtime, SaltDprMode mode);
int salt_dpr_runtime_invariant(const SaltDprRuntime *runtime);
int salt_dpr_target_unload(SaltDprRuntime *runtime,
                           const uint8_t checkpoint_sha256[32]);
int salt_dpr_reference_acquire(SaltDprRuntime *runtime);
int salt_dpr_reference_release(SaltDprRuntime *runtime);
int salt_dpr_target_restore(SaltDprRuntime *runtime,
                            const uint8_t checkpoint_sha256[32]);
int salt_dpr_verify_begin(SaltDprRuntime *runtime);
int salt_dpr_verify_finish_pending(SaltDprRuntime *runtime,
                                   int32_t token_id, uint64_t position);
/* Return 1 and the pending token/position, 0 when no token is pending, or -1
 * for malformed runtime state. */
int salt_dpr_pending_read(const SaltDprRuntime *runtime,
                          int32_t *token_id, uint64_t *position);
int salt_dpr_pending_commit(SaltDprRuntime *runtime,
                            int32_t token_id, uint64_t resulting_position);
int salt_dpr_runtime_fail(SaltDprRuntime *runtime);

#endif /* SALT_DPR_H */
