#ifndef SALT_TEXT_EXEC_H
#define SALT_TEXT_EXEC_H

#include <stddef.h>
#include <stdint.h>
#include "salt/tensorops.h"

/* A physical lane of the existing PREFILL traversal, not a second controller. */
typedef struct SaltTextPrefillTeam {
    SaltTensorHostGraphState *graph;
    uint32_t worker;
    uint32_t workers;
} SaltTextPrefillTeam;
const SaltTextPrefillTeam *salt_text_prefill_team_current(void);
int salt_text_prefill_team_sync(int status);
int salt_text_prefill_team_owner(void);
/* Execute one indivisible physical operation exactly once. Its body must not
 * submit CPU work back to the occupied pool. Completion precedes team reuse. */
int salt_text_prefill_team_once(int (*operation)(void *), void *context);
int salt_text_prefill_team_slice(int workers,
    void (*operation)(int worker, void *context), void *context);

typedef enum SaltTextRuntimePhase {
    SALT_TEXT_PHASE_PREFILL = 0,
    SALT_TEXT_PHASE_DECODE = 1,
    SALT_TEXT_PHASE_TARGET_VERIFY = 2,
} SaltTextRuntimePhase;

typedef struct SaltTextPhaseScopeOps {
    int (*begin)(void *context, SaltTextRuntimePhase phase);
    int (*end)(void *context, SaltTextRuntimePhase phase,
               int operation_status);
} SaltTextPhaseScopeOps;

typedef struct SaltTextPhaseScope {
    const SaltTextPhaseScopeOps *ops;
    void *context;
} SaltTextPhaseScope;

typedef enum SaltTextPhaseLookaheadStatus {
    SALT_TEXT_LOOKAHEAD_EMPTY = 0,
    SALT_TEXT_LOOKAHEAD_ELIGIBLE = 1,
    SALT_TEXT_LOOKAHEAD_PREPARED = 2,
    SALT_TEXT_LOOKAHEAD_READY = 3,
    SALT_TEXT_LOOKAHEAD_CLAIMED = 4,
    SALT_TEXT_LOOKAHEAD_REFUSED = 5,
    SALT_TEXT_LOOKAHEAD_CANCELLED = 6,
} SaltTextPhaseLookaheadStatus;

typedef struct SaltTextPhaseLookaheadRequest {
    SaltTextRuntimePhase source_phase;
    SaltTextRuntimePhase target_phase;
    uint64_t source_generation;
    uint64_t result_generation;
    uint32_t source_position;
    uint32_t result_position;
} SaltTextPhaseLookaheadRequest;

typedef struct SaltTextPhaseLookaheadOps {
    /* Preparation is resource-only: 0 prepared, 1 refused/no-op, -1 invalid.
     * It may populate or advise only existing bounded resources. */
    int (*prepare)(void *context,
                   const SaltTextPhaseLookaheadRequest *request);
    /* Claim transfers a matching READY preparation to the committed target
     * phase. It must not perform arithmetic, K/V writes, or publication. */
    int (*claim)(void *context,
                 const SaltTextPhaseLookaheadRequest *request);
    /* Cancel unwinds any preparation that owns a lease or other bounded
     * resource. It is called on phase failure and stale identity. */
    int (*cancel)(void *context,
                  const SaltTextPhaseLookaheadRequest *request);
} SaltTextPhaseLookaheadOps;

typedef struct SaltTextPhaseLookahead {
    SaltTextPhaseLookaheadRequest request;
    const SaltTextPhaseLookaheadOps *ops;
    void *context;
    SaltTextPhaseLookaheadStatus status;
} SaltTextPhaseLookahead;

/* Execute one existing operation body inside one status-bearing physical phase
 * scope. The wrapper owns no scheduling, resources, state, or arithmetic. Once
 * begin succeeds, end executes exactly once on success and failure. */
int salt_text_phase_scope_run(
    const SaltTextPhaseScope *scope, SaltTextRuntimePhase phase,
    int (*operation)(void *operation_context), void *operation_context);

/* Claim or cancel one preparation against the exact committed phase identity.
 * claim returns 1 when a matching preparation was claimed, 0 when no usable
 * preparation exists, and -1 on callback/lifecycle failure. */
int salt_text_phase_lookahead_claim(
    SaltTextPhaseLookahead *lookahead, SaltTextRuntimePhase phase,
    uint64_t generation, uint32_t position);
int salt_text_phase_lookahead_cancel(SaltTextPhaseLookahead *lookahead);

/* Model-neutral layer-major text prefill control plane. Model adapters bind
 * authenticated tensors and layered arithmetic callbacks; the engine owns
 * chunking, layer order, release/fence order, final-head order, and position
 * publication. Operation callbacks plan internal projection/expert work; the
 * control plane never replaces them with token-serial arithmetic. */
typedef struct SaltTextPrefillProfile {
    double total_s;
    double setup_s;
    double embed_s;
    double attention_s;
    double feed_forward_s;
    double release_s;
    double finish_s;
    double validate_s;
    double publish_s;
    double cleanup_s;
    int chunks;
    int grouped_chunks;
    int released_layers;
    int retained_layers;
} SaltTextPrefillProfile;

typedef enum SaltTextResourcePolicy {
    SALT_TEXT_RELEASE_PER_LAYER = 0,
    SALT_TEXT_RETAIN_OPERATION = 1,
} SaltTextResourcePolicy;

typedef struct {
    float *states;
    float *branches;
    size_t element_capacity;
} SaltTextPrefillScratch;

typedef struct {
    int n_layers;
    int hidden;
    int max_batch;
    SaltTextPrefillProfile *profile;
    SaltTextPrefillScratch *scratch;
    float *position_logits;
    size_t position_logits_stride;
    size_t position_logits_width;
    SaltTextResourcePolicy resource_policy;
    SaltTextPhaseLookahead *lookahead;
    const SaltTextPhaseLookaheadOps *lookahead_ops;
    void *lookahead_context;
    uint64_t transition_generation;
    SaltTextRuntimePhase next_phase;
    int (*transaction_run)(void *transaction_context,
                           int (*operation)(void *operation_context),
                           void *operation_context);
    void *transaction_context;
    SaltTensorHostGraphState *team_state;
    SaltTensorHostGraphParallelRun team_run;
    void *team_context;
    uint32_t team_workers;
} SaltTextPrefillPlan;

typedef struct {
    int (*embed)(void *context, const int *tokens, int batch, float *states);
    int (*attention)(void *context, int layer, int start_position, int batch,
                     const float *states, float *outputs);
    int (*feed_forward)(void *context, int layer, int start_position, int batch,
                        const float *states, float *outputs);
    void (*release_layer)(void *context, int layer);
    int (*finish)(void *context, const float *last_state, float *logits);
    int (*finish_batch)(void *context, const float *states, int batch,
                        float *position_logits, size_t logits_stride);
    int (*publish_position)(void *context, int position);
} SaltTextPrefillOps;

int salt_text_prefill_execute(const SaltTextPrefillPlan *plan,
                              const SaltTextPrefillOps *ops,
                              void *context, int start_position,
                              const int *tokens, int token_count,
                              float *logits);

#endif
