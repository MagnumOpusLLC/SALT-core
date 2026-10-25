#ifndef SALT_GEMMA4_OPERATION_H
#define SALT_GEMMA4_OPERATION_H

#include "gemma4_text.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef SaltTextGenerated SaltGemma4TargetGenerateResult;

int salt_gemma4_text_scheduler_binding(
    SaltGemma4Text *model, SaltTextGenerationBinding *binding,
    SaltTextTokenEpochController **controller);

int salt_gemma4_text_target_epoch_active(
    SaltGemma4Text *model, SaltTextTokenEpochController *controller,
    const int32_t *route_token_ids, int candidate_count,
    const float *parent_logits,
    SaltTextTokenExactLookupCommit exact_lookup_commit,
    void *exact_context, SaltTextTokenEpochResult *result);

/* Bind existing model storage/executor to core projection refill and TARGET.
 * Candidate extraction runs no model steps and does not mutate KV. logits is
 * the authoritative parent and, on success, the pending target distribution.
 * proposal_logits is existing non-state vocabulary scratch. Return 1 only for
 * a clean proposal-unavailable result or terminal ordinary-close handoff. */
int salt_gemma4_text_target_generate(
    SaltGemma4Text *model, float *logits, int maximum_tokens,
    float *proposal_logits, SaltGemma4TargetGenerateResult *result);
int salt_gemma4_text_consume_known(
    SaltGemma4Text *model, int token, float *logits);
int salt_gemma4_text_is_stop_token(int token);

#ifdef __cplusplus
}
#endif

#endif
