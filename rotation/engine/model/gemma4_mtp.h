#ifndef SALT_GEMMA4_MTP_H
#define SALT_GEMMA4_MTP_H

#include <stddef.h>
#include <stdint.h>

#include "salt/attn.h"

typedef struct SaltGemma4Mtp SaltGemma4Mtp;

typedef struct SaltGemma4MtpTargetOps {
    void *context;
    SaltKvCache *compute_pool;
    int (*target_embedding)(void *context, int32_t token, float *embedding);
    int (*shared_attention)(void *context, int full_attention, int position,
                            const float *queries, float *outputs);
    int (*is_stop)(void *context, int32_t token);
} SaltGemma4MtpTargetOps;

typedef struct SaltGemma4MtpStats {
    uint64_t rounds;
    uint64_t steps;
    uint64_t root_matches;
    uint64_t root_misses;
    uint64_t proposed_tokens;
    uint64_t matvecs;
    uint64_t metal_matvecs;
    uint64_t attention_calls;
} SaltGemma4MtpStats;

SaltGemma4Mtp *salt_gemma4_mtp_load(
    const char *source_path, const char *receipt_path,
    const SaltGemma4MtpTargetOps *target,
    char *error, size_t error_size);
void salt_gemma4_mtp_free(SaltGemma4Mtp *mtp);

/* Draft one causal trajectory from the current target boundary. The returned
 * sequence always starts at authoritative_root. A zero return with count=1
 * means the trained assistant did not match that root, so no suffix exists.
 * The assistant writes no target KV/state and consumes the target's existing
 * final hidden state and shared K/V views only. */
int salt_gemma4_mtp_draft(
    SaltGemma4Mtp *mtp, int32_t last_committed_token,
    const float *target_hidden, uint32_t target_position,
    int32_t authoritative_root,
    uint32_t maximum_tokens, int32_t *tokens, uint32_t capacity,
    uint32_t *count);

int salt_gemma4_mtp_stats(const SaltGemma4Mtp *mtp,
                          SaltGemma4MtpStats *stats);
uint64_t salt_gemma4_mtp_resident_bytes(const SaltGemma4Mtp *mtp);
uint64_t salt_gemma4_mtp_startup_bytes(void);

#endif
