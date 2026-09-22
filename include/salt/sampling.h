#ifndef SALT_SAMPLING_H
#define SALT_SAMPLING_H

#include <stdint.h>

#define SALT_SAMPLER_GREEDY_V1 1u
#define SALT_SAMPLER_TEMPERATURE_COUNTER_V1 2u
#define SALT_SAMPLER_MAX_TOP_K 256u

/* Request policy only. Position identifies the authoritative target draw;
 * candidate exploration never advances another draw or owns sampler state. */
typedef struct SaltSamplerConfig {
    uint32_t abi;
    float temperature;
    uint64_t seed;
    uint32_t top_k;
} SaltSamplerConfig;

/* Model-neutral, allocation-free target selection. The counter ABI preserves
 * the existing binary32 temperature/top-k and SplitMix64 arithmetic exactly;
 * it is distinct from the legacy xorshift samplers in head.c. On refusal the
 * output is unchanged. No KV, model state, or input logits are mutated. */
int salt_sampler_select(const SaltSamplerConfig *config,
                        const float *logits, int count,
                        uint64_t position, int *token_out);

#endif
