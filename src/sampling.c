/* sampling.c -- portable target sampling, independent of model/transport.
 * Ownership-only extraction of the existing counter sampler. Do not replace
 * its arithmetic with the separate legacy xorshift ABI in head.c. */
#include "salt/sampling.h"
#include "salt/bitmath.h"

#include <math.h>

static uint64_t sampler_counter_draw(uint64_t seed, uint64_t position) {
    uint64_t value = seed +
        (position + UINT64_C(1)) * UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31);
}

static int sampler_select(const SaltSamplerConfig *config,
                           const float *logits, int count, float top_p,
                           uint64_t position, int *token_out) {
    float maximum;
    int best = 0;
    if (!config || !logits || count < 1 || !token_out || !isfinite(logits[0]) ||
        !isfinite(top_p) || top_p <= 0.0f || top_p > 1.0f)
        return -1;
    maximum = logits[0];
    for (int token = 1; token < count; token++) {
        if (!isfinite(logits[token])) return -1;
        if (logits[token] > maximum) {
            maximum = logits[token];
            best = token;
        }
    }
    if (config->abi == SALT_SAMPLER_GREEDY_V1) {
        *token_out = best;
        return 0;
    }
    if (config->abi == SALT_SAMPLER_TEMPERATURE_COUNTER_V1 &&
        isfinite(config->temperature) && config->temperature > 0.0f &&
        config->top_k >= 1u && config->top_k <= SALT_SAMPLER_MAX_TOP_K) {
        int candidate_ids[SALT_SAMPLER_MAX_TOP_K];
        float candidate_logits[SALT_SAMPLER_MAX_TOP_K];
        double total = 0.0, cumulative = 0.0, target;
        uint64_t draw = sampler_counter_draw(config->seed, position);
        uint32_t candidate_count = 0;
        int selected;
        for (int token = 0; token < count; token++) {
            uint32_t insertion;
            float value = logits[token];
            if (candidate_count == config->top_k &&
                value <= candidate_logits[candidate_count - 1u])
                continue;
            insertion = candidate_count < config->top_k
                ? candidate_count++ : candidate_count - 1u;
            while (insertion > 0u &&
                   value > candidate_logits[insertion - 1u]) {
                candidate_logits[insertion] = candidate_logits[insertion - 1u];
                candidate_ids[insertion] = candidate_ids[insertion - 1u];
                insertion--;
            }
            candidate_logits[insertion] = value;
            candidate_ids[insertion] = token;
        }
        if (candidate_count < 1u) return -1;
        maximum = candidate_logits[0];
        if (top_p < 1.0f) {
            double mass = 0.0, prefix = 0.0, threshold;
            /* Only k candidates can survive the subsequent top-k operation.
             * Their cutoff still uses ALL vocabulary mass, not top-k mass.
             * No full-vocabulary sort, scratch array, or modified logits. */
            for (int token = 0; token < count; token++) {
                float weight = salt_expf(logits[token] - maximum);
                if (!isfinite(weight) || weight < 0.0f) return -1;
                mass += (double)weight;
            }
            if (!(mass > 0.0) || !isfinite(mass)) return -1;
            threshold = (double)top_p * mass;
            for (uint32_t candidate = 0; candidate < candidate_count; candidate++) {
                prefix += (double)salt_expf(candidate_logits[candidate] - maximum);
                if (prefix >= threshold) {
                    candidate_count = candidate + 1u;
                    break;
                }
            }
        }
        selected = candidate_ids[0];
        for (uint32_t candidate = 0; candidate < candidate_count; candidate++) {
            float weight = salt_expf(
                (candidate_logits[candidate] - maximum) /
                config->temperature);
            if (!isfinite(weight) || weight < 0.0f) return -1;
            total += (double)weight;
        }
        if (!(total > 0.0) || !isfinite(total)) return -1;
        target = ((double)(draw >> 11) / 9007199254740992.0) * total;
        for (uint32_t candidate = 0; candidate < candidate_count; candidate++) {
            float weight = salt_expf(
                (candidate_logits[candidate] - maximum) /
                config->temperature);
            cumulative += (double)weight;
            if (target < cumulative) {
                selected = candidate_ids[candidate];
                break;
            }
        }
        *token_out = selected;
        return 0;
    }
    return -1;
}

int salt_sampler_select(const SaltSamplerConfig *config,
                        const float *logits, int count,
                        uint64_t position, int *token_out) {
    return sampler_select(config, logits, count, 1.0f, position, token_out);
}

int salt_sampler_select_top_p(const SaltSamplerConfig *config,
                              const float *logits, int count, float top_p,
                              uint64_t position, int *token_out) {
    return sampler_select(config, logits, count, top_p, position, token_out);
}
