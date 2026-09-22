#include "salt/sampling.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const float logits[] = {-2.0f, 1.0f, 1.0f, 0.5f, -1.0f};
    float original[5], invalid[5];
    SaltSamplerConfig config = {SALT_SAMPLER_GREEDY_V1, 0.0f, 0, 1};
    int token = -7, replay = -8, seen[5] = {0};
    memcpy(original, logits, sizeof original);
    assert(salt_sampler_select(&config, logits, 5, 0, &token) == 0);
    assert(token == 1); /* Canonical lowest-index tie. */
    config.abi = SALT_SAMPLER_TEMPERATURE_COUNTER_V1;
    config.temperature = 0.8f;
    config.seed = 42;
    config.top_k = 3;
    for (uint64_t position = 0; position < 512; position++) {
        assert(salt_sampler_select(&config, logits, 5, position, &token) == 0);
        assert(salt_sampler_select(&config, logits, 5, position, &replay) == 0);
        assert(token == replay && token >= 1 && token <= 3);
        seen[token] = 1;
    }
    assert(seen[1] && seen[2] && seen[3]);
    config.top_k = 1;
    assert(salt_sampler_select(&config, logits, 5, UINT64_MAX, &token) == 0);
    assert(token == 1);
    config.top_k = SALT_SAMPLER_MAX_TOP_K;
    assert(salt_sampler_select(&config, logits, 5, 5, &token) == 0);
    assert(memcmp(original, logits, sizeof original) == 0);
    for (unsigned invalid_case = 0; invalid_case < 7; invalid_case++) {
        config.abi = SALT_SAMPLER_TEMPERATURE_COUNTER_V1;
        config.temperature = 0.8f;
        config.top_k = 3;
        memcpy(invalid, logits, sizeof invalid);
        if (invalid_case == 0) config.abi = 99;
        if (invalid_case == 1) config.temperature = 0.0f;
        if (invalid_case == 2) config.temperature = NAN;
        if (invalid_case == 3) config.top_k = 0;
        if (invalid_case == 4) config.top_k = SALT_SAMPLER_MAX_TOP_K + 1;
        if (invalid_case == 5) invalid[0] = INFINITY;
        if (invalid_case == 6) invalid[4] = NAN;
        token = -7;
        assert(salt_sampler_select(&config, invalid, 5, 0, &token) == -1);
        assert(token == -7);
    }
    assert(salt_sampler_select(NULL, logits, 5, 0, &token) == -1);
    assert(salt_sampler_select(&config, NULL, 5, 0, &token) == -1);
    assert(salt_sampler_select(&config, logits, 0, 0, &token) == -1);
    assert(salt_sampler_select(&config, logits, 5, 0, NULL) == -1);
    puts("sampling: PASS greedy/ties/top-k/counter-replay/refusal/input-immutable");
    return 0;
}
