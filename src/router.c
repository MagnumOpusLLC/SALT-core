/* router.c -- biased score SELECTS, unbiased score WEIGHTS (invariant 3).
 *
 * Fixture mode scores are deterministic hashes of (state, layer, expert);
 * with real weights the body of the expert loop becomes a dot product
 * against the resident router matrix. The selection/weighting split is
 * the part that must never change. */
#include "salt/salt.h"

#include <stdlib.h>

uint64_t salt_mix64(uint64_t x) {          /* splitmix64 finalizer */
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static uint64_t h3(uint64_t a, uint64_t b, uint64_t c) {
    return salt_mix64(a ^ salt_mix64(b * UINT64_C(0x100000001B3) + c));
}

void salt_router(int *idx, float *w, const SaltCfg *cfg,
                 uint64_t state, int layer, double locality) {
    const int E = cfg->n_experts, K = cfg->topk;
    float *sc = (float *)malloc((size_t)E * sizeof(float));
    float *ch = (float *)malloc((size_t)E * sizeof(float));
    if (!sc || !ch) { free(sc); free(ch); return; }

    for (int e = 0; e < E; e++) {
        double acc = (double)(h3(state, (uint64_t)layer, (uint64_t)e) & 0xFFFFFF) /
                     (double)0x1000000;
        double bias = (double)(h3(state ^ UINT64_C(0xB1A5), (uint64_t)layer,
                                  (uint64_t)e) & 0xFFFFFF) / (double)0x1000000;
        double sel = acc + (bias - 0.5) * 0.1;
        if (locality > 0.0) {
            /* popular set: experts 0..63 per layer, popularity falls off */
            double pop = 1.0 / (1.0 + (double)(e % 64));
            sel = acc * (1.0 + locality * pop);
        }
        sc[e] = (float)acc;                 /* UNBIASED score */
        ch[e] = (float)sel;                 /* SELECTION score */
    }

    char *taken = (char *)calloc((size_t)E, 1);
    if (!taken) { free(sc); free(ch); return; }
    for (int j = 0; j < K; j++) {           /* top-k by repeated max */
        int best = -1;
        for (int e = 0; e < E; e++)
            if (!taken[e] && (best < 0 || ch[e] > ch[best])) best = e;
        taken[best] = 1;
        idx[j] = best;
        w[j]   = sc[best];                  /* UNBIASED score again */
    }
    free(taken);
    free(sc);
    free(ch);
}
