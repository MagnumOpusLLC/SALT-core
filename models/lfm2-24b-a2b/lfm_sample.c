/* LFM2.5-24B-A2B greedy sampler. See lfm_sample.h. */

#include "lfm_sample.h"

int32_t lfm_sample_argmax(const float *logits, uint32_t count) {
    uint32_t i, best;

    if (!logits || count == 0u)
        return -1;

    best = 0u;
    for (i = 1u; i < count; i++) {
        /* Strictly greater, so a tie keeps the earlier id and NaN never wins. */
        if (logits[i] > logits[best])
            best = i;
    }
    return (int32_t)best;
}
