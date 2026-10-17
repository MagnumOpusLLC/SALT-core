/* LFM2.5-24B-A2B greedy sampler.
 *
 * The last stage of the forward loop: pick the token from a logit row. It is
 * deliberately the smallest thing in the package -- no temperature, no top-k, no
 * RNG -- because the loop's first run has to be reproducible before it can be
 * anything else, and a sampler is exactly where a plausible-looking wrong number
 * would hide.
 *
 * Model-owned. The engine holds no LFM arithmetic; this holds no engine
 * arithmetic. */

#ifndef SALT_LFM_SAMPLE_H
#define SALT_LFM_SAMPLE_H

#include <stdint.h>

/* Greedy argmax over `count` logits.
 *
 * Ties resolve to the LOWEST id: the comparison is strictly greater, so an equal
 * later value never displaces an earlier one. That makes the result a function of
 * the logits alone, which is what lets a run be compared to another run.
 *
 * A NaN never wins (every comparison against it is false), so a NaN anywhere in
 * the row cannot silently become the selection -- it is skipped, and if the whole
 * row is NaN the result is id 0, which a caller can detect by checking the logits
 * it passed in.
 *
 * Returns -1 for a null row or an empty one. */
int32_t lfm_sample_argmax(const float *logits, uint32_t count);

#endif /* SALT_LFM_SAMPLE_H */
