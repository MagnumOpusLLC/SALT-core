/*
 * head.h -- output head + embedding + sampling (issue #6 step 3).
 *
 * head.json / embed.json are written by tools/convert-salt.py convert
 * from the checkpoint's head.weight/head.scale/embed.weight (bytes
 * copied as-is). Logits = head . state (F8_E4M3, E8M0 group scales).
 * Sampling is DETERMINISTIC: xorshift64 seeded by the caller, softmax
 * over the full vocab, single PRNG draw -- same seed, same token.
 */
#ifndef SALT_HEAD_H
#define SALT_HEAD_H

#include "salt/salt.h"

#include <stdint.h>
#include <math.h>

typedef struct SaltHead {
    uint8_t *buf;           /* head.bin mmap'd/read whole */
    long     buf_n;
    long     w_off, w_nbytes, s_off, s_nbytes, b_off, b_nbytes;
    long     dims[4];
    int      rank, w_dtype;
    int      w_bits;        /* MLX affine bits (4 or 8); U32 packs 32/bits */
    long     sdims[4];
    int      srank;
} SaltHead;

typedef struct SaltEmbed {
    uint8_t *buf;
    long     buf_n;
    long     dims[4];
    int      rank, dtype;
    int      w_bits;        /* MLX affine bits (4 or 8); U32 packs 32/bits */
    long     s_off, s_nbytes, b_off, b_nbytes;
    long     sdims[4];
    int      srank;
} SaltEmbed;

int salt_head_load(SaltHead *h, const char *json_path);
void salt_head_free(SaltHead *h);
int salt_embed_load(SaltEmbed *e, const char *json_path);
void salt_embed_free(SaltEmbed *e);

/* logits[V] = head . state[H]; V = head rows. */
int salt_head_logits(const SaltHead *h, const float *state, float *logits);

/* state[H] = embed row tok (F32 or BF16; F8 refused for now). */
int salt_embed_gather(const SaltEmbed *e, int tok, float *out);

/* Deterministic top-k-free softmax sample: mutates *rng. */
int salt_sample(const float *logits, int V, uint64_t *rng);
int salt_sample_topp(const float *logits, int V, double p, uint64_t *rng,
                     int maxk);
int salt_sample_vllm(const float *logits, int V, int top_k, double top_p,
                     float temp, uint64_t *rng);

/* Additive frequency penalty applied to a logits buffer in place:
 * logits[i] -= freq_pen * count_i for every token in the recent
 * window (vLLM frequency_penalty semantics). Unlike the
 * multiplicative repetition penalty (logits /= pen^count, which a
 * 30-logit attractor peak can overwhelm), the additive form scales
 * linearly with count and subtracts directly from the logit gap:
 * count=9, freq_pen=1.0 -> -9 logits, which breaks any loop.
 * Pass the same recent-window counts the repetition penalty uses.
 * Returns 0. */
void salt_apply_freq_penalty(float *logits, int V, const int *recent,
                             int n_recent, float freq_pen);

/* Presence penalty (vLLM semantics, the Qwen3.6 model card's
 * recommended loop-killer): logits[i] -= pres_pen ONCE per token
 * present in the window, independent of count (card: 1.5, range
 * 0-2). Returns 0. */
void salt_apply_presence_penalty(float *logits, int V, const int *recent,
                                 int n_recent, float pres_pen);

/* Existing SALT generated-window repetition convention: divide each seen
 * logit by penalty^count, forming the power with ordered binary32 multiplies.
 * Both signs are divided, as in the historical Qwen path; this is NOT the
 * sign-aware HF set-style convention. No allocation, RNG, or retained state.
 * The caller supplies the bounded window and a private, existing logit seat.
 * Invalid input is refused before any write. */
#define SALT_REPETITION_MAX_WINDOW 64u
static inline int salt_apply_repetition_penalty(
        float *logits, int vocabulary, const int32_t *recent,
        uint32_t count, float penalty) {
    if (!logits || vocabulary < 1 || count > SALT_REPETITION_MAX_WINDOW ||
        (count && !recent) || !isfinite(penalty) ||
        penalty < 1.0f || penalty > 2.0f) return -1;
    for (uint32_t i = 0; i < count; i++)
        if (recent[i] < 0 || recent[i] >= vocabulary ||
            !isfinite(logits[recent[i]])) return -1;
    if (penalty == 1.0f) return 0;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t earlier = 0, occurrences = 0;
        for (; earlier < i; earlier++)
            if (recent[earlier] == recent[i]) break;
        if (earlier != i) continue;
        for (uint32_t j = i; j < count; j++)
            if (recent[j] == recent[i]) occurrences++;
        float factor = penalty;
        for (uint32_t n = 1; n < occurrences; n++) factor *= penalty;
        logits[recent[i]] /= factor;
    }
    return 0;
}

/* Greedy: the argmax token id (the SALT_GREEDY sampling mode). */
int salt_argmax(const float *logits, int V);

#endif /* SALT_HEAD_H */
