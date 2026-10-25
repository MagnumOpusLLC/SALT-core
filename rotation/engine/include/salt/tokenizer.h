/*
 * tokenizer.h -- HF tokenizer.json support for ds4f-disk (issue #6
 * step 5): byte-level BPE decode (ids -> text) and encode (text ->
 * ids). Handles the gpt-2/DeepSeek byte<->unicode mapping, the vocab
 * object, the merges list, and plain-string vocabs (non-byte-level).
 *
 * Zero deps beyond libc; the JSON is parsed by the vendored
 * src/json.h (included by tokenizer.c).
 */
#ifndef SALT_TOKENIZER_H
#define SALT_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef struct SaltTokenizer {
    int      mode;                /* SALT_TOKENIZER_* */
    int      nvocab;              /* vocab size */
    char   **vocab;               /* id -> token string (unicode form) */
    /* string -> id hash */
    char   **vkeys;               /* table: NUL-terminated token */
    int     *vids;
    int      vcap;                /* power of two */
    /* pair (left_id, right_id) -> (rank, merged_id) */
    uint64_t *pkeys;              /* (left<<32)|right */
    int32_t *pranks, *pmerged;
    int      pcap;
    /* byte<->unicode tables (gpt-2 style) */
    uint8_t rev[0x180];           /* char -> byte (0xFF = plain char) */
    uint16_t fwd[256];            /* byte -> char codepoint (>= 0x100 for
                                    the unsafe bytes) */
    int      nbytes_unsafe;       /* how many bytes map to 0x100+ */
    /* added_tokens (special tokens): content -> id. Matched verbatim
     * (longest first) BEFORE byte-level BPE, like HF's added_tokens
     * handling. Up to 64, sorted by decreasing content length. */
    int      nadded;
    char    *added[64];           /* content (raw UTF-8 as in json) */
    int      added_id[64];
    size_t   added_len[64];
} SaltTokenizer;

enum {
    SALT_TOKENIZER_GPT2_BYTE_LEVEL = 0,
    SALT_TOKENIZER_METASPACE_BYTE_FALLBACK = 1,
};

/* Load a tokenizer.json. Returns 0 on success. */
int  salt_tokenizer_load(SaltTokenizer *t, const char *path);

/* Load through an already authenticated, retained descriptor. The loader
 * resolves only the operating system's descriptor alias, never the original
 * namespace path. */
int  salt_tokenizer_load_fd(SaltTokenizer *t, int fd);

void salt_tokenizer_free(SaltTokenizer *t);

/* Encode UTF-8 text to token ids (byte-level BPE). Returns the number
 * of ids (<= max_ids) or -1 on failure. */
int salt_tokenizer_encode(const SaltTokenizer *t, const char *utf8,
                          int *ids, int max_ids);

/* Decode token ids to UTF-8 text. Returns bytes written (<= out_n-1,
 * always NUL-terminated) or -1 on failure. */
int salt_tokenizer_decode(const SaltTokenizer *t, const int *ids, int n,
                          char *out, int out_n);

#endif /* SALT_TOKENIZER_H */
