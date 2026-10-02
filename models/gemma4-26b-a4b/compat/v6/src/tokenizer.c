/*
 * tokenizer.c -- HF tokenizer.json: byte-level BPE decode + encode.
 *
 * Model formats accepted:
 *   - "model.vocab": object token -> id (string keys, int values)
 *   - "model.merges": array of [left, right] string pairs
 * Byte-level vocabs store tokens as the gpt-2 unicode-transformed
 * strings (\u0100-\u0143 for the 68 unsafe bytes); decode reverses
 * that mapping per char, encode applies it per byte. Plain-string
 * vocabs pass through as UTF-8.
 */
#include "salt/salt.h"
#include "salt/tokenizer.h"
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* ---------------- byte <-> unicode tables (gpt-2 style) ----------- */
static void byte_tables(SaltTokenizer *t) {
    for (int c = 0; c < 0x180; c++) t->rev[c] = 0xFF;
    for (int c = 0x20; c <= 0x7E; c++) t->rev[c] = (uint8_t)c;
    for (int c = 0xA1; c <= 0xAC; c++) t->rev[c] = (uint8_t)c;
    for (int c = 0xAE; c <= 0xFF; c++) t->rev[c] = (uint8_t)c;
    /* forward (encode) safety: byte 32 is NOT safe -> U+0120 ("Ġ"),
     * the byte-level BPE space (gpt-2 table: safe = 33..126, 161..172,
     * 174..255) */
    int u = 0x100;
    for (int b = 0; b < 256; b++) {
        int safe = (b >= 33 && b <= 126) ||
                   (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        if (!safe) t->rev[u++] = (uint8_t)b;
    }
    t->nbytes_unsafe = u - 0x100;
    for (int b = 0; b < 256; b++) t->fwd[b] = (uint16_t)b;
    u = 0;
    for (int b = 0; b < 256; b++) {
        int safe = (b >= 33 && b <= 126) ||
                   (b >= 161 && b <= 172) ||
                   (b >= 174 && b <= 255);
        if (!safe) t->fwd[b] = (uint16_t)(0x100 + u++);
    }
}

/* ---------------- utf-8 helpers ------------------------------------ */
static uint32_t utf8_get(const unsigned char **pp) {
    const unsigned char *p = *pp;
    uint32_t c = *p++;
    if (c < 0x80) { *pp = p; return c; }
    int n = 0;
    if ((c & 0xE0) == 0xC0) { n = 1; c &= 0x1F; }
    else if ((c & 0xF0) == 0xE0) { n = 2; c &= 0x0F; }
    else if ((c & 0xF8) == 0xF0) { n = 3; c &= 0x07; }
    else { *pp = p; return 0xFFFD; }
    for (int i = 0; i < n; i++) c = (c << 6) | (*p++ & 0x3F);
    *pp = p;
    return c;
}

static int utf8_put(uint32_t c, char *out, int max) {
    if (c < 0x80) { if (max < 1) return 0; out[0] = (char)c; return 1; }
    if (c < 0x800) { if (max < 2) return 0;
        out[0] = (char)(0xC0 | (c >> 6)); out[1] = (char)(0x80 | (c & 0x3F));
        return 2; }
    if (c < 0x10000) { if (max < 3) return 0;
        out[0] = (char)(0xE0 | (c >> 12)); out[1] = (char)(0x80 | ((c >> 6) & 0x3F));
        out[2] = (char)(0x80 | (c & 0x3F)); return 3; }
    if (max < 4) return 0;
    out[0] = (char)(0xF0 | (c >> 18)); out[1] = (char)(0x80 | ((c >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((c >> 6) & 0x3F)); out[3] = (char)(0x80 | (c & 0x3F));
    return 4;
}

static int json_hex(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

/* src/json.h intentionally exposes raw string spans. Tokenizer vocab keys,
 * merge atoms, and added-token contents need their JSON escapes resolved
 * before they participate in string identity (notably "\\n" -> newline). */
static char *json_unescape(const char *s, size_t n, size_t *out_n) {
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    size_t i = 0, o = 0;
    while (i < n) {
        unsigned char c = (unsigned char)s[i++];
        if (c != '\\') {
            out[o++] = (char)c;
            continue;
        }
        if (i >= n) { free(out); return NULL; }
        c = (unsigned char)s[i++];
        if (c == '"' || c == '\\' || c == '/') out[o++] = (char)c;
        else if (c == 'b') out[o++] = '\b';
        else if (c == 'f') out[o++] = '\f';
        else if (c == 'n') out[o++] = '\n';
        else if (c == 'r') out[o++] = '\r';
        else if (c == 't') out[o++] = '\t';
        else if (c == 'u') {
            if (i + 4 > n) { free(out); return NULL; }
            uint32_t cp = 0;
            for (int k = 0; k < 4; k++) {
                int h = json_hex((unsigned char)s[i++]);
                if (h < 0) { free(out); return NULL; }
                cp = (cp << 4) | (uint32_t)h;
            }
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (i + 6 > n || s[i] != '\\' || s[i + 1] != 'u') {
                    free(out); return NULL;
                }
                i += 2;
                uint32_t low = 0;
                for (int k = 0; k < 4; k++) {
                    int h = json_hex((unsigned char)s[i++]);
                    if (h < 0) { free(out); return NULL; }
                    low = (low << 4) | (uint32_t)h;
                }
                if (low < 0xDC00 || low > 0xDFFF) { free(out); return NULL; }
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                free(out); return NULL;
            }
            if (cp == 0) { free(out); return NULL; }
            int written = utf8_put(cp, out + o, (int)(n - o));
            if (written <= 0) { free(out); return NULL; }
            o += (size_t)written;
        } else {
            free(out); return NULL;
        }
    }
    out[o] = 0;
    *out_n = o;
    return out;
}

/* ---------------- hashing ------------------------------------------- */
static uint64_t fnv(const char *s, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 0x100000001B3ull; }
    return h;
}

static int vocab_find(const SaltTokenizer *t, const char *s, size_t n) {
    if (!t->vcap) return -1;
    size_t i = (size_t)(fnv(s, n) & (uint64_t)(t->vcap - 1));
    while (t->vkeys[i]) {
        if (strlen(t->vkeys[i]) == n && !memcmp(t->vkeys[i], s, n))
            return t->vids[i];
        i = (i + 1) & (size_t)(t->vcap - 1);
    }
    return -1;
}

static void vocab_insert(SaltTokenizer *t, const char *s, size_t n, int id) {
    if (!t->vcap) return;
    size_t i = (size_t)(fnv(s, n) & (uint64_t)(t->vcap - 1));
    while (t->vkeys[i]) i = (i + 1) & (size_t)(t->vcap - 1);
    t->vkeys[i] = (char *)malloc(n + 1);
    if (!t->vkeys[i]) return;
    memcpy(t->vkeys[i], s, n);
    t->vkeys[i][n] = 0;
    t->vids[i] = id;
}

static void pair_put(SaltTokenizer *t, int a, int b, int rank, int merged) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0 && t->pkeys[i] != k) i = (i + 1) & (size_t)(t->pcap - 1);
    t->pkeys[i] = k;
    t->pranks[i] = rank;
    t->pmerged[i] = merged;
}

/* rank of the pair; INT32_MAX when absent */
static int pair_rank(const SaltTokenizer *t, int a, int b) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0) {
        if (t->pkeys[i] == k) return t->pranks[i];
        i = (i + 1) & (size_t)(t->pcap - 1);
    }
    return INT32_MAX;
}

static int pair_merged(const SaltTokenizer *t, int a, int b) {
    uint64_t k = ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
    size_t i = (size_t)(fnv((const char *)&k, 8) & (uint64_t)(t->pcap - 1));
    while (t->pkeys[i] != 0) {
        if (t->pkeys[i] == k) return t->pmerged[i];
        i = (i + 1) & (size_t)(t->pcap - 1);
    }
    return -1;
}

static int json_string_is(const JEntry *e, const char *value) {
    size_t n = strlen(value);
    return e && e->type == 1 &&
           (size_t)(e->str_end - e->str) == n &&
           memcmp(e->str, value, n) == 0;
}

static int json_object_string_is(const JEntry *obj, const char *key,
                                 const char *value) {
    return obj && obj->type == 2 &&
           json_string_is(json_get(obj->child, obj->nchild, key), value);
}

static int json_pattern_is(const JEntry *obj, const char *value) {
    const JEntry *pattern = obj && obj->type == 2
        ? json_get(obj->child, obj->nchild, "pattern") : NULL;
    return pattern && pattern->type == 2 &&
           json_object_string_is(pattern, "String", value);
}

/* Return 1 for the exact Gemma tokenizer pipeline, 0 for the established
 * GPT-2 byte-level path, and -1 for a partially matching byte-fallback
 * pipeline that this loader must not silently misinterpret. */
static int tokenizer_mode(const JDoc *doc, const JEntry *model) {
    const JEntry *normalizer = json_get(doc->root, doc->nroot, "normalizer");
    const JEntry *pre = json_get(doc->root, doc->nroot, "pre_tokenizer");
    const JEntry *decoder = json_get(doc->root, doc->nroot, "decoder");
    const JEntry *byte_fallback = json_get(model->child, model->nchild,
                                           "byte_fallback");
    const JEntry *fuse_unk = json_get(model->child, model->nchild,
                                      "fuse_unk");
    int requests_byte_fallback = byte_fallback && byte_fallback->type == 4 &&
                                 byte_fallback->inum == 1;
    int requests_metaspace = normalizer && normalizer->type == 2 &&
                             json_object_string_is(normalizer, "content", "▁");
    if (!requests_byte_fallback && !requests_metaspace)
        return SALT_TOKENIZER_GPT2_BYTE_LEVEL;

    int exact = json_object_string_is(model, "type", "BPE") &&
                requests_byte_fallback && fuse_unk && fuse_unk->type == 4 &&
                fuse_unk->inum == 1 &&
                json_object_string_is(normalizer, "type", "Replace") &&
                json_pattern_is(normalizer, " ") &&
                json_object_string_is(pre, "type", "Split") &&
                json_pattern_is(pre, " ") &&
                json_object_string_is(pre, "behavior", "MergedWithPrevious");
    const JEntry *invert = pre && pre->type == 2
        ? json_get(pre->child, pre->nchild, "invert") : NULL;
    exact = exact && invert && invert->type == 4 && invert->inum == 0 &&
            json_object_string_is(decoder, "type", "Sequence");
    const JEntry *decoders = decoder && decoder->type == 2
        ? json_get(decoder->child, decoder->nchild, "decoders") : NULL;
    if (!decoders || decoders->type != 3 || decoders->nchild != 3)
        exact = 0;
    if (exact) {
        const JEntry *replace = &decoders->child[0];
        const JEntry *fallback = &decoders->child[1];
        const JEntry *fuse = &decoders->child[2];
        exact = json_object_string_is(replace, "type", "Replace") &&
                json_pattern_is(replace, "▁") &&
                json_object_string_is(replace, "content", " ") &&
                json_object_string_is(fallback, "type", "ByteFallback") &&
                json_object_string_is(fuse, "type", "Fuse");
    }
    return exact ? SALT_TOKENIZER_METASPACE_BYTE_FALLBACK : -1;
}

/* ---------------- load ---------------------------------------------- */
int salt_tokenizer_load(SaltTokenizer *t, const char *path) {
    memset(t, 0, sizeof *t);
    byte_tables(t);

    /* a directory arg means "find the tokenizer in here": try
     * tokenizer.json first (HF byte-level BPE) */
    char auto_path[4096];
    struct stat st;
    if (stat(path, &st) == 0 && S_ISDIR(st.st_mode)) {
        snprintf(auto_path, sizeof auto_path, "%s/tokenizer.json", path);
        path = auto_path;
    }

    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tokenizer: cannot read %s\n", path); return -1; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 2 || sz > (1L << 30)) { fclose(f); return -1; }
    char *buf = (char *)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    buf[sz] = 0;

    JDoc *doc = json_parse(buf, (size_t)sz);
    if (!doc) { fprintf(stderr, "tokenizer: bad json\n"); free(buf); return -1; }
    const JEntry *model = json_get(doc->root, doc->nroot, "model");
    if (!model || model->type != 2) {
        fprintf(stderr, "tokenizer: no model object\n");
        json_free(doc); free(buf); return -1;
    }
    t->mode = tokenizer_mode(doc, model);
    if (t->mode < 0) {
        fprintf(stderr, "tokenizer: unsupported byte-fallback pipeline\n");
        json_free(doc); free(buf); return -1;
    }

    /* vocab: first pass counts, then size the hash, then insert */
    const JEntry *vocab = json_get(model->child, model->nchild, "vocab");
    const JEntry *merges = json_get(model->child, model->nchild, "merges");
    if (!vocab || vocab->type != 2 || !merges || merges->type != 3) {
        fprintf(stderr, "tokenizer: no vocab/merges\n");
        json_free(doc); free(buf); return -1;
    }
    int nv = vocab->nchild;
    t->nvocab = nv;
    t->vocab = (char **)calloc((size_t)nv, sizeof(char *));
    if (!t->vocab) { json_free(doc); free(buf); return -1; }
    int vcap = 16;
    while (vcap < nv * 2) vcap <<= 1;
    t->vcap = vcap;
    t->vkeys = (char **)calloc((size_t)vcap, sizeof(char *));
    t->vids = (int *)calloc((size_t)vcap, sizeof(int));
    if (!t->vkeys || !t->vids) { json_free(doc); free(buf); return -1; }

    for (int i = 0; i < nv; i++) {
        const JEntry *e = &vocab->child[i];
        size_t kl = (size_t)(e->key_end - e->key);
        int id = (int)e->inum;
        if (id < 0 || id >= nv) continue;
        size_t token_len = 0;
        char *token = json_unescape(e->key, kl, &token_len);
        if (!token) { json_free(doc); free(buf); return -1; }
        if (t->vocab[id]) { free(t->vocab[id]); }
        t->vocab[id] = token;
        vocab_insert(t, token, token_len, id);
    }

    int nm = merges->nchild;
    int pcap = 16;
    while (pcap < nm * 2) pcap <<= 1;
    t->pcap = pcap;
    t->pkeys = (uint64_t *)calloc((size_t)pcap, sizeof(uint64_t));
    t->pranks = (int32_t *)malloc((size_t)pcap * sizeof(int32_t));
    t->pmerged = (int32_t *)malloc((size_t)pcap * sizeof(int32_t));
    if (!t->pkeys || !t->pranks || !t->pmerged) { json_free(doc); free(buf); return -1; }

    for (int i = 0; i < nm; i++) {
        const JEntry *m = &merges->child[i];
        char *merge_string = NULL, *left = NULL, *right = NULL;
        const char *ls = NULL, *rs = NULL;
        size_t ll = 0, rl = 0;
        if (m->type == 1) {
            /* HF byte-level BPE: merge = single space-separated string
             * ("i n"). Split on the FIRST space (tokens never contain
             * a literal space -- the byte-level encoding maps spaces
             * to U+0120 "Ġ"). */
            size_t sl = 0;
            merge_string = json_unescape(m->str,
                (size_t)(m->str_end - m->str), &sl);
            if (!merge_string) { json_free(doc); free(buf); return -1; }
            char *sp = (char *)memchr(merge_string, ' ', sl);
            if (!sp) { free(merge_string); continue; }
            *sp = 0;
            ls = merge_string; ll = (size_t)(sp - merge_string);
            rs = sp + 1; rl = sl - ll - 1;
        } else if (m->type == 3 && m->nchild >= 2) {
            /* array-of-strings form ("[left, right]") */
            const JEntry *l = &m->child[0], *r = &m->child[1];
            if (l->type != 1 || r->type != 1) continue;
            left = json_unescape(l->str, (size_t)(l->str_end - l->str), &ll);
            right = json_unescape(r->str, (size_t)(r->str_end - r->str), &rl);
            if (!left || !right) {
                free(left); free(right);
                json_free(doc); free(buf); return -1;
            }
            ls = left; rs = right;
        } else {
            continue;
        }
        int li = vocab_find(t, ls, ll);
        int ri = vocab_find(t, rs, rl);
        if (li < 0 || ri < 0) {
            free(merge_string); free(left); free(right);
            continue;
        }
        char *both = (char *)malloc(ll + rl + 1);
        if (!both) {
            free(merge_string); free(left); free(right);
            json_free(doc); free(buf); return -1;
        }
        memcpy(both, ls, ll);
        memcpy(both + ll, rs, rl);
        both[ll + rl] = 0;
        int mi = vocab_find(t, both, ll + rl);
        free(both);
        if (mi >= 0) pair_put(t, li, ri, i, mi);
        free(merge_string); free(left); free(right);
    }

    /* added_tokens: special tokens like <|im_start|>. HF matches these
     * VERBATIM (longest-first) before BPE -- the engine must too, or
     * chat-format prompts get BPE-split into garbage (e.g. the chars of
     * "<|im_start|>" instead of its single id). */
    const JEntry *added = json_get(doc->root, doc->nroot, "added_tokens");
    t->nadded = 0;
    if (added && added->type == 3) {
        for (int i = 0; i < added->nchild && t->nadded < 64; i++) {
            const JEntry *e = &added->child[i];
            if (e->type != 2) continue;
            const JEntry *ce = json_get(e->child, e->nchild, "content");
            const JEntry *ie = json_get(e->child, e->nchild, "id");
            if (!ce || ce->type != 1 || !ie || ie->type != 0) continue;
            size_t cl = (size_t)(ce->str_end - ce->str);
            int id = (int)ie->inum;
            if (cl == 0 || id < 0) continue;
            size_t content_len = 0;
            char *cp = json_unescape(ce->str, cl, &content_len);
            if (!cp) continue;
            t->added[t->nadded] = cp;
            t->added_id[t->nadded] = id;
            t->added_len[t->nadded] = content_len;
            t->nadded++;
        }
        /* sort by decreasing content length (longest-first match) */
        for (int i = 0; i < t->nadded; i++) {
            for (int j = i + 1; j < t->nadded; j++) {
                if (t->added_len[j] > t->added_len[i]) {
                    char *tmpc = t->added[i]; t->added[i] = t->added[j];
                    t->added[j] = tmpc;
                    int tmpi = t->added_id[i]; t->added_id[i] = t->added_id[j];
                    t->added_id[j] = tmpi;
                    size_t tmpl = t->added_len[i];
                    t->added_len[i] = t->added_len[j];
                    t->added_len[j] = tmpl;
                }
            }
        }
        if (getenv("SALT_DEBUG_TOK")) {
            fprintf(stderr, "[tok] %d added tokens:\n", t->nadded);
            for (int i = 0; i < t->nadded && i < 40; i++)
                fprintf(stderr, "  %d len=%zu id=%d %.30s\n", i,
                        t->added_len[i], t->added_id[i], t->added[i]);
        }
    }

    json_free(doc);
    free(buf);
    return 0;
}

int salt_tokenizer_load_fd(SaltTokenizer *t, int fd) {
    char path[64];
    int n;
    if (!t || fd < 0) return -1;
#if defined(__APPLE__)
    n = snprintf(path, sizeof path, "/dev/fd/%d", fd);
#else
    n = snprintf(path, sizeof path, "/proc/self/fd/%d", fd);
#endif
    if (n < 1 || (size_t)n >= sizeof path) return -1;
    return salt_tokenizer_load(t, path);
}

void salt_tokenizer_free(SaltTokenizer *t) {
    if (t->vocab) {
        for (int i = 0; i < t->nvocab; i++) free(t->vocab[i]);
        free(t->vocab);
    }
    if (t->vkeys) {
        for (int i = 0; i < t->vcap; i++) free(t->vkeys[i]);
        free(t->vkeys);
    }
    free(t->vids);
    free(t->pkeys);
    free(t->pranks);
    free(t->pmerged);
    for (int i = 0; i < t->nadded; i++) free(t->added[i]);
    t->nadded = 0;
    memset(t, 0, sizeof *t);
}

/* ---------------- decode --------------------------------------------- */
static int hex_digit(unsigned char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int decode_metaspace_token(const char *token, char *out, int max) {
    size_t n = strlen(token);
    if (n == 6 && !memcmp(token, "<0x", 3) && token[5] == '>') {
        int hi = hex_digit((unsigned char)token[3]);
        int lo = hex_digit((unsigned char)token[4]);
        if (hi >= 0 && lo >= 0 && max > 0) {
            out[0] = (char)((hi << 4) | lo);
            return 1;
        }
    }
    int written = 0;
    const unsigned char *p = (const unsigned char *)token;
    while (*p && written < max) {
        if (p[0] == 0xE2 && p[1] == 0x96 && p[2] == 0x81) {
            out[written++] = ' ';
            p += 3;
        } else {
            out[written++] = (char)*p++;
        }
    }
    return written;
}

int salt_tokenizer_decode(const SaltTokenizer *t, const int *ids, int n,
                          char *out, int out_n) {
    int o = 0;
    for (int i = 0; i < n && o < out_n; i++) {
        /* added/special tokens (ids above the base vocab, e.g.
         * <|im_end|>): emit their literal content -- HF decodes
         * these verbatim, they are NOT in the base vocab table. */
        int ai = -1;
        for (int a = 0; a < t->nadded; a++)
            if (t->added_id[a] == ids[i]) { ai = a; break; }
        if (ai >= 0) {
            for (size_t k = 0; k < t->added_len[ai] && o < out_n; k++)
                out[o++] = t->added[ai][k];
            continue;
        }
        if (ids[i] < 0 || ids[i] >= t->nvocab || !t->vocab[ids[i]]) return -1;
        if (t->mode == SALT_TOKENIZER_METASPACE_BYTE_FALLBACK) {
            o += decode_metaspace_token(t->vocab[ids[i]], out + o, out_n - o);
            continue;
        }
        const unsigned char *p = (const unsigned char *)t->vocab[ids[i]];
        while (*p && o < out_n) {
            uint32_t cp = utf8_get(&p);
            if (cp < 0x180 && t->rev[cp] != 0xFF) {
                out[o++] = (char)t->rev[cp];
            } else {
                o += utf8_put(cp, out + o, out_n - o);
            }
        }
    }
    if (o < out_n) out[o] = 0;
    else out[out_n - 1] = 0;
    return o;
}

/* ---------------- encode --------------------------------------------- */
static size_t utf8_width_bounded(const unsigned char *p, size_t n) {
    if (n == 0 || p[0] < 0x80) return n ? 1 : 0;
    size_t want = (p[0] & 0xE0) == 0xC0 ? 2 :
                  (p[0] & 0xF0) == 0xE0 ? 3 :
                  (p[0] & 0xF8) == 0xF0 ? 4 : 1;
    if (want > n) return 1;
    for (size_t i = 1; i < want; i++)
        if ((p[i] & 0xC0) != 0x80) return 1;
    return want;
}

static int bpe_merge_ids(const SaltTokenizer *t, int *seq, int ns) {
    while (ns > 1) {
        int bi = -1, br = INT32_MAX;
        for (int i = 0; i + 1 < ns; i++) {
            int r = pair_rank(t, seq[i], seq[i + 1]);
            if (r < br) { br = r; bi = i; }
        }
        if (bi < 0) break;
        int left = seq[bi], right = seq[bi + 1];
        int merged = pair_merged(t, left, right);
        if (merged < 0) break;
        int out = 0;
        for (int i = 0; i < ns; ) {
            if (i + 1 < ns && seq[i] == left && seq[i + 1] == right) {
                seq[out++] = merged;
                i += 2;
            } else {
                seq[out++] = seq[i++];
            }
        }
        ns = out;
    }
    return ns;
}

static int added_at(const SaltTokenizer *t, const unsigned char *p,
                    size_t available, size_t *matched_len) {
    int matched = -1;
    *matched_len = 0;
    for (int a = 0; a < t->nadded; a++) {
        size_t n = t->added_len[a];
        if (n <= available && n > *matched_len &&
            memcmp(p, t->added[a], n) == 0) {
            matched = a;
            *matched_len = n;
        }
    }
    return matched;
}

static int encode_metaspace_span(const SaltTokenizer *t,
                                 const unsigned char *p, size_t n,
                                 int *ids, int max_ids) {
    if (n == 0 || max_ids <= 0) return 0;
    int *seq = (int *)malloc(n * sizeof(int));
    if (!seq) return -1;
    int ns = 0;
    for (size_t off = 0; off < n; ) {
        if (p[off] == ' ') {
            int id = vocab_find(t, "▁", strlen("▁"));
            if (id < 0) { free(seq); return -1; }
            seq[ns++] = id;
            off++;
            continue;
        }
        size_t width = utf8_width_bounded(p + off, n - off);
        int id = vocab_find(t, (const char *)p + off, width);
        if (id >= 0) {
            seq[ns++] = id;
        } else {
            for (size_t i = 0; i < width; i++) {
                char byte_token[7];
                snprintf(byte_token, sizeof byte_token, "<0x%02X>", p[off + i]);
                id = vocab_find(t, byte_token, 6);
                if (id < 0) { free(seq); return -1; }
                seq[ns++] = id;
            }
        }
        off += width;
    }
    ns = bpe_merge_ids(t, seq, ns);
    if (ns > max_ids) ns = max_ids;
    memcpy(ids, seq, (size_t)ns * sizeof(int));
    free(seq);
    return ns;
}

static int encode_metaspace(const SaltTokenizer *t, const char *utf8,
                            int *ids, int max_ids) {
    const unsigned char *p = (const unsigned char *)utf8;
    size_t left = strlen(utf8);
    int total = 0;
    while (left > 0 && total < max_ids) {
        size_t special_len = 0;
        int special = added_at(t, p, left, &special_len);
        if (special >= 0) {
            ids[total++] = t->added_id[special];
            p += special_len;
            left -= special_len;
            continue;
        }
        size_t span = 1;
        while (span < left) {
            size_t ignored = 0;
            if (added_at(t, p + span, left - span, &ignored) >= 0) break;
            span++;
        }
        int count = encode_metaspace_span(t, p, span, ids + total,
                                          max_ids - total);
        if (count < 0) return -1;
        total += count;
        p += span;
        left -= span;
    }
    return total;
}

int salt_tokenizer_encode(const SaltTokenizer *t, const char *utf8,
                          int *ids, int max_ids) {
    if (t->mode == SALT_TOKENIZER_METASPACE_BYTE_FALLBACK)
        return encode_metaspace(t, utf8, ids, max_ids);
    const unsigned char *p = (const unsigned char *)utf8;
    int n = 0, first = 1;

    /* added-token helper: match at position p, longest first */
    while (*p && n < max_ids) {
        int matched = -1;
        size_t mlen = 0;
        for (int a = 0; a < t->nadded; a++) {
            size_t al = t->added_len[a];
            if (al > mlen && memcmp(p, t->added[a], al) == 0) {
                matched = a;
                mlen = al;
            }
        }
        if (getenv("SALT_DEBUG_TOK") && *p == '<') {
            fprintf(stderr, "[tokenc] at '<' nadded=%d matched=%d mlen=%zu "
                    "ctx: %.12s\n", t->nadded, matched, mlen, (const char *)p);
        }
        if (matched >= 0) {
            ids[n++] = t->added_id[matched];
            p += mlen;
            first = 0;
            continue;
        }
        /* newline: HF emits the "\n" char directly into the byte-level
         * stream (token "Ċ", id 198) -- it is NOT a space and must not
         * become the "Ġ " prefix. */
        if (*p == '\n') {
            char cbuf[8];
            int cl = utf8_put(t->fwd[0x0A], cbuf, 8);
            int id = vocab_find(t, cbuf, (size_t)cl);
            if (id < 0) return -1;
            ids[n++] = id;
            p++;
            first = 0;
            continue;
        }
        size_t sp = 0;
        while (p[sp] == ' ' || p[sp] == '\t' ||
               p[sp] == '\r') sp++;
        size_t start = sp;
        /* stop the word at any position where an added (special) token
         * starts -- otherwise "is<|im_end|>" becomes ONE BPE word and the
         * special token gets split into its literal chars. */
        while (p[start] && p[start] != ' ' && p[start] != '\t' &&
               p[start] != '\n' && p[start] != '\r') {
            int at_added = 0;
            for (int a = 0; a < t->nadded && !at_added; a++)
                if (memcmp(p + start, t->added[a], t->added_len[a]) == 0)
                    at_added = 1;
            if (at_added) break;
            start++;
        }
        if (start == 0) break;              /* no word left */
        /* word = p[0..start): [spaces][non-space run] */
        int wids[1024], wi = 0;
        /* leading spaces: byte 0x20 maps to its unicode char (U+0120,
         * the byte-level BPE space) */
        if (!first || sp > 0) {
            for (size_t i = 0; i < sp && wi < 1024; i++) {
                char cbuf[8];
                int cl = utf8_put(t->fwd[0x20], cbuf, 8);
                int id = vocab_find(t, cbuf, (size_t)cl);
                if (id < 0) return -1;
                wids[wi++] = id;
            }
        }
        /* non-space chars via the forward byte->char table */
        for (size_t i = sp; i < start && wi < 1024; i++) {
            uint32_t cp = t->fwd[p[i]];
            char cbuf[8];
            int cl = utf8_put(cp, cbuf, 8);
            int id = vocab_find(t, cbuf, (size_t)cl);
            if (id < 0) return -1;
            wids[wi++] = id;
        }
        /* BPE by lowest merge rank, merge all occurrences */
        int seq[1024], ns = 0;
        for (int i = 0; i < wi && ns < 1024; i++) seq[ns++] = wids[i];
        while (ns > 1) {
            int bi = -1, br = INT32_MAX;
            for (int i = 0; i + 1 < ns; i++) {
                int r = pair_rank(t, seq[i], seq[i + 1]);
                if (r < br) { br = r; bi = i; }
            }
            if (bi < 0) break;
            int l = seq[bi], r = seq[bi + 1];
            int m = pair_merged(t, l, r);
            if (m < 0) break;
            int nw2 = 0;
            for (int i = 0; i < ns; ) {
                if (i + 1 < ns && seq[i] == l && seq[i + 1] == r) {
                    seq[nw2++] = m;
                    i += 2;
                } else {
                    seq[nw2++] = seq[i++];
                }
            }
            ns = nw2;
        }
        if (ns > max_ids - n) ns = max_ids - n;
        for (int i = 0; i < ns; i++) ids[n + i] = seq[i];
        n += ns;
        p += start;
        first = 0;
    }
    return n;
}
