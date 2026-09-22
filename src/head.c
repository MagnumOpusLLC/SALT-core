/* head.c -- output head + embedding + sampling (issue #6 step 3). */
#include "salt/head.h"
#include "salt/kernels.h"
#include "salt/gpu.h"
#include "salt/bitmath.h"
#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* GPU path is opt-in (--gpu / SALT_GPU=1). Init lazily on first use;
 * if Metal is unavailable the call returns -1 and we fall back to CPU
 * for the rest of the run. CPU stays the default (byte-deterministic). */
static int gpu_tried = 0, gpu_ok = 0;
static int gpu_want(void) {
    const char *e = getenv("SALT_GPU");
    return e && *e && strcmp(e, "0") != 0;
}

static int gpu_trunk_want(void) {
    const char *e = getenv("SALT_GPU_TRUNK");
    return gpu_want() && e && *e && strcmp(e, "0") != 0;
}

static long read_file_buf(const char *path, uint8_t **out) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long sz = ftell(f);
    if (sz <= 0 || sz > (1L << 33)) { fclose(f); return -1; }
    rewind(f);
    uint8_t *b = (uint8_t *)malloc((size_t)sz);
    if (!b) { fclose(f); return -1; }
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b); fclose(f); return -1;
    }
    fclose(f);
    *out = b;
    return sz;
}

static long jnum(const JEntry *e, long dflt) {
    return e ? (long)e->inum : dflt;
}

static int jshape(const JEntry *e, long *dims, int *rank) {
    *rank = 0;
    if (!e || e->type != 3) return -1;
    int n = e->nchild;
    if (n > 4) n = 4;
    for (int i = 0; i < n; i++) dims[i] = (long)e->child[i].inum;
    *rank = n;
    return 0;
}

int salt_head_load(SaltHead *h, const char *json_path) {
    memset(h, 0, sizeof *h);
    long jlen;
    char *js = NULL;
    {
        uint8_t *tmp;
        jlen = read_file_buf(json_path, &tmp);
        if (jlen < 0) {
            fprintf(stderr, "head: cannot read %s\n", json_path);
            return -1;
        }
        js = (char *)tmp;
    }
    JDoc *doc = json_parse(js, (size_t)jlen);
    if (!doc) { free(js); return -1; }
    const JEntry *bin = json_get(doc->root, doc->nroot, "bin");
    const JEntry *w = json_get(doc->root, doc->nroot, "weight");
    const JEntry *s = json_get(doc->root, doc->nroot, "scale");
    if (!bin || bin->type != 1) { json_free(doc); free(js); return -1; }
    /* weight layout: nested object {off,nbytes,dtype,shape} (MLX
     * triplet) or flat top-level keys (legacy). */
    int w_nested = (w && w->type == 2);
    /* json_get takes the CHILD ARRAY: nested objects use w->child,
     * flat layout uses doc->root itself */
    const JEntry *warr = w_nested ? w->child : doc->root;
    int wcnt = w_nested ? w->nchild : doc->nroot;
    /* scale is OPTIONAL (a checkpoint may keep the head unquantized);
     * null/absent means no scales -> 1.0 everywhere */
    if (s && s->type == 2 && s->child) {
        h->s_off = jnum(json_get(s->child, s->nchild, "off"), 0);
        h->s_nbytes = jnum(json_get(s->child, s->nchild, "nbytes"), 0);
        jshape(json_get(s->child, s->nchild, "shape"),
               h->sdims, &h->srank);
    }
    size_t bl = (size_t)(bin->str_end - bin->str);
    char bin_path[4096];
    if (bl >= sizeof bin_path) { json_free(doc); free(js); return -1; }
    memcpy(bin_path, bin->str, bl);
    bin_path[bl] = 0;
    /* resolve relative to the json's directory (in place, back-to-front
     * so no overlap: shift the name right, then copy the dir prefix) */
    {
        char *slash = strrchr(json_path, '/');
        if (slash) {
            size_t dl = (size_t)(slash - json_path);
            if (dl + bl + 1 > sizeof bin_path) {
                json_free(doc); free(js); return -1;
            }
            memmove(bin_path + dl + 1, bin_path, bl + 1);
            memcpy(bin_path, json_path, dl);
            bin_path[dl] = '/';
        }
    }
    h->w_off = jnum(json_get(warr, wcnt, "off"), 0);
    h->w_nbytes = jnum(json_get(warr, wcnt, "nbytes"), 0);
    h->w_bits = (int)jnum(json_get(warr, wcnt, "bits"), 0);
    if (h->w_bits != 4 && h->w_bits != 8) h->w_bits = 4;
    const JEntry *wdt = json_get(warr, wcnt, "dtype");
    if (wdt && wdt->type == 1) {
        size_t wn = (size_t)(wdt->str_end - wdt->str);
        if (wn == 7 && !memcmp(wdt->str, "F8_E4M3", 7)) h->w_dtype = 2;
        else if (wn == 3 && !memcmp(wdt->str, "F32", 3)) h->w_dtype = 0;
        else if (wn == 4 && !memcmp(wdt->str, "BF16", 4)) h->w_dtype = 4;
        else if (wn == 2 && !memcmp(wdt->str, "I8", 2)) h->w_dtype = 1;
        else if ((wn == 3 && !memcmp(wdt->str, "F16", 3)) ||
                 (wn == 4 && !memcmp(wdt->str, "FP16", 4))) h->w_dtype = 5;
        else if (wn == 3 && !memcmp(wdt->str, "U32", 3)) h->w_dtype = 6;
        else h->w_dtype = 3;
    }
    jshape(json_get(warr, wcnt, "shape"), h->dims, &h->rank);
    /* MLX bias triplet (per-64-group BF16), optional like scale */
    {
        const JEntry *b = json_get(doc->root, doc->nroot, "bias");
        if (b && b->type == 2 && b->child) {
            h->b_off = jnum(json_get(b->child, b->nchild, "off"), 0);
            h->b_nbytes = jnum(json_get(b->child, b->nchild, "nbytes"), 0);
        }
    }
    json_free(doc);
    free(js);

    h->buf_n = read_file_buf(bin_path, &h->buf);
    if (h->buf_n < 0 || h->w_off + h->w_nbytes > h->buf_n ||
        h->s_off + h->s_nbytes > h->buf_n) {
        fprintf(stderr, "head: %s too small for layout\n", bin_path);
        salt_head_free(h);
        return -1;
    }
    if (h->w_dtype != 0 && h->w_dtype != 1 && h->w_dtype != 2 &&
        h->w_dtype != 4 && h->w_dtype != 5 && h->w_dtype != 6) {
        fprintf(stderr,
                "head: weight dtype \"%.*s\" unsupported "
                "(F32/I8/F8_E4M3/BF16/F16/U32)\n",
                wdt ? (int)(wdt->str_end - wdt->str) : 0,
                wdt ? wdt->str : "?");
        salt_head_free(h);
        return -1;
    }
    return 0;
}

void salt_head_free(SaltHead *h) {
    free(h->buf);
    memset(h, 0, sizeof *h);
}

int salt_embed_load(SaltEmbed *e, const char *json_path) {
    memset(e, 0, sizeof *e);
    long jlen;
    char *js = NULL;
    {
        uint8_t *tmp;
        jlen = read_file_buf(json_path, &tmp);
        if (jlen < 0) return -1;
        js = (char *)tmp;
    }
    JDoc *doc = json_parse(js, (size_t)jlen);
    if (!doc) { free(js); return -1; }
    const JEntry *bin = json_get(doc->root, doc->nroot, "bin");
    const JEntry *w = json_get(doc->root, doc->nroot, "weight");
    if (!bin || bin->type != 1) {
        json_free(doc); free(js); return -1;
    }
    int w_nested = (w && w->type == 2);
    /* json_get takes the CHILD ARRAY: nested objects use w->child,
     * flat layout uses doc->root itself */
    const JEntry *warr = w_nested ? w->child : doc->root;
    int wcnt = w_nested ? w->nchild : doc->nroot;
    const JEntry *dt = json_get(warr, wcnt, "dtype");
    e->w_bits = (int)jnum(json_get(warr, wcnt, "bits"), 0);
    if (e->w_bits != 4 && e->w_bits != 8) e->w_bits = 4;
    jshape(json_get(warr, wcnt, "shape"), e->dims, &e->rank);
    const JEntry *sc = json_get(doc->root, doc->nroot, "scale");
    if (sc && sc->type == 2 && sc->child) {
        e->s_off = jnum(json_get(sc->child, sc->nchild, "off"), 0);
        e->s_nbytes = jnum(json_get(sc->child, sc->nchild, "nbytes"), 0);
        jshape(json_get(sc->child, sc->nchild, "shape"),
               e->sdims, &e->srank);
    }
    {
        const JEntry *b = json_get(doc->root, doc->nroot, "bias");
        if (b && b->type == 2 && b->child) {
            e->b_off = jnum(json_get(b->child, b->nchild, "off"), 0);
            e->b_nbytes = jnum(json_get(b->child, b->nchild, "nbytes"), 0);
        }
    }
    e->dtype = 3;
    if (dt && dt->type == 1) {
        size_t dn = (size_t)(dt->str_end - dt->str);
        if (dn == 3 && !memcmp(dt->str, "F32", 3)) e->dtype = 0;
        else if (dn == 4 && !memcmp(dt->str, "BF16", 4)) e->dtype = 4;
        else if (dn == 2 && !memcmp(dt->str, "I8", 2)) e->dtype = 1;
        else if ((dn == 3 && !memcmp(dt->str, "F16", 3)) ||
                 (dn == 4 && !memcmp(dt->str, "FP16", 4))) e->dtype = 5;
        else if (dn == 7 && !memcmp(dt->str, "F8_E4M3", 7)) e->dtype = 2;
        else if (dn == 3 && !memcmp(dt->str, "U32", 3)) e->dtype = 6;
    }
    /* copy the bin path BEFORE freeing the json buffer */
    {
        size_t bl = (size_t)(bin->str_end - bin->str);
        char bin_path[4096];
        if (bl >= sizeof bin_path) { json_free(doc); free(js); return -1; }
        memcpy(bin_path, bin->str, bl);
        bin_path[bl] = 0;
        json_free(doc);
        free(js);
        if (e->dtype != 0 && e->dtype != 1 && e->dtype != 2 &&
            e->dtype != 4 && e->dtype != 5 && e->dtype != 6) {
            fprintf(stderr,
                    "embed: dtype %d \"%.*s\" unsupported "
                    "(F32/I8/F8_E4M3/BF16/F16/U32)\n",
                    e->dtype,
                    dt ? (int)(dt->str_end - dt->str) : 0,
                    dt ? dt->str : "?");
            return -1;
        }
        char *slash = strrchr(json_path, '/');
        if (slash) {
            size_t dl = (size_t)(slash - json_path);
            if (dl + bl + 1 > sizeof bin_path) return -1;
            memmove(bin_path + dl + 1, bin_path, bl + 1);
            memcpy(bin_path, json_path, dl);
            bin_path[dl] = '/';
        }
        e->buf_n = read_file_buf(bin_path, &e->buf);
    }
    if (e->buf_n < 0) { salt_embed_free(e); return -1; }
    return 0;
}

void salt_embed_free(SaltEmbed *e) {
    free(e->buf);
    memset(e, 0, sizeof *e);
}

int salt_head_logits(const SaltHead *h, const float *state, float *logits) {
    if (!h || !h->buf || h->rank != 2) {
        fprintf(stderr, "head logits: h=%p buf=%p rank=%d\n",
                (const void *)h, h ? (const void *)h->buf : NULL,
                h ? h->rank : -99);
        return -1;
    }
    long V = h->dims[0], H = h->dims[1];
    const uint8_t *scales = NULL;
    int SR = 1, SC = 1;
    if (h->s_nbytes > 0) {
        scales = h->buf + h->s_off;
        if (h->srank == 2) {
            SR = (int)h->sdims[0];
            SC = (int)h->sdims[1];
        } else if (h->srank == 1) {
            SC = (int)h->sdims[0];
        }
    }
    if (h->w_dtype == 2) {
        salt_f8_matvec(h->buf + h->w_off, scales,
                       (int)V, (int)H, SR, SC, state, logits);
    } else if (h->w_dtype == 6) {
        /* MLX affine head: U32 words + BF16 scale/bias per 64-group.
         * dims are PACKED cols; decoded H = dims[1] * (32/bits) --
         * bits follows the tensor (4-bit default, 8-bit on repos
         * that quantize the head wider). */
        long Hdec = H * (32 / (long)h->w_bits);
        const uint16_t *bias = h->b_nbytes > 0
            ? (const uint16_t *)(const void *)(h->buf + h->b_off) : NULL;
        /* The separately loaded head is not part of a canonical trunk layer
         * lease.  In trunk mapped-only mode it remains explicitly CPU-owned
         * rather than using an anonymous Metal weight buffer. */
        if (!gpu_trunk_want() && !salt_gpu_mapped_only() &&
            gpu_want() && !gpu_tried) {
            gpu_tried = 1;
            gpu_ok = salt_gpu_init() == 0;
        }
        if (!gpu_trunk_want() && !salt_gpu_mapped_only() && gpu_ok) {
            if (salt_gpu_q4_matvec(
                    (const uint32_t *)(const void *)(h->buf + h->w_off),
                    (const uint16_t *)(const void *)(h->buf + h->s_off),
                    bias, (int)V, (int)Hdec, state, logits) == 0)
                return 0;
            gpu_ok = 0;          /* fell through: use CPU from now on */
        }
        salt_q4_matvec(
            (const uint32_t *)(const void *)(h->buf + h->w_off),
            (const uint16_t *)(const void *)(h->buf + h->s_off),
            bias, (int)V, (int)Hdec, state, logits);
    } else if (h->w_dtype == 1) {
        salt_i8_matvec(h->buf + h->w_off, scales,
                       (int)V, (int)H, SR, SC, state, logits);
    } else if (h->w_dtype == 0) {
        salt_f32_matvec((const float *)(const void *)(h->buf + h->w_off),
                        (int)V, (int)H, state, logits);
    } else if (h->w_dtype == 5) {
        salt_f16_matvec((const uint16_t *)(const void *)(h->buf + h->w_off),
                        (int)V, (int)H, state, logits);
    } else {
        salt_bf16_matvec((const uint16_t *)(const void *)(h->buf + h->w_off),
                         (int)V, (int)H, state, NULL, logits);
    }
    return 0;
}

int salt_argmax(const float *logits, int V) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

int salt_embed_gather(const SaltEmbed *e, int tok, float *out) {
    if (!e || !e->buf || e->rank != 2) return -1;
    long V = e->dims[0], H = e->dims[1];
    if (tok < 0 || tok >= V) return -1;
    size_t esz = e->dtype == 0 ? 4 : (e->dtype == 4 || e->dtype == 5) ? 2 : 1;
    if (e->dtype == 6) esz = 4;          /* U32 packed words */
    const uint8_t *row = e->buf + (size_t)tok * H * esz;
    if (e->dtype == 0) {
        memcpy(out, row, (size_t)H * sizeof(float));
    } else if (e->dtype == 2) {
        const uint8_t *scales = NULL;
        int SR = 1, SC = 1;
        if (e->s_nbytes > 0) {
            scales = e->buf + e->s_off;
            if (e->srank == 2) {
                SR = (int)e->sdims[0];
                SC = (int)e->sdims[1];
            } else if (e->srank == 1) {
                SC = (int)e->sdims[0];
            }
        }
        salt_f8_decode_row(row, scales, (int)V, (int)H, SR, SC, 0, out);
    } else if (e->dtype == 1) {
        int SR = 1, SC = 1;
        const uint8_t *scales = NULL;
        if (e->s_nbytes > 0) {
            scales = e->buf + e->s_off;
            if (e->srank == 2) {
                SR = (int)e->sdims[0];
                SC = (int)e->sdims[1];
            } else if (e->srank == 1) {
                SC = (int)e->sdims[0];
            }
        }
        int sr = (int)(((int64_t)tok * SR) / V);
        for (int i = 0; i < (int)H; i++) {
            int sc = (int)(((int64_t)i * SC) / H);
            float s = scales ? salt_e8m0_value(scales[sr * SC + sc]) : 1.0f;
            out[i] = (float)(int8_t)row[i] * s;
        }
    } else if (e->dtype == 5) {
        const uint16_t *r16 = (const uint16_t *)(const void *)row;
        for (int i = 0; i < (int)H; i++) out[i] = salt_f16_to_f32(r16[i]);
    } else if (e->dtype == 6) {
        /* MLX affine embed: token row = U32 words, BF16 scale/bias
         * per 64-group. Packing follows the tensor bits: 4-bit = 8
         * elems/word, 8-bit = 4 elems/word. scales/biases are
         * [V x G] with G = decoded_cols/64. */
        int eb = e->w_bits > 0 ? e->w_bits : 4;
        if (eb != 4 && eb != 8) eb = 4;
        long Hdec = H * (32 / (long)eb);
        long G = Hdec / 64;
        const uint32_t *w32 = (const uint32_t *)(const void *)row;
        const uint16_t *scales = e->s_nbytes > 0
            ? (const uint16_t *)(const void *)(e->buf + e->s_off)
            : NULL;
        const uint16_t *bias = e->b_nbytes > 0
            ? (const uint16_t *)(const void *)(e->buf + e->b_off)
            : NULL;
        for (int i = 0; i < (int)Hdec; i++) {
            int q;
            if (eb == 8)
                q = (int)((w32[i >> 2] >> (8 * (i & 3))) & 0xFFu);
            else
                q = (int)((w32[i >> 3] >> (4 * (i & 7))) & 0xFu);
            int g = i / 64;
            float s = 1.0f, b = 0.0f;
            if (scales) {
                uint32_t sb = (uint32_t)scales[(size_t)tok * G + g] << 16;
                memcpy(&s, &sb, 4);
            }
            if (bias) {
                uint32_t bb = (uint32_t)bias[(size_t)tok * G + g] << 16;
                memcpy(&b, &bb, 4);
            }
            if (getenv("SALT_NAN_PROBE") && tok == 760 && i < 8)
                fprintf(stderr, "[embg] i=%d q=%d g=%d s=%.8g b=%.8g -> %.8g\n",
                        i, q, g, s, b, (float)q * s + b);
            out[i] = (float)q * s + b;
        }
    } else {
        const uint16_t *r16 = (const uint16_t *)(const void *)row;
        for (int i = 0; i < (int)H; i++) {
            uint32_t bits = (uint32_t)r16[i] << 16;
            memcpy(&out[i], &bits, 4);
        }
        if (getenv("SALT_NAN_PROBE") && tok < 3) {
            fprintf(stderr, "[embg] tok=%d dtype=%d esz=%zu H=%ld "
                    "r16[0..3]=%u %u %u %u out[0..3]=%.6g %.6g %.6g %.6g\n",
                    tok, e->dtype, esz, H,
                    (unsigned)r16[0], (unsigned)r16[1],
                    (unsigned)r16[2], (unsigned)r16[3],
                    out[0], out[1], out[2], out[3]);
        }
    }
    return 0;
}

int salt_sample(const float *logits, int V, uint64_t *rng) {
    /* softmax (max-subtract, fp32) */
    float mx = logits[0];
    for (int i = 1; i < V; i++)
        if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    float *w = (float *)malloc((size_t)V * sizeof(float));
    if (!w) return 0;
    for (int i = 0; i < V; i++) {
        w[i] = salt_expf(logits[i] - mx);
        sum += (double)w[i];
    }
    /* xorshift64 draw */
    uint64_t x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    double target = (double)(x >> 11) / 9007199254740992.0;  /* [0,1) */
    double acc = 0.0;
    int tok = 0;
    for (int i = 0; i < V; i++) {
        acc += (double)w[i] / sum;
        if (target < acc) { tok = i; break; }
    }
    free(w);
    return tok;
}

/* Nucleus (top-p) sampling with a bounded candidate window.
 *
 * logits in, p in (0,1]. Keeps the smallest set of top tokens whose
 * cumulative softmax mass >= p and samples within it. The candidate
 * window is capped at maxk (pass 0 for a safe default of 512): with a
 * tight p (0.8-0.95) the nucleus is far smaller than the 248K vocab,
 * which avoids the full-vocab softmax/malloc of salt_sample per token.
 *
 * Returns the sampled token id; falls back to argmax on allocation
 * failure (never returns an invalid id). */
int salt_sample_topp(const float *logits, int V, double p, uint64_t *rng,
                     int maxk) {
    if (V < 1) return 0;
    if (maxk <= 0 || maxk > V) maxk = V < 512 ? V : 512;
    /* collect the maxk highest-logit indices (partial selection) */
    int *idx = (int *)malloc((size_t)maxk * sizeof(int));
    float *w = (float *)malloc((size_t)maxk * sizeof(float));
    if (!idx || !w) { free(idx); free(w); return salt_argmax(logits, V); }
    for (int k = 0; k < maxk; k++) idx[k] = k;
    /* simple insertion sort of the top maxk by logit (descending) */
    for (int i = 0; i < maxk; i++) {
        int best = i;
        for (int j = i + 1; j < maxk; j++)
            if (logits[idx[j]] > logits[idx[best]]) best = j;
        int t = idx[i]; idx[i] = idx[best]; idx[best] = t;
    }
    float mx = logits[idx[0]];
    double sum = 0.0;
    int n = 0;
    for (int k = 0; k < maxk; k++) {
        double e = salt_expf(logits[idx[k]] - mx);
        sum += e;
        w[k] = (float)e;
        if (sum >= p) { n = k + 1; break; }
    }
    if (n < 1) n = maxk;
    /* xorshift64 draw within the nucleus */
    uint64_t x = *rng;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *rng = x;
    double target = (double)(x >> 11) / 9007199254740992.0 * sum;
    double acc = 0.0;
    int tok = idx[0];
    for (int k = 0; k < n; k++) {
        acc += w[k];
        if (target < acc) { tok = idx[k]; break; }
    }
    free(idx);
    free(w);
    return tok;
}

/* Additive frequency penalty (vLLM frequency_penalty semantics):
 * logits[i] -= freq_pen * count_i for tokens in the recent window.
 * Count array sized V; zeroed after use. O(V + n_recent). */
void salt_apply_freq_penalty(float *logits, int V, const int *recent,
                             int n_recent, float freq_pen) {
    if (!logits || V < 1 || !recent || n_recent < 1 || freq_pen <= 0.0f)
        return;
    int *cnt = (int *)calloc((size_t)V, sizeof(int));
    if (!cnt) return;
    for (int r = 0; r < n_recent; r++) {
        int t = recent[r];
        if (t >= 0 && t < V) cnt[t]++;
    }
    for (int i = 0; i < V; i++)
        if (cnt[i] > 0) logits[i] -= freq_pen * (float)cnt[i];
    free(cnt);
}

/* Presence penalty (vLLM semantics, the Qwen3.6 model card's
 * recommended loop-killer): logits[i] -= pres_pen ONCE for every
 * token that has appeared in the window, regardless of count.
 * Unlike frequency_penalty it does not scale with repetition, so
 * it discourages revisiting any seen token without crushing
 * legitimate re-use (the card: presence_penalty 1.5 thinking /
 * instruct general tasks, range 0-2). */
void salt_apply_presence_penalty(float *logits, int V, const int *recent,
                                 int n_recent, float pres_pen) {
    if (!logits || V < 1 || !recent || n_recent < 1 || pres_pen <= 0.0f)
        return;
    int *seen = (int *)calloc((size_t)V, sizeof(int));
    if (!seen) return;
    for (int r = 0; r < n_recent; r++) {
        int t = recent[r];
        if (t >= 0 && t < V) seen[t] = 1;
    }
    for (int i = 0; i < V; i++)
        if (seen[i]) logits[i] -= pres_pen;
    free(seen);
}

/* vLLM-style sampling: top-k -> top-p -> temperature -> sample.
 *
 * This is what the community engines do for quantized models (the
 * model's own generation_config.json ships top_k=20 top_p=0.95).
 * The naive salt_sample() softmaxes the FULL 248K vocab -- with a
 * normal 1.5-logit top1/top5 gap the mass spreads over a quarter
 * million tokens and EVERY draw looks arbitrary ("garbage"). Top-k
 * first (single pass, insertion into a k-slot array -- k is small,
 * e.g. 20-50) keeps the meaningful candidates; top-p then trims
 * the tail; temperature shapes the rest. Order matches vLLM:
 *   logits/temp -> top-k -> top-p -> softmax -> sample.
 * Returns the sampled id (never invalid). */
int salt_sample_vllm(const float *logits, int V, int top_k, double top_p,
                     float temp, uint64_t *rng) {
    if (V < 1) return 0;
    if (top_k <= 0 || top_k > V) top_k = 20;
    if (top_p <= 0.0) top_p = 0.95;
    if (top_p > 1.0) top_p = 1.0;
    if (temp <= 0.0f) temp = 1.0f;
    /* single pass: keep the top_k highest logits in idx (insertion) */
    int *idx = (int *)malloc((size_t)top_k * sizeof(int));
    float *w  = (float *)malloc((size_t)top_k * sizeof(float));
    if (!idx || !w) { free(idx); free(w); return salt_argmax(logits, V); }
    int n = 0;
    for (int i = 0; i < V; i++) {
        float li = (temp != 1.0f) ? logits[i] / temp : logits[i];
        if (n < top_k) {
            int pos = n++;
            while (pos > 0 && w[pos - 1] < li) { w[pos] = w[pos - 1]; idx[pos] = idx[pos - 1]; pos--; }
            w[pos] = li; idx[pos] = i;
        } else if (li > w[n - 1]) {
            int pos = n - 1;
            while (pos > 0 && w[pos - 1] < li) { w[pos] = w[pos - 1]; idx[pos] = idx[pos - 1]; pos--; }
            w[pos] = li; idx[pos] = i;
        }
    }
    /* softmax over the k-set (max-subtract) */
    float mx = w[0];
    double sum = 0.0;
    for (int k = 0; k < n; k++) {
        double e = salt_expf(w[k] - mx);
        w[k] = (float)e;
        sum += e;
    }
    /* top-p: trim to the smallest prefix with cumulative mass >= p */
    int m = n;
    if (top_p < 1.0) {
        double acc = 0.0;
        for (int k = 0; k < n; k++) {
            acc += (double)w[k] / sum;
            if (acc >= top_p) { m = k + 1; break; }
        }
    }
    /* xorshift64 draw within the surviving set */
    uint64_t x = *rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *rng = x;
    double target = (double)(x >> 11) / 9007199254740992.0 * sum;
    double acc = 0.0;
    int tok = idx[0];
    for (int k = 0; k < m; k++) {
        acc += (double)w[k];
        if (target < acc) { tok = idx[k]; break; }
    }
    free(idx);
    free(w);
    return tok;
}
