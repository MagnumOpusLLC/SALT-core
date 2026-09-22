/*
 * json.h -- minimal header-only JSON parser, sized for exactly the
 * shapes this project reads: config.json (flat numeric object) and
 * safetensors headers (object of {"dtype","shape","data_offsets"}).
 *
 * Vendor note: written for ds4f-disk; intentionally no schema, no
 * defaults, no error recovery. If it cannot parse, it returns NULL.
 */
#ifndef SALT_JSON_H
#define SALT_JSON_H

#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct JEntry {
    const char *key, *key_end;
    int         type;              /* 0 num, 1 str, 2 obj, 3 arr, 4 bool, 5 null */
    double      num;
    int64_t     inum;
    uint64_t    unum;              /* exact for non-negative integer tokens */
    int         num_integral;
    int         num_unsigned;
    const char *str, *str_end;
    struct JEntry *child;
    int         nchild;
} JEntry;

typedef struct JDoc {
    const char *buf;
    size_t      len;
    JEntry *root;                  /* root object entries */
    int     nroot;
    JEntry *pool;
    int     npool, cap;
} JDoc;

static JEntry *j_new(JDoc *d) {
    if (d->npool == d->cap) {
        int ncap = d->cap ? d->cap * 2 : 64;
        JEntry *np = (JEntry *)realloc(d->pool, (size_t)ncap * sizeof(JEntry));
        if (!np) return NULL;
        d->pool = np;
        d->cap  = ncap;
    }
    JEntry *e = &d->pool[d->npool++];
    memset(e, 0, sizeof *e);
    return e;
}

static const char *j_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end);

static const char *j_str_raw(JDoc *d, const char **pp, const char *end,
                             const char **s_out, const char **e_out) {
    (void)d;
    const char *p = j_ws(*pp, end);
    if (p >= end || *p != '"') return NULL;
    p++;
    const char *s = p;
    while (p < end && *p != '"') {
        if (*p == '\\') p++;
        p++;
    }
    if (p >= end) return NULL;
    *s_out = s;
    *e_out = p;
    *pp = p + 1;
    return p;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end);

/* Reserve n consecutive zeroed entries; objects use this for their
 * children block so the block stays contiguous even when a child is
 * itself an object with its own block. */
static int j_reserve(JDoc *d, int n) {
    while (d->npool + n > d->cap) {
        int ncap = d->cap ? d->cap * 2 : 64;
        JEntry *np = (JEntry *)realloc(d->pool, (size_t)ncap * sizeof(JEntry));
        if (!np) return -1;
        d->pool = np;
        d->cap  = ncap;
    }
    int base = d->npool;
    memset(&d->pool[base], 0, (size_t)n * sizeof(JEntry));
    d->npool += n;
    return base;
}

/* Two passes. Pass 1 counts the children (its allocations are garbage);
 * pass 2 reserves one contiguous block of exactly n and fills it. A
 * child's own children live in their own block, so parent and child
 * blocks never interleave. */
static JEntry *j_object(JDoc *d, const char **pp, const char *end) {
    const char *obj_start = *pp;
    const char *p = j_ws(obj_start, end);
    if (p >= end || *p != '{') return NULL;
    p++;

    int n = 0;
    {
        const char *q = j_ws(p, end);
        if (q >= end) return NULL;
        if (*q != '}') {
            while (1) {
                const char *s, *e;
                if (!j_str_raw(d, &q, end, &s, &e)) return NULL;
                q = j_ws(q, end);
                if (q >= end || *q != ':') return NULL;
                q++;
                const char *vp = q;
                if (!j_value(d, &vp, end)) return NULL;
                n++;
                q = j_ws(vp, end);
                if (q >= end) return NULL;
                if (*q == ',') { q++; continue; }
                if (*q == '}') break;
                return NULL;
            }
        }
    }

    JEntry *e = j_new(d);
    if (!e) return NULL;
    int obj_i = d->npool - 1;               /* index, not pointer: pool may realloc */
    e->type = 2;
    int base = j_reserve(d, n);
    if (base < 0) return NULL;

    const char *p2 = j_ws(obj_start, end);
    p2++;                                   /* '{' */
    for (int i = 0; i < n; i++) {
        const char *s, *e;
        if (!j_str_raw(d, &p2, end, &s, &e)) return NULL;
        p2 = j_ws(p2, end);
        p2++;                               /* ':' */
        JEntry *v = j_value(d, &p2, end);
        if (!v) return NULL;
        if (base + i >= d->cap) return NULL;  /* count pass diverged */
        d->pool[base + i] = *v;             /* re-fetch pool each iteration */
        d->pool[base + i].key = s;
        d->pool[base + i].key_end = e;
        p2 = j_ws(p2, end);
        if (p2 < end && *p2 == ',') p2++;
    }
    if (p2 < end && *p2 == '}') p2++;
    if (obj_i >= d->npool) return NULL;
    JEntry *obj = &d->pool[obj_i];
    obj->child = &d->pool[base];
    obj->nchild = n;
    *pp = p2;
    return obj;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end) {
    const char *p = j_ws(*pp, end);
    if (p >= end) return NULL;
    if (*p == '{') return j_object(d, pp, end);
    if (*p == '"') {
        JEntry *e = j_new(d);
        if (!e) return NULL;
        e->type = 1;
        const char *s, *x;
        if (!j_str_raw(d, pp, end, &s, &x)) return NULL;
        e->str = s; e->str_end = x;
        return e;
    }
    if (*p == '[') {
        p++;
        int n = 0;
        /* pass 1: count elements (allocations are garbage, counted --
         * same path as j_object, so the dry-run preallocation holds) */
        {
            const char *q = p;
            while (1) {
                q = j_ws(q, end);
                if (q >= end) return NULL;
                if (*q == ']') break;
                if (!j_value(d, &q, end)) return NULL;
                n++;
                q = j_ws(q, end);
                if (q >= end) return NULL;
                if (*q == ',') { q++; continue; }
                if (*q == ']') break;
                return NULL;
            }
        }
        JEntry *e = j_new(d);
        if (!e) return NULL;
        int e_i = d->npool - 1;             /* index, not pointer: pool may realloc */
        e->type = 3;
        int base = j_reserve(d, n);
        if (base < 0) return NULL;
        for (int i = 0; i < n; i++) {
            p = j_ws(p, end);
            JEntry *v = j_value(d, &p, end);
            if (!v) return NULL;
            if (base + i >= d->cap) return NULL;  /* count pass diverged */
            d->pool[base + i] = *v;     /* element COPY into the block */
            p = j_ws(p, end);
            if (p < end && *p == ',') p++;
        }
        if (p < end && *p == ']') p++;
        /* set child AFTER the fill loop: recursive j_value calls above may
         * realloc the pool, so &d->pool[base] taken before the loop would
         * dangle (ASan-confirmed heap-use-after-free on the real trunk.json) */
        e = &d->pool[e_i];
        e->child = &d->pool[base];
        e->nchild = n;
        *pp = p;
        return e;
    }
    if (*p == 't' || *p == 'f' || *p == 'n') {
        JEntry *e = j_new(d);
        if (!e) return NULL;
        e->type = (*p == 't' || *p == 'f') ? 4 : 5;
        if (e->type == 4) {
            e->num_integral = 1;
            e->inum = (*p == 't') ? 1 : 0;
            e->unum = (uint64_t)e->inum;
            e->num_unsigned = 1;
        }
        p += (*p == 't') ? 4 : (*p == 'f') ? 5 : 4;
        *pp = p;
        return e;
    }
    /* number */
    {
        JEntry *e = j_new(d);
        if (!e) return NULL;
        e->type = 0;
        char tmp[64];
        size_t i = 0;
        while (p < end && i < sizeof tmp - 1 &&
               ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' ||
                *p == '.' || *p == 'e' || *p == 'E')) {
            tmp[i++] = *p++;
        }
        tmp[i] = 0;
        if (i == 0 || (p < end &&
            ((*p >= '0' && *p <= '9') || *p == '-' || *p == '+' ||
             *p == '.' || *p == 'e' || *p == 'E')))
            return NULL;
        char *number_end = NULL;
        errno = 0;
        e->num = strtod(tmp, &number_end);
        if (errno == ERANGE || number_end != tmp + i) return NULL;
        e->num_integral =
            strchr(tmp, '.') == NULL && strchr(tmp, 'e') == NULL &&
            strchr(tmp, 'E') == NULL;
        if (e->num_integral && tmp[0] != '-') {
            errno = 0;
            number_end = NULL;
            unsigned long long value = strtoull(tmp, &number_end, 10);
            if (errno == ERANGE || number_end != tmp + i) return NULL;
            e->num_unsigned = 1;
            e->unum = (uint64_t)value;
            e->inum = value <= (unsigned long long)INT64_MAX
                ? (int64_t)value : INT64_MAX;
        } else if (e->num_integral) {
            errno = 0;
            number_end = NULL;
            long long value = strtoll(tmp, &number_end, 10);
            if (errno == ERANGE || number_end != tmp + i) return NULL;
            e->inum = (int64_t)value;
        } else if (e->num >= (double)INT64_MIN &&
                   e->num <= (double)INT64_MAX) {
            e->inum = (int64_t)e->num;
        }
        *pp = p;
        return e;
    }
}

static JDoc *json_parse(const char *buf, size_t len) {
    /* Dry-run first: count how many entries the parse allocates, then
     * preallocate exactly that capacity. The pool then NEVER reallocs
     * during the real parse, which keeps every stored child pointer
     * valid -- realloc would otherwise dangle them. */
    JDoc count = {0};
    {
        const char *p = buf, *end = buf + len;
        p = j_ws(p, end);
        if (p >= end || *p != '{' || !j_object(&count, &p, end)) {
            free(count.pool);
            return NULL;
        }
    }
    JDoc *d = (JDoc *)calloc(1, sizeof(JDoc));
    if (!d) { free(count.pool); return NULL; }
    d->buf = buf;
    d->len = len;
    d->cap = count.npool + 16;
    d->pool = (JEntry *)calloc((size_t)d->cap, sizeof(JEntry));
    if (!d->pool) { free(count.pool); free(d); return NULL; }
    free(count.pool);

    const char *p = buf, *end = buf + len;
    p = j_ws(p, end);
    if (p >= end || *p != '{') { free(d->pool); free(d); return NULL; }
    JEntry *root = j_object(d, &p, end);
    if (!root) { free(d->pool); free(d); return NULL; }
    d->root  = root->child;
    d->nroot = root->nchild;
    return d;
}

static void json_free(JDoc *d) {
    if (!d) return;
    free(d->pool);
    free(d);
}

static const JEntry *json_get(const JEntry *obj, int n, const char *key) {
    size_t klen = strlen(key);
    for (int i = 0; i < n; i++) {
        const JEntry *e = &obj[i];
        if ((size_t)(e->key_end - e->key) == klen &&
            memcmp(e->key, key, klen) == 0)
            return e;
    }
    return NULL;
}

#endif /* SALT_JSON_H */
