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
#include <salt/rotation/fail.h>
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

static const char *j_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end, JEntry *out);

static const char *j_str_raw(JDoc *d, const char **pp, const char *end,
                             const char **s_out, const char **e_out) {
    (void)d;
    const char *p = j_ws(*pp, end);
    if (p >= end || *p != '"') return NULL;
    p++;
    const char *s = p;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) return NULL;
        }
        p++;
    }
    if (p >= end) return NULL;
    *s_out = s;
    *e_out = p;
    *pp = p + 1;
    return p;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end, JEntry *out);

/* Scan structure without constructing entries. Each JSON value contributes
 * exactly one entry; object keys remain slices of the immutable input. */
static int j_scan_value(const char **pp, const char *end,
                        size_t *entries, int *children) {
    const char *p = j_ws(*pp, end);
    int n = 0;
    if (p >= end) return SALT_TFAIL(-1);
    if (entries) {
        if (*entries >= (size_t)INT_MAX) return SALT_TFAIL(-1);
        (*entries)++;
    }
    if (*p == '{' || *p == '[') {
        int object = *p == '{';
        char close = object ? '}' : ']';
        p = j_ws(p + 1, end);
        if (p >= end) return SALT_TFAIL(-1);
        if (*p != close) {
            for (;;) {
                if (object) {
                    const char *s, *e;
                    if (!j_str_raw(NULL, &p, end, &s, &e))
                        return SALT_TFAIL(-1);
                    p = j_ws(p, end);
                    if (p >= end || *p != ':') return SALT_TFAIL(-1);
                    p++;
                }
                if (n == INT_MAX || j_scan_value(&p, end, entries, NULL) != 0)
                    return SALT_TFAIL(-1);
                n++;
                p = j_ws(p, end);
                if (p >= end) return SALT_TFAIL(-1);
                if (*p == close) break;
                if (*p != ',') return SALT_TFAIL(-1);
                p = j_ws(p + 1, end);
                if (p >= end || *p == close) return SALT_TFAIL(-1);
            }
        }
        p++;
    } else if (*p == '"') {
        const char *s, *e;
        if (!j_str_raw(NULL, &p, end, &s, &e)) return SALT_TFAIL(-1);
    } else if (*p == 't' || *p == 'f' || *p == 'n') {
        const char *literal = *p == 't' ? "true" : (*p == 'f' ? "false" : "null");
        size_t bytes = strlen(literal);
        if ((size_t)(end - p) < bytes || memcmp(p, literal, bytes) != 0)
            return SALT_TFAIL(-1);
        p += bytes;
    } else {
        const char *start = p;
        while (p < end && ((*p >= '0' && *p <= '9') || *p == '-' ||
               *p == '+' || *p == '.' || *p == 'e' || *p == 'E')) p++;
        /* Conversion and exact integer validation remain in j_value. */
        if (p == start || (size_t)(p - start) >= 64u) return SALT_TFAIL(-1);
    }
    if (children) *children = n;
    *pp = p;
    return 0;
}

/* The complete pool is allocated once by json_parse. Filling a child directly
 * into its reserved seat creates no temporary records and never moves pointers. */
static int j_reserve(JDoc *d, int n) {
    if (n < 0 || d->npool > d->cap || n > d->cap - d->npool)
        return SALT_TFAIL(-1);
    int base = d->npool;
    d->npool += n;
    return base;
}

static JEntry *j_object(JDoc *d, const char **pp, const char *end, JEntry *out) {
    const char *p = j_ws(*pp, end), *scan = p;
    int n;
    if (p >= end || *p != '{' || j_scan_value(&scan, end, NULL, &n) != 0)
        return NULL;
    int base = j_reserve(d, n);
    if (base < 0) return NULL;
    out->type = 2;
    out->child = &d->pool[base];
    out->nchild = n;
    p = j_ws(p + 1, end);
    for (int i = 0; i < n; i++) {
        const char *s, *e;
        if (!j_str_raw(d, &p, end, &s, &e)) return NULL;
        p = j_ws(p, end);
        if (p >= end || *p != ':') return NULL;
        p++;
        JEntry *child = &d->pool[base + i];
        if (!j_value(d, &p, end, child)) return NULL;
        child->key = s;
        child->key_end = e;
        p = j_ws(p, end);
        if (p < end && *p == ',') p++;
    }
    if (p >= end || *p != '}') return NULL;
    *pp = p + 1;
    return out;
}

static JEntry *j_value(JDoc *d, const char **pp, const char *end, JEntry *e) {
    const char *p = j_ws(*pp, end);
    if (p >= end) return NULL;
    if (*p == '{') return j_object(d, pp, end, e);
    if (*p == '"') {
        e->type = 1;
        const char *s, *x;
        if (!j_str_raw(d, pp, end, &s, &x)) return NULL;
        e->str = s; e->str_end = x;
        return e;
    }
    if (*p == '[') {
        const char *scan = p;
        int n;
        if (j_scan_value(&scan, end, NULL, &n) != 0) return NULL;
        int base = j_reserve(d, n);
        if (base < 0) return NULL;
        e->type = 3;
        e->child = &d->pool[base];
        e->nchild = n;
        p = j_ws(p + 1, end);
        for (int i = 0; i < n; i++) {
            if (!j_value(d, &p, end, &d->pool[base + i])) return NULL;
            p = j_ws(p, end);
            if (p < end && *p == ',') p++;
        }
        if (p >= end || *p != ']') return NULL;
        *pp = p + 1;
        return e;
    }
    if (*p == 't' || *p == 'f' || *p == 'n') {
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
    /* The scan uses no entry storage. Allocate one stable seat per JSON value,
     * then fill children directly into their contiguous reserved blocks. */
    if (!buf) return NULL;
    const char *p = buf, *end = buf + len;
    size_t entries = 0;
    p = j_ws(p, end);
    if (p >= end || *p != '{' || j_scan_value(&p, end, &entries, NULL) != 0 ||
        j_ws(p, end) != end || entries == 0 ||
        entries > SIZE_MAX / sizeof(JEntry)) return NULL;
    JDoc *d = (JDoc *)calloc(1, sizeof(JDoc));
    if (!d) return NULL;
    d->buf = buf;
    d->len = len;
    d->cap = (int)entries;
    d->pool = (JEntry *)calloc(entries, sizeof(JEntry));
    if (!d->pool) { free(d); return NULL; }
    d->npool = 1;  /* root owns the first seat */
    p = buf;
    JEntry *root = j_object(d, &p, end, &d->pool[0]);
    if (!root || d->npool != d->cap || j_ws(p, end) != end) {
        free(d->pool); free(d); return NULL;
    }
    d->root = root->child;
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
