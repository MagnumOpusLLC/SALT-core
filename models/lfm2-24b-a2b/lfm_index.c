#include "lfm_index.h"

#include <stdio.h>
#include <string.h>

/* One line, copied so the parse can work on a terminated buffer. */
#define LFM_INDEX_LINE_MAX 512

static int parse_u64(const char *s, uint64_t *out) {
    unsigned long long v = 0;
    int n = 0;
    if (!s || !out || sscanf(s, "%llu%n", &v, &n) != 1 || n < 1)
        return -1;
    *out = (uint64_t)v;
    return 0;
}

static int parse_i64(const char *s, long long *out) {
    long long v = 0;
    int n = 0;
    if (!s || !out || sscanf(s, "%lld%n", &v, &n) != 1 || n < 1)
        return -1;
    *out = v;
    return 0;
}

/* Split a line in place on spaces. Returns the field count. */
static int split(char *line, char **fields, int max) {
    int n = 0;
    char *p = line;
    while (*p && n < max) {
        while (*p == ' ' || *p == '\t')
            *p++ = '\0';
        if (!*p)
            break;
        fields[n++] = p;
        while (*p && *p != ' ' && *p != '\t')
            p++;
    }
    return n;
}

int lfm_index_parse(LfmIndex *index, const char *text, size_t length) {
    size_t i = 0;
    int seen_header = 0;

    if (!index || !index->entries || index->capacity < 1 || !text)
        return -1;

    index->count = 0;
    for (int l = 0; l < LFM_LAYERS; l++)
        index->slot_base[l] = -1;

    while (i < length) {
        char line[LFM_INDEX_LINE_MAX];
        char *fields[16];
        int nf, field = 0;
        size_t start = i, len;

        while (i < length && text[i] != '\n')
            i++;
        len = i - start;
        if (i < length)
            i++;
        if (len >= sizeof line)
            return -1; /* a line longer than the format allows */
        memcpy(line, text + start, len);
        line[len] = '\0';
        /* Trailing CR from a CRLF index would otherwise corrupt the last field. */
        while (len > 0 && (line[len - 1] == '\r' || line[len - 1] == '\n'))
            line[--len] = '\0';
        if (len == 0)
            continue;

        nf = split(line, fields, 16);
        if (nf < 1)
            continue;

        if (strcmp(fields[0], "LFMIDX1") == 0) {
            if (nf < 9 || seen_header)
                return -1;
            if (parse_u64(fields[2], &index->slot_bytes))
                return -1;
            {
                long long v;
                if (parse_i64(fields[3], &v))
                    return -1;
                index->layers = (int)v;
                if (parse_i64(fields[4], &v))
                    return -1;
                index->experts = (int)v;
                if (parse_i64(fields[5], &v))
                    return -1;
                index->moe_layers = (int)v;
                if (parse_i64(fields[6], &v))
                    return -1;
                index->expert_slots = (int)v;
            }
            if (parse_u64(fields[7], &index->trunk_bytes) ||
                parse_u64(fields[8], &index->pool_bytes))
                return -1;
            seen_header = 1;
            continue;
        }

        if (strcmp(fields[0], "L") == 0) {
            long long layer, base;
            if (nf < 3 || parse_i64(fields[1], &layer) ||
                parse_i64(fields[2], &base))
                return -1;
            if (layer < 0 || layer >= LFM_LAYERS)
                return -1;
            index->slot_base[layer] = (int)base;
            continue;
        }

        if (strcmp(fields[0], "T") == 0 || strcmp(fields[0], "E") == 0) {
            LfmIndexEntry e;
            long long layer, expert, enc, rows, cols;
            int is_expert = fields[0][0] == 'E';

            memset(&e, 0, sizeof e);
            /* T: name layer enc rows cols v_off v_bytes s_off s_bytes b_off b_bytes
             * E: component layer expert enc rows cols v_off ... (one more field)
             * `field` is the ENCODING column: 3 for T, 4 for E, because
             * fields[2] is the layer in both. */
            field = is_expert ? 4 : 3;
            if (nf < field + 9)
                return -1;
            if (strlen(fields[1]) >= LFM_INDEX_NAME_MAX)
                return -1;
            memcpy(e.name, fields[1], strlen(fields[1]) + 1);
            if (parse_i64(fields[2], &layer))
                return -1;
            if (is_expert) {
                if (parse_i64(fields[3], &expert))
                    return -1;
                e.expert = (int)expert;
                if (layer < 0 || layer >= LFM_LAYERS)
                    return -1;
                e.slot_base = index->slot_base[layer];
                if (e.slot_base < 0)
                    return -1; /* an expert entry for a dense layer */
            } else {
                e.expert = -1;
                e.slot_base = -1;
            }
            e.layer = (int)layer;
            if (parse_i64(fields[field], &enc) ||
                parse_i64(fields[field + 1], &rows) ||
                parse_i64(fields[field + 2], &cols))
                return -1;
            e.encoding = (uint32_t)enc;
            e.rows = (uint32_t)rows;
            e.cols = (uint32_t)cols;
            if (parse_u64(fields[field + 3], &e.value_offset) ||
                parse_u64(fields[field + 4], &e.value_bytes) ||
                parse_u64(fields[field + 5], &e.scale_offset) ||
                parse_u64(fields[field + 6], &e.scale_bytes) ||
                parse_u64(fields[field + 7], &e.bias_offset) ||
                parse_u64(fields[field + 8], &e.bias_bytes))
                return -1;
            if (index->count >= index->capacity)
                return -1;
            index->entries[index->count++] = e;
            continue;
        }

        return -1; /* an unknown line type is a format change, not noise */
    }

    return seen_header ? 0 : -1;
}

int lfm_index_lookup(void *context, int layer, const char *name,
                     LfmTensorSpec *out) {
    LfmIndex *index = (LfmIndex *)context;
    if (!index || !name || !out)
        return -1;
    for (int i = 0; i < index->count; i++) {
        const LfmIndexEntry *e = &index->entries[i];
        if (e->expert >= 0)
            continue; /* T entries only: this fills layer descriptors */
        if (e->layer != layer)
            continue;
        if (strcmp(e->name, name) != 0)
            continue;
        memset(out, 0, sizeof *out);
        out->rows = e->rows;
        out->cols = e->cols;
        out->encoding = (SaltTensorEncoding)e->encoding;
        /* The index carries no group size, but an affine encoding is invalid
         * without one -- the engine's validation rejects group_size 0 before it
         * looks at anything else. 64 is the MLX group this container was
         * quantised with; it is a property of the package, not a guess. */
        if (e->encoding == SALT_TENSOR_ENCODING_AFFINE_Q4 ||
            e->encoding == SALT_TENSOR_ENCODING_AFFINE_Q6 ||
            e->encoding == SALT_TENSOR_ENCODING_AFFINE_Q8)
            out->group_size = 64;
        out->source_class = SALT_TENSOR_SOURCE_STATIC;
        out->resource_kind = SALT_TENSOR_RESOURCE_TRUNK;
        /* A STATIC tensor must carry logical_resource_id 0: the engine rejects
         * any non-zero value for a non-SELECTED source. Inventing a per-tensor
         * id here failed validation for every trunk tensor. */
        out->logical_resource_id = 0;
        /* The engine requires both of these on every descriptor. */
        out->host_ops = index->host_ops;
        out->stable_handle = ++index->stable_next;
        out->value_offset = e->value_offset;
        out->value_bytes = e->value_bytes;
        out->scale_offset = e->scale_offset;
        out->scale_bytes = e->scale_bytes;
        out->bias_offset = e->bias_offset;
        out->bias_bytes = e->bias_bytes;
        return 0;
    }
    return -1;
}

const LfmIndexEntry *lfm_index_expert(const LfmIndex *index, int layer,
                                      int expert, const char *component) {
    if (!index || !component)
        return NULL;
    for (int i = 0; i < index->count; i++) {
        const LfmIndexEntry *e = &index->entries[i];
        if (e->expert != expert || e->layer != layer)
            continue;
        if (strcmp(e->name, component) == 0)
            return e;
    }
    return NULL;
}
