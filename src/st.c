/* st.c -- safetensors index reader. The pointer map: tensor name ->
 * absolute file offset, built BEFORE any weight bytes are touched. */
#include "salt/salt.h"
#include "json.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint64_t rd_le64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

int salt_st_open(SaltSt *st, const char *path) {
    JDoc *doc = NULL;
    memset(st, 0, sizeof *st);
    st->fd = open(path, O_RDONLY);
    if (st->fd < 0) {
        fprintf(stderr, "st: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t lenb[8];
    ssize_t got = pread(st->fd, lenb, 8, 0);
    if (got != 8) { fprintf(stderr, "st: short header length\n"); goto fail; }
    uint64_t raw_hlen = rd_le64(lenb);
    if (raw_hlen == 0 || raw_hlen > (1U << 28)) {
        fprintf(stderr, "st: implausible header length %llu\n",
                (unsigned long long)raw_hlen);
        goto fail;
    }
    int64_t hlen = (int64_t)raw_hlen;
    st->hdr = (char *)malloc((size_t)hlen + 1);
    if (!st->hdr) goto fail;
    got = pread(st->fd, st->hdr, (size_t)hlen, 8);
    if (got != hlen) {
        fprintf(stderr, "st: short header read\n");
        goto fail;
    }
    st->hdr[hlen] = 0;
    st->hdr_len = 8 + hlen;

    doc = json_parse(st->hdr, (size_t)hlen);
    if (!doc) {
        fprintf(stderr, "st: safetensors header is not parseable JSON\n");
        goto fail;
    }
    /* payload starts at 8 + hlen; data_offsets are relative to it */
    st->t = (SaltTensor *)calloc((size_t)doc->nroot, sizeof(SaltTensor));
    if (!st->t) goto fail;
    st->n = doc->nroot;
    for (int i = 0; i < doc->nroot; i++) {
        const JEntry *e = &doc->root[i];
        if (e->type != 2) continue;              /* skip non-object values */
        const JEntry *dtype = json_get(e->child, e->nchild, "dtype");
        const JEntry *offs  = json_get(e->child, e->nchild, "data_offsets");
        if (!offs || offs->type != 3 || offs->nchild < 2) continue;
        int64_t a = offs->child[0].inum;
        int64_t b = offs->child[1].inum;
        st->t[i].name     = e->key;                /* points into st->hdr */
        st->t[i].name_len = (int)(e->key_end - e->key);
        st->t[i].off      = st->hdr_len + a;
        st->t[i].nbytes   = b - a;
        (void)dtype;
    }
    json_free(doc);
    return 0;

fail:
    json_free(doc);
    salt_st_close(st);
    st->fd = -1;
    return -1;
}

void salt_st_close(SaltSt *st) {
    if (st->fd >= 0) close(st->fd);
    free(st->hdr);
    free(st->t);
    memset(st, 0, sizeof *st);
}

const SaltTensor *salt_st_find(const SaltSt *st, const char *name) {
    size_t klen = strlen(name);
    for (int i = 0; i < st->n; i++) {
        const SaltTensor *t = &st->t[i];
        if (t->name && (size_t)t->name_len == klen &&
            memcmp(t->name, name, klen) == 0)
            return t;
    }
    return NULL;
}

int salt_pool_build(SaltExpertPool *pool, const SaltSt *st, const SaltCfg *cfg) {
    if (!pool || !st || !cfg || cfg->n_layers < 1 || cfg->n_experts < 1 ||
        cfg->expert_nbytes < 1 ||
        (size_t)cfg->n_layers > SIZE_MAX / (size_t)cfg->n_experts)
        return -1;
    memset(pool, 0, sizeof *pool);
    pool->fd = dup(st->fd);
    if (pool->fd < 0) {
        fprintf(stderr, "pool: cannot duplicate source fd: %s\n",
                strerror(errno));
        return -1;
    }
    pool->owns_fd = 1;
    pool->n_layers  = cfg->n_layers;
    pool->n_experts = cfg->n_experts;
    pool->nbytes    = cfg->expert_nbytes;
    size_t total = (size_t)cfg->n_layers * (size_t)cfg->n_experts;
    pool->ref = (SaltExpertRef *)calloc(total, sizeof(SaltExpertRef));
    if (!pool->ref) {
        salt_pool_close(pool);
        return -1;
    }

    char name[64];
    int missing = 0;
    for (int L = 0; L < cfg->n_layers; L++) {
        for (int e = 0; e < cfg->n_experts; e++) {
            snprintf(name, sizeof name, "e.%d.%d", L, e);
            const SaltTensor *t = salt_st_find(st, name);
            if (!t || t->nbytes != cfg->expert_nbytes) {
                if (!missing)
                    fprintf(stderr, "pool: missing or mis-sized %s\n", name);
                missing++;
                continue;
            }
            pool->ref[(size_t)L * cfg->n_experts + e].off    = t->off;
            pool->ref[(size_t)L * cfg->n_experts + e].nbytes = t->nbytes;
        }
    }
    if (missing) {
        fprintf(stderr, "pool: %d expert tensors missing from index\n", missing);
        free(pool->ref);
        pool->ref = NULL;
        salt_pool_close(pool);
        return -1;
    }
    return 0;
}

static uint64_t rd_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

int salt_pool_open_packed(SaltExpertPool *pool, const char *path,
                          const SaltCfg *cfg) {
    if (!pool || !path || !cfg || cfg->n_layers < 1 || cfg->n_experts < 1 ||
        cfg->expert_nbytes < 1)
        return -1;
    memset(pool, 0, sizeof *pool);
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "pool: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t hdr[24];
    if (pread(fd, hdr, 24, 0) != 24) {
        fprintf(stderr, "pool: short header in %s\n", path);
        close(fd);
        return -1;
    }
    uint64_t nbytes = rd_u64(hdr);
    uint64_t n_layers = rd_u64(hdr + 8);
    uint64_t n_experts = rd_u64(hdr + 16);
    if (nbytes == 0 || nbytes > INT64_MAX ||
        nbytes != (uint64_t)cfg->expert_nbytes ||
        n_layers != (uint64_t)cfg->n_layers ||
        n_experts != (uint64_t)cfg->n_experts) {
        fprintf(stderr,
                "pool: header mismatch (nbytes %llu, %llu layers x %llu "
                "experts; config %d x %d)\n",
                (unsigned long long)nbytes, (unsigned long long)n_layers,
                (unsigned long long)n_experts,
                cfg->n_layers, cfg->n_experts);
        close(fd);
        return -1;
    }
    if (n_layers > UINT64_MAX / n_experts) {
        close(fd);
        return -1;
    }
    uint64_t slot_count = n_layers * n_experts;
    if (slot_count > (UINT64_MAX - 24) / nbytes) {
        close(fd);
        return -1;
    }
    uint64_t expected_nbytes = 24 + slot_count * nbytes;
    if (expected_nbytes > SIZE_MAX || expected_nbytes > INT64_MAX) {
        close(fd);
        return -1;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        fprintf(stderr, "pool: cannot stat %s: %s\n", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (sb.st_size < 0 || (uint64_t)sb.st_size != expected_nbytes) {
        fprintf(stderr,
                "pool: extent mismatch (%lld bytes; expected %llu)\n",
                (long long)sb.st_size,
                (unsigned long long)expected_nbytes);
        close(fd);
        return -1;
    }
    pool->fd = fd;
    pool->owns_fd = 1;      /* conservation: salt_pool_close releases */
#ifdef __APPLE__
    /* NOTE: F_NOCACHE is deliberately NOT set. The mmap below wants
     * the unified page cache: experts fault in once and are reused
     * across tokens. Runtime enforcement uses application footprint;
     * clean file-backed pages stay a separate comparison field. Resident
     * pages are bounded by
     * madvise(MADV_DONTNEED) on cache eviction, so the touched-but-
     * evicted pages get flushed back to the OS -- no unbounded
     * resident growth (the "touched mem did not get flushed" trap).
     * SALT_COLD=1 opts into F_NOCACHE: every read goes straight to
     * disk (deterministic cold call without sudo purge). */
    if (getenv("SALT_COLD") && *getenv("SALT_COLD") == '1') {
        if (fcntl(fd, F_NOCACHE, 1) != 0)
            fprintf(stderr, "pool: F_NOCACHE failed: %s\n", strerror(errno));
    }
#endif
    /* FLAT MAP (zero-copy): mmap the whole pool. The ref[] table is
     * the flat map: off = 24 + (L*n_experts+e)*nbytes is DERIVED by
     * arithmetic, no file-tree lookup. The cache's pair table records
     * which experts are resident (key -> pointer into this mapping);
     * the fill (readahead + touch) is parallel and direct from known
     * offsets; eviction flushes the pages (DONTNEED). */
    size_t mlen = (size_t)expected_nbytes;
    uint8_t *map = (uint8_t *)mmap(NULL, mlen, PROT_READ, MAP_PRIVATE,
                                   fd, 0);
    if (map == MAP_FAILED) {
        fprintf(stderr, "pool: mmap failed (%zu bytes): %s\n", mlen,
                strerror(errno));
        salt_pool_close(pool);
        return -1;
    }
    pool->map = map;
    pool->map_len = mlen;
    pool->n_layers = (int)n_layers;
    pool->n_experts = (int)n_experts;
    pool->nbytes = (int64_t)nbytes;
    pool->ref = (SaltExpertRef *)calloc((size_t)slot_count,
                                        sizeof(SaltExpertRef));
    if (!pool->ref) {
        salt_pool_close(pool);
        pool->map_len = 0;
        pool->n_layers = pool->n_experts = 0;
        pool->nbytes = 0;
        return -1;
    }
    for (uint64_t i = 0; i < slot_count; i++) {
        pool->ref[i].off = (int64_t)(24 + i * nbytes);
        pool->ref[i].nbytes = (int64_t)nbytes;
    }
    return 0;
}

const uint8_t *salt_pool_expert_ptr(const SaltExpertPool *pool,
                                    size_t ordinal, int64_t min_nbytes) {
    size_t count;
    const SaltExpertRef *ref;
    if (!pool || !pool->map || !pool->ref || min_nbytes < 0 ||
        pool->n_layers < 1 || pool->n_experts < 1 ||
        (size_t)pool->n_layers > SIZE_MAX / (size_t)pool->n_experts)
        return NULL;
    count = (size_t)pool->n_layers * (size_t)pool->n_experts;
    if (ordinal >= count) return NULL;
    ref = &pool->ref[ordinal];
    if (ref->off < 0 || ref->nbytes < min_nbytes ||
        (uint64_t)ref->off > (uint64_t)pool->map_len ||
        (uint64_t)ref->nbytes >
            (uint64_t)pool->map_len - (uint64_t)ref->off)
        return NULL;
    return pool->map + (size_t)ref->off;
}

/* Conservation pairing for both pool constructors: release the packed
 * path's fd/mapping or the build path's duplicated source fd. The ref
 * table is freed by the caller (main.c) after this. */
int salt_pool_close(SaltExpertPool *pool) {
    if (!pool) return 0;
    if (pool->map && pool->map != MAP_FAILED) {
        if (munmap(pool->map, pool->map_len) != 0) return -1;
        pool->map = NULL;
        pool->map_len = 0;
    }
    if (pool->owns_fd && pool->fd >= 0) {
        if (close(pool->fd) != 0) return -1;
        pool->fd = -1;
        pool->owns_fd = 0;
    }
    return 0;
}
