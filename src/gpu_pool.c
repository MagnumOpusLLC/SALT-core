#if !defined(__APPLE__) && !defined(_GNU_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "salt/gpu_pool.h"
#include "salt/moe.h"
#include "sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define INDEX_HEADER_NBYTES 92u
#define INDEX_SEGMENT_NBYTES 48u
#define INDEX_RECORD_NBYTES 40u
#define DISPATCH_HEADER_NBYTES 68u
#define DISPATCH_RECORD_NBYTES 96u
#define GPU_POOL_VERSION 1u

static uint64_t off_t_max_u64(void) {
    if (sizeof(off_t) >= sizeof(int64_t)) return (uint64_t)INT64_MAX;
    if (sizeof(off_t) >= sizeof(int32_t)) return (uint64_t)INT32_MAX;
    return (uint64_t)INT16_MAX;
}

static const uint8_t INDEX_MAGIC[8] = {'S','A','L','T','G','P','0','1'};
static const uint8_t DISPATCH_MAGIC[8] = {'S','A','L','T','D','S','P','1'};

typedef struct SaltGpuPoolShard {
    uint64_t file_offset;
    size_t   map_nbytes;
    uint64_t payload_offset;
    uint64_t payload_nbytes;
} SaltGpuPoolShard;

typedef struct SaltGpuPoolReady {
    uint8_t  index_sha256[32];
    uint8_t  dispatch_sha256[32];
    uint8_t  manifest_sha256[32];
    uint64_t source_device;
    uint64_t source_inode;
    uint64_t source_nbytes;
    uint64_t source_mtime_ns;
    uint64_t source_ctime_ns;
} SaltGpuPoolReady;

typedef struct SaltGpuPoolImpl {
    SaltGpuPoolShard *shard;
    char             *source_path;
    uint8_t           layout_identity[32];
    SaltGpuPoolReady  ready;
} SaltGpuPoolImpl;

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static int read_full(int fd, void *dst, size_t nbytes, off_t off) {
    uint8_t *p = (uint8_t *)dst;
    size_t done = 0;
    while (done < nbytes) {
        ssize_t n = pread(fd, p + done, nbytes - done, off + (off_t)done);
        if (n <= 0) return -1;
        done += (size_t)n;
    }
    return 0;
}

static int make_path(char *dst, size_t cap, const char *root,
                     const char *suffix) {
    int n = snprintf(dst, cap, "%s/%s", root, suffix);
    return n >= 0 && (size_t)n < cap ? 0 : -1;
}

static int ready_u64(FILE *f, const char *key, uint64_t *value) {
    char line[160], *end = NULL;
    size_t n = strlen(key);
    if (!fgets(line, sizeof line, f) || strncmp(line, key, n) != 0 ||
        line[n] != '=') return -1;
    errno = 0;
    unsigned long long v = strtoull(line + n + 1, &end, 10);
    if (errno || end == line + n + 1 ||
        (*end != '\n' && !(*end == '\r' && end[1] == '\n'))) return -1;
    *value = (uint64_t)v;
    return 0;
}

static int ready_digest(FILE *f, const char *key, uint8_t out[32]) {
    char line[160];
    size_t n = strlen(key);
    if (!fgets(line, sizeof line, f) || strncmp(line, key, n) != 0 ||
        line[n] != '=' ||
        salt_sha256_hex_parse(line + n + 1, out) != 0) return -1;
    const char *tail = line + n + 1 + 64;
    return *tail == '\n' || (*tail == '\r' && tail[1] == '\n') ? 0 : -1;
}

static int load_ready(const char *directory, SaltGpuPoolReady *ready) {
    char path[PATH_MAX], line[160];
    if (make_path(path, sizeof path, directory, "READY") != 0) return -1;
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    memset(ready, 0, sizeof *ready);
    int ok = fgets(line, sizeof line, f) &&
        strcmp(line, "salt-gpu-ledger-v1\n") == 0 &&
        ready_digest(f, "index_sha256", ready->index_sha256) == 0 &&
        ready_digest(f, "dispatch_sha256", ready->dispatch_sha256) == 0 &&
        ready_digest(f, "manifest_sha256", ready->manifest_sha256) == 0 &&
        ready_u64(f, "source_device", &ready->source_device) == 0 &&
        ready_u64(f, "source_inode", &ready->source_inode) == 0 &&
        ready_u64(f, "source_nbytes", &ready->source_nbytes) == 0 &&
        ready_u64(f, "source_mtime_ns", &ready->source_mtime_ns) == 0 &&
        ready_u64(f, "source_ctime_ns", &ready->source_ctime_ns) == 0 &&
        fgets(line, sizeof line, f) == NULL;
    fclose(f);
    return ok ? 0 : -1;
}

static int verify_path_sha256(const char *directory, const char *name,
                              const uint8_t expected[32]) {
    char path[PATH_MAX];
    uint8_t actual[32];
    if (make_path(path, sizeof path, directory, name) != 0 ||
        salt_sha256_path(path, actual) != 0) return -1;
    return memcmp(actual, expected, sizeof actual) == 0 ? 0 : -1;
}

static uint64_t stat_mtime_ns(const struct stat *st) {
#ifdef __APPLE__
    return (uint64_t)st->st_mtimespec.tv_sec * 1000000000u +
           (uint64_t)st->st_mtimespec.tv_nsec;
#else
    return (uint64_t)st->st_mtim.tv_sec * 1000000000u +
           (uint64_t)st->st_mtim.tv_nsec;
#endif
}

static uint64_t stat_ctime_ns(const struct stat *st) {
#ifdef __APPLE__
    return (uint64_t)st->st_ctimespec.tv_sec * 1000000000u +
           (uint64_t)st->st_ctimespec.tv_nsec;
#else
    return (uint64_t)st->st_ctim.tv_sec * 1000000000u +
           (uint64_t)st->st_ctim.tv_nsec;
#endif
}

static int source_stat_matches(const SaltGpuPoolReady *ready,
                               const struct stat *st) {
    return st->st_size >= 0 &&
        (uint64_t)st->st_dev == ready->source_device &&
        (uint64_t)st->st_ino == ready->source_inode &&
        (uint64_t)st->st_size == ready->source_nbytes &&
        stat_mtime_ns(st) == ready->source_mtime_ns &&
        stat_ctime_ns(st) == ready->source_ctime_ns;
}

static void close_impl(SaltGpuPool *pool) {
    SaltGpuPoolImpl *impl = (SaltGpuPoolImpl *)pool->impl;
    if (!impl) return;
    free(impl->source_path);
    free(impl->shard);
    free(impl);
    pool->impl = NULL;
}

static int source_fd_valid(const SaltGpuPool *pool, int fd) {
    const SaltGpuPoolImpl *impl = pool ? (const SaltGpuPoolImpl *)pool->impl : NULL;
    uint8_t hdr[24];
    struct stat st;
    return impl && fd >= 0 && read_full(fd, hdr, sizeof hdr, 0) == 0 &&
           fstat(fd, &st) == 0 && source_stat_matches(&impl->ready, &st) &&
           (uint64_t)st.st_size == pool->source_nbytes &&
           rd64(hdr) == pool->expert_nbytes &&
           rd64(hdr + 8) == pool->n_layers &&
           rd64(hdr + 16) == pool->n_experts;
}

static int validate_source(SaltGpuPool *pool, const char *source_path) {
    SaltGpuPoolImpl *impl = (SaltGpuPoolImpl *)pool->impl;
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    int fd = open(source_path, flags);
    if (fd < 0) {
        fprintf(stderr, "gpu-pool: cannot open source %s: %s\n",
                source_path, strerror(errno));
        return -1;
    }
    int ok = source_fd_valid(pool, fd);
    close(fd);
    if (!ok) {
        fprintf(stderr, "gpu-pool: canonical source identity/header/extent mismatch\n");
        return -1;
    }
    size_t path_nbytes = strlen(source_path) + 1;
    impl->source_path = (char *)malloc(path_nbytes);
    if (!impl->source_path) return -1;
    memcpy(impl->source_path, source_path, path_nbytes);
    return 0;
}

static int range_in_ref(const SaltGpuPoolRef *ref, uint64_t off,
                        uint64_t nbytes) {
    if (!nbytes || off < ref->local_offset || off > UINT64_MAX - nbytes)
        return 0;
    uint64_t end = ref->local_offset + ref->nbytes;
    return off <= end && nbytes <= end - off;
}

static int ranges_overlap(uint64_t aoff, uint64_t an,
                          uint64_t boff, uint64_t bn) {
    return aoff < boff + bn && boff < aoff + an;
}

static int valid_q4_dispatch(const SaltGpuDispatchTemplate *t) {
    uint64_t elements = (uint64_t)t->R * t->C;
    if (!t->R || !t->C || t->fmt != 1 || t->bits != 4 ||
        t->block_size != 64 || elements % 64u != 0) return 0;
    uint64_t groups = elements / 64u;
    return t->v_nbytes == elements / 2u &&
        t->s_nbytes == groups * 2u && t->b_nbytes == groups * 2u &&
        (t->voff & 3u) == 0 && (t->soff & 1u) == 0 &&
        (t->boff & 1u) == 0 &&
        !ranges_overlap(t->voff, t->v_nbytes, t->soff, t->s_nbytes) &&
        !ranges_overlap(t->voff, t->v_nbytes, t->boff, t->b_nbytes) &&
        !ranges_overlap(t->soff, t->s_nbytes, t->boff, t->b_nbytes);
}

static int dispatch_pair_overlaps(const SaltGpuDispatchTemplate *a,
                                  const SaltGpuDispatchTemplate *b) {
    const uint64_t ao[3] = {a->voff, a->soff, a->boff};
    const uint64_t an[3] = {a->v_nbytes, a->s_nbytes, a->b_nbytes};
    const uint64_t bo[3] = {b->voff, b->soff, b->boff};
    const uint64_t bn[3] = {b->v_nbytes, b->s_nbytes, b->b_nbytes};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (ranges_overlap(ao[i], an[i], bo[j], bn[j])) return 1;
    return 0;
}

static int load_dispatch(SaltGpuPool *pool, const char *directory,
                         const uint8_t expected_sha256[32]) {
    SaltGpuPoolImpl *impl = (SaltGpuPoolImpl *)pool->impl;
    char path[PATH_MAX];
    if (make_path(path, sizeof path, directory, "dispatch.bin") != 0)
        return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gpu-pool: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t actual_sha256[32];
    if (salt_sha256_fd(fd, actual_sha256) != 0 ||
        memcmp(actual_sha256, expected_sha256, sizeof actual_sha256) != 0) {
        fprintf(stderr, "gpu-pool: dispatch SHA-256 mismatch\n");
        close(fd);
        return -1;
    }
    uint8_t hdr[DISPATCH_HEADER_NBYTES];
    struct stat st;
    if (read_full(fd, hdr, sizeof hdr, 0) != 0 || fstat(fd, &st) != 0 ||
        st.st_size < 0) {
        fprintf(stderr, "gpu-pool: cannot read dispatch header/extent\n");
        close(fd);
        return -1;
    }
    uint32_t version = rd32(hdr + 8);
    uint32_t header_nbytes = rd32(hdr + 12);
    uint32_t record_nbytes = rd32(hdr + 16);
    uint32_t count = rd32(hdr + 20);
    uint32_t n_layers = rd32(hdr + 24);
    uint32_t n_experts = rd32(hdr + 28);
    uint32_t reserved = rd32(hdr + 32);
    uint64_t expected_count = (uint64_t)pool->n_layers * pool->n_experts * 3u;
    uint64_t expected_file = DISPATCH_HEADER_NBYTES +
        expected_count * DISPATCH_RECORD_NBYTES;
    if (!impl || memcmp(hdr, DISPATCH_MAGIC, 8) != 0 ||
        version != GPU_POOL_VERSION ||
        header_nbytes != DISPATCH_HEADER_NBYTES ||
        record_nbytes != DISPATCH_RECORD_NBYTES || count != expected_count ||
        n_layers != pool->n_layers || n_experts != pool->n_experts ||
        reserved != 0 ||
        memcmp(hdr + 36, impl->layout_identity,
               sizeof impl->layout_identity) != 0 ||
        (uint64_t)st.st_size != expected_file ||
        expected_count > SIZE_MAX / sizeof(SaltGpuDispatchTemplate)) {
        fprintf(stderr, "gpu-pool: dispatch version/geometry/extent mismatch\n");
        close(fd);
        return -1;
    }
    pool->dispatch = (SaltGpuDispatchTemplate *)calloc(
        (size_t)expected_count, sizeof *pool->dispatch);
    if (!pool->dispatch) {
        close(fd);
        return -1;
    }
    pool->n_dispatch = expected_count;

    uint8_t raw[DISPATCH_RECORD_NBYTES];
    for (uint64_t i = 0; i < expected_count; i++) {
        if (read_full(fd, raw, sizeof raw,
                      (off_t)(DISPATCH_HEADER_NBYTES +
                              i * DISPATCH_RECORD_NBYTES)) != 0) {
            fprintf(stderr, "gpu-pool: short dispatch record %llu\n",
                    (unsigned long long)i);
            close(fd);
            return -1;
        }
        SaltGpuDispatchTemplate *t = &pool->dispatch[i];
        t->layer = rd32(raw);
        t->expert = rd32(raw + 4);
        t->component = rd32(raw + 8);
        t->shard_id = rd32(raw + 12);
        t->voff = rd64(raw + 16);
        t->soff = rd64(raw + 24);
        t->boff = rd64(raw + 32);
        t->v_nbytes = rd64(raw + 40);
        t->s_nbytes = rd64(raw + 48);
        t->b_nbytes = rd64(raw + 56);
        t->R = rd32(raw + 64);
        t->C = rd32(raw + 68);
        t->fmt = rd32(raw + 72);
        t->bits = rd32(raw + 76);
        t->block_size = rd32(raw + 80);
        t->pipeline_id = rd32(raw + 84);
        t->threadgroup_id = rd32(raw + 88);
        t->flags = rd32(raw + 92);
        uint32_t expected_layer = (uint32_t)(i / (pool->n_experts * 3u));
        uint32_t expected_expert = (uint32_t)((i / 3u) % pool->n_experts);
        uint32_t expected_component = (uint32_t)(i % 3u);
        const SaltGpuPoolRef *ref = salt_gpu_pool_ref(
            pool, expected_layer, expected_expert);
        if (!ref || t->layer != expected_layer ||
            t->expert != expected_expert ||
            t->component != expected_component ||
            t->shard_id != ref->shard_id || t->flags != 0 ||
            !range_in_ref(ref, t->voff, t->v_nbytes) ||
            !range_in_ref(ref, t->soff, t->s_nbytes) ||
            !range_in_ref(ref, t->boff, t->b_nbytes) ||
            !valid_q4_dispatch(t) ||
            t->pipeline_id != 1 || t->threadgroup_id != 256) {
            fprintf(stderr, "gpu-pool: invalid dispatch record %llu\n",
                    (unsigned long long)i);
            close(fd);
            return -1;
        }
    }
    for (uint64_t i = 0; i < expected_count; i += 3) {
        if (dispatch_pair_overlaps(&pool->dispatch[i], &pool->dispatch[i + 1]) ||
            dispatch_pair_overlaps(&pool->dispatch[i], &pool->dispatch[i + 2]) ||
            dispatch_pair_overlaps(&pool->dispatch[i + 1], &pool->dispatch[i + 2])) {
            fprintf(stderr, "gpu-pool: overlapping dispatch records at expert %llu\n",
                    (unsigned long long)(i / 3));
            close(fd);
            return -1;
        }
    }
    close(fd);
    return 0;
}

static int retain_source_path(SaltGpuPool *pool, const char *source_path) {
    SaltGpuPoolImpl *impl = pool ? (SaltGpuPoolImpl *)pool->impl : NULL;
    if (!impl || !source_path || !*source_path) return -1;
    size_t nbytes = strlen(source_path) + 1;
    impl->source_path = (char *)malloc(nbytes);
    if (!impl->source_path) return -1;
    memcpy(impl->source_path, source_path, nbytes);
    return 0;
}

static int derive_dispatch(SaltGpuPool *pool, const SaltPoolLayout *layout) {
    uint64_t count = (uint64_t)pool->n_layers * pool->n_experts * 3u;
    if (count > SIZE_MAX / sizeof(*pool->dispatch)) return -1;
    pool->dispatch = (SaltGpuDispatchTemplate *)calloc(
        (size_t)count, sizeof(*pool->dispatch));
    if (!pool->dispatch) return -1;
    pool->n_dispatch = count;
    for (uint32_t layer = 0; layer < pool->n_layers; layer++) {
        for (uint32_t expert = 0; expert < pool->n_experts; expert++) {
            const SaltExpertLayout *expert_layout = &layout->exp[
                (size_t)layer * pool->n_experts + expert];
            const SaltGpuPoolRef *ref = salt_gpu_pool_ref(pool, layer, expert);
            if (!ref || expert_layout->n != 3 || expert_layout->chain != 1)
                return -1;
            for (uint32_t component = 0; component < 3; component++) {
                const SaltMoETensor *tensor = &expert_layout->t[component];
                SaltGpuDispatchTemplate *dispatch = &pool->dispatch[
                    ((size_t)layer * pool->n_experts + expert) * 3u + component];
                if (tensor->rank != 2 || tensor->fmt != 1 ||
                    tensor->bits != 4 || tensor->rel_v < 0 ||
                    tensor->rel_s < 0 || tensor->rel_b < 0 ||
                    tensor->v_nbytes < 1 || tensor->s_nbytes < 1 ||
                    tensor->dims[0] < 1 || tensor->dims[1] < 1 ||
                    (uint64_t)tensor->dims[0] > UINT32_MAX ||
                    (uint64_t)tensor->dims[1] > UINT32_MAX ||
                    (uint64_t)tensor->rel_v > UINT64_MAX - ref->local_offset ||
                    (uint64_t)tensor->rel_s > UINT64_MAX - ref->local_offset ||
                    (uint64_t)tensor->rel_b > UINT64_MAX - ref->local_offset)
                    return -1;
                dispatch->layer = layer;
                dispatch->expert = expert;
                dispatch->component = component;
                dispatch->shard_id = layer;
                dispatch->voff = ref->local_offset + (uint64_t)tensor->rel_v;
                dispatch->soff = ref->local_offset + (uint64_t)tensor->rel_s;
                dispatch->boff = ref->local_offset + (uint64_t)tensor->rel_b;
                dispatch->v_nbytes = (uint64_t)tensor->v_nbytes;
                dispatch->s_nbytes = (uint64_t)tensor->s_nbytes;
                dispatch->b_nbytes = (uint64_t)tensor->s_nbytes;
                dispatch->R = (uint32_t)tensor->dims[0];
                dispatch->C = (uint32_t)tensor->dims[1];
                dispatch->fmt = (uint32_t)tensor->fmt;
                dispatch->bits = (uint32_t)tensor->bits;
                dispatch->block_size = layout->gpu.q4_default.block_size;
                dispatch->pipeline_id = layout->gpu.q4_default.pipeline_id;
                dispatch->threadgroup_id =
                    layout->gpu.q4_default.threadgroup_id;
                dispatch->flags = layout->gpu.q4_default.flags;
            }
        }
    }
    if (layout->gpu.n_override) {
        uint8_t *seen = (uint8_t *)calloc((size_t)count, 1);
        if (!seen) return -1;
        for (size_t i = 0; i < layout->gpu.n_override; i++) {
            const SaltGpuTensorOverride *override = &layout->gpu.override[i];
            if (override->layer >= pool->n_layers ||
                override->expert >= pool->n_experts ||
                override->component >= 3) {
                free(seen);
                return -1;
            }
            size_t index = ((size_t)override->layer * pool->n_experts +
                            override->expert) * 3u + override->component;
            if (seen[index]) {
                free(seen);
                return -1;
            }
            seen[index] = 1;
            SaltGpuDispatchTemplate *dispatch = &pool->dispatch[index];
            dispatch->block_size = override->policy.block_size;
            dispatch->pipeline_id = override->policy.pipeline_id;
            dispatch->threadgroup_id = override->policy.threadgroup_id;
            dispatch->flags = override->policy.flags;
        }
        free(seen);
    }
    for (uint64_t i = 0; i < count; i++) {
        const SaltGpuDispatchTemplate *dispatch = &pool->dispatch[i];
        const SaltGpuPoolRef *ref = salt_gpu_pool_ref(
            pool, dispatch->layer, dispatch->expert);
        if (!ref || dispatch->shard_id != ref->shard_id ||
            dispatch->flags != 0 || dispatch->pipeline_id != 1 ||
            dispatch->threadgroup_id != 256 ||
            !range_in_ref(ref, dispatch->voff, dispatch->v_nbytes) ||
            !range_in_ref(ref, dispatch->soff, dispatch->s_nbytes) ||
            !range_in_ref(ref, dispatch->boff, dispatch->b_nbytes) ||
            !valid_q4_dispatch(dispatch))
            return -1;
    }
    for (uint64_t i = 0; i < count; i += 3) {
        if (dispatch_pair_overlaps(&pool->dispatch[i], &pool->dispatch[i + 1]) ||
            dispatch_pair_overlaps(&pool->dispatch[i], &pool->dispatch[i + 2]) ||
            dispatch_pair_overlaps(&pool->dispatch[i + 1],
                                   &pool->dispatch[i + 2]))
            return -1;
    }
    return 0;
}

int salt_gpu_pool_from_layout(SaltGpuPool *pool,
                              const SaltPoolLayout *layout,
                              const SaltExpertPool *source,
                              const char *source_path) {
    if (!pool || !layout || !source || !source_path ||
        !layout->gpu.present || layout->gpu.version != GPU_POOL_VERSION ||
        layout->gpu.view_partition != SALT_GPU_VIEW_PARTITION_LAYER ||
        layout->gpu.resource_alignment < 4096 ||
        (layout->gpu.resource_alignment &
         (layout->gpu.resource_alignment - 1)) != 0 ||
        !layout->index_sha256_valid || !layout->source_identity.present ||
        !layout->exp || layout->n_layers < 1 || layout->n_experts < 1 ||
        layout->expert_nbytes < 1 || source->fd < 0 || !source->ref ||
        source->n_layers != layout->n_layers ||
        source->n_experts != layout->n_experts ||
        source->nbytes != layout->expert_nbytes)
        return -1;
    memset(pool, 0, sizeof *pool);
    uint64_t n_layers = (uint64_t)layout->n_layers;
    uint64_t n_experts = (uint64_t)layout->n_experts;
    uint64_t expert_nbytes = (uint64_t)layout->expert_nbytes;
    if (n_layers > UINT32_MAX || n_experts > UINT32_MAX ||
        n_layers > UINT64_MAX / n_experts)
        return -1;
    uint64_t record_count = n_layers * n_experts;
    if (record_count > SIZE_MAX / sizeof(*pool->ref) ||
        expert_nbytes > (UINT64_MAX - 24) / record_count)
        return -1;
    uint64_t source_nbytes = 24 + record_count * expert_nbytes;
    if (layout->source_identity.nbytes != source_nbytes ||
        source->map_len != (size_t)source_nbytes)
        return -1;
    pool->alignment = layout->gpu.resource_alignment;
    pool->n_layers = (uint32_t)n_layers;
    pool->n_experts = (uint32_t)n_experts;
    pool->n_shards = (uint32_t)n_layers;
    pool->expert_nbytes = expert_nbytes;
    pool->source_nbytes = source_nbytes;
    pool->ref = (SaltGpuPoolRef *)calloc(
        (size_t)record_count, sizeof(*pool->ref));
    SaltGpuPoolImpl *impl = (SaltGpuPoolImpl *)calloc(1, sizeof(*impl));
    if (!pool->ref || !impl) {
        free(pool->ref);
        free(impl);
        memset(pool, 0, sizeof *pool);
        return -1;
    }
    pool->impl = impl;
    impl->ready.source_device = layout->source_identity.device;
    impl->ready.source_inode = layout->source_identity.inode;
    impl->ready.source_nbytes = layout->source_identity.nbytes;
    impl->ready.source_mtime_ns = layout->source_identity.mtime_ns;
    impl->ready.source_ctime_ns = layout->source_identity.ctime_ns;
    memcpy(impl->layout_identity, layout->index_sha256,
           sizeof impl->layout_identity);
    impl->shard = (SaltGpuPoolShard *)calloc(
        (size_t)n_layers, sizeof(*impl->shard));
    if (!impl->shard) goto fail;
    if (expert_nbytes > UINT64_MAX / n_experts) goto fail;
    uint64_t layer_payload = expert_nbytes * n_experts;
    for (uint32_t layer = 0; layer < pool->n_layers; layer++) {
        if ((uint64_t)layer > (UINT64_MAX - 24) / layer_payload) goto fail;
        uint64_t payload_file_offset = 24 + (uint64_t)layer * layer_payload;
        uint64_t file_offset = payload_file_offset &
            ~((uint64_t)pool->alignment - 1);
        uint64_t payload_offset = payload_file_offset - file_offset;
        if (payload_offset > SIZE_MAX || layer_payload > SIZE_MAX - payload_offset)
            goto fail;
        uint64_t map_nbytes = payload_offset + layer_payload;
        if (file_offset > source_nbytes ||
            map_nbytes > source_nbytes - file_offset)
            goto fail;
        impl->shard[layer].file_offset = file_offset;
        impl->shard[layer].map_nbytes = (size_t)map_nbytes;
        impl->shard[layer].payload_offset = payload_offset;
        impl->shard[layer].payload_nbytes = layer_payload;
        for (uint32_t expert = 0; expert < pool->n_experts; expert++) {
            size_t ordinal = (size_t)layer * pool->n_experts + expert;
            SaltGpuPoolRef *ref = &pool->ref[ordinal];
            ref->layer = layer;
            ref->expert = expert;
            ref->shard_id = layer;
            ref->local_offset = payload_offset +
                (uint64_t)expert * expert_nbytes;
            ref->nbytes = expert_nbytes;
            ref->source_offset = file_offset + ref->local_offset;
            if (source->ref[ordinal].off < 0 ||
                source->ref[ordinal].nbytes != layout->expert_nbytes ||
                (uint64_t)source->ref[ordinal].off != ref->source_offset ||
                ref->source_offset > source_nbytes ||
                ref->nbytes > source_nbytes - ref->source_offset)
                goto fail;
        }
    }
    if (!source_fd_valid(pool, source->fd) ||
        retain_source_path(pool, source_path) != 0 ||
        derive_dispatch(pool, layout) != 0)
        goto fail;
    return 0;

fail:
    fprintf(stderr, "gpu-pool: unified CPU index derivation failed\n");
    salt_gpu_pool_close(pool);
    return -1;
}

int salt_gpu_pool_open(SaltGpuPool *pool, const char *directory,
                       const char *source_pool,
                       int expected_layers, int expected_experts,
                       uint64_t expected_expert_nbytes) {
    if (!pool || !directory || !source_pool ||
        expected_layers < 1 || expected_experts < 1 ||
        !expected_expert_nbytes) return -1;
    memset(pool, 0, sizeof *pool);

    char path[PATH_MAX];
    SaltGpuPoolReady ready;
    if (load_ready(directory, &ready) != 0) {
        fprintf(stderr, "gpu-pool: READY invalid in %s\n", directory);
        return -1;
    }
    if (verify_path_sha256(directory, "manifest.json",
                           ready.manifest_sha256) != 0) {
        fprintf(stderr, "gpu-pool: manifest SHA-256 mismatch\n");
        return -1;
    }
    if (make_path(path, sizeof path, directory, "INCOMPLETE") != 0 ||
        access(path, F_OK) == 0) {
        fprintf(stderr, "gpu-pool: artifact is incomplete in %s\n", directory);
        return -1;
    }
    if (make_path(path, sizeof path, directory, "index.bin") != 0) return -1;
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "gpu-pool: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    uint8_t index_sha256[32];
    if (salt_sha256_fd(fd, index_sha256) != 0 ||
        memcmp(index_sha256, ready.index_sha256, sizeof index_sha256) != 0) {
        fprintf(stderr, "gpu-pool: index SHA-256 mismatch\n");
        close(fd);
        return -1;
    }
    uint8_t hdr[INDEX_HEADER_NBYTES];
    struct stat st;
    if (read_full(fd, hdr, sizeof hdr, 0) != 0 || fstat(fd, &st) != 0 ||
        st.st_size < 0) {
        fprintf(stderr, "gpu-pool: cannot read index header/extent\n");
        close(fd);
        return -1;
    }
    uint32_t version = rd32(hdr + 8);
    uint32_t header_nbytes = rd32(hdr + 12);
    uint32_t record_nbytes = rd32(hdr + 16);
    uint32_t alignment = rd32(hdr + 20);
    uint32_t n_layers = rd32(hdr + 24);
    uint32_t n_experts = rd32(hdr + 28);
    uint32_t n_shards = rd32(hdr + 32);
    uint64_t expert_nbytes = rd64(hdr + 36);
    uint64_t source_nbytes = rd64(hdr + 44);
    uint64_t n_records = rd64(hdr + 52);
    uint64_t expected_records = (uint64_t)n_layers * n_experts;
    uint64_t expected_index = INDEX_HEADER_NBYTES +
        (uint64_t)n_shards * INDEX_SEGMENT_NBYTES +
        expected_records * INDEX_RECORD_NBYTES;
    if (memcmp(hdr, INDEX_MAGIC, 8) != 0 || version != GPU_POOL_VERSION ||
        header_nbytes != INDEX_HEADER_NBYTES ||
        record_nbytes != INDEX_RECORD_NBYTES ||
        alignment < 4096 || (alignment & (alignment - 1)) ||
        n_layers != (uint32_t)expected_layers ||
        n_experts != (uint32_t)expected_experts || n_shards != n_layers ||
        expert_nbytes != expected_expert_nbytes ||
        source_nbytes != ready.source_nbytes ||
        n_records != expected_records || expected_index > off_t_max_u64() ||
        (uint64_t)st.st_size != expected_index ||
        expected_records > SIZE_MAX / sizeof(SaltGpuPoolRef)) {
        fprintf(stderr, "gpu-pool: index version/geometry/extent mismatch\n");
        close(fd);
        return -1;
    }

    pool->alignment = alignment;
    pool->n_layers = n_layers;
    pool->n_experts = n_experts;
    pool->n_shards = n_shards;
    pool->expert_nbytes = expert_nbytes;
    pool->source_nbytes = source_nbytes;
    pool->ref = (SaltGpuPoolRef *)calloc(
        (size_t)expected_records, sizeof(SaltGpuPoolRef));
    SaltGpuPoolImpl *impl = (SaltGpuPoolImpl *)calloc(1, sizeof *impl);
    if (!pool->ref || !impl) {
        close(fd);
        free(pool->ref);
        free(impl);
        memset(pool, 0, sizeof *pool);
        return -1;
    }
    pool->impl = impl;
    impl->ready = ready;
    memcpy(impl->layout_identity, hdr + 60, sizeof impl->layout_identity);
    impl->shard = (SaltGpuPoolShard *)calloc(n_shards, sizeof *impl->shard);
    if (!impl->shard) {
        close(fd);
        salt_gpu_pool_close(pool);
        return -1;
    }

    if (expert_nbytes > UINT64_MAX / n_experts) {
        close(fd);
        salt_gpu_pool_close(pool);
        return -1;
    }
    uint64_t layer_payload = expert_nbytes * n_experts;
    uint8_t segment_raw[INDEX_SEGMENT_NBYTES];
    for (uint32_t i = 0; i < n_shards; i++) {
        off_t off = (off_t)(INDEX_HEADER_NBYTES +
                            (uint64_t)i * INDEX_SEGMENT_NBYTES);
        if (read_full(fd, segment_raw, sizeof segment_raw, off) != 0) {
            fprintf(stderr, "gpu-pool: short segment record %u\n", i);
            close(fd);
            salt_gpu_pool_close(pool);
            return -1;
        }
        uint32_t segment_id = rd32(segment_raw);
        uint32_t layer = rd32(segment_raw + 4);
        uint32_t flags = rd32(segment_raw + 8);
        uint32_t reserved = rd32(segment_raw + 12);
        uint64_t file_offset = rd64(segment_raw + 16);
        uint64_t map_nbytes = rd64(segment_raw + 24);
        uint64_t payload_offset = rd64(segment_raw + 32);
        uint64_t payload_nbytes = rd64(segment_raw + 40);
        uint64_t payload_file_offset = 24 + (uint64_t)i * layer_payload;
        uint64_t expected_file_offset =
            payload_file_offset & ~((uint64_t)alignment - 1);
        uint64_t expected_payload_offset =
            payload_file_offset - expected_file_offset;
        uint64_t expected_map_nbytes = expected_payload_offset + layer_payload;
        if (segment_id != i || layer != i || flags || reserved ||
            file_offset != expected_file_offset ||
            map_nbytes != expected_map_nbytes ||
            payload_offset != expected_payload_offset ||
            payload_nbytes != layer_payload || map_nbytes > SIZE_MAX ||
            file_offset > source_nbytes ||
            map_nbytes > source_nbytes - file_offset) {
            fprintf(stderr, "gpu-pool: invalid segment record %u\n", i);
            close(fd);
            salt_gpu_pool_close(pool);
            return -1;
        }
        impl->shard[i].file_offset = file_offset;
        impl->shard[i].map_nbytes = (size_t)map_nbytes;
        impl->shard[i].payload_offset = payload_offset;
        impl->shard[i].payload_nbytes = payload_nbytes;
    }

    uint8_t raw[INDEX_RECORD_NBYTES];
    uint64_t records_offset = INDEX_HEADER_NBYTES +
        (uint64_t)n_shards * INDEX_SEGMENT_NBYTES;
    for (uint64_t i = 0; i < n_records; i++) {
        if (read_full(fd, raw, sizeof raw,
                      (off_t)(records_offset + i * INDEX_RECORD_NBYTES)) != 0) {
            fprintf(stderr, "gpu-pool: short index record %llu\n",
                    (unsigned long long)i);
            close(fd);
            salt_gpu_pool_close(pool);
            return -1;
        }
        SaltGpuPoolRef *r = &pool->ref[i];
        r->layer = rd32(raw);
        r->expert = rd32(raw + 4);
        r->shard_id = rd32(raw + 8);
        r->flags = rd32(raw + 12);
        r->local_offset = rd64(raw + 16);
        r->nbytes = rd64(raw + 24);
        r->source_offset = rd64(raw + 32);
        uint32_t expected_layer = (uint32_t)(i / n_experts);
        uint32_t expected_expert = (uint32_t)(i % n_experts);
        const SaltGpuPoolShard *s = &impl->shard[expected_layer];
        uint64_t expected_local = s->payload_offset +
            (uint64_t)expected_expert * expert_nbytes;
        uint64_t expected_source = s->file_offset + expected_local;
        if (r->layer != expected_layer || r->expert != expected_expert ||
            r->shard_id != expected_layer || r->flags != 0 ||
            r->local_offset != expected_local || r->nbytes != expert_nbytes ||
            r->source_offset != expected_source ||
            r->source_offset > source_nbytes ||
            r->nbytes > source_nbytes - r->source_offset) {
            fprintf(stderr, "gpu-pool: invalid index record %llu\n",
                    (unsigned long long)i);
            close(fd);
            salt_gpu_pool_close(pool);
            return -1;
        }
    }
    close(fd);

    if (validate_source(pool, source_pool) != 0 ||
        load_dispatch(pool, directory, ready.dispatch_sha256) != 0) {
        salt_gpu_pool_close(pool);
        return -1;
    }
    for (uint64_t i = 0; i < n_records; i++) {
        const SaltGpuPoolRef *r = &pool->ref[i];
        const SaltGpuPoolShard *s = &impl->shard[r->shard_id];
        if (r->local_offset < s->payload_offset ||
            r->local_offset > s->map_nbytes ||
            r->nbytes > s->map_nbytes - r->local_offset ||
            r->local_offset - s->payload_offset > s->payload_nbytes ||
            r->nbytes > s->payload_nbytes -
                (r->local_offset - s->payload_offset)) {
            fprintf(stderr, "gpu-pool: record %llu is outside shard %u\n",
                    (unsigned long long)i, r->shard_id);
            salt_gpu_pool_close(pool);
            return -1;
        }
    }
    return 0;
}

const SaltGpuPoolRef *salt_gpu_pool_ref(const SaltGpuPool *pool,
                                        uint32_t layer, uint32_t expert) {
    if (!pool || !pool->ref || layer >= pool->n_layers ||
        expert >= pool->n_experts) return NULL;
    return &pool->ref[(size_t)layer * pool->n_experts + expert];
}

const uint8_t *salt_gpu_pool_layer_expert(const SaltGpuPool *pool,
                                          const SaltGpuPoolLayer *lease,
                                          uint32_t expert) {
    if (!pool || !lease || !lease->base || lease->layer >= pool->n_layers)
        return NULL;
    const SaltGpuPoolRef *r = salt_gpu_pool_ref(pool, lease->layer, expert);
    SaltGpuPoolImpl *impl = pool ? (SaltGpuPoolImpl *)pool->impl : NULL;
    if (!r || !impl || r->shard_id != lease->shard_id ||
        r->shard_id >= pool->n_shards) return NULL;
    const SaltGpuPoolShard *s = &impl->shard[r->shard_id];
    if (lease->mapped_nbytes != s->map_nbytes ||
        r->local_offset > lease->mapped_nbytes ||
        r->nbytes > lease->mapped_nbytes - r->local_offset) return NULL;
    return lease->base + (size_t)r->local_offset;
}

const SaltGpuDispatchTemplate *salt_gpu_pool_dispatch(
    const SaltGpuPool *pool, uint32_t layer, uint32_t expert,
    uint32_t component) {
    if (!pool || !pool->dispatch || layer >= pool->n_layers ||
        expert >= pool->n_experts || component >= 3) return NULL;
    size_t i = ((size_t)layer * pool->n_experts + expert) * 3u + component;
    if (i >= pool->n_dispatch) return NULL;
    return &pool->dispatch[i];
}

int salt_gpu_pool_layer_acquire(const SaltGpuPool *pool, uint32_t layer,
                                int writable_private,
                                SaltGpuPoolLayer *lease) {
    SaltGpuPoolImpl *impl = pool ? (SaltGpuPoolImpl *)pool->impl : NULL;
    if (!impl || !lease || layer >= pool->n_layers) return -1;
    memset(lease, 0, sizeof *lease);
    lease->fd = -1;
    const SaltGpuPoolShard *s = &impl->shard[layer];
    if (!impl->source_path || !s->map_nbytes ||
        s->file_offset > off_t_max_u64())
        return -1;
    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
    int fd = open(impl->source_path, flags);
    if (fd < 0 || !source_fd_valid(pool, fd)) {
        if (fd >= 0) close(fd);
        fprintf(stderr, "gpu-pool: source changed before layer %u acquire\n", layer);
        return -1;
    }
    int prot = PROT_READ | (writable_private ? PROT_WRITE : 0);
    uint8_t *base = (uint8_t *)mmap(
        NULL, s->map_nbytes, prot, MAP_PRIVATE, fd, (off_t)s->file_offset);
    if (base == MAP_FAILED) {
        close(fd);
        return -1;
    }
    if (!source_fd_valid(pool, fd)) {
        munmap(base, s->map_nbytes);
        close(fd);
        fprintf(stderr, "gpu-pool: source changed during layer %u acquire\n", layer);
        return -1;
    }
    lease->layer = layer;
    lease->shard_id = layer;
    lease->fd = fd;
    lease->base = base;
    lease->file_offset = s->file_offset;
    lease->mapped_nbytes = s->map_nbytes;
    lease->payload_offset = s->payload_offset;
    lease->payload_nbytes = s->payload_nbytes;
    return 0;
}

int salt_gpu_pool_layer_release(SaltGpuPoolLayer *lease) {
    if (!lease) return -1;
    if ((!lease->base) != (lease->mapped_nbytes == 0)) return -1;
    if (lease->base && lease->mapped_nbytes) {
        if (munmap(lease->base, lease->mapped_nbytes) != 0) return -1;
        lease->base = NULL;
        lease->mapped_nbytes = 0;
        lease->file_offset = 0;
        lease->payload_offset = 0;
        lease->payload_nbytes = 0;
    }
    if (lease->fd >= 0) {
        if (close(lease->fd) != 0) return -1;
        lease->fd = -1;
    }
    memset(lease, 0, sizeof *lease);
    lease->fd = -1;
    return 0;
}

void salt_gpu_pool_close(SaltGpuPool *pool) {
    if (!pool) return;
    close_impl(pool);
    free(pool->ref);
    free(pool->dispatch);
    memset(pool, 0, sizeof *pool);
}
