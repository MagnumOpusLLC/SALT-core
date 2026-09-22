#define _POSIX_C_SOURCE 200809L

#include "gemma4_vision.h"
#include "salt/attn.h"
#include "salt/bitmath.h"
#include "gemma4.h"
#include "salt/gpu.h"
#include "salt/gpu_resource.h"
#include "salt/kernels.h"
#include "salt/tensorops.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/../Frameworks/vecLib.framework/Headers/cblas.h>
#if defined(__clang__)
/* The stable LP64 CBLAS ABI is deprecated by the installed SDK while its
 * ILP64 umbrella header does not compile with command-line clang. */
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#endif

#define G4V_LAYERS 27
#define G4V_HIDDEN 1152
#define G4V_INTERMEDIATE 4304
#define G4V_HEADS 16
#define G4V_HEAD_DIM 72
#define G4V_PATCH 16
#define G4V_PATCH_VALUES (G4V_PATCH * G4V_PATCH * 3)
#define G4V_POSITION_SIZE 10240
#define G4V_POOL_KERNEL 3
#define G4V_TEXT_HIDDEN 2816
#define G4V_EXPECTED_BINDINGS 356
#define G4V_MAX_MAPS 4
#define G4V_MAX_PATCHES 630
#define G4V_MAX_FEATURES (G4V_MAX_PATCHES / 9)
#define G4V_EPS 0.000001f
#define G4V_ROPE_THETA 100.0f
#define G4V_MATRIX_MAX ((size_t)G4V_INTERMEDIATE * G4V_HIDDEN)

static const char G4V_SOURCE_SHA[] =
    "312e73b836f39d06f5c982f4b5a83575d7c4bb23539b787293a247cd2bc71949";
static const char G4V_ROLE_SHA[] =
    "fb4980b498cbe3985220560eca5ab0636fd1298d2f05c94f6367c93fb129c46c";

typedef struct {
    int set;
    int count;
    const unsigned char *bytes;
} G4VB;

typedef struct {
    int set;
    int bits, rows, cols;
    const uint32_t *weight;
    const uint16_t *scales;
    const uint16_t *biases;
} G4VQ;

typedef struct {
    int fd;
    char name[64];
    unsigned long long dev, ino;
    long long mtime_ns, ctime_ns;
    size_t size;
    unsigned char *base;
} G4VMap;

typedef struct {
    G4VB input_norm, post_attention_norm;
    G4VB pre_ffn_norm, post_ffn_norm;
    G4VB q_norm, k_norm;
    G4VB q_proj, k_proj, v_proj, o_proj;
    G4VB gate_proj, up_proj, down_proj;
} G4VLayer;

struct SaltGemma4Vision {
    G4VMap maps[G4V_MAX_MAPS];
    int map_count;
    int binding_count;
    G4VB patch_proj;
    G4VB position_table;
    G4VB std_bias;
    G4VB std_scale;
    G4VLayer layers[G4V_LAYERS];
    G4VQ projection;
    unsigned char *forward_arena;
    size_t forward_arena_bytes;
    float *matrix_scratch;
    float *pixels;
    float *state_a;
    float *state_b;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *branch;
    float *gate;
    float *up;
    float *scores;
    float *pooled;
    float *compact;
    float *prepared;
    float *bias;
    float *scale;
    unsigned char *valid;
    SaltKvCache compute_pool;
    int compute_pool_ready;
    int workers;
    uint64_t pool_submissions;
    int gpu_hmm_enabled;
    int gpu_resource_bound;
    int gpu_backend_owned;
    uint64_t gpu_submissions;
};

static void set_error(char *error, size_t size, const char *message) {
    if (!error || size == 0) return;
    snprintf(error, size, "%s", message ? message : "unknown Gemma 4 vision error");
}

static void set_role_error(char *error, size_t size,
                           const char *prefix, const char *role) {
    if (!error || size == 0) return;
    snprintf(error, size, "%s%s", prefix, role ? role : "<null>");
}

static long long timespec_ns(time_t sec, long nsec) {
    return (long long)sec * 1000000000LL + (long long)nsec;
}

static void stat_times(const struct stat *st, long long *mtime, long long *ctime) {
#if defined(__APPLE__)
    *mtime = timespec_ns(st->st_mtime, st->st_mtimensec);
    *ctime = timespec_ns(st->st_ctime, st->st_ctimensec);
#else
    *mtime = timespec_ns(st->st_mtim.tv_sec, st->st_mtim.tv_nsec);
    *ctime = timespec_ns(st->st_ctim.tv_sec, st->st_ctim.tv_nsec);
#endif
}

static int map_identity_matches(const G4VMap *map) {
    struct stat st;
    long long mtime, ctime;
    if (!map || map->fd < 0 || fstat(map->fd, &st) != 0 ||
        !S_ISREG(st.st_mode)) return 0;
    stat_times(&st, &mtime, &ctime);
    return (unsigned long long)st.st_dev == map->dev &&
           (unsigned long long)st.st_ino == map->ino &&
           (size_t)st.st_size == map->size &&
           mtime == map->mtime_ns && ctime == map->ctime_ns;
}

static int all_identities_match(const SaltGemma4Vision *model) {
    if (!model || model->map_count < 1) return 0;
    for (int i = 0; i < model->map_count; i++)
        if (!map_identity_matches(&model->maps[i])) return 0;
    return 1;
}

static G4VMap *find_map(SaltGemma4Vision *model, int fd) {
    for (int i = 0; i < model->map_count; i++)
        if (model->maps[i].fd == fd) return &model->maps[i];
    return NULL;
}

static int checked_extent(const G4VMap *map, unsigned long long offset,
                          unsigned long long bytes, const void **pointer) {
    if (!map || !pointer || bytes == 0 || offset > map->size ||
        bytes > map->size - (size_t)offset) return -1;
    *pointer = map->base + (size_t)offset;
    return 0;
}

static int add_map(SaltGemma4Vision *model, const char *name, int fd,
                   unsigned long long dev, unsigned long long ino,
                   unsigned long long bytes, long long mtime, long long ctime) {
    struct stat st;
    long long actual_mtime, actual_ctime;
    if (!model || model->map_count >= G4V_MAX_MAPS || fd < 0 || !name ||
        strlen(name) >= 64 || bytes == 0 || bytes > SIZE_MAX ||
        fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) return -1;
    stat_times(&st, &actual_mtime, &actual_ctime);
    if ((unsigned long long)st.st_dev != dev ||
        (unsigned long long)st.st_ino != ino ||
        (unsigned long long)st.st_size != bytes ||
        actual_mtime != mtime || actual_ctime != ctime || find_map(model, fd))
        return -1;
    /* Immutable authenticated vision tensors remain clean file-backed pages;
     * PROT_READ prevents mutation while MAP_SHARED avoids private compression. */
    void *base = mmap(NULL, (size_t)bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) return -1;
    G4VMap *map = &model->maps[model->map_count++];
    memset(map, 0, sizeof *map);
    map->fd = fd;
    snprintf(map->name, sizeof map->name, "%s", name);
    map->dev = dev;
    map->ino = ino;
    map->mtime_ns = mtime;
    map->ctime_ns = ctime;
    map->size = (size_t)bytes;
    map->base = (unsigned char *)base;
    return 0;
}

static int load_bf16(SaltGemma4Vision *model, int count, int fd,
                     unsigned long long offset, unsigned long long bytes,
                     G4VB *out) {
    const void *pointer = NULL;
    if (!model || !out || out->set || count < 1 ||
        bytes != (unsigned long long)count * 2u ||
        checked_extent(find_map(model, fd), offset, bytes, &pointer) != 0)
        return -1;
    out->set = 1;
    out->count = count;
    out->bytes = (const unsigned char *)pointer;
    return 0;
}

static int load_q4(SaltGemma4Vision *model, int bits, int rows, int cols,
                   int wfd, unsigned long long woff, unsigned long long wn,
                   int sfd, unsigned long long soff, unsigned long long sn,
                   int bfd, unsigned long long boff, unsigned long long bn,
                   G4VQ *out) {
    const void *weight = NULL, *scales = NULL, *biases = NULL;
    unsigned long long ew, ea;
    if (!model || !out || out->set || bits != 4 || rows < 1 || cols < 1 ||
        cols % 64 != 0) return -1;
    ew = (unsigned long long)rows * (unsigned long long)cols / 2u;
    ea = (unsigned long long)rows * (unsigned long long)(cols / 64) * 2u;
    if (wn != ew || sn != ea || bn != ea ||
        checked_extent(find_map(model, wfd), woff, wn, &weight) != 0 ||
        checked_extent(find_map(model, sfd), soff, sn, &scales) != 0 ||
        checked_extent(find_map(model, bfd), boff, bn, &biases) != 0)
        return -1;
    out->set = 1;
    out->bits = bits;
    out->rows = rows;
    out->cols = cols;
    out->weight = (const uint32_t *)weight;
    out->scales = (const uint16_t *)scales;
    out->biases = (const uint16_t *)biases;
    return 0;
}

static int layer_suffix(const char *role, int *layer, const char **suffix) {
    int used = -1;
    if (!role || !layer || !suffix ||
        sscanf(role, "vision.layers.%d.%n", layer, &used) != 1 || used < 0 ||
        *layer < 0 || *layer >= G4V_LAYERS || role[used] == 0)
        return -1;
    *suffix = role + used;
    return 0;
}

static G4VB *b_role(SaltGemma4Vision *model, const char *role) {
    int layer;
    const char *suffix;
    if (!strcmp(role, "vision.patch_embedder.input_proj")) return &model->patch_proj;
    if (!strcmp(role, "vision.patch_embedder.position_embedding_table"))
        return &model->position_table;
    if (!strcmp(role, "vision.std_bias")) return &model->std_bias;
    if (!strcmp(role, "vision.std_scale")) return &model->std_scale;
    if (layer_suffix(role, &layer, &suffix) != 0) return NULL;
    G4VLayer *l = &model->layers[layer];
    if (!strcmp(suffix, "input_layernorm")) return &l->input_norm;
    if (!strcmp(suffix, "post_attention_layernorm")) return &l->post_attention_norm;
    if (!strcmp(suffix, "pre_feedforward_layernorm")) return &l->pre_ffn_norm;
    if (!strcmp(suffix, "post_feedforward_layernorm")) return &l->post_ffn_norm;
    if (!strcmp(suffix, "self_attn.q_norm")) return &l->q_norm;
    if (!strcmp(suffix, "self_attn.k_norm")) return &l->k_norm;
    if (!strcmp(suffix, "self_attn.q_proj")) return &l->q_proj;
    if (!strcmp(suffix, "self_attn.k_proj")) return &l->k_proj;
    if (!strcmp(suffix, "self_attn.v_proj")) return &l->v_proj;
    if (!strcmp(suffix, "self_attn.o_proj")) return &l->o_proj;
    if (!strcmp(suffix, "mlp.gate_proj")) return &l->gate_proj;
    if (!strcmp(suffix, "mlp.up_proj")) return &l->up_proj;
    if (!strcmp(suffix, "mlp.down_proj")) return &l->down_proj;
    return NULL;
}

static int b_count(const G4VB *b, int count) {
    return b && b->set && b->count == count;
}

static int validate_closure(SaltGemma4Vision *model) {
    if (!model || model->binding_count != G4V_EXPECTED_BINDINGS ||
        !b_count(&model->patch_proj, G4V_HIDDEN * G4V_PATCH_VALUES) ||
        !b_count(&model->position_table,
                 2 * G4V_POSITION_SIZE * G4V_HIDDEN) ||
        !b_count(&model->std_bias, G4V_HIDDEN) ||
        !b_count(&model->std_scale, G4V_HIDDEN) ||
        !model->projection.set || model->projection.bits != 4 ||
        model->projection.rows != G4V_TEXT_HIDDEN ||
        model->projection.cols != G4V_HIDDEN)
        return -1;
    for (int i = 0; i < G4V_LAYERS; i++) {
        G4VLayer *l = &model->layers[i];
        if (!b_count(&l->input_norm, G4V_HIDDEN) ||
            !b_count(&l->post_attention_norm, G4V_HIDDEN) ||
            !b_count(&l->pre_ffn_norm, G4V_HIDDEN) ||
            !b_count(&l->post_ffn_norm, G4V_HIDDEN) ||
            !b_count(&l->q_norm, G4V_HEAD_DIM) ||
            !b_count(&l->k_norm, G4V_HEAD_DIM) ||
            !b_count(&l->q_proj, G4V_HIDDEN * G4V_HIDDEN) ||
            !b_count(&l->k_proj, G4V_HIDDEN * G4V_HIDDEN) ||
            !b_count(&l->v_proj, G4V_HIDDEN * G4V_HIDDEN) ||
            !b_count(&l->o_proj, G4V_HIDDEN * G4V_HIDDEN) ||
            !b_count(&l->gate_proj, G4V_INTERMEDIATE * G4V_HIDDEN) ||
            !b_count(&l->up_proj, G4V_INTERMEDIATE * G4V_HIDDEN) ||
            !b_count(&l->down_proj, G4V_HIDDEN * G4V_INTERMEDIATE))
            return -1;
    }
    return 0;
}

static int g4v_gpu_register_bf16(SaltGemma4Vision *model,
                                 const G4VB *matrix, int rows, int cols) {
    if (!model || !matrix || !matrix->set ||
        matrix->count != rows * cols || rows < 1 || cols < 8 || (cols & 7))
        return -1;
    return salt_gpu_bf16_resource_slot(
        SALT_GPU_RESOURCE_TRUNK, 14, matrix->bytes,
        (const uint16_t *)(const void *)matrix->bytes, rows, cols);
}

static int g4v_gpu_hmm_init(SaltGemma4Vision *model) {
    const char *requested = getenv("SALT_CUDA_PAGEABLE_MMAP");
    const char *vision_only = getenv("SALT_GEMMA_VISION_CUDA_HMM");
    if (!model || (requested && strcmp(requested, "0") &&
                   strcmp(requested, "1")) ||
        (vision_only && strcmp(vision_only, "0") &&
         strcmp(vision_only, "1")))
        return -1;
    if (!requested || !strcmp(requested, "0"))
        return vision_only && !strcmp(vision_only, "1") ? -1 : 0;
    if (vision_only && !strcmp(vision_only, "1") &&
        !salt_gpu_pageable_mmap_active()) {
        if (salt_gpu_init() != 0 || !salt_gpu_pageable_mmap_active())
            return -1;
        model->gpu_backend_owned = 1;
    }
    if (!salt_gpu_pageable_mmap_active() || model->map_count != 1 ||
        salt_gpu_weight_resource_describe(
            SALT_GPU_RESOURCE_TRUNK, 14,
            model->maps[0].base, model->maps[0].size) != 0)
        return -1;
    model->gpu_resource_bound = 1;
    if (salt_gpu_weight_resource_activate(
            SALT_GPU_RESOURCE_TRUNK, 14,
            SALT_GPU_WEIGHT_ADDRESS_PAGEABLE) != 0 ||
        g4v_gpu_register_bf16(model, &model->patch_proj,
            G4V_HIDDEN, G4V_PATCH_VALUES) != 0)
        goto fail;
    for (int layer = 0; layer < G4V_LAYERS; layer++) {
        G4VLayer *entry = &model->layers[layer];
        if (g4v_gpu_register_bf16(model, &entry->q_proj,
                G4V_HIDDEN, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->k_proj,
                G4V_HIDDEN, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->v_proj,
                G4V_HIDDEN, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->o_proj,
                G4V_HIDDEN, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->gate_proj,
                G4V_INTERMEDIATE, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->up_proj,
                G4V_INTERMEDIATE, G4V_HIDDEN) != 0 ||
            g4v_gpu_register_bf16(model, &entry->down_proj,
                G4V_HIDDEN, G4V_INTERMEDIATE) != 0)
            goto fail;
    }
    if (salt_gpu_weight_resource_slot(
            SALT_GPU_RESOURCE_TRUNK, 14,
            model->projection.weight, model->projection.weight,
            model->projection.scales, model->projection.biases,
            4, G4V_TEXT_HIDDEN, G4V_HIDDEN) != 0)
        goto fail;
    model->gpu_hmm_enabled = 1;
    return 0;
fail:
    (void)salt_gpu_weight_resource_unbind(
        SALT_GPU_RESOURCE_TRUNK, 14);
    model->gpu_resource_bound = 0;
    if (model->gpu_backend_owned) {
        (void)salt_gpu_free();
        model->gpu_backend_owned = 0;
    }
    return -1;
}

static int g4v_forward_arena_init(SaltGemma4Vision *model) {
    size_t floats = 0, bytes;
    float *cursor;
    if (!model || model->forward_arena) return -1;
#define ADD_FLOATS(count) do { \
    size_t add = (size_t)(count); \
    if (floats > SIZE_MAX - add) return -1; \
    floats += add; \
} while (0)
    ADD_FLOATS(G4V_MATRIX_MAX);
    ADD_FLOATS((size_t)G4V_MAX_PATCHES * G4V_PATCH_VALUES);
    for (int i = 0; i < 8; i++)
        ADD_FLOATS((size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    for (int i = 0; i < 2; i++)
        ADD_FLOATS((size_t)G4V_MAX_PATCHES * G4V_INTERMEDIATE);
    ADD_FLOATS((size_t)G4V_MAX_PATCHES * G4V_MAX_PATCHES);
    for (int i = 0; i < 3; i++)
        ADD_FLOATS((size_t)G4V_MAX_FEATURES * G4V_HIDDEN);
    ADD_FLOATS((size_t)2 * G4V_HIDDEN);
#undef ADD_FLOATS
    if (floats > (SIZE_MAX - G4V_MAX_FEATURES) / sizeof(float)) return -1;
    bytes = floats * sizeof(float) + G4V_MAX_FEATURES;
    model->forward_arena = (unsigned char *)calloc(1, bytes);
    if (!model->forward_arena) return -1;
    model->forward_arena_bytes = bytes;
    cursor = (float *)(void *)model->forward_arena;
#define BIND_FLOAT(field, count) do { \
    model->field = cursor; \
    cursor += (size_t)(count); \
} while (0)
    BIND_FLOAT(matrix_scratch, G4V_MATRIX_MAX);
    BIND_FLOAT(pixels, (size_t)G4V_MAX_PATCHES * G4V_PATCH_VALUES);
    BIND_FLOAT(state_a, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(state_b, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(norm, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(q, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(k, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(v, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(attn, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(branch, (size_t)G4V_MAX_PATCHES * G4V_HIDDEN);
    BIND_FLOAT(gate, (size_t)G4V_MAX_PATCHES * G4V_INTERMEDIATE);
    BIND_FLOAT(up, (size_t)G4V_MAX_PATCHES * G4V_INTERMEDIATE);
    BIND_FLOAT(scores, (size_t)G4V_MAX_PATCHES * G4V_MAX_PATCHES);
    BIND_FLOAT(pooled, (size_t)G4V_MAX_FEATURES * G4V_HIDDEN);
    BIND_FLOAT(compact, (size_t)G4V_MAX_FEATURES * G4V_HIDDEN);
    BIND_FLOAT(prepared, (size_t)G4V_MAX_FEATURES * G4V_HIDDEN);
    BIND_FLOAT(bias, G4V_HIDDEN);
    BIND_FLOAT(scale, G4V_HIDDEN);
#undef BIND_FLOAT
    model->valid = model->forward_arena + floats * sizeof(float);
    return model->valid + G4V_MAX_FEATURES ==
        model->forward_arena + model->forward_arena_bytes ? 0 : -1;
}

SaltGemma4Vision *salt_gemma4_vision_load(FILE *binding, int workers,
                                          char *error, size_t error_size) {
    char line[640];
    int auth_seen = 0, model_seen = 0, end_seen = 0;
    uint16_t endian = 1;
    SaltGemma4Vision *model = NULL;
    if (!binding || workers < 1 || workers > 32 ||
        *(unsigned char *)&endian != 1) {
        set_error(error, error_size, "invalid Gemma 4 vision loader arguments");
        return NULL;
    }
    model = (SaltGemma4Vision *)calloc(1, sizeof *model);
    if (!model) {
        set_error(error, error_size, "Gemma 4 vision loader allocation failed");
        return NULL;
    }
    model->workers = workers;
    if (!fgets(line, sizeof line, binding) ||
        strcmp(line, "SALT_GEMMA4_VISION_BINDING_V1\n")) {
        set_error(error, error_size, "invalid Gemma 4 vision binding header");
        goto fail;
    }
    while (fgets(line, sizeof line, binding)) {
        size_t length = strlen(line);
        if (length == 0 || line[length - 1] != '\n') {
            set_error(error, error_size, "unterminated Gemma 4 vision binding line");
            goto fail;
        }
        line[length - 1] = 0;
        if (!strncmp(line, "AUTH ", 5)) {
            char source[65], role[65], extra;
            if (auth_seen || sscanf(line, "AUTH %64s %64s %c",
                    source, role, &extra) != 2 ||
                strcmp(source, G4V_SOURCE_SHA) || strcmp(role, G4V_ROLE_SHA)) {
                set_error(error, error_size, "Gemma 4 vision authority drift");
                goto fail;
            }
            auth_seen = 1;
        } else if (!strncmp(line, "MODEL ", 6)) {
            int layers, hidden, intermediate, heads, head_dim, patch;
            int position_size, pool, text_hidden;
            float eps, theta;
            char extra;
            if (model_seen || sscanf(line,
                "MODEL %d %d %d %d %d %d %d %d %d %f %f %c",
                &layers, &hidden, &intermediate, &heads, &head_dim, &patch,
                &position_size, &pool, &text_hidden, &eps, &theta, &extra) != 11 ||
                layers != G4V_LAYERS || hidden != G4V_HIDDEN ||
                intermediate != G4V_INTERMEDIATE || heads != G4V_HEADS ||
                head_dim != G4V_HEAD_DIM || patch != G4V_PATCH ||
                position_size != G4V_POSITION_SIZE || pool != G4V_POOL_KERNEL ||
                text_hidden != G4V_TEXT_HIDDEN || eps != G4V_EPS ||
                theta != G4V_ROPE_THETA) {
                set_error(error, error_size, "Gemma 4 vision model drift");
                goto fail;
            }
            model_seen = 1;
        } else if (!strncmp(line, "FD ", 3)) {
            char name[64], extra;
            int fd;
            unsigned long long dev, ino, bytes;
            long long mtime, ctime;
            if (sscanf(line, "FD %63s %d %llu %llu %llu %lld %lld %c",
                    name, &fd, &dev, &ino, &bytes, &mtime, &ctime, &extra) != 7 ||
                add_map(model, name, fd, dev, ino, bytes, mtime, ctime) != 0) {
                set_error(error, error_size, "invalid retained vision descriptor");
                goto fail;
            }
        } else if (!strncmp(line, "B ", 2)) {
            char role[128], extra;
            int count, fd;
            unsigned long long offset, bytes;
            if (sscanf(line, "B %127s %d %d %llu %llu %c",
                    role, &count, &fd, &offset, &bytes, &extra) != 5) {
                set_error(error, error_size, "invalid vision BF16 record");
                goto fail;
            }
            G4VB *target = b_role(model, role);
            if (!target || load_bf16(model, count, fd, offset, bytes, target) != 0) {
                set_role_error(error, error_size, "invalid vision BF16 role: ", role);
                goto fail;
            }
            model->binding_count++;
        } else if (!strncmp(line, "Q ", 2)) {
            char role[128], extra;
            int bits, rows, cols, wfd, sfd, bfd;
            unsigned long long woff, wn, soff, sn, boff, bn;
            if (sscanf(line,
                "Q %127s %d %d %d %d %llu %llu %d %llu %llu %d %llu %llu %c",
                role, &bits, &rows, &cols, &wfd, &woff, &wn,
                &sfd, &soff, &sn, &bfd, &boff, &bn, &extra) != 13 ||
                strcmp(role, "vision_to_text.projection") ||
                load_q4(model, bits, rows, cols, wfd, woff, wn,
                        sfd, soff, sn, bfd, boff, bn, &model->projection) != 0) {
                set_role_error(error, error_size, "invalid vision projection role: ", role);
                goto fail;
            }
            model->binding_count++;
        } else if (!strcmp(line, "END")) {
            end_seen = 1;
            break;
        } else {
            set_error(error, error_size, "unknown Gemma 4 vision binding record");
            goto fail;
        }
    }
    if (!auth_seen || !model_seen || !end_seen || model->map_count != 1 ||
        validate_closure(model) != 0 || !all_identities_match(model)) {
        set_error(error, error_size, "incomplete Gemma 4 vision binding closure");
        goto fail;
    }
    if (g4v_gpu_hmm_init(model) != 0) {
        set_error(error, error_size,
                  "Gemma 4 vision CUDA HMM initialization failed");
        goto fail;
    }
    if (g4v_forward_arena_init(model) != 0) {
        set_error(error, error_size, "Gemma 4 vision forward arena allocation failed");
        goto fail;
    }
#if !defined(__APPLE__)
    model->compute_pool.nthreads = workers;
    model->compute_pool.max_tokens = 1;
    model->compute_pool.aq4_threads = workers;
    model->compute_pool.apool_threads = workers;
    if (salt_attn_pool_init(&model->compute_pool) != 0) {
        set_error(error, error_size,
                  "Gemma 4 vision compute pool initialization failed");
        goto fail;
    }
    model->compute_pool_ready = 1;
#endif
    return model;

fail:
    salt_gemma4_vision_free(model);
    return NULL;
}

int salt_gemma4_vision_close(SaltGemma4Vision **model_pointer) {
    SaltGemma4Vision *model;
    if (!model_pointer) return -1;
    model = *model_pointer;
    if (!model) return 0;
    if (model->gpu_resource_bound &&
        salt_gpu_weight_resource_unbind(
            SALT_GPU_RESOURCE_TRUNK, 14) != 0)
        return -1;
    model->gpu_resource_bound = 0;
    model->gpu_hmm_enabled = 0;
    if (model->gpu_backend_owned && salt_gpu_free() != 0)
        return -1;
    model->gpu_backend_owned = 0;
    salt_kv_free(&model->compute_pool);
    model->compute_pool_ready = 0;
    for (int i = 0; i < model->map_count; i++)
        if (model->maps[i].base)
            munmap(model->maps[i].base, model->maps[i].size);
    free(model->forward_arena);
    model->forward_arena = NULL;
    model->matrix_scratch = NULL;
    free(model);
    *model_pointer = NULL;
    return 0;
}

void salt_gemma4_vision_free(SaltGemma4Vision *model) {
    SaltGemma4Vision *owned = model;
    if (salt_gemma4_vision_close(&owned) != 0)
        fputs("Gemma 4 vision teardown failed; retaining source mapping\n",
              stderr);
}

int salt_gemma4_vision_worker_stats(const SaltGemma4Vision *model,
                                    int *workers, uint64_t *submissions) {
    if (!model || !workers || !submissions) return -1;
    *workers = model->workers;
    *submissions = model->pool_submissions;
    return 0;
}

int salt_gemma4_vision_gpu_stats(const SaltGemma4Vision *model,
                                 int *pageable_hmm,
                                 uint64_t *submissions) {
    if (!model || !pageable_hmm || !submissions) return -1;
    *pageable_hmm = model->gpu_hmm_enabled;
    *submissions = model->gpu_submissions;
    return 0;
}

static float bf16_value(const unsigned char *bytes, size_t index) {
    uint16_t encoded;
    uint32_t bits;
    float value;
    memcpy(&encoded, bytes + index * 2u, sizeof encoded);
    bits = (uint32_t)encoded << 16;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
        bits &= UINT32_C(0xffff0000);
    }
    memcpy(&value, &bits, sizeof value);
    return value;
}

float salt_gemma4_vision_postnorm_residual_bf16(float residual,
                                                 float normalized) {
    return round_bf16(residual + round_bf16(normalized));
}

float salt_gemma4_vision_gelu_mul_bf16(float gate, float up) {
    return round_bf16(round_bf16(salt_gemma4_gelu_tanh(gate)) * up);
}

static void round_array(float *values, size_t count) {
    for (size_t i = 0; i < count; i++) values[i] = round_bf16(values[i]);
}

static int decode_vector(const G4VB *tensor, float *out, int count) {
    if (!tensor || !tensor->set || tensor->count != count || !out) return -1;
    for (int i = 0; i < count; i++) {
        out[i] = bf16_value(tensor->bytes, (size_t)i);
        if (!isfinite(out[i])) return -1;
    }
    return 0;
}

#if !defined(__APPLE__)
typedef struct {
    const unsigned char *weights;
    int rows, cols, tokens, workers;
    const float *input;
    float *output;
} G4VBf16PoolTask;

static void g4v_bf16_pool_worker(int worker, void *argument) {
    G4VBf16PoolTask *task = (G4VBf16PoolTask *)argument;
    int begin = task->tokens * worker / task->workers;
    int end = task->tokens * (worker + 1) / task->workers;
    for (int token = begin; token < end; token++) {
        const float *input = task->input + (size_t)token * task->cols;
        float *output = task->output + (size_t)token * task->rows;
        for (int row = 0; row < task->rows; row++) {
            float sum = 0.0f;
            size_t base = (size_t)row * task->cols;
            for (int column = 0; column < task->cols; column++) {
                float product = bf16_value(
                    task->weights, base + (size_t)column) * input[column];
                sum += product;
            }
            output[row] = sum;
        }
    }
}
#endif

static int bf16_batch(SaltGemma4Vision *model, const G4VB *matrix,
                      int rows, int cols, int tokens,
                      const float *input, float *output) {
    size_t weight_count = (size_t)rows * (size_t)cols;
    size_t output_count = (size_t)tokens * (size_t)rows;
    int computed = 0;
    if (!model || !matrix || !matrix->set || matrix->count != (int)weight_count ||
        weight_count > G4V_MATRIX_MAX || tokens < 1 || !input || !output)
        return -1;
    if (model->gpu_hmm_enabled) {
        if (salt_gpu_bf16_matvec_batch(
                matrix->bytes, rows, cols, tokens, input, output) != 0)
            return -1;
        model->gpu_submissions++;
        computed = 1;
    }
#if !defined(__APPLE__)
    if (!computed) {
        int active = model->workers < tokens ? model->workers : tokens;
        G4VBf16PoolTask task = {
            matrix->bytes, rows, cols, tokens, active, input, output,
        };
        if (active > 1) {
            if (!model->compute_pool_ready ||
                salt_attn_pool_run_n(&model->compute_pool, active,
                    g4v_bf16_pool_worker, &task) != 0)
                return -1;
            model->pool_submissions++;
        } else {
            g4v_bf16_pool_worker(0, &task);
        }
        computed = 1;
    }
#endif
    if (!computed) {
        for (size_t i = 0; i < weight_count; i++) {
            float value = bf16_value(matrix->bytes, i);
            if (!isfinite(value)) return -1;
            model->matrix_scratch[i] = value;
        }
#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tokens, rows, cols, 1.0f,
                    input, cols, model->matrix_scratch, cols,
                    0.0f, output, rows);
        computed = 1;
#endif
        if (!computed) {
            for (int t = 0; t < tokens; t++) {
                for (int r = 0; r < rows; r++) {
                    float sum = 0.0f;
                    const float *w = model->matrix_scratch + (size_t)r * cols;
                    const float *x = input + (size_t)t * cols;
                    for (int c = 0; c < cols; c++) {
                        float product = w[c] * x[c];
                        sum += product;
                    }
                    output[(size_t)t * rows + r] = sum;
                }
            }
        }
    }
    for (size_t i = 0; i < output_count; i++) {
        if (!isfinite(output[i])) return -1;
        output[i] = round_bf16(output[i]);
    }
    return 0;
}

static int q4_batch(SaltGemma4Vision *model, const G4VQ *matrix,
                    int tokens, const float *input, float *output) {
    if (!model || !matrix || !matrix->set || matrix->bits != 4 ||
        matrix->rows != G4V_TEXT_HIDDEN || matrix->cols != G4V_HIDDEN ||
        tokens < 1 || tokens > G4V_MAX_FEATURES || !input || !output)
        return -1;
    if (model->gpu_hmm_enabled) {
        const float *inputs[G4V_MAX_FEATURES];
        float *outputs[G4V_MAX_FEATURES];
        for (int token = 0; token < tokens; token++) {
            inputs[token] = input + (size_t)token * G4V_HIDDEN;
            outputs[token] = output + (size_t)token * G4V_TEXT_HIDDEN;
        }
        if (salt_gpu_proj_batch(
                matrix->weight, matrix->scales, matrix->biases,
                matrix->rows, matrix->cols, tokens,
                inputs, outputs, matrix->weight) != 0)
            return -1;
        model->gpu_submissions++;
    } else {
        for (int token = 0; token < tokens; token++)
            salt_q4_matvec(matrix->weight, matrix->scales, matrix->biases,
                matrix->rows, matrix->cols,
                input + (size_t)token * G4V_HIDDEN,
                output + (size_t)token * G4V_TEXT_HIDDEN);
    }
    for (size_t i = 0; i < (size_t)tokens * G4V_TEXT_HIDDEN; i++) {
        if (!isfinite(output[i])) return -1;
        output[i] = round_bf16(output[i]);
    }
    return 0;
}

static int norm_batch(float *out, const float *input, const G4VB *weight,
                      int tokens, int width, int with_scale) {
    float stack_weight[G4V_HIDDEN];
    float *decoded = NULL;
    if (!out || !input || tokens < 1 || width < 1 || width > G4V_HIDDEN)
        return -1;
    if (with_scale) {
        if (!weight || decode_vector(weight, stack_weight, width) != 0) return -1;
        decoded = stack_weight;
    }
    for (int t = 0; t < tokens; t++) {
        if (salt_gemma4_rmsnorm(out + (size_t)t * width,
                input + (size_t)t * width, decoded, width, G4V_EPS,
                with_scale) != 0) return -1;
    }
    round_array(out, (size_t)tokens * width);
    return 0;
}

static int add_positions(float *states, const G4VB *table,
                         const int *positions, const unsigned char *padding,
                         int tokens) {
    size_t plane = (size_t)G4V_POSITION_SIZE * G4V_HIDDEN;
    if (!states || !table || !table->set ||
        table->count != 2 * G4V_POSITION_SIZE * G4V_HIDDEN ||
        !positions || !padding) return -1;
    for (int t = 0; t < tokens; t++) {
        int x = positions[2 * t], y = positions[2 * t + 1];
        if (padding[t] > 1 || (!padding[t] &&
            (x < 0 || y < 0 || x >= G4V_POSITION_SIZE || y >= G4V_POSITION_SIZE)))
            return -1;
        if (padding[t]) continue;
        for (int h = 0; h < G4V_HIDDEN; h++) {
            float px = bf16_value(table->bytes, (size_t)x * G4V_HIDDEN + h);
            float py = bf16_value(table->bytes, plane + (size_t)y * G4V_HIDDEN + h);
            size_t index = (size_t)t * G4V_HIDDEN + h;
            states[index] = round_bf16(states[index] + round_bf16(px + py));
        }
    }
    return 0;
}

static int attention(float *output, const float *q, const float *k,
                     const float *v, const unsigned char *padding,
                     int tokens, float *scores) {
    int valid = 0;
    if (!output || !q || !k || !v || !padding || !scores || tokens < 1)
        return -1;
    for (int t = 0; t < tokens; t++) {
        if (padding[t] > 1) return -1;
        if (!padding[t]) valid++;
    }
    if (valid < 1) return -1;
    memset(output, 0, (size_t)tokens * G4V_HIDDEN * sizeof(float));
    for (int head = 0; head < G4V_HEADS; head++) {
        const float *qh = q + (size_t)head * G4V_HEAD_DIM;
        const float *kh = k + (size_t)head * G4V_HEAD_DIM;
        const float *vh = v + (size_t)head * G4V_HEAD_DIM;
        float *oh = output + (size_t)head * G4V_HEAD_DIM;
        int qk_computed = 0, value_computed = 0;
#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    tokens, tokens, G4V_HEAD_DIM, 1.0f,
                    qh, G4V_HIDDEN, kh, G4V_HIDDEN,
                    0.0f, scores, tokens);
        qk_computed = 1;
#endif
        if (!qk_computed) {
            for (int i = 0; i < tokens; i++)
                for (int j = 0; j < tokens; j++) {
                    float sum = 0.0f;
                    for (int d = 0; d < G4V_HEAD_DIM; d++) {
                        float product = qh[(size_t)i * G4V_HIDDEN + d] *
                                        kh[(size_t)j * G4V_HIDDEN + d];
                        sum += product;
                    }
                    scores[(size_t)i * tokens + j] = sum;
                }
        }
        for (int i = 0; i < tokens; i++) {
            float maximum = -INFINITY, sum = 0.0f;
            float *row = scores + (size_t)i * tokens;
            for (int j = 0; j < tokens; j++) {
                if (padding[j]) row[j] = -INFINITY;
                else {
                    if (!isfinite(row[j])) return -1;
                    row[j] = round_bf16(row[j]);
                    if (row[j] > maximum) maximum = row[j];
                }
            }
            for (int j = 0; j < tokens; j++) {
                if (padding[j]) row[j] = 0.0f;
                else {
                    row[j] = salt_expf(row[j] - maximum);
                    sum += row[j];
                }
            }
            if (!(sum > 0.0f) || !isfinite(sum)) return -1;
            for (int j = 0; j < tokens; j++)
                row[j] = round_bf16(row[j] / sum);
        }
#if defined(__APPLE__)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tokens, G4V_HEAD_DIM, tokens, 1.0f,
                    scores, tokens, vh, G4V_HIDDEN,
                    0.0f, oh, G4V_HIDDEN);
        value_computed = 1;
#endif
        if (!value_computed) {
            for (int i = 0; i < tokens; i++)
                for (int d = 0; d < G4V_HEAD_DIM; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < tokens; j++) {
                        float product = scores[(size_t)i * tokens + j] *
                                        vh[(size_t)j * G4V_HIDDEN + d];
                        sum += product;
                    }
                    oh[(size_t)i * G4V_HIDDEN + d] = sum;
                }
        }
    }
    round_array(output, (size_t)tokens * G4V_HIDDEN);
    return 0;
}

static int postnorm_residual_batch(float *out, const float *residual,
                                   const float *branch, const G4VB *weight,
                                   int tokens) {
    float decoded[G4V_HIDDEN];
    if (!out || !residual || !branch ||
        decode_vector(weight, decoded, G4V_HIDDEN) != 0) return -1;
    for (int t = 0; t < tokens; t++) {
        float *row = out + (size_t)t * G4V_HIDDEN;
        const float *residual_row = residual + (size_t)t * G4V_HIDDEN;
        if (salt_gemma4_rmsnorm(row,
                branch + (size_t)t * G4V_HIDDEN,
                decoded, G4V_HIDDEN, G4V_EPS, 1) != 0)
            return -1;
        for (int h = 0; h < G4V_HIDDEN; h++)
            row[h] = salt_gemma4_vision_postnorm_residual_bf16(
                residual_row[h], row[h]);
    }
    return 0;
}

static int run_layer(SaltGemma4Vision *model, G4VLayer *layer,
                     float *current, float *next,
                     const int *positions, const unsigned char *padding,
                     int tokens, float *norm, float *q, float *k, float *v,
                     float *attn, float *branch, float *gate, float *up,
                     float *scores) {
    if (norm_batch(norm, current, &layer->input_norm,
                   tokens, G4V_HIDDEN, 1) != 0 ||
        bf16_batch(model, &layer->q_proj, G4V_HIDDEN, G4V_HIDDEN,
                   tokens, norm, q) != 0 ||
        bf16_batch(model, &layer->k_proj, G4V_HIDDEN, G4V_HIDDEN,
                   tokens, norm, k) != 0 ||
        bf16_batch(model, &layer->v_proj, G4V_HIDDEN, G4V_HIDDEN,
                   tokens, norm, v) != 0)
        return -1;
    float q_weight[G4V_HEAD_DIM], k_weight[G4V_HEAD_DIM];
    if (decode_vector(&layer->q_norm, q_weight, G4V_HEAD_DIM) != 0 ||
        decode_vector(&layer->k_norm, k_weight, G4V_HEAD_DIM) != 0)
        return -1;
    for (int t = 0; t < tokens; t++) {
        for (int h = 0; h < G4V_HEADS; h++) {
            float *qh = q + (size_t)t * G4V_HIDDEN + (size_t)h * G4V_HEAD_DIM;
            float *kh = k + (size_t)t * G4V_HIDDEN + (size_t)h * G4V_HEAD_DIM;
            float *vh = v + (size_t)t * G4V_HIDDEN + (size_t)h * G4V_HEAD_DIM;
            if (salt_gemma4_rmsnorm(qh, qh, q_weight,
                    G4V_HEAD_DIM, G4V_EPS, 1) != 0 ||
                salt_gemma4_rmsnorm(kh, kh, k_weight,
                    G4V_HEAD_DIM, G4V_EPS, 1) != 0 ||
                salt_gemma4_rmsnorm(vh, vh, NULL,
                    G4V_HEAD_DIM, G4V_EPS, 0) != 0 ||
                salt_gemma4_vision_rope(qh, G4V_HEAD_DIM,
                    positions[2 * t], positions[2 * t + 1],
                    G4V_ROPE_THETA) != 0 ||
                salt_gemma4_vision_rope(kh, G4V_HEAD_DIM,
                    positions[2 * t], positions[2 * t + 1],
                    G4V_ROPE_THETA) != 0)
                return -1;
            round_array(qh, G4V_HEAD_DIM);
            round_array(kh, G4V_HEAD_DIM);
            round_array(vh, G4V_HEAD_DIM);
        }
    }
    if (attention(attn, q, k, v, padding, tokens, scores) != 0 ||
        bf16_batch(model, &layer->o_proj, G4V_HIDDEN, G4V_HIDDEN,
                   tokens, attn, branch) != 0 ||
        postnorm_residual_batch(next, current, branch,
                                &layer->post_attention_norm, tokens) != 0)
        return -1;
    memcpy(current, next, (size_t)tokens * G4V_HIDDEN * sizeof(float));
    if (norm_batch(norm, current, &layer->pre_ffn_norm,
                   tokens, G4V_HIDDEN, 1) != 0 ||
        bf16_batch(model, &layer->gate_proj, G4V_INTERMEDIATE, G4V_HIDDEN,
                   tokens, norm, gate) != 0 ||
        bf16_batch(model, &layer->up_proj, G4V_INTERMEDIATE, G4V_HIDDEN,
                   tokens, norm, up) != 0)
        return -1;
    for (size_t i = 0; i < (size_t)tokens * G4V_INTERMEDIATE; i++)
        gate[i] = salt_gemma4_vision_gelu_mul_bf16(gate[i], up[i]);
    if (bf16_batch(model, &layer->down_proj, G4V_HIDDEN, G4V_INTERMEDIATE,
                   tokens, gate, branch) != 0 ||
        postnorm_residual_batch(next, current, branch,
                                &layer->post_ffn_norm, tokens) != 0)
        return -1;
    return 0;
}

typedef struct G4VForwardRun {
    SaltGemma4Vision *model;
    const float *patches;
    const int *positions;
    const unsigned char *padding;
    int n_patches;
    float *features;
    int feature_capacity;
    int output_length;
    int valid_count;
    float *current;
    float *next;
    char *error;
    size_t error_size;
} G4VForwardRun;

static int g4v_forward_node(void *opaque, uint32_t node) {
    G4VForwardRun *run = (G4VForwardRun *)opaque;
    SaltGemma4Vision *model;
    size_t pixel_count;
    if (!run || !(model = run->model)) return -1;
    pixel_count = (size_t)run->n_patches * G4V_PATCH_VALUES;
    if (node == 0u) {
        memcpy(model->pixels, run->patches, pixel_count * sizeof(float));
        if (salt_gemma4_vision_normalize_pixels(
                model->pixels, (int)pixel_count) != 0 ||
            bf16_batch(model, &model->patch_proj, G4V_HIDDEN,
                G4V_PATCH_VALUES, run->n_patches,
                model->pixels, model->state_a) != 0 ||
            add_positions(model->state_a, &model->position_table,
                run->positions, run->padding, run->n_patches) != 0) {
            set_error(run->error, run->error_size,
                      "Gemma 4 vision patch embedding failed");
            return -1;
        }
        run->current = model->state_a;
        run->next = model->state_b;
        return 0;
    }
    if (node <= G4V_LAYERS) {
        uint32_t layer = node - 1u;
        fprintf(stderr, "[vision] layer=%u/%d\n", layer + 1u, G4V_LAYERS);
        fflush(stderr);
        if (run_layer(model, &model->layers[layer],
                run->current, run->next, run->positions, run->padding,
                run->n_patches, model->norm, model->q, model->k, model->v,
                model->attn, model->branch, model->gate, model->up,
                model->scores) != 0) {
            set_error(run->error, run->error_size,
                      "Gemma 4 vision encoder layer failed");
            return -1;
        }
        {
            float *swap = run->current;
            run->current = run->next;
            run->next = swap;
        }
        return 0;
    }
    if (node == G4V_LAYERS + 1u) {
        if (salt_gemma4_vision_pool(model->pooled, model->valid,
                run->current, run->positions, run->padding,
                run->n_patches, run->output_length, G4V_HIDDEN) != 0) {
            set_error(run->error, run->error_size,
                      "Gemma 4 vision pooling failed");
            return -1;
        }
        run->valid_count = 0;
        for (int token = 0; token < run->output_length; token++) {
            if (!model->valid[token]) continue;
            memcpy(model->compact + (size_t)run->valid_count * G4V_HIDDEN,
                   model->pooled + (size_t)token * G4V_HIDDEN,
                   G4V_HIDDEN * sizeof(float));
            run->valid_count++;
        }
        if (run->valid_count < 1 || run->valid_count > run->feature_capacity ||
            decode_vector(&model->std_bias, model->bias, G4V_HIDDEN) != 0 ||
            decode_vector(&model->std_scale, model->scale, G4V_HIDDEN) != 0 ||
            salt_gemma4_vision_standardize(model->compact,
                model->bias, model->scale,
                run->valid_count, G4V_HIDDEN) != 0 ||
            salt_gemma4_vision_prepare_projection(model->prepared,
                model->compact, run->valid_count, G4V_HIDDEN, G4V_EPS) != 0) {
            set_error(run->error, run->error_size,
                      "Gemma 4 vision projection preparation failed");
            return -1;
        }
        if (q4_batch(model, &model->projection, run->valid_count,
                model->prepared, run->features) != 0) {
            set_error(run->error, run->error_size,
                      "Gemma 4 image projection failed");
            return -1;
        }
        return 0;
    }
    return -1;
}

int salt_gemma4_vision_forward(SaltGemma4Vision *model,
                               const float *patches,
                               const int *positions_xy,
                               const unsigned char *padding,
                               int n_patches,
                               float *features,
                               int feature_token_capacity,
                               char *error, size_t error_size) {
    G4VForwardRun run;
    SaltTensorProgramCallbacks program;
    uint32_t executed = 0;
    if (!model || !patches || !positions_xy || !padding || !features ||
        n_patches < G4V_POOL_KERNEL * G4V_POOL_KERNEL ||
        n_patches > G4V_MAX_PATCHES || n_patches % 9 != 0 ||
        feature_token_capacity < 1 || !model->forward_arena ||
        !all_identities_match(model)) {
        set_error(error, error_size, "invalid Gemma 4 vision execution arguments");
        return -1;
    }
    memset(model->forward_arena, 0, model->forward_arena_bytes);
    run = (G4VForwardRun) {
        model, patches, positions_xy, padding, n_patches,
        features, feature_token_capacity, n_patches / 9, 0,
        NULL, NULL, error, error_size,
    };
    program = (SaltTensorProgramCallbacks) {
        G4V_LAYERS + 2u, g4v_forward_node,
    };
    if (salt_tensor_program_execute(&program, &run, &executed) != 0 ||
        executed != program.node_count)
        return -1;
    if (!all_identities_match(model)) {
        set_error(error, error_size, "Gemma 4 vision descriptor identity changed");
        return -1;
    }
    return run.valid_count;
}

static int supported_soft_tokens(int value) {
    return value == 70;
}

static int read_ppm_token(FILE *stream, char *token, size_t capacity) {
    int ch;
    size_t used = 0;
    if (!stream || !token || capacity < 2) return -1;
    do {
        ch = fgetc(stream);
        if (ch == '#') {
            do ch = fgetc(stream); while (ch != '\n' && ch != EOF);
        }
    } while (ch != EOF && isspace((unsigned char)ch));
    if (ch == EOF) return -1;
    while (ch != EOF && !isspace((unsigned char)ch)) {
        if (used + 1 >= capacity) return -1;
        token[used++] = (char)ch;
        ch = fgetc(stream);
    }
    token[used] = 0;
    return used ? 0 : -1;
}

static int parse_int_token(FILE *stream, int *value) {
    char token[32], *end = NULL;
    long parsed;
    if (read_ppm_token(stream, token, sizeof token) != 0) return -1;
    errno = 0;
    parsed = strtol(token, &end, 10);
    if (errno || !end || *end || parsed < 1 || parsed > INT_MAX) return -1;
    *value = (int)parsed;
    return 0;
}

static int canonical_size(int height, int width, int max_soft_tokens,
                          int *target_height, int *target_width) {
    double target_pixels, factor, ideal_h, ideal_w;
    int side, h, w, max_side;
    if (height < 1 || width < 1 || !supported_soft_tokens(max_soft_tokens) ||
        !target_height || !target_width) return -1;
    target_pixels = (double)max_soft_tokens * 9.0 * 256.0;
    factor = sqrt(target_pixels / ((double)height * width));
    ideal_h = factor * height;
    ideal_w = factor * width;
    side = 48;
    h = (int)floor(ideal_h / side) * side;
    w = (int)floor(ideal_w / side) * side;
    if (h == 0 && w == 0) return -1;
    max_side = (max_soft_tokens * 9 / 9) * side;
    if (h == 0) {
        uint64_t ratio = (uint64_t)width / (uint64_t)height;
        h = side;
        w = ratio > (uint64_t)max_side / (uint64_t)side
            ? max_side : (int)(ratio * (uint64_t)side);
    } else if (w == 0) {
        uint64_t ratio = (uint64_t)height / (uint64_t)width;
        w = side;
        h = ratio > (uint64_t)max_side / (uint64_t)side
            ? max_side : (int)(ratio * (uint64_t)side);
    }
    if ((long long)h * w > (long long)max_soft_tokens * 9 * 256)
        return -1;
    *target_height = h;
    *target_width = w;
    return 0;
}

int salt_gemma4_ppm_to_patches(const char *path, int max_soft_tokens,
                               float **patches,
                               int **positions_xy,
                               unsigned char **padding,
                               int *n_patches,
                               int *valid_soft_tokens,
                               int *width, int *height,
                               char *error, size_t error_size) {
    FILE *stream = NULL;
    unsigned char *pixels = NULL;
    float *out_patches = NULL;
    int *out_positions = NULL;
    unsigned char *out_padding = NULL;
    char magic[8];
    int w, h, maximum, target_w, target_h, patch_w, patch_h;
    int max_patches, actual_patches;
    size_t pixel_bytes, patch_values;
    if (!path || !patches || !positions_xy || !padding || !n_patches ||
        !valid_soft_tokens || !width || !height ||
        !supported_soft_tokens(max_soft_tokens)) {
        set_error(error, error_size, "invalid Gemma 4 PPM processor arguments");
        return -1;
    }
    *patches = NULL; *positions_xy = NULL; *padding = NULL;
    stream = fopen(path, "rb");
    if (!stream || read_ppm_token(stream, magic, sizeof magic) != 0 ||
        strcmp(magic, "P6") || parse_int_token(stream, &w) != 0 ||
        parse_int_token(stream, &h) != 0 || parse_int_token(stream, &maximum) != 0 ||
        maximum != 255 || canonical_size(h, w, max_soft_tokens,
                                          &target_h, &target_w) != 0) {
        set_error(error, error_size, "invalid Gemma 4 P6 image");
        goto fail;
    }
    if (target_h != h || target_w != w) {
        set_error(error, error_size,
                  "Gemma 4 image requires unimplemented bicubic resize");
        goto fail;
    }
    if (h % G4V_PATCH || w % G4V_PATCH) {
        set_error(error, error_size, "Gemma 4 image dimensions are not patch aligned");
        goto fail;
    }
    if ((size_t)w > SIZE_MAX / (size_t)h / 3u) goto fail;
    pixel_bytes = (size_t)w * h * 3u;
    pixels = (unsigned char *)malloc(pixel_bytes);
    if (!pixels || fread(pixels, 1, pixel_bytes, stream) != pixel_bytes) {
        set_error(error, error_size, "short Gemma 4 P6 pixel payload");
        goto fail;
    }
    max_patches = max_soft_tokens * 9;
    patch_w = w / G4V_PATCH;
    patch_h = h / G4V_PATCH;
    actual_patches = patch_w * patch_h;
    if (actual_patches > max_patches || actual_patches % 9 != 0) {
        set_error(error, error_size, "Gemma 4 image patch budget mismatch");
        goto fail;
    }
    patch_values = (size_t)max_patches * G4V_PATCH_VALUES;
    out_patches = (float *)calloc(patch_values, sizeof(float));
    out_positions = (int *)malloc((size_t)max_patches * 2u * sizeof(int));
    out_padding = (unsigned char *)malloc((size_t)max_patches);
    if (!out_patches || !out_positions || !out_padding) {
        set_error(error, error_size, "Gemma 4 PPM patch allocation failed");
        goto fail;
    }
    for (int p = 0; p < max_patches; p++) {
        out_positions[2 * p] = -1;
        out_positions[2 * p + 1] = -1;
        out_padding[p] = 1;
    }
    for (int py = 0; py < patch_h; py++) {
        for (int px = 0; px < patch_w; px++) {
            int p = py * patch_w + px;
            out_positions[2 * p] = px;
            out_positions[2 * p + 1] = py;
            out_padding[p] = 0;
            for (int y = 0; y < G4V_PATCH; y++)
                for (int x = 0; x < G4V_PATCH; x++)
                    for (int c = 0; c < 3; c++) {
                        size_t src = ((size_t)(py * G4V_PATCH + y) * w +
                                      (size_t)(px * G4V_PATCH + x)) * 3u + c;
                        size_t dst = (size_t)p * G4V_PATCH_VALUES +
                                     ((size_t)y * G4V_PATCH + x) * 3u + c;
                        out_patches[dst] = (float)pixels[src] / 255.0f;
                    }
        }
    }
    *patches = out_patches;
    *positions_xy = out_positions;
    *padding = out_padding;
    *n_patches = max_patches;
    *valid_soft_tokens = actual_patches / 9;
    *width = w;
    *height = h;
    free(pixels);
    fclose(stream);
    return 0;

fail:
    free(out_padding); free(out_positions); free(out_patches); free(pixels);
    if (stream) fclose(stream);
    return -1;
}

void salt_gemma4_free_patches(float *patches, int *positions_xy,
                              unsigned char *padding) {
    free(padding);
    free(positions_xy);
    free(patches);
}
