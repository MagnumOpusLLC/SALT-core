#define _DARWIN_C_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#include "gemma4_mtp.h"
#include "gemma4.h"
#include "salt/gpu.h"
#include "salt/kernels.h"
#include "salt/rotation/fail.h"
#include "../../mtp/source_layout_q4.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(O_CLOEXEC)
#define O_CLOEXEC 0
#endif
#if !defined(O_NOFOLLOW)
#define O_NOFOLLOW 0
#endif

enum {
    MTP_LAYERS = 4,
    MTP_HIDDEN = 1024,
    MTP_INTERMEDIATE = 8192,
    MTP_TARGET_HIDDEN = 2816,
    MTP_VOCAB = 262144,
    MTP_HEADS = 16,
    MTP_SLIDING_HEAD_DIM = 256,
    MTP_FULL_HEAD_DIM = 512,
};

typedef struct MtpLayer {
    const unsigned char *q_proj;
    const unsigned char *o_proj;
    const unsigned char *gate_proj;
    const unsigned char *up_proj;
    const unsigned char *down_proj;
    const float *input_norm;
    const float *q_norm;
    const float *post_attention_norm;
    const float *pre_feedforward_norm;
    const float *post_feedforward_norm;
    float layer_scalar;
    int full_attention;
    int head_dim;
} MtpLayer;

struct SaltGemma4Mtp {
    int fd;
    const unsigned char *mapping;
    size_t mapping_bytes;
    size_t mapping_page_bytes;
    SaltGpuSharedBuffer source_buffer;
    SaltGpuSharedBuffer scratch_buffer;
    int metal;
    SaltGemma4MtpTargetOps target;
    const unsigned char *embedding_head;
    const unsigned char *pre_projection;
    const unsigned char *post_projection;
    const float *final_norm;
    MtpLayer layers[MTP_LAYERS];
    float *scratch;
    size_t scratch_floats;
    float *target_embedding;
    float *combined;
    float *state_a;
    float *state_b;
    float *normalized;
    float *queries;
    float *attention;
    float *branch;
    float *gate;
    float *up;
    float *ffn;
    float *logits;
    float *projected;
    SaltGemma4MtpStats stats;
};

typedef struct MtpReceipt {
    unsigned long long dev;
    unsigned long long ino;
    unsigned long long size;
    long long mtime_ns;
    char revision[65];
    char sha256[65];
} MtpReceipt;

typedef struct MtpBf16Task {
    const unsigned char *weights;
    const float *input;
    float *output;
    int rows;
    int columns;
    int workers;
    int failed[32];
} MtpBf16Task;

static void set_error(char *error, size_t error_size, const char *message) {
    if (error && error_size) snprintf(error, error_size, "%s", message ? message : "unknown MTP error");
}

static uint64_t stat_mtime_ns(const struct stat *st) {
#if defined(__APPLE__)
    return (uint64_t)st->st_mtimespec.tv_sec * UINT64_C(1000000000) +
           (uint64_t)st->st_mtimespec.tv_nsec;
#else
    return (uint64_t)st->st_mtim.tv_sec * UINT64_C(1000000000) +
           (uint64_t)st->st_mtim.tv_nsec;
#endif
}

static int read_receipt(const char *path, MtpReceipt *receipt) {
    FILE *file;
    char line[256];
    unsigned found = 0;
    if (!path || !receipt || !(file = fopen(path, "r"))) return SALT_TFAIL(-1);
    memset(receipt, 0, sizeof *receipt);
    if (!fgets(line, sizeof line, file) ||
        strcmp(line, "SALT_GEMMA4_MTP_Q4_RECEIPT_V1\n")) {
        fclose(file);
        return SALT_TFAIL(-1);
    }
    while (fgets(line, sizeof line, file)) {
        char extra;
        if (sscanf(line, "revision %64s %c", receipt->revision, &extra) == 1 && !(found & 1u)) found |= 1u;
        else if (sscanf(line, "sha256 %64s %c", receipt->sha256, &extra) == 1 && !(found & 2u)) found |= 2u;
        else if (sscanf(line, "dev %llu %c", &receipt->dev, &extra) == 1 && !(found & 4u)) found |= 4u;
        else if (sscanf(line, "ino %llu %c", &receipt->ino, &extra) == 1 && !(found & 8u)) found |= 8u;
        else if (sscanf(line, "size %llu %c", &receipt->size, &extra) == 1 && !(found & 16u)) found |= 16u;
        else if (sscanf(line, "mtime_ns %lld %c", &receipt->mtime_ns, &extra) == 1 && !(found & 32u)) found |= 32u;
        else if (!strcmp(line, "END\n") || !strcmp(line, "END")) { found |= 64u; break; }
        else { fclose(file); return SALT_TFAIL(-1); }
    }
    int bad = ferror(file);
    fclose(file);
    if (bad || found != 127u ||
        strcmp(receipt->revision, SALT_MTP_Q4_SOURCE_REVISION) ||
        strcmp(receipt->sha256, SALT_MTP_Q4_SOURCE_SHA256) ||
        receipt->size != SALT_MTP_Q4_SOURCE_BYTES)
        return SALT_TFAIL(-1);
    return 0;
}

static float f16_to_float(uint16_t encoded) {
    uint32_t sign = (uint32_t)(encoded & UINT16_C(0x8000)) << 16;
    uint32_t exponent = (encoded >> 10) & 31u;
    uint32_t mantissa = encoded & 1023u;
    uint32_t bits;
    float value;
    if (exponent == 0u) {
        if (mantissa == 0u) bits = sign;
        else {
            exponent = 113u;
            while ((mantissa & 1024u) == 0u) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 1023u;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    memcpy(&value, &bits, sizeof value);
    return value;
}

static uint16_t load_bf16(const void *pointer) {
    uint16_t value;
    memcpy(&value, pointer, sizeof value);
    return value;
}

static const unsigned char *weight_pointer(const SaltGemma4Mtp *mtp,
                                           uint64_t offset, uint64_t bytes) {
    if (!mtp || !mtp->mapping || offset > mtp->mapping_bytes ||
        bytes > mtp->mapping_bytes - (size_t)offset)
        return NULL;
    return mtp->mapping + (size_t)offset;
}

static const float *float_pointer(const SaltGemma4Mtp *mtp,
                                  uint64_t offset, uint32_t count) {
    uint64_t bytes = (uint64_t)count * sizeof(float);
    if (!mtp || !mtp->mapping || (offset & 3u) || offset > mtp->mapping_bytes ||
        bytes > mtp->mapping_bytes - (size_t)offset)
        return NULL;
    return (const float *)(const void *)(mtp->mapping + (size_t)offset);
}

static float q4_weight(const unsigned char *block, int index) {
    uint16_t encoded_scale = load_bf16(block);
    float scale = f16_to_float(encoded_scale);
    unsigned byte = block[2u + (unsigned)(index & 15)];
    int code = index < 16 ? (int)(byte & 15u) : (int)(byte >> 4);
    return scale * (float)(code - 8);
}

static void mtp_q4_worker(int worker, void *opaque) {
    MtpBf16Task *task = (MtpBf16Task *)opaque;
    int chunk = task->rows / task->workers + (task->rows % task->workers != 0);
    int64_t wide_first = (int64_t)worker * chunk;
    int64_t wide_last = wide_first + chunk;
    int first = wide_first < task->rows ? (int)wide_first : task->rows;
    int last = wide_last < task->rows ? (int)wide_last : task->rows;
    size_t row_bytes = (size_t)(task->columns / 32) * 18u;
    if (first >= last || task->columns % 32 != 0) return;
    for (int row = first; row < last; row++) {
        const unsigned char *source = task->weights + (size_t)row * row_bytes;
        float accum[8] = {0};
        for (int column = 0; column < task->columns; column++) {
            const unsigned char *block = source + (size_t)(column / 32) * 18u;
            accum[column & 7] += q4_weight(block, column & 31) * task->input[column];
        }
        task->output[row] =
            ((accum[0] + accum[2]) + (accum[4] + accum[6])) +
            ((accum[1] + accum[3]) + (accum[5] + accum[7]));
    }
}

static int mtp_matvec(SaltGemma4Mtp *mtp, const unsigned char *weights,
                      int rows, int columns, const float *input, float *output) {
    MtpBf16Task task;
    int workers;
    if (!mtp || !weights || rows < 1 || columns < 1 || !input || !output ||
        columns % 32 != 0 ||
        !mtp->target.compute_pool || mtp->target.compute_pool->aq4_threads < 1 ||
        mtp->target.compute_pool->aq4_threads > 32)
        return SALT_TFAIL(-1);
    if (mtp->metal) {
        size_t weight_offset, input_offset, output_offset;
        if (weights < mtp->mapping || input < mtp->scratch ||
            output < mtp->scratch)
            return SALT_TFAIL(-1);
        weight_offset = (size_t)(weights - mtp->mapping);
        input_offset = (size_t)(input - mtp->scratch);
        output_offset = (size_t)(output - mtp->scratch);
        if (input_offset >= mtp->scratch_floats ||
            output_offset >= mtp->scratch_floats ||
            salt_gpu_mtp_q4_0_matvec(
                &mtp->source_buffer, weight_offset,
                &mtp->scratch_buffer, input_offset, output_offset,
                mtp->scratch_floats * sizeof(float), rows, columns) != 0)
            return SALT_TFAIL(-1);
        mtp->stats.matvecs++;
        mtp->stats.metal_matvecs++;
        return 0;
    }
    workers = mtp->target.compute_pool->aq4_threads;
    if (workers > rows) workers = rows;
    memset(&task, 0, sizeof task);
    task.weights = weights;
    task.input = input;
    task.output = output;
    task.rows = rows;
    task.columns = columns;
    task.workers = workers;
    if (salt_attn_pool_run_n(mtp->target.compute_pool, workers,
                             mtp_q4_worker, &task) != 0)
        return SALT_TFAIL(-1);
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return SALT_TFAIL(-1);
    mtp->stats.matvecs++;
    return 0;
}

static int mtp_argmax(const float *logits) {
    int best = -1;
    float best_value = -INFINITY;
    if (!logits) return SALT_TFAIL(-1);
    for (int token = 0; token < MTP_VOCAB; token++) {
        float value = logits[token];
        if (!isfinite(value)) return SALT_TFAIL(-1);
        if (best < 0 || value > best_value) {
            best = token;
            best_value = value;
        }
    }
    return best;
}

static int mtp_forward_step(SaltGemma4Mtp *mtp, int32_t token,
                            const float *hidden, int position,
                            int32_t *prediction, float *next_hidden) {
    float *current, *next;
    if (!mtp || token < 0 || token >= MTP_VOCAB || !hidden || position < 0 ||
        !prediction || !next_hidden ||
        mtp->target.target_embedding(mtp->target.context, token,
                                     mtp->target_embedding) != 0)
        return SALT_TFAIL(-1);
    memcpy(mtp->combined, mtp->target_embedding,
           MTP_TARGET_HIDDEN * sizeof(float));
    memcpy(mtp->combined + MTP_TARGET_HIDDEN, hidden,
           MTP_TARGET_HIDDEN * sizeof(float));
    if (mtp_matvec(mtp, mtp->pre_projection, MTP_HIDDEN,
                   2 * MTP_TARGET_HIDDEN, mtp->combined, mtp->state_a) != 0)
        return SALT_TFAIL(-1);
    current = mtp->state_a;
    next = mtp->state_b;
    for (int layer_index = 0; layer_index < MTP_LAYERS; layer_index++) {
        MtpLayer *layer = &mtp->layers[layer_index];
        int q_width = MTP_HEADS * layer->head_dim;
        if (salt_gemma4_rmsnorm(mtp->normalized, current,
                layer->input_norm, MTP_HIDDEN, 1.0e-6f, 1) != 0 ||
            mtp_matvec(mtp, layer->q_proj, q_width, MTP_HIDDEN,
                       mtp->normalized, mtp->queries) != 0)
            return SALT_TFAIL(-1);
        for (int head = 0; head < MTP_HEADS; head++) {
            float *query = mtp->queries + (size_t)head * layer->head_dim;
            if (salt_gemma4_rmsnorm(query, query, layer->q_norm,
                    layer->head_dim, 1.0e-6f, 1) != 0 ||
                salt_gemma4_text_rope(query, layer->head_dim,
                    layer->full_attention ? 128 : layer->head_dim,
                    layer->head_dim, position,
                    layer->full_attention ? 1000000.0f : 10000.0f) != 0)
                return SALT_TFAIL(-1);
        }
        if (mtp->target.shared_attention(mtp->target.context,
                layer->full_attention, position,
                mtp->queries, mtp->attention) != 0 ||
            mtp_matvec(mtp, layer->o_proj, MTP_HIDDEN, q_width,
                       mtp->attention, mtp->branch) != 0 ||
            salt_gemma4_rmsnorm(mtp->branch, mtp->branch,
                layer->post_attention_norm, MTP_HIDDEN, 1.0e-6f, 1) != 0)
            return SALT_TFAIL(-1);
        mtp->stats.attention_calls++;
        for (int index = 0; index < MTP_HIDDEN; index++)
            next[index] = current[index] + mtp->branch[index];
        if (salt_gemma4_rmsnorm(mtp->normalized, next,
                layer->pre_feedforward_norm, MTP_HIDDEN, 1.0e-6f, 1) != 0 ||
            mtp_matvec(mtp, layer->gate_proj, MTP_INTERMEDIATE, MTP_HIDDEN,
                       mtp->normalized, mtp->gate) != 0 ||
            mtp_matvec(mtp, layer->up_proj, MTP_INTERMEDIATE, MTP_HIDDEN,
                       mtp->normalized, mtp->up) != 0)
            return SALT_TFAIL(-1);
        for (int index = 0; index < MTP_INTERMEDIATE; index++)
            mtp->gate[index] = salt_gemma4_gelu_tanh(mtp->gate[index]) *
                               mtp->up[index];
        if (mtp_matvec(mtp, layer->down_proj, MTP_HIDDEN, MTP_INTERMEDIATE,
                       mtp->gate, mtp->ffn) != 0 ||
            salt_gemma4_rmsnorm(mtp->ffn, mtp->ffn,
                layer->post_feedforward_norm, MTP_HIDDEN, 1.0e-6f, 1) != 0)
            return SALT_TFAIL(-1);
        for (int index = 0; index < MTP_HIDDEN; index++)
            next[index] = (next[index] + mtp->ffn[index]) * layer->layer_scalar;
        {
            float *swap = current;
            current = next;
            next = swap;
        }
    }
    if (salt_gemma4_rmsnorm(mtp->normalized, current, mtp->final_norm,
            MTP_HIDDEN, 1.0e-6f, 1) != 0 ||
        mtp_matvec(mtp, mtp->embedding_head, MTP_VOCAB, MTP_HIDDEN,
                   mtp->normalized, mtp->logits) != 0 ||
        mtp_matvec(mtp, mtp->post_projection, MTP_TARGET_HIDDEN, MTP_HIDDEN,
                   mtp->normalized, next_hidden) != 0)
        return SALT_TFAIL(-1);
    *prediction = mtp_argmax(mtp->logits);
    mtp->stats.steps++;
    return *prediction >= 0 ? 0 : -1;
}

static int bind_layer(SaltGemma4Mtp *mtp, int layer_index,
                      uint64_t input_norm, uint64_t scalar,
                      uint64_t down, uint64_t gate, uint64_t up,
                      uint64_t post_attention, uint64_t post_feedforward,
                      uint64_t pre_feedforward, uint64_t output,
                      uint64_t q_norm, uint64_t query) {
    MtpLayer *layer;
    const float *scalar_value;
    if (!mtp || layer_index < 0 || layer_index >= MTP_LAYERS)
        return SALT_TFAIL(-1);
    layer = &mtp->layers[layer_index];
    layer->full_attention = layer_index == 3;
    layer->head_dim = layer->full_attention ? MTP_FULL_HEAD_DIM : MTP_SLIDING_HEAD_DIM;
    layer->input_norm = float_pointer(mtp, input_norm, MTP_HIDDEN);
    layer->post_attention_norm = float_pointer(
        mtp, post_attention, MTP_HIDDEN);
    layer->pre_feedforward_norm = float_pointer(
        mtp, pre_feedforward, MTP_HIDDEN);
    layer->post_feedforward_norm = float_pointer(
        mtp, post_feedforward, MTP_HIDDEN);
    layer->q_norm = float_pointer(
        mtp, q_norm, (uint32_t)layer->head_dim);
    scalar_value = float_pointer(mtp, scalar, 1u);
    layer->q_proj = weight_pointer(mtp, query,
        (uint64_t)MTP_HEADS * layer->head_dim * MTP_HIDDEN / 32u * 18u);
    layer->o_proj = weight_pointer(mtp, output,
        (uint64_t)MTP_HIDDEN * MTP_HEADS * layer->head_dim / 32u * 18u);
    layer->gate_proj = weight_pointer(mtp, gate,
        (uint64_t)MTP_INTERMEDIATE * MTP_HIDDEN / 32u * 18u);
    layer->up_proj = weight_pointer(mtp, up,
        (uint64_t)MTP_INTERMEDIATE * MTP_HIDDEN / 32u * 18u);
    layer->down_proj = weight_pointer(mtp, down,
        (uint64_t)MTP_HIDDEN * MTP_INTERMEDIATE / 32u * 18u);
    if (!layer->q_proj || !layer->o_proj || !layer->gate_proj ||
        !layer->up_proj || !layer->down_proj || !layer->input_norm ||
        !layer->post_attention_norm || !layer->pre_feedforward_norm ||
        !layer->post_feedforward_norm || !layer->q_norm || !scalar_value)
        return SALT_TFAIL(-1);
    layer->layer_scalar = *scalar_value;
    return isfinite(layer->layer_scalar) ? 0 : -1;
}

SaltGemma4Mtp *salt_gemma4_mtp_load(
        const char *source_path, const char *receipt_path,
        const SaltGemma4MtpTargetOps *target,
        char *error, size_t error_size) {
    MtpReceipt receipt;
    struct stat st;
    SaltGemma4Mtp *mtp = NULL;
    uint32_t gguf_header[2] = {0};
    size_t scratch_floats;
    if (!source_path || !receipt_path || !target || !target->context ||
        !target->compute_pool || !target->target_embedding ||
        !target->shared_attention || !target->is_stop ||
        read_receipt(receipt_path, &receipt) != 0) {
        set_error(error, error_size, "invalid Gemma 4 MTP source receipt");
        return NULL;
    }
    mtp = (SaltGemma4Mtp *)calloc(1, sizeof *mtp);
    if (!mtp) {
        set_error(error, error_size, "Gemma 4 MTP allocation failed");
        return NULL;
    }
    mtp->fd = -1;
    mtp->fd = open(source_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (mtp->fd < 0 || fstat(mtp->fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        (unsigned long long)st.st_dev != receipt.dev ||
        (unsigned long long)st.st_ino != receipt.ino ||
        (unsigned long long)st.st_size != receipt.size ||
        stat_mtime_ns(&st) != (uint64_t)receipt.mtime_ns ||
        (uint64_t)st.st_size != SALT_MTP_Q4_SOURCE_BYTES ||
        pread(mtp->fd, gguf_header, sizeof gguf_header, 0) !=
            (ssize_t)sizeof gguf_header ||
        gguf_header[0] != UINT32_C(0x46554747) || gguf_header[1] != 3u) {
        set_error(error, error_size, "Gemma 4 MTP source identity mismatch");
        salt_gemma4_mtp_free(mtp);
        return NULL;
    }
    mtp->mapping_bytes = (size_t)st.st_size;
    mtp->mapping = mmap(NULL, mtp->mapping_bytes, PROT_READ, MAP_SHARED,
                        mtp->fd, 0);
    if (mtp->mapping == MAP_FAILED) {
        mtp->mapping = NULL;
        set_error(error, error_size, "Gemma 4 MTP source mapping failed");
        salt_gemma4_mtp_free(mtp);
        return NULL;
    }
    mtp->target = *target;

    scratch_floats = MTP_TARGET_HIDDEN + 2u * MTP_TARGET_HIDDEN +
        2u * MTP_HIDDEN + MTP_HIDDEN + 2u * (MTP_HEADS * MTP_FULL_HEAD_DIM) +
        MTP_HIDDEN + 2u * MTP_INTERMEDIATE + MTP_HIDDEN +
        MTP_VOCAB + MTP_TARGET_HIDDEN;
    mtp->scratch_floats = scratch_floats;
    if (salt_gpu_metal_present()) {
        long page = sysconf(_SC_PAGESIZE);
        if (page < 1 || mtp->mapping_bytes > SIZE_MAX - ((size_t)page - 1u)) {
            set_error(error, error_size, "Gemma 4 MTP Metal mapping geometry failed");
            salt_gemma4_mtp_free(mtp);
            return NULL;
        }
        mtp->mapping_page_bytes =
            (mtp->mapping_bytes + (size_t)page - 1u) /
            (size_t)page * (size_t)page;
        if (salt_gpu_shared_buffer_map_readonly(
                &mtp->source_buffer, mtp->mapping,
                mtp->mapping_page_bytes) != 0 ||
            salt_gpu_shared_buffer_alloc(&mtp->scratch_buffer,
                scratch_floats * sizeof(float) + 64u) != 0 ||
            salt_gpu_mtp_q4_0_prepare() != 0) {
            set_error(error, error_size, "Gemma 4 MTP Metal realization failed");
            salt_gemma4_mtp_free(mtp);
            return NULL;
        }
        mtp->scratch = (float *)mtp->scratch_buffer.contents;
        mtp->metal = 1;
    } else {
        mtp->scratch = (float *)calloc(scratch_floats, sizeof(float));
    }
    if (!mtp->scratch) {
        set_error(error, error_size, "Gemma 4 MTP scratch allocation failed");
        salt_gemma4_mtp_free(mtp);
        return NULL;
    }
    {
        float *cursor = mtp->scratch;
#define TAKE(name, count) do { mtp->name = cursor; cursor += (count); } while (0)
        TAKE(target_embedding, MTP_TARGET_HIDDEN);
        TAKE(combined, 2u * MTP_TARGET_HIDDEN);
        TAKE(state_a, MTP_HIDDEN);
        TAKE(state_b, MTP_HIDDEN);
        TAKE(normalized, MTP_HIDDEN);
        TAKE(queries, MTP_HEADS * MTP_FULL_HEAD_DIM);
        TAKE(attention, MTP_HEADS * MTP_FULL_HEAD_DIM);
        TAKE(branch, MTP_HIDDEN);
        TAKE(gate, MTP_INTERMEDIATE);
        TAKE(up, MTP_INTERMEDIATE);
        TAKE(ffn, MTP_HIDDEN);
        TAKE(logits, MTP_VOCAB);
        TAKE(projected, MTP_TARGET_HIDDEN);
#undef TAKE
        if ((size_t)(cursor - mtp->scratch) != scratch_floats) {
            set_error(error, error_size, "Gemma 4 MTP scratch layout mismatch");
            salt_gemma4_mtp_free(mtp);
            return NULL;
        }
    }
    mtp->final_norm = float_pointer(
        mtp, SALT_MTP_Q4_OUTPUT_NORM_WEIGHT_OFFSET, MTP_HIDDEN);
    mtp->embedding_head = weight_pointer(mtp,
        SALT_MTP_Q4_TOKEN_EMBD_WEIGHT_OFFSET,
        SALT_MTP_Q4_TOKEN_EMBD_WEIGHT_BYTES);
    mtp->pre_projection = weight_pointer(mtp,
        SALT_MTP_Q4_NEXTN_PRE_PROJECTION_WEIGHT_OFFSET,
        SALT_MTP_Q4_NEXTN_PRE_PROJECTION_WEIGHT_BYTES);
    mtp->post_projection = weight_pointer(mtp,
        SALT_MTP_Q4_NEXTN_POST_PROJECTION_WEIGHT_OFFSET,
        SALT_MTP_Q4_NEXTN_POST_PROJECTION_WEIGHT_BYTES);
#define BIND_LAYER(index) bind_layer(mtp, index, \
    SALT_MTP_Q4_BLK_##index##_ATTN_NORM_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_LAYER_OUTPUT_SCALE_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_FFN_DOWN_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_FFN_GATE_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_FFN_UP_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_POST_ATTENTION_NORM_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_POST_FFW_NORM_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_FFN_NORM_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_ATTN_OUTPUT_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_ATTN_Q_NORM_WEIGHT_OFFSET, \
    SALT_MTP_Q4_BLK_##index##_ATTN_Q_WEIGHT_OFFSET)
    if (!mtp->embedding_head || !mtp->pre_projection || !mtp->post_projection ||
        !mtp->final_norm ||
        BIND_LAYER(0) != 0 || BIND_LAYER(1) != 0 ||
        BIND_LAYER(2) != 0 || BIND_LAYER(3) != 0) {
#undef BIND_LAYER
        set_error(error, error_size, "Gemma 4 MTP tensor layout mismatch");
        salt_gemma4_mtp_free(mtp);
        return NULL;
    }
#undef BIND_LAYER
    return mtp;
}

void salt_gemma4_mtp_free(SaltGemma4Mtp *mtp) {
    if (!mtp) return;
    if (mtp->scratch_buffer.backend)
        (void)salt_gpu_shared_buffer_free(&mtp->scratch_buffer);
    else
        free(mtp->scratch);
    if (mtp->source_buffer.backend)
        (void)salt_gpu_shared_buffer_free(&mtp->source_buffer);
    if (mtp->mapping) munmap((void *)(uintptr_t)mtp->mapping, mtp->mapping_bytes);
    if (mtp->fd >= 0) close(mtp->fd);
    free(mtp);
}

int salt_gemma4_mtp_draft(
        SaltGemma4Mtp *mtp, int32_t last_committed_token,
        const float *target_hidden, uint32_t target_position,
        int32_t authoritative_root,
        uint32_t maximum_tokens, int32_t *tokens, uint32_t capacity,
        uint32_t *count) {
    const float *hidden = target_hidden;
    int32_t input = last_committed_token;
    int32_t prediction = -1;
    int position;
    if (count) *count = 0;
    if (!mtp || last_committed_token < 0 || last_committed_token >= MTP_VOCAB ||
        !target_hidden || authoritative_root < 0 ||
        authoritative_root >= MTP_VOCAB || target_position > INT_MAX ||
        maximum_tokens < 1u ||
        maximum_tokens > capacity || !tokens || !count)
        return SALT_TFAIL(-1);
    position = (int)target_position;
    mtp->stats.rounds++;
    if (mtp_forward_step(mtp, input, hidden, position,
                         &prediction, mtp->projected) != 0)
        return SALT_TFAIL(-1);
    tokens[0] = authoritative_root;
    *count = 1u;
    if (prediction != authoritative_root) {
        mtp->stats.root_misses++;
        return 0;
    }
    mtp->stats.root_matches++;
    mtp->stats.proposed_tokens++;
    input = authoritative_root;
    hidden = mtp->projected;
    for (uint32_t index = 1u; index < maximum_tokens; index++) {
        if (mtp->target.is_stop(mtp->target.context, input)) break;
        if (mtp_forward_step(mtp, input, hidden, position,
                             &prediction, mtp->projected) != 0)
            return SALT_TFAIL(-1);
        tokens[index] = prediction;
        *count = index + 1u;
        mtp->stats.proposed_tokens++;
        input = prediction;
        hidden = mtp->projected;
    }
    return 0;
}

int salt_gemma4_mtp_stats(const SaltGemma4Mtp *mtp,
                          SaltGemma4MtpStats *stats) {
    if (!mtp || !stats) return SALT_TFAIL(-1);
    *stats = mtp->stats;
    return 0;
}

uint64_t salt_gemma4_mtp_resident_bytes(const SaltGemma4Mtp *mtp) {
    if (!mtp) return 0;
    return (uint64_t)mtp->mapping_bytes +
        (uint64_t)mtp->scratch_floats * sizeof(float) +
        (mtp->metal ? 64u : 0u) + sizeof *mtp;
}

uint64_t salt_gemma4_mtp_startup_bytes(void) {
    const uint64_t scratch_floats = MTP_TARGET_HIDDEN + 2u * MTP_TARGET_HIDDEN +
        2u * MTP_HIDDEN + MTP_HIDDEN +
        2u * (MTP_HEADS * MTP_FULL_HEAD_DIM) + MTP_HIDDEN +
        2u * MTP_INTERMEDIATE + MTP_HIDDEN + MTP_VOCAB + MTP_TARGET_HIDDEN;
    return SALT_MTP_Q4_SOURCE_BYTES +
        scratch_floats * sizeof(float) + 64u + sizeof(SaltGemma4Mtp);
}
