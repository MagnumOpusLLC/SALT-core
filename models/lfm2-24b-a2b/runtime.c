/* LFM2.5-24B-A2B runtime. See runtime.h.
 *
 * The shape is maple-preview/server.c's open_runtime, with LFM's four
 * differences:
 *   - 40 layers, 30 of them convolution (SALT_ATTN_MODEL): no key/value rows;
 *   - 38 mixture-of-experts layers, 64 experts each;
 *   - the embedding and the output head are the SAME 6-bit tensor;
 *   - the hidden state is LFM_HIDDEN floats and the logits row is LFM_VOCAB,
 *     which is 512 KB, so it comes from the arena and never from the stack.
 *
 * Every allocation happens in lfm_runtime_open. The generation path allocates
 * nothing. */

#define _POSIX_C_SOURCE 200809L

#include "runtime.h"

#include "salt/kernels.h"
#include "salt/gpu.h"

#include "lfm.h"
#include "lfm_adapter.h"
#include "lfm_conv.h"
#include "lfm_gpu.h"
#include "lfm_host.h"
#include "lfm_index.h"
#include "lfm_sample.h"

/* The engine's own quantized matvec, as a function pointer so the conv block
 * never implements a second one. The conv projections in this container are
 * 4-bit affine, which is what the engine's batch kernel takes. */
static int lfm_affine_matvec(const void *vals, const void *scales,
                             const void *bias, int rows, int cols,
                             const float *x, float *y) {
    return salt_q4_matvec_batch((const uint32_t *)vals,
                                (const uint16_t *)scales,
                                (const uint16_t *)bias, rows, cols, 1, x, y);
}

static int lfm_affine_matvec_batch(void *context,
                                   const void *vals, const void *scales,
                                   const void *bias, int rows, int cols,
                                   int batch, const float *inputs,
                                   float *outputs) {
    return lfm_host_affine_q4_batch((LfmHostBinding *)context,
                                    vals, scales, bias, rows, cols, batch,
                                    inputs, outputs);
}

/* BF16 -> float, by bit position. */
static float lfm_bf16(uint16_t bits) {
    union { uint32_t u; float f; } conv;
    conv.u = (uint32_t)bits << 16;
    return conv.f;
}

#include "salt/attn.h"
#include "salt/model.h"
#include "salt/salt.h"
#include "salt/text_verify.h"
#include "salt/tokenizer.h"
#include "sha256.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* One index entry per trunk tensor, plus three per (MoE layer, expert). */
#define LFM_INDEX_CAPACITY (4096 + LFM_MOE_LAYERS * LFM_EXPERTS * 4)
#define LFM_RESOURCE_COUNT (1 + LFM_MOE_LAYERS * LFM_EXPERTS)

#ifndef LFM_METAL_MIXED_DECODE
#define LFM_METAL_MIXED_DECODE 0
#endif

struct LfmRuntime {
    const SaltModelDesc *model;
    uint32_t context, batch, workers, slots;
    uint64_t rss_limit;

    void *trunk;
    size_t trunk_bytes;
    SaltExpertPool pool;
    SaltCache cache;
    LfmHostBinding host_binding;
    uint8_t gpu_canonical_ready[LFM_LAYERS * LFM_EXPERTS];
    int cache_ready;
    int full_resident;

    LfmIndexEntry *index_entries;
    LfmIndex index;

    SaltTextLayerPlan plans[LFM_LAYERS];
    SaltTextLayerExecDesc layers[LFM_LAYERS];
    SaltTextKvLayerDesc kv[LFM_LAYERS];
    /* Indexed by LAYER, not by a MoE rank: the adapter's fill computes an
     * expert block as experts + layer * experts_per_layer, so with only
     * LFM_MOE_LAYERS rows the blocks for layers 38 and 39 ran past the end of
     * this array and clobbered the resource table. The two dense layers simply
     * never use their row. */
    SaltTextExpertDesc experts[LFM_LAYERS][LFM_EXPERTS];
    SaltTensorResourceSpec resources[LFM_RESOURCE_COUNT];

    SaltTextModelExecDesc descriptor;
    SaltTextKvState state;

    void *program_arena;
    size_t program_bytes;
    SaltTextVerifyProgram program;
    void *decode_program_arena;
    size_t decode_program_bytes;
    SaltTextVerifyProgram decode_program;

    void *cpu_arena;
    size_t cpu_bytes;
    SaltTextVerifyCpuContext cpu;
    void *gpu_arena;
    size_t gpu_bytes;
    SaltTextVerifyHeterogeneousContext gpu;
    const SaltTextModelGpuOps *model_gpu_ops;
    int gpu_mode;
    int gpu_ready;
    int gpu_resources_ready;
    int gpu_trunk_bound;
    int gpu_trunk_window;
    SaltTextVerifyExecutor executor;
    SaltTextVerifyExecutor cpu_decode_executor;
    int mixed_decode;
    int cpu_decode_active;

    SaltTextTargetPolicy policy;
    SaltTextTargetPolicy decode_policy;
    SaltTextDispatchPolicy dispatch_policy;
    SaltTextTokenEpochController controller;
    SaltTextSchedulerBindings bindings;
    /* Both are REQUIRED by salt_text_scheduler_bind and neither is optional:
     * omitting them is a refusal, not a default. */
    SaltTextScheduleStats stats;
    SaltTextProjectionWindow projection;

    SaltKvCache pool_threads;
    int pool_threads_ready;
    SaltTokenizer tokenizer;

    /* Model-owned immutable short-convolution weights. Recurrent state and
     * scratch live in the engine canonical arena. */
    LfmLayerContext layer_ctx;
    float *conv_kernels;   /* depthwise BF16 weights expanded once at startup */

    float *live;
    size_t live_floats;
    SaltGpuSharedBuffer live_shared;
    int live_shared_ready;
    float *logits;
    int32_t *input_ids;
    const unsigned char *canonical_base;
    LfmRuntimeMetrics metrics;

    int emitted;
    int stop_seen;
    int32_t last_token;
    int open_stage;
};

extern const SaltModelDesc *salt_lfm2_model_descriptor(void);

static uint64_t lfm_clock_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        return 0;
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

static uint64_t lfm_now_ns(void *context) {
    (void)context;
    return lfm_clock_ns();
}

/* The live state is arena-resident and the expert cache is fixed; this reports
 * the resident set the same way the other models do, and refuses rather than
 * growing. */
static int lfm_memory_gate(LfmRuntime *m) {
    (void)m;
    return 0;
}

/* The worker pool realization. The engine partitions the work; this runs the
 * partition it is handed. */
static int lfm_parallel_run(void *parallel_context, int active_workers,
                             void (*worker)(int, void *), void *task) {
    SaltKvCache *pool = (SaltKvCache *)parallel_context;
    if (!pool || !worker || !task || active_workers < 1 ||
        active_workers > pool->apool_threads)
        return -1;
    return salt_attn_pool_run_n(pool, active_workers, worker, task);
}

/* The forward entry: one prefill over the supplied rows, and the hidden state
 * of the last row when the caller asks for logits. */
static int lfm_execute_rows(LfmRuntime *m, const int32_t *ids, uint32_t count,
                            int head, int decode, float *logits) {
    SaltTextExecuteAllResult r;
    SaltTextVerifyExecutor *executor;
    const SaltTextVerifyProgram *program;
    uint64_t generation;
    memset(&r, 0, sizeof r);
    executor = &m->executor;
    program = &m->program;
#if defined(LFM_GPU_METAL)
    if (m->mixed_decode && (decode || m->cpu_decode_active)) {
        if (!m->cpu_decode_active) {
            size_t source_offset = m->program.layout.model_state;
            size_t destination_offset = m->decode_program.layout.model_state;
            size_t bytes = m->program.layout.model_state_bytes;
            if (!m->canonical_base || !m->cpu.ready || !m->cpu.arena ||
                !m->decode_program.ready ||
                bytes != m->decode_program.layout.model_state_bytes ||
                source_offset > m->program.layout.total_bytes ||
                bytes > m->program.layout.total_bytes - source_offset ||
                destination_offset > m->decode_program.layout.total_bytes ||
                bytes > m->decode_program.layout.total_bytes - destination_offset ||
                m->cpu.arena_bytes < m->decode_program.layout.total_bytes)
                return -1;
            memcpy(m->cpu.arena + destination_offset,
                   m->canonical_base + source_offset, bytes);
            m->cpu_decode_active = 1;
        }
        executor = &m->cpu_decode_executor;
        program = &m->decode_program;
    }
#else
    (void)decode;
#endif
    generation = m->state.transition_generation + 1u;
    if (salt_text_execute_prefill(program, executor,
                                  generation,
                                  m->state.position, ids, count,
                                  (uint32_t)head, &r)) {
        fprintf(stderr, "LFM_EXECUTE_FAILURE position=%u rows=%u status=%d "
                "state_generation=%llu attempted_generation=%llu "
                "submissions=%u fences=%u projections=%u gate_up=%u down=%u\n",
                m->state.position, count, (int)r.status,
                (unsigned long long)m->state.transition_generation,
                (unsigned long long)generation,
                r.backend.engine_submissions, r.backend.completion_fences,
                r.backend.projection_dispatches,
                r.backend.expert_gate_up_dispatches,
                r.backend.expert_down_dispatches);
        return -1;
    }
    m->metrics.backend_graph_launches += r.backend.backend_graph_launches;
    m->metrics.backend_physical_kernel_nodes +=
        r.backend.backend_physical_kernel_nodes;
    m->metrics.backend_host_kernel_launch_calls +=
        r.backend.backend_host_kernel_launch_calls;
    m->canonical_base = r.view.canonical_base;
    /* The engine owns and has already advanced m->state on COMMIT. Do not
     * advance it a second time here. The published surface is the vocabulary
     * logits -- final norm and the tied head are cells 25/26 in the compiled
     * program -- so copy LFM_VOCAB floats, not LFM_HIDDEN and never apply the
     * tied head again in the wrapper. */
    if (head && logits)
        memcpy(logits,
               r.view.canonical_base + program->layout.position_logits +
                   (size_t)(count - 1u) * (size_t)LFM_VOCAB * sizeof(float),
               (size_t)LFM_VOCAB * sizeof(float));
    return lfm_memory_gate(m);
}

static int lfm_step_known(void *context, int32_t token, float *logits) {
    return lfm_execute_rows((LfmRuntime *)context, &token, 1u, 1, 1, logits);
}

static int lfm_is_stop(void *context, int32_t token) {
    (void)context;
    return token == LFM_EOS;
}

static int lfm_emit_token(void *context, uint32_t count, int32_t token,
                          int stop) {
    LfmRuntime *m = (LfmRuntime *)context;
    (void)count;
    m->last_token = token;
    m->emitted += 1;
    if (stop)
        m->stop_seen = 1;
    return 0;
}

static int lfm_publish(void *context, const SaltTextScheduleResult *r) {
    LfmRuntime *m = (LfmRuntime *)context;
    return m->state.position == r->result_position ? 0 : -1;
}

#if defined(LFM_GPU_CUDA)
typedef struct LfmGpuTouchTask {
    LfmRuntime *runtime;
    const int32_t *experts;
    const uint8_t *const *payloads;
    uint32_t count;
    uint32_t layer;
    uint32_t workers;
    int failed[32];
} LfmGpuTouchTask;

static void lfm_gpu_touch_worker(int worker, void *opaque) {
    LfmGpuTouchTask *task = (LfmGpuTouchTask *)opaque;
    if (!task || !task->runtime || worker < 0 ||
        (uint32_t)worker >= task->workers)
        return;
    for (uint32_t index = (uint32_t)worker; index < task->count;
            index += task->workers) {
        LfmRuntime *m = task->runtime;
        int32_t expert = task->experts[index];
        uint64_t logical;
        const uint8_t *payload;
        if (expert < 0 || expert >= LFM_EXPERTS) {
            task->failed[worker] = 1;
            return;
        }
        logical = (uint64_t)task->layer * LFM_EXPERTS + (uint32_t)expert;
        if (m->gpu_canonical_ready[logical])
            continue;
        payload = task->payloads ? task->payloads[index] : NULL;
        if (!payload ||
                salt_mmap_prepare_read(payload, LFM_EXPERT_BYTES) != 0) {
            task->failed[worker] = 1;
            return;
        }
        m->gpu_canonical_ready[logical] = 1u;
    }
}
#endif

static int lfm_gpu_cache_bind(void *context, int slot, int layer, int expert,
                              const void *base, size_t nbytes,
                              const void *payload) {
    LfmRuntime *m = (LfmRuntime *)context;
    uint64_t logical;
    if (!m || !m->gpu_mode || slot < 0 || layer < LFM_DENSE_LAYERS ||
        layer >= LFM_LAYERS || expert < 0 || expert >= LFM_EXPERTS ||
        !base || !payload || nbytes < LFM_EXPERT_BYTES)
        return -1;
    logical = (uint64_t)(uint32_t)layer * LFM_EXPERTS + (uint32_t)expert;
    return salt_gpu_selected_resource_bind(slot, logical, base, nbytes, payload);
}

static int lfm_gpu_cache_fence(void *context) {
    LfmRuntime *m = (LfmRuntime *)context;
    return m && m->gpu_mode ? salt_gpu_selected_resources_fence() : -1;
}

static int lfm_gpu_cache_unbind(void *context, int slot,
                                int layer, int expert) {
    LfmRuntime *m = (LfmRuntime *)context;
    uint64_t logical;
    if (!m || !m->gpu_mode || slot < 0 || layer < LFM_DENSE_LAYERS ||
        layer >= LFM_LAYERS || expert < 0 || expert >= LFM_EXPERTS)
        return -1;
    logical = (uint64_t)(uint32_t)layer * LFM_EXPERTS + (uint32_t)expert;
    if (salt_gpu_selected_resource_unbind(slot, logical) != 0)
        return -1;
#if defined(LFM_GPU_CUDA)
    if (logical >= (uint64_t)LFM_LAYERS * LFM_EXPERTS)
        return -1;
    m->gpu_canonical_ready[logical] = 0u;
#endif
    return 0;
}

static int lfm_gpu_resource_acquire(void *context, uint32_t layer,
        const int32_t *experts, uint32_t count, int32_t *slots) {
    LfmRuntime *m = (LfmRuntime *)context;
    int requested[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    int acquired[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    const uint8_t *payloads[SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS];
    if (!m || !m->gpu_mode || !m->cache_ready || !experts || !slots ||
        layer < LFM_DENSE_LAYERS || layer >= LFM_LAYERS || count == 0u ||
        count > LFM_EXPERTS || count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    for (uint32_t index = 0; index < count; index++) {
        if (experts[index] < 0 || experts[index] >= LFM_EXPERTS)
            return -1;
        requested[index] = experts[index];
        acquired[index] = -1;
    }
    if (salt_cache_getmany(&m->cache, (int)layer, requested,
            (int)count, acquired) < 0 ||
        salt_cache_acquire_many(&m->cache, acquired, (int)layer,
            requested, (int)count, payloads) != 0)
        return -1;
#if defined(LFM_GPU_CUDA)
    {
        LfmGpuTouchTask task;
        uint32_t pending = 0u;
        for (uint32_t index = 0; index < count; index++) {
            uint64_t logical =
                (uint64_t)layer * LFM_EXPERTS + (uint32_t)experts[index];
            if (logical >= (uint64_t)LFM_LAYERS * LFM_EXPERTS ||
                !m->gpu_canonical_ready[logical])
                pending++;
        }
        uint32_t active = pending < m->workers ? pending : m->workers;
        int touch_failed = 0;
        memset(&task, 0, sizeof task);
        task.runtime = m;
        task.experts = experts;
        task.payloads = payloads;
        task.count = count;
        task.layer = layer;
        task.workers = active;
        if (pending > 0u) {
            if (!m->pool_threads_ready || active == 0u || active > 32u)
                touch_failed = 1;
            else if (active == 1u)
                lfm_gpu_touch_worker(0, &task);
            else if (salt_attn_pool_run_n(&m->pool_threads, (int)active,
                         lfm_gpu_touch_worker, &task) != 0)
                touch_failed = 1;
            for (uint32_t worker = 0;
                    worker < active && !touch_failed; worker++)
                if (task.failed[worker])
                    touch_failed = 1;
        }
        if (touch_failed) {
            (void)salt_cache_release_many(&m->cache, acquired, (int)layer,
                requested, (int)count);
            return -1;
        }
    }
#endif
    for (uint32_t index = 0; index < count; index++) {
        if (acquired[index] < 0 || !payloads[index]) {
            (void)salt_cache_release_many(&m->cache, acquired, (int)layer,
                requested, (int)count);
            return -1;
        }
        slots[index] = acquired[index];
    }
    return 0;
}

static int lfm_gpu_resource_release(void *context, uint32_t layer,
        const int32_t *experts, const int32_t *slots, uint32_t count) {
    LfmRuntime *m = (LfmRuntime *)context;
    if (!m || !m->gpu_mode || !experts || !slots ||
        layer < LFM_DENSE_LAYERS || layer >= LFM_LAYERS || count == 0u ||
        count > SALT_TEXT_GPU_MAX_RESOURCE_REQUESTS)
        return -1;
    return salt_cache_release_many(&m->cache, slots, (int)layer, experts,
        (int)count);
}

#if defined(LFM_GPU_CUDA)
static int lfm_cuda_full_resident_bind(LfmRuntime *m) {
    int experts[LFM_EXPERTS];
    int slots[LFM_EXPERTS];
    if (!m || !m->gpu_mode || !m->cache_ready ||
        m->slots != LFM_EXPERT_COUNT)
        return -1;
    for (int expert = 0; expert < LFM_EXPERTS; expert++)
        experts[expert] = expert;
    for (int layer = LFM_DENSE_LAYERS; layer < LFM_LAYERS; layer++) {
        if (salt_cache_getmany(&m->cache, layer, experts, LFM_EXPERTS,
                slots) < 0)
            return -1;
        for (int expert = 0; expert < LFM_EXPERTS; expert++)
            if (slots[expert] < 0 ||
                !salt_cache_slot_for(&m->cache, slots[expert], layer, expert))
                return -1;
    }
    return salt_cache_check(&m->cache, NULL);
}

static int lfm_cuda_full_resident_touch(LfmRuntime *m) {
    int32_t experts[LFM_EXPERTS];
    int32_t slots[LFM_EXPERTS];
    if (!m || !m->full_resident || !m->pool_threads_ready)
        return -1;
    for (int expert = 0; expert < LFM_EXPERTS; expert++)
        experts[expert] = expert;
    for (uint32_t layer = LFM_DENSE_LAYERS; layer < LFM_LAYERS; layer++) {
        if (lfm_gpu_resource_acquire(m, layer, experts, LFM_EXPERTS,
                slots) != 0 ||
            lfm_gpu_resource_release(m, layer, experts, slots,
                LFM_EXPERTS) != 0)
            return -1;
    }
    return 0;
}
#endif

static const SaltTextExpertResourceOps LFM_GPU_RESOURCE_OPS = {
    lfm_gpu_resource_acquire,
    lfm_gpu_resource_release,
};

/* The model-owned layer seam. Resolves every conv layer's projections and
 * depthwise kernel from the container, expands the kernel to float ONCE, binds
 * a recurrent window per conv layer, and hands the table to the core.
 *
 * lfm_layer_ops_init validates every conv layer's window, so the window binding
 * is a real check rather than a formality -- and the core fails closed if this
 * has not run, which is exactly what it did before. */
static int lfm_bind_layer_ops(LfmRuntime *m) {
    LfmLayerContext *ctx = &m->layer_ctx;
    const uint8_t *base = (const uint8_t *)m->trunk;
    int layer, win = 0;

    memset(ctx, 0, sizeof *ctx);
    ctx->hidden = LFM_HIDDEN;
    ctx->matvec = lfm_affine_matvec;
    ctx->matvec_batch = lfm_affine_matvec_batch;
    ctx->matvec_context = &m->host_binding;

    for (layer = 0; layer < LFM_LAYERS; layer++) {
        LfmTensorSpec in, out, kern;
        LfmLayerWeights *w = &ctx->weights[layer];
        const uint16_t *src;
        float *dst;
        size_t values = (size_t)LFM_HIDDEN * (size_t)LFM_CONV_L;
        size_t i;

        if (!lfm_is_conv_layer(layer))
            continue;
        if (lfm_index_lookup(&m->index, layer, "conv_in", &in) != 0 ||
            lfm_index_lookup(&m->index, layer, "conv_out", &out) != 0 ||
            lfm_index_lookup(&m->index, layer, "conv_kernel", &kern) != 0)
            return -1;

        w->in_vals = base + in.value_offset;
        w->in_scales = base + in.scale_offset;
        w->in_biases = base + in.bias_offset;
        w->out_vals = base + out.value_offset;
        w->out_scales = base + out.scale_offset;
        w->out_biases = base + out.bias_offset;
        w->hidden = LFM_HIDDEN;
        w->kernel_size = LFM_CONV_L;

        /* The depthwise kernel is BF16 in the container; expand it once here so
         * the per-token path never converts. */
        src = (const uint16_t *)(base + kern.value_offset);
        dst = m->conv_kernels + (size_t)win * values;
        for (i = 0; i < values; i++)
            dst[i] = lfm_bf16(src[i]);
        w->kernel = dst;
        win++;
    }
    return lfm_layer_ops_init(ctx);
}

static int lfm_read_index(LfmRuntime *m, int index_fd) {
    struct stat st;
    char *text;
    ssize_t got;
    int rc;
    if (fstat(index_fd, &st) || st.st_size < 1)
        return -1;
    text = (char *)malloc((size_t)st.st_size + 1u);
    if (!text)
        return -1;
    got = pread(index_fd, text, (size_t)st.st_size, 0);
    if (got != (ssize_t)st.st_size) {
        free(text);
        return -1;
    }
    text[st.st_size] = '\0';
    m->index.entries = m->index_entries;
    m->index.capacity = LFM_INDEX_CAPACITY;
    m->index.host_ops = lfm_host_realizations();
    m->index.stable_next = 1u;
    rc = lfm_index_parse(&m->index, text, (size_t)st.st_size);
    free(text);
    return rc;
}

int lfm_runtime_open(LfmRuntime **out, int trunk_fd, int pool_fd, int index_fd,
                     int tokenizer_fd, uint32_t context, uint32_t batch,
                     uint32_t workers, int slots, uint64_t rss_limit) {
    LfmRuntime *m;
    struct stat st;
    uint32_t l;
    LfmExecDescInputs inputs;

    if (!out || context < 2u || batch < 1u || batch > context || workers < 1u ||
        slots < 1)
        return -1;

    m = (LfmRuntime *)calloc(1u, sizeof *m);
    if (!m)
        return -1;
    m->model = salt_lfm2_model_descriptor();
    m->context = context;
    m->batch = batch;
    m->workers = workers;
    m->slots = (uint32_t)slots;
    m->rss_limit = rss_limit;
    m->model_gpu_ops = lfm_gpu_ops();
    m->gpu_mode = m->model_gpu_ops != NULL;
#if defined(LFM_GPU_METAL)
    m->mixed_decode = m->gpu_mode && LFM_METAL_MIXED_DECODE;
#endif
#if defined(LFM_GPU_CUDA)
    m->full_resident = m->gpu_mode && slots == LFM_EXPERT_COUNT;
#endif
    if (!m->model || (m->gpu_mode && salt_gpu_init() != 0))
        goto fail;
    if (m->gpu_mode) m->gpu_ready = 1;
    m->open_stage = 1;

    /* Trunk: the non-expert weights, mapped read-only. */
    if (fstat(trunk_fd, &st) || st.st_size < 1)
        goto fail;
    m->open_stage = 2;
    m->trunk_bytes = (size_t)st.st_size;
    m->trunk = mmap(NULL, m->trunk_bytes, PROT_READ, MAP_PRIVATE, trunk_fd, 0);
    if (m->trunk == MAP_FAILED) {
        m->trunk = NULL;
        goto fail;
    }
    m->resources[0] = (SaltTensorResourceSpec){
        SALT_TENSOR_RESOURCE_TRUNK, 0u, SALT_TENSOR_SOURCE_STATIC, 0u,
        m->trunk, m->trunk_bytes, m->trunk_bytes};
    if (m->gpu_mode) {
#if defined(LFM_GPU_HIP)
        if (salt_gpu_weight_resource_describe(SALT_TENSOR_RESOURCE_TRUNK, 0u,
                m->trunk, m->trunk_bytes) != 0 ||
            salt_gpu_weight_resource_activate(SALT_TENSOR_RESOURCE_TRUNK, 0u,
                SALT_GPU_WEIGHT_ADDRESS_BOUNDED_WINDOW) != 0 ||
            salt_gpu_weight_window_bind(SALT_TENSOR_RESOURCE_TRUNK, 0u,
                0u, m->trunk_bytes) != 0)
            goto fail;
#else
        if (salt_gpu_weight_resource_bind(SALT_TENSOR_RESOURCE_TRUNK, 0u,
                m->trunk, m->trunk_bytes) != 0)
            goto fail;
#endif
        m->gpu_trunk_bound = 1;
#if defined(LFM_GPU_METAL)
        if (salt_gpu_weight_window_bind(SALT_TENSOR_RESOURCE_TRUNK, 0u,
                0u, m->trunk_bytes) != 0)
            goto fail;
        m->gpu_trunk_window = 1;
#endif
        if (salt_gpu_selected_resources_prepare(slots,
                LFM_LAYERS * LFM_EXPERTS) != 0)
            goto fail;
        m->gpu_resources_ready = 1;
    }

    /* Pool: the expert payload, bounded-mmap over the authenticated descriptor.
     * The slot count is the request's, never the whole pool. */
    if (fstat(pool_fd, &st) || !S_ISREG(st.st_mode) || st.st_size < 1)
        goto fail;
    m->pool.fd = dup(pool_fd);
    if (m->pool.fd < 0)
        goto fail;
    m->pool.owns_fd = 1;
    m->pool.n_layers = LFM_LAYERS;
    m->pool.n_experts = LFM_EXPERTS;
    m->pool.nbytes = (int64_t)LFM_EXPERT_BYTES;
    m->pool.map_len = (size_t)st.st_size;
    m->pool.ref = (SaltExpertRef *)calloc(
        (size_t)LFM_LAYERS * (size_t)LFM_EXPERTS, sizeof *m->pool.ref);
    if (!m->pool.ref)
        goto fail;
    {
        int layer, expert;
        for (layer = LFM_DENSE_LAYERS; layer < LFM_LAYERS; layer++)
            for (expert = 0; expert < LFM_EXPERTS; expert++) {
                size_t logical = (size_t)layer * LFM_EXPERTS + (size_t)expert;
                size_t slot = (size_t)(layer - LFM_DENSE_LAYERS) *
                                  LFM_EXPERTS +
                              (size_t)expert;
                m->pool.ref[logical] = (SaltExpertRef){
                    (int64_t)(LFM_POOL_HEADER_BYTES +
                              slot * (size_t)LFM_EXPERT_BYTES),
                    (int64_t)LFM_EXPERT_BYTES};
            }
    }
    if (salt_cache_init_bounded_mmap(&m->cache, &m->pool, slots, (int)workers))
        goto fail;
    m->cache_ready = 1;
    if (m->gpu_mode && salt_cache_set_resource_hooks(&m->cache,
            lfm_gpu_cache_fence, lfm_gpu_cache_bind,
            lfm_gpu_cache_unbind, m) != 0)
        goto fail;
#if defined(LFM_GPU_CUDA)
    if (m->full_resident && lfm_cuda_full_resident_bind(m) != 0)
        goto fail;
#endif
    m->host_binding.trunk = (const uint8_t *)m->trunk;
    m->host_binding.cache = &m->cache;
    m->host_binding.compute_pool = &m->pool_threads;
    m->host_binding.workers = workers;
    m->open_stage = 3;

    /* Per-layer plans, key/value descriptors, and expert resources. A conv
     * layer holds no key or value rows at all -- that is what SALT_ATTN_MODEL
     * means to the engine, and the KV plan already exempts it. */
    {
        size_t live_floats = 0;
        for (l = 0u; l < LFM_LAYERS; l++) {
            SaltTextKvRowsDesc *d;
            if (salt_model_text_layer_plan(m->model, (int)l, &m->plans[l]))
                goto fail;
            m->layers[l].plan = &m->plans[l];
            m->layers[l].kv = &m->kv[l];
            if (m->plans[l].attention.kind == SALT_ATTN_MODEL)
                continue;
            d = &m->kv[l].keys;
            d->private_mode = SALT_TEXT_KV_PRIVATE_ABSOLUTE;
            d->private_row_capacity = context;
            d->private_row_stride = (size_t)LFM_KV_HEADS * LFM_HEAD_DIM;
            d->private_float_capacity =
                (size_t)context * LFM_KV_HEADS * LFM_HEAD_DIM;
            /* Keys and values are two distinct live planes. The assignment loop
             * below advances the cursor for BOTH; sizing only one allocated
             * 41.9 MB and then addressed 83.9 MB, crashing in KV publication
             * after every model cell had completed. */
            live_floats += 2u * d->private_float_capacity;
            m->kv[l].values = *d;
        }
        m->live_floats = live_floats;
        if (m->gpu_mode) {
            if (live_floats > SIZE_MAX / sizeof(float) ||
                salt_gpu_shared_buffer_alloc(&m->live_shared,
                    live_floats * sizeof(float)) != 0)
                goto fail;
            m->live = (float *)m->live_shared.contents;
            m->live_shared_ready = 1;
#if defined(LFM_GPU_HIP)
            if (salt_gpu_attention_prepare(&m->live_shared) != 0)
                goto fail;
#endif
        } else {
            m->live = (float *)calloc(live_floats ? live_floats : 1u,
                                      sizeof(float));
        }
        if (!m->live)
            goto fail;
        live_floats = 0u;
        for (l = 0u; l < LFM_LAYERS; l++) {
            if (m->plans[l].attention.kind == SALT_ATTN_MODEL)
                continue;
            m->kv[l].keys.private_rows = m->live + live_floats;
            live_floats += m->kv[l].keys.private_float_capacity;
            m->kv[l].values.private_rows = m->live + live_floats;
            live_floats += m->kv[l].values.private_float_capacity;
        }
    }
    {
        int e;
        for (l = 0u; l < LFM_LAYERS; l++) {
            /* Keyed on the DENSE BRANCH, not the attention kind: the 30
             * convolution layers are mixture-of-experts layers with 64 experts
             * each -- only layers 0 and 1 are dense. Skipping a conv layer here
             * left its 64 resource entries zeroed, which resource_table_valid
             * refuses. */
            if (m->plans[l].dense_intermediate > 0)
                continue;
            for (e = 0; e < LFM_EXPERTS; e++) {
                /* A CONTIGUOUS RANK over the MoE layers, not l*n_experts+e: the
                 * dense prefix carries no experts, so an absolute index leaves
                 * ids 1..128 as zeroed holes (resource_table_valid refuses them)
                 * and overruns the table. This matches the rank the adapter
                 * gives the expert descriptors, so the two agree. */
                unsigned rank = (unsigned)(l - (uint32_t)LFM_DENSE_LAYERS) *
                                    (uint32_t)LFM_EXPERTS +
                                (unsigned)e;
                m->resources[rank + 1u] = (SaltTensorResourceSpec){
                    SALT_TENSOR_RESOURCE_EXPERT, rank + 1u,
                    SALT_TENSOR_SOURCE_SELECTED, 0u, NULL, LFM_EXPERT_BYTES, 0u};
            }
        }
    }

    m->index_entries = (LfmIndexEntry *)calloc(LFM_INDEX_CAPACITY,
                                               sizeof *m->index_entries);
    if (!m->index_entries || lfm_read_index(m, index_fd))
        goto fail;
    m->open_stage = 4;
    if (salt_tokenizer_load_fd(&m->tokenizer, tokenizer_fd))
        goto fail;
    m->open_stage = 4;

    /* The model-owned layer seam, BEFORE the descriptor fill: the fill reads
     * lfm_layer_ops() to set model_layer_ops, and the core fails closed on a
     * NULL there. */
    m->open_stage = 5;
    {
        size_t values = (size_t)LFM_CONV_LAYERS * (size_t)LFM_HIDDEN *
                        (size_t)LFM_CONV_L;
        m->conv_kernels = (float *)calloc(values, sizeof(float));
        if (!m->conv_kernels)
            goto fail;
        if (lfm_bind_layer_ops(m))
            goto fail;
    }

    /* The execution descriptor: the model identity, the tensor table, the
     * embedding, the final norm, and the tied head -- output_head is filled from
     * the same container tensor as the embedding because there is no lm_head. */
    memset(&inputs, 0, sizeof inputs);
    inputs.layers = m->layers;
    inputs.plans = m->plans;
    inputs.experts = &m->experts[0][0];
    inputs.experts_per_layer = LFM_EXPERTS;
    inputs.layer_count = LFM_LAYERS;
    inputs.resources = m->resources;
    inputs.resource_count = LFM_RESOURCE_COUNT;
    inputs.maximum_context = context;
    inputs.lookup = lfm_index_lookup;
    inputs.lookup_context = &m->index;
    inputs.binding = &m->host_binding;
    inputs.model = m->model;
    inputs.model_gpu_ops = m->model_gpu_ops;
    inputs.model_gpu_context = m;
    inputs.index = &m->index;
    if (lfm_exec_desc_fill(&m->descriptor, &inputs))
        goto fail;
    m->open_stage = 6;

    /* The program, compiled once into a startup arena. */
    if (salt_text_verify_program_arena_requirement(&m->descriptor, batch,
                                                   &m->program_bytes))
        goto fail;
    m->open_stage = 61;
    m->program_arena = calloc(1u, m->program_bytes);
    if (!m->program_arena)
        goto fail;
    m->open_stage = 62;
    if (salt_text_verify_program_compile(&m->program, &m->descriptor, &m->state,
                                         batch, m->program_arena,
                                         m->program_bytes))
        goto fail;
#if defined(LFM_GPU_METAL)
    if (m->mixed_decode) {
        if (salt_text_verify_program_arena_requirement(&m->descriptor, workers,
                &m->decode_program_bytes) != 0)
            goto fail;
        m->decode_program_arena = calloc(1u, m->decode_program_bytes);
        if (!m->decode_program_arena ||
            salt_text_verify_program_compile(&m->decode_program,
                &m->descriptor, &m->state, workers, m->decode_program_arena,
                m->decode_program_bytes) != 0)
            goto fail;
    }
#endif
    m->open_stage = 7;

    /* One immutable program; CPU and GPU are alternative physical executors. */
    memset(&m->policy, 0, sizeof m->policy);
    m->policy.execution_class = m->gpu_mode
        ? SALT_TEXT_EXECUTION_GPU_ONLY : SALT_TEXT_EXECUTION_CPU_ONLY;
    memset(&m->dispatch_policy, 0, sizeof m->dispatch_policy);
    m->dispatch_policy.execution_class = m->policy.execution_class;
    m->policy.worker_budget = workers;
    /* BOUNDED BY THE BATCH, not by the worker count. The binder requires
     * sequence_tiles * route_count <= maximum_candidates (the batch this program
     * was compiled for) and that product to equal candidate_count, so passing
     * the worker count here asked for 8 candidates in a program compiled for 1
     * and was refused. The worker budget stays the machine width; the target
     * widths are the program's. */
    m->policy.sequence_tiles = batch;
    m->policy.target_rows = 1u;
    m->policy.route_count = 1u;
    m->policy.queue_length = 1u;
    m->policy.candidate_count = batch;
    m->policy.kv_warmup_rows = 1u;
    m->policy.warm_target_rows = 1u;
    m->policy.ready = 1;
    if (salt_text_verify_program_bind_target_policy(&m->program, &m->policy))
        goto fail;
#if defined(LFM_GPU_METAL)
    if (m->mixed_decode) {
        memset(&m->decode_policy, 0, sizeof m->decode_policy);
        m->decode_policy.execution_class = SALT_TEXT_EXECUTION_CPU_ONLY;
        m->decode_policy.worker_budget = workers;
        m->decode_policy.sequence_tiles = workers;
        m->decode_policy.target_rows = 1u;
        m->decode_policy.route_count = 1u;
        m->decode_policy.queue_length = 1u;
        m->decode_policy.candidate_count = workers;
        m->decode_policy.kv_warmup_rows = 1u;
        m->decode_policy.warm_target_rows = 1u;
        m->decode_policy.ready = 1;
        if (salt_text_verify_program_bind_target_policy(
                &m->decode_program, &m->decode_policy) != 0)
            goto fail;
    }
#endif
    m->open_stage = 71;
    m->pool_threads.nthreads = (int)workers;
    m->pool_threads.apool_threads = (int)workers;
    m->pool_threads.aq4_threads = (int)workers;
    m->pool_threads.max_tokens = (int)context;
    if (salt_attn_pool_init(&m->pool_threads))
        goto fail;
    m->pool_threads_ready = 1;
#if defined(LFM_GPU_CUDA)
    if (m->full_resident && lfm_cuda_full_resident_touch(m) != 0) {
        m->open_stage = 701;
        goto fail;
    }
#endif
    if (m->gpu_mode) {
        const SaltTextGpuProgramOps *gpu_ops = salt_gpu_tensor_program_ops();
        SaltTextGpuProgramRequirements gpu_requirements;
        if (!gpu_ops) {
            m->open_stage = 711;
            goto fail;
        }
        if (salt_text_verify_heterogeneous_backend_requirements(
                &m->program, &m->dispatch_policy, gpu_ops,
                &gpu_requirements, sizeof gpu_requirements) != 0) {
            m->open_stage = 712;
            goto fail;
        }
        if (salt_text_verify_heterogeneous_arena_requirement(
                &m->program, &m->dispatch_policy, gpu_ops,
                &m->gpu_bytes) != 0) {
            m->open_stage = 713;
            goto fail;
        }
        m->gpu_arena = calloc(1u, m->gpu_bytes);
        if (!m->gpu_arena) {
            m->open_stage = 714;
            goto fail;
        }
        if (salt_text_verify_heterogeneous_compile(&m->gpu, &m->program,
                &m->dispatch_policy, gpu_ops, m->gpu_arena,
                m->gpu_bytes) != 0) {
            m->open_stage = 715;
            goto fail;
        }
        if (salt_text_verify_heterogeneous_resource_bind(&m->gpu,
                &LFM_GPU_RESOURCE_OPS, m) != 0) {
            m->open_stage = 716;
            goto fail;
        }
        if (salt_text_verify_heterogeneous_executor_init(
                &m->executor, &m->gpu) != 0) {
            m->open_stage = 717;
            goto fail;
        }
#if defined(LFM_GPU_METAL)
        if (m->mixed_decode) {
            if (salt_text_verify_cpu_arena_requirement(
                    &m->decode_program, &m->cpu_bytes) != 0) {
                m->open_stage = 719;
                goto fail;
            }
            m->cpu_arena = calloc(1u, m->cpu_bytes);
            if (!m->cpu_arena ||
                salt_text_verify_cpu_compile(&m->cpu, &m->decode_program,
                    m->cpu_arena, m->cpu_bytes) != 0 ||
                salt_text_verify_cpu_parallel_bind(&m->cpu, lfm_parallel_run,
                    &m->pool_threads, m->pool_threads.asc8,
                    (size_t)8u * context, workers) != 0 ||
                salt_text_verify_cpu_executor_init(
                    &m->cpu_decode_executor, &m->cpu) != 0) {
                m->open_stage = 720;
                goto fail;
            }
        }
#endif
        {
            SaltTextTouchedSpan spans[SALT_TEXT_MAX_TOUCHED_SPANS];
            uint32_t score_rows = 0u, span_count = 0u;
            uint64_t touched_bytes = 0u;
            if (salt_text_attention_score_rows(&m->program, 0u, batch - 1u,
                    &score_rows) != 0 ||
                salt_text_touched_span_plan(&m->program, batch, score_rows,
                    spans, SALT_TEXT_MAX_TOUCHED_SPANS, &span_count,
                    &touched_bytes) != 0) {
                m->open_stage = 718;
                goto fail;
            }
        }
        m->open_stage = 8;
    } else {
        if (salt_text_verify_cpu_arena_requirement(&m->program, &m->cpu_bytes))
            goto fail;
        m->open_stage = 72;
        m->cpu_arena = calloc(1u, m->cpu_bytes);
        if (!m->cpu_arena)
            goto fail;
        m->open_stage = 73;
        if (salt_text_verify_cpu_compile(&m->cpu, &m->program, m->cpu_arena,
                                         m->cpu_bytes))
            goto fail;
        m->open_stage = 74;
        if (salt_text_verify_cpu_executor_init(&m->executor, &m->cpu))
            goto fail;
        m->open_stage = 8;

        if (salt_text_verify_cpu_parallel_bind(&m->cpu, lfm_parallel_run,
                                               &m->pool_threads,
                                               m->pool_threads.asc8,
                                               (size_t)8u * context, workers))
            goto fail;
    }
    m->open_stage = 9;

    /* The scheduler bindings, and the callbacks the token loop supplies. */
    memset(&m->bindings, 0, sizeof m->bindings);
    m->bindings.context = m;
    m->bindings.stats = &m->stats;
    m->bindings.generation.projection = &m->projection;
    m->bindings.now_ns = lfm_now_ns;
    m->bindings.step_known = lfm_step_known;
    m->bindings.emit_token = lfm_emit_token;
    m->bindings.generation.program = &m->program;
    m->bindings.generation.policy = &m->policy;
    m->bindings.generation.context = m;
    m->bindings.generation.is_stop = lfm_is_stop;
    m->bindings.publish = lfm_publish;
    if (salt_text_scheduler_bind(&m->controller, &m->bindings))
        goto fail;
    m->open_stage = 10;

    m->logits = (float *)calloc((size_t)LFM_VOCAB, sizeof(float));
    m->input_ids = (int32_t *)calloc(context, sizeof(int32_t));
    if (!m->logits || !m->input_ids)
        goto fail;

    fprintf(stderr,
            "LFM_READY backend=%s expert_mode=%s context=%u batch=%u workers=%u "
            "expert_slots=%d expert_bytes=%lld live_kv_bytes=%zu "
            "index_entries=%d\n",
            m->gpu_mode ? (m->mixed_decode ? "gpu-prefill-cpu-decode" : "gpu")
                        : "cpu",
            m->full_resident ? "full-resident" : "limited",
            context, batch, workers, slots, (long long)LFM_EXPERT_BYTES,
            m->live_floats * sizeof(float), m->index.count);
    *out = m;
    return 0;

fail:
    fprintf(stderr, "LFM_OPEN_FAIL stage=%d\n", m->open_stage);
    lfm_runtime_close(m);
    return -1;
}

int lfm_runtime_encode(LfmRuntime *m, const char *text, int32_t *ids,
                       uint32_t capacity) {
    if (!m || !text || !ids || capacity == 0u || sizeof(int) != sizeof(int32_t))
        return -1;
    return salt_tokenizer_encode(&m->tokenizer, text, (int *)(void *)ids,
                                 (int)capacity);
}

int lfm_runtime_decode(LfmRuntime *m, const int32_t *ids, uint32_t count,
                       char *text, uint32_t capacity) {
    if (!m || !ids || !text || capacity == 0u || count > (uint32_t)INT_MAX ||
        capacity > (uint32_t)INT_MAX || sizeof(int) != sizeof(int32_t))
        return -1;
    return salt_tokenizer_decode(&m->tokenizer, (const int *)(const void *)ids,
                                 (int)count, text, (int)capacity);
}

int lfm_runtime_generate(LfmRuntime *m, const int32_t *prompt_ids,
                         uint32_t prompt_count, uint32_t max_output,
                         int32_t *out_ids) {
    uint32_t produced = 0u;
    uint64_t request_begin, prefill_end, request_end;
    int32_t next;

    if (!m || !out_ids)
        return -1;
    if (prompt_count && !prompt_ids)
        return -1;
    memset(&m->metrics, 0, sizeof m->metrics);
    request_begin = lfm_clock_ns();

    /* Execute the prompt in model-owned B-sized PREFILL chunks. Every chunk
     * commits directly into the same live state; only the final chunk publishes
     * vocabulary logits for the first decode step. */
    if (prompt_count) {
        uint32_t first = 0u;
        if (m->state.position + prompt_count > m->context || m->batch == 0u)
            return -1;
        while (first < prompt_count) {
            uint32_t active = prompt_count - first;
            int final;
            if (active > m->batch) active = m->batch;
            if (m->cpu_decode_active && active > m->workers)
                active = m->workers;
            final = first + active == prompt_count;
            if (lfm_execute_rows(m, prompt_ids + first, active, final, 0,
                    final ? m->logits : NULL))
                return -1;
            first += active;
        }
    }
    prefill_end = lfm_clock_ns();

    while (produced < max_output) {
        next = lfm_sample_argmax(m->logits, LFM_VOCAB);
        if (next < 0)
            return -1;
        out_ids[produced++] = next;
        if (next == LFM_EOS)
            break;
        if (m->state.position + 1u > m->context)
            break;
        if (lfm_execute_rows(m, &next, 1u, 1, 1, m->logits))
            return -1;
    }
    request_end = lfm_clock_ns();
    if (request_begin == 0u || prefill_end < request_begin ||
        request_end < prefill_end)
        return -1;
    m->metrics.request_ns = request_end - request_begin;
    m->metrics.prefill_ns = prefill_end - request_begin;
    m->metrics.decode_ns = request_end - prefill_end;
    m->metrics.prompt_tokens = prompt_count;
    m->metrics.completion_tokens = produced;
    return (int)produced;
}

int lfm_runtime_last_metrics(const LfmRuntime *m, LfmRuntimeMetrics *metrics) {
    if (!m || !metrics || m->metrics.request_ns == 0u)
        return -1;
    *metrics = m->metrics;
    return 0;
}

static void lfm_digest_hex(const uint8_t digest[32], char out[65]) {
    static const char HEX[] = "0123456789abcdef";
    for (size_t index = 0; index < 32u; index++) {
        out[index * 2u] = HEX[digest[index] >> 4u];
        out[index * 2u + 1u] = HEX[digest[index] & 15u];
    }
    out[64] = '\0';
}

int lfm_runtime_state_digests(const LfmRuntime *m,
        LfmRuntimeStateDigests *digests) {
    const SaltTextVerifyProgram *active_program;
    SaltSha256 kv_hash;
    uint8_t state_bytes[12], digest[32];
    uint64_t kv_bytes = 0u;
    if (!m || !digests || !m->canonical_base || !m->live ||
        m->state.position > m->context)
        return -1;
    active_program = m->cpu_decode_active ? &m->decode_program : &m->program;
    memset(digests, 0, sizeof *digests);
#if defined(LFM_GPU_HIP)
    if (m->gpu_mode) {
        uint64_t copied = 0u;
        if (!m->live_shared_ready ||
            salt_gpu_attention_materialize((SaltGpuSharedBuffer *)&m->live_shared,
                (int)m->state.position, &copied) != 0)
            return -1;
    }
#endif
    state_bytes[0] = (uint8_t)m->state.position;
    state_bytes[1] = (uint8_t)(m->state.position >> 8u);
    state_bytes[2] = (uint8_t)(m->state.position >> 16u);
    state_bytes[3] = (uint8_t)(m->state.position >> 24u);
    for (uint32_t byte = 0; byte < 8u; byte++)
        state_bytes[4u + byte] =
            (uint8_t)(m->state.transition_generation >> (8u * byte));
    if (salt_sha256_bytes(state_bytes, sizeof state_bytes, digest) != 0)
        return -1;
    lfm_digest_hex(digest, digests->state_sha256);
    salt_sha256_init(&kv_hash);
    for (uint32_t layer = 0; layer < LFM_LAYERS; layer++) {
        const SaltTextKvRowsDesc *keys = &m->kv[layer].keys;
        const SaltTextKvRowsDesc *values = &m->kv[layer].values;
        size_t floats;
        if (m->plans[layer].attention.kind == SALT_ATTN_MODEL)
            continue;
        if (keys->private_row_stride != values->private_row_stride ||
            m->state.position > keys->private_row_capacity ||
            m->state.position > values->private_row_capacity ||
            (size_t)m->state.position > SIZE_MAX / keys->private_row_stride)
            return -1;
        floats = (size_t)m->state.position * keys->private_row_stride;
        if (floats > SIZE_MAX / sizeof(float) ||
            floats > keys->private_float_capacity ||
            floats > values->private_float_capacity)
            return -1;
        salt_sha256_update(&kv_hash, keys->private_rows,
            floats * sizeof(float));
        salt_sha256_update(&kv_hash, values->private_rows,
            floats * sizeof(float));
        kv_bytes += (uint64_t)floats * 2u * sizeof(float);
    }
    salt_sha256_final(&kv_hash, digest);
    lfm_digest_hex(digest, digests->kv_sha256);
    if (m->gpu_mode && !m->cpu_decode_active) {
        const SaltGpuSharedBuffer *canonical = m->gpu.canonical;
        if (!canonical || canonical->contents != m->canonical_base ||
            !canonical->backend ||
            active_program->layout.model_state > canonical->nbytes ||
            active_program->layout.model_state_bytes > canonical->nbytes -
                active_program->layout.model_state ||
            lfm_gpu_canonical_readback(canonical->contents, canonical->backend,
                active_program->layout.model_state,
                active_program->layout.model_state_bytes) != 0)
            return -1;
    }
    if (salt_sha256_bytes(
            m->canonical_base + active_program->layout.model_state,
            active_program->layout.model_state_bytes, digest) != 0)
        return -1;
    lfm_digest_hex(digest, digests->model_state_sha256);
    if (!m->logits || salt_sha256_bytes(m->logits,
            (size_t)LFM_VOCAB * sizeof(float), digest) != 0)
        return -1;
    lfm_digest_hex(digest, digests->logits_sha256);
    digests->kv_bytes = kv_bytes;
    digests->model_state_bytes = active_program->layout.model_state_bytes;
    digests->transition_generation = m->state.transition_generation;
    digests->position = m->state.position;
    return 0;
}

void lfm_runtime_close(LfmRuntime *m) {
    if (!m)
        return;
    if (m->gpu.ready)
        (void)salt_text_verify_heterogeneous_destroy(&m->gpu);
    if (m->pool_threads_ready) {
        salt_kv_free(&m->pool_threads);
        m->pool_threads_ready = 0;
    }
    if (m->cache_ready)
        salt_cache_free(&m->cache);
    if (m->live_shared_ready)
        (void)salt_gpu_shared_buffer_free(&m->live_shared);
    if (m->gpu_trunk_window)
        (void)salt_gpu_weight_window_unbind(SALT_TENSOR_RESOURCE_TRUNK, 0u);
    if (m->gpu_trunk_bound)
        (void)salt_gpu_weight_resource_unbind(SALT_TENSOR_RESOURCE_TRUNK, 0u);
    if (m->gpu_ready)
        (void)salt_gpu_free();
    if (m->pool.owns_fd)
        (void)salt_pool_close(&m->pool);
    salt_tokenizer_free(&m->tokenizer);
    if (m->trunk)
        munmap(m->trunk, m->trunk_bytes);
    free(m->program_arena);
    free(m->decode_program_arena);
    free(m->cpu_arena);
    free(m->gpu_arena);
    free(m->conv_kernels);
    if (!m->live_shared_ready)
        free(m->live);
    free(m->logits);
    free(m->input_ids);
    free(m->index_entries);
    free(m);
}
