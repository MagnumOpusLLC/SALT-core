/* main.c -- salt CLI. Memory plan first (refuse to start past 95%),
 * then the decode loop: bind trunk layer, route, fetch experts, compute
 * sink, record trace. Run report separates application footprint from
 * resident/touched working set.
 *
 * Exit codes: 0 ok; 1 config/usage; 2 I/O; 3 memory limit
 * (graceful stop); 4 completed with dropped experts (silent numerical
 * corruption must not exit 0). */
#include "salt/salt.h"
#include "salt/kernels.h"
#include "salt/moe.h"
#include "attn.h"
#include "salt/layer.h"
#include "salt/model.h"
#include "salt/head.h"
#include "salt/tokenizer.h"
#include "salt/gpu.h"
#include "salt/gpu_pool.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

enum { SALT_PREFILL_BATCH_MAX = 4096 };

static SaltMemSnapshot memory_snapshot(void) {
    SaltMemSnapshot s;
    if (salt_mem_snapshot(&s) != 0) {
        memset(&s, 0, sizeof s);
        s.application_b = -1;
    }
    return s;
}

static void memory_sample_peak(SaltMemSnapshot *peak,
                               const SaltMemSnapshot *sample) {
    if (!peak || !sample) return;
    if (sample->application_b > peak->application_b)
        peak->application_b = sample->application_b;
    if (sample->application_os_peak_b > peak->application_os_peak_b)
        peak->application_os_peak_b = sample->application_os_peak_b;
    if (sample->resident_b > peak->resident_b)
        peak->resident_b = sample->resident_b;
    if (sample->resident_os_peak_b > peak->resident_os_peak_b)
        peak->resident_os_peak_b = sample->resident_os_peak_b;
}

static int memory_limit_breached(const SaltMemSnapshot *sample,
                                 double limit_gb, const char *checkpoint,
                                 int index, int total) {
    if (limit_gb <= 0.0) return 0;
    if (!sample || sample->application_b < 0) {
        fprintf(stderr,
                "\nMEMORY LIMIT: application-footprint measurement failed "
                "at %s -- fail closed (exit 3)\n", checkpoint);
        return 1;
    }
    double current_gb = (double)sample->application_b / 1e9;
    if (current_gb < limit_gb) return 0;
    fprintf(stderr,
            "\nMEMORY LIMIT: application footprint %.2f GB >= %.2f GB "
            "(--mem-limit-gb) at %s %d/%d%s -- stopping gracefully "
            "(exit 3)\n",
            current_gb, limit_gb, checkpoint, index, total,
            sample->application_is_approximate
                ? " (Linux process-local approximation)" : "");
    return 1;
}

static int gpu_prefill_phase_check(
        int captured, int gpu_chunks, uint64_t prefill_trunk_batches,
        uint64_t trunk_after_prefill, uint64_t direct_after_prefill,
        uint64_t direct_jobs_after_prefill,
        const SaltGpuBatchStats *request_start) {
    SaltGpuBatchStats now;
    const char *mix = getenv("SALT_GPU_INDEXED_MIX");
    int indexed_mix = mix && *mix && *mix != '0';
    salt_gpu_batch_stats_get(&now);
    int counters_valid = captured && request_start &&
        trunk_after_prefill >= request_start->trunk_batches &&
        now.trunk_batches >= trunk_after_prefill &&
        direct_after_prefill >= request_start->direct_output_batches &&
        now.direct_output_batches >= direct_after_prefill &&
        direct_jobs_after_prefill >= request_start->direct_output_jobs &&
        now.direct_output_jobs >= direct_jobs_after_prefill &&
        now.expert_layer_batches >= request_start->expert_layer_batches &&
        now.arena_batches >= request_start->arena_batches &&
        now.mapped_only_misses >= request_start->mapped_only_misses;
    uint64_t decode_trunk_batches = counters_valid
        ? now.trunk_batches - trunk_after_prefill : UINT64_MAX;
    uint64_t request_expert_batches = counters_valid
        ? now.expert_layer_batches - request_start->expert_layer_batches
        : UINT64_MAX;
    uint64_t request_arena_batches = counters_valid
        ? now.arena_batches - request_start->arena_batches : UINT64_MAX;
    uint64_t request_mapped_misses = counters_valid
        ? now.mapped_only_misses - request_start->mapped_only_misses
        : UINT64_MAX;
    uint64_t prefill_direct_batches = counters_valid
        ? direct_after_prefill - request_start->direct_output_batches
        : UINT64_MAX;
    uint64_t prefill_direct_jobs = counters_valid
        ? direct_jobs_after_prefill - request_start->direct_output_jobs
        : UINT64_MAX;
    uint64_t decode_direct_batches = counters_valid
        ? now.direct_output_batches - direct_after_prefill : UINT64_MAX;
    uint64_t decode_direct_jobs = counters_valid
        ? now.direct_output_jobs - direct_jobs_after_prefill : UINT64_MAX;
    fprintf(stderr, "gpu-phase: chunks=%d prefill trunk=%llu "
            "decode trunk=%llu direct=%llu/%llu decode-direct=%llu/%llu\n",
            gpu_chunks, (unsigned long long)prefill_trunk_batches,
            (unsigned long long)decode_trunk_batches,
            (unsigned long long)prefill_direct_batches,
            (unsigned long long)prefill_direct_jobs,
            (unsigned long long)decode_direct_batches,
            (unsigned long long)decode_direct_jobs);
    if (!counters_valid ||
        prefill_trunk_batches !=
            trunk_after_prefill - request_start->trunk_batches ||
        decode_trunk_batches != 0 || request_expert_batches != 0 ||
        request_arena_batches != 0 || request_mapped_misses != 0 ||
        decode_direct_batches != 0 || decode_direct_jobs != 0 ||
        (indexed_mix && gpu_chunks > 0 &&
         (prefill_direct_batches == 0 || prefill_direct_jobs == 0)) ||
        (gpu_chunks > 0 && prefill_trunk_batches == 0) ||
        (gpu_chunks == 0 && prefill_trunk_batches != 0)) {
        fprintf(stderr, "gpu-phase: request isolation failed "
                "expert=%llu arena=%llu mapped-misses=%llu\n",
                (unsigned long long)request_expert_batches,
                (unsigned long long)request_arena_batches,
                (unsigned long long)request_mapped_misses);
        return -1;
    }
    return 0;
}

static int parse_nonnegative_double(const char *text, double *value_out) {
    char *end = NULL;
    double value;
    if (!text || !*text || !value_out) return -1;
    errno = 0;
    value = strtod(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(value) || value < 0.0)
        return -1;
    *value_out = value;
    return 0;
}

static int gpu_address_audit_enabled(void) {
    const char *p = getenv("SALT_GPU_ADDRESS_AUDIT");
    return p && *p && *p != '0';
}

static int kv_header_pref(const uint64_t hdr[8], int *pref) {
    if (!pref || hdr[3] > (uint64_t)INT_MAX) return -1;
    *pref = (int)hdr[3];
    return 0;
}

/* Persisted recurrent-state dimensions are model geometry, never
 * file-owned allocation metadata. */
static int kv_state_geometry(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                             const SaltLayerDesc *layers, int n_layers,
                             int *v_heads, int *kd, int *vd, int *conv_rows) {
    int found = 0;
    int vh, kh, kdim, vdim, expected_rows;
    long model_rows = 0;

    if (!cfg || !tl || !layers || !v_heads || !kd || !vd || !conv_rows)
        return -1;
    *v_heads = *kd = *vd = *conv_rows = 0;
    for (int L = 0; L < n_layers; L++) {
        int pi;
        long rows;
        if (layers[L].attn != LAYER_LIN) continue;
        pi = tl->q3_pqkv[L];
        if (pi < tl->t_off[L] || pi >= tl->t_off[L + 1] ||
            tl->t[pi].rank != 2)
            return -1;
        rows = tl->t[pi].dims[0];
        if (found && rows != model_rows)
            return -1;
        model_rows = rows;
        found = 1;
    }
    if (found) {
        if (salt_model_linear_geometry(cfg, &kh, &vh, &kdim, &vdim,
                                       &expected_rows) != 0 ||
            model_rows != expected_rows)
            return -1;
        *v_heads = vh;
        *kd = kdim;
        *vd = vdim;
        *conv_rows = (int)model_rows;
    }
    return 0;
}

static int kv_header_state_matches(const uint64_t hdr[8], int v_heads,
                                   int kd, int vd, int conv_rows) {
    return hdr[4] == (uint64_t)v_heads && hdr[5] == (uint64_t)kd &&
           hdr[6] == (uint64_t)vd && hdr[7] == (uint64_t)conv_rows;
}

static void usage(const char *argv0) {
    fprintf(stderr,
        "usage: %s MODEL_DIR --trunk TRUNK --offsets OFFSETS [opts]\n"
        "  MODEL_DIR           dir containing config.json (and model.safetensors\n"
        "                      for the fixture path)\n"
        "  --trunk FILE        packed dense layers (tools/pack-trunk)\n"
        "  --offsets FILE      trunk offset table (trunk.offsets)\n"
        "  --pool FILE         packed expert pool (tools/convert-salt.py);\n"
        "                      when given, model.safetensors is not needed\n"
        "  --layout-trunk FILE trunk.json tensor layout (enables real MoE)\n"
        "  --layout-pool FILE  pool-mxfp4.json tensor layout (real MoE)\n"
        "  --dump-state FILE   write final hidden state (fp32) after run\n"
        "  --head FILE         head.json (logits; enables text mode)\n"
        "  --embed FILE        embed.json (embedding table; text mode)\n"
        "  --prompt-ids S      comma-separated token ids (first = seed)\n"
        "  --pids-file FILE    read comma/space-separated token ids from a file\n"
        "  --tokenizer FILE    tokenizer.json (ids <-> text; decodes output)\n"
        "  --text S            prompt text, encoded via --tokenizer\n"
        "  --eos-ids LIST      comma-separated stop token ids (e.g. 248046);\n"
        "                        generation stops when any is sampled\n"
        "  --gpu              offload the output-head matvec to Metal\n"
        "                        (macOS; env SALT_GPU=1; default CPU)\n"
        "  --cache-gb X        expert cache budget in GB       (default 5;\n"
        "                        env SALT_CACHE_GB; CLI wins)\\n"
        "  --trunk-gb X        trunk pin budget in GB          (default 4)\n"
        "  --pin-layers N      explicit pinned trunk prefix    (default auto)\n"
        "  --nring N           trunk ring slots, >= 2          (default 2)\n"
        "  --gen N             tokens to generate              (default 4)\n"
        "  --mem-limit-gb X    application-footprint hard ceiling in GB\n"
        "                      (default 6; 0 = no limit)\n"
        "  --prompt S          seed string for hidden state    (default \"salt\")\n"
        "  --locality F        router popularity boost, 0..1   (default 0)\n"
        "  --trace FILE        write (layer,expert) request log\n"
        "  --threads N         parallel expert-read threads    (default 4)\n"
        "  --preset NAME       conserve | full | laptop | server\n"
        "                        (default conserve)\n"
        "                        conserve: cache 2, trunk 2, mem 6, threads 4\n"
        "                        full:     cache 2, trunk 2, mem 8, threads 8\n"
        "  --no-refuse         start even if the plan says no\n"
        "  --no-simd           force scalar kernels (default: SIMD when "
        "available)\n",
        argv0);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    double wall_s, fetch_wait_s;
    uint64_t physical_read_bytes, major_faults;
    int physical_read_valid, major_faults_valid;
    int64_t logical_miss_bytes, requests, hits, fetch_jobs;
} RoofSnap;

static RoofSnap roof_snap(const SaltCache *c) {
    RoofSnap s = {0};
    s.wall_s = now_s();
    s.fetch_wait_s = c->fetch_wait_s;
    SaltIoSnapshot io;
    if (salt_io_snapshot(&io) == 0) {
        s.physical_read_bytes = io.process_read_bytes;
        s.major_faults = io.major_faults;
        s.physical_read_valid = io.process_read_bytes_valid;
        s.major_faults_valid = io.major_faults_valid;
    }
    s.logical_miss_bytes = c->nread;
    s.requests = c->nreq;
    s.hits = c->nhit;
    s.fetch_jobs = c->nfetch_jobs;
    return s;
}

static int roof_counter_delta(uint64_t start, int start_valid,
                              uint64_t end, int end_valid,
                              uint64_t *delta) {
    if (!delta || !start_valid || !end_valid || end < start) return 0;
    *delta = end - start;
    return 1;
}

static void roof_rate_text(char *dst, size_t dst_n, uint64_t delta,
                           int valid, double divisor) {
    if (!dst || dst_n == 0) return;
    if (!valid || divisor <= 0.0) {
        snprintf(dst, dst_n, "n/a");
        return;
    }
    snprintf(dst, dst_n, "%.2f", (double)delta / divisor);
}

static int env_enabled(const char *name) {
    const char *value = getenv(name);
    return value && *value && *value != '0';
}

static void sha256_hex(const uint8_t digest[32], char hex[65]) {
    static const char digit[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[i * 2] = digit[digest[i] >> 4];
        hex[i * 2 + 1] = digit[digest[i] & 15];
    }
    hex[64] = '\0';
}

static int gpu_pool_tensor_matches(
    const SaltGpuPool *pool, uint32_t layer, uint32_t expert,
    uint32_t component, const SaltMoETensor *tensor) {
    const SaltGpuPoolRef *ref;
    const SaltGpuDispatchTemplate *dispatch;
    uint64_t voff, soff, boff;
    if (!pool || !tensor || tensor->rank != 2 || tensor->fmt != 1 ||
        tensor->bits != 4 || tensor->rel_v < 0 || tensor->rel_s < 0 ||
        tensor->rel_b < 0 || tensor->v_nbytes < 1 ||
        tensor->s_nbytes < 1 || tensor->dims[0] < 1 ||
        tensor->dims[1] < 1 || (uint64_t)tensor->dims[0] > UINT32_MAX ||
        (uint64_t)tensor->dims[1] > UINT32_MAX)
        return 0;
    ref = salt_gpu_pool_ref(pool, layer, expert);
    dispatch = salt_gpu_pool_dispatch(pool, layer, expert, component);
    if (!ref || !dispatch ||
        (uint64_t)tensor->rel_v > UINT64_MAX - ref->local_offset ||
        (uint64_t)tensor->rel_s > UINT64_MAX - ref->local_offset ||
        (uint64_t)tensor->rel_b > UINT64_MAX - ref->local_offset)
        return 0;
    voff = ref->local_offset + (uint64_t)tensor->rel_v;
    soff = ref->local_offset + (uint64_t)tensor->rel_s;
    boff = ref->local_offset + (uint64_t)tensor->rel_b;
    return dispatch->R == (uint32_t)tensor->dims[0] &&
           dispatch->C == (uint32_t)tensor->dims[1] &&
           dispatch->fmt == (uint32_t)tensor->fmt &&
           dispatch->bits == (uint32_t)tensor->bits &&
           dispatch->voff == voff && dispatch->soff == soff &&
           dispatch->boff == boff &&
           dispatch->v_nbytes == (uint64_t)tensor->v_nbytes &&
           dispatch->s_nbytes == (uint64_t)tensor->s_nbytes &&
           dispatch->b_nbytes == (uint64_t)tensor->s_nbytes;
}

static int gpu_pool_layer_prepare(
    const SaltGpuPool *pool, const SaltPoolLayout *layout,
    uint32_t layer, const int *experts, int nexperts,
    SaltGpuPoolDeviceLayer *device_layer) {
    if (!pool || !layout || !experts || nexperts < 1 || !device_layer ||
        layer >= (uint32_t)layout->n_layers)
        return -1;
    if (salt_gpu_pool_device_layer_acquire(
            pool, layer, 1, device_layer) != 0)
        return (device_layer->lease.base || device_layer->lease.fd >= 0)
            ? -2 : -1;
    for (int j = 0; j < nexperts; j++) {
        int expert = experts[j];
        if (expert < 0 || expert >= layout->n_experts) goto fail;
        const SaltExpertLayout *el =
            &layout->exp[(size_t)layer * layout->n_experts + expert];
        for (uint32_t component = 0; component < 3; component++) {
            if (!gpu_pool_tensor_matches(pool, layer, (uint32_t)expert,
                                         component, &el->t[component]) ||
                salt_gpu_pool_device_layer_register(
                    pool, device_layer, (uint32_t)expert, component,
                    &el->t[component]) != 0)
                goto fail;
        }
    }
    return 0;
fail:
    if (device_layer->bound &&
        salt_gpu_pool_device_layer_release(device_layer) != 0)
        return -2;
    return -1;
}

static int gpu_trunk_layer_release(SaltTrunk *trunk, int enabled,
                                   int *device_active) {
    if (!enabled) return 0;
    if (!trunk || !device_active) return -1;
    if (*device_active) {
        if (salt_gpu_resident_trunk_unmap() != 0) return -1;
        *device_active = 0;
    }
    salt_gpu_set_mapped_only(0);
    return salt_trunk_unbind(trunk);
}

static void startup_storage_release(SaltExpertPool *pool, SaltTrunk *trunk,
                                    SaltTrunk *gpu_prefill_trunk,
                                    int gpu_prefill_trunk_open) {
    if (gpu_prefill_trunk_open && gpu_prefill_trunk &&
        salt_trunk_close(gpu_prefill_trunk) != 0)
        fprintf(stderr, "gpu-trunk: startup prefill-handle close failed\n");
    if (trunk && salt_trunk_close(trunk) != 0)
        fprintf(stderr, "trunk: startup close failed\n");
    if (pool) {
        if (salt_pool_close(pool) != 0) {
            fprintf(stderr, "pool: startup close failed\n");
        } else {
            free(pool->ref);
            pool->ref = NULL;
        }
    }
}

static const uint8_t *runtime_trunk_bind(SaltTrunk *trunk, int layer,
                                         int gpu_trunk_enabled,
                                         int *device_active,
                                         int *release_failed) {
    static int address_source_reported = 0;
    if (release_failed) *release_failed = 0;
    if (!gpu_trunk_enabled) return salt_trunk_bind(trunk, layer);
    if (gpu_trunk_layer_release(trunk, 1, device_active) != 0) {
        if (release_failed) *release_failed = 1;
        return NULL;
    }
    const uint8_t *payload = salt_trunk_bind(trunk, layer);
    size_t mapped_nbytes = 0;
    const void *base = salt_trunk_mapping(trunk, &mapped_nbytes);
    if (!payload || !base || !mapped_nbytes) goto prepare_fail;

    /* Explicit resident intent is fail-closed. Never reinterpret a failed
     * canonical registration as permission to use CPU or arena weights. */
    salt_gpu_set_mapped_only(1);
    if (salt_gpu_init() == 0) {
        int map_rc = salt_gpu_resident_trunk_map(base, mapped_nbytes);
        if (map_rc == 0) {
            if (gpu_address_audit_enabled() && !address_source_reported) {
                int64_t map_file_off = trunk->lay[layer].off -
                    (int64_t)trunk->map_data_offset;
                fprintf(stderr,
                        "[gpu-address] source sample=1 layer=%d "
                        "layer_file_off=%lld layer_bytes=%lld "
                        "map_file_off=%lld map_bytes=%zu payload_rel=%zu "
                        "map_base=%p payload=%p\n",
                        layer, (long long)trunk->lay[layer].off,
                        (long long)trunk->lay[layer].nbytes,
                        (long long)map_file_off, mapped_nbytes,
                        trunk->map_data_offset, base, payload);
                address_source_reported = 1;
            }
            *device_active = 1;
            return payload;
        }
    }
    fprintf(stderr, "gpu-trunk: canonical Metal registration failed at L%d\n",
            layer);

prepare_fail:
    if (gpu_trunk_layer_release(trunk, 1, device_active) != 0 &&
        release_failed)
        *release_failed = 1;
    return NULL;
}

/* name ends with suffix (NUL-terminated; mirrors moe.c's static) */
static int name_ends(const char *name, const char *suffix) {
    size_t nl = strlen(name), sl = strlen(suffix);
    return nl >= sl && !memcmp(name + nl - sl, suffix, sl);
}

static int qwen_state_view(const SaltStateModelDesc *state_model,
                           const SaltCfg *cfg, const SaltKvCache *cache,
                           int position, int kvlat, SaltStateView *view) {
    uint64_t bytes_per_row;
    if (!state_model || !cfg || !cache || !view || position < 0 ||
        cfg->n_layers < 1 || kvlat < 1)
        return -1;
    bytes_per_row = (uint64_t)(uint32_t)cfg->n_layers *
                    (uint32_t)kvlat * sizeof(float);
    memset(view, 0, sizeof *view);
    view->schema_version = SALT_STATE_SCHEMA_VERSION;
    view->session_epoch = 1u;
    view->lifetime_turn = (uint64_t)(uint32_t)position;
    view->position = (uint64_t)(uint32_t)position;
    view->mindset_end = state_model->default_mindset_end;
    view->bytes_per_row = bytes_per_row;
    view->mindset_kind = state_model->mindset_kind;
    view->facts_kind = state_model->facts_kind;
    view->facts_optional = state_model->facts_optional;
    return salt_state_view_validate(state_model, view);
}

static int qwen_state_transaction_begin(
        SaltStateControl *control, const SaltStateModelDesc *state_model,
        const SaltCfg *cfg, const SaltKvCache *cache, int position, int kvlat,
        SaltStateTransactionKind kind, SaltStateArtifactKind artifact,
        SaltStateTransactionReason reason, SaltStateTransaction *transaction) {
    SaltStateView before;
    if (qwen_state_view(state_model, cfg, cache, position, kvlat, &before) != 0)
        return -1;
    return salt_state_transaction_begin(control, &before, kind, artifact,
        reason, transaction);
}

static int qwen_state_transaction_finish(
        const SaltStateModelDesc *state_model, const SaltCfg *cfg,
        const SaltKvCache *cache, int position, int kvlat,
        SaltStateTransaction *transaction) {
    SaltStateView after;
    if (qwen_state_view(state_model, cfg, cache, position, kvlat, &after) != 0)
        return -1;
    return salt_state_transaction_finish(transaction, &after);
}

static int qwen_state_save_full(const char *path, const SaltCfg *cfg,
                                const SaltKvCache *cache, int position,
                                int kvlat) {
    FILE *stream;
    uint64_t header[8];
    size_t row;
    if (!path || !*path || !cfg || !cache || !cache->kv || position < 0 ||
        position > cache->max_tokens || kvlat != cache->kvlat)
        return -1;
    stream = fopen(path, "wb");
    if (!stream) return -1;
    header[0] = UINT64_C(0x5354414C544B5631);
    header[1] = (uint64_t)(uint32_t)cfg->n_layers;
    header[2] = (uint64_t)(uint32_t)kvlat;
    header[3] = (uint64_t)(uint32_t)position;
    header[4] = (uint64_t)(uint32_t)cache->lin_vh;
    header[5] = (uint64_t)(uint32_t)cache->lin_kd;
    header[6] = (uint64_t)(uint32_t)cache->lin_vd;
    header[7] = (uint64_t)(uint32_t)cache->conv_rows;
    if (fwrite(header, sizeof header, 1, stream) != 1) goto fail;
    row = (size_t)kvlat * sizeof(float);
    for (int layer = 0; layer < cfg->n_layers; layer++) {
        const float *base = cache->kv +
            (size_t)layer * (size_t)cache->max_tokens * (size_t)kvlat;
        if (fwrite(base, row, (size_t)position, stream) !=
                (size_t)position)
            goto fail;
    }
    if ((cache->lin_alloc &&
         fwrite(cache->lin, sizeof(float),
                (size_t)cfg->n_layers * (size_t)cache->lin_vh *
                (size_t)cache->lin_kd * (size_t)cache->lin_vd, stream) !=
            (size_t)cfg->n_layers * (size_t)cache->lin_vh *
                (size_t)cache->lin_kd * (size_t)cache->lin_vd) ||
        (cache->conv_alloc &&
         fwrite(cache->conv, sizeof(float),
                (size_t)cfg->n_layers * 4u * (size_t)cache->conv_rows,
                stream) !=
            (size_t)cfg->n_layers * 4u * (size_t)cache->conv_rows) ||
        fflush(stream) != 0 || ferror(stream))
        goto fail;
    return fclose(stream) == 0 ? 0 : -1;

fail:
    fclose(stream);
    return -1;
}

int main(int argc, char **argv) {
    fprintf(stderr, "salt build %s\n",
#ifdef SALT_GIT
            SALT_GIT
#else
            "dev"
#endif
            );
    const char *model_dir = NULL, *trunk_path = NULL, *off_path = NULL;
    const char *trace_path = NULL, *prompt = "salt", *pool_path = NULL;
    /* EOS stop ids (--eos-ids "248044,248046"): generation stops when a
     * sampled token matches. 0 = disabled. */
    int eos_ids[16], n_eos = 0;
    const char *tl_path = NULL, *pl_path = NULL, *dump_path = NULL;
    const char *head_path = NULL, *embed_path = NULL, *prompt_ids = NULL;
    const char *pids_file = NULL;
    const char *kv_save_path = NULL;   /* --kv-save PATH: write the KV
                                          cache after prefill (doc-QA
                                          reuse; buy speed from storage) */
    const char *kv_load_path = NULL;   /* --kv-load PATH: restore the
                                          KV cache before prefill and
                                          start the chunk pass at the
                                          saved token count */
    const char *tok_path = NULL, *text_arg = NULL;
    const char *tokids_path = NULL;   /* --dump-tokids PATH: append raw
                                         generated token ids (int32) */
    const char *kv_save_after = NULL; /* --kv-save-after PATH: save KV
                                         AFTER generation, rows
                                         [0..npids+gen) -- captures the
                                         whole QA session (v1) */
    const char *kv_save_mind = NULL;  /* --kv-save-mindset PATH: save
                                         ONLY the lin delta + conv ring
                                         (the mental model, the ICL) */
    const char *kv_save_mem = NULL;   /* --kv-save-memory PATH: save
                                         ONLY the GQA K/V rows (the
                                         episodic transcript) */
    const char *kv_load_mind = NULL;  /* --kv-load-mindset PATH */
    const char *kv_load_mem = NULL;   /* --kv-load-memory PATH */
    const char *serve_fifo = NULL;   /* --serve-fifo PATH: persistent
                                        loop (EC2/hosting); reads
                                        comma-separated pids, one per
                                        line, from the fifo */
    double cache_gb = 2.0, trunk_gb = 2.0, locality = 0.0;
    double mem_limit_gb = 6.0;   /* hard cap: engine headroom is
                                    ~8.8GB on the M4 Pro; 6 keeps the
                                    OS baseline (17GB) + engine safe.
                                    Coherent budget set (measured):
                                    cache 2 + trunk budget 2 (the
                                    whole 864.6MB trunk pins) + base
                                    ~1.6GB = ~4.5-5GB PEAK RSS. */
    int pin_layers = -1, nring = 2, gen = 4, threads = 4, refuse = 1;
    /* --gpu and SALT_GPU=1 both funnel through the env var; head.c
     * reads SALT_GPU to decide whether to try the Metal path. */
    /* SALT_STRIP_THINK=1 (opt-in, default off): suppress the model's
     * <think>...</think> reasoning trace in the decoded stdout. The
     * trace is still GENERATED (it conditions the answer via the KV
     * cache); only the printed text is filtered. Standard serving
     * pattern for reasoning models (Qwen3/DeepSeek-R1/o-series). */
    int strip_think = 0;
    const char *env_st = getenv("SALT_STRIP_THINK");
    if (!env_st) env_st = getenv("SALT_THINK_HIDE"); /* alias */
    if (env_st && *env_st && strcmp(env_st, "0") != 0) strip_think = 1;
    /* SALT_CACHE_GB env overrides the default (CLI --cache-gb still wins).
     * Default 2 GB: the 3-4 GB resident target; 8+ GB only when asked. */
    const char *env_cache_gb = getenv("SALT_CACHE_GB");
    int cache_gb_env = env_cache_gb && *env_cache_gb;
    if (cache_gb_env) cache_gb = atof(env_cache_gb);
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--preset")) {
            if (++i >= argc) return usage(argv[0]), 1;
            if (!strcmp(argv[i], "conserve")) {
                /* the measured safe operating point (M4 Pro):
                 * cache 2 + trunk 2 (whole 864.6MB pins) + base
                 * ~1.6GB = ~4.5-5GB PEAK RSS, under mem-limit 6. */
                if (!cache_gb_env) cache_gb = 2;
                trunk_gb = 2; mem_limit_gb = 6; threads = 4;
            }
            else if (!strcmp(argv[i], "full")) {
                /* max performance within the frame: same weight
                 * budgets (cache 2 hard cap), mem headroom to 8,
                 * all 8 spawn threads. */
                if (!cache_gb_env) cache_gb = 2;
                trunk_gb = 2; mem_limit_gb = 8; threads = 8;
            }
            else if (!strcmp(argv[i], "laptop")) {
                if (!cache_gb_env) cache_gb = 8;
                trunk_gb = 4;
            }
            else if (!strcmp(argv[i], "server")) {
                if (!cache_gb_env) cache_gb = 64;
                trunk_gb = 48; nring = 4;
            }
            else { fprintf(stderr, "unknown preset %s\n", argv[i]); return 1; }
        } else if (!strcmp(argv[i], "--cache-gb") && i + 1 < argc) cache_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gpu")) setenv("SALT_GPU", "1", 1);
        else if (!strcmp(argv[i], "--trunk-gb") && i + 1 < argc) trunk_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--pin-layers") && i + 1 < argc) pin_layers = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nring") && i + 1 < argc) nring = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mem-limit-gb") && i + 1 < argc) {
            if (parse_nonnegative_double(argv[++i], &mem_limit_gb) != 0) {
                fprintf(stderr, "invalid --mem-limit-gb value: %s\n", argv[i]);
                return 1;
            }
        }
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--locality") && i + 1 < argc) locality = atof(argv[++i]);
        else if (!strcmp(argv[i], "--trace") && i + 1 < argc) trace_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trunk") && i + 1 < argc) trunk_path = argv[++i];
        else if (!strcmp(argv[i], "--offsets") && i + 1 < argc) off_path = argv[++i];
        else if (!strcmp(argv[i], "--pool") && i + 1 < argc) pool_path = argv[++i];
        else if (!strcmp(argv[i], "--layout-trunk") && i + 1 < argc) tl_path = argv[++i];
        else if (!strcmp(argv[i], "--layout-pool") && i + 1 < argc) pl_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-state") && i + 1 < argc) dump_path = argv[++i];
        else if (!strcmp(argv[i], "--head") && i + 1 < argc) head_path = argv[++i];
        else if (!strcmp(argv[i], "--embed") && i + 1 < argc) embed_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt-ids") && i + 1 < argc) prompt_ids = argv[++i];
        else if (!strcmp(argv[i], "--pids-file") && i + 1 < argc) pids_file = argv[++i];
        else if (!strcmp(argv[i], "--kv-save") && i + 1 < argc) kv_save_path = argv[++i];
        else if (!strcmp(argv[i], "--kv-load") && i + 1 < argc) kv_load_path = argv[++i];
        else if (!strcmp(argv[i], "--serve-fifo") && i + 1 < argc) serve_fifo = argv[++i];
        else if (!strcmp(argv[i], "--tokenizer") && i + 1 < argc) tok_path = argv[++i];
        else if (!strcmp(argv[i], "--dump-tokids") && i + 1 < argc) tokids_path = argv[++i];
        else if (!strcmp(argv[i], "--kv-save-after") && i + 1 < argc) kv_save_after = argv[++i];
        else if (!strcmp(argv[i], "--kv-save-mindset") && i + 1 < argc) kv_save_mind = argv[++i];
        else if (!strcmp(argv[i], "--kv-save-memory") && i + 1 < argc) kv_save_mem = argv[++i];
        else if (!strcmp(argv[i], "--kv-load-mindset") && i + 1 < argc) kv_load_mind = argv[++i];
        else if (!strcmp(argv[i], "--kv-load-memory") && i + 1 < argc) kv_load_mem = argv[++i];
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text_arg = argv[++i];
        else if (!strcmp(argv[i], "--eos-ids") && i + 1 < argc) {
            char *dup = strdup(argv[++i]);
            if (dup) {
                char *save = NULL;
                for (char *tk = strtok_r(dup, ",", &save);
                     tk && n_eos < 16; tk = strtok_r(NULL, ",", &save))
                    eos_ids[n_eos++] = atoi(tk);
                free(dup);
            }
        }
        else if (!strcmp(argv[i], "--no-refuse")) refuse = 0;
        else if (!strcmp(argv[i], "--no-simd")) salt_kernels_set_simd(0);
        else if (!model_dir) model_dir = argv[i];
        else { usage(argv[0]); return 1; }
    }
    if (!model_dir || !trunk_path || !off_path) { usage(argv[0]); return 1; }
    if (gen < 1) gen = 1;
    if (nring < 2) nring = 2;
    if ((tl_path || pl_path) && !pool_path) {
        fprintf(stderr, "--layout-* requires --pool\n");
        return 1;
    }

    SaltCfg cfg;
    if (salt_cfg_load(&cfg, model_dir) != 0) return 1;
    int gpu_intent = env_enabled("SALT_GPU");
    int gpu_moe_intent = gpu_intent && env_enabled("SALT_GPU_MOE");
    int gpu_trunk_intent = gpu_intent && env_enabled("SALT_GPU_TRUNK");
    int gpu_trunk_prefill_requested =
        env_enabled("SALT_GPU_TRUNK_PREFILL_ONLY");
    /* Subordinate GPU policy is inert while the model's master switch is off.
     * This lets a CPU-default blueprint describe its qualified opt-in mode. */
    int gpu_trunk_prefill_only = gpu_intent && gpu_trunk_prefill_requested;
    if (gpu_trunk_prefill_requested && gpu_intent && !gpu_trunk_intent) {
        fprintf(stderr, "gpu-trunk: SALT_GPU_TRUNK_PREFILL_ONLY requires "
                "SALT_GPU=1 and SALT_GPU_TRUNK=1\n");
        return 1;
    }
    if (gpu_trunk_prefill_only && gpu_moe_intent) {
        fprintf(stderr, "gpu-trunk: prefill-only mix requires "
                "SALT_GPU_MOE=0 for CPU decode\n");
        return 1;
    }
    const char *legacy_gpu_ledger = getenv("SALT_GPU_LEDGER");
    const char *gpu_resident = getenv("SALT_GPU_RESIDENT");
    if (legacy_gpu_ledger && *legacy_gpu_ledger)
        fprintf(stderr, "gpu-index: SALT_GPU_LEDGER is migration-only "
                "and ignored by production\n");
    if (gpu_trunk_intent && gpu_resident && *gpu_resident == '0') {
        fprintf(stderr, "gpu-trunk: SALT_GPU_TRUNK requires bounded "
                "source-backed residency\n");
        return 1;
    }

    SaltExpertPool pool;
    const char *pool_src;
    if (pool_path) {
        /* packed pool from tools/convert-salt.py: no safetensors needed */
        if (salt_pool_open_packed(&pool, pool_path, &cfg) != 0) return 2;
        pool_src = "packed";
    } else {
        char st_path[4096];
        snprintf(st_path, sizeof st_path, "%s/model.safetensors", model_dir);
        SaltSt st;
        if (salt_st_open(&st, st_path) != 0) return 2;
        if (salt_pool_build(&pool, &st, &cfg) != 0) return 2;
        salt_st_close(&st);
        pool_src = "safetensors";
    }

    SaltTrunk trunk;
    if (salt_trunk_open(&trunk, trunk_path, off_path) != 0) return 2;
    if (trunk.n_layers != cfg.n_layers &&
        trunk.n_layers != cfg.n_layers + 1) {
        /* +1 tolerated: the Qwen3.5 builder rides the final norm in a
         * synthetic last trunk layer (before lm_head) */
        fprintf(stderr, "trunk has %d layers, config says %d\n",
                trunk.n_layers, cfg.n_layers);
        startup_storage_release(&pool, &trunk, NULL, 0);
        return 1;
    }

    /* pin plan: explicit or fixed-point against the trunk budget */
    int npin;
    int64_t slot;
    if (pin_layers >= 0) {
        npin = pin_layers > trunk.n_layers ? trunk.n_layers : pin_layers;
        slot = 0;
        for (int L = npin; L < trunk.n_layers; L++)
            if (trunk.lay[L].nbytes > slot) slot = trunk.lay[L].nbytes;
        if (slot <= 0) slot = 4096;
    } else {
        salt_trunk_plan(&trunk, (int64_t)(trunk_gb * 1e9), &npin, &slot, nring);
    }
    if (gpu_trunk_intent && !gpu_trunk_prefill_only) {
        npin = 0;
        slot = 0;
    }

    int64_t cache_bytes = (int64_t)(cache_gb * 1e9);
    int nslot = (int)(cache_bytes / cfg.expert_nbytes);
    if (nslot < 1) nslot = 1;
    int64_t shared_bytes = (int64_t)cfg.n_shared * cfg.n_layers * cfg.expert_nbytes;

    SaltMemPlan plan;
    salt_mem_plan(&plan, &cfg, 0, cache_bytes, shared_bytes, npin, slot,
                  gpu_trunk_intent && !gpu_trunk_prefill_only ? 0 : nring);
    /* the plan needs the REAL pinned bytes, not a budget guess */
    plan.trunk_pin_b = 0;
    for (int L = 0; L < npin; L++) plan.trunk_pin_b += (double)trunk.lay[L].nbytes;
    plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                  plan.shared_b + plan.state_b + plan.index_b;
    salt_mem_print(&plan);
    if (refuse && salt_mem_refuses(&plan)) {
        fprintf(stderr,
                "\nREFUSING TO START: this needs %.1f GB and the machine has "
                "%.1f GB available, a shortfall of %.1f GB.\n"
                "Options: a larger box, a smaller --cache-gb/--trunk-gb, or "
                "fewer --pin-layers.\n",
                plan.need_b / 1e9, plan.have_b / 1e9,
                (plan.need_b - plan.have_b) / 1e9);
        startup_storage_release(&pool, &trunk, NULL, 0);
        return 1;
    }

    SaltTrunk gpu_prefill_trunk;
    int gpu_prefill_trunk_open = 0;
    if (gpu_trunk_prefill_only) {
        fprintf(stderr,
                "gpu-trunk: phase mix prefill=bounded-metal decode=cpu\n");
        fprintf(stderr, "moe: starting CPU decode trunk "
                "(npin=%d slot=%lld ring=%d)\n",
                npin, (long long)slot, nring);
        if (salt_trunk_start(&trunk, npin, slot, nring) != 0) {
            startup_storage_release(&pool, &trunk, NULL, 0);
            return 2;
        }
        if (salt_trunk_open(&gpu_prefill_trunk, trunk_path, off_path) != 0) {
            startup_storage_release(&pool, &trunk, NULL, 0);
            return 2;
        }
        gpu_prefill_trunk_open = 1;
        if (gpu_prefill_trunk.n_layers != trunk.n_layers ||
            salt_trunk_start_mapped(&gpu_prefill_trunk) != 0) {
            startup_storage_release(&pool, &trunk, &gpu_prefill_trunk,
                                    gpu_prefill_trunk_open);
            return 2;
        }
    } else if (gpu_trunk_intent) {
        fprintf(stderr, "moe: starting trunk (bounded canonical layers)\n");
        if (salt_trunk_start_mapped(&trunk) != 0) {
            startup_storage_release(&pool, &trunk, NULL, 0);
            return 2;
        }
    } else {
        fprintf(stderr, "moe: starting trunk (npin=%d slot=%lld ring=%d)\n",
                npin, (long long)slot, nring);
        if (salt_trunk_start(&trunk, npin, slot, nring) != 0) {
            startup_storage_release(&pool, &trunk, NULL, 0);
            return 2;
        }
    }
    fprintf(stderr, "moe: trunk started; cache init (%d slots, %d threads)\n",
            nslot, threads);
    SaltCache cache;
    if (salt_cache_init(&cache, &pool, nslot, threads) != 0) {
        startup_storage_release(&pool, &trunk, &gpu_prefill_trunk,
                                gpu_prefill_trunk_open);
        return 2;
    }
    fprintf(stderr,
            "moe: cache ready (layer-quota=%s reserved=%d shared=%d sum=%d)\n",
            cache.quota_enabled ? "on" : "off", cache.reserved_slots,
            cache.shared_slots, cache.reserved_slots + cache.shared_slots);

    FILE *trf = NULL;
    if (trace_path) {
        trf = fopen(trace_path, "w");
        if (!trf) {
            fprintf(stderr, "cannot open %s\n", trace_path);
            salt_cache_free(&cache);
            startup_storage_release(&pool, &trunk, &gpu_prefill_trunk,
                                    gpu_prefill_trunk_open);
            return 2;
        }
        fprintf(trf, "# expert_bytes=%lld\n", (long long)cfg.expert_nbytes);
    }

    /* real MoE mode: both layouts present and the pool is mxfp4 */
    SaltTrunkLayout tl;
    SaltPoolLayout pl;
    SaltGpuPool gpu_pool;
    int gpu_pool_ok = 0;
    int gpu_trunk_map_active = 0;
    int gpu_trunk_release_failed = 0;
    int gpu_prefill_defer_prior = 0;
    int gpu_prefill_defer_active = 0;
    SaltTrunk *gpu_runtime_trunk = gpu_trunk_prefill_only
        ? &gpu_prefill_trunk : &trunk;
    uint64_t gpu_prefill_trunk_start_batches = 0;
    uint64_t gpu_prefill_trunk_batches = 0;
    uint64_t gpu_prefill_trunk_after_batches = 0;
    uint64_t gpu_prefill_direct_after_batches = 0;
    uint64_t gpu_prefill_direct_after_jobs = 0;
    SaltGpuBatchStats gpu_request_start_stats = {0};
    int gpu_prefill_gpu_chunks = 0;
    int gpu_prefill_stats_captured = 0;
    int gpu_prefill_phase_validated = 0;
    SaltKvCache kvc;
    SaltStateControl state_control = {0};
    SaltStateTransaction state_transaction = {0};
    const SaltStateModelDesc *state_model = NULL;
    int kv_ok = 0;
    int rc = 0;                  /* serve-loop exit code (function
                                    scope: goto sites are outside the
                                    head block) */
    int moe_mode = 0;
    int kvlat = 0;              /* per-token KV bytes/layer (GQA/MLA) */
    float *state = NULL, *scratch = NULL, *xin_buf = NULL;
    float *prev_state = NULL, *prev_hin = NULL;
    int mhc_streams = 1;      /* residual streams (n_hc), 1 without mHC */
    int kv_state_vh = 0, kv_state_kd = 0, kv_state_vd = 0;
    int kv_state_conv = 0;
    float **jscratch = NULL;
    long scratch_n = 0;
    int64_t n_matvec = 0, n_decode = 0;
    /* SALT_WATERFALL: per-phase accumulators (seconds) across the
     * whole decode loop -- attn, router, expert fetch, moe, head,
     * misc. Printed after the run report. */
    double wf_attn = 0, wf_route = 0, wf_fetch = 0, wf_moe = 0,
           wf_head = 0, wf_misc = 0;
    double wf_pbind = 0, wf_pattn = 0, wf_pgqa = 0, wf_prfm = 0;
    double wf_proute = 0, wf_pfetch = 0, wf_pmoe = 0;
    int wf_on = getenv("SALT_WATERFALL") ? 1 : 0;
    /* SALT_STEP_MS: per-TOKEN phase breakdown (attn/route/fetch/moe/
     * head) printed after each token's head -- the decode-side step
     * profiler. Per-layer detail comes from SALT_CHUNK_MS on the
     * chunk side. */
    int step_ms = getenv("SALT_STEP_MS") ? 1 : 0;
    /* L2 (read-head): SALT_PIN_HOT=1 enables online hot-expert pinning
     * (promote a slot to pinned once its hit count reaches SALT_PIN_HITS,
     * default 8). Bit-fidelity-neutral: eviction policy never touches
     * the decode math, only which experts stay resident. */
    int pin_hot = getenv("SALT_PIN_HOT") ? 1 : 0;
    int pin_hits = 8;
    {
        const char *ph = getenv("SALT_PIN_HITS");
        if (ph) pin_hits = atoi(ph);
        if (pin_hits < 1) pin_hits = 1;
    }
    double st_attn = 0, st_route = 0, st_fetch = 0, st_moe = 0,
           st_head = 0, st_wall = 0;
    /* the layer-kind registry: the model-specific surface (which
     * layer is a GQA/LIN/MoE) as DATA -- the engine's walk is one
     * uniform loop over this. A different model = a different
     * registry fill, no engine change. Built once; the per-layer
     * dispatch below reads layers[L]. */
    SaltLayerDesc layers[SALT_MAX_LAYERS];
    int n_layers_reg = 0;
    if (tl_path && pl_path) {
        fprintf(stderr, "moe: loading trunk layout %s\n", tl_path);
        if (salt_trunk_layout_load(&tl, tl_path) != 0) return 1;
        fprintf(stderr, "moe: trunk layout ok (%d layers)\n", tl.n_layers);
        n_layers_reg = salt_layers_build(&tl, layers, SALT_MAX_LAYERS);
        if (kv_state_geometry(&cfg, &tl, layers, n_layers_reg,
                              &kv_state_vh, &kv_state_kd, &kv_state_vd,
                              &kv_state_conv) != 0) {
            fprintf(stderr, "moe: invalid linear-attention geometry\n");
            return 1;
        }
        {
            int gqa = 0, lin = 0;
            for (int i = 0; i < n_layers_reg; i++)
                if (layers[i].attn == LAYER_GQA) gqa++;
                else if (layers[i].attn == LAYER_LIN) lin++;
            fprintf(stderr, "moe: layer registry: %d layers "
                    "(%d GQA, %d LIN, all MoE)\n", n_layers_reg, gqa, lin);
        }
        fprintf(stderr, "moe: loading pool layout %s\n", pl_path);
        if (salt_pool_layout_load_ex(&pl, pl_path, &cfg, gpu_moe_intent) != 0)
            return 1;
        char pool_index_hex[65];
        sha256_hex(pl.index_sha256, pool_index_hex);
        fprintf(stderr, "moe: pool layout ok (%d x %d, max_rc=%lld, "
                "sha256=%s)\n", pl.n_layers, pl.n_experts,
                (long long)pl.max_rc, pool_index_hex);
        if (gpu_moe_intent) {
            if (!pool_path || !pl.gpu.present ||
                salt_gpu_pool_from_layout(&gpu_pool, &pl, &pool, pool_path) != 0) {
                fprintf(stderr, "gpu-index: unified expert metadata/source "
                        "validation failed\n");
                return 1;
            }
            gpu_pool_ok = 1;
            fprintf(stderr, "gpu-index: unified expert view v%u source="
                    "%llu:%llu bytes=%llu partition=layer\n",
                    pl.gpu.version,
                    (unsigned long long)pl.source_identity.device,
                    (unsigned long long)pl.source_identity.inode,
                    (unsigned long long)pl.source_identity.nbytes);
        }
        moe_mode = 1;
        scratch_n = pl.max_rc;
        scratch = (float *)malloc((size_t)scratch_n * sizeof(float));
        if (!scratch) { fprintf(stderr, "moe: scratch alloc failed\n"); return 2; }
        /* job-scratch pool: topk-1 extra max_rc buffers, allocated ONCE
         * (per-call malloc page-faults ~16 MB x topk x layers) */
        jscratch = (float **)calloc((size_t)cfg.topk, sizeof(float *));
        if (!jscratch) { fprintf(stderr, "moe: jscratch alloc failed\n"); return 2; }
        for (int k = 0; k < cfg.topk - 1; k++) {
            jscratch[k] = (float *)malloc((size_t)scratch_n * sizeof(float));
            if (!jscratch[k]) {
                fprintf(stderr, "moe: jscratch[%d] alloc failed\n", k);
                return 2;
            }
        }
        plan.state_b += (double)scratch_n * 4.0 * (double)cfg.topk;
        /* KV cache: MLA (DS-V4, kvlat) or Qwen3 GQA (krows+vrows from
         * the first self_attn layer). Qwen3: 512+512=1024 floats/token. */
        kvlat = tl.kvlat;
        if (kvlat < 1) {
            for (int L2 = 0; L2 < cfg.n_layers && kvlat < 1; L2++) {
                if (tl.q3_k[L2] >= 0) {
                    kvlat = (int)tl.t[tl.q3_k[L2]].dims[0] +
                            (int)tl.t[tl.q3_v[L2]].dims[0];
                }
            }
        }
        if (kvlat < 1) kvlat = 1;
        state_model = salt_model_state(salt_model_get("qwen36"));
        memset(&state_transaction, 0, sizeof state_transaction);
        if (!state_model ||
            salt_state_control_init(
                &state_control, state_model, NULL, NULL, NULL) != 0) {
            fprintf(stderr, "state: Qwen materialization control init failed\n");
            return 1;
        }
        plan.state_b += (double)kvlat * 4.0 * (double)cfg.n_layers *
                        (double)gen;
        plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                      plan.shared_b + plan.state_b + plan.index_b;
        int nreal = 0;
        for (int L = 0; L < cfg.n_layers; L++)
            if (tl.gate[L] >= 0) nreal++;
        fprintf(stderr, "router: real matvec on %d/%d layers, "
                        "others hash fallback\n", nreal, cfg.n_layers);
        int nha = 0, nhf = 0;
        for (int L = 0; L < cfg.n_layers; L++) {
            if (tl.hc_attn_fn[L] >= 0) nha++;
            if (tl.hc_ffn_fn[L] >= 0) nhf++;
        }
        fprintf(stderr, "hc: mHC on %d/%d attn, %d/%d ffn\n",
                nha, cfg.n_layers, nhf, cfg.n_layers);
        fprintf(stderr, "attn: %s (heads %d, qk_rope %d)\n",
                cfg.n_heads > 0 ? "real MLA" : "kvhalf fallback",
                cfg.n_heads, cfg.qk_rope);
        for (int L = 0; L < 1 && L < cfg.n_layers; L++) {
            int idx[3] = { tl.hc_attn_fn[L], tl.hc_attn_base[L],
                           tl.hc_attn_scale[L] };
            const char *nm[3] = { "attn_fn", "attn_base", "attn_scale" };
            for (int k = 0; k < 3; k++)
                if (idx[k] >= 0)
                    fprintf(stderr, "hc L%d %s: dtype %d shape [%ld",
                            L, nm[k], tl.t[idx[k]].dtype,
                            (long)tl.t[idx[k]].dims[0]);
            int idx2[3] = { tl.hc_ffn_fn[L], tl.hc_ffn_base[L],
                            tl.hc_ffn_scale[L] };
            const char *nm2[3] = { "ffn_fn", "ffn_base", "ffn_scale" };
            for (int k = 0; k < 3; k++)
                if (idx2[k] >= 0)
                    fprintf(stderr, " L%d %s: dtype %d shape [%ld",
                            L, nm2[k], tl.t[idx2[k]].dtype,
                            (long)tl.t[idx2[k]].dims[0]);
            fprintf(stderr, "\n");
        }
    }

    uint64_t hstate = salt_mix64(0);
    for (const char *p = prompt; *p; p++)
        hstate = salt_mix64(hstate ^ (uint8_t)*p);

    /* text mode (issue #6 step 3): embed -> layers -> head -> sample */
    SaltHead head;
    SaltEmbed embed;
    float *logits = NULL;
    int *pids = NULL;
    int npids = 0, text_mode = 0;
    int kv_prefilled = 0;   /* session memory: rows already computed.
                               Lives ACROSS requests in session mode
                               (SALT_SESSION=1); reset per request
                               otherwise. */
    uint64_t rng = hstate;
    SaltTokenizer tok;
    memset(&head, 0, sizeof head);
    memset(&embed, 0, sizeof embed);
    memset(&tok, 0, sizeof tok);
    if (tok_path &&
        salt_tokenizer_load(&tok, tok_path) != 0) {
        fprintf(stderr, "tokenizer load failed\n");
        return 1;
    }
    if (head_path && embed_path) {
        if (salt_head_load(&head, head_path) != 0) {
            fprintf(stderr, "head load failed (%s)\n", head_path);
            return 1;
        }
        if (salt_embed_load(&embed, embed_path) != 0) {
            fprintf(stderr, "embed load failed (%s)\n", embed_path);
            return 1;
        }
        long head_hidden = -1, embed_hidden = -1;
        if (head.rank == 2 && head.dims[0] > 0 && head.dims[1] > 0) {
            head_hidden = head.dims[1];
            if (head.w_dtype == 6) {
                int pack = 32 / head.w_bits;
                head_hidden = head_hidden <= LONG_MAX / pack
                    ? head_hidden * pack : -1;
            }
        }
        if (embed.rank == 2 && embed.dims[0] > 0 && embed.dims[1] > 0) {
            embed_hidden = embed.dims[1];
            if (embed.dtype == 6) {
                int pack = 32 / embed.w_bits;
                embed_hidden = embed_hidden <= LONG_MAX / pack
                    ? embed_hidden * pack : -1;
            }
        }
        if (head_hidden != cfg.hidden || embed_hidden != cfg.hidden ||
            head.dims[0] != embed.dims[0]) {
            fprintf(stderr,
                    "head/embed geometry mismatch: head [%ld,%ld] -> H=%ld, "
                    "embed [%ld,%ld] -> H=%ld, model H=%d\n",
                    head.dims[0], head.dims[1], head_hidden,
                    embed.dims[0], embed.dims[1], embed_hidden, cfg.hidden);
            salt_head_free(&head);
            salt_embed_free(&embed);
            return 1;
        }
        long V = head.dims[0];
        logits = (float *)malloc((size_t)V * sizeof(float));
        if (!logits) return 2;
        /* ---- PERSISTENT SERVE LOOP (--serve-fifo) -----------------
         * The expensive init (trunk bind, expert cache, pool, layout,
         * head/embed/tokenizer) ran ONCE above. This loop re-enters
         * at the prompt parse for each request: read one line of
         * comma-separated pids from the fifo, run prefill+gen, reset
         * the per-request resources, repeat. Non-serve mode falls
         * through exactly as before (byte-identical). */
        FILE *sf = NULL;
        if (serve_fifo) {
            sf = fopen(serve_fifo, "r");
            if (!sf) {
                fprintf(stderr, "serve: cannot open fifo %s\n", serve_fifo);
                return 2;
            }
            fprintf(stderr, "[serve] listening on %s\n", serve_fifo);
        }
serve_next:
       salt_state_transaction_abort(&state_transaction);
       if (gpu_prefill_defer_active) {
           if (salt_gpu_set_defer(gpu_prefill_defer_prior) != 0) {
               fprintf(stderr, "gpu-trunk: prefill defer-policy restore failed\n");
               rc = 2;
           }
           gpu_prefill_defer_active = 0;
       }
       if (gpu_trunk_layer_release(gpu_runtime_trunk, gpu_trunk_intent,
                                   &gpu_trunk_map_active) != 0) {
           fprintf(stderr, "gpu-trunk: synchronized release failed\n");
           gpu_trunk_release_failed = 1;
           rc = 2;
       }
       /* Execution errors are terminal in both one-shot and FIFO modes.
         * A partially updated recurrent state cannot be reused safely.
         * Framing errors that occur before a request starts leave rc == 0
         * and may skip to the next FIFO line. */
        if (rc != 0) goto serve_done;
        if (serve_fifo) {
            /* read one request from the fifo. Line format:
             *   "<gen>:<comma-separated pids>"
             * (gen = tokens to generate for this request; the KV
             * cache is re-sized to npids+gen below). */
            char line[1 << 20];
            if (!fgets(line, sizeof line, sf)) goto serve_done;
            /* free the previous request's pids, parse fresh */
            free(pids); pids = NULL; npids = 0;
            char *body = strchr(line, ':');
            if (!body) {
                fprintf(stderr, "[serve] malformed line (no ':'): %.40s\n",
                        line);
                goto serve_next;
            }
            *body++ = 0;
            int req_gen = atoi(line);
            if (req_gen > 0) gen = req_gen;
            char *save = NULL;
            for (char *tk = strtok_r(body, ", \t\r\n", &save); tk;
                 tk = strtok_r(NULL, ", \t\r\n", &save)) {
                int *np = (int *)realloc(pids,
                    (size_t)(npids + 1) * sizeof(int));
                if (!np) { rc = 2; goto serve_done; }
                pids = np;
                pids[npids++] = atoi(tk);
            }
            if (npids < 1) {
                fprintf(stderr, "[serve] empty prompt line, skipping\n");
                goto serve_next;
            }
            fprintf(stderr, "[serve] prompt %d ids, gen %d\n", npids, gen);
            goto serve_have_pids;
        }
        if (text_arg) {
            /* --text: encode the prompt with the tokenizer */
            if (tok_path == NULL) {
                fprintf(stderr, "--text requires --tokenizer\n");
                rc = 2; goto serve_next;
            }
            int tmp[512];
            int n = salt_tokenizer_encode(&tok, text_arg, tmp, 512);
            if (n <= 0) {
                fprintf(stderr, "prompt encode failed\n");
                rc = 2; goto serve_next;
            }
            pids = (int *)malloc((size_t)n * sizeof(int));
            if (!pids) { rc = 2; goto serve_next; }
            for (int i = 0; i < n; i++) pids[i] = tmp[i];
            npids = n;
            fprintf(stderr, "prompt ids: %d", n);
            for (int i = 0; i < n; i++)
                fprintf(stderr, " %d", pids[i]);
            fprintf(stderr, "\n");
        } else if (prompt_ids) {
            char *dup = strdup(prompt_ids);
            if (!dup) {
                fprintf(stderr, "salt: strdup(prompt_ids) failed\n");
                free(pids);
                rc = 2; goto serve_next;
            }
            char *save = NULL;
            for (char *tok = strtok_r(dup, ",", &save); tok;
                 tok = strtok_r(NULL, ",", &save)) {
                int *np = (int *)realloc(pids,
                    (size_t)(npids + 1) * sizeof(int));
                if (!np) { rc = 2; goto serve_next; }
                pids = np;
                pids[npids++] = atoi(tok);
            }
            free(dup);
        } else if (pids_file) {
            /* --pids-file: comma/space-separated ids from a file (for
             * long prompts that exceed argv limits, e.g. 20K tokens) */
            FILE *pf = fopen(pids_file, "rb");
            if (!pf) {
                fprintf(stderr, "cannot read %s\n", pids_file);
                rc = 2; goto serve_next;
            }
            fseek(pf, 0, SEEK_END);
            long fsz = ftell(pf);
            fseek(pf, 0, SEEK_SET);
            if (fsz < 1 || fsz > (1L << 30)) { fclose(pf); rc = 2; goto serve_next; }
            char *fbuf = (char *)malloc((size_t)fsz + 1);
            if (!fbuf) { fclose(pf); rc = 2; goto serve_next; }
            if (fread(fbuf, 1, (size_t)fsz, pf) != (size_t)fsz) {
                free(fbuf); fclose(pf); rc = 2; goto serve_next;
            }
            fclose(pf);
            fbuf[fsz] = 0;
            char *save = NULL;
            for (char *tok = strtok_r(fbuf, ", \t\r\n", &save); tok;
                 tok = strtok_r(NULL, ", \t\r\n", &save)) {
                int *np = (int *)realloc(pids,
                    (size_t)(npids + 1) * sizeof(int));
                if (!np) { free(fbuf); rc = 2; goto serve_next; }
                pids = np;
                pids[npids++] = atoi(tok);
            }
            free(fbuf);
        }
serve_have_pids:
        if (gpu_trunk_prefill_only) {
            salt_gpu_batch_stats_get(&gpu_request_start_stats);
            gpu_prefill_trunk_start_batches =
                gpu_request_start_stats.trunk_batches;
            gpu_prefill_trunk_batches = 0;
            gpu_prefill_trunk_after_batches =
                gpu_request_start_stats.trunk_batches;
            gpu_prefill_direct_after_batches =
                gpu_request_start_stats.direct_output_batches;
            gpu_prefill_direct_after_jobs =
                gpu_request_start_stats.direct_output_jobs;
            gpu_prefill_gpu_chunks = 0;
            gpu_prefill_stats_captured = 0;
            gpu_prefill_phase_validated = 0;
        }
        text_mode = 1;
        plan.state_b += (double)head.buf_n + (double)embed.buf_n +
                        (double)V * 4.0;
        plan.need_b = plan.trunk_pin_b + plan.trunk_ring_b + plan.cache_b +
                      plan.shared_b + plan.state_b + plan.index_b;
    }
    if (moe_mode && !kv_ok) {
        /* KV capacity: the cache must hold the RESTORED rows (pref
         * from --kv-load, 0 without) PLUS this request's prompt and
         * generated tokens. In serve+resume mode the request pids
         * are request-LOCAL (new tokens only -- the conversation is
         * the KV cache, not the pid list); the load below restores
         * rows [0..pref) and the chunk pass writes [pref..pref+npids).
         * Peek the header so the buffer is sized for the whole
         * session, not just the new slice. In SESSION mode
         * (SALT_SESSION=1) the buffer is sized to the BUDGET
         * (SALT_KV_BUDGET_GB, default 1.0): the KV cache is the
         * memory, the budget is the memory limit, and crossing it
         * triggers the forget policy (drop rows, keep delta state). */
        int kv_pref = 0;
        if (kv_load_path) {
            FILE *kf = fopen(kv_load_path, "rb");
            if (kf) {
                uint64_t hdr[8];
                if (fread(hdr, sizeof hdr, 1, kf) == 1 &&
                    hdr[0] == 0x5354414C544B5631ULL) {
                    if (hdr[1] != (uint64_t)cfg.n_layers ||
                        hdr[2] != (uint64_t)kvlat ||
                        !kv_header_state_matches(hdr, kv_state_vh,
                            kv_state_kd, kv_state_vd, kv_state_conv) ||
                        kv_header_pref(hdr, &kv_pref) != 0) {
                        fprintf(stderr,
                                "kv-load: invalid sizing header in %s\n",
                                kv_load_path);
                        fclose(kf); rc = 2; goto serve_next;
                    }
                }
                fclose(kf);
            }
        }
        /* R5 split load: the MEMORY side carries transcript rows too --
         * its pref must size the buffer the same way (the kv pointer's
         * session extent), or the resumed GQA reads past the cache. */
        if (kv_load_mem) {
            FILE *kf = fopen(kv_load_mem, "rb");
            if (kf) {
                uint64_t hdr[8];
                if (fread(hdr, sizeof hdr, 1, kf) == 1 &&
                    hdr[0] == 0x5354414C544B5631ULL) {
                    int mem_pref;
                    if (hdr[1] != (uint64_t)cfg.n_layers ||
                        hdr[2] != (uint64_t)kvlat ||
                        !kv_header_state_matches(hdr, kv_state_vh,
                            kv_state_kd, kv_state_vd, kv_state_conv) ||
                        kv_header_pref(hdr, &mem_pref) != 0) {
                        fprintf(stderr,
                                "kv-load-memory: invalid sizing header in %s\n",
                                kv_load_mem);
                        fclose(kf); rc = 2; goto serve_next;
                    }
                    if (mem_pref > kv_pref) kv_pref = mem_pref;
                }
                fclose(kf);
            }
        }
        if ((uint64_t)kv_pref + (uint64_t)npids + (uint64_t)gen >
                (uint64_t)INT_MAX) {
            fprintf(stderr, "kv-load: token capacity exceeds INT_MAX\n");
            rc = 2; goto serve_next;
        }
        int kv_max = kv_pref + npids + gen;
        if (getenv("SALT_SESSION")) {
            double budget_gb = 1.0;
            const char *bg = getenv("SALT_KV_BUDGET_GB");
            if (bg && atof(bg) > 0.0) budget_gb = atof(bg);
            long bytes_per_tok = (long)cfg.n_layers * (long)kvlat * 4L;
            int budget_toks = (int)(budget_gb * 1e9 / (double)bytes_per_tok);
            if (budget_toks < npids + gen) budget_toks = npids + gen;
            fprintf(stderr, "[session] KV budget %.2f GB = %d tokens "
                    "(%ld B/tok; load pref %d)\n",
                    budget_gb, budget_toks, bytes_per_tok, kv_pref);
            kv_max = budget_toks;
        }
        if (salt_kv_init(&kvc, cfg.n_layers, kvlat, kv_max) != 0) {
            fprintf(stderr, "moe: kv cache init failed\n");
            rc = 2; goto serve_next;
        }
        kvc.nthreads = threads;  /* attention spawn count: propagate
                                    --threads (fetch pool got it via
                                    salt_cache_init) */
        if (salt_attn_pool_init(&kvc) != 0) {
            fprintf(stderr, "moe: attention pool init failed\n");
            rc = 2; goto serve_next;
        }
        /* Per-token attention work arena. GQA and linear requirements are
         * derived from validated model/tensor geometry by their shared
         * sizing helpers.
         * Compute the bound from the trunk dims and allocate ONCE --
         * never malloc per layer/token (macOS zone high-water). */
        {
            long need = 1024;
            int Hh = cfg.hidden;
            for (int L2 = 0; L2 < cfg.n_layers; L2++) {
                int qi = tl.q3_q[L2], pi = tl.q3_pqkv[L2];
                if (qi >= 0) {
                    int ki = tl.q3_k[L2], vi = tl.q3_v[L2], oi = tl.q3_o[L2];
                    int qrows, krows, vrows, orows;
                    long ocols = oi >= 0
                        ? salt_trunk_tensor_cols(&tl.t[oi]) : -1;
                    long n = ki >= 0 && vi >= 0 && oi >= 0 &&
                        salt_trunk_tensor_rows(&tl.t[qi], &qrows) == 0 &&
                        salt_trunk_tensor_rows(&tl.t[ki], &krows) == 0 &&
                        salt_trunk_tensor_rows(&tl.t[vi], &vrows) == 0 &&
                        salt_trunk_tensor_rows(&tl.t[oi], &orows) == 0 &&
                        ocols > 0 && ocols <= INT_MAX
                        ? salt_attn_gqa_scratch_floats(
                              &cfg, qrows, krows, vrows, orows, (int)ocols,
                              kvc.max_tokens)
                        : -1;
                    if (n < 0) {
                        fprintf(stderr, "invalid GQA scratch geometry\n");
                        rc = 2; goto serve_next;
                    }
                    if (n > need) need = n;
                }
                if (pi >= 0) {
                    int zi = tl.q3_pz[L2], oi = tl.q3_opa[L2];
                    int qkv_rows, z_rows, o_rows;
                    long n = zi >= 0 && oi >= 0 &&
                        salt_trunk_tensor_rows(&tl.t[pi], &qkv_rows) == 0 &&
                        salt_trunk_tensor_rows(&tl.t[zi], &z_rows) == 0 &&
                        salt_trunk_tensor_rows(&tl.t[oi], &o_rows) == 0
                        ? salt_attn_linear_scratch_floats(
                              &cfg, Hh, qkv_rows, z_rows, o_rows, threads)
                        : -1;
                    if (n < 0) {
                        fprintf(stderr, "invalid linear-attention scratch geometry\n");
                        rc = 2; goto serve_next;
                    }
                    if (n > need) need = n;
                }
            }
            if (salt_kv_scratch_init(&kvc, need) != 0) {
                fprintf(stderr, "moe: kv scratch init failed\n");
                rc = 2; goto serve_next;
            }
            fprintf(stderr, "kv scratch: %ld floats (%.1f MB)\n", need,
                    (double)need * 4.0 / 1048576.0);
        }
        kv_ok = 1;
    }

    /* KV persistence load (--kv-load): restore the resume state saved
     * by an earlier --kv-save -- GQA K/V rows [0..prefilled), linear
     * delta states, and the conv ring. The chunk pass then starts at
     * prefilled instead of 0, so only the NEW tokens (the question)
     * are prefilled. Layout remap is free: the higher-level
     * abstraction reads K/V by absolute position and the conv ring by
     * (position % Q3_CONV_K), so restoring the semantic contents at
     * the same absolute positions is all that matters. */
    if (kv_load_path && moe_mode && kv_ok &&
        (getenv("SALT_SESSION") == NULL || kv_prefilled == 0)) {
        if (qwen_state_transaction_begin(&state_control, state_model,
                &cfg, &kvc, kv_prefilled, kvlat,
                SALT_STATE_TRANSACTION_IMPORT, SALT_STATE_ARTIFACT_FULL,
                SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
            fprintf(stderr, "kv-load: engine transaction rejected\n");
            rc = 2; goto serve_next;
        }
        FILE *kf = fopen(kv_load_path, "rb");
        if (!kf) {
            fprintf(stderr, "kv-load: cannot open %s\n", kv_load_path);
            rc = 2; goto serve_next;
        }
        uint64_t hdr[8];
        if (fread(hdr, sizeof hdr, 1, kf) != 1 ||
            hdr[0] != 0x5354414C544B5631ULL) {
            fprintf(stderr, "kv-load: bad header in %s\n", kv_load_path);
            fclose(kf); rc = 2; goto serve_next;
        }
        int pref = 0;
        /* request-local pids: the restore target is pref (rows
         * [0..pref) already computed), and this request's pids are
         * NEW tokens at [pref..pref+npids). The only real constraint
         * is that the KV buffer (sized pref+npids+gen above) holds
         * them. */
        if (hdr[1] != (uint64_t)cfg.n_layers ||
            hdr[2] != (uint64_t)kvlat ||
            !kv_header_state_matches(hdr, kv_state_vh, kv_state_kd,
                                     kv_state_vd, kv_state_conv) ||
            kv_header_pref(hdr, &pref) != 0 ||
            (getenv("SALT_SESSION") == NULL &&
             (uint64_t)pref + (uint64_t)npids + (uint64_t)gen >
                 (uint64_t)kvc.max_tokens)) {
            fprintf(stderr, "kv-load: dim mismatch (layers %llu/%d kvlat "
                    "%llu/%d pref %llu npids %d max %d)\n",
                    (unsigned long long)hdr[1], cfg.n_layers,
                    (unsigned long long)hdr[2], kvlat,
                    (unsigned long long)hdr[3], npids,
                    kvc.max_tokens);
            fclose(kf); rc = 2; goto serve_next;
        }
        int load_rows = pref;
        if (getenv("SALT_SESSION") &&
            (size_t)pref > (size_t)kvc.max_tokens) {
            /* session budget smaller than the saved transcript:
             * recover the DELTA STATE (the compressed memory), skip
             * the rows (transcript forgotten at load). */
            fprintf(stderr, "[session] cache %d rows > budget %d: "
                    "restoring delta state only (transcript "
                    "forgotten at load)\n", pref, kvc.max_tokens);
            load_rows = 0;
            kv_prefilled = 0;
        }
        /* Lazily allocate only the model-owned recurrent geometry. */
        if (kv_state_vh > 0 &&
            (salt_kv_lin_init(&kvc, kv_state_vh, kv_state_kd,
                              kv_state_vd) != 0 ||
             salt_kv_conv_init(&kvc, kv_state_conv) != 0)) {
            fprintf(stderr, "kv-load: lin/conv init failed\n");
            fclose(kf); rc = 2; goto serve_next;
        }
        /* per-layer restore: the in-memory stride is max_tokens, not
         * pref -- read each layer's rows [0..load_rows) from the
         * compact file layout and place them at L*max_tokens*kvlat.
         * When load_rows == 0 (session budget skip), the rows are
         * skipped with fseek so lin/conv still read at the right
         * file offset. */
        {
            size_t row = (size_t)kvlat * sizeof(float);
            if (load_rows == 0) {
                if (fseek(kf, (long)((size_t)cfg.n_layers * (size_t)pref *
                                     row), SEEK_CUR) != 0) {
                    fprintf(stderr, "kv-load: fseek rows failed\n");
                    fclose(kf); rc = 2; goto serve_next;
                }
            } else {
            for (int L = 0; L < cfg.n_layers; L++) {
                float *base = kvc.kv +
                    (size_t)L * (size_t)kvc.max_tokens * (size_t)kvlat;
                if (fread(base, row, (size_t)load_rows, kf) !=
                    (size_t)load_rows) {
                    fprintf(stderr, "kv-load: short kv read at L%d\n", L);
                    fclose(kf); rc = 2; goto serve_next;
                }
            }
            }
        }
        if (kv_state_vh > 0 && !kvc.lin_alloc &&
            salt_kv_lin_init(&kvc, kv_state_vh, kv_state_kd,
                             kv_state_vd) != 0) {
            fprintf(stderr, "kv-load: lin init failed\n");
            fclose(kf); rc = 2; goto serve_next;
        }
        if (kvc.lin_alloc) {
            size_t lnn = (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                         (size_t)kvc.lin_kd * (size_t)kvc.lin_vd;
            if (fread(kvc.lin, sizeof(float), lnn, kf) != lnn) {
                fprintf(stderr, "kv-load: short lin read\n");
                fclose(kf); rc = 2; goto serve_next;
            }
        }
        if (kv_state_conv > 0 && !kvc.conv_alloc &&
            salt_kv_conv_init(&kvc, kv_state_conv) != 0) {
            fprintf(stderr, "kv-load: conv init failed\n");
            fclose(kf); rc = 2; goto serve_next;
        }
        if (kvc.conv_alloc) {
            size_t cn = (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows;
            if (fread(kvc.conv, sizeof(float), cn, kf) != cn) {
                fprintf(stderr, "kv-load: short conv read\n");
                fclose(kf); rc = 2; goto serve_next;
            }
        }
        fclose(kf);
        kv_prefilled = load_rows;
        if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                kv_prefilled, kvlat, &state_transaction) != 0) {
            fprintf(stderr, "kv-load: engine transaction failed\n");
            rc = 2; goto serve_next;
        }
        fprintf(stderr, "[kv-load] %s (%d/%d rows restored, resuming "
                "prefill at %d)\n", kv_load_path, load_rows, pref,
                load_rows);
    }
    /* R5 split load -- MINDSET only: the lin delta + conv ring
     * (the mental model). The KV rows are left untouched (the
     * memory side is loaded/owned separately). The lin/conv are
     * lazily allocated by the first linear-attention token, so
     * init them here from the header dims if needed. */
    if (kv_load_mind && moe_mode && kv_ok) {
        if (qwen_state_transaction_begin(&state_control, state_model,
                &cfg, &kvc, kv_prefilled, kvlat,
                SALT_STATE_TRANSACTION_IMPORT, SALT_STATE_ARTIFACT_MINDSET,
                SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
            fprintf(stderr, "kv-load-mindset: engine transaction rejected\n");
            rc = 2; goto serve_next;
        }
        FILE *kf = fopen(kv_load_mind, "rb");
        if (!kf) {
            fprintf(stderr, "kv-load-mindset: cannot open %s\n",
                    kv_load_mind);
            rc = 2; goto serve_next;
        }
        uint64_t hdr[8];
        if (fread(hdr, sizeof hdr, 1, kf) != 1 ||
            hdr[0] != 0x5354414C544B5631ULL) {
            fprintf(stderr, "kv-load-mindset: bad header in %s\n",
                    kv_load_mind);
            fclose(kf); rc = 2; goto serve_next;
        }
        if ((uint64_t)cfg.n_layers != hdr[1] ||
            !kv_header_state_matches(hdr, kv_state_vh, kv_state_kd,
                                     kv_state_vd, kv_state_conv)) {
            fprintf(stderr, "kv-load-mindset: shape mismatch %s\n",
                    kv_load_mind);
            fclose(kf); rc = 2; goto serve_next;
        }
        if (kv_state_vh > 0 && !kvc.lin_alloc &&
            salt_kv_lin_init(&kvc, kv_state_vh, kv_state_kd,
                             kv_state_vd) != 0) {
            fprintf(stderr, "kv-load-mindset: lin init failed\n");
            fclose(kf); rc = 2; goto serve_next;
        }
        if (kvc.lin_alloc) {
            size_t lnn = (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                         (size_t)kvc.lin_kd * (size_t)kvc.lin_vd;
            if (fread(kvc.lin, sizeof(float), lnn, kf) != lnn) {
                fprintf(stderr, "kv-load-mindset: short lin read\n");
                fclose(kf); rc = 2; goto serve_next;
            }
        }
        if (kv_state_conv > 0 && !kvc.conv_alloc &&
            salt_kv_conv_init(&kvc, kv_state_conv) != 0) {
            fprintf(stderr, "kv-load-mindset: conv init failed\n");
            fclose(kf); rc = 2; goto serve_next;
        }
        if (kvc.conv_alloc) {
            size_t cn = (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows;
            if (fread(kvc.conv, sizeof(float), cn, kf) != cn) {
                fprintf(stderr, "kv-load-mindset: short conv read\n");
                fclose(kf); rc = 2; goto serve_next;
            }
        }
        fclose(kf);
        if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                kv_prefilled, kvlat, &state_transaction) != 0) {
            fprintf(stderr, "kv-load-mindset: engine transaction failed\n");
            rc = 2; goto serve_next;
        }
        fprintf(stderr, "[kv-load-mindset] %s (lin+conv restored, "
                "memory untouched)\n", kv_load_mind);
    }
    /* R5 split load -- MEMORY only: the GQA K/V rows (the episodic
     * transcript). The lin delta + conv ring are left untouched. */
    if (kv_load_mem && moe_mode && kv_ok) {
        if (qwen_state_transaction_begin(&state_control, state_model,
                &cfg, &kvc, kv_prefilled, kvlat,
                SALT_STATE_TRANSACTION_IMPORT, SALT_STATE_ARTIFACT_FACTS,
                SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
            fprintf(stderr, "kv-load-memory: engine transaction rejected\n");
            rc = 2; goto serve_next;
        }
        FILE *kf = fopen(kv_load_mem, "rb");
        if (!kf) {
            fprintf(stderr, "kv-load-memory: cannot open %s\n",
                    kv_load_mem);
            rc = 2; goto serve_next;
        }
        uint64_t hdr[8];
        if (fread(hdr, sizeof hdr, 1, kf) != 1 ||
            hdr[0] != 0x5354414C544B5631ULL) {
            fprintf(stderr, "kv-load-memory: bad header in %s\n",
                    kv_load_mem);
            fclose(kf); rc = 2; goto serve_next;
        }
        if ((uint64_t)cfg.n_layers != hdr[1] ||
            (uint64_t)kvlat != hdr[2] ||
            !kv_header_state_matches(hdr, kv_state_vh, kv_state_kd,
                                     kv_state_vd, kv_state_conv)) {
            fprintf(stderr, "kv-load-memory: shape mismatch %s\n",
                    kv_load_mem);
            fclose(kf); rc = 2; goto serve_next;
        }
        int pref = 0;
        if (kv_header_pref(hdr, &pref) != 0 || pref > kvc.max_tokens) {
            fprintf(stderr, "kv-load-memory: pref %llu > max %d\n",
                    (unsigned long long)hdr[3], kvc.max_tokens);
            fclose(kf); rc = 2; goto serve_next;
        }
        size_t row = (size_t)kvlat * sizeof(float);
        for (int L = 0; L < cfg.n_layers; L++) {
            const float *base = kvc.kv +
                (size_t)L * (size_t)kvc.max_tokens * (size_t)kvlat;
            if (fread((void *)base, row, (size_t)pref, kf) != (size_t)pref) {
                fprintf(stderr, "kv-load-memory: short kv read at L%d\n", L);
                fclose(kf); rc = 2; goto serve_next;
            }
        }
        fclose(kf);
        if (pref > kv_prefilled)
            kv_prefilled = pref;
        if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                kv_prefilled, kvlat, &state_transaction) != 0) {
            fprintf(stderr, "kv-load-memory: engine transaction failed\n");
            rc = 2; goto serve_next;
        }
        fprintf(stderr, "[kv-load-memory] %s (%d rows restored, "
                "mindset untouched)\n", kv_load_mem, pref);
    }

    /* ---- session memory budget (SALT_SESSION=1) --------------------
     * The KV cache is the memory; the budget is its limit. When this
     * request would push the session past the buffer, apply the
     * FORGET policy: drop the exact transcript rows + conv ring
     * (salt_kv_forget), KEEP the linear delta states (the compressed
     * memory), and restart rows at 0. The model keeps its
     * understanding; it loses exact recall of older tokens. The last
     * checkpoint on disk (--kv-save-after) still holds the full
     * session, so the forget is reversible by reloading it. */
    if (getenv("SALT_SESSION") && kv_ok &&
        (size_t)kv_prefilled + (size_t)npids + (size_t)gen >
            (size_t)kvc.max_tokens) {
        if (!kv_save_after)
            fprintf(stderr, "[session] WARNING: no --kv-save-after "
                    "checkpoint -- forgetting is irreversible\n");
        fprintf(stderr, "[session] memory full at %d+%d+%d > %d rows: "
                "FORGETTING %d transcript rows, delta state kept\n",
                kv_prefilled, npids, gen, kvc.max_tokens, kv_prefilled);
        salt_kv_forget(&kvc);
        kv_prefilled = 0;
    }

    if (moe_mode) {
        /* mHC: the residual stream is n_hc x H (n_hc from the first
         * hc fn tensor's column count / hidden); 1 without hc. */
        int nstreams = 1;
        if (tl.hc_attn_fn[0] >= 0 &&
            tl.t[tl.hc_attn_fn[0]].rank == 2 &&
            tl.t[tl.hc_attn_fn[0]].dims[1] % cfg.hidden == 0)
            nstreams = (int)(tl.t[tl.hc_attn_fn[0]].dims[1] /
                             cfg.hidden);
        state = (float *)malloc((size_t)cfg.hidden * nstreams *
                                sizeof(float));
        if (!state) { rc = 2; goto serve_next; }
        if (text_mode) {
            if (npids > 0) {
                if (salt_embed_gather(&embed, pids[0], state) != 0) {
                    fprintf(stderr, "embedding token out of range: %d\n",
                            pids[0]);
                    rc = 2; goto serve_next;
                }
                for (int j = 1; j < nstreams; j++)
                    memcpy(state + (size_t)j * cfg.hidden, state,
                           (size_t)cfg.hidden * sizeof(float));
            } else {
                for (int i = 0; i < cfg.hidden * nstreams; i++)
                    state[i] = 0.0f;
            }
        } else {
            for (int i = 0; i < cfg.hidden; i++) {
                uint64_t h = salt_mix64(hstate ^ salt_mix64((uint64_t)i));
                state[i] = (float)((double)((int64_t)(h % 1000)) / 100.0 -
                                   5.0);
            }
            for (int j = 1; j < nstreams; j++)
                memcpy(state + (size_t)j * cfg.hidden, state,
                       (size_t)cfg.hidden * sizeof(float));
        }
        mhc_streams = nstreams;
    }

    int idx[64];
    float w[64];
    int slots[64];
    float scores[256];
    const uint8_t *es[64];
    /* zero the whole stack arrays: topk fills only [0..topk), and an
     * uninitialized read past it is the flaky nondeterminism (values
     * vary with ASLR; -O2 reads them at ~3-25% of runs). */
    memset(idx, 0, sizeof idx);
    memset(w, 0, sizeof w);
    memset(slots, 0, sizeof slots);
    memset(scores, 0, sizeof scores);
    memset(es, 0, sizeof es);
    /* mHC layer-input buffer for the ffn router (H floats) */
    xin_buf = (float *)malloc((size_t)cfg.hidden * sizeof(float));
    if (!xin_buf) { rc = 2; goto serve_next; }
    /* dbg10 fix: the token-to-token delta needs per-layer prev states
     * (the single prev_state compared adjacent layers -- the
     * layer-transition magnitude, not the token's movement) */
    prev_state = (float *)malloc(
        (size_t)cfg.hidden * mhc_streams * SALT_MAX_LAYERS *
        sizeof(float));
    if (!prev_state) { rc = 2; goto serve_next; }
    prev_hin = (float *)malloc(
        (size_t)cfg.hidden * sizeof(float));
    if (!prev_hin) { rc = 2; goto serve_next; }
    int last_tok = npids > 0 ? pids[0] : -1;
    /* repetition penalty: SALT_REP_PENALTY (multiplicative, HF-style).
     * Recent generated tokens get logits[i] /= penalty before sampling
     * so greedy/temp can't lock into "... X. X. X. ..." loops. */
    int recent_toks[64], n_recent = 0;
    float rep_pen = 0.0f;
    {
        const char *rp = getenv("SALT_REP_PENALTY");
        if (rp && *rp) rep_pen = (float)atof(rp);
    }

    double t0 = now_s();
    SaltMemSnapshot memory_run_start = memory_snapshot();
    SaltMemSnapshot memory_decode_entry = memory_run_start;
    SaltMemSnapshot memory_final = memory_run_start;
    SaltMemSnapshot memory_sampled_peak = memory_run_start;
    const char *roofp = getenv("SALT_ROOFLINE");
    int roofline_on = roofp && *roofp && *roofp != '0';
    RoofSnap roof_start = {0}, roof_decode = {0}, roof_first = {0};
    int roof_decode_seen = 0, roof_first_seen = 0;
    if (roofline_on) roof_start = roof_snap(&cache);
    const char *mpp = getenv("SALT_MOE_PROFILE");
    int moe_profile_on = mpp && *mpp && *mpp != '0';
    if (moe_profile_on) salt_moe_batch_profile_reset();
    /* ---- CHUNKED PREFILL (M1 of prefill-batch, opt-in) ------------
     * SALT_PREFILL_CHUNK=1: process the prompt layer-major in chunks
     * of B tokens. The qkv+z projections batch across the chunk (the
     * dequant amortizes); the conv ring + delta state stay serial in
     * token order via lin_body. The trunk is bound ONCE per layer per
     * chunk (vs once per token), so the layer weights amortize Bx.
     * The head is skipped for prompt tokens (logits are never sampled
     * there -- generation re-embeds from last_tok). Bit-identical KV
     * cache fill vs the serial path. */
    const char *pc = getenv("SALT_PREFILL_CHUNK");
    if (gpu_trunk_prefill_only) {
        gpu_prefill_defer_prior = salt_gpu_defer();
        gpu_prefill_defer_active = 1;
    }
    if (gpu_trunk_prefill_only &&
        !(pc && *pc == '1' && text_mode && npids > 1 && moe_mode))
        fprintf(stderr, "gpu-trunk: no eligible chunked prefill; using CPU\n");
    if (pc && *pc == '1' && text_mode && npids > 1 && moe_mode) {
        int CHUNK = 64;
        int gpu_prefill_min_b = 1024;
        const char *ce = getenv("SALT_PREFILL_B");
        if (ce) {
            /* discrete batch size: the CPU path's chunk. The cap is
             * 4096 (the GPU proj-batch can take the bigger chunks --
             * SALT_GPU_PREFILL_B). At CPU 512 is the measured sweet
             * spot (the tiled batch plateaus past it). */
            int v = atoi(ce);
            if (v >= 1 && v <= SALT_PREFILL_BATCH_MAX) CHUNK = v;
        }
        {
            const char *tb = getenv("SALT_GPU_TRUNK_B");
            int v = tb ? atoi(tb) : 0;
            if (v > 0) gpu_prefill_min_b = v;
        }
        /* Grouped-expert crossover depends on the model's active
         * parameter/expert geometry. The selected model blueprint must
         * provide SALT_M2_BATCH_B; there is no universal engine default. */
        int m2_batch_min_b = 0;
        const char *m2be = getenv("SALT_M2_BATCH_B");
        if (m2be) {
            int v = atoi(m2be);
            if (v >= 1 && v <= 4096) m2_batch_min_b = v;
        }
        const char *m2e = getenv("SALT_M2_BATCH");
        int m2_batch_requested = m2e && *m2e && *m2e != '0';
        int m2_batch_on = salt_moe_batch_should_run(
            m2_batch_requested, m2_batch_min_b, m2_batch_min_b);
        fprintf(stderr, "[prefill] chunked prompt pass, B=%d npids=%d%s\n",
                CHUNK, npids, kv_prefilled > 0 ? " (resumed)" : "");
        if (m2_batch_on)
            fprintf(stderr, "[prefill] grouped MoE batch for actual B >= %d\n",
                    m2_batch_min_b);
        else if (m2_batch_requested)
            fprintf(stderr, "[prefill] grouped MoE disabled: "
                    "SALT_M2_BATCH_B is not configured for this model\n");
        for (int c0 = kv_prefilled; c0 < kv_prefilled + npids; c0 += CHUNK) {
            int B = (c0 + CHUNK < kv_prefilled + npids) ? CHUNK
                    : (kv_prefilled + npids) - c0;
            int gpu_chunk_enabled = gpu_trunk_intent;
            if (gpu_trunk_prefill_only) {
                gpu_chunk_enabled = B >= 4 && B >= gpu_prefill_min_b;
                if (gpu_chunk_enabled) {
                    gpu_prefill_gpu_chunks++;
                } else if (gpu_trunk_map_active &&
                           gpu_trunk_layer_release(&gpu_prefill_trunk, 1,
                                                   &gpu_trunk_map_active) != 0) {
                    fprintf(stderr, "gpu-trunk: CPU-tail transition failed\n");
                    gpu_trunk_release_failed = 1;
                    rc = 2; goto serve_next;
                }
            }
            /* Keep legacy singleton routing disabled for CPU-owned tail chunks. */
            salt_attn_set_batch_b(gpu_chunk_enabled ? B : 1);
            /* per-token states: [B][H * mhc_streams] */
            long stn = (long)cfg.hidden * mhc_streams;
            float *pstates = (float *)malloc((size_t)B * (size_t)stn *
                                             sizeof(float));
            
            float *xin_b = (float *)malloc((size_t)cfg.hidden *
                                           sizeof(float));
            /* batched gate scores: [B][n_experts] */
            float *scores_b = (float *)malloc(
                (size_t)B * (size_t)cfg.n_experts * sizeof(float));
            float *ps_arr[SALT_PREFILL_BATCH_MAX];
            const float *shared_states[SALT_PREFILL_BATCH_MAX];
            if (!pstates || !xin_b || !scores_b) {
                fprintf(stderr, "prefill chunk alloc failed\n");
                rc = 2; goto serve_next;
            }
            for (int b = 0; b < B; b++) {
                int pid = pids[c0 - kv_prefilled + b];
                if (salt_embed_gather(&embed, pid,
                                      pstates + (size_t)b * stn) != 0) {
                    fprintf(stderr, "embedding token out of range: %d\n",
                            pid);
                    free(pstates); free(xin_b); free(scores_b);
                    rc = 2; goto serve_next;
                }
                for (int j = 1; j < mhc_streams; j++)
                    memcpy(pstates + (size_t)b * stn + (size_t)j * cfg.hidden,
                           pstates + (size_t)b * stn,
                           (size_t)cfg.hidden * sizeof(float));
                ps_arr[b] = pstates + (size_t)b * stn;
                shared_states[b] = ps_arr[b];
            }
            /* layer-major: one trunk pass per chunk */
            SaltTrunk *prefill_trunk =
                gpu_trunk_prefill_only && gpu_chunk_enabled
                    ? &gpu_prefill_trunk : &trunk;
            salt_trunk_rewind(prefill_trunk);
            double _temb = now_s(), _tbind = 0.0, _tgqa = 0.0, _tattn = 0.0;
            double _trfm = 0.0, _troute = 0.0, _tpfetch = 0.0, _tpmoe = 0.0;
            for (int L = 0; L < cfg.n_layers; L++) {
                if (gpu_chunk_enabled || gpu_moe_intent)
                    salt_gpu_set_defer(1);   /* the layer's proj commits */
                double _tb0 = now_s();
                const uint8_t *tr = runtime_trunk_bind(
                    prefill_trunk, L, gpu_chunk_enabled,
                    &gpu_trunk_map_active,
                    &gpu_trunk_release_failed);
                _tbind += now_s() - _tb0;
                if (!tr) {
                    fprintf(stderr, gpu_trunk_release_failed
                            ? "gpu-trunk: release failed before L%d\n"
                            : "trunk bind failed at L%d\n", L);
                    rc = 2;
                    goto serve_next;
                }
                int use_real = tl.gate[L] >= 0;
                /* attention: linear layers batched, gqa serial */
                if (use_real && kv_ok && !getenv("SALT_SKIP_ATTN")) {
                    /* the uniform walk: the registry's kind picks the
                     * chunked phase -- LIN batched, GQA batched-proj +
                     * serial body, anything else serial. */
                    SaltLayerAttnKind kind =
                        L < n_layers_reg ? layers[L].attn : LAYER_NONE;
                    double _ta0 = now_s();
                    if (kind == LAYER_LIN) {
                        /* batch the linear layer over the chunk */
                        if (salt_attn_linear_chunk(&cfg, &tl, L, tr, ps_arr,
                                              c0, B, &kvc) != 0) {
                            fprintf(stderr, "linear_step_chunk failed at L%d\n", L);
                            rc = 2; goto serve_next;
                        }
                        if (getenv("SALT_PROJ_MS") && L == 0 && c0 == 512) {
                            FILE *pa = fopen("/tmp/postlin-L0c1.bin", "wb");
                            if (pa) {
                                for (int b = 0; b < B; b++)
                                    fwrite(ps_arr[b], sizeof(float),
                                           (size_t)cfg.hidden, pa);
                                fclose(pa);
                            }
                        }
                        /* bisect: L0 linear-chunk OUTPUT (pre-MoE) for
                         * the first chunk -- is the token-0 divergence
                         * already present before the MoE? */
                        if (getenv("SALT_NAN_PROBE") && L == 0 && c0 == 0) {
                            FILE *pa = fopen("/tmp/q36-pre-moe-L0-c0.bin", "wb");
                            if (pa) {
                                fwrite(pstates, sizeof(float),
                                       (size_t)B * stn, pa);
                                fclose(pa);
                            }
                        }
                    } else if (kind == LAYER_GQA) {
                        /* M3-lite: batch the GQA projections (q/k/v +
                         * o_proj); the softmax body stays serial */
                        double _tg0 = now_s();
                        if (salt_attn_gqa_chunk(&cfg, &tl, L, tr, ps_arr,
                                           c0, B, &kvc) != 0) {
                            if (gpu_chunk_enabled) {
                                fprintf(stderr,
                                        "gpu-trunk: GQA chunk failed at L%d\n",
                                        L);
                                rc = 2;
                                goto serve_next;
                            }
                            /* fallback: serial per token */
                            for (int b = 0; b < B; b++) {
                                float *st = pstates + (size_t)b * stn;
                                if (salt_attn_qwen_step(&cfg, &tl, L, tr, st,
                                                        &kvc, c0 + b) != 0) {
                                    rc = 2;
                                    goto serve_next;
                                }
                            }
                        }
                        if (getenv("SALT_NAN_PROBE") && L == 3) {
                            double sm = 0.0;
                            for (int i = 0; i < cfg.hidden; i++)
                                sm += (double)pstates[8 * stn + i] *
                                      pstates[8 * stn + i];
                            fprintf(stderr, "[chk-gqa] L3 t8 post-gqa rms "
                                    "%.6g\n", sqrt(sm / cfg.hidden));
                        }
                        _tgqa += now_s() - _tg0;
                    } else {
                        for (int b = 0; b < B; b++) {
                            float *st = pstates + (size_t)b * stn;
                            if (salt_attn_step(&cfg, &tl, L, tr, st,
                                               &kvc, c0 + b) != 0) {
                                rc = 2;
                                goto serve_next;
                            }
                        }
                    }
                    _tattn += now_s() - _ta0;
                }
                /* M2 batched rfm measured SLOWER than serial at B=64
                 * (249-330ms/layer vs 134-167ms: routing diversity
                 * gives ~2-4 tokens/expert, so the dequant savings
                 * lose to the gather + spawn overhead). Reverted to
                 * the per-token loop; salt_moe_step_batch stays as a
                 * -1-fallback for future larger-B work.
                 * The UNION-FETCH part of M2 is kept here though:
                 * the B*topk expert lookups per layer dedupe to
                 * ~130 unique ids, fetched in ONE getmany instead of
                 * 64 -- cut redundant disk reads + per-call overhead
                 * without touching the moe math (bit-fidelity-
                 * neutral: fetch only controls resident bytes). */
                double _tf0 = now_s();
                double _tr_acc = 0, _tfe_acc = 0, _tm_acc = 0;
                /* union fetch: dedupe the B*topk expert ids per layer
                 * (~130 unique of 512) into ONE getmany (chunked at
                 * 64), cutting redundant disk reads + per-call
                 * overhead. Bit-fidelity-neutral: fetch only decides
                 * which bytes are resident, never the math. */
                int uniq_n = 0;
                int emap[256];
                size_t nsel = (size_t)B * cfg.topk;
                int *tok_slot = (int *)malloc(nsel * sizeof(int));
                int *idx_all = (int *)malloc(nsel * sizeof(int));
                float *w_all = (float *)malloc(nsel * sizeof(float));
                if (!tok_slot || !idx_all || !w_all) {
                    free(tok_slot); free(idx_all); free(w_all);
                    rc = 2; goto serve_next;
                }
                memset(emap, 0xFF, sizeof emap);
                for (int b = 0; b < B; b++) {
                    double _tr0 = now_s();
                    float *st = pstates + (size_t)b * stn;
                    if (use_real) {
                        const SaltTrunkTensor *gt = &tl.t[tl.gate[L]];
                        const float *gbias = NULL;
                        if (tl.gate_bias[L] >= 0) {
                            const SaltTrunkTensor *bt = &tl.t[tl.gate_bias[L]];
                            gbias = (const float *)(const void *)(tr + bt->off);
                        }
                        const float *rstate = st;
                        if (tl.ffn_norm[L] >= 0 && !getenv("SALT_NO_NORMS")) {
                            memcpy(xin_b, st, (size_t)cfg.hidden * sizeof(float));
                            double sn = 0.0;
                            for (int i = 0; i < cfg.hidden; i++)
                                sn += (double)xin_b[i] * xin_b[i];
                            float rn = sqrtf((float)(sn / (double)cfg.hidden) + 1e-6f);
                            const uint16_t *pnw = (const uint16_t *)(const void *)(
                                tr + tl.t[tl.ffn_norm[L]].off);
                            for (int i = 0; i < cfg.hidden; i++) {
                                uint32_t pb2 = (uint32_t)pnw[i] << 16;
                                float pw;
                                memcpy(&pw, &pb2, 4);
                                xin_b[i] = xin_b[i] / rn * pw;
                            }
                            rstate = xin_b;
                        }
                        if (gt->dtype == 5) {
                            const SaltTrunkTensor *gsc = NULL, *gbs = NULL;
                            for (int qi = tl.t_off[L]; qi < tl.t_off[L + 1]; qi++) {
                                if (name_ends(tl.t[qi].name, ".mlp.gate.scales"))
                                    gsc = &tl.t[qi];
                                else if (name_ends(tl.t[qi].name, ".mlp.gate.biases"))
                                    gbs = &tl.t[qi];
                            }
                            long pcp = gt->dims[1];
                            if (gt->bits == 8)
                                salt_q8_matvec(
                                    (const uint32_t *)(const void *)(tr + gt->off),
                                    gsc ? (const uint16_t *)(const void *)(tr + gsc->off) : NULL,
                                    gbs ? (const uint16_t *)(const void *)(tr + gbs->off) : NULL,
                                    (int)gt->dims[0],
                                    (int)(pcp * (32 / gt->bits)), rstate, scores_b);
                            else
                                salt_q4_matvec(
                                    (const uint32_t *)(const void *)(tr + gt->off),
                                    gsc ? (const uint16_t *)(const void *)(tr + gsc->off) : NULL,
                                    gbs ? (const uint16_t *)(const void *)(tr + gbs->off) : NULL,
                                    (int)gt->dims[0],
                                    (int)(pcp * (32 / gt->bits)), rstate, scores_b);
                        } else if (gt->dtype == 4)
                            salt_bf16_matvec(
                                (const uint16_t *)(const void *)(tr + gt->off),
                                cfg.n_experts, cfg.hidden, rstate, gbias, scores_b);
                        else
                            salt_router_scores(
                                (const float *)(const void *)(tr + gt->off), gbias,
                                cfg.n_experts, cfg.hidden, rstate, scores_b);
                        salt_topk(scores_b, cfg.n_experts, cfg.topk, idx, w);
                    } else {
                        uint64_t hstate = salt_mix64(0);
                        hstate = salt_mix64(hstate ^ salt_checksum(tr, trunk.lay[L].nbytes));
                        salt_router(idx, w, &cfg, hstate, L, locality);
                    }
                    _tr_acc += now_s() - _tr0;
                    /* store per-token routing + dedupe into the union */
                    for (int j = 0; j < cfg.topk; j++) {
                        int e = idx[j];
                        idx_all[b * cfg.topk + j] = e;
                        w_all[b * cfg.topk + j] = w[j];
                        if (emap[e] < 0) emap[e] = uniq_n++;
                        tok_slot[b * cfg.topk + j] = emap[e];
                    }
                }
                /* one union fetch (chunked at 64 per getmany call) */
                {
                    int *uslot = (int *)malloc((size_t)uniq_n * sizeof(int));
                    if (!uslot) { rc = 2; goto serve_next; }
                    memset(uslot, 0xFF, (size_t)uniq_n * sizeof(int));
                    double _tfe0 = now_s();
                    for (int u0 = 0; u0 < uniq_n; u0 += 64) {
                        int un = uniq_n - u0 < 64 ? uniq_n - u0 : 64;
                        int uexp[64];
                        int usl[64];
                        for (int i = 0; i < un; i++)
                            uexp[i] = -1;   /* filled below via emap */
                        /* rebuild the unique id list: scan emap */
                        for (int e = 0, cnt = 0;
                             e < cfg.n_experts && cnt < uniq_n; e++)
                            if (emap[e] >= u0 && emap[e] < u0 + un)
                                uexp[emap[e] - u0] = e;
                        if (salt_cache_getmany(&cache, L, uexp, un, usl) < 0) {
                            fprintf(stderr,
                                    "expert cache fetch failed at prefill L%d\n", L);
                            free(uslot); free(tok_slot); free(idx_all); free(w_all);
                            rc = 2;
                            goto serve_next;
                        }
                        for (int i = 0; i < un; i++)
                            uslot[u0 + i] = usl[i];
                    }
                    /* A later chunked getmany() may recycle a slot from an
                     * earlier chunk when capacity is constrained. Never
                     * consume a retained raw index without proving that it
                     * still names the requested key. */
                    for (int e = 0; e < cfg.n_experts; e++) {
                        if (emap[e] < 0) continue;
                        if (!salt_cache_slot_for(&cache, uslot[emap[e]], L, e)) {
                            fprintf(stderr,
                                    "expert cache handle stale at prefill L%d E%d\n",
                                    L, e);
                            free(uslot); free(tok_slot); free(idx_all); free(w_all);
                            rc = 2;
                            goto serve_next;
                        }
                    }
                    _tfe_acc += now_s() - _tfe0;
                    /* moe pass: per token, map union slots -> es.
                     * M6: the shared expert (dense, same weights for
                     * every token) is batched first via the
                     * bit-identical kernel; moe_step folds each
                     * token's sout/sgate into acc at the serial
                     * point (residual order unchanged).
                     * M2 re-test (SALT_M2_BATCH=1): route the whole
                     * layer through salt_moe_step_batch -- grouped
                     * routed experts + internal shared batch. The
                     * old rejection (2x slower at B=64) predates
                     * M4-M6; re-measured on the new baseline. */
                    double _tm0 = now_s();
                    if (use_real) {
                        if (salt_moe_batch_should_run(
                                m2_batch_requested, m2_batch_min_b, B) &&
                            !gpu_moe_intent) {
                            /* column-major normed xins[H][B] (M2
                             * layout; it does NOT norm internally) */
                            float *xcol_m = (float *)malloc(
                                (size_t)cfg.hidden * (size_t)B *
                                sizeof(float));
                            int *sel_f = (int *)malloc(
                                (size_t)B * cfg.topk * sizeof(int));
                            const uint8_t **es_f = (const uint8_t **)
                                malloc((size_t)B * cfg.topk *
                                       sizeof(uint8_t *));
                            if (xcol_m && sel_f && es_f) {
                                const uint16_t *pnw = NULL;
                                if (tl.ffn_norm[L] >= 0 &&
                                    !getenv("SALT_NO_NORMS"))
                                    pnw = (const uint16_t *)(const void *)(
                                        tr + tl.t[tl.ffn_norm[L]].off);
                                for (int b = 0; b < B; b++) {
                                    const float *st = pstates +
                                        (size_t)b * stn;
                                    for (int i = 0; i < cfg.hidden; i++)
                                        xcol_m[(size_t)i * B + b] = st[i];
                                    if (pnw) {
                                        double ss = 0.0;
                                        const float *sr = pstates +
                                            (size_t)b * stn;
                                        for (int i = 0; i < cfg.hidden; i++)
                                            ss += (double)sr[i] * sr[i];
                                        float r = sqrtf((float)
                                            (ss / (double)cfg.hidden) +
                                            1e-6f);
                                        for (int i = 0; i < cfg.hidden;
                                             i++) {
                                            uint32_t bits =
                                                (uint32_t)pnw[i] << 16;
                                            float bv;
                                            memcpy(&bv, &bits, 4);
                                            xcol_m[(size_t)i * B + b] =
                                                sr[i] / r * bv;
                                        }
                                    }
                                    for (int j = 0; j < cfg.topk; j++) {
                                        sel_f[(size_t)b * cfg.topk + j] =
                                            idx_all[b * cfg.topk + j];
                                        int expert =
                                            idx_all[b * cfg.topk + j];
                                        es_f[(size_t)b * cfg.topk + j] =
                                            salt_cache_slot_for(
                                                &cache,
                                                uslot[tok_slot[
                                                    b * cfg.topk + j]],
                                                L, expert);
                                    }
                                }
                                int batch_rc = salt_moe_step_batch(
                                    &cfg, &tl, L, tr, &pl, xcol_m, B,
                                    cfg.topk, sel_f, w_all, es_f,
                                    (float *const *)ps_arr, &n_matvec,
                                    &n_decode);
                                free(xcol_m); free(sel_f); free(es_f);
                                if (batch_rc != 0) {
                                    fprintf(stderr,
                                            "[m2] batch failed at L%d\n", L);
                                    rc = 2; goto serve_next;
                                }
                            } else {
                                free(xcol_m); free(sel_f); free(es_f);
                                rc = 2; goto serve_next;
                            }
                        } else {
                        float *sb = NULL, *sg2 = NULL;
                        int shared_failed = 0;
                        int shared_present =
                            tl.se_g[L] >= 0 && tl.se_u[L] >= 0 &&
                            tl.se_d[L] >= 0 && tl.se_gs[L] >= 0 &&
                            tl.se_us[L] >= 0 && tl.se_ds[L] >= 0 &&
                            tl.se_r[L] >= 0 && tl.se_rs[L] >= 0 &&
                            tl.se_rb[L] >= 0;
                        int skip_shared = getenv("SALT_SKIP_SHARED") &&
                            *getenv("SALT_SKIP_SHARED") != '0';
                        sb = (float *)malloc((size_t)B *
                                             (size_t)cfg.hidden *
                                             sizeof(float));
                        sg2 = (float *)malloc((size_t)B * sizeof(float));
                        if (!skip_shared && shared_present &&
                            (!sb || !sg2 ||
                             salt_moe_shared_batch(&cfg, &tl, L, tr,
                                                   shared_states, B,
                                                   sb, sg2) != 0))
                            shared_failed = 1;
                        if (shared_failed) {
                            free(sb); free(sg2);
                            sb = NULL; sg2 = NULL;
                            if (gpu_chunk_enabled) {
                                fprintf(stderr,
                                    "[shared] mapped trunk failed at L%d\n",
                                    L);
                                rc = 2; goto serve_next;
                            }
                        }
                        if (getenv("SALT_NAN_PROBE") && L == 0 && c0 == 0) {
                            FILE *psf = fopen("/tmp/q36-shared-L0-c0.bin", "wb");
                            if (psf) {
                                fwrite(sb, sizeof(float),
                                       (size_t)B * cfg.hidden, psf);
                                fclose(psf);
                            }
                            FILE *pgf = fopen("/tmp/q36-sharedgate-L0-c0.bin", "wb");
                            if (pgf) {
                                fwrite(sg2, sizeof(float), (size_t)B, pgf);
                                fclose(pgf);
                            }
                        }
                        for (int b = 0; b < B; b++) {
                            float *st = pstates + (size_t)b * stn;
                            int *tid = idx_all + (size_t)b * cfg.topk;
                            float *tw = w_all + (size_t)b * cfg.topk;
                            for (int j = 0; j < cfg.topk; j++)
                                es[j] = salt_cache_slot_for(
                                    &cache,
                                    uslot[tok_slot[b * cfg.topk + j]],
                                    L, tid[j]);
                            if (getenv("SALT_PROJ_MS") && L == 0 && c0 == 512
                                && b == 0) {
                                FILE *pe = fopen("/tmp/es-L0c1.bin", "wb");
                                if (pe) {
                                    for (int j = 0; j < cfg.topk; j++) {
                                        fwrite(&es[j], sizeof(void *), 1, pe);
                                        if (es[j])
                                            fwrite(es[j], 1, 4096, pe);
                                        else {
                                            static const char zz[4096];
                                            fwrite(zz, 1, 4096, pe);
                                        }
                                    }
                                    fclose(pe);
                                }
                                FILE *pr = fopen("/tmp/route-L0c1.bin", "wb");
                                if (pr) {
                                    fwrite(tid, sizeof(int),
                                           (size_t)cfg.topk, pr);
                                    fwrite(tw, sizeof(float),
                                           (size_t)cfg.topk, pr);
                                    fclose(pr);
                                }
                            }
                            if (getenv("SALT_SKIP_ROUTED") &&
                                *getenv("SALT_SKIP_ROUTED") != '0') {
                                /* diagnostic: drop the routed-expert
                                 * contribution entirely (shared expert
                                 * still runs) -- tests whether the
                                 * routed MoE matters for the output. */
                            } else {
                                int gpu_prefill_guard =
                                    gpu_moe_intent;
                                int prior_mapped_only =
                                    salt_gpu_mapped_only();
                                int moe_rc;
                                if (gpu_prefill_guard)
                                    salt_gpu_set_mapped_only(1);
                                moe_rc = salt_moe_step(
                                    &cfg, &tl, L, tr, &pl, es, tid, tw, st,
                                    scratch, scratch_n, jscratch, &n_matvec,
                                    &n_decode,
                                    sb ? sb + (size_t)b * cfg.hidden : NULL,
                                    sb ? sg2[b] : 0.0f);
                                if (gpu_prefill_guard)
                                    salt_gpu_set_mapped_only(
                                        prior_mapped_only);
                                if (moe_rc != 0) {
                                    fprintf(stderr,
                                            "moe step failed at L%d\n", L);
                                    free(sb); free(sg2);
                                    rc = 2; goto serve_next;
                                }
                            }
                        }
                        free(sb); free(sg2);
                        }
                    }
                    _tm_acc += now_s() - _tm0;
                    /* SWEEP: dump the LAST token's state after every
                     * layer (SALT_LAYER_SWEEP=1) -- Q1/Q2 comparison
                     * pinpoints where input-dependence dies. */
                    if (getenv("SALT_LAYER_SWEEP") && B > 1) {
                        char pth[128];
                        snprintf(pth, sizeof pth,
                                 "/tmp/q36-last-L%02d.bin", L);
                        FILE *sf = fopen(pth, "wb");
                        if (sf) {
                            fwrite(pstates + (size_t)(B - 1) * stn,
                                   sizeof(float), (size_t)cfg.hidden, sf);
                            fclose(sf);
                        }
                    }
                    if (getenv("SALT_ALL_STATES") && B > 1 &&
                        (L == 0 || L == 1)) {
                        char pth[128];
                        snprintf(pth, sizeof pth,
                                 "/tmp/q36-all-L%02d.bin", L);
                        FILE *sf = fopen(pth, "wb");
                        if (sf) {
                            fwrite(pstates, sizeof(float),
                                   (size_t)B * stn, sf);
                            fclose(sf);
                        }
                    }
                    free(uslot);   /* after the moe pass: uslot maps
                                      union expert -> cache slot for
                                      the per-token moe (lifetime must
                                      cover the whole chunk delivery) */
                }
                if (getenv("SALT_PROJ_MS") && L == 0 && c0 == 512) {
                    FILE *pm = fopen("/tmp/postmoe-L0c1.bin", "wb");
                    if (pm) {
                        for (int b = 0; b < B; b++)
                            fwrite(ps_arr[b], sizeof(float),
                                   (size_t)cfg.hidden, pm);
                        fclose(pm);
                    }
                }
                free(tok_slot); free(idx_all); free(w_all);
                double _rfm_wall = now_s() - _tf0;
                _trfm += _rfm_wall;
                _troute += _tr_acc;
                _tpfetch += _tfe_acc;
                _tpmoe += _tm_acc;
                if (getenv("SALT_CHUNK_MS")) {
                    fprintf(stderr, "[chunk-moe] L%d B=%d rfm %.2f ms "
                            "(route %.2f fetch %.2f moe %.2f)\n", L, B,
                            _rfm_wall * 1e3,
                            _tr_acc * 1e3, _tfe_acc * 1e3, _tm_acc * 1e3);
                }
                /* NAN_PROBE state dump at token 8 (matches the serial
                 * path's /tmp/q36-eng-L%d-t8.bin) so the chunk fill
                 * can be diffed layer by layer vs the serial fill. */
                if (getenv("SALT_NAN_PROBE") && c0 <= 8 && c0 + B > 8) {
                    int tb = 8 - c0;
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L%d-t8.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(pstates + (size_t)tb * stn, sizeof(float),
                               (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                /* all tokens at L<4 (the GQA coupling at L3 reads the
                 * past tokens' K cache -- need per-token diffs) */
                if (getenv("SALT_NAN_PROBE") && L < 4) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L%d-t%d.bin",
                             L, c0);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(pstates, sizeof(float),
                               (size_t)B * stn, sf);
                        fclose(sf);
                    }
                }
                /* the layer's wait: the deferred proj commits sync
                 * here (the chunk's proj group: 1 wait per layer). */
                if (salt_gpu_sync() != 0) {
                    fprintf(stderr, "gpu-trunk: prefill completion failed at L%d\n",
                            L);
                    rc = 2;
                    goto serve_done;
                }
            }
            /* KV DUMP: after the chunk, dump the GQA layers' KV
             * slices (kvc.kv [L][max_tokens][kvlat]) -- Q1/Q2
             * comparison shows whether the question tokens' K/V
             * differ (write path) or not (read/attention path). */
            if (getenv("SALT_KV_DUMP") && c0 == 0) {
                for (int L2 = 0; L2 < cfg.n_layers; L2++) {
                    if (tl.q3_q[L2] < 0) continue;   /* linear layer */
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-kv-L%02d.bin", L2);
                    FILE *kf = fopen(pth, "wb");
                    if (kf) {
                        fwrite(kvc.kv + (size_t)L2 * kvc.max_tokens *
                               kvc.kvlat, sizeof(float),
                               (size_t)kvc.max_tokens * kvc.kvlat, kf);
                        fclose(kf);
                    }
                }
            }
            if (wf_on) {
                wf_pbind += _tbind;
                wf_pattn += _tattn;   /* all attention; GQA is nested */
                wf_pgqa += _tgqa;
                wf_prfm += _trfm;
                wf_proute += _troute;
                wf_pfetch += _tpfetch;
                wf_pmoe += _tpmoe;
            }
            if (getenv("SALT_CHUNK_MS"))
                fprintf(stderr, "[chunk-sum] c%d B=%d bind %.2fs lin %.2fs "
                        "gqa %.2fs route %.2fs fetch %.2fs moe %.2fs "
                        "rfm-glue %.2fs other %.2fs\n", c0 / CHUNK, B,
                        _tbind, _tattn - _tgqa, _tgqa,
                        _troute, _tpfetch, _tpmoe,
                        _trfm - _troute - _tpfetch - _tpmoe,
                        now_s() - _temb - _tbind - _tattn - _trfm);
            /* save the LAST prompt token's post-prompt state: the
             * serial path captures last_tok from its logits (the
             * first gen token's prediction); the chunk pass skipped
             * the head, so we need the state to run it once below. */
            if (c0 + B >= npids) {
                int lb = npids - 1 - c0;
                if (lb >= 0 && lb < B)
                    memcpy(state, pstates + (size_t)lb * stn,
                           (size_t)stn * sizeof(float));
            }
            {
                SaltMemSnapshot memory_now = memory_snapshot();
                memory_sample_peak(&memory_sampled_peak, &memory_now);
                int breached = memory_limit_breached(
                    &memory_now, mem_limit_gb, "prefill chunk",
                    c0 / CHUNK + 1, (npids + CHUNK - 1) / CHUNK);
                free(pstates); free(xin_b); free(scores_b);
                if (breached) {
                    if (trf) { fclose(trf); trf = NULL; }
                    rc = 3;
                    goto serve_done;
                }
            }
        }
        fprintf(stderr, "[prefill] chunked pass done, starting gen\n");
        if (getenv("SALT_PROJ_MS")) {
            FILE *sf = fopen("/tmp/postmoe-L39.bin", "wb");
            if (sf) {
                fwrite(state, sizeof(float),
                       (size_t)cfg.hidden * (size_t)mhc_streams, sf);
                fclose(sf);
            }
        }
        /* KV persistence (--kv-save): serialize the whole resume
         * state -- GQA K/V rows [0..npids), linear delta states, and
         * the conv ring. A later --kv-load restores these and starts
         * the chunk pass at npids, skipping the prefill entirely
         * (doc-QA: pay 135s once, reuse in ~0.2s from storage). */
        if (kv_save_path) {
            int export_position = kv_prefilled + npids;
            if (qwen_state_transaction_begin(&state_control, state_model,
                    &cfg, &kvc, export_position, kvlat,
                    SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_FULL,
                    SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
                fprintf(stderr, "kv-save: engine transaction rejected\n");
                rc = 2; goto serve_next;
            }
            FILE *kf = fopen(kv_save_path, "wb");
            if (!kf) {
                fprintf(stderr, "kv-save: cannot open %s\n", kv_save_path);
                rc = 2; goto serve_next;
            }
            uint64_t hdr[8];
            hdr[0] = 0x5354414C544B5631ULL;          /* magic SALTKV01 */
            hdr[1] = (uint64_t)cfg.n_layers;
            hdr[2] = (uint64_t)kvlat;
            hdr[3] = (uint64_t)(kv_prefilled + npids); /* prefilled count */
            hdr[4] = (uint64_t)kvc.lin_vh;
            hdr[5] = (uint64_t)kvc.lin_kd;
            hdr[6] = (uint64_t)kvc.lin_vd;
            hdr[7] = (uint64_t)kvc.conv_rows;
            if (fwrite(hdr, sizeof hdr, 1, kf) != 1) {
                fprintf(stderr, "kv-save: header write failed\n");
                fclose(kf); rc = 2; goto serve_next;
            }
            /* kv region: per-layer rows [0..pref), unpacked from the
             * in-memory [L][max_tokens][kvlat] stride. Writing the
             * raw buffer contiguously would shift every layer after 0
             * by (max_tokens - pref) rows -- the layer stride in
             * memory is max_tokens, NOT pref. */
            size_t row = (size_t)kvlat * sizeof(float);
            for (int L = 0; L < cfg.n_layers; L++) {
                const float *base = kvc.kv +
                    (size_t)L * (size_t)kvc.max_tokens * (size_t)kvlat;
                if (fwrite(base, row, (size_t)(kv_prefilled + npids), kf) !=
                    (size_t)(kv_prefilled + npids)) {
                    fprintf(stderr, "kv-save: kv write failed at L%d\n", L);
                    fclose(kf); rc = 2; goto serve_next;
                }
            }
            if ((kvc.lin_alloc &&
                 fwrite(kvc.lin, sizeof(float),
                        (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                        (size_t)kvc.lin_kd * (size_t)kvc.lin_vd, kf) !=
                    (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                        (size_t)kvc.lin_kd * (size_t)kvc.lin_vd) ||
                (kvc.conv_alloc &&
                 fwrite(kvc.conv, sizeof(float),
                        (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows,
                        kf) !=
                    (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows)) {
                fprintf(stderr, "kv-save: lin/conv write failed\n");
                fclose(kf);
                rc = 2; goto serve_next;
            }
            fclose(kf);
            if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                    export_position, kvlat, &state_transaction) != 0) {
                fprintf(stderr, "kv-save: engine transaction failed\n");
                rc = 2; goto serve_next;
            }
            fprintf(stderr, "[kv-save] %s (%d tokens, kvlat %d)\n",
                    kv_save_path, npids, kvlat);
        }
        /* R5 split save/load -- the two-state persistence surface:
         * the MINDSET (lin delta + conv ring: the mental model, the
         * ICL -- fixed size, survives forget) and the MEMORY (GQA
         * K/V rows: the episodic transcript -- row-addressable).
         * Same SALTKV01 header (the shape info); the file SIZE
         * determines which region is present (kv-only vs lin/conv). */
        if (kv_save_mind) {
            int export_position = kv_prefilled + npids;
            if (qwen_state_transaction_begin(&state_control, state_model,
                    &cfg, &kvc, export_position, kvlat,
                    SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_MINDSET,
                    SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
                fprintf(stderr, "kv-save-mindset: engine transaction rejected\n");
                rc = 2; goto serve_next;
            }
            FILE *kf = fopen(kv_save_mind, "wb");
            if (!kf) {
                fprintf(stderr, "kv-save-mindset: cannot open %s\n",
                        kv_save_mind);
                rc = 2; goto serve_next;
            }
            uint64_t hdr[8];
            hdr[0] = 0x5354414C544B5631ULL;
            hdr[1] = (uint64_t)cfg.n_layers;
            hdr[2] = (uint64_t)kvlat;
            hdr[3] = (uint64_t)(kv_prefilled + npids);
            hdr[4] = (uint64_t)kvc.lin_vh;
            hdr[5] = (uint64_t)kvc.lin_kd;
            hdr[6] = (uint64_t)kvc.lin_vd;
            hdr[7] = (uint64_t)kvc.conv_rows;
            fwrite(hdr, sizeof hdr, 1, kf);
            if ((kvc.lin_alloc &&
                 fwrite(kvc.lin, sizeof(float),
                        (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                        (size_t)kvc.lin_kd * (size_t)kvc.lin_vd, kf) !=
                    (size_t)cfg.n_layers * (size_t)kvc.lin_vh *
                        (size_t)kvc.lin_kd * (size_t)kvc.lin_vd) ||
                (kvc.conv_alloc &&
                 fwrite(kvc.conv, sizeof(float),
                        (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows,
                        kf) !=
                    (size_t)cfg.n_layers * 4 * (size_t)kvc.conv_rows)) {
                fprintf(stderr, "kv-save-mindset: lin/conv write failed\n");
                fclose(kf); rc = 2; goto serve_next;
            }
            fclose(kf);
            if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                    export_position, kvlat, &state_transaction) != 0) {
                fprintf(stderr, "kv-save-mindset: engine transaction failed\n");
                rc = 2; goto serve_next;
            }
            fprintf(stderr, "[kv-save-mindset] %s (lin+conv only)\n",
                    kv_save_mind);
        }
        if (kv_save_mem) {
            int export_position = kv_prefilled + npids;
            if (qwen_state_transaction_begin(&state_control, state_model,
                    &cfg, &kvc, export_position, kvlat,
                    SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_FACTS,
                    SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
                fprintf(stderr, "kv-save-memory: engine transaction rejected\n");
                rc = 2; goto serve_next;
            }
            FILE *kf = fopen(kv_save_mem, "wb");
            if (!kf) {
                fprintf(stderr, "kv-save-memory: cannot open %s\n",
                        kv_save_mem);
                rc = 2; goto serve_next;
            }
            uint64_t hdr[8];
            hdr[0] = 0x5354414C544B5631ULL;
            hdr[1] = (uint64_t)cfg.n_layers;
            hdr[2] = (uint64_t)kvlat;
            hdr[3] = (uint64_t)(kv_prefilled + npids);
            hdr[4] = (uint64_t)kvc.lin_vh;
            hdr[5] = (uint64_t)kvc.lin_kd;
            hdr[6] = (uint64_t)kvc.lin_vd;
            hdr[7] = (uint64_t)kvc.conv_rows;
            fwrite(hdr, sizeof hdr, 1, kf);
            size_t row = (size_t)kvlat * sizeof(float);
            for (int L = 0; L < cfg.n_layers; L++) {
                const float *base = kvc.kv +
                    (size_t)L * (size_t)kvc.max_tokens * (size_t)kvlat;
                if (fwrite(base, row, (size_t)(kv_prefilled + npids), kf) !=
                    (size_t)(kv_prefilled + npids)) {
                    fprintf(stderr, "kv-save-memory: kv write failed at L%d\n",
                            L);
                    fclose(kf); rc = 2; goto serve_next;
                }
            }
            fclose(kf);
            if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                    export_position, kvlat, &state_transaction) != 0) {
                fprintf(stderr, "kv-save-memory: engine transaction failed\n");
                rc = 2; goto serve_next;
            }
            fprintf(stderr, "[kv-save-memory] %s (%d rows)\n",
                    kv_save_mem, kv_prefilled + npids);
        }
        /* capture the FIRST gen token's prediction from the last
         * prompt token's logits -- the serial path does this at
         * t == npids-1 (main.c:1298). The chunk pass saved the last
         * prompt state into `state`; run the head once here so
         * last_tok is the model's prediction, not pids[0].
         * The final-norm tensor lives in the synthetic last trunk
         * layer (L = n_layers-1); the chunk pass consumed 0..39 so
         * the reader has streamed through 40 -- bind directly. DO
         * NOT rewind here: that resets consumed/ready and bind(40)
         * deadlocks (the reader stalls at consumed+window with
         * nring=2 before the main ever consumes 0..39 again). */
        {
            const float *hstate_in = state;
            if (tl.final_norm >= 0) {
                /* Final norm (BF16 [hidden]) lives in the synthetic
                 * last trunk layer. The ring stream is sized/raced
                 * around the layer pass; pread the 4 KB tensor
                 * directly instead of bind()ing layer n_layers-1
                 * (which deadlocks when the reader's window stalls
                 * at consumed+1 with nring=2). */
                const SaltTrunkTensor *fn = &tl.t[tl.final_norm];
                static uint16_t *fnw = NULL;
                static long fnw_cap = 0;
                if (fn->nbytes > fnw_cap) {
                    uint16_t *nb = (uint16_t *)realloc(
                        fnw, (size_t)fn->nbytes);
                    if (!nb) { rc = 2; goto serve_next; }
                    fnw = nb; fnw_cap = fn->nbytes;
                }
                const SaltTrunkLayer *fnl =
                    &trunk.lay[tl.n_layers - 1];
                if (pread(trunk.fd, fnw, (size_t)fn->nbytes,
                          fnl->off + fn->off) != fn->nbytes) {
                    fprintf(stderr, "final-norm pread failed\n");
                    rc = 2; goto serve_next;
                }
                double ss = 0.0;
                for (int i = 0; i < cfg.hidden; i++)
                    ss += (double)hstate_in[i] * hstate_in[i];
                float r = sqrtf((float)(ss / (double)cfg.hidden) + 1e-6f);
                for (int i = 0; i < cfg.hidden; i++) {
                    uint32_t bits = (uint32_t)fnw[i] << 16;
                    float w;
                    memcpy(&w, &bits, 4);
                    xin_buf[i] = hstate_in[i] / r * w;
                }
                hstate_in = xin_buf;
            }
            if (salt_head_logits(&head, hstate_in, logits) != 0) {
                fprintf(stderr, "head logits failed (chunk last-tok)\n");
                rc = 2; goto serve_next;
            }
            if (getenv("SALT_GREEDY"))
                last_tok = salt_argmax(logits, (int)head.dims[0]);
            else
                last_tok = salt_sample(logits, (int)head.dims[0], &rng);
        }
    }
    if (gpu_trunk_prefill_only) {
        SaltGpuBatchStats phase_stats;
        if (gpu_trunk_layer_release(&gpu_prefill_trunk, 1,
                                    &gpu_trunk_map_active) != 0) {
            fprintf(stderr, "gpu-trunk: prefill/decode boundary release "
                    "failed\n");
            gpu_trunk_release_failed = 1;
            rc = 2; goto serve_next;
        }
        if (salt_gpu_set_defer(gpu_prefill_defer_prior) != 0) {
            fprintf(stderr, "gpu-trunk: prefill/decode defer-policy restore "
                    "failed\n");
            rc = 2; goto serve_next;
        }
        gpu_prefill_defer_active = 0;
        salt_gpu_batch_stats_get(&phase_stats);
        if (phase_stats.trunk_batches < gpu_prefill_trunk_start_batches) {
            fprintf(stderr, "gpu-trunk: batch counter moved backwards\n");
            rc = 2; goto serve_next;
        }
        gpu_prefill_trunk_batches = phase_stats.trunk_batches -
            gpu_prefill_trunk_start_batches;
        gpu_prefill_trunk_after_batches = phase_stats.trunk_batches;
        gpu_prefill_direct_after_batches = phase_stats.direct_output_batches;
        gpu_prefill_direct_after_jobs = phase_stats.direct_output_jobs;
        gpu_prefill_stats_captured = 1;
        fprintf(stderr, "[gpu-phase] prefill complete chunks=%d trunk=%llu "
                "direct=%llu/%llu arena=%llu mapped-misses=%llu; decode=cpu\n",
                gpu_prefill_gpu_chunks,
                (unsigned long long)gpu_prefill_trunk_batches,
                (unsigned long long)(phase_stats.direct_output_batches -
                    gpu_request_start_stats.direct_output_batches),
                (unsigned long long)(phase_stats.direct_output_jobs -
                    gpu_request_start_stats.direct_output_jobs),
                (unsigned long long)phase_stats.arena_batches,
                (unsigned long long)phase_stats.mapped_only_misses);
    }
    /* prompt pass + generation: the first npids iterations feed the
     * prompt tokens (no sampling, no output) so the KV/state caches
     * see the whole context; the next gen iterations generate.
     * When the chunked prefill ran, it already filled the KV cache
     * for the whole prompt -- the loop starts at the first
     * GENERATION position (absolute: kv_prefilled + npids) so the
     * prompt is NOT re-processed serially. In resume mode the
     * restored rows live at [0..kv_prefilled) and this request's
     * prompt at [kv_prefilled..kv_prefilled+npids). */
    int t_start = kv_prefilled;   /* skip restored rows (0 when fresh) */
    if (pc && *pc == '1' && text_mode && npids > 1 && moe_mode)
        t_start = kv_prefilled + npids;   /* chunked prefill did it */
    int total_toks = kv_prefilled + npids + gen;
    int gen_done = gen;          /* actual generated rows (EOS may cut) */
    /* SALT_STRIP_THINK state (opt-in). The wrapper's chat framing
     * PREPENDS <think> (248068) as prompt tokens, so the block is
     * already open before generation starts -- seed in_think from
     * the prompt's LAST think-tag (open or closed). Otherwise the
     * whole reasoning trace leaks to stdout. */
    int in_think = 0;
    int max_think = 0, think_count = 0;
    /* the reasoning tokens come from the model registry -- never
     * hardcoded in the core (the qwen36's 248068/248069). */
    {
        const char *mname = getenv("SALT_MODEL");
        const SaltModelDesc *md0 = salt_model_get(
            mname && *mname ? mname : "qwen36");
        int tok_open = md0 ? md0->think_open : 0;
        int tok_close = md0 ? md0->think_close : 0;
        const char *mt = getenv("SALT_MAX_THINK");
        if (mt && atoi(mt) > 0) max_think = atoi(mt);
        for (int i = npids - 1; i >= 0; i--) {
            if (tok_open && pids[i] == tok_open) { in_think = 1; break; }
            if (tok_close && pids[i] == tok_close) break;
        }
        if (getenv("SALT_STRIP_THINK") &&
            strcmp(getenv("SALT_STRIP_THINK"), "0") != 0 && in_think)
            fprintf(stderr, "[strip] think block open at gen start\n");
    }
    /* startup timing: everything before decode entry
     * (process init, trunk bind, cache alloc, layout load, warmup) */
    memory_decode_entry = memory_snapshot();
    memory_sample_peak(&memory_sampled_peak, &memory_decode_entry);
    if (memory_limit_breached(&memory_decode_entry, mem_limit_gb,
                              "decode entry", 0, gen)) {
        if (trf) { fclose(trf); trf = NULL; }
        rc = 3;
        goto serve_done;
    }
    fprintf(stderr, "[startup] %.3f s to decode entry (t=%d)\n",
            now_s() - t0, t_start);
    for (int t = t_start; t < total_toks; t++) {
        salt_attn_set_batch_b(1);   /* the gen: B=1 -- the CPU path */
        int gen_t = t - (kv_prefilled + npids); /* >= 0 past prompt */
        if (moe_profile_on && gen_t == 0)
            salt_moe_decode_profile_reset();
        if (roofline_on && gen_t == 0 && !roof_decode_seen) {
            roof_decode = roof_snap(&cache);
            roof_decode_seen = 1;
        }
        if (step_ms) st_wall = now_s();
        if (getenv("SALT_TIME_LAYERS")) {
            fprintf(stderr, "[toktime] t=%d start %.3fs\n", t,
                    now_s() - t0);
        }
        /* Graceful hard limit: sample the current application footprint at
         * each token boundary. On breach
         * stop cleanly with exit 3 (memory limit) -- never let the OS
         * OOM-kill mid-stream, and never mask it as a normal run. */
        {
            SaltMemSnapshot memory_now = memory_snapshot();
            memory_sample_peak(&memory_sampled_peak, &memory_now);
            if (memory_limit_breached(&memory_now, mem_limit_gb,
                                      "token start", gen_t, gen)) {
                    if (trf) { fclose(trf); trf = NULL; }
                    if (dump_path && moe_mode) {
                        FILE *df = fopen(dump_path, "wb");
                        if (df) {
                            fwrite(state, sizeof(float),
                                   (size_t)cfg.hidden * (size_t)mhc_streams,
                                   df);
                            fclose(df);
                        }
                    }
                    rc = 3;
                    goto serve_done;
            }
        }
        if (text_mode && gen_t >= 0) {
            if (salt_embed_gather(&embed, last_tok, state) != 0) {
                fprintf(stderr, "embedding token out of range: %d\n",
                        last_tok);
                rc = 2; goto serve_next;
            }
            for (int j = 1; j < mhc_streams; j++)
                memcpy(state + (size_t)j * cfg.hidden, state,
                       (size_t)cfg.hidden * sizeof(float));
            if (getenv("SALT_NAN_PROBE") && t == 8) {
                FILE *ef = fopen("/tmp/q36-eng-embed-t8.bin", "wb");
                if (ef) {
                    fwrite(state, sizeof(float), (size_t)cfg.hidden, ef);
                    fclose(ef);
                }
                fprintf(stderr, "[emb8] tok %d rms %.6g\n", last_tok,
                        sqrt((double)cfg.hidden) > 0
                            ? 0.0 : 0.0);
                double em = 0.0;
                for (int i = 0; i < cfg.hidden; i++)
                    em += (double)state[i] * state[i];
                fprintf(stderr, "[emb8] tok %d rms %.6g\n", last_tok,
                        sqrt(em / cfg.hidden));
            }
        } else if (text_mode) {
            /* resume mode: pids are request-local, absolute position
             * is kv_prefilled + t' (t' = t - kv_prefilled) */
            int pid = pids[t - kv_prefilled];
            if (salt_embed_gather(&embed, pid, state) != 0) {
                fprintf(stderr, "embedding token out of range: %d\n", pid);
                rc = 2; goto serve_next;
            }
            if (getenv("SALT_NAN_PROBE") && t == 0) {
                fprintf(stderr, "[emb] t=%d npids=%d pids[0..5]=%d %d %d %d %d %d "
                        "gathered=%d\n", t, npids, pids[0], pids[1], pids[2],
                        pids[3], pids[4], pids[5], pids[t - kv_prefilled]);
                double em = 0.0;
                for (int i = 0; i < cfg.hidden; i++)
                    em += (double)state[i] * state[i];
                fprintf(stderr, "[emb] rms %.6g tok %d\n",
                        sqrt(em / cfg.hidden), pids[t - kv_prefilled]);
                if (getenv("SALT_DUMP_EMBED")) {
                    FILE *ef = fopen("/tmp/q36-eng-embed.bin", "wb");
                    if (ef) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, ef);
                        fclose(ef);
                    }
                }
            }
        }
        if (moe_mode) {
            /* multi-token generation re-streams the trunk once per
             * token: restart the reader's pass before the first bind */
            salt_trunk_rewind(&trunk);
        }
        for (int L = 0; L < cfg.n_layers; L++) {
            if (gpu_moe_intent ||
                (gpu_trunk_intent && !gpu_trunk_prefill_only))
                salt_gpu_set_defer(1);   /* the layer's proj commits */
            double _l0 = getenv("SALT_LAYER_MS") ? now_s() : 0;
            double _l2 = 0;
            if (getenv("SALT_TIME_LAYERS") && t == 0 &&
                (L % 4 == 0 || L == cfg.n_layers - 1))
                fprintf(stderr, "[L] t0 L%d at %.3fs\n", L, now_s() - t0);
            if (getenv("SALT_NAN_PROBE")) {
                int bad = 0;
                for (int i = 0; i < cfg.hidden; i++)
                    if (state[i] != state[i]) { bad = 1; break; }
                if (bad)
                    fprintf(stderr, "[nan] t%d before L%d\n", t, L);
            }
            if (moe_mode && t == 0 && L < 3)
                fprintf(stderr, "moe: token 0 layer %d\n", L);
            const uint8_t *tr = runtime_trunk_bind(
                &trunk, L, gpu_trunk_intent && !gpu_trunk_prefill_only,
                &gpu_trunk_map_active,
                &gpu_trunk_release_failed);
            double _l1 = getenv("SALT_LAYER_MS") ? now_s() : 0;
            if (getenv("SALT_NAN_PROBE") && L >= 38 && t == 0) {
                fprintf(stderr, "[gbin] t0 L%d bind done tr=%p\n",
                        L, (const void *)tr);
                fflush(stderr);
            }
            if (!tr) {
                fprintf(stderr, gpu_trunk_release_failed
                        ? "gpu-trunk: release failed before layer %d\n"
                        : "trunk bind failed at layer %d\n", L);
                rc = 2;
                goto serve_next;
            }

            int use_real = moe_mode && tl.gate[L] >= 0;
            int overlap = !(gpu_trunk_intent && !gpu_trunk_prefill_only) &&
                          getenv("SALT_GPU_OVERLAP") &&
                          *getenv("SALT_GPU_OVERLAP") != '0';
            /* MLA attention first: it reads/writes state, and the
             * router below sees the post-attention state (real order) */
            if (use_real && kv_ok && !getenv("SALT_SKIP_ATTN")) {
                /* the uniform walk: the registry's kind dispatches the
                 * phase. GQA/LIN run the qwen two-phase (proj/body);
                 * anything else falls to the generic attention. The
                 * q3_* branch chain is gone from the engine's loop. */
                SaltLayerAttnKind kind =
                    L < n_layers_reg ? layers[L].attn : LAYER_NONE;
                int attn_rc;
                double _ta = now_s();
                if (kind == LAYER_GQA && overlap && gen_t > 0) {
                    /* the pipelined schedule: this layer's proj ran
                     * already (prefetched while the previous layer's
                     * moe ran -- the state snapshot completed before
                     * that moe's fold); the body runs now on the
                     * prefetched q/k/v's. */
                    attn_rc = salt_attn_qwen_body(&cfg, &tl, L, tr, state,
                                                  &kvc, t);
                } else if (kind == LAYER_GQA || kind == LAYER_LIN)
                    attn_rc = salt_attn_qwen_step(&cfg, &tl, L, tr, state,
                                                  &kvc, t);
                else
                    attn_rc = salt_attn_step(&cfg, &tl, L, tr, state,
                                              &kvc, t);
                if (wf_on) wf_attn += now_s() - _ta;
                if (step_ms) st_attn += now_s() - _ta;
                _l2 = getenv("SALT_LAYER_MS") ? now_s() : 0;
                if (attn_rc != 0) {
                    fprintf(stderr, "attn step failed at layer %d\n", L);
                    rc = 2; goto serve_next;
                }
                if (getenv("SALT_PROJ_MS") && gen_t == 0 &&
                    (L == 0 || L == 39)) {
                    char pth[64];
                    snprintf(pth, sizeof pth, "/tmp/dcd-attnL%d.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float),
                               (size_t)cfg.hidden * mhc_streams, sf);
                        fclose(sf);
                    }
                }
            }
            if (getenv("SALT_DEBUG2")) {
                uint64_t ck = salt_mix64(0);
                for (int i = 0; i < cfg.hidden; i++) {
                    uint32_t bits;
                    memcpy(&bits, &state[i], 4);
                    ck = salt_mix64(ck ^ bits);
                }
                fprintf(stderr, "[dbg2] t%d L%d after attn %016llx\n", t, L,
                        (unsigned long long)ck);
            }
            if (getenv("SALT_DEBUG3")) {
                uint32_t u0, u1;
                memcpy(&u0, &state[0], 4);
                memcpy(&u1, &state[1], 4);
                fprintf(stderr, "[dbg3] t%d L%d attn state[0..1] %08x %08x\n",
                        t, L, u0, u1);
            }
            if (getenv("SALT_DEBUG6")) {
                double s2 = 0.0;
                long n = (long)cfg.hidden * mhc_streams;
                for (long i = 0; i < n; i++)
                    s2 += (double)state[i] * state[i];
                fprintf(stderr, "[dbg6] t%d L%d rms=%.6g after attn\n",
                        t, L, sqrt(s2 / (double)n));
            }
            if (getenv("SALT_LAYER_SWEEP") && t == npids - 1 &&
                !(getenv("SALT_PREFILL_CHUNK") &&
                  *getenv("SALT_PREFILL_CHUNK") == '1')) {
                /* serial-prefill sweep: last prompt token per-layer */
                char pth[128];
                snprintf(pth, sizeof pth, "/tmp/q36-serial-L%02d.bin", L);
                FILE *sf = fopen(pth, "wb");
                if (sf) {
                    fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                    fclose(sf);
                }
            }
            if (getenv("SALT_DEBUG10")) {
                /* the token-information trace: the state's delta from
                 * the PREVIOUS TOKEN at the SAME layer -- where the
                 * embed's difference dies (the frozen direction) */
                const float *pv = prev_state +
                    (size_t)L * cfg.hidden * mhc_streams;
                if (t > 0) {
                    double d2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (long i = 0; i < n; i++) {
                        float d = state[i] - pv[i];
                        d2 += (double)d * d;
                    }
                    fprintf(stderr, "[dbg10] t%d L%d tok_delta=%.6g\n",
                            t, L, sqrt(d2 / (double)n));
                }
                memcpy(prev_state + (size_t)L * cfg.hidden * mhc_streams,
                       state, (size_t)cfg.hidden * mhc_streams *
                       sizeof(float));
            }
            double _tr = now_s();
            if (use_real) {
                const SaltTrunkTensor *gt = &tl.t[tl.gate[L]];
                const float *gbias = NULL;
                if (tl.gate_bias[L] >= 0) {
                    const SaltTrunkTensor *bt = &tl.t[tl.gate_bias[L]];
                    gbias = (const float *)(const void *)(tr + bt->off);
                }
                /* the ffn router sees the mHC layer input (A-combined
                 * streams) when the checkpoint has hc_ffn tensors */
                const float *rstate = state;
                if (tl.ffn_norm[L] >= 0 && !getenv("SALT_NO_NORMS")) {
                    /* the router sees post_attention_layernorm(state),
                     * exactly like the expert input in moe_step */
                    memcpy(xin_buf, state,
                           (size_t)cfg.hidden * sizeof(float));
                    double sn = 0.0;
                    for (int i = 0; i < cfg.hidden; i++)
                        sn += (double)xin_buf[i] * xin_buf[i];
                    float rn = sqrtf((float)(sn / (double)cfg.hidden) +
                                     1e-6f);
                    const uint16_t *pnw = (const uint16_t *)(const void *)(
                        tr + tl.t[tl.ffn_norm[L]].off);
                    for (int i = 0; i < cfg.hidden; i++) {
                        uint32_t pb2 = (uint32_t)pnw[i] << 16;
                        float pw;
                        memcpy(&pw, &pb2, 4);
                        xin_buf[i] = xin_buf[i] / rn * pw;
                    }
                    rstate = xin_buf;
                }
                if (tl.hc_ffn_fn[L] >= 0) {
                    float A[8], C[8], B[64];
                    int nhc = 1;
                    int hok = salt_hc_params(
                        &tl, tl.hc_ffn_fn[L], tl.hc_ffn_base[L],
                        tl.hc_ffn_scale[L], tr, cfg.hidden, state,
                        &nhc, A, C, B);
                    if (hok < 0) { rc = 2; goto serve_next; }
                    if (hok > 0) {
                        salt_hc_combine(nhc, cfg.hidden, A, state, xin_buf);
                        rstate = xin_buf;
                        if (getenv("SALT_DEBUG5")) {
                            fprintf(stderr,
                                    "[dbg5] t%d L%d ffn A=", t, L);
                            for (int j = 0; j < nhc; j++)
                                fprintf(stderr, " %.4f", (double)A[j]);
                            fprintf(stderr, " C=");
                            for (int j = 0; j < nhc; j++)
                                fprintf(stderr, " %.4f", (double)C[j]);
                            fprintf(stderr, " B[0][*]=");
                            for (int k = 0; k < nhc; k++)
                                fprintf(stderr, " %.4f",
                                        (double)B[0 * nhc + k]);
                            fprintf(stderr, "\n");
                        }
                    }
                }
                if (gt->dtype == 5) {
                    /* MLX 4-bit router gate: U32 nibbles + BF16 scales +
                     * BF16 biases per group. Rows = experts, cols =
                     * packed_cols * 8. Locate the sibling tensors. */
                    const SaltTrunkTensor *gsc = NULL, *gbs = NULL;
                    for (int qi = tl.t_off[L]; qi < tl.t_off[L + 1]; qi++) {
                        if (name_ends(tl.t[qi].name, ".mlp.gate.scales"))
                            gsc = &tl.t[qi];
                        else if (name_ends(tl.t[qi].name, ".mlp.gate.biases"))
                            gbs = &tl.t[qi];
                    }
                    long pc = gt->dims[1];
                    if (gt->bits == 8)
                        salt_q8_matvec(
                            (const uint32_t *)(const void *)(tr + gt->off),
                            gsc ? (const uint16_t *)(const void *)(tr + gsc->off)
                                : NULL,
                            gbs ? (const uint16_t *)(const void *)(tr + gbs->off)
                                : NULL,
                            (int)gt->dims[0], (int)(pc * (32 / gt->bits)),
                            rstate, scores);
                    else
                        salt_q4_matvec(
                            (const uint32_t *)(const void *)(tr + gt->off),
                            gsc ? (const uint16_t *)(const void *)(tr + gsc->off)
                                : NULL,
                            gbs ? (const uint16_t *)(const void *)(tr + gbs->off)
                                : NULL,
                            (int)gt->dims[0], (int)(pc * (32 / gt->bits)),
                            rstate, scores);
                } else if (gt->dtype == 4)      /* BF16 */
                    salt_bf16_matvec(
                        (const uint16_t *)(const void *)(tr + gt->off),
                        cfg.n_experts, cfg.hidden, rstate, gbias, scores);
                else                     /* F32 */
                    salt_router_scores(
                        (const float *)(const void *)(tr + gt->off), gbias,
                        cfg.n_experts, cfg.hidden, rstate, scores);
                salt_topk(scores, cfg.n_experts, cfg.topk, idx, w);
                if (getenv("SALT_NAN_PROBE") && L < 3) {
                    fprintf(stderr, "[rtop] L%d sel=%d %d %d %d %d %d %d %d "
                            "w=%.4g %.4g %.4g %.4g\n", L, idx[0], idx[1],
                            idx[2], idx[3], idx[4], idx[5], idx[6], idx[7],
                            w[0], w[1], w[2], w[3]);
                }
                if (getenv("SALT_DEBUG4")) {
                    uint64_t ck = salt_mix64(0);
                    double s2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (int i = 0; i < cfg.hidden; i++) {
                        uint32_t bits;
                        memcpy(&bits, &state[i], 4);
                        ck = salt_mix64(ck ^ bits);
                    }
                    for (long i = 0; i < n; i++)
                        s2 += (double)state[i] * state[i];
                    fprintf(stderr, "[dbg4] t%d L%d scores %.6g %.6g %.6g %.6g"
                            " idx %d%d%d rms %.6g stateck %016llx\n", t, L,
                            (double)scores[0], (double)scores[1],
                            (double)scores[2], (double)scores[3],
                            idx[0], idx[1], idx[2],
                            sqrt(s2 / (double)n),
                            (unsigned long long)ck);
                }
            } else {
                /* hash-fallback layer: the trunk checksum feeds hstate,
                 * which drives the hash router. Only pay for it here —
                 * in moe mode with real gates it was 5.26 GB/token of
                 * single-threaded hashing that fed nothing. */
                hstate = salt_mix64(
                    hstate ^ salt_checksum(tr, trunk.lay[L].nbytes));
                salt_router(idx, w, &cfg, hstate, L, locality);
            }
            if (wf_on) wf_route += now_s() - _tr;
            if (step_ms) st_route += now_s() - _tr;
            double _tf = now_s();
            if (salt_cache_getmany(&cache, L, idx, cfg.topk, slots) < 0) {
                fprintf(stderr, "expert cache fetch failed at decode L%d\n", L);
                rc = 2;
                goto serve_next;
            }
            /* L2 (read-head): online hot-expert pinning. After a fetch,
             * any slot whose hit count crossed the threshold is promoted
             * to pinned so it is never evicted again. Gated by
             * SALT_PIN_HOT=1; threshold SALT_PIN_HITS (default 8). */
            if (pin_hot) {
                for (int j = 0; j < cfg.topk; j++) {
                    int s = slots[j];
                    if (s >= 0 && cache.hits[s] >= (uint64_t)pin_hits)
                        salt_cache_pin(&cache, s, 1);
                }
            }
            if (wf_on) wf_fetch += now_s() - _tf;
            if (step_ms) st_fetch += now_s() - _tf;
            if (use_real) {
                for (int j = 0; j < cfg.topk; j++) {
                    es[j] = salt_cache_slot_for(
                        &cache, slots[j], L, idx[j]);
                    if (!es[j]) {
                        fprintf(stderr,
                                "expert cache handle stale at decode L%d E%d\n",
                                L, idx[j]);
                        rc = 2;
                        goto serve_next;
                    }
                }
                double _tm = now_s();
                if (getenv("SALT_PROJ_MS") && gen_t == 0 &&
                    (L == 0 || L == 39)) {
                    char pth[64];
                    snprintf(pth, sizeof pth, "/tmp/dcd-preL%d.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float),
                               (size_t)cfg.hidden * mhc_streams, sf);
                        fclose(sf);
                    }
                }
                if (getenv("SALT_PROJ_MS") && gen_t == 0 &&
                    (L == 0 || L == 39)) {
                    char pth[64];
                    snprintf(pth, sizeof pth, "/tmp/dcd-esL%d.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        for (int j = 0; j < cfg.topk; j++) {
                            fwrite(&es[j], sizeof(void *), 1, sf);
                            if (es[j])
                                fwrite(es[j], 1, 4096, sf);
                            else {
                                static const char zz[4096];
                                fwrite(zz, 1, 4096, sf);
                            }
                        }
                        fclose(sf);
                    }
                }
                SaltGpuPoolDeviceLayer gpu_layer;
                int gpu_layer_ok = 0;
                int gpu_attempt = gpu_moe_intent;
                int prior_mapped_only = salt_gpu_mapped_only();
                int prior_defer = salt_gpu_defer();
                if (gpu_attempt) {
                    int prepare_rc = -1;
                    /* Explicit mapped residency is fail-closed: backend,
                     * acquisition, registration, and cleanup must succeed. */
                    salt_gpu_set_mapped_only(1);
                    if (salt_gpu_init() == 0)
                        prepare_rc = gpu_pool_layer_prepare(
                            &gpu_pool, &pl, (uint32_t)L, idx,
                            cfg.topk, &gpu_layer);
                    if (prepare_rc == 0) {
                        gpu_layer_ok = 1;
                    } else if (prepare_rc == -2) {
                        salt_gpu_set_mapped_only(prior_mapped_only);
                        fprintf(stderr, "gpu-index: L%d fallback release failed\n", L);
                        rc = 2; goto serve_done;
                    } else {
                        salt_gpu_set_mapped_only(prior_mapped_only);
                        fprintf(stderr,
                                "gpu-index: L%d canonical registration failed\n",
                                L);
                        rc = 2; goto serve_done;
                    }
                }
                if (gpu_layer_ok) {
                    /* gate/up must complete before the CPU SiLU dependency;
                     * down must complete before unbind/unmap. */
                    if (salt_gpu_set_defer(0) != 0) {
                        int release_rc =
                            salt_gpu_pool_device_layer_release(&gpu_layer);
                        int restore_rc = salt_gpu_set_defer(prior_defer);
                        salt_gpu_set_mapped_only(prior_mapped_only);
                        fprintf(stderr,
                                "gpu-index: L%d Metal completion failed%s%s\n",
                                L, release_rc ? "; resource retained" : "",
                                restore_rc ? "; policy restore failed" : "");
                        rc = 2; goto serve_done;
                    }
                }
                if (salt_moe_step(&cfg, &tl, L, tr, &pl, es, idx, w,
                                  state, scratch, scratch_n, jscratch,
                                  &n_matvec, &n_decode, NULL, 0.0f) != 0) {
                    int release_rc = 0;
                    if (gpu_layer_ok) {
                        release_rc =
                            salt_gpu_pool_device_layer_release(&gpu_layer);
                        if (salt_gpu_set_defer(prior_defer) != 0)
                            release_rc = -1;
                    }
                    if (gpu_attempt)
                        salt_gpu_set_mapped_only(prior_mapped_only);
                    if (release_rc != 0) {
                        fprintf(stderr,
                                "gpu-index: L%d failure cleanup release failed\n",
                                L);
                        rc = 2; goto serve_done;
                    }
                    fprintf(stderr, "moe step failed at layer %d\n", L);
                    rc = 2; goto serve_next;
                }
                if (gpu_layer_ok) {
                    int release_rc =
                        salt_gpu_pool_device_layer_release(&gpu_layer);
                    int restore_rc = salt_gpu_set_defer(prior_defer);
                    salt_gpu_set_mapped_only(prior_mapped_only);
                    if (release_rc != 0) {
                        fprintf(stderr, "gpu-index: L%d synchronized release failed\n", L);
                        rc = 2; goto serve_done;
                    }
                    if (restore_rc != 0) {
                        fprintf(stderr, "gpu-index: L%d policy restore failed\n", L);
                        rc = 2; goto serve_done;
                    }
                } else if (gpu_attempt) {
                    salt_gpu_set_mapped_only(prior_mapped_only);
                }
                if (getenv("SALT_PROJ_MS") && (gen_t == 0 || gen_t == 1) &&
                    (L == 0 || L == 39)) {
                    char pth[64];
                    snprintf(pth, sizeof pth, "/tmp/dcd-postL%d-g%d.bin",
                             L, gen_t);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float),
                               (size_t)cfg.hidden * mhc_streams, sf);
                        fclose(sf);
                    }
                }
                if (wf_on) wf_moe += now_s() - _tm;
            if (step_ms) st_moe += now_s() - _tm;
                if (getenv("SALT_DEBUG6")) {
                    double s2 = 0.0;
                    long n = (long)cfg.hidden * mhc_streams;
                    for (long i = 0; i < n; i++)
                        s2 += (double)state[i] * state[i];
                    fprintf(stderr, "[dbg6] t%d L%d rms=%.6g after ffn\n",
                            t, L, sqrt(s2 / (double)n));
                }
                if (getenv("SALT_DEBUG2")) {
                    uint64_t ck = salt_mix64(0);
                    for (int i = 0; i < cfg.hidden; i++)
                        ck = salt_mix64(ck ^ (uint64_t)(state[i] * 1e6f));
                    fprintf(stderr, "[dbg2] t%d L%d after moe  %016llx\n",
                            t, L, (unsigned long long)ck);
                }
            } else {
                for (int j = 0; j < cfg.topk; j++) {
                    const uint8_t *es = salt_cache_slot_for(
                        &cache, slots[j], L, idx[j]);
                    if (!es) {
                        fprintf(stderr,
                                "expert cache handle stale at decode L%d E%d\n",
                                L, idx[j]);
                        rc = 2;
                        goto serve_next;
                    }
                    hstate = salt_mix64(hstate ^
                                        salt_checksum(es, cfg.expert_nbytes));
                    if (trf) fprintf(trf, "%d,%d\n", L, idx[j]);
                }
            }
            if (trf && use_real)
                for (int j = 0; j < cfg.topk; j++)
                    fprintf(trf, "%d,%d\n", L, idx[j]);
            if (getenv("SALT_NAN_PROBE")) {
                int bad = 0;
                for (int i = 0; i < cfg.hidden; i++)
                    if (state[i] != state[i]) { bad = 1; break; }
                if (bad)
                    fprintf(stderr, "[nan] t%d after L%d\n", t, L);
                if (t == 8) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L%d-t8.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (L < 4) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L%d-t%d.bin",
                             L, t);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (L == 0) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L0-t%d.bin", t);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (t == 7 && L % 4 == 0) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-L%d-t7.bin", L);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (t == 7 && L == cfg.n_layers - 1) {
                    FILE *sf = fopen("/tmp/q36-eng-L39-t7.bin", "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (L == cfg.n_layers - 1 && t < 21) {
                    char pth[128];
                    snprintf(pth, sizeof pth, "/tmp/q36-eng-final-t%d.bin", t);
                    FILE *sf = fopen(pth, "wb");
                    if (sf) {
                        fwrite(state, sizeof(float), (size_t)cfg.hidden, sf);
                        fclose(sf);
                    }
                }
                if (getenv("SALT_NAN_PROBE") && L == cfg.n_layers - 1 &&
                    t > 0 && t % 2000 == 0) {
                    double sm = 0.0;
                    for (int i = 0; i < cfg.hidden; i++)
                        sm += (double)state[i] * state[i];
                    fprintf(stderr, "[stab] t=%d final-state rms %.6g\n",
                            t, sqrt(sm / cfg.hidden));
                }
                if (L < 2 || L % 8 == 0 || L >= 27 || t < 2 ||
                    (t == 1 && L < 8)) {
                    if (t < 64 || (t % 2000 == 0)) {
                        double sm = 0.0;
                        for (int i = 0; i < cfg.hidden; i++)
                            sm += (double)state[i] * state[i];
                        fprintf(stderr, "[st] t%d after L%d rms %.6g\n",
                                t, L, sqrt(sm / cfg.hidden));
                    }
                }
            }
            /* the pipelined prefetch (SALT_GPU_OVERLAP=1): proj(L+1)
             * dispatched now -- the state is the post-moe(L) (the
             * fold done), and the proj's FIRST step is the normed
             * snapshot, so the next layer's body finds the q/k/v's
             * prefetched. The GPU's async work overlaps the next
             * layer's CPU body. */
            if (overlap && gen_t > 0 && L + 1 < cfg.n_layers) {
                const uint8_t *trn = salt_trunk_bind(&trunk, L + 1);
                if (trn)
                    salt_attn_qwen_proj(&cfg, &tl, L + 1, trn, state,
                                        &kvc, t);
            }
            double _l3 = getenv("SALT_LAYER_MS") ? now_s() : 0;
            /* the layer's wait: the deferred proj commits sync here
             * (the 4-proj group: 4 commits, 1 wait). */
            if (salt_gpu_sync() != 0) {
                fprintf(stderr, "gpu-trunk: decode completion failed at L%d\n",
                        L);
                rc = 2;
                goto serve_done;
            }
            if (getenv("SALT_LAYER_MS"))
                fprintf(stderr, "[layms] L%d bind=%.2f attn=%.2f "
                        "moe=%.2f glue=%.2f ms\n", L,
                        (_l1-_l0)*1e3, (_l2-_l1)*1e3, (_l3-_l2)*1e3,
                        (now_s()-_l3)*1e3);
        }
        if (text_mode) {
            /* the head reads the mHC-contracted stream when the
             * checkpoint has the global hc_head (learned output
             * contraction); otherwise stream 0 (single-stream state) */
            const float *hstate_in = state;
            /* SALT_HEAD_RAW: the head reads a raw stream instead of the
             * hc_head A-combine -- the DEBUG11 evidence: the state moves
             * (delta 2.18) but the A-combined head input is frozen
             * (3.5e-7): the streams' movements cancel in the A
             * projection and the logits freeze. SALT_HEAD_STREAM picks
             * the stream (default 0). */
            if (getenv("SALT_HEAD_RAW")) {
                int hstream = 0;
                const char *hs = getenv("SALT_HEAD_STREAM");
                if (hs) hstream = atoi(hs);
                if (hstream < 0) hstream = 0;
                if (hstream >= mhc_streams) hstream = mhc_streams - 1;
                hstate_in = state + (size_t)hstream * cfg.hidden;
            } else if (tl.hc_head_fn >= 0) {
                const uint8_t *trh = runtime_trunk_bind(
                    &trunk, cfg.n_layers - 1,
                    gpu_trunk_intent && !gpu_trunk_prefill_only,
                    &gpu_trunk_map_active, &gpu_trunk_release_failed);
                if (!trh) {
                    fprintf(stderr, gpu_trunk_release_failed
                            ? "gpu-trunk: release failed before head\n"
                            : "trunk bind failed before head\n");
                    rc = 2;
                    goto serve_next;
                }
                float A[8], C[8], B[64];
                int nhc = 1;
                int hok = salt_hc_params(
                    &tl, tl.hc_head_fn, tl.hc_head_base, tl.hc_head_scale,
                    trh, cfg.hidden, state, &nhc, A, C, B);
                if (hok < 0) { rc = 2; goto serve_next; }
                if (hok > 0) {
                    salt_hc_combine(nhc, cfg.hidden, A, state, xin_buf);
                    hstate_in = xin_buf;
                }
            }
            if (getenv("SALT_DEBUG11")) {
                /* the head-input trace: the hstate_in's delta from the
                 * previous token (the A-combined projection -- the state
                 * moves (dbg10) but the logits freeze; is the movement
                 * lost in the hc_head combine?) */
                if (t == 0) {
                    memcpy(prev_hin, hstate_in,
                           (size_t)cfg.hidden * sizeof(float));
                } else {
                    double d2 = 0.0;
                    for (int i = 0; i < cfg.hidden; i++) {
                        float d = hstate_in[i] - prev_hin[i];
                        d2 += (double)d * d;
                    }
                    fprintf(stderr, "[dbg11] t%d head_in_delta=%.6g\n",
                            t, sqrtf((float)(d2 / (double)cfg.hidden)));
                    memcpy(prev_hin, hstate_in,
                           (size_t)cfg.hidden * sizeof(float));
                }
            }
            if (tl.final_norm >= 0) {
                /* Qwen3.5 final norm before lm_head (BF16 [hidden]);
                 * the tensor lives in the synthetic last trunk layer.
                 * Norm into xin_buf (free after the layer loop). */
                double ss0 = 0.0;
                int bad0 = 0;
                for (int i = 0; i < cfg.hidden; i++) {
                    ss0 += (double)hstate_in[i] * (double)hstate_in[i];
                    if (hstate_in[i] != hstate_in[i]) bad0 = 1;
                }
                if (getenv("SALT_NAN_PROBE")) {
                    fprintf(stderr, "[fin] t%d pre-norm rms %.6g bad %d\n",
                            t, sqrt(ss0 / cfg.hidden), bad0);
                    fflush(stderr);
                }
                /* pread the 4 KB final norm directly (same reasoning
                 * as the prefill path: bind()ing the synthetic last
                 * layer deadlocks the ring stream). */
                const SaltTrunkTensor *fn = &tl.t[tl.final_norm];
                static uint16_t *fnw = NULL;
                static long fnw_cap = 0;
                if (fn->nbytes > fnw_cap) {
                    uint16_t *nb = (uint16_t *)realloc(
                        fnw, (size_t)fn->nbytes);
                    if (!nb) { rc = 2; goto serve_next; }
                    fnw = nb; fnw_cap = fn->nbytes;
                }
                const SaltTrunkLayer *fnl =
                    &trunk.lay[tl.n_layers - 1];
                if (pread(trunk.fd, fnw, (size_t)fn->nbytes,
                          fnl->off + fn->off) != fn->nbytes) {
                    fprintf(stderr, "final-norm pread failed (gen)\n");
                    rc = 2; goto serve_next;
                }
                if (getenv("SALT_NAN_PROBE")) {
                    float w0, w1;
                    uint32_t b0 = (uint32_t)fnw[0] << 16;
                    uint32_t b1 = (uint32_t)fnw[1] << 16;
                    memcpy(&w0, &b0, 4);
                    memcpy(&w1, &b1, 4);
                    fprintf(stderr, "[fin] final_norm idx %d off %ld "
                            "nbytes %ld w[0]=%.4f w[1]=%.4f\n",
                            tl.final_norm, (long)fn->off, (long)fn->nbytes,
                            w0, w1);
                    fflush(stderr);
                }
                {
                    double ss = 0.0;
                    for (int i = 0; i < cfg.hidden; i++)
                        ss += (double)hstate_in[i] * hstate_in[i];
                    float r = sqrtf((float)(ss / (double)cfg.hidden) + 1e-6f);
                    for (int i = 0; i < cfg.hidden; i++) {
                        uint32_t bits = (uint32_t)fnw[i] << 16;
                        float w;
                        memcpy(&w, &bits, 4);
                        xin_buf[i] = hstate_in[i] / r * w;
                    }
                    hstate_in = xin_buf;
                }
            }
            double _th = now_s();
            if (salt_head_logits(&head, hstate_in, logits) != 0) {
                fprintf(stderr, "head logits failed\n");
                rc = 2; goto serve_next;
            }
            if (wf_on) wf_head += now_s() - _th;
            if (step_ms) st_head += now_s() - _th;
            if (step_ms) {
                double wall = st_attn + st_route + st_fetch + st_moe +
                              st_head;
                fprintf(stderr,
                        "[step] t=%d attn %.2f route %.2f fetch %.2f "
                        "moe %.2f head %.2f | sum %.2f ms (%.1f%% of "
                        "wall)\n",
                        t, st_attn * 1e3, st_route * 1e3,
                        st_fetch * 1e3, st_moe * 1e3, st_head * 1e3,
                        wall * 1e3, wall / (now_s() - st_wall) * 100.0);
                st_attn = st_route = st_fetch = st_moe = st_head = 0;
                st_wall = now_s();
            }
            if (getenv("SALT_DEBUG7")) {
                /* the logits shape: peaked vs flat, and the top-5 ids
                 * + their decoded tokens (the soup diagnosis) */
                long V = head.dims[0];
                int top[5];
                for (int k = 0; k < 5; k++) top[k] = 0;
                for (long i = 0; i < V; i++) {
                    for (int k = 0; k < 5; k++) {
                        if (logits[i] > logits[top[k]]) {
                            for (int kk = 4; kk > k; kk--)
                                top[kk] = top[kk - 1];
                            top[k] = (int)i;
                            break;
                        }
                    }
                }
                double sr2 = 0.0;
                for (long i = 0; i < (long)cfg.hidden * 4; i++)
                    sr2 += (double)state[i] * state[i];
                float srms = sqrtf((float)(sr2 / (double)(cfg.hidden * 4)));
                double hr2 = 0.0;
                for (long i = 0; i < (long)cfg.hidden; i++)
                    hr2 += (double)hstate_in[i] * hstate_in[i];
                float hrms = sqrtf((float)(hr2 / (double)cfg.hidden));
                fprintf(stderr, "logits: t%-2d state_rms %.3f head_rms %.3f"
                                " top5", t, srms, hrms);
                for (int k = 0; k < 5; k++) {
                    char tb[64];
                    salt_tokenizer_decode(&tok, &top[k], 1, tb, 64);
                    fprintf(stderr, " [%6d %.3f %s]", top[k], logits[top[k]], tb);
                }
                fprintf(stderr, "\n");
            }
            int tokid = -1;
            if (gen_t >= 0 && rep_pen > 1.0f) {
                if (getenv("SALT_DEBUG7") && n_recent == 0)
                    fprintf(stderr, "[pen] rep_pen=%.2f window=%d "
                            "vocab=%ld\n", rep_pen, 64, head.dims[0]);
                /* Multiplicative repetition penalty over the recent
                 * generated window, applied PER OCCURRENCE (llama.cpp
                 * semantics: logits /= pen^count). The HF set-style
                 * (one division per unique token) does not break
                 * short repetition cycles -- a 7-token loop repeated
                 * 9x inside the 64-window keeps winning greedy.
                 * This also allows a lower base penalty (1.3) to
                 * break loops without flattening legitimate text.
                 * O(V + 64) via a count array (a naive V*64 scan is
                 * ~16M ops/token -- too slow on the decode path). */
                static int *rcnt = NULL;
                static long rcnt_cap = 0;
                if (!rcnt) {
                    rcnt_cap = head.dims[0] > 0 ? head.dims[0] : 1;
                    rcnt = (int *)calloc((size_t)rcnt_cap, sizeof(int));
                }
                if (rcnt) {
                    for (int r = 0; r < n_recent; r++) {
                        int t = recent_toks[r];
                        if (t >= 0 && t < rcnt_cap) rcnt[t]++;
                    }
                    for (int i = 0; i < (int)head.dims[0]; i++) {
                        int cnt = rcnt[i];
                        if (cnt > 1) {
                            float p = rep_pen;
                            for (int k = 1; k < cnt; k++) p *= rep_pen;
                            logits[i] /= p;
                        } else if (cnt == 1) {
                            logits[i] /= rep_pen;
                        }
                    }
                    memset(rcnt, 0, (size_t)head.dims[0] * sizeof(int));
                }
                /* additive frequency penalty (SALT_FREQ_PENALTY,
                 * vLLM frequency_penalty semantics): logits[i] -=
                 * freq_pen * count_i. The multiplicative form above
                 * is overwhelmed by 30-logit attractor peaks; the
                 * additive form subtracts directly from the gap. */
                {
                    const char *fp = getenv("SALT_FREQ_PENALTY");
                    if (fp && atof(fp) > 0.0 && n_recent > 0)
                        salt_apply_freq_penalty(logits,
                                                (int)head.dims[0],
                                                recent_toks, n_recent,
                                                (float)atof(fp));
                }
                /* presence penalty (SALT_PRESENCE_PENALTY, vLLM
                 * semantics): logits[i] -= pres_pen ONCE per seen
                 * token. The Qwen3.6 model card's recommended
                 * loop-killer (1.5 thinking / instruct general). */
                {
                    const char *pp = getenv("SALT_PRESENCE_PENALTY");
                    if (pp && atof(pp) > 0.0 && n_recent > 0)
                        salt_apply_presence_penalty(logits,
                                                    (int)head.dims[0],
                                                    recent_toks, n_recent,
                                                    (float)atof(pp));
                }
            }
            if (gen_t < 0) {
                /* prompt pass: no output -- but the LAST prompt token's
                 * logits predict the FIRST generated token. Capture that
                 * sample so generation continues from it; otherwise the
                 * first gen step re-feeds pids[0] and the model echoes
                 * the prompt forever (saw: "capital of France isThe..."). */
                if (t == npids - 1) {
                    if (getenv("SALT_GREEDY"))
                        last_tok = salt_argmax(logits, (int)head.dims[0]);
                    else {
                        int top_k = 20;
                        double top_p = 0.95;
                        float temp = 1.0f;
                        const char *tk = getenv("SALT_TOPK");
                        if (tk && atoi(tk) > 0) top_k = atoi(tk);
                        const char *tp = getenv("SALT_TOP_P");
                        if (tp && atof(tp) > 0.0) top_p = atof(tp);
                        const char *te = getenv("SALT_TEMP");
                        if (te && atof(te) > 0.0) temp = (float)atof(te);
                        last_tok = salt_sample_vllm(
                            logits, (int)head.dims[0], top_k, top_p,
                            temp, &rng);
                    }
                }
            } else if (getenv("SALT_GREEDY"))
                tokid = salt_argmax(logits, (int)head.dims[0]);
            else {
                /* vLLM-style sampling (top-k -> top-p -> temp) --
                 * the community recipe for quantized models; the
                 * model's own generation_config ships top_k=20
                 * top_p=0.95. SALT_TOPK overrides the k, SALT_TOP_P
                 * the p, SALT_TEMP the temperature. The naive
                 * full-vocab softmax sampler is NOT used here: over
                 * a 248K vocab with a normal top1/top5 logit gap
                 * every draw is arbitrary ("garbage"). */
                int top_k = 20;
                double top_p = 0.95;
                float temp = 1.0f;
                const char *tk = getenv("SALT_TOPK");
                if (tk && atoi(tk) > 0) top_k = atoi(tk);
                const char *tp = getenv("SALT_TOP_P");
                if (tp && atof(tp) > 0.0) top_p = atof(tp);
                const char *te = getenv("SALT_TEMP");
                if (te && atof(te) > 0.0) temp = (float)atof(te);
                tokid = salt_sample_vllm(logits, (int)head.dims[0],
                                         top_k, top_p, temp, &rng);
            }
            /* EOS stop: if the sampled token is a stop id, end
             * generation here (the model's <|im_end|>/<|endoftext|>
             * are emitted as ordinary tokens otherwise and the run
             * pads to --gen). --eos-ids 248044,248046 */
            if (gen_t >= 0 && n_eos > 0) {
                int hit_eos = 0;
                for (int e = 0; e < n_eos; e++)
                    if (tokid == eos_ids[e]) { hit_eos = 1; break; }
                if (hit_eos) {
                    gen_done = gen_t + 1;  /* EOS consumed: it IS a row */
                    if (tok_path) {
                        char tbuf[256]; tbuf[0] = 0;
                        int tl = salt_tokenizer_decode(&tok, &tokid, 1,
                                                       tbuf, 256);
                        if (tl > 0) fwrite(tbuf, 1, (size_t)tl, stdout);
                    }
                    fprintf(stderr, "\n[EOS] token %d at gen %d -- "
                            "stopping\n", tokid, gen_t);
                    if (trf) fclose(trf), trf = NULL;
                    break;   /* out of the decode loop; report emits */
                }
                /* stuck-token detector (period-adaptive): scan the
                 * recent window for a repetition CYCLE -- the
                 * smallest p where the last p tokens match the p
                 * before them (t[i] == t[i-p] over the window tail).
                 * A fixed same-token-count rule is content-blind:
                 * a 2-token loop ("urbanized") trips it, but a
                 * longer-phrase loop (period >= 16) never accumulates
                 * N hits of any single token inside a small window.
                 * Period detection catches ANY loop regardless of
                 * its length. SALT_LOOP_P = max period scanned
                 * (default 32); SALT_LOOP_P=0 disables.
                 * Deterministic: fixed scan, fixed window. */
                if (gen_t >= 0) {
                    int max_p = 32;
                    const char *lp = getenv("SALT_LOOP_P");
                    if (lp && atoi(lp) > 0) max_p = atoi(lp);
                    else if (lp && atoi(lp) == 0) max_p = 0;
                    if (max_p > 0 && n_recent >= 4) {
                        /* window: recent_toks[0] = newest. A cycle of
                         * period p means every window slot matches
                         * the slot p ahead of it. */
                        int hit_p = 0;
                        for (int p = 1; p <= max_p && p * 2 <= n_recent;
                             p++) {
                            int matches = 1;
                            for (int i = 0; i < p * 2 && i < n_recent; i++)
                                if (recent_toks[i] !=
                                    recent_toks[i + p]) {
                                    matches = 0;
                                    break;
                                }
                            if (matches) { hit_p = p; break; }
                        }
                        if (hit_p > 0) {
                            /* candidate NOT consumed: it is not a row */
                            gen_done = gen_t;
                            fprintf(stderr,
                                    "\n[loop] period %d detected at gen %d "
                                    "-- stopping\n",
                                    hit_p, gen_t);
                            if (trf) fclose(trf), trf = NULL;
                            break;
                        }
                    }
                }
            }
            if (gen_t >= 0) {
            if (tok_path) {
                char tbuf[256];
                tbuf[0] = 0;
                int tl = salt_tokenizer_decode(&tok, &tokid, 1, tbuf, 256);
                if (tl < 0) {
                    /* unknown token id (shouldn't happen now that
                     * added tokens decode; never print a stale
                     * buffer) */
                    tbuf[0] = 0;
                }
                if (getenv("SALT_PROJ_MS")) {
                    FILE *tf = fopen("/tmp/dcd-tokid.bin", "ab");
                    if (tf) {
                        fwrite(&tokid, sizeof(int), 1, tf);
                        fclose(tf);
                    }
                }
                /* opt-in think-trace suppression: <think>...</think>
                 * is still generated (conditions the answer) but not
                 * printed. Token-boundary safe: the tokenizer keeps
                 * the tags as single added tokens. SALT_MAX_THINK
                 * caps the reasoning (the thinking-effort knob):
                 * after N think-tokens the engine forces the close
                 * and the answer proceeds. */
                if (strip_think) {
                    if (in_think) {
                        if (strstr(tbuf, "</think>")) in_think = 0;
                        else if (max_think > 0 &&
                                 ++think_count > max_think) {
                            in_think = 0;
                            printf("</think>");
                        }
                    } else if (strstr(tbuf, "<think>")) {
                        in_think = 1;
                    } else {
                        printf("%s", tbuf);
                    }
                } else {
                    printf("%s", tbuf);
                }
                fflush(stdout);
            } else {
                printf("%s%d", gen_t ? " " : "", tokid);
                fflush(stdout);
            }
            last_tok = tokid;
            if (roofline_on && gen_t == 0 && !roof_first_seen) {
                roof_first = roof_snap(&cache);
                roof_first_seen = 1;
            }
            /* the recent window feeds the repetition penalty, the
             * additive penalties, and the stuck-token detector.
             * Shift-window (slot 0 = newest): a naive ring with a
             * count capped at 64 never advances (n_recent % 64 == 0
             * forever after the fill), silently freezing the window
             * at [latest, first-63] -- which broke ALL three
             * consumers (measured: reps=0 mid-loop). memmove of 64
             * ints is 256 B/token, nothing against ~1.2 GB/token
             * of expert traffic. */
            if (gen_t >= 0) {
                memmove(&recent_toks[1], &recent_toks[0],
                        (64 - 1) * sizeof(int));
                recent_toks[0] = tokid;
                if (n_recent < 64) n_recent++;
            }
            if (gen_t >= 0 && tokids_path) {
                FILE *tf = fopen(tokids_path, "ab");
                if (tf) {
                    fwrite(&tokid, sizeof(int), 1, tf);
                    fclose(tf);
                }
            }
            }
        }
        if (getenv("SALT_DEBUG")) {
            uint64_t ck = salt_mix64(0);
            for (int i = 0; i < cfg.hidden; i++)
                ck = salt_mix64(ck ^ (uint64_t)(state[i] * 1e6f));
            fprintf(stderr, "[dbg] token %d state ck %016llx\n", t,
                    (unsigned long long)ck);
        }
        if (gen_t >= 0) {
            SaltMemSnapshot memory_now = memory_snapshot();
            memory_sample_peak(&memory_sampled_peak, &memory_now);
            if (memory_limit_breached(&memory_now, mem_limit_gb,
                                      "token complete", gen_t + 1, gen)) {
                if (trf) { fclose(trf); trf = NULL; }
                rc = 3;
                goto serve_done;
            }
        }
    }
    /* EOS and loop detection break out before the ordinary per-token tail.
     * Enforce once at the decode boundary so persistent serving cannot accept
     * a request that completed its final work above the hard limit. */
    {
        SaltMemSnapshot memory_decode_exit = memory_snapshot();
        memory_sample_peak(&memory_sampled_peak, &memory_decode_exit);
        if (memory_limit_breached(&memory_decode_exit, mem_limit_gb,
                                  "decode exit", gen_done, gen)) {
            if (trf) { fclose(trf); trf = NULL; }
            rc = 3;
            goto serve_done;
        }
    }
    if (gpu_trunk_prefill_only) {
        if (gpu_prefill_phase_check(
                gpu_prefill_stats_captured, gpu_prefill_gpu_chunks,
                gpu_prefill_trunk_batches, gpu_prefill_trunk_after_batches,
                gpu_prefill_direct_after_batches,
                gpu_prefill_direct_after_jobs,
                &gpu_request_start_stats) != 0) {
            rc = 2;
            goto serve_done;
        }
        gpu_prefill_phase_validated = 1;
    }
    if (text_mode) printf("\n");
    double dt = now_s() - t0;
    if (trf) fclose(trf);

    /* --kv-save-after: serialize the FULL session state (restored
     * rows [0..kv_prefilled) + this request's prompt rows
     * [kv_prefilled..kv_prefilled+npids) + generated rows). A later
     * --kv-load resumes with the whole conversation baked in --
     * the KV cache IS the memory; there is no context window. */
    if (kv_save_after && moe_mode && kv_ok && !serve_fifo) {
        long pref = (long)kv_prefilled + (long)npids + (long)gen_done;
        if (pref < 0 || pref > INT_MAX ||
            qwen_state_transaction_begin(&state_control, state_model,
                &cfg, &kvc, (int)pref, kvlat,
                SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_FULL,
                SALT_STATE_REASON_EXPLICIT, &state_transaction) != 0) {
            fprintf(stderr, "kv-save-after: engine transaction rejected\n");
            rc = 2; goto serve_next;
        }
        if (qwen_state_save_full(
                kv_save_after, &cfg, &kvc, (int)pref, kvlat) != 0) {
            fprintf(stderr, "kv-save-after: export failed\n");
            rc = 2; goto serve_next;
        }
        if (qwen_state_transaction_finish(state_model, &cfg, &kvc,
                (int)pref, kvlat, &state_transaction) != 0) {
            fprintf(stderr, "kv-save-after: engine transaction failed\n");
            rc = 2; goto serve_next;
        }
        fprintf(stderr, "[kv-save-after] %s (%ld tokens incl %d gen)\n",
                kv_save_after, pref, gen);
    }

    if (dump_path && moe_mode) {
        int bad = 0;
        for (int i = 0; i < cfg.hidden; i++)
            if (!(state[i] == state[i]) || state[i] > 1e30f ||
                state[i] < -1e30f) { bad = 1; break; }
        if (bad) {
            fprintf(stderr, "REFUSE: final state not finite, not dumping\n");
            rc = 2; goto serve_next;
        }
        FILE *df = fopen(dump_path, "wb");
        if (!df) { fprintf(stderr, "cannot open %s\n", dump_path); rc = 2; goto serve_next; }
        fwrite(state, sizeof(float),
               (size_t)cfg.hidden * (size_t)mhc_streams, df);
        fclose(df);
    }

    int64_t read_b = trunk.nread + cache.nread;
    double gb_tok = (double)read_b / 1e9 / (double)gen;
    double hit = cache.nreq ? (double)cache.nhit / (double)cache.nreq : 0.0;
    memory_final = memory_snapshot();
    memory_sample_peak(&memory_sampled_peak, &memory_final);

    fprintf(stderr,
            "\n--- run report ---\n"
            "config: %d layers x %d experts, topk %d, expert %lld bytes\n"
            "pool:   %s\n"
            "trunk:  pin %d/%d layers, ring %d x %lld bytes\n"
            "cache:  %d slots (%d MB), %d fetch threads\n"
            "%d tokens in %.1f s, %.2f s/token\n"
            "GB read per token: %.2f  (trunk %lld MB, experts %lld MB)\n"
            "cache: %lld requests, %lld hits (%.1f%%), %lld dropped\n"
            "%s%s",
            cfg.n_layers, cfg.n_experts, cfg.topk, (long long)cfg.expert_nbytes,
            pool_src,
            trunk.npin, trunk.n_layers, trunk.nring, (long long)trunk.slot,
            cache.nslot, (int)(cache_bytes / (1 << 20)),
            threads,
            gen, dt, dt / (double)gen,
            gb_tok, (long long)(trunk.nread / (1 << 20)),
            (long long)(cache.nread / (1 << 20)),
            (long long)cache.nreq, (long long)cache.nhit, hit * 100.0,
            (long long)cache.ndrop,
            moe_mode ? "moe: real matvec compute (kernels)\n" : "",
            salt_kernels_simd() ? "kernels: simd\n" : "kernels: scalar\n");
    fprintf(stderr,
            "MEMORY (absolute decimal GB%s)\n"
            "  application footprint current       %.2f / limit %.2f\n"
            "  application footprint observed peak %.2f\n"
            "  application footprint OS peak       %.2f\n"
            "  resident/touched current             %.2f\n"
            "  resident/touched observed peak       %.2f\n"
            "  resident/touched OS peak             %.2f\n"
            "  anonymous/internal                   %.2f\n"
            "  compressed                           %.2f\n"
            "  file-backed/external                 %.2f\n"
            "  shared + page tables + swap          %.2f / %.2f / %.2f\n",
            memory_final.application_is_approximate
                ? "; application metric is Linux approximation" : "",
            (double)memory_final.application_b / 1e9, mem_limit_gb,
            (double)memory_sampled_peak.application_b / 1e9,
            (double)memory_final.application_os_peak_b / 1e9,
            (double)memory_final.resident_b / 1e9,
            (double)memory_sampled_peak.resident_b / 1e9,
            (double)memory_final.resident_os_peak_b / 1e9,
            (double)memory_final.anonymous_b / 1e9,
            (double)memory_final.compressed_b / 1e9,
            (double)memory_final.file_backed_b / 1e9,
            (double)memory_final.shared_b / 1e9,
            (double)memory_final.page_table_b / 1e9,
            (double)memory_final.swap_b / 1e9);

    if (wf_on) {
        double wf_sum = wf_attn + wf_route + wf_fetch + wf_moe + wf_head;
        wf_misc = dt - wf_sum;
        if (wf_misc < 0) wf_misc = 0;
        /* NOTE: the timers span the WHOLE run (prompt pass + gen), so
         * per-token is over total_toks, NOT gen -- gen alone inflated
         * every phase ~376x on long-context runs and broke the
         * sum-to-total check. */
        int wt = total_toks > 0 ? total_toks : gen;
        fprintf(stderr,
                "\n--- waterfall (total %.1f s over %d tokens, %.1f ms/token) ---\n"
                "attn:   %6.1f ms/token (%4.1f%%)\n"
                "router: %6.1f ms/token (%4.1f%%)\n"
                "fetch:  %6.1f ms/token (%4.1f%%)\n"
                "moe:    %6.1f ms/token (%4.1f%%)\n"
                "head:   %6.1f ms/token (%4.1f%%)\n"
                "prefill: %6.1f ms/token (%4.1f%%)  <-- the chunked decode\n"
                "  bind:   %6.2f ms/token (the trunk pread)\n"
                "  linear-attention layers: %6.2f ms/token\n"
                "  GQA layers:              %6.2f ms/token\n"
                "  route:                   %6.2f ms/token\n"
                "  expert fetch:            %6.2f ms/token\n"
                "  MoE compute:             %6.2f ms/token\n"
                "  RFM glue:                %6.2f ms/token\n"
                "  outer prefill glue:      %6.2f ms/token\n"
                "memory (GB resident/touched / application footprint; sampled):\n"
                "  run start:                %6.2f / %6.2f\n"
                "  decode entry:             %6.2f / %6.2f\n"
                "  final:                    %6.2f / %6.2f\n"
                "  sampled peak:             %6.2f / %6.2f\n",
                dt, wt, dt / (double)wt * 1e3,
                wf_attn / wt * 1e3, wf_attn / dt * 100.0,
                wf_route / wt * 1e3, wf_route / dt * 100.0,
                wf_fetch / wt * 1e3, wf_fetch / dt * 100.0,
                wf_moe / wt * 1e3, wf_moe / dt * 100.0,
                wf_head / wt * 1e3, wf_head / dt * 100.0,
                wf_misc / wt * 1e3, wf_misc / dt * 100.0,
                wf_pbind / wt * 1e3,
                (wf_pattn - wf_pgqa) / wt * 1e3,
                wf_pgqa / wt * 1e3,
                wf_proute / wt * 1e3,
                wf_pfetch / wt * 1e3,
                wf_pmoe / wt * 1e3,
                (wf_prfm - wf_proute - wf_pfetch - wf_pmoe) / wt * 1e3,
                (wf_misc - wf_pbind - wf_pattn - wf_prfm)
                    / wt * 1e3,
                (double)memory_run_start.resident_b / 1e9,
                (double)memory_run_start.application_b / 1e9,
                (double)memory_decode_entry.resident_b / 1e9,
                (double)memory_decode_entry.application_b / 1e9,
                (double)memory_final.resident_b / 1e9,
                (double)memory_final.application_b / 1e9,
                (double)memory_sampled_peak.resident_b / 1e9,
                (double)memory_sampled_peak.application_b / 1e9);
    }

    if (roofline_on && roof_decode_seen) {
        RoofSnap roof_end = roof_snap(&cache);
        uint64_t pphys = 0, pfaults = 0;
        int pphys_valid = roof_counter_delta(
            roof_start.physical_read_bytes, roof_start.physical_read_valid,
            roof_decode.physical_read_bytes, roof_decode.physical_read_valid,
            &pphys);
        int pfaults_valid = roof_counter_delta(
            roof_start.major_faults, roof_start.major_faults_valid,
            roof_decode.major_faults, roof_decode.major_faults_valid,
            &pfaults);
        char pphys_text[32], pfaults_text[32];
        roof_rate_text(pphys_text, sizeof pphys_text, pphys, pphys_valid,
                       1048576.0 * (double)npids);
        roof_rate_text(pfaults_text, sizeof pfaults_text, pfaults,
                       pfaults_valid, (double)npids);
        int64_t plog = roof_decode.logical_miss_bytes -
            roof_start.logical_miss_bytes;
        int64_t preq = roof_decode.requests - roof_start.requests;
        int64_t phit = roof_decode.hits - roof_start.hits;
        double pwait = roof_decode.fetch_wait_s - roof_start.fetch_wait_s;
        int64_t pjobs = roof_decode.fetch_jobs - roof_start.fetch_jobs;
        double pwall = roof_decode.wall_s - roof_start.wall_s;
        fprintf(stderr, "\n--- disk roofline (phase-exclusive) ---\n");
        fprintf(stderr,
                "cache=%s touch=%s; logical misses are payload bytes, "
                "os-read is a process-attributed transport-inclusive OS "
                "counter; n/a means unavailable\n",
                cache.mode == 1 ? "zerocopy" :
                    (cache.mode == 2 ? "mixture" : "arena"),
                getenv("SALT_FETCH_TOUCH") ? getenv("SALT_FETCH_TOUCH") : "0");
        fprintf(stderr,
                "prefill: tokens=%d wall=%.3fs logical=%.2f MiB/token "
                "os-read-MiB/token=%s major-faults/token=%s hit=%.1f%% "
                "fetch-wait=%.3f ms/token jobs=%lld\n",
                npids, pwall,
                npids > 0 ? (double)plog / 1048576.0 / npids : 0.0,
                pphys_text, pfaults_text,
                preq > 0 ? (double)phit / preq * 100.0 : 0.0,
                npids > 0 ? pwait * 1e3 / npids : 0.0,
                (long long)pjobs);
        if (roof_first_seen) {
            uint64_t fphys = 0, ffaults = 0;
            int fphys_valid = roof_counter_delta(
                roof_decode.physical_read_bytes,
                roof_decode.physical_read_valid,
                roof_first.physical_read_bytes,
                roof_first.physical_read_valid, &fphys);
            int ffaults_valid = roof_counter_delta(
                roof_decode.major_faults, roof_decode.major_faults_valid,
                roof_first.major_faults, roof_first.major_faults_valid,
                &ffaults);
            int64_t flog = roof_first.logical_miss_bytes -
                roof_decode.logical_miss_bytes;
            int64_t freq = roof_first.requests - roof_decode.requests;
            int64_t fhit = roof_first.hits - roof_decode.hits;
            double fwait = roof_first.fetch_wait_s -
                roof_decode.fetch_wait_s;
            int64_t fjobs = roof_first.fetch_jobs - roof_decode.fetch_jobs;
            double first_wall = roof_first.wall_s - roof_decode.wall_s;
            int post_n = gen_done > 1 ? gen_done - 1 : 0;
            uint64_t dphys = 0, dfaults = 0;
            int dphys_valid = roof_counter_delta(
                roof_first.physical_read_bytes,
                roof_first.physical_read_valid,
                roof_end.physical_read_bytes, roof_end.physical_read_valid,
                &dphys);
            int dfaults_valid = roof_counter_delta(
                roof_first.major_faults, roof_first.major_faults_valid,
                roof_end.major_faults, roof_end.major_faults_valid,
                &dfaults);
            int64_t dlog = roof_end.logical_miss_bytes -
                roof_first.logical_miss_bytes;
            int64_t dreq = roof_end.requests - roof_first.requests;
            int64_t dhit = roof_end.hits - roof_first.hits;
            double dwait = roof_end.fetch_wait_s - roof_first.fetch_wait_s;
            int64_t djobs = roof_end.fetch_jobs - roof_first.fetch_jobs;
            double dwall = roof_end.wall_s - roof_first.wall_s;
            char fphys_text[32], ffaults_text[32];
            char dphys_text[32], dfaults_text[32];
            roof_rate_text(fphys_text, sizeof fphys_text, fphys,
                           fphys_valid, 1048576.0);
            roof_rate_text(ffaults_text, sizeof ffaults_text, ffaults,
                           ffaults_valid, 1.0);
            roof_rate_text(dphys_text, sizeof dphys_text, dphys,
                           dphys_valid,
                           1048576.0 * (double)post_n);
            roof_rate_text(dfaults_text, sizeof dfaults_text, dfaults,
                           dfaults_valid, (double)post_n);
            fprintf(stderr,
                    "first-token: TTFT=%.3fs decode=%.3fs "
                    "logical=%.2f MiB os-read-MiB=%s major-faults=%s "
                    "hit=%.1f%% "
                    "fetch-wait=%.3fms jobs=%lld\n",
                    roof_first.wall_s - roof_start.wall_s, first_wall,
                    (double)flog / 1048576.0,
                    fphys_text, ffaults_text,
                    freq > 0 ? (double)fhit / freq * 100.0 : 0.0,
                    fwait * 1e3, (long long)fjobs);
            fprintf(stderr,
                    "post-first: tokens=%d wall=%.3fs rate=%.3f tok/s "
                    "logical=%.2f MiB/token os-read-MiB/token=%s "
                    "major-faults/token=%s "
                    "hit=%.1f%% fetch-wait=%.3f ms/token jobs=%lld\n",
                    post_n, dwall,
                    dwall > 0 ? (double)post_n / dwall : 0.0,
                    post_n > 0 ? (double)dlog / 1048576.0 / post_n : 0.0,
                    dphys_text, dfaults_text,
                    dreq > 0 ? (double)dhit / dreq * 100.0 : 0.0,
                    post_n > 0 ? dwait * 1e3 / post_n : 0.0,
                    (long long)djobs);
        }
        if (cache.mode == 1 && !getenv("SALT_FETCH_TOUCH"))
            fprintf(stderr,
                    "ownership: lazy mmap faults may be charged to "
                    "projection; fetch-wait is advisory/barrier time only\n");
    }

    if (moe_profile_on) {
        SaltMoEBatchProfile mp;
        salt_moe_batch_profile_get(&mp);
        if (mp.calls > 0) {
            double mt = mp.setup_s + mp.select_s + mp.group_s +
                mp.gather_s + mp.gate_up_s + mp.activation_s +
                mp.down_pack_s + mp.down_s + mp.scatter_s +
                mp.shared_s + mp.residual_s + mp.glue_s;
            double md = npids > 0 ? (double)npids : 1.0;
            fprintf(stderr,
                    "\n--- grouped MoE profile (exclusive, prefill) ---\n"
                    "calls=%lld groups=%lld selections=%lld tokens=%d\n"
                    "  setup/zero/alloc: %7.3f ms/token\n"
                    "  selection census: %7.3f ms/token\n"
                    "  group/sort/alloc:  %7.3f ms/token\n"
                    "  input gather:      %7.3f ms/token\n"
                    "  gate+up projection:%7.3f ms/token\n"
                    "  SiLU multiply:     %7.3f ms/token\n"
                    "  down-input pack:   %7.3f ms/token\n"
                    "  down projection:   %7.3f ms/token\n"
                    "  weighted scatter:  %7.3f ms/token\n"
                    "  shared expert:     %7.3f ms/token\n"
                    "  residual add:      %7.3f ms/token\n"
                    "  cleanup/probe:     %7.3f ms/token\n"
                    "  total:             %7.3f ms/token\n",
                    (long long)mp.calls, (long long)mp.groups,
                    (long long)mp.selections, npids,
                    mp.setup_s / md * 1e3, mp.select_s / md * 1e3,
                    mp.group_s / md * 1e3, mp.gather_s / md * 1e3,
                    mp.gate_up_s / md * 1e3,
                    mp.activation_s / md * 1e3,
                    mp.down_pack_s / md * 1e3, mp.down_s / md * 1e3,
                    mp.scatter_s / md * 1e3, mp.shared_s / md * 1e3,
                    mp.residual_s / md * 1e3, mp.glue_s / md * 1e3,
                    mt / md * 1e3);
        }
        SaltMoEDecodeProfile dp;
        salt_moe_decode_profile_get(&dp);
        if (dp.calls > 0) {
            double dd = gen_done > 0 ? (double)gen_done : 1.0;
            double other = dp.total_s - dp.scheduler_window_s -
                dp.shared_post_window_s - dp.combine_s;
            fprintf(stderr,
                    "\n--- decode MoE profile (critical wall + overlap) ---\n"
                    "calls=%lld routed-jobs=%lld tokens=%d\n"
                    "  scheduler window:  %7.3f ms/token\n"
                    "  shared caller:     %7.3f ms/token (nested diagnostic)\n"
                    "  shared post-window:%7.3f ms/token\n"
                    "  routed combine:    %7.3f ms/token\n"
                    "  fold/norm/resid:    %7.3f ms/token\n"
                    "  total:              %7.3f ms/token\n",
                    (long long)dp.calls, (long long)dp.routed_jobs, gen_done,
                    dp.scheduler_window_s / dd * 1e3,
                    dp.shared_s / dd * 1e3,
                    dp.shared_post_window_s / dd * 1e3,
                    dp.combine_s / dd * 1e3, other / dd * 1e3,
                    dp.total_s / dd * 1e3);
        }
    }

    if (cache.ndrop > 0) {
        fprintf(stderr,
                "\nWARNING: %lld expert fetch(es) were dropped (all slots "
                "pinned/inflight). Output is silently corrupt; treating the "
                "run as failed.\n", (long long)cache.ndrop);
        rc = 4;
    }
    if (moe_mode)
        fprintf(stderr, "moe: %lld matvecs, %lld decoded elements\n",
                (long long)n_matvec, (long long)n_decode);

    /* serve mode: mark the end of this response on stdout with a
     * NUL byte (decoded text is UTF-8; a lone \0 cannot collide).
     * serve.py reads up to the NUL to frame one completion. */
    if (serve_fifo) {
        fputc(0, stdout);
        fflush(stdout);
        fprintf(stderr, "[serve] response done\n");
    }

    /* serve loop: free the per-request resources (state/scratch/
     * xin_buf/prev_state/prev_hin -- pids is freed at serve_next's
     * parse) and read the next prompt. The EXPENSIVE resources
     * (trunk, expert cache, pool, head, embed, tokenizer, logits,
     * jscratch) stay loaded across requests. The KV CACHE is NOT
     * freed in session mode (SALT_SESSION=1): it IS the memory --
     * the next request's pids are request-local new tokens appended
     * at kv_prefilled (never-forget; no context window). */
    if (serve_fifo) {
        free(state); state = NULL;
        free(xin_buf); xin_buf = NULL;
        free(prev_state); prev_state = NULL;
        free(prev_hin); prev_hin = NULL;
        if (kv_ok && !getenv("SALT_SESSION")) { salt_kv_free(&kvc); kv_ok = 0; }
        if (kv_ok && getenv("SALT_SESSION"))
            kv_prefilled += npids + gen_done;  /* memory grows */
        goto serve_next;
    }
serve_done:
    salt_state_transaction_abort(&state_transaction);
    if (serve_fifo && kv_save_after && kv_ok && kv_prefilled > 0) {
        if (qwen_state_transaction_begin(&state_control, state_model,
                &cfg, &kvc, kv_prefilled, kvlat,
                SALT_STATE_TRANSACTION_EXPORT, SALT_STATE_ARTIFACT_FULL,
                SALT_STATE_REASON_SESSION_CLOSE, &state_transaction) != 0 ||
            qwen_state_save_full(
                kv_save_after, &cfg, &kvc, kv_prefilled, kvlat) != 0 ||
            qwen_state_transaction_finish(state_model, &cfg, &kvc,
                kv_prefilled, kvlat, &state_transaction) != 0) {
            salt_state_transaction_abort(&state_transaction);
            fprintf(stderr, "kv-save-close: engine transaction failed\n");
            rc = 2;
        } else {
            fprintf(stderr, "[kv-save-close] %s (%d tokens)\n",
                    kv_save_after, kv_prefilled);
        }
    }
    if (gpu_prefill_defer_active) {
        if (salt_gpu_set_defer(gpu_prefill_defer_prior) != 0) {
            fprintf(stderr, "gpu-trunk: final defer-policy restore failed\n");
            rc = 2;
        }
        gpu_prefill_defer_active = 0;
    }
    if (gpu_trunk_layer_release(gpu_runtime_trunk, gpu_trunk_intent,
                                &gpu_trunk_map_active) != 0) {
        fprintf(stderr, "gpu-trunk: final synchronized release failed\n");
        gpu_trunk_release_failed = 1;
        rc = 2;
    }
    free(state);
    free(scratch);
    free(xin_buf);
    free(prev_state);
    free(prev_hin);
    free(logits);
    free(pids);
    salt_head_free(&head);
    salt_embed_free(&embed);
    salt_tokenizer_free(&tok);
    if (jscratch) {
        for (int k = 0; k < cfg.topk - 1; k++) free(jscratch[k]);
        free(jscratch);
    }
    if (kv_ok) salt_kv_free(&kvc);
    if (gpu_intent) {
        SaltGpuBatchStats gpu_stats;
        salt_gpu_batch_stats_get(&gpu_stats);
        fprintf(stderr, "gpu-resource: batches expert-layer=%llu trunk=%llu "
                "direct=%llu/%llu arena=%llu mapped-misses=%llu\n",
                (unsigned long long)gpu_stats.expert_layer_batches,
                (unsigned long long)gpu_stats.trunk_batches,
                (unsigned long long)gpu_stats.direct_output_batches,
                (unsigned long long)gpu_stats.direct_output_jobs,
                (unsigned long long)gpu_stats.arena_batches,
                (unsigned long long)gpu_stats.mapped_only_misses);
        if (gpu_trunk_prefill_only && !gpu_prefill_phase_validated) {
            if (!gpu_prefill_stats_captured) {
                if (rc == 0) {
                    fprintf(stderr, "gpu-phase: missing prefill snapshot\n");
                    rc = 2;
                }
            } else if (gpu_prefill_phase_check(
                           gpu_prefill_stats_captured, gpu_prefill_gpu_chunks,
                           gpu_prefill_trunk_batches,
                           gpu_prefill_trunk_after_batches,
                           gpu_prefill_direct_after_batches,
                           gpu_prefill_direct_after_jobs,
                           &gpu_request_start_stats) != 0 && rc == 0) {
                rc = 2;
            }
        }
        if ((gpu_trunk_intent && !gpu_trunk_prefill_only &&
             gpu_stats.trunk_batches == 0) ||
            (gpu_moe_intent && gpu_stats.expert_layer_batches == 0) ||
            ((gpu_trunk_intent || gpu_moe_intent) &&
             (gpu_stats.arena_batches != 0 ||
              gpu_stats.mapped_only_misses != 0))) {
            fprintf(stderr, "gpu-resource: explicit mapped execution did not "
                    "satisfy source-backed ownership\n");
            rc = 2;
        }
        salt_gpu_set_mapped_only(0);
    }
    if (gpu_pool_ok) salt_gpu_pool_close(&gpu_pool);
    if (moe_mode) salt_pool_layout_free(&pl);
    if (cache.accounting || cache.check_enabled || cache.page_probe_enabled ||
        cache.pretouch_enabled)
        salt_cache_report(&cache, stderr);
    if (cache.accounting || cache.check_enabled) {
        if (salt_cache_check(&cache, stderr) != 0 && rc == 0) rc = 2;
    }
    salt_cache_free(&cache);
    if (gpu_intent && !gpu_trunk_release_failed && salt_gpu_free() != 0) {
        fprintf(stderr, "gpu: synchronized teardown failed\n");
        gpu_trunk_release_failed = 1;
        rc = 2;
    }
    if (!gpu_trunk_release_failed) {
        if (gpu_prefill_trunk_open &&
            salt_trunk_close(&gpu_prefill_trunk) != 0) {
            fprintf(stderr, "gpu-trunk: prefill handle close failed\n");
            rc = 2;
        }
        if (salt_trunk_close(&trunk) != 0) {
            fprintf(stderr, "trunk: synchronized close failed\n");
            rc = 2;
        }
    }
    if (salt_pool_close(&pool) != 0) {
        fprintf(stderr, "pool: canonical close failed\n");
        rc = 2;
    } else {
        free(pool.ref);
    }
    return rc;
}
