#define _POSIX_C_SOURCE 200809L

/* attn.c -- MLA attention step + KV cache (issue #6, step 2). */
#include "salt/attn.h"
#include "salt/kernels.h"
#include "salt/moe.h"
#include "salt/bitmath.h"
#include "salt/simd.h"
#include "compiler.h"

#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double attn_soa_now_s(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void attn_soa_count(int enabled, uint64_t *value, uint64_t add) {
    if (!enabled || !value) return;
    *value = *value > UINT64_MAX - add ? UINT64_MAX : *value + add;
}

static int g_weighted_value_simd = 1;

int salt_attn_set_weighted_value_simd(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    g_weighted_value_simd = enabled;
    return 0;
}

int salt_attn_weighted_value_accumulate(
        float *output, const float *values, int value_stride,
        const float *scores, int count, int dimension, float denominator) {
    if (!output || !values || !scores || count < 1 || dimension < 1 ||
        value_stride < dimension || !(denominator > 0.0f) ||
        !isfinite(denominator) ||
        (size_t)(count - 1) >
            (SIZE_MAX - (size_t)dimension) / (size_t)value_stride)
        return -1;
    if (g_weighted_value_simd && salt_kernels_simd()) {
        salt_simd_weighted_value_accumulate(
            output, values, value_stride, scores,
            count, dimension, denominator);
        return 0;
    }
    for (int position = 0; position < count; position++) {
        const float *value = values + (size_t)position * value_stride;
        float weight = scores[position] / denominator;
        for (int d = 0; d < dimension; d++)
            output[d] += weight * value[d];
    }
    return 0;
}

static const float *attn_fold_span_row(
        const SaltAttnHeadFold *fold, int value_row, int position) {
    int offset = position;
    for (int span = 0; span < fold->span_count; span++) {
        const SaltAttnKvSpan *item = &fold->spans[span];
        if (offset < item->count) {
            const float *base = value_row ? item->values : item->keys;
            return base + (size_t)offset * fold->kv_row_stride +
                (size_t)fold->kv_head * fold->head_dim;
        }
        offset -= item->count;
    }
    return NULL;
}

int salt_attn_head_fold(const SaltAttnHeadFold *fold) {
    int span_rows = 0;
    float maximum = -INFINITY, denominator = 0.0f;
    if (!fold || !fold->query || !fold->output || !fold->scores ||
        fold->row_count < 1 || fold->head_dim < 1 || fold->kv_head < 0 ||
        !isfinite(fold->score_scale) ||
        ((fold->spans == NULL) == (fold->row_resolver == NULL)))
        return -1;
    if (fold->spans) {
        if (fold->span_count < 1 ||
            fold->span_count > SALT_ATTN_HEAD_MAX_SPANS ||
            fold->kv_row_stride < fold->head_dim)
            return -1;
        for (int span = 0; span < fold->span_count; span++) {
            const SaltAttnKvSpan *item = &fold->spans[span];
            if (!item->keys || !item->values || item->count < 1 ||
                item->count > fold->row_count - span_rows)
                return -1;
            span_rows += item->count;
        }
        if (span_rows != fold->row_count) return -1;
    } else if (fold->span_count != 0 || fold->kv_row_stride != 0) {
        return -1;
    }
    for (int position = 0; position < fold->row_count; position++) {
        const float *key = fold->spans
            ? attn_fold_span_row(fold, 0, position)
            : fold->row_resolver(fold->row_context, 0, position,
                                 fold->kv_head, fold->head_dim);
        float score = 0.0f;
        if (!key) return -1;
        for (int d = 0; d < fold->head_dim; d++)
            score += fold->query[d] * key[d];
        score *= fold->score_scale;
        fold->scores[position] = score;
        if (score > maximum) maximum = score;
    }
    for (int position = 0; position < fold->row_count; position++) {
        fold->scores[position] =
            salt_expf(fold->scores[position] - maximum);
        denominator += fold->scores[position];
    }
    if (!(denominator > 0.0f) || !isfinite(denominator)) return -1;
    memset(fold->output, 0, (size_t)fold->head_dim * sizeof(float));
    if (fold->spans) {
        int score_offset = 0;
        for (int span = 0; span < fold->span_count; span++) {
            const SaltAttnKvSpan *item = &fold->spans[span];
            const float *values = item->values +
                (size_t)fold->kv_head * fold->head_dim;
            if (salt_attn_weighted_value_accumulate(
                    fold->output, values, fold->kv_row_stride,
                    fold->scores + score_offset, item->count,
                    fold->head_dim, denominator) != 0)
                return -1;
            score_offset += item->count;
        }
    } else {
        for (int position = 0; position < fold->row_count; position++) {
            const float *value = fold->row_resolver(
                fold->row_context, 1, position,
                fold->kv_head, fold->head_dim);
            float weight = fold->scores[position] / denominator;
            if (!value) return -1;
            for (int d = 0; d < fold->head_dim; d++)
                fold->output[d] += weight * value[d];
        }
    }
    return 0;
}

enum { SALT_ATTN_HEAD_MAX_WORKERS = 256 };

typedef struct SaltAttnHeadTask {
    SaltKvCache *pool;
    const SaltAttnHeadOperation *operation;
    int workers;
    int failed[SALT_ATTN_HEAD_MAX_WORKERS];
} SaltAttnHeadTask;

static int g_head_parallel = 1;

int salt_attn_set_head_parallel(int enabled) {
    if (enabled != 0 && enabled != 1) return -1;
    g_head_parallel = enabled;
    return 0;
}

static int attn_head_operation_validate(
        const SaltKvCache *pool, const SaltAttnHeadOperation *operation,
        int *total_rows) {
    int rows = 0;
    if (!pool || !operation || !total_rows || !pool->apool_sync_init ||
        !pool->ath || !pool->asc8 || pool->apool_threads < 1 ||
        pool->apool_threads > SALT_ATTN_HEAD_MAX_WORKERS ||
        pool->max_tokens < 1 || !operation->queries || !operation->outputs ||
        operation->head_count < 1 || operation->head_dim < 1 ||
        operation->kv_heads < 1 ||
        operation->head_count % operation->kv_heads != 0 ||
        operation->query_stride < operation->head_dim ||
        operation->output_stride < operation->head_dim ||
        operation->head_dim > INT_MAX / operation->kv_heads ||
        operation->kv_row_stride < operation->head_dim * operation->kv_heads ||
        operation->span_count < 1 ||
        operation->span_count > SALT_ATTN_HEAD_MAX_SPANS)
        return -1;
    if ((size_t)(operation->head_count - 1) >
            (SIZE_MAX - (size_t)operation->head_dim) /
                (size_t)operation->query_stride ||
        (size_t)(operation->head_count - 1) >
            (SIZE_MAX - (size_t)operation->head_dim) /
                (size_t)operation->output_stride)
        return -1;
    for (int span = 0; span < operation->span_count; span++) {
        const SaltAttnKvSpan *item = &operation->spans[span];
        if (!item->keys || !item->values || item->count < 1 ||
            item->count > pool->max_tokens - rows)
            return -1;
        rows += item->count;
    }
    *total_rows = rows;
    return 0;
}

static void attn_head_worker(int worker, void *opaque) {
    SaltAttnHeadTask *task = (SaltAttnHeadTask *)opaque;
    const SaltAttnHeadOperation *operation;
    float *scores;
    int groups, total_rows = 0;
    if (!task || !task->pool || !(operation = task->operation) ||
        worker < 0 || worker >= task->workers) {
        if (task && worker >= 0 && worker < SALT_ATTN_HEAD_MAX_WORKERS)
            task->failed[worker] = 1;
        return;
    }
    for (int span = 0; span < operation->span_count; span++)
        total_rows += operation->spans[span].count;
    scores = task->pool->asc8 +
        (size_t)worker * 8u * (size_t)task->pool->max_tokens;
    groups = operation->head_count / operation->kv_heads;
    for (int head = worker; head < operation->head_count;
            head += task->workers) {
        SaltAttnHeadFold fold;
        memset(&fold, 0, sizeof fold);
        fold.query = operation->queries +
            (size_t)head * operation->query_stride;
        fold.output = operation->outputs +
            (size_t)head * operation->output_stride;
        fold.scores = scores;
        fold.row_count = total_rows;
        fold.head_dim = operation->head_dim;
        fold.kv_head = head / groups;
        fold.score_scale = 1.0f;
        fold.kv_row_stride = operation->kv_row_stride;
        fold.spans = operation->spans;
        fold.span_count = operation->span_count;
        fold.span_accumulate = salt_attn_weighted_value_accumulate;
        if (salt_attn_head_fold(&fold) != 0) {
            task->failed[worker] = 1;
            return;
        }
    }
}

int salt_attn_heads_execute(
        SaltKvCache *pool, const SaltAttnHeadOperation *operation) {
    SaltAttnHeadTask task;
    int total_rows, workers;
    if (attn_head_operation_validate(pool, operation, &total_rows) != 0)
        return -1;
    (void)total_rows;
    memset(&task, 0, sizeof task);
    task.pool = pool;
    task.operation = operation;
    workers = pool->apool_threads;
    if (workers > operation->head_count) workers = operation->head_count;
    if (!g_head_parallel || workers == 1) {
        task.workers = 1;
        attn_head_worker(0, &task);
        return task.failed[0] ? -1 : 0;
    }
    task.workers = workers;
    if (salt_attn_pool_run_n(
            pool, workers, attn_head_worker, &task) != 0)
        return -1;
    for (int worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    return 0;
}

static float bf16_f(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* in-place RMSNorm with BF16 weights, eps 1e-6 */
static void rmsnorm(const uint16_t *w, int dim, float *x) {
    double ss = 0.0;
    for (int i = 0; i < dim; i++) ss += (double)x[i] * x[i];
    float r = sqrtf((float)(ss / (double)dim) + 1e-6f);
    for (int i = 0; i < dim; i++) x[i] = x[i] / r * bf16_f(w[i]);
}

int salt_kv_init(SaltKvCache *c, int n_layers, int kvlat, int max_tokens) {
    memset(c, 0, sizeof *c);
    if (n_layers < 1 || kvlat < 1 || max_tokens < 1) return -1;
    c->kv = (float *)calloc(
        (size_t)n_layers * (size_t)max_tokens * (size_t)kvlat,
        sizeof(float));
    if (!c->kv) return -1;
    c->n_layers = n_layers;
    c->kvlat = kvlat;
    c->max_tokens = max_tokens;
    return 0;
}

/* Per-token work arena for the attention paths: allocated ONCE with the
 * exact worst-case float count (computed by the caller from the trunk
 * dims). Removes ~1.1M malloc/free per 20K-token run -- macOS keeps
 * freed large blocks at the zone high-water (~2.9 GB resident empty
 * zones observed). */
int salt_kv_scratch_init(SaltKvCache *c, long floats) {
    if (!c || c->scratch || floats < 1 ||
        (size_t)floats > SIZE_MAX / sizeof(float))
        return -1;
    c->scratch = (float *)malloc((size_t)floats * sizeof(float));
    if (!c->scratch) return -1;
    c->scratch_n = floats;
    c->scratch_owned = 1;
    return 0;
}

int salt_kv_scratch_bind(SaltKvCache *c, float *scratch, long floats) {
    if (!c || !scratch || floats < 1 ||
        (size_t)floats > SIZE_MAX / sizeof(float) ||
        c->scratch)
        return -1;
    c->scratch = scratch;
    c->scratch_n = floats;
    c->scratch_owned = 0;
    return 0;
}

int salt_kv_scratch_unbind(SaltKvCache *c, float *scratch) {
    if (!c || !scratch || c->scratch_owned || c->scratch != scratch)
        return -1;
    c->scratch = NULL;
    c->scratch_n = 0;
    return 0;
}

static void attn_pool_teardown(SaltKvCache *c) {
    if (!c) return;
    if (c->apool_sync_init && c->ath_count > 0) {
        pthread_mutex_lock(&c->amu);
        c->ashutdown = 1;
        pthread_cond_broadcast(&c->acv_work);
        pthread_cond_broadcast(&c->acv_wide);
        pthread_mutex_unlock(&c->amu);
    }
    for (int i = 0; i < c->ath_count; i++)
        pthread_join(c->ath[i], NULL);
    free(c->ath);
    free(c->asc8);
    free(c->awg8);
    c->ath = NULL;
    c->asc8 = NULL;
    c->awg8 = NULL;
    c->ath_count = 0;
    if (c->apool_sync_init) {
        pthread_cond_destroy(&c->acv_flow_work);
        pthread_cond_destroy(&c->acv_flow);
        pthread_cond_destroy(&c->acv_work);
        pthread_cond_destroy(&c->acv_wide);
        pthread_cond_destroy(&c->acv_done);
        pthread_mutex_destroy(&c->amu);
        c->apool_sync_init = 0;
    }
}

void salt_kv_free(SaltKvCache *c) {
    if (!c) return;
    attn_pool_teardown(c);
    free(c->kv);
    free(c->lin);
    free(c->conv);
    if (c->scratch_owned) free(c->scratch);
    memset(c, 0, sizeof *c);
}

/* Forget policy (session mode): drop the exact per-token K/V rows
 * and the conv ring (the transcript), KEEP the linear delta states
 * (the compressed memory -- they carry the accumulated history and
 * are fixed-size by design). The next token starts fresh rows at
 * position 0; the conv ring rebuilds in Q3_CONV_K tokens. The
 * caller resets its kv_prefilled counter. */
void salt_kv_forget(SaltKvCache *c) {
    if (!c) return;
    if (c->kv)
        memset(c->kv, 0,
               (size_t)c->n_layers * (size_t)c->max_tokens *
                   (size_t)c->kvlat * sizeof(float));
    if (c->conv)
        memset(c->conv, 0, 4UL * (size_t)c->conv_rows * sizeof(float));
    /* c->lin (the delta states) is deliberately NOT touched: that is
     * the compressed memory that survives the forget. */
}

/* ---- persistent attention pool ---------------------------------- */
/* Batch runner: a submit stores afn/aarg, signals the selected worker
 * prefix, and waits for that complete prefix. Base workers and optional
 * wide workers use separate condition variables so ordinary attention
 * does not wake the extra Q4 projection workers. */

/* per-thread create arg: the pool passes the tid explicitly so no
 * worker has to scan ath[] for pthread_self() (threads start before
 * the array is fully populated -- the scan was racy). */
typedef struct { SaltKvCache *c; int tid; } AttnPoolArg;

static void attn_pool_graph_worker(int tid, void *opaque) {
    SaltKvCache *c = (SaltKvCache *)opaque;
    uint64_t seen = 0;
    pthread_cond_t *phase_cv = tid < c->nthreads
        ? &c->acv_work : &c->acv_wide;
    pthread_mutex_lock(&c->amu);
    c->agraph_ready++;
    pthread_cond_broadcast(&c->acv_done);
    for (;;) {
        while (!c->ashutdown && !c->agraph_stop &&
               (c->agraph_phase == seen ||
                tid >= c->agraph_phase_active))
            pthread_cond_wait(phase_cv, &c->amu);
        if (c->ashutdown || c->agraph_stop) {
            pthread_mutex_unlock(&c->amu);
            return;
        }
        seen = c->agraph_phase;
        void (*fn)(int, void *) = c->agraph_fn;
        void *arg = c->agraph_arg;
        double action_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        pthread_mutex_unlock(&c->amu);
        fn(tid, arg);
        double action_s = c->asoa_profile
            ? attn_soa_now_s() - action_started : 0.0;
        pthread_mutex_lock(&c->amu);
        if (c->asoa_profile) {
            c->asoa_worker_action_sum_s += action_s;
            if (action_s > c->asoa_phase_worker_max_s)
                c->asoa_phase_worker_max_s = action_s;
        }
        c->agraph_phase_done++;
        if (c->agraph_phase_done >= c->agraph_phase_active)
            pthread_cond_broadcast(&c->acv_done);
    }
}

SALT_THREAD_LOCAL SaltKvCache *attn_team_pool;
SALT_THREAD_LOCAL int attn_team_tid;
SALT_THREAD_LOCAL int attn_team_in_phase;

/* SC publication plus SC park registration prevents the lost-wakeup case:
 * either the publisher sees a sleeper, or that sleeper sees the new phase
 * on its mutex-protected second check and never waits. No dynamic storage. */
static void attn_team_wake(SaltKvCache *c) {
    if (__atomic_load_n(&c->ateam_sleepers, __ATOMIC_SEQ_CST) != 0) {
        pthread_mutex_lock(&c->amu);
        pthread_cond_broadcast(&c->acv_flow_work);
        pthread_mutex_unlock(&c->amu);
    }
}

static uint64_t attn_team_next(SaltKvCache *c, uint64_t seen) {
    uint64_t phase;
    for (unsigned spin = 0; spin < 4096u; spin++) {
        phase = __atomic_load_n(&c->aflow_phase, __ATOMIC_SEQ_CST);
        if (phase != seen) return phase;
    }
    pthread_mutex_lock(&c->amu);
    __atomic_add_fetch(&c->ateam_sleepers, 1, __ATOMIC_SEQ_CST);
    while ((phase = __atomic_load_n(&c->aflow_phase, __ATOMIC_SEQ_CST)) == seen)
        pthread_cond_wait(&c->acv_flow_work, &c->amu);
    __atomic_sub_fetch(&c->ateam_sleepers, 1, __ATOMIC_SEQ_CST);
    pthread_mutex_unlock(&c->amu);
    return phase;
}

static void attn_team_profile_action(SaltKvCache *c, double begin) {
    if (c->asoa_profile) {
        double elapsed = attn_soa_now_s() - begin;
        pthread_mutex_lock(&c->amu);
        c->asoa_worker_action_sum_s += elapsed;
        if (elapsed > c->asoa_phase_worker_max_s)
            c->asoa_phase_worker_max_s = elapsed;
        pthread_mutex_unlock(&c->amu);
    }
}

static void attn_pool_team_worker(int tid, SaltKvCache *c) {
    uint64_t seen = 0;
    attn_team_pool = c;
    attn_team_tid = tid;
    attn_team_in_phase = 0;
    if (tid == 0) {
        int result;
        pthread_mutex_lock(&c->amu);
        c->aflow_coordinator = pthread_self();
        pthread_mutex_unlock(&c->amu);
        result = c->aflow_coordinator_fn(c->aflow_coordinator_arg);
        /* Every synchronous phase has drained before returning, including
         * failure. Stop is another publication, never an early worker exit. */
        __atomic_store_n(&c->aflow_phase, UINT64_MAX, __ATOMIC_SEQ_CST);
        attn_team_wake(c);
        pthread_mutex_lock(&c->amu);
        c->aflow_result = result;
        c->aflow_stop = 1;
        pthread_mutex_unlock(&c->amu);
    } else {
        for (;;) {
            uint64_t phase = attn_team_next(c, seen);
            double started;
            if (phase == UINT64_MAX) break;
            seen = phase;
            started = c->asoa_profile ? attn_soa_now_s() : 0.0;
            if (tid < c->aflow_phase_active) {
                attn_team_in_phase = 1;
                c->aflow_phase_fn(tid, c->aflow_phase_arg);
                attn_team_in_phase = 0;
                attn_team_profile_action(c, started);
            }
            /* Inactive members acknowledge too: the next phase must not
             * overwrite shared descriptors while any member can read them. */
            if (__atomic_add_fetch(&c->aflow_phase_done, 1, __ATOMIC_SEQ_CST)
                    == c->apool_threads - 1 &&
                __atomic_load_n(&c->ateam_join_sleeping, __ATOMIC_SEQ_CST)) {
                pthread_mutex_lock(&c->amu);
                pthread_cond_signal(&c->acv_flow);
                pthread_mutex_unlock(&c->amu);
            }
        }
    }
    attn_team_pool = NULL;
    attn_team_in_phase = 0;
}

static int attn_pool_team_phase(SaltKvCache *c, int nactive,
        void (*fn)(int, void *), void *arg) {
    uint64_t phase;
    double started;
    if (attn_team_pool != c || attn_team_tid != 0 || attn_team_in_phase ||
        !fn || nactive < 1 || nactive > c->apool_threads)
        return -1;
    phase = __atomic_load_n(&c->aflow_phase, __ATOMIC_SEQ_CST);
    if (phase >= UINT64_MAX - 1u) return -1;
    c->aflow_phase_fn = fn;
    c->aflow_phase_arg = arg;
    c->aflow_phase_active = nactive;
    __atomic_store_n(&c->aflow_phase_done, 0, __ATOMIC_SEQ_CST);
    if (c->asoa_profile) c->asoa_phase_worker_max_s = 0.0;
    attn_soa_count(c->asoa_profile, &c->asoa_phase_actions, 1);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_actions, (uint64_t)(unsigned)nactive);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_capacity,
                   (uint64_t)(unsigned)c->apool_threads);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_headroom,
                   (uint64_t)(unsigned)(c->apool_threads - nactive));
    if (c->asoa_profile && (uint32_t)nactive > c->asoa_peak_active)
        c->asoa_peak_active = (uint32_t)nactive;
    __atomic_store_n(&c->aflow_phase, phase + 1u, __ATOMIC_SEQ_CST);
    attn_team_wake(c);
    started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    attn_team_in_phase = 1;
    fn(0, arg);
    attn_team_in_phase = 0;
    attn_team_profile_action(c, started);
    for (unsigned spin = 0; spin < 4096u; spin++)
        if (__atomic_load_n(&c->aflow_phase_done, __ATOMIC_SEQ_CST)
                == c->apool_threads - 1) break;
    if (__atomic_load_n(&c->aflow_phase_done, __ATOMIC_SEQ_CST)
            != c->apool_threads - 1) {
        pthread_mutex_lock(&c->amu);
        __atomic_store_n(&c->ateam_join_sleeping, 1, __ATOMIC_SEQ_CST);
        while (__atomic_load_n(&c->aflow_phase_done, __ATOMIC_SEQ_CST)
                != c->apool_threads - 1)
            pthread_cond_wait(&c->acv_flow, &c->amu);
        __atomic_store_n(&c->ateam_join_sleeping, 0, __ATOMIC_SEQ_CST);
        pthread_mutex_unlock(&c->amu);
    }
    if (c->asoa_profile)
        c->asoa_worker_critical_s += c->asoa_phase_worker_max_s;
    c->aflow_phase_fn = NULL;
    c->aflow_phase_arg = NULL;
    c->aflow_phase_active = 0;
    return 0;
}

static void attn_pool_flow_worker(int tid, void *opaque) {
    SaltKvCache *c = (SaltKvCache *)opaque;
    if (__atomic_load_n(&c->ateam_mode, __ATOMIC_ACQUIRE)) {
        attn_pool_team_worker(tid, c);
        return;
    }
    pthread_cond_t *phase_cv = &c->acv_flow_work;
    uint64_t seen = 0;
    if (tid == 0) {
        int (*coordinator_fn)(void *);
        void *coordinator_arg;
        int result;
        pthread_mutex_lock(&c->amu);
        c->aflow_coordinator = pthread_self();
        coordinator_fn = c->aflow_coordinator_fn;
        coordinator_arg = c->aflow_coordinator_arg;
        pthread_mutex_unlock(&c->amu);
        result = coordinator_fn ? coordinator_fn(coordinator_arg) : -1;
        if (c->aflow_lock_held)
            c->aflow_lock_held = 0;
        else
            pthread_mutex_lock(&c->amu);
        c->aflow_result = result;
        c->aflow_stop = 1;
        pthread_cond_broadcast(&c->acv_flow_work);
        attn_soa_count(c->asoa_profile,
                       &c->asoa_internal_signal_calls, 1);
        pthread_mutex_unlock(&c->amu);
        return;
    }
    pthread_mutex_lock(&c->amu);
    for (;;) {
        while (!c->ashutdown && !c->aflow_stop &&
               (c->aflow_phase == seen || tid >= c->aflow_phase_active))
            pthread_cond_wait(phase_cv, &c->amu);
        if (c->ashutdown || c->aflow_stop) {
            pthread_mutex_unlock(&c->amu);
            return;
        }
        seen = c->aflow_phase;
        void (*fn)(int, void *) = c->aflow_phase_fn;
        void *arg = c->aflow_phase_arg;
        double action_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        pthread_mutex_unlock(&c->amu);
        fn(tid, arg);
        double action_s = c->asoa_profile
            ? attn_soa_now_s() - action_started : 0.0;
        pthread_mutex_lock(&c->amu);
        if (c->asoa_profile) {
            c->asoa_worker_action_sum_s += action_s;
            if (action_s > c->asoa_phase_worker_max_s)
                c->asoa_phase_worker_max_s = action_s;
        }
        c->aflow_phase_done++;
        if (c->aflow_phase_done >= c->aflow_phase_active - 1)
            pthread_cond_broadcast(&c->acv_flow);
    }
}

static void *attn_pool_worker(void *p) {
    AttnPoolArg *pa = (AttnPoolArg *)p;
    SaltKvCache *c = pa->c;
    int tid = pa->tid;
    free(pa);
    uint64_t last = 0;        /* last generation this worker ran */
    pthread_cond_t *work_cv = tid < c->nthreads
        ? &c->acv_work : &c->acv_wide;
    for (;;) {
        pthread_mutex_lock(&c->amu);
        while (!c->ashutdown &&
               (c->abatch == last || tid >= c->aactive))
            pthread_cond_wait(work_cv, &c->amu);
        if (c->ashutdown) { pthread_mutex_unlock(&c->amu); return NULL; }
        last = c->abatch;                 /* claim THIS generation */
        void (*fn)(int, void *) = c->afn;
        void *arg = c->aarg;
        int control_dispatch = fn == attn_pool_graph_worker ||
            fn == attn_pool_flow_worker;
        double action_started = c->asoa_profile && !control_dispatch
            ? attn_soa_now_s() : 0.0;
        pthread_mutex_unlock(&c->amu);

        fn(tid, arg);

        double action_s = c->asoa_profile && !control_dispatch
            ? attn_soa_now_s() - action_started : 0.0;
        pthread_mutex_lock(&c->amu);
        if (c->asoa_profile && !control_dispatch) {
            c->asoa_worker_action_sum_s += action_s;
            if (action_s > c->asoa_phase_worker_max_s)
                c->asoa_phase_worker_max_s = action_s;
        }
        c->adone++;
        if (c->adone >= c->aactive)
            pthread_cond_broadcast(&c->acv_done);
        pthread_mutex_unlock(&c->amu);
    }
}

int salt_attn_pool_init(SaltKvCache *c) {
    const char *soa_profile;
    if (!c) return -1;
    if (c->apool_sync_init || c->ath || c->asc8 || c->awg8)
        return -1;
    if (c->nthreads < 1) return 0;
    soa_profile = getenv("SALT_CPU_SOA");
    if (soa_profile && strcmp(soa_profile, "0") != 0 &&
        strcmp(soa_profile, "1") != 0)
        return -1;
    c->asoa_profile = soa_profile && strcmp(soa_profile, "1") == 0;
    if (c->aq4_threads < 1) {
        int q4_threads = 8;
        const char *e = getenv("SALT_ATTN_THREADS");
        if (e) {
            int v = atoi(e);
            if (v >= 1 && v <= 32) q4_threads = v;
        }
        c->aq4_threads = q4_threads;
    }
    if (c->aq4_threads < 1 || c->aq4_threads > 32) return -1;
    {
        const char *e = getenv("SALT_Q4_MATVEC2_POOL");
        c->aq4_pool_enabled = (!e || *e != '0');
    }
    if (c->apool_threads < 1) {
        int workers = c->nthreads;
        if (c->aq4_pool_enabled && c->aq4_threads > workers)
            workers = c->aq4_threads;
        c->apool_threads = workers;
    }
    if (c->apool_threads < c->nthreads || c->apool_threads > 256 ||
        (c->aq4_pool_enabled && c->apool_threads < c->aq4_threads))
        return -1;
    c->ashutdown = 0;
    c->abatch = 0;
    c->adone = 0;
    c->arunning = 0;
    c->ath_count = 0;
    if (pthread_mutex_init(&c->amu, NULL) != 0) return -1;
    c->ateam_mode = 0;
    c->ateam_sleepers = 0;
    c->ateam_join_sleeping = 0;
    if (pthread_cond_init(&c->acv_work, NULL) != 0) {
        pthread_mutex_destroy(&c->amu);
        return -1;
    }
    if (pthread_cond_init(&c->acv_wide, NULL) != 0) {
        pthread_cond_destroy(&c->acv_work);
        pthread_mutex_destroy(&c->amu);
        return -1;
    }
    if (pthread_cond_init(&c->acv_done, NULL) != 0) {
        pthread_cond_destroy(&c->acv_wide);
        pthread_cond_destroy(&c->acv_work);
        pthread_mutex_destroy(&c->amu);
        return -1;
    }
    if (pthread_cond_init(&c->acv_flow, NULL) != 0) {
        pthread_cond_destroy(&c->acv_done);
        pthread_cond_destroy(&c->acv_wide);
        pthread_cond_destroy(&c->acv_work);
        pthread_mutex_destroy(&c->amu);
        return -1;
    }
    if (pthread_cond_init(&c->acv_flow_work, NULL) != 0) {
        pthread_cond_destroy(&c->acv_flow);
        pthread_cond_destroy(&c->acv_done);
        pthread_cond_destroy(&c->acv_wide);
        pthread_cond_destroy(&c->acv_work);
        pthread_mutex_destroy(&c->amu);
        return -1;
    }
    c->apool_sync_init = 1;
    /* persistent per-thread score/weight buffers: 8 heads x max_tokens
     * (worst case is the K-reuse worker's 8*cap; the plain worker's
     * 1*cap is a subset). */
    if (c->max_tokens < 1 || (size_t)c->max_tokens > SIZE_MAX / 8) {
        attn_pool_teardown(c);
        return -1;
    }
    size_t per = (size_t)8 * (size_t)c->max_tokens;
    if ((size_t)c->apool_threads > SIZE_MAX / per / sizeof(float)) {
        attn_pool_teardown(c);
        return -1;
    }
    c->asc8 = (float *)calloc((size_t)c->apool_threads * per,
                              sizeof(float));
    c->awg8 = (float *)calloc((size_t)c->apool_threads * per,
                              sizeof(float));
    if (!c->asc8 || !c->awg8) {
        attn_pool_teardown(c);
        return -1;
    }
    c->ath = (pthread_t *)calloc((size_t)c->apool_threads,
                                 sizeof(pthread_t));
    if (!c->ath) {
        attn_pool_teardown(c);
        return -1;
    }
    for (int i = 0; i < c->apool_threads; i++) {
        AttnPoolArg *pa = (AttnPoolArg *)malloc(sizeof(AttnPoolArg));
        if (!pa) {
            attn_pool_teardown(c);
            return -1;
        }
        pa->c = c;
        pa->tid = i;
        if (pthread_create(&c->ath[i], NULL, attn_pool_worker, pa) != 0) {
            free(pa);
            attn_pool_teardown(c);
            return -1;
        }
        c->ath_count++;
    }
    return 0;
}

/* Submit a batch to the first nactive resident workers. Inactive workers
 * leave their last-generation value unchanged, so a later wider submission
 * executes exactly once for its own generation. A second or recursive
 * submitter is rejected while the first generation is active. */
static int attn_pool_flow_phase_locked(
        SaltKvCache *c, int nactive,
        void (*fn)(int tid, void *arg), void *arg,
        double lock_started, int new_lock) {
    double action_started, action_s, wait_started;
    if (!c->aflow_running || c->aflow_stop || c->ashutdown ||
        !pthread_equal(c->aflow_coordinator, pthread_self()) ||
        c->aflow_phase == UINT64_MAX || c->aflow_phase_active != 0 ||
        c->aflow_phase_done != 0) {
        attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
        pthread_mutex_unlock(&c->amu);
        return -1;
    }
    attn_soa_count(c->asoa_profile,
                   &c->asoa_internal_lock_acquires,
                   new_lock ? 1u : 0u);
    if (c->asoa_profile && new_lock)
        c->asoa_internal_lock_wait_s += attn_soa_now_s() - lock_started;
    c->aflow_phase_fn = fn;
    c->aflow_phase_arg = arg;
    c->aflow_phase_active = nactive;
    c->aflow_phase_done = 0;
    if (c->asoa_profile) c->asoa_phase_worker_max_s = 0.0;
    c->aflow_phase++;
    attn_soa_count(c->asoa_profile, &c->asoa_phase_actions, 1);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_actions,
                   (uint64_t)(uint32_t)nactive);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_capacity,
                   (uint64_t)(uint32_t)c->apool_threads);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_headroom,
                   (uint64_t)(uint32_t)(c->apool_threads - nactive));
    if (c->asoa_profile && (uint32_t)nactive > c->asoa_peak_active)
        c->asoa_peak_active = (uint32_t)nactive;
    if (nactive > 1) {
        pthread_cond_broadcast(&c->acv_flow_work);
        attn_soa_count(c->asoa_profile,
                       &c->asoa_internal_signal_calls, 1);
    }
    action_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    pthread_mutex_unlock(&c->amu);
    fn(0, arg);
    action_s = c->asoa_profile ? attn_soa_now_s() - action_started : 0.0;
    pthread_mutex_lock(&c->amu);
    if (c->asoa_profile) {
        c->asoa_worker_action_sum_s += action_s;
        if (action_s > c->asoa_phase_worker_max_s)
            c->asoa_phase_worker_max_s = action_s;
    }
    while (c->aflow_phase_done < nactive - 1) {
        wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile,
                       &c->asoa_internal_wait_calls, 1);
        pthread_cond_wait(&c->acv_flow, &c->amu);
        if (c->asoa_profile)
            c->asoa_internal_wait_s += attn_soa_now_s() - wait_started;
    }
    if (c->asoa_profile)
        c->asoa_worker_critical_s += c->asoa_phase_worker_max_s;
    c->aflow_phase_fn = NULL;
    c->aflow_phase_arg = NULL;
    c->aflow_phase_active = 0;
    c->aflow_phase_done = 0;
    c->aflow_lock_held = 1;
    return 0;
}

int salt_attn_pool_run_n(SaltKvCache *c, int nactive,
                         void (*fn)(int tid, void *arg), void *arg) {
    double lock_started, wait_started;
    if (!c || !c->apool_sync_init || !c->ath || !fn || nactive < 1 ||
        nactive > c->apool_threads)
        return -1;
    if (__atomic_load_n(&c->ateam_mode, __ATOMIC_ACQUIRE))
        return attn_pool_team_phase(c, nactive, fn, arg);
    if (nactive == 1 && c->aflow_running && !c->aflow_stop &&
        pthread_equal(c->aflow_coordinator, pthread_self())) {
        double action_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile, &c->asoa_phase_actions, 1);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_actions, 1);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_capacity,
                       (uint64_t)(uint32_t)c->apool_threads);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_headroom,
                       (uint64_t)(uint32_t)(c->apool_threads - 1));
        if (c->asoa_profile && c->asoa_peak_active == 0)
            c->asoa_peak_active = 1;
        fn(0, arg);
        if (c->asoa_profile) {
            double action_s = attn_soa_now_s() - action_started;
            c->asoa_worker_action_sum_s += action_s;
            c->asoa_worker_critical_s += action_s;
        }
        return 0;
    }
    if (c->aflow_running && c->aflow_lock_held &&
        pthread_equal(c->aflow_coordinator, pthread_self()))
        return attn_pool_flow_phase_locked(
            c, nactive, fn, arg, 0.0, 0);
    lock_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    pthread_mutex_lock(&c->amu);
    if (c->aflow_running)
        return attn_pool_flow_phase_locked(
            c, nactive, fn, arg, lock_started, 1);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_lock_acquires, 1);
    if (c->asoa_profile)
        c->asoa_flow_lock_wait_s += attn_soa_now_s() - lock_started;
    if (c->agraph_running) {
        if (!pthread_equal(c->agraph_owner, pthread_self()) ||
            c->ashutdown || c->agraph_stop || !c->arunning ||
            c->agraph_phase == UINT64_MAX ||
            c->agraph_phases == UINT64_MAX ||
            c->agraph_phase_active != 0 || c->agraph_phase_done != 0) {
            attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
            pthread_mutex_unlock(&c->amu);
            return -1;
        }
        c->agraph_fn = fn;
        c->agraph_arg = arg;
        c->agraph_phase_active = nactive;
        c->agraph_phase_done = 0;
        if (c->asoa_profile) c->asoa_phase_worker_max_s = 0.0;
        c->agraph_phase++;
        c->agraph_phases++;
        attn_soa_count(c->asoa_profile, &c->asoa_phase_actions, 1);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_actions, (uint64_t)(uint32_t)nactive);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_capacity,
                       (uint64_t)(uint32_t)c->apool_threads);
        attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_headroom,
                       (uint64_t)(uint32_t)(c->apool_threads - nactive));
        if (c->asoa_profile && (uint32_t)nactive > c->asoa_peak_active)
            c->asoa_peak_active = (uint32_t)nactive;
        pthread_cond_broadcast(&c->acv_work);
        attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
        if (nactive > c->nthreads)
            pthread_cond_broadcast(&c->acv_wide);
        if (nactive > c->nthreads)
            attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
        while (c->agraph_phase_done < c->agraph_phase_active) {
            wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
            attn_soa_count(c->asoa_profile, &c->asoa_flow_wait_calls, 1);
            pthread_cond_wait(&c->acv_done, &c->amu);
            if (c->asoa_profile)
                c->asoa_flow_wait_s += attn_soa_now_s() - wait_started;
        }
        if (c->asoa_profile)
            c->asoa_worker_critical_s += c->asoa_phase_worker_max_s;
        c->agraph_fn = NULL;
        c->agraph_arg = NULL;
        c->agraph_phase_active = 0;
        c->agraph_phase_done = 0;
        pthread_mutex_unlock(&c->amu);
        return 0;
    }
    if (c->arunning || c->ashutdown || c->abatch == UINT64_MAX) {
        attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
        pthread_mutex_unlock(&c->amu);
        return -1;
    }
    c->arunning = 1;
    c->afn = fn;
    c->aarg = arg;
    c->aactive = nactive;
    c->adone = 0;
    if (c->asoa_profile) c->asoa_phase_worker_max_s = 0.0;
    c->abatch++;
    attn_soa_count(c->asoa_profile, &c->asoa_phase_actions, 1);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_actions, (uint64_t)(uint32_t)nactive);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_capacity,
                   (uint64_t)(uint32_t)c->apool_threads);
    attn_soa_count(c->asoa_profile, &c->asoa_worker_slot_headroom,
                   (uint64_t)(uint32_t)(c->apool_threads - nactive));
    if (c->asoa_profile && (uint32_t)nactive > c->asoa_peak_active)
        c->asoa_peak_active = (uint32_t)nactive;
    pthread_cond_broadcast(&c->acv_work);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
    if (nactive > c->nthreads)
        pthread_cond_broadcast(&c->acv_wide);
    if (nactive > c->nthreads)
        attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
    while (c->adone < c->aactive) {
        wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile, &c->asoa_flow_wait_calls, 1);
        pthread_cond_wait(&c->acv_done, &c->amu);
        if (c->asoa_profile)
            c->asoa_flow_wait_s += attn_soa_now_s() - wait_started;
    }
    if (c->asoa_profile)
        c->asoa_worker_critical_s += c->asoa_phase_worker_max_s;
    c->afn = NULL;
    c->aarg = NULL;
    c->aactive = 0;
    c->arunning = 0;
    pthread_mutex_unlock(&c->amu);
    return 0;
}

int salt_attn_pool_graph_begin(SaltKvCache *c) {
    double lock_started, wait_started;
    if (!c || !c->apool_sync_init || !c->ath || c->apool_threads < 1)
        return -1;
    lock_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    pthread_mutex_lock(&c->amu);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_lock_acquires, 1);
    if (c->asoa_profile)
        c->asoa_flow_lock_wait_s += attn_soa_now_s() - lock_started;
    if (c->arunning || c->agraph_running || c->ashutdown ||
        c->abatch == UINT64_MAX || c->agraph_sessions == UINT64_MAX) {
        attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
        pthread_mutex_unlock(&c->amu);
        return -1;
    }
    c->arunning = 1;
    c->agraph_running = 1;
    c->agraph_stop = 0;
    c->agraph_ready = 0;
    c->agraph_phase_done = 0;
    c->agraph_phase_active = 0;
    c->agraph_phase = 0;
    c->agraph_fn = NULL;
    c->agraph_arg = NULL;
    c->agraph_owner = pthread_self();
    c->afn = attn_pool_graph_worker;
    c->aarg = c;
    c->aactive = c->apool_threads;
    c->adone = 0;
    c->abatch++;
    c->agraph_sessions++;
    pthread_cond_broadcast(&c->acv_work);
    pthread_cond_broadcast(&c->acv_wide);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 2);
    while (c->agraph_ready < c->aactive) {
        wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile, &c->asoa_flow_wait_calls, 1);
        attn_soa_count(c->asoa_profile, &c->asoa_graph_begin_wait_calls, 1);
        pthread_cond_wait(&c->acv_done, &c->amu);
        if (c->asoa_profile)
            c->asoa_flow_wait_s += attn_soa_now_s() - wait_started;
    }
    pthread_mutex_unlock(&c->amu);
    return 0;
}

int salt_attn_pool_graph_end(SaltKvCache *c) {
    double lock_started, wait_started;
    if (!c || !c->apool_sync_init) return -1;
    lock_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    pthread_mutex_lock(&c->amu);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_lock_acquires, 1);
    if (c->asoa_profile)
        c->asoa_flow_lock_wait_s += attn_soa_now_s() - lock_started;
    if (!c->agraph_running || !c->arunning || c->ashutdown ||
        !pthread_equal(c->agraph_owner, pthread_self()) ||
        c->agraph_phase_active != 0 || c->agraph_phase_done != 0) {
        attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
        pthread_mutex_unlock(&c->amu);
        return -1;
    }
    c->agraph_stop = 1;
    pthread_cond_broadcast(&c->acv_work);
    pthread_cond_broadcast(&c->acv_wide);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 2);
    while (c->adone < c->aactive) {
        wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile, &c->asoa_flow_wait_calls, 1);
        attn_soa_count(c->asoa_profile, &c->asoa_graph_end_wait_calls, 1);
        pthread_cond_wait(&c->acv_done, &c->amu);
        if (c->asoa_profile)
            c->asoa_flow_wait_s += attn_soa_now_s() - wait_started;
    }
    c->afn = NULL;
    c->aarg = NULL;
    c->aactive = 0;
    c->arunning = 0;
    c->agraph_running = 0;
    c->agraph_stop = 0;
    c->agraph_ready = 0;
    c->agraph_phase_done = 0;
    c->agraph_phase_active = 0;
    c->agraph_phase = 0;
    c->agraph_fn = NULL;
    c->agraph_arg = NULL;
    pthread_mutex_unlock(&c->amu);
    return 0;
}

static int attn_pool_flow_run(SaltKvCache *c,
                              int (*fn)(void *arg), void *arg, int team) {
    double lock_started, wait_started;
    int result;
    if (!c || !fn || !c->apool_sync_init || !c->ath ||
        c->apool_threads < 1)
        return -1;
    lock_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
    pthread_mutex_lock(&c->amu);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_lock_acquires, 1);
    if (c->asoa_profile)
        c->asoa_flow_lock_wait_s += attn_soa_now_s() - lock_started;
    if (c->arunning || c->agraph_running || c->aflow_running ||
        c->ashutdown || c->abatch == UINT64_MAX ||
        c->aflow_sessions == UINT64_MAX) {
        attn_soa_count(c->asoa_profile, &c->asoa_rejected_actions, 1);
        pthread_mutex_unlock(&c->amu);
        return -1;
    }
    c->arunning = 1;
    c->aflow_running = 1;
    c->aflow_stop = 0;
    c->aflow_result = -1;
    c->aflow_phase_active = 0;
    c->aflow_phase_done = 0;
    c->aflow_lock_held = 0;
    c->aflow_phase = 0;
    c->aflow_coordinator_fn = fn;
    c->aflow_coordinator_arg = arg;
    c->aflow_phase_fn = NULL;
    c->aflow_phase_arg = NULL;
    c->afn = attn_pool_flow_worker;
    __atomic_store_n(&c->ateam_sleepers, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&c->ateam_join_sleeping, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&c->ateam_mode, team, __ATOMIC_RELEASE);
    c->aarg = c;
    c->aactive = c->apool_threads;
    c->adone = 0;
    c->abatch++;
    c->aflow_sessions++;
    pthread_cond_broadcast(&c->acv_work);
    attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
    if (c->apool_threads > c->nthreads) {
        pthread_cond_broadcast(&c->acv_wide);
        attn_soa_count(c->asoa_profile, &c->asoa_flow_signal_calls, 1);
    }
    while (c->adone < c->aactive) {
        wait_started = c->asoa_profile ? attn_soa_now_s() : 0.0;
        attn_soa_count(c->asoa_profile, &c->asoa_flow_wait_calls, 1);
        pthread_cond_wait(&c->acv_done, &c->amu);
        if (c->asoa_profile)
            c->asoa_flow_wait_s += attn_soa_now_s() - wait_started;
    }
    result = c->aflow_result;
    __atomic_store_n(&c->ateam_mode, 0, __ATOMIC_RELEASE);
    c->afn = NULL;
    c->aarg = NULL;
    c->aactive = 0;
    c->arunning = 0;
    c->aflow_running = 0;
    c->aflow_stop = 0;
    c->aflow_result = -1;
    c->aflow_phase_active = 0;
    c->aflow_phase_done = 0;
    c->aflow_lock_held = 0;
    c->aflow_phase = 0;
    c->aflow_coordinator_fn = NULL;
    c->aflow_coordinator_arg = NULL;
    c->aflow_phase_fn = NULL;
    c->aflow_phase_arg = NULL;
    pthread_mutex_unlock(&c->amu);
    return result;
}

int salt_attn_pool_flow_run(SaltKvCache *c,
                            int (*fn)(void *arg), void *arg) {
    return attn_pool_flow_run(c, fn, arg, 0);
}

int salt_attn_pool_team_run(SaltKvCache *c,
                            int (*fn)(void *arg), void *arg) {
    if (attn_team_pool) return -1;
    return attn_pool_flow_run(c, fn, arg, 1);
}

typedef struct AttnWfqCall {
    SaltKvCache *pool;
    SaltAreaWfqRuntime *runtime;
    SaltAreaWfqItemExecute item_execute;
    void *item_context;
    SaltAreaWfqResult *result;
} AttnWfqCall;

static int attn_wfq_pool_run(
        void *opaque, int active_workers,
        void (*worker)(int worker, void *task), void *task) {
    SaltKvCache *pool = (SaltKvCache *)opaque;
    return salt_attn_pool_run_n(pool, active_workers, worker, task);
}

static int attn_wfq_coordinator(void *opaque) {
    AttnWfqCall *call = (AttnWfqCall *)opaque;
    if (!call) return -1;
    return salt_area_wfq_execute(
        call->runtime, attn_wfq_pool_run, call->pool,
        call->item_execute, call->item_context, call->result);
}

int salt_attn_pool_wfq_execute(
        SaltKvCache *c, SaltAreaWfqRuntime *runtime,
        SaltAreaWfqItemExecute item_execute, void *item_context,
        SaltAreaWfqResult *result) {
    AttnWfqCall call;
    if (!c || !runtime || !runtime->ready || !item_execute ||
        !item_context || !result || runtime->plan.workers == 0 ||
        runtime->plan.workers > (uint32_t)c->apool_threads)
        return -1;
    memset(&call, 0, sizeof call);
    call.pool = c;
    call.runtime = runtime;
    call.item_execute = item_execute;
    call.item_context = item_context;
    call.result = result;
    return salt_attn_pool_flow_run(c, attn_wfq_coordinator, &call);
}

int salt_attn_pool_stats(SaltKvCache *c, SaltAttnPoolStats *stats) {
    if (!c || !stats || !c->apool_sync_init) return -1;
    if (pthread_mutex_lock(&c->amu) != 0) return -1;
    memset(stats, 0, sizeof *stats);
    stats->resident_workers = (uint32_t)c->apool_threads;
    stats->base_workers = (uint32_t)c->nthreads;
    stats->wide_workers = (uint32_t)(c->apool_threads - c->nthreads);
    stats->peak_active_workers = c->asoa_peak_active;
    stats->graph_sessions = c->agraph_sessions;
    stats->flow_sessions = c->aflow_sessions;
    stats->phase_actions = c->asoa_phase_actions;
    stats->worker_actions = c->asoa_worker_actions;
    stats->worker_slot_capacity = c->asoa_worker_slot_capacity;
    stats->worker_slot_headroom = c->asoa_worker_slot_headroom;
    stats->flow_lock_acquires = c->asoa_flow_lock_acquires;
    stats->flow_signal_calls = c->asoa_flow_signal_calls;
    stats->flow_wait_calls = c->asoa_flow_wait_calls;
    stats->graph_begin_wait_calls = c->asoa_graph_begin_wait_calls;
    stats->graph_end_wait_calls = c->asoa_graph_end_wait_calls;
    stats->internal_lock_acquires = c->asoa_internal_lock_acquires;
    stats->internal_signal_calls = c->asoa_internal_signal_calls;
    stats->internal_wait_calls = c->asoa_internal_wait_calls;
    stats->rejected_actions = c->asoa_rejected_actions;
    stats->flow_lock_wait_s = c->asoa_flow_lock_wait_s;
    stats->flow_wait_s = c->asoa_flow_wait_s;
    stats->worker_action_sum_s = c->asoa_worker_action_sum_s;
    stats->worker_critical_s = c->asoa_worker_critical_s;
    stats->internal_lock_wait_s = c->asoa_internal_lock_wait_s;
    stats->internal_wait_s = c->asoa_internal_wait_s;
    if (pthread_mutex_unlock(&c->amu) != 0) return -1;
    return 0;
}

void salt_attn_pool_run(SaltKvCache *c, void (*fn)(int tid, void *arg),
                        void *arg) {
    (void)salt_attn_pool_run_n(c, c->nthreads, fn, arg);
}

static int kv_size_mul(size_t a, size_t b, size_t *out) {
    if (!out || (b != 0 && a > SIZE_MAX / b)) return -1;
    *out = a * b;
    return 0;
}

int salt_kv_lin_init(SaltKvCache *c, int v_heads, int kd, int vd) {
    size_t n;
    if (!c || c->n_layers < 1 || v_heads < 1 || kd < 1 || vd < 1)
        return -1;
    if (c->lin_alloc)
        return c->lin_vh == v_heads && c->lin_kd == kd &&
               c->lin_vd == vd ? 0 : -1;
    n = (size_t)c->n_layers;
    if (kv_size_mul(n, (size_t)v_heads, &n) != 0 ||
        kv_size_mul(n, (size_t)kd, &n) != 0 ||
        kv_size_mul(n, (size_t)vd, &n) != 0 ||
        n > SIZE_MAX / sizeof(float))
        return -1;
    c->lin = (float *)calloc(n, sizeof(float));
    if (!c->lin) return -1;
    c->lin_vh = v_heads;
    c->lin_kd = kd;
    c->lin_vd = vd;
    c->lin_alloc = 1;
    return 0;
}

/* Allocate the linear-attn conv1d ring (Q3_CONV_K x qkv_rows per layer).
 * Persistent across tokens -- the causal conv needs the past qkv. */
int salt_kv_conv_init(SaltKvCache *c, int qkv_rows) {
    size_t n;
    if (!c || c->n_layers < 1 || qkv_rows < 1) return -1;
    if (c->conv_alloc) return c->conv_rows == qkv_rows ? 0 : -1;
    if (c->conv) {
        free(c->conv);
        c->conv = NULL;
        c->conv_alloc = 0;
    }
    n = (size_t)c->n_layers;
    if (kv_size_mul(n, 4, &n) != 0 ||
        kv_size_mul(n, (size_t)qkv_rows, &n) != 0 ||
        n > SIZE_MAX / sizeof(float))
        return -1;
    c->conv = (float *)calloc(n, sizeof(float));
    if (!c->conv) return -1;
    c->conv_rows = qkv_rows;
    c->conv_alloc = 1;
    return 0;
}

/* E8M0 group-scale matvec wrapper: pulls SR/SC from the scale tensor.
 * Large matrices (R >= 2048) run row-partitioned across SALT_ATTN_THREADS
 * workers (default 8) -- the attention's wo_a/wo_b/wq_b are the dominant
 * arithmetic once the real MLA runs (issue #6). */
typedef struct {
    const uint8_t *W, *S;
    int R, C, SR, SC;
    const float *x;
    float *y;
    int r0, r1;
    int simd;
} F8RowJob;

static void *f8_row_worker(void *p) {
    F8RowJob *j = (F8RowJob *)p;
    if (j->simd)
        salt_simd_f8_matvec(j->W, j->S, j->R, j->C, j->SR, j->SC,
                            j->x, j->y, j->r0, j->r1);
    else
        salt_f8_matvec_rows(j->W, j->S, j->R, j->C, j->SR, j->SC,
                            j->x, j->y, j->r0, j->r1);
    return NULL;
}

static void f8_matvec_t(const SaltTrunkLayout *tl, int wi, int si,
                        const uint8_t *tr, int R, int C, const float *x,
                        float *y) {
    int SR = 1, SC = 1;
    if (si >= 0) {
        const SaltTrunkTensor *s = &tl->t[si];
        if (s->rank == 2) {
            SR = (int)s->dims[0];
            SC = (int)s->dims[1];
        } else if (s->rank == 1) {
            SC = (int)s->dims[0];
        }
    }
    if (R >= 2048 && C >= 256) {
        int nth = 8;
        const char *env = getenv("SALT_ATTN_THREADS");
        if (env) {
            int v = atoi(env);
            if (v >= 1 && v <= 32) nth = v;
        }
        if (nth > 1) {
            static pthread_t th[16];
            static F8RowJob job[16];
            if (nth > 16) nth = 16;
            int use_simd = 0;
            if (salt_kernels_simd()) {
                int ssc = SC < 1 ? 1 : SC;
                if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0))
                    use_simd = 1;
            }
            int chunk = (R + nth - 1) / nth;
            for (int t = 0; t < nth; t++) {
                job[t].W = tr + tl->t[wi].off;
                job[t].S = tr + tl->t[si].off;
                job[t].R = R; job[t].C = C;
                job[t].SR = SR; job[t].SC = SC;
                job[t].x = x; job[t].y = y;
                job[t].r0 = t * chunk;
                job[t].r1 = (t + 1) * chunk < R ? (t + 1) * chunk : R;
                job[t].simd = use_simd;
                if (job[t].r0 >= R) { job[t].r1 = job[t].r0; continue; }
                pthread_create(&th[t], NULL, f8_row_worker, &job[t]);
            }
            for (int t = 0; t < nth; t++)
                if (job[t].r1 > job[t].r0) pthread_join(th[t], NULL);
            return;
        }
    }
    salt_f8_matvec(tr + tl->t[wi].off, tr + tl->t[si].off,
                   R, C, SR, SC, x, y);
}

int salt_attn_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                   const uint8_t *tr, float *state, SaltKvCache *kv,
                   int token) {
    if (!tl || !cfg) return -1;
    int qn = tl->attn_qn[L], kvn = tl->attn_kvn[L];
    int wqa = tl->attn_wqa[L], wqa_s = tl->attn_wqa_s[L];
    int wqb = tl->attn_wqb[L], wqb_s = tl->attn_wqb_s[L];
    int wkv = tl->attn_wkv[L], wkv_s = tl->attn_wkv_s[L];
    int woa = tl->attn_woa[L], woa_s = tl->attn_woa_s[L];
    int wob = tl->attn_wob[L], wob_s = tl->attn_wob_s[L];
    int woc = tl->attn_woc[L], woc_s = tl->attn_woc_s[L];
    int sink_i = tl->attn_sink[L];
    /* incomplete graph -> skip the layer (graceful degradation).
     * NOTE: wo_c is OPTIONAL -- the real V4 chain is wo_a + wo_b
     * only; requiring wo_c silently disabled attention on the real
     * checkpoint (woc stayed -1 and the step returned 0). */
    if (qn < 0 || kvn < 0 || wqa < 0 || wqa_s < 0 ||
        wkv < 0 || wkv_s < 0 ||
        woa < 0 || woa_s < 0 || wob < 0 || wob_s < 0)
        return 0;
    int H = cfg->hidden;
    int qlat = (int)tl->t[wqa].dims[0];
    int kvlat = (int)tl->t[wkv].dims[0];
    int qdim = (wqb >= 0) ? (int)tl->t[wqb].dims[0] : qlat;
    int kvhalf = kvlat / 2;
    if (kvlat < 2 || kvhalf < 1 || H < 1) return -1;
    if (!kv || !kv->kv || token < 0 || token >= kv->max_tokens) return 0;

    /* ---- the real MLA (V4-class, layout-driven) ----
     * per-head q width qh = qdim/heads; qn = qh - qk_rope (the rope
     * term skipped in v1); v width vh = wo_a.dims[1]/heads; the latent
     * is [k_nope qn; v vh]; outv = concat of per-head v-sums. The
     * kvhalf single-head path stays as the fallback (fixtures). */
    int heads = cfg->n_heads;
    int qh = qdim, qn_nope = qdim, vh = kvhalf, outv_n = kvhalf;
    int real_mla = 0;
    long ar = H, br = H;
    if (heads > 0 && qdim % heads == 0 && tl->t[woa].rank == 2 &&
        (int)tl->t[woa].dims[1] % heads == 0 &&
        tl->t[wob].rank == 2) {
        qh = qdim / heads;
        vh = (int)tl->t[woa].dims[1] / heads;
        qn_nope = qh - (cfg->qk_rope > 0 ? cfg->qk_rope : 0);
        if (qn_nope < 1) qn_nope = qh;
        if (vh >= 1 && vh <= kvlat && qn_nope <= kvlat &&
            qn_nope + vh <= kvlat + 1) {
            real_mla = 1;
            outv_n = heads * vh;
            ar = tl->t[woa].dims[0];
            br = tl->t[wob].dims[0];
        }
    }
    int w = token + 1;
    int w_sc = real_mla ? heads * w : w;
    /* the value's offset within the latent: default the tail (the
     * engine's [k_nope 448; v 64] assumption), overridable -- the
     * vstd-0 evidence says the tail is the rope slot, not the value;
     * SALT_V_OFFSET sweeps the candidate value locations. */
    int voff = kvlat - vh;
    {
        const char *vset = getenv("SALT_V_OFFSET");
        if (vset) {
            int v = atoi(vset);
            if (v >= 0 && v + vh <= kvlat) voff = v;
        }
    }
    int cha_n = (int)(ar > H ? ar : H);
    int chb_n = (int)(br > H ? br : H);
    /* calloc: any untouched tail (scores/wgt at short windows, or a
     * skipped chain) must be zero, not malloc garbage -- the softmax
     * and combine read them (determinism, issue #6 step 5). */
    float *buf = (float *)calloc(
        (size_t)(qlat + qdim + outv_n + kvlat + 2 * w_sc +
                 cha_n + chb_n + 2 * H + 1),
        sizeof(float));
    if (!buf) return -1;
    float *ql = buf;
    float *q = ql + qlat;
    float *outv = q + qdim;
    float *kvlat_buf = outv + outv_n;
    float *scores = kvlat_buf + kvlat;
    float *wgt = scores + w_sc;
    float *cha = wgt + w_sc;
    float *chb = cha + cha_n;
    float *chc = chb + chb_n;

    /* mHC (issue #6 step 6): F_attn sees x_in = A·vec(X) (the n_hc
     * residual streams combined); the update is
     * new[j*H+i] = sum_k B[j][k]*state[k*H+i] + C[j]*F[i]. xin is H
     * floats beyond the chain (alloc has 4*H + 1). */
    float A[8], C[8], B[64];
    int nhc = 1;
    int hc_ok = salt_hc_params(tl, tl->hc_attn_fn[L], tl->hc_attn_base[L],
                               tl->hc_attn_scale[L], tr, H, state,
                               &nhc, A, C, B);
    if (hc_ok < 0) { free(buf); return -1; }
    if (hc_ok > 0) {
        const char *cs = getenv("SALT_C_SCALE");
        if (cs) {
            float sc = (float)atof(cs);
            if (sc > 0.0f)
                for (int j = 0; j < nhc; j++) C[j] *= sc;
        }
    }
    float *xin = chc + H;
    if (hc_ok)
        salt_hc_combine(nhc, H, A, state, xin);
    else
        xin = state;
    /* the real model's input_layernorm: norm the attention's input
     * with the checkpoint's attn_norm (was never applied -- the raw
     * A-combined state fed the projections and the state grew
     * unbounded) */
    if (tl->attn_norm[L] >= 0 && !getenv("SALT_NO_NORMS"))
        rmsnorm((const uint16_t *)(const void *)(
                    tr + tl->t[tl->attn_norm[L]].off),
                H, xin);
    /* layer-input RMS: the F-rescale target (see the update below) */
    double x2 = 0.0;
    for (int i = 0; i < H; i++) x2 += (double)xin[i] * xin[i];
    float rms_in = sqrtf((float)(x2 / (double)H));

    /* kv latent, normed, cached */
    f8_matvec_t(tl, wkv, wkv_s, tr, kvlat, H, xin, kvlat_buf);
    rmsnorm((const uint16_t *)(const void *)(tr + tl->t[kvn].off),
            kvlat, kvlat_buf);
    memcpy(kv->kv + ((size_t)L * kv->max_tokens + token) * kvlat,
           kvlat_buf, (size_t)kvlat * sizeof(float));

    /* q = wq_b . RMSNorm(wq_a . x, q_norm) */
    f8_matvec_t(tl, wqa, wqa_s, tr, qlat, H, xin, ql);
    rmsnorm((const uint16_t *)(const void *)(tr + tl->t[qn].off),
            qlat, ql);
    if (wqb >= 0)
        f8_matvec_t(tl, wqb, wqb_s, tr, qdim, qlat, ql, q);
    else
        memcpy(q, ql, (size_t)qlat * sizeof(float));

    /* scores over cached positions 0..token (causal), + sink boost.
     * Real MLA: per head, q_nope . k_nope / sqrt(qh) (the rope term
     * is the documented v1 skip). Fallback: the old single-head
     * kvhalf path. */
    float dscale = salt_rsqrtf((float)(real_mla ? qh : qdim));
    int sink_n = 0;
    const float *sinkv = NULL;
    if (sink_i >= 0 && !getenv("SALT_NO_SINK")) {
        sink_n = (int)tl->t[sink_i].nbytes / (int)sizeof(float);
        sinkv = (const float *)(const void *)(tr + tl->t[sink_i].off);
    }
    if (real_mla) {
        /* rope frequencies: the tyrope (yarn-style correction, from the
         * DeepSeek V3.2/V4 rope_scaling: freq_inter = 1/(factor*theta^r),
         * freq_extra = 1/theta^r, a linear ramp between the correction
         * dims for beta_fast and beta_slow). Plain rotary when the
         * params are absent. */
        int qr = cfg->qk_rope > 0 ? cfg->qk_rope : 0;
        static float rope_freq[32];
        static int rope_freq_ready = 0;
        if (qr > 0 && !rope_freq_ready) {
            double theta = cfg->rope_theta > 0 ? cfg->rope_theta : 10000.0;
            double factor = cfg->rope_factor, bf = cfg->rope_beta_fast,
                   bs = cfg->rope_beta_slow, mxp = cfg->rope_max_pos;
            if (factor > 1.0 && mxp > 1.0 && bf > 0.0 && bs > 0.0) {
                double lt = log(theta);
                double corr_bf = (qr * log(mxp / (bf * 2.0 * 3.14159265358979323846))) /
                                 (2.0 * lt);
                double corr_bs = (qr * log(mxp / (bs * 2.0 * 3.14159265358979323846))) /
                                 (2.0 * lt);
                int low = (int)floor(corr_bf);
                int high = (int)ceil(corr_bs);
                if (low < 0) low = 0;
                if (high >= qr / 2) high = qr / 2 - 1;
                if (high < low) high = low;
                for (int i = 0; i < qr / 2; i++) {
                    double ex = 1.0 / pow(theta, (2.0 * i) / (double)qr);
                    double it = 1.0 / (factor *
                                       pow(theta, (2.0 * i) / (double)qr));
                    double ramp = 0.0;
                    if (high > low) {
                        double r = ((double)i - low) / (double)(high - low);
                        ramp = r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
                    }
                    rope_freq[i] = (float)(it * (1.0 - ramp) + ex * ramp);
                }
            } else {
                for (int i = 0; i < qr / 2; i++)
                    rope_freq[i] = 1.0f / salt_powf(
                        (float)theta, (float)(2 * i) / (float)qr);
            }
            rope_freq_ready = 1;
        }
        for (int h = 0; h < heads; h++) {
            const float *q_h = q + (size_t)h * qh;
            float *sc = scores + (size_t)h * w;
            float *wg = wgt + (size_t)h * w;
            /* rotate the query's rope part once per head/token */
            float qr_buf[64];
            if (qr > 0 && qr <= 64) {
                memcpy(qr_buf, q_h + qn_nope, (size_t)qr * sizeof(float));
                for (int i = 0; i + 1 < qr; i += 2) {
                    float a = (float)token * rope_freq[i / 2];
                    float c = salt_cosf(a), s = salt_sinf(a);
                    float x = qr_buf[i], y = qr_buf[i + 1];
                    qr_buf[i] = x * c - y * s;
                    qr_buf[i + 1] = x * s + y * c;
                }
            }
            /* CSA step 1: the sliding window (the checkpoint's
             * sliding_window=128). SALT_WINDOW overrides; 0 = full. */
            int win = 0;
            const char *we = getenv("SALT_WINDOW");
            if (we) win = atoi(we);
            int t2a = 0;
            if (win > 0 && token - win + 1 > t2a) t2a = token - win + 1;
            for (int t2 = t2a; t2 <= token; t2++) {
                const float *k2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat;
                float acc = 0.0f;
                for (int i = 0; i < qn_nope; i++)
                    acc += q_h[i] * k2[i];
                /* rope term: the k's rope part (the latent's last qr,
                 * the shared KV -- same values as the v) rotated by
                 * the key position, dotted with the rotated q_rope */
                if (qr > 0) {
                    float kr_buf[64];
                    memcpy(kr_buf, k2 + qn_nope,
                           (size_t)qr * sizeof(float));
                    for (int i = 0; i + 1 < qr; i += 2) {
                        float a = (float)t2 * rope_freq[i / 2];
                        float c = salt_cosf(a), s = salt_sinf(a);
                        float x = kr_buf[i], y = kr_buf[i + 1];
                        kr_buf[i] = x * c - y * s;
                        kr_buf[i + 1] = x * s + y * c;
                    }
                    for (int i = 0; i < qr; i++)
                        acc += qr_buf[i] * kr_buf[i];
                }
                sc[t2] = acc * dscale;
                if (sinkv && t2 < sink_n) sc[t2] += sinkv[t2];
            }
            float mx = sc[0];
            for (int t2 = 1; t2 <= token; t2++)
                if (sc[t2] > mx) mx = sc[t2];
            float sum = 0.0f;
            for (int t2 = 0; t2 <= token; t2++) {
                wg[t2] = salt_expf(sc[t2] - mx);
                sum += wg[t2];
            }
            float *ov = outv + (size_t)h * vh;
            for (int j = 0; j < vh; j++) ov[j] = 0.0f;
            for (int t2 = 0; t2 <= token; t2++) {
                const float *v2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat +
                    (voff);
                float wgtn = wg[t2] / sum;
                for (int j = 0; j < vh; j++) ov[j] += wgtn * v2[j];
            }
            if (h == 0 && getenv("SALT_DEBUG9")) {
                /* the frozen-F probe: score spread, the v's variation
                 * across the cache, the softmax concentration */
                float smin = sc[0], smax = sc[0];
                for (int t2 = 1; t2 <= token; t2++) {
                    if (sc[t2] < smin) smin = sc[t2];
                    if (sc[t2] > smax) smax = sc[t2];
                }
                int np = token + 1;
                float vstd = 0.0f, vmean = 0.0f;
                for (int j = 0; j < vh; j++) {
                    float m = 0.0f, s = 0.0f;
                    for (int t2 = 0; t2 <= token; t2++) {
                        float v = kv->kv[
                            ((size_t)L * kv->max_tokens + t2) * kvlat +
                            (voff) + j];
                        m += v; s += v * v;
                    }
                    m /= (float)np;
                    s = sqrtf(s / (float)np - m * m);
                    vstd += s; vmean += m;
                }
                vstd /= (float)vh;
                vmean /= (float)vh;
                float wmax = 0.0f;
                for (int t2 = 0; t2 <= token; t2++) {
                    float wn = wg[t2] / sum;
                    if (wn > wmax) wmax = wn;
                }
                fprintf(stderr,
                        "c9: L%-2d t%-3d scores [%.3f, %.3f] "
                        "spread %.3f vstd %.4f vmean %.4f wmax %.3f\n",
                        L, token, smin, smax, smax - smin, vstd, vmean,
                        wmax);
            }
        }
    } else {
        for (int t2 = 0; t2 <= token; t2++) {
            const float *k2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat;
            float acc = 0.0f;
            for (int i = 0; i < kvhalf; i++) acc += q[i] * k2[i];
            scores[t2] = acc * dscale;
            if (sinkv && t2 < sink_n) scores[t2] += sinkv[t2];
        }
        float mx = scores[0];
        for (int t2 = 1; t2 <= token; t2++)
            if (scores[t2] > mx) mx = scores[t2];
        float sum = 0.0f;
        for (int t2 = 0; t2 <= token; t2++) {
            wgt[t2] = salt_expf(scores[t2] - mx);
            sum += wgt[t2];
        }
        for (int i = 0; i < kvhalf; i++) outv[i] = 0.0f;
        for (int t2 = 0; t2 <= token; t2++) {
            const float *v2 = kv->kv +
                ((size_t)L * kv->max_tokens + t2) * kvlat + kvhalf;
            float w = wgt[t2] / sum;
            for (int i = 0; i < kvhalf; i++) outv[i] += w * v2[i];
        }
    }

    /* output chain. Real MLA: wo_a [ar x outv_n] then wo_b [br x ar]
     * (the real V4 chain, no wo_c). Fallback: the old H-width chain. */
    if (real_mla) {
        f8_matvec_t(tl, woa, woa_s, tr, (int)ar, outv_n, outv, cha);
        f8_matvec_t(tl, wob, wob_s, tr, (int)br, (int)ar, cha, chb);
        memcpy(chc, chb, (size_t)H * sizeof(float));
    } else {
        f8_matvec_t(tl, woa, woa_s, tr, H, kvhalf, outv, cha);
        f8_matvec_t(tl, wob, wob_s, tr, H, H, cha, chb);
        f8_matvec_t(tl, woc, woc_s, tr, H, H, chb, chc);
    }
    if (hc_ok) {
        /* F-rescale: the approximate attention reads amplify (the real
         * model bounds F by training against the manifold-constrained
         * residual). Rescale F to the layer-input RMS so the mHC
         * update is finite; the real fix is the exact MLA column
         * reads. */
        double s2 = 0.0;
        for (int i = 0; i < H; i++) s2 += (double)chc[i] * chc[i];
        float rms_f = sqrtf((float)(s2 / (double)H)) + 1e-30f;
        if (hc_ok) {
            /* F-rescale: OFF by default -- the A/B showed the raw attention
             * output routes better (82.2% vs 75.1% hits, bytes/token 0.31
             * vs 0.43). SALT_F_RESCALE=1 opts INTO the rms_in clamp. */
            float gain = rms_in / rms_f;
            if (getenv("SALT_F_RESCALE"))
                if (gain > 0.0f && gain < 1e30f)
                    for (int i = 0; i < H; i++) chc[i] *= gain;
            if (getenv("SALT_DEBUG8")) {
                double f2 = 0.0;
                for (int i = 0; i < H; i++)
                    f2 += (double)chc[i] * chc[i];
                fprintf(stderr, "c8: L%-2d C[", L);
                for (int j = 0; j < nhc; j++)
                    fprintf(stderr, " %.4f", C[j]);
                fprintf(stderr, "] A[");
                for (int j = 0; j < nhc; j++)
                    fprintf(stderr, " %.4f", A[j]);
                fprintf(stderr, "] Frms %.4f rms_in %.4f\n",
                        sqrtf((float)(f2 / (double)H)), rms_in);
            }
        }
        /* new[j*H+i] = sum_k B[j][k]*state[k*H+i] + C[j]*chc[i].
         * SALT_NO_B_MIX: identity-B (s + C*F -- the real model's
         * residual-stream shape); the Sinkhorn B's contraction pins
         * the state to a fixed point and the logits freeze. */
        float mix[8];
        int no_b = getenv("SALT_NO_B_MIX") ? 1 : 0;
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < nhc; j++) {
                float s = no_b ? state[j * H + i] : 0.0f;
                if (!no_b)
                    for (int k = 0; k < nhc; k++)
                        s += B[j * nhc + k] * state[k * H + i];
                mix[j] = s + C[j] * chc[i];
            }
            for (int j = 0; j < nhc; j++) state[j * H + i] = mix[j];
        }
        /* state-rescale: the mHC update carries the input forward
         * (B ~ identity) plus up to C*F, so the state lands at
         * ~(1+C)*rms_in and compounds. Bound the whole state to the
         * layer-input RMS -- the real model's trained F achieves this
         * internally; the interim preserves the mHC shape (B-mixing,
         * C distribution) with a bounded magnitude. */
        double t2 = 0.0;
        for (int i = 0; i < nhc * H; i++) t2 += (double)state[i] * state[i];
        float rms_s = sqrtf((float)(t2 / (double)(nhc * H))) + 1e-30f;
        float sgain = rms_in / rms_s;
        /* the rms_in target is a fixed point at the embed's scale
         * (~0.001): the state stays dead and the head reads ~0 logits
         * (the flat softmax = the multi-language soup). SALT_STATE_RMS_TARGET
         * overrides the target with a fixed norm (1.0 = the normed
         * scale the trained head expects). */
        {
            const char *tgt = getenv("SALT_STATE_RMS_TARGET");
            if (tgt) {
                float t = (float)atof(tgt);
                if (t > 0.0f) sgain = t / rms_s;
            }
        }
        if (!getenv("SALT_NO_STATE_RESCALE"))
            if (sgain > 0.0f && sgain < 1e30f)
                for (int i = 0; i < nhc * H; i++) state[i] *= sgain;
    } else {
        for (int i = 0; i < H; i++) state[i] += chc[i];
    }

    if (getenv("SALT_DEBUG13") && token > 0) {
        /* the per-stream token-deltas: where the state's movement
         * lives (which of the 4 mHC streams carries the token) */
        static float prev_s13[SALT_MAX_LAYERS][8][4096];
        const float *pv = &prev_s13[L][0][0];
        for (int j = 0; j < nhc; j++) {
            double d2 = 0.0;
            const float *cur = state + (size_t)j * H;
            for (int i = 0; i < H; i++) {
                float d = cur[i] - pv[j * H + i];
                d2 += (double)d * d;
            }
            fprintf(stderr, "[dbg13] L%d t%d s%d delta=%.6g\n",
                    L, token, j, sqrtf((float)(d2 / (double)H)));
        }
        memcpy(prev_s13[L], state, (size_t)nhc * H * sizeof(float));
    }
    if (getenv("SALT_DEBUG12") && token > 0) {
        /* the F_attn token-delta: does the attention output carry the
         * token's movement? (the state's tok_delta decays ~24%/layer;
         * if F's delta is ~0, the attention path itself is the
         * contraction) */
        static float prev_f[SALT_MAX_LAYERS][4096];
        const float *pv = prev_f[L];
        double d2 = 0.0;
        for (int i = 0; i < H; i++) {
            float d = chc[i] - pv[i];
            d2 += (double)d * d;
        }
        fprintf(stderr, "[dbg12] L%d t%d Fattn_delta=%.6g\n",
                L, token, sqrtf((float)(d2 / (double)H)));
        memcpy(prev_f[L], chc, (size_t)H * sizeof(float));
    }

    free(buf);
    return 0;
}
