#define _POSIX_C_SOURCE 200809L

#include "salt/attn_batch.h"
#include "salt/attn.h"
#include "salt/bitmath.h"
#include "thread-lifecycle.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <string.h>

typedef struct {
    const SaltAttentionBatchJob *job;
    int worker;
    int workers;
    int failed;
} AttentionWorker;

static void *attention_worker(void *opaque) {
    AttentionWorker *worker = (AttentionWorker *)opaque;
    const SaltAttentionBatchJob *job = worker->job;
    const SaltAttentionDesc *attention = &job->attention;
    int total = job->batch * attention->n_heads;
    int groups = attention->n_heads / attention->n_kv_heads;
    int score_capacity = job->start_position + job->batch;
    float *scores = NULL;
    if (job->pool && job->pool->asc8 && worker->worker >= 0 &&
        worker->worker < job->pool->apool_threads &&
        score_capacity <= 8 * job->pool->max_tokens) {
        scores = job->pool->asc8 +
            (size_t)worker->worker * 8u * (size_t)job->pool->max_tokens;
    }
    if (!scores) {
        worker->failed = 1;
        return NULL;
    }
    for (int task = worker->worker; task < total; task += worker->workers) {
        int token = task / attention->n_heads;
        int head = task % attention->n_heads;
        int kv_head = head / groups;
        int position = job->start_position + token;
        int first = attention->kind == SALT_ATTN_FULL
            ? 0 : position - attention->window + 1;
        const float *query = job->queries +
            (size_t)token * job->query_stride +
            (size_t)head * attention->head_dim;
        float *output = job->outputs +
            (size_t)token * job->query_stride +
            (size_t)head * attention->head_dim;
        SaltAttnKvSpan spans[SALT_ATTN_HEAD_MAX_SPANS];
        SaltAttnHeadFold fold;
        int span_count = 0;
        if (first < 0) first = 0;
        if (first < job->shared_tokens) {
            int end = position + 1;
            if (end > job->shared_tokens) end = job->shared_tokens;
            spans[span_count++] = (SaltAttnKvSpan) {
                job->shared_keys + (size_t)first * job->kv_stride,
                job->shared_values + (size_t)first * job->kv_stride,
                end - first,
            };
        }
        if (position >= job->shared_tokens) {
            int start = first;
            if (start < job->shared_tokens) start = job->shared_tokens;
            spans[span_count++] = (SaltAttnKvSpan) {
                job->private_keys + (size_t)start * job->kv_stride,
                job->private_values + (size_t)start * job->kv_stride,
                position - start + 1,
            };
        }
        memset(&fold, 0, sizeof fold);
        fold.query = query;
        fold.output = output;
        fold.scores = scores;
        fold.row_count = position - first + 1;
        fold.head_dim = attention->head_dim;
        fold.kv_head = kv_head;
        fold.score_scale = 1.0f;
        fold.kv_row_stride = job->kv_stride;
        fold.spans = spans;
        fold.span_count = span_count;
        fold.span_accumulate = salt_attn_weighted_value_accumulate;
        if (salt_attn_head_fold(&fold) != 0) {
            worker->failed = 1;
            break;
        }
    }
    return NULL;
}

typedef struct {
    AttentionWorker *workers;
} AttentionPoolRun;

static void attention_pool_worker(int worker, void *opaque) {
    AttentionPoolRun *run = (AttentionPoolRun *)opaque;
    if (!run || !run->workers) return;
    (void)attention_worker(&run->workers[worker]);
}

int salt_attention_batch_run(const SaltAttentionBatchJob *job, int threads) {
    AttentionWorker workers[32];
    AttentionPoolRun pool_run;
    int total;
    if (!job || !job->queries || !job->private_keys || !job->private_values ||
        !job->outputs || !job->pool || job->start_position < 0 || job->batch < 1 ||
        job->start_position > INT_MAX - job->batch ||
        job->attention.n_heads < 1 || job->attention.n_kv_heads < 1 ||
        job->attention.n_heads % job->attention.n_kv_heads != 0 ||
        job->attention.head_dim < 1 ||
        job->query_stride != job->attention.n_heads * job->attention.head_dim ||
        job->kv_stride != job->attention.n_kv_heads * job->attention.head_dim ||
        job->shared_tokens < 0 ||
        (job->shared_tokens > 0 &&
         (!job->shared_keys || !job->shared_values)))
        return -1;
    if (job->batch > INT_MAX / job->attention.n_heads) return -1;
    total = job->batch * job->attention.n_heads;
    if (threads < 1) threads = 1;
    if (threads > 32) threads = 32;
    if (threads > total) threads = total;
    for (int thread = 0; thread < threads; thread++) {
        workers[thread].job = job;
        workers[thread].worker = thread;
        workers[thread].workers = threads;
        workers[thread].failed = 0;
    }
    if (!job->pool->asc8 || threads > job->pool->apool_threads ||
        job->start_position + job->batch > 8 * job->pool->max_tokens)
        return -1;
    pool_run.workers = workers;
    if (salt_attn_pool_run_n(job->pool, threads,
            attention_pool_worker, &pool_run) != 0)
        return -1;
    for (int thread = 0; thread < threads; thread++)
        if (workers[thread].failed) return -1;
    return 0;
}
