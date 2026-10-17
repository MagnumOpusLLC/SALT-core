/* LFM2.5-24B-A2B host realization.
 *
 * The engine's SaltTensorHostOps table for LFM's tensors. It is a DISPATCH, not
 * new arithmetic: the 4-bit track calls the engine's existing CPU batch kernel,
 * and the 6-bit track uses this model's own packed-code decode (lfm_quant.h).
 *
 * The base for every tensor is its `binding` -- the mapped resource -- plus the
 * storage offsets the adapter recorded. Nothing here re-derives an address, and
 * nothing here is model-specific beyond the encoding dispatch.
 *
 * Model-owned. The engine declares the interface and calls it; it contains no
 * LFM arithmetic. */

#include "lfm_host.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "salt/kernels.h"
#include "salt/simd.h"

/* BF16 -> float, by bit position. No library call, so the result is identical
 * everywhere. */
static float bf16_to_float(uint16_t bits) {
    uint32_t widened = (uint32_t)bits << 16;
    float out;
    union { uint32_t u; float f; } conv;
    conv.u = widened;
    out = conv.f;
    return out;
}

static const uint8_t *lfm_acquire_tensor(const SaltTensorDesc *t, int *slot) {
    const LfmHostBinding *binding;
    uint64_t logical;
    int layer, expert;

    if (!t || !slot || !(binding = (const LfmHostBinding *)t->binding))
        return NULL;
    *slot = -1;
    if (t->storage.source_class == SALT_TENSOR_SOURCE_STATIC)
        return binding->trunk;
    if (t->storage.source_class != SALT_TENSOR_SOURCE_SELECTED ||
        !binding->cache)
        return NULL;

    logical = t->storage.logical_resource_id;
    if (logical >= (uint64_t)LFM_LAYERS * LFM_EXPERTS)
        return NULL;
    layer = (int)(logical / LFM_EXPERTS);
    expert = (int)(logical % LFM_EXPERTS);
    if (salt_cache_getmany(binding->cache, layer, &expert, 1, slot) < 0)
        return NULL;
    return salt_cache_acquire(binding->cache, *slot, layer, expert);
}

static int lfm_release_tensor(const SaltTensorDesc *t, int slot) {
    const LfmHostBinding *binding;
    uint64_t logical;
    if (slot < 0)
        return 0;
    if (!t || !(binding = (const LfmHostBinding *)t->binding) || !binding->cache)
        return -1;
    logical = t->storage.logical_resource_id;
    return salt_cache_release(binding->cache, slot,
                              (int)(logical / LFM_EXPERTS),
                              (int)(logical % LFM_EXPERTS));
}

static int affine6_matvec_batch(const uint32_t *vals, const uint16_t *scales,
                                const uint16_t *biases, int rows, int cols,
                                int batch, const float *xs, float *ys) {
    return lfm_affine6_matvec_batch(
        vals, scales, biases, rows, cols, batch, xs, ys);
}

/* F32 and BF16 rows, which LFM uses for its norms and its router. */
static int plain_matvec_batch_rows(const SaltTensorDesc *t, const void *vals,
        int rows, int cols, int batch, const float *xs, float *ys,
        int row_start, int row_end) {
    if (!t || !vals || !xs || !ys || rows < 1 || cols < 1 || batch < 1 ||
        row_start < 0 || row_start > row_end || row_end > rows)
        return -1;
    for (int row = row_start; row < row_end; row++) {
        for (int b = 0; b < batch; b++) {
            float acc = 0.0f;
            const float *x = xs + (size_t)b * (size_t)cols;
            for (int c = 0; c < cols; c++) {
                float w;
                if (t->storage.encoding == SALT_TENSOR_ENCODING_F32)
                    w = ((const float *)vals)[(size_t)row * (size_t)cols +
                                              (size_t)c];
                else
                    w = bf16_to_float(
                        ((const uint16_t *)vals)[(size_t)row * (size_t)cols +
                                                 (size_t)c]);
                acc += w * x[c];
            }
            ys[(size_t)b * (size_t)rows + (size_t)row] = acc;
        }
    }
    return 0;
}

static int plain_matvec_batch(const SaltTensorDesc *t, const void *vals,
                              int rows, int cols, int batch,
                              const float *xs, float *ys) {
    return plain_matvec_batch_rows(
        t, vals, rows, cols, batch, xs, ys, 0, rows);
}

static int lfm_project_batch(const SaltTensorDesc *t, const float *xs,
                             uint32_t batch, float *ys, void *scratch,
                             size_t scratch_bytes) {
    const uint8_t *base;
    const void *vals, *scales, *biases;
    int rows, cols, slot, rc = -1;
    (void)scratch;
    (void)scratch_bytes;
    if (!t || !xs || !ys || batch < 1 || t->rows < 1 || t->cols < 1)
        return -1;
    base = lfm_acquire_tensor(t, &slot);
    if (!base)
        return -1;
    vals = base + t->storage.value_offset;
    scales = base + t->storage.scale_offset;
    biases = base + t->storage.bias_offset;
    rows = (int)t->rows;
    cols = (int)t->cols;

    switch (t->storage.encoding) {
    case SALT_TENSOR_ENCODING_AFFINE_Q4:
        /* The batch kernel carries a bounded per-row group table; LFM dense-down
         * has 11,776 columns = 184 groups and exceeds it. The engine's single-row
         * kernel supports arbitrary C through its canonical fallback. This is a
         * physical dispatch between TWO ENGINE kernels, not new model arithmetic. */
        if (batch == 1u ||
            cols / SALT_MLX4_GROUP > SALT_SIMD_Q4_MAX_GROUPS_PER_ROW) {
            rc = 0;
            for (uint32_t b = 0; b < batch && rc == 0; b++)
                rc = salt_q4_matvec_status((const uint32_t *)vals,
                                           (const uint16_t *)scales,
                                           (const uint16_t *)biases,
                                           rows, cols,
                                           xs + (size_t)b * (size_t)cols,
                                           ys + (size_t)b * (size_t)rows);
        } else {
            rc = salt_q4_matvec_batch((const uint32_t *)vals,
                                      (const uint16_t *)scales,
                                      (const uint16_t *)biases,
                                      rows, cols, (int)batch, xs, ys);
        }
        if (rc != 0)
            fprintf(stderr,
                    "LFM_Q4_FAIL rows=%d cols=%d batch=%u vo=%llu so=%llu "
                    "bo=%llu vb=%llu sb=%llu bb=%llu rc=%d\n",
                    rows, cols, batch,
                    (unsigned long long)t->storage.value_offset,
                    (unsigned long long)t->storage.scale_offset,
                    (unsigned long long)t->storage.bias_offset,
                    (unsigned long long)t->storage.value_bytes,
                    (unsigned long long)t->storage.scale_bytes,
                    (unsigned long long)t->storage.bias_bytes, rc);
        break;
    case SALT_TENSOR_ENCODING_AFFINE_Q6:
        rc = affine6_matvec_batch((const uint32_t *)vals,
                                  (const uint16_t *)scales,
                                  (const uint16_t *)biases,
                                  rows, cols, (int)batch, xs, ys);
        break;
    case SALT_TENSOR_ENCODING_F32:
    case SALT_TENSOR_ENCODING_BF16:
        rc = plain_matvec_batch(t, vals, rows, cols, (int)batch, xs, ys);
        break;
    default:
        rc = -1; /* an encoding this realization does not execute */
        break;
    }
    if (lfm_release_tensor(t, slot) != 0)
        rc = -1;
    return rc;
}

static size_t lfm_scratch_requirement(const SaltTensorDesc *t) {
    (void)t;
    return 0; /* the kernels in use need no caller-supplied scratch */
}

static int lfm_gather_rows(const SaltTensorDesc *t, const int32_t *row_ids,
                           uint32_t row_count, float *output, void *scratch,
                           size_t scratch_bytes) {
    const uint8_t *base;
    uint32_t i;
    int cols, slot, rc = 0;
    (void)scratch;
    (void)scratch_bytes;
    if (!t || !row_ids || !output || t->rows < 1 || t->cols < 1)
        return -1;
    for (i = 0; i < row_count; i++)
        if (row_ids[i] < 0 || (uint32_t)row_ids[i] >= t->rows)
            return -1;
    base = lfm_acquire_tensor(t, &slot);
    if (!base)
        return -1;
    cols = (int)t->cols;
    for (i = 0; i < row_count && rc == 0; i++) {
        int32_t id = row_ids[i];
        if (t->storage.encoding == SALT_TENSOR_ENCODING_F32) {
            const float *src = (const float *)(base + t->storage.value_offset);
            for (int c = 0; c < cols; c++)
                output[(size_t)i * (size_t)cols + (size_t)c] =
                    src[(size_t)id * (size_t)cols + (size_t)c];
        } else if (t->storage.encoding == SALT_TENSOR_ENCODING_AFFINE_Q6) {
            /* The embedding is a 6-bit affine row per token, and it is also the
             * output head. Gathering it means decoding one row -- not copying
             * bytes, because the packing spans words. */
            const uint32_t *vals =
                (const uint32_t *)(base + t->storage.value_offset);
            const uint16_t *sc =
                (const uint16_t *)(base + t->storage.scale_offset);
            const uint16_t *bi =
                (const uint16_t *)(base + t->storage.bias_offset);
            size_t words_per_row = (size_t)cols * LFM_AFFINE6_BITS / 32u;
            size_t groups_per_row = (size_t)cols / LFM_AFFINE6_GROUP;
            const uint32_t *row_words = vals + (size_t)id * words_per_row;
            const uint16_t *row_sc = sc + (size_t)id * groups_per_row;
            const uint16_t *row_bi = bi + (size_t)id * groups_per_row;
            for (int c = 0; c < cols; c++) {
                unsigned code = lfm_affine6_code(row_words, (unsigned)c);
                size_t g = (size_t)c / LFM_AFFINE6_GROUP;
                output[(size_t)i * (size_t)cols + (size_t)c] =
                    (float)code * bf16_to_float(row_sc[g]) +
                    bf16_to_float(row_bi[g]);
            }
        } else if (t->storage.encoding == SALT_TENSOR_ENCODING_BF16) {
            /* The norms are BF16: a row is a plain vector of BF16 values, so
             * gathering it is a decode, not a byte copy. */
            const uint16_t *src =
                (const uint16_t *)(base + t->storage.value_offset);
            int c;
            for (c = 0; c < cols; c++)
                output[(size_t)i * (size_t)cols + (size_t)c] =
                    bf16_to_float(src[(size_t)id * (size_t)cols + (size_t)c]);
        } else {
            rc = -1; /* a packed 4-bit row has no row-dequant entry here */
        }
    }
    if (lfm_release_tensor(t, slot) != 0)
        rc = -1;
    return rc;
}

static int lfm_matvec(const SaltTensorDesc *t, const float *input,
                      float *output, void *scratch, size_t scratch_bytes) {
    return lfm_project_batch(t, input, 1u, output, scratch, scratch_bytes);
}

static int lfm_matvec_batch(const SaltTensorDesc *t, const float *inputs,
                            uint32_t row_count, float *outputs, void *scratch,
                            size_t scratch_bytes) {
    return lfm_project_batch(t, inputs, row_count, outputs, scratch,
                             scratch_bytes);
}

typedef struct LfmPreparedHostJob {
    const SaltTensorDesc *tensor;
    const void *values;
    const void *scales;
    const void *biases;
    const float *inputs;
    float *outputs;
    int rows;
    int cols;
    int batch;
} LfmPreparedHostJob;

typedef struct LfmHostPoolTask {
    LfmPreparedHostJob *jobs;
    uint32_t job_count;
    uint32_t workers;
    int failed[32];
} LfmHostPoolTask;

static int lfm_project_prepared_rows(const LfmPreparedHostJob *job,
        int row_start, int row_end) {
    if (!job || row_start < 0 || row_start > row_end || row_end > job->rows)
        return -1;
    switch (job->tensor->storage.encoding) {
    case SALT_TENSOR_ENCODING_AFFINE_Q4:
        if (job->cols / SALT_MLX4_GROUP >
                SALT_SIMD_Q4_MAX_GROUPS_PER_ROW) {
            for (int batch = 0; batch < job->batch; batch++)
                if (salt_q4_matvec_rows_status(
                        (const uint32_t *)job->values,
                        (const uint16_t *)job->scales,
                        (const uint16_t *)job->biases,
                        job->rows, job->cols,
                        job->inputs + (size_t)batch * (size_t)job->cols,
                        job->outputs + (size_t)batch * (size_t)job->rows,
                        row_start, row_end) != 0)
                    return -1;
            return 0;
        }
        return salt_q4_matvec_batch_rows(
            (const uint32_t *)job->values,
            (const uint16_t *)job->scales,
            (const uint16_t *)job->biases,
            job->rows, job->cols, job->batch,
            job->inputs, job->outputs, row_start, row_end);
    case SALT_TENSOR_ENCODING_AFFINE_Q6:
        return lfm_affine6_matvec_batch_rows(
            job->values, job->scales, job->biases,
            job->rows, job->cols, job->batch,
            job->inputs, job->outputs, row_start, row_end);
    case SALT_TENSOR_ENCODING_F32:
    case SALT_TENSOR_ENCODING_BF16:
        return plain_matvec_batch_rows(
            job->tensor, job->values, job->rows, job->cols, job->batch,
            job->inputs, job->outputs, row_start, row_end);
    default:
        return -1;
    }
}

static void lfm_host_pool_worker(int worker, void *opaque) {
    LfmHostPoolTask *task = (LfmHostPoolTask *)opaque;
    if (!task || worker < 0 || (uint32_t)worker >= task->workers)
        return;
    for (uint32_t index = 0; index < task->job_count; index++) {
        LfmPreparedHostJob *job = &task->jobs[index];
        int row_start = (int)(((int64_t)job->rows * worker) / task->workers);
        int row_end = (int)(((int64_t)job->rows * (worker + 1)) /
                            task->workers);
        if (row_start < row_end &&
            lfm_project_prepared_rows(job, row_start, row_end) != 0) {
            task->failed[worker] = 1;
            return;
        }
    }
}

int lfm_host_affine_q4_batch(LfmHostBinding *binding,
        const void *vals, const void *scales, const void *biases,
        int rows, int cols, int batch, const float *inputs, float *outputs) {
    SaltTensorDesc tensor;
    LfmPreparedHostJob job;
    LfmHostPoolTask task;
    uint32_t workers;
    if (!binding || !binding->compute_pool || !vals || !scales || !biases ||
        !inputs || !outputs || rows < 1 || cols < 1 || batch < 1 ||
        binding->workers == 0u || binding->compute_pool->apool_threads < 1)
        return -1;
    memset(&tensor, 0, sizeof tensor);
    tensor.storage.encoding = SALT_TENSOR_ENCODING_AFFINE_Q4;
    memset(&job, 0, sizeof job);
    job.tensor = &tensor;
    job.values = vals;
    job.scales = scales;
    job.biases = biases;
    job.inputs = inputs;
    job.outputs = outputs;
    job.rows = rows;
    job.cols = cols;
    job.batch = batch;
    workers = binding->workers;
    if (workers > (uint32_t)binding->compute_pool->apool_threads)
        workers = (uint32_t)binding->compute_pool->apool_threads;
    if (workers > 32u) workers = 32u;
    if (workers == 0u) return -1;
    memset(&task, 0, sizeof task);
    task.jobs = &job;
    task.job_count = 1u;
    task.workers = workers;
    if (workers == 1u)
        lfm_host_pool_worker(0, &task);
    else if (salt_attn_pool_run_n(binding->compute_pool, (int)workers,
                 lfm_host_pool_worker, &task) != 0)
        return -1;
    for (uint32_t worker = 0; worker < workers; worker++)
        if (task.failed[worker]) return -1;
    return 0;
}

static int lfm_host_execute_jobs(const SaltTensorHostBatch *jobs,
        uint32_t job_count, uint32_t worker_limit) {
    LfmPreparedHostJob prepared[256];
    int unique_experts[LFM_EXPERTS], unique_slots[LFM_EXPERTS];
    const uint8_t *unique_payloads[LFM_EXPERTS];
    uint64_t unique_logical[LFM_EXPERTS];
    uint32_t unique_count = 0u, workers;
    int selected_layer = -1, leases = 0, result = -1;
    const LfmHostBinding *binding = NULL;
    LfmHostPoolTask task;
    if (!jobs || job_count < 1u || job_count > 256u || worker_limit < 1u)
        return -1;
    memset(prepared, 0, sizeof prepared);
    memset(unique_slots, 0xff, sizeof unique_slots);
    memset(unique_payloads, 0, sizeof unique_payloads);
    for (uint32_t index = 0; index < job_count; index++) {
        const SaltTensorDesc *tensor = jobs[index].tensor;
        const LfmHostBinding *owner;
        if (!tensor || !jobs[index].inputs || !jobs[index].outputs ||
            jobs[index].row_count == 0u || tensor->rows == 0u ||
            tensor->rows > INT32_MAX || tensor->cols == 0u ||
            tensor->cols > INT32_MAX || jobs[index].row_count > INT32_MAX ||
            !(owner = (const LfmHostBinding *)tensor->binding) ||
            (binding && binding != owner))
            goto done;
        binding = owner;
        prepared[index].tensor = tensor;
        prepared[index].inputs = jobs[index].inputs;
        prepared[index].outputs = jobs[index].outputs;
        prepared[index].rows = (int)tensor->rows;
        prepared[index].cols = (int)tensor->cols;
        prepared[index].batch = (int)jobs[index].row_count;
        if (tensor->storage.source_class == SALT_TENSOR_SOURCE_SELECTED) {
            uint64_t logical = tensor->storage.logical_resource_id;
            int layer, expert, found = -1;
            if (!binding->cache || logical >= (uint64_t)LFM_LAYERS * LFM_EXPERTS)
                goto done;
            layer = (int)(logical / LFM_EXPERTS);
            expert = (int)(logical % LFM_EXPERTS);
            if (selected_layer >= 0 && selected_layer != layer)
                goto done;
            selected_layer = layer;
            for (uint32_t item = 0; item < unique_count; item++)
                if (unique_logical[item] == logical) {
                    found = (int)item;
                    break;
                }
            if (found < 0) {
                if (unique_count >= LFM_EXPERTS)
                    goto done;
                found = (int)unique_count;
                unique_logical[unique_count] = logical;
                unique_experts[unique_count] = expert;
                unique_count++;
            }
        } else if (tensor->storage.source_class != SALT_TENSOR_SOURCE_STATIC ||
                   !binding->trunk) {
            goto done;
        }
    }
    if (!binding || !binding->compute_pool || binding->workers == 0u ||
        binding->compute_pool->apool_threads < 1)
        goto done;
    if (unique_count > 0u) {
        if (selected_layer < 0 ||
            salt_cache_getmany(binding->cache, selected_layer, unique_experts,
                (int)unique_count, unique_slots) < 0 ||
            salt_cache_acquire_many(binding->cache, unique_slots, selected_layer,
                unique_experts, (int)unique_count, unique_payloads) != 0)
            goto done;
        leases = 1;
    }
    for (uint32_t index = 0; index < job_count; index++) {
        const SaltTensorDesc *tensor = prepared[index].tensor;
        const uint8_t *base = binding->trunk;
        if (tensor->storage.source_class == SALT_TENSOR_SOURCE_SELECTED) {
            uint64_t logical = tensor->storage.logical_resource_id;
            base = NULL;
            for (uint32_t item = 0; item < unique_count; item++)
                if (unique_logical[item] == logical) {
                    base = unique_payloads[item];
                    break;
                }
            if (!base) goto done;
        }
        prepared[index].values = base + tensor->storage.value_offset;
        prepared[index].scales = base + tensor->storage.scale_offset;
        prepared[index].biases = base + tensor->storage.bias_offset;
    }
    workers = binding->workers;
    if (workers > worker_limit) workers = worker_limit;
    if (workers > (uint32_t)binding->compute_pool->apool_threads)
        workers = (uint32_t)binding->compute_pool->apool_threads;
    if (workers > 32u) workers = 32u;
    if (workers == 0u) goto done;
    memset(&task, 0, sizeof task);
    task.jobs = prepared;
    task.job_count = job_count;
    task.workers = workers;
    if (workers == 1u)
        lfm_host_pool_worker(0, &task);
    else if (salt_attn_pool_run_n(binding->compute_pool, (int)workers,
                 lfm_host_pool_worker, &task) != 0)
        goto done;
    result = 0;
    for (uint32_t worker = 0; worker < workers; worker++)
        if (task.failed[worker]) result = -1;
done:
    if (leases && salt_cache_release_many(binding->cache, unique_slots,
            selected_layer, unique_experts, (int)unique_count) != 0)
        result = -1;
    return result;
}

static int lfm_area_batch(const SaltTensorHostBatch *jobs, uint32_t job_count,
                          uint32_t active_rows, SaltAreaScanPlan *plan,
                          void *scratch, size_t scratch_bytes) {
    const LfmHostBinding *binding;
    SaltAreaScanRequest request;
    uint64_t output_tiles = 0u, candidate_tiles = 0u;
    (void)scratch;
    (void)scratch_bytes;
    if (!jobs || job_count < 1u || job_count > 256u || active_rows < 1u ||
        !plan || !jobs[0].tensor ||
        !(binding = (const LfmHostBinding *)jobs[0].tensor->binding) ||
        !binding->compute_pool || binding->workers == 0u)
        return -1;
    for (uint32_t index = 0; index < job_count; index++) {
        if (!jobs[index].tensor || jobs[index].tensor->binding != binding ||
            jobs[index].tensor->rows == 0u || jobs[index].row_count == 0u ||
            UINT64_MAX - output_tiles < jobs[index].tensor->rows ||
            jobs[index].tensor->rows > UINT64_MAX / jobs[index].row_count ||
            UINT64_MAX - candidate_tiles <
                jobs[index].tensor->rows * (uint64_t)jobs[index].row_count)
            return -1;
        output_tiles += jobs[index].tensor->rows;
        candidate_tiles +=
            jobs[index].tensor->rows * (uint64_t)jobs[index].row_count;
    }
    memset(&request, 0, sizeof request);
    request.requested_rows = active_rows;
    request.admitted_rows = active_rows;
    request.resident_workers = binding->workers;
    request.active_worker_limit = binding->workers;
    request.narrow_workers = binding->workers;
    request.wide_allowed = 1u;
    request.output_row_tiles = output_tiles;
    request.candidate_output_tiles = candidate_tiles;
    if (salt_area_scan_plan(&request, plan) != 0 || plan->active_workers == 0u)
        return -1;
    return lfm_host_execute_jobs(jobs, job_count, plan->active_workers);
}

static int lfm_multi_batch(const SaltTensorHostBatch *jobs, uint32_t job_count,
                           void *scratch, size_t scratch_bytes) {
    const LfmHostBinding *binding;
    (void)scratch;
    (void)scratch_bytes;
    if (!jobs || job_count < 1u || !jobs[0].tensor ||
        !(binding = (const LfmHostBinding *)jobs[0].tensor->binding) ||
        binding->workers == 0u)
        return -1;
    return lfm_host_execute_jobs(jobs, job_count, binding->workers);
}

static const SaltTensorHostOps LFM_HOST = {
    .fixed_scratch_requirement = lfm_scratch_requirement,
    .gather_rows = lfm_gather_rows,
    .matvec = lfm_matvec,
    .matvec_batch = lfm_matvec_batch,
    .matvec_multi_batch = lfm_multi_batch,
    .matvec_area_batch = lfm_area_batch,
};

const SaltTensorHostRealizations *lfm_host_realizations(void) {
    static const SaltTensorHostRealizations REALIZATIONS = { .cpu = &LFM_HOST };
    return &REALIZATIONS;
}
