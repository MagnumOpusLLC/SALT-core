#include "salt/int2.h"
#include "salt/kernels.h"
#include "salt/simd.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static float row_scale(const void *source, int row) {
    const uint8_t *p = (const uint8_t *)source + (size_t)row * 4u;
    uint32_t bits = (uint32_t)p[0] | (uint32_t)p[1] << 8 |
        (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
    float result;
    memcpy(&result, &bits, sizeof result);
    return result;
}
static float weight(const uint8_t *row, int column, float scale) {
    int code = (row[(unsigned)column / 4u] >>
        (2u * ((unsigned)column % 4u))) & 3;
    return (float)(code - 2) * scale;
}
int salt_int2_decode_row(const uint8_t *values, size_t value_bytes,
    const void *scales, size_t scale_bytes, int rows, int cols, int row,
    float *output) {
    size_t stride;
    float scale;
    if (!values || !scales || !output || rows < 1 || cols < 1 ||
        row < 0 || row >= rows) return -1;
    stride = ((size_t)cols + 3u) / 4u;
    if ((size_t)rows > SIZE_MAX / stride ||
        (size_t)rows > SIZE_MAX / 4u || value_bytes < (size_t)rows * stride ||
        scale_bytes < (size_t)rows * 4u) return -1;
    scale = row_scale(scales, row);
    if (!isfinite(scale) || scale < 0.0f) return -1;
    for (int c = 0; c < cols; c++)
        output[c] = weight(values + (size_t)row * stride, c, scale);
    return 0;
}
int salt_int2_matvec_batch_rows(const uint8_t *values, const void *scales,
    int rows, int cols, int batch, const float *inputs, float *outputs,
    int first_row, int end_row) {
    size_t stride;
    if (!values || !scales || !inputs || !outputs || rows < 1 || cols < 1 ||
        batch < 1 || first_row < 0 || end_row < first_row || end_row > rows ||
        (size_t)rows > SIZE_MAX / ((size_t)batch * sizeof(float)) ||
        (size_t)cols > SIZE_MAX / ((size_t)batch * sizeof(float))) return -1;
    stride = ((size_t)cols + 3u) / 4u;
    if ((size_t)rows > SIZE_MAX / stride) return -1;
    for (int r = first_row; r < end_row; r++)
        if (!isfinite(row_scale(scales, r)) || row_scale(scales, r) < 0.0f)
            return -1;
    if (salt_kernels_simd()) {
        int rc = salt_simd_int2_matvec_batch_rows(values, scales, rows, cols,
            batch, inputs, outputs, first_row, end_row);
        if (rc != 1) return rc;
    }
    for (int r = first_row; r < end_row; r++) {
        const uint8_t *w = values + (size_t)r * stride;
        float scale = row_scale(scales, r);
        for (int b = 0; b < batch; b++) {
            const float *x = inputs + (size_t)b * cols;
            float a[8] = {0};
            int c = 0;
            for (; c + 7 < cols; c += 8)
                for (int lane = 0; lane < 8; lane++) {
                    float product = weight(w, c + lane, scale) * x[c + lane];
                    a[lane] += product;
                }
            float sum = ((a[0] + a[2]) + (a[4] + a[6])) +
                        ((a[1] + a[3]) + (a[5] + a[7]));
            for (; c < cols; c++) {
                float product = weight(w, c, scale) * x[c];
                sum += product;
            }
            if (!isfinite(sum)) return -1;
            outputs[(size_t)b * rows + r] = sum;
        }
    }
    return 0;
}

int salt_int2_multi_batch_worker(const SaltInt2BatchJob *jobs, int job_count,
                                int worker_index, int worker_count) {
    uint64_t total = 0, base = 0, first, end;
    if (!jobs || job_count < 1 || worker_count < 1 || worker_index < 0 ||
        worker_index >= worker_count) return -1;
    for (int j = 0; j < job_count; j++) {
        const SaltInt2BatchJob *p = &jobs[j];
        uint64_t row_work;
        if (!p->values || !p->scales || !p->inputs || !p->outputs ||
            p->rows < 1 || p->cols < 1 || p->batch < 1) return -1;
        row_work = (uint64_t)(unsigned)p->cols * (unsigned)p->batch;
        if ((uint64_t)(unsigned)p->rows > (UINT64_MAX - total) / row_work)
            return -1;
        total += (uint64_t)(unsigned)p->rows * row_work;
    }
    first = (total / (unsigned)worker_count) * (unsigned)worker_index +
        (total % (unsigned)worker_count) * (unsigned)worker_index / (unsigned)worker_count;
    end = (total / (unsigned)worker_count) * ((unsigned)worker_index + 1u) +
        (total % (unsigned)worker_count) * ((unsigned)worker_index + 1u) / (unsigned)worker_count;
    for (int j = 0; j < job_count; j++) {
        const SaltInt2BatchJob *p = &jobs[j];
        uint64_t row_work = (uint64_t)(unsigned)p->cols * (unsigned)p->batch;
        uint64_t limit = base + (uint64_t)(unsigned)p->rows * row_work;
        if (first < limit && end > base) {
            uint64_t lo = first > base ? first - base : 0;
            uint64_t hi = end < limit ? end - base : limit - base;
            int r0 = (int)(lo / row_work + (lo % row_work != 0));
            int r1 = (int)(hi / row_work + (hi % row_work != 0));
            if (r0 < r1 && salt_int2_matvec_batch_rows(p->values, p->scales,
                    p->rows, p->cols, p->batch, p->inputs, p->outputs, r0, r1))
                return -1;
        }
        base = limit;
    }
    return 0;
}
