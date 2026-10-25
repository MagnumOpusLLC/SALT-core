#ifndef SALT_INT2_H
#define SALT_INT2_H
#include <stddef.h>
#include <stdint.h>
/* Row-major, four low-bit-first codes/byte, independently padded rows.
 * Weight = (code - 2) * little-endian F32 row scale. No payload expansion.
 * The dot uses the existing BF16 reference's eight-lane, unfused fold.
 * Caller owns validated source spans and disjoint, startup-owned outputs. */
int salt_int2_matvec_batch_rows(const uint8_t *values, const void *scales,
    int rows, int cols, int batch, const float *inputs, float *outputs,
    int first_row, int end_row);
int salt_int2_decode_row(const uint8_t *values, size_t value_bytes,
    const void *scales, size_t scale_bytes, int rows, int cols, int row,
    float *output);
/* Quant-level form of SALT's concatenated row-batch contract. Bindings borrow
 * canonical payloads under existing leases; core partitions complete output
 * rows. No scheduling state, allocation, acquisition, or publication here. */
typedef struct SaltInt2BatchJob {
    const uint8_t *values;
    const void *scales;
    int rows, cols, batch;
    const float *inputs;
    float *outputs;
} SaltInt2BatchJob;
int salt_int2_multi_batch_worker(const SaltInt2BatchJob *jobs, int job_count,
                                int worker_index, int worker_count);
#endif
