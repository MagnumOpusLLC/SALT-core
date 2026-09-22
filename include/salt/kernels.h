/*
 * kernels.h -- scalar mxfp4 math (issue #2, milestone step 2).
 *
 * The mxfp4 pool format written by tools/convert-salt.py:
 *   values:  2 elements per byte, even index in LOW nibble
 *   scales:  one E8M0 byte per block (16 or 32 elements),
 *            value = 2^(b - 127); b = 0 encodes 2^-127
 *   element = +/-{0, 0.5, 1, 1.5, 2, 3, 4, 6} * scale  (MX E2M1)
 *
 * These are the scalar-correct reference kernels: portable C99, no
 * SIMD. The SIMD paths (AVX2/NEON) must verify bit-identical against
 * these on fixtures before they are allowed to replace them.
 */
#ifndef SALT_KERNELS_H
#define SALT_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#define SALT_MXFP4_BLOCK16 16
#define SALT_MXFP4_BLOCK32 32

/* E8M0 block scale value: 2^(b-127); b=0 -> 2^-127. */
float salt_e8m0_value(uint8_t b);

/* Decode n flat mxfp4 elements (even index = low nibble, one E8M0
 * scale per bsize elements) into fp32. */
void salt_mx4_decode(const uint8_t *vals, const uint8_t *scales,
                       int n, int bsize, float *out);

/* y[r] = sum_c W[r,c] * x[c]; W is row-major mxfp4 [R x C].
 * scratch must hold R*C floats. */
void salt_mx4_matvec(const uint8_t *vals, const uint8_t *scales,
                       int R, int C, int bsize, const float *x, float *y,
                       float *scratch);

/* Router: scores[e] = sum_c W[e,c] * x[c] + bias[e]; W is resident
 * fp32 [E x H], bias optional. */
void salt_router_scores(const float *W, const float *bias, int E, int H,
                        const float *x, float *scores);

/* Plain fp32 matvec: y[r] = sum_c W[r,c] * x[c], row-major [R x C]. */
void salt_f32_matvec(const float *W, int R, int C, const float *x,
                     float *y);

/* BF16 matvec: W is brain-float16 (truncated fp32), decoded on the fly.
 * scores[r] = sum_c W[r,c] * x[c] (+ bias[r] when given). */
void salt_bf16_matvec(const uint16_t *W, int R, int C, const float *x,
                      const float *bias, float *y);

/* F8_E4M3 matvec (issue #6): W is F8 bytes [R x C], E8M0 block scales
 * [SR x SC]; element (r,c) uses scale[r*SR/R][c*SC/C] (the checkpoint's
 * per-group scheme). y[r] = sum_c W[r,c] * x[c]. Scalar reference;
 * SIMD path lands with the attention step. */
void salt_f8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC,
                    const float *x, float *y);

/* SIMD dispatch (issue #5): when enabled AND available, mxfp4 decode /
 * matvec / bf16 matvec route to the NEON or AVX2 path. Decode stays
 * bit-identical; matvec accumulates in lane order (tolerance-verified).
 * Default: enabled. */
void salt_kernels_set_simd(int on);
void salt_kernels_set_tiled(int on);   /* A12: force tiled batch path */
int  salt_kernels_simd(void);
/* Thread-context flag for the row-split dispatch: the 8 expert worker
 * threads already saturate the P-cores, so matvecs they call must NOT
 * spawn more threads; the main thread (attention/head path) has idle
 * cores and SHOULD split. exp_run sets the flag around its body. */
void salt_kernels_set_in_expert(int in_expert);
int  salt_kernels_in_expert(void);
/* Resource scope is separate from worker-thread context: grouped routed
 * experts may run on the caller thread and still must bypass trunk GPU
 * routing while retaining the ordinary CPU batch implementation. */
void salt_kernels_set_cpu_expert_scope(int on);
int  salt_kernels_cpu_expert_scope(void);

/* I8 matvec: W is int8 bytes with optional E8M0 block scales [SR x SC]
 * (NULL scales = 1.0). Used by I8-quantized checkpoints (the V4-Flash
 * checkpoint's experts are I8, and the head often is too). */
void salt_i8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y);

/* F8_E4M3 with BF16 block scales (the OFFICIAL Qwen FP8 layout:
 * weight_scale_inv is BF16 [SR,SC]; the E8M0 variant above loses the
 * scale mantissa -> flat logits). */
void salt_f8_matvec_bf16(const uint8_t *W, const uint16_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y);

/* row-range variant for threaded callers: computes rows [r0, r1)
 * with the GLOBAL row scale mapping (sr = (r*SR)/R for the absolute
 * row r). Scalar only (the SIMD path handles the whole matrix). */
void salt_f8_matvec_rows(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1);

/* F16 (IEEE half) matvec + scalar conversion. */
float salt_f16_to_f32(uint16_t h);
void salt_f16_matvec(const uint16_t *W, int R, int C, const float *x,
                     float *y);

/* single F8_E4M3 byte to float (the scalar LUT; used by the hc_*
 * hyper-connection scales). */
float salt_f8_value(uint8_t b);

/* F8 decode of one row of a [V x H] F8_E4M3 tensor with E8M0 block
 * scales [SR x SC] (element (r,c) uses scale[r*SR/V][c*SC/H]); NULL
 * scales means 1.0 everywhere. Used by the embedding gather. */
void salt_f8_decode_row(const uint8_t *W, const uint8_t *scales,
                        int V, int H, int SR, int SC, int row, float *out);

/* ------------------------------------------------------------------ */
/* MLX 4-bit (Qwen3.5-35B-A3B MLX quant) kernel                       */
/* ------------------------------------------------------------------ */
/*
 * The MLX 4-bit pool format (split-mlx-switchmlp.py output):
 *   values:  one U32 per 8 elements, low nibble first:
 *            elem(k) = (u32 >> (4*(k%8))) & 0xF, k in [0, R*C)
 *   scales:  one BF16 per group of G=64 elements, row-major:
 *            scale[g] with g = (r*C + c) / G
 *   biases:  one BF16 per group (same layout as scales)
 *   element(r,c) = q * scale[g] + bias[g]
 *
 * MLX affine stores q in [0,15] and learns the per-group bias; there is no
 * implicit -8 offset. This is unlike mxfp4's E2M1. Scalar reference; SIMD
 * implementations must remain bit-identical.
 */
#define SALT_MLX4_GROUP 64

/* Decode n elements (flattened row-major [R x C]) into fp32. */
void salt_q4_decode(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int n, float *out);

/* y[r] = sum_c W[r,c] * x[c]; W is MLX 4-bit packed. */
int salt_q4_matvec_status(const uint32_t *vals, const uint16_t *scales,
                          const uint16_t *biases, int R, int C,
                          const float *x, float *y);
void salt_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int R, int C,
                      const float *x, float *y);

/* y[r] = sum_c W[r,c] * x[c]; W is MLX 8-bit affine packed (4
 * elems/U32 word, byte LUT over 0..255). Used for the 8-bit router
 * gates and shared-expert gates on the 3.6 repo (config
 * quantization map bits=8). Scalar LUT only -- the 4-bit SIMD path
 * is untouched (bit-identical contract). */
void salt_q8_matvec(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int R, int C,
                      const float *x, float *y);

/* Q8 weight-stationary batch over one disjoint output-row interval. Each row
 * is decoded once, then consumed by B token rows with the serial accumulator
 * order. Intended for a session-owned persistent row executor. */
int salt_q8_matvec_batch_rows(const uint32_t *vals, const uint16_t *scales,
                              const uint16_t *biases, int R, int C, int B,
                              const float *xs, float *ys, int r0, int r1);
int salt_q8_matvec_batch_rows_scratch(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    const float *xs, float *ys, int r0, int r1,
    float *row_scratch, size_t row_capacity);

/* Rank-1 update (the mental-model op, dger-style): Y[kd][vd] +=
 * u[kd] * v[vd]^T  -- the delta-state write S += k (x) delta.
 * Column-major Y (kd rows x vd cols, row stride vd). The scalar
 * accumulation order (per output element, u[i]*v[j] in i-major,
 * j-major) is the bit-identity anchor; the SIMD path replicates it
 * with 4-wide lanes in the same order. Deterministic. */
void salt_rank1(float *Y, const float *u, const float *v,
                int kd, int vd);

/* Two independent MLX4 matvecs (same x, disjoint outputs) in ONE
 * 8-way row split over the combined row space -- overlaps the linear
 * attention qkv+z projections instead of two sequential spawns. */
int salt_q4_matvec2(const uint32_t *v1, const uint16_t *s1,
                      const uint16_t *b1, int R1,
                      const uint32_t *v2, const uint16_t *s2,
                      const uint16_t *b2, int R2, int C,
                      const float *x, float *y1, float *y2);

/* Batched prefill matvec (M1 of prefill-batch): Y[B][R] = W[R x C] *
 * X[B][C]. Dequant once per weight row, reuse across B token
 * vectors. Bit-identical per (row, token) to salt_q4_matvec. */
int salt_q4_matvec_batch(const uint32_t *vals, const uint16_t *scales,
                           const uint16_t *biases, int R, int C, int B,
                           const float *xs, float *ys);

/* Execute one disjoint output-row range of the same CPU batch kernel without
 * creating workers. A session-owned persistent executor partitions [0,R) and
 * invokes this synchronously from each active worker. */
int salt_q4_matvec_batch_rows(const uint32_t *vals, const uint16_t *scales,
                              const uint16_t *biases, int R, int C, int B,
                              const float *xs, float *ys, int r0, int r1);
int salt_q4_matvec_batch_tile(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    const float *xs, float *ys,
    int b0, int b1, int r0, int r1);
int salt_q4_matvec_batch_rows_scratch(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    const float *xs, float *ys, int r0, int r1,
    float *row_scratch, size_t row_capacity);

/* The multi-weight batch (the GQA q/k/v fusion): N jobs with their
 * OWN weights run as ONE thread-shared pass -- the weighted partition
 * spans the jobs' concatenated rows. Bit-identical per (job, row,
 * token) to the single-weight batch. */
typedef struct {
    const uint32_t *vals; const uint16_t *scales, *biases;
    int R, C, B;
    const float *xs;   /* [B][C] row-major */
    float *ys;         /* [B][R] row-major */
    int r0, r1;        /* row range [r0, r1) of W */
} SaltBatchJob;

int salt_q4_multi_batch(SaltBatchJob *jobs, int njobs);
/* Execute one canonical concatenated output-row range across heterogeneous
 * jobs. Each job retains its own weights, B, inputs, and direct output range;
 * only row ownership is partitioned. No workers are created here. */
int salt_q4_multi_batch_rows(SaltBatchJob *jobs, int njobs,
                             int64_t concat_r0, int64_t concat_r1);
int salt_q4_multi_batch_rows_scratch(
    SaltBatchJob *jobs, int njobs, int64_t concat_r0, int64_t concat_r1,
    float *row_scratch, size_t row_capacity);

#endif /* SALT_KERNELS_H */
