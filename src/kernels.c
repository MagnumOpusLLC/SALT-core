/* kernels.c -- scalar mxfp4 reference kernels (portable C99). */
#include "salt/kernels.h"
#include "salt/gpu.h"
#include "salt/simd.h"
#include "compiler.h"
#include "thread-lifecycle.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

static int g_simd = 1;
SALT_THREAD_LOCAL int g_in_expert = 0;
SALT_THREAD_LOCAL int g_cpu_expert_scope = 0;

/* env cache: SALT_ATTN_THREADS gates every threaded matvec dispatch
 * (hundreds of calls/token); getenv is a libc lock + environ scan.
 * The value is fixed at process launch -> parse ONCE (constructor
 * runs before main, no thread race). Semantics unchanged: unset ->
 * 8, else clamp [1,32]. */
static int g_attn_threads_value = 8;
static pthread_once_t g_attn_threads_once = PTHREAD_ONCE_INIT;
static void kernels_env_init(void) {
    const char *env = getenv("SALT_ATTN_THREADS");
    if (env) {
        int v = atoi(env);
        if (v >= 1 && v <= 32) g_attn_threads_value = v;
    }
}

static int attn_threads(void) {
    if (pthread_once(&g_attn_threads_once, kernels_env_init) != 0)
        return 1;
    return g_attn_threads_value;
}

void salt_kernels_set_simd(int on) { g_simd = on ? 1 : 0; }

int salt_kernels_simd(void) { return g_simd && salt_simd_available(); }

void salt_kernels_set_in_expert(int in_expert) { g_in_expert = in_expert ? 1 : 0; }

int salt_kernels_in_expert(void) { return g_in_expert; }

void salt_kernels_set_cpu_expert_scope(int on) {
    g_cpu_expert_scope = on ? 1 : 0;
}

int salt_kernels_cpu_expert_scope(void) { return g_cpu_expert_scope; }

float salt_e8m0_value(uint8_t b) {
    return ldexpf(1.0f, (int)b - 127);   /* b = 0 -> 2^-127 */
}

static float e2m1_mag(int idx) {
    static const float M[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                               2.0f, 3.0f, 4.0f, 6.0f};
    return M[idx & 7];
}

void salt_mx4_decode(const uint8_t *vals, const uint8_t *scales,
                       int n, int bsize, float *out) {
    if (salt_kernels_simd()) {
        salt_simd_mx4_decode(vals, scales, n, bsize, out);
        return;
    }
    for (int i = 0; i < n; i++) {
        int nib = (vals[i >> 1] >> ((i & 1) ? 4 : 0)) & 0xF;
        float v = e2m1_mag(nib);
        if (nib & 8) v = -v;
        out[i] = v * salt_e8m0_value(scales[i / bsize]);
    }
}

void salt_mx4_matvec(const uint8_t *vals, const uint8_t *scales,
                       int R, int C, int bsize, const float *x, float *y,
                       float *scratch) {
    if (salt_kernels_simd()) {
        salt_simd_mx4_matvec(vals, scales, R, C, bsize, x, y, scratch);
        return;
    }
    salt_mx4_decode(vals, scales, R * C, bsize, scratch);
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = scratch + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

void salt_router_scores(const float *W, const float *bias, int E, int H,
                        const float *x, float *scores) {
    for (int e = 0; e < E; e++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)e * H;
        for (int c = 0; c < H; c++) acc += wr[c] * x[c];
        scores[e] = acc + (bias ? bias[e] : 0.0f);
    }
}

void salt_f32_matvec(const float *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += wr[c] * x[c];
        y[r] = acc;
    }
}

static float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;   /* bf16 = top half of fp32 */
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static uint16_t load_u16_unaligned(const uint16_t *base, size_t index) {
    uint16_t value;
    memcpy(&value, (const unsigned char *)(const void *)base + index * 2,
           sizeof value);
    return value;
}

static uint32_t load_u32_unaligned(const uint32_t *base, size_t index) {
    uint32_t value;
    memcpy(&value, (const unsigned char *)(const void *)base + index * 4,
           sizeof value);
    return value;
}

void salt_bf16_matvec(const uint16_t *W, int R, int C, const float *x,
                      const float *bias, float *y) {
    if (salt_kernels_simd()) {
        salt_simd_bf16_matvec(W, R, C, x, bias, y);
        return;
    }
    /* BIT-EXACTNESS: this scalar body MUST produce the same bits as
     * salt_simd_bf16_matvec. Mirror the NEON structure exactly:
     * 8 accumulators a0..a7 (lane j accumulates elements c%8==j --
     * the NEON holds a0..a3 in acc0, a4..a7 in acc1); per-element
     * rounding t=w*x then +=acc (mul rounds, add rounds, NO fusing
     * -- fp-contract=off); final tree: t0=(a0+a2)+(a4+a6),
     * t1=(a1+a3)+(a5+a7), s=t0+t1; then the scalar tail for
     * c past the 8-aligned block. Verified by the KV md5 A/B. */
    for (int r = 0; r < R; r++) {
        const uint16_t *wr = W + (size_t)r * C;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
        int c = 0;
        for (; c + 7 < C; c += 8) {
            float t;
            t = bf16_to_f32(wr[c + 0]) * x[c + 0]; a0 += t;
            t = bf16_to_f32(wr[c + 1]) * x[c + 1]; a1 += t;
            t = bf16_to_f32(wr[c + 2]) * x[c + 2]; a2 += t;
            t = bf16_to_f32(wr[c + 3]) * x[c + 3]; a3 += t;
            t = bf16_to_f32(wr[c + 4]) * x[c + 4]; a4 += t;
            t = bf16_to_f32(wr[c + 5]) * x[c + 5]; a5 += t;
            t = bf16_to_f32(wr[c + 6]) * x[c + 6]; a6 += t;
            t = bf16_to_f32(wr[c + 7]) * x[c + 7]; a7 += t;
        }
        /* final pairwise tree (exact NEON order) */
        float s = ((a0 + a2) + (a4 + a6)) + ((a1 + a3) + (a5 + a7));
        for (; c < C; c++) s += bf16_to_f32(wr[c]) * x[c];
        y[r] = s + (bias ? bias[r] : 0.0f);
    }
}

/* ------------------------------------------------------------------ */
/* MLX 4-bit (Qwen3.5-35B-A3B) -- scalar reference                     */
/* ------------------------------------------------------------------ */
/*
 * Packing (verified against the real repo, 2026-08-06):
 *   values  U32, 8 nibbles per word, low nibble first:
 *           elem(k) = (v[k/8] >> (4*(k%8))) & 0xF
 *   scales  BF16, one per 64 elements, row-major group index
 *   biases  BF16, same layout as scales
 *   element = q * scale[g] + bias[g], g = k / 64
 */

void salt_q4_decode(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int n, float *out) {
    float lut[16];
    long gcur = -1;
    for (int k = 0; k < n; k++) {
        int g = k / SALT_MLX4_GROUP;
        if (g != gcur) {
            gcur = g;
            float s = bf16_to_f32(load_u16_unaligned(scales, (size_t)g));
            float b = biases ?
                bf16_to_f32(load_u16_unaligned(biases, (size_t)g)) : 0.0f;
            /* MLX affine dequant: w = scale*q + bias (q raw 0..15).
             * NO -8 offset -- that was a systematic per-group error */
            for (int q = 0; q < 16; q++)
                lut[q] = (float)q * s + b;
        }
        int q = (int)((load_u32_unaligned(vals, (size_t)k >> 3) >>
                       (4 * (k & 7))) & 0xFu);
        out[k] = lut[q];
    }
}

/* row-partitioned q4 matvec worker (large matrices: expert MLPs,
 * the 248320-row head) */
typedef struct {
    const uint32_t *vals;
    const uint16_t *scales, *biases;
    int R, C, r0, r1;
    const float *x;
    float *y;
} Mlx4RowJob;

static float q4_canonical_row(const uint32_t *vals,
                              const uint16_t *scales,
                              const uint16_t *biases,
                              int C, const float *x, int r) {
    float acc[32];
    for (int i = 0; i < 32; i++) acc[i] = 0.0f;
    for (int c = 0; c < C; c++) {
        size_t k = (size_t)r * (size_t)C + (size_t)c;
        size_t g = k / SALT_MLX4_GROUP;
        uint32_t word = load_u32_unaligned(vals, k >> 3);
        int q = (int)((word >> (4 * (int)(k & 7u))) & 0xFu);
        float scale = bf16_to_f32(load_u16_unaligned(scales, g));
        float bias = biases ?
            bf16_to_f32(load_u16_unaligned(biases, g)) : 0.0f;
        float dequant = (float)q * scale;
        dequant += bias;
        float product = dequant * x[c];
        acc[c & 31] += product;
    }
    float lane0 = 0.0f, lane1 = 0.0f;
    float lane2 = 0.0f, lane3 = 0.0f;
    for (int a = 0; a < 8; a++) {
        lane0 += acc[4 * a];
        lane1 += acc[4 * a + 1];
        lane2 += acc[4 * a + 2];
        lane3 += acc[4 * a + 3];
    }
    float t0 = lane0 + lane2;
    float t1 = lane1 + lane3;
    return t0 + t1;
}

static void q4_canonical_rows(const uint32_t *vals,
                              const uint16_t *scales,
                              const uint16_t *biases,
                              int R, int C, const float *x, float *y,
                              int r0, int r1) {
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++)
        y[r] = q4_canonical_row(vals, scales, biases, C, x, r);
}

static void *q4_row_worker(void *arg) {
    Mlx4RowJob *j = (Mlx4RowJob *)arg;
    salt_simd_q4_matvec(j->vals, j->scales, j->biases, j->R, j->C,
                        j->x, j->y, j->r0, j->r1);
    return NULL;
}

int salt_q4_matvec_status(const uint32_t *vals, const uint16_t *scales,
                          const uint16_t *biases, int R, int C,
                          const float *x, float *y) {
    if (!vals || !scales || !x || !y || R < 1 || C < 1 ||
        (size_t)R > SIZE_MAX / (size_t)C)
        return -1;
    /* SIMD is a physical realization of the same logical 32-accumulator
     * graph. Row partitioning changes only ownership, never operation order. */
    if (salt_kernels_simd() && (C % 8) == 0 &&
        (C % SALT_MLX4_GROUP) == 0 &&
        C / SALT_MLX4_GROUP <= SALT_SIMD_Q4_MAX_GROUPS_PER_ROW) {
        int nth = 1;
        if (!salt_kernels_in_expert() && R >= 2048)
            nth = attn_threads();
        if (nth > 1 && nth <= 16 && R >= 2048 &&
            !salt_kernels_in_expert()) {
            pthread_t th[16];
            Mlx4RowJob job[16];
            unsigned char started[16] = {0};
            int chunk = R / nth + (R % nth != 0);
            for (int t = 0; t < nth; t++) {
                int64_t wr0 = (int64_t)t * chunk;
                int64_t wr1 = wr0 + chunk;
                int r0 = wr0 < R ? (int)wr0 : R;
                int r1 = wr1 < R ? (int)wr1 : R;
                if (r0 >= r1) continue;
                job[t].vals = vals; job[t].scales = scales;
                job[t].biases = biases; job[t].R = R; job[t].C = C;
                job[t].x = x; job[t].y = y;
                job[t].r0 = r0; job[t].r1 = r1;
                if (pthread_create(&th[t], NULL, q4_row_worker, &job[t]) == 0)
                    started[t] = 1;
                else
                    (void)q4_row_worker(&job[t]);
            }
            for (int t = 0; t < nth; t++)
                if (started[t])
                    (void)salt_join_one_or_exit(
                        th[t], pthread_join, "q4-matvec", t);
            return 0;
        }
        salt_simd_q4_matvec(vals, scales, biases, R, C, x, y, 0, R);
        return 0;
    }
    q4_canonical_rows(vals, scales, biases, R, C, x, y, 0, R);
    return 0;
}

void salt_q4_matvec(const uint32_t *vals, const uint16_t *scales,
                    const uint16_t *biases, int R, int C,
                    const float *x, float *y) {
    (void)salt_q4_matvec_status(vals, scales, biases, R, C, x, y);
}

/* 8-bit affine (MLX 8-bit: 4 elems/U32 word, signed int8 center
 * 0? MLX affine 8-bit stores the quantized value in the byte, the
 * LUT is v*s+b over the FULL byte range (0..255). This is the
 * decoder for the 3.6 repo's 8-bit router gates and shared-expert
 * gate (config quantization map: bits=8). Kept separate from
 * salt_q4_matvec so the 4-bit path stays bit-identical. */
void salt_q8_matvec(const uint32_t *vals, const uint16_t *scales,
                      const uint16_t *biases, int R, int C,
                      const float *x, float *y) {
    const unsigned char *b8 = (const unsigned char *)(const void *)vals;
    float lut[256];
    long gcur = -1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const float *xr = x;
        for (int c = 0; c < C; c++) {
            long k = (long)r * C + c;
            long g = k / SALT_MLX4_GROUP;
            if (g != gcur) {
                gcur = g;
                float s = bf16_to_f32(
                    load_u16_unaligned(scales, (size_t)g));
                float b = biases ? bf16_to_f32(
                    load_u16_unaligned(biases, (size_t)g)) : 0.0f;
                for (int q = 0; q < 256; q++)
                    lut[q] = (float)q * s + b;
            }
            int q = b8[k];
            acc += lut[q] * xr[c];
        }
        y[r] = acc;
    }
}

int salt_q8_matvec_batch_rows(const uint32_t *vals, const uint16_t *scales,
                              const uint16_t *biases, int R, int C, int B,
                              const float *xs, float *ys, int r0, int r1) {
    const unsigned char *bytes = (const unsigned char *)(const void *)vals;
    SALT_THREAD_LOCAL float *row = NULL;
    SALT_THREAD_LOCAL size_t row_capacity = 0;
    if (!vals || !scales || !xs || !ys || R < 1 || C < 1 || B < 1 ||
        r0 < 0 || r0 > r1 || r1 > R ||
        (size_t)R > SIZE_MAX / (size_t)C ||
        (size_t)B > SIZE_MAX / (size_t)C ||
        (size_t)B > SIZE_MAX / (size_t)R)
        return -1;
    if ((size_t)C > row_capacity) {
        float *grown = (float *)realloc(row, (size_t)C * sizeof(float));
        if (!grown) return -1;
        row = grown;
        row_capacity = (size_t)C;
    }
    for (int r = r0; r < r1; r++) {
        float lut[256];
        size_t group_current = SIZE_MAX;
        for (int c = 0; c < C; c++) {
            size_t element = (size_t)r * (size_t)C + (size_t)c;
            size_t group = element / SALT_MLX4_GROUP;
            if (group != group_current) {
                float scale = bf16_to_f32(
                    load_u16_unaligned(scales, group));
                float bias = biases ? bf16_to_f32(
                    load_u16_unaligned(biases, group)) : 0.0f;
                for (int q = 0; q < 256; q++)
                    lut[q] = (float)q * scale + bias;
                group_current = group;
            }
            row[c] = lut[bytes[element]];
        }
        for (int token = 0; token < B; token++) {
            const float *input = xs + (size_t)token * (size_t)C;
            float accumulator = 0.0f;
            for (int c = 0; c < C; c++)
                accumulator += row[c] * input[c];
            ys[(size_t)token * (size_t)R + (size_t)r] = accumulator;
        }
    }
    return 0;
}

/* Rank-1 update (dger-style): Y[kd][vd] += u[kd] x v[vd]^T.
 * The mental-model op of the delta rule (the state write). The
 * scalar anchor's accumulation order (i-major over kd, j-major over
 * vd, each element y[i*vd+j] += u[i]*v[j]) is the bit-identity
 * contract; the SIMD path replicates it 4-wide in the same order. */
void salt_rank1(float *Y, const float *u, const float *v,
                int kd, int vd) {
#ifdef __aarch64__
    if (salt_kernels_simd()) {
        for (int i = 0; i < kd; i++) {
            float32x4_t ui = vdupq_n_f32(u[i]);
            float *yi = Y + (size_t)i * vd;
            int j = 0;
            for (; j + 4 <= vd; j += 4) {
                float32x4_t acc = vld1q_f32(yi + j);
                acc = vfmaq_f32(acc, ui, vld1q_f32(v + j));
                vst1q_f32(yi + j, acc);
            }
            for (; j < vd; j++)
                yi[j] += u[i] * v[j];
        }
        return;
    }
#endif
    for (int i = 0; i < kd; i++) {
        float ui = u[i];
        float *yi = Y + (size_t)i * vd;
        for (int j = 0; j < vd; j++)
            yi[j] += ui * v[j];
    }
}

/* fp8_e4m3fn decode table (issue #6); e=0 subnormal-ish, e=15 clamp. */
static float fp8_lut[256];
static int  fp8_lut_ready = 0;

static void fp8_lut_build(void) {
    for (int b = 0; b < 256; b++) {
        int s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
        float v;
        if (e == 0)
            v = (float)m * 0.001953125f;         /* m * 2^-9 */
        else if (e == 0xF)
            v = 448.0f;                          /* inf/nan -> E4M3FN max */
        else
            v = (1.0f + (float)m / 8.0f) * ldexpf(1.0f, e - 7);
        fp8_lut[b] = s ? -v : v;
    }
    fp8_lut_ready = 1;   /* benign race: identical values either way */
}

void salt_f8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    /* SIMD when the scale blocks are 16-aligned: SC == 1 (per-row) or
     * the block width C/SC is a multiple of 16. Otherwise scalar. */
    if (salt_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            salt_simd_f8_matvec(W, scales, R, C, SR, ssc, x, y, 0, R);
            return;
        }
    }
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);  /* col block */
            float s = scales ? salt_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

/* F8_E4M3 with BF16 block scales (the OFFICIAL Qwen FP8 layout:
 * weight_scale_inv is BF16 [SR,SC] per block, NOT E8M0). Reads the
 * scale as BF16 -- the E8M0 encode loses the mantissa (~41% per-block
 * scale error -> flattened logits -> the "!" attractor). */
void salt_f8_matvec_bf16(const uint8_t *W, const uint16_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y) {
    /* SIMD only when the whole logical row is 16-aligned and the scale
     * blocks are 16-aligned: SC == 1 (per-row) or block width C/SC is
     * a multiple of 16. */
    if (salt_kernels_simd() && C % 16 == 0) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            salt_simd_f8_matvec_bf16(W, scales, R, C, SR, ssc, x, y, 0, R);
            return;
        }
    }
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    /* BIT-EXACTNESS: this scalar body MUST produce the same bits as
     * salt_simd_f8_matvec_bf16. Mirror the NEON structure exactly:
     * 16 accumulators acc[0..15] (acc[i] accumulates element c+i of
     * each 16-block -- the NEON holds acc[0..3] in a0, [4..7] in a1,
     * [8..11] in a2, [12..15] in a3); per-element rounding
     * t=(V*s) then t*x then +=acc (NO fused FMA -- fp-contract=off);
     * final tree: r0=acc0+acc2, r1=acc1+acc3, r2=acc4+acc6,
     * r3=acc5+acc7, t0=r0+r2, t1=r1+r3, t0+=acc8+acc10,
     * t1+=acc9+acc11, t0+=acc12+acc14, t1+=acc13+acc15, s=t0+t1.
     * Verified by the KV md5 A/B (SIMD on vs --no-simd). */
    for (int r = 0; r < R; r++) {
        int sr = (int)(((int64_t)r * SR) / R);      /* row block */
        const uint8_t *wr = W + (size_t)r * C;
        float acc[16];
        for (int i = 0; i < 16; i++) acc[i] = 0.0f;
        int c = 0;
        for (; c + 15 < C; c += 16) {
            int sc = (int)(((int64_t)c * SC) / C);  /* col block */
            float s = scales ? bf16_to_f32(scales[sr * SC + sc]) : 1.0f;
            for (int i = 0; i < 16; i++) {
                /* t = V*s (rounded); t = t*x (rounded); acc += t */
                float t = fp8_lut[wr[c + i]] * s;
                t = t * x[c + i];
                acc[i] += t;
            }
        }
        /* final pairwise tree (exact NEON order) */
        float r0 = acc[0] + acc[2];
        float r1 = acc[1] + acc[3];
        float r2 = acc[4] + acc[6];
        float r3 = acc[5] + acc[7];
        float t0 = r0 + r2;
        float t1 = r1 + r3;
        t0 += acc[8] + acc[10];
        t1 += acc[9] + acc[11];
        t0 += acc[12] + acc[14];
        t1 += acc[13] + acc[15];
        float s = t0 + t1;
        for (; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);  /* col block */
            float sl = scales ? bf16_to_f32(scales[sr * SC + sc]) : 1.0f;
            s += fp8_lut[wr[c]] * sl * x[c];
        }
        y[r] = s;
    }
}

void salt_f8_matvec_rows(const uint8_t *W, const uint8_t *scales,
                         int R, int C, int SR, int SC, const float *x,
                         float *y, int r0, int r1) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    if (r0 < 0) r0 = 0;
    if (r1 > R) r1 = R;
    for (int r = r0; r < r1; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);      /* GLOBAL row block */
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? salt_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += fp8_lut[wr[c]] * s * x[c];
        }
        y[r] = acc;
    }
}

void salt_f8_decode_row(const uint8_t *W, const uint8_t *scales,
                        int V, int H, int SR, int SC, int row, float *out) {
    if (!fp8_lut_ready) fp8_lut_build();
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    const uint8_t *wr = W + (size_t)row * H;
    int sr = (int)(((int64_t)row * SR) / V);
    for (int c = 0; c < H; c++) {
        int sc = (int)(((int64_t)c * SC) / H);
        float s = scales ? salt_e8m0_value(scales[sr * SC + sc]) : 1.0f;
        out[c] = fp8_lut[wr[c]] * s;
    }
}

float salt_f8_value(uint8_t b) {
    if (!fp8_lut_ready) fp8_lut_build();
    return fp8_lut[b];
}

void salt_i8_matvec(const uint8_t *W, const uint8_t *scales,
                    int R, int C, int SR, int SC, const float *x,
                    float *y) {
    if (salt_kernels_simd()) {
        int ssc = SC < 1 ? 1 : SC;
        if (ssc == 1 || (C % ssc == 0 && ((C / ssc) % 16) == 0)) {
            salt_simd_i8_matvec(W, scales, R, C, SR, ssc, x, y);
            return;
        }
    }
    if (SR < 1) SR = 1;
    if (SC < 1) SC = 1;
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        int sr = (int)(((int64_t)r * SR) / R);
        const uint8_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) {
            int sc = (int)(((int64_t)c * SC) / C);
            float s = scales ? salt_e8m0_value(scales[sr * SC + sc])
                             : 1.0f;
            acc += (float)(int8_t)wr[c] * s * x[c];
        }
        y[r] = acc;
    }
}

float salt_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400u)) { man <<= 1; exp--; }
            man &= 0x3FFu;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

void salt_f16_matvec(const uint16_t *W, int R, int C, const float *x,
                     float *y) {
    for (int r = 0; r < R; r++) {
        float acc = 0.0f;
        const uint16_t *wr = W + (size_t)r * C;
        for (int c = 0; c < C; c++) acc += salt_f16_to_f32(wr[c]) * x[c];
        y[r] = acc;
    }
}

/* ---- combined two-matvec row split (qkv + z, one spawn) ---------- */
typedef struct {
    const uint32_t *v1; const uint16_t *s1, *b1; int R1;
    const uint32_t *v2; const uint16_t *s2, *b2; int R2;
    int C;
    const float *x;
    float *y1, *y2;
    int r0, r1;          /* combined row range [r0, r1) over R1+R2 */
} Mlx4RowJob2;

static void *q4_row_worker2(void *arg) {
    Mlx4RowJob2 *j = (Mlx4RowJob2 *)arg;
    /* combined row space: [0,R1) -> matvec 1, [R1, R1+R2) -> matvec 2 */
    if (j->r1 <= j->R1) {
        salt_simd_q4_matvec(j->v1, j->s1, j->b1, j->R1, j->C,
                              j->x, j->y1, j->r0, j->r1);
    } else if (j->r0 >= j->R1) {
        salt_simd_q4_matvec(j->v2, j->s2, j->b2, j->R2, j->C,
                              j->x, j->y2, j->r0 - j->R1, j->r1 - j->R1);
    } else {
        salt_simd_q4_matvec(j->v1, j->s1, j->b1, j->R1, j->C,
                              j->x, j->y1, j->r0, j->R1);
        salt_simd_q4_matvec(j->v2, j->s2, j->b2, j->R2, j->C,
                              j->x, j->y2, 0, j->r1 - j->R1);
    }
    return NULL;
}

/* Two independent MLX4 matvecs with the same x, computed in ONE
 * 8-way row split over the combined row space (R1+R2). The linear
 * attention qkv (8192 rows) and z (4096) projections both read xin
 * and write disjoint outputs -- running them sequentially cost two
 * spawn/join cycles per layer; this overlaps them. Same per-row
 * math as salt_q4_matvec (bit-identical rows), just partitioned
 * together. Returns 0 if both computed, -1 on fallback. */
int salt_q4_matvec2(const uint32_t *v1, const uint16_t *s1,
                      const uint16_t *b1, int R1,
                      const uint32_t *v2, const uint16_t *s2,
                      const uint16_t *b2, int R2, int C,
                      const float *x, float *y1, float *y2) {
    /* CPU-only fusion is not a valid source-backed trunk dispatch. */
    if (salt_gpu_mapped_only()) return -1;
    if (!salt_kernels_simd() || (C % 8) != 0 ||
        (C % SALT_MLX4_GROUP) != 0 || R1 < 1 || R2 < 1)
        return -1;
    if (salt_kernels_in_expert()) return -1;   /* expert threads: no spawn */
    int configured_threads = attn_threads();
    if (!v1 || !s1 || !v2 || !s2 || !x || !y1 || !y2 ||
        R1 > INT_MAX - R2 || configured_threads > 32)
        return -1;
    int total = R1 + R2;
    int nth = configured_threads;
    if (nth < 2) {
        salt_simd_q4_matvec(v1, s1, b1, R1, C, x, y1, 0, R1);
        salt_simd_q4_matvec(v2, s2, b2, R2, C, x, y2, 0, R2);
        return 0;
    }
    int chunk = total / nth + (total % nth != 0);
    pthread_t th[32];
    Mlx4RowJob2 job[32];
    unsigned char started[32] = {0};
    for (int t = 0; t < nth; t++) {
        int64_t wr0 = (int64_t)t * chunk;
        int64_t wr1 = wr0 + chunk;
        int r0 = wr0 < total ? (int)wr0 : total;
        int r1 = wr1 < total ? (int)wr1 : total;
        if (r0 >= r1) continue;
        job[t].v1 = v1; job[t].s1 = s1; job[t].b1 = b1; job[t].R1 = R1;
        job[t].v2 = v2; job[t].s2 = s2; job[t].b2 = b2; job[t].R2 = R2;
        job[t].C = C; job[t].x = x; job[t].y1 = y1; job[t].y2 = y2;
        job[t].r0 = r0; job[t].r1 = r1;
        if (pthread_create(&th[t], NULL, q4_row_worker2, &job[t]) == 0)
            started[t] = 1;
        else
            (void)q4_row_worker2(&job[t]);
    }
    for (int t = 0; t < nth; t++)
        if (started[t])
            (void)salt_join_one_or_exit(
                th[t], pthread_join, "q4-matvec2", t);
    return 0;
}

/* ---- batched prefill matvec (M1 of prefill-batch) ---------------- */
/* Y[B][R] = W[R x C] * X, where X is ROW-MAJOR [B][C] (each token's
 * row contiguous -- the layout the per-token SIMD FMA loop needs).
 * The 4-bit row is decoded ONCE (dequant amortized over the B token
 * vectors), then each token runs the EXACT salt_simd_q4_matvec
 * accumulation topology: 8 vector accumulators with the (c>>2)&7
 * column map, same FMA chains, same reduction order -- so results
 * are BIT-IDENTICAL per (row, token) to the serial SIMD matvec, not
 * merely close. The recurrent delta-rule state amplifies any 1-ULP
 * difference into divergent generation; this preserves the map. */
static char q4_batch_worker_failed;

static void *q4_batch_worker(void *arg) {
    SaltBatchJob *j = (SaltBatchJob *)arg;
    const int C = j->C, B = j->B;
    /* decoded row scratch (per worker): C floats, grown on demand */
    SALT_THREAD_LOCAL float *trow = NULL;
    SALT_THREAD_LOCAL size_t trow_cap = 0;
    if ((size_t)C > trow_cap) {
        float *nb = (float *)realloc(trow, (size_t)C * sizeof(float));
        if (!nb) return &q4_batch_worker_failed;
        trow = nb; trow_cap = (size_t)C;
    }
    float *wrow = trow;
    for (int r = j->r0; r < j->r1; r++) {
        /* decode this row ONCE. Group index ABSOLUTE across R*C
         * ((r*C+c)/64), exactly like simd.c:158 -- per-row c/64
         * reads row 0's scales (the divergence bug). */
        const uint32_t *vr = j->vals + (size_t)r * (C / 8);
        int ng = (C + SALT_MLX4_GROUP - 1) / SALT_MLX4_GROUP;
        float srow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        float brow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        if (ng > SALT_SIMD_Q4_MAX_GROUPS_PER_ROW)
            return &q4_batch_worker_failed;
        for (int g = 0; g < ng; g++) {
            size_t absg = ((size_t)r * C + (size_t)g * SALT_MLX4_GROUP)
                          / SALT_MLX4_GROUP;
            uint32_t sb = (uint32_t)j->scales[absg] << 16;
            uint32_t bb = j->biases ? (uint32_t)j->biases[absg] << 16 : 0;
            memcpy(&srow[g], &sb, 4);
            memcpy(&brow[g], &bb, 4);
        }
#ifdef __aarch64__
        if (salt_kernels_simd()) {
            /* vector decode into wrow: same FMA per element as the
             * serial kernel's d0..d3 (bv + q*sv), stored once */
            int c = 0, w = 0;
            for (; c + 15 < C; c += 16, w += 2) {
                int g = c / SALT_MLX4_GROUP;
                float32x4_t sv = vdupq_n_f32(srow[g]);
                float32x4_t bv = vdupq_n_f32(brow[g]);
                uint32_t u0 = vr[w], u1 = vr[w + 1];
                uint8x8_t b0 = vreinterpret_u8_u32(vdup_n_u32(u0));
                uint8x8_t b1 = vreinterpret_u8_u32(vdup_n_u32(u1));
                uint8x8_t lo0 = vand_u8(b0, vdup_n_u8(0x0F));
                uint8x8_t hi0 = vand_u8(vshr_n_u8(b0, 4), vdup_n_u8(0x0F));
                uint8x8_t lo1 = vand_u8(b1, vdup_n_u8(0x0F));
                uint8x8_t hi1 = vand_u8(vshr_n_u8(b1, 4), vdup_n_u8(0x0F));
                uint8x8_t n0 = vzip_u8(lo0, hi0).val[0];
                uint8x8_t n1 = vzip_u8(lo1, hi1).val[0];
                uint32x4_t q00 = vmovl_u16(vget_low_u16(vmovl_u8(n0)));
                uint32x4_t q01 = vmovl_u16(vget_high_u16(vmovl_u8(n0)));
                uint32x4_t q10 = vmovl_u16(vget_low_u16(vmovl_u8(n1)));
                uint32x4_t q11 = vmovl_u16(vget_high_u16(vmovl_u8(n1)));
                vst1q_f32(wrow + c,     vmlaq_f32(bv, vcvtq_f32_u32(q00), sv));
                vst1q_f32(wrow + c + 4, vmlaq_f32(bv, vcvtq_f32_u32(q01), sv));
                vst1q_f32(wrow + c + 8, vmlaq_f32(bv, vcvtq_f32_u32(q10), sv));
                vst1q_f32(wrow + c + 12, vmlaq_f32(bv, vcvtq_f32_u32(q11), sv));
            }
            /* scalar tail decode (C % 16 != 0) */
            for (; c < C; c++) {
                long k = (size_t)r * C + c;
                int gl = (int)((c) / SALT_MLX4_GROUP);
                long wl = (k - (size_t)r * C) >> 3;
                int q = (int)((vr[wl] >> (4 * (k & 7))) & 0xFu);
                wrow[c] = (float)q * srow[gl] + brow[gl];
            }
            /* per-token FMA: the EXACT serial topology -- 8 vector
             * accumulators, (c>>2)&7 map, same reduction order */
            for (int t = 0; t < B; t++) {
                const float *xt = j->xs + (size_t)t * C;
                float32x4_t acc[8];
                for (int a = 0; a < 8; a++) acc[a] = vdupq_n_f32(0.0f);
                int c = 0;
                for (; c + 15 < C; c += 16) {
                    int ab = (c >> 2) & 7;   /* 0 or 4 for c%16==0 */
                    acc[ab + 0] = vmlaq_f32(acc[ab + 0],
                                            vld1q_f32(wrow + c),
                                            vld1q_f32(xt + c));
                    acc[ab + 1] = vmlaq_f32(acc[ab + 1],
                                            vld1q_f32(wrow + c + 4),
                                            vld1q_f32(xt + c + 4));
                    acc[ab + 2] = vmlaq_f32(acc[ab + 2],
                                            vld1q_f32(wrow + c + 8),
                                            vld1q_f32(xt + c + 8));
                    acc[ab + 3] = vmlaq_f32(acc[ab + 3],
                                            vld1q_f32(wrow + c + 12),
                                            vld1q_f32(xt + c + 12));
                }
                float32x4_t s4 = vdupq_n_f32(0.0f);
                for (int a = 0; a < 8; a++) s4 = vaddq_f32(s4, acc[a]);
                float32x2_t tt = vadd_f32(vget_low_f32(s4),
                                          vget_high_f32(s4));
                float s = vget_lane_f32(tt, 0) + vget_lane_f32(tt, 1);
                for (; c < C; c++) s += wrow[c] * xt[c];  /* tail */
                j->ys[(size_t)t * j->R + r] = s;
            }
        } else
#endif
        {
            /* scalar fallback: decode wrow scalar, c-order accumulate.
             * BIT-FIDELITY: this runs on x86 when R < nth (e.g. the
             * R=1 shared-expert gate); it must match the NEON 8-acc
             * (c>>2)&7 map + pairwise reduce exactly, otherwise the
             * cross-ISA KV diverges (measured: gate off by 4.5e-6). */
            for (int c = 0; c < C; c++) {
                long k = (size_t)r * C + c;
                int gl = (int)(k / SALT_MLX4_GROUP);
                long wl = (k - (size_t)r * C) >> 3;
                int q = (int)((vr[wl] >> (4 * (k & 7))) & 0xFu);
                wrow[c] = (float)q * srow[gl] + brow[gl];
            }
            for (int t = 0; t < B; t++) {
                const float *xt = j->xs + (size_t)t * C;
                /* 8 accumulators, (c>>2)&7 map, c ascending -- the
                 * EXACT SIMD topology (simd.c single-row kernel and
                 * the NEON batch worker). */
                float acc[8][4];
                memset(acc, 0, sizeof acc);
                int c = 0;
                for (; c + 15 < C; c += 16) {
                    int ab = (c >> 2) & 7;   /* 0 or 4 for c%16==0 */
                    for (int l = 0; l < 4; l++) {
                        float *ac = acc[ab + l];
                        const float *wr16 = wrow + c + 4 * l;
                        const float *xt16 = xt + c + 4 * l;
                        ac[0] += wr16[0] * xt16[0];
                        ac[1] += wr16[1] * xt16[1];
                        ac[2] += wr16[2] * xt16[2];
                        ac[3] += wr16[3] * xt16[3];
                    }
                }
                /* reduce: lane-wise sum over 8 accs, then
                 * (s4[0]+s4[2]) + (s4[1]+s4[3]) -- the SIMD tree */
                float s4[4] = {0, 0, 0, 0};
                for (int a = 0; a < 8; a++)
                    for (int l = 0; l < 4; l++) s4[l] += acc[a][l];
                float s = (s4[0] + s4[2]) + (s4[1] + s4[3]);
                for (; c < C; c++) s += wrow[c] * xt[c];  /* tail */
                j->ys[(size_t)t * j->R + r] = s;
            }
        }
    }
    return NULL;
}

/* A12: weight-stationary register-tiled batch GEMM (aarch64).
 * Same contract as q4_batch_worker, SAME per-token bit-fidelity:
 * the inner per-token loop keeps the exact 8-accumulator (c>>2)&7
 * map, c ascending, same reduction order. The change is only the
 * loop nesting: decode a 32-float weight block into 8 NEON regs
 * ONCE, then FMA it against every token's x block before loading
 * the next weight block. The decoded weight row was previously
 * re-read from L1/L2 once per token (B*C loads); now it stays in
 * registers (C loads per row). x traffic unchanged (B*C). */
#ifdef __aarch64__
/* ---- the multi-weight batch (the GQA q/k/v fusion) ---------------- */
/* N jobs with their OWN weights run as ONE thread-shared pass: the
 * weighted partition spans the jobs' concatenated rows, each worker's
 * range may cross the job boundaries. The per-row decode + the FMA
 * order are the q4_batch_worker's exactly (the kv-gate). */
typedef struct {
    SaltBatchJob *jobs;
    int njobs;
    long r0, r1;          /* the CONCAT row range */
} Mlx4MultiArg;

static void *q4_multi_worker(void *arg) {
    Mlx4MultiArg *ma = (Mlx4MultiArg *)arg;
    SaltBatchJob *jobs = ma->jobs;
    int njobs = ma->njobs;
    SALT_THREAD_LOCAL float *trow = NULL;
    SALT_THREAD_LOCAL size_t trow_cap = 0;
    /* the job cursor: the job holding concat-row r0 */
    int j = 0;
    long base = 0;
    while (j < njobs - 1 && base + jobs[j].R <= ma->r0) {
        base += jobs[j].R;
        j++;
    }
    for (long r = ma->r0; r < ma->r1; r++) {
        while (j < njobs - 1 && base + jobs[j].R <= r) {
            base += jobs[j].R;
            j++;
        }
        SaltBatchJob *jb = &jobs[j];
        const int C = jb->C, B = jb->B;
        if ((size_t)C > trow_cap) {
            float *nb = (float *)realloc(trow, (size_t)C * sizeof(float));
            if (!nb) return &q4_batch_worker_failed;
            trow = nb;
            trow_cap = (size_t)C;
        }
        float *wrow = trow;
        const int rr = (int)(r - base);
        const uint32_t *vr = jb->vals + (size_t)rr * (C / 8);
        int ng = (C + SALT_MLX4_GROUP - 1) / SALT_MLX4_GROUP;
        float srow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        float brow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        if (ng > SALT_SIMD_Q4_MAX_GROUPS_PER_ROW)
            return &q4_batch_worker_failed;
        for (int g = 0; g < ng; g++) {
            size_t absg = ((size_t)rr * C + (size_t)g * SALT_MLX4_GROUP)
                          / SALT_MLX4_GROUP;
            uint32_t sb = (uint32_t)jb->scales[absg] << 16;
            uint32_t bb = jb->biases ? (uint32_t)jb->biases[absg] << 16 : 0;
            memcpy(&srow[g], &sb, 4);
            memcpy(&brow[g], &bb, 4);
        }
#ifdef __aarch64__
        if (salt_kernels_simd()) {
            int c = 0, w = 0;
            for (; c + 15 < C; c += 16, w += 2) {
                int g = c / SALT_MLX4_GROUP;
                float32x4_t sv = vdupq_n_f32(srow[g]);
                float32x4_t bv = vdupq_n_f32(brow[g]);
                uint32_t u0 = vr[w], u1 = vr[w + 1];
                uint8x8_t b0 = vreinterpret_u8_u32(vdup_n_u32(u0));
                uint8x8_t b1 = vreinterpret_u8_u32(vdup_n_u32(u1));
                uint8x8_t lo0 = vand_u8(b0, vdup_n_u8(0x0F));
                uint8x8_t hi0 = vand_u8(vshr_n_u8(b0, 4), vdup_n_u8(0x0F));
                uint8x8_t lo1 = vand_u8(b1, vdup_n_u8(0x0F));
                uint8x8_t hi1 = vand_u8(vshr_n_u8(b1, 4), vdup_n_u8(0x0F));
                uint8x8_t n0 = vzip_u8(lo0, hi0).val[0];
                uint8x8_t n1 = vzip_u8(lo1, hi1).val[0];
                uint32x4_t q00 = vmovl_u16(vget_low_u16(vmovl_u8(n0)));
                uint32x4_t q01 = vmovl_u16(vget_high_u16(vmovl_u8(n0)));
                uint32x4_t q10 = vmovl_u16(vget_low_u16(vmovl_u8(n1)));
                uint32x4_t q11 = vmovl_u16(vget_high_u16(vmovl_u8(n1)));
                vst1q_f32(wrow + c,     vmlaq_f32(bv, vcvtq_f32_u32(q00), sv));
                vst1q_f32(wrow + c + 4, vmlaq_f32(bv, vcvtq_f32_u32(q01), sv));
                vst1q_f32(wrow + c + 8, vmlaq_f32(bv, vcvtq_f32_u32(q10), sv));
                vst1q_f32(wrow + c + 12, vmlaq_f32(bv, vcvtq_f32_u32(q11), sv));
            }
            for (; c < C; c++) {
                long k = (size_t)rr * C + c;
                int gl = (int)(c / SALT_MLX4_GROUP);
                long wl = (k - (size_t)rr * C) >> 3;
                int q = (int)((vr[wl] >> (4 * (k & 7))) & 0xFu);
                wrow[c] = (float)q * srow[gl] + brow[gl];
            }
            for (int t = 0; t < B; t++) {
                const float *xt = jb->xs + (size_t)t * C;
                float32x4_t acc[8];
                for (int a = 0; a < 8; a++) acc[a] = vdupq_n_f32(0.0f);
                int c = 0;
                for (; c + 15 < C; c += 16) {
                    int ab = (c >> 2) & 7;
                    acc[ab + 0] = vmlaq_f32(acc[ab + 0],
                                            vld1q_f32(wrow + c),
                                            vld1q_f32(xt + c));
                    acc[ab + 1] = vmlaq_f32(acc[ab + 1],
                                            vld1q_f32(wrow + c + 4),
                                            vld1q_f32(xt + c + 4));
                    acc[ab + 2] = vmlaq_f32(acc[ab + 2],
                                            vld1q_f32(wrow + c + 8),
                                            vld1q_f32(xt + c + 8));
                    acc[ab + 3] = vmlaq_f32(acc[ab + 3],
                                            vld1q_f32(wrow + c + 12),
                                            vld1q_f32(xt + c + 12));
                }
                float32x4_t s4 = vdupq_n_f32(0.0f);
                for (int a = 0; a < 8; a++) s4 = vaddq_f32(s4, acc[a]);
                float32x2_t tt = vadd_f32(vget_low_f32(s4),
                                          vget_high_f32(s4));
                float s = vget_lane_f32(tt, 0) + vget_lane_f32(tt, 1);
                for (; c < C; c++) s += wrow[c] * xt[c];
                jb->ys[(size_t)t * jb->R + rr] = s;
            }
        } else
#endif
        {
            for (int t = 0; t < B; t++) {
                const float *xt = jb->xs + (size_t)t * C;
                float s = 0.0f;
                for (int c = 0; c < C; c++) {
                    long k = (size_t)rr * C + c;
                    int gl = (int)(c / SALT_MLX4_GROUP);
                    long wl = (k - (size_t)rr * C) >> 3;
                    int q = (int)((vr[wl] >> (4 * (k & 7))) & 0xFu);
                    float wv = (float)q * srow[gl] + brow[gl];
                    s += wv * xt[c];
                }
                jb->ys[(size_t)t * jb->R + rr] = s;
            }
        }
    }
    return NULL;
}

int salt_q4_multi_batch(SaltBatchJob *jobs, int njobs) {
    /* CPU-only q/k/v fusion must not bypass mapped-only routing. */
    if (salt_gpu_mapped_only()) return -1;
    if (!jobs || njobs < 1 || !salt_kernels_simd()) return -1;
    long total = 0;
    for (int j = 0; j < njobs; j++) total += jobs[j].R;
    int nth = attn_threads();
    if (nth < 2 || total < nth) {
        for (int j = 0; j < njobs; j++)
            if (q4_batch_worker(&jobs[j]) != NULL) return -1;
        return 0;
    }
    long per = (total + nth - 1) / nth;
    pthread_t th[32];
    Mlx4MultiArg ma[32];
    int nspawn = 0;
    for (int t = 0; t < nth; t++) {
        long r0 = (long)t * per;
        long r1 = r0 + per;
        if (r1 > total) r1 = total;
        if (r0 >= r1) break;
        ma[t].jobs = jobs;
        ma[t].njobs = njobs;
        ma[t].r0 = r0;
        ma[t].r1 = r1;
        if (pthread_create(&th[t], NULL, q4_multi_worker, &ma[t]) != 0)
            break;
        nspawn++;
    }
    int failed = nspawn == 0 || ma[nspawn - 1].r1 < total;
    for (int t = 0; t < nspawn; t++) {
        void *worker_rc = salt_join_one_or_exit(
            th[t], pthread_join, "q4-multi-batch", t);
        if (worker_rc != NULL)
            failed = 1;
    }
    return failed ? -1 : 0;
}

static void *q4_batch_worker_tiled(void *arg) {
    SaltBatchJob *j = (SaltBatchJob *)arg;
    const int C = j->C, B = j->B;
    for (int r = j->r0; r < j->r1; r++) {
        const uint32_t *vr = j->vals + (size_t)r * (C / 8);
        int ng = (C + SALT_MLX4_GROUP - 1) / SALT_MLX4_GROUP;
        float srow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        float brow[SALT_SIMD_Q4_MAX_GROUPS_PER_ROW];
        if (ng > SALT_SIMD_Q4_MAX_GROUPS_PER_ROW)
            return &q4_batch_worker_failed;
        for (int g = 0; g < ng; g++) {
            size_t absg = ((size_t)r * C + (size_t)g * SALT_MLX4_GROUP)
                          / SALT_MLX4_GROUP;
            uint32_t sb = (uint32_t)j->scales[absg] << 16;
            uint32_t bb = j->biases ? (uint32_t)j->biases[absg] << 16 : 0;
            memcpy(&srow[g], &sb, 4);
            memcpy(&brow[g], &bb, 4);
        }
        /* Preserve one accumulator vector set per grouped TARGET row so
         * each decoded weight block serves the complete bounded N-row
         * expert group.  MT=64 covers the qualified TARGET ceiling and
         * bounds accumulator scratch at 8 KiB per worker; larger prefill
         * batches retain tiled traversal in 64-row chunks. */
        enum { MT = 64, WB = 32 };  /* grouped rows, weight block */
        int cb = 0;
        /* token micro-tile OUTER: accs [MT][8] persist across ALL cb
         * blocks (each token's full C-dot accumulates in registers,
         * reduced once at the end -- the previous version reset them
         * per block, which was WRONG and stored C/32 partials). */
        for (int t0 = 0; t0 < B; t0 += MT) {
            int nt = (B - t0 < MT) ? (B - t0) : MT;
            float32x4_t a[MT][8];
            for (int t = 0; t < nt; t++)
                for (int a8 = 0; a8 < 8; a8++)
                    a[t][a8] = vdupq_n_f32(0.0f);
            for (cb = 0; cb + (WB - 1) < C; cb += WB) {
                /* srow[] holds this row's groups pre-resolved to
                 * ABSOLUTE scale offsets; group for column cb is
                 * cb/64 (32-block never straddles 64). */
                float32x4_t sv = vdupq_n_f32(srow[cb / SALT_MLX4_GROUP]);
                float32x4_t bv = vdupq_n_f32(brow[cb / SALT_MLX4_GROUP]);
                /* decode WB floats into 8 NEON regs, ONCE per block */
                float32x4_t w0, w1, w2, w3, w4, w5, w6, w7;
                uint32_t u0 = vr[cb / 8 + 0], u1 = vr[cb / 8 + 1];
                uint32_t u2 = vr[cb / 8 + 2], u3 = vr[cb / 8 + 3];
                uint8x8_t b0 = vreinterpret_u8_u32(vdup_n_u32(u0));
                uint8x8_t b1 = vreinterpret_u8_u32(vdup_n_u32(u1));
                uint8x8_t b2 = vreinterpret_u8_u32(vdup_n_u32(u2));
                uint8x8_t b3 = vreinterpret_u8_u32(vdup_n_u32(u3));
                uint8x8_t lo0 = vand_u8(b0, vdup_n_u8(0x0F));
                uint8x8_t hi0 = vand_u8(vshr_n_u8(b0, 4), vdup_n_u8(0x0F));
                uint8x8_t lo1 = vand_u8(b1, vdup_n_u8(0x0F));
                uint8x8_t hi1 = vand_u8(vshr_n_u8(b1, 4), vdup_n_u8(0x0F));
                uint8x8_t lo2 = vand_u8(b2, vdup_n_u8(0x0F));
                uint8x8_t hi2 = vand_u8(vshr_n_u8(b2, 4), vdup_n_u8(0x0F));
                uint8x8_t lo3 = vand_u8(b3, vdup_n_u8(0x0F));
                uint8x8_t hi3 = vand_u8(vshr_n_u8(b3, 4), vdup_n_u8(0x0F));
                uint8x8_t n0 = vzip_u8(lo0, hi0).val[0];
                uint8x8_t n1 = vzip_u8(lo1, hi1).val[0];
                uint8x8_t n2 = vzip_u8(lo2, hi2).val[0];
                uint8x8_t n3 = vzip_u8(lo3, hi3).val[0];
                uint32x4_t q00 = vmovl_u16(vget_low_u16(vmovl_u8(n0)));
                uint32x4_t q01 = vmovl_u16(vget_high_u16(vmovl_u8(n0)));
                uint32x4_t q10 = vmovl_u16(vget_low_u16(vmovl_u8(n1)));
                uint32x4_t q11 = vmovl_u16(vget_high_u16(vmovl_u8(n1)));
                uint32x4_t q20 = vmovl_u16(vget_low_u16(vmovl_u8(n2)));
                uint32x4_t q21 = vmovl_u16(vget_high_u16(vmovl_u8(n2)));
                uint32x4_t q30 = vmovl_u16(vget_low_u16(vmovl_u8(n3)));
                uint32x4_t q31 = vmovl_u16(vget_high_u16(vmovl_u8(n3)));
                w0 = vmlaq_f32(bv, vcvtq_f32_u32(q00), sv);
                w1 = vmlaq_f32(bv, vcvtq_f32_u32(q01), sv);
                w2 = vmlaq_f32(bv, vcvtq_f32_u32(q10), sv);
                w3 = vmlaq_f32(bv, vcvtq_f32_u32(q11), sv);
                w4 = vmlaq_f32(bv, vcvtq_f32_u32(q20), sv);
                w5 = vmlaq_f32(bv, vcvtq_f32_u32(q21), sv);
                w6 = vmlaq_f32(bv, vcvtq_f32_u32(q30), sv);
                w7 = vmlaq_f32(bv, vcvtq_f32_u32(q31), sv);
                for (int t = 0; t < nt; t++) {
                    const float *xt = j->xs + (size_t)(t0 + t) * C + cb;
                    int ab = (cb >> 2) & 7;   /* 0 or 4 for cb%16==0 */
                    a[t][ab + 0] = vmlaq_f32(a[t][ab + 0], w0,
                                             vld1q_f32(xt + 0));
                    a[t][ab + 1] = vmlaq_f32(a[t][ab + 1], w1,
                                             vld1q_f32(xt + 4));
                    a[t][ab + 2] = vmlaq_f32(a[t][ab + 2], w2,
                                             vld1q_f32(xt + 8));
                    a[t][ab + 3] = vmlaq_f32(a[t][ab + 3], w3,
                                             vld1q_f32(xt + 12));
                    a[t][ab + 4] = vmlaq_f32(a[t][ab + 4], w4,
                                             vld1q_f32(xt + 16));
                    a[t][ab + 5] = vmlaq_f32(a[t][ab + 5], w5,
                                             vld1q_f32(xt + 20));
                    a[t][ab + 6] = vmlaq_f32(a[t][ab + 6], w6,
                                             vld1q_f32(xt + 24));
                    a[t][ab + 7] = vmlaq_f32(a[t][ab + 7], w7,
                                             vld1q_f32(xt + 28));
                }
            }
            for (int t = 0; t < nt; t++) {
                float32x4_t s4 = vdupq_n_f32(0.0f);
                for (int a8 = 0; a8 < 8; a8++)
                    s4 = vaddq_f32(s4, a[t][a8]);
                float32x2_t tt = vadd_f32(vget_low_f32(s4),
                                          vget_high_f32(s4));
                float s = vget_lane_f32(tt, 0) + vget_lane_f32(tt, 1);
                j->ys[(size_t)(t0 + t) * j->R + r] = s;
            }
        }
        /* scalar tail for C % 32 != 0: decode wrow + exact order */
        if (cb < C) {
            float wrow[64];
            int c = cb;
            for (; c < C; c++) {
                long k = (size_t)r * C + c;
                int gl = (int)(k / SALT_MLX4_GROUP);
                long wl = (k - (size_t)r * C) >> 3;
                int q = (int)((vr[wl] >> (4 * (k & 7))) & 0xFu);
                wrow[c - cb] = (float)q * srow[gl] + brow[gl];
            }
            for (int t = 0; t < B; t++) {
                const float *xt = j->xs + (size_t)t * C;
                float s = 0.0f;
                for (int cc = cb; cc < C; cc++)
                    s += wrow[cc - cb] * xt[cc];
                j->ys[(size_t)t * j->R + r] = s;
            }
        }
    }
    return NULL;
}
#endif

/* A12: tiled-batch GEMM (weight-stationary register tiles). Proven
 * bit-identical at every shape incl. the C=200 tail case (probe
 * 2026-08-09: 0 diffs, 1.6-1.8x at real shapes). Enabled by default on
 * aarch64; SALT_A12=0 disables (the probe's A/B hook lives in
 * bench-kernels). */
static int g_tiled_force = -1;   /* -1 = env/default, 0/1 = forced */
void salt_kernels_set_tiled(int on) { g_tiled_force = on; }
#ifdef __aarch64__
static int a12_tiled_ok(void) {
    if (g_tiled_force >= 0) return g_tiled_force;
    static int a12 = -1;
    if (a12 < 0) {
        const char *e = getenv("SALT_A12");
        /* The tiled worker's reduce tree is bit-identical to the
         * serial fold on arm64 (probe 2026-08-09: 0 diffs at every
         * shape incl. C=200 tail; KV md5 f06b0040... matches at the
         * real B=512 shape). DEFAULT ON for aarch64: ~28% on the
         * batched projections. SALT_A12=0 disables (restores the
         * non-tiled worker; only needed to compare against x86,
         * whose AVX2 fold runs the non-tiled math). */
        int on = 1;
        if (e && *e == '0') on = 0;
        a12 = on;
    }
    return a12;
}
#endif

/* x86-64 batch worker: no NEON worker here, so partition rows across
 * threads and call the portable AVX2 single-row kernel per token.
 * Same row math as salt_q4_matvec (bit-identical rows), just
 * batched and row-partitioned like the aarch64 path. */
#ifndef __aarch64__
static void *q4_batch_worker_x86(void *arg) {
    SaltBatchJob *j = (SaltBatchJob *)arg;
    for (int b = 0; b < j->B; b++)
        salt_simd_q4_matvec(j->vals, j->scales, j->biases,
                              j->R, j->C,
                              j->xs + (size_t)b * j->C,
                              j->ys + (size_t)b * j->R,
                              j->r0, j->r1);
    return NULL;
}
#endif

int salt_q4_matvec_batch_rows(const uint32_t *vals, const uint16_t *scales,
                              const uint16_t *biases, int R, int C, int B,
                              const float *xs, float *ys, int r0, int r1) {
    SaltBatchJob job;
    if (!vals || !scales || !xs || !ys || R < 1 || C < 1 || B < 1 ||
        (C % 8) != 0 || r0 < 0 || r0 > r1 || r1 > R ||
        C / SALT_MLX4_GROUP > SALT_SIMD_Q4_MAX_GROUPS_PER_ROW)
        return -1;
    if (r0 == r1) return 0;
    memset(&job, 0, sizeof job);
    job.vals = vals;
    job.scales = scales;
    job.biases = biases;
    job.R = R;
    job.C = C;
    job.B = B;
    job.xs = xs;
    job.ys = ys;
    job.r0 = r0;
    job.r1 = r1;
#ifdef __aarch64__
    if (a12_tiled_ok() && (C % 32) == 0)
        return q4_batch_worker_tiled(&job) != NULL ? -1 : 0;
    return q4_batch_worker(&job) != NULL ? -1 : 0;
#else
    return q4_batch_worker_x86(&job) != NULL ? -1 : 0;
#endif
}

int salt_q4_matvec_batch_tile(
        const uint32_t *vals, const uint16_t *scales,
        const uint16_t *biases, int R, int C, int B,
        const float *xs, float *ys,
        int b0, int b1, int r0, int r1) {
    if (!vals || !scales || !xs || !ys || R < 1 || C < 1 || B < 1 ||
        b0 < 0 || b0 > b1 || b1 > B || r0 < 0 || r0 > r1 || r1 > R)
        return -1;
    if (b0 == b1 || r0 == r1) return 0;
    return salt_q4_matvec_batch_rows(
        vals, scales, biases, R, C, b1 - b0,
        xs + (size_t)b0 * C, ys + (size_t)b0 * R,
        r0, r1);
}

int salt_q4_multi_batch_rows(SaltBatchJob *jobs, int njobs,
                             int64_t concat_r0, int64_t concat_r1) {
    int64_t total = 0;
    if (!jobs || njobs < 1 || concat_r0 < 0 || concat_r0 > concat_r1)
        return -1;
    for (int job = 0; job < njobs; job++) {
        if (!jobs[job].vals || !jobs[job].scales || !jobs[job].xs ||
            !jobs[job].ys || jobs[job].R < 1 || jobs[job].C < 1 ||
            jobs[job].B < 1 || jobs[job].R > INT64_MAX - total)
            return -1;
        total += jobs[job].R;
    }
    if (concat_r1 > total) return -1;
    if (concat_r0 == concat_r1) return 0;
    int64_t base = 0;
    for (int job = 0; job < njobs; job++) {
        int64_t lo = concat_r0 > base ? concat_r0 - base : 0;
        int64_t hi = concat_r1 < base + jobs[job].R
            ? concat_r1 - base : jobs[job].R;
        if (lo < hi &&
            salt_q4_matvec_batch_rows(
                jobs[job].vals, jobs[job].scales, jobs[job].biases,
                jobs[job].R, jobs[job].C, jobs[job].B,
                jobs[job].xs, jobs[job].ys, (int)lo, (int)hi) != 0)
            return -1;
        base += jobs[job].R;
        if (base >= concat_r1) break;
    }
    return 0;
}

#ifndef __aarch64__
int salt_q4_multi_batch(SaltBatchJob *jobs, int njobs) {
    int64_t total = 0;
    /* Match the AArch64 admission contract. The portable row-range helper
     * below owns exact per-job output placement and x86 row arithmetic. */
    if (salt_gpu_mapped_only()) return -1;
    if (!jobs || njobs < 1 || !salt_kernels_simd()) return -1;
    for (int job = 0; job < njobs; job++) {
        if (jobs[job].R < 1 || jobs[job].R > INT64_MAX - total)
            return -1;
        total += jobs[job].R;
    }
    return salt_q4_multi_batch_rows(jobs, njobs, 0, total);
}
#endif

int salt_q4_matvec_batch(const uint32_t *vals, const uint16_t *scales,
                           const uint16_t *biases, int R, int C, int B,
                           const float *xs, float *ys) {
    /* Expert-worker batches remain owned by moe.c's fail-closed GPU-MoE
     * route. A caller-thread grouped CPU batch instead carries an explicit
     * resource scope: it bypasses trunk GPU routing while retaining the
     * ordinary CPU batch implementation. */
    if (salt_kernels_in_expert()) return -1;
    if (salt_kernels_cpu_expert_scope())
        goto gpu_trunk_done;
    /* main-pipeline GPU routing (SALT_GPU_TRUNK=1): the chunk
     * prefill's big projections (the qkv/z/o, R >= 512, B up to the
     * chunk size) go through the Metal proj-batch -- one weight
     * uploaded once, the kernel re-reads it per job. The tiny a/b
     * gates and the q8 tensors stay CPU; -1 falls through to the
     * CPU batch below.
     *
     * The batch size is discrete AND per-layer: SALT_GPU_PREFILL_B
     * is the default (1024; the literal "MAX" = the whole prompt),
     * and the per-shape knobs (SALT_GPU_B_QKV/Z/O) select the B by
     * the layer's projection shape -- the parallel steps tuned to
     * the layer, not a global constant. */
    const char *gbe = getenv("SALT_GPU_PREFILL_B");
    int gB;
    if (gbe && !strcmp(gbe, "MAX")) gB = 1 << 20;   /* the whole prompt */
    else gB = gbe ? atoi(gbe) : 1024;
    {
        const char *bq = getenv("SALT_GPU_B_QKV");
        const char *bz = getenv("SALT_GPU_B_Z");
        const char *bo = getenv("SALT_GPU_B_O");
        if (bq && R >= 8192) gB = atoi(bq);
        else if (bz && R >= 4096) gB = atoi(bz);
        else if (bo && R >= 2048) gB = atoi(bo);
    }
    const char *gpu_trunk = getenv("SALT_GPU_TRUNK");
    int gpu_trunk_on = gpu_trunk && *gpu_trunk != '0';
    int mapped_required = gpu_trunk_on && salt_gpu_mapped_only();
    if (gpu_trunk_on && R >= 512 && B >= 4 && B <= gB &&
        gB <= (1 << 20)) {
        /* The engine fallback threshold is 1024. Model blueprints may
         * override SALT_GPU_TRUNK_B after exactness and crossover
         * qualification; Qwen3.6-35B-A3B's prefill-only path uses 200.
         * Decode remains CPU and SALT_GPU itself remains opt-in. */
        int tB = 1024;
        {
            const char *tb = getenv("SALT_GPU_TRUNK_B");
            if (tb && atoi(tb) > 0) tB = atoi(tb);
        }
        if (B < tB) {
            if (mapped_required) return -1;
            goto gpu_trunk_done;
        }
        const float **xj = (const float **)malloc((size_t)B * sizeof(float *));
        float **yj = (float **)malloc((size_t)B * sizeof(float *));
        if (!xj || !yj) { free(xj); free(yj); return -1; }
        for (int b = 0; b < B; b++) {
            xj[b] = xs + (size_t)b * C;
            yj[b] = ys + (size_t)b * R;
        }
        int rc = salt_gpu_proj_batch(vals, scales, biases, R, C, B,
                                     xj, yj, (const void *)vals);
        free(xj);
        free(yj);
        if (rc == 0)
            return 0;
        if (mapped_required) return -1;
    }
    if (mapped_required && R >= 512) return -1;
    gpu_trunk_done:
    if (!salt_kernels_simd() || (C % 8) != 0 || R < 1 || B < 1)
        return -1;
    int nth = attn_threads();   /* cached at first dispatch */
#ifdef __aarch64__
    int use_tiled = a12_tiled_ok() && (C % 32) == 0;
#endif
    if (nth < 2 || R < nth) {
        SaltBatchJob j;
        j.vals = vals; j.scales = scales; j.biases = biases;
        j.R = R; j.C = C; j.B = B; j.xs = xs; j.ys = ys;
        j.r0 = 0; j.r1 = R;
#ifdef __aarch64__
        if (use_tiled)
            return q4_batch_worker_tiled(&j) != NULL ? -1 : 0;
        else
#endif
        return q4_batch_worker(&j) != NULL ? -1 : 0;
    }
    pthread_t th[32];
    SaltBatchJob job[32];
    int nspawn = 0;
#ifdef __aarch64__
    /* weighted row partition: the equal-COMPUTE split, not the
     * equal-rows split. The rows in one batch share C, so both are
     * identical here -- but the weighted form (per-row weight = C)
     * stays exact when the per-layer dispatch mixes shapes, and it
     * guarantees the max per-thread MAC imbalance is < 1 row. */
    {
        long total = (long)R * (long)C;
        long per = (total + nth - 1) / nth;
        long acc = 0;
        for (int t = 0; t < nth; t++) {
            long want = (long)(t + 1) * per;
            if (want > total) want = total;
            int r0 = (int)(acc / (long)C);
            int r1 = (int)((want + (long)C - 1) / (long)C);
            if (r1 > R) r1 = R;
            if (r0 >= R) continue;
            job[t].vals = vals; job[t].scales = scales; job[t].biases = biases;
            job[t].R = R; job[t].C = C; job[t].B = B;
            job[t].xs = xs; job[t].ys = ys;
            job[t].r0 = r0; job[t].r1 = r1;
            if (pthread_create(&th[t], NULL,
                               use_tiled ? q4_batch_worker_tiled
                                         : q4_batch_worker, &job[t]) != 0)
                break;
            nspawn++;
            acc = want;
        }
    }
#else
    int chunk = (R + nth - 1) / nth;
    for (int t = 0; t < nth; t++) {
        int r0 = t * chunk;
        int r1 = (t + 1) * chunk < R ? (t + 1) * chunk : R;
        if (r0 >= R) continue;
        job[t].vals = vals; job[t].scales = scales; job[t].biases = biases;
        job[t].R = R; job[t].C = C; job[t].B = B;
        job[t].xs = xs; job[t].ys = ys;
        job[t].r0 = r0; job[t].r1 = r1;
        if (pthread_create(&th[t], NULL, q4_batch_worker_x86, &job[t]) != 0)
            break;
        nspawn++;
    }
#endif
    int failed = nspawn == 0 || job[nspawn - 1].r1 < R;
    for (int t = 0; t < nspawn; t++) {
        void *worker_rc = salt_join_one_or_exit(
            th[t], pthread_join, "q4-batch", t);
        if (worker_rc != NULL)
            failed = 1;
    }
    return failed ? -1 : 0;
}

static int q4_mixed_cpu_range(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C,
    int first_job, int job_count, const float *xs, float *ys) {
    int prior_scope, rc;
    if (job_count < 1) return 0;
    prior_scope = salt_kernels_cpu_expert_scope();
    salt_kernels_set_cpu_expert_scope(1);
    rc = salt_q4_matvec_batch(
        vals, scales, biases, R, C, job_count,
        xs + (size_t)first_job * C,
        ys + (size_t)first_job * R);
    salt_kernels_set_cpu_expert_scope(prior_scope);
    return rc;
}

int salt_gpu_mixed_proj_batch(
    const uint32_t *vals, const uint16_t *scales,
    const uint16_t *biases, int R, int C, int B,
    int gpu_first_job, int gpu_jobs, const float *xs,
    SaltGpuSharedBuffer *output, size_t y_float_offset,
    const void *id) {
    uint64_t output_span;
    size_t output_end;
    int cpu_first_job, cpu_jobs;
    float *ys;
    const char *test_order;
    if (!vals || !scales || !xs || !output || !output->contents ||
        !output->backend || !id || R < 1 || C < 1 || B < 2 ||
        gpu_jobs < 1 || gpu_jobs >= B || gpu_first_job < 0 ||
        gpu_first_job > B - gpu_jobs ||
        (gpu_first_job != 0 && gpu_first_job + gpu_jobs != B))
        return -1;
    output_span = (uint64_t)(uint32_t)B * (uint32_t)R;
    if (output_span > UINT32_MAX ||
        (uint64_t)y_float_offset > (uint64_t)UINT32_MAX - output_span ||
        output_span > (uint64_t)SIZE_MAX ||
        y_float_offset > SIZE_MAX - (size_t)output_span)
        return -1;
    output_end = y_float_offset + (size_t)output_span;
    if (output_end > output->nbytes / sizeof(float)) return -1;
    ys = (float *)output->contents + y_float_offset;
    if (gpu_first_job == 0) {
        cpu_first_job = gpu_jobs;
        cpu_jobs = B - gpu_jobs;
    } else {
        cpu_first_job = 0;
        cpu_jobs = gpu_first_job;
    }
    test_order = getenv("SALT_GPU_MIX_TEST_ORDER");

    /* Deterministic ordering probes may force either executor to complete
     * first. Production leaves the variable unset: Metal submits
     * asynchronously, CPU computes the disjoint complement concurrently,
     * and the caller fences only at the actual consumer. */
    if (test_order && !strcmp(test_order, "cpu-first")) {
        if (q4_mixed_cpu_range(vals, scales, biases, R, C,
                               cpu_first_job, cpu_jobs, xs, ys) != 0)
            return -1;
        if (salt_gpu_proj_batch_indexed(
                vals, scales, biases, R, C, B,
                gpu_first_job, gpu_jobs, xs, output, y_float_offset,
                id) != 0)
            return -1;
    } else {
        if (salt_gpu_proj_batch_indexed(
                vals, scales, biases, R, C, B,
                gpu_first_job, gpu_jobs, xs, output, y_float_offset,
                id) != 0)
            return -1;
        if (test_order && !strcmp(test_order, "gpu-first") &&
            salt_gpu_sync() != 0)
            return -1;
        if (q4_mixed_cpu_range(vals, scales, biases, R, C,
                               cpu_first_job, cpu_jobs, xs, ys) != 0) {
            (void)salt_gpu_sync();
            return -1;
        }
    }
    if (getenv("SALT_GPU_MIX_DIAG"))
        fprintf(stderr,
                "[gpu-mix] B=%d GPU=[%d,%d) CPU=[%d,%d) dst=%zu direct=1\n",
                B, gpu_first_job, gpu_first_job + gpu_jobs,
                cpu_first_job, cpu_first_job + cpu_jobs, y_float_offset);
    return 0;
}
