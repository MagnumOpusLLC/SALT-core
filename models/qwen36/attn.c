/*
 * attn_qwen.c -- Qwen3.5-35B-A3B attention: full-GQA layers + linear
 * (Mamba2-class) layers, MLX 4-bit projections.
 *
 * Full-GQA layer (10 of 40, full_attention_interval=4):
 *   q = q_proj(x)   [16 heads x 512]  (512 = 256 + 256 gate, the
 *                                      attn_output_gate convention)
 *   k = k_proj(x)   [2 kv-heads x 256]
 *   v = v_proj(x)   [2 kv-heads x 256]
 *   q = RMSNorm(q, q_norm), k = RMSNorm(k, k_norm)
 *   attn = softmax(q k^T / sqrt(d)) v   (GQA: 16 q-heads share 2 kv)
 *   out = o_proj(attn)                  (2048)
 *
 * Linear-attention layer (30 of 40): Mamba2-class -- conv1d, A_log,
 * dt_bias, in_proj_qkv/z/a/b, out_proj. The recurrent state machinery
 * is the next block; this module currently SKIPS linear layers
 * gracefully (the MLP still runs, so the forward is incomplete but
 * deterministic).
 *
 * All projections are MLX 4-bit (U32 nibbles + BF16 scale/bias per
 * 64-group); the trunk layout carries the role-matched triplets.
 */
#include "attn.h"
#include "salt/quant.h"
#include "salt/kernels.h"
#include "salt/moe.h"
#include "salt/bitmath.h"
#include "deltachunk.h"
#include "salt/gpu.h"
#include "salt/model.h"
#include "salt/simd.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#include <pthread.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    const uint32_t *v1, *v2;
    const uint16_t *s1, *b1, *s2, *b2;
    const float *x;
    float *y1, *y2;
    int R1, R2, C, nthreads;
} Q4Matvec2PoolCtx;

static void q4_matvec2_pool_worker(int tid, void *opaque) {
    Q4Matvec2PoolCtx *ctx = (Q4Matvec2PoolCtx *)opaque;
    int64_t total = (int64_t)ctx->R1 + (int64_t)ctx->R2;
    int64_t chunk = (total + (int64_t)ctx->nthreads - 1) /
                    (int64_t)ctx->nthreads;
    int64_t lo = (int64_t)tid * chunk;
    int64_t hi = lo + chunk;
    if (hi > total) hi = total;
    int r0 = (int)lo;
    int r1 = (int)hi;
    if (r0 >= r1) return;
    if (r0 < ctx->R1) {
        int a1 = r1 < ctx->R1 ? r1 : ctx->R1;
        salt_simd_q4_matvec(ctx->v1, ctx->s1, ctx->b1,
                            ctx->R1, ctx->C, ctx->x, ctx->y1, r0, a1);
    }
    if (r1 > ctx->R1) {
        int b0 = r0 > ctx->R1 ? r0 - ctx->R1 : 0;
        int b1 = r1 - ctx->R1;
        salt_simd_q4_matvec(ctx->v2, ctx->s2, ctx->b2,
                            ctx->R2, ctx->C, ctx->x, ctx->y2, b0, b1);
    }
}

int salt_attn_q4_matvec2(SaltKvCache *kv,
                         const uint32_t *v1, const uint16_t *s1,
                         const uint16_t *b1, int R1,
                         const uint32_t *v2, const uint16_t *s2,
                         const uint16_t *b2, int R2, int C,
                         const float *x, float *y1, float *y2) {
    if (!kv || !kv->aq4_pool_enabled || !kv->apool_sync_init || !kv->ath ||
        kv->aq4_threads < 1 || kv->aq4_threads > kv->apool_threads ||
        kv->ath_count != kv->apool_threads ||
        !salt_kernels_simd() || salt_kernels_in_expert() ||
        salt_gpu_mapped_only() || !v1 || !s1 || !v2 || !s2 ||
        !x || !y1 || !y2 || R1 <= 0 || R2 <= 0 ||
        (int64_t)R1 + (int64_t)R2 >
            (int64_t)INT_MAX - (kv->aq4_threads - 1) ||
        C <= 0 || (C % 8) != 0 ||
        (C % SALT_MLX4_GROUP) != 0)
        return -1;
    Q4Matvec2PoolCtx ctx = {
        .v1 = v1, .v2 = v2, .s1 = s1, .b1 = b1, .s2 = s2, .b2 = b2,
        .x = x, .y1 = y1, .y2 = y2, .R1 = R1, .R2 = R2, .C = C,
        .nthreads = kv->aq4_threads
    };
    return salt_attn_pool_run_n(kv, kv->aq4_threads,
                                q4_matvec2_pool_worker, &ctx);
}

static int q4_matvec2_decode(SaltKvCache *kv,
                             const uint32_t *v1, const uint16_t *s1,
                             const uint16_t *b1, int R1,
                             const uint32_t *v2, const uint16_t *s2,
                             const uint16_t *b2, int R2, int C,
                             const float *x, float *y1, float *y2) {
    if (kv && kv->aq4_pool_enabled &&
        salt_attn_q4_matvec2(kv, v1, s1, b1, R1, v2, s2, b2, R2, C,
                             x, y1, y2) == 0)
        return 0;
    return salt_q4_matvec2(v1, s1, b1, R1, v2, s2, b2, R2, C,
                           x, y1, y2);
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

/* Decoded (logical) column count for a trunk tensor: the layout's
 * dims[1] is the PACKED U32 count, and each U32 holds 32/bits
 * elements (4-bit -> 8, 8-bit -> 4). The 3.6 repo quantizes router
 * gates at 8-bit while the dense projections stay 4-bit, so the
 * packing ratio MUST follow the per-tensor bits, never a constant. */
static int proj_shape_ok(const SaltTrunkLayout *tl, int wi, int si,
                         int bi, int R, int C) {
    if (!tl || !tl->t || !tl->t_off || tl->n_layers < 1 || R < 1 || C < 1)
        return 0;
    int nt = tl->t_off[tl->n_layers];
    if (wi < 0 || wi >= nt || tl->t[wi].rank != 2 ||
        tl->t[wi].dims[0] != R || salt_trunk_tensor_cols(&tl->t[wi]) != C)
        return 0;
    if (tl->t[wi].dtype != 4 && (si < 0 || si >= nt)) return 0;
    if (bi >= nt) return 0;
    return 1;
}


/* mlx matvec for a trunk tensor triplet (weight/scales/biases roles),
 * dispatching on the tensor's OWN bit width (4-bit SIMD path or the
 * 8-bit scalar LUT path -- the 3.6 router gates). Returns 0 on
 * success, -1 if a required role is missing. */
/* the current batch B -- the gen's 1, the chunked prefill's chunk. */
static int _cur_batch_b = 1;
void salt_attn_set_batch_b(int b) { _cur_batch_b = b > 0 ? b : 1; }
int salt_attn_batch_b(void) { return _cur_batch_b; }

static int gpu_address_audit_enabled(void) {
    const char *p = getenv("SALT_GPU_ADDRESS_AUDIT");
    return p && *p && *p != '0';
}

static void gpu_address_audit_index(const SaltTrunkLayout *tl, int wi,
                                    int si, int bi, const uint8_t *tr,
                                    int R, int C, int B) {
    static int reported = 0;
    int layer = -1;
    if (reported || !gpu_address_audit_enabled() || !salt_gpu_mapped_only() ||
        !tl || !tl->t || !tl->t_off || !tr || wi < 0)
        return;
    for (int L = 0; L < tl->n_layers; L++) {
        if (wi >= tl->t_off[L] && wi < tl->t_off[L + 1]) {
            layer = L;
            break;
        }
    }
    fprintf(stderr,
            "[gpu-address] index sample=1 layer=%d tensor_index=%d "
            "tensor=%.95s rows=%d cols=%d batch=%d "
            "layer_rel_v=%ld layer_rel_s=%ld layer_rel_b=%ld "
            "key=%p host_v=%p host_s=%p host_b=%p\n",
            layer, wi, tl->t[wi].name, R, C, B,
            tl->t[wi].off, si >= 0 ? tl->t[si].off : -1L,
            bi >= 0 ? tl->t[bi].off : -1L,
            (const void *)(tr + tl->t[wi].off),
            (const void *)(tr + tl->t[wi].off),
            si >= 0 ? (const void *)(tr + tl->t[si].off) : NULL,
            bi >= 0 ? (const void *)(tr + tl->t[bi].off) : NULL);
    reported = 1;
}

static int q4_proj(const SaltTrunkLayout *tl, int wi, int si, int bi,
                     const uint8_t *tr, int R, int C, const float *x,
                     float *y) {
    if (!tr || !x || !y || !proj_shape_ok(tl, wi, si, bi, R, C)) return -1;
    if (getenv("SALT_NAN_PROBE") && wi == 12) {
        fprintf(stderr, "[sq8p] wi=%d si=%d dtype=%d bits=%d R=%d C=%d\n",
                wi, si, tl->t[wi].dtype, tl->t[wi].bits, R, C);
    }
    const uint16_t *bias = bi >= 0
        ? (const uint16_t *)(const void *)(tr + tl->t[bi].off) : NULL;
    if (tl->t[wi].dtype == 2) {
        /* F8_E4M3 trunk tensor (official Qwen FP8): 1 B/val, per-tensor
         * weight_scale_inv -- BF16 [SR,SC] block scales (NOT E8M0: the
         * E8M0 encode loses the mantissa -> flat logits). */
        if (si < 0) return -1;
        int SR = tl->t[si].rank > 0 ? (int)tl->t[si].dims[0] : 4;
        int SC = tl->t[si].rank > 1 ? (int)tl->t[si].dims[1] : 16;
        if (getenv("SALT_NAN_PROBE")) {
            static int dbg_wi = -1;
            if (dbg_wi != wi) {
                dbg_wi = wi;
                const uint16_t *s0 = (const uint16_t *)(const void *)
                                     (tr + tl->t[si].off);
                double xr2 = 0;
                for (int i = 0; i < C; i++)
                    xr2 += (double)x[i] * x[i];
                fprintf(stderr,
                        "[f8proj] wi=%d si=%d R=%d C=%d SR=%d SC=%d "
                        "woff=%ld soff=%ld s0=%u x-rms=%.4g "
                        "w0=%u w1=%u w2=%u w3=%u\n",
                        wi, si, R, C, SR, SC, tl->t[wi].off,
                        tl->t[si].off, (unsigned)s0[0],
                        sqrt(xr2 / C), (unsigned)((const uint8_t *)
                        (const void *)(tr + tl->t[wi].off))[0],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[1],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[2],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[3]);
            }
        }
        salt_f8_matvec_bf16(
            (const uint8_t *)(const void *)(tr + tl->t[wi].off),
            (const uint16_t *)(const void *)(tr + tl->t[si].off),
            R, C, SR, SC, x, y);
    } else if (tl->t[wi].dtype == 4) {
        /* BF16 trunk tensor (official FP8 keeps in_proj_a/b and the
         * norms BF16): 2 B/val, no scale. */
        const SaltQuant *qm = &salt_quant_bf16;
        qm->matvec((const void *)(tr + tl->t[wi].off), NULL, bias,
                   R, C, x, y);
    } else if (tl->t[wi].bits == 8) {
        if (getenv("SALT_NAN_PROBE") && wi == 12) {
            const uint8_t *w0p = (const uint8_t *)(const void *)
                (tr + tl->t[wi].off);
            const uint16_t *s0p = si >= 0
                ? (const uint16_t *)(const void *)(tr + tl->t[si].off)
                : NULL;
            fprintf(stderr, "[sq8] wi=%d si=%d R=%d C=%d woff=%zu soff=%zu "
                    "w[0..7]=%u %u %u %u %u %u %u %u s0=%.6g\n",
                    wi, si, R, C, (size_t)tl->t[wi].off,
                    (size_t)(si >= 0 ? tl->t[si].off : -1),
                    (unsigned)w0p[0], (unsigned)w0p[1],
                    (unsigned)w0p[2], (unsigned)w0p[3],
                    (unsigned)w0p[4], (unsigned)w0p[5],
                    (unsigned)w0p[6], (unsigned)w0p[7],
                    s0p ? (double)bf16_f(s0p[0]) : -9.0);
        }
        const SaltQuant *qm = salt_quant_get(8);
        if (qm && qm->matvec)
            qm->matvec((const void *)(tr + tl->t[wi].off),
                       (const void *)(tr + tl->t[si].off),
                       bias, R, C, x, y);
    } else {
        /* Main-pipeline GPU routing (SALT_GPU_TRUNK=1): mapped-only
         * operation scope requires every eligible Q4 projection to use
         * the canonical source mapping. Legacy unguarded mode retains its
         * batch threshold and CPU fallback. */
        const char *gpu_trunk = getenv("SALT_GPU_TRUNK");
        int gpu_trunk_on = gpu_trunk && *gpu_trunk != '0';
        int mapped_required = gpu_trunk_on && salt_gpu_mapped_only();
        /* Eligible Q4 projections use source-backed Metal regardless of
         * batch size while mapped-only is active, and failure is terminal.
         * Outside that operation scope, the legacy singleton path remains
         * thresholded and may fall back to CPU. Tiny a/b gates (R=32) and
         * non-Q4 tensors remain CPU-owned. */
        if (gpu_trunk_on && R >= 512 &&
            (mapped_required || salt_attn_batch_b() >= 64)) {
            if (mapped_required)
                gpu_address_audit_index(tl, wi, si, bi, tr, R, C, 1);
            if (salt_gpu_q4_matvec(
                    (const uint32_t *)(const void *)(tr + tl->t[wi].off),
                    (const uint16_t *)(const void *)(tr + tl->t[si].off),
                    bias, R, C, x, y) == 0)
                return 0;
            if (mapped_required) return -1;
        }
        const SaltQuant *qm = salt_quant_get(
            tl->t[wi].bits > 0 ? tl->t[wi].bits : 4);
        if (qm && qm->matvec)
            qm->matvec((const void *)(tr + tl->t[wi].off),
                       si >= 0 ? (const void *)(tr + tl->t[si].off) : NULL,
                       bias, R, C, x, y);
        else
            salt_q4_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[wi].off),
                (const uint16_t *)(const void *)(tr + tl->t[si].off),
                bias, R, C, x, y);
    }
    return 0;
}

typedef struct Q4ChunkBuffer {
    SaltGpuSharedBuffer shared;
    float *data;
    int shared_owned;
} Q4ChunkBuffer;

static int q4_indexed_mix_enabled(void) {
    const char *value = getenv("SALT_GPU_INDEXED_MIX");
    return salt_gpu_mapped_only() && value && *value && *value != '0';
}

static int q4_chunk_buffer_alloc(Q4ChunkBuffer *buffer, size_t nbytes,
                                 int require_shared) {
    if (!buffer || nbytes < 1) return -1;
    memset(buffer, 0, sizeof *buffer);
    if (require_shared) {
        if (salt_gpu_shared_buffer_alloc(&buffer->shared, nbytes) != 0)
            return -1;
        buffer->data = (float *)buffer->shared.contents;
        buffer->shared_owned = 1;
        return 0;
    }
    buffer->data = (float *)malloc(nbytes);
    return buffer->data ? 0 : -1;
}

static void q4_chunk_buffer_release(Q4ChunkBuffer *buffer) {
    if (!buffer) return;
    if (buffer->shared_owned) {
        if (salt_gpu_shared_buffer_free(&buffer->shared) != 0) {
            fprintf(stderr, "gpu-mix: shared output cleanup failed\n");
            fflush(stderr);
            _Exit(125);
        }
    } else {
        free(buffer->data);
    }
    memset(buffer, 0, sizeof *buffer);
}

/* Batched projection with F8 dispatch: the chunk-prefill path uses
 * salt_q4_matvec_batch for the 4-bit tensors (dequant once, reuse
 * over B). F8_E4M3 tensors (dtype==2) cannot use the 4-bit batch
 * kernel -- run B serial salt_f8_matvec calls (correct, per-token;
 * the batch is a perf optimization, not a correctness requirement). */
static int q4_batch_proj(const SaltTrunkLayout *tl, int wi, int si,
                           int bi, const uint8_t *tr, int R, int C, int B,
                           const float *xs, float *ys,
                           SaltGpuSharedBuffer *output,
                           size_t y_float_offset) {
    if (!tr || !xs || !ys || B < 1 ||
        !proj_shape_ok(tl, wi, si, bi, R, C))
        return -1;
    if (getenv("SALT_NAN_PROBE")) {
        static int dbg_first = 1;
        if (dbg_first) {
            dbg_first = 0;
            fprintf(stderr, "[bproj] wi=%d si=%d dtype=%d R=%d C=%d B=%d\n",
                    wi, si, tl->t[wi].dtype, R, C, B);
        }
    }
    if (tl->t[wi].dtype == 2) {
        int SR = si >= 0 && tl->t[si].rank > 0 ? (int)tl->t[si].dims[0] : 4;
        int SC = si >= 0 && tl->t[si].rank > 1 ? (int)tl->t[si].dims[1] : 16;
        if (getenv("SALT_NAN_PROBE")) {
            static int dbg_L = -1;
            if (dbg_L < 0) {
                dbg_L = 1;
                const uint16_t *s0 = (const uint16_t *)(const void *)
                                     (tr + tl->t[si].off);
                double xr2 = 0;
                for (int i = 0; i < C; i++)
                    xr2 += (double)xs[i] * xs[i];
                fprintf(stderr,
                        "[f8proj] L? wi=%d si=%d R=%d C=%d B=%d "
                        "SR=%d SC=%d woff=%ld soff=%ld s0=%u x-rms=%.4g "
                        "w[0..3]=%u %u %u %u\n",
                        wi, si, R, C, B, SR, SC, tl->t[wi].off,
                        tl->t[si].off, (unsigned)s0[0],
                        sqrt(xr2 / C), (unsigned)((const uint8_t *)
                        (const void *)(tr + tl->t[wi].off))[0],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[1],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[2],
                        (unsigned)((const uint8_t *)(const void *)
                        (tr + tl->t[wi].off))[3]);
            }
        }
        for (int b = 0; b < B; b++)
            salt_f8_matvec_bf16(
                (const uint8_t *)(const void *)(tr + tl->t[wi].off),
                (const uint16_t *)(const void *)(tr + tl->t[si].off),
                R, C, SR, SC, xs + (size_t)b * C, ys + (size_t)b * R);
        return 0;
    }
    if (tl->t[wi].dtype == 4) {
        for (int b = 0; b < B; b++)
            salt_bf16_matvec(
                (const uint16_t *)(const void *)(tr + tl->t[wi].off),
                R, C, xs + (size_t)b * C, NULL,
                ys + (size_t)b * R);
        return 0;
    }
    if (tl->t[wi].bits == 8) {
        /* MLX 8-bit affine: no batched kernel -- B serial LUT calls
         * (correct; the batch is a perf optimization only). */
        const uint16_t *bias = bi >= 0
            ? (const uint16_t *)(const void *)(tr + tl->t[bi].off) : NULL;
        if (getenv("SALT_NAN_PROBE") && B > 21 && R == 2048) {
        }
        for (int b = 0; b < B; b++)
            salt_q8_matvec(
                (const uint32_t *)(const void *)(tr + tl->t[wi].off),
                (const uint16_t *)(const void *)(tr + tl->t[si].off),
                bias, R, C, xs + (size_t)b * C, ys + (size_t)b * R);
        if (getenv("SALT_NAN_PROBE") && B > 21 && R == 8192) {
            double xr2 = 0.0, yr2 = 0.0;
            for (int i = 0; i < C; i++)
                xr2 += (double)xs[21 * C + i] * xs[21 * C + i];
            for (int i = 0; i < R; i++)
                yr2 += (double)ys[21 * R + i] * ys[21 * R + i];
            const uint8_t *w0p = (const uint8_t *)(const void *)
                (tr + tl->t[wi].off);
            const uint16_t *s0p = si >= 0
                ? (const uint16_t *)(const void *)(tr + tl->t[si].off)
                : NULL;
            fprintf(stderr, "[boqkv] wi=%d si=%d R=%d C=%d x-rms=%.6g "
                    "y-rms=%.6g s0=%.6g w[0..7]=%u %u %u %u %u %u %u %u "
                    "y[0..2]=%.6g %.6g %.6g\n",
                    wi, si, R, C, sqrt(xr2 / C), sqrt(yr2 / R),
                    s0p ? (double)bf16_f(s0p[0]) : -9.0,
                    (unsigned)w0p[0], (unsigned)w0p[1],
                    (unsigned)w0p[2], (unsigned)w0p[3],
                    (unsigned)w0p[4], (unsigned)w0p[5],
                    (unsigned)w0p[6], (unsigned)w0p[7],
                    (double)ys[21 * R], (double)ys[21 * R + 1],
                    (double)ys[21 * R + 2]);
        }
        return 0;
    }
    if (salt_gpu_mapped_only() && R >= 512)
        gpu_address_audit_index(tl, wi, si, bi, tr, R, C, B);
    if (q4_indexed_mix_enabled() && R >= 512 && B >= 2) {
        const char *pct_text = getenv("SALT_GPU_MIX_GPU_PCT");
        const char *side = getenv("SALT_GPU_MIX_GPU_SIDE");
        int pct = pct_text ? atoi(pct_text) : 50;
        int gpu_jobs, gpu_first_job;
        if (!output || !output->contents || !output->backend ||
            (float *)output->contents + y_float_offset != ys)
            return -1;
        if (pct < 1 || pct > 99) pct = 50;
        gpu_jobs = (B * pct + 99) / 100;
        if (gpu_jobs < 1) gpu_jobs = 1;
        if (gpu_jobs >= B) gpu_jobs = B - 1;
        gpu_first_job = side && !strcmp(side, "back") ? B - gpu_jobs : 0;
        return salt_gpu_mixed_proj_batch(
            (const uint32_t *)(const void *)(tr + tl->t[wi].off),
            (const uint16_t *)(const void *)(tr + tl->t[si].off),
            bi >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[bi].off)
                    : NULL,
            R, C, B, gpu_first_job, gpu_jobs, xs, output,
            y_float_offset, (const void *)&tl->t[wi]);
    }
    return salt_q4_matvec_batch(
        (const uint32_t *)(const void *)(tr + tl->t[wi].off),
        (const uint16_t *)(const void *)(tr + tl->t[si].off),
        bi >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[bi].off)
                : NULL,
        R, C, B, xs, ys);
}

/* Full-GQA step for layer L. state[hidden] in/out (residual added).
 * token = current position (cache write slot).
 *
 * Reference: modeling_qwen3_5.py Qwen3_5Attention.
 *   q, gate = chunk(q_proj(x) [16 x 512], 2)   -> q[16x256], gate[16x256]
 *   q = q_norm(q); k = k_norm(k_proj(x)); v = v_proj(x)
 *   q, k = RoPE(q, k)          (partial_rotary 0.25 -> 64 dims, mrope)
 *   attn = softmax(q k^T / sqrt(256)) v         (GQA, repeat_kv 8)
 *   out = o_proj(attn * sigmoid(gate))
 */
/* forward decl: the shared GQA serial body (defined after gqa_step) --
 * both gqa_step (serial) and gqa_chunk (batched projections) call it */
static int gqa_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                    const uint8_t *tr, SaltKvCache *kv, int token,
                    float *q, float *k, float *v, float *attn_out);
static int gqa_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                    const uint8_t *tr, float *state, SaltKvCache *kv,
                    int token) {
    int qi = tl->q3_q[L], qs = tl->q3_qs[L], qb = tl->q3_qb[L];
    int ki = tl->q3_k[L], ks = tl->q3_ks[L], kb = tl->q3_kb[L];
    int vi = tl->q3_v[L], vs = tl->q3_vs[L], vb = tl->q3_vb[L];
    int oi = tl->q3_o[L], os = tl->q3_os[L], ob = tl->q3_ob[L];
    int qn = tl->q3_qn[L], kn = tl->q3_kn[L];
    int iln = tl->attn_norm[L];
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 || qn < 0 || kn < 0)
        return 0;                    /* incomplete graph: skip */
    int H = cfg->hidden;
    int qrows, krows, vrows, orows, qcols, ocols;
    int heads, kv_heads, kh, rope_dim;
    long qcols_l = salt_trunk_tensor_cols(&tl->t[qi]);
    long ocols_l = salt_trunk_tensor_cols(&tl->t[oi]);
    if (salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0 ||
        qcols_l != H || ocols_l < 1 || ocols_l > INT_MAX ||
        salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                (int)ocols_l, &heads, &kv_heads, &kh,
                                &rope_dim) != 0) {
        fprintf(stderr, "qwen attn: L%d bad dims q=%ldx%ld k=%ldx%ld "
                "v=%ldx%ld o=%ldx%ld heads=%d kvh=%d\n", L,
                tl->t[qi].dims[0], qcols_l,
                tl->t[ki].dims[0], qcols_l,
                tl->t[vi].dims[0], qcols_l,
                tl->t[oi].dims[0], ocols_l, cfg->n_heads,
                cfg->n_kv_heads);
        return -1;
    }
    qcols = (int)qcols_l;
    ocols = (int)ocols_l;
    (void)rope_dim;
    if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens) return 0;

    /* per-token KV storage: [n_layers][max_tokens][krows + vrows] */
    int kvlat = krows + vrows;
    if (kv->kvlat != kvlat && kv->kvlat != 0) return 0;

    long scratch_need = salt_attn_gqa_scratch_floats(
        cfg, qrows, krows, vrows, orows, ocols, kv->max_tokens);
    if (scratch_need < 0 || !kv->scratch || kv->scratch_n < scratch_need)
        return -1;
    if (getenv("SALT_NAN_PROBE") && L == 3) {
        fprintf(stderr, "[gqa] L%d enter scratch_n=%ld need=%d\n", L,
                kv->scratch_n, (int)scratch_need);
        fprintf(stderr, "[gqa] L%d orows=%d ocols=%d heads=%d kh=%d "
                        "qrows=%d\n", L, orows, ocols, heads, kh, qrows);
    }

    double _g0 = getenv("SALT_GQA_MS") ? now_s() : 0;
    float *buf = kv->scratch;   /* arena, sized at init (never per-call) */
    float *xin = buf;               /* input_layernorm(state) */
    float *q = xin + H;             /* qrows: q(16x256) + gate(16x256) */
    float *k = q + qrows;
    float *v = k + krows;
    float *o = v + vrows;
    float *attn_out = o + orows;

    /* reference: x = x + attn(input_layernorm(x)) -- the projections
     * read the NORMED input, the residual adds to the raw state. */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[iln].off);
        memcpy(xin, state, (size_t)H * sizeof(float));
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
        float r = sqrtf((float)(ss / (double)H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            uint32_t bits = (uint32_t)nw[i] << 16;
            float w;
            memcpy(&w, &bits, 4);
            xin[i] = xin[i] / r * w;
        }
    } else {
        memcpy(xin, state, (size_t)H * sizeof(float));
    }

    /* projections */
    if (getenv("SALT_NAN_PROBE") && L == 3) {
        const uint16_t *s0p = (const uint16_t *)(const void *)(tr +
                              tl->t[qs].off);
        fprintf(stderr, "[gqa-proj] L%d qi=%d qs=%d q_woff=%ld q_soff=%ld "
                "q_srank=%d q_sdims=%ldx%ld s0[0..3]=%.6g %.6g %.6g %.6g "
                "s0[16..19]=%.6g %.6g %.6g %.6g\n",
                L, qi, qs, tl->t[qi].off, tl->t[qs].off,
                tl->t[qs].rank, tl->t[qs].dims[0], tl->t[qs].dims[1],
                bf16_f(s0p[0]), bf16_f(s0p[1]), bf16_f(s0p[2]), bf16_f(s0p[3]),
                bf16_f(s0p[16]), bf16_f(s0p[17]), bf16_f(s0p[18]),
                bf16_f(s0p[19]));
    }
    /* the q/k/v fusion (the default ON): the 3 projections run as
     * ONE thread-shared multi-weight batch -- the weighted partition
     * spans the concatenated rows (8192+512+512), each worker's range
     * crossing the job boundaries. Per-row bit-identical.
     * SALT_QKV_BATCH=0 disables. */
    {
        const char *qbe = getenv("SALT_QKV_BATCH");
        int qb_on = qbe ? *qbe != '0' : 1;
        if (qb_on && !getenv("SALT_GPU_TRUNK")) {
        SaltBatchJob qkv[3];
        int wis[3] = {qi, ki, vi}, sis[3] = {qs, ks, vs};
        int bis[3] = {qb, kb, vb};
        int Rjs[3] = {qrows, krows, vrows};
        float *yout[3] = {q, k, v};
        int ok = 1;
        for (int j = 0; j < 3; j++) {
            int bits = wis[j] >= 0 && tl->t[wis[j]].bits > 0
                ? tl->t[wis[j]].bits : 4;
            if (wis[j] < 0 || sis[j] < 0 ||
                tl->t[wis[j]].dtype == 2 || tl->t[wis[j]].dtype == 4 ||
                bits != 4) {
                ok = 0;
                break;
            }
            qkv[j].vals = (const uint32_t *)(const void *)(tr + tl->t[wis[j]].off);
            qkv[j].scales = (const uint16_t *)(const void *)(tr + tl->t[sis[j]].off);
            qkv[j].biases = bis[j] >= 0
                ? (const uint16_t *)(const void *)(tr + tl->t[bis[j]].off) : NULL;
            qkv[j].R = Rjs[j]; qkv[j].C = qcols; qkv[j].B = 1;
            qkv[j].xs = xin; qkv[j].ys = yout[j];
            qkv[j].r0 = 0; qkv[j].r1 = Rjs[j];
        }
        if (ok && salt_q4_multi_batch(qkv, 3) == 0)
            goto qkv_done;
        }
    }
    if (q4_proj(tl, qi, qs, qb, tr, qrows, qcols, xin, q) != 0 ||
        q4_proj(tl, ki, ks, kb, tr, krows, qcols, xin, k) != 0 ||
        q4_proj(tl, vi, vs, vb, tr, vrows, qcols, xin, v) != 0) {
        /* buf is the shared arena (kv->scratch) -- never freed here */
        return -1;
    }
    qkv_done:
        ;   /* the C99 label needs a statement before the decl */
    double _g1 = getenv("SALT_GQA_MS") ? now_s() : 0;
    /* split q | gate and per-head q_norm (moved into gqa_body) */
    if (gqa_body(cfg, tl, L, tr, kv, token, q, k, v, attn_out) != 0)
        return -1;
    double _g2 = getenv("SALT_GQA_MS") ? now_s() : 0;
    /* the body's arena slices (gate/qq/kk/cos/sin/scores/wgt) follow
     * attn_out inside kv->scratch -- the probes below read them. */
    {
        float *gate = attn_out + ocols;
        float *qq = gate + (size_t)heads * kh;
        if (getenv("SALT_NAN_PROBE") && L == 3) {
            double q2 = 0.0, a2 = 0.0, g2 = 0.0;
            float gmin = 1e30f, gmax = -1e30f;
            for (int i = 0; i < heads * kh; i++) {
                q2 += (double)qq[i] * qq[i];
                a2 += (double)attn_out[i] * attn_out[i];
                if (gate[i] < gmin) gmin = gate[i];
                if (gate[i] > gmax) gmax = gate[i];
            }
            for (int i = 0; i < vrows; i++) g2 += (double)v[i] * v[i];
            fprintf(stderr, "[gqa] L3 q-rms %.6g attn-rms %.6g v-rms %.6g "
                    "gate[%.4g, %.4g] heads=%d kh=%d qh=%d qrows=%d\n",
                    sqrt(q2 / (heads * kh)), sqrt(a2 / (heads * kh)),
                    sqrt(g2 / vrows), gmin, gmax, heads, kh,
                    qrows / heads, qrows);
        }
    }
    /* o_proj(attn_out) -> o; residual add. */
    if (q4_proj(tl, oi, os, ob, tr, orows, ocols, attn_out, o) != 0) {
        return -1;
    }
    double _g3 = getenv("SALT_GQA_MS") ? now_s() : 0;
    if (getenv("SALT_GQA_MS"))
        fprintf(stderr, "[gqams] L%d norm+proj=%.2f body=%.2f o=%.2f ms\n",
                L, (_g1-_g0)*1e3, (_g2-_g1)*1e3, (_g3-_g2)*1e3);
    if (getenv("SALT_NAN_PROBE") && (L == 3 || L == 7)) {
        double o2 = 0.0, a2 = 0.0;
        for (int i = 0; i < H; i++) o2 += (double)o[i] * o[i];
        for (int i = 0; i < ocols; i++)
            a2 += (double)attn_out[i] * attn_out[i];
        fprintf(stderr, "[gqa] L%d o_proj-rms %.6g attn-rms(%d) %.6g "
                "ratio %.3g\n", L, sqrt(o2 / H), ocols, sqrt(a2 / ocols),
                sqrt(o2 / H) / (sqrt(a2 / ocols) + 1e-30f));
        if (token == 1 && L == 7) {
            FILE *gf = fopen("/tmp/q36-eng-gqa7.bin", "wb");
            if (gf) {
                fwrite(attn_out, sizeof(float), (size_t)ocols, gf);
                fclose(gf);
            }
        }
        if (token == 7 && L == 3) {
            FILE *gf = fopen("/tmp/q36-eng-gqa-L3-t7.bin", "wb");
            if (gf) {
                fwrite(attn_out, sizeof(float), (size_t)ocols, gf);
                fclose(gf);
            }
        }
    }
    for (int i = 0; i < H; i++) state[i] += o[i];

    return 0;
}

/* GQA prep: q|gate split, q/k RMSNorm, RoPE, cache write. This half
 * has NO dependency on the past (pure per-token), so the chunk path
 * can run it for all B tokens FIRST (filling the whole chunk's K/V
 * cache), then batch the attention. gqa_body = gqa_prep + the serial
 * attention below. Bit-fidelity anchor: identical code on both
 * paths. Returns 0 on success; fills qq (normed+rotated q), gate,
 * and writes k/v into kv->kv at [token]. */
static int gqa_prep(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                    const uint8_t *tr, SaltKvCache *kv, int token,
                    float *q, float *k, float *v, float *qq, float *gate,
                    float *cos_t, float *sin_t) {
    int qi = tl->q3_q[L], qn = tl->q3_qn[L], kn = tl->q3_kn[L];
    int ki = tl->q3_k[L], vi = tl->q3_v[L], oi = tl->q3_o[L];
    int qrows, krows, vrows, orows, heads, kv_heads, kh, rd;
    long ocols;
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 || qn < 0 || kn < 0 ||
        salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0)
        return -1;
    ocols = salt_trunk_tensor_cols(&tl->t[oi]);
    if (ocols < 1 || ocols > INT_MAX ||
        salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                (int)ocols, &heads, &kv_heads, &kh,
                                &rd) != 0 ||
        tl->t[qn].rank < 1 || tl->t[qn].dims[0] != kh ||
        tl->t[kn].rank < 1 || tl->t[kn].dims[0] != kh)
        return -1;
    int qh = 2 * kh;
    if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens) return 0;
    int kvlat = krows + vrows;
    if (kv->kvlat != kvlat && kv->kvlat != 0) return 0;

    memset(gate, 0, (size_t)heads * kh * sizeof(float));
    memset(qq, 0, (size_t)heads * kh * sizeof(float));
    for (int h = 0; h < heads; h++) {
        float *hq = q + (size_t)h * qh;
        memcpy(qq + (size_t)h * kh, hq, (size_t)kh * sizeof(float));
        memcpy(gate + (size_t)h * kh, hq + kh, (size_t)kh * sizeof(float));
    }
    if (getenv("SALT_NAN_PROBE") && L == 3 && getenv("SALT_DUMP_STATE")) {
        FILE *rf = fopen("/tmp/q36-qraw-L3-h0.bin", "wb");
        if (rf) {
            fwrite(qq, sizeof(float), (size_t)kh, rf);
            fclose(rf);
        }
    }
    /* q_norm / k_norm: RMSNorm per head over the full 256 dims */
    const uint16_t *qwn = (const uint16_t *)(const void *)(tr + tl->t[qn].off);
    const uint16_t *kwn = (const uint16_t *)(const void *)(tr + tl->t[kn].off);
    for (int h = 0; h < heads; h++)
        rmsnorm(qwn, kh, qq + (size_t)h * kh);
    for (int h = 0; h < kv_heads; h++)
        rmsnorm(kwn, kh, k + (size_t)h * kh);

    /* RoPE on the model-owned rotary slice, interleaved.
     * Plain text decode: position = token (mrope section T only). */
    {
        double theta = cfg->rope_theta;
        memset(cos_t, 0, (size_t)rd * sizeof(float));
        memset(sin_t, 0, (size_t)rd * sizeof(float));
        for (int i = 0; i < rd / 2; i++) {
            double inv = 1.0 / pow(theta, (double)(2 * i) / (double)rd);
            double f = (double)token * inv;
            cos_t[i] = (float)cos(f);
            sin_t[i] = (float)sin(f);
        }
        for (int h = 0; h < heads; h++) {
            float *hq = qq + (size_t)h * kh;
            for (int i = 0; i < rd / 2; i++) {
                float x0 = hq[i], x1 = hq[i + rd / 2];
                float c = cos_t[i], s = sin_t[i];
                hq[i] = x0 * c - x1 * s;
                hq[i + rd / 2] = x0 * s + x1 * c;
            }
        }
        for (int h = 0; h < kv_heads; h++) {
            float *hk = k + (size_t)h * kh;
            for (int i = 0; i < rd / 2; i++) {
                float x0 = hk[i], x1 = hk[i + rd / 2];
                float c = cos_t[i], s = sin_t[i];
                hk[i] = x0 * c - x1 * s;
                hk[i + rd / 2] = x0 * s + x1 * c;
            }
        }
    }

    /* write k/v into the cache at token */
    float *ck = kv->kv + ((size_t)L * kv->max_tokens + token) * kvlat;
    if (getenv("SALT_NAN_PROBE") && L == 3 && token >= 100)
        fprintf(stderr, "[gqa-w] L3 token=%d ck=%p kv=%p max=%d "
                "kvlat=%d rows=%d\n", token, (void *)ck, (void *)kv->kv,
                kv->max_tokens, kvlat, krows + vrows);
    memcpy(ck, k, (size_t)krows * sizeof(float));
    memcpy(ck + krows, v, (size_t)vrows * sizeof(float));
    return 0;
}

/* ---- threaded GQA attention (M7) ---------------------------------- */
/* One (b,h) attention pair in the EXACT serial order (t2 ascending,
 * i ascending, same expf/div sequence). Each pair is fully
 * independent (disjoint aob slices, private sc/wg), so any thread
 * assignment is bit-identical to the serial gqa_body loop. Used by
 * the chunk path (B tokens x heads) AND the serial decode path
 * (B=1: 16 heads across 8 threads). */typedef struct {
    const SaltKvCache *kv;
    const float *qqb, *gateb;
    float *aob;
    int L, t0, B, heads, kv_heads, kh, krows, vrows, kvlat, ocols;
    float dscale;
    int tid, nth;              /* thread index, thread count */
    /* phase-probe accumulators (SALT_GQA_MS): per-thread sums */
    double ts_score, ts_soft, ts_wsum;
} GqaAtnThr;

static void *gqa_atn_thread(void *arg) {
    GqaAtnThr *w = (GqaAtnThr *)arg;
    /* private score/weight buffers (per-thread, no sharing); drawn
     * from the persistent pool when up (first 1*cap of the 8*cap
     * per-thread region), else heap. */
    size_t cap = (size_t)w->kv->max_tokens;
    float *sc, *wg;
    int pooled = w->kv->ath != NULL;
    if (pooled) {
        size_t per = (size_t)8 * cap;
        sc = w->kv->asc8 + (size_t)w->tid * per;
        wg = w->kv->awg8 + (size_t)w->tid * per;
    } else {
        sc = (float *)calloc(cap, sizeof(float));
        wg = (float *)calloc(cap, sizeof(float));
        if (!sc || !wg) { free(sc); free(wg); return NULL; }
    }
    int use_simd = 0;
#ifdef __aarch64__
    use_simd = salt_kernels_simd() && w->kh % 4 == 0;
#endif
    int total = w->B * w->heads;
    /* round-robin over (b,h) pairs: thread tid handles p ≡ tid (mod N) */
    for (int p = w->tid; p < total; p += w->nth) {
        int b = p / w->heads, h = p % w->heads;
        int npos = w->t0 + b + 1;
        const float *qh = w->qqb + ((size_t)b * w->heads + h) * w->kh;
        int khh = h / (w->heads / w->kv_heads);
        float mx = -1e30f;
#ifdef __aarch64__
        if (use_simd) {
            /* scores 4-wide over t2 (outputs); each t2-lane keeps the
             * exact serial i-accumulation order (M9 pattern). The
             * k2 rows are kvlat-strided, so lanes load scalar and
             * assemble -- same FMA sequence per lane as the scalar
             * loop, bit-identical. */
            float32x4_t dsc = vdupq_n_f32(w->dscale);
            int nq = npos & ~3;
            for (int t2 = 0; t2 < nq; t2 += 4) {
                const float *k0 = w->kv->kv +
                    ((size_t)w->L * w->kv->max_tokens + t2 + 0) * w->kvlat +
                    (size_t)khh * w->kh;
                const float *k1 = k0 + w->kvlat;
                const float *k2 = k1 + w->kvlat;
                const float *k3 = k2 + w->kvlat;
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < w->kh; i++) {
                    float32x4_t qv = vdupq_n_f32(qh[i]);
                    float32x4_t kv = vdupq_n_f32(0.0f);
                    kv = vld1q_lane_f32(k0 + i, kv, 0);
                    kv = vld1q_lane_f32(k1 + i, kv, 1);
                    kv = vld1q_lane_f32(k2 + i, kv, 2);
                    kv = vld1q_lane_f32(k3 + i, kv, 3);
                    acc = vaddq_f32(acc, vmulq_f32(qv, kv));
                }
                float32x4_t s4 = vmulq_f32(acc, dsc);
                vst1q_f32(sc + t2, s4);
            }
            for (int t2 = nq; t2 < npos; t2++) {
                const float *k2 = w->kv->kv +
                    ((size_t)w->L * w->kv->max_tokens + t2) * w->kvlat +
                    (size_t)khh * w->kh;
                float acc = 0.0f;
                for (int i = 0; i < w->kh; i++) acc += qh[i] * k2[i];
                sc[t2] = acc * w->dscale;
            }
            for (int t2 = 0; t2 < npos; t2++)
                if (sc[t2] > mx) mx = sc[t2];
        } else
#endif
        {
        for (int t2 = 0; t2 < npos; t2++) {
            const float *k2 = w->kv->kv +
                ((size_t)w->L * w->kv->max_tokens + t2) * w->kvlat +
                (size_t)khh * w->kh;
            float acc = 0.0f;
            for (int i = 0; i < w->kh; i++) acc += qh[i] * k2[i];
            sc[t2] = acc * w->dscale;
            if (sc[t2] > mx) mx = sc[t2];
        }
        }
        float sum = 0.0f;
        for (int t2 = 0; t2 < npos; t2++) {
            wg[t2] = salt_expf(sc[t2] - mx);
            sum += wg[t2];
        }
        for (int t2 = 0; t2 < npos; t2++) wg[t2] /= sum;
        float *aor = w->aob + ((size_t)b * w->ocols) + (size_t)h * w->kh;
#ifdef __aarch64__
        if (use_simd) {
            /* weighted sum 4-wide over i (outputs); each i-lane
             * accumulates over t2 in serial order (M9 pattern). v2
             * rows are contiguous per t2 -> clean vector loads.
             * aob is zeroed by the caller (threads write disjoint
             * h slices), so the accumulators start at 0. */
            for (int t2 = 0; t2 < npos; t2++) {
                const float *v2 = w->kv->kv +
                    ((size_t)w->L * w->kv->max_tokens + t2) * w->kvlat +
                    w->krows + (size_t)khh * w->kh;
                float32x4_t wv = vdupq_n_f32(wg[t2]);
                for (int i = 0; i < w->kh; i += 4) {
                    float32x4_t vv = vld1q_f32(v2 + i);
                    float32x4_t av = vld1q_f32(aor + i);
                    vst1q_f32(aor + i, vaddq_f32(av, vmulq_f32(wv, vv)));
                }
            }
        } else
#endif
        {
        for (int t2 = 0; t2 < npos; t2++) {
            const float *v2 = w->kv->kv +
                ((size_t)w->L * w->kv->max_tokens + t2) * w->kvlat +
                w->krows + (size_t)khh * w->kh;
            float wv = wg[t2];
            for (int i = 0; i < w->kh; i++) aor[i] += wv * v2[i];
        }
        }
        /* gate multiply (same as gqa_body) */
        const float *gh = w->gateb + ((size_t)b * w->heads + h) * w->kh;
        for (int i = 0; i < w->kh; i++) {
            float gv = gh[i];
            aor[i] *= 1.0f / (1.0f + salt_expf(-gv));
        }
    }
    if (!pooled) { free(sc); free(wg); }
    return NULL;
}

/* ---- A-full: K-reuse GQA score GEMM (2026-08-08) ------------------ */
/* The 16 q-heads map to 2 kv-heads (8 q-heads per kv-head). The old
 * worker re-read each kv-head's K/V rows once PER q-head (8x
 * redundant). This variant makes one thread own a (b, kv-head) unit:
 * it streams each K row ONCE (sequential, prefetch-friendly) into 8
 * q-head accumulators (4-wide NEON over the h dimension, serial i
 * reduction per score -- bit-identical), then reuses each V row
 * across the 8 heads' weighted sums. K/V read traffic drops 8x, and
 * the dominant access becomes a sequential stream instead of 8-way
 * strided. Enabled only when heads/kv_heads == 8 and kh%4==0. */
#define A_FULL_HPS 8               /* q-heads per kv-head (16/2) */

#ifdef __aarch64__
/* NEON-only worker: the K-reuse restructure depends on the 4-wide
 * lane gather/broadcast pattern. On non-aarch64 (x86 EC2, edge) the
 * spawn selects the plain worker's scalar path instead -- same math,
 * bit-identical by construction (the NEON lanes keep the exact serial
 * i-accumulation per score, and the scalar path IS that serial
 * order). No AVX2 port of this worker: the plain worker's scalar path
 * is the low-compute baseline the edge/EC2 deployments run. */
static void *gqa_atn_thread_kr(void *arg) {
    GqaAtnThr *w = (GqaAtnThr *)arg;
    size_t cap = (size_t)w->kv->max_tokens;
    /* per-head score/weight buffers: 8 heads x npos. Drawn from the
     * persistent attention pool when it is up (no per-call calloc);
     * fall back to heap otherwise. */
    float *sc8, *wg8;
    int pooled = w->kv->ath != NULL;
    if (pooled) {
        size_t per = (size_t)8 * cap;
        sc8 = w->kv->asc8 + (size_t)w->tid * per;
        wg8 = w->kv->awg8 + (size_t)w->tid * per;
    } else {
        sc8 = (float *)calloc((size_t)8 * cap, sizeof(float));
        wg8 = (float *)calloc((size_t)8 * cap, sizeof(float));
        if (!sc8 || !wg8) { free(sc8); free(wg8); return NULL; }
    }
    float32x4_t dsc = vdupq_n_f32(w->dscale);
    /* the transposed-q staging, hoisted OUT of the pair loop: alloca
     * in a loop grows the stack per iteration (8 KB x the pairs) and
     * never unwinds until the scope exits -- across B=103 pairs that
     * is ~824 KB per worker, crossing the thread's stack guard at a
     * run-dependent point -> the nondeterministic SIGBUS. One fixed
     * buffer reused per pair. */
    float *qqt = (float *)alloca((size_t)w->kh * 8 * sizeof(float));
    const int hps = A_FULL_HPS;
    const int gqa_ms = getenv("SALT_GQA_MS") != NULL;
    int total = w->B * w->kv_heads;          /* (b, kv-head) units */
    for (int p = w->tid; p < total; p += w->nth) {
        int b = p / w->kv_heads, khh = p % w->kv_heads;
        int npos = w->t0 + b + 1;
        if (getenv("SALT_NAN_PROBE") && w->L == 3 && w->tid == 0 &&
            (b < 12))
            fprintf(stderr, "[gqa-kr] L3 b=%d khh=%d npos=%d cap=%zu "
                    "tid=%d p=%d total=%d\n", b, khh, npos, cap,
                    w->tid, p, total);
        const int h0 = khh * hps;            /* first q-head of this kv */
        const float *kbase = w->kv->kv +
            (size_t)w->L * w->kv->max_tokens * w->kvlat +
            (size_t)khh * w->kh;             /* row t2 at kbase+t2*kvlat */
        const float *vbase = kbase + w->krows;
        double t0s = 0.0, t1s = 0.0, t2s = 0.0;
        if (gqa_ms) t0s = now_s();
        /* ---- scores: one sequential pass over K, 8 heads at once.
         * For each t2: acc[h] = sum_i qh[h][i]*k2[i] (i ascending,
         * serial per score -- bit-identical to the old loops). K row
         * read once, feeds 8 accumulators (2x 4-wide NEON).
         *
         * ILP: two K rows are live at once (t2, t2+1), giving 4
         * independent FMA chains (a0/a1 for t2, a2/a3 for t2+1).
         * The score loop is latency-bound on its FMA chains (2 chains
         * = ~2 MACs/cycle on NEON); pairing doubles the chains to
         * ~4 MACs/cycle. Each score's i-accumulation order is
         * unchanged, so results are bit-identical. */
        int nq2 = npos & ~3;
        int t2;
        /* the transposed q: qqt[i*8 + hh] = qh_hh[i] -- the 8 heads
         * contiguous per i, so the score loop loads 2 vectors
         * instead of 8 strided gathers. Same numbers, same FMA
         * order per (hh, i) -- bit-identical. (qqt hoisted: alloca
         * in a loop grows the stack unbounded.) */
        for (int hh = 0; hh < 8; hh++) {
            const float *qh = w->qqb +
                ((size_t)b * w->heads + h0 + hh) * w->kh;
            for (int i = 0; i < w->kh; i++)
                qqt[(size_t)i * 8 + hh] = qh[i];
        }
        for (t2 = 0; t2 + 1 < nq2; t2 += 2) {
            const float *k2 = kbase + (size_t)t2 * w->kvlat;
            const float *k2n = k2 + w->kvlat;   /* row t2+1 */
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            for (int i = 0; i < w->kh; i++) {
                float k0 = k2[i], k1 = k2n[i];  /* 2 loads, 16 uses */
                float32x4_t kv0 = vdupq_n_f32(k0), kv1 = vdupq_n_f32(k1);
                /* lane j = head j's q at i (contiguous in qqt) --
                 * each lane's accumulator j keeps the exact serial
                 * i-accumulation of head j -- bit-identical. */
                float32x4_t q0 = vld1q_f32(qqt + (size_t)i * 8);
                float32x4_t q1 = vld1q_f32(qqt + (size_t)i * 8 + 4);
                a0 = vaddq_f32(a0, vmulq_f32(q0, kv0));
                a1 = vaddq_f32(a1, vmulq_f32(q1, kv0));
                a2 = vaddq_f32(a2, vmulq_f32(q0, kv1));
                a3 = vaddq_f32(a3, vmulq_f32(q1, kv1));
            }
            float s0[4], s1[4], s2[4], s3[4];
            vst1q_f32(s0, vmulq_f32(a0, dsc));
            vst1q_f32(s1, vmulq_f32(a1, dsc));
            vst1q_f32(s2, vmulq_f32(a2, dsc));
            vst1q_f32(s3, vmulq_f32(a3, dsc));
            for (int hh = 0; hh < 4; hh++) {
                sc8[(size_t)hh * cap + t2] = s0[hh];
                sc8[(size_t)(hh + 4) * cap + t2] = s1[hh];
                sc8[(size_t)hh * cap + t2 + 1] = s2[hh];
                sc8[(size_t)(hh + 4) * cap + t2 + 1] = s3[hh];
            }
        }
        for (; t2 < npos; t2++) {
            const float *k2 = kbase + (size_t)t2 * w->kvlat;
            for (int hh = 0; hh < hps; hh++) {
                const float *qh = w->qqb +
                    ((size_t)b * w->heads + h0 + hh) * w->kh;
                float acc = 0.0f;
                for (int i = 0; i < w->kh; i++) acc += qh[i] * k2[i];
                sc8[(size_t)hh * cap + t2] = acc * w->dscale;
            }
        }
        if (gqa_ms) { t1s = now_s(); w->ts_score += t1s - t0s; }
        /* ---- softmax per head (t2 ascending, exact order). The
         * weights are stored TRANSPOSED (wg8[t2*hps + hh], 8 heads
         * contiguous per t2) so the wsum below reads them as two
         * 4-wide vector loads instead of 8 scattered loads. Softmax
         * itself is ~0% of the phase, so the strided writes here are
         * free; the FMA order per (hh, t2) is unchanged. */
        for (int hh = 0; hh < hps; hh++) {
            float *s = sc8 + (size_t)hh * cap;
            float *g = wg8 + (size_t)hh;      /* stride hps per t2 */
            float mx = -1e30f;
            for (t2 = 0; t2 < npos; t2++) if (s[t2] > mx) mx = s[t2];
            float sum = 0.0f;
            for (t2 = 0; t2 < npos; t2++) {
                float e = salt_expf(s[t2] - mx);
                g[(size_t)t2 * hps] = e;
                sum += e;
            }
            for (t2 = 0; t2 < npos; t2++) g[(size_t)t2 * hps] /= sum;
        }
        if (gqa_ms) { t2s = now_s(); w->ts_soft += t2s - t1s; }
        /* ---- weighted sum: one pass over V, 8 heads accumulate.
         * vv is loaded ONCE per (t2, i-block) and feeds all 8 aor
         * slices (V read traffic /8). Accumulation order per
         * (hh, i-lane) is t2 ascending -- bit-identical.
         *
         * Register-resident aor: the 8 accumulators live in NEON
         * registers across the whole t2 loop and are stored ONCE per
         * i-block. The old code did vld1q+vst1q round-trips per
         * (t2, hh) -- 8 loads + 8 stores per V row. Now it is 8
         * vmlaq per V row with no memory ops on aor; each lane's
         * FMA chain over t2 is unchanged, so results are
         * bit-identical.
         *
         * Transposed weights: wg8[t2*hps+hh] puts the 8 heads'
         * weights for a t2 in 32 contiguous bytes -- two vld1q
         * (w01, w45) replace eight wg8[hh*cap+t2] scattered loads
         * (cap floats apart, 8 L2 hits per t2). vdupq_lane broadcasts
         * each weight to all 4 lanes; same FMA sequence per head. */
        for (int i = 0; i < w->kh; i += 4) {
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f), acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f), acc7 = vdupq_n_f32(0.0f);
            for (t2 = 0; t2 < npos; t2++) {
                const float *v2 = vbase + (size_t)t2 * w->kvlat;
                float32x4_t vv = vld1q_f32(v2 + i);
                const float *wgt = wg8 + (size_t)t2 * hps;
                float32x4_t w01 = vld1q_f32(wgt);       /* heads 0-3 */
                float32x4_t w45 = vld1q_f32(wgt + 4);   /* heads 4-7 */
                acc0 = vaddq_f32(acc0, vmulq_f32(vdupq_laneq_f32(w01, 0), vv));
                acc1 = vaddq_f32(acc1, vmulq_f32(vdupq_laneq_f32(w01, 1), vv));
                acc2 = vaddq_f32(acc2, vmulq_f32(vdupq_laneq_f32(w01, 2), vv));
                acc3 = vaddq_f32(acc3, vmulq_f32(vdupq_laneq_f32(w01, 3), vv));
                acc4 = vaddq_f32(acc4, vmulq_f32(vdupq_laneq_f32(w45, 0), vv));
                acc5 = vaddq_f32(acc5, vmulq_f32(vdupq_laneq_f32(w45, 1), vv));
                acc6 = vaddq_f32(acc6, vmulq_f32(vdupq_laneq_f32(w45, 2), vv));
                acc7 = vaddq_f32(acc7, vmulq_f32(vdupq_laneq_f32(w45, 3), vv));
            }
            for (int hh = 0; hh < hps; hh++) {
                float *aor = w->aob + ((size_t)b * w->ocols) +
                             (size_t)(h0 + hh) * w->kh;
                float32x4_t acc = hh < 4 ? (hh == 0 ? acc0 : hh == 1 ? acc1
                                          : hh == 2 ? acc2 : acc3)
                                          : (hh == 4 ? acc4 : hh == 5 ? acc5
                                          : hh == 6 ? acc6 : acc7);
                vst1q_f32(aor + i, acc);
            }
        }
        if (gqa_ms) { w->ts_wsum += now_s() - t2s; }
        /* ---- gate multiply per head ---- */
        for (int hh = 0; hh < hps; hh++) {
            float *aor = w->aob + ((size_t)b * w->ocols) +
                         (size_t)(h0 + hh) * w->kh;
            const float *gh = w->gateb +
                ((size_t)b * w->heads + h0 + hh) * w->kh;
            for (int i = 0; i < w->kh; i++) {
                float gv = gh[i];
                aor[i] *= 1.0f / (1.0f + salt_expf(-gv));
            }
        }
    }
    if (!pooled) { free(sc8); free(wg8); }
    return NULL;
}
#endif /* __aarch64__ -- K-reuse worker */

/* pool adapters: the pool calls fn(tid, arg) where arg is the ctx
 * array; pick this thread's ctx entry and run the worker body. */
#ifdef __aarch64__
static void gqa_atn_pool_kr(int tid, void *arg) {
    gqa_atn_thread_kr(&((GqaAtnThr *)arg)[tid]);
}
#endif
static void gqa_atn_pool_plain(int tid, void *arg) {
    gqa_atn_thread(&((GqaAtnThr *)arg)[tid]);
}

/* spawn the threaded attention over B*heads pairs. aob must be
 * zeroed by the caller (threads write disjoint h slices). The thread
 * count comes from --threads (kv->nthreads), NOT a hardcoded value:
 * the pool has exactly kv->nthreads workers and every batch dispatches
 * to all of them (the done-barrier waits for all), so the ctx array is
 * sized to that count. */
static void gqa_atn_spawn(const SaltKvCache *kv, const float *qqb,
                          const float *gateb, float *aob, int L, int t0,
                          int B, int heads, int kv_heads, int kh,
                          int krows, int vrows, int ocols, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (kv->nthreads > nthreads) nthreads = kv->nthreads;
    GqaAtnThr *ctx = (GqaAtnThr *)calloc((size_t)nthreads, sizeof(GqaAtnThr));
    if (!ctx) { nthreads = 1; ctx = (GqaAtnThr *)calloc(1, sizeof(GqaAtnThr)); }
    if (!ctx) return;
    pthread_t *th = (pthread_t *)calloc((size_t)nthreads, sizeof(pthread_t));
    float dscale = 1.0f / sqrtf((float)kh);
    /* A-full: K-reuse worker when the geometry is exactly 8 q-heads
     * per kv-head and kh is 4-aligned (this model: 16/2, kh=256).
     * Same spawn contract; aob still zeroed by the caller. */
    int use_kr = 0;
#ifdef __aarch64__
    /* A-full: K-reuse worker when the geometry is exactly 8 q-heads
     * per kv-head and kh is 4-aligned (this model: 16/2, kh=256).
     * NEON-only; non-aarch64 keeps the plain worker's scalar path. */
    use_kr = (heads == kv_heads * 8) && (kh % 4 == 0) && B >= 1 &&
             salt_kernels_simd();
#endif
    for (int ti = 0; ti < nthreads; ti++) {
        ctx[ti].kv = kv; ctx[ti].qqb = qqb; ctx[ti].gateb = gateb;
        ctx[ti].aob = aob; ctx[ti].L = L; ctx[ti].t0 = t0;
        ctx[ti].B = B; ctx[ti].heads = heads;
        ctx[ti].kv_heads = kv_heads; ctx[ti].kh = kh;
        ctx[ti].krows = krows; ctx[ti].vrows = vrows;
        ctx[ti].kvlat = krows + vrows; ctx[ti].ocols = ocols;
        ctx[ti].dscale = dscale; ctx[ti].tid = ti; ctx[ti].nth = nthreads;
        ctx[ti].ts_score = 0.0; ctx[ti].ts_soft = 0.0;
        ctx[ti].ts_wsum = 0.0;
    }
    if (kv->ath) {
        /* persistent pool: zero the per-thread sc/wg regions (the
         * workers accumulate into them; the pool allocates them once
         * at init, so each spawn must clear before use). */
        size_t per = (size_t)8 * (size_t)kv->max_tokens;
        memset(kv->asc8, 0, (size_t)nthreads * per * sizeof(float));
        memset(kv->awg8, 0, (size_t)nthreads * per * sizeof(float));
#ifdef __aarch64__
        salt_attn_pool_run((SaltKvCache *)kv,
                           use_kr ? gqa_atn_pool_kr : gqa_atn_pool_plain,
                           ctx);
#else
        salt_attn_pool_run((SaltKvCache *)kv, gqa_atn_pool_plain, ctx);
#endif
    } else {
        for (int ti = 0; ti < nthreads; ti++) {
#ifdef __aarch64__
            if (pthread_create(&th[ti], NULL,
                               use_kr ? gqa_atn_thread_kr : gqa_atn_thread,
                               &ctx[ti]) != 0) {
#else
            if (pthread_create(&th[ti], NULL, gqa_atn_thread,
                               &ctx[ti]) != 0) {
#endif
                nthreads = ti;              /* spawn what we can */
                break;
            }
        }
        for (int ti = 0; ti < nthreads; ti++) pthread_join(th[ti], NULL);
    }
    if (use_kr && getenv("SALT_GQA_MS")) {
        double sc = 0, so = 0, ws = 0;
        for (int ti = 0; ti < nthreads; ti++) {
            sc += ctx[ti].ts_score; so += ctx[ti].ts_soft;
            ws += ctx[ti].ts_wsum;
        }
        double tot = sc + so + ws;
        fprintf(stderr,
                "[gqa-ms] L%d B=%d npos~%d: score %.1fs (%.0f%%) soft %.1fs "
                "(%.0f%%) wsum %.1fs (%.0f%%)\n",
                L, B, t0 + B, sc, 100.0 * sc / tot, so, 100.0 * so / tot,
                ws, 100.0 * ws / tot);
    }
    free(ctx);
    free(th);
}

/* ---- M9: threaded Gated DeltaNet decode (head-parallel) ------------ */
/* The delta-rule state update is O(1) per token (fixed-size state
 * per value head) and every value head is fully independent
 * (disjoint state slices in kv->lin, disjoint readout rows, private
 * model-sized delta scratch). The serial lin_body head loop is split across
 * threads by head range; every head runs the EXACT serial math, so
 * the result is bit-identical (same pattern as M7/M8). */
typedef struct {
    const SaltKvCache *kv;
    const float *qkv;
    float *readout, *delta;
    const float *b32, *g32;
    int L, token, k_heads, v_heads, kd, vd;
    int use_simd;
    int h0, h1;              /* head range for this thread */
} LinDeltaThr;

static void *lin_delta_thread(void *arg) {
    LinDeltaThr *w = (LinDeltaThr *)arg;
    const int kd = w->kd, vd = w->vd, k_heads = w->k_heads;
    float *S = w->kv->lin + (size_t)w->L * w->v_heads * kd * vd;
    float *delta = w->delta;
    for (int h = w->h0; h < w->h1; h++) {
        int khh = h / (w->v_heads / w->k_heads);
        const float *kh = w->qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
        const float *qh = w->qkv + (size_t)khh * kd;
        const float *vh = w->qkv + (size_t)(2 * k_heads * kd) +
                          (size_t)h * vd;
        float *Sh = S + (size_t)h * kd * vd;
        float beta = 1.0f / (1.0f + salt_expf(-w->b32[h]));   /* sigmoid */
        float decay = salt_expf(w->g32[h]);
#ifdef __aarch64__
        if (w->use_simd) {
            /* NEON delta rule: 4-wide over vd, same i-accumulation
             * order as the scalar loop (bit-identical, e2e-verified) */
            float32x4_t dcy = vdupq_n_f32(decay);
            for (int i = 0; i < kd * vd; i += 4) {
                float32x4_t s = vld1q_f32(Sh + i);
                vst1q_f32(Sh + i, vmulq_f32(s, dcy));
            }
            for (int j = 0; j < vd; j += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < kd; i++) {
                    float32x4_t s = vld1q_f32(Sh + (size_t)i * vd + j);
                    acc = vaddq_f32(acc, vmulq_f32(s, vdupq_n_f32(kh[i])));
                }
                float32x4_t vv = vld1q_f32(vh + j);
                float32x4_t dl = vmulq_f32(vsubq_f32(vv, acc),
                                           vdupq_n_f32(beta));
                vst1q_f32(delta + j, dl);
            }
            for (int i = 0; i < kd; i++) {
                float32x4_t kk = vdupq_n_f32(kh[i]);
                float *row = Sh + (size_t)i * vd;
                for (int j = 0; j < vd; j += 4) {
                    float32x4_t s = vld1q_f32(row + j);
                    float32x4_t d = vld1q_f32(delta + j);
                    vst1q_f32(row + j, vaddq_f32(s, vmulq_f32(kk, d)));
                }
            }
            float *oh = w->readout + (size_t)h * vd;
            for (int j = 0; j < vd; j += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < kd; i++) {
                    float32x4_t s = vld1q_f32(Sh + (size_t)i * vd + j);
                    acc = vaddq_f32(acc, vmulq_f32(s, vdupq_n_f32(qh[i])));
                }
                vst1q_f32(oh + j, acc);
            }
        } else
#endif
        {
            /* scalar delta rule (the serial reference path) */
            for (int i = 0; i < kd * vd; i++) Sh[i] *= decay;
            for (int j = 0; j < vd; j++) {
                float acc = 0.0f;
                for (int i = 0; i < kd; i++)
                    acc += Sh[(size_t)i * vd + j] * kh[i];
                delta[j] = (vh[j] - acc) * beta;
            }
            for (int i = 0; i < kd; i++) {
                float *row = Sh + (size_t)i * vd;
                float kk = kh[i];
                for (int j = 0; j < vd; j++)
                    row[j] += kk * delta[j];
            }
            float *oh = w->readout + (size_t)h * vd;
            for (int j = 0; j < vd; j++) {
                float acc = 0.0f;
                for (int i = 0; i < kd; i++)
                    acc += Sh[(size_t)i * vd + j] * qh[i];
                oh[j] = acc;
            }
        }
    }
    return NULL;
}

/* pool adapter for lin_delta: same tid->ctx dispatch. */
static void lin_delta_pool_adapter(int tid, void *arg) {
    lin_delta_thread(&((LinDeltaThr *)arg)[tid]);
}

/* split the v_heads value heads across threads by contiguous range
 * (head h -> thread h*v_heads/nthreads); each thread's heads keep
 * serial order. Thread count comes from --threads (kv->nthreads). */
static void lin_delta_spawn(const SaltKvCache *kv, const float *qkv,
                            float *readout, const float *b32,
                            const float *g32, float *delta_scratch,
                            int L, int token,
                            int k_heads, int v_heads, int kd, int vd,
                            int use_simd, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (kv->nthreads > nthreads) nthreads = kv->nthreads;
    /* The pool dispatches to every kv worker, including empty head ranges. */
    LinDeltaThr base;
    memset(&base, 0, sizeof base);
    base.kv = kv; base.qkv = qkv; base.readout = readout;
    base.delta = delta_scratch; base.b32 = b32; base.g32 = g32;
    base.L = L; base.token = token; base.k_heads = k_heads;
    base.v_heads = v_heads; base.kd = kd; base.vd = vd;
    base.use_simd = use_simd;
    LinDeltaThr *ctx = (LinDeltaThr *)calloc((size_t)nthreads,
                                              sizeof(LinDeltaThr));
    if (!ctx) {
        base.h0 = 0; base.h1 = v_heads;
        lin_delta_thread(&base);
        return;
    }
    for (int ti = 0; ti < nthreads; ti++) {
        ctx[ti] = base;
        ctx[ti].delta = delta_scratch + (size_t)ti * vd;
        ctx[ti].h0 = ti * v_heads / nthreads;
        ctx[ti].h1 = (ti + 1) * v_heads / nthreads;
    }
    if (kv->ath) {
        salt_attn_pool_run((SaltKvCache *)kv, lin_delta_pool_adapter, ctx);
    } else {
        pthread_t *th = (pthread_t *)calloc((size_t)nthreads,
                                             sizeof(pthread_t));
        int created = 0;
        if (th) {
            for (; created < nthreads; created++)
                if (pthread_create(&th[created], NULL, lin_delta_thread,
                                   &ctx[created]) != 0)
                    break;
            for (int ti = 0; ti < created; ti++) pthread_join(th[ti], NULL);
        }
        for (int ti = created; ti < nthreads; ti++)
            lin_delta_thread(&ctx[ti]);
        free(th);
    }
    free(ctx);
}

static int linear_size_add(size_t *total, size_t value) {
    if (*total > SIZE_MAX - value) return -1;
    *total += value;
    return 0;
}

static int linear_size_mul(size_t a, size_t b, size_t *out) {
    if (b != 0 && a > SIZE_MAX / b) return -1;
    *out = a * b;
    return 0;
}

long salt_attn_gqa_scratch_floats(const SaltCfg *cfg, int q_rows,
                                  int k_rows, int v_rows, int o_rows,
                                  int o_cols, int max_tokens) {
    int heads, kv_heads, head_dim, rope_dim;
    size_t total = 0, term;
    if (max_tokens < 1 ||
        salt_model_gqa_geometry(cfg, q_rows, k_rows, v_rows, o_rows,
                                o_cols, &heads, &kv_heads, &head_dim,
                                &rope_dim) != 0)
        return -1;
    (void)kv_heads;
    if (linear_size_add(&total, (size_t)cfg->hidden) != 0 ||
        linear_size_add(&total, (size_t)q_rows) != 0 ||
        linear_size_add(&total, (size_t)k_rows) != 0 ||
        linear_size_add(&total, (size_t)v_rows) != 0 ||
        linear_size_add(&total, (size_t)o_rows) != 0 ||
        linear_size_add(&total, (size_t)o_cols) != 0 ||
        linear_size_mul((size_t)heads, (size_t)head_dim, &term) != 0 ||
        linear_size_mul(term, 3, &term) != 0 ||
        linear_size_add(&total, term) != 0 ||
        linear_size_mul((size_t)rope_dim, 2, &term) != 0 ||
        linear_size_add(&total, term) != 0 ||
        linear_size_mul((size_t)max_tokens, 2, &term) != 0 ||
        linear_size_add(&total, term) != 0 ||
        total > (size_t)LONG_MAX)
        return -1;
    return (long)total;
}

long salt_attn_linear_scratch_floats(const SaltCfg *cfg, int hidden,
                                     int qkv_rows, int z_rows, int o_rows,
                                     int nthreads) {
    int kh, vh, kd, vd, expected_qkv;
    size_t total = 0, term, legacy, work, nth;
    if (hidden < 1 || qkv_rows < 1 || z_rows < 1 || o_rows != hidden ||
        salt_model_linear_geometry(cfg, &kh, &vh, &kd, &vd,
                                   &expected_qkv) != 0 ||
        qkv_rows != expected_qkv || z_rows != vh * vd)
        return -1;
    nth = (size_t)(nthreads > 0 ? nthreads : 1);
    if (linear_size_add(&total, (size_t)hidden) != 0 ||
        linear_size_add(&total, (size_t)qkv_rows) != 0 ||
        linear_size_add(&total, (size_t)z_rows) != 0 ||
        linear_size_add(&total, (size_t)o_rows) != 0 ||
        linear_size_mul((size_t)vh, (size_t)vd, &term) != 0 ||
        linear_size_add(&total, term) != 0 ||
        linear_size_mul((size_t)hidden, 2, &legacy) != 0 ||
        linear_size_add(&legacy, 1) != 0 ||
        linear_size_mul((size_t)vh, 3, &work) != 0 ||
        linear_size_mul(nth, (size_t)vd, &term) != 0 ||
        linear_size_add(&work, term) != 0 ||
        linear_size_add(&total, work > legacy ? work : legacy) != 0 ||
        total > (size_t)LONG_MAX)
        return -1;
    return (long)total;
}

/* ------------------------------------------------------------------ */
/* GQA serial body: q|gate split, norms, RoPE, cache write, softmax   */
/* attention, gate multiply -> attn_out. q/k/v are PRECOMPUTED (the   */
/* caller projects them: q4_proj serial, or the batched kernel in   */
/* the chunk path). This body is the bit-fidelity anchor -- both      */
/* paths run the identical code on the same q/k/v, so results are     */
/* bit-identical as long as the projections are.                      */
/* ------------------------------------------------------------------ */
static int gqa_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                    const uint8_t *tr, SaltKvCache *kv, int token,
                    float *q, float *k, float *v, float *attn_out) {
    int qi = tl->q3_q[L];
    int ki = tl->q3_k[L], vi = tl->q3_v[L], oi = tl->q3_o[L];
    int qrows, krows, vrows, orows, ocols;
    int heads, kv_heads, kh, rope_dim;
    long ocols_l;
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 ||
        salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0)
        return -1;
    ocols_l = salt_trunk_tensor_cols(&tl->t[oi]);
    if (ocols_l < 1 || ocols_l > INT_MAX ||
        salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                (int)ocols_l, &heads, &kv_heads, &kh,
                                &rope_dim) != 0)
        return -1;
    ocols = (int)ocols_l;
    if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens) return 0;
    int kvlat = krows + vrows;
    if (kv->kvlat != kvlat && kv->kvlat != 0) return 0;
    long scratch_need = salt_attn_gqa_scratch_floats(
        cfg, qrows, krows, vrows, orows, ocols, kv->max_tokens);
    if (scratch_need < 0 || !kv->scratch || kv->scratch_n < scratch_need)
        return -1;

    /* working slices live in kv->scratch (the shared arena), NOT
     * relative to attn_out: the chunk path passes an attn_out that
     * is a per-token row inside its own buffer, and the arena must
     * be stable across the B serial body calls. The layout matches
     * gqa_step's (xin..o unused here, but the same offsets keep the
     * serial path bit-identical). */
    float *buf = kv->scratch;
    float *q2 = buf + (size_t)cfg->hidden;              /* qrows */
    float *k2 = q2 + qrows;                             /* krows */
    float *v2 = k2 + krows;                             /* vrows */
    float *o2 = v2 + vrows;                             /* orows */
    float *a2 = o2 + orows;                             /* ocols */
    float *gate = a2 + ocols;
    float *qq = gate + (size_t)heads * kh;
    float *kk = qq + (size_t)heads * kh;
    float *cos_t = kk + (size_t)heads * kh;
    float *sin_t = cos_t + rope_dim;
    float *scores = sin_t + rope_dim;
    float *wgt = scores + (size_t)kv->max_tokens;
    memset(kk, 0, (size_t)heads * kh * sizeof(float));
    if (gqa_prep(cfg, tl, L, tr, kv, token, q, k, v, qq, gate,
                 cos_t, sin_t) != 0)
        return -1;

    /* attention: 16 q-heads over 2 kv-heads (repeat_kv 8), 0..token.
     * M8: threaded over the (b=1,h) pairs -- gqa_atn_spawn runs each
     * head's scores->softmax->v-sum->gate in the EXACT serial order,
     * so every attn_out[h] row is bit-identical to the serial loop.
     * The SALT_DEBUG_ATTN probe keeps the serial path (it prints
     * per-head state that the worker doesn't). */
    memset(attn_out, 0, (size_t)ocols * sizeof(float));
    if (getenv("SALT_DEBUG_ATTN")) {
        int npos = token + 1;
        float dscale = 1.0f / sqrtf((float)kh);
        for (int h = 0; h < heads; h++) {
            const float *qh_ptr = qq + (size_t)h * kh;
            int khh = h / (heads / kv_heads);
            float mx = -1e30f;
            for (int t2 = 0; t2 < npos; t2++) {
                const float *k2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat +
                    (size_t)khh * kh;
                float acc = 0.0f;
                for (int i = 0; i < kh; i++) acc += qh_ptr[i] * k2[i];
                scores[t2] = acc * dscale;
                if (scores[t2] > mx) mx = scores[t2];
            }
            if ((L == 3 || L == 7) && h == 0) {
                char afn[64], qfn[64];
                snprintf(afn, sizeof afn, "/tmp/q36-attn-L%d-h0.bin", L);
                snprintf(qfn, sizeof qfn, "/tmp/q36-q-L%d-h0.bin", L);
                FILE *af = fopen(afn, "wb");
                if (af) {
                    fwrite(scores, sizeof(float), (size_t)npos, af);
                    fclose(af);
                }
                FILE *qf = fopen(qfn, "wb");
                if (qf) {
                    fwrite(qh_ptr, sizeof(float), (size_t)kh, qf);
                    fclose(qf);
                }
            }
            float sum = 0.0f;
            for (int t2 = 0; t2 < npos; t2++) {
                wgt[t2] = salt_expf(scores[t2] - mx);
                sum += wgt[t2];
            }
            if (h == 0 && L == 3) {
                float wmax = 0.0f;
                int warg = -1;
                for (int t2 = 0; t2 < npos; t2++) {
                    wgt[t2] /= sum;
                    if (wgt[t2] > wmax) { wmax = wgt[t2]; warg = t2; }
                }
                fprintf(stderr, "[attn] L%d t%d npos=%d topw=%.3f at pos %d "
                        "score[0]=%.3f score[last]=%.3f\n", L, token, npos,
                        wmax, warg, scores[0], scores[npos - 1]);
            } else {
                for (int t2 = 0; t2 < npos; t2++) wgt[t2] /= sum;
            }
            for (int t2 = 0; t2 < npos; t2++) {
                const float *v2 = kv->kv +
                    ((size_t)L * kv->max_tokens + t2) * kvlat + krows +
                    (size_t)khh * kh;
                float w = wgt[t2];
                for (int i = 0; i < kh; i++)
                    attn_out[(size_t)h * kh + i] += w * v2[i];
            }
        }
        /* gate: attn_out *= sigmoid(gate) */
        for (int h = 0; h < heads; h++) {
            const float *gh = gate + (size_t)h * kh;
            float *ao = attn_out + (size_t)h * kh;
            for (int i = 0; i < kh; i++) {
                float gv = gh[i];
                ao[i] *= 1.0f / (1.0f + salt_expf(-gv));
            }
        }
    } else {
        gqa_atn_spawn(kv, qq, gate, attn_out, L, token, 1, heads,
                      kv_heads, kh, krows, vrows, ocols, kv->nthreads);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Linear attention: Gated DeltaNet (Qwen3-Next class)                 */
/* ------------------------------------------------------------------ */
/*
 * Reference: transformers modeling_qwen3_5.py Qwen3_5GatedDeltaNet.
 * Per layer, per token (decode, seq_len=1):
 *   qkv = in_proj_qkv(x)            [8192] = q(2048=16x128) k(2048) v(4096=32x128)
 *   qkv = silu(conv1d(qkv))         depthwise, kernel 4 (causal, pad 3)
 *   z   = in_proj_z(x)              [4096] = 32 value heads x 128
 *   b   = sigmoid(in_proj_b(x))     [32] beta per value head
 *   a   = in_proj_a(x)              [32]
 *   g   = -exp(A_log) * softplus(a + dt_bias)      [32] log-decay
 *   q,k = repeat_interleave(2)      (16 key heads -> 32 value heads)
 *   q   = l2norm(q)/sqrt(128); k = l2norm(k)
 *   state = state * exp(g)
 *   kv_mem = sum(state * k, dim=k)              (delta-rule correction)
 *   delta  = (v - kv_mem) * beta
 *   state = state + k (x) delta
 *   out   = sum(state * q, dim=k)
 *   out   = RMSNormGated(out, z)   = norm.weight * out * silu(z)
 *   out_proj(out) -> [2048], residual add
 *
 * The recurrent state per value head is [kd x vd] (128x128); 32 heads
 * x 2 MB/layer live in the kv cache's lin arena.
 */
#define Q3_CONV_K 4

/* The SERIAL body of the Gated DeltaNet layer: a/b gates ->
 * conv1d -> q/k norm -> delta-rule state update -> readout ->
 * RMSNormGated -> o_proj -> residual. Everything after the qkv+z
 * projections. This part MUST run per token (conv ring + delta state
 * are serial chains); the projections that feed it are batchable
 * (linear_step_chunk, the prefill path). Indices/geometry are
 * re-derived from tl/cfg/L. buf layout (same as linear_step):
 * [xin H][qkv qkv_rows][z z_rows][o o_rows][readout v_heads*vd][qk]. */
/* The linear-attention body for ONE token (the serial reference path).
 * Exported (not static) so the chunked delta kernel (deltachunk.c) can
 * call the EXACT serial math per token -- the B=1 degenerate anchor for
 * the M3 chunked formulation. Same signature/behavior as before. */
int salt_attn_lin_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                       const uint8_t *tr, SaltKvCache *kv, int token,
                       float *state, float *qkv, float *z, float *buf,
                       int skip_o) {
    int pi = tl->q3_pqkv[L];
    int zi = tl->q3_pz[L];
    int ai = tl->q3_pa[L], as_ = tl->q3_pas[L], ab = tl->q3_pab[L];
    int bi = tl->q3_pb[L], bs_ = tl->q3_pbs[L], bb = tl->q3_pbb[L];
    int ci = tl->q3_conv[L];
    int oi = tl->q3_opa[L], os_ = tl->q3_opas[L], ob = tl->q3_opab[L];
    int ni = tl->q3_lnorm[L];
    int ai_ = tl->q3_a_log[L], di = tl->q3_dt[L];
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];
    int z_rows = (int)tl->t[zi].dims[0];
    int o_rows = (int)tl->t[oi].dims[0];
    long model_cols = salt_trunk_tensor_cols(&tl->t[pi]);
    int cols = H;
    int k_heads, v_heads, kd, vd, expected_qkv;
    if (salt_model_linear_geometry(cfg, &k_heads, &v_heads, &kd, &vd,
                                   &expected_qkv) != 0 ||
        qkv_rows != expected_qkv || z_rows != v_heads * vd ||
        model_cols != H || o_rows != H)
        return -1;
    float *xin = buf;
    float *o = z + z_rows;
    float *readout = o + o_rows;
    float *conv_ring = kv->conv + (size_t)L * 4 * qkv_rows;
    /* Work arena after readout: a, b, reusable tmp/g, then one delta row
     * per attention worker. The sizing helper shared by all callers owns
     * this layout. */
    float *a32 = readout + (size_t)v_heads * vd;
    float *b32 = a32 + v_heads;
    float *g32 = b32 + v_heads;
    float *delta = g32 + v_heads;
    double _l0 = getenv("SALT_LIN_MS") ? now_s() : 0;
    double _l2 = 0, _l3 = 0, _l4 = 0;
    {
        if (q4_proj(tl, ai, as_, ab, tr, v_heads, cols, xin, g32) != 0 ||
            q4_proj(tl, bi, bs_, bb, tr, v_heads, cols, xin, a32) != 0) {
            fprintf(stderr, "[linbody] L%d a/b proj FAIL ai=%d bi=%d cols=%d\n", L, ai, bi, cols);
            return -1;
        }
        memcpy(b32, a32, (size_t)v_heads * sizeof(float));
        memcpy(a32, g32, (size_t)v_heads * sizeof(float));
    }
    double _l1 = getenv("SALT_LIN_MS") ? now_s() : 0;
    /* causal conv1d (depthwise, kernel 4) with SILU activation. Weight
     * [8192, 4, 1] BF16, layout [channel][k][1]. Ring holds the past
     * Q3_CONV_K qkv vectors. The reference's causal_conv1d_update
     * concatenates [x_{t-3}, x_{t-2}, x_{t-1}, x_t] and convolves with
     * w[0..3] -- so w[0] is the OLDEST tap and w[3] the newest.
     * conv(x) = silu(sum_k w[k]*x[t-3+k]) */
    {
        const uint16_t *cw = (const uint16_t *)(const void *)(tr +
                              tl->t[ci].off);
        int base = (token % Q3_CONV_K) * qkv_rows;
        memcpy(conv_ring + base, qkv, (size_t)qkv_rows * sizeof(float));
        if (getenv("SALT_NAN_PROBE") && L == 0 && token == 0) {
            FILE *pf = fopen("/tmp/q36-eng-qkvpre.bin", "wb");
            if (pf) {
                fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
                fclose(pf);
            }
            FILE *xf = fopen("/tmp/q36-eng-xin-attn.bin", "wb");
            if (xf) {
                fwrite(xin, sizeof(float), (size_t)H, xf);
                fclose(xf);
            }
            fprintf(stderr, "[convw] ci=%d off=%zu cw[0..7]=%.6g %.6g %.6g %.6g "
                    "%.6g %.6g %.6g %.6g\n", ci,
                    (size_t)tl->t[ci].off,
                    bf16_f(cw[0]), bf16_f(cw[1]), bf16_f(cw[2]), bf16_f(cw[3]),
                    bf16_f(cw[4]), bf16_f(cw[5]), bf16_f(cw[6]), bf16_f(cw[7]));
        }
    _l2 = getenv("SALT_LIN_MS") ? now_s() : 0;
        for (int ch = 0; ch < qkv_rows; ch++) {
            float acc = 0.0f;
            for (int k = 0; k < Q3_CONV_K; k++) {
                int tpos = token - (Q3_CONV_K - 1 - k);   /* oldest..newest */
                if (tpos < 0) continue;
                int slot = (tpos % Q3_CONV_K) * qkv_rows + ch;
                float w = bf16_f(cw[(size_t)ch * Q3_CONV_K + k]);
                acc += conv_ring[slot] * w;
            }
            float sig = 1.0f / (1.0f + salt_expf(-acc));
            qkv[ch] = acc * sig;             /* silu */
        }
        if (getenv("SALT_NAN_PROBE") && L == 0 && token == 8 &&
            v_heads * vd >= 4) {
            double q2 = 0, k2 = 0, v2 = 0;
            int kspan = k_heads * kd;
            for (int i = 0; i < kspan; i++) q2 += (double)qkv[i] * qkv[i];
            for (int i = kspan; i < 2 * kspan; i++)
                k2 += (double)qkv[i] * qkv[i];
            for (int i = 2 * kspan; i < qkv_rows; i++)
                v2 += (double)qkv[i] * qkv[i];
            fprintf(stderr, "[postconv] L0 t8 q-rms %.6g k-rms %.6g "
                    "v-rms %.6g v[0..3] %.6g %.6g %.6g %.6g\n",
                    sqrt(q2 / kspan), sqrt(k2 / kspan),
                    sqrt(v2 / (v_heads * vd)), qkv[2 * kspan],
                    qkv[2 * kspan + 1], qkv[2 * kspan + 2],
                    qkv[2 * kspan + 3]);
        }
        if (getenv("SALT_NAN_PROBE") && L == 1 && token == 21) {
            FILE *qf = fopen("/tmp/q36-L1-qkv-t21.bin", "wb");
            if (qf) {
                fwrite(qkv, sizeof(float), (size_t)qkv_rows, qf);
                fclose(qf);
            }
        }
    }
    /* q/k normalization, EXACTLY the reference (qwen3_5.py
     * Qwen3_5MoeAttention.forward, use_qk_l2norm_in_kernel=True):
     *   l2norm(x) = x / sqrt(sum(x^2) + 1e-6)     (SUM, FLA-style)
     *   query = l2norm(query) * (1/sqrt(kd))
     *   key   = l2norm(key)                        (NO extra scale)
     * The previous code used a MEAN-based rmsnorm and swapped the
     * scales (q got 1/kd, k got 1/sqrt(kd)) -- wrong math that
     * distorted the q/k magnitudes entering the delta rule. */
    {
        for (int h = 0; h < k_heads; h++) {
            float *qh = qkv + (size_t)h * kd;
            float *kh = qkv + (size_t)(k_heads * kd) + (size_t)h * kd;
            for (int which = 0; which < 2; which++) {
                float *vec = which ? kh : qh;
                double ss = 0.0;
                for (int i = 0; i < kd; i++)
                    ss += (double)vec[i] * vec[i];
                float inv = (float)(1.0 / sqrt(ss + 1e-6));
                if (!which) inv *= (float)(1.0 / sqrt((double)kd));
                for (int i = 0; i < kd; i++)
                    vec[i] = vec[i] * inv;
            }
        }
    }
    /* log-decay per value head: g = -exp(A_log) * softplus(a + dt_bias).
     * A_log's dtype follows the checkpoint: F32 on the earlier fixture,
     * BF16 on the 3.6 MLX repo (64 B vs 128 B -- reading the wrong
     * width past the tensor end decodes garbage -> NaN decay -> the
     * delta rule freezes and every token collapses to the same id). */
    {
        const SaltTrunkTensor *at = &tl->t[ai_];
        const uint16_t *dtb = (const uint16_t *)(const void *)(tr +
                              tl->t[di].off);
        for (int h = 0; h < v_heads; h++) {
            float Al;
            if (at->dtype == 4) {  /* BF16 */
                const uint16_t *alb = (const uint16_t *)(const void *)
                                      (tr + at->off);
                Al = bf16_f(alb[h]);
            } else {               /* F32 */
                const float *Alf = (const float *)(const void *)
                                   (tr + at->off);
                Al = Alf[h];
            }
            float dt = bf16_f(dtb[h]);
            float sp = (a32[h] + dt) > 0
                ? (a32[h] + dt) + salt_log1pf(salt_expf(-(a32[h] + dt)))
                : salt_log1pf(salt_expf(a32[h] + dt));
            g32[h] = -salt_expf(Al) * sp;
        }
        if (getenv("SALT_NAN_PROBE") && L == 1 && v_heads >= 28 &&
            getenv("SALT_DUMP_STATE")) {
            fprintf(stderr, "[decay-all] L%d heads 24-27: "
                    "decay %.6g %.6g %.6g %.6g beta %.6g %.6g %.6g %.6g\n",
                    L, salt_expf(g32[24]), salt_expf(g32[25]),
                    salt_expf(g32[26]), salt_expf(g32[27]),
                    b32[24], b32[25], b32[26], b32[27]);
        }
        if (getenv("SALT_NAN_PROBE") && L == 5) {
            fprintf(stderr, "[decay] L%d a[0..3]=[%.4g %.4g %.4g %.4g] "
                    "dt[0..3]=[%.4g %.4g %.4g %.4g] "
                    "g[0..3]=[%.4g %.4g %.4g %.4g] expg=%.4g\n",
                    L, a32[0], a32[1], a32[2], a32[3],
                    bf16_f(dtb[0]), bf16_f(dtb[1]), bf16_f(dtb[2]),
                    bf16_f(dtb[3]),
                    g32[0], g32[1], g32[2], g32[3], salt_expf(g32[0]));
        }
    }
    /* delta-rule state update + readout (the FLA recurrent rule):
     *   state = state * exp(g); kv_mem = sum_k(state * k)
     *   delta = (v - kv_mem) * beta; state += k (x) delta
     *   out = sum_k(state * q)
     * Value heads map evenly onto key heads (repeat_interleave ratio
     * comes from model-owned geometry).
     * M9: the 32 value heads are fully independent (disjoint state
     * slices, disjoint readout rows) -> threaded by head range via
     * lin_delta_spawn, every head in the EXACT serial order
     * (bit-identical). The SALT_NAN_PROBE path stays serial (it
     * dumps per-head state). */
    float *S = kv->lin + (size_t)L * v_heads * kd * vd;
    memset(readout, 0, (size_t)v_heads * vd * sizeof(float));
    if (getenv("SALT_NAN_PROBE")) {
        double sm = 0.0;
        for (int i = 0; i < kd * vd; i++)
            if (S[i] == S[i]) sm += (double)S[i] * S[i];
        if (L % 8 == 0 && token == 6)
            fprintf(stderr, "[lin] L%d pre-state rms %.6g\n", L, sqrt(sm));
        if (L == 0 && (token == 6 || token == 12 || token == 18 ||
                       token == 20 || token == 21)) {
            char pth[128];
            snprintf(pth, sizeof pth,
                     "/tmp/q36-eng-L0-state-t%d.bin", token);
            FILE *sf = fopen(pth, "wb");
            if (sf) {
                fwrite(S, sizeof(float), (size_t)v_heads * kd * vd, sf);
                fclose(sf);
            }
        }
        if (L <= 2 && (token == 6 || token == 12 || token == 21)) {
            char pth[128];
            snprintf(pth, sizeof pth, "/tmp/q36-delta-L%d-t%d.bin", L, token);
            FILE *sf = fopen(pth, "wb");
            if (sf) {
                fwrite(S, sizeof(float), (size_t)v_heads * kd * vd, sf);
                fclose(sf);
            }
        }
    }
    if (getenv("SALT_NAN_PROBE")) {
        /* serial path (debug dumps per-head state) */
#ifdef __aarch64__
    if (salt_kernels_simd() && kd % 4 == 0 && vd % 4 == 0) {
        /* NEON delta rule: 4-wide over vd. Each output element keeps
         * the SAME i-accumulation order as the scalar loop, and the
         * vmlaq FMA matches -O2 fp-contract=on, so results are
         * bit-identical to the scalar path -- verified by e2e. */
        for (int h = 0; h < v_heads; h++) {
            int khh = h / (v_heads / k_heads);
            const float *kh = qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
            const float *qh = qkv + (size_t)khh * kd;
            const float *vh = qkv + (size_t)(2 * k_heads * kd) +
                              (size_t)h * vd;
            float *Sh = S + (size_t)h * kd * vd;
            float beta = 1.0f / (1.0f + salt_expf(-b32[h]));
            float decay = salt_expf(g32[h]);
            float32x4_t dcy = vdupq_n_f32(decay);
            /* state *= decay (elementwise, 4-wide) */
            for (int i = 0; i < kd * vd; i += 4) {
                float32x4_t s = vld1q_f32(Sh + i);
                vst1q_f32(Sh + i, vmulq_f32(s, dcy));
            }
            /* kv_mem = sum_k(state * k) -> [vd]; delta = (v - kv_mem)*beta */
            for (int j = 0; j < vd; j += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < kd; i++) {
                    float32x4_t s = vld1q_f32(Sh + (size_t)i * vd + j);
                    acc = vaddq_f32(acc, vmulq_f32(s, vdupq_n_f32(kh[i])));
                }
                float32x4_t vv = vld1q_f32(vh + j);
                float32x4_t dl = vmulq_f32(vsubq_f32(vv, acc),
                                           vdupq_n_f32(beta));
                vst1q_f32(delta + j, dl);
            }
            /* state += k (x) delta (row-wise, 4-wide) */
            for (int i = 0; i < kd; i++) {
                float32x4_t kk = vdupq_n_f32(kh[i]);
                float *row = Sh + (size_t)i * vd;
                for (int j = 0; j < vd; j += 4) {
                    float32x4_t s = vld1q_f32(row + j);
                    float32x4_t d = vld1q_f32(delta + j);
                    vst1q_f32(row + j, vaddq_f32(s, vmulq_f32(kk, d)));
                }
            }
            /* readout: out_h = sum_k(state * q) */
            float *oh = readout + (size_t)h * vd;
            for (int j = 0; j < vd; j += 4) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int i = 0; i < kd; i++) {
                    float32x4_t s = vld1q_f32(Sh + (size_t)i * vd + j);
                    acc = vaddq_f32(acc, vmulq_f32(s, vdupq_n_f32(qh[i])));
                }
                vst1q_f32(oh + j, acc);
            }
            if (getenv("SALT_NAN_PROBE") && L <= 2 && token >= 18 &&
                h == 0) {
                double sm = 0.0;
                for (int i = 0; i < kd * vd; i++)
                    if (Sh[i] == Sh[i]) sm += (double)Sh[i] * Sh[i];
                double k2 = 0.0, v2 = 0.0, d2 = 0.0, q2 = 0.0;
                for (int i = 0; i < kd; i++) {
                    k2 += (double)kh[i] * kh[i];
                    q2 += (double)qh[i] * qh[i];
                }
                for (int i = 0; i < vd; i++) v2 += (double)vh[i] * vh[i];
                for (int i = 0; i < vd; i++) d2 += (double)delta[i] * delta[i];
                fprintf(stderr, "[delta] L%d h0 decay %.6g beta %.6g k-rms %.6g "
                        "q-rms %.6g v-rms %.6g delta-rms %.6g state-rms %.6g\n",
                        L, decay, beta, sqrt(k2 / kd), sqrt(q2 / kd),
                        sqrt(v2 / vd), sqrt(d2 / vd), sqrt(sm / (kd * vd)));
            }
        }
    } else
#endif
    {
    for (int h = 0; h < v_heads; h++) {
        int khh = h / (v_heads / k_heads);
        const float *kh = qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
        const float *qh = qkv + (size_t)khh * kd;
        const float *vh = qkv + (size_t)(2 * k_heads * kd) +
                          (size_t)h * vd;
        float *Sh = S + (size_t)h * kd * vd;
        float beta = 1.0f / (1.0f + salt_expf(-b32[h]));   /* sigmoid */
        float decay = salt_expf(g32[h]);
        /* state *= decay */
        for (int i = 0; i < kd * vd; i++) Sh[i] *= decay;
        /* kv_mem = sum_k(state * k) -> [vd]; delta = (v - kv_mem)*beta */
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += Sh[(size_t)i * vd + j] * kh[i];
            delta[j] = (vh[j] - acc) * beta;
        }
        /* state += k (x) delta */
        for (int i = 0; i < kd; i++) {
            float *row = Sh + (size_t)i * vd;
            float kk = kh[i];
            for (int j = 0; j < vd; j++)
                row[j] += kk * delta[j];
        }
    _l3 = getenv("SALT_LIN_MS") ? now_s() : 0;
        /* readout: out_h = sum_k(state * q) */
        float *oh = readout + (size_t)h * vd;
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += Sh[(size_t)i * vd + j] * qh[i];
            oh[j] = acc;
        }
        if (getenv("SALT_NAN_PROBE") && L == 0 && token == 8 && h == 0) {
            double sm = 0.0;
            for (int i = 0; i < kd * vd; i++)
                if (Sh[i] == Sh[i]) sm += (double)Sh[i] * Sh[i];
            double k2 = 0.0, v2 = 0.0, d2 = 0.0, q2 = 0.0;
            for (int i = 0; i < kd; i++) {
                k2 += (double)kh[i] * kh[i];
                q2 += (double)qh[i] * qh[i];
            }
            for (int i = 0; i < vd; i++) v2 += (double)vh[i] * vh[i];
            for (int i = 0; i < vd; i++) d2 += (double)delta[i] * delta[i];
            fprintf(stderr, "[delta] L%d h0 decay %.6g beta %.6g k-rms %.6g "
                    "q-rms %.6g v-rms %.6g delta-rms %.6g state-rms %.6g\n",
                    L, decay, beta, sqrt(k2 / kd), sqrt(q2 / kd),
                    sqrt(v2 / vd), sqrt(d2 / vd), sqrt(sm / (kd * vd)));
            FILE *vf = fopen("/tmp/q36-eng-L0-k.bin", "wb");
            if (vf) { fwrite(kh, sizeof(float), (size_t)kd, vf); fclose(vf); }
            vf = fopen("/tmp/q36-eng-L0-q.bin", "wb");
            if (vf) { fwrite(qh, sizeof(float), (size_t)kd, vf); fclose(vf); }
            vf = fopen("/tmp/q36-eng-L0-v.bin", "wb");
            if (vf) { fwrite(vh, sizeof(float), (size_t)vd, vf); fclose(vf); }
            vf = fopen("/tmp/q36-eng-L0-delta.bin", "wb");
            if (vf) { fwrite(delta, sizeof(float), (size_t)vd, vf); fclose(vf); }
            vf = fopen("/tmp/q36-eng-L0-S.bin", "wb");
            if (vf) { fwrite(Sh, sizeof(float), (size_t)kd * vd, vf); fclose(vf); }
        }
    }
    }
    } else {
        /* M9 threaded path: independent value heads across workers, each in
         * the exact serial order (bit-identical to the loop above) */
    _l3 = getenv("SALT_LIN_MS") ? now_s() : 0;
        lin_delta_spawn(kv, qkv, readout, b32, g32, delta, L, token,
                        k_heads, v_heads, kd, vd,
                        salt_kernels_simd() && kd % 4 == 0 && vd % 4 == 0,
                        kv->nthreads);
    }
    /* RMSNormGated: norm.weight * out * silu(z) -- the learned norm is
     * on the OUTPUT, gated by silu(z). The norm is PER-HEAD (each
     * value-head slice is normalized independently), matching the
     * reference's mean over the last dim of [N, head_v_dim]. */
    _l4 = getenv("SALT_LIN_MS") ? now_s() : 0;
    {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[ni].off);
        for (int h = 0; h < v_heads; h++) {
            float *oh = readout + (size_t)h * vd;
            float *zh = z + (size_t)h * vd;
            double ss = 0.0;
            for (int i = 0; i < vd; i++)
                ss += (double)oh[i] * oh[i];
            float r = sqrtf((float)(ss / (double)vd) + 1e-6f);
            if (getenv("SALT_NAN_PROBE") && L == 0 && token == 0 && h == 0)
                fprintf(stderr, "[norm] L0 t0 h0 pre-r=%.6g first=%.6g\n",
                        r, oh[0]);
            if (getenv("SALT_NAN_PROBE") && L == 0 && token == 21 &&
                getenv("SALT_DUMP_Z")) {
                FILE *rf = fopen("/tmp/q36-eng-readout.bin", "wb");
                if (rf) {
                    fwrite(readout, sizeof(float),
                           (size_t)v_heads * vd, rf);
                    fclose(rf);
                }
                FILE *zf = fopen("/tmp/q36-eng-z.bin", "wb");
                if (zf) {
                    fwrite(z, sizeof(float),
                           (size_t)v_heads * vd, zf);
                    fclose(zf);
                }
            }
            for (int i = 0; i < vd; i++) {
                float zv = zh[i];
                float sig = 1.0f / (1.0f + salt_expf(-zv));
                oh[i] = oh[i] / r * bf16_f(nw[i]) * (zv * sig);
            }
            if (getenv("SALT_NAN_PROBE") && L == 0 && token == 0 && h == 0) {
                double hn = 0.0;
                for (int i = 0; i < vd; i++)
                    hn += (double)oh[i] * oh[i];
                fprintf(stderr, "[norm] h0 post-rms %.6g oh[0..3] %.6g %.6g "
                        "%.6g %.6g\n", sqrt(hn / vd), oh[0], oh[1], oh[2],
                        oh[3]);
                fprintf(stderr, "[norm] h0 zh[0..3] %.6g %.6g %.6g %.6g "
                        "zh[64..67] %.6g %.6g %.6g %.6g zh[124..127] %.6g "
                        "%.6g %.6g %.6g\n", zh[0], zh[1], zh[2], zh[3],
                        zh[64], zh[65], zh[66], zh[67],
                        zh[124], zh[125], zh[126], zh[127]);
            }
        }
    }
    /* out_proj(readout) -> o, residual add. When skip_o is set (the
     * chunk path), the o_proj is deferred and batched over the chunk
     * (M4: same 8.4 MB tensor, B readouts); the residual is also
     * deferred so the caller can add after the batched o. */
    double _to = 0;
    if (getenv("SALT_PROJ_MS")) _to = now_s();
    if (!skip_o &&
        q4_proj(tl, oi, os_, ob, tr, o_rows, v_heads * vd, readout, o)
        != 0) {
        fprintf(stderr, "[linbody] L%d o proj FAIL oi=%d os_=%d\n", L, oi, os_);
        return -1;
    }
    if (getenv("SALT_NAN_PROBE") && L == 0 && token == 21) {
        FILE *of = fopen("/tmp/q36-eng-o.bin", "wb");
        if (of) {
            fwrite(o, sizeof(float), (size_t)o_rows, of);
            fclose(of);
        }
    }
    if (getenv("SALT_PROJ_MS")) {
        static double _oacc = 0; static long _ocnt = 0;
        _oacc += now_s() - _to; _ocnt++;
        if (_ocnt <= 3 || _ocnt % 100 == 0)
            fprintf(stderr, "[proj-o] L%d o: %.2f ms (avg %.2f)\n",
                    L, (now_s() - _to) * 1e3, _oacc / _ocnt * 1e3);
    }
    if (getenv("SALT_NAN_PROBE") && L == 0 &&
        (token == 0 || token == 1)) {
        double r2 = 0.0, z2 = 0.0, o2 = 0.0, q2 = 0.0;
        for (int i = 0; i < v_heads * vd; i++) {
            r2 += (double)readout[i] * readout[i];
            z2 += (double)z[i] * z[i];
        }
        for (int i = 0; i < H; i++) o2 += (double)o[i] * o[i];
        for (int i = 0; i < k_heads * kd; i++) q2 += (double)qkv[i] * qkv[i];
        if (getenv("SALT_DUMP_Z")) {
            char qfn[128];
            snprintf(qfn, sizeof qfn, "/tmp/q36-eng-qkv-t%d.bin", token);
            FILE *zf = fopen(qfn, "wb");
            if (zf) {
                fwrite(qkv, sizeof(float), (size_t)qkv_rows, zf);
                fclose(zf);
            }
            FILE *bf = fopen("/tmp/q36-eng-ab.bin", "wb");
            if (bf) {
                fwrite(a32, sizeof(float), (size_t)v_heads, bf);
                fwrite(b32, sizeof(float), (size_t)v_heads, bf);
                fclose(bf);
            }
            if (token == 1) {
                FILE *sf = fopen("/tmp/q36-eng-state-t1.bin", "wb");
                if (sf) {
                    fwrite(state, sizeof(float), (size_t)H, sf);
                    fclose(sf);
                }
            }
        }
        /* per-head r of head 0 */
        double h0 = 0.0;
        for (int i = 0; i < vd; i++) h0 += (double)readout[i] * readout[i];
        fprintf(stderr, "[linout] L%d t%d readout-rms %.6g z-rms %.6g "
                "o-rms %.6g q-pre rms %.6g head0-r %.6g nw0..3 %.4g %.4g %.4g %.4g "
                "z[0..3] %.4g %.4g %.4g %.4g xin[0..3] %.4g %.4g %.4g %.4g\n",
                L, token,
                sqrt(r2 / (v_heads * vd)), sqrt(z2 / (v_heads * vd)),
                sqrt(o2 / H), sqrt(q2 / (k_heads * kd)),
                sqrt(h0 / (double)vd),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[0]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[1]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[2]),
                bf16_f(((const uint16_t *)(const void *)(tr + tl->t[ni].off))[3]),
                z[0], z[1], z[2], z[3],
                xin[0], xin[1], xin[2], xin[3]);
    }
    double _l5 = getenv("SALT_LIN_MS") ? now_s() : 0;
    if (!skip_o)
        for (int i = 0; i < H; i++) state[i] += o[i];
    if (getenv("SALT_LIN_MS"))
        fprintf(stderr, "[linms] L%d ab=%.2f conv=%.2f delta=%.2f "
                "ro=%.2f o=%.2f ms\n", L,
                (_l1-_l0)*1e3, (_l2-_l1)*1e3, (_l3-_l2)*1e3,
                (_l4-_l3)*1e3, (_l5-_l4)*1e3);
    return 0;
}

static int linear_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                       const uint8_t *tr, float *state, SaltKvCache *kv,
                       int token) {
    int pi = tl->q3_pqkv[L], ps = tl->q3_pqkvs[L], pb = tl->q3_pqkvb[L];
    int zi = tl->q3_pz[L], zs = tl->q3_pzs[L], zb = tl->q3_pzb[L];
    int ai = tl->q3_pa[L];
    int bi = tl->q3_pb[L];
    int ci = tl->q3_conv[L];
    int oi = tl->q3_opa[L];
    int ni = tl->q3_lnorm[L];
    int ai_ = tl->q3_a_log[L], di = tl->q3_dt[L];
    int iln = tl->attn_norm[L];
    if (pi < 0 || zi < 0 || ai < 0 || bi < 0 || ci < 0 || oi < 0 ||
        ni < 0 || ai_ < 0 || di < 0) {
        if (getenv("SALT_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d SKIP pi=%d zi=%d ai=%d bi=%d "
                    "ci=%d oi=%d ni=%d Al=%d dt=%d\n", L, pi, zi, ai, bi,
                    ci, oi, ni, ai_, di);
        return 0;                    /* incomplete graph: skip */
    }
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];      /* 8192 */
    int z_rows = (int)tl->t[zi].dims[0];        /* 4096 */
    int o_rows = (int)tl->t[oi].dims[0];        /* 2048 */
    long model_cols = salt_trunk_tensor_cols(&tl->t[pi]);
    int cols = H;
    /* Linear-attention geometry is model-owned config data. */
    int k_heads, v_heads, kd, vd, expected_qkv;
    if (salt_model_linear_geometry(cfg, &k_heads, &v_heads, &kd, &vd,
                                   &expected_qkv) != 0) {
        fprintf(stderr, "qwen lin: invalid model geometry\n");
        return -1;
    }
    if (qkv_rows != expected_qkv) {
        if (getenv("SALT_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d GEOM qkv_rows=%d want=%d\n", L,
                    qkv_rows, expected_qkv);
        fprintf(stderr, "qwen lin: L%d qkv %d != k %d*%d*2 + v %d*%d\n",
                L, qkv_rows, k_heads, kd, v_heads, vd);
        return -1;
    }
    if (model_cols != H || z_rows != v_heads * vd || o_rows != H) {
        if (getenv("SALT_NAN_PROBE") && L < 3)
            fprintf(stderr, "[lin] L%d GEOM cols=%ld z_rows=%d o_rows=%d\n",
                    L, model_cols, z_rows, o_rows);
        return -1;
    }
    if (getenv("SALT_NAN_PROBE") && L < 3)
        fprintf(stderr, "[lin] L%d enter ok dtype_qkv=%d dtype_z=%d "
                "pi=%d zi=%d qkv_rows=%d z_rows=%d\n",
                L, tl->t[pi].dtype, tl->t[zi].dtype, pi, zi,
                qkv_rows, z_rows);
    if (!kv || token < 0 || token >= kv->max_tokens) return 0;
    if (!kv->lin_alloc) {
        if (salt_kv_lin_init(kv, v_heads, kd, vd) != 0) {
            fprintf(stderr, "[lin] L%d kv_lin_init FAILED vh=%d kd=%d vd=%d\n", L, v_heads, kd, vd);
            return -1;
        }
    }
    if (!kv->conv_alloc) {
        if (salt_kv_conv_init(kv, qkv_rows) != 0) {
            fprintf(stderr, "[lin] L%d kv_conv_init FAILED qkv_rows=%d\n", L, qkv_rows);
            return -1;
        }
    }
    float *buf = kv->scratch;   /* arena, sized at init (never per-call) */
    float *xin = buf;               /* input_layernorm(state) */
    float *qkv = xin + H;
    float *z = qkv + qkv_rows;
    /* NOTE: linear_step and gqa_step share kv->scratch. They are called
     * sequentially (one per layer), so reuse is safe. */
    memset(buf, 0, (size_t)(H + qkv_rows + z_rows + o_rows +
                 v_heads * vd + v_heads * kd + 2 * H + 1) *
                 sizeof(float));

    /* reference: x = x + GatedDeltaNet(input_layernorm(x)) */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[iln].off);
        memcpy(xin, state, (size_t)H * sizeof(float));
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
        float r = sqrtf((float)(ss / (double)H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            uint32_t bits = (uint32_t)nw[i] << 16;
            float w;
            memcpy(&w, &bits, 4);
            xin[i] = xin[i] / r * w;
        }
    } else {
        memcpy(xin, state, (size_t)H * sizeof(float));
    }

    double _tproj = 0;
    if (getenv("SALT_PROJ_MS")) _tproj = now_s();
    if (tl->t[pi].dtype == 2 || tl->t[zi].dtype == 2 ||
        tl->t[pi].bits == 8 || tl->t[zi].bits == 8) {
        /* F8_E4M3 serial path: the fused q4 matvec2 kernel is 4-bit-only
         * and would read F8 bytes as U32 nibbles (garbage -> NaN).
         * MLX 8-bit (U32 bits==8): the fused kernel is ALSO 4-bit-only
         * and nibble-decodes the 8-bit qkv -- same class of bug. Both
         * go through the dtype-aware per-tensor projections. */
        if (q4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, xin, qkv) != 0 ||
            q4_proj(tl, zi, zs, zb, tr, z_rows, cols, xin, z) != 0) {
            fprintf(stderr, "[lin] L%d f8 proj FAILED pi=%d zi=%d cols=%d\n",
                    L, pi, zi, cols);
            return -1;
        }
    } else if (q4_matvec2_decode(kv,
            (const uint32_t *)(const void *)(tr + tl->t[pi].off),
            (const uint16_t *)(const void *)(tr + tl->t[ps].off),
            pb >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[pb].off)
                    : NULL, qkv_rows,
            (const uint32_t *)(const void *)(tr + tl->t[zi].off),
            (const uint16_t *)(const void *)(tr + tl->t[zs].off),
            zb >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[zb].off)
                    : NULL, z_rows,
            cols, xin, qkv, z) != 0) {
        fprintf(stderr, "[lin] L%d matvec2 FAILED, fallback\n", L);
        if (q4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, xin, qkv) != 0 ||
            q4_proj(tl, zi, zs, zb, tr, z_rows, cols, xin, z) != 0) {
            fprintf(stderr, "[lin] L%d fallback proj FAILED pi=%d zi=%d cols=%d\n", L, pi, zi, cols);
            return -1;
        }
    }
    if (getenv("SALT_PROJ_MS")) {
        static double _pacc = 0; static long _pcnt = 0;
        _pacc += now_s() - _tproj; _pcnt++;
        if (_pcnt <= 3 || _pcnt % 100 == 0)
            fprintf(stderr, "[proj] L%d qkv+z: %.2f ms (avg %.2f)\n",
                    L, (now_s() - _tproj) * 1e3, _pacc / _pcnt * 1e3);
    }
    if (getenv("SALT_PROJ_MS") && L == 3 && token == 8) {
        fprintf(stderr, "[qkvd] serial dump L3 t8 firing\n");
        FILE *pf = fopen("/tmp/qkv-serial-L3.bin", "wb");
        if (pf) {
            fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
            fwrite(z, sizeof(float), (size_t)z_rows, pf);
            fclose(pf);
        }
    }
    return salt_attn_lin_body(cfg, tl, L, tr, kv, token, state, qkv, z, buf, 0);

    return 0;
}

/* CHUNKED prefill variant of linear_step (M1 of prefill-batch):
 * processes B prompt tokens at layer L together. The qkv+z
 * projections (the batchable part) run ONCE over the B x-vectors
 * with the dequant amortized; the serial chain (conv1d ring + delta
 * state) runs per token via lin_body, in token order, so results are
 * bit-identical to B sequential linear_step calls. states[B][H] in,
 * residual-added out. Returns 0 ok, -1 fail (caller falls back). */
int salt_attn_linear_chunk(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                             int L, const uint8_t *tr,
                             float *const *states, int t0, int B,
                             SaltKvCache *kv) {
    int pi = tl->q3_pqkv[L], ps = tl->q3_pqkvs[L], pb = tl->q3_pqkvb[L];
    int zi = tl->q3_pz[L], zs = tl->q3_pzs[L], zb = tl->q3_pzb[L];
    int iln = tl->attn_norm[L];
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];      /* 8192 */
    int z_rows = (int)tl->t[zi].dims[0];        /* 4096 */
    long model_cols = salt_trunk_tensor_cols(&tl->t[pi]);
    int cols = H;
    if (B < 1) return 0;
    if (!kv) return -1;
    int k_heads, v_heads, kd, vd, expected_qkv;
    if (model_cols != H ||
        salt_model_linear_geometry(cfg, &k_heads, &v_heads, &kd, &vd,
                                   &expected_qkv) != 0 ||
        qkv_rows != expected_qkv || z_rows != v_heads * vd)
        return -1;
    double _tch = now_s();
    static double _chk_acc[4] = {0,0,0,0}; static long _chk_n = 0;
    if (!kv->lin_alloc)
        if (salt_kv_lin_init(kv, v_heads, kd, vd) != 0)
            return -1;
    if (!kv->conv_alloc)
        if (salt_kv_conv_init(kv, qkv_rows) != 0) return -1;

    /* chunk scratch: per token xin[H] + qkv + z, plus the shared
     * per-token body buffer (lin_body's buf layout:
     * [xin H][qkv qkv_rows][z z_rows][o o_rows][readout v_heads*vd]
     * [qk]). B=64 -> ~5.5 MB total, negligible. The batch matvec
     * takes xins as ROW-MAJOR [B][H] (per-token rows contiguous --
     * the layout the bit-identical SIMD FMA loop needs); qkvs/zbtok
     * are [B][rows] row-major for the serial lin_body feed. */
    long per = (long)H + qkv_rows + z_rows;
    Q4ChunkBuffer xs_storage;
    int indexed_mix = q4_indexed_mix_enabled();
    if (per < 1 || (size_t)B > SIZE_MAX / (size_t)per / sizeof(float) ||
        q4_chunk_buffer_alloc(&xs_storage,
            (size_t)B * (size_t)per * sizeof(float), indexed_mix) != 0)
        return -1;
    float *xs = xs_storage.data;
    float *xins = xs;                       /* [B][H] row-major */
    float *qkvs = xins + (size_t)B * H;     /* [B][qkv_rows] row-major */
    float *zbtok = qkvs + (size_t)B * qkv_rows;  /* [B][z_rows] */
    int o_rows = (int)tl->t[tl->q3_opa[L]].dims[0];   /* 2048 = H */
    long bneed = salt_attn_linear_scratch_floats(
        cfg, H, qkv_rows, z_rows, o_rows, kv->nthreads);
    if (bneed < 0) { q4_chunk_buffer_release(&xs_storage); return -1; }
    float *bbuf = (float *)malloc((size_t)bneed * sizeof(float));
    if (!bbuf) { q4_chunk_buffer_release(&xs_storage); return -1; }
    /* zero the WHOLE body buffer like the serial path's
     * memset(buf, 0, ...): lin_body reads regions it did not write
     * (the o/readout/qk slack), and garbage there is a structural
     * divergence -- the chunk fill diverged from serial at L3 for
     * exactly this reason (2048/2048 state floats differed). */
    memset(bbuf, 0, (size_t)bneed * sizeof(float));

    /* 1) per-token input_layernorm into xins[B][H] (row-major) */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                              tl->t[iln].off);
        for (int b = 0; b < B; b++) {
            const float *st = states[b];
            float *xin = xins + (size_t)b * H;
            double ss = 0.0;
            for (int i = 0; i < H; i++) ss += (double)st[i] * st[i];
            float r = sqrtf((float)(ss / (double)H) + 1e-6f);
            if (getenv("SALT_NAN_PROBE") && L < 2 && b < 2)
                fprintf(stderr, "[chnorm] L%d b=%d st-rms=%.6g "
                        "st[0..3]=%.6g %.6g %.6g %.6g w[0..3]=%.6g "
                        "%.6g %.6g %.6g\n",
                        L, b, r, st[0], st[1], st[2], st[3],
                        bf16_f(nw[0]), bf16_f(nw[1]),
                        bf16_f(nw[2]), bf16_f(nw[3]));
            for (int i = 0; i < H; i++) {
                uint32_t bits = (uint32_t)nw[i] << 16;
                float w;
                memcpy(&w, &bits, 4);
                xin[i] = st[i] / r * w;
            }
        }
    } else {
        for (int b = 0; b < B; b++)
            memcpy(xins + (size_t)b * H, states[b],
                   (size_t)H * sizeof(float));
    }

    /* 2) batched qkv + z: dequant once per row, reuse over B tokens */
    double _tp0 = getenv("SALT_PROJ_MS") ? now_s() : 0;
    if (getenv("SALT_NAN_PROBE") && L < 2) {
        const uint16_t *s0 = (const uint16_t *)(const void *)(tr +
                              tl->t[ps].off);
        double xr2 = 0;
        for (int i = 0; i < cols; i++)
            xr2 += (double)xins[i] * xins[i];
        fprintf(stderr, "[chkqkv] L%d pi=%d ps=%d qkv_dtype=%d "
                "qkv_rows=%d cols=%d B=%d s0=%u srank=%d sdims=%ld,%ld "
                "x-rms=%.4g\n",
                L, pi, ps, tl->t[pi].dtype, qkv_rows, cols, B,
                (unsigned)s0[0], tl->t[ps].rank,
                tl->t[ps].dims[0], tl->t[ps].dims[1],
                sqrt(xr2 / cols));
    }
    if (q4_batch_proj(tl, pi, ps, pb, tr, qkv_rows, cols, B, xins,
                      qkvs, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(qkvs - xs)) != 0) {
        q4_chunk_buffer_release(&xs_storage); free(bbuf);
        return -1;                 /* caller falls back to serial */
    }
    double _tqkv = getenv("SALT_PROJ_MS") ? now_s() : 0;
    if (q4_batch_proj(tl, zi, zs, zb, tr, z_rows, cols, B, xins,
                      zbtok, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(zbtok - xs)) != 0) {
        (void)salt_gpu_sync();
        q4_chunk_buffer_release(&xs_storage); free(bbuf);
        return -1;                 /* caller falls back to serial */
    }
    /* qkvs/z are consumed by the CPU recurrent body below.  The layer-end
     * wait is too late: deferred Metal ownership ends at this host-read
     * boundary. */
    if (salt_gpu_sync() != 0) {
        q4_chunk_buffer_release(&xs_storage); free(bbuf);
        return -1;
    }
    if (getenv("SALT_PROJ_MS")) {
        static double _qa = 0, _za = 0; static long _qc = 0, _zc = 0;
        _qa += _tqkv - _tp0; _qc++;
        _za += now_s() - _tqkv; _zc++;
        if (_qc <= 5 || _qc % 40 == 0)
            fprintf(stderr, "[proj-chunk] L%d B=%d qkv %.2f ms (avg %.2f) "
                    "z %.2f ms (avg %.2f)\n", L, B,
                    (_tqkv - _tp0) * 1e3, _qa / _qc * 1e3,
                    (now_s() - _tqkv) * 1e3, _za / _zc * 1e3);
    }
    if (getenv("SALT_PROJ_MS") && L == 0 && t0 == 512 && t0 + B > 512) {
        int tb = 0;
        float *qkv = qkvs + (size_t)tb * qkv_rows;
        float *z = zbtok + (size_t)tb * z_rows;
        FILE *pf = fopen("/tmp/qkv-L0c1.bin", "wb");
        if (pf) {
            fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
            fwrite(z, sizeof(float), (size_t)z_rows, pf);
            fclose(pf);
        }
        FILE *px = fopen("/tmp/xins-L0c1.bin", "wb");
        if (px) {
            fwrite(xins, sizeof(float), (size_t)B * H, px);
            fclose(px);
        }
        /* lin state for L0 after chunk 0 = kv->lin + 0 */
        FILE *pl = fopen("/tmp/lin-L0.bin", "wb");
        if (pl) {
            fwrite(kv->lin, sizeof(float),
                   (size_t)kv->lin_vh * kv->lin_kd * kv->lin_vd, pl);
            fclose(pl);
        }
        FILE *pc = fopen("/tmp/conv-L0.bin", "wb");
        if (pc) {
            fwrite(kv->conv, sizeof(float), (size_t)4 * kv->conv_rows, pc);
            fclose(pc);
        }
    }
    if (getenv("SALT_PROJ_MS") && L == 1 && t0 == 512 && t0 + B > 512) {
        int tb = 0;
        float *qkv = qkvs + (size_t)tb * qkv_rows;
        float *z = zbtok + (size_t)tb * z_rows;
        FILE *pf = fopen("/tmp/qkv-chunk-L1c1.bin", "wb");
        if (pf) {
            fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
            fwrite(z, sizeof(float), (size_t)z_rows, pf);
            fclose(pf);
        }
        FILE *px = fopen("/tmp/xins-L1c1.bin", "wb");
        if (px) {
            fwrite(xins, sizeof(float), (size_t)B * H, px);
            fclose(px);
        }
    }
    if (getenv("SALT_PROJ_MS") && L == 1 && t0 == 0 && t0 + B > 8) {
        int tb = 8 - t0;
        float *qkv = qkvs + (size_t)tb * qkv_rows;
        float *z = zbtok + (size_t)tb * z_rows;
        FILE *pf = fopen("/tmp/qkv-chunk-L3.bin", "wb");
        if (pf) {
            fwrite(qkv, sizeof(float), (size_t)qkv_rows, pf);
            fwrite(z, sizeof(float), (size_t)z_rows, pf);
            fclose(pf);
        }
    }

    /* 3) serial per-token body: conv ring + delta state in token
     * order. o_proj is DEFERRED (skip_o): the readout is collected
     * per token and the shared 8.4 MB o tensor is batched in step 4
     * (M4 -- same bit-identical kernel, B readouts amortize the
     * dequant). */
    int ro_cols = v_heads * vd;                /* readout width */
    float *readouts = (float *)malloc((size_t)B * (size_t)ro_cols *
                                      sizeof(float));
    Q4ChunkBuffer obatch_storage;
    if ((size_t)B > SIZE_MAX / (size_t)o_rows / sizeof(float) ||
        q4_chunk_buffer_alloc(&obatch_storage,
            (size_t)B * (size_t)o_rows * sizeof(float), indexed_mix) != 0) {
        q4_chunk_buffer_release(&xs_storage); free(bbuf); free(readouts);
        return -1;
    }
    float *obatch = obatch_storage.data;
    if (!readouts) { q4_chunk_buffer_release(&xs_storage); free(bbuf);
                     q4_chunk_buffer_release(&obatch_storage); return -1; }
    double _tl0 = getenv("SALT_BODY_MS") ? now_s() : 0;
    if (getenv("SALT_DELTA_CHUNK")) {
        /* M3 chunked delta kernel (deltachunk.c). M3B1: calls the
         * exact serial per-token body per token -- the degenerate
         * anchor. M3-full swaps in the chunked quadratic. */
        float *xarr[B];
        for (int b = 0; b < B; b++) xarr[b] = xins + (size_t)b * H;
        if (salt_delta_chunk_body(cfg, tl, L, tr, kv, t0, B, states,
                                  qkvs, zbtok, xarr, readouts) != 0) {
            q4_chunk_buffer_release(&xs_storage); free(bbuf); free(readouts);
            q4_chunk_buffer_release(&obatch_storage);
            return -1;
        }
    } else {
    for (int b = 0; b < B; b++) {
        float *qkv = qkvs + (size_t)b * qkv_rows;
        float *z = zbtok + (size_t)b * z_rows;
        /* bbuf layout: [xin H][qkv][z][o][readout][qk]; lin_body reads
         * xin from buf (the a/b gates project xin) -- we pass the
         * token's xin as buf start so the a/b projection is right. */
        float *bqkv = bbuf + H;
        float *bz = bqkv + qkv_rows;
        memcpy(bbuf, xins + (size_t)b * H, (size_t)H * sizeof(float));
        memcpy(bqkv, qkv, (size_t)qkv_rows * sizeof(float));
        memcpy(bz, z, (size_t)z_rows * sizeof(float));
        if (salt_attn_lin_body(cfg, tl, L, tr, kv, t0 + b, states[b], bqkv, bz,
                     bbuf, 1) != 0) {
            q4_chunk_buffer_release(&xs_storage); free(bbuf); free(readouts);
            q4_chunk_buffer_release(&obatch_storage);
            return -1;
        }
        /* readout = o region + o_rows (lin_body's layout, skip_o
         * leaves it at the post-RMSNormGated state) */
        memcpy(readouts + (size_t)b * ro_cols,
               bbuf + H + qkv_rows + z_rows + o_rows,
               (size_t)ro_cols * sizeof(float));
    }
    }
    if (getenv("SALT_NAN_PROBE") && L == 0 && t0 == 0 && B > 21) {
        FILE *rf = fopen("/tmp/q36-eng-readout-chunk21.bin", "wb");
        if (rf) {
            fwrite(readouts + 21 * ro_cols, sizeof(float),
                   (size_t)ro_cols, rf);
            fclose(rf);
        }
    }
    if (getenv("SALT_BODY_MS"))
        fprintf(stderr, "[body] L%d B=%d serial-body %.2f ms "
                "(avg %.2f)\n", L, B, (now_s() - _tl0) * 1e3,
                (now_s() - _tl0) / B * 1e3);
    if (getenv("SALT_PROJ_MS") && L == 0 && t0 == 512) {
        FILE *pr = fopen("/tmp/readouts-L0c1.bin", "wb");
        if (pr) {
            fwrite(readouts, sizeof(float), (size_t)B * ro_cols, pr);
            fclose(pr);
        }
        FILE *ps = fopen("/tmp/postattn-L0c1.bin", "wb");
        if (ps) {
            for (int b = 0; b < B; b++)
                fwrite(states[b], sizeof(float), (size_t)H, ps);
            fclose(ps);
        }
    }
    /* 4) batched o_proj: one row-decode, B readouts through the same
     * accumulator map -- bit-identical to B serial q4_proj calls */
    double _to0 = getenv("SALT_PROJ_MS") ? now_s() : 0;
    if (q4_batch_proj(tl, tl->q3_opa[L], tl->q3_opas[L],
                        tl->q3_opab[L], tr, o_rows, ro_cols, B, readouts,
                        obatch,
                        obatch_storage.shared_owned ? &obatch_storage.shared
                                                   : NULL,
                        0) != 0) {
        (void)salt_gpu_sync();
        q4_chunk_buffer_release(&xs_storage); free(bbuf); free(readouts);
        q4_chunk_buffer_release(&obatch_storage);
        return -1;                 /* caller falls back to serial */
    }
    /* The residual add below consumes o_proj immediately. */
    if (salt_gpu_sync() != 0) {
        q4_chunk_buffer_release(&xs_storage); free(bbuf); free(readouts);
        q4_chunk_buffer_release(&obatch_storage);
        return -1;
    }
    if (getenv("SALT_NAN_PROBE") && L == 0 && t0 == 0 && B > 21) {
        FILE *of = fopen("/tmp/q36-eng-o-batch-t21.bin", "wb");
        if (of) {
            fwrite(obatch + 21 * o_rows, sizeof(float), (size_t)o_rows, of);
            fclose(of);
        }
    }
    if (getenv("SALT_PROJ_MS")) {
        static double _oa = 0; static long _oc = 0;
        _oa += now_s() - _to0; _oc++;
        if (_oc <= 5 || _oc % 40 == 0)
            fprintf(stderr, "[proj-chunk] L%d B=%d o %.2f ms (avg %.2f)\n",
                    L, B, (now_s() - _to0) * 1e3, _oa / _oc * 1e3);
    }
    /* 5) residual add (deferred from lin_body) */
    if (getenv("SALT_NAN_PROBE") && L == 0 && t0 == 0 && B > 21) {
        double s2 = 0;
        for (int i = 0; i < H; i++) s2 += (double)states[21][i] * states[21][i];
        fprintf(stderr, "[resid] L0 t21 PRE-residual state rms %.6g\n",
                sqrt(s2 / H));
    }
    for (int b = 0; b < B; b++) {
        float *st = states[b];
        const float *o = obatch + (size_t)b * o_rows;
        for (int i = 0; i < H; i++) st[i] += o[i];
    }
    if (getenv("SALT_NAN_PROBE") && L == 0 && t0 == 0 && B > 21) {
        double s2 = 0;
        for (int i = 0; i < H; i++) s2 += (double)states[21][i] * states[21][i];
        fprintf(stderr, "[resid] L0 t21 POST-residual state rms %.6g\n",
                sqrt(s2 / H));
    }
    if (getenv("SALT_NAN_PROBE") && L < 2) {
        double o2 = 0, s2 = 0;
        int onan = 0, snan = 0;
        for (int b = 0; b < B && b < 4; b++) {
            const float *o = obatch + (size_t)b * o_rows;
            const float *st = states[b];
            for (int i = 0; i < H; i++) {
                if (o[i] != o[i]) onan++;
                if (st[i] != st[i]) snan++;
                o2 += (double)o[i] * o[i];
                s2 += (double)st[i] * st[i];
            }
        }
        fprintf(stderr, "[ores] L%d o-rms=%.6g post-rms=%.6g "
                "o-nan=%d st-nan=%d\n",
                L, sqrt(o2 / (4 * H)), sqrt(s2 / (4 * H)), onan, snan);
    }
    free(readouts); q4_chunk_buffer_release(&obatch_storage);
    _chk_acc[0] += now_s() - _tch; _chk_n++;
    if (_chk_n <= 40 || _chk_n % 80 == 0)
        fprintf(stderr, "[chunk] L%d B=%d total %.2f ms (avg %.2f)\n",
                L, B, (now_s() - _tch) * 1e3, _chk_acc[0] / _chk_n * 1e3);
    q4_chunk_buffer_release(&xs_storage); free(bbuf);
    return 0;
}

/* ---- M3-lite: chunked GQA (batched projections) ------------------- */
/* Full-GQA layers (L3, L39): q/k/v + o_proj are batched over the
 * chunk with the bit-identical kernel (one row-decode, B tokens
 * through the same accumulator map); the attention body (norms,
 * RoPE, softmax over past tokens, gate) stays SERIAL per token via
 * gqa_body -- it reads the growing K cache, inherently sequential.
 * Same code as gqa_step for the body, same kernel as M1 for the
 * projections, so results are bit-identical to serial. */

int salt_attn_gqa_chunk(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                        int L, const uint8_t *tr, float *const *states,
                        int t0, int B, SaltKvCache *kv) {
    int qi = tl->q3_q[L], qs = tl->q3_qs[L], qb = tl->q3_qb[L];
    int ki = tl->q3_k[L], ks = tl->q3_ks[L], kb = tl->q3_kb[L];
    int vi = tl->q3_v[L], vs = tl->q3_vs[L], vb = tl->q3_vb[L];
    int oi = tl->q3_o[L], os = tl->q3_os[L], ob = tl->q3_ob[L];
    int iln = tl->attn_norm[L];
    if (qi < 0 || ki < 0 || vi < 0 || oi < 0 || !cfg || !kv) return -1;
    int H = cfg->hidden;
    int qrows, krows, vrows, orows, qcols, ocols;
    int heads, kv_heads, kh, rope_dim;
    long qcols_l = salt_trunk_tensor_cols(&tl->t[qi]);
    long ocols_l = salt_trunk_tensor_cols(&tl->t[oi]);
    size_t per_token = 0, total_floats, term;
    if (B < 1 || t0 < 0 || !kv->kv || B > kv->max_tokens - t0 ||
        salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
        salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0 ||
        qcols_l != H || ocols_l < 1 || ocols_l > INT_MAX ||
        salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                (int)ocols_l, &heads, &kv_heads, &kh,
                                &rope_dim) != 0)
        return -1;
    qcols = (int)qcols_l;
    ocols = (int)ocols_l;
    if (kv->kvlat != krows + vrows && kv->kvlat != 0)
        return -1;
    if (linear_size_add(&per_token, (size_t)H) != 0 ||
        linear_size_add(&per_token, (size_t)qrows) != 0 ||
        linear_size_add(&per_token, (size_t)krows) != 0 ||
        linear_size_add(&per_token, (size_t)vrows) != 0 ||
        linear_size_add(&per_token, (size_t)orows) != 0 ||
        linear_size_add(&per_token, (size_t)ocols) != 0 ||
        linear_size_mul((size_t)B, per_token, &total_floats) != 0 ||
        linear_size_mul((size_t)B, (size_t)ocols, &term) != 0 ||
        linear_size_mul(term, 2, &term) != 0 ||
        linear_size_add(&total_floats, term) != 0 ||
        linear_size_mul((size_t)rope_dim, 2, &term) != 0 ||
        linear_size_add(&total_floats, term) != 0 ||
        total_floats > SIZE_MAX / sizeof(float))
        return -1;
    Q4ChunkBuffer xs_storage;
    int indexed_mix = q4_indexed_mix_enabled();
    if (q4_chunk_buffer_alloc(&xs_storage,
                              total_floats * sizeof(float), indexed_mix) != 0)
        return -1;
    float *xs = xs_storage.data;
    if (getenv("SALT_NAN_PROBE") && L == 3)
        fprintf(stderr, "[gqa-chunk] L%d B=%d malloc total=%zu "
                "qrows=%d krows=%d vrows=%d orows=%d ocols=%d "
                "xs=%p\n", L, B, total_floats, qrows, krows, vrows, orows,
                ocols, (void *)xs);
    float *xins = xs;                          /* [B][H] row-major */
    float *qb_ = xins + (size_t)B * H;         /* [B][qrows] */
    float *kb_ = qb_ + (size_t)B * qrows;      /* [B][krows] */
    float *vb_ = kb_ + (size_t)B * krows;      /* [B][vrows] */
    float *ob_ = vb_ + (size_t)B * vrows;      /* [B][orows] */
    float *aob = ob_ + (size_t)B * orows;      /* [B][ocols] */

    /* 1) per-token input_layernorm into xins[B][H] (row-major) */
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                              tl->t[iln].off);
        for (int b = 0; b < B; b++) {
            const float *st = states[b];
            float *xin = xins + (size_t)b * H;
            double ss = 0.0;
            for (int i = 0; i < H; i++) ss += (double)st[i] * st[i];
            float r = sqrtf((float)(ss / (double)H) + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint32_t bits = (uint32_t)nw[i] << 16;
                float w;
                memcpy(&w, &bits, 4);
                xin[i] = st[i] / r * w;
            }
        }
    } else {
        for (int b = 0; b < B; b++)
            memcpy(xins + (size_t)b * H, states[b],
                   (size_t)H * sizeof(float));
    }

    /* 2) batched q/k/v projections (bit-identical kernel) */
    if (getenv("SALT_NAN_PROBE") && L == 3)
        fprintf(stderr, "[gqa-chunk] L3 pre-proj B=%d qb_=%p kb_=%p "
                "vb_=%p xins=%p\n", B, (void *)qb_, (void *)kb_,
                (void *)vb_, (void *)xins);
    if (q4_batch_proj(tl, qi, qs, qb, tr, qrows, qcols, B, xins,
                      qb_, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(qb_ - xs)) != 0) {
        q4_chunk_buffer_release(&xs_storage);
        return -1;                 /* caller falls back to serial */
    }
    if (getenv("SALT_NAN_PROBE") && L == 3)
        fprintf(stderr, "[gqa-chunk] L3 q done\n");
    if (q4_batch_proj(tl, ki, ks, kb, tr, krows, qcols, B, xins,
                      kb_, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(kb_ - xs)) != 0) {
        (void)salt_gpu_sync();
        q4_chunk_buffer_release(&xs_storage);
        return -1;
    }
    if (getenv("SALT_NAN_PROBE") && L == 3)
        fprintf(stderr, "[gqa-chunk] L3 k done\n");
    if (q4_batch_proj(tl, vi, vs, vb, tr, vrows, qcols, B, xins,
                      vb_, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(vb_ - xs)) != 0) {
        (void)salt_gpu_sync();
        q4_chunk_buffer_release(&xs_storage);
        return -1;
    }
    /* q/k/v feed the serial GQA body and KV update. */
    if (salt_gpu_sync() != 0) {
        q4_chunk_buffer_release(&xs_storage);
        return -1;
    }
    if (getenv("SALT_NAN_PROBE") && L == 3)
        fprintf(stderr, "[gqa-chunk] L3 v done\n");
    if (getenv("SALT_PROJ_MS") && L == 27 && t0 == 0) {
        FILE *pf = fopen("/tmp/gqa-proj-L27.bin", "wb");
        if (pf) {
            fwrite(qb_, sizeof(float), (size_t)B * qrows, pf);
            fwrite(kb_, sizeof(float), (size_t)B * krows, pf);
            fwrite(vb_, sizeof(float), (size_t)B * vrows, pf);
            fclose(pf);
        }
    }

    /* 3) two-phase batched attention:
     *   3a) prep ALL B tokens first (split|gate, norms, RoPE, cache
     *       write) -- pure per-token, fills the whole chunk's K/V
     *       cache + stores qq/gate per token. This is the bit-
     *       fidelity anchor (identical gqa_prep code).
     *   3b) attention parallelized over (b,h) pairs: each pair's
     *       scores->softmax->v-sum runs in the EXACT serial order
     *       (t2 ascending, i ascending, same expf/div sequence), so
     *       every aob[b][h] row is bit-identical to the serial
     *       gqa_body. 8 threads over B*heads pairs. */
    {
        float *qqb = aob + (size_t)B * ocols;    /* [B][heads*kh] */
        float *gateb = qqb + (size_t)B * heads * kh;
        float *cosb = gateb + (size_t)B * heads * kh;
        float *sinb = cosb + rope_dim;
        /* 3a: prep all tokens (serial, fills the cache + qq/gate) */
        for (int b = 0; b < B; b++) {
            if (getenv("SALT_NAN_PROBE") && L == 3 &&
                (b < 3 || b >= B - 2))
                fprintf(stderr, "[gqa-prep] L3 b=%d t0+b=%d "
                        "kbase=%p kvlat=%d\n", b, t0 + b,
                        (void *)(kv->kv + ((size_t)L * kv->max_tokens +
                        (size_t)(t0 + b)) * kv->kvlat), kv->kvlat);
            if (gqa_prep(cfg, tl, L, tr, kv, t0 + b,
                         qb_ + (size_t)b * qrows,
                         kb_ + (size_t)b * krows,
                         vb_ + (size_t)b * vrows,
                         qqb + (size_t)b * heads * kh,
                         gateb + (size_t)b * heads * kh,
                         cosb, sinb) != 0) {
                q4_chunk_buffer_release(&xs_storage);
                return -1;
            }
        }
        /* 3b: threaded attention over (b,h) pairs. aob zeroed ONCE
         * (threads write disjoint h slices of each b). Per-pair math
         * is the exact serial order -- bit-identical to gqa_body. */
        if (getenv("SALT_NAN_PROBE") && L == 3)
            fprintf(stderr, "[gqa-3b] L3 B=%d spawn: qqb=%p gateb=%p "
                    "aob=%p heads=%d kh=%d ocols=%d\n", B,
                    (void *)qqb, (void *)gateb, (void *)aob,
                    heads, kh, ocols);
        memset(aob, 0, (size_t)B * (size_t)ocols * sizeof(float));
        gqa_atn_spawn(kv, qqb, gateb, aob, L, t0, B, heads, kv_heads,
                      kh, krows, vrows, ocols, kv->nthreads);
        if (getenv("SALT_NAN_PROBE") && L == 3)
            fprintf(stderr, "[gqa-3b] L3 B=%d spawn done\n", B);
    }

    /* 4) batched o_proj over the attention outputs */
    if (q4_batch_proj(tl, oi, os, ob, tr, orows, ocols, B, aob,
                      ob_, xs_storage.shared_owned ? &xs_storage.shared : NULL,
                      (size_t)(ob_ - xs)) != 0) {
        (void)salt_gpu_sync();
        q4_chunk_buffer_release(&xs_storage);
        return -1;                 /* caller falls back to serial */
    }
    if (salt_gpu_sync() != 0) {
        q4_chunk_buffer_release(&xs_storage);
        return -1;
    }

    /* 5) residual add */
    for (int b = 0; b < B; b++) {
        float *st = states[b];
        const float *o = ob_ + (size_t)b * orows;
        for (int i = 0; i < H; i++) st[i] += o[i];
    }
    q4_chunk_buffer_release(&xs_storage);
    return 0;
}

/* The linear layer's projection phase: the normed snapshot + the
 * qkv+z projections (the GPU-routed q4_proj's) -- the prefetchable
 * half of linear_step. The body consumes the prefetched qkv/z's. */
static int linear_proj(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                       const uint8_t *tr, float *state, SaltKvCache *kv,
                       int token) {
    int pi = tl->q3_pqkv[L], ps = tl->q3_pqkvs[L], pb = tl->q3_pqkvb[L];
    int zi = tl->q3_pz[L], zs = tl->q3_pzs[L], zb = tl->q3_pzb[L];
    int ni = tl->q3_lnorm[L];
    int iln = tl->attn_norm[L];
    if (pi < 0 || zi < 0) return 0;
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];
    int z_rows = (int)tl->t[zi].dims[0];
    int cols = (int)salt_trunk_tensor_cols(&tl->t[pi]);
    if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens) return 0;
    float *buf = kv->scratch;
    float *xin = buf;
    float *qkv = xin + H;
    float *z = qkv + qkv_rows;
    if (iln >= 0) {
        const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                             tl->t[iln].off);
        memcpy(xin, state, (size_t)H * sizeof(float));
        double ss = 0.0;
        for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
        float r = sqrtf((float)(ss / (double)H) + 1e-6f);
        for (int i = 0; i < H; i++) {
            uint32_t bits = (uint32_t)nw[i] << 16;
            float w;
            memcpy(&w, &bits, 4);
            xin[i] = xin[i] / r * w;
        }
    } else {
        memcpy(xin, state, (size_t)H * sizeof(float));
    }
    if (tl->t[pi].dtype == 2 || tl->t[zi].dtype == 2 ||
        tl->t[pi].bits == 8 || tl->t[zi].bits == 8) {
        if (q4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, xin, qkv) != 0 ||
            q4_proj(tl, zi, zs, zb, tr, z_rows, cols, xin, z) != 0)
            return -1;
    } else if (q4_matvec2_decode(kv,
            (const uint32_t *)(const void *)(tr + tl->t[pi].off),
            (const uint16_t *)(const void *)(tr + tl->t[ps].off),
            pb >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[pb].off)
                    : NULL, qkv_rows,
            (const uint32_t *)(const void *)(tr + tl->t[zi].off),
            (const uint16_t *)(const void *)(tr + tl->t[zs].off),
            zb >= 0 ? (const uint16_t *)(const void *)(tr + tl->t[zb].off)
                    : NULL, z_rows,
            cols, xin, qkv, z) != 0) {
        if (q4_proj(tl, pi, ps, pb, tr, qkv_rows, cols, xin, qkv) != 0 ||
            q4_proj(tl, zi, zs, zb, tr, z_rows, cols, xin, z) != 0)
            return -1;
    }
    (void)ni;
    return 0;
}

/* ------------------------------------------------------------------
 * Pipelined split (SALT_GPU_OVERLAP=1): salt_attn_qwen_proj runs the
 * layer's normed snapshot + the GPU-routed projections; the body
 * consumes the prefetched q/k/v's. gqa_step = proj + body exactly
 * (bit-identical to the serial path by construction).
 * ------------------------------------------------------------------ */
int salt_attn_qwen_proj(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                        const uint8_t *tr, float *state, SaltKvCache *kv,
                        int token) {
    /* GQA layers: the q/k/v projections (the GPU-routed q4_proj's).
     * The normed snapshot is the first step -- the body's state fold
     * never races the prefetch (the proj reads the state BEFORE the
     * body writes it). */
    if (tl->q3_q[L] >= 0) {
        int qi = tl->q3_q[L], qs = tl->q3_qs[L], qb = tl->q3_qb[L];
        int ki = tl->q3_k[L], ks = tl->q3_ks[L], kb = tl->q3_kb[L];
        int vi = tl->q3_v[L], vs = tl->q3_vs[L], vb = tl->q3_vb[L];
        int oi = tl->q3_o[L];
        int iln = tl->attn_norm[L];
        int H = cfg->hidden;
        if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens)
            return 0;
        int qrows, krows, vrows, orows, qcols, ocols;
        int heads, kv_heads, kh, rope_dim;
        long qcols_l, ocols_l, scratch_need;
        if (ki < 0 || vi < 0 || oi < 0 ||
            salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0)
            return -1;
        qcols_l = salt_trunk_tensor_cols(&tl->t[qi]);
        ocols_l = salt_trunk_tensor_cols(&tl->t[oi]);
        if (qcols_l != H || ocols_l < 1 || ocols_l > INT_MAX ||
            salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                    (int)ocols_l, &heads, &kv_heads, &kh,
                                    &rope_dim) != 0)
            return -1;
        qcols = (int)qcols_l;
        ocols = (int)ocols_l;
        scratch_need = salt_attn_gqa_scratch_floats(
            cfg, qrows, krows, vrows, orows, ocols, kv->max_tokens);
        if (scratch_need < 0 || !kv->scratch || kv->scratch_n < scratch_need)
            return -1;
        (void)heads; (void)kv_heads; (void)kh; (void)rope_dim;
        if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens)
            return 0;
        float *buf = kv->scratch;
        float *xin = buf;
        float *q = xin + H;
        float *k = q + qrows;
        float *v = k + krows;
        if (iln >= 0) {
            const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                                 tl->t[iln].off);
            memcpy(xin, state, (size_t)H * sizeof(float));
            double ss = 0.0;
            for (int i = 0; i < H; i++) ss += (double)xin[i] * xin[i];
            float r = sqrtf((float)(ss / (double)H) + 1e-6f);
            for (int i = 0; i < H; i++) {
                uint32_t bits = (uint32_t)nw[i] << 16;
                float w;
                memcpy(&w, &bits, 4);
                xin[i] = xin[i] / r * w;
            }
        } else {
            memcpy(xin, state, (size_t)H * sizeof(float));
        }
        if (q4_proj(tl, qi, qs, qb, tr, qrows, qcols, xin, q) != 0 ||
            q4_proj(tl, ki, ks, kb, tr, krows, qcols, xin, k) != 0 ||
            q4_proj(tl, vi, vs, vb, tr, vrows, qcols, xin, v) != 0)
            return -1;
        return 0;
    }
    /* Linear layers: the qkv+z projections. */
    if (tl->q3_pqkv[L] >= 0 || tl->q3_conv[L] >= 0)
        return linear_proj(cfg, tl, L, tr, state, kv, token);
    return 0;
}

int salt_attn_qwen_body(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                        const uint8_t *tr, float *state, SaltKvCache *kv,
                        int token) {
    if (tl->q3_q[L] >= 0) {
        /* the body consumes the prefetched q/k/v's in the scratch
         * (the same offsets salt_attn_qwen_proj wrote), then the
         * o_proj + the state fold -- the exact gqa_step tail. */
        int qi = tl->q3_q[L];
        int ki = tl->q3_k[L], vi = tl->q3_v[L];
        int oi = tl->q3_o[L], os = tl->q3_os[L], ob = tl->q3_ob[L];
        int H = cfg->hidden;
        if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens)
            return 0;
        int qrows, krows, vrows, orows, ocols;
        int heads, kv_heads, kh, rope_dim;
        long ocols_l, scratch_need;
        if (ki < 0 || vi < 0 || oi < 0 ||
            salt_trunk_tensor_rows(&tl->t[qi], &qrows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[ki], &krows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[vi], &vrows) != 0 ||
            salt_trunk_tensor_rows(&tl->t[oi], &orows) != 0)
            return -1;
        ocols_l = salt_trunk_tensor_cols(&tl->t[oi]);
        if (ocols_l < 1 || ocols_l > INT_MAX ||
            salt_model_gqa_geometry(cfg, qrows, krows, vrows, orows,
                                    (int)ocols_l, &heads, &kv_heads, &kh,
                                    &rope_dim) != 0)
            return -1;
        ocols = (int)ocols_l;
        scratch_need = salt_attn_gqa_scratch_floats(
            cfg, qrows, krows, vrows, orows, ocols, kv->max_tokens);
        if (scratch_need < 0 || !kv->scratch || kv->scratch_n < scratch_need)
            return -1;
        (void)heads; (void)kv_heads; (void)kh; (void)rope_dim;
        if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens)
            return 0;
        float *buf = kv->scratch;
        float *q = buf + H;
        float *k = q + qrows;
        float *v = k + krows;
        float *o = v + vrows;
        float *attn_out = o + orows;
        if (gqa_body(cfg, tl, L, tr, kv, token, q, k, v, attn_out) != 0)
            return -1;
        /* o_proj(attn_out) -> o; residual add (the gqa_step tail). */
        if (q4_proj(tl, oi, os, ob, tr, orows, ocols, attn_out, o) != 0)
            return -1;
        for (int i = 0; i < H; i++) state[i] += o[i];
        return 0;
    }
    /* Linear layers: the lin_body (the delta+readout+o) on the
     * prefetched qkv/z's, then the residual fold. */
    if (tl->q3_pqkv[L] >= 0 || tl->q3_conv[L] >= 0) {
        int pi = tl->q3_pqkv[L];
        int qkv_rows = (int)tl->t[pi].dims[0];
        int H = cfg->hidden;
        if (token < 0 || !kv || !kv->kv || token >= kv->max_tokens)
            return 0;
        float *buf = kv->scratch;
        float *qkv = buf + (size_t)H;
        float *z = qkv + qkv_rows;
        return salt_attn_lin_body(cfg, tl, L, tr, kv, token, state,
                                  qkv, z, buf, 0);
    }
    return 0;
}

/* Dispatch: full-GQA layers (self_attn roles present) vs linear. */
int salt_attn_qwen_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                        const uint8_t *tr, float *state, SaltKvCache *kv,
                        int token) {
    if (!tl || !cfg) return -1;
    if (getenv("SALT_NAN_PROBE") && (L == 3 || L == 7) &&
        getenv("SALT_DUMP_STATE")) {
        char sf[64];
        snprintf(sf, sizeof sf, "/tmp/q36-L%d-in.bin", L);
        FILE *sfp = fopen(sf, "wb");
        if (sfp) {
            fwrite(state, sizeof(float), (size_t)cfg->hidden, sfp);
            fclose(sfp);
        }
    }
    if (tl->q3_q[L] >= 0)
        return gqa_step(cfg, tl, L, tr, state, kv, token);
    if (tl->q3_conv[L] >= 0 || tl->q3_pqkv[L] >= 0)
        return linear_step(cfg, tl, L, tr, state, kv, token);
    return 0;                     /* no attention graph: skip */
}
