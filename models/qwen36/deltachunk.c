/* deltachunk.c -- M3 chunked delta-rule kernel (parallel prefill).
 *
 * M3-full (SALT_DELTA_CHUNK=2): the chunked quadratic.
 *
 * Serial convention (lin_body): decay BEFORE kv_mem --
 *   S_{t+1} = d_t·S_t + k_t ⊗ β_t(v_t − d_t·S_t·k_t),  d_t = exp(g_t)
 *
 * Chunked form: normalize S'_t = S_t/D_t, D_t = Π_{u<t} d_u,
 * v̂_t = v_t/D_{t+1}. Then
 *   S'_{t+1} = S'_t + k_t ⊗ β_t(v̂_t − S'_t·k_t)        (vanilla delta)
 *   δ̂ = (I + β⊙tril(KK^T))^{-1} · β⊙(v̂ − S_0·K)         (WY solve, O(B²))
 *   out_t = D_{t+1}·(S_0·q_t + Σ_{u≤t}(k_u·q_t)·δ̂_u)     (triangular out)
 *   S_out = D_B·(S_0 + K^T·δ̂)                             (state update)
 *
 * The dense parts (K·K^T, S_0·K, S_0·Q, K^T·δ̂) are matmul-shaped
 * (thread/SIMD-friendly); the only serial spine is the O(BT²) forward
 * substitution. Sub-chunk BT=64 (FLA's chunk size; the BT×BT solve
 * fits the scratch arena). 32 value heads are independent -> threaded
 * via the pool (M9 pattern) -- serial over heads for now (correctness
 * anchor first; head threading is the same M9 pattern, next).
 *
 * Correctness level: SELF-CONSISTENCY. The chunked accumulation order
 * differs from the serial recurrence (dividing v by D then re-scaling
 * vs in-place decay), so results match serial within fp tolerance, not
 * bit-for-bit. Save/load/resume must reproduce the NEW state; the
 * serial path (SALT_DELTA_CHUNK unset / =1) stays byte-identical.
 *
 * B=1 degenerate: BT=1 -> A empty, δ̂ = β⊙(v̂ − S_0·K),
 * out = d·(S_0·q + (k·q)·δ̂), S_out = d·(S_0 + k⊗δ̂) -- exactly the
 * serial recurrence's VALUE (verified by derivation; fp rounding
 * differs, hence tolerance-level equality, not bit-for-bit).
 */
#include "deltachunk.h"
#include "attn.h"
#include "salt/kernels.h"
#include "salt/bitmath.h"
#include "salt/model.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

/* ---- mode 1: degenerate passthrough (bit-identical anchor) ---------- */

/* BF16 -> f32 (same conversion as attn_qwen.c's static bf16_f). */
static float dc_bf16(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static int chunk_body_serial(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                             int L, const uint8_t *tr, SaltKvCache *kv,
                             int t0, int B, float *const *states,
                             float *qkvs, float *zbtok, float *const *xins,
                             float *readouts) {
    int pi = tl->q3_pqkv[L];
    int zi = tl->q3_pz[L];
    if (pi < 0 || zi < 0) return -1;
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];
    int z_rows = (int)tl->t[zi].dims[0];
    int o_rows = (int)tl->t[tl->q3_opa[L]].dims[0];
    int k_heads, v_heads, kd, vd, expected_qkv;
    if (salt_model_linear_geometry(cfg, &k_heads, &v_heads, &kd, &vd,
                                   &expected_qkv) != 0 ||
        qkv_rows != expected_qkv || z_rows != v_heads * vd)
        return -1;
    int ro_cols = v_heads * vd;

    long bneed = salt_attn_linear_scratch_floats(
        cfg, H, qkv_rows, z_rows, o_rows, kv->nthreads);
    if (bneed < 0 || !kv->scratch || kv->scratch_n < bneed) return -1;
    float *bbuf = kv->scratch;
    memset(bbuf, 0, (size_t)bneed * sizeof(float));

    for (int b = 0; b < B; b++) {
        float *qkv = qkvs + (size_t)b * qkv_rows;
        float *z = zbtok + (size_t)b * z_rows;
        float *bqkv = bbuf + H;
        float *bz = bqkv + qkv_rows;
        memcpy(bbuf, xins[b], (size_t)H * sizeof(float));
        memcpy(bqkv, qkv, (size_t)qkv_rows * sizeof(float));
        memcpy(bz, z, (size_t)z_rows * sizeof(float));
        if (salt_attn_lin_body(cfg, tl, L, tr, kv, t0 + b, states[b],
                               bqkv, bz, bbuf, 1) != 0) {
            return -1;
        }
        memcpy(readouts + (size_t)b * ro_cols,
               bbuf + H + qkv_rows + z_rows + o_rows,
               (size_t)ro_cols * sizeof(float));
    }
    return 0;
}

/* ---- M3-full: chunked quadratic -------------------------------------- */
#define M3_BT 64
#define M3_MAX_THREADS 64

/* Per-head working context for one sub-chunk: shared reads (qkvs,
 * zbtok, S0t state slices, betas/decs) + the worker's OWN staging
 * scratch (hset). Heads are independent; workers split the range. */
typedef struct {
    const float *qkvs, *zbtok;
    float *S0t, *readouts;
    const float *betas, *decs;
    const uint16_t *nw;
    int qkv_rows, z_rows, k_heads, v_heads, kd, vd, bt, t0s;
    float *hset;   /* this worker's staging base */
} M3HeadCtx;

typedef struct {
    M3HeadCtx *ctx;
    int h0, h1;
} M3HeadJob;

/* One value head's chunked delta over a BT-token sub-chunk.
 *   S0   [kd][vd]   incoming state (TRUE domain; D resets per chunk)
 *   q,k  [BT][kd]   per-token q/k slices for this head (k: khh = h/2)
 *   v    [BT][vd]   per-token v slice
 *   beta [BT]       sigmoid(b32[h]) per token
 *   d    [BT]       decay exp(g32[h]) per token
 *   D    [BT+1]     cumulative decay prefix (D[0]=1, D[b+1]=D[b]·d[b])
 *   out  [BT][vd]   readout (pre-RMSNormGated) -- MAY alias v (all v
 *                    reads happen before the first out write)
 *   S1   [kd][vd]   out: updated state in TRUE domain (S1 = S0 + K^T·δ̂)
 *                    -- caller multiplies by D[BT] for the carry
 * Scratch: A [BT][BT] strictly-lower (I + β⊙tril(KK^T)), delt [BT][vd].
 */
static void chunk_head_delta(const float *S0, const float *q, const float *k,
                             const float *v, const float *beta,
                             const float *d, const float *D,
                             int kd, int vd, int BT,
                             float *A, float *delt, float *S1,
                             float *out) {
    (void)d;
    if (getenv("SALT_NAN_PROBE") && BT > 1) {
        static int _hd_cnt = 0;
        int hh = _hd_cnt++;
        double s0r = 0, qr = 0, kr = 0, vr = 0, dr = 0;
        for (int i = 0; i < kd * vd; i++) s0r += (double)S0[i] * S0[i];
        for (int i = 0; i < BT * kd; i++) { qr += (double)q[i] * q[i];
                                            kr += (double)k[i] * k[i]; }
        for (int i = 0; i < BT * vd; i++) vr += (double)v[i] * v[i];
        for (int i = 0; i <= BT; i++) dr += (double)D[i] * D[i];
        fprintf(stderr, "[m3hd] h=%d BT=%d S0-rms=%.4g q-rms=%.4g "
                "k-rms=%.4g v-rms=%.4g D-rms=%.4g D[0]=%.4g D[%d]=%.4g\n",
                hh, BT, sqrt(s0r / (kd * vd)), sqrt(qr / (BT * kd)),
                sqrt(kr / (BT * kd)), sqrt(vr / (BT * vd)),
                sqrt(dr / (BT + 1)), (double)D[0], BT, (double)D[BT]);
    }
    /* A[t][u] = beta[t] * (k_t·k_u) for u < t; 0 on/above diagonal */
    for (int t = 1; t < BT; t++) {
        for (int u = 0; u < t; u++) {
            float kk = 0.0f;
            for (int i = 0; i < kd; i++)
                kk += k[(size_t)t * kd + i] * k[(size_t)u * kd + i];
            A[(size_t)t * BT + u] = beta[t] * kk;
        }
    }
    /* rhs_t = beta[t]*(v̂_t − S0·k_t); v̂_t = v_t/D[t+1] */
    for (int t = 0; t < BT; t++) {
        float s0k[256];
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += S0[(size_t)i * vd + j] * k[(size_t)t * kd + i];
            s0k[j] = acc;
        }
        for (int j = 0; j < vd; j++) {
            float vhat = v[(size_t)t * vd + j] / D[t + 1];
            delt[(size_t)t * vd + j] = beta[t] * (vhat - s0k[j]);
        }
    }
    /* forward substitution: (I + A) δ̂ = rhs  (A strictly lower) */
    for (int t = 0; t < BT; t++) {
        for (int u = 0; u < t; u++) {
            float a = A[(size_t)t * BT + u];
            if (a == 0.0f) continue;
            for (int j = 0; j < vd; j++)
                delt[(size_t)t * vd + j] -= a * delt[(size_t)u * vd + j];
        }
    }
    /* out_t = D[t+1] * (S0·q_t + Σ_{u≤t} (k_u·q_t)·δ̂_u) */
    for (int t = 0; t < BT; t++) {
        float s0q[256];
        for (int j = 0; j < vd; j++) {
            float acc = 0.0f;
            for (int i = 0; i < kd; i++)
                acc += S0[(size_t)i * vd + j] * q[(size_t)t * kd + i];
            s0q[j] = acc;
        }
        for (int j = 0; j < vd; j++) {
            float acc = s0q[j];
            for (int u = 0; u <= t; u++) {
                float kq = 0.0f;
                for (int i = 0; i < kd; i++)
                    kq += k[(size_t)u * kd + i] * q[(size_t)t * kd + i];
                acc += kq * delt[(size_t)u * vd + j];
            }
            out[(size_t)t * vd + j] = D[t + 1] * acc;
        }
    }
    /* S1 = S0 + K^T·δ̂ (TRUE domain; caller scales by D[BT] for carry) */
    memcpy(S1, S0, (size_t)kd * vd * sizeof(float));
    for (int u = 0; u < BT; u++)
        salt_rank1(S1, k + (size_t)u * kd, delt + (size_t)u * vd, kd, vd);
}

/* One head's work for one sub-chunk: stage the per-head slices, run
 * the chunked quadratic (or the exact serial recurrence when the
 * cumulative decay underflows the D-normalization), write the state
 * carry into S0t[h] and the RMSNormGated readouts. Deterministic:
 * the chunked heads use the pinned WY-solve arithmetic; the
 * fast-decay heads use the exact lin_body recurrence. */
static void m3_head_work(M3HeadCtx *c, int h) {
    const int kd = c->kd, vd = c->vd, bt = c->bt, t0s = c->t0s;
    const int k_heads = c->k_heads, v_heads = c->v_heads;
    const int qkv_rows = c->qkv_rows, z_rows = c->z_rows;
    int khh = h / (c->v_heads / c->k_heads);
    /* worker staging: Amat BT^2, delt BT*vd, headq BT*kd, headk BT*kd,
     * headv BT*vd, betah BT, dech BT, Dcum BT+1, S0n kd*vd, S1n kd*vd */
    float *Amat = c->hset;
    float *delt = Amat + (size_t)M3_BT * M3_BT;
    float *headq = delt + (size_t)M3_BT * vd;
    float *headk = headq + (size_t)M3_BT * kd;
    float *headv = headk + (size_t)M3_BT * kd;
    float *betah = headv + (size_t)M3_BT * vd;
    float *dech = betah + M3_BT;
    float *Dcum = dech + M3_BT;
    float *S0n = Dcum + (M3_BT + 1);
    float *S1n = S0n + (size_t)kd * vd;

    Dcum[0] = 1.0f;
    for (int b = 0; b < bt; b++) {
        const float *qkv = c->qkvs + (size_t)(t0s + b) * qkv_rows;
        const float *qh = qkv + (size_t)khh * kd;
        const float *kh = qkv + (size_t)(k_heads * kd) + (size_t)khh * kd;
        const float *vh = qkv + (size_t)(2 * k_heads * kd) +
                          (size_t)h * vd;
        memcpy(headq + (size_t)b * kd, qh, (size_t)kd * sizeof(float));
        memcpy(headk + (size_t)b * kd, kh, (size_t)kd * sizeof(float));
        memcpy(headv + (size_t)b * vd, vh, (size_t)vd * sizeof(float));
        betah[b] = c->betas[(size_t)b * v_heads + h];
        dech[b] = c->decs[(size_t)b * v_heads + h];
        Dcum[b + 1] = Dcum[b] * dech[b];
    }
    /* incoming state for this sub-chunk (TRUE domain) */
    memcpy(S0n, c->S0t + (size_t)h * kd * vd,
           (size_t)kd * vd * sizeof(float));
    if (Dcum[bt] < 1e-30f) {
        /* Fast-decay head: the chunked D-normalization would
         * underflow (vhat = v/D -> inf). Run the EXACT serial
         * recurrence for this head over the sub-chunk --
         * deterministic, stable, and the same math as the
         * lin_body (decay BEFORE kv_mem, in-place). */
        memcpy(S1n, S0n, (size_t)kd * vd * sizeof(float));
        float delta[vd];
        for (int b = 0; b < bt; b++) {
            const float *kh = headk + (size_t)b * kd;
            const float *qh = headq + (size_t)b * kd;
            const float *vh = headv + (size_t)b * vd;

            for (int i = 0; i < kd * vd; i++)
                S1n[i] *= dech[b];
            for (int j = 0; j < vd; j++) {
                float kv_mem = 0.0f;
                for (int i = 0; i < kd; i++)
                    kv_mem += S1n[(size_t)i * vd + j] * kh[i];
                delta[j] = (vh[j] - kv_mem) * betah[b];
            }
            salt_rank1(S1n, kh, delta, kd, vd);
            /* readout: S1n·q -> headv[b] (pre-norm) */
            for (int j = 0; j < vd; j++) {
                float acc = 0.0f;
                for (int i = 0; i < kd; i++)
                    acc += S1n[(size_t)i * vd + j] * qh[i];
                headv[(size_t)b * vd + j] = acc;
            }
        }
        memcpy(c->S0t + (size_t)h * kd * vd, S1n,
               (size_t)kd * vd * sizeof(float));
    } else {
        chunk_head_delta(S0n, headq, headk, headv, betah, dech, Dcum,
                         kd, vd, bt, Amat, delt, S1n, headv);
        /* state carry: S_out = D[bt] · S1n */
        for (int i = 0; i < kd * vd; i++)
            c->S0t[(size_t)h * kd * vd + i] = S1n[i] * Dcum[bt];
    }
    /* readouts: RMSNormGated per token (exact lin_body math) */
    for (int b = 0; b < bt; b++) {
        float *oh = headv + (size_t)b * vd;
        const float *zh = c->zbtok + (size_t)(t0s + b) * z_rows +
                          (size_t)h * vd;
        double ss = 0.0;
        for (int i = 0; i < vd; i++)
            ss += (double)oh[i] * oh[i];
        float r = sqrtf((float)(ss / (double)vd) + 1e-6f);
        for (int i = 0; i < vd; i++) {
            float zv = zh[i];
            float sig = 1.0f / (1.0f + salt_expf(-zv));
            oh[i] = oh[i] / r * dc_bf16(c->nw[i]) * (zv * sig);
        }
        memcpy(c->readouts + (size_t)(t0s + b) * v_heads * vd +
               (size_t)h * vd,
               oh, (size_t)vd * sizeof(float));
    }
}

static void *m3_head_worker(void *arg) {
    M3HeadJob *jb = (M3HeadJob *)arg;
    for (int h = jb->h0; h < jb->h1; h++)
        m3_head_work(jb->ctx, h);
    return NULL;
}

/* M3-full chunk body. The per-token pre-body (a/b gates, conv ring,
 * q/k norm, g/beta) replicates lin_body EXACTLY (same inputs, same
 * op order -- bit-identical per token); only the delta recurrence is
 * replaced by the chunked quadratic. Per-token beta/decay are staged
 * as [bt][v_heads] because the gates depend on xin (token-dependent).
 * Mode 2 = SALT_DELTA_CHUNK=2. */
static int chunk_body_chunked(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                              int L, const uint8_t *tr, SaltKvCache *kv,
                              int t0, int B,
                              float *qkvs, float *zbtok, float *const *xins,
                              float *readouts) {
    int pi = tl->q3_pqkv[L];
    int zi = tl->q3_pz[L];
    int ai = tl->q3_pa[L], as_ = tl->q3_pas[L], ab = tl->q3_pab[L];
    int bi = tl->q3_pb[L], bs_ = tl->q3_pbs[L], bb = tl->q3_pbb[L];
    int ci = tl->q3_conv[L];
    int ai_ = tl->q3_a_log[L], di = tl->q3_dt[L];
    int ni = tl->q3_lnorm[L];
    if (pi < 0 || zi < 0 || ai < 0 || bi < 0 || ci < 0 || ai_ < 0 ||
        di < 0 || ni < 0)
        return -1;
    int H = cfg->hidden;
    int qkv_rows = (int)tl->t[pi].dims[0];
    int z_rows = (int)tl->t[zi].dims[0];
    int o_rows = (int)tl->t[tl->q3_opa[L]].dims[0];
    int cols = (int)tl->t[pi].dims[1] * (32 / (tl->t[pi].bits > 0 ? tl->t[pi].bits : 4));
    int k_heads, v_heads, kd, vd, expected_qkv;
    if (salt_model_linear_geometry(cfg, &k_heads, &v_heads, &kd, &vd,
                                   &expected_qkv) != 0 ||
        qkv_rows != expected_qkv)
        return -1;
    if (z_rows != v_heads * vd) return -1;

    /* Scratch layout (all from kv->scratch, arena pre-sized at init):
     *   bbuf   [bneed]            serial-path buffer (unused here)
     *   Amat   [BT][BT]           strictly-lower kernel matrix
     *   delt   [BT][vd]           normalized deltas
     *   headq  [BT][kd]           per-head q slice
     *   headk  [BT][kd]           per-head k slice
     *   headv  [BT][vd]           per-head v slice / out
     *   betas  [BT][v_heads]      per-token per-head beta
     *   decs   [BT][v_heads]      per-token per-head decay
     *   betah  [BT]               per-head beta staging
     *   dech   [BT]               per-head decay staging
     *   Dcum   [BT+1]             cumulative decay prefix
     *   S0n    [kd][vd]           incoming state (true domain)
     *   S1n    [kd][vd]           outgoing state (true domain)
     *   S0t    [kd][vd]           full serial-domain state scratch */
    long bneed = (long)H + qkv_rows + z_rows + o_rows +
                 (long)v_heads * vd + 2 * H + 1;
    /* Layout after bbuf (all floats):
     *   Amat BT·BT + delt BT·vd + headq BT·kd + headk BT·kd +
     *   headv BT·vd + betas BT·v_heads + decs BT·v_heads +
     *   betah BT + dech BT + Dcum BT+1 + S0n kd·vd + S1n kd·vd +
     *   S0t v_heads·kd·vd   (the full layer state working copy) */
    long extra = (long)M3_BT * M3_BT + (long)M3_BT * vd * 3 +
                 (long)M3_BT * kd * 2 + (long)M3_BT * v_heads * 2 +
                 (long)M3_BT + (long)M3_BT + (long)(M3_BT + 1) +
                 2 * (long)kd * vd + (long)v_heads * kd * vd;
    /* threaded per-worker staging: sub-chunks x nthreads per-head
     * sets (Amat BT^2 + delt BT*vd + headq/headk BT*kd + headv BT*vd
     * + betah/dech BT + Dcum BT+1 + S0n + S1n). */
    {
        int nth2 = kv->nthreads > 1 ? kv->nthreads : 1;
        if (nth2 > v_heads) nth2 = v_heads;
        long perh = (long)M3_BT * M3_BT + (long)M3_BT * vd * 2 +
                    (long)M3_BT * kd * 2 + (long)M3_BT + (long)M3_BT +
                    (long)(M3_BT + 1) + 2 * (long)kd * vd;
        long sub2 = ((long)B + M3_BT - 1) / M3_BT;
        extra += sub2 * nth2 * perh;
    }
    if (!kv->scratch || kv->scratch_n < bneed + extra) {
        fprintf(stderr, "[deltachunk] M3-full needs %ld floats, arena %ld "
                "-- falling back\n", bneed + extra, kv->scratch_n);
        return -1;
    }
    float *bbuf = kv->scratch;
    memset(bbuf, 0, (size_t)bneed * sizeof(float));
    float *Amat = bbuf + bneed;                  /* [BT][BT] */
    float *delt = Amat + (size_t)M3_BT * M3_BT;  /* [BT][vd] */
    float *headq = delt + (size_t)M3_BT * vd;    /* [BT][kd] */
    float *headk = headq + (size_t)M3_BT * kd;   /* [BT][kd] */
    float *headv = headk + (size_t)M3_BT * kd;   /* [BT][vd] */
    float *betas = headv + (size_t)M3_BT * vd;   /* [BT][v_heads] */
    float *decs = betas + (size_t)M3_BT * v_heads; /* [BT][v_heads] */
    float *betah = decs + (size_t)M3_BT * v_heads; /* [BT] */
    float *dech = betah + (size_t)M3_BT;         /* [BT] */
    float *Dcum = dech + (size_t)M3_BT;          /* [BT+1] */
    float *S0n = Dcum + (size_t)(M3_BT + 1);     /* [kd][vd] */
    float *S1n = S0n + (size_t)kd * vd;          /* [kd][vd] */
    /* S0t holds ALL v_heads states [v_heads][kd][vd] (the working
     * copy of kv->lin for this layer); sized accordingly. */
    float *S0t = S1n + (size_t)kd * vd;          /* [v_heads][kd][vd] */

    const uint16_t *cw = (const uint16_t *)(const void *)(tr +
                          tl->t[ci].off);
    const float *Al = (const float *)(const void *)(tr + tl->t[ai_].off);
    const uint16_t *dtb = (const uint16_t *)(const void *)(tr +
                          tl->t[di].off);
    const uint16_t *nw = (const uint16_t *)(const void *)(tr +
                          tl->t[ni].off);
    float *conv_ring = kv->conv + (size_t)L * 4 * qkv_rows;
    float *S = kv->lin + (size_t)L * v_heads * kd * vd;
    memcpy(S0t, S, (size_t)v_heads * kd * vd * sizeof(float));

    int sub = (B + M3_BT - 1) / M3_BT;
    for (int sc = 0; sc < sub; sc++) {
        int t0s = sc * M3_BT;
        int bt = (t0s + M3_BT < B) ? M3_BT : B - t0s;
        if (bt < 1) break;

        /* 1) per-token pre-body (serial over bt): replicate lin_body's
         * a/b gates, conv ring + silu, q/k norm, g32/beta/decay --
         * EXACT same math and op order (bit-identical per token). */
        for (int b = 0; b < bt; b++) {
            int tok = t0 + t0s + b;
            float *xin = xins[t0s + b];
            float *qkv = qkvs + (size_t)(t0s + b) * qkv_rows;
            float a32[32], b32[32], tmp[32];
            /* a/b gate projections (R=32, from xin); precision-generic
             * dispatch: 8-bit affine via salt_q8_matvec, else the
             * 4-bit kernel (mirrors the lin_body's q4_proj). */
            const uint16_t *as_bias = ab >= 0
                ? (const uint16_t *)(const void *)(tr + tl->t[ab].off)
                : NULL;
            const uint16_t *bs_bias = bb >= 0
                ? (const uint16_t *)(const void *)(tr + tl->t[bb].off)
                : NULL;
            if (tl->t[ai].bits == 8) {
                salt_q8_matvec(
                    (const uint32_t *)(const void *)(tr + tl->t[ai].off),
                    (const uint16_t *)(const void *)(tr + tl->t[as_].off),
                    as_bias, 32, cols, xin, tmp);
                salt_q8_matvec(
                    (const uint32_t *)(const void *)(tr + tl->t[bi].off),
                    (const uint16_t *)(const void *)(tr + tl->t[bs_].off),
                    bs_bias, 32, cols, xin, a32);
            } else {
                salt_q4_matvec(
                    (const uint32_t *)(const void *)(tr + tl->t[ai].off),
                    (const uint16_t *)(const void *)(tr + tl->t[as_].off),
                    as_bias, 32, cols, xin, tmp);
                salt_q4_matvec(
                    (const uint32_t *)(const void *)(tr + tl->t[bi].off),
                    (const uint16_t *)(const void *)(tr + tl->t[bs_].off),
                    bs_bias, 32, cols, xin, a32);
            }
            memcpy(b32, a32, sizeof b32);   /* b = second proj result */
            memcpy(a32, tmp, sizeof a32);
            /* conv ring + silu (ring gets the RAW projection, then
             * qkv is overwritten with silu(conv) -- as lin_body) */
            int base = (tok % 4) * qkv_rows;
            memcpy(conv_ring + base, qkv, (size_t)qkv_rows * sizeof(float));
            for (int ch = 0; ch < qkv_rows; ch++) {
                float acc = 0.0f;
                for (int kk = 0; kk < 4; kk++) {
                    int tpos = tok - (3 - kk);
                    if (tpos < 0) continue;
                    int slot = (tpos % 4) * qkv_rows + ch;
                    float w = dc_bf16(cw[(size_t)ch * 4 + kk]);
                    acc += conv_ring[slot] * w;
                }
                float sig = 1.0f / (1.0f + salt_expf(-acc));
                qkv[ch] = acc * sig;
            }
            /* q/k norm (exact reference: eps on the MEAN) */
            for (int h = 0; h < k_heads; h++) {
                float *qh = qkv + (size_t)h * kd;
                float *kh = qkv + (size_t)(k_heads * kd) + (size_t)h * kd;
                for (int which = 0; which < 2; which++) {
                    float *vec = which ? kh : qh;
                    double ss = 0.0;
                    for (int i = 0; i < kd; i++)
                        ss += (double)vec[i] * vec[i];
                    float rms = sqrtf((float)(ss / (double)kd) + 1e-6f);
                    float scale = which ? (1.0f / sqrtf((float)kd))
                                        : (1.0f / (float)kd);
                    for (int i = 0; i < kd; i++)
                        vec[i] = vec[i] / rms * scale;
                }
            }
            /* g32 = -exp(A_log)*softplus(a+dt_bias); beta=sigmoid(b) */
            for (int h = 0; h < v_heads; h++) {
                float dt = dc_bf16(dtb[h]);
                float sp = (a32[h] + dt) > 0
                    ? (a32[h] + dt) + salt_log1pf(salt_expf(-(a32[h] + dt)))
                    : salt_log1pf(salt_expf(a32[h] + dt));
                betas[(size_t)b * v_heads + h] =
                    1.0f / (1.0f + salt_expf(-b32[h]));
                decs[(size_t)b * v_heads + h] =
                    salt_expf(-salt_expf(Al[h]) * sp);
            }
        }

        /* scratch layout after S0t: nthreads per-head working sets
         * (Amat BT·BT + delt BT·vd + headq BT·kd + headk BT·kd +
         * headv BT·vd + betah BT + dech BT + Dcum BT+1 + S0n kd·vd +
         * S1n kd·vd). Grown at init (main.c's m3 sizing). */
        int nth = kv->nthreads > 1 ? kv->nthreads : 1;
        if (nth > v_heads) nth = v_heads;
        long perh = (long)M3_BT * M3_BT + (long)M3_BT * vd * 2 +
                    (long)M3_BT * kd * 2 + (long)M3_BT + (long)M3_BT +
                    (long)(M3_BT + 1) + 2 * (long)kd * vd;

        int sub = (B + M3_BT - 1) / M3_BT;
        for (int sc = 0; sc < sub; sc++) {
            int t0s = sc * M3_BT;
            int bt = (t0s + M3_BT < B) ? M3_BT : B - t0s;
            if (bt < 1) break;
            /* per-head work -- the 32 heads are INDEPENDENT (disjoint
             * state slices, disjoint readout rows, shared qkv/z reads):
             * split over nthreads workers, each with its own staging. */
            M3HeadCtx hc;
            hc.qkvs = qkvs; hc.zbtok = zbtok; hc.S0t = S0t;
            hc.readouts = readouts;
            hc.qkv_rows = qkv_rows; hc.z_rows = z_rows;
            hc.k_heads = k_heads; hc.v_heads = v_heads;
            hc.kd = kd; hc.vd = vd; hc.bt = bt; hc.t0s = t0s;
            hc.betas = betas; hc.decs = decs; hc.nw = nw;
            hc.hset = S0t + (size_t)v_heads * kd * vd +
                      (size_t)sc * nth * perh;
            if (nth > 1) {
                M3HeadJob jbs[M3_MAX_THREADS];
                M3HeadCtx wctx[M3_MAX_THREADS];
                pthread_t th[M3_MAX_THREADS];
                for (int w = 0; w < nth; w++) {
                    M3HeadJob *jb = &jbs[w];
                    M3HeadCtx *wc = &wctx[w];
                    *wc = hc;
                    wc->hset = S0t + (size_t)v_heads * kd * vd +
                               ((size_t)sc * nth + (size_t)w) * perh;
                    jb->ctx = wc;
                    jb->h0 = (w * v_heads) / nth;
                    jb->h1 = ((w + 1) * v_heads) / nth;
                    pthread_create(&th[w], NULL, m3_head_worker, jb);
                }
                for (int w = 0; w < nth; w++)
                    pthread_join(th[w], NULL);
            } else {
                M3HeadJob jb0;
                jb0.ctx = &hc;
                jb0.h0 = 0;
                jb0.h1 = v_heads;
                m3_head_worker(&jb0);
            }
        }
    }

    /* commit the updated state back to kv->lin */
    memcpy(S, S0t, (size_t)v_heads * kd * vd * sizeof(float));
    return 0;
}

int salt_delta_chunk_body(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                          int L, const uint8_t *tr, SaltKvCache *kv,
                          int t0, int B, float *const *states,
                          float *qkvs, float *zbtok, float *const *xins,
                          float *readouts) {
    const char *mode = getenv("SALT_DELTA_CHUNK");
    if (mode && strcmp(mode, "2") == 0) {
        /* Keep the experimental implementation compiled, but never execute
         * it: its normalization/A_log semantics and sub-chunk schedule are
         * not qualified against the production serial anchor. */
        (void)chunk_body_chunked;
        fprintf(stderr, "deltachunk: mode 2 is unqualified and disabled\n");
        return -1;
    }
    return chunk_body_serial(cfg, tl, L, tr, kv, t0, B, states,
                             qkvs, zbtok, xins, readouts);
}
