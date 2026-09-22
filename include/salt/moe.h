/*
 * moe.h -- trunk/pool tensor layouts + the real MoE compute step
 * (issue #2, milestone step 3).
 *
 * The layouts come from the converter (trunk.json) and the quantizer
 * (pool-mxfp4.json). The MoE step runs REAL math: router matvec on the
 * resident fp32 gate matrix, top-k selection, and the mxfp4 expert
 * chain w1 -> w2 -> w3 against the pool bytes the cache fetched.
 */
#ifndef SALT_MOE_H
#define SALT_MOE_H

#include "salt/salt.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#define SALT_MAX_LAYERS 256
#define SALT_MAX_TENSORS_PER_EXPERT 8

typedef struct SaltMoETensor {
    long dims[4];
    int  rank;
    long rel_v;         /* values offset, relative to slot start */
    long rel_s;         /* block scales offset, relative to slot */
    long rel_b;         /* biases offset (q4), relative to slot; -1 none */
    long v_nbytes, s_nbytes;
    int  bsize;         /* 16 or 32 (mxfp4) */
    int  fmt;           /* 0 = mxfp4, 1 = q4/q8 affine (bits), 2 = F8 */
    int  bits;          /* fmt=1: 4 or 8 (MLX affine width) */
    int  s_rank;        /* F8 (fmt=2): scale grid dims (SR,SC) from the
                           manifest s_shape -- down_proj is [16,4] while
                           gate/up are [4,16] */
    long s_dims[2];
} SaltMoETensor;

typedef struct SaltExpertLayout {
    int n;
    int chain;          /* 0 = DS-V4 sequential w1->w2->w3,
                           1 = Qwen3 parallel gate||up -> silu -> down */
    SaltMoETensor t[SALT_MAX_TENSORS_PER_EXPERT];
} SaltExpertLayout;

enum {
    SALT_GPU_VIEW_PARTITION_NONE = 0,
    SALT_GPU_VIEW_PARTITION_LAYER = 1,
};

typedef struct SaltLayoutSourceIdentity {
    uint64_t device, inode, nbytes, mtime_ns, ctime_ns;
    int present;
} SaltLayoutSourceIdentity;

typedef struct SaltGpuTensorPolicy {
    uint32_t block_size;
    uint32_t pipeline_id;
    uint32_t threadgroup_id;
    uint32_t flags;
} SaltGpuTensorPolicy;

typedef struct SaltGpuTensorOverride {
    uint32_t tensor_ordinal;     /* ordinal in the parent CPU tensor array */
    uint32_t layer, expert, component;
    SaltGpuTensorPolicy policy; /* root defaults with this override applied */
} SaltGpuTensorOverride;

typedef struct SaltGpuLayoutExtension {
    uint32_t version;
    uint32_t resource_alignment;
    uint32_t view_partition;
    SaltGpuTensorPolicy q4_default;
    SaltGpuTensorOverride *override;
    size_t n_override;
    int present;                /* parsed only for an engaged GPU reader */
} SaltGpuLayoutExtension;

typedef struct SaltPoolLayout {
    int      n_layers, n_experts;
    int64_t  expert_nbytes;
    int64_t  max_rc;            /* largest R*C over all expert tensors */
    int      expert_scale_rows; /* F8 (fmt=2): weight_scale_inv block rows */
    int      expert_scale_cols; /* F8 (fmt=2): weight_scale_inv block cols */
    SaltExpertLayout *exp;      /* [n_layers*n_experts] */
    uint8_t index_sha256[32];   /* exact bytes consumed by this parse */
    int index_sha256_valid;
    SaltLayoutSourceIdentity source_identity;
    SaltGpuLayoutExtension gpu;
} SaltPoolLayout;

/* Load pool-mxfp4.json (as written by tools/convert-salt.py quantize). */
int salt_pool_layout_load(SaltPoolLayout *pl, const char *path,
                          const SaltCfg *cfg);
/* Same authoritative parse, with optional GPU fields retrieved into a
 * process-local sidecar. CPU callers keep using salt_pool_layout_load and
 * allocate no GPU override table. */
int salt_pool_layout_load_ex(SaltPoolLayout *pl, const char *path,
                             const SaltCfg *cfg, int gpu_engaged);
void salt_pool_layout_free(SaltPoolLayout *pl);

typedef struct SaltTrunkTensor {
    char name[96];
    int  dtype;                 /* 0=F32, 1=I8, 2=F8_E4M3, 4=BF16, 3=other */
    int  bits;                  /* MLX affine bit width (4 or 8); U32 word
                                   packs 32/bits elements. Router gates on
                                   the 3.6 repo are 8-bit. */
    long dims[4];
    int  rank;
    long off;                   /* relative to layer payload start */
    long nbytes;
} SaltTrunkTensor;

/* Narrow a positive tensor row count only when it is representable by
 * the C kernel API. */
static inline int salt_trunk_tensor_rows(const SaltTrunkTensor *t,
                                         int *rows) {
    if (!t || !rows || t->rank < 1 || t->dims[0] < 1 ||
        t->dims[0] > INT_MAX)
        return -1;
    *rows = (int)t->dims[0];
    return 0;
}

/* Logical decoded column count. Quantized U32 tensors store packed words in
 * dims[1]; other tensors store logical columns directly. */
static inline long salt_trunk_tensor_cols(const SaltTrunkTensor *t) {
    if (!t || t->rank < 2 || t->dims[1] < 1) return -1;
    if (t->dtype == 2 || t->dtype == 4) return t->dims[1];
    int bits = t->bits > 0 ? t->bits : 4;
    if (bits != 4 && bits != 8) bits = 4;
    long ratio = 32 / bits;
    if (t->dims[1] > LONG_MAX / ratio) return -1;
    return t->dims[1] * ratio;
}

typedef struct SaltTrunkLayout {
    int   n_layers;
    SaltTrunkTensor *t;         /* flat, layer-major */
    int  *t_off;                /* [n_layers+1] index into t */
    int   gate[SALT_MAX_LAYERS]; /* tensor idx or -1 */
    int   gate_bias[SALT_MAX_LAYERS];
    int   down[SALT_MAX_LAYERS];
    int   up[SALT_MAX_LAYERS];
    /* MLA attention roles (issue #6); _s = the E8M0 scale sibling */
    int   attn_qn[SALT_MAX_LAYERS];
    int   attn_kvn[SALT_MAX_LAYERS];
    int   attn_wqa[SALT_MAX_LAYERS],  attn_wqa_s[SALT_MAX_LAYERS];
    int   attn_wqb[SALT_MAX_LAYERS],  attn_wqb_s[SALT_MAX_LAYERS];
    int   attn_wkv[SALT_MAX_LAYERS],  attn_wkv_s[SALT_MAX_LAYERS];
    int   attn_woa[SALT_MAX_LAYERS],  attn_woa_s[SALT_MAX_LAYERS];
    int   attn_wob[SALT_MAX_LAYERS],  attn_wob_s[SALT_MAX_LAYERS];
    int   attn_woc[SALT_MAX_LAYERS],  attn_woc_s[SALT_MAX_LAYERS];
    int   attn_sink[SALT_MAX_LAYERS];
    int   attn_norm[SALT_MAX_LAYERS]; /* input_layernorm (BF16) */
    int   ffn_norm[SALT_MAX_LAYERS]; /* post_attention_layernorm */
    /* mHC (issue #6 step 6): per layer, three tensors per connection:
     * fn = the W projections (cols: pre, post, res), base = the S
     * biases (same order), scale = the alpha gating scalars
     * (alpha_pre, alpha_post, alpha_res -- NVIDIA bridge mapping).
     * n_hc = 1: the residual transform B is constrained to 1. */
    int   hc_attn_fn[SALT_MAX_LAYERS], hc_attn_base[SALT_MAX_LAYERS],
          hc_attn_scale[SALT_MAX_LAYERS];
    int   hc_ffn_fn[SALT_MAX_LAYERS],  hc_ffn_base[SALT_MAX_LAYERS],
          hc_ffn_scale[SALT_MAX_LAYERS];
    /* global HC head (learned output contraction over the streams) */
    int   hc_head_fn, hc_head_base, hc_head_scale;
    /* Qwen3.5 attention: GQA (self_attn q/k/v/o) + linear_attn
     * (conv1d, A_log, dt_bias, in_proj_qkv/z/a/b, out_proj, norm).
     * Each projection has weight + scales + biases triplet. */
    int q3_q[SALT_MAX_LAYERS], q3_qs[SALT_MAX_LAYERS], q3_qb[SALT_MAX_LAYERS];
    int q3_k[SALT_MAX_LAYERS], q3_ks[SALT_MAX_LAYERS], q3_kb[SALT_MAX_LAYERS];
    int q3_v[SALT_MAX_LAYERS], q3_vs[SALT_MAX_LAYERS], q3_vb[SALT_MAX_LAYERS];
    int q3_o[SALT_MAX_LAYERS], q3_os[SALT_MAX_LAYERS], q3_ob[SALT_MAX_LAYERS];
    int q3_qn[SALT_MAX_LAYERS], q3_kn[SALT_MAX_LAYERS];
    int q3_conv[SALT_MAX_LAYERS], q3_a_log[SALT_MAX_LAYERS];
    int q3_dt[SALT_MAX_LAYERS];
    int q3_pqkv[SALT_MAX_LAYERS], q3_pqkvs[SALT_MAX_LAYERS],
        q3_pqkvb[SALT_MAX_LAYERS];
    int q3_pz[SALT_MAX_LAYERS], q3_pzs[SALT_MAX_LAYERS],
        q3_pzb[SALT_MAX_LAYERS];
    int q3_pa[SALT_MAX_LAYERS], q3_pas[SALT_MAX_LAYERS],
        q3_pab[SALT_MAX_LAYERS];
    int q3_pb[SALT_MAX_LAYERS], q3_pbs[SALT_MAX_LAYERS],
        q3_pbb[SALT_MAX_LAYERS];
    int q3_opa[SALT_MAX_LAYERS], q3_opas[SALT_MAX_LAYERS],
        q3_opab[SALT_MAX_LAYERS];
    int q3_lnorm[SALT_MAX_LAYERS];
    /* Qwen3.5 shared expert: dense every-token MLP added after the
     * routed experts -- y += sigmoid(shared_expert_gate(x)) *
     * shared_expert(x). Triplets like the routed experts. */
    int se_g[SALT_MAX_LAYERS], se_gs[SALT_MAX_LAYERS], se_gb[SALT_MAX_LAYERS];
    int se_u[SALT_MAX_LAYERS], se_us[SALT_MAX_LAYERS], se_ub[SALT_MAX_LAYERS];
    int se_d[SALT_MAX_LAYERS], se_ds[SALT_MAX_LAYERS], se_db[SALT_MAX_LAYERS];
    int se_r[SALT_MAX_LAYERS], se_rs[SALT_MAX_LAYERS], se_rb[SALT_MAX_LAYERS];
    int   final_norm;            /* final norm before lm_head (-1 = none) */
    int   kvlat;                /* wkv output dim (0 = no attention) */
} SaltTrunkLayout;

/* Load trunk.json (as written by tools/convert-salt.py convert). */
int salt_trunk_layout_load(SaltTrunkLayout *tl, const char *path);

/* mHC (DeepSeek-V4 paper eq. 1-8, n_hc general):
 *   X_{l+1} = B·X + C·F(A·X)     X in R^{n_hc x H}, vec(X) = state
 *   xhat = RMSNorm(vec(X))
 *   A = sigmoid(alpha_pre * (xhat . W_pre) + S_pre)          [n_hc]
 *   C = 2*sigmoid(alpha_post * (xhat . W_post) + S_post)     [n_hc]
 *   B = Sinkhorn(exp(alpha_res * Mat(xhat . W_res) + S_res)) [n_hc x n_hc]
 * fn = [(n_hc*(2+n_hc)) x (n_hc*H)]: rows [0,n_hc) W_pre,
 * [n_hc,2n_hc) W_post, [2n_hc,..) W_res (the checkpoint stores the
 * projections transposed). base = [n_hc*(2+n_hc)] (S in the same row
 * order), scale = [3] (alpha_pre, alpha_post, alpha_res).
 * Returns n_hc (>0) with A/C/B set; 0 when absent; -1 on mismatch. */
int salt_hc_params(const SaltTrunkLayout *tl, int fn_i, int base_i,
                   int sc_i, const uint8_t *tr, int H,
                   const float *state, int *n_hc_out,
                   float *A, float *C, float *B);

/* Combine the n_hc residual streams with A into the layer input
 * x_in[i] = sum_j A[j] * state[j*H + i]. Needs A from hc_params;
 * x_in must hold H floats. */
void salt_hc_combine(int n_hc, int H, const float *A, const float *state,
                     float *x_in);

/* Top-k over scores: descending, tie-break by expert index (earlier
 * expert wins ties -- deterministic). */
void salt_topk(const float *scores, int E, int k, int *idx, float *w);

/* Real MoE step for layer L. state[hidden] in/out. tr = trunk layer
 * payload; es[j] = cache slot payload for sel[j] (topk, already
 * fetched). scratch holds max_rc floats; job_scratch[k] for k in
 * [0, topk) holds another max_rc floats each, allocated ONCE by the
 * caller (the parallel expert chains must not malloc per call --
 * fresh mmap'd scratch page-faults ~16 MB x topk x layers). Counters
 * accumulate. */
int salt_moe_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                  const uint8_t *tr, const SaltPoolLayout *pl,
                  const uint8_t *const *es, const int *sel, const float *wsel,
                  float *state, float *scratch, long scratch_n,
                  float *const *job_scratch,
                  int64_t *n_matvec, int64_t *n_decode,
                  const float *shared_sout, float shared_sgate);
/* M6 shared-expert batch: compute the dense shared-expert contribution
 * for B tokens (norm + se_g/se_u/se_d/se_r) in ONE batched kernel
 * pass -- the weights are identical for every token, so the dequant
 * amortizes 64x with zero routing diversity (unlike grouped routed
 * experts, which M2 measured slower). Outputs: sout[B][H] and
 * sgate[B]; pass each token's values to salt_moe_step to fold into
 * acc at the serial point (bit-identical residual order). Returns 0
 * on success, -1 if the layer has no shared expert or the kernel
 * falls back. */
int salt_moe_shared_batch(const SaltCfg *cfg, const SaltTrunkLayout *tl,
                          int L, const uint8_t *tr,
                          const float *const *states, int B,
                          float *sout, float *sgate);

typedef struct {
    double setup_s, select_s, group_s, gather_s;
    double gate_up_s, activation_s, down_pack_s, down_s;
    double scatter_s, shared_s, residual_s, glue_s;
    int64_t calls, groups, selections;
} SaltMoEBatchProfile;

/* Opt-in, operation-neutral grouped-MoE profile. Reset enables timing
 * for the calling thread; get returns exclusive aggregate buckets. */
void salt_moe_batch_profile_reset(void);
void salt_moe_batch_profile_get(SaltMoEBatchProfile *out);

typedef struct {
    double total_s, scheduler_window_s, combine_s;
    double shared_s, shared_post_window_s;
    int64_t calls, routed_jobs;
} SaltMoEDecodeProfile;

/* Decode-side wall buckets. scheduler_window_s runs from routed dispatch to
 * routed completion and includes any overlapping shared work. shared_s is
 * nested diagnostic work; only shared_post_window_s is exclusive. */
void salt_moe_decode_profile_reset(void);
void salt_moe_decode_profile_get(SaltMoEDecodeProfile *out);

/* The grouped-MoE crossover is model-owned and applies to each actual
 * chunk independently, including a short tail chunk. */
int salt_moe_batch_should_run(int requested, int min_b, int actual_b);

/* M2 batched moe for the chunked prefill: B tokens x topk selections
 * grouped by expert, one batched matvec per distinct expert (dequant
 * amortized over the group), combine in selection order. xins is
 * [H][B] column-major; es/sel/wsel are [B][topk]; states[B][H]
 * in/out (residual add). Returns -1 when the layer needs a path the
 * batch can't do (mHC / up tensor) -- caller falls back to serial
 * salt_moe_step per token. */
int salt_moe_step_batch(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                        const uint8_t *tr, const SaltPoolLayout *pl,
                        const float *xins, int B, int topk,
                        const int *sel, const float *wsel,
                        const uint8_t *const *es,
                        float *const *states,
                        int64_t *n_matvec, int64_t *n_decode);

#endif /* SALT_MOE_H */
