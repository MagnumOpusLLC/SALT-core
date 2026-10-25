/*
 * attn.h -- MLA attention step + KV cache (issue #6, step 2).
 *
 * The real V4-Flash attention is MLA (multi-head latent attention):
 *   ql = wq_a . x            (q latent)
 *   qn = RMSNorm(ql, q_norm)
 *   q  = wq_b . qn           (up-project to full width)
 *   kv = RMSNorm(wkv . x, kv_norm)      (kv latent, cached per token)
 *   k  = kv[:kvlat/2], v = kv[kvlat/2:]
 *   scores = q . k / sqrt(d) + sink[pos] (sink positions always attend)
 *   out    = softmax(scores) . v
 *   state += wo_c . wo_b . wo_a . out
 *
 * The KV cache is per-layer (each layer computes its own kv latent);
 * MLA's shared-cache memory optimization is the roadmap refinement.
 * All tensors come from the trunk layout (F8 + E8M0 group scales,
 * BF16 norms, F32 sink). Attention is skipped for layers whose graph
 * is incomplete, so the engine degrades gracefully.
 */
#ifndef SALT_ATTN_H
#define SALT_ATTN_H

#include "salt/salt.h"
#include "salt/area_scan.h"
#include "salt/moe.h"

typedef struct SaltAttnPoolStats {
    uint32_t resident_workers;
    uint32_t base_workers;
    uint32_t wide_workers;
    uint32_t peak_active_workers;
    uint64_t graph_sessions;
    uint64_t flow_sessions;
    uint64_t phase_actions;
    uint64_t worker_actions;
    uint64_t worker_slot_capacity;
    uint64_t worker_slot_headroom;
    uint64_t flow_lock_acquires;
    uint64_t flow_signal_calls;
    uint64_t flow_wait_calls;
    uint64_t graph_begin_wait_calls;
    uint64_t graph_end_wait_calls;
    uint64_t internal_lock_acquires;
    uint64_t internal_signal_calls;
    uint64_t internal_wait_calls;
    uint64_t rejected_actions;
    double flow_lock_wait_s;
    double flow_wait_s;
    double worker_action_sum_s;
    double worker_critical_s;
    double internal_lock_wait_s;
    double internal_wait_s;
} SaltAttnPoolStats;

enum { SALT_ATTN_HEAD_MAX_SPANS = 2 };

typedef struct SaltAttnKvSpan {
    const float *keys;
    const float *values;
    int count;
} SaltAttnKvSpan;

/* Resolve one canonical K/V row for tree-shaped target history. The resolver
 * returns a pointer already advanced to kv_head's first dimension. */
typedef const float *(*SaltAttnKvRowResolver)(
    const void *context, int value_row, int position,
    int kv_head, int head_dim);

typedef int (*SaltAttnSpanAccumulator)(
    float *output, const float *values, int value_stride,
    const float *scores, int count, int dimension, float denominator);

/* One pure per-head arithmetic fold. Exactly one source mode is selected:
 * contiguous canonical spans (ordinary B1 and grouped prefill), or a row
 * resolver (target tree history). score_scale is explicit so adapters retain
 * their qualified semantics. */
typedef struct SaltAttnHeadFold {
    const float *query;
    float *output;
    float *scores;
    int row_count;
    int head_dim;
    int kv_head;
    float score_scale;
    int kv_row_stride;
    const SaltAttnKvSpan *spans;
    int span_count;
    SaltAttnSpanAccumulator span_accumulate;
    SaltAttnKvRowResolver row_resolver;
    const void *row_context;
} SaltAttnHeadFold;

int salt_attn_head_fold(const SaltAttnHeadFold *fold);

/* One exact MHA/GQA body over one or two contiguous canonical KV spans. The
 * model adapter supplies geometry and storage views; the engine owns worker
 * assignment, score scratch, reduction order, and output placement. */
typedef struct SaltAttnHeadOperation {
    const float *queries;
    float *outputs;
    int head_count;
    int head_dim;
    int kv_heads;
    int query_stride;
    int output_stride;
    int kv_row_stride;
    int span_count;
    SaltAttnKvSpan spans[SALT_ATTN_HEAD_MAX_SPANS];
} SaltAttnHeadOperation;

typedef struct SaltKvCache {
    float *kv;              /* [n_layers * max_tokens * kvlat] (MLA/GQA) */
    int    n_layers, kvlat, max_tokens;
    float *lin;             /* linear-attn recurrent state
                              [n_layers][v_heads][kd][vd] (Gated DeltaNet) */
    int    lin_vh, lin_kd, lin_vd, lin_alloc;
    float *conv;            /* linear-attn conv1d ring, PERSISTENT across
                              tokens: [n_layers][Q3_CONV_K][qkv_rows] */
    int    conv_rows, conv_alloc;
    float *scratch;         /* per-token work arena (GQA + linear attn),
                              allocated once at init: covers the worst
                              per-layer buffer need. Removes ~1.1M
                              malloc/free per 20K-token run (macOS keeps
                              freed large blocks at the zone high-water,
                              ~2.9 GB resident). */
    long   scratch_n;       /* floats */
    int    scratch_owned;   /* nonzero only for salt_kv_scratch_init storage */
    int    nthreads;        /* ordinary attention worker count (--threads) */
    /* persistent attention pool (2026-08-09): workers live across
     * spawn calls instead of pthread_create/join per layer. The old
     * per-call calloc of the sc/wg buffers is the real tax -- macOS
     * keeps freed large blocks at the zone high-water (~2.9GB
     * resident, same class of problem the scratch arena solves).
     * Batch semantics: a submit selects an active worker prefix; each
     * selected worker runs its tid slice and the submitter waits for
     * the complete prefix (done-barrier, not claim-based). */
    pthread_t       *ath;   /* [apool_threads] persistent workers */
    pthread_mutex_t amu;
    pthread_cond_t  acv_work, acv_wide, acv_done, acv_flow, acv_flow_work;
    int             apool_sync_init;
    int             ath_count;       /* successfully created workers */
    int             apool_threads;   /* resident workers; >= nthreads */
    int             aq4_threads;     /* SALT_ATTN_THREADS, fixed at init */
    int             aq4_pool_enabled;/* SALT_Q4_MATVEC2_POOL, fixed at init */
    int             aactive;         /* active workers in this generation */
    int             arunning;        /* one submitter owns the active generation */
    int             ashutdown;
    uint64_t        abatch;          /* monotonic active batch id */
    int             adone;           /* workers done in active batch */
    void          (*afn)(int tid, void *arg);
    void           *aarg;
    int             agraph_running;
    int             agraph_stop;
    int             agraph_ready;
    int             agraph_phase_done;
    int             agraph_phase_active;
    uint64_t        agraph_phase;
    void          (*agraph_fn)(int tid, void *arg);
    void           *agraph_arg;
    pthread_t       agraph_owner;
    uint64_t        agraph_sessions;
    uint64_t        agraph_phases;
    int             aflow_running;
    int             aflow_stop;
    int             aflow_result;
    int             aflow_phase_active;
    int             aflow_phase_done;
    int             aflow_lock_held;
    uint64_t        aflow_phase;
    int           (*aflow_coordinator_fn)(void *arg);
    void           *aflow_coordinator_arg;
    void          (*aflow_phase_fn)(int tid, void *arg);
    void           *aflow_phase_arg;
    pthread_t       aflow_coordinator;
    uint64_t        aflow_sessions;
    /* Optional graph-scoped team on this SAME resident pool. Phase payload
     * stays in aflow_*; only parking metadata is additional. Atomic compiler
     * adapters publish phases/completion, never floating-point reductions. */
    int             ateam_mode;
    int             ateam_sleepers;
    int             ateam_join_sleeping;
    int             asoa_profile;
    uint32_t        asoa_peak_active;
    uint64_t        asoa_phase_actions;
    uint64_t        asoa_worker_actions;
    uint64_t        asoa_worker_slot_capacity;
    uint64_t        asoa_worker_slot_headroom;
    uint64_t        asoa_flow_lock_acquires;
    uint64_t        asoa_flow_signal_calls;
    uint64_t        asoa_flow_wait_calls;
    uint64_t        asoa_graph_begin_wait_calls;
    uint64_t        asoa_graph_end_wait_calls;
    uint64_t        asoa_internal_lock_acquires;
    uint64_t        asoa_internal_signal_calls;
    uint64_t        asoa_internal_wait_calls;
    uint64_t        asoa_rejected_actions;
    double          asoa_flow_lock_wait_s;
    double          asoa_flow_wait_s;
    double          asoa_worker_action_sum_s;
    double          asoa_worker_critical_s;
    double          asoa_phase_worker_max_s;
    double          asoa_internal_lock_wait_s;
    double          asoa_internal_wait_s;
    float          *asc8, *awg8;  /* per-resident-worker buffers
                                     [apool_threads][8*max_tokens] */
    float          *aq4_rows;      /* [apool_threads][aq4_row_capacity] */
    size_t          aq4_row_capacity;
} SaltKvCache;

int  salt_kv_init(SaltKvCache *c, int n_layers, int kvlat, int max_tokens);
void salt_kv_free(SaltKvCache *c);
/* Forget policy (session mode): drop the exact per-token K/V rows
 * and the conv ring (the transcript), KEEP the linear delta states
 * (the compressed memory). The next token starts fresh rows at
 * position 0; the delta states carry the accumulated history. */
void salt_kv_forget(SaltKvCache *c);
int  salt_attn_pool_init(SaltKvCache *c);
void salt_attn_pool_run(SaltKvCache *c, void (*fn)(int tid, void *arg),
                        void *arg);
/* One submission may be active at a time. Concurrent or recursive
 * submissions are rejected with -1. */
int  salt_attn_pool_run_n(SaltKvCache *c, int nactive,
                          void (*fn)(int tid, void *arg), void *arg);
/* Execute independent attention heads on the existing resident pool. Every
 * score preserves dimension order; every softmax and weighted-value result
 * preserves canonical KV-position order. */
int  salt_attn_heads_execute(SaltKvCache *c,
                             const SaltAttnHeadOperation *operation);
/* Qualification control selected before worker/model startup. Production
 * default is enabled; disabled mode executes the same descriptor serially. */
int  salt_attn_set_head_parallel(int enabled);
/* Keep resident workers inside one caller-owned token generation. Existing
 * run_n calls become internal dependency phases; end is the one caller-visible
 * pool completion. */
int  salt_attn_pool_graph_begin(SaltKvCache *c);
int  salt_attn_pool_graph_end(SaltKvCache *c);
/* Run one caller-owned token/control function on worker 0. Calls to run_n made
 * by that coordinator execute worker 0 directly and advance workers 1..N-1
 * inside the same resident generation. The external caller waits only for the
 * complete flow. The coordinator callback and every phase argument must remain
 * valid until this synchronous call returns. */
int  salt_attn_pool_flow_run(SaltKvCache *c,
                             int (*fn)(void *arg), void *arg);
/* Same coordinator/phase contract as flow_run, but ready phases use atomic
 * publication/completion rather than a mutex/condvar rendezvous per operator.
 * Idle team members park using the existing pool condition variables. */
int  salt_attn_pool_team_run(SaltKvCache *c,
                             int (*fn)(void *arg), void *arg);
/* Execute one fixed-capacity W/F/Q scheduler inside one resident pool flow.
 * Every internal Q phase retains worker identity; the caller observes one
 * synchronous completion. */
int salt_attn_pool_wfq_execute(
    SaltKvCache *c, SaltAreaWfqRuntime *runtime,
    SaltAreaWfqItemExecute item_execute, void *item_context,
    SaltAreaWfqResult *result);
/* Read cumulative caller-flow and worker-action telemetry. Snapshot before and
 * after one token to obtain exact action/wait/headroom deltas. Worker idle
 * sleeps are intentionally excluded because they are scheduler-dependent. */
int  salt_attn_pool_stats(SaltKvCache *c, SaltAttnPoolStats *stats);
/* Decode-side fused qkv+z MLX4 projection on the KV-owned worker pool. */
int  salt_attn_q4_matvec2(SaltKvCache *c,
                          const uint32_t *v1, const uint16_t *s1,
                          const uint16_t *b1, int R1,
                          const uint32_t *v2, const uint16_t *s2,
                          const uint16_t *b2, int R2, int C,
                          const float *x, float *y1, float *y2);
/* Allocate the per-token attention work arena (floats). Call after
 * kv_init, before the token loop; frees with salt_kv_free. */
int  salt_kv_scratch_init(SaltKvCache *c, long floats);
/* Replace owned scratch during model initialization with fixed caller-owned
 * storage (for example one backend-shared scheduler seat). */
int  salt_kv_scratch_bind(SaltKvCache *c, float *scratch, long floats);
/* Detach the exact caller-owned allocation before its backend is destroyed. */
int  salt_kv_scratch_unbind(SaltKvCache *c, float *scratch);
/* Allocate the linear-attention state arena (v_heads x kd x vd per
 * layer). Returns 0 on success, -1 on failure. */
int  salt_kv_lin_init(SaltKvCache *c, int v_heads, int kd, int vd);
/* Allocate the linear-attn conv1d ring (4 x qkv_rows per layer). */
int  salt_kv_conv_init(SaltKvCache *c, int qkv_rows);

/* Required shared arena size for one GQA token. Returns floats, or -1 for
 * invalid model/tensor geometry or overflow. */
long salt_attn_gqa_scratch_floats(const SaltCfg *cfg, int q_rows,
                                  int k_rows, int v_rows, int o_rows,
                                  int o_cols, int max_tokens);

/* Required shared arena size for one linear-attention body. Returns floats,
 * or -1 for invalid geometry/overflow. */
long salt_attn_linear_scratch_floats(const SaltCfg *cfg, int hidden,
                                     int qkv_rows, int z_rows, int o_rows,
                                     int nthreads);

/* Attention for layer L. tr = trunk layer payload; state[hidden] in/out
 * (residual added). token = current token index (cache position).
 * Returns 0 (ok, possibly skipped), -1 on layout/config error. */
int salt_attn_step(const SaltCfg *cfg, const SaltTrunkLayout *tl, int L,
                   const uint8_t *tr, float *state, SaltKvCache *kv,
                   int token);

/* the current batch B (the gen: 1; the chunked prefill: the chunk B)
 * -- set by the engine's loops, read by the trunk's GPU routing so
 * the B=1 gen stays on the CPU while the big-B prefill offloads. */
void salt_attn_set_batch_b(int b);
int salt_attn_batch_b(void);

/* Accumulate one contiguous canonical value-row segment into one attention
 * head output. The caller zeroes output once and invokes segments in ascending
 * value-position order. The scalar path is the operation-order anchor; an
 * available SIMD backend vectorizes only disjoint output dimensions. */
int salt_attn_weighted_value_accumulate(
    float *output, const float *values, int value_stride,
    const float *scores, int count, int dimension, float denominator);
/* Qualification control selected before worker/model startup. Production
 * default is enabled; invalid values reject without changing the selection. */
int salt_attn_set_weighted_value_simd(int enabled);

#endif /* SALT_ATTN_H */
