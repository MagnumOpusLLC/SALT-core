/* gate: MLA attention step vs hand-computed identity-weight reference
 * (issue #6 step 2). Builds a mini trunk payload + manual layout with
 * identity projections so every stage reduces to simple arithmetic:
 *   wq_a/wq_b/wkv/wo_b/wo_c = identity, wo_a = [x0,x1] -> out[:2],
 *   q_norm/kv_norm = 1.0, sink = 0, scales = 1.0.
 * Then: kv_latent = x[:4]; q = x[:4]; k = x[:2], v = x[2:4];
 * scores[0] = (x0^2 + x1^2)/2; softmax over 1 pos = 1;
 * out = v = x[2:4]; wo_a -> [x2,x3,0,0,0,0,0,0]; state += that. */
#include "salt/attn.h"
#include "salt/kernels.h"
#include "salt/model.h"
#include "salt/salt.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N (16)   /* max tensors in the mini layout */

static uint8_t *add_tensor(uint8_t *p, const uint8_t *data, long nbytes) {
    memcpy(p, data, (size_t)nbytes);
    return p + nbytes;
}

static void pool_count(int tid, void *arg) {
    int *seen = (int *)arg;
    seen[tid]++;
}

typedef struct {
    SaltKvCache *pool;
    int nested_rc;
    int nested_seen;
} PoolRecursive;

static void pool_recursive(int tid, void *arg) {
    PoolRecursive *r = (PoolRecursive *)arg;
    (void)tid;
    r->nested_rc = salt_attn_pool_run_n(r->pool, 1, pool_count,
                                        &r->nested_seen);
}

typedef struct {
    SaltKvCache *pool;
    int seen[4];
} PoolFlow;

static int pool_flow(void *arg) {
    PoolFlow *flow = (PoolFlow *)arg;
    return !flow ||
        salt_attn_pool_run_n(flow->pool, 2, pool_count, flow->seen) != 0 ||
        salt_attn_pool_run_n(flow->pool, 2, pool_count, flow->seen) != 0 ||
        salt_attn_pool_run_n(flow->pool, 4, pool_count, flow->seen) != 0 ||
        salt_attn_pool_run_n(flow->pool, 1, pool_count, flow->seen) != 0
        ? -1 : 0;
}

/* F8 identity row: row r has 1.0 at col r, 0 elsewhere; C cols */
static void f8_identity(uint8_t *out, int R, int C, int row0, int col0) {
    memset(out, 0, (size_t)R * C);
    for (int r = 0; r < R; r++)
        if (r + row0 < C + 0 && r + row0 < C)   /* identity within [C] */
            out[r * C + (r + col0)] = 0x38;     /* 1.0 in E4M3 */
}

int main(void) {
    SaltCfg cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.hidden = 8;
    cfg.latent = 4;
    cfg.moe_inter = 8;
    cfg.n_layers = 2;
    cfg.n_experts = 4;
    cfg.topk = 3;

    /* payload layout: H=8, qlat=4, kvlat=4, qdim=4, kvhalf=2 */
    long off = 0;
    long offs[16];
    uint8_t payload[4096];
    memset(payload, 0, sizeof payload);
    uint8_t *p = payload;
    /* 0 wq_a [4x8] F8 identity */
    f8_identity(p, 4, 8, 0, 0); offs[0] = off; p = add_tensor(p, payload + offs[0], 32); off += 32;
    /* 1 wq_a scale [1x1] = 127 */
    p[0] = 127; offs[1] = off; p += 1; off += 1;
    /* 2 wq_b [4x4] identity */
    f8_identity(p, 4, 4, 0, 0); offs[2] = off; p = add_tensor(p, payload + offs[2], 16); off += 16;
    /* 3 wq_b scale 127 */
    p[0] = 127; offs[3] = off; p += 1; off += 1;
    /* 4 wkv [4x8] identity */
    f8_identity(p, 4, 8, 0, 0); offs[4] = off; p = add_tensor(p, payload + offs[4], 32); off += 32;
    /* 5 wkv scale 127 */
    p[0] = 127; offs[5] = off; p += 1; off += 1;
    /* 6 wo_a [8x2]: rows 0,1 identity on cols 0,1; rest 0 */
    memset(p, 0, 16);
    p[0 * 2 + 0] = 0x38;
    p[1 * 2 + 1] = 0x38;
    offs[6] = off; p += 16; off += 16;
    /* 7 wo_a scale 127 */
    p[0] = 127; offs[7] = off; p += 1; off += 1;
    /* 8 wo_b [8x8] identity */
    f8_identity(p, 8, 8, 0, 0); offs[8] = off; p = add_tensor(p, payload + offs[8], 64); off += 64;
    /* 9 wo_b scale 127 */
    p[0] = 127; offs[9] = off; p += 1; off += 1;
    /* 10 wo_c [8x8] identity */
    f8_identity(p, 8, 8, 0, 0); offs[10] = off; p = add_tensor(p, payload + offs[10], 64); off += 64;
    /* 11 wo_c scale 127 */
    p[0] = 127; offs[11] = off; p += 1; off += 1;
    /* 12 q_norm [4] BF16 1.0 */
    { uint16_t one = 0x3F80; memcpy(p, &one, 2); memcpy(p + 2, &one, 2);
      memcpy(p + 4, &one, 2); memcpy(p + 6, &one, 2); }
    offs[12] = off; p += 8; off += 8;
    /* 13 kv_norm [4] BF16 1.0 */
    { uint16_t one = 0x3F80; memcpy(p, &one, 2); memcpy(p + 2, &one, 2);
      memcpy(p + 4, &one, 2); memcpy(p + 6, &one, 2); }
    offs[13] = off; p += 8; off += 8;
    /* 14 sink [64] F32 zeros */
    offs[14] = off; p += 256; off += 256;

    SaltTrunkLayout tl;
    memset(&tl, 0, sizeof tl);
    tl.n_layers = 1;
    tl.t = (SaltTrunkTensor *)calloc(16, sizeof(SaltTrunkTensor));
    tl.t_off = (int *)calloc(2, sizeof(int));
    tl.gate[0] = -1;
    tl.hc_attn_fn[0] = tl.hc_attn_base[0] = tl.hc_attn_scale[0] = -1;
    tl.hc_ffn_fn[0] = tl.hc_ffn_base[0] = tl.hc_ffn_scale[0] = -1;
    tl.attn_qn[0] = 12;
    tl.attn_kvn[0] = 13;
    tl.attn_wqa[0] = 0;  tl.attn_wqa_s[0] = 1;
    tl.attn_wqb[0] = 2;  tl.attn_wqb_s[0] = 3;
    tl.attn_wkv[0] = 4;  tl.attn_wkv_s[0] = 5;
    tl.attn_woa[0] = 6;  tl.attn_woa_s[0] = 7;
    tl.attn_wob[0] = 8;  tl.attn_wob_s[0] = 9;
    tl.attn_woc[0] = 10; tl.attn_woc_s[0] = 11;
    tl.attn_sink[0] = 14;
    tl.attn_norm[0] = -1;   /* the layer norms: not in the
                              fixture (the attention is tested raw) */
    tl.ffn_norm[0] = -1;
    tl.kvlat = 4;
    int idx = 0;
    /* wq_a [4,8] F8 */
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[0]; tl.t[idx].nbytes = 32; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[1]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 4;
    tl.t[idx].off = offs[2]; tl.t[idx].nbytes = 16; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[3]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 4; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[4]; tl.t[idx].nbytes = 32; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[5]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 2;
    tl.t[idx].off = offs[6]; tl.t[idx].nbytes = 16; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[7]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[8]; tl.t[idx].nbytes = 64; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[9]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 8; tl.t[idx].dims[1] = 8;
    tl.t[idx].off = offs[10]; tl.t[idx].nbytes = 64; tl.t[idx].dtype = 2; idx++;
    tl.t[idx].rank = 2; tl.t[idx].dims[0] = 1; tl.t[idx].dims[1] = 1;
    tl.t[idx].off = offs[11]; tl.t[idx].nbytes = 1; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 4; tl.t[idx].nbytes = 8;
    tl.t[idx].off = offs[12]; tl.t[idx].dtype = 4; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 4; tl.t[idx].nbytes = 8;
    tl.t[idx].off = offs[13]; tl.t[idx].dtype = 4; idx++;
    tl.t[idx].rank = 1; tl.t[idx].dims[0] = 64; tl.t[idx].nbytes = 256;
    tl.t[idx].off = offs[14]; tl.t[idx].dtype = 0; idx++;
    tl.t_off[0] = 0;
    tl.t_off[1] = idx;

    SaltKvCache kv;
    if (salt_kv_init(&kv, 1, 4, 4) != 0) { printf("kv init FAIL\n"); return 1; }

    /* Model-owned recurrent geometry is immutable once allocated, and
     * allocator products must refuse overflow rather than wrap. */
    {
        SaltCfg geom;
        SaltKvCache kg, huge, pool, pool_disabled, external;
        SaltKvCache pool_overflow, pool_width_overflow;
        float external_scratch[16];
        memset(&geom, 0, sizeof geom);
        geom.linear_num_key_heads = 1;
        geom.linear_num_value_heads = 7;
        geom.linear_key_head_dim = 5;
        geom.linear_value_head_dim = 3;
        if (salt_attn_linear_scratch_floats(&geom, 13, 31, 21, 13, 4)
            != 132 ||
            salt_attn_linear_scratch_floats(&geom, 13, 31, 21, 11, 4)
            >= 0) {
            printf("linear scratch geometry FAIL\n");
            return 1;
        }
        {
            SaltCfg gqa;
            int heads, kv_heads, head_dim, rope_dim;
            memset(&gqa, 0, sizeof gqa);
            gqa.hidden = 96;
            gqa.n_heads = 6;
            gqa.n_kv_heads = 2;
            gqa.qk_rope = 8;
            gqa.rope_theta = 100000.0;
            if (salt_model_gqa_geometry(&gqa, 192, 32, 32, 96, 96,
                                        &heads, &kv_heads, &head_dim,
                                        &rope_dim) != 0 ||
                heads != 6 || kv_heads != 2 || head_dim != 16 ||
                rope_dim != 8 ||
                salt_attn_gqa_scratch_floats(&gqa, 192, 32, 32, 96,
                                              96, 17) != 882 ||
                salt_attn_gqa_scratch_floats(&gqa, 192, 32, 31, 96,
                                              96, 17) >= 0 ||
                salt_attn_gqa_scratch_floats(&gqa, 192, 32, 32, 96,
                                              95, 17) >= 0) {
                printf("GQA scratch geometry FAIL\n");
                return 1;
            }
            gqa.qk_rope = 9;
            if (salt_attn_gqa_scratch_floats(&gqa, 192, 32, 32, 96,
                                              96, 17) >= 0) {
                printf("GQA rotary width guard FAIL\n");
                return 1;
            }
            gqa.qk_rope = 8;
            gqa.rope_theta = 0.0;
            if (salt_attn_gqa_scratch_floats(&gqa, 192, 32, 32, 96,
                                              96, 17) >= 0) {
                printf("GQA rotary theta guard FAIL\n");
                return 1;
            }
        }
        if (salt_kv_init(&kg, 2, 4, 4) != 0 ||
            salt_kv_lin_init(&kg, 2, 3, 5) != 0 ||
            salt_kv_lin_init(&kg, 2, 3, 5) != 0 ||
            salt_kv_lin_init(&kg, 2, 3, 6) == 0 ||
            salt_kv_conv_init(&kg, 17) != 0 ||
            salt_kv_conv_init(&kg, 17) != 0 ||
            salt_kv_conv_init(&kg, 18) == 0) {
            printf("recurrent geometry guard FAIL\n");
            return 1;
        }
        salt_kv_free(&kg);
        const char *prior_pool_env = getenv("SALT_Q4_MATVEC2_POOL");
        char *saved_pool_env = prior_pool_env ? strdup(prior_pool_env) : NULL;
        if ((prior_pool_env && !saved_pool_env) ||
            setenv("SALT_Q4_MATVEC2_POOL", "1", 1) != 0)
            return 1;
        if (salt_kv_init(&pool, 1, 1, 2) != 0) return 1;
        pool.nthreads = 2;
        pool.apool_threads = 4;
        pool.aq4_threads = 4;
        if (salt_attn_pool_init(&pool) != 0 || pool.ath_count != 4 ||
            !pool.aq4_pool_enabled) {
            printf("attention pool init FAIL\n");
            return 1;
        }
        {
            int seen[2] = {0, 0};
            salt_attn_pool_run(&pool, pool_count, seen);
            salt_attn_pool_run(&pool, pool_count, seen);
            if (seen[0] != 2 || seen[1] != 2 ||
                salt_attn_pool_run_n(&pool, 0, pool_count, seen) == 0 ||
                salt_attn_pool_run_n(&pool, 5, pool_count, seen) == 0 ||
                salt_attn_pool_init(&pool) == 0) {
                printf("attention pool lifecycle FAIL\n");
                return 1;
            }
            PoolRecursive recursive = { .pool = &pool, .nested_rc = 0,
                                        .nested_seen = 0 };
            if (salt_attn_pool_run_n(&pool, 1, pool_recursive,
                                     &recursive) != 0 ||
                recursive.nested_rc == 0 || recursive.nested_seen != 0) {
                printf("attention pool recursive submit guard FAIL\n");
                return 1;
            }
        }
        {
            int seen[4] = {0, 0, 0, 0};
            uint64_t generation = pool.abatch;
            SaltAttnPoolStats before, after;
            int soa_enabled = getenv("SALT_CPU_SOA") != NULL;
            PoolRecursive recursive = { .pool = &pool, .nested_rc = 0,
                                        .nested_seen = 0 };
            if (salt_attn_pool_graph_end(&pool) == 0 ||
                salt_attn_pool_stats(&pool, &before) != 0 ||
                salt_attn_pool_graph_begin(&pool) != 0 ||
                pool.abatch != generation + 1u || !pool.agraph_running ||
                salt_attn_pool_run_n(&pool, 2, pool_count, seen) != 0 ||
                salt_attn_pool_run_n(&pool, 2, pool_count, seen) != 0 ||
                salt_attn_pool_run_n(&pool, 4, pool_count, seen) != 0 ||
                seen[0] != 3 || seen[1] != 3 ||
                seen[2] != 1 || seen[3] != 1 ||
                salt_attn_pool_run_n(&pool, 1, pool_recursive,
                                     &recursive) != 0 ||
                recursive.nested_rc == 0 || recursive.nested_seen != 0 ||
                salt_attn_pool_graph_end(&pool) != 0 ||
                salt_attn_pool_stats(&pool, &after) != 0 ||
                pool.agraph_running || pool.arunning ||
                pool.abatch != generation + 1u ||
                after.graph_sessions - before.graph_sessions != 1u ||
                after.resident_workers != 4u || after.base_workers != 2u ||
                after.wide_workers != 2u ||
                (soa_enabled ?
                    (after.phase_actions - before.phase_actions != 4u ||
                     after.worker_actions - before.worker_actions != 9u ||
                     after.worker_slot_capacity - before.worker_slot_capacity != 16u ||
                     after.worker_slot_headroom - before.worker_slot_headroom != 7u ||
                     after.flow_lock_acquires - before.flow_lock_acquires != 7u ||
                     after.flow_signal_calls - before.flow_signal_calls != 9u ||
                     after.rejected_actions - before.rejected_actions != 1u ||
                     after.peak_active_workers != 4u ||
                     after.worker_action_sum_s <= before.worker_action_sum_s ||
                     after.worker_critical_s <= before.worker_critical_s) :
                    (after.phase_actions != before.phase_actions ||
                     after.worker_actions != before.worker_actions ||
                     after.worker_slot_capacity != before.worker_slot_capacity ||
                     after.worker_slot_headroom != before.worker_slot_headroom ||
                     after.flow_lock_acquires != before.flow_lock_acquires ||
                     after.flow_signal_calls != before.flow_signal_calls ||
                     after.rejected_actions != before.rejected_actions ||
                     after.peak_active_workers != 0u ||
                     after.worker_action_sum_s != before.worker_action_sum_s ||
                     after.worker_critical_s != before.worker_critical_s))) {
                printf("attention pool graph session FAIL\n");
                return 1;
            }
            printf("attention pool SoA profile: PASS enabled=%d actions=4 "
                   "worker_slots=9 capacity=16 headroom=7\n",
                   soa_enabled);
        }
        {
            PoolFlow flow = { .pool = &pool, .seen = {0, 0, 0, 0} };
            SaltAttnPoolStats before, after;
            int soa_enabled = getenv("SALT_CPU_SOA") != NULL;
            if (salt_attn_pool_stats(&pool, &before) != 0 ||
                salt_attn_pool_flow_run(&pool, pool_flow, &flow) != 0 ||
                salt_attn_pool_stats(&pool, &after) != 0 ||
                flow.seen[0] != 4 || flow.seen[1] != 3 ||
                flow.seen[2] != 1 || flow.seen[3] != 1 ||
                after.flow_sessions - before.flow_sessions != 1u ||
                (soa_enabled ?
                    (after.phase_actions - before.phase_actions != 4u ||
                     after.worker_actions - before.worker_actions != 9u ||
                     after.worker_slot_capacity - before.worker_slot_capacity != 16u ||
                     after.worker_slot_headroom - before.worker_slot_headroom != 7u ||
                     after.flow_lock_acquires - before.flow_lock_acquires != 1u ||
                     after.flow_signal_calls - before.flow_signal_calls != 2u ||
                     after.internal_lock_acquires - before.internal_lock_acquires != 1u ||
                     after.internal_signal_calls - before.internal_signal_calls != 4u) :
                    (after.phase_actions != before.phase_actions ||
                     after.flow_lock_acquires != before.flow_lock_acquires ||
                     after.internal_lock_acquires != before.internal_lock_acquires))) {
                printf("attention pool one-caller flow FAIL\n");
                return 1;
            }
            printf("attention pool one-caller flow: PASS enabled=%d\n",
                   soa_enabled);
        }
        {
            float queries[12], keys0[6], keys1[6];
            float values0[6], values1[6];
            float scalar[12], parallel[12];
            SaltAttnHeadOperation operation;
            SaltAttnPoolStats before, after;
            int soa_enabled = getenv("SALT_CPU_SOA") != NULL;
            for (int i = 0; i < 12; i++)
                queries[i] = (float)(i - 5) * 0.125f;
            for (int i = 0; i < 6; i++) {
                keys0[i] = (float)(i + 1) * 0.0625f;
                keys1[i] = (float)(7 - i) * -0.03125f;
                values0[i] = (float)(i - 2) * 0.25f;
                values1[i] = (float)(3 - i) * 0.1875f;
            }
            memset(&operation, 0, sizeof operation);
            memset(scalar, 0, sizeof scalar);
            memset(parallel, 0, sizeof parallel);
            operation.queries = queries;
            operation.head_count = 4;
            operation.head_dim = 3;
            operation.kv_heads = 2;
            operation.query_stride = 3;
            operation.output_stride = 3;
            operation.kv_row_stride = 6;
            operation.span_count = 2;
            operation.spans[0] = (SaltAttnKvSpan) {
                .keys = keys0, .values = values0, .count = 1,
            };
            operation.spans[1] = (SaltAttnKvSpan) {
                .keys = keys1, .values = values1, .count = 1,
            };
            operation.outputs = scalar;
            if (salt_attn_set_head_parallel(0) != 0 ||
                salt_attn_heads_execute(&pool, &operation) != 0 ||
                salt_attn_pool_stats(&pool, &before) != 0 ||
                salt_attn_set_head_parallel(1) != 0)
                return 1;
            operation.outputs = parallel;
            if (salt_attn_heads_execute(&pool, &operation) != 0 ||
                salt_attn_pool_stats(&pool, &after) != 0 ||
                memcmp(scalar, parallel, sizeof scalar) != 0 ||
                salt_attn_set_head_parallel(2) == 0 ||
                (soa_enabled &&
                 after.phase_actions - before.phase_actions != 1u)) {
                printf("attention head parallel exactness FAIL\n");
                return 1;
            }
            printf("attention head parallel exactness: PASS enabled=%d\n",
                   soa_enabled);
        }
        {
            enum { R1 = 5, R2 = 3, C = 64 };
            uint32_t v1[R1 * C / 8], v2[R2 * C / 8];
            uint16_t s1[R1 * C / 64], s2[R2 * C / 64];
            uint16_t b1[R1 * C / 64], b2[R2 * C / 64];
            float x[C], want1[R1], want2[R2], got1[R1], got2[R2];
            for (int i = 0; i < R1 * C / 8; i++)
                v1[i] = 0x11111111u * (uint32_t)(i / (C / 8) + 1);
            for (int i = 0; i < R2 * C / 8; i++)
                v2[i] = 0x11111111u * (uint32_t)(i / (C / 8) + 5);
            for (int i = 0; i < R1 * C / 64; i++) {
                s1[i] = 0x3f80; b1[i] = 0;
            }
            for (int i = 0; i < R2 * C / 64; i++) {
                s2[i] = 0x3f80; b2[i] = 0;
            }
            for (int i = 0; i < C; i++) x[i] = 1.0f;
            if (salt_q4_matvec2(v1, s1, b1, R1, v2, s2, b2, R2, C,
                                x, want1, want2) != 0) {
                printf("attention pool q4 reference FAIL\n");
                return 1;
            }
            if (salt_attn_q4_matvec2(&pool, v1, s1, b1, INT_MAX - 1,
                                     v2, s2, b2, 1, C, x,
                                     got1, got2) == 0) {
                printf("attention pool q4 extreme geometry guard FAIL\n");
                return 1;
            }
            uint64_t generation = pool.abatch;
            if (salt_attn_q4_matvec2(&pool, v1, s1, b1, R1,
                                     v2, s2, b2, R2, C, x,
                                     got1, got2) != 0 ||
                pool.abatch != generation + 1 ||
                memcmp(got1, want1, sizeof got1) != 0 ||
                memcmp(got2, want2, sizeof got2) != 0) {
                printf("attention pool q4 first generation FAIL\n");
                return 1;
            }
            int seen[2] = {0, 0};
            salt_attn_pool_run(&pool, pool_count, seen);
            if (pool.abatch != generation + 2 ||
                seen[0] != 1 || seen[1] != 1 ||
                salt_attn_q4_matvec2(&pool, v1, s1, b1, R1,
                                     v2, s2, b2, R2, C, x,
                                     got1, got2) != 0 ||
                pool.abatch != generation + 3 ||
                memcmp(got1, want1, sizeof got1) != 0 ||
                memcmp(got2, want2, sizeof got2) != 0) {
                printf("attention pool q4 generation/exactness FAIL\n");
                return 1;
            }
        }
        salt_kv_free(&pool);
        if (setenv("SALT_Q4_MATVEC2_POOL", "0", 1) != 0 ||
            salt_kv_init(&pool_disabled, 1, 1, 2) != 0)
            return 1;
        pool_disabled.nthreads = 2;
        pool_disabled.aq4_threads = 4;
        if (salt_attn_pool_init(&pool_disabled) != 0 ||
            pool_disabled.aq4_pool_enabled ||
            pool_disabled.apool_threads != 2 ||
            pool_disabled.ath_count != 2) {
            printf("attention pool disabled lifecycle FAIL\n");
            return 1;
        }
        salt_kv_free(&pool_disabled);
        if (saved_pool_env) {
            if (setenv("SALT_Q4_MATVEC2_POOL", saved_pool_env, 1) != 0)
                return 1;
        } else if (unsetenv("SALT_Q4_MATVEC2_POOL") != 0) {
            return 1;
        }
        free(saved_pool_env);
        memset(&pool_overflow, 0, sizeof pool_overflow);
        pool_overflow.nthreads = INT_MAX;
        pool_overflow.max_tokens = INT_MAX;
        if (salt_attn_pool_init(&pool_overflow) == 0) {
            printf("attention pool overflow guard FAIL\n");
            return 1;
        }
        salt_kv_free(&pool_overflow);
        memset(&pool_width_overflow, 0, sizeof pool_width_overflow);
        pool_width_overflow.nthreads = 2;
        pool_width_overflow.apool_threads = 257;
        pool_width_overflow.max_tokens = 2;
        if (salt_attn_pool_init(&pool_width_overflow) == 0) {
            printf("attention pool width overflow guard FAIL\n");
            return 1;
        }
        salt_kv_free(&pool_width_overflow);
        memset(&external, 0, sizeof external);
        if (salt_kv_scratch_init(&external, 8) != 0 ||
            salt_kv_scratch_bind(&external, external_scratch, 16) == 0)
            return 1;
        salt_kv_free(&external);
        memset(&external, 0, sizeof external);
        if (salt_kv_scratch_bind(&external, external_scratch, 16) != 0 ||
            external.scratch != external_scratch || external.scratch_n != 16 ||
            external.scratch_owned != 0 ||
            salt_kv_scratch_bind(&external, external_scratch, 16) == 0 ||
            salt_kv_scratch_unbind(&external, external_scratch + 1) == 0 ||
            salt_kv_scratch_unbind(&external, external_scratch) != 0 ||
            external.scratch != NULL || external.scratch_n != 0) {
            printf("external scratch ownership FAIL\n");
            return 1;
        }
        salt_kv_free(&external);
        memset(&huge, 0, sizeof huge);
        huge.n_layers = 0x7fffffff;
        if (salt_kv_lin_init(&huge, 0x7fffffff, 0x7fffffff,
                             0x7fffffff) == 0 ||
            salt_kv_conv_init(&huge, 0x7fffffff) == 0 ||
            salt_kv_scratch_init(&huge, LONG_MAX) == 0) {
            printf("recurrent geometry overflow guard FAIL\n");
            return 1;
        }
    }

    /* naive reference mirroring the step's formulas (identity weights):
     * ql = kv_lat = x[:4]; RMSNorm -> n; k = n[:2], v = n[2:4];
     * score = (q.k)/2 (qdim 4); softmax; out = sum w*v; wo_a puts
     * out into rows 0,1; state += that. */
    float n0cache[4];

    /* token 0: state = [1,2,3,4,0,0,0,0] */
    float state[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    float want0[8];
    {
        double ss = 0;
        for (int i = 0; i < 4; i++) ss += (double)state[i] * state[i];
        float r = sqrtf((float)(ss / 4.0) + 1e-6f);
        float n[4];
        for (int i = 0; i < 4; i++) { n[i] = state[i] / r; n0cache[i] = n[i]; }
        for (int i = 0; i < 8; i++) want0[i] = state[i];
        want0[0] += n[2];   /* wo_a: out = v = n[2:4] into rows 0,1 */
        want0[1] += n[3];
    }
    int rc = salt_attn_step(&cfg, &tl, 0, payload, state, &kv, 0);
    if (rc != 0) { printf("attn step FAIL rc=%d\n", rc); return 1; }
    for (int i = 0; i < 8; i++) {
        if (fabsf(state[i] - want0[i]) > 1e-4f) {
            printf("token0 state[%d] = %g, want %g\n", i, state[i], want0[i]);
            return 1;
        }
    }

    /* token 1: cache has token0 n; state2 = [1,0,0,0,5,6,0,0] */
    float state2[8] = {1, 0, 0, 0, 5, 6, 0, 0};
    float want1[8];
    {
        double ss = 0;
        for (int i = 0; i < 4; i++) ss += (double)state2[i] * state2[i];
        float r = sqrtf((float)(ss / 4.0) + 1e-6f);
        float n[4];
        for (int i = 0; i < 4; i++) n[i] = state2[i] / r;
        float s0 = (n0cache[0] * n[0] + n0cache[1] * n[1]) * 0.5f;
        float s1 = (n[0] * n[0] + n[1] * n[1]) * 0.5f;
        float mx = s0 > s1 ? s0 : s1;
        float e0 = expf(s0 - mx), e1 = expf(s1 - mx);
        float w0 = e0 / (e0 + e1), w1 = e1 / (e0 + e1);
        for (int i = 0; i < 8; i++) want1[i] = state2[i];
        want1[0] += w0 * n0cache[2] + w1 * n[2];
        want1[1] += w0 * n0cache[3] + w1 * n[3];
    }
    rc = salt_attn_step(&cfg, &tl, 0, payload, state2, &kv, 1);
    if (rc != 0) { printf("attn step2 FAIL rc=%d\n", rc); return 1; }
    for (int i = 0; i < 8; i++) {
        if (fabsf(state2[i] - want1[i]) > 1e-4f) {
            printf("token1 state[%d] = %g, want %g\n", i, state2[i], want1[i]);
            return 1;
        }
    }

    /* determinism: re-run token 0 on a fresh state, identical result */
    {
        float s[8] = {1, 2, 3, 4, 0, 0, 0, 0};
        SaltKvCache kv2;
        salt_kv_init(&kv2, 1, 4, 4);
        salt_attn_step(&cfg, &tl, 0, payload, s, &kv2, 0);
        for (int i = 0; i < 8; i++)
            if (s[i] != want0[i]) {
                printf("determinism mismatch %d: %g vs %g\n",
                       i, s[i], want0[i]);
                return 1;
            }
        salt_kv_free(&kv2);
    }

    salt_kv_free(&kv);
    free(tl.t);
    free(tl.t_off);
    printf("test_attn ok\n");
    return 0;
}
