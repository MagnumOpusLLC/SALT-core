/* LFM2.5-24B-A2B descriptor and geometry gate.
 *
 * Asserts the declared model-owned geometry and records the exact boundary of
 * what the current engine can plan. The conv layers are expected to REFUSE:
 * that refusal is the measured statement of the missing short-convolution
 * stage, not a placeholder. When the engine gains the stage this test flips to
 * requiring a successful plan, and the assertions below are where that change
 * lands. */

#include "salt/model.h"
#include "lfm.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;

/* The model registry declares this in models/registry.c; the descriptor test
 * links the model object directly. */
extern const SaltModelDesc *salt_lfm2_model_descriptor(void);

static void check(int ok, const char *what) {
    if (!ok) {
        printf("FAIL %s\n", what);
        failures++;
    } else {
        printf("ok   %s\n", what);
    }
}

static int float_eq(float a, float b) { return fabsf(a - b) <= 1e-6f * (1.0f + fabsf(b)); }

int main(void) {
    const SaltModelDesc *m = salt_lfm2_model_descriptor();
    const SaltTextGraphDesc *g;
    int attention_layers = 0, conv_layers = 0, plan_ok = 0, conv_plan_ok = 0;

    check(m != NULL, "descriptor present");
    if (!m)
        return 1;

    /* ---- declared identity ---------------------------------------------- */
    check(!strcmp(m->name, LFM_NAME), "name");
    check(m->n_layers == 40, "40 layers");
    check(m->n_experts == 64, "64 experts");
    check(m->topk == 4, "top-4 routing");
    check(m->n_shared == 0, "no shared expert");
    check(m->hidden == 2048, "hidden 2048");
    check(m->moe_inter == 1536, "expert intermediate 1536");
    check(m->n_heads == 32, "32 query heads");
    check(m->n_kv_heads == 8, "8 kv heads");
    check(m->qk_rope == 64, "head dimension 64");
    check(float_eq((float)m->rope_theta, 1e7f), "rope theta 1e7");
    check(m->think_open == LFM_THINK_OPEN && m->think_close == LFM_THINK_CLOSE,
          "thinking tokens 124901/124902");
    check(m->runtime_ready == 1, "runtime_ready after real-package qualification");
    check(m->state != NULL && m->state->mindset_kind == SALT_MINDSET_PREFIX,
          "mindset prefix state descriptor");
    check(m->state != NULL && m->state->facts_kind == SALT_FACTS_KV_ROWS,
          "facts are KV rows");

    /* ---- declared graph policy ------------------------------------------ */
    g = m->text_graph;
    check(g != NULL, "text graph present");
    if (!g)
        return 1;
    check(g->period == 4 && g->full_slot == 2, "attention exactly at i % 4 == 2");
    check(g->dense_intermediate == 11776, "dense intermediate 11776");
    check(g->dense_layers == 2, "dense branch on layers 0 and 1 only");
    check(g->expert_activation == SALT_ACT_SILU, "plain SiLU, no clamp");
    check(g->activation_limit == 0.0f, "no activation clamp limit");
    check(g->router_rmsnorm == 0, "no router rmsnorm");
    check(g->router_kind == SALT_ROUTER_MODEL,
          "sigmoid router with selection-only expert bias");
    check(g->router_normalize_topk == 1 && g->router_scale == 1.0f,
          "selected sigmoid weights normalize and retain scale 1");
    check(g->parallel_dense_routed == 0, "no parallel dense branch");
    check(g->residual_postnorm == 0, "no residual postnorm");
    check(g->final_layer_scale == 0, "no final layer scale");
    check(g->tied_embeddings == 1, "tied output head");
    check(g->logit_softcap == 0.0f, "no logit softcap");
    check(g->n_eos == 1 && g->eos[0] == LFM_EOS, "EOS 124900 only");
    check(g->full.kind == SALT_ATTN_FULL && g->full.causal == 1, "full causal attention");
    check(g->full.n_heads == 32 && g->full.n_kv_heads == 8 && g->full.head_dim == 64,
          "GQA 32/8/64");
    check(g->full.rope_kind == SALT_ROPE_DEFAULT && g->full.rope_dim == 64 &&
          g->full.rope_base_dim == 64, "full-head-dimension RoPE");
    check(g->full.window == 0, "no sliding window on attention layers");
    check(float_eq(g->full.score_scale, 0.125f), "score scale 64**-0.5");
    check(g->full.raw_values == 1, "raw values (no value rmsnorm)");
    check(g->sliding.kind == SALT_ATTN_MODEL,
          "conv layers declared as the conv kind");

    /* ---- derived package geometry --------------------------------------- */
    check(LFM_MOE_LAYERS == 38, "38 MoE layers");
    check(LFM_EXPERT_COUNT == 2432, "2432 expert slots");
    check(LFM_EXPERT_BYTES == 5308416u, "expert slot 5308416 bytes");
    check(LFM_EXPERT_BYTES % 4096u == 0u, "expert slot is page aligned");
    check(LFM_POOL_BYTES == 12910071808ull, "pool extent 12910071808 bytes");
    check(lfm_kv_floats(2048) * 4u == 83886080ull, "live KV at CTX2048 is 83886080 bytes");
    check(lfm_kv_floats(512) * 4u == 20971520ull, "live KV at CTX512 is 20971520 bytes");
    check(lfm_conv_state_floats() == 184320ull, "conv state is 184320 floats");
    check(lfm_conv_state_floats() * 4u == 737280ull, "conv state is 737280 bytes");

    /* ---- measured engine boundary --------------------------------------- */
    for (int layer = 0; layer < LFM_LAYERS; layer++) {
        SaltAttentionDesc a;
        int want_attention = lfm_is_attention_layer(layer);
        int rc = salt_model_attention(m, layer, &a);
        if (want_attention) {
            attention_layers++;
            check(rc == 0, "attention layer plans");
            if (rc == 0) {
                check(a.kind == SALT_ATTN_FULL, "attention kind FULL");
                check(a.n_heads == 32 && a.n_kv_heads == 8 && a.head_dim == 64,
                      "attention GQA geometry");
            }
        } else {
            conv_layers++;
            check(rc == 0, "conv layer declares its geometry");
            if (rc == 0) {
                check(a.kind == SALT_ATTN_MODEL, "conv layer is the conv kind");
                check(a.window == LFM_CONV_L, "conv window is the kernel size");
                check(a.n_heads == 0 && a.n_kv_heads == 0 && a.head_dim == 0,
                      "conv layer carries no heads");
                check(a.rope_kind == SALT_ROPE_NONE && a.rope_dim == 0,
                      "conv layer carries no RoPE");
                check(a.causal == 1, "conv layer is causal");
            }
        }
    }
    check(attention_layers == 10, "exactly 10 attention layers");
    check(conv_layers == 30, "exactly 30 conv layers");

    for (int layer = 0; layer < LFM_LAYERS; layer++) {
        SaltTextLayerPlan plan;
        int rc = salt_model_text_layer_plan(m, layer, &plan);
        if (lfm_is_attention_layer(layer)) {
            check(rc == 0, "attention layer produces a layer plan");
            if (rc == 0) {
                plan_ok++;
                check(plan.hidden == 2048, "plan hidden 2048");
                check(plan.n_experts == 64 && plan.top_k_experts == 4,
                      "plan expert geometry");
                check(plan.expert_intermediate == 1536, "plan expert intermediate");
                check(plan.dense_intermediate == 0,
                      "MoE layer carries no dense branch (dense is layers 0/1 only)");
                check(plan.activation == SALT_ACT_SILU, "plan activation SiLU");
                check(plan.router_kind == SALT_ROUTER_MODEL &&
                      plan.router_normalize_topk == 1 &&
                      plan.router_scale == 1.0f,
                      "plan carries sigmoid routing policy");
                check(plan.tied_embeddings == 1, "plan tied head");
                check(plan.logit_softcap == 0.0f, "plan softcap absent");
                check(plan.runtime_ready == 1, "plan carries runtime_ready 1");
            }
        } else {
            check(rc == 0, "conv layer produces a layer plan");
            if (rc == 0) {
                conv_plan_ok++;
                check(plan.attention.kind == SALT_ATTN_MODEL,
                      "conv plan carries the conv kind");
            }
        }
    }
    check(plan_ok == 10, "10 attention layers plan");
    check(conv_plan_ok == 30, "30 conv layers plan");

    /* The whole-model plan now builds, because every layer has a
     * representable geometry. This describes the PLAN, not runnability: the
     * program compiler admits only FULL or SLIDING, so a conv layer still
     * fails closed there until the conv cell kind exists. */
    {
        SaltTextLayerPlan plan[LFM_LAYERS];
        int rows = salt_model_text_plan(m, plan, LFM_LAYERS);
        check(rows == LFM_LAYERS, "whole-model plan builds all 40 layers");
        if (rows == LFM_LAYERS) {
            check(plan[0].attention.kind == SALT_ATTN_MODEL &&
                  plan[2].attention.kind == SALT_ATTN_FULL,
                  "plan carries conv and full kinds on the right layers");
        check(plan[2].attention.k_norm_absent == 0,
              "attention layer REQUIRES a K norm (the container has one)");
        check(plan[0].attention.k_norm_absent == 0,
              "conv layer does not claim a K norm");
        }
    }
    {
        SaltKvLayerPlan plan[LFM_LAYERS];
        size_t total = 0;
        int rc = salt_model_kv_plan(m, 2048, plan, LFM_LAYERS, &total);
        check(rc == 0, "KV plan builds");
        if (rc == 0) {
            check(total == (size_t)lfm_kv_floats(2048),
                  "KV plan charges only the 10 attention layers");
            check(plan[0].k_width == 0 && plan[0].v_width == 0 &&
                  plan[0].token_capacity == 0,
                  "conv layer is charged no KV");
            check(plan[2].k_width == LFM_KV_HEADS * LFM_HEAD_DIM &&
                  plan[2].v_width == LFM_KV_HEADS * LFM_HEAD_DIM &&
                  plan[2].token_capacity == 2048,
                  "attention layer is charged its full KV");
        }
    }

    printf("\nLFM_GATE attention_layers=%d conv_layers=%d plan_ok=%d conv_plan_ok=%d failures=%d\n",
           attention_layers, conv_layers, plan_ok, conv_plan_ok, failures);
    if (failures) {
        printf("LFM_GATE_FAIL\n");
        return 1;
    }
    printf("LFM_GATE_PASS\n");
    return 0;
}
