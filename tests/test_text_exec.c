#include "salt/text_exec.h"

#include <stdio.h>
#include <string.h>

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "text exec: %s\n", message); \
        return 1; \
    } \
} while (0)

typedef struct {
    int hidden;
    int embed_batches[4];
    int embed_calls;
    int attention_calls;
    int ffn_calls;
    int ffn_batches[8];
    int releases;
    int published_position;
    int publish_calls;
    int transaction_calls;
    int transaction_failure;
    int phase_begins;
    int phase_ends;
    int phase_operation_calls;
    int phase_begin_failure;
    int phase_end_failure;
    int phase_operation_failure;
    int phase_end_operation_status;
    SaltTextRuntimePhase phase_seen;
    int fail_layer;
    int finish_failure;
    int lookahead_prepare_result;
    int lookahead_prepare_calls;
    int lookahead_claim_calls;
    int lookahead_cancel_calls;
    SaltTextPhaseLookaheadRequest lookahead_request;
    float finished[4];
    int events[32];
    int event_count;
} FakeModel;

static int fake_phase_begin(void *opaque, SaltTextRuntimePhase phase) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model) return -1;
    model->phase_begins++;
    model->phase_seen = phase;
    return model->phase_begin_failure ? -1 : 0;
}

static int fake_phase_end(
        void *opaque, SaltTextRuntimePhase phase, int operation_status) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model || phase != model->phase_seen) return -1;
    model->phase_ends++;
    model->phase_end_operation_status = operation_status;
    return model->phase_end_failure ? -1 : 0;
}

static int fake_phase_operation(void *opaque) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model) return -1;
    model->phase_operation_calls++;
    return model->phase_operation_failure ? -1 : 0;
}

static void record_event(FakeModel *model, int event) {
    if (model->event_count <
            (int)(sizeof model->events / sizeof model->events[0]))
        model->events[model->event_count++] = event;
}

static int fake_embed(void *opaque, const int *tokens, int batch,
                      float *states) {
    FakeModel *model = (FakeModel *)opaque;
    record_event(model, 100 + batch);
    model->embed_batches[model->embed_calls++] = batch;
    for (int token = 0; token < batch; token++)
        for (int d = 0; d < model->hidden; d++)
            states[token * model->hidden + d] =
                (float)(tokens[token] * 10 + d);
    return 0;
}

static int fake_attention(void *opaque, int layer, int start, int batch,
                          const float *states, float *outputs) {
    FakeModel *model = (FakeModel *)opaque;
    (void)start;
    record_event(model, 200 + layer);
    model->attention_calls++;
    if (layer == model->fail_layer) return -1;
    for (int i = 0; i < batch * model->hidden; i++)
        outputs[i] = states[i] + (float)(layer + 1);
    return 0;
}

static int fake_ffn(void *opaque, int layer, int start, int batch,
                    const float *states, float *outputs) {
    FakeModel *model = (FakeModel *)opaque;
    (void)start;
    record_event(model, 300 + layer);
    model->ffn_batches[model->ffn_calls++] = batch;
    for (int i = 0; i < batch * model->hidden; i++)
        outputs[i] = states[i] + (float)(10 * (layer + 1));
    return 0;
}

static void fake_release(void *opaque, int layer) {
    FakeModel *model = (FakeModel *)opaque;
    record_event(model, 400 + layer);
    model->releases++;
}

static int fake_finish(void *opaque, const float *last_state, float *logits) {
    FakeModel *model = (FakeModel *)opaque;
    record_event(model, 500);
    if (model->finish_failure) return -1;
    for (int d = 0; d < model->hidden; d++) {
        model->finished[d] = last_state[d];
        logits[d] = last_state[d];
    }
    return 0;
}

static int fake_lookahead_prepare(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model || !request) return -1;
    record_event(model, 450);
    model->lookahead_prepare_calls++;
    model->lookahead_request = *request;
    return model->lookahead_prepare_result;
}

static int fake_lookahead_claim(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model || !request ||
            memcmp(request, &model->lookahead_request,
                   sizeof *request) != 0)
        return -1;
    record_event(model, 451);
    model->lookahead_claim_calls++;
    return 0;
}

static int fake_lookahead_cancel(
        void *opaque, const SaltTextPhaseLookaheadRequest *request) {
    FakeModel *model = (FakeModel *)opaque;
    if (!model || !request ||
            memcmp(request, &model->lookahead_request,
                   sizeof *request) != 0)
        return -1;
    record_event(model, 452);
    model->lookahead_cancel_calls++;
    return 0;
}

static const SaltTextPhaseLookaheadOps fake_lookahead_ops = {
    fake_lookahead_prepare,
    fake_lookahead_claim,
    fake_lookahead_cancel,
};

static int fake_transaction(
        void *opaque, int (*operation)(void *), void *operation_context) {
    FakeModel *model = (FakeModel *)opaque;
    int rc;
    model->transaction_calls++;
    rc = operation ? operation(operation_context) : -1;
    return model->transaction_failure ? -1 : rc;
}

static int fake_publish(void *opaque, int position) {
    FakeModel *model = (FakeModel *)opaque;
    record_event(model, 700 + position);
    model->published_position = position;
    model->publish_calls++;
    return 0;
}

static SaltTextPrefillOps fake_ops(void) {
    SaltTextPrefillOps ops;
    memset(&ops, 0, sizeof ops);
    ops.embed = fake_embed;
    ops.attention = fake_attention;
    ops.feed_forward = fake_ffn;
    ops.release_layer = fake_release;
    ops.finish = fake_finish;
    ops.publish_position = fake_publish;
    return ops;
}

int main(void) {
    float states[12], branches[12];
    SaltTextPrefillScratch scratch = {
        states, branches, sizeof states / sizeof states[0]
    };
    SaltTextPrefillProfile profile;
    SaltTextPrefillPlan plan = {
        .n_layers = 3,
        .hidden = 4,
        .max_batch = 3,
        .profile = &profile,
        .scratch = &scratch,
        .resource_policy = SALT_TEXT_RELEASE_PER_LAYER,
    };
    SaltTextPrefillOps ops = fake_ops();
    int tokens[5] = {2, 3, 5, 7, 11};
    float logits[4];
    FakeModel model;
    const SaltTextPhaseScopeOps phase_ops = {
        fake_phase_begin, fake_phase_end,
    };
    SaltTextPhaseScope phase_scope = { &phase_ops, &model };
    const int expected_events[] = {
        103,
        200, 300, 400, 201, 301, 401, 202, 302, 402,
        102,
        200, 300, 400, 201, 301, 401, 202, 302, 402,
        500, 714,
    };
    memset(&model, 0, sizeof model);
    CHECK(salt_text_phase_scope_run(
              &phase_scope, SALT_TEXT_PHASE_PREFILL,
              fake_phase_operation, &model) == 0 &&
          model.phase_begins == 1 && model.phase_operation_calls == 1 &&
          model.phase_ends == 1 && model.phase_seen == SALT_TEXT_PHASE_PREFILL &&
          model.phase_end_operation_status == 0,
          "successful phase scope changed");
    memset(&model, 0, sizeof model);
    model.phase_operation_failure = 1;
    CHECK(salt_text_phase_scope_run(
              &phase_scope, SALT_TEXT_PHASE_DECODE,
              fake_phase_operation, &model) != 0 &&
          model.phase_begins == 1 && model.phase_operation_calls == 1 &&
          model.phase_ends == 1 && model.phase_end_operation_status != 0,
          "failed operation skipped phase cleanup");
    memset(&model, 0, sizeof model);
    model.phase_begin_failure = 1;
    CHECK(salt_text_phase_scope_run(
              &phase_scope, SALT_TEXT_PHASE_TARGET_VERIFY,
              fake_phase_operation, &model) != 0 &&
          model.phase_begins == 1 && model.phase_operation_calls == 0 &&
          model.phase_ends == 0,
          "failed phase begin executed operation or cleanup");
    memset(&model, 0, sizeof model);
    model.phase_end_failure = 1;
    CHECK(salt_text_phase_scope_run(
              &phase_scope, SALT_TEXT_PHASE_PREFILL,
              fake_phase_operation, &model) != 0 &&
          model.phase_begins == 1 && model.phase_operation_calls == 1 &&
          model.phase_ends == 1,
          "failed phase completion was accepted");

    memset(&model, 0, sizeof model);
    model.hidden = 4;
    model.fail_layer = -1;
    plan.transaction_run = fake_transaction;
    plan.transaction_context = &model;
    CHECK(salt_text_prefill_execute(
              &plan, &ops, &model, 9, tokens, 5, logits) == 0,
          "valid execution failed");
    CHECK(model.embed_calls == 2 && model.embed_batches[0] == 3 &&
              model.embed_batches[1] == 2,
          "outer chunk widths changed");
    CHECK(model.attention_calls == 6 && model.ffn_calls == 6,
          "layer schedule changed");
    CHECK(model.ffn_batches[0] == 3 && model.ffn_batches[1] == 3 &&
              model.ffn_batches[2] == 3 && model.ffn_batches[3] == 2 &&
              model.ffn_batches[4] == 2 && model.ffn_batches[5] == 2,
          "control plane substituted token-serial FFN calls");
    CHECK(model.releases == 6,
          "one release/fence per completed layer was not issued");
    CHECK(profile.released_layers == 6 && profile.retained_layers == 0,
          "host release telemetry drifted");
    CHECK(model.publish_calls == 1 && model.published_position == 14,
          "position was not published exactly once at completion");
    CHECK(model.transaction_calls == 1,
          "prefill did not execute in one admitted transaction");
    CHECK(logits[0] == 176.0f && logits[3] == 179.0f,
          "final head consumed the wrong canonical row");
    CHECK(model.event_count == (int)(sizeof expected_events /
              sizeof expected_events[0]) &&
          memcmp(model.events, expected_events, sizeof expected_events) == 0,
          "engine phase order changed");

    {
        SaltTextPhaseLookahead lookahead;
        memset(&lookahead, 0, sizeof lookahead);
        memset(&model, 0, sizeof model);
        model.hidden = 4;
        model.fail_layer = -1;
        plan.lookahead = &lookahead;
        plan.lookahead_ops = &fake_lookahead_ops;
        plan.lookahead_context = &model;
        plan.transition_generation = 41;
        plan.next_phase = SALT_TEXT_PHASE_DECODE;
        CHECK(salt_text_prefill_execute(
                  &plan, &ops, &model, 9, tokens, 5, logits) == 0,
              "lookahead execution failed");
        CHECK(model.lookahead_prepare_calls == 1 &&
                  lookahead.status == SALT_TEXT_LOOKAHEAD_READY &&
                  lookahead.request.source_phase == SALT_TEXT_PHASE_PREFILL &&
                  lookahead.request.target_phase == SALT_TEXT_PHASE_DECODE &&
                  lookahead.request.source_generation == 41 &&
                  lookahead.request.result_generation == 42 &&
                  lookahead.request.source_position == 9 &&
                  lookahead.request.result_position == 14,
              "lookahead identity or readiness changed");
        CHECK(model.event_count >= 3 &&
                  model.events[model.event_count - 3] == 500 &&
                  model.events[model.event_count - 2] == 450 &&
                  model.events[model.event_count - 1] == 714,
              "lookahead did not follow final head and precede publication");
        CHECK(salt_text_phase_lookahead_claim(
                  &lookahead, SALT_TEXT_PHASE_DECODE, 42, 15) == 0 &&
                  lookahead.status == SALT_TEXT_LOOKAHEAD_CANCELLED &&
                  model.lookahead_cancel_calls == 1 &&
                  model.lookahead_claim_calls == 0,
              "stale lookahead was claimed or not cancelled");

        memset(&model, 0, sizeof model);
        model.hidden = 4;
        model.fail_layer = -1;
        CHECK(salt_text_prefill_execute(
                  &plan, &ops, &model, 9, tokens, 5, logits) == 0 &&
              salt_text_phase_lookahead_claim(
                  &lookahead, SALT_TEXT_PHASE_DECODE, 42, 14) == 1 &&
              lookahead.status == SALT_TEXT_LOOKAHEAD_CLAIMED &&
              model.lookahead_claim_calls == 1 &&
              model.lookahead_cancel_calls == 0,
              "matching lookahead claim failed");

        memset(&model, 0, sizeof model);
        model.hidden = 4;
        model.fail_layer = -1;
        model.lookahead_prepare_result = 1;
        CHECK(salt_text_prefill_execute(
                  &plan, &ops, &model, 9, tokens, 5, logits) == 0 &&
              lookahead.status == SALT_TEXT_LOOKAHEAD_REFUSED &&
              salt_text_phase_lookahead_claim(
                  &lookahead, SALT_TEXT_PHASE_DECODE, 42, 14) == 0,
              "refused lookahead changed ordinary prefill");

        memset(&lookahead, 0, sizeof lookahead);
        memset(&model, 0, sizeof model);
        model.hidden = 4;
        model.fail_layer = -1;
        model.finish_failure = 1;
        CHECK(salt_text_prefill_execute(
                  &plan, &ops, &model, 9, tokens, 5, logits) != 0 &&
              model.publish_calls == 0 &&
              model.lookahead_prepare_calls == 0 &&
              model.lookahead_cancel_calls == 0 &&
              lookahead.status == SALT_TEXT_LOOKAHEAD_EMPTY,
              "failed final head issued a lookahead");

        memset(&model, 0, sizeof model);
        model.hidden = 4;
        model.fail_layer = -1;
        model.transaction_failure = 1;
        CHECK(salt_text_prefill_execute(
                  &plan, &ops, &model, 9, tokens, 5, logits) != 0 &&
              model.publish_calls == 1 &&
              model.lookahead_cancel_calls == 1 &&
              lookahead.status == SALT_TEXT_LOOKAHEAD_CANCELLED,
              "failed outer transaction retained a READY lookahead");

        plan.lookahead = NULL;
        plan.lookahead_ops = NULL;
        plan.lookahead_context = NULL;
        plan.transition_generation = 0;
        plan.next_phase = SALT_TEXT_PHASE_PREFILL;
    }

    memset(&model, 0, sizeof model);
    model.hidden = 4;
    model.fail_layer = -1;
    plan.resource_policy = SALT_TEXT_RETAIN_OPERATION;
    CHECK(salt_text_prefill_execute(
              &plan, &ops, &model, 9, tokens, 5, logits) == 0,
          "operation-retained execution failed");
    CHECK(model.releases == 0,
          "operation-retained execution released completed layers");
    CHECK(profile.released_layers == 0 && profile.retained_layers == 6,
          "operation-retained telemetry drifted");

    memset(&model, 0, sizeof model);
    model.hidden = 4;
    model.fail_layer = 1;
    CHECK(salt_text_prefill_execute(
              &plan, &ops, &model, 9, tokens, 5, logits) != 0,
          "failing layer was accepted");
    CHECK(model.publish_calls == 0,
          "failed execution published position");
    CHECK(model.releases == 1,
          "operation-retained failure did not release only the attempted layer");
    CHECK(profile.released_layers == 1 && profile.retained_layers == 1,
          "operation-retained failure telemetry drifted");
    CHECK(model.event_count == 5 &&
              model.events[0] == 103 && model.events[1] == 200 &&
              model.events[2] == 300 && model.events[3] == 201 &&
              model.events[4] == 401,
          "failed phase/release order changed");

    plan.resource_policy = SALT_TEXT_RELEASE_PER_LAYER;
    plan.max_batch = 0;
    CHECK(salt_text_prefill_execute(
              &plan, &ops, &model, 0, tokens, 5, logits) != 0,
          "invalid engine plan was accepted");
    puts("engine text prefill control plane: PASS");
    return 0;
}
