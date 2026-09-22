#include "salt/state.h"

#include <string.h>

static int bool_value(int value) {
    return value == 0 || value == 1;
}

static int digest_nonzero(const uint8_t digest[SALT_STATE_SHA256_BYTES]) {
    uint8_t value = 0;
    for (size_t i = 0; i < SALT_STATE_SHA256_BYTES; i++) value |= digest[i];
    return value != 0;
}

static int digest_zero(const uint8_t digest[SALT_STATE_SHA256_BYTES]) {
    return !digest_nonzero(digest);
}

static int artifact_supported(const SaltStateModelDesc *model,
                              SaltStateArtifactKind artifact) {
    if (!model) return 0;
    switch (artifact) {
    case SALT_STATE_ARTIFACT_FULL:
        return 1;
    case SALT_STATE_ARTIFACT_MINDSET:
        return model->mindset_kind != SALT_MINDSET_NONE;
    case SALT_STATE_ARTIFACT_FACTS:
        return model->facts_kind != SALT_FACTS_NONE;
    default:
        return 0;
    }
}

static int export_view_unchanged(const SaltStateView *before,
                                 const SaltStateView *after) {
    return before && after &&
        before->schema_version == after->schema_version &&
        before->session_epoch == after->session_epoch &&
        before->lifetime_turn == after->lifetime_turn &&
        before->history_turns == after->history_turns &&
        before->position == after->position &&
        before->mindset_end == after->mindset_end &&
        before->bytes_per_row == after->bytes_per_row &&
        before->mindset_kind == after->mindset_kind &&
        before->facts_kind == after->facts_kind &&
        before->facts_optional == after->facts_optional &&
        before->proof_computed == after->proof_computed &&
        memcmp(before->state_sha256, after->state_sha256,
               SALT_STATE_SHA256_BYTES) == 0 &&
        memcmp(before->mindset_sha256, after->mindset_sha256,
               SALT_STATE_SHA256_BYTES) == 0 &&
        memcmp(before->facts_sha256, after->facts_sha256,
               SALT_STATE_SHA256_BYTES) == 0;
}

const char *salt_state_mindset_name(SaltMindsetKind kind) {
    switch (kind) {
    case SALT_MINDSET_NONE: return "none";
    case SALT_MINDSET_PREFIX: return "prefix";
    case SALT_MINDSET_RECURRENT: return "recurrent";
    default: return NULL;
    }
}

const char *salt_state_facts_name(SaltFactsKind kind) {
    switch (kind) {
    case SALT_FACTS_NONE: return "none";
    case SALT_FACTS_KV_ROWS: return "kv-rows";
    default: return NULL;
    }
}

int salt_state_view_validate(const SaltStateModelDesc *model,
                             const SaltStateView *view) {
    if (!model || !view ||
        model->schema_version != SALT_STATE_SCHEMA_VERSION ||
        view->schema_version != SALT_STATE_SCHEMA_VERSION ||
        !salt_state_mindset_name(model->mindset_kind) ||
        !salt_state_facts_name(model->facts_kind) ||
        model->mindset_kind != view->mindset_kind ||
        model->facts_kind != view->facts_kind ||
        !bool_value(model->facts_optional) ||
        !bool_value(model->clear_facts_supported) ||
        !bool_value(model->hard_reset_supported) ||
        !bool_value(view->facts_optional) ||
        !bool_value(view->proof_computed) ||
        model->facts_optional != view->facts_optional ||
        view->session_epoch == 0 ||
        view->history_turns > view->lifetime_turn ||
        view->mindset_end > view->position ||
        view->mindset_end < model->default_mindset_end)
        return -1;
    if ((view->proof_computed &&
            (!digest_nonzero(view->state_sha256) ||
             !digest_nonzero(view->mindset_sha256) ||
             !digest_nonzero(view->facts_sha256))) ||
        (!view->proof_computed &&
            (!digest_zero(view->state_sha256) ||
             !digest_zero(view->mindset_sha256) ||
             !digest_zero(view->facts_sha256))))
        return -1;
    if (model->mindset_kind == SALT_MINDSET_NONE && view->mindset_end != 0)
        return -1;
    /* PREFIX models may authenticate a longer runtime mindset than their
     * model-owned default. The common validation above keeps that boundary at
     * or above the default and at or below the committed position. */
    if (model->facts_kind == SALT_FACTS_NONE &&
        (view->position != view->mindset_end || view->facts_optional))
        return -1;
    if (model->facts_kind == SALT_FACTS_KV_ROWS && view->bytes_per_row == 0)
        return -1;
    return 0;
}

int salt_state_clear_transition_validate(const SaltStateModelDesc *model,
                                         const SaltStateView *before,
                                         const SaltStateView *after) {
    if (salt_state_view_validate(model, before) != 0 ||
        salt_state_view_validate(model, after) != 0 ||
        !model->clear_facts_supported ||
        before->session_epoch != after->session_epoch ||
        before->lifetime_turn != after->lifetime_turn ||
        after->history_turns != 0 ||
        before->mindset_end != after->mindset_end ||
        after->position != after->mindset_end ||
        before->mindset_kind != after->mindset_kind ||
        before->facts_kind != after->facts_kind ||
        before->facts_optional != after->facts_optional ||
        before->bytes_per_row != after->bytes_per_row ||
        before->proof_computed != after->proof_computed)
        return -1;
    if (before->proof_computed &&
        (memcmp(before->mindset_sha256, after->mindset_sha256,
                SALT_STATE_SHA256_BYTES) != 0 ||
         memcmp(after->state_sha256, after->mindset_sha256,
                SALT_STATE_SHA256_BYTES) != 0))
        return -1;
    return 0;
}

int salt_state_control_init(SaltStateControl *control,
                            const SaltStateModelDesc *model,
                            SaltStateMaterializeFn materialize,
                            SaltStateFinalizeFn finalize,
                            void *context) {
    if (!control || !model ||
        model->schema_version != SALT_STATE_SCHEMA_VERSION ||
        !salt_state_mindset_name(model->mindset_kind) ||
        !salt_state_facts_name(model->facts_kind))
        return -1;
    memset(control, 0, sizeof *control);
    control->schema_version = SALT_STATE_CONTROL_SCHEMA_VERSION;
    control->model = model;
    control->materialize = materialize;
    control->finalize = finalize;
    control->context = context;
    return 0;
}

int salt_state_transaction_begin(
        SaltStateControl *control, const SaltStateView *before,
        SaltStateTransactionKind kind, SaltStateArtifactKind artifact,
        SaltStateTransactionReason reason, SaltStateTransaction *transaction) {
    uint64_t generation;
    if (!control || !before || !transaction ||
        control->schema_version != SALT_STATE_CONTROL_SCHEMA_VERSION ||
        !control->model || control->active ||
        (kind != SALT_STATE_TRANSACTION_EXPORT &&
         kind != SALT_STATE_TRANSACTION_IMPORT) ||
        (reason != SALT_STATE_REASON_EXPLICIT &&
         reason != SALT_STATE_REASON_SESSION_CLOSE) ||
        (kind == SALT_STATE_TRANSACTION_IMPORT &&
         reason != SALT_STATE_REASON_EXPLICIT) ||
        !artifact_supported(control->model, artifact) ||
        salt_state_view_validate(control->model, before) != 0 ||
        control->generation == UINT64_MAX)
        return -1;
    generation = control->generation + 1u;
    memset(transaction, 0, sizeof *transaction);
    control->generation = generation;
    control->active = 1;
    transaction->schema_version = SALT_STATE_CONTROL_SCHEMA_VERSION;
    transaction->control = control;
    transaction->before = *before;
    transaction->generation = generation;
    transaction->kind = kind;
    transaction->artifact = artifact;
    transaction->reason = reason;
    transaction->active = 1;
    if (control->materialize &&
        control->materialize(control->context, kind, artifact, reason) != 0) {
        salt_state_transaction_abort(transaction);
        return -1;
    }
    return 0;
}

int salt_state_transaction_finish(SaltStateTransaction *transaction,
                                  const SaltStateView *after) {
    SaltStateControl *control;
    if (!transaction || !after || !transaction->active ||
        transaction->schema_version != SALT_STATE_CONTROL_SCHEMA_VERSION ||
        !(control = transaction->control) || !control->active ||
        control->generation != transaction->generation ||
        control->model == NULL ||
        salt_state_view_validate(control->model, after) != 0 ||
        transaction->before.session_epoch != after->session_epoch ||
        (transaction->kind == SALT_STATE_TRANSACTION_EXPORT &&
         !export_view_unchanged(&transaction->before, after)) ||
        (control->finalize &&
         control->finalize(control->context, transaction->kind,
             transaction->artifact, transaction->reason, after) != 0))
        return -1;
    control->active = 0;
    memset(transaction, 0, sizeof *transaction);
    return 0;
}

void salt_state_transaction_abort(SaltStateTransaction *transaction) {
    SaltStateControl *control;
    if (!transaction) return;
    control = transaction->control;
    if (transaction->active && control && control->active &&
        control->generation == transaction->generation)
        control->active = 0;
    memset(transaction, 0, sizeof *transaction);
}
