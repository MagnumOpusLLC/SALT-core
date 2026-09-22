#include "salt/model.h"
#include "salt/state.h"

#include <stdio.h>
#include <string.h>

static void digest_fill(uint8_t out[32], unsigned seed) {
    for (int i = 0; i < 32; i++) out[i] = (uint8_t)(seed + (unsigned)i);
}

typedef struct {
    int calls;
    int final_calls;
    SaltStateTransactionKind kind;
    SaltStateArtifactKind artifact;
    SaltStateTransactionReason reason;
} MaterializeProbe;

static int materialize_probe(
        void *opaque, SaltStateTransactionKind kind,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason) {
    MaterializeProbe *probe = (MaterializeProbe *)opaque;
    if (!probe) return -1;
    probe->calls++;
    probe->kind = kind;
    probe->artifact = artifact;
    probe->reason = reason;
    return 0;
}

static int finalize_probe(
        void *opaque, SaltStateTransactionKind kind,
        SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
        const SaltStateView *view) {
    MaterializeProbe *probe = (MaterializeProbe *)opaque;
    if (!probe || !view) return -1;
    probe->final_calls++;
    probe->kind = kind;
    probe->artifact = artifact;
    probe->reason = reason;
    return 0;
}

static int exercise(const char *name, SaltMindsetKind mindset) {
    const SaltModelDesc *model = salt_model_get(name);
    const SaltStateModelDesc *state = salt_model_state(model);
    SaltStateView before, after, bad, off_before, off_after;
    SaltStateControl control;
    SaltStateTransaction transaction, nested;
    MaterializeProbe probe;
    if (!model || !state || state->mindset_kind != mindset ||
        state->facts_kind != SALT_FACTS_KV_ROWS || !state->facts_optional ||
        !state->clear_facts_supported || !state->hard_reset_supported)
        return -1;
    memset(&before, 0, sizeof before);
    before.schema_version = SALT_STATE_SCHEMA_VERSION;
    before.session_epoch = 9;
    before.lifetime_turn = 4;
    before.history_turns = 3;
    before.position = 11;
    before.mindset_end = state->default_mindset_end;
    if (mindset == SALT_MINDSET_PREFIX) before.mindset_end += 3;
    before.bytes_per_row = 450560;
    before.mindset_kind = state->mindset_kind;
    before.facts_kind = state->facts_kind;
    before.facts_optional = state->facts_optional;
    before.proof_computed = 1;
    digest_fill(before.mindset_sha256, 7);
    digest_fill(before.state_sha256, 19);
    digest_fill(before.facts_sha256, 31);
    if (salt_state_view_validate(state, &before) != 0)
        return -1;
    after = before;
    after.history_turns = 0;
    after.position = after.mindset_end;
    memcpy(after.state_sha256, after.mindset_sha256, 32);
    digest_fill(after.facts_sha256, 41);
    if (salt_state_clear_transition_validate(state, &before, &after) != 0)
        return -1;
    bad = after;
    bad.session_epoch++;
    if (salt_state_clear_transition_validate(state, &before, &bad) == 0)
        return -1;
    bad = after;
    bad.mindset_sha256[0] ^= 1u;
    if (salt_state_clear_transition_validate(state, &before, &bad) == 0)
        return -1;
    off_before = before;
    off_before.proof_computed = 0;
    memset(off_before.state_sha256, 0, 32);
    memset(off_before.mindset_sha256, 0, 32);
    memset(off_before.facts_sha256, 0, 32);
    off_after = after;
    off_after.proof_computed = 0;
    memset(off_after.state_sha256, 0, 32);
    memset(off_after.mindset_sha256, 0, 32);
    memset(off_after.facts_sha256, 0, 32);
    if (salt_state_view_validate(state, &off_before) != 0 ||
        salt_state_clear_transition_validate(state, &off_before, &off_after) != 0)
        return -1;
    bad = off_before;
    bad.state_sha256[0] = 1;
    if (salt_state_view_validate(state, &bad) == 0) return -1;
    memset(&probe, 0, sizeof probe);
    memset(&transaction, 0, sizeof transaction);
    memset(&nested, 0, sizeof nested);
    if (salt_state_control_init(
            &control, state, materialize_probe, finalize_probe, &probe) != 0 ||
        salt_state_transaction_begin(
            &control, &off_before, SALT_STATE_TRANSACTION_EXPORT,
            SALT_STATE_ARTIFACT_FULL, SALT_STATE_REASON_EXPLICIT,
            &transaction) != 0 ||
        probe.calls != 1 || probe.kind != SALT_STATE_TRANSACTION_EXPORT ||
        probe.artifact != SALT_STATE_ARTIFACT_FULL ||
        probe.reason != SALT_STATE_REASON_EXPLICIT ||
        salt_state_transaction_begin(
            &control, &off_before, SALT_STATE_TRANSACTION_EXPORT,
            SALT_STATE_ARTIFACT_FULL, SALT_STATE_REASON_EXPLICIT,
            &nested) == 0 ||
        salt_state_transaction_finish(&transaction, &off_before) != 0 ||
        probe.final_calls != 1 || control.active)
        return -1;
    if (salt_state_transaction_begin(
            &control, &off_before, SALT_STATE_TRANSACTION_EXPORT,
            SALT_STATE_ARTIFACT_MINDSET, SALT_STATE_REASON_SESSION_CLOSE,
            &transaction) != 0)
        return -1;
    bad = off_before;
    bad.position++;
    if (salt_state_transaction_finish(&transaction, &bad) == 0)
        return -1;
    salt_state_transaction_abort(&transaction);
    if (control.active ||
        salt_state_transaction_begin(
            &control, &off_before, SALT_STATE_TRANSACTION_IMPORT,
            SALT_STATE_ARTIFACT_FACTS, SALT_STATE_REASON_EXPLICIT,
            &transaction) != 0)
        return -1;
    bad = off_before;
    bad.position++;
    bad.lifetime_turn++;
    if (salt_state_transaction_finish(&transaction, &bad) != 0 ||
        control.active ||
        salt_state_transaction_begin(
            &control, &off_before, SALT_STATE_TRANSACTION_IMPORT,
            SALT_STATE_ARTIFACT_FULL, SALT_STATE_REASON_SESSION_CLOSE,
            &transaction) == 0)
        return -1;
    return 0;
}

int main(void) {
    if (exercise("qwen36", SALT_MINDSET_RECURRENT) != 0) {
        fputs("FAIL qwen state adapter\n", stderr);
        return 1;
    }
    if (exercise("gemma4-26b-a4b", SALT_MINDSET_PREFIX) != 0) {
        fputs("FAIL gemma state adapter\n", stderr);
        return 1;
    }
    puts("model-agnostic state control: PASS");
    return 0;
}
