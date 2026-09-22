#ifndef SALT_STATE_H
#define SALT_STATE_H

#include <stddef.h>
#include <stdint.h>

#define SALT_STATE_SHA256_BYTES 32
#define SALT_STATE_SCHEMA_VERSION 1u
#define SALT_STATE_CONTROL_SCHEMA_VERSION 1u

typedef enum {
    SALT_MINDSET_NONE = 0,
    SALT_MINDSET_PREFIX = 1,
    SALT_MINDSET_RECURRENT = 2
} SaltMindsetKind;

typedef enum {
    SALT_FACTS_NONE = 0,
    SALT_FACTS_KV_ROWS = 1
} SaltFactsKind;

typedef struct {
    unsigned schema_version;
    SaltMindsetKind mindset_kind;
    SaltFactsKind facts_kind;
    int facts_optional;
    int clear_facts_supported;
    int hard_reset_supported;
    uint64_t default_mindset_end;
} SaltStateModelDesc;

typedef struct {
    unsigned schema_version;
    uint64_t session_epoch;
    uint64_t lifetime_turn;
    uint64_t history_turns;
    uint64_t position;
    uint64_t mindset_end;
    uint64_t bytes_per_row;
    SaltMindsetKind mindset_kind;
    SaltFactsKind facts_kind;
    int facts_optional;
    int proof_computed;
    uint8_t state_sha256[SALT_STATE_SHA256_BYTES];
    uint8_t mindset_sha256[SALT_STATE_SHA256_BYTES];
    uint8_t facts_sha256[SALT_STATE_SHA256_BYTES];
} SaltStateView;

typedef enum {
    SALT_STATE_ARTIFACT_FULL = 1,
    SALT_STATE_ARTIFACT_MINDSET = 2,
    SALT_STATE_ARTIFACT_FACTS = 3
} SaltStateArtifactKind;

typedef enum {
    SALT_STATE_TRANSACTION_EXPORT = 1,
    SALT_STATE_TRANSACTION_IMPORT = 2
} SaltStateTransactionKind;

typedef enum {
    SALT_STATE_REASON_EXPLICIT = 1,
    SALT_STATE_REASON_SESSION_CLOSE = 2
} SaltStateTransactionReason;

typedef int (*SaltStateMaterializeFn)(
    void *context, SaltStateTransactionKind transaction,
    SaltStateArtifactKind artifact, SaltStateTransactionReason reason);
typedef int (*SaltStateFinalizeFn)(
    void *context, SaltStateTransactionKind transaction,
    SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
    const SaltStateView *committed_view);

typedef struct SaltStateControl {
    unsigned schema_version;
    const SaltStateModelDesc *model;
    SaltStateMaterializeFn materialize;
    SaltStateFinalizeFn finalize;
    void *context;
    uint64_t generation;
    int active;
} SaltStateControl;

typedef struct SaltStateTransaction {
    unsigned schema_version;
    SaltStateControl *control;
    SaltStateView before;
    uint64_t generation;
    SaltStateTransactionKind kind;
    SaltStateArtifactKind artifact;
    SaltStateTransactionReason reason;
    int active;
} SaltStateTransaction;

/* Validate a model adapter's state view without interpreting model payloads. */
int salt_state_view_validate(const SaltStateModelDesc *model,
                             const SaltStateView *view);

/* Validate the model-agnostic lifecycle invariant for factual clear. */
int salt_state_clear_transition_validate(const SaltStateModelDesc *model,
                                         const SaltStateView *before,
                                         const SaltStateView *after);

/* Engine-owned state materialization transaction. Model adapters retain their
 * format and row geometry; backends may only materialize existing live
 * committed storage through the callback. The caller performs model-specific
 * streaming between begin and finish. Export must leave StateView unchanged. */
int salt_state_control_init(SaltStateControl *control,
                            const SaltStateModelDesc *model,
                            SaltStateMaterializeFn materialize,
                            SaltStateFinalizeFn finalize,
                            void *context);
int salt_state_transaction_begin(
    SaltStateControl *control, const SaltStateView *before,
    SaltStateTransactionKind kind, SaltStateArtifactKind artifact,
    SaltStateTransactionReason reason, SaltStateTransaction *transaction);
int salt_state_transaction_finish(SaltStateTransaction *transaction,
                                  const SaltStateView *after);
void salt_state_transaction_abort(SaltStateTransaction *transaction);

const char *salt_state_mindset_name(SaltMindsetKind kind);
const char *salt_state_facts_name(SaltFactsKind kind);

#endif /* SALT_STATE_H */
