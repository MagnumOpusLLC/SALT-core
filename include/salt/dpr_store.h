#ifndef SALT_DPR_STORE_H
#define SALT_DPR_STORE_H

#include "salt/dpr.h"

#include <stddef.h>
#include <stdint.h>

#define SALT_DPR_EDGE_FILE_BYTES 512u
#define SALT_DPR_MENTOR_FAMILY_FILE_BYTES 256u
#define SALT_DPR_MENTOR_BINDING_FILE_BYTES 256u
#define SALT_DPR_ATTENTION_PLAN_FILE_BYTES 2560u
#define SALT_DPR_REFERENCE_HEADER_BYTES 256u
#define SALT_DPR_MENTOR_BINDING_VERSION 1u
#define SALT_DPR_STORE_NONE UINT32_MAX
#define SALT_DPR_PATH_MAX 4096u

typedef struct SaltDprStoredEdge {
    SaltDprEdge edge;
    uint8_t edge_sha256[32];
    uint32_t next_index;
} SaltDprStoredEdge;

typedef struct SaltDprStore {
    SaltDprStoredEdge *edges;
    size_t edge_capacity;
    size_t edge_count;
    uint32_t *buckets;
    size_t bucket_count;
    size_t rejected_edges;
} SaltDprStore;

typedef struct SaltDprMentorFamilyMaterial {
    uint32_t schema_version;
    uint8_t mindset_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint8_t nomogram_root_sha256[32];
    uint8_t provenance_sha256[32];
} SaltDprMentorFamilyMaterial;

typedef struct SaltDprMentorFamilyStore {
    SaltDprMentorFamily *families;
    uint32_t *next_indices;
    size_t family_capacity;
    size_t family_count;
    uint32_t *buckets;
    size_t bucket_count;
    size_t rejected_families;
} SaltDprMentorFamilyStore;

typedef struct SaltDprMentorBindingMaterial {
    uint32_t schema_version;
    uint8_t family_sha256[32];
    uint8_t edge_sha256[32];
    uint8_t source_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
} SaltDprMentorBindingMaterial;

typedef struct SaltDprMentorBinding {
    uint32_t schema_version;
    uint8_t family_sha256[32];
    uint8_t edge_sha256[32];
    uint8_t source_node_sha256[32];
    uint8_t state_compatibility_sha256[32];
    uint8_t policy_sha256[32];
    uint8_t binding_sha256[32];
    size_t family_index;
    size_t edge_index;
} SaltDprMentorBinding;

typedef struct SaltDprMentorBindingStore {
    SaltDprMentorBinding *bindings;
    uint32_t *next_indices;
    size_t binding_capacity;
    size_t binding_count;
    uint32_t *buckets;
    size_t bucket_count;
    size_t rejected_bindings;
} SaltDprMentorBindingStore;

typedef struct SaltDprStoredAttentionPlan {
    SaltDprAttentionPlan plan;
    uint8_t plan_sha256[32];
    uint32_t next_index;
} SaltDprStoredAttentionPlan;

typedef struct SaltDprAttentionPlanStore {
    SaltDprStoredAttentionPlan *plans;
    size_t plan_capacity;
    size_t plan_count;
    uint32_t *buckets;
    size_t bucket_count;
    size_t rejected_plans;
    size_t runtime_index;
    int runtime_present;
} SaltDprAttentionPlanStore;

typedef struct SaltDprReferenceLease {
    int fd;
    uint64_t bytes;
    uint64_t position;
    uint64_t device;
    uint64_t inode;
    uint64_t mtime_ns;
    uint64_t ctime_ns;
    uint8_t edge_sha256[32];
    uint8_t reference_sha256[32];
    char path[SALT_DPR_PATH_MAX + 1u];
} SaltDprReferenceLease;

typedef struct SaltDprReferenceMeasurement {
    uint8_t file_sha256[32];
    uint64_t bytes;
    uint64_t device;
    uint64_t inode;
    uint64_t mtime_ns;
    uint64_t ctime_ns;
} SaltDprReferenceMeasurement;

int salt_dpr_store_prepare(const char *root);
int salt_dpr_edge_path(const char *root, const uint8_t edge_sha256[32],
                       char *out, size_t out_size);
int salt_dpr_reference_receipt_path(
    const char *root, const uint8_t edge_sha256[32],
    char *out, size_t out_size);
int salt_dpr_mentor_family_path(
    const char *root, const uint8_t family_sha256[32],
    char *out, size_t out_size);
int salt_dpr_mentor_family_encode(
    const SaltDprMentorFamilyMaterial *material,
    uint8_t out[SALT_DPR_MENTOR_FAMILY_FILE_BYTES],
    uint8_t family_sha256[32]);
int salt_dpr_mentor_family_decode(
    const uint8_t bytes[SALT_DPR_MENTOR_FAMILY_FILE_BYTES],
    SaltDprMentorFamily *family);
int salt_dpr_mentor_family_publish(
    const char *root, const SaltDprMentorFamilyMaterial *material,
    uint8_t family_sha256[32]);
int salt_dpr_mentor_binding_path(
    const char *root, const uint8_t binding_sha256[32],
    char *out, size_t out_size);
int salt_dpr_mentor_binding_encode(
    const SaltDprMentorBindingMaterial *material,
    uint8_t out[SALT_DPR_MENTOR_BINDING_FILE_BYTES],
    uint8_t binding_sha256[32]);
int salt_dpr_mentor_binding_decode(
    const uint8_t bytes[SALT_DPR_MENTOR_BINDING_FILE_BYTES],
    SaltDprMentorBinding *binding);
int salt_dpr_mentor_binding_publish(
    const char *root, const SaltDprMentorBindingMaterial *material,
    uint8_t binding_sha256[32]);
int salt_dpr_attention_plan_path(
    const char *root, const uint8_t plan_sha256[32],
    char *out, size_t out_size);
int salt_dpr_attention_plan_encode(
    const SaltDprAttentionPlan *plan,
    uint8_t out[SALT_DPR_ATTENTION_PLAN_FILE_BYTES],
    uint8_t plan_sha256[32]);
int salt_dpr_attention_plan_decode(
    const uint8_t bytes[SALT_DPR_ATTENTION_PLAN_FILE_BYTES],
    SaltDprAttentionPlan *plan, uint8_t plan_sha256[32]);
int salt_dpr_attention_plan_publish(
    const char *root, const SaltDprAttentionPlan *plan,
    uint8_t plan_sha256[32]);
int salt_dpr_edge_encode(const SaltDprEdge *edge,
                         uint8_t out[SALT_DPR_EDGE_FILE_BYTES],
                         uint8_t edge_sha256[32]);
int salt_dpr_edge_decode(const uint8_t bytes[SALT_DPR_EDGE_FILE_BYTES],
                         SaltDprEdge *edge, uint8_t edge_sha256[32]);
int salt_dpr_edge_publish(const char *root, const SaltDprEdge *edge,
                          uint8_t edge_sha256[32]);

int salt_dpr_reference_measure(const char *path,
                               SaltDprReferenceMeasurement *measurement);
int salt_dpr_reference_bind(const char *root,
                            const uint8_t edge_sha256[32],
                            const SaltDprEdge *edge,
                            const char *reference_path,
                            const SaltDprReferenceMeasurement *measurement);
int salt_dpr_reference_open(const char *root,
                            const SaltDprStoredEdge *stored_edge,
                            SaltDprReferenceLease *lease);
void salt_dpr_reference_close(SaltDprReferenceLease *lease);

int salt_dpr_mentor_family_store_init(
    SaltDprMentorFamilyStore *store, SaltDprMentorFamily *family_slots,
    uint32_t *next_indices, size_t family_capacity,
    uint32_t *buckets, size_t bucket_count);
int salt_dpr_mentor_family_store_load(
    SaltDprMentorFamilyStore *store, const char *root);
int salt_dpr_mentor_family_store_find(
    const SaltDprMentorFamilyStore *store,
    const uint8_t family_sha256[32], size_t *family_index);

int salt_dpr_mentor_binding_store_init(
    SaltDprMentorBindingStore *store, SaltDprMentorBinding *binding_slots,
    uint32_t *next_indices, size_t binding_capacity,
    uint32_t *buckets, size_t bucket_count);
int salt_dpr_mentor_binding_store_load(
    SaltDprMentorBindingStore *store, const char *root,
    const SaltDprMentorFamilyStore *family_store,
    const SaltDprStore *edge_store);
int salt_dpr_attention_plan_store_init(
    SaltDprAttentionPlanStore *store,
    SaltDprStoredAttentionPlan *plan_slots, size_t plan_capacity,
    uint32_t *buckets, size_t bucket_count);
int salt_dpr_attention_plan_store_load(
    SaltDprAttentionPlanStore *store, const char *root);
int salt_dpr_attention_plan_store_upsert_runtime(
    SaltDprAttentionPlanStore *store, const SaltDprAttentionPlan *plan,
    uint8_t plan_sha256[32], size_t *stored_index);
int salt_dpr_attention_plan_store_clear_runtime(
    SaltDprAttentionPlanStore *store);
int salt_dpr_attention_plan_store_select(
    const SaltDprAttentionPlanStore *store,
    const SaltDprComputeIntent *intent,
    uint32_t max_layers, uint32_t max_experts, uint64_t max_context,
    size_t *selected_index);
int salt_dpr_mentor_candidates_build(
    const SaltDprMentorBindingStore *binding_store,
    const SaltDprMentorFamilyStore *family_store,
    const SaltDprStore *edge_store,
    const uint8_t *retained, const SaltDprChart *effective_charts,
    SaltDprMode mode,
    const uint8_t target_node_sha256[32],
    uint64_t serial_ns_per_token, uint64_t minimum_samples,
    SaltDprSelectorCandidate *candidates, size_t candidate_capacity,
    size_t *candidate_count);
int salt_dpr_mentor_walk_nomogram(
    const SaltDprMentorBindingStore *binding_store,
    const SaltDprMentorFamilyStore *family_store,
    const SaltDprStore *edge_store, const uint8_t *retained,
    const SaltDprChart *effective_charts,
    size_t family_index, SaltDprMode mode,
    const uint8_t start_node_sha256[32],
    uint64_t serial_ns_per_token, uint64_t minimum_samples,
    int32_t vocab_size, uint64_t max_context,
    SaltDprStopTokenFn stop_token, void *stop_opaque,
    uint32_t max_tokens, int32_t *candidate_token_ids,
    size_t *edge_indices, size_t *binding_indices,
    uint32_t *candidate_count);

int salt_dpr_store_init(SaltDprStore *store,
                        SaltDprStoredEdge *edge_slots,
                        size_t edge_capacity,
                        uint32_t *buckets,
                        size_t bucket_count);
int salt_dpr_store_load(SaltDprStore *store, const char *root);
int salt_dpr_store_select(const SaltDprStore *store,
                          const uint8_t source_node_sha256[32],
                          SaltDprMode mode,
                          uint64_t serial_ns_per_token,
                          uint64_t minimum_samples,
                          int32_t vocab_size,
                          uint64_t max_context,
                          SaltDprStopTokenFn stop_token,
                          void *stop_opaque,
                          size_t *selected_index);
int salt_dpr_store_select_kind(const SaltDprStore *store,
                               const uint8_t source_node_sha256[32],
                               SaltDprMode mode, SaltDprEdgeKind kind,
                               uint64_t serial_ns_per_token,
                               uint64_t minimum_samples,
                               int32_t vocab_size, uint64_t max_context,
                               SaltDprStopTokenFn stop_token,
                               void *stop_opaque,
                               size_t *selected_index);
int salt_dpr_store_walk_nomogram(const SaltDprStore *store,
                               const uint8_t start_node_sha256[32],
                               SaltDprMode mode,
                               uint64_t serial_ns_per_token,
                               uint64_t minimum_samples,
                               int32_t vocab_size, uint64_t max_context,
                               SaltDprStopTokenFn stop_token,
                               void *stop_opaque,
                               uint32_t max_tokens,
                               int32_t *candidate_token_ids,
                               size_t *edge_indices,
                               uint32_t *candidate_count);

#endif /* SALT_DPR_STORE_H */
