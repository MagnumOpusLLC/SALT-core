#ifndef SALT_GEMMA4_TEXT_H
#define SALT_GEMMA4_TEXT_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "salt/dpr.h"
#include "salt/state.h"
#include "salt/text_verify.h"

typedef struct SaltGemma4Text SaltGemma4Text;

enum {
    SALT_GEMMA4_TEXT_NFQ_MAX_CANDIDATES = 128,
    SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES = 356
};

typedef struct SaltGemma4MemoryStats {
    uint64_t proof_validation_calls;
    uint64_t kv_arena_bytes;
    uint64_t startup_limit_bytes;
    uint64_t startup_forecast_bytes;
    uint64_t startup_source_bytes;
    uint64_t startup_registered_bytes;
    uint64_t startup_pageable_bytes;
    uint64_t startup_expert_residency_bytes;
    uint64_t startup_host_runtime_bytes;
    uint64_t startup_host_kv_bytes;
    uint64_t startup_shared_arena_bytes;
    uint64_t startup_backend_pinned_bytes;
    uint64_t startup_backend_device_bytes;
    uint64_t startup_device_kv_bytes;
    uint64_t startup_cache_metadata_bytes;
    uint64_t startup_rope_bytes;
    uint64_t startup_descriptor_bytes;
    int startup_admitted;
    uint64_t expert_budget_bytes;
    uint64_t expert_slot_bytes;
    uint64_t expert_capacity_bytes;
    uint64_t expert_resident_logical_bytes;
    uint64_t expert_peak_logical_bytes;
    uint64_t expert_requests;
    uint64_t expert_hits;
    uint64_t expert_misses;
    uint64_t expert_evictions;
    uint64_t expert_release_calls;
    uint64_t expert_release_failures;
    uint64_t expert_released_page_bytes;
    uint64_t expert_mapped_bytes;
    uint64_t expert_peak_mapped_bytes;
    uint64_t expert_unmap_calls;
    uint64_t expert_unmap_failures;
    uint64_t expert_unmapped_bytes;
    uint64_t expert_prepare_calls;
    uint64_t expert_prepare_failures;
    uint64_t expert_prepared_bytes;
    uint64_t expert_union_fetch_calls;
    uint64_t expert_union_fetch_experts;
    uint64_t expert_gate_up_epochs;
    uint64_t expert_down_epochs;
    uint64_t expert_matrix_gate_up_waves;
    uint64_t expert_matrix_down_waves;
    uint64_t expert_matrix_projection_jobs;
    uint64_t prefill_expert_matrix_flow_sessions;
    uint64_t target_matrix_flow_sessions;
    uint64_t target_gpu_expert_batches;
    uint64_t target_gpu_expert_jobs;
    uint64_t q4_pool_submissions;
    uint64_t q4_pool_fallbacks;
    uint64_t q4_multi_pool_submissions;
    uint64_t q4_multi_pool_jobs;
    uint64_t qkv_multi_pool_submissions;
    uint64_t q4_multi_pool_fallbacks;
    uint64_t q8_pool_submissions;
    uint64_t q8_pool_fallbacks;
    uint64_t expert_pool_submissions;
    uint64_t expert_pool_fallbacks;
    uint64_t gpu_dense_submissions;
    uint64_t gpu_router_submissions;
    uint64_t gpu_expert_gate_up_submissions;
    uint64_t gpu_expert_down_submissions;
    uint64_t gpu_expert_gate_up_commands;
    uint64_t gpu_expert_down_commands;
    uint64_t gpu_decode_expert_gate_up_commands;
    uint64_t gpu_decode_expert_down_commands;
    uint64_t gpu_head_submissions;
    uint64_t gpu_decode_submissions;
    uint64_t gpu_failures;
    uint64_t gpu_shared_output_batches;
    uint64_t gpu_shared_output_jobs;
    uint64_t gpu_shared_arena_bytes;
    uint64_t gpu_weight_described_bytes;
    uint64_t gpu_weight_registered_bytes;
    uint64_t gpu_weight_pageable_bytes;
    uint64_t gpu_weight_active_window_bytes;
    uint64_t gpu_weight_peak_window_bytes;
    uint64_t gpu_weight_copied_bytes;
    uint32_t gpu_weight_addressability;
    uint32_t gpu_weight_described_resources;
    uint32_t gpu_weight_active_resources;
    uint32_t gpu_weight_active_windows;
    uint32_t token_program_cells;
    uint32_t token_caller_submissions;
    uint32_t token_internal_barriers;
    uint32_t token_completion_fences;
    uint32_t token_intermediate_publications;
    uint32_t token_final_publications;
    uint64_t token_cpu_pool_sessions;
    uint64_t token_cpu_pool_phases;
    uint64_t gpu_attention_batches;
    uint64_t gpu_attention_tasks;
    uint64_t gpu_attention_kv_bytes;
    uint64_t prefill_ffn_arena_bytes;
    uint64_t prefill_ffn_arena_calls;
    uint64_t dense_payload_bytes;
    int dense_retained_immutable;
    int expert_capacity_slots;
    int expert_resident_slots;
    int expert_peak_slots;
    int expert_preload_enabled;
    int expert_preloaded_slots;
    int expert_preload_layer_count;
    int expert_preload_protected_slots;
    uint64_t expert_preload_layer_mask;
    uint64_t expert_preload_fetch_jobs;
    int expert_inflight_slots;
    int expert_peak_inflight_slots;
    int q4_pool_workers;
} SaltGemma4MemoryStats;

typedef struct SaltGemma4AttentionPrepareStats {
    uint32_t requested_items;
    uint32_t prepared_items;
    uint32_t resident_hits;
    uint32_t fetch_jobs;
} SaltGemma4AttentionPrepareStats;

typedef struct SaltGemma4AttentionLeaseEntry {
    uint32_t layer;
    uint32_t expert;
    int32_t slot;
} SaltGemma4AttentionLeaseEntry;

/* Caller-owned operation lease over the existing expert cache. Exact router
 * results may mark covered entries, but never consume this descriptor as route
 * authority. */
typedef struct SaltGemma4AttentionPlanLease {
    SaltGemma4AttentionLeaseEntry entries[SALT_DPR_MAX_RELEVANCE_ITEMS];
    uint64_t used_mask;
    uint32_t count;
    uint32_t exact_routes;
    uint32_t ready_matches;
    int active;
} SaltGemma4AttentionPlanLease;

typedef struct SaltGemma4RouteObservation {
    uint32_t kind;
    uint64_t position;
    uint32_t layer;
    uint32_t expert;
} SaltGemma4RouteObservation;

enum {
    SALT_GEMMA4_ROUTE_DECODE = 1,
    SALT_GEMMA4_ROUTE_PREFILL = 2
};

typedef struct SaltGemma4RouteObserver {
    SaltGemma4RouteObservation *items;
    size_t capacity;
    size_t count;
    int overflow;
    /* Zero records both kinds; otherwise record only this exact kind. */
    uint32_t route_kind;
} SaltGemma4RouteObserver;

/* Consume a private SALT_GEMMA4_TEXT_BINDING_V3 stream whose descriptors are
 * already authenticated and retained by the caller. The loader revalidates
 * descriptor identity and exact tensor closure before mapping any bytes. */
SaltGemma4Text *salt_gemma4_text_load(FILE *binding, int max_context,
                                      int expert_workers,
                                      char *error, size_t error_size);

void salt_gemma4_text_free(SaltGemma4Text *model);
/* Propagate the existing request-scoped strict-proof boundary into physical
 * resource verification. This changes no arithmetic, KV, or commit semantics. */
int salt_gemma4_text_set_proof_state(SaltGemma4Text *model, int enabled);

int salt_gemma4_text_encode(const SaltGemma4Text *model, const char *text,
                            int *tokens, int token_capacity);
int salt_gemma4_text_decode(const SaltGemma4Text *model, const int *tokens,
                            int token_count, char *text, int text_capacity);

/* Execute one sequential token. logits may be NULL during prefill; otherwise
 * it must hold 262144 floats. The model owns and advances its bounded KV
 * state. This local QA API does not set runtime_ready or expose serving. */
int salt_gemma4_text_step(SaltGemma4Text *model, int token, float *logits);

/* Append a text prompt through the established layer-major Q4 batch path.
 * Existing KV/state is the prefix; logits predicts the first generated token.
 * This private runtime surface does not change readiness. */
int salt_gemma4_text_prefill(SaltGemma4Text *model,
                             const int *tokens, int token_count,
                             float *logits);

/* Private engine-level target-block transition. The startup-compiled portable
 * executor verifies candidate tokens against parent_logits, commits only
 * target-generated accepted-prefix K/V rows, scrubs every rejected suffix row,
 * and returns one unconsumed correction/bonus token. It never steps the
 * pending token and does not change serving readiness. */
int salt_gemma4_text_target_block(SaltGemma4Text *model,
                                  const int *candidate_tokens,
                                  int candidate_count,
                                  const float *parent_logits,
                                  SaltTextVerifyResult *result);

/* Tree-causal target frontier. Nodes are parent-before-child and use unique
 * startup-owned tentative state seats. Target logits alone select the winning
 * root/child path; only that path commits and the returned pending token is not
 * stepped. */
int salt_gemma4_text_target_frontier(
    SaltGemma4Text *model,
    const SaltTextTargetNode *nodes, int node_count,
    const float *parent_logits, SaltTextVerifyResult *result);
int salt_gemma4_text_target_cascade(
    SaltGemma4Text *model,
    const int32_t *seed_token_ids, int tile_count, int frontier_width,
    const float *parent_logits,
    SaltTextTargetNode *node_scratch, int node_scratch_capacity,
    SaltTextVerifyResult *result);
int salt_gemma4_text_target_route(
    SaltGemma4Text *model,
    const int32_t *route_token_ids, int candidate_count,
    const float *parent_logits, SaltTextVerifyResult *result);
int salt_gemma4_text_target_epoch(
    SaltGemma4Text *model, SaltTextTokenEpochController *controller,
    const int32_t *route_token_ids, int candidate_count,
    const float *parent_logits,
    SaltTextTokenExactLookupCommit exact_lookup_commit,
    void *exact_context, SaltTextTokenEpochResult *result);
int salt_gemma4_text_target_policy_get(
    const SaltGemma4Text *model, SaltTextTargetPolicy *policy);

/* Candidate-source-neutral progressive target verification. Tiles start at
 * seed_candidates and double up to maximum_tile_candidates after full hits. */
int salt_gemma4_text_target_progressive(
    SaltGemma4Text *model,
    const int *candidate_tokens,
    int candidate_count,
    const float *parent_logits,
    int seed_candidates,
    int maximum_tile_candidates,
    SaltTextProgressiveResult *result);

/* DPR target verification: append one candidate block through the same
 * layer-major prefill path and emit target logits after every candidate token.
 * position_logits is [token_count][logits_stride], stride >= vocab size. */
int salt_gemma4_text_prefill_verify(SaltGemma4Text *model,
                                    const int *tokens, int token_count,
                                    float *position_logits,
                                    size_t logits_stride,
                                    float *final_logits);

/* Scrub target-verification rows after new_position and publish the accepted
 * prefix position. */
int salt_gemma4_text_rollback_position(SaltGemma4Text *model,
                                       int new_position);

/* Execute an exact one-image prompt as a layer-major prefill. The prompt must
 * contain one contiguous run of mm_token_type==1 placeholders, bracketed by
 * the pinned <|image> / <image|> token IDs, and image_features must contain
 * one 2816-float row per placeholder. Global layers remain causal; sliding
 * layers use sliding_window AND (causal OR same-image-block). On success the
 * model owns a populated KV cache at token_count and logits predicts the first
 * generated token. This private QA API does not enable serving or readiness. */
int salt_gemma4_text_prefill_image(SaltGemma4Text *model,
                                   const int *tokens,
                                   const unsigned char *mm_token_type,
                                   int token_count,
                                   const float *image_features,
                                   int image_feature_tokens,
                                   float *logits);

/* Private QA-only in-memory KV snapshot boundary. A snapshot contains the
 * populated key/value prefix and position, but not first-token logits; callers
 * that resume generation must retain the logits produced at the prefill
 * boundary separately. The format is bound to the exact model instance and
 * validates geometry, finite values, payload SHA-256, and restored destination
 * bytes. It is process-local and is not a persistence or serving API. */
size_t salt_gemma4_text_kv_snapshot_size(const SaltGemma4Text *model);
size_t salt_gemma4_text_kv_snapshot_header_size(void);
int salt_gemma4_text_kv_snapshot(const SaltGemma4Text *model,
                                 void *buffer, size_t buffer_size);
/* Destructive proof helper: overwrite every live prefix K/V value with NaN.
 * A subsequent warm decode may proceed only after a successful restore. */
int salt_gemma4_text_kv_poison_prefix(SaltGemma4Text *model, int position,
                                      size_t *poisoned_bytes);
int salt_gemma4_text_kv_restore(SaltGemma4Text *model,
                                const void *buffer, size_t buffer_size);

/* Transferable cold-start cache format (G4KVC006). Unlike the private QA
 * snapshot above, this format is bound to the manifest-derived portable
 * compatibility identity spanning package, tokenizer/control, template,
 * arithmetic, and engine semantics rather than a process or platform build.
 * Each host authenticates its native binary separately. Integer header fields
 * are little-endian. Sliding layers store exact finite IEEE-754 binary32 K and
 * V rows. Full shared-projection layers store the normalized V base once and
 * reconstruct their exact scaled/RoPE K rows during import. */
size_t salt_gemma4_text_kv_export_size(const SaltGemma4Text *model);
int salt_gemma4_text_kv_export(const SaltGemma4Text *model,
                               void *buffer, size_t buffer_size);
int salt_gemma4_text_kv_import(SaltGemma4Text *model,
                               const void *buffer, size_t buffer_size);
/* Bounded-memory G4KVC006 transfer over an already-open stream. These APIs
 * never allocate or mmap a second KV-sized object. Export scans the live rows
 * to authenticate them, then writes the canonical header/payload. Import
 * writes into the one startup-owned KV arena and scrubs it on any failure. */
int salt_gemma4_text_kv_export_stream(const SaltGemma4Text *model, FILE *stream);
int salt_gemma4_text_kv_import_stream(SaltGemma4Text *model, FILE *stream);
int salt_gemma4_text_kv_import_stream_sha256(
    SaltGemma4Text *model, FILE *stream, uint8_t file_sha256[32]);

/* Bind Gemma's existing live state and G4KVC006 adapter to the engine-owned
 * state materialization transaction. Artifact serialization remains
 * model-specific; begin/finish/abort and backend materialization are common. */
int salt_gemma4_text_state_view(const SaltGemma4Text *model,
                                SaltStateView *view);
int salt_gemma4_text_state_transaction_begin(
    SaltGemma4Text *model, SaltStateTransactionKind kind,
    SaltStateArtifactKind artifact, SaltStateTransactionReason reason,
    SaltStateTransaction *transaction);
int salt_gemma4_text_state_transaction_finish(
    SaltGemma4Text *model, SaltStateTransaction *transaction);
/* Parent exact-transition operation path. The stream is an already-materialized
 * authenticated G4KVC006 reference at source_position + token_count. Read only
 * rows [source_position, source_position + token_count) and append them directly
 * to the one live session KV arena. Existing prefix rows are never rewritten.
 * Complete payload hashing remains the caller's strict-proof boundary. */
int salt_gemma4_text_kv_import_delta_stream(
    SaltGemma4Text *model, FILE *stream,
    int source_position, int token_count);
/* Import replaceable factual rows from a complete G4KVC006 reference that was
 * compiled under the exact immutable shared mindset prefix. The live model
 * must be positioned at mindset_end and every shared layer must end there.
 * Prefix rows are byte-compared before any factual row can commit. */
int salt_gemma4_text_kv_import_facts_stream(
    SaltGemma4Text *model, FILE *stream,
    int mindset_end, int token_count);
/* Attach a validated G4KVC006 hybrid payload as an immutable prefix. The caller
 * owns the read-only mapping and must retain it until the model is freed. Full
 * shared-projection V rows remain mapped while exact K rows are reconstructed
 * into the existing live arena; sliding-layer suffixes use the model-owned ring. */
int salt_gemma4_text_kv_attach_shared(SaltGemma4Text *model,
                                      const void *buffer, size_t buffer_size);
int salt_gemma4_text_shared_position(const SaltGemma4Text *model);

int salt_gemma4_text_position(const SaltGemma4Text *model);
int salt_gemma4_text_context_capacity(const SaltGemma4Text *model);
int salt_gemma4_text_prefill_chunk_capacity(const SaltGemma4Text *model);
size_t salt_gemma4_text_kv_capacity_bytes(const SaltGemma4Text *model);
size_t salt_gemma4_text_kv_required_bytes(const SaltGemma4Text *model,
                                          int position);
int salt_gemma4_text_vocab_size(const SaltGemma4Text *model);
int salt_gemma4_text_memory_stats(const SaltGemma4Text *model,
                                  SaltGemma4MemoryStats *stats);
/* Apply one validated A0 Mindset Attention plan as resource preparation only.
 * Return 0 on success, 1 when the valid plan uses a relevance kind this Gemma
 * track does not support, or -1 on malformed input/cache failure. Arithmetic,
 * router selection, KV, and target position are unchanged. */
int salt_gemma4_text_prepare_attention_plan(
    SaltGemma4Text *model, const SaltDprComputeIntent *intent,
    const SaltDprAttentionPlan *plan,
    SaltGemma4AttentionPrepareStats *stats);
int salt_gemma4_text_prepare_attention_plan_leased(
    SaltGemma4Text *model, const SaltDprComputeIntent *intent,
    const SaltDprAttentionPlan *plan, SaltGemma4AttentionPlanLease *lease,
    SaltGemma4AttentionPrepareStats *stats);
int salt_gemma4_text_release_attention_plan_lease(
    SaltGemma4Text *model, SaltGemma4AttentionPlanLease *lease);
/* Caller-owned exact-route observation. It records canonical router outputs
 * only when an explicit NM/qualification caller attaches it and never
 * influences selection, cache admission, arithmetic, or state. */
int salt_gemma4_text_route_observer_begin(
    SaltGemma4Text *model, SaltGemma4RouteObserver *observer);
int salt_gemma4_text_route_observer_end(SaltGemma4Text *model);
/* Compile exact in-memory route observations into the existing caller-owned NM
 * attention-plan template. Only matching route_kind observations contribute.
 * Plan metadata/provenance/policy remain caller-owned; model state is unchanged. */
int salt_gemma4_text_route_observer_compile_attention_plan(
    const SaltGemma4RouteObserver *observer, uint32_t route_kind,
    uint32_t maximum_per_layer, uint32_t maximum_items,
    SaltDprAttentionPlan *plan);
/* Merge matching exact observations into a caller-owned bounded history.
 * The merge is all-or-nothing on malformed input or counter overflow. */
int salt_gemma4_text_route_observer_update_coverage_ledger(
    const SaltGemma4RouteObserver *observer, uint32_t route_kind,
    SaltDprExpertCoverageLedger *ledger);
/* QA-only same-binary factor-ownership control. Must be selected at position 0;
 * arithmetic and state transitions remain identical. Persistent serving does
 * not expose this function as a request option. */
int salt_gemma4_text_use_ondemand_rope_for_proof(SaltGemma4Text *model,
                                                  int enabled);
/* One-shot proof boundary: revalidate authenticated file identities and the
 * complete bound tensor/layer closure. Normal operation never repeats it. */
int salt_gemma4_text_validate_proof(SaltGemma4Text *model);
/* Architecture/build-independent complete-state and factual-state audits.
 * Host binary authentication remains separate through
 * salt_gemma4_text_build_identity_sha256(). */
int salt_gemma4_text_state_sha256(const SaltGemma4Text *model,
                                  uint8_t out[32]);
/* Architecture/build-independent strict transition audit. Binds compatible
 * model geometry, canonical position, and complete live KV rows, but never the
 * host binary build receipt. Ordinary DPR lookup uses the incremental chain in
 * salt/dpr.h and must not call this per token. */
int salt_gemma4_text_transition_sha256(const SaltGemma4Text *model,
                                       uint8_t out[32]);
int salt_gemma4_text_build_identity_sha256(const SaltGemma4Text *model,
                                           uint8_t out[32]);
int salt_gemma4_text_facts_sha256(const SaltGemma4Text *model,
                                  int mindset_end, uint8_t out[32]);
int salt_gemma4_text_prefix_sha256(const SaltGemma4Text *model,
                                   int position, uint8_t out[32]);
/* Validate all layer ranges, scrub every factual K/V row at and after
 * mindset_end, then publish position=mindset_end. */
int salt_gemma4_text_clear_facts(SaltGemma4Text *model, int mindset_end);

#endif /* SALT_GEMMA4_TEXT_H */
