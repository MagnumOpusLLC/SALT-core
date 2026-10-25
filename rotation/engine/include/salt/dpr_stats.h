#ifndef SALT_DPR_STATS_H
#define SALT_DPR_STATS_H

#include "salt/dpr_store.h"

#include <stddef.h>
#include <stdint.h>

#define SALT_DPR_STATS_VERSION 1u
#define SALT_DPR_STATS_MIN_SAMPLES 3u
#define SALT_DPR_STATS_HISTOGRAM_BINS (SALT_DPR_MAX_HORIZON + 1u)
#define SALT_DPR_STATS_FILE_BYTES 768u

typedef struct SaltDprEdgeStats {
    uint32_t schema_version;
    SaltDprMode mode;
    uint32_t horizon;
    uint32_t reserved;
    uint8_t edge_sha256[32];
    uint64_t lookup_count;
    uint64_t hit_count;
    uint64_t round_count;
    uint64_t fallback_count;
    uint64_t lookup_ns_total;
    uint64_t recover_ns_total;
    uint64_t verify_ns_total;
    uint64_t accepted_tokens_total;
    uint64_t committed_tokens_total;
    uint64_t last_used_epoch;
    uint64_t persist_failures;
    uint64_t accepted_prefix_histogram[SALT_DPR_STATS_HISTOGRAM_BINS];
} SaltDprEdgeStats;

typedef struct SaltDprRetentionItem {
    size_t edge_index;
    uint8_t edge_sha256[32];
    uint64_t reference_bytes;
    SaltDprChart static_chart;
    const SaltDprEdgeStats *observed;
} SaltDprRetentionItem;

int salt_dpr_stats_init(SaltDprEdgeStats *stats,
                        const uint8_t edge_sha256[32],
                        SaltDprMode mode, uint32_t horizon);
int salt_dpr_stats_validate(const SaltDprEdgeStats *stats);
int salt_dpr_stats_observe(SaltDprEdgeStats *stats,
                           uint64_t lookup_ns, uint64_t recover_ns,
                           uint64_t verify_ns, uint32_t accepted_tokens,
                           uint32_t committed_tokens, int hit, int fallback,
                           uint64_t epoch);
int salt_dpr_stats_chart(const SaltDprEdgeStats *stats, SaltDprChart *chart);
int salt_dpr_stats_is_green(const SaltDprEdgeStats *stats,
                            uint64_t serial_ns_per_token,
                            uint64_t minimum_samples);
int salt_dpr_stats_store_select(
    const SaltDprStore *store, const SaltDprEdgeStats *stats,
    const uint8_t *retained, const uint8_t source_node_sha256[32],
    SaltDprMode mode, uint64_t serial_ns_per_token,
    uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
    SaltDprStopTokenFn stop_token, void *stop_opaque,
    size_t *selected_index);
int salt_dpr_stats_store_select_kind(
    const SaltDprStore *store, const SaltDprEdgeStats *stats,
    const uint8_t *retained, const uint8_t source_node_sha256[32],
    SaltDprMode mode, SaltDprEdgeKind kind,
    uint64_t serial_ns_per_token, uint64_t minimum_samples,
    int32_t vocab_size, uint64_t max_context,
    SaltDprStopTokenFn stop_token, void *stop_opaque,
    size_t *selected_index);
int salt_dpr_stats_walk_nomogram(
    const SaltDprStore *store, const SaltDprEdgeStats *stats,
    const uint8_t *retained, const uint8_t start_node_sha256[32],
    SaltDprMode mode, uint64_t serial_ns_per_token,
    uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
    SaltDprStopTokenFn stop_token, void *stop_opaque,
    uint32_t max_tokens, int32_t *candidate_token_ids,
    size_t *edge_indices, uint32_t *candidate_count);
/* Exact first-token coverage over retained roots. This extends eligibility,
 * not acceptance: the caller must still target-verify every returned token.
 * Retention is restored before returning, including on selector refusal. */
int salt_dpr_stats_walk_nomogram_coverage(
    const SaltDprStore *store, const SaltDprEdgeStats *stats,
    uint8_t *retained, const uint8_t start_node_sha256[32],
    SaltDprMode mode, uint64_t serial_ns_per_token,
    uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
    SaltDprStopTokenFn stop_token, void *stop_opaque,
    int32_t required_first_token, uint32_t max_tokens,
    int32_t *candidate_token_ids, size_t *edge_indices,
    uint32_t *candidate_count, uint32_t *root_window_count);
int salt_dpr_stats_select_prefill(
    const SaltDprStore *store, const SaltDprEdgeStats *stats,
    const uint8_t *retained, const uint8_t source_node_sha256[32],
    SaltDprMode mode, uint64_t serial_ns_per_token,
    uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
    const int32_t *prompt_token_ids, uint32_t max_tokens,
    size_t *selected_index);

int salt_dpr_stats_prepare(const char *root);
int salt_dpr_stats_path(const char *root, const uint8_t edge_sha256[32],
                        char *out, size_t out_size);
/* load: 0 loaded, 1 absent, -1 invalid/error. */
int salt_dpr_stats_load(const char *root, const uint8_t edge_sha256[32],
                        SaltDprEdgeStats *stats);
int salt_dpr_stats_save(const char *root, const SaltDprEdgeStats *stats);

int salt_dpr_retention_plan(const SaltDprRetentionItem *items,
                            size_t item_count, uint64_t budget_bytes,
                            uint64_t serial_ns_per_token,
                            uint8_t *retained, size_t *order,
                            size_t order_capacity);

#endif
