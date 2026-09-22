#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#else
#define _POSIX_C_SOURCE 200809L
#endif

#include "salt/dpr_stats.h"
#include "sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define DPR_STATS_PATH_MAX 4096u

static const uint8_t STATS_MAGIC[8] = {'S','A','L','T','D','P','R','S'};

static int digest_is_zero(const uint8_t value[32]) {
    uint8_t total = 0;
    if (!value) return 1;
    for (size_t i = 0; i < 32; i++) total |= value[i];
    return total == 0;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (!out || left > UINT64_MAX - right) return -1;
    *out = left + right;
    return 0;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *out) {
    if (!out || (right && left > UINT64_MAX / right)) return -1;
    *out = left * right;
    return 0;
}

static void le32_store(uint8_t *out, uint32_t value) {
    for (unsigned i = 0; i < 4; i++) out[i] = (uint8_t)(value >> (8u * i));
}

static void le64_store(uint8_t *out, uint64_t value) {
    for (unsigned i = 0; i < 8; i++) out[i] = (uint8_t)(value >> (8u * i));
}

static uint32_t le32_load(const uint8_t *in) {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; i++) value |= (uint32_t)in[i] << (8u * i);
    return value;
}

static uint64_t le64_load(const uint8_t *in) {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; i++) value |= (uint64_t)in[i] << (8u * i);
    return value;
}

static int fraction_compare(uint64_t left_num, uint64_t left_den,
                            uint64_t right_num, uint64_t right_den) {
    int reversed = 0;
    if (!left_den || !right_den) return 0;
    for (;;) {
        uint64_t left_q = left_num / left_den;
        uint64_t right_q = right_num / right_den;
        uint64_t left_r, right_r;
        if (left_q != right_q) {
            int result = left_q < right_q ? -1 : 1;
            return reversed ? -result : result;
        }
        left_r = left_num % left_den;
        right_r = right_num % right_den;
        if (!left_r || !right_r) {
            int result;
            if (!left_r && !right_r) return 0;
            result = !left_r ? -1 : 1;
            return reversed ? -result : result;
        }
        left_num = left_den;
        left_den = left_r;
        right_num = right_den;
        right_den = right_r;
        reversed = !reversed;
    }
}

static int private_directory(const char *path) {
    struct stat value;
    if (!path || lstat(path, &value) != 0 || !S_ISDIR(value.st_mode) ||
        (value.st_mode & 077u) != 0 || value.st_uid != geteuid())
        return -1;
    return 0;
}

static int private_file_stat(const struct stat *value) {
    return value && S_ISREG(value->st_mode) && value->st_nlink == 1 &&
           (value->st_mode & 077u) == 0 && value->st_uid == geteuid() ? 0 : -1;
}

static void digest_hex(const uint8_t digest[32], char out[65]) {
    static const char digits[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        out[2 * i] = digits[digest[i] >> 4];
        out[2 * i + 1] = digits[digest[i] & 15u];
    }
    out[64] = 0;
}

int salt_dpr_stats_init(SaltDprEdgeStats *stats,
                        const uint8_t edge_sha256[32],
                        SaltDprMode mode, uint32_t horizon) {
    if (!stats || !edge_sha256 || digest_is_zero(edge_sha256) ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC) ||
        horizon == 0 || horizon > SALT_DPR_MAX_HORIZON)
        return -1;
    memset(stats, 0, sizeof *stats);
    stats->schema_version = SALT_DPR_STATS_VERSION;
    stats->mode = mode;
    stats->horizon = horizon;
    memcpy(stats->edge_sha256, edge_sha256, 32);
    return 0;
}

int salt_dpr_stats_validate(const SaltDprEdgeStats *stats) {
    uint64_t maximum, histogram_rounds = 0, histogram_accepted = 0;
    if (!stats || stats->schema_version != SALT_DPR_STATS_VERSION ||
        (stats->mode != SALT_DPR_PERSIST &&
         stats->mode != SALT_DPR_DYNAMIC) ||
        stats->horizon == 0 || stats->horizon > SALT_DPR_MAX_HORIZON ||
        stats->reserved != 0 || digest_is_zero(stats->edge_sha256) ||
        stats->hit_count != stats->round_count ||
        stats->hit_count > stats->lookup_count ||
        stats->fallback_count > stats->lookup_count ||
        stats->last_used_epoch < stats->lookup_count ||
        mul_u64(stats->round_count, stats->horizon, &maximum) != 0 ||
        stats->accepted_tokens_total > maximum ||
        stats->committed_tokens_total < stats->accepted_tokens_total ||
        stats->committed_tokens_total >
            stats->accepted_tokens_total + stats->round_count)
        return -1;
    for (uint32_t i = 0; i < SALT_DPR_STATS_HISTOGRAM_BINS; i++) {
        uint64_t weighted;
        if (i > stats->horizon && stats->accepted_prefix_histogram[i])
            return -1;
        if (add_u64(histogram_rounds,
                    stats->accepted_prefix_histogram[i],
                    &histogram_rounds) != 0 ||
            mul_u64(i, stats->accepted_prefix_histogram[i], &weighted) != 0 ||
            add_u64(histogram_accepted, weighted,
                    &histogram_accepted) != 0)
            return -1;
    }
    return histogram_rounds == stats->round_count &&
           histogram_accepted == stats->accepted_tokens_total ? 0 : -1;
}

int salt_dpr_stats_observe(SaltDprEdgeStats *stats,
                           uint64_t lookup_ns, uint64_t recover_ns,
                           uint64_t verify_ns, uint32_t accepted_tokens,
                           uint32_t committed_tokens, int hit, int fallback,
                           uint64_t epoch) {
    SaltDprEdgeStats next;
    if (salt_dpr_stats_validate(stats) != 0 ||
        (hit != 0 && hit != 1) || (fallback != 0 && fallback != 1) ||
        (hit && fallback) || accepted_tokens > stats->horizon ||
        (!hit && (accepted_tokens || committed_tokens || verify_ns)) ||
        (hit && (committed_tokens < accepted_tokens ||
                 committed_tokens > accepted_tokens + 1u)) ||
        epoch <= stats->last_used_epoch)
        return -1;
    next = *stats;
    if (add_u64(next.lookup_count, 1, &next.lookup_count) != 0 ||
        add_u64(next.lookup_ns_total, lookup_ns,
                &next.lookup_ns_total) != 0 ||
        add_u64(next.recover_ns_total, recover_ns,
                &next.recover_ns_total) != 0)
        return -1;
    if (fallback && add_u64(next.fallback_count, 1,
                            &next.fallback_count) != 0)
        return -1;
    if (hit) {
        if (add_u64(next.hit_count, 1, &next.hit_count) != 0 ||
            add_u64(next.round_count, 1, &next.round_count) != 0 ||
            add_u64(next.verify_ns_total, verify_ns,
                    &next.verify_ns_total) != 0 ||
            add_u64(next.accepted_tokens_total, accepted_tokens,
                    &next.accepted_tokens_total) != 0 ||
            add_u64(next.committed_tokens_total, committed_tokens,
                    &next.committed_tokens_total) != 0 ||
            add_u64(next.accepted_prefix_histogram[accepted_tokens], 1,
                    &next.accepted_prefix_histogram[accepted_tokens]) != 0)
            return -1;
    }
    next.last_used_epoch = epoch;
    if (salt_dpr_stats_validate(&next) != 0) return -1;
    *stats = next;
    return 0;
}

int salt_dpr_stats_chart(const SaltDprEdgeStats *stats, SaltDprChart *chart) {
    if (salt_dpr_stats_validate(stats) != 0 || !chart || !stats->round_count)
        return -1;
    memset(chart, 0, sizeof *chart);
    chart->horizon = stats->horizon;
    chart->lookup_ns = stats->lookup_ns_total / stats->round_count;
    chart->recover_ns = stats->recover_ns_total / stats->round_count;
    chart->verify_ns = stats->verify_ns_total / stats->round_count;
    chart->accepted_tokens_total = stats->accepted_tokens_total;
    chart->rounds = stats->round_count;
    chart->sample_count = stats->round_count;
    return salt_dpr_chart_validate(chart);
}

int salt_dpr_stats_is_green(const SaltDprEdgeStats *stats,
                            uint64_t serial_ns_per_token,
                            uint64_t minimum_samples) {
    uint64_t cost, committed;
    if (salt_dpr_stats_validate(stats) != 0 || !serial_ns_per_token ||
        stats->round_count < minimum_samples || !stats->round_count ||
        add_u64(stats->lookup_ns_total, stats->recover_ns_total, &cost) != 0 ||
        add_u64(cost, stats->verify_ns_total, &cost) != 0 ||
        add_u64(stats->accepted_tokens_total, stats->round_count,
                &committed) != 0)
        return 0;
    return fraction_compare(cost, committed, serial_ns_per_token, 1) < 0;
}

static size_t node_bucket(const uint8_t node[32], size_t bucket_count) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < 32; i++) {
        hash ^= node[i];
        hash *= UINT64_C(1099511628211);
    }
    return (size_t)(hash % bucket_count);
}

int salt_dpr_stats_store_select_kind(
        const SaltDprStore *store, const SaltDprEdgeStats *stats,
        const uint8_t *retained, const uint8_t source_node_sha256[32],
        SaltDprMode mode, SaltDprEdgeKind kind,
        uint64_t serial_ns_per_token,
        uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
        SaltDprStopTokenFn stop_token, void *stop_opaque,
        size_t *selected_index) {
    uint32_t index;
    size_t bucket, walked = 0, best = SIZE_MAX;
    SaltDprChart best_chart;
    if (!store || !store->edges || !store->buckets || !stats || !retained ||
        !source_node_sha256 || !selected_index || !store->bucket_count ||
        !serial_ns_per_token ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC) ||
        (kind != SALT_DPR_EDGE_PARENT_EXACT &&
         kind != SALT_DPR_EDGE_NOMOGRAM_DRAFT &&
         kind != SALT_DPR_EDGE_PARENT_PREFILL &&
         kind != SALT_DPR_EDGE_NOMOGRAM_PREFILL))
        return -1;
    *selected_index = SIZE_MAX;
    memset(&best_chart, 0, sizeof best_chart);
    bucket = node_bucket(source_node_sha256, store->bucket_count);
    index = store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        const SaltDprStoredEdge *candidate;
        SaltDprChart candidate_chart;
        int comparison = 0;
        if (index >= store->edge_count || ++walked > store->edge_count)
            return -1;
        candidate = &store->edges[index];
        candidate_chart = candidate->edge.chart;
        if (retained[index] && candidate->edge.mode == mode &&
            candidate->edge.kind == kind &&
            memcmp(candidate->edge.source_node_sha256,
                   source_node_sha256, 32) == 0 &&
            salt_dpr_edge_validate(
                &candidate->edge, vocab_size, max_context,
                stop_token, stop_opaque) == 0 &&
            salt_dpr_chart_is_green(
                &candidate->edge.chart,
                serial_ns_per_token, minimum_samples)) {
            if (salt_dpr_stats_validate(&stats[index]) != 0 ||
                memcmp(stats[index].edge_sha256,
                       candidate->edge_sha256, 32) != 0 ||
                stats[index].mode != candidate->edge.mode ||
                stats[index].horizon != candidate->edge.horizon)
                return -1;
            if (stats[index].round_count >= minimum_samples) {
                if (!salt_dpr_stats_is_green(
                        &stats[index], serial_ns_per_token,
                        minimum_samples)) {
                    index = candidate->next_index;
                    continue;
                }
                if (salt_dpr_stats_chart(
                        &stats[index], &candidate_chart) != 0)
                    return -1;
            }
            if (best == SIZE_MAX) {
                best = index;
                best_chart = candidate_chart;
            } else if (salt_dpr_chart_compare(
                    &candidate_chart, &best_chart, &comparison) != 0) {
                return -1;
            } else if (comparison < 0 ||
                       (comparison == 0 &&
                        memcmp(candidate->edge_sha256,
                               store->edges[best].edge_sha256, 32) < 0)) {
                best = index;
                best_chart = candidate_chart;
            }
        }
        index = candidate->next_index;
    }
    if (best == SIZE_MAX) return 0;
    *selected_index = best;
    return 1;
}

int salt_dpr_stats_store_select(
        const SaltDprStore *store, const SaltDprEdgeStats *stats,
        const uint8_t *retained, const uint8_t source_node_sha256[32],
        SaltDprMode mode, uint64_t serial_ns_per_token,
        uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
        SaltDprStopTokenFn stop_token, void *stop_opaque,
        size_t *selected_index) {
    return salt_dpr_stats_store_select_kind(
        store, stats, retained, source_node_sha256, mode,
        SALT_DPR_EDGE_PARENT_EXACT, serial_ns_per_token, minimum_samples,
        vocab_size, max_context, stop_token, stop_opaque, selected_index);
}

int salt_dpr_stats_walk_nomogram(
        const SaltDprStore *store, const SaltDprEdgeStats *stats,
        const uint8_t *retained, const uint8_t start_node_sha256[32],
        SaltDprMode mode, uint64_t serial_ns_per_token,
        uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
        SaltDprStopTokenFn stop_token, void *stop_opaque,
        uint32_t max_tokens, int32_t *candidate_token_ids,
        size_t *edge_indices, uint32_t *candidate_count) {
    uint8_t node[32], visited[SALT_DPR_MAX_HORIZON][32];
    uint32_t count = 0;
    if (!store || !stats || !retained || !start_node_sha256 ||
        !candidate_token_ids || !candidate_count || max_tokens == 0 ||
        max_tokens > SALT_DPR_MAX_HORIZON)
        return -1;
    memcpy(node, start_node_sha256, 32);
    while (count < max_tokens) {
        size_t selected = SIZE_MAX;
        int found;
        for (uint32_t i = 0; i < count; i++)
            if (memcmp(visited[i], node, 32) == 0) return -1;
        memcpy(visited[count], node, 32);
        found = salt_dpr_stats_store_select_kind(
            store, stats, retained, node, mode, SALT_DPR_EDGE_NOMOGRAM_DRAFT,
            serial_ns_per_token, minimum_samples, vocab_size, max_context,
            stop_token, stop_opaque, &selected);
        if (found < 0) return -1;
        if (found == 0) break;
        candidate_token_ids[count] =
            store->edges[selected].edge.candidate_token_ids[0];
        if (edge_indices) edge_indices[count] = selected;
        memcpy(node, store->edges[selected].edge.next_node_sha256, 32);
        count++;
    }
    *candidate_count = count;
    return count == 0 ? 0 : 1;
}

int salt_dpr_stats_walk_nomogram_coverage(
        const SaltDprStore *store, const SaltDprEdgeStats *stats,
        uint8_t *retained, const uint8_t start_node_sha256[32],
        SaltDprMode mode, uint64_t serial_ns_per_token,
        uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
        SaltDprStopTokenFn stop_token, void *stop_opaque,
        int32_t required_first_token, uint32_t max_tokens,
        int32_t *candidate_token_ids, size_t *edge_indices,
        uint32_t *candidate_count, uint32_t *root_window_count) {
    uint8_t node[32], visited[SALT_DPR_MAX_HORIZON][32];
    size_t masked[SALT_DPR_MAX_HORIZON];
    uint8_t masked_values[SALT_DPR_MAX_HORIZON];
    uint32_t count = 0, roots = 0;
    if (!store || !stats || !retained || !start_node_sha256 ||
        required_first_token < 0 || required_first_token >= vocab_size ||
        !candidate_token_ids || !candidate_count || !root_window_count ||
        max_tokens == 0 || max_tokens > SALT_DPR_MAX_HORIZON)
        return -1;
    memcpy(node, start_node_sha256, 32);
    while (count < max_tokens) {
        size_t selected = SIZE_MAX;
        int found;
        for (uint32_t prior = 0; prior < count; prior++)
            if (memcmp(visited[prior], node, 32) == 0) return -1;
        memcpy(visited[count], node, 32);
        if (count == 0) {
            uint32_t masked_count = 0;
            int exact_match = 0;
            found = 0;
            while (roots < SALT_DPR_MAX_HORIZON) {
                found = salt_dpr_stats_store_select_kind(
                    store, stats, retained, node, mode,
                    SALT_DPR_EDGE_NOMOGRAM_DRAFT,
                    serial_ns_per_token, minimum_samples,
                    vocab_size, max_context, stop_token, stop_opaque,
                    &selected);
                if (found <= 0) break;
                roots++;
                if (store->edges[selected].edge.candidate_token_ids[0] ==
                        required_first_token) {
                    exact_match = 1;
                    break;
                }
                masked_values[masked_count] = retained[selected];
                retained[selected] = 0;
                masked[masked_count++] = selected;
            }
            for (uint32_t index = 0; index < masked_count; index++)
                retained[masked[index]] = masked_values[index];
            if (!exact_match && found >= 0) {
                found = 0;
                selected = SIZE_MAX;
            }
        } else {
            found = salt_dpr_stats_store_select_kind(
                store, stats, retained, node, mode,
                SALT_DPR_EDGE_NOMOGRAM_DRAFT,
                serial_ns_per_token, minimum_samples,
                vocab_size, max_context, stop_token, stop_opaque,
                &selected);
        }
        if (found < 0) return -1;
        if (found == 0 || selected == SIZE_MAX) break;
        candidate_token_ids[count] =
            store->edges[selected].edge.candidate_token_ids[0];
        if (edge_indices) edge_indices[count] = selected;
        memcpy(node, store->edges[selected].edge.next_node_sha256, 32);
        count++;
    }
    *candidate_count = count;
    *root_window_count = roots;
    return count == 0 ? 0 : 1;
}

int salt_dpr_stats_select_prefill(
        const SaltDprStore *store, const SaltDprEdgeStats *stats,
        const uint8_t *retained, const uint8_t source_node_sha256[32],
        SaltDprMode mode, uint64_t serial_ns_per_token,
        uint64_t minimum_samples, int32_t vocab_size, uint64_t max_context,
        const int32_t *prompt_token_ids, uint32_t max_tokens,
        size_t *selected_index) {
    uint32_t index;
    size_t bucket, walked = 0, best = SIZE_MAX;
    SaltDprChart best_chart;
    if (!store || !store->edges || !store->buckets || !stats || !retained ||
        !source_node_sha256 || !prompt_token_ids || !selected_index ||
        !store->bucket_count || !serial_ns_per_token || max_tokens == 0 ||
        max_tokens > SALT_DPR_MAX_HORIZON ||
        (mode != SALT_DPR_PERSIST && mode != SALT_DPR_DYNAMIC))
        return -1;
    *selected_index = SIZE_MAX;
    memset(&best_chart, 0, sizeof best_chart);
    bucket = node_bucket(source_node_sha256, store->bucket_count);
    index = store->buckets[bucket];
    while (index != SALT_DPR_STORE_NONE) {
        const SaltDprStoredEdge *candidate;
        SaltDprChart candidate_chart;
        int comparison = 0;
        if (index >= store->edge_count || ++walked > store->edge_count)
            return -1;
        candidate = &store->edges[index];
        candidate_chart = candidate->edge.chart;
        if (retained[index] && candidate->edge.mode == mode &&
            candidate->edge.kind == SALT_DPR_EDGE_PARENT_PREFILL &&
            candidate->edge.horizon <= max_tokens &&
            memcmp(candidate->edge.source_node_sha256,
                   source_node_sha256, 32) == 0 &&
            memcmp(candidate->edge.candidate_token_ids, prompt_token_ids,
                   (size_t)candidate->edge.horizon * sizeof(int32_t)) == 0 &&
            salt_dpr_edge_validate(&candidate->edge, vocab_size, max_context,
                                   NULL, NULL) == 0 &&
            salt_dpr_chart_is_green(&candidate->edge.chart,
                                    serial_ns_per_token, minimum_samples)) {
            if (salt_dpr_stats_validate(&stats[index]) != 0 ||
                memcmp(stats[index].edge_sha256,
                       candidate->edge_sha256, 32) != 0)
                return -1;
            if (stats[index].round_count >= minimum_samples) {
                if (!salt_dpr_stats_is_green(&stats[index],
                                             serial_ns_per_token,
                                             minimum_samples)) {
                    index = candidate->next_index;
                    continue;
                }
                if (salt_dpr_stats_chart(&stats[index], &candidate_chart) != 0)
                    return -1;
            }
            if (best == SIZE_MAX) {
                best = index;
                best_chart = candidate_chart;
            } else if (salt_dpr_chart_compare(
                    &candidate_chart, &best_chart, &comparison) != 0) {
                return -1;
            } else if (comparison < 0 ||
                       (comparison == 0 &&
                        memcmp(candidate->edge_sha256,
                               store->edges[best].edge_sha256, 32) < 0)) {
                best = index;
                best_chart = candidate_chart;
            }
        }
        index = candidate->next_index;
    }
    if (best == SIZE_MAX) return 0;
    *selected_index = best;
    return 1;
}

int salt_dpr_stats_prepare(const char *root) {
    char path[DPR_STATS_PATH_MAX];
    int length;
    if (private_directory(root) != 0) return -1;
    length = snprintf(path, sizeof path, "%s/stats", root);
    if (length < 0 || (size_t)length >= sizeof path) return -1;
    if (mkdir(path, 0700) != 0 && errno != EEXIST) return -1;
    return private_directory(path);
}

int salt_dpr_stats_path(const char *root, const uint8_t edge_sha256[32],
                        char *out, size_t out_size) {
    char hex[65];
    int length;
    if (!root || !*root || !edge_sha256 || digest_is_zero(edge_sha256) ||
        !out || !out_size)
        return -1;
    digest_hex(edge_sha256, hex);
    length = snprintf(out, out_size, "%s/stats/%s.stat", root, hex);
    return length < 0 || (size_t)length >= out_size ? -1 : 0;
}

static int encode_stats(const SaltDprEdgeStats *stats,
                        uint8_t out[SALT_DPR_STATS_FILE_BYTES]) {
    SaltSha256 hasher;
    if (salt_dpr_stats_validate(stats) != 0 || !out) return -1;
    memset(out, 0, SALT_DPR_STATS_FILE_BYTES);
    memcpy(out, STATS_MAGIC, 8);
    le32_store(out + 8, SALT_DPR_STATS_VERSION);
    le32_store(out + 12, SALT_DPR_STATS_FILE_BYTES);
    le32_store(out + 16, (uint32_t)stats->mode);
    le32_store(out + 20, stats->horizon);
    memcpy(out + 32, stats->edge_sha256, 32);
    le64_store(out + 64, stats->lookup_count);
    le64_store(out + 72, stats->hit_count);
    le64_store(out + 80, stats->round_count);
    le64_store(out + 88, stats->fallback_count);
    le64_store(out + 96, stats->lookup_ns_total);
    le64_store(out + 104, stats->recover_ns_total);
    le64_store(out + 112, stats->verify_ns_total);
    le64_store(out + 120, stats->accepted_tokens_total);
    le64_store(out + 128, stats->committed_tokens_total);
    le64_store(out + 136, stats->last_used_epoch);
    le64_store(out + 144, stats->persist_failures);
    for (uint32_t i = 0; i < SALT_DPR_STATS_HISTOGRAM_BINS; i++)
        le64_store(out + 152 + (size_t)i * 8u,
                   stats->accepted_prefix_histogram[i]);
    salt_sha256_init(&hasher);
    salt_sha256_update(&hasher, out, SALT_DPR_STATS_FILE_BYTES - 32u);
    salt_sha256_final(&hasher, out + SALT_DPR_STATS_FILE_BYTES - 32u);
    return 0;
}

static int decode_stats(const uint8_t in[SALT_DPR_STATS_FILE_BYTES],
                        SaltDprEdgeStats *stats) {
    uint8_t digest[32];
    if (!in || !stats || memcmp(in, STATS_MAGIC, 8) ||
        le32_load(in + 8) != SALT_DPR_STATS_VERSION ||
        le32_load(in + 12) != SALT_DPR_STATS_FILE_BYTES ||
        salt_sha256_bytes(in, SALT_DPR_STATS_FILE_BYTES - 32u, digest) != 0 ||
        memcmp(digest, in + SALT_DPR_STATS_FILE_BYTES - 32u, 32))
        return -1;
    memset(stats, 0, sizeof *stats);
    stats->schema_version = le32_load(in + 8);
    stats->mode = (SaltDprMode)le32_load(in + 16);
    stats->horizon = le32_load(in + 20);
    memcpy(stats->edge_sha256, in + 32, 32);
    stats->lookup_count = le64_load(in + 64);
    stats->hit_count = le64_load(in + 72);
    stats->round_count = le64_load(in + 80);
    stats->fallback_count = le64_load(in + 88);
    stats->lookup_ns_total = le64_load(in + 96);
    stats->recover_ns_total = le64_load(in + 104);
    stats->verify_ns_total = le64_load(in + 112);
    stats->accepted_tokens_total = le64_load(in + 120);
    stats->committed_tokens_total = le64_load(in + 128);
    stats->last_used_epoch = le64_load(in + 136);
    stats->persist_failures = le64_load(in + 144);
    for (uint32_t i = 0; i < SALT_DPR_STATS_HISTOGRAM_BINS; i++)
        stats->accepted_prefix_histogram[i] =
            le64_load(in + 152 + (size_t)i * 8u);
    return salt_dpr_stats_validate(stats);
}

static int read_stats_file(const char *path, SaltDprEdgeStats *stats) {
    uint8_t bytes[SALT_DPR_STATS_FILE_BYTES];
    struct stat value;
    int fd = -1, result = -1;
    size_t offset = 0;
    fd = open(path, O_RDONLY | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT ? 1 : -1;
    if (fstat(fd, &value) != 0 || private_file_stat(&value) != 0 ||
        value.st_size != (off_t)sizeof bytes)
        goto done;
    while (offset < sizeof bytes) {
        ssize_t count = read(fd, bytes + offset, sizeof bytes - offset);
        if (count <= 0) goto done;
        offset += (size_t)count;
    }
    if (fstat(fd, &value) != 0 || value.st_size != (off_t)sizeof bytes)
        goto done;
    result = decode_stats(bytes, stats);
done:
    if (fd >= 0) close(fd);
    return result;
}

int salt_dpr_stats_load(const char *root, const uint8_t edge_sha256[32],
                        SaltDprEdgeStats *stats) {
    char path[DPR_STATS_PATH_MAX];
    int result;
    if (!stats || salt_dpr_stats_path(
            root, edge_sha256, path, sizeof path) != 0)
        return -1;
    result = read_stats_file(path, stats);
    if (result == 0 && memcmp(stats->edge_sha256, edge_sha256, 32)) return -1;
    return result;
}

static int write_all(int fd, const uint8_t *bytes, size_t count) {
    size_t offset = 0;
    while (offset < count) {
        ssize_t written = write(fd, bytes + offset, count - offset);
        if (written <= 0) return -1;
        offset += (size_t)written;
    }
    return 0;
}

int salt_dpr_stats_save(const char *root, const SaltDprEdgeStats *stats) {
    uint8_t bytes[SALT_DPR_STATS_FILE_BYTES];
    char target[DPR_STATS_PATH_MAX], staging[DPR_STATS_PATH_MAX];
    char directory[DPR_STATS_PATH_MAX];
    struct stat value;
    SaltDprEdgeStats readback;
    int fd = -1, directory_fd = -1, result = -1, length;
    if (salt_dpr_stats_prepare(root) != 0 || encode_stats(stats, bytes) != 0 ||
        salt_dpr_stats_path(root, stats->edge_sha256,
                            target, sizeof target) != 0)
        return -1;
    length = snprintf(directory, sizeof directory, "%s/stats", root);
    if (length < 0 || (size_t)length >= sizeof directory) return -1;
    for (unsigned attempt = 0; attempt < 100; attempt++) {
        length = snprintf(staging, sizeof staging, "%s/.%ld.%u.tmp",
                          directory, (long)getpid(), attempt);
        if (length < 0 || (size_t)length >= sizeof staging) return -1;
        fd = open(staging, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
        if (fd >= 0) break;
        if (errno != EEXIST) return -1;
    }
    if (fd < 0 || write_all(fd, bytes, sizeof bytes) != 0 || fsync(fd) != 0 ||
        close(fd) != 0)
        goto done;
    fd = -1;
    errno = 0;
    if (lstat(target, &value) == 0) {
        if (private_file_stat(&value) != 0) goto done;
    } else if (errno != ENOENT) {
        goto done;
    }
    if (rename(staging, target) != 0) goto done;
    staging[0] = 0;
    directory_fd = open(directory, O_RDONLY | O_NOFOLLOW);
    if (directory_fd < 0 || fsync(directory_fd) != 0) goto done;
    if (read_stats_file(target, &readback) != 0 ||
        memcmp(stats, &readback, sizeof *stats))
        goto done;
    result = 0;
done:
    if (directory_fd >= 0) close(directory_fd);
    if (fd >= 0) close(fd);
    if (staging[0]) unlink(staging);
    return result;
}

static int item_benefit(const SaltDprRetentionItem *item,
                        uint64_t serial_ns_per_token, uint64_t *benefit) {
    uint64_t committed, serial, cost, per_round;
    if (!item || !benefit || !item->reference_bytes ||
        digest_is_zero(item->edge_sha256) || !serial_ns_per_token)
        return -1;
    if (item->observed &&
        salt_dpr_stats_validate(item->observed) == 0 &&
        item->observed->round_count >= SALT_DPR_STATS_MIN_SAMPLES) {
        if (add_u64(item->observed->accepted_tokens_total,
                    item->observed->round_count, &committed) != 0 ||
            mul_u64(serial_ns_per_token, committed, &serial) != 0 ||
            add_u64(item->observed->lookup_ns_total,
                    item->observed->recover_ns_total, &cost) != 0 ||
            add_u64(cost, item->observed->verify_ns_total, &cost) != 0)
            return -1;
    } else {
        if (salt_dpr_chart_validate(&item->static_chart) != 0 ||
            !item->static_chart.rounds ||
            add_u64(item->static_chart.accepted_tokens_total,
                    item->static_chart.rounds, &committed) != 0 ||
            mul_u64(serial_ns_per_token, committed, &serial) != 0 ||
            add_u64(item->static_chart.lookup_ns,
                    item->static_chart.recover_ns, &per_round) != 0 ||
            add_u64(per_round, item->static_chart.draft_ns,
                    &per_round) != 0 ||
            add_u64(per_round, item->static_chart.verify_ns,
                    &per_round) != 0 ||
            mul_u64(per_round, item->static_chart.rounds, &cost) != 0)
            return -1;
    }
    *benefit = serial > cost ? serial - cost : 0;
    return 0;
}

int salt_dpr_retention_plan(const SaltDprRetentionItem *items,
                            size_t item_count, uint64_t budget_bytes,
                            uint64_t serial_ns_per_token,
                            uint8_t *retained, size_t *order,
                            size_t order_capacity) {
    uint64_t remaining = budget_bytes;
    size_t retained_count = 0;
    if ((!items && item_count) || (!retained && item_count) ||
        (!order && order_capacity) || order_capacity < item_count ||
        !serial_ns_per_token)
        return -1;
    memset(retained, 0, item_count);
    for (size_t i = 0; i < order_capacity; i++) order[i] = SIZE_MAX;
    for (;;) {
        size_t best = SIZE_MAX;
        uint64_t best_benefit = 0;
        for (size_t i = 0; i < item_count; i++) {
            uint64_t benefit = 0;
            int comparison;
            if (retained[i] != 0 ||
                item_benefit(&items[i], serial_ns_per_token, &benefit) != 0 ||
                benefit == 0)
                continue;
            if (best == SIZE_MAX) {
                best = i;
                best_benefit = benefit;
                continue;
            }
            comparison = fraction_compare(
                benefit, items[i].reference_bytes,
                best_benefit, items[best].reference_bytes);
            if (comparison > 0 ||
                (comparison == 0 &&
                 memcmp(items[i].edge_sha256,
                        items[best].edge_sha256, 32) < 0)) {
                best = i;
                best_benefit = benefit;
            }
        }
        if (best == SIZE_MAX) break;
        if (items[best].reference_bytes <= remaining) {
            retained[best] = 1;
            remaining -= items[best].reference_bytes;
            order[retained_count++] = items[best].edge_index;
        } else {
            retained[best] = 2;
        }
    }
    for (size_t i = 0; i < item_count; i++)
        if (retained[i] == 2) retained[i] = 0;
    return 0;
}
