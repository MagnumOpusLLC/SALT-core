#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L

#include "gemma4_text.h"
#include "gemma4_operation.h"
#include "salt/attn.h"
#include "salt/simd.h"
#include "salt/gpu.h"
#include "salt/dpr.h"
#include "sha256.h"
#include "gemma4_kv_file.h"
#include "gemma4-memory.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define QA_TARGET_ROUTE_COUNT 2

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int dpr_node_sha256(const SaltGemma4Text *model,
                           const int *tokens, int token_count,
                           uint8_t out[32]) {
    static const char sampler_config[] =
        "salt-greedy-v1;temperature=0;target-rng=none";
    SaltDprNodeMaterial material;
    uint8_t chain[32], next[32];
    if (!model || token_count < 0 || (token_count && !tokens) || !out ||
        salt_gemma4_text_position(model) != token_count)
        return -1;
    memset(&material, 0, sizeof material);
    if (salt_sha256_hex_parse(
            SALT_GEMMA4_KV_COMPAT_SHA256,
            material.state_compatibility_sha256) != 0 ||
        salt_sha256_bytes(sampler_config, sizeof sampler_config - 1u,
                          material.sampler_config_sha256) != 0)
        return -1;
    if (salt_dpr_transition_chain_init(
            material.state_compatibility_sha256,
            material.sampler_config_sha256,
            SALT_DPR_SAMPLER_GREEDY_V1, 0, chain) != 0)
        return -1;
    for (int token = 0; token < token_count; token++) {
        if (salt_dpr_transition_chain_advance(
                chain, (int32_t)tokens[token],
                (uint64_t)token + 1u, next) != 0)
            return -1;
        memcpy(chain, next, 32);
    }
    memcpy(material.state_sha256, chain, 32);
    material.sampler_abi = SALT_DPR_SAMPLER_GREEDY_V1;
    material.position = (uint64_t)salt_gemma4_text_position(model);
    return salt_dpr_node_sha256(&material, out);
}

static void print_sha256(FILE *stream, const char *name,
                         const uint8_t value[32]) {
    static const char digits[] = "0123456789abcdef";
    fprintf(stream, "%s=", name);
    for (int i = 0; i < 32; i++) {
        fputc(digits[value[i] >> 4], stream);
        fputc(digits[value[i] & 15u], stream);
    }
    fputc('\n', stream);
}

static int write_proof_artifact(
        const char *path, const void *buffer, size_t bytes) {
    const unsigned char *cursor = (const unsigned char *)buffer;
    int fd;
    if (!path || !*path || !buffer || bytes < 1) return -1;
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) return -1;
    while (bytes) {
        ssize_t written = write(fd, cursor, bytes);
        if (written <= 0) {
            close(fd);
            return -1;
        }
        cursor += (size_t)written;
        bytes -= (size_t)written;
    }
    if (fsync(fd) != 0 || close(fd) != 0) return -1;
    return 0;
}

static int parse_positive(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || !end || *end || parsed < 1 || parsed > 1000000) return -1;
    *value = (int)parsed;
    return 0;
}

static int qa_target_backend_engaged(
        const SaltTextVerifyBackendStats *backend,
        int target_frontier, int target_gpu_program,
        int target_cuda_program, int dpr_proof,
        int target_route_f, int target_route_q) {
    uint32_t rows, selected_jobs;
    if (!backend || dpr_proof < 1 || target_route_f < 1 ||
        target_route_q < 1)
        return 0;
    rows = (uint32_t)dpr_proof *
        (uint32_t)(target_frontier ? target_route_f : 1);
    /* Frontier queue, helper, cancellation, and ready-depth counts describe
     * the selected physical traversal.  They remain in the receipt below but
     * must not govern token/state/KV correctness or force runtime scheduling
     * to reproduce a historical Q-runway shape. */
    if (!target_gpu_program) {
        if (backend->cpu_matrix_pool_phases == 0u ||
            backend->cpu_qkv_waves == 0u ||
            backend->cpu_dense_gate_up_waves == 0u ||
            backend->cpu_expert_gate_up_waves == 0u ||
            backend->cpu_expert_down_waves == 0u ||
            backend->backend_template_count != 0u ||
            backend->backend_template_reuses != 0u ||
            backend->backend_dynamic_patches != 0u ||
            backend->backend_selected_jobs != 0u ||
            backend->backend_graph_launches != 0u ||
            backend->backend_physical_kernel_nodes != 0u ||
            backend->backend_host_kernel_launch_calls != 0u)
            return 0;
        if (backend->cpu_graph_sessions != 0u &&
            (backend->cpu_graph_nodes == 0u ||
             backend->cpu_graph_barriers == 0u))
            return 0;
        return 1;
    }
    selected_jobs = rows * 8u;
    if (backend->projection_dispatches == 0u ||
        backend->expert_gate_up_dispatches == 0u ||
        backend->expert_down_dispatches == 0u ||
        backend->backend_template_count == 0u ||
        backend->backend_template_reuses == 0u)
        return 0;
    if (target_cuda_program && !salt_gpu_pageable_mmap_active() &&
        (backend->backend_graph_count == 0u ||
         backend->backend_graph_launches == 0u ||
         backend->backend_graph_parameter_patches == 0u ||
         backend->backend_selected_job_capacity < selected_jobs ||
         backend->backend_selected_jobs != selected_jobs ||
         backend->backend_physical_kernel_nodes == 0u ||
         backend->backend_host_kernel_launch_calls == 0u))
        return 0;
    if (target_cuda_program && salt_gpu_pageable_mmap_active() &&
        ((backend->backend_graph_count == 0u
             ? (backend->backend_graph_launches != 0u ||
                backend->backend_graph_parameter_patches != 0u)
             : backend->backend_graph_launches == 0u) ||
         backend->backend_dynamic_patches == 0u ||
         backend->backend_selected_job_capacity < selected_jobs ||
         backend->backend_selected_jobs != selected_jobs ||
         backend->backend_physical_kernel_nodes == 0u ||
         backend->backend_host_kernel_launch_calls == 0u))
        return 0;
    if (!target_cuda_program &&
        ((backend->backend_physical_kernel_nodes == 0u &&
          backend->backend_graph_launches == 0u) ||
         (backend->backend_host_kernel_launch_calls == 0u &&
          backend->backend_graph_parameter_patches == 0u)))
        return 0;
    return 1;
}

static void qa_print_target_cpu_waterfall(
        const SaltTextVerifyBackendStats *backend) {
    if (!backend || backend->cpu_target_interpret_ns == 0) return;
    fprintf(stdout,
        "GEMMA4_TARGET_CPU_WATERFALL clear_ns=%llu setup_ns=%llu "
        "interpret_ns=%llu",
        (unsigned long long)backend->cpu_target_clear_ns,
        (unsigned long long)backend->cpu_target_setup_ns,
        (unsigned long long)backend->cpu_target_interpret_ns);
    for (uint32_t kind = 1;
         kind < SALT_TEXT_EXECUTION_CELL_KIND_COUNT; kind++)
        fprintf(stdout, " kind_%u_ns=%llu kind_%u_calls=%u",
            kind, (unsigned long long)backend->cpu_target_cell_ns[kind],
            kind, backend->cpu_target_cell_calls[kind]);
    fprintf(stdout, "\n");
}

static int parse_token_csv(const char *text, int expected, int *tokens) {
    const char *cursor = text;
    if (!text || !*text || expected < 1 || !tokens) return -1;
    for (int i = 0; i < expected; i++) {
        unsigned long long value = 0;
        const char *start = cursor;
        if (*cursor < '0' || *cursor > '9' ||
            (*cursor == '0' && cursor[1] >= '0' && cursor[1] <= '9'))
            return -1;
        while (*cursor >= '0' && *cursor <= '9') {
            unsigned digit = (unsigned)(*cursor - '0');
            if (value > (unsigned long long)(INT_MAX - (int)digit) / 10u)
                return -1;
            value = value * 10u + digit;
            cursor++;
        }
        if (cursor == start || value > INT_MAX) return -1;
        tokens[i] = (int)value;
        if (i + 1 < expected) {
            if (*cursor != ',') return -1;
            cursor++;
        }
    }
    return *cursor ? -1 : 0;
}

static int greedy_token(const float *logits, int count) {
    int best = -1;
    float best_value = -INFINITY;
    for (int i = 0; i < count; i++) {
        if (!isfinite(logits[i])) return -1;
        if (best < 0 || logits[i] > best_value) {
            best = i;
            best_value = logits[i];
        }
    }
    return best;
}

static int is_stop_token(int token) {
    return token == 1 || token == 106 || token == 50;
}

static void print_ids(FILE *stream, const char *label,
                      const int *tokens, int count) {
    fprintf(stream, "%s=", label);
    for (int i = 0; i < count; i++)
        fprintf(stream, "%s%d", i ? "," : "", tokens[i]);
    fputc('\n', stream);
}

static int gpu_stats_delta(const SaltGpuBatchStatsV2 *before,
                           const SaltGpuBatchStatsV2 *after,
                           SaltGpuBatchStatsV2 *delta) {
    if (!before || !after || !delta) return -1;
    memset(delta, 0, sizeof *delta);
#define GPU_STATS_DELTA(field) do { \
        if (after->field < before->field || after->field == UINT64_MAX) \
            return -1; \
        delta->field = after->field - before->field; \
    } while (0)
    GPU_STATS_DELTA(arena_batches);
    GPU_STATS_DELTA(trunk_batches);
    GPU_STATS_DELTA(expert_layer_batches);
    GPU_STATS_DELTA(mapped_only_misses);
    GPU_STATS_DELTA(direct_output_batches);
    GPU_STATS_DELTA(direct_output_jobs);
    GPU_STATS_DELTA(nvfp4_logical_packed_weight_bytes);
    GPU_STATS_DELTA(nvfp4_logical_weight_scale_bytes);
    GPU_STATS_DELTA(nvfp4_activation_input_bytes);
    GPU_STATS_DELTA(nvfp4_activation_qdq_bytes);
    GPU_STATS_DELTA(nvfp4_projection_output_bytes);
    GPU_STATS_DELTA(nvfp4_kernel_launches);
    GPU_STATS_DELTA(nvfp4_completion_fences);
    GPU_STATS_DELTA(nvfp4_h2d_ns);
    GPU_STATS_DELTA(nvfp4_qdq_ns);
    GPU_STATS_DELTA(nvfp4_projection_ns);
    GPU_STATS_DELTA(nvfp4_d2h_ns);
#undef GPU_STATS_DELTA
    return 0;
}

static int gpu_stats_snapshot(SaltGpuBatchStatsV2 *stats) {
    return salt_gpu_batch_stats_get_v2(stats, sizeof *stats);
}

static void print_nvfp4_telemetry(FILE *stream, const char *phase,
                                  const SaltGpuBatchStatsV2 *stats) {
    fprintf(stream,
        "GEMMA4_NVFP4_TELEMETRY phase=%s "
        "logical_packed_weight_bytes=%llu logical_weight_scale_bytes=%llu "
        "activation_input_bytes=%llu activation_qdq_bytes=%llu "
        "projection_output_bytes=%llu kernel_launches=%llu "
        "completion_fences=%llu h2d_ns=%llu qdq_ns=%llu "
        "projection_ns=%llu d2h_ns=%llu\n",
        phase,
        (unsigned long long)stats->nvfp4_logical_packed_weight_bytes,
        (unsigned long long)stats->nvfp4_logical_weight_scale_bytes,
        (unsigned long long)stats->nvfp4_activation_input_bytes,
        (unsigned long long)stats->nvfp4_activation_qdq_bytes,
        (unsigned long long)stats->nvfp4_projection_output_bytes,
        (unsigned long long)stats->nvfp4_kernel_launches,
        (unsigned long long)stats->nvfp4_completion_fences,
        (unsigned long long)stats->nvfp4_h2d_ns,
        (unsigned long long)stats->nvfp4_qdq_ns,
        (unsigned long long)stats->nvfp4_projection_ns,
        (unsigned long long)stats->nvfp4_d2h_ns);
}

typedef struct {
    int output_count;
    int stop_token;
    double first_token_ready;
    double end;
} DecodeResult;

static int generate(SaltGemma4Text *model, float *logits, float *proposal_logits,
                    int *output_ids, int generation_cap,
                    const char *phase, DecodeResult *result) {
    int next;
    if (!model || !logits || !proposal_logits || !output_ids || generation_cap < 1 ||
        !phase || !result)
        return -1;
    memset(result, 0, sizeof *result);
    result->stop_token = -1;
    result->first_token_ready = -1.0;
    next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
    if (next < 0) {
        fprintf(stderr, "gemma4 qa runner: %s non-finite first logits\n", phase);
        return -1;
    }
    while (result->output_count < generation_cap) {
        SaltGemma4TargetGenerateResult selection;
        /* Production cheap NFQ selection; X=1 admits no TARGET model work. */
        if (salt_gemma4_text_target_generate(model, logits, 1,
                proposal_logits, &selection) != 1 ||
            selection.candidate_count == 0u || selection.target_model_rows != 0u)
            return -1;
        if (result->first_token_ready < 0.0)
            result->first_token_ready = now_seconds();
        output_ids[result->output_count++] = next;
        fprintf(stderr, "[%s-decode] token=%d/%d id=%d stop=%d\n",
                phase, result->output_count, generation_cap,
                next, is_stop_token(next));
        if (is_stop_token(next)) {
            result->stop_token = next;
            break;
        }
        if (result->output_count >= generation_cap) break;
        double start = now_seconds();
        if (salt_gemma4_text_step(model, next, logits) != 0) {
            fprintf(stderr,
                    "gemma4 qa runner: %s decode failed after output %d\n",
                    phase, result->output_count);
            return -1;
        }
        fprintf(stderr, "[%s-decode-step] consumed=%d wall_s=%.6f\n",
                phase, next, now_seconds() - start);
        next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
        if (next < 0) {
            fprintf(stderr, "gemma4 qa runner: %s non-finite logits\n", phase);
            return -1;
        }
    }
    result->end = now_seconds();
    return result->first_token_ready >= 0.0 ? 0 : -1;
}

static int write_route_observations(
        const char *path, const SaltGemma4RouteObserver *observer) {
    int fd;
    FILE *stream;
    if (!path || !*path || !observer || !observer->items || observer->overflow)
        return -1;
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0 || !(stream = fdopen(fd, "wb"))) {
        if (fd >= 0) close(fd);
        return -1;
    }
    if (fprintf(stream, "SALT_GEMMA4_ROUTE_OBSERVATIONS_V1 count=%zu\n",
                observer->count) < 0)
        goto fail;
    for (size_t i = 0; i < observer->count; i++)
        if (fprintf(stream, "%u,%llu,%u,%u\n",
                    observer->items[i].kind,
                    (unsigned long long)observer->items[i].position,
                    observer->items[i].layer,
                    observer->items[i].expert) < 0)
            goto fail;
    if (fflush(stream) != 0 || fsync(fileno(stream)) != 0 ||
        fclose(stream) != 0)
        return -1;
    return 0;
fail:
    fclose(stream);
    return -1;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --ctx 200 --gen 50 --prompt TEXT [--workers 8] "
        "[--kv-load FILE] [--kv-save FILE] "
        "[--load-only] [--compare-kv-warm] [--package-proof] "
        "[--ondemand-rope-proof] [--dpr-proof N] [--target-frontier] "
        "[--decode-candidates CSV] "
        "[--route-observe FILE] "
        "[--dpr-reference-save FILE] [--dpr-reference-ids CSV] "
        "[--dpr-prefill-reference-save FILE --dpr-prefill-tokens N]\n",
        program);
}

static int qa_target_exact_miss(
        void *opaque, uint64_t epoch_generation,
        uint64_t parent_generation, uint32_t parent_position,
        SaltTextVerifyResult *result) {
    (void)opaque;
    (void)epoch_generation;
    (void)parent_generation;
    (void)parent_position;
    if (!result) return -1;
    memset(result, 0, sizeof *result);
    return 0;
}

int main(int argc, char **argv) {
    int context = 0, generation_cap = 0, workers = 8;
    int weighted_value_simd = 1;
    int head_parallel = 1;
    int q4_row_pair = 1;
    int ffn_detail = 0;
    int load_only = 0, compare_kv_warm = 0, package_proof = 0;
    int ondemand_rope_proof = 0, dpr_proof = 0, progressive_seed = 0;
    int optimized_decode = 0;
    int target_frontier = 0;
    int target_n_parallel = 0;
    int target_route_f = QA_TARGET_ROUTE_COUNT;
    int target_route_q = 4;
    int target_rejection_cases = -1;
    int expert_matrix_waves = 0;
    int prefill_expert_matrix_flow = 0;
    int target_matrix_flow = 0;
    int prefill_gpu_experts = 0;
    int target_gpu_experts = 0;
    int target_gpu_program = 0;
    int target_cuda_program = 0;
    int target_rocm_program = 0;
    int dpr_prefill_tokens = 0;
    int exit_code = 1;
    const char *prompt = NULL, *kv_load_path = NULL, *kv_save_path = NULL;
    const char *dpr_reference_save_path = NULL;
    const char *dpr_reference_ids_text = NULL;
    const char *decode_candidates_text = NULL;
    const char *dpr_prefill_reference_save_path = NULL;
    const char *route_observe_path = NULL;
    SaltGemma4Text *model = NULL;
    int *prompt_ids = NULL, *output_ids = NULL, *warm_output_ids = NULL;
    float *logits = NULL, *prefill_logits = NULL, *proposal_logits = NULL;
    int *dpr_draft_ids = NULL, *dpr_serial_ids = NULL;
    int *target_seed_ids = NULL;
    void *dpr_final_snapshot = NULL, *dpr_serial_snapshot = NULL;
    char *response = NULL, *warm_response = NULL;
    void *kv_snapshot = NULL;
    void *final_kv_snapshot = NULL;
    size_t kv_snapshot_bytes = 0, kv_header_bytes = 0;
    size_t final_kv_snapshot_bytes = 0, final_kv_header_bytes = 0;
    size_t kv_bytes_per_token = 0, kv_poisoned_bytes = 0;
    char error[256];
    DecodeResult cold = {0}, warm = {0};
    SaltGemma4MemoryStats memory_stats = {0};
    SaltGemma4MemoryStats prefill_memory_stats = {0};
    uint8_t final_state_sha256[32] = {0};
    uint8_t final_facts_sha256[32] = {0};
    uint8_t final_kv_sha256[32] = {0};
    uint8_t final_build_identity_sha256[32] = {0};
    SaltGemma4RouteObserver route_observer = {0};
    SaltGemma4RouteObservation *route_items = NULL;
    int route_observer_active = 0;
    SaltTextTargetPolicy target_policy = {0};
    uint32_t target_route_width = 0u;
    G4ProcessMemory process_memory = {0};
    SaltGpuBatchStatsV2 gpu_request_start = {0}, gpu_after_prefill = {0};
    SaltGpuBatchStatsV2 gpu_after_cold = {0}, gpu_after_warm = {0};
    SaltGpuBatchStatsV2 gpu_request_end = {0}, gpu_request_delta = {0};
    SaltGpuBatchStatsV2 gpu_prefill_delta = {0}, gpu_cold_delta = {0};
    SaltGpuBatchStatsV2 gpu_warm_delta = {0}, gpu_commit_delta = {0};
    double load_start, load_end, request_start, prefill_done;
    double state_capture_start = 0.0, state_capture_end = 0.0;
    double kv_capture_start = 0.0, kv_capture_end = 0.0;
    double logit_capture_start = 0.0, logit_capture_end = 0.0;
    double poison_start = 0.0, poison_end = 0.0;
    double cold_decode_start, warm_start = 0.0;
    double kv_restore_start = 0.0, kv_restore_end = 0.0;
    double logit_restore_start = 0.0, logit_restore_end = 0.0;
    int prompt_count = -1, response_capacity;
    int response_length = -1, warm_response_length = -1;
    int kv_loaded_tokens = 0, kv_saved_tokens = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            if (parse_positive(argv[++i], &context) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--gen") && i + 1 < argc) {
            if (parse_positive(argv[++i], &generation_cap) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--workers") && i + 1 < argc) {
            if (parse_positive(argv[++i], &workers) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--kv-load") && i + 1 < argc) {
            kv_load_path = argv[++i];
        } else if (!strcmp(argv[i], "--kv-save") && i + 1 < argc) {
            kv_save_path = argv[++i];
        } else if (!strcmp(argv[i], "--load-only")) {
            load_only = 1;
        } else if (!strcmp(argv[i], "--compare-kv-warm")) {
            compare_kv_warm = 1;
        } else if (!strcmp(argv[i], "--package-proof")) {
            package_proof = 1;
        } else if (!strcmp(argv[i], "--ondemand-rope-proof")) {
            ondemand_rope_proof = 1;
        } else if (!strcmp(argv[i], "--dpr-proof") && i + 1 < argc) {
            if (parse_positive(argv[++i], &dpr_proof) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--decode-candidates") &&
                   i + 1 < argc) {
            decode_candidates_text = argv[++i];
        } else if (!strcmp(argv[i], "--target-frontier")) {
            target_frontier = 1;
        } else if (!strcmp(argv[i], "--dpr-reference-save") &&
                   i + 1 < argc) {
            dpr_reference_save_path = argv[++i];
        } else if (!strcmp(argv[i], "--dpr-reference-ids") &&
                   i + 1 < argc) {
            dpr_reference_ids_text = argv[++i];
        } else if (!strcmp(argv[i], "--dpr-prefill-reference-save") &&
                   i + 1 < argc) {
            dpr_prefill_reference_save_path = argv[++i];
        } else if (!strcmp(argv[i], "--dpr-prefill-tokens") &&
                   i + 1 < argc) {
            if (parse_positive(argv[++i], &dpr_prefill_tokens) != 0) {
                usage(argv[0]);
                return 2;
            }
        } else if (!strcmp(argv[i], "--route-observe") && i + 1 < argc) {
            route_observe_path = argv[++i];
            if (!*route_observe_path) return 2;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    optimized_decode = decode_candidates_text != NULL;
    {
        const char *setting = getenv("SALT_TARGET_TILE_SEED");
        if (setting && parse_positive(setting, &progressive_seed) != 0) {
            fputs("gemma4 qa runner: SALT_TARGET_TILE_SEED must be positive\n",
                  stderr);
            return 2;
        }
    }


    {
        const char *setting = getenv("SALT_TARGET_REJECTION_CASES");
        if (setting) {
            char *end = NULL;
            long parsed;
            errno = 0;
            parsed = strtol(setting, &end, 10);
            if (errno || !end || *end || parsed < 0 ||
                parsed > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES) {
                fputs("gemma4 qa runner: SALT_TARGET_REJECTION_CASES must be "
                      "in [0, 128]\n", stderr);
                return 2;
            }
            target_rejection_cases = (int)parsed;
        }
    }
    {
        const char *setting = getenv("SALT_EXPERT_MATRIX_WAVES");
        if (setting && strcmp(setting, "0") != 0 && strcmp(setting, "1") != 0) {
            fputs("gemma4 qa runner: SALT_EXPERT_MATRIX_WAVES must be 0 or 1\n",
                  stderr);
            return 2;
        }
        expert_matrix_waves = setting && strcmp(setting, "1") == 0;
    }
    {
        const char *prefill = getenv("SALT_PREFILL_EXPERT_MATRIX_FLOW");
        const char *prefill_gpu = getenv("SALT_PREFILL_GPU_EXPERTS");
        const char *gpu_experts = getenv("SALT_TARGET_GPU_EXPERTS");
        if ((prefill && strcmp(prefill, "0") != 0 && strcmp(prefill, "1") != 0) ||
            (prefill_gpu && strcmp(prefill_gpu, "0") != 0 &&
             strcmp(prefill_gpu, "1") != 0) ||
            (gpu_experts && strcmp(gpu_experts, "0") != 0 &&
             strcmp(gpu_experts, "1") != 0)) {
            fputs("gemma4 qa runner: matrix-flow flags must be 0 or 1\n",
                  stderr);
            return 2;
        }
        prefill_expert_matrix_flow = prefill && strcmp(prefill, "1") == 0;
        prefill_gpu_experts = prefill_gpu && strcmp(prefill_gpu, "1") == 0;
        target_gpu_experts = gpu_experts && strcmp(gpu_experts, "1") == 0;
    }
    if (target_rejection_cases < 0)
        target_rejection_cases = optimized_decode ? 0 : dpr_proof;
    if (context < 1 || generation_cap < 1 || generation_cap > context ||
        workers < 1 || workers > 8 || !prompt || !*prompt ||
        (!optimized_decode && dpr_proof > generation_cap - 1) ||
        (optimized_decode &&
         (!*decode_candidates_text || dpr_proof != 0 ||
          !target_frontier || target_rejection_cases != 0)) ||
        (!optimized_decode && target_frontier &&
         (uint64_t)(uint32_t)dpr_proof * (uint32_t)target_route_f >
             SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES) ||
        (!optimized_decode && target_frontier &&
         (dpr_proof == 0 || progressive_seed != 0 ||
          dpr_proof > SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES)) ||
        (!optimized_decode && target_rejection_cases > dpr_proof) ||
        (dpr_reference_save_path != NULL && dpr_proof == 0) ||
        (progressive_seed != 0 &&
         (dpr_proof == 0 || progressive_seed > dpr_proof)) ||
        (dpr_reference_ids_text != NULL &&
         dpr_reference_save_path == NULL) ||
        ((dpr_prefill_reference_save_path != NULL) !=
         (dpr_prefill_tokens != 0)) ||
        (load_only && (compare_kv_warm || kv_load_path || kv_save_path)) ||
        (compare_kv_warm && (kv_load_path || kv_save_path)) ||
        (dpr_proof && (load_only || compare_kv_warm ||
                       kv_load_path || kv_save_path ||
                       dpr_prefill_reference_save_path)) ||
        (dpr_prefill_reference_save_path &&
         (load_only || compare_kv_warm || kv_load_path || kv_save_path)) ||
        (optimized_decode &&
         (load_only || compare_kv_warm || kv_load_path ||
          dpr_reference_save_path || dpr_reference_ids_text ||
          dpr_prefill_reference_save_path || route_observe_path)) ||
        (route_observe_path &&
         (load_only || compare_kv_warm || dpr_proof || kv_load_path ||
          kv_save_path || dpr_prefill_reference_save_path))) {
        usage(argv[0]);
        return 2;
    }

    {
        const char *setting = getenv("SALT_ATTN_WEIGHTED_VALUE_SIMD");
        if (setting) {
            if (!strcmp(setting, "0")) weighted_value_simd = 0;
            else if (!strcmp(setting, "1")) weighted_value_simd = 1;
            else {
                fputs("gemma4 qa runner: SALT_ATTN_WEIGHTED_VALUE_SIMD "
                      "must be 0 or 1\n", stderr);
                return 2;
            }
        }
        if (salt_attn_set_weighted_value_simd(weighted_value_simd) != 0)
            return 2;
        setting = getenv("SALT_ATTN_HEAD_PARALLEL");
        if (setting) {
            if (!strcmp(setting, "0")) head_parallel = 0;
            else if (!strcmp(setting, "1")) head_parallel = 1;
            else {
                fputs("gemma4 qa runner: SALT_ATTN_HEAD_PARALLEL "
                      "must be 0 or 1\n", stderr);
                return 2;
            }
        }
        if (salt_attn_set_head_parallel(head_parallel) != 0)
            return 2;
        setting = getenv("SALT_Q4_ROW_PAIR");
        if (setting) {
            if (!strcmp(setting, "0")) q4_row_pair = 0;
            else if (!strcmp(setting, "1")) q4_row_pair = 1;
            else {
                fputs("gemma4 qa runner: SALT_Q4_ROW_PAIR must be 0 or 1\n",
                      stderr);
                return 2;
            }
        }
        if (salt_simd_set_q4_row_pair(q4_row_pair) != 0)
            return 2;
        setting = getenv("SALT_FFN_DETAIL");
        if (setting) {
            if (!strcmp(setting, "0")) ffn_detail = 0;
            else if (!strcmp(setting, "1")) ffn_detail = 1;
            else {
                fputs("gemma4 qa runner: SALT_FFN_DETAIL must be 0 or 1\n",
                      stderr);
                return 2;
            }
        }
    }
    load_start = now_seconds();
    model = salt_gemma4_text_load(
        stdin, context, workers, error, sizeof error);
    load_end = now_seconds();
    if (!model) {
        fprintf(stderr, "gemma4 qa runner: load failed: %s\n", error);
        return 1;
    }
    if (salt_gemma4_text_target_policy_get(model, &target_policy) != 0) {
        fputs("gemma4 qa runner: engine TARGET policy unavailable\n", stderr);
        goto done;
    }
    target_route_f = (int)target_policy.route_count;
    target_route_q = (int)target_policy.queue_length;
    target_route_width =
        target_policy.target_rows * target_policy.route_count;
    if (target_route_width > target_policy.candidate_count)
        goto done;
    target_n_parallel = target_policy.n_parallel != 0u;
    target_matrix_flow = target_policy.matrix_flow != 0u;
    target_gpu_program =
        target_policy.execution_class == SALT_TEXT_EXECUTION_GPU_ONLY;
    target_cuda_program = target_gpu_program && salt_gpu_cuda_present();
    target_rocm_program = target_gpu_program && salt_gpu_rocm_present();
    if (optimized_decode) dpr_proof = (int)target_policy.target_rows;
    if ((target_frontier &&
         dpr_proof != (int)target_policy.target_rows) ||
        (target_frontier &&
         target_route_width >
             SALT_GEMMA4_TEXT_TARGET_MAX_CANDIDATES) ||
        dpr_proof > generation_cap - 1 ||
        target_rejection_cases > dpr_proof) {
        fputs("gemma4 qa runner: workload disagrees with engine TARGET policy\n",
              stderr);
        goto done;
    }
    if (!load_only &&
        (prefill_gpu_experts || target_gpu_experts || target_gpu_program) &&
        dpr_proof != 0 && dpr_reference_ids_text != NULL) {
        fputs("gemma4 qa runner: GPU expert engagement requires a normal "
              "decode transaction\n", stderr);
        goto done;
    }
    if (target_gpu_program && target_gpu_experts) {
        fputs("gemma4 qa runner: complete and compatibility GPU target modes "
              "are mutually exclusive\n", stderr);
        goto done;
    }
    if (package_proof && salt_gemma4_text_validate_proof(model) != 0) {
        fputs("gemma4 qa runner: one-shot proof validation failed\n", stderr);
        goto done;
    }
    if (ondemand_rope_proof &&
        salt_gemma4_text_use_ondemand_rope_for_proof(model, 1) != 0) {
        fputs("gemma4 qa runner: on-demand RoPE proof selection failed\n", stderr);
        goto done;
    }
    if (g4_print_process_memory(stderr, "after_load", NULL) != 0) {
        fputs("gemma4 qa runner: load memory sample failed\n", stderr);
        goto done;
    }
    if (kv_load_path && g4_kv_file_load(
            model, kv_load_path, context, &kv_loaded_tokens,
            error, sizeof error) != 0) {
        fprintf(stderr, "gemma4 qa runner: KV cache load failed: %s\n", error);
        goto done;
    }

    prompt_ids = (int *)malloc((size_t)context * sizeof *prompt_ids);
    output_ids = (int *)malloc((size_t)generation_cap * sizeof *output_ids);
    logits = (float *)malloc(
        (size_t)salt_gemma4_text_vocab_size(model) * sizeof *logits);
    proposal_logits = (float *)malloc(
        (size_t)salt_gemma4_text_vocab_size(model) * sizeof *proposal_logits);
    response_capacity = generation_cap * 64 + 1024;
    response = (char *)malloc((size_t)response_capacity);
    if (compare_kv_warm) {
        warm_output_ids = (int *)malloc(
            (size_t)generation_cap * sizeof *warm_output_ids);
        warm_response = (char *)malloc((size_t)response_capacity);
    }
    if (compare_kv_warm || dpr_proof)
        prefill_logits = (float *)malloc(
            (size_t)salt_gemma4_text_vocab_size(model) * sizeof *prefill_logits);
    if (dpr_proof) {
        dpr_draft_ids = (int *)malloc((size_t)dpr_proof * sizeof *dpr_draft_ids);
        if (!optimized_decode)
            dpr_serial_ids = (int *)malloc(
                (size_t)(dpr_proof + 1) * sizeof *dpr_serial_ids);
        if (target_frontier)
            target_seed_ids = (int *)malloc(
                (size_t)target_policy.candidate_count *
                sizeof *target_seed_ids);
    }
    if (route_observe_path) {
        size_t capacity;
        if ((size_t)context > SIZE_MAX / (30u * 8u)) goto done;
        capacity = (size_t)context * 30u * 8u;
        route_items = (SaltGemma4RouteObservation *)calloc(
            capacity, sizeof *route_items);
        if (!route_items) goto done;
        route_observer.items = route_items;
        route_observer.capacity = capacity;
    }
    if (!prompt_ids || !output_ids || !logits || !proposal_logits || !response) {
        fputs("gemma4 qa runner: allocation failed\n", stderr);
        goto done;
    }
    if (compare_kv_warm && (!warm_output_ids || !warm_response)) {
        fputs("gemma4 qa runner: KV-warm allocation failed\n", stderr);
        goto done;
    }
    if ((compare_kv_warm || dpr_proof) && !prefill_logits) {
        fputs("gemma4 qa runner: prefill-logit allocation failed\n", stderr);
        goto done;
    }
    if (dpr_proof && (!dpr_draft_ids ||
                      (!optimized_decode && !dpr_serial_ids) ||
                      (target_frontier && !target_seed_ids))) {
        fputs("gemma4 qa runner: decode allocation failed\n", stderr);
        goto done;
    }
    if (optimized_decode) {
        int vocab = salt_gemma4_text_vocab_size(model);
        if (parse_token_csv(
                decode_candidates_text, (int)target_route_width,
                target_seed_ids) != 0)
            goto done;
        for (uint32_t i = 0; i < target_route_width; i++)
            if (target_seed_ids[i] < 0 || target_seed_ids[i] >= vocab ||
                is_stop_token(target_seed_ids[i]))
                goto done;
        memcpy(dpr_draft_ids, target_seed_ids,
               (size_t)dpr_proof * sizeof *dpr_draft_ids);
    }
    if (dpr_reference_ids_text) {
        int vocab = salt_gemma4_text_vocab_size(model);
        if (parse_token_csv(
                dpr_reference_ids_text, dpr_proof, dpr_draft_ids) != 0)
            goto done;
        for (int i = 0; i < dpr_proof; i++)
            if (dpr_draft_ids[i] < 0 || dpr_draft_ids[i] >= vocab ||
                is_stop_token(dpr_draft_ids[i]))
                goto done;
    }

    prompt_count = salt_gemma4_text_encode(model, prompt, prompt_ids, context);
    if (prompt_count < 1 || kv_loaded_tokens > context - prompt_count ||
        kv_loaded_tokens + prompt_count > context - generation_cap) {
        fprintf(stderr,
            "gemma4 qa runner: prompt/generation exceed context: cached=%d prompt=%d ctx=%d gen=%d\n",
            kv_loaded_tokens, prompt_count, context, generation_cap);
        goto done;
    }
    if (load_only) {
        fprintf(stdout,
            "GEMMA4_QA_LOADER_OK ctx=%d gen_cap=%d workers=%d "
            "prompt_tokens=%d runtime_ready=true\n",
            context, generation_cap, workers, prompt_count);
        print_ids(stdout, "prompt_ids", prompt_ids, prompt_count);
        exit_code = 0;
        goto done;
    }
    if (route_observe_path) {
        if (salt_gemma4_text_route_observer_begin(
                model, &route_observer) != 0)
            goto done;
        route_observer_active = 1;
    }
    if (dpr_prefill_reference_save_path) {
        int saved_position = 0;
        uint8_t source_node[32], result_node[32];
        if (dpr_prefill_tokens >= prompt_count ||
            dpr_node_sha256(model, NULL, 0, source_node) != 0 ||
            salt_gemma4_text_prefill(
                model, prompt_ids, dpr_prefill_tokens, logits) != 0 ||
            dpr_node_sha256(
                model, prompt_ids, dpr_prefill_tokens, result_node) != 0 ||
            g4_kv_file_save(
                model, dpr_prefill_reference_save_path, &saved_position,
                error, sizeof error) != 0 ||
            saved_position != dpr_prefill_tokens) {
            fprintf(stderr,
                "gemma4 qa runner: DPR prefill reference failed: %s\n", error);
            goto done;
        }
        fprintf(stdout,
            "GEMMA4_QA_DPR_PREFILL_REFERENCE_V1 prompt_tokens=%d "
            "cached_tokens=%d position=%d runtime_ready=true\n",
            prompt_count, dpr_prefill_tokens, saved_position);
        print_sha256(stdout, "dpr_source_node_sha256", source_node);
        print_sha256(stdout, "dpr_result_node_sha256", result_node);
        print_ids(stdout, "prompt_ids", prompt_ids, prompt_count);
        print_ids(stdout, "dpr_prefill_ids", prompt_ids, dpr_prefill_tokens);
        exit_code = 0;
        goto done;
    }

    fprintf(stderr,
        "GEMMA4_QA_EXECUTION_START ctx=%d gen_cap=%d workers=%d "
        "prompt_tokens=%d mode=greedy kv_compare=%d rope_factors=%d "
        "weighted_value_simd=%d head_parallel=%d q4_row_pair=%d "
        "ffn_detail=%d "
        "runtime_ready=true "
        "load_s=%.6f\n",
        context, generation_cap, workers, prompt_count, compare_kv_warm,
        !ondemand_rope_proof, weighted_value_simd, head_parallel, q4_row_pair,
        ffn_detail,
        load_end - load_start);
    print_ids(stderr, "prompt_ids", prompt_ids, prompt_count);
    fflush(stderr);

    if (gpu_stats_snapshot(&gpu_request_start) != 0) {
        fputs("gemma4 qa runner: request GPU telemetry snapshot failed\n", stderr);
        goto done;
    }
    request_start = now_seconds();
    {
        const char *chunked = getenv("SALT_PREFILL_CHUNK");
        if (chunked && *chunked != '0' && prompt_count > 1) {
            if (salt_gemma4_text_prefill(
                    model, prompt_ids, prompt_count, logits) != 0) {
                fputs("gemma4 qa runner: layer-major prefill failed\n", stderr);
                goto done;
            }
            fprintf(stderr, "[prefill] B=%d ms_per_token=%.3f\n",
                    prompt_count,
                    (now_seconds() - request_start) * 1000.0 / prompt_count);
        } else for (int i = 0; i < prompt_count; i++) {
            double start = now_seconds();
            if (salt_gemma4_text_step(model, prompt_ids[i],
                    i == prompt_count - 1 ? logits : NULL) != 0) {
                fprintf(stderr,
                    "gemma4 qa runner: prefill failed at token %d/%d\n",
                    i + 1, prompt_count);
                goto done;
            }
            fprintf(stderr, "[prefill] token=%d/%d id=%d wall_s=%.6f\n",
                    i + 1, prompt_count, prompt_ids[i], now_seconds() - start);
            fflush(stderr);
        }
    }

    prefill_done = now_seconds();
    if (gpu_stats_snapshot(&gpu_after_prefill) != 0) {
        fputs("gemma4 qa runner: prefill GPU telemetry snapshot failed\n", stderr);
        goto done;
    }
    if (salt_gemma4_text_memory_stats(model, &prefill_memory_stats) != 0) {
        fputs("gemma4 qa runner: prefill model telemetry snapshot failed\n",
              stderr);
        goto done;
    }
    if (g4_print_process_memory(stderr, "after_prefill", NULL) != 0) {
        fputs("gemma4 qa runner: prefill memory sample failed\n", stderr);
        goto done;
    }
    if (compare_kv_warm || (dpr_proof && !optimized_decode)) {
        state_capture_start = now_seconds();
        kv_snapshot_bytes = salt_gemma4_text_kv_snapshot_size(model);
        kv_header_bytes = salt_gemma4_text_kv_snapshot_header_size();
        if (kv_snapshot_bytes == 0 || !(kv_snapshot = malloc(kv_snapshot_bytes))) {
            fputs("gemma4 qa runner: KV snapshot allocation failed\n", stderr);
            goto done;
        }
        if (kv_header_bytes == 0 || kv_snapshot_bytes <= kv_header_bytes ||
            (kv_snapshot_bytes - kv_header_bytes) % (size_t)prompt_count != 0) {
            fputs("gemma4 qa runner: invalid KV snapshot geometry\n", stderr);
            goto done;
        }
        kv_bytes_per_token =
            (kv_snapshot_bytes - kv_header_bytes) / (size_t)prompt_count;
        kv_capture_start = now_seconds();
        if (salt_gemma4_text_kv_snapshot(
                model, kv_snapshot, kv_snapshot_bytes) != 0 ||
            salt_gemma4_text_position(model) != prompt_count) {
            fputs("gemma4 qa runner: KV snapshot failed\n", stderr);
            goto done;
        }
        kv_capture_end = now_seconds();
        logit_capture_start = now_seconds();
        memcpy(prefill_logits, logits,
               (size_t)salt_gemma4_text_vocab_size(model) * sizeof *logits);
        logit_capture_end = now_seconds();
        state_capture_end = logit_capture_end;
    } else if (optimized_decode) {
        state_capture_start = now_seconds();
        memcpy(prefill_logits, logits,
               (size_t)salt_gemma4_text_vocab_size(model) * sizeof *logits);
        state_capture_end = now_seconds();
        logit_capture_start = state_capture_start;
        logit_capture_end = state_capture_end;
    }

    if (dpr_proof) {
        int vocab = salt_gemma4_text_vocab_size(model);
        size_t dpr_snapshot_bytes;
        FILE *stream_checkpoint = NULL;
        void *stream_snapshot = NULL;
        double draft_start, draft_end, recover_start, recover_end;
        double verify_start, verify_end, pending_start, pending_end;
        double serial_start, serial_verify_end = 0.0, serial_end;
        unsigned char dpr_state_sha[32], serial_state_sha[32];
        SaltTextVerifyResult transition;
        SaltTextProgressiveResult progressive;
        SaltTextVerifyBackendStats accepted_backend;
        SaltTextTokenEpochController target_epoch;
        SaltGemma4MemoryStats serial_memory_before, serial_memory_after;
        SaltGemma4MemoryStats flow_memory_before, flow_memory_after;
        uint32_t progressive_repair_tiles = 0;
        uint32_t progressive_repair_submitted_tiles = 0;
        int bonus, target_rc;
        if (!optimized_decode) {
            stream_checkpoint = tmpfile();
            stream_snapshot = malloc(kv_snapshot_bytes);
            if (!stream_checkpoint || !stream_snapshot ||
                salt_gemma4_text_state_sha256(model, dpr_state_sha) != 0 ||
                salt_gemma4_text_kv_export_stream(model, stream_checkpoint) != 0 ||
                salt_gemma4_text_kv_poison_prefix(
                    model, prompt_count, &kv_poisoned_bytes) != 0 ||
                fseek(stream_checkpoint, 0, SEEK_SET) != 0 ||
                salt_gemma4_text_kv_import_stream(model, stream_checkpoint) != 0 ||
                salt_gemma4_text_state_sha256(model, serial_state_sha) != 0 ||
                memcmp(dpr_state_sha, serial_state_sha, 32u) != 0 ||
                salt_gemma4_text_kv_snapshot(
                    model, stream_snapshot,
                    kv_snapshot_bytes) != 0 ||
                memcmp(kv_snapshot, stream_snapshot, kv_snapshot_bytes) != 0) {
                if (stream_checkpoint) fclose(stream_checkpoint);
                free(stream_snapshot);
                fputs("gemma4 qa runner: DPR streaming seat proof failed\n", stderr);
                goto done;
            }
            fclose(stream_checkpoint);
            free(stream_snapshot);
            draft_start = now_seconds();
            for (int i = 0; i < dpr_proof; i++) {
                int token = dpr_reference_ids_text
                    ? dpr_draft_ids[i] : greedy_token(logits, vocab);
                if (token < 0 || is_stop_token(token) ||
                    salt_gemma4_text_step(model, token, logits) != 0) {
                    fputs("gemma4 qa runner: DPR cached-edge draft failed\n", stderr);
                    goto done;
                }
                dpr_draft_ids[i] = token;
            }
            draft_end = now_seconds();
            bonus = greedy_token(logits, vocab);
            if (bonus < 0) goto done;
            if (dpr_reference_save_path) {
                int saved_position = 0;
                if (g4_kv_file_save(
                        model, dpr_reference_save_path, &saved_position,
                        error, sizeof error) != 0 ||
                    saved_position != prompt_count + dpr_proof) {
                    fputs("gemma4 qa runner: DPR reference save failed\n", stderr);
                    goto done;
                }
            }
            if (dpr_reference_ids_text) {
                fputs("dpr_reference_ids=", stdout);
                for (int i = 0; i < dpr_proof; i++)
                    printf("%s%d", i ? "," : "", dpr_draft_ids[i]);
                fputc('\n', stdout);
                exit_code = 0;
                goto done;
            }
            recover_start = now_seconds();
            if (salt_gemma4_text_kv_restore(
                    model, kv_snapshot, kv_snapshot_bytes) != 0)
                goto done;
            memcpy(logits, prefill_logits, (size_t)vocab * sizeof *logits);
            recover_end = now_seconds();
        } else {
            draft_start = draft_end = now_seconds();
            recover_start = recover_end = draft_end;
            bonus = -1;
        }
        memset(&flow_memory_before, 0, sizeof flow_memory_before);
        memset(&flow_memory_after, 0, sizeof flow_memory_after);
        if (salt_gemma4_text_memory_stats(model, &flow_memory_before) != 0 ||
            (prefill_expert_matrix_flow &&
             flow_memory_before.prefill_expert_matrix_flow_sessions == 0u) ||
            (prefill_gpu_experts &&
             (prefill_memory_stats.gpu_expert_gate_up_commands == 0u ||
              prefill_memory_stats.gpu_expert_down_commands == 0u)))
            goto done;
        verify_start = now_seconds();
        memset(&transition, 0, sizeof transition);
        memset(&progressive, 0, sizeof progressive);
        memset(&accepted_backend, 0, sizeof accepted_backend);
        memset(&target_epoch, 0, sizeof target_epoch);
        if (target_frontier) {
            SaltTextTokenEpochResult epoch_result;
            if (!optimized_decode)
                for (int route = 0; route < target_route_f; route++)
                    for (int tile = 0; tile < dpr_proof; tile++)
                        target_seed_ids[route * dpr_proof + tile] =
                            (dpr_draft_ids[tile] + route) % vocab;
            if (!optimized_decode)
                for (uint32_t base = target_route_width; base-- > 0u;) {
                    int token = target_seed_ids[base];
                    for (uint32_t queue = 0u;
                         queue < target_policy.queue_length; queue++)
                        target_seed_ids[
                            base * target_policy.queue_length + queue] = token;
                }
            memset(&epoch_result, 0, sizeof epoch_result);
            target_rc = salt_text_token_epoch_init(&target_epoch) != 0
                ? -1 : salt_gemma4_text_target_epoch(
                model,
                &target_epoch,
                (const int32_t *)(const void *)target_seed_ids,
                (int)target_policy.candidate_count, prefill_logits,
                qa_target_exact_miss, NULL, &epoch_result);
            transition = epoch_result.target;
        } else if (progressive_seed) {
            target_rc = salt_gemma4_text_target_progressive(
                model, dpr_draft_ids, dpr_proof, prefill_logits,
                progressive_seed, dpr_proof, &progressive);
            transition = progressive.target;
        } else {
            target_rc = salt_gemma4_text_target_block(
                model, dpr_draft_ids, dpr_proof, prefill_logits, &transition);
        }
        if (target_frontier)
            fprintf(stderr,
                "GEMMA4_TARGET_FRONTIER_OBS f=%d n=%d rc=%d status=%d "
                "accepted=%u committed=%u produced=%u pending=%d "
                "winner_index=%u winner_id=%u "
                "lookups=%llu exact_hits=%llu exact_misses=%llu "
                "cold_searches=%llu "
                "sessions=%u queued=%u executed=%u helpers=%u reassign=%u "
                "cancelled=%u peak_ready=%u submissions=%u fences=%u\n",
                target_route_f, dpr_proof, target_rc, (int)transition.status,
                transition.accepted_count, transition.committed_count,
                transition.produced_count, transition.pending_token_id,
                transition.winning_node_index, transition.winning_node_id,
                (unsigned long long)target_epoch.lookup_count,
                (unsigned long long)target_epoch.exact_hit_count,
                (unsigned long long)target_epoch.exact_miss_count,
                (unsigned long long)target_epoch.cold_search_count,
                transition.backend.production_frontier_sessions,
                transition.backend.production_frontier_tasks_queued,
                transition.backend.production_frontier_tasks_executed,
                transition.backend.production_frontier_helper_executions,
                transition.backend.production_frontier_worker_reassignments,
                transition.backend.production_frontier_queued_cancellations,
                transition.backend.production_frontier_peak_ready_depth,
                transition.backend.engine_submissions,
                transition.backend.completion_fences);

        if (optimized_decode) bonus = transition.pending_token_id;

        if (optimized_decode) {
            accepted_backend = transition.backend;
            verify_end = now_seconds();
            qa_print_target_cpu_waterfall(&accepted_backend);
        }

        if (target_rc != 0 ||
            transition.status != SALT_TEXT_VERIFY_COMMITTED ||
            transition.accepted_count != (uint32_t)dpr_proof ||
            transition.committed_count != (uint32_t)dpr_proof ||
            transition.produced_count != (uint32_t)dpr_proof + 1u ||
            transition.pending_token_id != bonus ||
            transition.result_position != (uint32_t)(prompt_count + dpr_proof) ||
            (target_frontier &&
             !((transition.winning_node_index == (uint32_t)dpr_proof - 1u &&
                transition.winning_node_id == (uint32_t)dpr_proof - 1u) ||
               (transition.winning_node_index == 0u &&
                transition.winning_node_id == 0u))) ||
            (target_frontier &&
             (target_epoch.lookup_count != 1u ||
              target_epoch.exact_hit_count != 0u ||
              target_epoch.exact_miss_count != 1u ||
              target_epoch.cold_search_count != 1u)) ||
            (target_frontier && target_n_parallel &&
             (transition.backend.area_matrix_parallel_dispatches == 0u ||
              transition.backend.area_candidate_output_tiles <=
                  transition.backend.area_output_row_tiles)) ||
            !qa_target_backend_engaged(
                &transition.backend, target_frontier, target_gpu_program,
                target_cuda_program, dpr_proof,
                target_route_f, target_route_q) ||
            transition.backend.canonical_clear_bytes == 0u ||
            transition.backend.engine_submissions !=
                (target_frontier ? 1u :
                 progressive_seed ? progressive.submitted_tile_count : 1u) ||
            transition.backend.completion_fences !=
                (target_frontier ? 1u :
                 progressive_seed ? progressive.submitted_tile_count : 1u) ||
            transition.backend.intermediate_host_publications != 0u) {
            fprintf(stderr,
                "GEMMA4_TARGET_BLOCK_FAIL rc=%d status=%d accepted=%u "
                "produced=%u committed=%u pending=%d position=%u generation=%llu "
                "submissions=%u fences=%u publications=%u projections=%u "
                "expert_gate_up=%u expert_down=%u templates=%u reuses=%u "
                "dynamic_patches=%u selected_capacity=%u selected_jobs=%u "
                "graphs=%u graph_launches=%u graph_patches=%u kernels=%u "
                "host_launches=%u canonical_clear_bytes=%llu\n",
                target_rc, (int)transition.status, transition.accepted_count,
                transition.produced_count, transition.committed_count,
                transition.pending_token_id, transition.result_position,
                (unsigned long long)transition.transition_generation,
                transition.backend.engine_submissions,
                transition.backend.completion_fences,
                transition.backend.intermediate_host_publications,
                transition.backend.projection_dispatches,
                transition.backend.expert_gate_up_dispatches,
                transition.backend.expert_down_dispatches,
                transition.backend.backend_template_count,
                transition.backend.backend_template_reuses,
                transition.backend.backend_dynamic_patches,
                transition.backend.backend_selected_job_capacity,
                transition.backend.backend_selected_jobs,
                transition.backend.backend_graph_count,
                transition.backend.backend_graph_launches,
                transition.backend.backend_graph_parameter_patches,
                transition.backend.backend_physical_kernel_nodes,
                transition.backend.backend_host_kernel_launch_calls,
                (unsigned long long)transition.backend.canonical_clear_bytes);
            goto done;
        }
        if (salt_gemma4_text_memory_stats(model, &flow_memory_after) != 0 ||
            (target_matrix_flow &&
             flow_memory_after.target_matrix_flow_sessions -
                 flow_memory_before.target_matrix_flow_sessions != 1u) ||
            (target_gpu_experts &&
             (flow_memory_after.target_gpu_expert_batches <=
                  flow_memory_before.target_gpu_expert_batches ||
              flow_memory_after.target_gpu_expert_jobs <=
                  flow_memory_before.target_gpu_expert_jobs)))
            goto done;
        accepted_backend = transition.backend;
        verify_end = now_seconds();
        if (optimized_decode) {
            uint32_t winning_route =
                transition.winning_node_index / target_policy.target_rows;
            if (winning_route >= target_policy.route_count)
                goto done;
            memcpy(output_ids,
                   target_seed_ids + winning_route * target_policy.target_rows,
                   (size_t)dpr_proof * sizeof *output_ids);
            cold.output_count = dpr_proof;
            cold.stop_token = -1;
            cold.first_token_ready = verify_end;
            cold.end = verify_end;
            cold_decode_start = verify_start;
            fprintf(stdout,
                "GEMMA4_OPTIMIZED_DECODE width=%d accepted=%u pending=%d "
                "source_position=%d result_position=%u decode_s=%.6f "
                "serial_control=0 rejection_cases=0 gpu_program=%d "
                "cuda_program=%d rocm_program=%d submissions=%u fences=%u "
                "publications=%u canonical_clear_bytes=%llu "
                "backend_kernel_nodes=%u backend_host_launches=%u "
                "runtime_ready=true\n",
                dpr_proof, transition.accepted_count,
                transition.pending_token_id, prompt_count,
                transition.result_position, verify_end - verify_start,
                target_gpu_program, target_cuda_program,
                target_rocm_program, accepted_backend.engine_submissions,
                accepted_backend.completion_fences,
                accepted_backend.intermediate_host_publications,
                (unsigned long long)accepted_backend.canonical_clear_bytes,
                accepted_backend.backend_physical_kernel_nodes,
                accepted_backend.backend_host_kernel_launch_calls);
            goto finalize_cold;
        }
        pending_start = verify_end;
        if (salt_gemma4_text_step(
                model, transition.pending_token_id, NULL) != 0)
            goto done;
        pending_end = now_seconds();
        if (g4_print_process_memory(stderr, "dpr_after_verify", NULL) != 0)
            goto done;
        dpr_snapshot_bytes = salt_gemma4_text_kv_snapshot_size(model);
        dpr_final_snapshot = malloc(dpr_snapshot_bytes);
        if (!dpr_final_snapshot ||
            salt_gemma4_text_kv_snapshot(
                model, dpr_final_snapshot, dpr_snapshot_bytes) != 0 ||
            salt_gemma4_text_state_sha256(model, dpr_state_sha) != 0)
            goto done;

        if (salt_gemma4_text_kv_restore(
                model, kv_snapshot, kv_snapshot_bytes) != 0)
            goto done;
        memcpy(logits, prefill_logits, (size_t)vocab * sizeof *logits);
        memset(&serial_memory_before, 0, sizeof serial_memory_before);
        memset(&serial_memory_after, 0, sizeof serial_memory_after);
        if (salt_gemma4_text_memory_stats(
                model, &serial_memory_before) != 0)
            goto done;
        serial_start = now_seconds();
        for (int i = 0; i < dpr_proof + 1; i++) {
            int token = greedy_token(logits, vocab);
            if (token < 0 || salt_gemma4_text_step(
                    model, token, i == dpr_proof ? NULL : logits) != 0)
                goto done;
            dpr_serial_ids[i] = token;
            if (i == dpr_proof - 1) serial_verify_end = now_seconds();
        }
        serial_end = now_seconds();
        if (serial_verify_end <= serial_start || serial_verify_end > serial_end)
            goto done;
        if (g4_print_process_memory(stderr, "dpr_after_serial", NULL) != 0)
            goto done;
        dpr_serial_snapshot = malloc(dpr_snapshot_bytes);
        if (!dpr_serial_snapshot ||
            salt_gemma4_text_kv_snapshot(
                model, dpr_serial_snapshot, dpr_snapshot_bytes) != 0 ||
            salt_gemma4_text_state_sha256(model, serial_state_sha) != 0)
            goto done;
        if (memcmp(dpr_state_sha, serial_state_sha, 32u) != 0 ||
            memcmp(dpr_final_snapshot, dpr_serial_snapshot,
                   dpr_snapshot_bytes) != 0) {
            const unsigned char *target_bytes =
                (const unsigned char *)dpr_final_snapshot;
            const unsigned char *serial_bytes =
                (const unsigned char *)dpr_serial_snapshot;
            size_t differing = 0, first = SIZE_MAX;
            for (size_t byte = 0; byte < dpr_snapshot_bytes; byte++)
                if (target_bytes[byte] != serial_bytes[byte]) {
                    if (first == SIZE_MAX) first = byte;
                    differing++;
                }
            fprintf(stderr,
                "GEMMA4_TARGET_IDENTITY_FAIL state_equal=%d "
                "first_differing_byte=%zu differing_bytes=%zu "
                "snapshot_bytes=%zu\n",
                memcmp(dpr_state_sha, serial_state_sha, 32u) == 0,
                first, differing, dpr_snapshot_bytes);
            goto done;
        }
        if (salt_gemma4_text_memory_stats(model, &serial_memory_after) != 0 ||
            (expert_matrix_waves &&
             (serial_memory_after.expert_matrix_gate_up_waves <=
                  serial_memory_before.expert_matrix_gate_up_waves ||
              serial_memory_after.expert_matrix_down_waves <=
                  serial_memory_before.expert_matrix_down_waves ||
              serial_memory_after.expert_matrix_projection_jobs <=
                  serial_memory_before.expert_matrix_projection_jobs)))
            goto done;
        for (int i = 0; i < dpr_proof; i++)
            if (dpr_draft_ids[i] != dpr_serial_ids[i]) goto done;
        if (bonus != dpr_serial_ids[dpr_proof]) goto done;
        for (int reject = 0; reject < target_rejection_cases; reject++) {
            int correction;
            SaltTextProgressiveResult rejected_progressive;
            memcpy(output_ids, dpr_draft_ids,
                   (size_t)dpr_proof * sizeof *output_ids);
            output_ids[reject] = (output_ids[reject] + 1) % vocab;
            if (output_ids[reject] == dpr_serial_ids[reject]) goto done;
            if (salt_gemma4_text_kv_restore(
                    model, kv_snapshot, kv_snapshot_bytes) != 0)
                goto done;
            memcpy(logits, prefill_logits, (size_t)vocab * sizeof *logits);
            memset(&transition, 0, sizeof transition);
            memset(&rejected_progressive, 0, sizeof rejected_progressive);
            if (target_frontier) {
                SaltTextTokenEpochController rejected_epoch;
                SaltTextTokenEpochResult epoch_result;
                int *failed = target_seed_ids +
                    reject;
                int original = *failed;
                *failed =
                    (original + target_route_f + 1) % vocab;
                memset(&rejected_epoch, 0, sizeof rejected_epoch);
                memset(&epoch_result, 0, sizeof epoch_result);
                target_rc = salt_text_token_epoch_init(&rejected_epoch) != 0
                    ? -1 : salt_gemma4_text_target_epoch(
                    model, &rejected_epoch,
                    (const int32_t *)(const void *)target_seed_ids,
                    (int)target_policy.candidate_count,
                    prefill_logits, qa_target_exact_miss, NULL,
                    &epoch_result);
                transition = epoch_result.target;
                *failed = original;
            } else {
                target_rc = progressive_seed
                    ? salt_gemma4_text_target_progressive(
                        model, output_ids, dpr_proof, prefill_logits,
                        progressive_seed, dpr_proof, &rejected_progressive)
                    : salt_gemma4_text_target_block(
                        model, output_ids, dpr_proof,
                        prefill_logits, &transition);
            }
            if (target_rc != 0)
                goto done;
            if (progressive_seed) transition = rejected_progressive.target;
            if (
                transition.status != SALT_TEXT_VERIFY_COMMITTED ||
                transition.accepted_count != (uint32_t)reject ||
                transition.committed_count != (uint32_t)reject ||
                transition.produced_count != (uint32_t)reject + 1u ||
                transition.result_position != (uint32_t)(prompt_count + reject) ||
                transition.backend.engine_submissions !=
                    (target_frontier ? (reject == 0 ? 0u : 1u) :
                     progressive_seed
                        ? rejected_progressive.submitted_tile_count
                        : (reject == 0 ? 0u : 1u)) ||
                transition.backend.completion_fences !=
                    (target_frontier ? (reject == 0 ? 0u : 1u) :
                     progressive_seed
                        ? rejected_progressive.submitted_tile_count
                        : (reject == 0 ? 0u : 1u)) ||
                transition.backend.intermediate_host_publications != 0u)
                goto done;
            correction = transition.pending_token_id;
            if (correction != dpr_serial_ids[reject]) goto done;
            if (progressive_seed) {
                int repair_count = dpr_proof - reject;
                const float *repair_parent = transition.pending_logits;
                SaltTextProgressiveResult repair_progressive;
                if (!repair_parent ||
                    transition.pending_logits_count != (uint32_t)vocab)
                    goto done;
                memset(&repair_progressive, 0, sizeof repair_progressive);
                if (salt_gemma4_text_target_progressive(
                        model, dpr_serial_ids + reject, repair_count,
                        repair_parent, progressive_seed, dpr_proof,
                        &repair_progressive) != 0 ||
                    repair_progressive.target.accepted_count !=
                        (uint32_t)repair_count ||
                    repair_progressive.target.committed_count !=
                        (uint32_t)repair_count ||
                    repair_progressive.target.pending_token_id != bonus ||
                    repair_progressive.target.result_position !=
                        (uint32_t)(prompt_count + dpr_proof) ||
                    salt_gemma4_text_step(
                        model, repair_progressive.target.pending_token_id,
                        NULL) != 0)
                    goto done;
                progressive_repair_tiles += repair_progressive.tile_count;
                progressive_repair_submitted_tiles +=
                    repair_progressive.submitted_tile_count;
            } else {
                if (salt_gemma4_text_step(model, correction, logits) != 0)
                    goto done;
                for (int i = reject + 1; i < dpr_proof + 1; i++) {
                    int token = greedy_token(logits, vocab);
                    if (token != dpr_serial_ids[i] ||
                        salt_gemma4_text_step(
                            model, token, i == dpr_proof ? NULL : logits) != 0)
                        goto done;
                }
            }
            if (salt_gemma4_text_state_sha256(model, dpr_state_sha) != 0 ||
                memcmp(dpr_state_sha, serial_state_sha, 32u) != 0 ||
                salt_gemma4_text_kv_snapshot(
                    model, dpr_final_snapshot, dpr_snapshot_bytes) != 0 ||
                memcmp(dpr_final_snapshot, dpr_serial_snapshot,
                       dpr_snapshot_bytes) != 0)
                goto done;
        }
        fprintf(stdout,
            "GEMMA4_DPR_PROOF block=%d accepted=%d bonus=%d "
            "target_frontier=%d sequence_tiles=%d "
            "frontier_width=%d queue_length=%d "
            "frontier_nodes=%d target_n_parallel=%d "
            "expert_matrix_waves=%d expert_matrix_gate_up_waves=%llu "
            "expert_matrix_down_waves=%llu expert_matrix_jobs=%llu "
            "prefill_matrix_flow=%d prefill_matrix_flow_sessions=%llu "
            "prefill_gpu_experts=%d prefill_gpu_gate_up_commands=%llu "
            "prefill_gpu_down_commands=%llu "
            "target_matrix_flow=%d target_matrix_flow_sessions=%llu "
            "target_gpu_experts=%d target_gpu_expert_batches=%llu "
            "target_gpu_expert_jobs=%llu target_gpu_program=%d "
            "target_cuda_program=%d target_rocm_program=%d "
            "cache_lookups=%llu exact_hits=%llu exact_misses=%llu "
            "cold_searches=%llu duplicate_admissions=%llu "
            "load_s=%.6f prefill_s=%.6f snapshot_s=%.6f "
            "draft_s=%.6f recover_s=%.6f verify_s=%.6f pending_s=%.6f "
            "serial_verify_s=%.6f serial_pending_s=%.6f serial_s=%.6f "
            "rejection_cases=%d state_bytes=%zu kv_arena_bytes=%zu "
            "engine_submissions=%u completion_fences=%u "
            "intermediate_publications=%u dependency_barriers=%u "
            "projection_dispatches=%u expert_gate_up_dispatches=%u "
            "expert_down_dispatches=%u area_m1_dispatches=%u "
            "area_mk_dispatches=%u area_mn_dispatches=%u "
            "area_peak_active_workers=%u area_output_row_tiles=%llu "
            "area_matrix_parallel_dispatches=%u "
            "area_candidate_output_tiles=%llu "
            "production_frontier_sessions=%u "
            "production_frontier_tasks_queued=%u "
            "production_frontier_tasks_executed=%u "
            "production_frontier_helper_executions=%u "
            "production_frontier_worker_reassignments=%u "
            "production_frontier_queued_cancellations=%u "
            "production_frontier_cancel_requested_completions=%u "
            "production_frontier_stale_rejections=%u "
            "production_frontier_peak_ready_depth=%u "
            "cpu_matrix_pool_phases=%u cpu_qkv_waves=%u "
            "cpu_dense_gate_up_waves=%u cpu_expert_gate_up_waves=%u "
            "cpu_expert_down_waves=%u cpu_graph_sessions=%u "
            "cpu_graph_nodes=%u cpu_graph_barriers=%u "
            "backend_template_count=%u backend_template_reuses=%u "
            "backend_dynamic_patches=%u "
            "backend_selected_job_capacity=%u backend_selected_jobs=%u "
            "backend_graph_count=%u backend_graph_launches=%u "
            "backend_graph_parameter_patches=%u "
            "backend_physical_kernel_nodes=%u "
            "backend_host_kernel_launch_calls=%u "
            "initial_transfer_bytes=%llu "
            "final_logits_transfer_bytes=%llu final_kv_publish_bytes=%llu "
            "tentative_scrub_bytes=%llu canonical_clear_bytes=%llu "
            "progressive_seed=%d "
            "progressive_tiles=%u progressive_submitted_tiles=%u "
            "progressive_largest_tile=%u progressive_repair_tiles=%u "
            "progressive_repair_submitted_tiles=%u differing_bytes=0 "
            "runtime_ready=true\n",
            dpr_proof, dpr_proof, bonus,
            target_frontier,
            target_frontier ? dpr_proof : 1,
            target_frontier ? target_route_f : 1,
            target_frontier ? target_route_q : 1,
            target_frontier ? target_route_f * dpr_proof : dpr_proof,
            target_n_parallel,
            expert_matrix_waves,
            (unsigned long long)(
                serial_memory_after.expert_matrix_gate_up_waves -
                serial_memory_before.expert_matrix_gate_up_waves),
            (unsigned long long)(
                serial_memory_after.expert_matrix_down_waves -
                serial_memory_before.expert_matrix_down_waves),
            (unsigned long long)(
                serial_memory_after.expert_matrix_projection_jobs -
                serial_memory_before.expert_matrix_projection_jobs),
            prefill_expert_matrix_flow,
            (unsigned long long)
                flow_memory_before.prefill_expert_matrix_flow_sessions,
            prefill_gpu_experts,
            (unsigned long long)
                prefill_memory_stats.gpu_expert_gate_up_commands,
            (unsigned long long)
                prefill_memory_stats.gpu_expert_down_commands,
            target_matrix_flow,
            (unsigned long long)(
                flow_memory_after.target_matrix_flow_sessions -
                flow_memory_before.target_matrix_flow_sessions),
            target_gpu_experts,
            (unsigned long long)(
                flow_memory_after.target_gpu_expert_batches -
                flow_memory_before.target_gpu_expert_batches),
            (unsigned long long)(
                flow_memory_after.target_gpu_expert_jobs -
                flow_memory_before.target_gpu_expert_jobs),
            target_gpu_program,
            target_cuda_program,
            target_rocm_program,
            (unsigned long long)target_epoch.lookup_count,
            (unsigned long long)target_epoch.exact_hit_count,
            (unsigned long long)target_epoch.exact_miss_count,
            (unsigned long long)target_epoch.cold_search_count,
            (unsigned long long)target_epoch.duplicate_admission_count,
            load_end - load_start, prefill_done - request_start,
            state_capture_end - state_capture_start,
            draft_end - draft_start, recover_end - recover_start,
            verify_end - verify_start, pending_end - pending_start,
            serial_verify_end - serial_start,
            serial_end - serial_verify_end, serial_end - serial_start,
            target_rejection_cases, dpr_snapshot_bytes,
            salt_gemma4_text_kv_capacity_bytes(model),
            accepted_backend.engine_submissions,
            accepted_backend.completion_fences,
            accepted_backend.intermediate_host_publications,
            accepted_backend.internal_dependency_barriers,
            accepted_backend.projection_dispatches,
            accepted_backend.expert_gate_up_dispatches,
            accepted_backend.expert_down_dispatches,
            accepted_backend.area_m1_dispatches,
            accepted_backend.area_mk_dispatches,
            accepted_backend.area_mn_dispatches,
            accepted_backend.area_peak_active_workers,
            (unsigned long long)accepted_backend.area_output_row_tiles,
            accepted_backend.area_matrix_parallel_dispatches,
            (unsigned long long)accepted_backend.area_candidate_output_tiles,
            accepted_backend.production_frontier_sessions,
            accepted_backend.production_frontier_tasks_queued,
            accepted_backend.production_frontier_tasks_executed,
            accepted_backend.production_frontier_helper_executions,
            accepted_backend.production_frontier_worker_reassignments,
            accepted_backend.production_frontier_queued_cancellations,
            accepted_backend.production_frontier_cancel_requested_completions,
            accepted_backend.production_frontier_stale_rejections,
            accepted_backend.production_frontier_peak_ready_depth,
            accepted_backend.cpu_matrix_pool_phases,
            accepted_backend.cpu_qkv_waves,
            accepted_backend.cpu_dense_gate_up_waves,
            accepted_backend.cpu_expert_gate_up_waves,
            accepted_backend.cpu_expert_down_waves,
            accepted_backend.cpu_graph_sessions,
            accepted_backend.cpu_graph_nodes,
            accepted_backend.cpu_graph_barriers,
            accepted_backend.backend_template_count,
            accepted_backend.backend_template_reuses,
            accepted_backend.backend_dynamic_patches,
            accepted_backend.backend_selected_job_capacity,
            accepted_backend.backend_selected_jobs,
            accepted_backend.backend_graph_count,
            accepted_backend.backend_graph_launches,
            accepted_backend.backend_graph_parameter_patches,
            accepted_backend.backend_physical_kernel_nodes,
            accepted_backend.backend_host_kernel_launch_calls,
            (unsigned long long)accepted_backend.initial_transfer_bytes,
            (unsigned long long)accepted_backend.final_logits_transfer_bytes,
            (unsigned long long)accepted_backend.final_kv_publish_bytes,
            (unsigned long long)accepted_backend.tentative_scrub_bytes,
            (unsigned long long)accepted_backend.canonical_clear_bytes,
            progressive_seed, progressive.tile_count,
            progressive.submitted_tile_count, progressive.largest_tile,
            progressive_repair_tiles, progressive_repair_submitted_tiles);
        print_ids(stdout, "dpr_draft_ids", dpr_draft_ids, dpr_proof);
        print_ids(stdout, "dpr_serial_ids", dpr_serial_ids, dpr_proof + 1);
        qa_print_target_cpu_waterfall(&accepted_backend);
        exit_code = 0;
        goto done;
    }

    cold_decode_start = now_seconds();
    if (generate(model, logits, proposal_logits, output_ids, generation_cap,
                 "cold", &cold) != 0)
        goto done;
    if (route_observer_active) {
        if (salt_gemma4_text_route_observer_end(model) != 0)
            goto done;
        route_observer_active = 0;
        if (write_route_observations(
                route_observe_path, &route_observer) != 0) {
            fputs("gemma4 qa runner: route observation publication failed\n",
                  stderr);
            goto done;
        }
    }
finalize_cold:
    if (gpu_stats_snapshot(&gpu_after_cold) != 0) {
        fputs("gemma4 qa runner: cold GPU telemetry snapshot failed\n", stderr);
        goto done;
    }
    if (g4_print_process_memory(stderr, "after_cold_decode", NULL) != 0) {
        fputs("gemma4 qa runner: cold memory sample failed\n", stderr);
        goto done;
    }
    if (package_proof) {
        const char *final_kv_save_path = getenv("SALT_QA_FINAL_KV_SAVE");
        final_kv_snapshot_bytes = salt_gemma4_text_kv_snapshot_size(model);
        final_kv_header_bytes = salt_gemma4_text_kv_snapshot_header_size();
        final_kv_snapshot = final_kv_snapshot_bytes
            ? malloc(final_kv_snapshot_bytes) : NULL;
        if (!final_kv_snapshot || final_kv_header_bytes == 0 ||
            final_kv_header_bytes >= final_kv_snapshot_bytes ||
            salt_gemma4_text_kv_snapshot(model, final_kv_snapshot,
                final_kv_snapshot_bytes) != 0 ||
            salt_gemma4_text_state_sha256(model, final_state_sha256) != 0 ||
            salt_gemma4_text_facts_sha256(model, 0, final_facts_sha256) != 0 ||
            salt_gemma4_text_build_identity_sha256(
                model, final_build_identity_sha256) != 0 ||
            salt_sha256_bytes(
                (unsigned char *)final_kv_snapshot + final_kv_header_bytes,
                final_kv_snapshot_bytes - final_kv_header_bytes,
                final_kv_sha256) != 0) {
            fputs("gemma4 qa runner: final proof digest failed\n", stderr);
            goto done;
        }
        if (final_kv_save_path && write_proof_artifact(final_kv_save_path,
                final_kv_snapshot, final_kv_snapshot_bytes) != 0) {
            fputs("gemma4 qa runner: final KV proof save failed\n", stderr);
            goto done;
        }
        fprintf(stderr, "GEMMA4_QA_FINAL_PROOF position=%d kv_bytes=%zu\n",
                salt_gemma4_text_position(model), final_kv_snapshot_bytes);
        print_sha256(stderr, "final_state_sha256", final_state_sha256);
        print_sha256(stderr, "final_facts_sha256", final_facts_sha256);
        print_sha256(stderr, "final_kv_payload_sha256", final_kv_sha256);
        print_sha256(stderr, "final_build_identity_sha256",
                     final_build_identity_sha256);
        free(final_kv_snapshot);
        final_kv_snapshot = NULL;
    }
    int decode_count = cold.output_count;
    if (decode_count > 0 && is_stop_token(output_ids[decode_count - 1]))
        decode_count--;
    response_length = salt_gemma4_text_decode(
        model, output_ids, decode_count, response, response_capacity);
    if (response_length < 0 || response_length >= response_capacity ||
        memchr(response, '\0', (size_t)response_length) != NULL) {
        fputs("gemma4 qa runner: output decode failed\n", stderr);
        goto done;
    }

    if (compare_kv_warm) {
        poison_start = now_seconds();
        if (salt_gemma4_text_kv_poison_prefix(
                model, prompt_count, &kv_poisoned_bytes) != 0 ||
            kv_poisoned_bytes != kv_snapshot_bytes - kv_header_bytes) {
            fputs("gemma4 qa runner: KV poison proof failed\n", stderr);
            goto done;
        }
        poison_end = now_seconds();
        warm_start = now_seconds();
        kv_restore_start = warm_start;
        if (salt_gemma4_text_kv_restore(
                model, kv_snapshot, kv_snapshot_bytes) != 0 ||
            salt_gemma4_text_position(model) != prompt_count) {
            fputs("gemma4 qa runner: KV restore failed\n", stderr);
            goto done;
        }
        kv_restore_end = now_seconds();
        logit_restore_start = kv_restore_end;
        memcpy(logits, prefill_logits,
               (size_t)salt_gemma4_text_vocab_size(model) * sizeof *logits);
        logit_restore_end = now_seconds();
        if (generate(model, logits, proposal_logits, warm_output_ids, generation_cap,
                     "warm", &warm) != 0)
            goto done;
        if (gpu_stats_snapshot(&gpu_after_warm) != 0) {
            fputs("gemma4 qa runner: warm GPU telemetry snapshot failed\n", stderr);
            goto done;
        }
        if (g4_print_process_memory(stderr, "after_warm_decode", NULL) != 0) {
            fputs("gemma4 qa runner: warm memory sample failed\n", stderr);
            goto done;
        }
        int warm_decode_count = warm.output_count;
        if (warm_decode_count > 0 &&
            is_stop_token(warm_output_ids[warm_decode_count - 1]))
            warm_decode_count--;
        warm_response_length = salt_gemma4_text_decode(
            model, warm_output_ids, warm_decode_count,
            warm_response, response_capacity);
        if (cold.output_count != warm.output_count ||
            cold.stop_token != warm.stop_token ||
            memcmp(output_ids, warm_output_ids,
                   (size_t)cold.output_count * sizeof *output_ids) ||
            warm_response_length < 0 ||
            warm_response_length >= response_capacity ||
            memchr(warm_response, '\0', (size_t)warm_response_length) != NULL ||
            response_length != warm_response_length ||
            memcmp(response, warm_response, (size_t)response_length) != 0) {
            fputs("gemma4 qa runner: cold/warm output drift\n", stderr);
            goto done;
        }
    } else {
        gpu_after_warm = gpu_after_cold;
    }

    if (kv_save_path) {
        int expected_position;
        if (optimized_decode && cold.stop_token == -1 &&
            cold.output_count == dpr_proof && cold.output_count >= 1) {
            /* TARGET already committed every accepted candidate.  Its pending
             * bonus was not stepped and is not part of the exported KV. */
            expected_position = kv_loaded_tokens + prompt_count +
                                cold.output_count;
        } else if (cold.stop_token == 106 && cold.output_count >= 1) {
            if (salt_gemma4_text_step(model, cold.stop_token, NULL) != 0) {
                fputs("gemma4 qa runner: KV cache final turn step failed\n", stderr);
                goto done;
            }
            expected_position = kv_loaded_tokens + prompt_count + cold.output_count;
        } else if (cold.stop_token == -1 &&
                   cold.output_count == generation_cap &&
                   cold.output_count >= 1) {
            /* Length-cap completion retains the last emitted token as the
             * pending token. Export only the committed target transitions. */
            expected_position = kv_loaded_tokens + prompt_count +
                                cold.output_count - 1;
        } else {
            fputs("gemma4 qa runner: KV cache save requires a committed turn\n",
                  stderr);
            goto done;
        }
        if (salt_gemma4_text_position(model) != expected_position ||
            g4_kv_file_save(model, kv_save_path, &kv_saved_tokens,
                            error, sizeof error) != 0 ||
            kv_saved_tokens != expected_position) {
            fprintf(stderr, "gemma4 qa runner: KV cache save failed: %s\n", error);
            goto done;
        }
    }

    struct rusage usage_data;
    memset(&usage_data, 0, sizeof usage_data);
    if (getrusage(RUSAGE_SELF, &usage_data) != 0)
        usage_data.ru_maxrss = -1;
    if (salt_gemma4_text_memory_stats(model, &memory_stats) != 0 ||
        g4_print_process_memory(stderr, "final", &process_memory) != 0) {
        fputs("gemma4 qa runner: final memory accounting failed\n", stderr);
        goto done;
    }
    if (gpu_stats_snapshot(&gpu_request_end) != 0) {
        fputs("gemma4 qa runner: final GPU telemetry snapshot failed\n", stderr);
        goto done;
    }
    if (gpu_stats_delta(&gpu_request_start, &gpu_after_prefill,
                        &gpu_prefill_delta) != 0 ||
        gpu_stats_delta(&gpu_after_prefill, &gpu_after_cold,
                        &gpu_cold_delta) != 0 ||
        gpu_stats_delta(&gpu_after_cold, &gpu_after_warm,
                        &gpu_warm_delta) != 0 ||
        gpu_stats_delta(&gpu_after_warm, &gpu_request_end,
                        &gpu_commit_delta) != 0 ||
        gpu_stats_delta(&gpu_request_start, &gpu_request_end,
                        &gpu_request_delta) != 0) {
        fputs("gemma4 qa runner: invalid GPU telemetry delta\n", stderr);
        goto done;
    }

    print_ids(stderr, "output_ids", output_ids, cold.output_count);
    if (compare_kv_warm)
        print_ids(stderr, "warm_output_ids", warm_output_ids, warm.output_count);
    if (gpu_request_delta.nvfp4_kernel_launches > 0) {
        print_nvfp4_telemetry(stderr, "prefill", &gpu_prefill_delta);
        print_nvfp4_telemetry(stderr, "cold_decode", &gpu_cold_delta);
        print_nvfp4_telemetry(stderr, "warm_decode", &gpu_warm_delta);
        print_nvfp4_telemetry(stderr, "commit", &gpu_commit_delta);
        print_nvfp4_telemetry(stderr, "request", &gpu_request_delta);
    }
    double prefill_s = prefill_done - request_start;
    double cold_first_token_ready_s =
        prefill_s + cold.first_token_ready - cold_decode_start;
    double cold_total_s = prefill_s + cold.end - cold_decode_start;
    fprintf(stderr,
        "GEMMA4_QA_EXECUTION_DONE prompt_tokens=%d output_steps=%d stop_token=%d "
        "response_bytes=%d warm_response_bytes=%d "
        "kv_loaded_tokens=%d kv_saved_tokens=%d "
        "state_envelope=%s stop_commit_steps=%d package_proof=%d "
        "prefill_s=%.6f first_token_ready_s=%.6f total_s=%.6f "
        "cold_first_token_ready_s=%.6f cold_request_s=%.6f maxrss_raw=%ld "
        "maxrss_unit=%s kv_compare=%d kv_prefill_position=%d "
        "kv_snapshot_version=2 kv_snapshot_header_bytes=%zu "
        "kv_snapshot_bytes_per_token=%zu kv_snapshot_bytes=%zu "
        "kv_poisoned=%d kv_poisoned_bytes=%zu kv_restore_verified=%d "
        "state_capture_s=%.6f kv_capture_s=%.6f logit_capture_s=%.6f "
        "kv_poison_s=%.6f kv_restore_s=%.6f logit_restore_s=%.6f "
        "state_restore_s=%.6f warm_first_token_ready_s=%.6f "
        "warm_total_s=%.6f memory_detailed=%d proof_validation_calls=%llu "
        "startup_admitted=%d startup_limit_bytes=%llu "
        "startup_forecast_bytes=%llu startup_source_bytes=%llu "
        "startup_registered_bytes=%llu startup_pageable_bytes=%llu "
        "startup_expert_residency_bytes=%llu "
        "startup_host_runtime_bytes=%llu startup_host_kv_bytes=%llu "
        "startup_shared_arena_bytes=%llu "
        "startup_backend_pinned_bytes=%llu "
        "startup_backend_device_bytes=%llu startup_device_kv_bytes=%llu "
        "startup_cache_metadata_bytes=%llu startup_rope_bytes=%llu "
        "startup_descriptor_bytes=%llu "
        "current_rss_bytes=%llu "
        "resident_peak_bytes=%llu physical_footprint_bytes=%llu "
        "internal_bytes=%llu external_bytes=%llu reusable_bytes=%llu "
        "compressed_bytes=%llu minor_faults=%llu major_faults=%llu "
        "input_blocks=%llu output_blocks=%llu "
        "expert_cache_mode=bounded-mmap-zero-copy "
        "expert_cache_budget_bytes=%llu expert_cache_capacity_slots=%d "
        "expert_cache_capacity_bytes=%llu expert_cache_resident_slots=%d "
        "expert_cache_peak_slots=%d expert_preload_enabled=%d "
        "expert_preloaded_slots=%d expert_preload_layer_count=%d "
        "expert_preload_layer_mask=%llu expert_preload_protected_slots=%d "
        "expert_preload_fetch_jobs=%llu "
        "expert_cache_resident_logical_bytes=%llu "
        "expert_cache_peak_logical_bytes=%llu expert_cache_requests=%llu "
        "expert_cache_hits=%llu expert_cache_misses=%llu "
        "expert_cache_evictions=%llu expert_cache_release_calls=%llu "
        "expert_cache_release_failures=%llu "
        "expert_cache_released_page_bytes=%llu expert_cache_mapped_bytes=%llu "
        "expert_cache_peak_mapped_bytes=%llu expert_cache_unmap_calls=%llu "
        "expert_cache_unmap_failures=%llu expert_cache_unmapped_bytes=%llu "
        "expert_prepare_calls=%llu expert_prepare_failures=%llu "
        "expert_prepared_bytes=%llu "
        "expert_union_fetch_calls=%llu expert_union_fetch_experts=%llu "
        "expert_gate_up_epochs=%llu expert_down_epochs=%llu "
        "q4_pool_submissions=%llu q4_pool_fallbacks=%llu q4_pool_workers=%d "
        "q4_multi_pool_submissions=%llu q4_multi_pool_jobs=%llu "
        "qkv_multi_pool_submissions=%llu "
        "q4_multi_pool_fallbacks=%llu "
        "q8_pool_submissions=%llu q8_pool_fallbacks=%llu "
        "expert_pool_submissions=%llu expert_pool_fallbacks=%llu "
        "gpu_dense_submissions=%llu gpu_router_submissions=%llu "
        "gpu_expert_gate_up_submissions=%llu "
        "gpu_expert_down_submissions=%llu gpu_head_submissions=%llu "
        "gpu_expert_gate_up_commands=%llu gpu_expert_down_commands=%llu "
        "gpu_decode_expert_gate_up_commands=%llu "
        "gpu_decode_expert_down_commands=%llu "
        "gpu_decode_submissions=%llu gpu_failures=%llu "
        "gpu_shared_output_batches=%llu gpu_shared_output_jobs=%llu "
        "gpu_shared_arena_bytes=%llu "
        "gpu_weight_addressability=%u gpu_weight_described_resources=%u "
        "gpu_weight_active_resources=%u gpu_weight_active_windows=%u "
        "token_program_cells=%u token_caller_submissions=%u "
        "token_internal_barriers=%u token_completion_fences=%u "
        "token_intermediate_publications=%u token_final_publications=%u "
        "token_cpu_pool_sessions=%llu token_cpu_pool_phases=%llu "
        "gpu_weight_described_bytes=%llu gpu_weight_registered_bytes=%llu "
        "gpu_weight_pageable_bytes=%llu "
        "gpu_weight_active_window_bytes=%llu "
        "gpu_weight_peak_window_bytes=%llu gpu_weight_copied_bytes=%llu "
        "gpu_attention_batches=%llu gpu_attention_tasks=%llu "
        "gpu_attention_kv_bytes=%llu "
        "prefill_ffn_arena_bytes=%llu prefill_ffn_arena_calls=%llu "
        "expert_inflight_slots=%d "
        "expert_peak_inflight_slots=%d dense_retained_immutable=%d "
        "dense_payload_bytes=%llu "
        "runtime_ready=true\n",
        prompt_count, cold.output_count, cold.stop_token,
        response_length, compare_kv_warm ? warm_response_length : 0,
        kv_loaded_tokens, kv_saved_tokens,
        compare_kv_warm ? "kv-restore-proof" :
            (kv_save_path ? "kv-save" : "none"),
        kv_save_path ? 1 : 0, package_proof,
        prefill_s, cold_first_token_ready_s, cold_total_s,
        cold_first_token_ready_s, cold_total_s, usage_data.ru_maxrss,
#if defined(__APPLE__)
        "bytes",
#else
        "KiB",
#endif
        compare_kv_warm, compare_kv_warm ? prompt_count : 0,
        kv_header_bytes, kv_bytes_per_token, kv_snapshot_bytes,
        compare_kv_warm, kv_poisoned_bytes, compare_kv_warm,
        compare_kv_warm ? state_capture_end - state_capture_start : 0.0,
        compare_kv_warm ? kv_capture_end - kv_capture_start : 0.0,
        compare_kv_warm ? logit_capture_end - logit_capture_start : 0.0,
        compare_kv_warm ? poison_end - poison_start : 0.0,
        compare_kv_warm ? kv_restore_end - kv_restore_start : 0.0,
        compare_kv_warm ? logit_restore_end - logit_restore_start : 0.0,
        compare_kv_warm ? logit_restore_end - warm_start : 0.0,
        compare_kv_warm ? warm.first_token_ready - warm_start : 0.0,
        compare_kv_warm ? warm.end - warm_start : 0.0,
        process_memory.detailed,
        (unsigned long long)memory_stats.proof_validation_calls,
        memory_stats.startup_admitted,
        (unsigned long long)memory_stats.startup_limit_bytes,
        (unsigned long long)memory_stats.startup_forecast_bytes,
        (unsigned long long)memory_stats.startup_source_bytes,
        (unsigned long long)memory_stats.startup_registered_bytes,
        (unsigned long long)memory_stats.startup_pageable_bytes,
        (unsigned long long)memory_stats.startup_expert_residency_bytes,
        (unsigned long long)memory_stats.startup_host_runtime_bytes,
        (unsigned long long)memory_stats.startup_host_kv_bytes,
        (unsigned long long)memory_stats.startup_shared_arena_bytes,
        (unsigned long long)memory_stats.startup_backend_pinned_bytes,
        (unsigned long long)memory_stats.startup_backend_device_bytes,
        (unsigned long long)memory_stats.startup_device_kv_bytes,
        (unsigned long long)memory_stats.startup_cache_metadata_bytes,
        (unsigned long long)memory_stats.startup_rope_bytes,
        (unsigned long long)memory_stats.startup_descriptor_bytes,
        (unsigned long long)process_memory.current_rss_bytes,
        (unsigned long long)process_memory.resident_peak_bytes,
        (unsigned long long)process_memory.physical_footprint_bytes,
        (unsigned long long)process_memory.internal_bytes,
        (unsigned long long)process_memory.external_bytes,
        (unsigned long long)process_memory.reusable_bytes,
        (unsigned long long)process_memory.compressed_bytes,
        (unsigned long long)process_memory.minor_faults,
        (unsigned long long)process_memory.major_faults,
        (unsigned long long)process_memory.input_blocks,
        (unsigned long long)process_memory.output_blocks,
        (unsigned long long)memory_stats.expert_budget_bytes,
        memory_stats.expert_capacity_slots,
        (unsigned long long)memory_stats.expert_capacity_bytes,
        memory_stats.expert_resident_slots,
        memory_stats.expert_peak_slots,
        memory_stats.expert_preload_enabled,
        memory_stats.expert_preloaded_slots,
        memory_stats.expert_preload_layer_count,
        (unsigned long long)memory_stats.expert_preload_layer_mask,
        memory_stats.expert_preload_protected_slots,
        (unsigned long long)memory_stats.expert_preload_fetch_jobs,
        (unsigned long long)memory_stats.expert_resident_logical_bytes,
        (unsigned long long)memory_stats.expert_peak_logical_bytes,
        (unsigned long long)memory_stats.expert_requests,
        (unsigned long long)memory_stats.expert_hits,
        (unsigned long long)memory_stats.expert_misses,
        (unsigned long long)memory_stats.expert_evictions,
        (unsigned long long)memory_stats.expert_release_calls,
        (unsigned long long)memory_stats.expert_release_failures,
        (unsigned long long)memory_stats.expert_released_page_bytes,
        (unsigned long long)memory_stats.expert_mapped_bytes,
        (unsigned long long)memory_stats.expert_peak_mapped_bytes,
        (unsigned long long)memory_stats.expert_unmap_calls,
        (unsigned long long)memory_stats.expert_unmap_failures,
        (unsigned long long)memory_stats.expert_unmapped_bytes,
        (unsigned long long)memory_stats.expert_prepare_calls,
        (unsigned long long)memory_stats.expert_prepare_failures,
        (unsigned long long)memory_stats.expert_prepared_bytes,
        (unsigned long long)memory_stats.expert_union_fetch_calls,
        (unsigned long long)memory_stats.expert_union_fetch_experts,
        (unsigned long long)memory_stats.expert_gate_up_epochs,
        (unsigned long long)memory_stats.expert_down_epochs,
        (unsigned long long)memory_stats.q4_pool_submissions,
        (unsigned long long)memory_stats.q4_pool_fallbacks,
        memory_stats.q4_pool_workers,
        (unsigned long long)memory_stats.q4_multi_pool_submissions,
        (unsigned long long)memory_stats.q4_multi_pool_jobs,
        (unsigned long long)memory_stats.qkv_multi_pool_submissions,
        (unsigned long long)memory_stats.q4_multi_pool_fallbacks,
        (unsigned long long)memory_stats.q8_pool_submissions,
        (unsigned long long)memory_stats.q8_pool_fallbacks,
        (unsigned long long)memory_stats.expert_pool_submissions,
        (unsigned long long)memory_stats.expert_pool_fallbacks,
        (unsigned long long)memory_stats.gpu_dense_submissions,
        (unsigned long long)memory_stats.gpu_router_submissions,
        (unsigned long long)memory_stats.gpu_expert_gate_up_submissions,
        (unsigned long long)memory_stats.gpu_expert_down_submissions,
        (unsigned long long)memory_stats.gpu_head_submissions,
        (unsigned long long)memory_stats.gpu_expert_gate_up_commands,
        (unsigned long long)memory_stats.gpu_expert_down_commands,
        (unsigned long long)memory_stats.gpu_decode_expert_gate_up_commands,
        (unsigned long long)memory_stats.gpu_decode_expert_down_commands,
        (unsigned long long)memory_stats.gpu_decode_submissions,
        (unsigned long long)memory_stats.gpu_failures,
        (unsigned long long)memory_stats.gpu_shared_output_batches,
        (unsigned long long)memory_stats.gpu_shared_output_jobs,
        (unsigned long long)memory_stats.gpu_shared_arena_bytes,
        memory_stats.gpu_weight_addressability,
        memory_stats.gpu_weight_described_resources,
        memory_stats.gpu_weight_active_resources,
        memory_stats.gpu_weight_active_windows,
        memory_stats.token_program_cells,
        memory_stats.token_caller_submissions,
        memory_stats.token_internal_barriers,
        memory_stats.token_completion_fences,
        memory_stats.token_intermediate_publications,
        memory_stats.token_final_publications,
        (unsigned long long)memory_stats.token_cpu_pool_sessions,
        (unsigned long long)memory_stats.token_cpu_pool_phases,
        (unsigned long long)memory_stats.gpu_weight_described_bytes,
        (unsigned long long)memory_stats.gpu_weight_registered_bytes,
        (unsigned long long)memory_stats.gpu_weight_pageable_bytes,
        (unsigned long long)memory_stats.gpu_weight_active_window_bytes,
        (unsigned long long)memory_stats.gpu_weight_peak_window_bytes,
        (unsigned long long)memory_stats.gpu_weight_copied_bytes,
        (unsigned long long)memory_stats.gpu_attention_batches,
        (unsigned long long)memory_stats.gpu_attention_tasks,
        (unsigned long long)memory_stats.gpu_attention_kv_bytes,
        (unsigned long long)memory_stats.prefill_ffn_arena_bytes,
        (unsigned long long)memory_stats.prefill_ffn_arena_calls,
        memory_stats.expert_inflight_slots,
        memory_stats.expert_peak_inflight_slots,
        memory_stats.dense_retained_immutable,
        (unsigned long long)memory_stats.dense_payload_bytes
    );
    fputs("GEMMA4_QA_RESPONSE_BEGIN\n", stdout);
    fwrite(response, 1, (size_t)response_length, stdout);
    if (response_length == 0 || response[response_length - 1] != '\n')
        fputc('\n', stdout);
    fputs("GEMMA4_QA_RESPONSE_END\n", stdout);
    if (compare_kv_warm) {
        fputs("GEMMA4_QA_KV_WARM_RESPONSE_BEGIN\n", stdout);
        fwrite(warm_response, 1, (size_t)warm_response_length, stdout);
        if (warm_response_length == 0 ||
            warm_response[warm_response_length - 1] != '\n')
            fputc('\n', stdout);
        fputs("GEMMA4_QA_KV_WARM_RESPONSE_END\n", stdout);
    }
    fflush(stdout);
    fflush(stderr);

    exit_code = 0;
done:
    if (route_observer_active)
        (void)salt_gemma4_text_route_observer_end(model);
    free(route_items);
    free(dpr_serial_snapshot);
    free(dpr_final_snapshot);
    free(target_seed_ids);
    free(dpr_serial_ids);
    free(dpr_draft_ids);
    free(final_kv_snapshot);
    free(kv_snapshot);
    free(warm_response);
    free(prefill_logits);
    free(warm_output_ids);
    free(response);
    free(proposal_logits);
    free(logits);
    free(output_ids);
    free(prompt_ids);
    salt_gemma4_text_free(model);
    return exit_code;
}
