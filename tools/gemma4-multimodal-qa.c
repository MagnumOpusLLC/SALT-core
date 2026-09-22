#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L

#include "gemma4_text.h"
#include "gemma4_vision.h"
#include "sha256.h"
#include "gemma4_kv_file.h"
#include "gemma4-memory.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define G4_IMAGE_TOKEN 258880
#define G4_TEXT_HIDDEN 2816

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return -1.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static int parse_positive(const char *text, int *value) {
    char *end = NULL;
    long parsed;
    if (!text || !value) return -1;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || !end || *end || parsed < 1 || parsed > 1000000) return -1;
    *value = (int)parsed;
    return 0;
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

static void print_hex(FILE *stream, const uint8_t digest[32]) {
    for (int i = 0; i < 32; i++) fprintf(stream, "%02x", digest[i]);
}

typedef struct {
    int output_count;
    int stop_token;
    double first_token_ready;
    double end;
} DecodeResult;

static int generate(SaltGemma4Text *model, float *logits,
                    int *output_ids, int generation_cap,
                    const char *phase, DecodeResult *result) {
    int next;
    if (!model || !logits || !output_ids || generation_cap < 1 ||
        !phase || !result)
        return -1;
    memset(result, 0, sizeof *result);
    result->stop_token = -1;
    result->first_token_ready = -1.0;
    next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
    if (next < 0) {
        fprintf(stderr,
                "gemma4 multimodal runner: %s non-finite first logits\n",
                phase);
        return -1;
    }
    while (result->output_count < generation_cap) {
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
        double step_start = now_seconds();
        if (salt_gemma4_text_step(model, next, logits) != 0) {
            fprintf(stderr,
                    "gemma4 multimodal runner: %s decode failed after output %d\n",
                    phase, result->output_count);
            return -1;
        }
        fprintf(stderr, "[%s-decode-step] consumed=%d wall_s=%.6f\n",
                phase, next, now_seconds() - step_start);
        next = greedy_token(logits, salt_gemma4_text_vocab_size(model));
        if (next < 0) {
            fprintf(stderr, "gemma4 multimodal runner: %s non-finite logits\n",
                    phase);
            return -1;
        }
    }
    result->end = now_seconds();
    return result->first_token_ready >= 0.0 ? 0 : -1;
}

static void usage(const char *program) {
    fprintf(stderr,
        "usage: %s --ctx 200 --gen 50 --prompt TEXT --image IMAGE "
        "[--workers 8] [--max-soft-tokens 70] [--kv-save FILE] "
        "[--compare-kv-warm] [--package-proof]\n",
        program);
}

int main(int argc, char **argv) {
    int context = 0, generation_cap = 0, workers = 8;
    int max_soft_tokens = 70, compare_kv_warm = 0, package_proof = 0;
    const char *prompt = NULL, *image = NULL, *kv_save_path = NULL;
    SaltGemma4Text *text_model = NULL;
    SaltGemma4Vision *vision_model = NULL;
    int *prompt_ids = NULL, *output_ids = NULL, *warm_output_ids = NULL;
    int *positions = NULL;
    unsigned char *mm_types = NULL, *padding = NULL;
    float *patches = NULL, *features = NULL, *logits = NULL;
    float *prefill_logits = NULL;
    char *response = NULL, *warm_response = NULL;
    void *kv_snapshot = NULL;
    size_t kv_snapshot_bytes = 0, kv_header_bytes = 0;
    size_t kv_bytes_per_token = 0, kv_poisoned_bytes = 0;
    char error[256] = {0};
    int n_patches = 0, expected_features = 0, width = 0, height = 0;
    int prompt_count = -1, actual_features = -1, exit_code = 1;
    DecodeResult cold = {0}, warm = {0};
    SaltGemma4MemoryStats memory_stats = {0};
    G4ProcessMemory process_memory = {0};
    uint8_t feature_digest[32];
    double load_start, load_end, image_start, image_end, vision_end;
    double prefill_end, request_start, cold_decode_start;
    double state_capture_start = 0.0, state_capture_end = 0.0;
    double kv_capture_start = 0.0, kv_capture_end = 0.0;
    double logit_capture_start = 0.0, logit_capture_end = 0.0;
    double poison_start = 0.0, poison_end = 0.0;
    double warm_start = 0.0;
    double kv_restore_start = 0.0, kv_restore_end = 0.0;
    double logit_restore_start = 0.0, logit_restore_end = 0.0;
    int response_length = -1, warm_response_length = -1;
    int kv_saved_tokens = 0;
    int vision_pool_workers = 0;
    uint64_t vision_pool_submissions = 0;
    int vision_gpu_hmm = 0;
    uint64_t vision_gpu_submissions = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            if (parse_positive(argv[++i], &context) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--gen") && i + 1 < argc) {
            if (parse_positive(argv[++i], &generation_cap) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--workers") && i + 1 < argc) {
            if (parse_positive(argv[++i], &workers) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--max-soft-tokens") && i + 1 < argc) {
            if (parse_positive(argv[++i], &max_soft_tokens) != 0) goto bad_args;
        } else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (!strcmp(argv[i], "--image") && i + 1 < argc) {
            image = argv[++i];
        } else if (!strcmp(argv[i], "--kv-save") && i + 1 < argc) {
            kv_save_path = argv[++i];
        } else if (!strcmp(argv[i], "--compare-kv-warm")) {
            compare_kv_warm = 1;
        } else if (!strcmp(argv[i], "--package-proof")) {
            package_proof = 1;
        } else {
            goto bad_args;
        }
    }
    if (context < 1 || generation_cap < 1 || generation_cap > context ||
        workers < 1 || workers > 8 || max_soft_tokens != 70 ||
        !prompt || !*prompt || !image || !*image ||
        (compare_kv_warm && kv_save_path))
        goto bad_args;

    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    load_start = now_seconds();
    text_model = salt_gemma4_text_load(stdin, context, workers,
                                       error, sizeof error);
    if (!text_model) {
        fprintf(stderr, "gemma4 multimodal runner: text load failed: %s\n", error);
        goto done;
    }
    if (package_proof && salt_gemma4_text_validate_proof(text_model) != 0) {
        fputs("gemma4 multimodal runner: one-shot proof validation failed\n",
              stderr);
        goto done;
    }
    vision_model = salt_gemma4_vision_load(
        stdin, workers, error, sizeof error);
    if (!vision_model) {
        fprintf(stderr, "gemma4 multimodal runner: vision load failed: %s\n", error);
        goto done;
    }
    load_end = now_seconds();
    if (g4_print_process_memory(stderr, "after_load", NULL) != 0) {
        fputs("gemma4 multimodal runner: load memory sample failed\n", stderr);
        goto done;
    }

    prompt_ids = (int *)malloc((size_t)context * sizeof *prompt_ids);
    output_ids = (int *)malloc((size_t)generation_cap * sizeof *output_ids);
    mm_types = (unsigned char *)calloc((size_t)context, 1);
    logits = (float *)malloc(
        (size_t)salt_gemma4_text_vocab_size(text_model) * sizeof *logits);
    response = (char *)malloc((size_t)generation_cap * 64u + 1024u);
    if (compare_kv_warm) {
        warm_output_ids = (int *)malloc(
            (size_t)generation_cap * sizeof *warm_output_ids);
        prefill_logits = (float *)malloc(
            (size_t)salt_gemma4_text_vocab_size(text_model) *
            sizeof *prefill_logits);
        warm_response = (char *)malloc(
            (size_t)generation_cap * 64u + 1024u);
    }
    if (!prompt_ids || !output_ids || !mm_types || !logits || !response) {
        fputs("gemma4 multimodal runner: allocation failed\n", stderr);
        goto done;
    }
    if (compare_kv_warm &&
        (!warm_output_ids || !prefill_logits || !warm_response)) {
        fputs("gemma4 multimodal runner: KV-warm allocation failed\n", stderr);
        goto done;
    }
    prompt_count = salt_gemma4_text_encode(
        text_model, prompt, prompt_ids, context);
    if (prompt_count < 1 || prompt_count + generation_cap > context) {
        fprintf(stderr,
            "gemma4 multimodal runner: prompt/generation exceed context: "
            "prompt=%d ctx=%d gen=%d\n",
            prompt_count, context, generation_cap);
        goto done;
    }
    int placeholder_count = 0;
    for (int i = 0; i < prompt_count; i++) {
        if (prompt_ids[i] == G4_IMAGE_TOKEN) {
            mm_types[i] = 1;
            placeholder_count++;
        }
    }

    image_start = now_seconds();
    if (salt_gemma4_ppm_to_patches(image, max_soft_tokens,
            &patches, &positions, &padding, &n_patches, &expected_features,
            &width, &height, error, sizeof error) != 0) {
        fprintf(stderr,
            "gemma4 multimodal runner: image processing failed: %s\n", error);
        goto done;
    }
    image_end = now_seconds();
    features = (float *)calloc(
        (size_t)expected_features * G4_TEXT_HIDDEN, sizeof(float));
    if (!features) {
        fputs("gemma4 multimodal runner: feature allocation failed\n", stderr);
        goto done;
    }
    actual_features = salt_gemma4_vision_forward(
        vision_model, patches, positions, padding, n_patches,
        features, expected_features, error, sizeof error);
    vision_end = now_seconds();
    if (actual_features < 0 || actual_features != expected_features ||
        actual_features != placeholder_count) {
        fprintf(stderr,
            "gemma4 multimodal runner: feature/placeholder mismatch: %s "
            "expected=%d actual=%d placeholders=%d\n",
            error, expected_features, actual_features, placeholder_count);
        goto done;
    }
    size_t feature_count = (size_t)actual_features * G4_TEXT_HIDDEN;
    if (salt_sha256_bytes(features, feature_count * sizeof(float),
                          feature_digest) != 0) {
        fputs("gemma4 multimodal runner: feature hash failed\n", stderr);
        goto done;
    }
    if (g4_print_process_memory(stderr, "after_vision_forward", NULL) != 0) {
        fputs("gemma4 multimodal runner: vision memory sample failed\n", stderr);
        goto done;
    }
    if (salt_gemma4_vision_worker_stats(
            vision_model, &vision_pool_workers,
            &vision_pool_submissions) != 0 ||
        salt_gemma4_vision_gpu_stats(
            vision_model, &vision_gpu_hmm,
            &vision_gpu_submissions) != 0) {
        fputs("gemma4 multimodal runner: vision execution stats failed\n", stderr);
        goto done;
    }
    if (salt_gemma4_vision_close(&vision_model) != 0) {
        fputs("gemma4 multimodal runner: vision release failed\n", stderr);
        goto done;
    }
    if (g4_print_process_memory(stderr, "after_vision_release", NULL) != 0) {
        fputs("gemma4 multimodal runner: vision release sample failed\n", stderr);
        goto done;
    }

    fprintf(stderr,
        "GEMMA4_MM_EXECUTION_START ctx=%d gen_cap=%d workers=%d "
        "prompt_tokens=%d image_tokens=%d image=%dx%d patch_capacity=%d "
        "mode=greedy kv_compare=%d runtime_ready=true load_s=%.6f image_s=%.6f "
        "vision_s=%.6f vision_pool_workers=%d "
        "vision_pool_submissions=%llu vision_gpu_hmm=%d "
        "vision_gpu_submissions=%llu feature_sha256=",
        context, generation_cap, workers, prompt_count, actual_features,
        width, height, n_patches, compare_kv_warm, load_end - load_start,
        image_end - image_start, vision_end - image_end,
        vision_pool_workers,
        (unsigned long long)vision_pool_submissions, vision_gpu_hmm,
        (unsigned long long)vision_gpu_submissions);
    print_hex(stderr, feature_digest);
    fputc('\n', stderr);
    print_ids(stderr, "prompt_ids", prompt_ids, prompt_count);

    request_start = now_seconds();
    if (salt_gemma4_text_prefill_image(text_model, prompt_ids, mm_types,
            prompt_count, features, actual_features, logits) != 0) {
        fputs("gemma4 multimodal runner: image prefill failed\n", stderr);
        goto done;
    }
    prefill_end = now_seconds();
    fprintf(stderr, "[multimodal-prefill] tokens=%d wall_s=%.6f\n",
            prompt_count, prefill_end - request_start);
    if (g4_print_process_memory(stderr, "after_prefill", NULL) != 0) {
        fputs("gemma4 multimodal runner: prefill memory sample failed\n", stderr);
        goto done;
    }

    if (compare_kv_warm) {
        state_capture_start = now_seconds();
        kv_snapshot_bytes = salt_gemma4_text_kv_snapshot_size(text_model);
        kv_header_bytes = salt_gemma4_text_kv_snapshot_header_size();
        if (kv_snapshot_bytes == 0 || !(kv_snapshot = malloc(kv_snapshot_bytes))) {
            fputs("gemma4 multimodal runner: KV snapshot allocation failed\n",
                  stderr);
            goto done;
        }
        if (kv_header_bytes == 0 || kv_snapshot_bytes <= kv_header_bytes ||
            (kv_snapshot_bytes - kv_header_bytes) % (size_t)prompt_count != 0) {
            fputs("gemma4 multimodal runner: invalid KV snapshot geometry\n",
                  stderr);
            goto done;
        }
        kv_bytes_per_token =
            (kv_snapshot_bytes - kv_header_bytes) / (size_t)prompt_count;
        kv_capture_start = now_seconds();
        if (salt_gemma4_text_kv_snapshot(
                text_model, kv_snapshot, kv_snapshot_bytes) != 0 ||
            salt_gemma4_text_position(text_model) != prompt_count) {
            fputs("gemma4 multimodal runner: KV snapshot failed\n", stderr);
            goto done;
        }
        kv_capture_end = now_seconds();
        logit_capture_start = now_seconds();
        memcpy(prefill_logits, logits,
               (size_t)salt_gemma4_text_vocab_size(text_model) *
               sizeof *logits);
        logit_capture_end = now_seconds();
        state_capture_end = logit_capture_end;
    }

    cold_decode_start = now_seconds();
    if (generate(text_model, logits, output_ids, generation_cap,
                 "cold", &cold) != 0)
        goto done;
    if (g4_print_process_memory(stderr, "after_cold_decode", NULL) != 0) {
        fputs("gemma4 multimodal runner: cold memory sample failed\n", stderr);
        goto done;
    }
    int decode_count = cold.output_count;
    if (decode_count > 0 && is_stop_token(output_ids[decode_count - 1]))
        decode_count--;
    int response_capacity = generation_cap * 64 + 1024;
    response_length = salt_gemma4_text_decode(
        text_model, output_ids, decode_count, response, response_capacity);
    if (response_length < 0 || response_length >= response_capacity ||
        memchr(response, '\0', (size_t)response_length) != NULL) {
        fputs("gemma4 multimodal runner: output decode failed\n", stderr);
        goto done;
    }

    if (compare_kv_warm) {
        poison_start = now_seconds();
        if (salt_gemma4_text_kv_poison_prefix(
                text_model, prompt_count, &kv_poisoned_bytes) != 0 ||
            kv_poisoned_bytes != kv_snapshot_bytes - kv_header_bytes) {
            fputs("gemma4 multimodal runner: KV poison proof failed\n", stderr);
            goto done;
        }
        poison_end = now_seconds();
        warm_start = now_seconds();
        kv_restore_start = warm_start;
        if (salt_gemma4_text_kv_restore(
                text_model, kv_snapshot, kv_snapshot_bytes) != 0 ||
            salt_gemma4_text_position(text_model) != prompt_count) {
            fputs("gemma4 multimodal runner: KV restore failed\n", stderr);
            goto done;
        }
        kv_restore_end = now_seconds();
        logit_restore_start = kv_restore_end;
        memcpy(logits, prefill_logits,
               (size_t)salt_gemma4_text_vocab_size(text_model) *
               sizeof *logits);
        logit_restore_end = now_seconds();
        if (generate(text_model, logits, warm_output_ids, generation_cap,
                     "warm", &warm) != 0)
            goto done;
        if (g4_print_process_memory(stderr, "after_warm_decode", NULL) != 0) {
            fputs("gemma4 multimodal runner: warm memory sample failed\n", stderr);
            goto done;
        }
        int warm_decode_count = warm.output_count;
        if (warm_decode_count > 0 &&
            is_stop_token(warm_output_ids[warm_decode_count - 1]))
            warm_decode_count--;
        warm_response_length = salt_gemma4_text_decode(
            text_model, warm_output_ids, warm_decode_count,
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
            fputs("gemma4 multimodal runner: cold/warm output drift\n", stderr);
            goto done;
        }
    }

    if (kv_save_path) {
        int expected_position;
        if (cold.stop_token != 106 || cold.output_count < 1) {
            fputs("gemma4 multimodal runner: KV cache save requires a completed turn\n",
                  stderr);
            goto done;
        }
        if (salt_gemma4_text_step(text_model, cold.stop_token, NULL) != 0) {
            fputs("gemma4 multimodal runner: KV cache final turn step failed\n",
                  stderr);
            goto done;
        }
        expected_position = prompt_count + cold.output_count;
        if (salt_gemma4_text_position(text_model) != expected_position ||
            g4_kv_file_save(text_model, kv_save_path, &kv_saved_tokens,
                            error, sizeof error) != 0 ||
            kv_saved_tokens != expected_position) {
            fprintf(stderr, "gemma4 multimodal runner: KV cache save failed: %s\n",
                    error);
            goto done;
        }
    }

    unsigned long long peak_bytes;
    if (salt_gemma4_text_memory_stats(text_model, &memory_stats) != 0 ||
        g4_print_process_memory(stderr, "final", &process_memory) != 0) {
        fputs("gemma4 multimodal runner: final memory accounting failed\n",
              stderr);
        goto done;
    }
    peak_bytes = process_memory.resident_peak_bytes;
    print_ids(stderr, "output_ids", output_ids, cold.output_count);
    if (compare_kv_warm)
        print_ids(stderr, "warm_output_ids", warm_output_ids, warm.output_count);
    double prefill_s = prefill_end - request_start;
    double text_first_token_ready_s =
        prefill_s + cold.first_token_ready - cold_decode_start;
    double text_total_s = prefill_s + cold.end - cold_decode_start;
    double cold_first_token_ready_s = prefill_end - image_start +
                                      cold.first_token_ready - cold_decode_start;
    double cold_request_s = prefill_end - image_start +
                            cold.end - cold_decode_start;
    double native_total_s = prefill_end - load_start +
                            cold.end - cold_decode_start;
    fprintf(stderr,
        "GEMMA4_MM_EXECUTION_DONE prompt_tokens=%d image_tokens=%d "
        "output_steps=%d stop_token=%d response_bytes=%d warm_response_bytes=%d "
        "kv_loaded_tokens=0 kv_saved_tokens=%d "
        "state_envelope=%s stop_commit_steps=%d package_proof=%d "
        "prefill_s=%.6f first_token_ready_s=%.6f "
        "text_s=%.6f native_total_s=%.6f peak_rss_gb=%.9f "
        "peak_rss_bytes=%llu cold_first_token_ready_s=%.6f "
        "cold_request_s=%.6f "
        "kv_compare=%d kv_prefill_position=%d kv_snapshot_bytes=%zu "
        "kv_snapshot_version=2 kv_snapshot_header_bytes=%zu "
        "kv_snapshot_bytes_per_token=%zu kv_poisoned=%d "
        "kv_poisoned_bytes=%zu kv_restore_verified=%d "
        "state_capture_s=%.6f kv_capture_s=%.6f logit_capture_s=%.6f "
        "kv_poison_s=%.6f kv_restore_s=%.6f logit_restore_s=%.6f "
        "state_restore_s=%.6f warm_first_token_ready_s=%.6f "
        "warm_total_s=%.6f memory_detailed=%d proof_validation_calls=%llu "
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
        "expert_union_fetch_calls=%llu expert_union_fetch_experts=%llu "
        "expert_gate_up_epochs=%llu expert_down_epochs=%llu "
        "q4_pool_submissions=%llu q4_pool_fallbacks=%llu q4_pool_workers=%d "
        "q4_multi_pool_submissions=%llu q4_multi_pool_jobs=%llu "
        "qkv_multi_pool_submissions=%llu "
        "q4_multi_pool_fallbacks=%llu "
        "q8_pool_submissions=%llu q8_pool_fallbacks=%llu "
        "expert_pool_submissions=%llu expert_pool_fallbacks=%llu "
        "gpu_dense_submissions=%llu "
        "gpu_expert_gate_up_submissions=%llu "
        "gpu_expert_down_submissions=%llu gpu_head_submissions=%llu "
        "gpu_decode_submissions=%llu gpu_failures=%llu "
        "gpu_weight_addressability=%u "
        "gpu_weight_described_resources=%u gpu_weight_active_resources=%u "
        "gpu_weight_described_bytes=%llu gpu_weight_registered_bytes=%llu "
        "gpu_weight_pageable_bytes=%llu gpu_weight_copied_bytes=%llu "
        "gpu_attention_batches=%llu gpu_attention_tasks=%llu "
        "gpu_attention_kv_bytes=%llu "
        "prefill_ffn_arena_bytes=%llu prefill_ffn_arena_calls=%llu "
        "expert_inflight_slots=%d "
        "expert_peak_inflight_slots=%d dense_retained_immutable=%d "
        "dense_payload_bytes=%llu "
        "runtime_ready=true\n",
        prompt_count, actual_features, cold.output_count, cold.stop_token,
        response_length, compare_kv_warm ? warm_response_length : 0,
        kv_saved_tokens,
        compare_kv_warm ? "kv-restore-proof" :
            (kv_save_path ? "kv-save" : "none"),
        kv_save_path ? 1 : 0, package_proof,
        prefill_s, text_first_token_ready_s, text_total_s, native_total_s,
        (double)peak_bytes / 1.0e9, peak_bytes,
        cold_first_token_ready_s, cold_request_s,
        compare_kv_warm, compare_kv_warm ? prompt_count : 0,
        kv_snapshot_bytes, kv_header_bytes, kv_bytes_per_token,
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
        (unsigned long long)memory_stats.gpu_expert_gate_up_submissions,
        (unsigned long long)memory_stats.gpu_expert_down_submissions,
        (unsigned long long)memory_stats.gpu_head_submissions,
        (unsigned long long)memory_stats.gpu_decode_submissions,
        (unsigned long long)memory_stats.gpu_failures,
        memory_stats.gpu_weight_addressability,
        memory_stats.gpu_weight_described_resources,
        memory_stats.gpu_weight_active_resources,
        (unsigned long long)memory_stats.gpu_weight_described_bytes,
        (unsigned long long)memory_stats.gpu_weight_registered_bytes,
        (unsigned long long)memory_stats.gpu_weight_pageable_bytes,
        (unsigned long long)memory_stats.gpu_weight_copied_bytes,
        (unsigned long long)memory_stats.gpu_attention_batches,
        (unsigned long long)memory_stats.gpu_attention_tasks,
        (unsigned long long)memory_stats.gpu_attention_kv_bytes,
        (unsigned long long)memory_stats.prefill_ffn_arena_bytes,
        (unsigned long long)memory_stats.prefill_ffn_arena_calls,
        memory_stats.expert_inflight_slots,
        memory_stats.expert_peak_inflight_slots,
        memory_stats.dense_retained_immutable,
        (unsigned long long)memory_stats.dense_payload_bytes);
    fputs("GEMMA4_MM_RESPONSE_BEGIN\n", stdout);
    fwrite(response, 1, (size_t)response_length, stdout);
    if (response_length == 0 || response[response_length - 1] != '\n')
        fputc('\n', stdout);
    fputs("GEMMA4_MM_RESPONSE_END\n", stdout);
    if (compare_kv_warm) {
        fputs("GEMMA4_MM_KV_WARM_RESPONSE_BEGIN\n", stdout);
        fwrite(warm_response, 1, (size_t)warm_response_length, stdout);
        if (warm_response_length == 0 ||
            warm_response[warm_response_length - 1] != '\n')
            fputc('\n', stdout);
        fputs("GEMMA4_MM_KV_WARM_RESPONSE_END\n", stdout);
    }
    fputs("runtime_ready=true\n", stdout);
    exit_code = 0;
    goto done;

bad_args:
    usage(argv[0]);
    return 2;

done:
    free(kv_snapshot);
    free(warm_response);
    free(prefill_logits);
    free(warm_output_ids);
    free(response);
    free(logits);
    free(features);
    salt_gemma4_free_patches(patches, positions, padding);
    free(mm_types);
    free(output_ids);
    free(prompt_ids);
    if (salt_gemma4_vision_close(&vision_model) != 0) exit_code = 1;
    salt_gemma4_text_free(text_model);
    return exit_code;
}
