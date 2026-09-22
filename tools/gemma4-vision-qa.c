#if defined(__APPLE__)
#define _DARWIN_C_SOURCE 1
#endif
#define _POSIX_C_SOURCE 200809L

#include "gemma4_vision.h"
#include "sha256.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

static double elapsed(const struct timespec *start, const struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) +
           (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
}

static unsigned long long peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return (unsigned long long)usage.ru_maxrss;
#else
    return (unsigned long long)usage.ru_maxrss * 1024u;
#endif
}

static void print_hex(const uint8_t digest[32]) {
    for (int i = 0; i < 32; i++) printf("%02x", digest[i]);
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

int main(int argc, char **argv) {
    const char *image = NULL;
    int max_soft_tokens = 70;
    char error[256] = {0};
    SaltGemma4Vision *model = NULL;
    float *patches = NULL, *features = NULL;
    int *positions = NULL;
    unsigned char *padding = NULL;
    int n_patches = 0, expected_features = 0, width = 0, height = 0;
    int actual_features = -1;
    int gpu_hmm = 0;
    uint64_t gpu_submissions = 0;
    uint8_t digest[32];
    struct timespec total_start, load_end, image_end, vision_end;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--image") && i + 1 < argc) image = argv[++i];
        else if (!strcmp(argv[i], "--max-soft-tokens") && i + 1 < argc) {
            if (parse_positive(argv[++i], &max_soft_tokens) != 0) {
                fprintf(stderr, "gemma4 vision runner: invalid soft-token budget\n");
                return 2;
            }
        } else {
            fprintf(stderr, "gemma4 vision runner: invalid argument\n");
            return 2;
        }
    }
    if (!image) {
        fprintf(stderr, "gemma4 vision runner: --image is required\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    setvbuf(stderr, NULL, _IOLBF, 0);
    clock_gettime(CLOCK_MONOTONIC, &total_start);
    model = salt_gemma4_vision_load(stdin, 1, error, sizeof error);
    if (!model) {
        fprintf(stderr, "gemma4 vision runner: load failed: %s\n", error);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &load_end);
    if (salt_gemma4_ppm_to_patches(image, max_soft_tokens,
            &patches, &positions, &padding, &n_patches, &expected_features,
            &width, &height, error, sizeof error) != 0) {
        fprintf(stderr, "gemma4 vision runner: image processing failed: %s\n", error);
        salt_gemma4_vision_free(model);
        return 1;
    }
    clock_gettime(CLOCK_MONOTONIC, &image_end);
    features = (float *)calloc((size_t)expected_features * 2816u, sizeof(float));
    if (!features) {
        fprintf(stderr, "gemma4 vision runner: feature allocation failed\n");
        salt_gemma4_free_patches(patches, positions, padding);
        salt_gemma4_vision_free(model);
        return 1;
    }
    actual_features = salt_gemma4_vision_forward(model, patches, positions,
        padding, n_patches, features, expected_features, error, sizeof error);
    clock_gettime(CLOCK_MONOTONIC, &vision_end);
    if (actual_features < 0 || actual_features != expected_features) {
        fprintf(stderr,
            "gemma4 vision runner: forward failed: %s expected=%d actual=%d\n",
            error, expected_features, actual_features);
        free(features);
        salt_gemma4_free_patches(patches, positions, padding);
        salt_gemma4_vision_free(model);
        return 1;
    }
    size_t feature_count = (size_t)actual_features * 2816u;
    if (salt_sha256_bytes(features, feature_count * sizeof(float), digest) != 0) {
        fprintf(stderr, "gemma4 vision runner: feature hash failed\n");
        free(features);
        salt_gemma4_free_patches(patches, positions, padding);
        salt_gemma4_vision_free(model);
        return 1;
    }
    double l2 = 0.0;
    float minimum = INFINITY, maximum = -INFINITY;
    for (size_t i = 0; i < feature_count; i++) {
        if (!isfinite(features[i])) {
            fprintf(stderr, "gemma4 vision runner: non-finite feature stream\n");
            free(features);
            salt_gemma4_free_patches(patches, positions, padding);
            salt_gemma4_vision_free(model);
            return 1;
        }
        if (features[i] < minimum) minimum = features[i];
        if (features[i] > maximum) maximum = features[i];
        l2 += (double)features[i] * features[i];
    }
    if (salt_gemma4_vision_gpu_stats(
            model, &gpu_hmm, &gpu_submissions) != 0) {
        fprintf(stderr, "gemma4 vision runner: execution stats failed\n");
        free(features);
        salt_gemma4_free_patches(patches, positions, padding);
        salt_gemma4_vision_free(model);
        return 1;
    }
    printf("GEMMA4_VISION_FEATURES_OK\n");
    printf("image_width=%d image_height=%d patch_capacity=%d valid_soft_tokens=%d\n",
           width, height, n_patches, actual_features);
    printf("feature_values=%zu feature_gb=%.9f feature_bytes=%zu feature_sha256=",
           feature_count, (double)(feature_count * sizeof(float)) / 1.0e9,
           feature_count * sizeof(float));
    print_hex(digest);
    putchar('\n');
    printf("feature_min=%.9g feature_max=%.9g feature_l2=%.12g\n",
           minimum, maximum, sqrt(l2));
    printf("feature_prefix=");
    for (int i = 0; i < 8 && (size_t)i < feature_count; i++)
        printf("%s%.9g", i ? "," : "", features[i]);
    putchar('\n');
    printf("load_seconds=%.6f image_seconds=%.6f vision_seconds=%.6f total_seconds=%.6f\n",
           elapsed(&total_start, &load_end), elapsed(&load_end, &image_end),
           elapsed(&image_end, &vision_end), elapsed(&total_start, &vision_end));
    printf("vision_gpu_hmm=%d vision_gpu_submissions=%llu\n",
           gpu_hmm, (unsigned long long)gpu_submissions);
    unsigned long long peak_bytes = peak_rss_bytes();
    printf("peak_rss_gb=%.9f peak_rss_bytes=%llu runtime_ready=true\n",
           (double)peak_bytes / 1.0e9, peak_bytes);

    free(features);
    salt_gemma4_free_patches(patches, positions, padding);
    return salt_gemma4_vision_close(&model) == 0 ? 0 : 1;
}
