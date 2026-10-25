#ifndef SALT_GEMMA4_VISION_H
#define SALT_GEMMA4_VISION_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct SaltGemma4Vision SaltGemma4Vision;

/* Consume a private SALT_GEMMA4_VISION_BINDING_V1 stream whose source
 * descriptors have already been authenticated and retained by the caller.
 * The loader revalidates descriptor identity and exact 356-role closure. */
SaltGemma4Vision *salt_gemma4_vision_load(FILE *binding, int workers,
                                          char *error, size_t error_size);
/* Status-bearing teardown for production callers. On backend release failure
 * the model and source mappings remain owned and *model is unchanged. */
int salt_gemma4_vision_close(SaltGemma4Vision **model);
void salt_gemma4_vision_free(SaltGemma4Vision *model);
int salt_gemma4_vision_worker_stats(const SaltGemma4Vision *model,
                                    int *workers, uint64_t *submissions);
int salt_gemma4_vision_gpu_stats(const SaltGemma4Vision *model,
                                 int *pageable_hmm,
                                 uint64_t *submissions);

/* Exact BF16 composition boundaries used by the pinned vision layer. */
float salt_gemma4_vision_postnorm_residual_bf16(float residual,
                                                 float normalized);
float salt_gemma4_vision_gelu_mul_bf16(float gate, float up);

/* Encode canonical [0,1] RGB patches. patches is row-major
 * [n_patches][16*16*3], positions_xy is [n_patches][2], and padding uses
 * 1=padding. n_patches includes padding and must be divisible by 9.
 * feature_token_capacity counts valid compacted tokens, not padded tokens.
 * Returns the number of valid projected 2816-wide soft tokens, or -1. */
int salt_gemma4_vision_forward(SaltGemma4Vision *model,
                               const float *patches,
                               const int *positions_xy,
                               const unsigned char *padding,
                               int n_patches,
                               float *features,
                               int feature_token_capacity,
                               char *error, size_t error_size);

/* Strict native P6 RGB processor for images already at the canonical size
 * selected by the pinned Gemma 4 aspect-ratio calculation. It intentionally
 * fails when bicubic resize would be required; no approximate resize is used.
 * The currently qualified native subset accepts max_soft_tokens=70 only.
 * The caller owns the three returned allocations. */
int salt_gemma4_ppm_to_patches(const char *path, int max_soft_tokens,
                               float **patches,
                               int **positions_xy,
                               unsigned char **padding,
                               int *n_patches,
                               int *valid_soft_tokens,
                               int *width, int *height,
                               char *error, size_t error_size);
void salt_gemma4_free_patches(float *patches, int *positions_xy,
                              unsigned char *padding);

#endif /* SALT_GEMMA4_VISION_H */
