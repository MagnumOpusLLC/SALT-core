#ifndef SALT_GEMMA4_H
#define SALT_GEMMA4_H

#include <stddef.h>

/* Production scalar anchors for Gemma 4. Optimized implementations must
 * preserve these operation boundaries and be qualified independently.
 * Unless a function is explicitly in-place, output ranges may not partially
 * overlap inputs. Exact in-place aliasing is permitted only for RMSNorm
 * out==x, router out==state, position out==patches, and projection
 * out==vision_states. On failure primary outputs are unchanged; FFN scratch
 * contents are workspace and undefined. */

float salt_gemma4_embedding_scale(int hidden);
float salt_gemma4_gelu_tanh(float x);

int salt_gemma4_rmsnorm(float *out, const float *x, const float *weight,
                        int n, float eps, int with_scale);

int salt_gemma4_router_input(float *out, const float *state,
                             const float *scale, int hidden, float eps);

int salt_gemma4_router_topk(const float *logits,
                            const float *per_expert_scale,
                            int n_experts, int topk,
                            int *selected, float *weights);

int salt_gemma4_residual_postnorm(float *out, const float *residual,
                                  const float *branch,
                                  const float *post_norm_weight,
                                  int n, float eps);

int salt_gemma4_parallel_ffn_combine(float *out, const float *residual,
                                     const float *dense,
                                     const float *routed,
                                     const float *dense_post_weight,
                                     const float *routed_post_weight,
                                     const float *final_post_weight,
                                     float layer_scalar, int n, float eps,
                                     float *scratch_2n);

int salt_gemma4_softcap(float *logits, int n, float cap);

/* Apply text RoPE in-place. rope_dim counts all rotated channels while
 * rope_base_dim is the denominator used to generate inverse frequencies.
 * Proportional RoPE therefore rotates rope_dim channels but preserves the
 * head_dim-sized rotate-half layout. The scalar anchor emulates BF16 tensor
 * operation boundaries. */
int salt_gemma4_text_rope(float *head, int head_dim, int rope_dim,
                          int rope_base_dim, int position,
                          float rope_theta);

/* Apply the same BF16-boundary text RoPE using precomputed deterministic
 * cosine/sine factors for rope_dim/2 pairs. The on-demand function above is
 * the proof oracle; this operation seam changes factor ownership only. */
int salt_gemma4_text_rope_factors(float *head, int head_dim, int rope_dim,
                                  const float *cosines, const float *sines);

/* Construct Gemma 4's no-cache masks, row-major [query, key]. Full layers are
 * causal. Sliding layers are:
 *   sliding_window AND (causal OR same_contiguous_vision_block)
 * mm_token_type 1 and 2 are vision; all other values are text. valid uses
 * 1=valid, 0=padding. mask_count and block_count are caller capacities.
 * Output matrices must not overlap each other or any input/scratch range. */
int salt_gemma4_attention_masks(unsigned char *full,
                                unsigned char *sliding,
                                size_t mask_count,
                                const unsigned char *mm_token_type,
                                const unsigned char *valid,
                                int n_tokens, int sliding_window,
                                int *block_ids, size_t block_count);

/* Vision inputs arrive as flattened 16x16x3 patches from the canonical image
 * processor. These helpers define the model-owned boundaries after decode and
 * resize; they do not implement image codecs. */
int salt_gemma4_vision_normalize_pixels(float *pixels, int n);

int salt_gemma4_vision_add_positions(float *out, const float *patches,
                                     const float *position_table,
                                     const int *positions_xy,
                                     const unsigned char *padding,
                                     int n_patches, int hidden,
                                     int position_embedding_size);

int salt_gemma4_vision_rope(float *head, int head_dim,
                            int position_x, int position_y,
                            float rope_theta);

int salt_gemma4_vision_pool(float *out, unsigned char *valid,
                            const float *hidden_states,
                            const int *positions_xy,
                            const unsigned char *padding,
                            int input_length, int output_length,
                            int hidden);

int salt_gemma4_vision_standardize(float *states, const float *bias,
                                   const float *scale, int tokens,
                                   int hidden);

int salt_gemma4_vision_prepare_projection(float *out,
                                          const float *vision_states,
                                          int tokens, int hidden, float eps);

int salt_gemma4_scatter_image_features(float *text_embeddings,
                                       const int *input_ids,
                                       int sequence_length, int text_hidden,
                                       int image_token_id,
                                       const float *image_features,
                                       int image_tokens);

#endif /* SALT_GEMMA4_H */
