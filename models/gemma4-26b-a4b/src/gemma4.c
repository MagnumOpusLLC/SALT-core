#include "gemma4.h"
#include "salt/bitmath.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static float round_bf16(float x) {
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1u);
        bits &= UINT32_C(0xffff0000);
    }
    memcpy(&x, &bits, sizeof x);
    return x;
}

static float bf16_add(float a, float b) {
    return round_bf16(round_bf16(a) + round_bf16(b));
}

static float bf16_mul(float a, float b) {
    return round_bf16(round_bf16(a) * round_bf16(b));
}

static int byte_ranges_overlap(const void *a, size_t a_size,
                               const void *b, size_t b_size);

static int array_bytes(int n, size_t element_size, size_t *bytes) {
    if (n < 1 || !bytes || (size_t)n > SIZE_MAX / element_size)
        return -1;
    *bytes = (size_t)n * element_size;
    return 0;
}

static int disallowed_overlap(const void *out, size_t out_size,
                              const void *in, size_t in_size,
                              int allow_exact) {
    if (allow_exact && out == in)
        return 0;
    return byte_ranges_overlap(out, out_size, in, in_size);
}

static int rmsnorm_inverse(const float *x, const float *weight,
                           int n, float eps, int with_scale,
                           float *inv_out) {
    float ss = 0.0f, inv;
    if (!x || !inv_out || n < 1 || !(eps >= 0.0f) || !isfinite(eps) ||
        (with_scale && !weight))
        return -1;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i]) || (with_scale && !isfinite(weight[i])))
            return -1;
        ss += x[i] * x[i];
    }
    /* Matches torch.pow(mean_squared, -0.5), not rsqrt or 1 / sqrt. */
    inv = salt_powf(ss / (float)n + eps, -0.5f);
    if (!isfinite(inv))
        return -1;
    for (int i = 0; i < n; i++) {
        float y = x[i] * inv;
        if (with_scale)
            y = y * weight[i];
        if (!isfinite(y))
            return -1;
    }
    *inv_out = inv;
    return 0;
}

float salt_gemma4_embedding_scale(int hidden) {
    if (hidden < 1) return 0.0f;
    /* The reference casts sqrt(hidden) to the embedding dtype. Gemma 4's
     * source embedding is BF16, so 2816 rounds from ~53.066 to exactly 53. */
    return round_bf16(salt_sqrtf((float)hidden));
}

float salt_gemma4_gelu_tanh(float x) {
    const float k = 0.7978845608028654f; /* sqrt(2 / pi) */
    return 0.5f * x * (1.0f + salt_tanhf(k * (x + 0.044715f * x * x * x)));
}

int salt_gemma4_rmsnorm(float *out, const float *x, const float *weight,
                        int n, float eps, int with_scale) {
    float inv;
    size_t bytes;
    if (!out || array_bytes(n, sizeof *out, &bytes) != 0 ||
        disallowed_overlap(out, bytes, x, bytes, 1) ||
        (with_scale && byte_ranges_overlap(out, bytes, weight, bytes)) ||
        rmsnorm_inverse(x, weight, n, eps, with_scale, &inv) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        float y = x[i] * inv;
        if (with_scale)
            y = y * weight[i];
        out[i] = y;
    }
    return 0;
}

int salt_gemma4_router_input(float *out, const float *state,
                             const float *scale, int hidden, float eps) {
    float inv, root;
    size_t bytes;
    if (!out || !state || !scale ||
        array_bytes(hidden, sizeof *out, &bytes) != 0 ||
        disallowed_overlap(out, bytes, state, bytes, 1) ||
        byte_ranges_overlap(out, bytes, scale, bytes))
        return -1;
    for (int i = 0; i < hidden; i++)
        if (!isfinite(scale[i])) return -1;
    if (rmsnorm_inverse(state, NULL, hidden, eps, 0, &inv) != 0)
        return -1;
    root = 1.0f / salt_sqrtf((float)hidden);
    for (int i = 0; i < hidden; i++) {
        float y = state[i] * inv;
        y = y * scale[i];
        y = y * root;
        if (!isfinite(y))
            return -1;
    }
    for (int i = 0; i < hidden; i++) {
        float y = state[i] * inv;
        y = y * scale[i];
        out[i] = y * root;
    }
    return 0;
}

int salt_gemma4_router_topk(const float *logits,
                            const float *per_expert_scale,
                            int n_experts, int topk,
                            int *selected, float *weights) {
    float mx = -INFINITY, all_sum = 0.0f, selected_sum = 0.0f;
    size_t input_bytes, selected_bytes, weight_bytes;
    if (!logits || !per_expert_scale || !selected || !weights ||
        n_experts < 1 || topk < 1 || topk > n_experts ||
        array_bytes(n_experts, sizeof *logits, &input_bytes) != 0 ||
        array_bytes(topk, sizeof *selected, &selected_bytes) != 0 ||
        array_bytes(topk, sizeof *weights, &weight_bytes) != 0 ||
        byte_ranges_overlap(selected, selected_bytes, weights, weight_bytes) ||
        byte_ranges_overlap(selected, selected_bytes, logits, input_bytes) ||
        byte_ranges_overlap(selected, selected_bytes,
                            per_expert_scale, input_bytes) ||
        byte_ranges_overlap(weights, weight_bytes, logits, input_bytes) ||
        byte_ranges_overlap(weights, weight_bytes,
                            per_expert_scale, input_bytes))
        return -1;
    for (int e = 0; e < n_experts; e++) {
        if (!isfinite(logits[e]) || !isfinite(per_expert_scale[e])) return -1;
        if (logits[e] > mx) mx = logits[e];
    }
    for (int j = 0; j < topk; j++) {
        selected[j] = -1;
        weights[j] = -INFINITY;
    }
    for (int e = 0; e < n_experts; e++) {
        int j;
        float s = logits[e];
        j = topk - 1;
        while (j >= 0 && (selected[j] < 0 || s > weights[j])) j--;
        if (j < topk - 1) {
            for (int q = topk - 2; q > j; q--) {
                selected[q + 1] = selected[q];
                weights[q + 1] = weights[q];
            }
            selected[j + 1] = e;
            weights[j + 1] = s;
        }
    }
    for (int e = 0; e < n_experts; e++)
        all_sum += salt_expf(logits[e] - mx);
    if (!(all_sum > 0.0f) || !isfinite(all_sum)) return -1;
    for (int j = 0; j < topk; j++) {
        weights[j] = salt_expf(weights[j] - mx) / all_sum;
        selected_sum += weights[j];
    }
    if (!(selected_sum > 0.0f) || !isfinite(selected_sum)) return -1;
    for (int j = 0; j < topk; j++) {
        float normalized = weights[j] / selected_sum;
        weights[j] = normalized * per_expert_scale[selected[j]];
    }
    return 0;
}

int salt_gemma4_residual_postnorm(float *out, const float *residual,
                                  const float *branch,
                                  const float *post_norm_weight,
                                  int n, float eps) {
    float inv;
    size_t bytes;
    if (!out || !residual || !branch || !post_norm_weight ||
        array_bytes(n, sizeof *out, &bytes) != 0 ||
        byte_ranges_overlap(out, bytes, residual, bytes) ||
        byte_ranges_overlap(out, bytes, branch, bytes) ||
        byte_ranges_overlap(out, bytes, post_norm_weight, bytes))
        return -1;
    for (int i = 0; i < n; i++)
        if (!isfinite(residual[i])) return -1;
    if (rmsnorm_inverse(branch, post_norm_weight, n, eps, 1, &inv) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        float normalized = (branch[i] * inv) * post_norm_weight[i];
        if (!isfinite(residual[i] + normalized))
            return -1;
    }
    for (int i = 0; i < n; i++) {
        float normalized = (branch[i] * inv) * post_norm_weight[i];
        out[i] = residual[i] + normalized;
    }
    return 0;
}

int salt_gemma4_parallel_ffn_combine(float *out, const float *residual,
                                     const float *dense,
                                     const float *routed,
                                     const float *dense_post_weight,
                                     const float *routed_post_weight,
                                     const float *final_post_weight,
                                     float layer_scalar, int n, float eps,
                                     float *scratch_2n) {
    float *sum, *final;
    size_t bytes, scratch_bytes;
    if (!out || !residual || !dense || !routed || !dense_post_weight ||
        !routed_post_weight || !final_post_weight || !scratch_2n || n < 1 ||
        !isfinite(layer_scalar) ||
        array_bytes(n, sizeof *out, &bytes) != 0 ||
        (size_t)n > SIZE_MAX / (2 * sizeof *scratch_2n))
        return -1;
    scratch_bytes = (size_t)n * 2 * sizeof *scratch_2n;
    if (byte_ranges_overlap(out, bytes, scratch_2n, scratch_bytes) ||
        byte_ranges_overlap(out, bytes, residual, bytes) ||
        byte_ranges_overlap(out, bytes, dense, bytes) ||
        byte_ranges_overlap(out, bytes, routed, bytes) ||
        byte_ranges_overlap(out, bytes, dense_post_weight, bytes) ||
        byte_ranges_overlap(out, bytes, routed_post_weight, bytes) ||
        byte_ranges_overlap(out, bytes, final_post_weight, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes, residual, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes, dense, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes, routed, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes,
                            dense_post_weight, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes,
                            routed_post_weight, bytes) ||
        byte_ranges_overlap(scratch_2n, scratch_bytes,
                            final_post_weight, bytes))
        return -1;
    for (int i = 0; i < n; i++)
        if (!isfinite(residual[i])) return -1;
    sum = scratch_2n;
    final = scratch_2n + n;
    if (salt_gemma4_rmsnorm(sum, dense, dense_post_weight,
                            n, eps, 1) != 0 ||
        salt_gemma4_rmsnorm(final, routed, routed_post_weight,
                            n, eps, 1) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        float combined = sum[i] + final[i];
        if (!isfinite(combined))
            return -1;
        sum[i] = combined;
    }
    if (salt_gemma4_rmsnorm(final, sum, final_post_weight,
                            n, eps, 1) != 0)
        return -1;
    for (int i = 0; i < n; i++) {
        float y = (residual[i] + final[i]) * layer_scalar;
        if (!isfinite(y))
            return -1;
    }
    for (int i = 0; i < n; i++)
        out[i] = (residual[i] + final[i]) * layer_scalar;
    return 0;
}

int salt_gemma4_softcap(float *logits, int n, float cap) {
    if (!logits || n < 1 || !(cap > 0.0f) || !isfinite(cap))
        return -1;
    for (int i = 0; i < n; i++)
        if (!isfinite(logits[i])) return -1;
    for (int i = 0; i < n; i++)
        logits[i] = cap * salt_tanhf(logits[i] / cap);
    return 0;
}

static int bf16_rope_pair(float a, float b, float c, float s,
                          float *out_a, float *out_b) {
    float ac, bs, bc, as;
    c = round_bf16(c);
    s = round_bf16(s);
    ac = bf16_mul(a, c);
    bs = bf16_mul(b, s);
    bc = bf16_mul(b, c);
    as = bf16_mul(a, s);
    *out_a = bf16_add(ac, -bs);
    *out_b = bf16_add(bc, as);
    return isfinite(*out_a) && isfinite(*out_b) ? 0 : -1;
}

int salt_gemma4_text_rope(float *head, int head_dim, int rope_dim,
                          int rope_base_dim, int position,
                          float rope_theta) {
    int half, active_pairs;
    if (!head || head_dim < 2 || head_dim % 2 != 0 ||
        rope_dim < 2 || rope_dim % 2 != 0 || rope_dim > head_dim ||
        rope_base_dim < rope_dim || rope_base_dim > head_dim ||
        position < 0 || !(rope_theta > 0.0f) || !isfinite(rope_theta))
        return -1;
    for (int i = 0; i < head_dim; i++)
        if (!isfinite(head[i]))
            return -1;
    half = head_dim / 2;
    active_pairs = rope_dim / 2;
    for (int j = 0; j < active_pairs; j++) {
        float exponent = (float)(2 * j) / (float)rope_base_dim;
        float angle = (float)position / salt_powf(rope_theta, exponent);
        float a, b;
        if (bf16_rope_pair(head[j], head[half + j],
                           salt_cosf(angle), salt_sinf(angle), &a, &b) != 0)
            return -1;
    }
    for (int i = 0; i < head_dim; i++)
        head[i] = round_bf16(head[i]);
    for (int j = 0; j < active_pairs; j++) {
        float exponent = (float)(2 * j) / (float)rope_base_dim;
        float angle = (float)position / salt_powf(rope_theta, exponent);
        float a, b;
        (void)bf16_rope_pair(head[j], head[half + j],
                             salt_cosf(angle), salt_sinf(angle), &a, &b);
        head[j] = a;
        head[half + j] = b;
    }
    return 0;
}

int salt_gemma4_text_rope_factors(float *head, int head_dim, int rope_dim,
                                  const float *cosines, const float *sines) {
    int half, active_pairs;
    if (!head || !cosines || !sines || head_dim < 2 || head_dim % 2 != 0 ||
        rope_dim < 2 || rope_dim % 2 != 0 || rope_dim > head_dim)
        return -1;
    for (int i = 0; i < head_dim; i++)
        if (!isfinite(head[i])) return -1;
    half = head_dim / 2;
    active_pairs = rope_dim / 2;
    for (int j = 0; j < active_pairs; j++) {
        float a, b;
        if (!isfinite(cosines[j]) || !isfinite(sines[j]) ||
            bf16_rope_pair(head[j], head[half + j],
                           cosines[j], sines[j], &a, &b) != 0)
            return -1;
    }
    for (int i = 0; i < head_dim; i++)
        head[i] = round_bf16(head[i]);
    for (int j = 0; j < active_pairs; j++) {
        float a, b;
        if (bf16_rope_pair(head[j], head[half + j],
                           cosines[j], sines[j], &a, &b) != 0)
            return -1;
        head[j] = a;
        head[half + j] = b;
    }
    return 0;
}

static int byte_ranges_overlap(const void *a, size_t a_size,
                               const void *b, size_t b_size) {
    uintptr_t ap, bp;
    if (!a || !b || a_size == 0 || b_size == 0)
        return 0;
    ap = (uintptr_t)a;
    bp = (uintptr_t)b;
    if (ap <= bp)
        return bp - ap < a_size;
    return ap - bp < b_size;
}

int salt_gemma4_attention_masks(unsigned char *full,
                                unsigned char *sliding,
                                size_t mask_count,
                                const unsigned char *mm_token_type,
                                const unsigned char *valid,
                                int n_tokens, int sliding_window,
                                int *block_ids, size_t block_count) {
    size_t n, count, block_bytes;
    int block = -1, previous_vision = 0;
    if (!full || !sliding || !mm_token_type || !valid || !block_ids ||
        n_tokens < 1 || sliding_window < 1)
        return -1;
    n = (size_t)n_tokens;
    if (n > SIZE_MAX / n || n > SIZE_MAX / sizeof *block_ids)
        return -1;
    count = n * n;
    block_bytes = n * sizeof *block_ids;
    if (mask_count < count || block_count < n ||
        byte_ranges_overlap(full, count, sliding, count) ||
        byte_ranges_overlap(full, count, mm_token_type, n) ||
        byte_ranges_overlap(full, count, valid, n) ||
        byte_ranges_overlap(full, count, block_ids, block_bytes) ||
        byte_ranges_overlap(sliding, count, mm_token_type, n) ||
        byte_ranges_overlap(sliding, count, valid, n) ||
        byte_ranges_overlap(sliding, count, block_ids, block_bytes) ||
        byte_ranges_overlap(block_ids, block_bytes, mm_token_type, n) ||
        byte_ranges_overlap(block_ids, block_bytes, valid, n))
        return -1;
    for (int i = 0; i < n_tokens; i++)
        if (valid[i] > 1)
            return -1;
    for (int i = 0; i < n_tokens; i++) {
        int vision = mm_token_type[i] == 1 || mm_token_type[i] == 2;
        if (vision && !previous_vision)
            block++;
        block_ids[i] = vision ? block : -1;
        previous_vision = vision;
    }
    for (int q = 0; q < n_tokens; q++) {
        for (int k = 0; k < n_tokens; k++) {
            size_t i = (size_t)q * n + (size_t)k;
            int both_valid = valid[q] && valid[k];
            int causal = k <= q;
            int same_block = block_ids[q] >= 0 &&
                             block_ids[q] == block_ids[k];
            int in_window = k > q - sliding_window;
            full[i] = (unsigned char)(both_valid && causal);
            sliding[i] = (unsigned char)(both_valid && in_window &&
                                         (causal || same_block));
        }
    }
    return 0;
}

int salt_gemma4_vision_normalize_pixels(float *pixels, int n) {
    if (!pixels || n < 1) return -1;
    for (int i = 0; i < n; i++)
        if (!isfinite(pixels[i])) return -1;
    for (int i = 0; i < n; i++)
        pixels[i] = round_bf16(2.0f * (pixels[i] - 0.5f));
    return 0;
}

int salt_gemma4_vision_add_positions(float *out, const float *patches,
                                     const float *position_table,
                                     const int *positions_xy,
                                     const unsigned char *padding,
                                     int n_patches, int hidden,
                                     int position_embedding_size) {
    size_t count, out_bytes, plane, table_bytes;
    if (!out || !patches || !position_table || !positions_xy || !padding ||
        n_patches < 1 || n_patches > INT_MAX / 2 || hidden < 1 ||
        position_embedding_size < 1 ||
        (size_t)n_patches > SIZE_MAX / (size_t)hidden ||
        (size_t)position_embedding_size > SIZE_MAX / (size_t)hidden)
        return -1;
    plane = (size_t)position_embedding_size * (size_t)hidden;
    count = (size_t)n_patches * (size_t)hidden;
    if (count > SIZE_MAX / sizeof *out ||
        plane > SIZE_MAX / (2 * sizeof *position_table))
        return -1;
    out_bytes = count * sizeof *out;
    table_bytes = plane * 2 * sizeof *position_table;
    if (disallowed_overlap(out, out_bytes, patches,
                           count * sizeof *patches, 1) ||
        byte_ranges_overlap(out, out_bytes, position_table, table_bytes) ||
        byte_ranges_overlap(out, out_bytes, positions_xy,
                            (size_t)n_patches * 2 * sizeof *positions_xy) ||
        byte_ranges_overlap(out, out_bytes, padding, (size_t)n_patches))
        return -1;
    for (int p = 0; p < n_patches; p++) {
        int x = positions_xy[2 * p];
        int y = positions_xy[2 * p + 1];
        if (padding[p] > 1)
            return -1;
        if (!padding[p] &&
            (x < 0 || y < 0 || x >= position_embedding_size ||
             y >= position_embedding_size))
            return -1;
        for (int h = 0; h < hidden; h++) {
            size_t i = (size_t)p * (size_t)hidden + (size_t)h;
            if (!isfinite(patches[i])) return -1;
            if (!padding[p] &&
                (!isfinite(position_table[(size_t)x * hidden + h]) ||
                 !isfinite(position_table[plane + (size_t)y * hidden + h])))
                return -1;
        }
    }
    for (int p = 0; p < n_patches; p++) {
        int x = positions_xy[2 * p];
        int y = positions_xy[2 * p + 1];
        for (int h = 0; h < hidden; h++) {
            size_t i = (size_t)p * (size_t)hidden + (size_t)h;
            float value;
            if (!padding[p]) {
                float position = bf16_add(
                    position_table[(size_t)x * hidden + h],
                    position_table[plane + (size_t)y * hidden + h]);
                value = bf16_add(patches[i], position);
            } else
                value = bf16_add(patches[i], 0.0f);
            out[i] = value;
        }
    }
    return 0;
}

int salt_gemma4_vision_rope(float *head, int head_dim,
                            int position_x, int position_y,
                            float rope_theta) {
    int half, quarter;
    if (!head || head_dim < 4 || head_dim % 4 != 0 ||
        !(rope_theta > 0.0f) || !isfinite(rope_theta))
        return -1;
    for (int i = 0; i < head_dim; i++)
        if (!isfinite(head[i])) return -1;
    half = head_dim / 2;
    quarter = half / 2;
    /* Upstream splits the head into contiguous x/y sections and applies
     * rotate_half independently inside each section. */
    for (int axis = 0; axis < 2; axis++) {
        int base = axis * half;
        float position = (float)(axis == 0 ? position_x : position_y);
        for (int j = 0; j < quarter; j++) {
            float exponent = (float)(2 * j) / (float)half;
            float angle = position / salt_powf(rope_theta, exponent);
            float a, b;
            if (bf16_rope_pair(head[base + j], head[base + quarter + j],
                               salt_cosf(angle), salt_sinf(angle), &a, &b) != 0)
                return -1;
        }
    }
    for (int i = 0; i < head_dim; i++)
        head[i] = round_bf16(head[i]);
    for (int axis = 0; axis < 2; axis++) {
        int base = axis * half;
        float position = (float)(axis == 0 ? position_x : position_y);
        for (int j = 0; j < quarter; j++) {
            float exponent = (float)(2 * j) / (float)half;
            float angle = position / salt_powf(rope_theta, exponent);
            float a, b;
            if (bf16_rope_pair(
                    head[base + j], head[base + quarter + j],
                    salt_cosf(angle), salt_sinf(angle), &a, &b) != 0)
                return -1;
            head[base + j] = a;
            head[base + quarter + j] = b;
        }
    }
    return 0;
}

int salt_gemma4_vision_pool(float *out, unsigned char *valid,
                            const float *hidden_states,
                            const int *positions_xy,
                            const unsigned char *padding,
                            int input_length, int output_length,
                            int hidden) {
    int ratio, kernel, max_x = 0;
    size_t input_count, output_count, grid_width;
    float root, weight;
    if (!out || !valid || !hidden_states || !positions_xy || !padding ||
        input_length < 1 || output_length < 1 || hidden < 1 ||
        input_length > INT_MAX / 2 || output_length > input_length ||
        input_length % output_length != 0 ||
        (size_t)input_length > SIZE_MAX / (size_t)hidden ||
        (size_t)output_length > SIZE_MAX / (size_t)hidden)
        return -1;
    input_count = (size_t)input_length * (size_t)hidden;
    output_count = (size_t)output_length * (size_t)hidden;
    if (input_count > SIZE_MAX / sizeof *hidden_states ||
        output_count > SIZE_MAX / sizeof *out ||
        byte_ranges_overlap(out, output_count * sizeof *out,
                            hidden_states, input_count * sizeof *hidden_states) ||
        byte_ranges_overlap(out, output_count * sizeof *out,
                            positions_xy,
                            (size_t)input_length * 2 * sizeof *positions_xy) ||
        byte_ranges_overlap(out, output_count * sizeof *out,
                            padding, (size_t)input_length) ||
        byte_ranges_overlap(valid, (size_t)output_length,
                            hidden_states, input_count * sizeof *hidden_states) ||
        byte_ranges_overlap(valid, (size_t)output_length,
                            positions_xy,
                            (size_t)input_length * 2 * sizeof *positions_xy) ||
        byte_ranges_overlap(valid, (size_t)output_length,
                            padding, (size_t)input_length) ||
        byte_ranges_overlap(out, output_count * sizeof *out,
                            valid, (size_t)output_length))
        return -1;
    ratio = input_length / output_length;
    kernel = (int)salt_sqrtf((float)ratio);
    if (kernel < 1 || kernel * kernel != ratio) return -1;
    weight = 1.0f / (float)ratio;
    for (int p = 0; p < input_length; p++) {
        int x = positions_xy[2 * p];
        int y = positions_xy[2 * p + 1];
        if (padding[p] > 1 || (!padding[p] && (x < 0 || y < 0)) ||
            x == INT_MAX)
            return -1;
        if (x < 0) x = 0;
        if (x + 1 > max_x) max_x = x + 1;
        for (int h = 0; h < hidden; h++)
            if (!isfinite(hidden_states[(size_t)p * hidden + h])) return -1;
    }
    grid_width = (size_t)max_x / (size_t)kernel;
    if (grid_width < 1) return -1;
    /* Validate every one-hot bucket before touching caller output. */
    for (int p = 0; p < input_length; p++) {
        int x = positions_xy[2 * p];
        int y = positions_xy[2 * p + 1];
        size_t bucket, row;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        row = (size_t)y / (size_t)kernel;
        if (row > SIZE_MAX / grid_width)
            return -1;
        bucket = (size_t)x / (size_t)kernel + grid_width * row;
        if (bucket >= (size_t)output_length) return -1;
    }
    memset(out, 0, output_count * sizeof *out);
    memset(valid, 0, (size_t)output_length * sizeof *valid);
    for (int p = 0; p < input_length; p++) {
        int x = positions_xy[2 * p];
        int y = positions_xy[2 * p + 1];
        size_t bucket;
        if (x < 0) x = 0;
        if (y < 0) y = 0;
        bucket = (size_t)x / (size_t)kernel +
                 grid_width * ((size_t)y / (size_t)kernel);
        valid[bucket] = 1;
        if (!padding[p]) {
            for (int h = 0; h < hidden; h++)
                out[bucket * (size_t)hidden + (size_t)h] +=
                    round_bf16(hidden_states[(size_t)p * hidden + h]) * weight;
        }
    }
    root = salt_sqrtf((float)hidden);
    for (size_t i = 0; i < output_count; i++)
        out[i] = round_bf16(out[i]) * root;
    return 0;
}

int salt_gemma4_vision_standardize(float *states, const float *bias,
                                   const float *scale, int tokens,
                                   int hidden) {
    size_t n;
    if (!states || !bias || !scale || tokens < 1 || hidden < 1 ||
        (size_t)tokens > SIZE_MAX / (size_t)hidden)
        return -1;
    n = (size_t)tokens * (size_t)hidden;
    for (int h = 0; h < hidden; h++)
        if (!isfinite(bias[h]) || !isfinite(scale[h])) return -1;
    for (size_t i = 0; i < n; i++)
        if (!isfinite(states[i])) return -1;
    for (int t = 0; t < tokens; t++)
        for (int h = 0; h < hidden; h++) {
            size_t i = (size_t)t * hidden + h;
            float centered = states[i] - bias[h];
            float value = centered * scale[h];
            if (!isfinite(value))
                return -1;
        }
    for (int t = 0; t < tokens; t++)
        for (int h = 0; h < hidden; h++) {
            size_t i = (size_t)t * hidden + h;
            states[i] = round_bf16((states[i] - bias[h]) * scale[h]);
        }
    return 0;
}

int salt_gemma4_vision_prepare_projection(float *out,
                                          const float *vision_states,
                                          int tokens, int hidden, float eps) {
    size_t count, bytes;
    if (!out || !vision_states || tokens < 1 || hidden < 1 ||
        (size_t)tokens > SIZE_MAX / (size_t)hidden)
        return -1;
    count = (size_t)tokens * (size_t)hidden;
    if (count > SIZE_MAX / sizeof *out)
        return -1;
    bytes = count * sizeof *out;
    if (disallowed_overlap(out, bytes, vision_states, bytes, 1))
        return -1;
    for (size_t i = 0; i < count; i++)
        if (!isfinite(vision_states[i])) return -1;
    for (int t = 0; t < tokens; t++) {
        float inv;
        if (rmsnorm_inverse(vision_states + (size_t)t * hidden, NULL,
                            hidden, eps, 0, &inv) != 0)
            return -1;
    }
    for (int t = 0; t < tokens; t++)
        if (salt_gemma4_rmsnorm(out + (size_t)t * hidden,
                                vision_states + (size_t)t * hidden,
                                NULL, hidden, eps, 0) != 0)
            return -1;
    for (size_t i = 0; i < count; i++)
        out[i] = round_bf16(out[i]);
    return 0;
}

int salt_gemma4_scatter_image_features(float *text_embeddings,
                                       const int *input_ids,
                                       int sequence_length, int text_hidden,
                                       int image_token_id,
                                       const float *image_features,
                                       int image_tokens) {
    int slots = 0, feature = 0;
    size_t feature_count, feature_bytes, id_bytes, text_count, text_bytes;
    if (!text_embeddings || !input_ids || sequence_length < 0 ||
        text_hidden < 1 || image_tokens < 0 ||
        (size_t)sequence_length > SIZE_MAX / (size_t)text_hidden ||
        (size_t)image_tokens > SIZE_MAX / (size_t)text_hidden)
        return -1;
    text_count = (size_t)sequence_length * (size_t)text_hidden;
    feature_count = (size_t)image_tokens * (size_t)text_hidden;
    if (text_count > SIZE_MAX / sizeof *text_embeddings ||
        feature_count > SIZE_MAX / sizeof *image_features ||
        (size_t)sequence_length > SIZE_MAX / sizeof *input_ids)
        return -1;
    text_bytes = text_count * sizeof *text_embeddings;
    feature_bytes = feature_count * sizeof *image_features;
    id_bytes = (size_t)sequence_length * sizeof *input_ids;
    if (byte_ranges_overlap(text_embeddings, text_bytes,
                            input_ids, id_bytes) ||
        (image_tokens > 0 &&
         byte_ranges_overlap(text_embeddings, text_bytes,
                             image_features, feature_bytes)))
        return -1;
    for (int i = 0; i < sequence_length; i++)
        if (input_ids[i] == image_token_id) slots++;
    if (slots != image_tokens || (image_tokens > 0 && !image_features))
        return -1;
    for (size_t i = 0; i < feature_count; i++)
        if (!isfinite(image_features[i])) return -1;
    for (int i = 0; i < sequence_length; i++) {
        if (input_ids[i] != image_token_id) continue;
        memcpy(text_embeddings + (size_t)i * text_hidden,
               image_features + (size_t)feature * text_hidden,
               (size_t)text_hidden * sizeof *text_embeddings);
        feature++;
    }
    return 0;
}
