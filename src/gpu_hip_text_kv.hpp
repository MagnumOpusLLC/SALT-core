#ifndef SALT_GPU_HIP_TEXT_KV_HPP
#define SALT_GPU_HIP_TEXT_KV_HPP

#include "salt/text_verify.h"

#include <stddef.h>
#include <stdint.h>

__device__ static const float *cuda_text_frontier_row(
        const float *tentative, const uint32_t *parents,
        const uint32_t *depths, uint32_t rows, uint32_t current,
        uint32_t target_depth, uint32_t stride) {
    uint32_t row = current, steps = 0u;
    if (!tentative || !parents || !depths || current >= rows) return NULL;
    while (depths[row] > target_depth) {
        if (++steps > rows) return NULL;
        row = parents[row];
        if (row == UINT32_MAX || row >= rows) return NULL;
    }
    if (depths[row] != target_depth) return NULL;
    return tentative + (size_t)row * stride;
}

__device__ static const float *cuda_text_committed_row(
        const float *rows, uint32_t position, uint32_t source_position,
        uint32_t width, uint32_t mode, uint32_t capacity) {
    uint32_t index;
    if (!rows || width == 0u || capacity == 0u ||
        position >= source_position)
        return NULL;
    if (mode == SALT_TEXT_KV_PRIVATE_ABSOLUTE) {
        if (position >= capacity) return NULL;
        index = position;
    } else if (mode == SALT_TEXT_KV_PRIVATE_RING) {
        uint32_t oldest = source_position > capacity
            ? source_position - capacity : 0u;
        if (position < oldest) return NULL;
        index = position % capacity;
    } else {
        return NULL;
    }
    return rows + (size_t)index * width;
}

#endif /* SALT_GPU_HIP_TEXT_KV_HPP */
