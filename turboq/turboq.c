#include "turboq.h"
#include "codebook.h"
#include "hd3.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define QK_TBQ4_0 256
#define QK_TBQ3_0 256

static float f16_to_f32(ggml_half h) {
    uint32_t w = ((uint32_t)h << 16) | 0x3c00;
    return *(float*)&w;
}

static ggml_half f32_to_f16(float x) {
    uint32_t u = *(uint32_t*)&x;
    uint16_t h = ((u >> 16) & 0x8000) | (((u >> 23) - 112) << 10) | ((u >> 13) & 0x3ff);
    return h;
}

static int find_nearest_centroid_4bit(float value, const float *centroids) {
    int best_idx = 0;
    float best_dist = fabsf(value - centroids[0]);
    for (int i = 1; i < 16; i++) {
        float dist = fabsf(value - centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

static int find_nearest_centroid_3bit(float value, const float *centroids) {
    int best_idx = 0;
    float best_dist = fabsf(value - centroids[0]);
    for (int i = 1; i < 8; i++) {
        float dist = fabsf(value - centroids[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = i;
        }
    }
    return best_idx;
}

void turboq_encode_f16(const ggml_half *x, void *y, size_t n, int bit_width,
                       uint32_t layer_idx, uint32_t head_idx) {
    if (bit_width != 4 && bit_width != 3) {
        return;
    }

    size_t padded_n = 1;
    while (padded_n < n) padded_n *= 2;

    float *buf = (float *)malloc(padded_n * sizeof(float));
    memset(buf, 0, padded_n * sizeof(float));

    float norm = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float v = f16_to_f32(x[i]);
        norm += v * v;
        buf[i] = v;
    }
    norm = sqrtf(norm);

    if (norm > 0.0f) {
        for (size_t i = 0; i < n; i++) {
            buf[i] /= norm;
        }
    }

    hd3_rotate_forward(buf, n, layer_idx, head_idx);

    if (bit_width == 4) {
        block_tbq4_0 *block = (block_tbq4_0 *)y;
        block->d = f32_to_f16(norm);
        memset(block->qs, 0, sizeof(block->qs));

        for (size_t i = 0; i < QK_TBQ4_0; i += 2) {
            int idx0 = find_nearest_centroid_4bit(buf[i], TURBOQ_CENTROIDS_4BIT);
            int idx1 = find_nearest_centroid_4bit(buf[i + 1], TURBOQ_CENTROIDS_4BIT);
            block->qs[i / 2] = (idx0 & 0xf) | ((idx1 & 0xf) << 4);
        }
    } else {
        block_tbq3_0 *block = (block_tbq3_0 *)y;
        block->d = f32_to_f16(norm);
        memset(block->qs, 0, sizeof(block->qs));

        size_t byte_idx = 0;
        uint8_t byte_val = 0;
        int bit_pos = 0;

        for (size_t i = 0; i < QK_TBQ3_0; i++) {
            int idx = find_nearest_centroid_3bit(buf[i], TURBOQ_CENTROIDS_3BIT);
            byte_val |= (idx & 0x7) << bit_pos;
            bit_pos += 3;
            if (bit_pos >= 8) {
                block->qs[byte_idx++] = byte_val;
                byte_val = 0;
                bit_pos = 0;
            }
        }
        if (bit_pos > 0) {
            block->qs[byte_idx] = byte_val;
        }
    }

    free(buf);
}

void turboq_decode_f16(const void *x, ggml_half *y, size_t n, int bit_width,
                       uint32_t layer_idx, uint32_t head_idx) {
    if (bit_width != 4 && bit_width != 3) {
        return;
    }

    size_t padded_n = 1;
    while (padded_n < n) padded_n *= 2;

    float *buf = (float *)malloc(padded_n * sizeof(float));
    memset(buf, 0, padded_n * sizeof(float));

    if (bit_width == 4) {
        const block_tbq4_0 *block = (const block_tbq4_0 *)x;
        float norm = f16_to_f32(block->d);

        for (size_t i = 0; i < QK_TBQ4_0; i += 2) {
            uint8_t byte = block->qs[i / 2];
            int idx0 = byte & 0xf;
            int idx1 = (byte >> 4) & 0xf;
            buf[i] = TURBOQ_CENTROIDS_4BIT[idx0];
            buf[i + 1] = TURBOQ_CENTROIDS_4BIT[idx1];
        }

        hd3_rotate_inverse(buf, n, layer_idx, head_idx);

        for (size_t i = 0; i < n; i++) {
            y[i] = f32_to_f16(buf[i] * norm);
        }
    } else {
        const block_tbq3_0 *block = (const block_tbq3_0 *)x;
        float norm = f16_to_f32(block->d);

        size_t byte_idx = 0;
        uint8_t byte_val = block->qs[0];
        int bit_pos = 0;

        for (size_t i = 0; i < QK_TBQ3_0; i++) {
            int idx = (byte_val >> bit_pos) & 0x7;
            buf[i] = TURBOQ_CENTROIDS_3BIT[idx];
            bit_pos += 3;
            if (bit_pos >= 8) {
                byte_idx++;
                byte_val = (byte_idx < 96) ? block->qs[byte_idx] : 0;
                bit_pos = 0;
            }
        }

        hd3_rotate_inverse(buf, n, layer_idx, head_idx);

        for (size_t i = 0; i < n; i++) {
            y[i] = f32_to_f16(buf[i] * norm);
        }
    }

    free(buf);
}

size_t turboq_encode_size(size_t n_values, int bit_width) {
    if (bit_width == 4) {
        return (n_values / QK_TBQ4_0) * sizeof(block_tbq4_0);
    } else if (bit_width == 3) {
        return (n_values / QK_TBQ3_0) * sizeof(block_tbq3_0);
    }
    return 0;
}
