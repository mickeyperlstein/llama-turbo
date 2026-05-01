#include "turboq.h"
#include "codebook.h"
#include "hd3.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define QK_TBQ4_0 256
#define QK_TBQ3_0 256

static float f16_to_f32(ggml_half h) {
    uint32_t s   = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t m   = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0x1f) {
        bits = (s << 31) | 0x7f800000u | (m << 13);
    } else if (exp == 0) {
        if (m == 0) { bits = s << 31; }
        else {
            exp = 1;
            while (!(m & 0x400u)) { m <<= 1; exp--; }
            bits = (s << 31) | ((exp + 112u) << 23) | ((m & 0x3ffu) << 13);
        }
    } else {
        bits = (s << 31) | ((exp + 112u) << 23) | (m << 13);
    }
    float f; memcpy(&f, &bits, 4); return f;
}

static ggml_half f32_to_f16(float x) {
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s   = u >> 31;
    uint32_t exp = (u >> 23) & 0xffu;
    uint32_t m   = u & 0x7fffffu;
    uint16_t h;
    if (exp == 0xff) {
        h = (uint16_t)((s << 15) | 0x7c00u | (m ? 0x200u : 0u));
    } else if (exp <= 112) {
        h = (uint16_t)(s << 15);
    } else if (exp >= 143) {
        h = (uint16_t)((s << 15) | 0x7c00u);
    } else {
        uint32_t he = exp - 112u;
        uint32_t hm = (m + 0x1000u) >> 13;
        if (hm & 0x400u) { he++; hm = 0; }
        h = (uint16_t)((s << 15) | (he << 10) | (hm & 0x3ffu));
    }
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

    // Scale to N(0,1) so Lloyd-Max centroids apply
    float scale = sqrtf((float)n);
    for (size_t i = 0; i < n; i++) buf[i] *= scale;

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

        uint64_t bit_buf = 0;
        int bits_in_buf = 0;
        size_t byte_idx = 0;

        for (size_t i = 0; i < QK_TBQ3_0; i++) {
            int idx = find_nearest_centroid_3bit(buf[i], TURBOQ_CENTROIDS_3BIT);
            bit_buf |= (uint64_t)(idx & 0x7) << bits_in_buf;
            bits_in_buf += 3;
            if (bits_in_buf >= 8) {
                block->qs[byte_idx++] = (uint8_t)(bit_buf & 0xff);
                bit_buf >>= 8;
                bits_in_buf -= 8;
            }
        }
        if (bits_in_buf > 0 && byte_idx < 96)
            block->qs[byte_idx] = (uint8_t)(bit_buf & 0xff);
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

        // Undo the sqrt(n) scale applied before quantization
        float inv_scale = 1.0f / sqrtf((float)n);
        for (size_t i = 0; i < n; i++) buf[i] *= inv_scale;

        hd3_rotate_inverse(buf, n, layer_idx, head_idx);

        for (size_t i = 0; i < n; i++) {
            y[i] = f32_to_f16(buf[i] * norm);
        }
    } else {
        const block_tbq3_0 *block = (const block_tbq3_0 *)x;
        float norm = f16_to_f32(block->d);

        uint64_t bit_buf = 0;
        int bits_in_buf = 0;
        size_t byte_idx = 0;

        for (size_t i = 0; i < QK_TBQ3_0; i++) {
            while (bits_in_buf < 3 && byte_idx < 96) {
                bit_buf |= (uint64_t)block->qs[byte_idx++] << bits_in_buf;
                bits_in_buf += 8;
            }
            buf[i] = TURBOQ_CENTROIDS_3BIT[bit_buf & 0x7];
            bit_buf >>= 3;
            bits_in_buf -= 3;
        }

        // Undo the sqrt(n) scale applied before quantization
        float inv_scale = 1.0f / sqrtf((float)n);
        for (size_t i = 0; i < n; i++) buf[i] *= inv_scale;

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
