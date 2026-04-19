#pragma once

#include <stdint.h>
#include <stddef.h>

typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    uint8_t qs[128];
} block_tbq4_0;

typedef struct {
    ggml_half d;
    uint8_t qs[96];
} block_tbq3_0;

void turboq_encode_f16(const ggml_half *x, void *y, size_t n, int bit_width,
                       uint32_t layer_idx, uint32_t head_idx);

void turboq_decode_f16(const void *x, ggml_half *y, size_t n, int bit_width,
                       uint32_t layer_idx, uint32_t head_idx);

size_t turboq_encode_size(size_t n_values, int bit_width);
