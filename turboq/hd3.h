#pragma once

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint64_t state;
} hd3_prng_t;

void hd3_prng_init(hd3_prng_t *prng, uint64_t seed);
uint64_t hd3_prng_next(hd3_prng_t *prng);

void hd3_rotate_forward(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx);
void hd3_rotate_inverse(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx);

void hd3_walsh_hadamard_transform(float *v, size_t dim);
