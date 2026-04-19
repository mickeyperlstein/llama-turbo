#include "hd3.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void hd3_prng_init(hd3_prng_t *prng, uint64_t seed) {
    prng->state = seed;
}

uint64_t hd3_prng_next(hd3_prng_t *prng) {
    return splitmix64(&prng->state);
}

static size_t next_power_of_2(size_t n) {
    if (n == 0) return 1;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

void hd3_walsh_hadamard_transform(float *v, size_t dim) {
    for (size_t h = 1; h < dim; h *= 2) {
        for (size_t i = 0; i < dim; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                float a = v[j];
                float b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
}

static void apply_random_signs(float *v, size_t dim, hd3_prng_t *prng) {
    for (size_t i = 0; i < dim; i++) {
        uint64_t rand = hd3_prng_next(prng);
        if (rand & 1) {
            v[i] = -v[i];
        }
    }
}

static void normalize_vector(float *v, size_t dim, float scale) {
    for (size_t i = 0; i < dim; i++) {
        v[i] *= scale;
    }
}

void hd3_rotate_forward(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx) {
    size_t padded_dim = next_power_of_2(dim);
    float *buffer = (float *)malloc(padded_dim * sizeof(float));
    memset(buffer, 0, padded_dim * sizeof(float));
    memcpy(buffer, v, dim * sizeof(float));

    uint64_t seed = ((uint64_t)layer_idx << 32) | head_idx;
    hd3_prng_t prng;
    hd3_prng_init(&prng, seed);

    for (int round = 0; round < 3; round++) {
        apply_random_signs(buffer, padded_dim, &prng);
        hd3_walsh_hadamard_transform(buffer, padded_dim);
    }

    float final_scale = 1.0f / powf((float)padded_dim, 1.5f);
    normalize_vector(buffer, padded_dim, final_scale);

    memcpy(v, buffer, dim * sizeof(float));
    free(buffer);
}

void hd3_rotate_inverse(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx) {
    size_t padded_dim = next_power_of_2(dim);
    float *buffer = (float *)malloc(padded_dim * sizeof(float));
    memset(buffer, 0, padded_dim * sizeof(float));
    memcpy(buffer, v, dim * sizeof(float));

    float final_inv_scale = powf((float)padded_dim, 1.5f);
    normalize_vector(buffer, padded_dim, final_inv_scale);

    uint64_t seed = ((uint64_t)layer_idx << 32) | head_idx;
    hd3_prng_t prng;
    hd3_prng_init(&prng, seed);

    for (int round = 0; round < 3; round++) {
        hd3_walsh_hadamard_transform(buffer, padded_dim);
        apply_random_signs(buffer, padded_dim, &prng);
    }

    memcpy(v, buffer, dim * sizeof(float));
    free(buffer);
}
