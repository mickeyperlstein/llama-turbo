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
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16; n |= n >> 32;
    return n + 1;
}

void hd3_walsh_hadamard_transform(float *v, size_t dim) {
    for (size_t h = 1; h < dim; h *= 2) {
        for (size_t i = 0; i < dim; i += h * 2) {
            for (size_t j = i; j < i + h; j++) {
                float a = v[j], b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
        }
    }
}

// NWHT = WHT / sqrt(n) — orthogonal, self-inverse (NWHT² = I)
static void nwht(float *v, size_t dim) {
    hd3_walsh_hadamard_transform(v, dim);
    float s = 1.0f / sqrtf((float)dim);
    for (size_t i = 0; i < dim; i++) v[i] *= s;
}

// Pack 1 sign bit per element into a compact array (0 = +1, 1 = -1)
static void gen_signs(uint8_t *out, size_t dim, hd3_prng_t *prng) {
    for (size_t i = 0; i < dim; ) {
        uint64_t r = hd3_prng_next(prng);
        for (int b = 0; b < 64 && i < dim; b++, i++)
            out[i] = (r >> b) & 1;
    }
}

static void apply_signs(float *v, const uint8_t *signs, size_t dim) {
    for (size_t i = 0; i < dim; i++)
        if (signs[i]) v[i] = -v[i];
}

// Forward: y = NWHT(D3(NWHT(D2(NWHT(D1(x))))))
// Each round: sign_r → NWHT
void hd3_rotate_forward(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx) {
    size_t pdim = next_power_of_2(dim);
    float   *buf  = (float *)  calloc(pdim, sizeof(float));
    uint8_t *s[3] = { (uint8_t *)malloc(pdim), (uint8_t *)malloc(pdim), (uint8_t *)malloc(pdim) };

    memcpy(buf, v, dim * sizeof(float));

    uint64_t seed = ((uint64_t)layer_idx << 32) | (uint64_t)head_idx;
    hd3_prng_t prng; hd3_prng_init(&prng, seed);
    for (int r = 0; r < 3; r++) gen_signs(s[r], pdim, &prng);

    for (int r = 0; r < 3; r++) {
        apply_signs(buf, s[r], pdim);
        nwht(buf, pdim);
    }

    memcpy(v, buf, dim * sizeof(float));
    free(buf); free(s[0]); free(s[1]); free(s[2]);
}

// Inverse: since NWHT is self-inverse (NWHT²=I) and each D_r is self-inverse (±1),
// inverse of round r (sign_r → NWHT) is (NWHT → sign_r), applied in reverse order r=2,1,0
void hd3_rotate_inverse(float *v, size_t dim, uint32_t layer_idx, uint32_t head_idx) {
    size_t pdim = next_power_of_2(dim);
    float   *buf  = (float *)  calloc(pdim, sizeof(float));
    uint8_t *s[3] = { (uint8_t *)malloc(pdim), (uint8_t *)malloc(pdim), (uint8_t *)malloc(pdim) };

    memcpy(buf, v, dim * sizeof(float));

    uint64_t seed = ((uint64_t)layer_idx << 32) | (uint64_t)head_idx;
    hd3_prng_t prng; hd3_prng_init(&prng, seed);
    for (int r = 0; r < 3; r++) gen_signs(s[r], pdim, &prng);

    for (int r = 2; r >= 0; r--) {
        nwht(buf, pdim);
        apply_signs(buf, s[r], pdim);
    }

    memcpy(v, buf, dim * sizeof(float));
    free(buf); free(s[0]); free(s[1]); free(s[2]);
}
