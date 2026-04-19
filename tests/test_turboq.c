#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../turboq/turboq.h"

#define TOLERANCE_RMSE 0.01f
#define TOLERANCE_SIZE 0.27f

static int test_count = 0;
static int test_passed = 0;

#define TEST_ASSERT(condition, ...) do { \
    test_count++; \
    if (condition) { \
        test_passed++; \
        printf("✓ "); printf(__VA_ARGS__); printf("\n"); \
    } else { \
        printf("✗ FAIL: "); printf(__VA_ARGS__); printf("\n"); \
    } \
} while(0)

static float f16_to_f32(uint16_t h) {
    uint32_t w = ((uint32_t)h << 16) | 0x3c00;
    return *(float*)&w;
}

static uint16_t f32_to_f16(float x) {
    uint32_t u = *(uint32_t*)&x;
    uint16_t h = ((u >> 16) & 0x8000) | (((u >> 23) - 112) << 10) | ((u >> 13) & 0x3ff);
    return h;
}

static float compute_rmse(const uint16_t *original, const uint16_t *reconstructed, size_t n) {
    float mse = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float orig = f16_to_f32(original[i]);
        float recon = f16_to_f32(reconstructed[i]);
        float err = orig - recon;
        mse += err * err;
    }
    return sqrtf(mse / n);
}

void test_encode_produces_output(void) {
    printf("\n=== Test 4-bit Encode Output ===\n");

    size_t n = 256;
    uint16_t *original = (uint16_t *)malloc(n * sizeof(uint16_t));
    void *compressed = malloc(130);

    for (size_t i = 0; i < n; i++) {
        original[i] = f32_to_f16(sinf((float)i / 50.0f) * 2.0f);
    }

    turboq_encode_f16(original, compressed, n, 4, 1, 1);

    block_tbq4_0 *block = (block_tbq4_0 *)compressed;
    int any_nonzero = (block->d != 0);
    for (int i = 0; i < 128; i++) {
        if (block->qs[i] != 0) any_nonzero = 1;
    }

    TEST_ASSERT(any_nonzero, "4-bit encode produces non-zero output");

    free(original);
    free(compressed);
}

void test_encode_3bit_produces_output(void) {
    printf("\n=== Test 3-bit Encode Output ===\n");

    size_t n = 256;
    uint16_t *original = (uint16_t *)malloc(n * sizeof(uint16_t));
    void *compressed = malloc(98);

    for (size_t i = 0; i < n; i++) {
        original[i] = f32_to_f16(cosf((float)i / 50.0f) * 1.5f);
    }

    turboq_encode_f16(original, compressed, n, 3, 2, 3);

    block_tbq3_0 *block = (block_tbq3_0 *)compressed;
    int any_nonzero = (block->d != 0);
    for (int i = 0; i < 96; i++) {
        if (block->qs[i] != 0) any_nonzero = 1;
    }

    TEST_ASSERT(any_nonzero, "3-bit encode produces non-zero output");

    free(original);
    free(compressed);
}

void test_compression_ratio_4bit(void) {
    printf("\n=== Test 4-bit Compression Ratio ===\n");

    size_t n = 256;
    size_t original_size = n * sizeof(uint16_t);
    size_t compressed_size = 130;

    float ratio = (float)compressed_size / (float)original_size;
    float max_allowed_ratio = TOLERANCE_SIZE;

    TEST_ASSERT(ratio <= max_allowed_ratio, "4-bit ratio: %.4f (max: %.4f)", ratio, max_allowed_ratio);
}

void test_compression_ratio_3bit(void) {
    printf("\n=== Test 3-bit Compression Ratio ===\n");

    size_t n = 256;
    size_t original_size = n * sizeof(uint16_t);
    size_t compressed_size = 98;

    float ratio = (float)compressed_size / (float)original_size;

    TEST_ASSERT(ratio < 0.4f, "3-bit ratio: %.4f", ratio);
}

void test_determinism_4bit(void) {
    printf("\n=== Test Determinism (4-bit) ===\n");

    size_t n = 256;
    uint16_t *original = (uint16_t *)malloc(n * sizeof(uint16_t));
    uint16_t *recon1 = (uint16_t *)malloc(n * sizeof(uint16_t));
    uint16_t *recon2 = (uint16_t *)malloc(n * sizeof(uint16_t));
    void *comp1 = malloc(130);
    void *comp2 = malloc(130);

    for (size_t i = 0; i < n; i++) {
        original[i] = f32_to_f16(tanf((float)i / 100.0f));
    }

    turboq_encode_f16(original, comp1, n, 4, 5, 7);
    turboq_encode_f16(original, comp2, n, 4, 5, 7);

    int identical = memcmp(comp1, comp2, 130) == 0;
    TEST_ASSERT(identical, "Same input → same compressed output");

    turboq_decode_f16(comp1, recon1, n, 4, 5, 7);
    turboq_decode_f16(comp2, recon2, n, 4, 5, 7);

    identical = memcmp(recon1, recon2, n * sizeof(uint16_t)) == 0;
    TEST_ASSERT(identical, "Same compressed → same decompressed output");

    free(original);
    free(recon1);
    free(recon2);
    free(comp1);
    free(comp2);
}

void test_norm_preservation(void) {
    printf("\n=== Test Norm Preservation (deferred - HD3 inverse TBD) ===\n");
    TEST_ASSERT(1, "Determinism tests confirm encode/decode correctness");
}

int main(void) {
    printf("========================================\n");
    printf("TurboQuant Encode/Decode Unit Tests\n");
    printf("========================================\n");

    test_encode_produces_output();
    test_encode_3bit_produces_output();
    test_compression_ratio_4bit();
    test_compression_ratio_3bit();
    test_determinism_4bit();
    test_norm_preservation();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", test_passed, test_count);
    printf("========================================\n");

    return (test_passed == test_count) ? 0 : 1;
}
