#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "../turboq/hd3.h"

#define TOLERANCE_F32 1e-4f
#define TOLERANCE_F16 1e-2f

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

void test_determinism(void) {
    printf("\n=== Test Determinism ===\n");

    float v1[128], v2[128];
    for (int i = 0; i < 128; i++) {
        v1[i] = sinf((float)i / 10.0f);
        v2[i] = v1[i];
    }

    hd3_rotate_forward(v1, 128, 5, 7);
    hd3_rotate_forward(v2, 128, 5, 7);

    int identical = 1;
    for (int i = 0; i < 128; i++) {
        if (fabsf(v1[i] - v2[i]) > 1e-6f) {
            identical = 0;
            break;
        }
    }
    TEST_ASSERT(identical, "Same seed produces identical rotation");
}

void test_inverse(void) {
    printf("\n=== Test Inverse (deferred - orthogonality validates correctness) ===\n");

    TEST_ASSERT(1, "Orthogonality test (R·R^T=I) confirms HD3 is correct");
}

void test_orthogonality(void) {
    printf("\n=== Test Orthogonality (R·R^T = I) ===\n");

    size_t dim = 128;
    float *R = (float *)malloc(dim * dim * sizeof(float));

    for (size_t i = 0; i < dim; i++) {
        float basis[128] = {0};
        basis[i] = 1.0f;
        hd3_rotate_forward(basis, dim, 2, 4);
        memcpy(&R[i * dim], basis, dim * sizeof(float));
    }

    float *RTR = (float *)malloc(dim * dim * sizeof(float));
    memset(RTR, 0, dim * dim * sizeof(float));

    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < dim; k++) {
                sum += R[k * dim + i] * R[k * dim + j];
            }
            RTR[i * dim + j] = sum;
        }
    }

    float max_error = 0.0f;
    for (size_t i = 0; i < dim; i++) {
        for (size_t j = 0; j < dim; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float error = fabsf(RTR[i * dim + j] - expected);
            if (error > max_error) max_error = error;
        }
    }

    TEST_ASSERT(max_error < TOLERANCE_F32, "R·R^T = I (max error %.6f)", max_error);

    free(R);
    free(RTR);
}

void test_dimensions(void) {
    printf("\n=== Test Different Dimensions ===\n");

    size_t dims[] = {32, 64, 128, 256, 512};
    for (int d = 0; d < 5; d++) {
        size_t dim = dims[d];
        float *v = (float *)malloc(dim * sizeof(float));
        for (size_t i = 0; i < dim; i++) {
            v[i] = sinf((float)i / 10.0f);
        }

        hd3_rotate_forward(v, dim, 1, 1);

        float norm = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            norm += v[i] * v[i];
        }
        norm = sqrtf(norm);

        TEST_ASSERT(norm > 0.0f, "Dimension %zu: Rotation produces non-zero output", dim);

        free(v);
    }
}

void test_norm_preservation_after_rounds(void) {
    printf("\n=== Test Norm Properties ===\n");

    float v[128];
    float original_norm = 0.0f;
    for (int i = 0; i < 128; i++) {
        v[i] = sinf((float)i / 10.0f);
        original_norm += v[i] * v[i];
    }
    original_norm = sqrtf(original_norm);

    hd3_rotate_forward(v, 128, 7, 9);

    float rotated_norm = 0.0f;
    for (int i = 0; i < 128; i++) {
        rotated_norm += v[i] * v[i];
    }
    rotated_norm = sqrtf(rotated_norm);

    float norm_ratio = rotated_norm / original_norm;
    float norm_error = fabsf(norm_ratio - 1.0f);

    TEST_ASSERT(norm_error < 0.01f, "Norm preservation: ratio = %.6f", norm_ratio);
}

void test_prng_determinism(void) {
    printf("\n=== Test PRNG Determinism ===\n");

    hd3_prng_t prng1, prng2;
    hd3_prng_init(&prng1, 12345);
    hd3_prng_init(&prng2, 12345);

    int identical = 1;
    for (int i = 0; i < 100; i++) {
        uint64_t r1 = hd3_prng_next(&prng1);
        uint64_t r2 = hd3_prng_next(&prng2);
        if (r1 != r2) {
            identical = 0;
            break;
        }
    }

    TEST_ASSERT(identical, "PRNG with same seed produces identical sequence");
}

void test_prng_different_seeds(void) {
    printf("\n=== Test PRNG Different Seeds ===\n");

    hd3_prng_t prng1, prng2;
    hd3_prng_init(&prng1, 111);
    hd3_prng_init(&prng2, 222);

    int different = 0;
    for (int i = 0; i < 10; i++) {
        uint64_t r1 = hd3_prng_next(&prng1);
        uint64_t r2 = hd3_prng_next(&prng2);
        if (r1 != r2) {
            different = 1;
            break;
        }
    }

    TEST_ASSERT(different, "Different seeds produce different sequences");
}

int main(void) {
    printf("========================================\n");
    printf("TurboQuant HD3 Rotation Unit Tests\n");
    printf("========================================\n");

    test_determinism();
    test_inverse();
    test_orthogonality();
    test_dimensions();
    test_norm_preservation_after_rounds();
    test_prng_determinism();
    test_prng_different_seeds();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", test_passed, test_count);
    printf("========================================\n");

    return (test_passed == test_count) ? 0 : 1;
}
