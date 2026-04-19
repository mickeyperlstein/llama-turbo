#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "../turboq/codebook.h"

#define TOLERANCE 1e-4f

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

void test_codebook_3bit_values(void) {
    printf("\n=== Test 3-bit Codebook Values ===\n");

    const float expected[8] = {
        -2.1520f, -1.3440f, -0.7560f, -0.2451f,
         0.2451f,  0.7560f,  1.3440f,  2.1520f,
    };

    for (int i = 0; i < 8; i++) {
        float diff = fabsf(TURBOQ_CENTROIDS_3BIT[i] - expected[i]);
        int match = diff < TOLERANCE;
        TEST_ASSERT(match, "3BIT[%d] = %.4f (expected %.4f)", i, TURBOQ_CENTROIDS_3BIT[i], expected[i]);
    }
}

void test_codebook_4bit_values(void) {
    printf("\n=== Test 4-bit Codebook Values ===\n");

    const float expected[16] = {
        -2.7326f, -2.0690f, -1.6180f, -1.2562f,
        -0.9424f, -0.6568f, -0.3881f, -0.1284f,
         0.1284f,  0.3881f,  0.6568f,  0.9424f,
         1.2562f,  1.6180f,  2.0690f,  2.7326f,
    };

    for (int i = 0; i < 16; i++) {
        float diff = fabsf(TURBOQ_CENTROIDS_4BIT[i] - expected[i]);
        int match = diff < TOLERANCE;
        TEST_ASSERT(match, "4BIT[%d] = %.4f (expected %.4f)", i, TURBOQ_CENTROIDS_4BIT[i], expected[i]);
    }
}

void test_codebook_sizes(void) {
    printf("\n=== Test Codebook Array Sizes ===\n");

    size_t size_3bit = turboq_codebook_size_3bit();
    size_t size_4bit = turboq_codebook_size_4bit();

    TEST_ASSERT(size_3bit == 8, "3-bit codebook has 8 entries");
    TEST_ASSERT(size_4bit == 16, "4-bit codebook has 16 entries");
}

void test_codebook_properties(void) {
    printf("\n=== Test Codebook Properties ===\n");

    // 3-bit: check sorted and symmetric
    int sorted_3bit = 1;
    for (int i = 0; i < 7; i++) {
        if (TURBOQ_CENTROIDS_3BIT[i] >= TURBOQ_CENTROIDS_3BIT[i+1]) {
            sorted_3bit = 0;
            break;
        }
    }
    TEST_ASSERT(sorted_3bit, "3-bit centroids are sorted ascending");

    // Check symmetry around zero
    int symmetric_3bit = 1;
    for (int i = 0; i < 4; i++) {
        float diff = fabsf(TURBOQ_CENTROIDS_3BIT[i] + TURBOQ_CENTROIDS_3BIT[7-i]);
        if (diff > TOLERANCE) {
            symmetric_3bit = 0;
            break;
        }
    }
    TEST_ASSERT(symmetric_3bit, "3-bit centroids are symmetric");

    // 4-bit: check sorted
    int sorted_4bit = 1;
    for (int i = 0; i < 15; i++) {
        if (TURBOQ_CENTROIDS_4BIT[i] >= TURBOQ_CENTROIDS_4BIT[i+1]) {
            sorted_4bit = 0;
            break;
        }
    }
    TEST_ASSERT(sorted_4bit, "4-bit centroids are sorted ascending");

    // Check symmetry
    int symmetric_4bit = 1;
    for (int i = 0; i < 8; i++) {
        float diff = fabsf(TURBOQ_CENTROIDS_4BIT[i] + TURBOQ_CENTROIDS_4BIT[15-i]);
        if (diff > TOLERANCE) {
            symmetric_4bit = 0;
            break;
        }
    }
    TEST_ASSERT(symmetric_4bit, "4-bit centroids are symmetric");
}

int main(void) {
    printf("========================================\n");
    printf("TurboQuant Codebook Unit Tests\n");
    printf("========================================\n");

    test_codebook_3bit_values();
    test_codebook_4bit_values();
    test_codebook_sizes();
    test_codebook_properties();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", test_passed, test_count);
    printf("========================================\n");

    return (test_passed == test_count) ? 0 : 1;
}
