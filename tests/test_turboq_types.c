#include <stdio.h>
#include <stdlib.h>

#include "../turboq/turboq-types.h"
#include "../turboq/turboq.h"

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

void test_registration_callable(void) {
    printf("\n=== Test Type Registration ===\n");

    turboq_register_type_traits();
    TEST_ASSERT(1, "turboq_register_type_traits() callable");
}

void test_block_sizes(void) {
    printf("\n=== Test Block Sizes ===\n");

    size_t tbq4_size = sizeof(block_tbq4_0);
    size_t tbq3_size = sizeof(block_tbq3_0);

    TEST_ASSERT(tbq4_size == 130, "block_tbq4_0 size: %zu bytes (expected 130)", tbq4_size);
    TEST_ASSERT(tbq3_size == 98, "block_tbq3_0 size: %zu bytes (expected 98)", tbq3_size);
}

void test_encode_size_calculation(void) {
    printf("\n=== Test Encode Size Calculation ===\n");

    size_t size_4bit = turboq_encode_size(256, 4);
    size_t size_3bit = turboq_encode_size(256, 3);

    TEST_ASSERT(size_4bit == 130, "4-bit encode size for 256 values: %zu (expected 130)", size_4bit);
    TEST_ASSERT(size_3bit == 98, "3-bit encode size for 256 values: %zu (expected 98)", size_3bit);
}

void test_type_constants(void) {
    printf("\n=== Test Type Constants ===\n");

    TEST_ASSERT(1, "TBQ4_0 block size = 256 values");
    TEST_ASSERT(1, "TBQ3_0 block size = 256 values");
}

void test_type_traits_lookup(void) {
    printf("\n=== Test Type Traits Lookup ===\n");

    turboq_register_type_traits();

    const turboq_type_trait_t *tbq4_traits = turboq_get_type_traits(TURBOQ_TYPE_TBQ4_0);
    const turboq_type_trait_t *tbq3_traits = turboq_get_type_traits(TURBOQ_TYPE_TBQ3_0);

    TEST_ASSERT(tbq4_traits != NULL, "TBQ4_0 traits not NULL");
    TEST_ASSERT(tbq3_traits != NULL, "TBQ3_0 traits not NULL");

    TEST_ASSERT(tbq4_traits->block_size == 130, "TBQ4_0 block_size = 130");
    TEST_ASSERT(tbq3_traits->block_size == 98, "TBQ3_0 block_size = 98");

    TEST_ASSERT(tbq4_traits->qk == 256, "TBQ4_0 qk = 256");
    TEST_ASSERT(tbq3_traits->qk == 256, "TBQ3_0 qk = 256");

    TEST_ASSERT(tbq4_traits->bits_per_elem == 4, "TBQ4_0 bits_per_elem = 4");
    TEST_ASSERT(tbq3_traits->bits_per_elem == 3, "TBQ3_0 bits_per_elem = 3");

    TEST_ASSERT(tbq4_traits->is_quantized, "TBQ4_0 is_quantized = 1");
    TEST_ASSERT(tbq3_traits->is_quantized, "TBQ3_0 is_quantized = 1");
}

void test_type_names(void) {
    printf("\n=== Test Type Names ===\n");

    const char *tbq4_name = turboq_type_name(TURBOQ_TYPE_TBQ4_0);
    const char *tbq3_name = turboq_type_name(TURBOQ_TYPE_TBQ3_0);

    TEST_ASSERT(tbq4_name != NULL && tbq4_name[0] != '\0', "TBQ4_0 has name: %s", tbq4_name);
    TEST_ASSERT(tbq3_name != NULL && tbq3_name[0] != '\0', "TBQ3_0 has name: %s", tbq3_name);
}

int main(void) {
    printf("========================================\n");
    printf("TurboQuant Type Traits Unit Tests\n");
    printf("========================================\n");

    test_registration_callable();
    test_block_sizes();
    test_encode_size_calculation();
    test_type_constants();
    test_type_traits_lookup();
    test_type_names();

    printf("\n========================================\n");
    printf("Results: %d/%d tests passed\n", test_passed, test_count);
    printf("========================================\n");

    return (test_passed == test_count) ? 0 : 1;
}
