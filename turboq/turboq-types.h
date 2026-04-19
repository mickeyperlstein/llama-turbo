#pragma once

#include <stdint.h>
#include <stddef.h>

typedef enum {
    TURBOQ_TYPE_TBQ4_0,
    TURBOQ_TYPE_TBQ3_0,
    TURBOQ_TYPE_COUNT
} turboq_type_t;

typedef struct {
    turboq_type_t type;
    const char *name;
    size_t block_size;           // bytes per block
    size_t qk;                   // quantization block size (values per block)
    uint8_t bits_per_elem;       // bits per element
    int is_quantized;
    void *encode_fn;
    void *decode_fn;
    void *size_fn;
} turboq_type_trait_t;

void turboq_register_type_traits(void);
const turboq_type_trait_t *turboq_get_type_traits(turboq_type_t type);
const char *turboq_type_name(turboq_type_t type);
