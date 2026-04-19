#include "turboq-types.h"
#include "turboq.h"
#include <stddef.h>

static int types_registered = 0;
static turboq_type_trait_t type_traits[TURBOQ_TYPE_COUNT];

void turboq_register_type_traits(void) {
    if (types_registered) return;

    type_traits[TURBOQ_TYPE_TBQ4_0] = (turboq_type_trait_t){
        .type = TURBOQ_TYPE_TBQ4_0,
        .name = "TBQ4_0",
        .block_size = sizeof(block_tbq4_0),
        .qk = 256,
        .bits_per_elem = 4,
        .is_quantized = 1,
        .encode_fn = (void *)turboq_encode_f16,
        .decode_fn = (void *)turboq_decode_f16,
        .size_fn = (void *)turboq_encode_size,
    };

    type_traits[TURBOQ_TYPE_TBQ3_0] = (turboq_type_trait_t){
        .type = TURBOQ_TYPE_TBQ3_0,
        .name = "TBQ3_0",
        .block_size = sizeof(block_tbq3_0),
        .qk = 256,
        .bits_per_elem = 3,
        .is_quantized = 1,
        .encode_fn = (void *)turboq_encode_f16,
        .decode_fn = (void *)turboq_decode_f16,
        .size_fn = (void *)turboq_encode_size,
    };

    types_registered = 1;
}

const turboq_type_trait_t *turboq_get_type_traits(turboq_type_t type) {
    turboq_register_type_traits();
    if (type >= TURBOQ_TYPE_COUNT) return NULL;
    return &type_traits[type];
}

const char *turboq_type_name(turboq_type_t type) {
    const turboq_type_trait_t *traits = turboq_get_type_traits(type);
    return traits ? traits->name : "unknown";
}
