#include "codebook.h"

const float TURBOQ_CENTROIDS_3BIT[8] = {
    -2.1520f,
    -1.3440f,
    -0.7560f,
    -0.2451f,
     0.2451f,
     0.7560f,
     1.3440f,
     2.1520f,
};

const float TURBOQ_CENTROIDS_4BIT[16] = {
    -2.7326f,
    -2.0690f,
    -1.6180f,
    -1.2562f,
    -0.9424f,
    -0.6568f,
    -0.3881f,
    -0.1284f,
     0.1284f,
     0.3881f,
     0.6568f,
     0.9424f,
     1.2562f,
     1.6180f,
     2.0690f,
     2.7326f,
};

size_t turboq_codebook_size_3bit(void) {
    return sizeof(TURBOQ_CENTROIDS_3BIT) / sizeof(float);
}

size_t turboq_codebook_size_4bit(void) {
    return sizeof(TURBOQ_CENTROIDS_4BIT) / sizeof(float);
}
