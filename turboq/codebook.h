#pragma once

#include <stddef.h>

extern const float TURBOQ_CENTROIDS_3BIT[8];
extern const float TURBOQ_CENTROIDS_4BIT[16];

size_t turboq_codebook_size_3bit(void);
size_t turboq_codebook_size_4bit(void);
