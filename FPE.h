#pragma once

#include <stdint.h>

typedef uint32_t uint32;

// data is the item index to shuffle
// numItems must be a power of 2
// numRounds must be greater than 0
// key may be any value, including 0

uint32 FPE_Encrypt(uint32 data, uint32 key, uint32 numItems, uint32 numRounds);

uint32 FPE_Decrypt(uint32 data, uint32 key, uint32 numItems, uint32 numRounds);
