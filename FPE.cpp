#include "FPE.h"

// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
static uint32 pcg_hash(uint32 input)
{
    uint32 state = input * 747796405u + 2891336453u;
    uint32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

static uint32 NumBits(uint32 value)
{
    uint32 ret = 0;
    while (value)
    {
        ret++;
        value /= 2;
    }
    return ret;
}

uint32 FPE_Encrypt(uint32 data, uint32 key, uint32 numItems, uint32 numRounds)
{
    // left bits <= right bits
    const uint32 c_numBits = NumBits(numItems - 1);
    const uint32 c_numLeftBits = c_numBits / 2;
    const uint32 c_numRightBits = c_numBits - c_numLeftBits;
    const uint32 c_leftMask = (1 << c_numLeftBits) - 1;
    const uint32 c_rightMask = (1 << c_numRightBits) - 1;

    uint32 ret = data;
    for (uint32 roundIndex = 0; roundIndex < numRounds; ++roundIndex)
    {
        // Break the data into 2 pieces
        uint32 left = (ret >> c_numRightBits);
        uint32 right = ret & c_rightMask;

        // swap them, and make the new left (the >= sized half) incorporate a hash of the left, and also the key
        uint32 newLeft = (right ^ pcg_hash(left ^ key)) & c_rightMask;
        uint32 newRight = left;

        // put the data back together for the next round
        ret = (newLeft << c_numLeftBits) | newRight;
    }
    return ret;
}

uint32 FPE_Decrypt(uint32 data, uint32 key, uint32 numItems, uint32 numRounds)
{
    // left bits <= right bits
    const uint32 c_numBits = NumBits(numItems - 1);
    const uint32 c_numLeftBits = c_numBits / 2;
    const uint32 c_numRightBits = c_numBits - c_numLeftBits;
    const uint32 c_leftMask = (1 << c_numLeftBits) - 1;
    const uint32 c_rightMask = (1 << c_numRightBits) - 1;

    uint32 ret = data;
    for (uint32 roundIndex = 0; roundIndex < numRounds; ++roundIndex)
    {
        // Break the data into 2 pieces.
        // Left and right are reversed because we are doing the encryption process in reverse.
        uint32 newLeft = (ret >> c_numLeftBits);
        uint32 newRight = ret & c_leftMask;

        // swap them, and make the right (the >= sized half) incorporate a hash of the left, and also the key
        uint32 left = newRight;
        uint32 right = (newLeft ^ pcg_hash(newRight ^ key)) & c_rightMask;

        // put the data back together for the next round
        ret = (left << c_numRightBits) | right;
    }
    return ret;
}
