#include "bucket_counter.h"

#include<stdlib.h>
#include<string.h>

static uint64_t *buk; // buckets
static uint64_t size = 0; // number of buckets
static uint64_t cap = 0; // capacity of buckets

void bukAdd(uint64_t idx)
{
    // if access out-of-bounds, expand the array
    if (idx >= cap) {
        uint64_t *newBuk;
        uint64_t newCap = cap * 2 + 1;

        // allocate the new array and copy from the original
        while (idx >= newCap) newCap *= 2;
        newBuk = (uint64_t*)calloc(newCap, sizeof(uint64_t));
        memcpy(newBuk, buk, sizeof(uint64_t)*size);

        // free old array and switch to new one
        free(buk);
        buk = newBuk;
        cap = newCap;
    }

    // add count
    buk[idx]++;
    size = max(size, idx + 1);
}

uint64_t *getBuk()
{
    return buk;
}

uint64_t getBukCnt(uint64_t idx)
{
    return buk[idx];
}

uint64_t getBukSize()
{
    return size;
}

void clearBuk()
{
    if (size > 0) free(buk);
    buk = NULL;
    size = 0;
    cap = 0;
}
