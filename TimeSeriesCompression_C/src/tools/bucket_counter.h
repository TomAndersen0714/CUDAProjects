#ifndef _BUCKET_COUNTER_H_
#define _BUCKET_COUNTER_H_

#include<stdint.h>

void bukAdd(uint64_t idx);

// get the bucket
uint64_t *getBuk();

// get the count of a specific bucket
uint64_t getBukCnt(uint64_t idx);

// get the number of buckets
uint64_t getBukSize();

// clear the buckets
void clearBuk();

#endif // _BUCKET_COUNTER_H_