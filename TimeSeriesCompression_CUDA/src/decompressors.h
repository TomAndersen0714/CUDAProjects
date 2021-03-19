#ifndef _DECOMPRESSORS_H_
#define _DECOMPRESSORS_H_

#include "utils/data_types.h"

// Decompress compressed and compacted timestamps data on GPU using Gorilla
ByteBuffer *timestamp_decompress_gorilla_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

// Decompress compressed and compacted values data on GPU using Gorilla
ByteBuffer *value_decompress_gorilla_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

/*
ByteBuffer* timestamp_decompress_rle(ByteBuffer* timestamps, uint64_t count);

ByteBuffer* value_decompress_bitpack(ByteBuffer* values, uint64_t count);

ByteBuffer* value_decompress_bucket(ByteBuffer* values, uint64_t count);*/

#endif // _DECOMPRESSORS_H_