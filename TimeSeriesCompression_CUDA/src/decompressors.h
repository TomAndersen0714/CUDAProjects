#ifndef _DECOMPRESSORS_H_
#define _DECOMPRESSORS_H_

#include "utils/data_types.h"

// Decompress compressed and compacted timestamps data on GPU using Gorilla
ByteBuffer *timestamp_decompress_gorilla_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

// Decompress compressed and compacted timestamps data on GPU using RLE
ByteBuffer *timestamp_decompress_rle_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

// Decompress compressed and compacted values data on GPU using Gorilla
ByteBuffer *value_decompress_gorilla_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

// Decompress compressed and compacted values data on GPU using Bucket
ByteBuffer *value_decompress_bucket_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

// Decompress compressed and compacted values data on GPU using Bitpack
ByteBuffer *value_decompress_bitpack_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
);

#endif // _DECOMPRESSORS_H_