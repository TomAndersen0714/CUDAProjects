#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "utils/data_types.h"

// Compress timestamps on GPU using Gorilla, return compressed data which is not compacted
CompressedData *timestamp_compress_gorilla_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress timestamps on GPU using RLE, return compressed data which is not compacted
CompressedData *timestamp_compress_rle_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress timestamps on GPU using RLE, return compressed data which is compacted
CompressedData *timestamp_compress_rle_gpu_c(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress values on GPU using Gorilla, return compressed data which is not compacted
CompressedData *value_compress_gorilla_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress values on GPU using Bucket, return compressed data which is not compacted
CompressedData *value_compress_bucket_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress values on GPU using Bucket, return compressed data which is compacted
CompressedData *value_compress_bucket_gpu_c(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress values on GPU using Bitpack, return compressed data which is not compacted
CompressedData *value_compress_bitpack_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Compress values on GPU using Bitpack, return compressed data which is compacted
CompressedData *value_compress_bitpack_gpu_c(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
);

// Trigger the lazy creation of the CUDA context
void warmUp();

#endif // _COMPRESSOR_H_