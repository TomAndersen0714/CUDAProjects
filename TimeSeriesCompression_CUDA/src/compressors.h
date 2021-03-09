#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "utils/data_types.h"


// Compress timestamps on GPU using Gorilla
void timestamp_compress_gorilla_gpu(
    ByteBuffer* tsByteBuffer,
    uint32_t blocks,
    uint32_t warps
);

/*
ByteBuffer* timestamp_compress_rle_gpu(ByteBuffer* tsByteBuffer);

ByteBuffer* value_compress_gorilla_gpu(ByteBuffer* valByteBuffer);

ByteBuffer* value_compress_bitpack_gpu(ByteBuffer* valByteBuffer);

ByteBuffer* value_compress_bucket_gpu(ByteBuffer* valByteBuffer);
*/

#endif // _COMPRESSOR_H_