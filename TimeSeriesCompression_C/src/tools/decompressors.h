#ifndef _DECOMPRESSORS_H_
#define _DECOMPRESSORS_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_reader.h"

ByteBuffer* timestamp_decompress_gorilla(ByteBuffer* timestamps, uint64_t count);

ByteBuffer* timestamp_decompress_rle(ByteBuffer* timestamps, uint64_t count);

ByteBuffer* value_decompress_gorilla(ByteBuffer* values, uint64_t count);

ByteBuffer* value_decompress_bitpack(ByteBuffer* values, uint64_t count);

ByteBuffer* value_decompress_bucket(ByteBuffer* values, uint64_t count);

#endif // _DECOMPRESSORS_H_