#ifndef _DECOMPRESSORS_H_
#define _DECOMPRESSORS_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_reader.h"

DataBuffer* timestamp_decompress_gorilla(ByteBuffer* timestamps, uint64_t length);

DataBuffer* timestamp_decompress_rle(ByteBuffer* timestamps, uint64_t length);

DataBuffer* value_decompress_gorilla(ByteBuffer* values, uint64_t length);

DataBuffer* value_decompress_bitpack(ByteBuffer* values, uint64_t length);

DataBuffer* value_decompress_bucket(ByteBuffer* values, uint64_t length);

#endif // _DECOMPRESSORS_H_