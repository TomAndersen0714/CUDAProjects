#ifndef _DECOMPRESSORS_H_
#define _DECOMPRESSORS_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_reader.h"

//DataBuffer* timestamp_decompress(CompressedData* timestamps);
//
//DataBuffer* value_decompress(CompressedData* values);

UncompressedData* timestamp_decompress_gorilla(ByteBuffer* timestamps);

UncompressedData* timestamp_decompress_rle(ByteBuffer* timestamps);

UncompressedData* value_decompress_gorilla(ByteBuffer* values);

UncompressedData* value_decompress_bitpack(ByteBuffer* values);

UncompressedData* value_decompress_bucket(ByteBuffer* values);

#endif // _DECOMPRESSORS_H_