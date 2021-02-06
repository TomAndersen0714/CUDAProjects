#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_writer.h"

ByteBuffer* timestamp_compress_gorilla(DataBuffer* timestamps);

ByteBuffer* timestamp_compress_rle(DataBuffer* timestamps);

ByteBuffer* value_compress_gorilla(DataBuffer* values);

ByteBuffer* value_compress_bitpack(DataBuffer* values);

ByteBuffer* value_compress_bucket(DataBuffer* values);

#endif // _COMPRESSOR_H_