#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_writer.h"

ByteBuffer* timestamp_compress_gorilla(UncompressedData* timestamps);

ByteBuffer* timestamp_compress_rle(UncompressedData* timestamps);

ByteBuffer* value_compress_gorilla(UncompressedData* values);

ByteBuffer* value_compress_bitpack(UncompressedData* values);

ByteBuffer* value_compress_bucket(UncompressedData* values);

#endif // _COMPRESSOR_H_