#ifndef _COMPRESSOR_H_
#define _COMPRESSOR_H_

#include "data_types.h"
#include "encode_utils.h"
#include "bit_writer.h"

ByteBuffer* timestamp_compress_gorilla(ByteBuffer* tsByteBuffer);

ByteBuffer* timestamp_compress_rle(ByteBuffer* tsByteBuffer);

ByteBuffer* value_compress_gorilla(ByteBuffer* valByteBuffer);

ByteBuffer* value_compress_bitpack(ByteBuffer* valByteBuffer);

ByteBuffer* value_compress_bucket(ByteBuffer* valByteBuffer);

#endif // _COMPRESSOR_H_