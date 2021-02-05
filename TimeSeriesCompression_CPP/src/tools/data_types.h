#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <vector>
#include <cstdint>

#define BITS_OF_BYTE 8
#define BITS_OF_INT 32
#define BITS_OF_FLOAT 32
#define BITS_OF_LONG 64
#define BITS_OF_DOUBLE 64

typedef unsigned char byte;
typedef std::vector<byte> ByteBuffer;

#endif // _DATA_TYPES_H_