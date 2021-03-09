#ifndef _BIT_READER_H_
#define _BIT_READER_H_

#include "data_types.h"

typedef struct _BitReader {
    ByteBuffer* byteBuffer;
    uint8_t leftBits;
    byte cacheByte;
    uint64_t cursor;
} BitReader;

// Construct a BitReader
inline BitReader* bitReaderConstructor(ByteBuffer* byteBuffer) {
    BitReader *bitReader = (BitReader*) malloc(sizeof(BitReader));
    assert(bitReader != NULL);
    bitReader->byteBuffer = byteBuffer;
    bitReader->leftBits = BITS_OF_BYTE;
    bitReader->cacheByte = byteBuffer->buffer[0]; // cache first byte
    bitReader->cursor = 1; // 'cursor' point to next byte
    return bitReader;
}

// Deconstruct a BitReader
inline void bitReaderDeconstructor(BitReader* bitReader) {
    free(bitReader);
}

// Get a new byte from buffer, if all bits in cached byte have been read.
inline void bitReaderFlipByte(BitReader* bitReader) {
    if (bitReader->leftBits == 0) {
        assert(bitReader->cursor < bitReader->byteBuffer->length);
        bitReader->cacheByte =
            bitReader->byteBuffer->buffer[bitReader->cursor++];
        bitReader->leftBits = BITS_OF_BYTE;
    }
}

// Read the next bit and returns true if is '1' bit and false if not.
inline bool bitReaderNextBit(BitReader* bitReader) {
    bool bit = bitReader->cacheByte >> (bitReader->leftBits - 1) & 1;
    bitReader->leftBits--;
    bitReaderFlipByte(bitReader);
    return bit;
}

// Read bit continuously, until next '0' bit is found or the number of read bits reach the value of 'maxBits'.
inline uint32_t bitReaderNextControlBits(BitReader* bitReader, uint32_t maxBits) {
    uint32_t controlBits = 0x00;
    bool bit;

    for (uint32_t i = 0; i < maxBits; i++) {
        controlBits <<= 1;
        bit = bitReaderNextBit(bitReader);
        if (bit) {
            // add a '1' bit to the end
            controlBits |= 1;
        }
        else {
            // add a '0' bit to the end
            break;
        }
    }
    return controlBits;
}

inline int64_t bitReaderNextLong(BitReader* bitReader, uint32_t bits) {

    int64_t value = 0;
    byte leastSignificantBits;

    while (bits > 0 && bits <= BITS_OF_LONG_LONG) {
        // If the number of bits to read is more than the left bits in cache byte
        if (bits > bitReader->leftBits || bits == BITS_OF_BYTE) {
            // Take only the least significant bits
            leastSignificantBits =
                bitReader->cacheByte & ((1 << bitReader->leftBits) - 1);
            // value = (value << bitReader->leftBits) + (leastSignificantBits & 0xFF);
            value = (value << bitReader->leftBits) | leastSignificantBits;
            bits -= bitReader->leftBits;
            bitReader->leftBits = 0;
        }
        else {
            // Shift to correct position and take only least significant bits
            leastSignificantBits =
                (bitReader->cacheByte >> (bitReader->leftBits - bits)) & ((1 << bits) - 1);
            // value = (value << bits) + (leastSignificantBits & 0xFF);
            value = (value << bits) | leastSignificantBits;
            bitReader->leftBits -= bits;
            bits = 0;
        }
        bitReaderFlipByte(bitReader);
    }
    return value;
}
#endif // _BIT_READER_H_