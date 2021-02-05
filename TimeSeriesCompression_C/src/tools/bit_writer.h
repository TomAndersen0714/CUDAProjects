#ifndef _BIT_READER_H_
#define _BIT_READER_H_

#include "data_types.h"

typedef struct _BitWriter {
    ByteBuffer* byteBuffer;
    uint8_t leftBits;
    byte cacheByte;
    //uint64_t cursor;
} BitWriter;

// Construct a BitWriter
inline BitWriter* bitWriterConstructor(ByteBuffer* byteBuffer) {
    BitWriter *bitWriter = malloc(sizeof(BitWriter));
    assert(bitWriter != NULL);
    bitWriter->byteBuffer = byteBuffer;
    bitWriter->leftBits = BITS_OF_BYTE;
    bitWriter->cacheByte = 0;
    return bitWriter;
}

// Deconstruct a BitWriter
inline void bitWriterDeconstructor(BitWriter* bitWriter) {
    free(bitWriter);
}

// If cached byte is full, then write it into buffer and get next empty byte
inline void bitWriterFlipByte(BitWriter* bitWriter) {
    // If cached byte is full
    if (bitWriter->leftBits == 0) {
        // write the cache byte into buffer
        bitWriter->byteBuffer->buffer[bitWriter->byteBuffer->length++] 
            = bitWriter->cacheByte;
        // If the left space of buffer is run out, realloc more memory space
        if (bitWriter->byteBuffer->length == bitWriter->byteBuffer->capacity) {
            uint64_t newCap =
                bitWriter->byteBuffer->capacity + (bitWriter->byteBuffer->capacity >> 1);
            void* newBuffer = realloc(
                bitWriter->byteBuffer->buffer, newCap
            );
            // If realloc succeed
            assert(newBuffer != NULL);
            bitWriter->byteBuffer->capacity = newCap;
            bitWriter->byteBuffer->buffer = newBuffer;
        }
        // Reset the params
        bitWriter->cacheByte = 0;
        bitWriter->leftBits = BITS_OF_BYTE;
    }
};

// Write the specific least significant bits of value into the buffer
inline void bitWriterWriteBits(BitWriter* bitWriter, uint64_t value, uint64_t bits) {
    int64_t shift;
    while (bits > 0) {
        shift = bits - bitWriter->leftBits;
        // If the left bits in cached byte is not enough to write following bits
        if (shift >= 0) {
            bitWriter->cacheByte |= (byte)((value >> shift) & ((1 << bitWriter->leftBits) - 1));
            bits -= bitWriter->leftBits;
            bitWriter->leftBits = 0;
        }
        // If it is enough to write following bits
        else {
            shift = -shift;
            bitWriter->cacheByte |= (byte)((value << shift) & ((1 << bitWriter->leftBits) - 1));
            bitWriter->leftBits -= (uint8_t)bits;
            bits = 0;
        }
        bitWriterFlipByte(bitWriter);
    }
}

// Write a 64-bits integer value into buffer
inline void bitWriterWriteLong(BitWriter* bitWriter, uint64_t value) {
    bitWriterWriteBits(bitWriter, value, BITS_OF_LONG_LONG);
}

// Write a double value into buffer
inline void bitWriterWriteDouble(BitWriter* bitWriter, double value) {
    bitWriterWriteBits(bitWriter, *((uint64_t*)&value), BITS_OF_DOUBLE);
}

// Write a '0' bit into cache byte
inline void bitWriterWriteZeroBit(BitWriter* bitWriter) {
    bitWriter->leftBits--;
    bitWriterFlipByte(bitWriter);
}

// Write a '1' bit into cache byte
inline void bitWriterWriteOneBit(BitWriter* bitWriter) {
    bitWriter->cacheByte |= (1 << (bitWriter->leftBits - 1));
    bitWriter->leftBits--;
    bitWriterFlipByte(bitWriter);
}

// Write the left bits in cached byte into the buffer.
inline void bitWriterFlush(BitWriter* bitWriter) {
    bitWriter->leftBits = 0;
    bitWriterFlipByte(bitWriter);
}

#endif // _BIT_READER_H_