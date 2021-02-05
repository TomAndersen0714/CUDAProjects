#include "bit_buffer_writer.h"

void BitBufferWriter::writeOneBit()
{
    cacheByte |= (1 << (leftBits - 1));
    leftBits--;
    flipByte();
}

void BitBufferWriter::writeZeroBit()
{
    leftBits--;
    flipByte();
}

void BitBufferWriter::writeBits(uint64_t value, int32_t bits)
{
    while (bits > 0) {
        int shift = bits - leftBits;
        if (shift >= 0) {
            cacheByte |= (byte)((value >> shift) & ((1 << leftBits) - 1));
            bits -= leftBits;
            leftBits = 0;
        }
        else {
            shift = leftBits - bits;
            cacheByte |= (byte)(value << shift);
            leftBits -= bits;
            bits = 0;
        }
        flipByte();
    }
}

void BitBufferWriter::writeByte(byte value)
{
    writeBits(value, BITS_OF_BYTE);
}

void BitBufferWriter::writeInt(uint32_t value)
{
    writeBits(value, BITS_OF_INT);
}

void BitBufferWriter::writeLong(uint64_t value)
{
    writeBits(value, BITS_OF_LONG);
}

void BitBufferWriter::writeFloat(float value)
{
    writeBits(*(uint32_t*)(&value), BITS_OF_FLOAT);
}

void BitBufferWriter::writeDouble(double value)
{
    writeBits(*(uint64_t*)(&value), BITS_OF_DOUBLE);
}

void BitBufferWriter::flush()
{
    leftBits = 0;
    flipByte();
    this->buffer = this->getBuffer();
}

// Push the current byte, and flush the cached byte.
void BitBufferWriter::flipByte()
{
    if (leftBits == 0) {

        buffer.push_back(cacheByte);
        cursor++;

        cacheByte = 0;
        leftBits = BITS_OF_BYTE;
    }
}
