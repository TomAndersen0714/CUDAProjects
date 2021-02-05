#include "bit_buffer_reader.h"

bool BitBufferReader::nextBit()
{
    return false;
}

int32_t BitBufferReader::nextControlBits(uint32_t maxBits)
{
    return int32_t();
}

byte BitBufferReader::nextByte()
{
    return byte();
}

byte BitBufferReader::nextByte(uint32_t bits)
{
    return byte();
}

int32_t BitBufferReader::nextInt()
{
    return int32_t();
}

int32_t BitBufferReader::nextInt(uint32_t bits)
{
    return int32_t();
}

int64_t BitBufferReader::nextLong()
{
    return int64_t();
}

int64_t BitBufferReader::nextLong(uint32_t bits)
{
    return int64_t();
}

float BitBufferReader::nextFloat()
{
    return 0.0f;
}

float BitBufferReader::nextFloat(uint32_t bits)
{
    return 0.0f;
}

double BitBufferReader::nextDouble()
{
    return 0.0;
}

double BitBufferReader::nextDouble(uint32_t bits)
{
    return 0.0;
}

// Read and cache next byte.
void BitBufferReader::flipByte()
{
    if (leftBits == 0) {
        cacheByte = buffer.at(cursor++);
        leftBits = BITS_OF_BYTE;
    }
}
