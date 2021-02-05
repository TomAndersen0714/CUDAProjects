#ifndef _BIT_BUFFER_WRITER_H_
#define _BIT_BUFFER_WRITER_H_

#include "bit_buffer.h"
#include "bit_writer.h"

class BitBufferWriter :public BitBuffer, public BitWriter {

public:

    BitBufferWriter() :
        BitBuffer() {
        cacheByte = buffer[cursor];
        leftBits = BITS_OF_BYTE;
    };

    BitBufferWriter(ByteBuffer byteBuffer) :
        BitBuffer(byteBuffer) {

    }

    ~BitBufferWriter() {

    }

    // Inherited from BitWriter
    void writeOneBit() override;

    void writeZeroBit() override;

    void writeBits(uint64_t value, int32_t bits) override;

    void writeByte(byte value) override;

    void writeInt(uint32_t value) override;

    void writeLong(uint64_t value) override;

    void writeFloat(float value) override;

    void writeDouble(double value) override;

    void flush() override;

    // Inherited from BitBuffer
    void flipByte() override;
};
#endif // _BIT_BUFFER_WRITER_H_