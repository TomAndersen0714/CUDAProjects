#include "bit_buffer.h"
#include "bit_reader.h"

class BitBufferReader:public BitBuffer,public BitReader
{

public:

    BitBufferReader(ByteBuffer inputByteBuffer) :
        BitBuffer(inputByteBuffer) {
        // Cache first byte.
        flipByte();
    }

    ~BitBufferReader() {

    };

private:

    // Inherited from BitBuffer
    virtual void flipByte() override;

    // Inherited from BitBuffer BitReader
    virtual bool nextBit() override;

    virtual int32_t nextControlBits(uint32_t maxBits) override;

    virtual byte nextByte() override;

    virtual byte nextByte(uint32_t bits) override;

    virtual int32_t nextInt() override;

    virtual int32_t nextInt(uint32_t bits) override;

    virtual int64_t nextLong() override;

    virtual int64_t nextLong(uint32_t bits) override;

    virtual float nextFloat() override;

    virtual float nextFloat(uint32_t bits) override;

    virtual double nextDouble() override;

    virtual double nextDouble(uint32_t bits) override;

};
