#ifndef _BIT_WRITER_H_
#define _BIT_WRITER_H_

#include "data_types.h"

class BitWriter {

public:

    virtual ~BitWriter() = 0;
    /**
    * Write a '1' bit into the buffer.
    */
    virtual void writeOneBit() = 0;

    /**
    * Write a '0' bit into the buffer.
    */
    virtual void writeZeroBit() = 0;


    /**
    * Write the specific least significant bits of value into the buffer.
    *
    * @param value value need to write
    * @param bits  the number of bits need to write
    */
    virtual void writeBits(uint64_t value, int32_t bits) = 0;

    /**
    * Write a single byte into the buffer.
    *
    * @param b the byte value need to write.
    */
    virtual void writeByte(byte value) = 0;

    /**
    * Write a int type value into the buffer.
    *
    * @param value the int type value need to write.
    */
    virtual void writeInt(uint32_t value) = 0;

    /**
    * Write a long type value into the buffer.
    *
    * @param value the long type value need to write.
    */
    virtual void writeLong(uint64_t value) = 0;

    /**
    * Write a float type value into the buffer.
    *
    * @param value the float type value need to write.
    */
    virtual void writeFloat(float value) = 0;

    /**
    * Write a double type value into the buffer.
    *
    * @param value the double type value need to write.
    */
    virtual void writeDouble(double value) = 0;

    /**
    * Write the cached byte(s) to the buffer.
    */
    virtual void flush() = 0;

};
#endif // _BIT_WRITER_H_