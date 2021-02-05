#ifndef _BIT_READER_H_
#define _BIT_READER_H_

#include "data_types.h"

class BitReader {

public:

    virtual ~BitReader() = 0;

    /**
    * Read the next bit and returns true if is '1' bit and false if not.
    *
    * @return true(i.e. ' 1 ' bit), false(i.e. '0' bit)
    */
    virtual bool nextBit() = 0;

    /**
    * Read bit continuously, until next '0' bit is found or the number of
    * read bits reach the value of 'maxBits'.
    *
    * @param maxBits How many bits at maximum until returning
    * @return Integer value of the read bits
    */
    virtual int32_t nextControlBits(uint32_t maxBits) = 0;

    /**
    * Read next 8 bits and returns the byte value.
    *
    * @return byte type value
    */
    virtual byte nextByte() = 0;

    /**
    * Read next n(n<=8) bits and return the corresponding byte type value.
    *
    * @param bits the number of bit need to read
    * @return byte type value
    */
    virtual byte nextByte(uint32_t bits) = 0;

    /**
    * Read next 32 bits and return the corresponding integer type value.
    *
    * @return integer type value.
    */
    virtual int32_t nextInt() = 0;

    /**
    * Read next n(n<=32) bits and return the corresponding integer type value.
    *
    * @param bits the number of bit to read.
    * @return integer type value.
    */
    virtual int32_t nextInt(uint32_t bits) = 0;

    /**
    * Read next 64 bits and return the corresponding long type value.
    *
    * @return long type value .
    */
    virtual int64_t nextLong() = 0;

    /**
    * Read next n(n<=64) bits and return the corresponding long type value.
    *
    * @param bits the number of bit to read.
    * @return long type value.
    */
    virtual int64_t nextLong(uint32_t bits) = 0;

    /**
    * Read next 32 bits and return the corresponding float type value.
    *
    * @return float type value.
    */
    virtual float nextFloat() = 0;

    /**
    * Read next n(n<=32) bits and return the corresponding float type value.
    *
    * @param bits the number of bit to read
    * @return float type value.
    */
    virtual float nextFloat(uint32_t bits) = 0;

    /**
    * Read next 64 bits and return the corresponding double type value.
    *
    * @return double type value.
    */
    virtual double nextDouble() = 0;

    /**
    * Read next n(n<=64) bits and return the corresponding double type value.
    *
    * @param bits the number of bit to read.
    * @return double type value.
    */
    virtual double nextDouble(uint32_t bits) = 0;

};

#endif // _BIT_READER_H_