#ifndef _BIT_BUFFER_H_
#define _BIT_BUFFER_H_

#include "data_types.h"

class BitBuffer {

public:

    virtual ~BitBuffer() = 0;

private:
    // Default capacity for buffer(i.e. 4KB)
    static const int32_t __DEFAULT_CAPACITY = 4096;

protected:

    ByteBuffer buffer; // Byte buffer for reader/writer
    uint32_t cursor = 0; // The pointer to the current byte in buffer
    byte cacheByte = 0; // The cached byte in buffer
    int leftBits = 0; // The number of unread bits in cached byte

    BitBuffer() :BitBuffer(__DEFAULT_CAPACITY) {
    }

    BitBuffer(int32_t capacity) {
        this->buffer = ByteBuffer();
        this->buffer.reserve(capacity);

        //this->buffer = ByteBuffer(capacity);
        //this->buffer.reserve(capacity);
    }

    BitBuffer(ByteBuffer byteBuffer) {
        this->buffer = byteBuffer;
    }

    /**
    * Flush the cached byte.
    */
    virtual void flipByte() = 0;

    /**
    * Returns the output buffer.
    */
    ByteBuffer getBuffer() {
        return this->buffer;
    }

};

#endif // _BIT_BUFFER_H_