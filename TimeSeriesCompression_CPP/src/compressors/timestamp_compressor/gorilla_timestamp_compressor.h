#ifndef _GORILLA_TIMESTAMP_COMPRESSOR_H_
#define _GORILLA_TIMESTAMP_COMPRESSOR_H_

#include "../../tools/encode_tool.h"

class GorillaTimestampCompressor {

private:
    int64_t prevTimestamp = 0;
    int32_t prevDelta = 0;

    static const int32_t DELTA_7_MASK = 0b10 << 7;
    static const int32_t DELTA_9_MASK = 0b110 << 9;
    static const int32_t DELTA_12_MASK = 0b1110 << 12;

public:
    void addTimestamp(int64_t timestamp);
    void close();
};


#endif // _GORILLA_TIMESTAMP_COMPRESSOR_H_