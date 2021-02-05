#ifndef _ENCODE_TOOL_H_
#define _ENCODE_TOOL_H_

#include <stdint.h>

// Encode a 32-bit signed value(i.e. Integer type value) to unsigned value.
inline int32_t encodeZigZag32(const int32_t n) {
    return (n << 1) ^ (n >> 31);
}

// Encode a 64-bit signed value(i.e. Integer type value) to unsigned value.
inline int64_t encodeZigZag64(const int64_t n) {
    return (n << 1) ^ (n >> 63);
}

// Decode a ZigZag-encoded 32-bit value.
inline int32_t decodeZigZag32(const int32_t n) {
    return ((uint32_t)n >> 1) ^ -(n & 1);
}

// Decode a ZigZag-encoded 64-bit value.
inline int64_t decodeZigZag64(const int64_t n) {
    return ((int64_t)n >> 1) ^ -(n & 1);
}

#endif // _ENCODE_TOOL_H_
