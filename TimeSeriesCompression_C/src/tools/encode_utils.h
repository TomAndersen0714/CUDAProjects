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
    return ((uint64_t)n >> 1) ^ -(n & 1);
}

// Leading zeros count for 32-bits number.
inline uint8_t leadingZerosCount32(uint32_t val) {
    if (val == 0) return 32;
    int n = 1;
    if (val >> 16 == 0) { n += 16; val <<= 16; }
    if (val >> 24 == 0) { n += 8; val <<= 8; }
    if (val >> 28 == 0) { n += 4; val <<= 4; }
    if (val >> 30 == 0) { n += 2; val <<= 2; }
    n -= val >> 31;
    return n;
}

// Leading zeros count for 64-bits number.
inline uint8_t leadingZerosCount64(uint64_t val) {
    if (val == 0) return 64;
    int n = 1;
    uint32_t x = val >> 32;
    if (x == 0) { n += 32; x = (int)val; }
    if (x >> 16 == 0) { n += 16; x <<= 16; }
    if (x >> 24 == 0) { n += 8; x <<= 8; }
    if (x >> 28 == 0) { n += 4; x <<= 4; }
    if (x >> 30 == 0) { n += 2; x <<= 2; }
    n -= x >> 31;
    return n;
}

// Trailing zeros count for 32-bits number.
inline uint8_t trailingZerosCount32(uint32_t val) {
    int y;
    if (val == 0) return 32;
    int n = 31;
    y = val << 16; if (y != 0) { n = n - 16; val = y; }
    y = val << 8; if (y != 0) { n = n - 8; val = y; }
    y = val << 4; if (y != 0) { n = n - 4; val = y; }
    y = val << 2; if (y != 0) { n = n - 2; val = y; }
    return n - ((uint32_t)(val << 1) >> 31);
}

// Trailing zeros count for 32-bits number.
inline uint8_t trailingZerosCount64(uint64_t val) {
    int x, y;
    if (val == 0) return 64;
    int n = 63;
    y = (int)val; if (y != 0) { n = n - 32; x = y; }
    else x = (int)((uint64_t)val >> 32);
    y = x << 16; if (y != 0) { n = n - 16; x = y; }
    y = x << 8; if (y != 0) { n = n - 8; x = y; }
    y = x << 4; if (y != 0) { n = n - 4; x = y; }
    y = x << 2; if (y != 0) { n = n - 2; x = y; }
    return n - ((uint32_t)(x << 1) >> 31);
}

#endif // _ENCODE_TOOL_H_
