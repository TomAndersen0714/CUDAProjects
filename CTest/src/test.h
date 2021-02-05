#pragma once

#include <time.h>
#include <stdint.h>

// Get unix timestamp in seconds at current moment.
inline int64_t unixSecondTimestamp() {
    time_t current;
    time(&current);
    return current;
}

// Get unix timestamp in seconds at current moment.
inline int64_t unixSecondTimestamp1() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec;
}

// Get unix timestamp in milliseconds at current moment.
inline int64_t unixMillisecondTimestamp() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}
