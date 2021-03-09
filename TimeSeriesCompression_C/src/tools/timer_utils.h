#ifndef _TIMER_UTILS_H_
#define _TIMER_UTILS_H_

#include <time.h>
#include <stdint.h>

// Get unix timestamp in seconds at current moment.
static inline uint64_t unixSecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec;
}

// Get unix timestamp in milliseconds at current moment.
static inline uint64_t unixMillisecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

#endif // _TIMER_UTILS_H_