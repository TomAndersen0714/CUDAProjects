#ifndef _CUDA_COMMON_UTILS_H_
#define _CUDA_COMMON_UTILS_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data_types.h"

// Define error check function.
#define checkCudaError(error) check((error), __FILE__, __LINE__)

static inline void check(const cudaError_t error, const char *const file, int const line) {
    if (error != cudaSuccess) {
        printf("Error: %s:%d,  ", file, line);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Warming up for GPU
__global__ static void warmingUp() {

}

// Verify the input params according to the 'count' for compression
static inline void verifyParams(
    const uint32_t count,
    uint16_t *frame_p,
    uint32_t *block_p,
    uint32_t *thdPB_p,
    uint32_t *thd_p
) {
    uint32_t
        frame, thd, thdsPB, block;
    //uint32_t
    //    padding = 0, // unused space in the used last frame
    //    left = 0, // unused threads in the last block
    //    *offs; // start offsets of data that each thread will compress

    block = *block_p;
    thdsPB = *thdPB_p;
    thd = *thd_p;
    frame = *frame_p;

    // if count <= MIN_FRAME_SIZE, use just one thread to compress
    if (count <= MIN_FRAME_SIZE) {
        frame = count; block = 1; thdsPB = 1; thd = 1;
    }
    else if (frame <= MIN_FRAME_SIZE) {// else if frame is too short
        frame = MIN_FRAME_SIZE;
        // recalculate the number of needed threads
        thd = (count + frame - 1) / frame;
        //padding = frame - count % frame;
        if (thd < MAX_THREADS_PER_BLOCK) {
            // use just one block to compress
            block = 1;
            //warps = (thd + WARPSIZE - 1) / WARPSIZE;
            thdsPB = thd;
        }
        else {
            // use block as less as possible to compress according to the 
            // number of threads within per block
            block = (thd + thdsPB - 1) / thdsPB;
            /*left = blocks*thdsPerBlock - thds;
            thds = blocks*thdsPerBlock;*/
        }
    }
    else if (frame > MAX_FRAME_SIZE) {// else if frame is too long
        frame = MAX_FRAME_SIZE;
        // recalculate the number of needed threads
        thd = (count + frame - 1) / frame;
        //padding = frame - count % frame;
        if (thd < MAX_THREADS_PER_BLOCK) {
            // use just one block to compress
            block = 1;
            //warps = (thd + WARPSIZE - 1) / WARPSIZE;
            thdsPB = thd;
        }
        else {
            // use block as less as possible to compress according to the 
            // number of threads within per block
            block = (thd + thdsPB - 1) / thdsPB;
            /*left = blocks*thdsPerBlock - thds;
            thds = blocks*thdsPerBlock;*/
        }
    }

    // construct the offsets array, each scope of frame is [offs[i],offs[i+1])
    /*offs = (uint32_t*)malloc(BYTES_OF_INT*(thds + 1));
    for (int i = 0; i <= thds - left - 1; i++)
        offs[i] = i*frame;
    offs[thds - left] = (thds - left)*frame - padding;
    for (int i = thds - left + 1; i <= thds; i++)
        offs[i] = offs[thds - left];*/
    /*offs = (uint32_t*)malloc(BYTES_OF_INT*(thds + 1));
    for (int i = 0; i < thds; i++) offs[i] = i*frame;
    offs[thds] = count;*/

    // modify input params and return result
    *frame_p = frame;
    *block_p = block;
    *thdPB_p = thdsPB;
    *thd_p = thd;
    //return offs;
}

//// Verify the input params for decompression
//static inline void verifyParams(
//    const uint32_t count,
//    const uint32_t thd,
//    const uint16_t frame,
//    uint32_t *block_p,
//    uint32_t *thdPB_p
//) {
//    uint32_t
//        block = *block_p,
//        thdPB = *thdPB_p;
//
//    // if thd <= MAX_THREADS_PER_BLOCK, use just one block to compress
//    if (thd <= MAX_THREADS_PER_BLOCK) {
//        block = 1; thdPB = thd;
//    }
//    else {
//        block = (thd + thdPB - 1) / thdPB;
//    }
//    // modify input params
//    *block_p = block;
//    *thdPB_p = thdPB;
//
//}

// since the 'Size and Alignment Requirement', we have to read the 
// the compressed data(bytes) byte-by-byte
__device__ static inline uint64_t readDeviceMem64(byte* buffer,int bits) {
    uint64_t tmp = 0; 
    byte b;
    for (int i = bits; i >= 0; i--) {
        b = *(buffer + i);
        tmp = tmp << BITS_OF_BYTE | b;
    }
    return tmp;
}

#endif // _CUDA_COMMON_UTILS_H_