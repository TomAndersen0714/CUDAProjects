#ifndef _CUDA_TOOL_H_
#define _CUDA_TOOL_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__device__ static uint64_t getThreadIdxOfGrid(void) {
    uint64_t threadOfBlock =
        blockDim.x * blockDim.y * blockDim.z;
    uint64_t blockIdxOfGrid =
        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    uint64_t threadIdxOfBlock =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    return threadOfBlock * blockIdxOfGrid + threadIdxOfBlock;
}

// Define error check function.
__host__ static void CHECK_ERROR(const cudaError_t error) {

    if (error != cudaSuccess) {
        printf("Error: %s:%d,  ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

// Get unix timestamp in seconds at current moment.
__host__ static int64_t unixSecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec;
}

// Get unix timestamp in milliseconds at current moment.
__host__ static int64_t unixMillisecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}


// __global__ method can be called on host and device(whth 3 compute capability).
__global__ static void checkIndex(void) {
    printf
    (
        "gridDim:(%d,%d,%d), blockDim:(%d,%d,%d),threadIdx:(%d,%d,%d), blockIdx:(%d,%d,%d),threadIdxOfGrid:%lld.\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
        getThreadIdxOfGrid()
    );
}

#endif // !__CUDA__TOOL__

