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

// Define error check function.
//#ifndef CHECK_ERROR
//#define CHECK_ERROR(cudaError_t call)                                                       \
//{                                                                               \
//    const cudaError_t error = call;                                             \
//    if(error != cudaError::cudaSuccess){                                        \
//        printf("Error: %s:%d,  ", __FILE__, __LINE__);                          \
//        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
//        exit(1);                                                                \
//    }                                                                           \
//}
//#endif // !CHECK_ERROR(call

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

// Generate random numbers for input float array.
__host__ static void initialData(float *arr, int num) {
    // Generate different seed for random numbers.
    time_t t;
    srand((unsigned int)time(&t));
    // Generate random numbers
    for (int i = 0; i < num; i++) {
        arr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Since array object will automatically convert to corresponding type point when it is passwd as 
// a variable, we need pass the length of array concurrently.
__host__ static void printResult(float arr[], int num) {
    if (arr == NULL) return;
    // int num = sizeof(arr) / sizeof(float); // error
    for (int i = 0; i < num; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

// Compare every element in two float array.
__host__ static void checkResult(float* cpuArray, float*gpuArray, const int N) {
    double epsilon = 1.0E-8;
    bool isMatch = true;
    for (int i = 0; i < N; i++) {
        if (fabs(cpuArray[i] - gpuArray[i]) > epsilon) {
            isMatch = false;
            printf("Arrays don't match!\n");
            printf("CPU: %5.2f, GPU: %5.2f, idx: %d", cpuArray[i], gpuArray[i], i);
            break;
        }
    }
    if (isMatch) printf("Arrays match succeed!\n");
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

__host__ static void sumArraysOnHost(float *a, float *b, float *c, const int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sumArraysOnDevice(float *a, float *b, float *c) {
    uint64_t threadIdxOfGrid = getThreadIdxOfGrid();
    c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
}

__global__ static void sumArraysOnDevice(float *a, float *b, float *c, const int n) {
    uint64_t threadIdxOfGrid = getThreadIdxOfGrid();
    if (threadIdxOfGrid < n) {
        c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
    }
}

__global__ static void warmingUp(float *a, float *b, float *c) {
    uint64_t threadIdxOfGrid = getThreadIdxOfGrid();
    c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
}

__global__ static void warmingUp(float *a, float *b, float *c, const int n) {
    uint64_t threadIdxOfGrid = getThreadIdxOfGrid();
    if (threadIdxOfGrid < n) {
        c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
    }
}

// Leading zeros count for 32-bits number.
__device__ static inline uint8_t leadingZerosCount32(uint32_t val) {
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
__device__ static inline uint8_t leadingZerosCount64(uint64_t val) {
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
__device__ static inline uint8_t trailingZerosCount32(uint32_t val) {
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
__device__ static inline uint8_t trailingZerosCount64(uint64_t val) {
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

#endif // !__CUDA__TOOL__

