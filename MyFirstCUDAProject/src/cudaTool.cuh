#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef __CUDA__TOOL__
#define __CUDA__TOOL__

typedef unsigned __int64 u_int64;

// Define error check function.
#ifndef CHECK_ERROR
#define CHECK_ERROR(call)                                                       \
{                                                                               \
    const cudaError_t error = call;                                             \
    if(error != cudaError::cudaSuccess){                                        \
        printf("Error: %s:%d,  ", __FILE__, __LINE__);                          \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \
}
#endif // !CHECK_ERROR(call

__device__ u_int64 getThreadIdxOfGrid(void) {
    u_int64 threadOfBlock =
        blockDim.x * blockDim.y * blockDim.z;
    u_int64 blockNumOfGrid =
        blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    u_int64 threadNumOfBlock =
        threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    return threadOfBlock * blockNumOfGrid + threadNumOfBlock;
}

// Get unix timestamp in seconds at current moment.
__host__ __int64 unixSecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec;
}

// Get unix timestamp in milliseconds at current moment.
__host__  __int64 unixMillisecondTimestamp(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Generate random numbers for input float array.
__host__ void initialData(float *arr, int num) {
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
__host__ void printResult(float arr[], int num) {
    if (arr == NULL) return;
    // int num = sizeof(arr) / sizeof(float); // error
    for (int i = 0; i < num; i++) {
        printf("%.2f ", arr[i]);
    }
    printf("\n");
}

// Compare every element in two float array.
__host__ void checkResult(float* cpuArray, float*gpuArray, const int N) {
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
__global__ void checkIndex(void) {
    printf
    (
        "gridDim:(%d,%d,%d), blockDim:(%d,%d,%d),threadIdx:(%d,%d,%d), blockIdx:(%d,%d,%d),threadIdxOfGrid:%lld.\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
        getThreadIdxOfGrid()
    );
}

__host__ void sumArraysOnHost(float *a, float *b, float *c, const int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sumArraysOnDevice(float *a, float *b, float *c) {
    u_int64 threadIdxOfGrid = getThreadIdxOfGrid();
    c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
}

__global__ void sumArraysOnDevice(float *a, float *b, float *c, const int n) {
    u_int64 threadIdxOfGrid = getThreadIdxOfGrid();
    if (threadIdxOfGrid < n) {
        c[threadIdxOfGrid] = a[threadIdxOfGrid] + b[threadIdxOfGrid];
    }
}



#endif // !__CUDA__TOOL__

