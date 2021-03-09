#ifndef _CUDA_COMMON_UTILS_H_
#define _CUDA_COMMON_UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// Define error check function.
#define checkCudaError(error) check((error), __FILE__, __LINE__)

void check(const cudaError_t error, const char *const file, int const line) {
    if (error != cudaSuccess) {
        printf("Error: %s:%d,  ", file, line);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Warming up for GPU
__global__ static void warmingUp() {

}
#endif // _CUDA_COMMON_UTILS_H_