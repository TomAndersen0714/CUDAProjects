#ifndef _CUDA_COMMON_UTILS_H_
#define _CUDA_COMMON_UTILS_H_

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// Define error check function.
static inline void checkCudaError(const cudaError_t error) {

    if (error != cudaSuccess) {
        printf("Error: %s:%d,  ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Warming up for GPU
__global__ static inline void warmingUp() {

}
#endif // _CUDA_COMMON_UTILS_H_