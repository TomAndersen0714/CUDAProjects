#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// __global__ method can be called on host or device(whth 3 compute capability).
__global__ void checkIndex(void) {
    printf(
        "gridDim:(%d,%d,%d), blockDim:(%d,%d,%d), threadIdx:(%d,%d,%d), blockIdx:(%d,%d,%d).\n",
        gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z,
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z
    );
}

// __global__ method can only be called on device.
__device__ void methodOnDevice(void) {
    
}

// __global__ method can only be called on host.
__host__ void methodOnHost(void) {
    
}

int main(int argc, char** argv) {
    // Define the number of elements;
    const int numOfElements = 32;

    // Define grid and block structures
    dim3 block(3);
    dim3 grid((numOfElements + block.x - 1) / block.x);

    // Check grid and block dimension on host side.
    printf("grid.x:%d, grid.y:%d, grid.z:%d.\n", grid.x, grid.y, grid.z);
    printf("block.x:%d, block.y:%d, block.z:%d.\n", block.x, block.y, block.z);

    // Check grid and block dimension on device side.
    checkIndex << <grid, block >> > ();

    // Let host wait for device finish.
    cudaDeviceSynchronize();

    // Reset device memory allocation and status.
    cudaDeviceReset();

    return 0;
}