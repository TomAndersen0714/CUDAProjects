#include <stdio.h>
#include <math.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>


int main(int argc, char **argv) {
    // Define the number of elements.
    int numOfElements = 1024;

    // Define grid and block structure.
    dim3 block(1024);

    dim3 grid((numOfElements + block.x - 1) / block.x); // 相当于向上取整
    printf("grid.x %d, block.x %d.\n", grid.x, block.x);

    // Reset block.
    block.x = 512;
    grid.x = (numOfElements + block.x - 1) / block.x;
    printf("grid.x %d, block.x %d.\n", grid.x, block.x);

    // Reset block.
    block.x = 256;
    grid.x = (numOfElements + block.x - 1) / block.x;
    printf("grid.x %d, block.x %d.\n", grid.x, block.x);

    // Reset block.
    block.x = 128;
    grid.x = (numOfElements + block.x - 1) / block.x;
    printf("grid.x %d, block.x %d.\n", grid.x, block.x);

    // Reset device(i.e. destroy all allocations and reset all state).
    cudaDeviceReset();

    return 0;
}