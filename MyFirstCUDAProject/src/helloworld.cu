#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>

// Define a kernal function by "__global__" keyword.
__global__ void helloFromGPU() {
    printf("Hello World! This is CPU(Host) thread %d .\n", threadIdx.x);
}

int main(void) {
    using namespace std;
    cout << "Hello world from CPU!\n" << endl;
    // Say "Hello World" from CPU(Host).
    printf("Hello world from CPU!\n");

    // Define a cuda status variable.
    cudaError_t cudaStatus;

    // Launch a kernal function on GPU(Device).
    helloFromGPU << <1, 10 >> > ();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addVectorKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaError::cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addVectorKernel!\n", cudaStatus);
        return 1;
    }

    return 0;
}