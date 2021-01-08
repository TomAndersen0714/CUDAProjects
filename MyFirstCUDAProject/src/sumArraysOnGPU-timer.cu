#include "cudaTool.cuh"

int main(int argv, char** args) {

    int devNum = 0;
    cudaDeviceProp deviceProp;
    // Get device properties.
    CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, devNum));
    printf("Using the device %d: %s\n", devNum, deviceProp.name);
    // Set the which device(GPU) to use.
    CHECK_ERROR(cudaSetDevice(devNum));

    // Set the device of vector.
    int nElem = 1 << 24;
    printf("Vector size: %d\n", nElem);

    // Malloc for host.
    size_t nBytes = nElem * sizeof(float);

    float *h_a, *h_b, *cpuResult, *gpuResult;
    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    cpuResult = (float*)malloc(nBytes);
    gpuResult = (float*)malloc(nBytes);

    // Initialize the data at host size.
    __int64 clock;
    initialData(h_a, nElem);
    initialData(h_b, nElem);

    // Compute result at host sied.
    clock = unixMillisecondTimestamp();
    sumArraysOnHost(h_a, h_b, cpuResult, nElem);
    printf("sumArraysOnHost elapsed time: %lld ms.\n", unixMillisecondTimestamp() - clock);

    // Malloc for device.
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, nBytes);
    cudaMalloc((float**)&d_b, nBytes);
    cudaMalloc((float**)&d_c, nBytes);

    // Transform data from host to device.
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Invoke kernal at host side.
    int sizeOfBlock = 512;
    dim3 block(sizeOfBlock);
    dim3 grid((nElem + block.x - 1) / block.x);

    clock = unixMillisecondTimestamp();
    sumArraysOnDevice << <grid, block >> > (d_a, d_b, d_c, nElem);
    CHECK_ERROR(cudaDeviceSynchronize());
    printf("sumArraysOnDevice elapsed time: %lld ms.\n", unixMillisecondTimestamp() - clock);

    // Copy compute result from device to host.
    cudaMemcpy(gpuResult, d_c, nBytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    // Check the result.
    checkResult(cpuResult, gpuResult, nElem);

    // Free the allocated memory in device and host.
    free(h_a);
    free(h_b);
    free(cpuResult);
    free(gpuResult);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}