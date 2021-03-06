#include "cuda_utils.cuh"

int main(void) {

    // Set the which device(GPU) to use.
    cudaSetDevice(0);

    // Initial the variables.
    int num = 32;
    size_t arrayByteSize = num * sizeof(float);
    __int64 clock;

    // Allocate memory space for arrays in host.
    float* h_a = (float *)malloc(arrayByteSize);
    float* h_b = (float *)malloc(arrayByteSize);
    float* cupResult = (float *)malloc(arrayByteSize);
    float* gpuResult = (float*)malloc(arrayByteSize);

    // Allocate memory space for arrays in device.
    float *d_a, *d_b, *d_c;
    CHECK_ERROR(cudaMalloc((float**)&d_a, arrayByteSize));
    CHECK_ERROR(cudaMalloc((float**)&d_b, arrayByteSize));
    CHECK_ERROR(cudaMalloc((float**)&d_c, arrayByteSize));

    // Initial the float array.
    initialData(h_a, num);
    initialData(h_b, num);

    // Copy arrays from host to device.The host will be hanged up until this
    // method return.
    CHECK_ERROR(cudaMemcpy(d_a, h_a, arrayByteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, h_b, arrayByteSize, cudaMemcpyKind::cudaMemcpyHostToDevice));

    clock = unixMillisecondTimestamp();
    //printf("%lld\n", unixMillisecondTimestamp());

    // Define the grid and block structure.
    dim3 block(num);
    dim3 grid((num + block.x - 1) / block.x);

    // Compute sum on device by kernal function.
    sumArraysOnDevice << <grid, block >> > (d_a, d_b, d_c);

    // Let host wait for device finish.
    CHECK_ERROR(cudaDeviceSynchronize());

    clock = unixMillisecondTimestamp() - clock;
    //printf("%lld\n", unixMillisecondTimestamp());
    printf("Tht consuming time on device(ms): %lld\n", clock);

    // Compute sum on host.
    sumArrayOnHost(h_a, h_b, cupResult, num);

    // Print the result in host.
    printResult(cupResult, num);
    // Copy the result from device to host.
    CHECK_ERROR(cudaMemcpy(gpuResult, d_c, arrayByteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    // Print the result in device.
    printResult(gpuResult, num);

    // Release the allocated memory space.
    free(h_a);
    free(h_b);
    free(cupResult);
    free(gpuResult);
    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    // Return exit signal.
    return 0;
}