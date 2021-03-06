#include "cuda_utils.cuh"

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
#endif // !CHECK_ERROR(call)


void initialData(float *arr, int num) {
    // Generate different seed for random numbers.
    time_t t;
    srand((unsigned int)time(&t));
    // Generate random numbers
    for (int i = 0; i < num; i++) {
        arr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// Since array object will convert to point automatically when it is passwd as 
// a variable, we need pass the length of array concurrently.
void printFloatArray(float arr[], int num) {
    if (arr == NULL) return;
    // int num = sizeof(arr) / sizeof(float);
    for (int i = 0; i < num; i++) {
        printf("%d: %.2f\n", i, arr[i]);
    }
    printf("\n");
}


int main() {
    // Set the device which to use.
    cudaSetDevice(0);

    // Initial the variables.
    int num = 32;
    size_t arrayByteSize = num * sizeof(float);

    // Allocate memory space for arrays in host.
    float *h_a = (float *)malloc(arrayByteSize);
    float * h_b = (float *)malloc(arrayByteSize);
    float * h_c = (float *)malloc(arrayByteSize);

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

    // Calculate sum on device by kernal function.
    sumArraysOnDevice <<<1, num >> > (d_a, d_b, d_c);

    // Let host wait for device finish.
    CHECK_ERROR(cudaDeviceSynchronize());

    // Copy the result from device to host.
    CHECK_ERROR(cudaMemcpy(h_c, d_c, arrayByteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    printFloatArray(h_c, num);

    // Release the memory allocated space.
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    // Return exit signal.
    return 0;
}