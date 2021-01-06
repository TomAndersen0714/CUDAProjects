#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void sumArraysOnDevice(float *a, float *b, float *c, const int num) {
    for (int i = 0; i < num; i++) {
        c[i] = a[i] + b[i];
    }
}

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
        printf("%.2f\n", arr[i]);
    }
    printf("\n");
}


int main() {
    // Initial the variables.
    int num = 32;
    size_t arrayByteSize = num * sizeof(float);

    // Allocate memory space for arrays in host.
    float *h_a = (float *)malloc(arrayByteSize);
    float * h_b = (float *)malloc(arrayByteSize);
    float * h_c = (float *)malloc(arrayByteSize);

    // Allocate memory space for arrays in device.
    float *d_a, *d_b, *d_c;
    cudaMalloc((float**)&d_a, arrayByteSize);
    cudaMalloc((float**)&d_b, arrayByteSize);
    cudaMalloc((float**)&d_c, arrayByteSize);

    // Initial the float array.
    initialData(h_a, num);
    initialData(h_b, num);

    // Copy arrays from host to device.The host will be hanged up until this
    // method return.
    cudaMemcpy(d_a, h_a, arrayByteSize, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, arrayByteSize, cudaMemcpyKind::cudaMemcpyHostToDevice);

    // Calculate sum on device by kernal function.
    //
    //

    // Copy the result from device to host.
    cudaMemcpy(h_c, d_c, arrayByteSize, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    printFloatArray(h_c, num);

    // Release the memory allocated space.
    free(d_a);
    free(d_b);
    free(d_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Return exit signal.
    return 0;
}