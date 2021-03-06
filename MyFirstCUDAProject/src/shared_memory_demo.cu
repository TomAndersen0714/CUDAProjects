#include "cuda_utils.cuh"

#define THREAD_SIZE 16
//__shared__ int shared_arr[THREAD_SIZE]; // static shared memory allocation
// 不论是声明为全局还是局部共享变量,效果都是相同的,但是为了避免歧义
// 还是建议将所有的 Shared Memory 声明在对应的 Device/Global 函数中


__global__ void kernal_1() {
    __shared__ int shared_arr[THREAD_SIZE];
    int threadIdxOfGrid = threadIdx.x + blockIdx.x*blockDim.x;
    shared_arr[threadIdxOfGrid] += 1;
    printf("%d:%d\n", threadIdxOfGrid, shared_arr[threadIdxOfGrid]);
}

__global__ void kernal_2() {
    __shared__ int shared_arr[THREAD_SIZE];
    int threadIdxOfGrid = threadIdx.x + blockIdx.x*blockDim.x;
    shared_arr[threadIdxOfGrid] += 1;
    printf("%d:%d\n", threadIdxOfGrid, shared_arr[threadIdxOfGrid]);
}

int main(void) {
    kernal_1 << <2, THREAD_SIZE >> > ();
    // 输出结果全为1,表明不同Block中的Shared Memory不同
    cudaDeviceSynchronize();
    printf("----------------------------------\n");
    kernal_2 << <3, THREAD_SIZE >> > ();
    // 输出结果中前2个Block中的结果全为2,最后一个Block结果全为1,说明不同Grid仍然
    // 可能会分配同一片Shared Memory,在使用Shared Memory时不能视默认值为0
    return 0;
}