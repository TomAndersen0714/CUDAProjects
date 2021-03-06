#include "cuda_utils.cuh"

#define THREAD_SIZE 16
//__shared__ int shared_arr[THREAD_SIZE]; // static shared memory allocation
// ����������Ϊȫ�ֻ��Ǿֲ��������,Ч��������ͬ��,����Ϊ�˱�������
// ���ǽ��齫���е� Shared Memory �����ڶ�Ӧ�� Device/Global ������


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
    // ������ȫΪ1,������ͬBlock�е�Shared Memory��ͬ
    cudaDeviceSynchronize();
    printf("----------------------------------\n");
    kernal_2 << <3, THREAD_SIZE >> > ();
    // ��������ǰ2��Block�еĽ��ȫΪ2,���һ��Block���ȫΪ1,˵����ͬGrid��Ȼ
    // ���ܻ����ͬһƬShared Memory,��ʹ��Shared Memoryʱ������Ĭ��ֵΪ0
    return 0;
}