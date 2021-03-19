#include "cuda_utils.cuh"

//__global__ void checkIndex(void); 
// 小结: 测试结果说明 CUDA 默认不支持外部链接

void checkDimension() {
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
    // equals to "checkIndex << <(32 + 3 - 1) / 3, 3 >> > ();"

    // Let host wait for device finish.
    cudaDeviceSynchronize();

    // Reset device memory allocation and status.
    cudaDeviceReset();
}

//////////////////////////////////////////////////////////////////////////
// 测试 PTX 指令获取 laneid 和 warpid

// Get lane id within current warp
static inline __device__ unsigned get_lane_id_warp() {
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

// Get warp id within current block
static inline __device__ unsigned get_warp_id_block() {
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// Get lane id within current warp
static inline __device__ unsigned get_lane_id_warp_1() {
    const unsigned int threadIdxBlock = threadIdx.x +
        threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    return threadIdxBlock & 0b11111;
}

// Get warp id within current block
static inline __device__ unsigned int get_warp_id_block_1() {
    const unsigned int threadIdxBlock = threadIdx.x +
        threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    return threadIdxBlock >> 5;
}

__global__ static void ptx_test_kernel() {
    const int actual_warpid = get_warp_id_block();
    const int actual_laneid = get_lane_id_warp();
    //const int expected_warpid = threadIdx.x / 32;
    const int expected_warpid = get_warp_id_block_1();
    //const int expected_laneid = threadIdx.x % 32;
    const int expected_laneid = get_lane_id_warp_1();
    if (expected_laneid == 0) {
        printf("[warp:] actual: %i  expected: %i\n", actual_warpid,
            expected_warpid);
        printf("[lane:] actual: %i  expected: %i\n", actual_laneid,
            expected_laneid);
    }
}

void test_warpid_and_laneid_instruction() {
    dim3 grid(8);
    dim3 block(32 * 2, 2, 2);

    ptx_test_kernel << <grid, block >> > ();
    cudaDeviceSynchronize();
}
// 小结: 此PTX指令可以直接获得Warp中当前 laneid,但是 Block中当前的 warpid 和计算结果
// 不同,因此还是建议直接使用公式进行计算, PTX 指令中返回的 warpid 与预期的不同,
// 因为 SM 中分配的 warpid 可能是 1/2 warp(未得到实际验证)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// 测试 __clzll 内建函数,统计前导零个数
__global__ static void test_clz_kernal() {
    printf(
        "hex: 0x%X,leading zeros: %d,lzc: %d\n",
        threadIdx.x, leadingZerosCount64(threadIdx.x), __clzll(threadIdx.x)
    );
}

void test_clz() {
    dim3 grid(1);
    dim3 block(32);

    test_clz_kernal << <grid, block >> > ();
    cudaDeviceSynchronize();
}
// 小结 __clzll 可以用于统计输入8字节整数的前导零个数
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// 测试 __ffsll 内建函数,用于获取首个非0bit位置
__global__ static void test_ffs_kernal() {
    printf(
        "hex: 0x%X,trailing zeros: %d,ffs: %d\n",
        threadIdx.x, trailingZerosCount64(threadIdx.x), __ffsll(threadIdx.x)
    );
}

void test_ffs() {
    dim3 grid(1);
    dim3 block(32);

    test_ffs_kernal <<<grid, block >>> ();
    cudaDeviceSynchronize();
}
// 小结 __ffsll/__ffs 内建函数返回的是"从低位到高位"首个非0bit的位置,即相当于尾端零的个数
// 但是需要注意的是,当输入参数为0时则返回的是0,而非64,因此在调用之前需要判断输入参数是否为0
// 如果输入参数为0,则直接返回64,否则再去调用此内建函数
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// 测试 cudaMemcpyToSymbol 修改 __constant__ 常量
__constant__ uint32_t c_count;
__constant__ uint32_t* c_offs;
__global__ static void test_cudaMemcpyToSymbol_kernal() {
    printf("%d\n", c_count);
    for (int i = 0; i < c_count; i++) {
        printf("%d ", c_offs[i]);
    }
    printf("\n");
}

void test_cudaMemcpyToSymbol() {
    uint32_t count = 4;
    uint32_t offs[] = { 1,2,3,4 };
    uint32_t *d_offs;
    cudaMalloc((void**)&d_offs, sizeof(uint32_t)*count);
    cudaMemcpy(d_offs, offs, sizeof(uint32_t)*count, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void*));
    cudaMemcpyToSymbol(c_count, &count, sizeof(uint32_t));
    cudaDeviceSynchronize();
    dim3 grid(1);
    dim3 block(1);
    test_cudaMemcpyToSymbol_kernal << <grid, block >> > ();
    cudaDeviceSynchronize();
}
// 小结: 使用 cudaMemcpyToSymbol 时,不论 src 参数是指针还是变量,都需要使用取地址
// 符号"&"来获取变量的地址; 且不论 dest 参数是指针还是变量,都必须直接输入其符号,不能
// 附加取地址符"&",虽然函数声明中说传递的是指针,intellisen会报错,但是编译依旧通过
// 官方文档上也是如此使用(PS:CUDA开发真的有点恶心人,对新人不友好)
//////////////////////////////////////////////////////////////////////////