#include "cuda_utils.cuh"

//__global__ void checkIndex(void); 
// С��: ���Խ��˵�� CUDA Ĭ�ϲ�֧���ⲿ����

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
// ���� PTX ָ���ȡ laneid �� warpid

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
// С��: ��PTXָ�����ֱ�ӻ��Warp�е�ǰ laneid,���� Block�е�ǰ�� warpid �ͼ�����
// ��ͬ,��˻��ǽ���ֱ��ʹ�ù�ʽ���м���, PTX ָ���з��ص� warpid ��Ԥ�ڵĲ�ͬ,
// ��Ϊ SM �з���� warpid ������ 1/2 warp(δ�õ�ʵ����֤)
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// ���� __clzll �ڽ�����,ͳ��ǰ�������
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
// С�� __clzll ��������ͳ������8�ֽ�������ǰ�������
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// ���� __ffsll �ڽ�����,���ڻ�ȡ�׸���0bitλ��
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
// С�� __ffsll/__ffs �ڽ��������ص���"�ӵ�λ����λ"�׸���0bit��λ��,���൱��β����ĸ���
// ������Ҫע�����,���������Ϊ0ʱ�򷵻ص���0,����64,����ڵ���֮ǰ��Ҫ�ж���������Ƿ�Ϊ0
// ����������Ϊ0,��ֱ�ӷ���64,������ȥ���ô��ڽ�����
//////////////////////////////////////////////////////////////////////////


//////////////////////////////////////////////////////////////////////////
// ���� cudaMemcpyToSymbol �޸� __constant__ ����
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
// С��: ʹ�� cudaMemcpyToSymbol ʱ,���� src ������ָ�뻹�Ǳ���,����Ҫʹ��ȡ��ַ
// ����"&"����ȡ�����ĵ�ַ; �Ҳ��� dest ������ָ�뻹�Ǳ���,������ֱ�����������,����
// ����ȡ��ַ��"&",��Ȼ����������˵���ݵ���ָ��,intellisen�ᱨ��,���Ǳ�������ͨ��
// �ٷ��ĵ���Ҳ�����ʹ��(PS:CUDA��������е������,�����˲��Ѻ�)
//////////////////////////////////////////////////////////////////////////