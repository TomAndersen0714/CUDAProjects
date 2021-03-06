//#include "cuda_utils.cuh"
#include "test.h"

int main(int argc, char** argv) {
    // 测试使用 PTX 指令集中的指令来获取 warpid 和 laneid
    //test_warpid_and_laneid_instruction();
    // 小结: 此PTX指令可以直接获得Warp中当前 laneid,但是 Block中当前的 warpid 和计算结果
    // 不同,因此还是建议直接使用公式进行计算, PTX 指令中返回的 warpid 与预期的不同,
    // 因为 SM 中分配的 warpid 可能是 1/2 warp(未得到实际验证)

    // 测试 __clzll 内建函数,统计前导零个数
    //test_clz();
    // 小结:__clzll 可以用于统计输入8字节整数的前导零个数

    // 测试 __ffsll 内建函数,统计前导零个数
    //test_ffs();
    // 小结:__ffsll/__ffs 内建函数返回的是"从低位到高位"首个非0bit的位置,即相当于尾端零的个数
    // 但是需要注意的是,当输入参数为0时则返回的是0,而非64,因此在调用之前需要事先手动判断输入参数
    // 是否为0,如果输入参数为0,则直接返回64,否则再去调用此内建函数

    // 测试 cudaMemcpyToSymbol 修改 __constant__ 常量
    test_cudaMemcpyToSymbol();
    // 小结: 使用 cudaMemcpyToSymbol 时,不论 src 参数是指针还是变量,都需要使用取地址
    // 符号"&"来获取变量的地址; 且不论 dest 参数是指针还是变量,都必须直接输入其符号,不能
    // 附加取地址符"&",虽然函数声明中说传递的是指针,intellisen会报错,但是编译依旧通过
    // 官方文档上也是如此使用(PS:CUDA开发真的有点恶心人,对新人不友好)
    return 0;
}