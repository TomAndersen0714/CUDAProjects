#ifndef _TEST_H_
#define _TEST_H_
#include "cuda_utils.cuh"

// 测试 cuda_utils.cuh: checkIndex
void checkDimension();

// 测试 PTX 指令获取 laneid 和 warpid
void test_warpid_and_laneid_instruction();

// 测试 __clzll 内建函数,统计前导零个数
void test_clz();

// 测试 __ffsll 内建函数,用于获取首个非0bit位置
void test_ffs();

// 测试 cudaMemcpyToSymbol 修改 __constant__ 常量
void test_cudaMemcpyToSymbol();


#endif // _TEST_H_