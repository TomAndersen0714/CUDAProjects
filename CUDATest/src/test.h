#ifndef _TEST_H_
#define _TEST_H_
#include "cuda_utils.cuh"

// ���� cuda_utils.cuh: checkIndex
void checkDimension();

// ���� PTX ָ���ȡ laneid �� warpid
void test_warpid_and_laneid_instruction();

// ���� __clzll �ڽ�����,ͳ��ǰ�������
void test_clz();

// ���� __ffsll �ڽ�����,���ڻ�ȡ�׸���0bitλ��
void test_ffs();

// ���� cudaMemcpyToSymbol �޸� __constant__ ����
void test_cudaMemcpyToSymbol();


#endif // _TEST_H_