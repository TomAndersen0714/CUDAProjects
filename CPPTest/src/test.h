#ifndef _TEST_H_
#define _TEST_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    void sayHello(void);

#ifdef __cplusplus
}
#endif

void test();

// 测试 C++11 之后 NULL 和 nullptr是否相同
void test_nullptr();

#endif // _TEST_H_