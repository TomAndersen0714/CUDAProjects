#include <stdio.h>
#include <math.h>
#include "test.h"


#ifndef __TEST_C__
#define __TEST_C__


__TEST_C__ int fun() {
    //printf("%lld", sizeof(float*));
    return 0;
}
#endif // __TEST_C__

int main() {
    // �������Ĳ����б�Ϊ��ʱ,�����Ǵ����޷��������,
    // ���Ǵ�������ĸ����޷�ȷ��,��ʵ�ʵ���ʱ�ܹ������κ��������κ����͵Ĳ���.
    // Ҫ�뷽�����ƴ������,������Ҫʹ��void
    fun();
    printf("Hello World!\n");
    printf("%f \n", (float)1 / 2);
    printf("%f \n", ceil((float)1 / 2));
    printf("%f \n", floor((float)1 / 2));
    printf("%f \n", round((float)1 / 2));
    printf("%lld\n", unixSecondTimestamp());
    printf("%lld\n", unixMillisecondTimestamp());
    return 0;
}