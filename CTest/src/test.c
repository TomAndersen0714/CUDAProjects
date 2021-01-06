#include <stdio.h>
#include <math.h>

#ifndef __TEST_C__
#define __TEST_C__

__TEST_C__ int fun(void) {
    //printf("%lld", sizeof(float*));
    return 0;
}
#endif // __TEST_C__

int main() {
    // 当函数的参数列表为空时,并不是代表无法传入参数,而是代表参数的个数无法确定.要想
    // 方法进制传入参数,还是需要使用void
    fun();
    printf("Hello World!\n");
    printf("%f \n", (float)1 / 2);
    printf("%f \n", ceil((float)1 / 2));
    printf("%f \n", floor((float)1 / 2));
    printf("%f \n", round((float)1 / 2));
    return 0;
}