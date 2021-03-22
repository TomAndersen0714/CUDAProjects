#include <stdio.h>

#include "test.h"

int main(int argc, char* argv[]) {


    //test();
    
    // 测试 C++11 之后 NULL,nullptr,(void*)0是否能够比较,是否相同
    test_nullptr();
    // 小结:结果表明 C++11 之后的 NULL,nullptr,(void*)0,0是可以直接进行相互比较的
    // 并且在数值上是相等的.因此在分配指针之后,依旧可以兼容C版本判断指针是否为空的方式,即
    // 判断指针是否为NULL

    return 0;
}