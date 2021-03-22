#include <stdio.h>

#include "test.h"

int main(int argc, char* argv[]) {
    
    // 测试
    //tests(argc, argv);

    // 测试从字符数组中解析数值
    testParseChars();
    // 小结: 虽然能够通过 sscanf 函数直接解析字符数组char*中的内容,但是这并不像是Java
    // 中的 buffer 对象,并不存在相关变量来控制访问的位置,而C语言中与Java Buffer类似
    // 的结构是 FILE,因为 FILE 的读取和写出行为都和 Java 中的 Buffer 十分类似,有相
    // 关内置变量用于控制读写位置

    return 0;
}