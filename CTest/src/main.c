#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <direct.h>
#include "time_helper.h"

#ifndef __TEST_C__
#define __TEST_C__
//#pragma message ( "your warning text here" )
// 测试msvc中预处理命令的使用

__TEST_C__ int fun(void) {
    //printf("%lld", sizeof(float*));
    return 0;
}

// 定义一个函数指针类型compare,此函数参数列表的类型为(int,int),返回值类型为int.即具有相同
// 参数列表类型,以及相同的返回值类型的函数都属于此类型
// 使用comprare声明的变量皆为此函数指针类型,如 Compare fp = &func;
typedef int(*Compare)(int, int);
// 等价于:
//typedef int _Compare(int,int);
//typedef _Compare* Compare;
// 也可以写成
//typedef int(*Compare)(int a, int b); // 但形参名无实际用处,可以省略

// 声明一个函数指针变量func,此函数参数列表的类型为(void),返回值类型为int.
int(*func)(void);

typedef struct _stu {
    int num;
    char name[5];
} student;

void printStudentName(student s) {
    s.num = 5;
    printf("%p\n", &s);
    printf("%d\n", s.num);
}

typedef enum _numbers {
    one, two
}numbers;
#endif // __TEST_C__

typedef union _MyUnion
{
    short int i;
    char a[2];
}MyUnion;

static const char* separator = "------------------------------";

int main(int argc, char* argv[]) {
    // 测试C语言中 const 关键字修饰指针的使用方式
    const int *ptr1;
    int *const ptr2 = &argc; // const 类型的变量必须指定初始值
    //*ptr1 = 3; // 表明无法通过ptr1修改对应的int值
    ptr1 = &argc;// 表明可以修改ptr1的指向
    *ptr2 = 3; // 表明可以通过ptr2修改对应的int值
    //ptr2 = &argc; // 表明不可以修改ptr2的指向
    // 小结: 使用const修饰指针时,默认修饰左侧符号,如果没有则修饰右侧,如果被修饰符号是*则表示
    // 指针的指向不可变(指针变量不可变),否则则表示指向的值不可变

    // 测试int32_t类型正整数和uint32_t类型正整数之间相互转换是否会丢失精度(在int32_t类型最大值范围内)
    int32_t int32_t_1 = 12;
    uint32_t uint32_t_1 = (uint32_t)int32_t_1;
    printf("%ld %lu\n", int32_t_1, uint32_t_1);
    // 测试结果显示,当未超过int32_t类型的范围时,int32_t 和 uint32_t 之间能够直接进行转换

    puts(separator);

    // 测试 cstbool.h(C99) 中 bool 的实际值
    bool bit1 = 1 == 1;
    bool bit2 = false;
    bool bit3 = 11 & 1;
    bool bit4 = -1;
    printf("%d %d %d %d\n", bit1, bit2, bit3, bit4);
    // 测试结果表明,在C语言中 bool 实际上使用 0 表示false,使用非0表示true

    puts(separator);

    // 测试修改*ptr对应值时,是否会影响ptr指向的原始变量值
    int a = 10, *ptr = &a;
    *ptr = 11;
    printf("%d\n", a); // 11
    // 测试结果表明,修改*ptr的值会改变ptr指向的原始变量值

    puts(separator);

    // 当函数的参数列表为空时,并不是代表无法传入参数,
    // 而是代表参数的个数无法确定,在实际调用时能够接收任何数量的任何类型的参数.
    // 要想方法禁止传入参数,还是需要使用void
    fun();
    printf("%lld\n", sizeof(&fun)); // 打印函数地址的长度(取地址符号&不可以省略)
    func = &fun; // 取地址符号&可以省略
    func();
    printf("Hello World!\n");

    puts(separator);

    // 测试浮点数能否使用移位运算符
    //printf("%lf", 5.5 >> 2);
    // 无法通过编译,即浮点数无法直接进行移位运算

    // 测试浮点数二进制转换成无符号整数
    double d = 5.5;
    uint64_t ud = *((uint64_t*)&d);
    printf("0x%016llX\n", d); // 直接打印浮点数二进制对应的16进制
    printf("%llu\n", d); // 直接按照unsigned long long打印,也能直接打印对应的二进制数值
    printf("0x%016llX\n", ud); // 将浮点数转换成unsigned long long类型并打印16进制
    printf("%llu\n", ud); // 将浮点数转换成unsigned long long类型进行打印

    puts(separator);

    // 测试ungetc函数的使用
    /*char buffer[BUFSIZ];
    setbuf(stdin, buffer);
    strcpy(buffer, (char*)"5.5");*/
    ungetc('5', stdin);
    putchar(getchar());
    ungetc('5', stdin);
    ungetc('.', stdin); // ungetc函数每次只能同时插入单个字符,并且不支持连续调用
    // 连续调用ungetc函数后续的字符也无法添加到标准输入缓冲区的头部
    //ungetc('5', stdin);
    putchar('\n');
    putchar('1');
    putchar(getchar());
    putchar('\n');
    putchar('2');
    putchar(getchar());
    putchar('\n');
    putchar('3');
    putchar(getchar());
    putchar('\n');
    putchar('4');
    putchar(getchar());

    // 验证在Windows-Dos控制台中,敲击回车键输入的是'\n',而非"\r\n"
    char c;
    while ((c = getchar()) != '#') {// 按'#'键结束输入
        if (c == '\r') puts("r");
        else if (c == '\n') puts("n");
        else putchar(c);
        //putchar('\n');
    }
    //putchar('\r');
    //putchar('\r\n');
    //putchar('\n');

    puts(separator);

    // 验证sscanf函数能否直接将double对应字符串读取成uint64_t变量
    // 即在读取字符串转换成double后,是直接写入地址,还是将其十进制转换成uint64_t再进行写地址
    uint64_t lvalue;
    // sscanf函数直接将字符串解析成对应的数值,并将其二进制值赋值到对应地址空间
    sscanf("5.5", "%lf", &lvalue); // 5.5
    printf("0x%llX\n", lvalue); // 0x4016000000000000
    printf("0x%llX\n", (uint64_t)5.5); // 0x5
    // 结果表明,sscanf是先将字符串格式化成对应类型的数值,然后将数值的二进制写入到指定地址中
    // 而非显式强制转换

    puts(separator);

    // 验证printf函数能否直接将uint64_t打印成double
    // 即是按照二进制转换成double进行打印,还是按照十进制转换成double进行打印
    lvalue = 0x4016000000000000;
    printf("%lf\n", lvalue); // 5.500000
    printf("%lf\n", 0x4016000000000000); // 5.500000
    printf("%lf\n", (double)5); // 5.000000
    printf("%lf\n", *((double*)&lvalue)); // 5.500000
    // 结果表明printf函数打印字符串时,依旧是将对应值按照二进制转换成输出格式进行打印
    // PS:但是还未弄清楚,字面量常量是如何在二进制层面进行转换的,printf函数能够将
    // 字面量作为实参赋值给形参,然后上述指针转换的方式进行二进制类型转换,但目前还不知
    // 道如何直接进行转换

    // 测试C语言中函数是否默认是 extern 的
    void sayHello(void);
    sayHello();
    // 小结:C中函数默认是 extern
    // 综合main.cc中的结果可得:C/C++中函数声明默认为 extern的

    puts(separator);

    // 测试 string.h 字符串拼接函数 strcat
    char *str1 = "Hello ", *str2 = "World!";
    char *str3 = (char *)malloc(strlen(str1) + strlen(str2) + 1);
    strcpy(str3, str1);
    strcat(str3, str2);
    printf("%d\t%d\n", strlen(str1), strlen(str2));
    printf("%s\n", str3);
    // 小结: strlen 函数返回的是字符串首个非'\0'字符之前的字符个数,且并不包括空字符

    // 输出当前可执行文件绝对路径
    printf("%s\n", argv[0]);

    // 输出当前的工作路径(direct.h)
    char* workDir = malloc(1024);
    _getcwd(workDir, 1024);
    printf("%s\n", workDir);
    // 小结:可执行文件所在路径并不一定就是工作路径

    // 测试C中是否能够通过数组索引的方式来访问地址
    printf("%c\t%c\n", workDir[0], workDir[1]);
    // 小结:测试结果说明,可以对指针使用数组索引的方式进行访问,默认每个元素的长度为
    // 指针变量对应类型的长度

    // 测试负整数赋值给 unsigned 是否会丢失精度
    int64_t x = -10;
    uint64_t y = x;
    printf("0x%llX\t0x%llX\n", x, y);
    // 小结:当负数赋值给对应类型 unsigned 变量时,不会丢失精度

    puts(separator);

    // 测试当前机器是大端存储还是小端存储
    // PS:小端存储为低地址存放低位字节,高地址存放高位字节,大端存储则相反
    MyUnion u;
    u.i = 0x2211;
    if (u.a[0] == 0x22) {
        printf("This is big-endian mode.\n");
    }
    else {
        printf("This is little-endian mode.\n");
    }
    // 小结:当前机器为小端存储(little-endian),因此将一个数字按照地址顺序输出字节
    // 时,将是低位字节在前,高位字节在后

    // 测试对一块内存连续释放两次会报什么错
    char* chars = (char*)malloc(16);
    free(chars);
    //free(chars);
    // 小结: 会抛出异常 _CrtlsValidHeapPointer(block)

    printf("%llu\n", 1ULL);


    return 0;
}