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
// ����msvc��Ԥ���������ʹ��

__TEST_C__ int fun(void) {
    //printf("%lld", sizeof(float*));
    return 0;
}

// ����һ������ָ������compare,�˺��������б������Ϊ(int,int),����ֵ����Ϊint.��������ͬ
// �����б�����,�Լ���ͬ�ķ���ֵ���͵ĺ��������ڴ�����
// ʹ��comprare�����ı�����Ϊ�˺���ָ������,�� Compare fp = &func;
typedef int(*Compare)(int, int);
// �ȼ���:
//typedef int _Compare(int,int);
//typedef _Compare* Compare;
// Ҳ����д��
//typedef int(*Compare)(int a, int b); // ���β�����ʵ���ô�,����ʡ��

// ����һ������ָ�����func,�˺��������б������Ϊ(void),����ֵ����Ϊint.
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
    // ����C������ const �ؼ�������ָ���ʹ�÷�ʽ
    const int *ptr1;
    int *const ptr2 = &argc; // const ���͵ı�������ָ����ʼֵ
    //*ptr1 = 3; // �����޷�ͨ��ptr1�޸Ķ�Ӧ��intֵ
    ptr1 = &argc;// ���������޸�ptr1��ָ��
    *ptr2 = 3; // ��������ͨ��ptr2�޸Ķ�Ӧ��intֵ
    //ptr2 = &argc; // �����������޸�ptr2��ָ��
    // С��: ʹ��const����ָ��ʱ,Ĭ������������,���û���������Ҳ�,��������η�����*���ʾ
    // ָ���ָ�򲻿ɱ�(ָ��������ɱ�),�������ʾָ���ֵ���ɱ�

    // ����int32_t������������uint32_t����������֮���໥ת���Ƿ�ᶪʧ����(��int32_t�������ֵ��Χ��)
    int32_t int32_t_1 = 12;
    uint32_t uint32_t_1 = (uint32_t)int32_t_1;
    printf("%ld %lu\n", int32_t_1, uint32_t_1);
    // ���Խ����ʾ,��δ����int32_t���͵ķ�Χʱ,int32_t �� uint32_t ֮���ܹ�ֱ�ӽ���ת��

    puts(separator);

    // ���� cstbool.h(C99) �� bool ��ʵ��ֵ
    bool bit1 = 1 == 1;
    bool bit2 = false;
    bool bit3 = 11 & 1;
    bool bit4 = -1;
    printf("%d %d %d %d\n", bit1, bit2, bit3, bit4);
    // ���Խ������,��C������ bool ʵ����ʹ�� 0 ��ʾfalse,ʹ�÷�0��ʾtrue

    puts(separator);

    // �����޸�*ptr��Ӧֵʱ,�Ƿ��Ӱ��ptrָ���ԭʼ����ֵ
    int a = 10, *ptr = &a;
    *ptr = 11;
    printf("%d\n", a); // 11
    // ���Խ������,�޸�*ptr��ֵ��ı�ptrָ���ԭʼ����ֵ

    puts(separator);

    // �������Ĳ����б�Ϊ��ʱ,�����Ǵ����޷��������,
    // ���Ǵ�������ĸ����޷�ȷ��,��ʵ�ʵ���ʱ�ܹ������κ��������κ����͵Ĳ���.
    // Ҫ�뷽����ֹ�������,������Ҫʹ��void
    fun();
    printf("%lld\n", sizeof(&fun)); // ��ӡ������ַ�ĳ���(ȡ��ַ����&������ʡ��)
    func = &fun; // ȡ��ַ����&����ʡ��
    func();
    printf("Hello World!\n");

    puts(separator);

    // ���Ը������ܷ�ʹ����λ�����
    //printf("%lf", 5.5 >> 2);
    // �޷�ͨ������,���������޷�ֱ�ӽ�����λ����

    // ���Ը�����������ת�����޷�������
    double d = 5.5;
    uint64_t ud = *((uint64_t*)&d);
    printf("0x%016llX\n", d); // ֱ�Ӵ�ӡ�����������ƶ�Ӧ��16����
    printf("%llu\n", d); // ֱ�Ӱ���unsigned long long��ӡ,Ҳ��ֱ�Ӵ�ӡ��Ӧ�Ķ�������ֵ
    printf("0x%016llX\n", ud); // ��������ת����unsigned long long���Ͳ���ӡ16����
    printf("%llu\n", ud); // ��������ת����unsigned long long���ͽ��д�ӡ

    puts(separator);

    // ����ungetc������ʹ��
    /*char buffer[BUFSIZ];
    setbuf(stdin, buffer);
    strcpy(buffer, (char*)"5.5");*/
    ungetc('5', stdin);
    putchar(getchar());
    ungetc('5', stdin);
    ungetc('.', stdin); // ungetc����ÿ��ֻ��ͬʱ���뵥���ַ�,���Ҳ�֧����������
    // ��������ungetc�����������ַ�Ҳ�޷���ӵ���׼���뻺������ͷ��
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

    // ��֤��Windows-Dos����̨��,�û��س����������'\n',����"\r\n"
    char c;
    while ((c = getchar()) != '#') {// ��'#'����������
        if (c == '\r') puts("r");
        else if (c == '\n') puts("n");
        else putchar(c);
        //putchar('\n');
    }
    //putchar('\r');
    //putchar('\r\n');
    //putchar('\n');

    puts(separator);

    // ��֤sscanf�����ܷ�ֱ�ӽ�double��Ӧ�ַ�����ȡ��uint64_t����
    // ���ڶ�ȡ�ַ���ת����double��,��ֱ��д���ַ,���ǽ���ʮ����ת����uint64_t�ٽ���д��ַ
    uint64_t lvalue;
    // sscanf����ֱ�ӽ��ַ��������ɶ�Ӧ����ֵ,�����������ֵ��ֵ����Ӧ��ַ�ռ�
    sscanf("5.5", "%lf", &lvalue); // 5.5
    printf("0x%llX\n", lvalue); // 0x4016000000000000
    printf("0x%llX\n", (uint64_t)5.5); // 0x5
    // �������,sscanf���Ƚ��ַ�����ʽ���ɶ�Ӧ���͵���ֵ,Ȼ����ֵ�Ķ�����д�뵽ָ����ַ��
    // ������ʽǿ��ת��

    puts(separator);

    // ��֤printf�����ܷ�ֱ�ӽ�uint64_t��ӡ��double
    // ���ǰ��ն�����ת����double���д�ӡ,���ǰ���ʮ����ת����double���д�ӡ
    lvalue = 0x4016000000000000;
    printf("%lf\n", lvalue); // 5.500000
    printf("%lf\n", 0x4016000000000000); // 5.500000
    printf("%lf\n", (double)5); // 5.000000
    printf("%lf\n", *((double*)&lvalue)); // 5.500000
    // �������printf������ӡ�ַ���ʱ,�����ǽ���Ӧֵ���ն�����ת���������ʽ���д�ӡ
    // PS:���ǻ�δŪ���,����������������ڶ����Ʋ������ת����,printf�����ܹ���
    // ��������Ϊʵ�θ�ֵ���β�,Ȼ������ָ��ת���ķ�ʽ���ж���������ת��,��Ŀǰ����֪
    // �����ֱ�ӽ���ת��

    // ����C�����к����Ƿ�Ĭ���� extern ��
    void sayHello(void);
    sayHello();
    // С��:C�к���Ĭ���� extern
    // �ۺ�main.cc�еĽ���ɵ�:C/C++�к�������Ĭ��Ϊ extern��

    puts(separator);

    // ���� string.h �ַ���ƴ�Ӻ��� strcat
    char *str1 = "Hello ", *str2 = "World!";
    char *str3 = (char *)malloc(strlen(str1) + strlen(str2) + 1);
    strcpy(str3, str1);
    strcat(str3, str2);
    printf("%d\t%d\n", strlen(str1), strlen(str2));
    printf("%s\n", str3);
    // С��: strlen �������ص����ַ����׸���'\0'�ַ�֮ǰ���ַ�����,�Ҳ����������ַ�

    // �����ǰ��ִ���ļ�����·��
    printf("%s\n", argv[0]);

    // �����ǰ�Ĺ���·��(direct.h)
    char* workDir = malloc(1024);
    _getcwd(workDir, 1024);
    printf("%s\n", workDir);
    // С��:��ִ���ļ�����·������һ�����ǹ���·��

    // ����C���Ƿ��ܹ�ͨ�����������ķ�ʽ�����ʵ�ַ
    printf("%c\t%c\n", workDir[0], workDir[1]);
    // С��:���Խ��˵��,���Զ�ָ��ʹ�����������ķ�ʽ���з���,Ĭ��ÿ��Ԫ�صĳ���Ϊ
    // ָ�������Ӧ���͵ĳ���

    // ���Ը�������ֵ�� unsigned �Ƿ�ᶪʧ����
    int64_t x = -10;
    uint64_t y = x;
    printf("0x%llX\t0x%llX\n", x, y);
    // С��:��������ֵ����Ӧ���� unsigned ����ʱ,���ᶪʧ����

    puts(separator);

    // ���Ե�ǰ�����Ǵ�˴洢����С�˴洢
    // PS:С�˴洢Ϊ�͵�ַ��ŵ�λ�ֽ�,�ߵ�ַ��Ÿ�λ�ֽ�,��˴洢���෴
    MyUnion u;
    u.i = 0x2211;
    if (u.a[0] == 0x22) {
        printf("This is big-endian mode.\n");
    }
    else {
        printf("This is little-endian mode.\n");
    }
    // С��:��ǰ����ΪС�˴洢(little-endian),��˽�һ�����ְ��յ�ַ˳������ֽ�
    // ʱ,���ǵ�λ�ֽ���ǰ,��λ�ֽ��ں�

    // ���Զ�һ���ڴ������ͷ����λᱨʲô��
    char* chars = (char*)malloc(16);
    free(chars);
    //free(chars);
    // С��: ���׳��쳣 _CrtlsValidHeapPointer(block)

    printf("%llu\n", 1ULL);


    return 0;
}