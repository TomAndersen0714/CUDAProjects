#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "test.h"

#ifndef __TEST_C__
#define __TEST_C__


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
//typedef int(*Compare)(int a, int b);
// ���β�����ʵ���ô�

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

static const char* separator = "------------------------------";

int main(int argc, char* argv[]) {
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
    printf("0x%llX\n", d); // ֱ�Ӵ�ӡ�����������ƶ�Ӧ��16����
    printf("%llu\n", d); // ֱ�Ӱ���unsigned long long��ӡ,Ҳ��ֱ�Ӵ�ӡ��Ӧ�Ķ�������ֵ
    printf("0x%llX\n", ud); // ��������ת����unsigned long long���Ͳ���ӡ16����
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

    // ��֤scanf�����ܷ�ֱ�ӽ�double��Ӧ�ַ�����ȡ��uint64_t����
    // ���ڶ�ȡ�ַ���ת����double��,��ֱ��д���ַ,���ǽ���ʮ����ת����uint64_t�ٽ���д��ַ
    uint64_t lvalue;
    // sscanf����ֱ�ӽ��ַ��������ɶ�Ӧ����ֵ,�����������ֵ��ֵ����Ӧ��ַ�ռ�
    sscanf("5.5", "%lf", &lvalue); // 5.5
    printf("0x%llX\n", lvalue); // 0x4016000000000000
    printf("0x%llX\n", (uint64_t)5.5); // 0x5
    // �������,scanf��ֱ��д���ַ,������ʽת��

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

    return 0;
}