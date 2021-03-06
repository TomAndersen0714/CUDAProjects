#include <iostream>
#include <cstdint>
#include <vector>

#include "test.h"

typedef struct _IEEE_FLOAT {
    unsigned int nMantissa : 23;  //β������
    unsigned int nExponent : 8;  //ָ������
    unsigned int nSign : 1;  //����λ
} IEEE_FLOAT;

int main(int argc, char* argv[]) {
    
    printf("%zd\n", sizeof(long));
    printf("%zd\n", sizeof(long long));

    printf("%ld\n", INT32_MAX);
    printf("0x%X\n", INT32_MAX);

    printf("%ld\n", INT32_MIN);
    printf("0x%X\n", INT32_MIN);

    printf("%f\n", static_cast<double>(INT32_MIN));
    printf("0x%X\n", static_cast<double>(INT32_MIN));

    printf("%d\n", static_cast<uint32_t>(5.5));
    printf("0x%X\n", static_cast<uint32_t>(5.5));
    printf("0x%X\n", 5.5);

    printf("(int)5.5 : %d\n", (int)5.5);
    printf("(float)5 : %f\n", (float)5);

    printf("%lld\n", static_cast<int64_t>(INT32_MAX));

    printf("%lld\n", static_cast<int64_t>(INT32_MIN));
    printf("%lld\n", (int64_t)(INT32_MIN));

    float a = 5.5;
    printf("0x%X\n", *(uint32_t *)&a);

    a = 19.625;
    IEEE_FLOAT* p = (IEEE_FLOAT*)&a;
    printf("%d, %#x, %#x\n", p->nSign, p->nExponent - 127, p->nMantissa);

    std::vector<int>a_vector;
    for (int i = 0; i < 30; i++) {
        a_vector.push_back(i);
    }
    std::cout << a_vector.capacity() << '\t' << a_vector.size() << '\t' << std::endl;
    // C++��֧����������(Reference),��C�����в���֧����������
    // PS: ������������ʱ������ *const ָ��,ʹ��ʱ������ָ������� *ptr
    int a1 = 10;
    int &a2 = a1;


    // ����C++�к����Ƿ�Ĭ���� extern ��
    void sayHello(void);
    sayHello();
    // С��:C++�к���Ĭ����extern��,�ۺ�main.c�еĽ���ɵ�,C/C++�к�������Ĭ��Ϊ extern��

    test();
    return 0;
}