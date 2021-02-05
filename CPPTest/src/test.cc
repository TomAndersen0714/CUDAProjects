#include <iostream>
#include <cstdint>
#include <vector>

typedef struct _FP_SIGLE {
    unsigned int nMantissa : 23;  //尾数部分
    unsigned int nExponent : 8;  //指数部分
    unsigned int nSign : 1;  //符号位
} FP_SINGLE;

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
    FP_SINGLE* p = (FP_SINGLE*)&a;
    printf("%d, %#x, %#x\n", p->nSign, p->nExponent - 127, p->nMantissa);

    std::vector<int>a_vector;
    for (int i = 0; i < 30; i++) {
        a_vector.push_back(i);
    }
    std::cout << a_vector.capacity() << '\t' << a_vector.size() << '\t' << std::endl;
    // C++中支持引用,而C语言中并不支持引用
    int a1 = 10;
    int &a2 = a1;
}