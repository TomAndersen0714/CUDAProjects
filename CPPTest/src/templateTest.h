#pragma once


template<typename T>
class TemplateTest {
public:
    T a = T();
    void printfTypeSize();
};


template<typename T>
void TemplateTest<T>::printfTypeSize()
{
    printf("%lld\n", sizeof(T));
    printf("%lld\n", sizeof(this->a));
}
