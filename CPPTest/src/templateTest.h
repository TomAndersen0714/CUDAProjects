template<typename T,int n>
class TemplateTest {
public:
    int array[n];
    T a = T();
    void printfTypeSize();
};


template<typename T,int n>
void TemplateTest<T,n>::printfTypeSize()
{
    printf("%lld\n", sizeof(T));
    printf("%lld\n", sizeof(this->a));
}
