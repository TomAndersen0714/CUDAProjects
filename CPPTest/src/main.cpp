#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void fun() {
    using namespace std;
    cout << "Hello World!" << endl;
}

void sumArrayOnHost(float *a, float *b, float *c, const int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void initialData(float *arr, int count) {
    // Generate different seed for random numbers.
    time_t t;
    srand((unsigned int)time(&t));
    // Generate random numbers
    for (int i = 0; i < count; i++) {
        arr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// 数组在作为参数进行传递时,会自动转换成对应类型的指针,将无法获取其实际长度,只能当成指针进行操作
// 需要手动传入数组长度
void printFloatArray(float arr[], int count) {
    if (arr == NULL) return;
    // int count = sizeof(arr) / sizeof(float);
    for (int i = 0; i < count; i++) {
        printf("%.2f\n", arr[i]);
    }
    printf("\n");
}

int main() {
    // Define size of array.
    int count = 32;
    size_t numberSize = count * sizeof(float);

    // Allocate the memory for float arrays.
    float *h_a = (float *)malloc(numberSize);
    float * h_b = (float *)malloc(numberSize);
    float * h_c = (float *)malloc(numberSize);

    // Initial the float array.
    initialData(h_a, count);
    initialData(h_b, count);

    // Calculate the sum of array.
    sumArrayOnHost(h_a, h_b, h_c, count);
    printFloatArray(h_c, count);

    // Release allocated memory.
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}


