#include <stdio.h>
#include "test.h"

// static extern void sayHello(void) // error: extern �� static �޷�ͬʱʹ��
void sayHello(void) {
    printf("Hello World!\n");
}

void test()
{
    printf("This is inline method.\n");
}
