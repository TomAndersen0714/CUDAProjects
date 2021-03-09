#include <stdio.h>
#include "test.h"

// static extern void sayHello(void) // error: extern 和 static 无法同时使用
void sayHello(void) {
    printf("Hello World!\n");
}

void test()
{
    printf("This is inline method.\n");
}
