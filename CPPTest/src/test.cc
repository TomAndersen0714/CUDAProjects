#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

    void sayHello(void) {
        printf("Hello World\n");
    }


#ifdef __cplusplus
}
#endif

extern inline void test() {
    printf("This is a inline test function.");
}