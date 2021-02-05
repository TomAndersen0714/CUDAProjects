// Used for C/C++ test.
#include <stdio.h>
#include <inttypes.h>

#include "../tools/encode_tool.h"

#include "test.h"

int main(int argc, char* argv[]) {
    
    //extern int a;

    for (int i = -30; i < 30; i++) {
        a = a + i;
        int64_t encodedNum = encodeZigZag64(i);

        printf("%I32d\t", i);
        printf("%I64d\t", encodedNum);
        printf("%I64d\n", decodeZigZag64(encodedNum));

    }
}