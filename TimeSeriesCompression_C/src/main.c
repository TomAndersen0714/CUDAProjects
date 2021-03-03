#include "tools/data_types.h"
#include "tools/compressors.h"
#include "tools/decompressors.h"
#include "tools/io_utils.h"
#include "test.h"

int main(int argc, char* argv[]) {

    // ���� io_utils.h
    //test_io_utils();

    // ���� Gorilla �㷨
    //test_gorilla();

    // ���� RLE �㷨
    //test_rle();

    // ���� BitPack �㷨
    //test_bitpack();

    // ���� Bucket �㷨
    //test_bucket();


    // ���� timer_utils.h
    /*printf("%llu\n", unixMillisecondTimestamp());
    printf("%llu\n", unixSecondTimestamp());*/

    // ����ͳ����Ϣ��ӡ
    test_statistic();

    return 0;
}
