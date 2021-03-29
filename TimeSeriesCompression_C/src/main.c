#include "tools/data_types.h"
#include "tools/compressors.h"
#include "tools/decompressors.h"
#include "tools/io_utils.h"
#include "test.h"

int main(int argc, char* argv[]) {

    // 测试 io_utils.h
    //test_io_utils();

    // 使用 io_utils.h:readUncompressedFile 转换测量值数据类型
    //test_readUncompressedFile();

    // 测试 io_utils.h:textToBinary
    //test_textToBinary();

    // 测试 io_utils.h:readUncompressedFile_b
    //test_readUncompressedFile_b();

    
    
    // 测试 RLE 算法
    //test_rle();

    // 测试 BitPack 算法
    //test_bitpack();

    // 测试 Bucket 算法
    //test_bucket();


    // 测试 timer_utils.h
    /*printf("%llu\n", unixMillisecondTimestamp());
    printf("%llu\n", unixSecondTimestamp());*/

    // 测试 data_types.h: printStat
    //test_statistic();

    // 测试 Gorilla 算法
    //test_gorilla();
    // 测试 gorilla 时间戳压缩性能
    //test_gorilla_t();
    // 测试 gorilla 测量值压缩性能
    test_gorilla_v();

    return 0;
}
