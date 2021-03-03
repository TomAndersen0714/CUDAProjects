#include "tools/data_types.h"
#include "tools/compressors.h"
#include "tools/decompressors.h"
#include "tools/io_utils.h"
#include "test.h"

int main(int argc, char* argv[]) {

    // ≤‚ ‘ io_utils.h
    //test_io_utils();

    // ≤‚ ‘ Gorilla À„∑®
    //test_gorilla();

    // ≤‚ ‘ RLE À„∑®
    //test_rle();

    // ≤‚ ‘ BitPack À„∑®
    //test_bitpack();

    // ≤‚ ‘ Bucket À„∑®
    //test_bucket();


    // ≤‚ ‘ timer_utils.h
    /*printf("%llu\n", unixMillisecondTimestamp());
    printf("%llu\n", unixSecondTimestamp());*/

    // ≤‚ ‘Õ≥º∆–≈œ¢¥Ú”°
    test_statistic();

    return 0;
}
