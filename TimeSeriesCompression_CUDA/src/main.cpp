#include "test.h"

int main(int argc, char* argv[]) {

    // TEST:io_utils
    //test_io_utils();
    

    // TEST gorilla.cu
    //test_compress_gorilla_gpu();

    // TEST data_types.h:printStat
    //test_printStat();

    
    //TEST data_types.h: compactData
    //TEST io_utils.h: readUncompressedFile_b
    //test_compactData();

    
    // TEST rle.cu: timestamp_compress_rle_gpu
    // TEST rle.cu: timestamp_decompress_rle_gpu    
    //test_compress_rle_gpu();

    // TEST bitpack.cu: timestamp_compress_bitpack_gpu
    // TEST bitpack.cu : timestamp_decompress_bitpack_gpu
    //test_compress_bitpack_gpu();

    //TEST bucket.cu: value_compress_bucket_gpu
    //TEST bucket.cu : value_decompress_bucket_gpu
    test_compress_bucket_gpu();

    return 0;
}