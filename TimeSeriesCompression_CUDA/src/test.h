#ifndef _TEST_H_
#define _TEST_H_

/**
 * TEST io_utils.cpp: writeCompressedData
 * TEST io_utils.cpp: readCompressedFile
 */
void test_io_utils();


/**
* TEST gorilla.cu: timestamp_compress_gorilla_gpu
* TEST gorilla.cu: timestamp_decompress_gorilla_gpu
* TEST gorilla.cu: value_compress_gorilla_gpu
* TEST gorilla.cu: value_decompress_gorilla_gpu
*/
void test_compress_gorilla_gpu();



/**
 * TEST data_types.h:printStat
 */
void test_printStat();


#endif // _TEST_H_