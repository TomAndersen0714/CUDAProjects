#include "compressors.h"
#include "decompressors.h"
#include "utils/data_types.h"
#include "utils/io_utils.h"
#include "utils/timer_utils.h"


/**
 * 测试 gorilla.cu: timestamp_compress_gorilla_gpu
 * 测试 gorilla.cu: timestamp_decompress_gorilla_gpu
 */
void test_timestamp_compress_gorilla_gpu() {
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "IoT1";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints* dataPoints;
    ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    //ValueType timestampType = _LONG_LONG, valueType = _DOUBLE;// 测试浮点型值情况

    //////////////////////////////////////////////////////////////////////////
    // 测试 gorilla.cu: timestamp_compress_gorilla_gpu
    inputFilePath = (char*)malloc(strlen(base_dir) + strlen(dataset) + 1);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    // Read the uncompressed data
    dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType
    );

    // Print uncompressed data points
    printDatapoints(dataPoints);

    // Construct the buffer for uncompressed timestamps
    ByteBuffer* tsByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte*)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);

    // Compress the data points
    timestamp_compress_gorilla_gpu(tsByteBuffer,1,16);

    // Print the compressed data

    //////////////////////////////////////////////////////////////////////////


}