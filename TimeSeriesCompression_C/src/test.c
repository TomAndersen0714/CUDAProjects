#include <string.h>

#include "tools/data_types.h"
#include "tools/compressors.h"
#include "tools/decompressors.h"
#include "tools/io_utils.h"
#include "tools/timer_utils.h"
#include "tools/bucket_counter.h"
#include "test.h"

//////////////////////////////////////////////////////////////////////////
// 测试 io_utils.h
//////////////////////////////////////////////////////////////////////////
void test_io_utils()
{

    // declare
    //char *base_dir = "C:/Users/DELL/Desktop/TSDataset/with timestamps/with abnormal timestamp/ATimeSeriesDataset-master/dataset/";
    char *base_dir = "dataset/";
    char *dataset = "testDataset";
    char *inputFilePath, *outputFilePath;
    FILE *inputFile, *outputFile;
    ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG; // 测试整型值情况
    //ValueType timestampType = _LONG_LONG, valueType = _DOUBLE;// 测试浮点型值情况

    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    //////////////////////////////////////////////////////////////////////////
    // 测试 readUncompressedFile

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);
    DataPoints *dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType);
    // Print the datapoints info
    printDataPoints(dataPoints);

    fclose(inputFile);
    free(inputFilePath);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 writeDecompressedData

    outputFilePath = "tmp/decompressedOutputDataset";
    outputFile = fopen(outputFilePath, "w");
    assert(outputFile != NULL);
    writeDecompressedData(outputFile, dataPoints);

    fclose(outputFile);
    freeDataPoints(dataPoints);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 writeCompressedData

    // Construct a 'CompressedData'
    CompressedData *compressedData = (CompressedData *)malloc(sizeof(CompressedData));
    compressedData->metadata = (Metadata *)malloc(sizeof(Metadata));
    compressedData->timestamps = (byte *)malloc(sizeof(byte) * 16);
    compressedData->values = (byte *)malloc(sizeof(byte) * 16);
    *(uint64_t *)(compressedData->timestamps) = 1519531200000;
    *((uint64_t *)(compressedData->timestamps) + 1) = 1519531260000;
    *(uint64_t *)(compressedData->values) = 96; // 测试整型值情况
    *((uint64_t *)(compressedData->values) + 1) = 134;
    //*(double*)(compressedData->values) = 96.0; // 测试浮点型值情况
    //*((double*)(compressedData->values) + 1) = 134.0;

    // Construct a 'Metadata'
    compressedData->metadata->tsComAndDecom = TS_GORILLA;
    compressedData->metadata->valComAndDecom = VAL_GORILLA;
    compressedData->metadata->timestampType = timestampType;
    compressedData->metadata->valueType = valueType;
    compressedData->metadata->tsLength = 16;
    compressedData->metadata->valLength = 16;
    compressedData->metadata->count = 2;
    printMetadata(compressedData->metadata);

    // Write out the compressed Data
    outputFilePath = "tmp/compressedOutputDataset";
    outputFile = fopen(outputFilePath, "w");
    assert(outputFile != NULL);
    writeCompressedData(outputFile, compressedData);

    fclose(outputFile);
    freeCompressedData(compressedData);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 readCompressedFile
    inputFilePath = "tmp/compressedOutputDataset";
    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);
    compressedData = readCompressedFile(inputFile);
    // Print metadata
    printMetadata(compressedData->metadata);

    fclose(inputFile);
    freeCompressedData(compressedData);
    //////////////////////////////////////////////////////////////////////////
}

//////////////////////////////////////////////////////////////////////////
// 测试 io_utils.h:textToBinary
//////////////////////////////////////////////////////////////////////////
void test_textToBinary()
{
    // declare
    char *inputPath, *ouputPath;

    inputPath = 
        "C:/Users/DELL/Desktop/TSDataset/experiment/IoT7_i";
    ouputPath = 
        "C:/Users/DELL/Desktop/TSDataset/experiment/IoT7_i_b";

    // transform the file
    textToBinary(inputPath, ouputPath, _LONG_LONG, _LONG_LONG, 1);

}

//////////////////////////////////////////////////////////////////////////
// 测试 io_utils.h:readUncompressedFile_b
//////////////////////////////////////////////////////////////////////////
void test_readUncompressedFile_b()
{
    // declare
    DataPoints *dataPoints;
    char *inputPath = "dataset/testDataset3_b";

    // read uncompressed file in binary format
    dataPoints = readUncompressedFile_b(inputPath);

    // print data points
    printDataPoints(dataPoints);

    // free memory
    freeDataPoints(dataPoints);

}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: timestamp_compress_gorilla
// 测试 decompressors.h: timestamp_decompress_gorilla
// 测试 compressors.h: value_compress_gorilla
// 测试 compressors.h: value_decompress_gorilla
//////////////////////////////////////////////////////////////////////////
void test_gorilla()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3_fb";
    DataPoints *dataPoints;
    //uint64_t timer, compressionTimeMillis, decompressionTimeMillis;
    clock_t timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: timestamp_compress_gorilla

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_gorilla(tsByteBuffer);

    //printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////



    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_gorilla

    // Construct the buffer for uncompressed values
    ByteBuffer *valBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    valBuffer->buffer = (byte *)dataPoints->values;
    valBuffer->length = dataPoints->count * sizeof(uint64_t);
    valBuffer->capacity = valBuffer->length;

    // Compress the values of data points
    ByteBuffer *compressedValues = value_compress_gorilla(valBuffer);
    // Print the compressed values
    //printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: timestamp_decompress_gorilla

    ByteBuffer *decompressedTimestamps = timestamp_decompress_gorilla(
        compressedTimestamps,
        dataPoints->count);
    //printDecompressedData(decompressedTimestamps, dataPoints->timestampType);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_decompress_gorilla

    ByteBuffer *decompressedValues = value_decompress_gorilla(
        compressedValues, dataPoints->count);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    printDecompressedData(decompressedTimestamps, dataPoints->timestampType);
    printDecompressedData(decompressedValues, dataPoints->valueType);

    // print the stat info
    printStat(
        dataPoints,
        compressedTimestamps,
        compressedValues,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // Free the allocated memory
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: timestamp_compress_rle
// 测试 decompressors.h: timestamp_decompress_rle
//////////////////////////////////////////////////////////////////////////
void test_rle()
{
    // declare
    const char inputFilePath[] =
        "C:/Users/DELL/Desktop/TSDataset/experiment/integer_timestamp/compression_ratio/IoT1_i_b";
    //"dataset/testDataset3_fb";
    const char dataset[] = "IoT1_i_b";
    DataPoints
        *dataPoints;
    clock_t
        timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);

    //timer = unixMillisecondTimestamp();
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: timestamp_compress_gorilla

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_rle(tsByteBuffer);

    //printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: timestamp_decompress_gorilla

    ByteBuffer *decompressedTimestamps = timestamp_decompress_rle(
        compressedTimestamps, dataPoints->count
    );

    //////////////////////////////////////////////////////////////////////////
    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    printDecompressedData(decompressedTimestamps, dataPoints->timestampType);

    // print stat result
    printf("%s\n", dataset);
    printStat1(
        dataPoints,
        compressedTimestamps,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // print distribution of timestamp sequence
    uint64_t *buk = getBuk();
    uint64_t size = getBukSize();

    for (uint16_t i = 0; i < size; i++) {
        printf("%llu\t", *(buk + i));
    }
    printf("\n");

    // free the allocated memory
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeDataPoints(dataPoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: value_compress_bitpack
// 测试 decompressors.h: value_decompress_bitpack
//////////////////////////////////////////////////////////////////////////
void test_bitpack()
{
    // declare
    char inputFilePath[] =
        "C:/Users/DELL/Desktop/TSDataset/experiment/integer_value/compression_ratio/Server106_i_b";
    DataPoints *dataPoints;
    //uint64_t timer, compressionTimeMillis, decompressionTimeMillis;
    clock_t timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);


    //timer = unixMillisecondTimestamp();
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_bitpack

    // Construct buffer for uncompressed values
    ByteBuffer *uncompressedValues = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    assert(uncompressedValues != NULL);
    uncompressedValues->length = dataPoints->count * sizeof(uint64_t);
    uncompressedValues->capacity = uncompressedValues->length;
    uncompressedValues->buffer = (byte*)dataPoints->values;

    // Compress values of data points
    ByteBuffer *compressedValues = value_compress_bitpack(uncompressedValues);
    //printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: value_decompress_bitpack
    ByteBuffer *decompressedValues = value_decompress_bitpack(
        compressedValues, dataPoints->count);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    //compareByteBuffer(uncompressedValues, decompressedValues, dataPoints->count);

    printDecompressedData(decompressedValues, dataPoints->valueType);

    printStat1(
        dataPoints,
        compressedValues,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // free the allocated memory
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: value_compress_bucket
// 测试 decompressors.h: value_decompress_bucket
//////////////////////////////////////////////////////////////////////////
void test_bucket()
{
    // declare
    char inputFilePath[] = 
        "C:/Users/DELL/Desktop/TSDataset/experiment/float_value/Server66_b";
    DataPoints *dataPoints;
    //uint64_t timer, compressionTimeMillis, decompressionTimeMillis;
    clock_t timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);


    //timer = unixMillisecondTimestamp();
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_bucket

    // Construct buffer for uncompressed values
    ByteBuffer *uncompressedValues = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    assert(uncompressedValues != NULL);
    uncompressedValues->length = dataPoints->count * sizeof(uint64_t);
    uncompressedValues->capacity = uncompressedValues->length;
    uncompressedValues->buffer = (byte*)dataPoints->values;

    // Compress values of data points
    ByteBuffer *compressedValues = value_compress_bucket(uncompressedValues);
    printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: value_decompress_bucket
    ByteBuffer *decompressedValues = value_decompress_bucket(
        compressedValues, dataPoints->count);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    //compareByteBuffer(uncompressedValues, decompressedValues, dataPoints->count);

    printDecompressedData(decompressedValues, dataPoints->valueType);

    printStat1(
        dataPoints,
        compressedValues,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // free the allocated memory
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 data_types.h: printStat
//////////////////////////////////////////////////////////////////////////
void test_statistic()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3_b";
    DataPoints *dataPoints;
    //uint64_t timer, compressionTimeMillis, decompressionTimeMillis;
    clock_t timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);


    //timer = unixMillisecondTimestamp();
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: timestamp_compress_gorilla

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_gorilla(tsByteBuffer);

    //printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_gorilla

    // Construct the buffer for uncompressed values
    ByteBuffer *valBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    valBuffer->buffer = (byte *)dataPoints->values;
    valBuffer->length = dataPoints->count * sizeof(uint64_t);
    valBuffer->capacity = valBuffer->length;

    // Compress the values of data points
    ByteBuffer *compressedValues = value_compress_gorilla(valBuffer);
    // Print the compressed values
    //printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    //compressionTimeMillis = unixMillisecondTimestamp() - timer;
    //timer = unixMillisecondTimestamp();
    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: timestamp_decompress_gorilla

    ByteBuffer *decompressedTimestamps = timestamp_decompress_gorilla(
        compressedTimestamps,
        dataPoints->count
    );
    //printDecompressedData(decompressedTimestamps, timestampType);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_decompress_gorilla

    ByteBuffer *decompressedValues = value_decompress_gorilla(
        compressedValues, dataPoints->count
    );
    //printDecompressedData(decompressedValues, valueType);
    //////////////////////////////////////////////////////////////////////////

    //decompressionTimeMillis = unixMillisecondTimestamp() - timer;
    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    //////////////////////////////////////////////////////////////////////////
    // 测试 data_types.h: printStat
    printStat(
        dataPoints,
        compressedTimestamps,
        compressedValues,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    //////////////////////////////////////////////////////////////////////////

    // Free the allocated memory
    //fclose(inputFile);
    //free(inputFilePath);
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);
}

void test_gorilla_t()
{
    // declare
    const char inputFilePath[] =
        "C:/Users/DELL/Desktop/TSDataset/experiment/integer_value/compression_ratio/Server97_i_b";
        //"dataset/testDataset3_fb";
    const char dataset[] = "Server97_i_b";
    DataPoints
        *dataPoints;
    clock_t
        timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    printDataPoints(dataPoints);

    //timer = unixMillisecondTimestamp();
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: timestamp_compress_gorilla

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_gorilla(tsByteBuffer);

    //printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: timestamp_decompress_gorilla

    ByteBuffer *decompressedTimestamps = timestamp_decompress_gorilla(
        compressedTimestamps, dataPoints->count
    );

    //////////////////////////////////////////////////////////////////////////
    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    printDecompressedData(decompressedTimestamps, dataPoints->timestampType);

    // print stat result
    printf("%s\n", dataset);
    printStat1(
        dataPoints,
        compressedTimestamps,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // free the allocated memory
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeDataPoints(dataPoints);
}

void test_gorilla_v()
{
    // declare
    const char inputFilePath[] =
        "C:/Users/DELL/Desktop/TSDataset/experiment/float_value/compression_ratio/IoT1_b";
    //"dataset/testDataset3_fb";
    const char dataset[] = "IoT1_b";
    DataPoints
        *dataPoints;
    clock_t
        timer, compressionTimeMillis, decompressionTimeMillis;

    // read the uncompressed data in binary format
    dataPoints = readUncompressedFile_b(inputFilePath);

    // Print uncompressed data points
    //printDataPoints(dataPoints);

    timer = clock();


    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_gorilla

    // Construct the buffer for uncompressed values
    ByteBuffer *valBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    valBuffer->buffer = (byte *)dataPoints->values;
    valBuffer->length = dataPoints->count * sizeof(uint64_t);
    valBuffer->capacity = valBuffer->length;

    // Compress the values of data points
    ByteBuffer *compressedValues = value_compress_gorilla(valBuffer);
    // Print the compressed values
    //printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    // print distribution of timestamp sequence
    uint64_t *buk = getBuk();
    uint64_t size = getBukSize();

    for (uint16_t i = 0; i < size; i++) {
        printf("%llu\n", *(buk + i));
    }
    printf("\n");
    clearBuk();


    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_decompress_gorilla

    ByteBuffer *decompressedValues = value_decompress_gorilla(
        compressedValues, dataPoints->count);

    //////////////////////////////////////////////////////////////////////////

    // print distribution of timestamp sequence
    buk = getBuk();
    size = getBukSize();
    for (uint16_t i = 0; i < size; i++) {
        printf("%llu\n", *(buk + i));
    }
    printf("\n");
    clearBuk();

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    //printDecompressedData(decompressedValues, dataPoints->valueType);

    // print stat result
    printf("%s\n", dataset);
    printStat1(
        dataPoints,
        compressedValues,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // Free the allocated memory
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);

}


