#ifndef _TEST_H_
#define _TEST_H_

#include <string.h>

#include "tools/data_types.h"
#include "tools/compressors.h"
#include "tools/decompressors.h"
#include "tools/io_utils.h"
#include "tools/timer_utils.h"

//////////////////////////////////////////////////////////////////////////
// 测试 io_utils.h
//////////////////////////////////////////////////////////////////////////
static void test_io_utils()
{

    // Declare variables
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
    printDatapoints(dataPoints);

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
// 测试 compressors.h: timestamp_compress_gorilla
// 测试 decompressors.h: timestamp_decompress_gorilla
// 测试 compressors.h: value_compress_gorilla
// 测试 compressors.h: value_decompress_gorilla
//////////////////////////////////////////////////////////////////////////
static void test_gorilla()
{
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "testDataset";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints *dataPoints;
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE; // 测试浮点型值情况

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: timestamp_compress_gorilla
    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    // Read the uncompressed data
    dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType);

    // Print uncompressed data points
    printDatapoints(dataPoints);

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_gorilla(tsByteBuffer);

    printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: timestamp_decompress_gorilla

    ByteBuffer *decompressedTimestamps = timestamp_decompress_gorilla(
        compressedTimestamps,
        dataPoints->count);
    printDecompressedData(decompressedTimestamps, timestampType);
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
    printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_decompress_gorilla

    ByteBuffer *decompressedValues = value_decompress_gorilla(
        compressedValues, dataPoints->count);
    printDecompressedData(decompressedValues, valueType);
    //////////////////////////////////////////////////////////////////////////

    // Free the allocated memory
    fclose(inputFile);
    free(inputFilePath);
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
static void test_rle()
{
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "testDataset";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints *dataPoints;
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE; // 测试浮点型值情况

    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    assert(inputFilePath != NULL);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    // Read the uncompressed data points
    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType);

    // Print the uncompressed data points
    printDatapoints(dataPoints);

    //////////////////////////////////////////////////////////////////////////
    // 测试: timestamp_compress_rle

    // Construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    assert(tsByteBuffer != NULL);
    tsByteBuffer->buffer = (byte *)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count * sizeof(uint64_t);
    tsByteBuffer->capacity = tsByteBuffer->length;
    assert(tsByteBuffer->buffer != NULL);

    // Compress the timestamps of data points
    ByteBuffer *compressedTimestamps =
        timestamp_compress_rle(tsByteBuffer);

    printCompressedData(compressedTimestamps);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试: timestamp_decompress_rle
    ByteBuffer *decompressedTimestamps =
        timestamp_decompress_rle(compressedTimestamps, dataPoints->count);

    printDecompressedData(decompressedTimestamps, timestampType);
    //////////////////////////////////////////////////////////////////////////

    // Free the allocated resources
    fclose(inputFile);
    free(inputFilePath);
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeDataPoints(dataPoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: value_compress_bitpack
// 测试 decompressors.h: value_decompress_bitpack
//////////////////////////////////////////////////////////////////////////
static void test_bitpack()
{
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "testDataset";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints *datapoints;
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE; // 测试浮点型值情况

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_bitpack

    // Read the uncompressed data
    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    assert(inputFilePath != NULL);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    datapoints = readUncompressedFile(
        inputFile, timestampType, valueType);
    // Print uncompressed data points
    printDatapoints(datapoints);

    // Construct the buffer for uncompreessed values
    ByteBuffer *uncompressedValues = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    assert(uncompressedValues != NULL);
    uncompressedValues->length = datapoints->count * sizeof(uint64_t);
    uncompressedValues->capacity = uncompressedValues->length;
    uncompressedValues->buffer = (byte *)datapoints->values;

    // Compress the values of data points
    ByteBuffer *compressedValues = value_compress_bitpack(uncompressedValues);

    printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: value_decompress_bitpack
    ByteBuffer *decompressedValues = value_decompress_bitpack(
        compressedValues, datapoints->count);
    printDecompressedData(decompressedValues, valueType);
    //////////////////////////////////////////////////////////////////////////

    // Free the allocated resources
    fclose(inputFile);
    free(inputFilePath);
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(datapoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 compressors.h: value_compress_bucket
// 测试 decompressors.h: value_decompress_bucket
//////////////////////////////////////////////////////////////////////////
static void test_bucket()
{
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "testDataset";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints *datapoints;
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE; // 测试浮点型值情况

    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_compress_bucket

    // Read compressed data
    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    assert(inputFilePath != NULL);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    datapoints = readUncompressedFile(
        inputFile, timestampType, valueType);
    // Print uncompressed datapoints
    printDatapoints(datapoints);

    // Construct buffer for uncompressed values
    ByteBuffer *uncompressedValues = (ByteBuffer *)malloc(sizeof(ByteBuffer));
    assert(uncompressedValues != NULL);
    uncompressedValues->length = datapoints->count * sizeof(uint64_t);
    uncompressedValues->capacity = uncompressedValues->length;
    uncompressedValues->buffer = (byte *)datapoints->values;

    // Compress values of data points
    ByteBuffer *compressedValues = value_compress_bucket(uncompressedValues);
    printCompressedData(compressedValues);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // 测试 decompressors.h: value_decompress_bucket
    ByteBuffer *decompressedValues = value_decompress_bucket(
        compressedValues, datapoints->count);
    printDecompressedData(decompressedValues, valueType);
    //////////////////////////////////////////////////////////////////////////

    // Free the allocated resources
    fclose(inputFile);
    free(inputFilePath);
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(datapoints);
}

//////////////////////////////////////////////////////////////////////////
// 测试 data_types.h: printStat
//////////////////////////////////////////////////////////////////////////
static void test_statistic()
{
    // Declare variables
    char *base_dir = "dataset/";
    char *dataset = "testDataset3";
    char *inputFilePath;
    FILE *inputFile;
    DataPoints *dataPoints;
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;// 测试整型值情况
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE; // 测试浮点型值情况
    //uint64_t timer, compressionTimeMillis, decompressionTimeMillis;
    clock_t timer, compressionTimeMillis, decompressionTimeMillis;

    // Read the uncompressed data
    inputFilePath = (char *)malloc(strlen(base_dir) + strlen(dataset) + 1);
    strcpy(inputFilePath, base_dir);
    strcat(inputFilePath, dataset);

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType);

    // Print uncompressed data points
    //printDatapoints(dataPoints);


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
        dataPoints->count);
    //printDecompressedData(decompressedTimestamps, timestampType);
    //////////////////////////////////////////////////////////////////////////
    
    //////////////////////////////////////////////////////////////////////////
    // 测试 compressors.h: value_decompress_gorilla
    
    ByteBuffer *decompressedValues = value_decompress_gorilla(
        compressedValues, dataPoints->count);
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
        decompressionTimeMillis);

    //////////////////////////////////////////////////////////////////////////

    // Free the allocated memory
    fclose(inputFile);
    free(inputFilePath);
    freeByteBuffer(compressedTimestamps);
    freeByteBuffer(decompressedTimestamps);
    freeByteBuffer(compressedValues);
    freeByteBuffer(decompressedValues);
    freeDataPoints(dataPoints);
}

#endif // _TEST_H_
