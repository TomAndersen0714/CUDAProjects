#include <string.h>

#include "compressors.h"
#include "decompressors.h"
#include "utils/data_types.h"
#include "utils/io_utils.h"
#include "utils/timer_utils.h"
#include "test.h"

/**
* TEST:io_utils
*/
void test_io_utils() {
    // declare
    char 
        *inputFilePath = "dataset/testDataset", 
        *outputFilePath;
    FILE *inputFile, *outputFile;
    ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG; // TEST整型值情况
    // ValueType timestampType = _LONG_LONG, valueType = _DOUBLE;// TEST浮点型值情况

    //////////////////////////////////////////////////////////////////////////
    // TEST:readUncompressedData
    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);
    DataPoints *dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType
    );
    // print the datapoints info
    printDatapoints(dataPoints);

    fclose(inputFile);
    //free(inputFilePath);

    //////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////
    // TEST:writeDecompressedData
    outputFilePath = "tmp/decompressedTest";
    outputFile = fopen(outputFilePath, "w");
    assert(outputFile != NULL);

    writeDecompressedData(outputFile, dataPoints);

    fclose(outputFile);
    freeDataPoints(dataPoints);
    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // TEST:writeCompressedData

    outputFilePath = "tmp/compressedTest";
    outputFile = fopen(outputFilePath, "w");
    assert(outputFile != NULL);

    // construct compressed data
    CompressedDPs *compressedDPs =
        (CompressedDPs*)malloc(sizeof(CompressedDPs));
    assert(compressedDPs != NULL);

    // construct metadata
    compressedDPs->metadata = (Metadata *)malloc(sizeof(Metadata));
    assert(compressedDPs->metadata != NULL);
    compressedDPs->metadata->count = 4;
    compressedDPs->metadata->frame = 4;
    compressedDPs->metadata->tsComAndDecom = TS_GORILLA;
    compressedDPs->metadata->valComAndDecom = VAL_GORILLA;
    compressedDPs->metadata->timestampType = timestampType;
    compressedDPs->metadata->valueType = valueType;

    // print metadata
    printMetadata(compressedDPs->metadata);
    
    // construct data to compress
    uint32_t thd = (compressedDPs->metadata->count + compressedDPs->metadata->frame - 1)
        / compressedDPs->metadata->frame;
    compressedDPs->tsLens = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    compressedDPs->valLens = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    compressedDPs->timestamps =
        (byte*)malloc(BYTES_OF_LONG_LONG*compressedDPs->metadata->count);
    compressedDPs->values =
        (byte*)malloc(BYTES_OF_LONG_LONG*compressedDPs->metadata->count);
    assert(compressedDPs->tsLens != NULL && compressedDPs->valLens != NULL
        && compressedDPs->timestamps != NULL && compressedDPs->values != NULL);

    compressedDPs->tsLens[0] = 8;
    compressedDPs->valLens[0] = 8;
    ((uint64_t*)compressedDPs->timestamps)[0] = 1519531200;
    ((uint64_t*)compressedDPs->values)[0] = 96; // test long long type value
    //((double*)compressedDPs->values)[0] = 96; // test double type value

    // 26(metadata)+4(sizes)+16(compressed data)
    writeCompressedData(outputFile, compressedDPs);

    fclose(outputFile);
    freeCompressedDPs(compressedDPs);
    //////////////////////////////////////////////////////////////////////////


    //////////////////////////////////////////////////////////////////////////
    // TEST:readCompressedData
    inputFilePath = outputFilePath;
    inputFile = fopen(outputFilePath, "r");
    assert(inputFile != NULL);

    compressedDPs = readCompressedFile(inputFile);

    // print metadata
    printMetadata(compressedDPs->metadata);
    printf("tsSizes[0]:%hd\tvalSizes[0]:%hd\n", 
        compressedDPs->tsLens[0], compressedDPs->valLens[0]);
    printf("timestamps[0]:%lld\tvalues[0]:%lld\n",
        ((uint64_t*)compressedDPs->timestamps)[0],
        ((uint64_t*)compressedDPs->values)[0]);

    /*// test double type value
    printf("timestamps[0]:%lld\tvalues[0]:%lld\n",
        ((uint64_t*)compressedDPs->timestamps)[0],
        ((uint64_t*)compressedDPs->values)[0]);*/

    fclose(inputFile);
    freeCompressedDPs(compressedDPs);
    //////////////////////////////////////////////////////////////////////////
}


/**
* TEST gorilla.cu: timestamp_compress_gorilla_gpu
* TEST gorilla.cu: timestamp_decompress_gorilla_gpu
* TEST gorilla.cu: value_compress_gorilla_gpu
* TEST gorilla.cu: value_decompress_gorilla_gpu
*/
void test_compress_gorilla_gpu() 
{
    // declare
    char inputFilePath[] = "dataset/testDataset2_b";
    DataPoints
        *dataPoints;
    uint32_t
        block = 1, warp = 1;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t
        timer;

    // read the uncompressed data
    dataPoints = readUncompressedFile_b(inputFilePath);

    // print the last 32 data points
    printDatapoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_compress_gorilla_gpu

    // construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(tsByteBuffer != NULL);
    tsByteBuffer->buffer = (byte*)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData *compressedData_t =
        timestamp_compress_gorilla_gpu(tsByteBuffer, block, warp);

    //////////////////////////////////////////////////////////////////////////
    
    // print the compressed data
    //printCompressedData(compressedData_t);

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_compress_gorilla_gpu

    // construct the buffer for uncompressed values
    ByteBuffer *valByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(valByteBuffer != NULL);
    valByteBuffer->buffer = (byte*)dataPoints->values;
    valByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData* compressedData_v =
        value_compress_gorilla_gpu(valByteBuffer, block, warp);

    //////////////////////////////////////////////////////////////////////////

    // print the compressed data
    //printCompressedData(compressedData_v);

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_decompress_gorilla_gpu

    // declare
    byte
        *buffer = compressedData_t->buffer,
        *slow = buffer,
        *fast = buffer;
    uint16_t
        frame = compressedData_t->frame,
        *lens = compressedData_t->lens;
    uint32_t
        count = compressedData_t->count,
        thd = (count + frame - 1) / frame;
    uint64_t
        frame_b = frame*BYTES_OF_LONG_LONG;

    // compact the compressed data
    for (uint32_t i = 0; i < thd; i++) {
        assert(lens[i] <= frame_b);
        memmove(slow, fast, lens[i]);
        slow += lens[i];
        fast += frame_b;
    }

    // decompress the compacted data
    ByteBuffer* decompressedData_t =
        timestamp_decompress_gorilla_gpu(compressedData_t, block, warp);

    //////////////////////////////////////////////////////////////////////////

    // print the decompressed data
    printDecompressedData(decompressedData_t, dataPoints->timestampType);

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_decompress_gorilla_gpu

    // declare
    buffer = compressedData_v->buffer;
    slow = buffer;
    fast = buffer;
    lens = compressedData_v->lens;

    // compact the compressed data
    for (uint32_t i = 0; i < thd; i++) {
        assert(lens[i] <= frame_b);
        memmove(slow, fast, lens[i]);
        slow += lens[i];
        fast += frame_b;
    }

    // decompress the compacted data
    ByteBuffer* decompressedData_v =
        value_decompress_gorilla_gpu(compressedData_v, block, warp);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    // print the decompressed data
    printDecompressedData(decompressedData_v, dataPoints->valueType);

    // print stat info 
    printStat(
        compressedData_t,
        compressedData_v,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // free memory
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_t);
    freeCompressedData(compressedData_v);
    freeByteBuffer(decompressedData_t);
    freeByteBuffer(decompressedData_v);
}

// TEST data_types.h:printStat
void test_printStat()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3";
    FILE *inputFile;
    DataPoints* dataPoints;
    // test metric value in double type
    ValueType timestampType = _LONG_LONG, valueType = _DOUBLE;
    // test metric value in long type
    //ValueType timestampType = _LONG_LONG, valueType = _LONG_LONG;
    uint32_t 
        block = 8, warp = 32;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t timer;

    inputFile = fopen(inputFilePath, "r");
    assert(inputFile != NULL);

    // read the uncompressed data
    dataPoints = readUncompressedFile(
        inputFile, timestampType, valueType
    );

    // print uncompressed data points
    /*printDatapoints(dataPoints);*/

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_compress_gorilla_gpu

    // construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(tsByteBuffer != NULL);
    tsByteBuffer->buffer = (byte*)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData *compressedData_t =
        timestamp_compress_gorilla_gpu(tsByteBuffer, block, warp);

    // print the compressed data
    /*printf("Compress timestamps(thread 0-3): \n");
    printCompressedData(compressedData_t);*/

    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_compress_gorilla_gpu

    // construct the buffer for uncompressed values
    ByteBuffer *valByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(valByteBuffer != NULL);
    valByteBuffer->buffer = (byte*)dataPoints->values;
    valByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData* compressedData_v =
        value_compress_gorilla_gpu(valByteBuffer, block, warp);

    // print the compressed data
    /*printf("Compress values(thread 0-3): \n");
    printCompressedData(compressedData_v);*/

    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_decompress_gorilla_gpu

    // declare
    byte
        *buffer = compressedData_t->buffer,
        *slow = buffer,
        *fast = buffer;
    uint16_t
        frame = compressedData_t->frame,
        *lens = compressedData_t->lens;
    uint32_t
        count = compressedData_t->count,
        thd = (count + frame - 1) / frame;
    uint64_t
        frame_b = frame*BYTES_OF_LONG_LONG;

    // compact the compressed data
    for (uint32_t i = 0; i < thd; i++) {
        assert(lens[i] <= frame_b);
        memmove(slow, fast, lens[i]);
        slow += lens[i];
        fast += frame_b;
    }

    // decompress the compacted data
    ByteBuffer* decompressedData_t =
        timestamp_decompress_gorilla_gpu(compressedData_t, block, warp);

    // print the decompressed data
    /*printf("Decompressed timestamps(the last 32): \n");
    printDecompressedData(decompressedData_t, timestampType);*/

    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_decompress_gorilla_gpu

    // declare
    buffer = compressedData_v->buffer;
    slow = buffer;
    fast = buffer;
    lens = compressedData_v->lens;

    // compact the compressed data
    for (uint32_t i = 0; i < thd; i++) {
        assert(lens[i] <= frame_b);
        memmove(slow, fast, lens[i]);
        slow += lens[i];
        fast += frame_b;
    }

    // decompress the compacted data
    ByteBuffer* decompressedData_v =
        value_decompress_gorilla_gpu(compressedData_v, block, warp);

    // print the decompressed data
    /*printf("Decompressed values(the last 32): \n");
    printDecompressedData(decompressedData_v, valueType);*/

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    printStat(
        compressedData_t, 
        compressedData_v,
        compressionTimeMillis, 
        decompressionTimeMillis
    );


    // free memory
    fclose(inputFile);
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_t);
    freeCompressedData(compressedData_v);
    freeByteBuffer(decompressedData_t);
    freeByteBuffer(decompressedData_v);
}

void test_compactData()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3_b";
    DataPoints *dataPoints;
    uint32_t 
        block = 8, warp = 32;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t timer;

    // read the uncompressed data
    dataPoints = readUncompressedFile_b(inputFilePath);

    // print the last 32 data points
    printDatapoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_compress_gorilla_gpu

    // construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(tsByteBuffer != NULL);
    tsByteBuffer->buffer = (byte*)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData *compressedData_t =
        timestamp_compress_gorilla_gpu(tsByteBuffer, block, warp);

    // print the compressed data
    //printCompressedData(compressedData_t);

    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_compress_gorilla_gpu

    // construct the buffer for uncompressed values
    ByteBuffer *valByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(valByteBuffer != NULL);
    valByteBuffer->buffer = (byte*)dataPoints->values;
    valByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData* compressedData_v =
        value_compress_gorilla_gpu(valByteBuffer, block, warp);

    // print the compressed data
    //printCompressedData(compressedData_v);

    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_decompress_gorilla_gpu

    // compact the compressed data
    compactData(compressedData_t);

    // decompress the compacted data
    ByteBuffer* decompressedData_t =
        timestamp_decompress_gorilla_gpu(compressedData_t, block, warp);

    // print the decompressed data
    printDecompressedData(decompressedData_t, dataPoints->timestampType);

    //////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_decompress_gorilla_gpu
    
    // compact the compressed data
    compactData(compressedData_v);

    // decompress the compacted data
    ByteBuffer* decompressedData_v =
        value_decompress_gorilla_gpu(compressedData_v, block, warp);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    // print the decompressed data
    printDecompressedData(decompressedData_v, dataPoints->valueType);

    printStat(
        compressedData_t,
        compressedData_v,
        compressionTimeMillis,
        decompressionTimeMillis
    );


    // free memory
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_t);
    freeCompressedData(compressedData_v);
    freeByteBuffer(decompressedData_t);
    freeByteBuffer(decompressedData_v);
}

void test_compress_rle_gpu()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3_b";
    DataPoints 
        *dataPoints;
    uint32_t
        block = 8, warp = 16;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t timer;

    // read the uncompressed data
    dataPoints = readUncompressedFile_b(inputFilePath);

    // print the last 32 data points
    printDatapoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_compress_rle_gpu

    // construct the buffer for uncompressed timestamps
    ByteBuffer *tsByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(tsByteBuffer != NULL);
    tsByteBuffer->buffer = (byte*)dataPoints->timestamps;
    tsByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData *compressedData_t =
        timestamp_compress_rle_gpu(tsByteBuffer, block, warp);

    // print the compressed data
    //printCompressedData(compressedData_t);
    
    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:timestamp_decompress_gorilla_gpu

    // compact the compressed data
    compactData(compressedData_t);

    // decompress the compacted data
    ByteBuffer* decompressedData_t =
        timestamp_decompress_rle_gpu(compressedData_t, block, warp);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    // print the decompressed data
    printDecompressedData(decompressedData_t, dataPoints->timestampType);

    printStat(
        compressedData_t,
        compressionTimeMillis,
        decompressionTimeMillis
    );

    // free memory
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_t);
    freeByteBuffer(decompressedData_t);
}

void test_compress_bitpack_gpu()
{
    // declare
    char inputFilePath[] = "dataset/Server35_b";
    DataPoints 
        *dataPoints;
    uint32_t
        block = 1, warp = 1;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t 
        timer;

    // read the uncompressed data
    dataPoints = readUncompressedFile_b(inputFilePath);

    // print the last 32 data points
    printDatapoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_compress_bitpack_gpu

    // construct the buffer for uncompressed values
    ByteBuffer *valByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(valByteBuffer != NULL);
    valByteBuffer->buffer = (byte*)dataPoints->values;
    valByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData* compressedData_v =
        value_compress_bitpack_gpu(valByteBuffer, block, warp);

    // print the compressed data
    //printCompressedData(compressedData_v);

    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_decompress_bitpack_gpu

    // compact the compressed data
    compactData(compressedData_v);

    // decompress the compacted data
    ByteBuffer* decompressedData_v =
        value_decompress_bitpack_gpu(compressedData_v, block, warp);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    // print the decompressed data
    printDecompressedData(decompressedData_v, dataPoints->valueType);

    printStat(
        compressedData_v,
        compressionTimeMillis,
        decompressionTimeMillis
    );


    // free memory
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_v);
    freeByteBuffer(decompressedData_v);
}

void test_compress_bucket_gpu()
{
    // declare
    char inputFilePath[] = "dataset/testDataset3_fb";
    DataPoints
        *dataPoints;
    uint32_t
        block = 8, warp = 16;
    uint64_t
        compressionTimeMillis, decompressionTimeMillis;
    clock_t
        timer;

    // read the uncompressed data
    dataPoints = readUncompressedFile_b(inputFilePath);

    // print the last 32 data points
    printDatapoints(dataPoints);

    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_compress_bitpack_gpu

    // construct the buffer for uncompressed values
    ByteBuffer *valByteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    assert(valByteBuffer != NULL);
    valByteBuffer->buffer = (byte*)dataPoints->values;
    valByteBuffer->length = dataPoints->count*BYTES_OF_LONG_LONG;

    // compress the data points, and get the compressed data 
    // which is not compacted
    CompressedData* compressedData_v =
        value_compress_bucket_gpu(valByteBuffer, block, warp);

    // print the compressed data
    //printCompressedData(compressedData_v);

    //////////////////////////////////////////////////////////////////////////

    compressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;
    timer = clock();

    //////////////////////////////////////////////////////////////////////////
    // TEST:value_decompress_bitpack_gpu

    // compact the compressed data
    compactData(compressedData_v);

    // decompress the compacted data
    ByteBuffer* decompressedData_v =
        value_decompress_bucket_gpu(compressedData_v, block, warp);

    //////////////////////////////////////////////////////////////////////////

    decompressionTimeMillis = (clock() - timer) * 1000 / CLOCKS_PER_SEC;

    // print the decompressed data
    printDecompressedData(decompressedData_v, dataPoints->valueType);

    printStat(
        compressedData_v,
        compressionTimeMillis,
        decompressionTimeMillis
    );


    // free memory
    freeDataPoints(dataPoints);
    freeCompressedData(compressedData_v);
    freeByteBuffer(decompressedData_v);
}

