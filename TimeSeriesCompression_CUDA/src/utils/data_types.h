#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#define BITS_OF_BYTE 8
#define BITS_OF_INT 32
#define BITS_OF_FLOAT 32
#define BITS_OF_LONG_LONG 64
#define BITS_OF_DOUBLE 64
#define BYTES_OF_BYTE 1
#define BYTES_OF_SHORT 2
#define BYTES_OF_INT 4
#define BYTES_OF_LONG_LONG 8
#define BYTES_OF_DOUBLE 8
#define DEFAULT_BUFFER_SIZE 1024
#define DEFAULT_SUBFRAME_SIZE 8
#define MIN_FRAME_SIZE 1024
#define MAX_FRAME_SIZE 65536
#define WARPSIZE 32
#define MAX_THREADS_PER_BLOCK 1024

typedef unsigned char byte;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

// Timestamp compression and decompression algorithms
typedef enum _TSComAndDecomAlgo {
    TS_GORILLA, TS_RLE
} TSComAndDecomAlgo;

// Value compression and decompression algorithms
typedef enum _ValueComAndDecomAlgo {
    VAL_GORILLA, VAL_BITPACK, VAL_BUCKET
} ValComAndDecomAlgo;

// Metric value type
typedef enum _ValueType {
    _LONG_LONG, _DOUBLE
} ValueType;

// Struct of data buffer
typedef struct _ByteBuffer {
    byte *buffer;
    uint64_t length; // The length of buffer in bytes
    //uint64_t capacity;
    //ValueType type;
} ByteBuffer;

// Struct of compressed data
typedef struct _CompressedData {
    byte *buffer; // buffer of compressed data
    //uint32_t *offs; // the start offset of each frame in 8 byte-wise
    uint16_t *lens; // the length of each compressed frame
    uint16_t frame; // the length of each frame
    uint32_t count; // the total number of data points
} CompressedData;

// Struct of data points
typedef struct _DataPoitns {
    uint64_t *timestamps;
    uint64_t *values;
    uint64_t count;
    ValueType timestampType;
    ValueType valueType;
} DataPoints;

// Metadata of compressed data points
typedef struct _Metadata {
    TSComAndDecomAlgo tsComAndDecom;
    ValComAndDecomAlgo valComAndDecom;
    ValueType timestampType;
    ValueType valueType;
    //uint64_t tsLength; // byte length of compressed timestamps
    //uint64_t valLength; // byte length of compressed values
    uint16_t frame; // the length of each frame
    uint32_t count; // the number of compressed data points
} Metadata;

// Struct of compressed data points
typedef struct _CompressedDPs {
    Metadata *metadata;
    uint16_t* tsLens;
    uint16_t* valLens;
    byte *timestamps;
    byte *values;
} CompressedDPs;

// For scalability, define function pointer type for compression and 
// decompression method
typedef ByteBuffer *(*compressMethod)(ByteBuffer*);
typedef ByteBuffer *(*decompressMethod)(CompressedDPs*);


// Compact the compressed data
static inline void compactData(CompressedData *compressedData) {

    // declare
    byte
        *buffer = compressedData->buffer,
        *slow = buffer,
        *fast = buffer;
    uint16_t
        frame = compressedData->frame,
        *lens = compressedData->lens;
    uint32_t
        count = compressedData->count,
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
}

static inline void freeByteBuffer(ByteBuffer* const byteBuffer) {
    free(byteBuffer->buffer);
    free(byteBuffer);
}
/*

static inline void freeDataBuffer(DataBuffer* const dataBuffer) {
    free(dataBuffer->buffer);
    free(dataBuffer);
}
*/

static inline void freeDataPoints(DataPoints* const dataPoints) {
    free(dataPoints->timestamps);
    free(dataPoints->values);
    free(dataPoints);
}

static inline void freeMetadata(Metadata* const metadata) {
    free(metadata);
}

static inline void freeCompressedData(CompressedData* compressedData) {
    free(compressedData->buffer);
    free(compressedData->lens);
    free(compressedData);
}

static inline void freeCompressedDPs(CompressedDPs* const compressedDPs) {
    freeMetadata(compressedDPs->metadata);
    free(compressedDPs->tsLens);
    free(compressedDPs->valLens);
    free(compressedDPs->timestamps);
    free(compressedDPs->values);
    free(compressedDPs);
}

// print the compressed data which is not compacted
static inline void printCompressedData(
    CompressedData* compressedData
) {
    uint32_t
        thd = (compressedData->count + compressedData->frame - 1)
        / compressedData->frame,
        frame_b = compressedData->frame*BYTES_OF_LONG_LONG,
        start = 0;
    uint16_t
        *lens = compressedData->lens;
    byte
        *buffer = compressedData->buffer;

    // restrict data to print
    if (thd > 4) thd = 4;

    printf("Compressed data(the first %u thread): \n", thd);
    for (uint32_t i = 0; i < thd; i++) {
        assert(lens[i] <= frame_b); // avoid accessing out of bounds
        printf("Thread-%u:\n", i);
        for (uint32_t j = start; j < start + lens[i]; j++) {
            printf("%02X ", buffer[j]);
        }
        start += frame_b;
        puts(""); // new line
    }
    puts("");
}

static inline void printDecompressedData(
    ByteBuffer *byteBuffer, ValueType dataType
) {
    // declare
    uint64_t
        *datas = (uint64_t*)byteBuffer->buffer;
    uint64_t
        count = byteBuffer->length / sizeof(uint64_t),
        len = 32, // the numeber of data to print
        offset = 0; // the offset of data to print

    // restrict the data to print
    offset = count > len ? count - len : 0;

    printf("Decompressed data(the last %llu):\n", len);
    if (dataType == _LONG_LONG) {
        for (uint64_t i = offset; i < count; i++) {
            printf("%lld\n", datas[i]);
        }
    }
    else {
        double *datas_d = (double*)byteBuffer->buffer;
        for (uint64_t i = offset; i < count; i++) {
            printf("%lf\n", datas_d[i]);
        }
    }
    puts("");
}

static inline void printDatapoints(const DataPoints* const dataPoints) {
    // restrict the number of datapoint to print
    uint64_t
        count = dataPoints->count,
        offset = 0; // the offset of data to print

    if (count > 32) offset = count - 32;

    // Print the datapoints info
    printf(
        "TimestampType: %d, ValueType: %d, Count: %llu \n",
        dataPoints->timestampType,
        dataPoints->valueType,
        dataPoints->count
    );

    // Print data points
    printf("Datapoints(the last %llu):\n", count - offset);
    if (dataPoints->timestampType == _LONG_LONG && dataPoints->valueType == _LONG_LONG) {
        for (uint64_t i = offset; i < count; i++) {
            printf(
                "%llu\t%llu\n",
                dataPoints->timestamps[i],
                dataPoints->values[i]
            );
        }
    }
    else if (dataPoints->timestampType == _LONG_LONG && dataPoints->valueType == _DOUBLE) {
        double *values = (double*)dataPoints->values;
        for (uint64_t i = offset; i < count; i++) {
            printf(
                "%llu\t%lf\n",
                dataPoints->timestamps[i],
                values[i]
            );
        }
    }
    puts(""); // print new line
}

static inline void printMetadata(const Metadata* const metadata) {
    printf(
        "TSComAndDecomAlgo: %d,\tValComAndDecomAlgo: %d\n",
        metadata->tsComAndDecom, metadata->valComAndDecom
    );
    printf(
        "TimestampType: %d,\tValueType: %d\n",
        metadata->timestampType, metadata->valueType
    );
    printf(
        "Frame: %hu,\tCount: %lu\n",
        metadata->frame, metadata->count
    );
}

/*
// Print the statistic info
static inline void printStat(
    DataPoints *datapoints,
    ByteBuffer *compressedTimestamps,
    ByteBuffer *compressedValues,
    uint64_t compressionTimeMillis,
    uint64_t decompressionTimeMillis
) {
    uint64_t uncompressedTimestampSize = datapoints->count * sizeof(uint64_t);
    uint64_t uncompressedValuesSize = datapoints->count * sizeof(uint64_t);
    uint64_t compressedTimestampsSize = compressedTimestamps->length;
    uint64_t compressedValuesSize = compressedValues->length;
    float timestampsCompRatio = (float)uncompressedTimestampSize / compressedTimestampsSize;
    float valuesCompRatio = (float)uncompressedValuesSize / compressedValuesSize;
    float compRatio = (float)(uncompressedTimestampSize + uncompressedValuesSize)
        / (compressedTimestampsSize + compressedValuesSize);
    double compSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / compressionTimeMillis * 1000;
    double decompSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / decompressionTimeMillis * 1000;

    // Print statistic info
    printf("Timestamps: %lluB -> %lluB\n", uncompressedTimestampSize, compressedTimestampsSize);
    printf("Timestamps compression ratio: %f\n", timestampsCompRatio);
    printf("Metric values: %lluB -> %lluB\n", uncompressedValuesSize, compressedValuesSize);
    printf("Metric values compression ratio: %f\n", valuesCompRatio);
    printf("Compression ratio: %f\n", compRatio);
    printf("Compression time: %llums\n", compressionTimeMillis);
    printf("Compression speed: %lfB/s, %lfKB/s, %lfMB/s\n", compSpeed, compSpeed / 1024, compSpeed / (1024 * 1024));
    printf("Decompression time: %llums\n", decompressionTimeMillis);
    printf("Decompression speed: %lfB/s, %lfKB/s, %lfMB/s\n", decompSpeed, decompSpeed / 1024, decompSpeed / (1024 * 1024));
}*/

// Print the statistical info of compressed timestamps or values
static inline void printStat(
    CompressedData *compressedData,
    uint64_t compressionTimeMillis,
    uint64_t decompressionTimeMillis
) {
    uint64_t
        uncompressedDataSize = compressedData->count*BYTES_OF_LONG_LONG,
        compressedDataSize = 0;
    uint32_t
        thd = (compressedData->count + compressedData->frame - 1)
        / compressedData->frame;

    for (uint32_t i = 0; i < thd; i++) {
        compressedDataSize += compressedData->lens[i];
    }

    // space for storing the length of compressed frames
    compressedDataSize += BYTES_OF_SHORT*thd;

    double
        compRatio = (double)uncompressedDataSize / compressedDataSize;
    double
        compSpeed = (double)uncompressedDataSize / compressionTimeMillis * 1000,
        decompSpeed = (double)uncompressedDataSize / decompressionTimeMillis * 1000;

    // print statistic info
    printf("Data size: %lluB -> %lluB\n", uncompressedDataSize, compressedDataSize);
    printf("Compression ratio: %f\n", compRatio);
    printf("Compression time: %llums\n", compressionTimeMillis);
    printf("Compression speed: %lfB/s, %lfKB/s, %lfMB/s\n", compSpeed, compSpeed / 1024, compSpeed / (1024 * 1024));
    printf("Decompression time: %llums\n", decompressionTimeMillis);
    printf("Decompression speed: %lfB/s, %lfKB/s, %lfMB/s\n", decompSpeed, decompSpeed / 1024, decompSpeed / (1024 * 1024));
}

// Print the statistical info of compressed data points
static inline void printStat(
    CompressedData *compressedTimestamps,
    CompressedData *compressedValues,
    uint64_t compressionTimeMillis,
    uint64_t decompressionTimeMillis
) {
    uint64_t
        uncompressedTimestampSize
        = compressedTimestamps->count*BYTES_OF_LONG_LONG,
        uncompressedValuesSize
        = compressedValues->count*BYTES_OF_LONG_LONG,
        compressedTimestampsSize = 0,
        compressedValuesSize = 0;
    uint32_t
        thd = (compressedTimestamps->count + compressedTimestamps->frame - 1)
        / compressedTimestamps->frame;
    for (uint32_t i = 0; i < thd; i++) {
        compressedTimestampsSize += compressedTimestamps->lens[i];
        compressedValuesSize += compressedValues->lens[i];
    }

    // space for storing the length of compressed frames
    compressedTimestampsSize += BYTES_OF_SHORT*thd;
    compressedValuesSize += BYTES_OF_SHORT*thd;

    float
        timestampsCompRatio =
        (float)uncompressedTimestampSize / compressedTimestampsSize,
        valuesCompRatio =
        (float)uncompressedValuesSize / compressedValuesSize,
        compRatio = (float)(uncompressedTimestampSize + uncompressedValuesSize)
        / (compressedTimestampsSize + compressedValuesSize);
    double
        compSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / compressionTimeMillis * 1000,
        decompSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / decompressionTimeMillis * 1000;

    // print statistic info
    printf("Timestamps: %lluB -> %lluB\n", uncompressedTimestampSize, compressedTimestampsSize);
    printf("Timestamps compression ratio: %f\n", timestampsCompRatio);
    printf("Metric values: %lluB -> %lluB\n", uncompressedValuesSize, compressedValuesSize);
    printf("Metric values compression ratio: %f\n", valuesCompRatio);
    printf("Compression ratio: %f\n", compRatio);
    printf("Compression time: %llums\n", compressionTimeMillis);
    printf("Compression speed: %lfB/s, %lfKB/s, %lfMB/s\n", compSpeed, compSpeed / 1024, compSpeed / (1024 * 1024));
    printf("Decompression time: %llums\n", decompressionTimeMillis);
    printf("Decompression speed: %lfB/s, %lfKB/s, %lfMB/s\n", decompSpeed, decompSpeed / 1024, decompSpeed / (1024 * 1024));
}

#endif // _DATA_TYPES_H_