#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <time.h>

#define BITS_OF_BYTE 8
#define BITS_OF_INT 32
#define BITS_OF_FLOAT 32
#define BITS_OF_LONG_LONG 64
#define BITS_OF_DOUBLE 64
#define BYTES_OF_INT 4
#define BYTES_OF_LONG_LONG 8
#define BYTES_OF_DOUBLE 8
#define DEFAULT_BUFFER_SIZE 1024
#define DEFAULT_FRAME_SIZE 8
#define MIN_FRAME_SIZE 32
#define WARPSIZE 32
#define MAX_THREADS_PER_BLOCK 1024

typedef unsigned char byte;


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
    uint64_t length; // The number of elements(bytes)
    //uint64_t capacity;
    //ValueType type;
} ByteBuffer;


/*
// Struct of uncompressed buffer
typedef struct _DataBuffer {
    uint64_t *buffer;
    uint64_t length; // The number of elements(uint64_t)
    //ValueType type;
} DataBuffer;
*/

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
    uint64_t tsLength; // byte length of compressed timestamps
    uint64_t valLength; // byte length of compressed values
    uint64_t count; // the number of compressed data points
} Metadata;

// Struct of compressed data points
typedef struct _CompressedData {
    Metadata *metadata;
    byte *timestamps;
    byte *values;
} CompressedData;

// For scalability, define function pointer type for compression and 
// decompression method
typedef ByteBuffer *(*compressMethod)(ByteBuffer*);
typedef ByteBuffer *(*decompressMethod)(CompressedData*);

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

static inline void freeCompressedData(CompressedData* const compressedData) {
    freeMetadata(compressedData->metadata);
    free(compressedData->timestamps);
    free(compressedData->values);
    free(compressedData);
}

/*
static inline void printCompressedData(ByteBuffer* byteBuffer) {
    for (int i = 0; i < byteBuffer->length; i++) {
        printf("%02X ", byteBuffer->buffer[i]);
    }
    puts(""); // print new line
}*/

static inline void printCompressedData(
    byte* compressedData,
    uint32_t* offs, // offsets of each frame
    uint32_t* length,// length of compressed data for each thread
    uint32_t threads // the total number of threads
) {
    // Restrict data to print
    if (threads > 4) threads = 4;
    int start, end; // start and end offset of compressed bytes
    for (int i = 0; i < threads; i++) {
        start = offs[i] * BYTES_OF_LONG_LONG;
        end = start + length[i];
        assert(end < offs[i + 1] * BYTES_OF_LONG_LONG);
        for (int j = start; j < end; j++) {
            printf("%02X ", compressedData[j]);
        }
        puts(""); // print new line
    }
    puts(""); // print new line
}

static inline void printDecompressedData(ByteBuffer* byteBuffer, ValueType dataType) {
    uint64_t* datas = (uint64_t*)byteBuffer->buffer;
    uint64_t count = byteBuffer->length / sizeof(uint64_t);

    if (dataType == _LONG_LONG) {
        for (int i = 0; i < count; i++) {
            printf("%lld\n", datas[i]);
        }
    }
    else {
        for (int i = 0; i < count; i++) {
            printf("%lf\n", datas[i]);
        }
    }
    puts("");
}

static inline void printDatapoints(const DataPoints* const dataPoints) {
    // restrict the number of datapoint to print
    int count = dataPoints->count;
    if (count > 32) count = 32;

    // Print the datapoints info
    printf(
        "TimestampType: %d, ValueType: %d, Count: %llu \n",
        dataPoints->timestampType,
        dataPoints->valueType,
        dataPoints->count
    );

    // Print data points
    printf("Timestamps:\tValues:\n");
    if (dataPoints->timestampType == _LONG_LONG
        &&dataPoints->valueType == _LONG_LONG
        ) {
        for (int i = 0; i < count; i++) {
            printf(
                "%llu\t%llu\n",
                dataPoints->timestamps[i],
                dataPoints->values[i]
            );
        }
    }
    else if (
        dataPoints->timestampType == _LONG_LONG
        &&dataPoints->valueType == _DOUBLE
        ) {
        for (int i = 0; i < count; i++) {
            printf(
                "%llu\t%lf\n",
                dataPoints->timestamps[i],
                dataPoints->values[i]
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
        "Timestamp length: %llu,\tValue length: %llu,\tCount: %llu\n",
        metadata->tsLength, metadata->valLength, metadata->count
    );
}

// Print the statistic info 
static inline void printStat(
    DataPoints* datapoints,
    ByteBuffer* compressedTimestamps,
    ByteBuffer* compressedValues,
    uint64_t compressionTimeMillis,
    uint64_t decompressionTimeMillis
) {
    uint64_t uncompressedTimestampSize = datapoints->count * sizeof(uint64_t);
    uint64_t uncompressedValuesSize = datapoints->count * sizeof(uint64_t);
    uint64_t compressedTimestmapsSize = compressedTimestamps->length;
    uint64_t compressedValuesSize = compressedValues->length;
    float timestampsCompRatio = (float)uncompressedTimestampSize / compressedTimestmapsSize;
    float valuesCompRatio = (float)uncompressedValuesSize / compressedValuesSize;
    float compRatio = (float)(uncompressedTimestampSize + uncompressedValuesSize)
        / (compressedTimestmapsSize + compressedValuesSize);
    double compSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / compressionTimeMillis * 1000;
    double decompSpeed = (double)(uncompressedTimestampSize + uncompressedValuesSize) / decompressionTimeMillis * 1000;

    // Print statistic info
    printf("Timestamps: %lluB -> %lluB\n", uncompressedTimestampSize, compressedTimestmapsSize);
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