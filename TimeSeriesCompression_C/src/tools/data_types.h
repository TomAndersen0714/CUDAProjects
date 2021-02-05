#ifndef _DATA_TYPES_H_
#define _DATA_TYPES_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define BITS_OF_BYTE 8
#define BITS_OF_INT 32
#define BITS_OF_FLOAT 32
#define BITS_OF_LONG_LONG 64
#define BITS_OF_DOUBLE 64
#define LINE_MAX_LENGTH 256
#define DEFAULT_BUFFER_SIZE 1024
#define DEFAULT_FRAME_SIZE 8
#define ZOOM_FACTOR 1.5

typedef unsigned char byte;


// Timestamp compression and decompression algorithms
typedef enum _TSComAndDecom {
    TS_GORILLA, TS_RLE
} TSComAndDecom;

// Value compression and decompression algorithms
typedef enum _ValueComAndDecom {
    VAL_GORILLA, VAL_BITPACK, VAL_BUCKET
} ValComAndDecom;

// Metric value type
typedef enum _ValueType {
    _LONG_LONG, _DOUBLE
} ValueType;

// Struct of compressed data
typedef struct _ByteBuffer {
    byte *buffer;
    uint64_t length;
    uint64_t capacity;
    //ValueType type;
} ByteBuffer;


// Struct of uncompressed buffer
typedef struct _UncompressedData {
    uint64_t *buffer;
    uint64_t length;
    ValueType type;
} UncompressedData;

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
    TSComAndDecom tsComAndDecom;
    ValComAndDecom valComAndDecom;
    ValueType timestampType;
    ValueType valueType;
    uint64_t tsLength;
    uint64_t valLength;
    uint64_t count;
} Metadata;

// Struct of compressed data points
typedef struct _CompressedData {
    Metadata *metadata;
    byte *timestamps;
    byte *values;
} CompressedData;

// For scalability, define function pointer type for compression and 
// decompression method
typedef ByteBuffer *(*compressMethod)(UncompressedData*);
typedef UncompressedData *(*decompressMethod)(CompressedData*);

#endif // _DATA_TYPES_H_