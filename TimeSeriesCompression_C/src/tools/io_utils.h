#ifndef _IO_UTILS_H_
#define _IO_UTILS_H_

#include "data_types.h"

// Read and parse the uncompressed file in text format.
DataPoints *readUncompressedFile(
    FILE *inputFile, ValueType tsType, ValueType valType
);

// Read and parse the uncompressed file in binary format.
DataPoints *readUncompressedFile_b(
    const char *const input
);

// Read and parse the comressed file.
CompressedData *readCompressedFile(
    FILE *inputFile
);

// Write the compressed data in binary format into specific file.
void writeCompressedData(
    FILE *outputFile, CompressedData* compressedData
);

// Write the decompressed data in text format into specific file.
void writeDecompressedData(
    FILE *outputFile, DataPoints *decompressedData
);

// Transform dataset from text format into binary
void textToBinary(
    char const *const input, char const *const output,
    ValueType tsType, ValueType valType
);

#endif // _IO_UTILS_H_