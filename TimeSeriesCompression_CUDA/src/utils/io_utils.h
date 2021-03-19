#ifndef _IO_UTILS_H_
#define _IO_UTILS_H_

#include "data_types.h"

// Read and parse the uncompressed file in text format.
DataPoints* readUncompressedData(
    FILE* inputFile,
    ValueType timestampType,
    ValueType valueType
);

// Read and parse the comressed file.
CompressedDPs* readCompressedData(FILE* inputFile);

// Write the compressed data in binary format into specific file.
void writeCompressedData(
    FILE* outputFile,
    CompressedDPs* compressedDPs
);

// Write the decompressed data in text format into specific file.
void writeDecompressedData(
    FILE* outputFile,
    DataPoints* decompressedDPs
);

#endif // _IO_UTILS_H_