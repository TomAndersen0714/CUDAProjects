#include "io_utils.h"

DataPoints *readUncompressedFile(FILE *inputFile, ValueType timestampType, ValueType valueType) {
    // Declare variables
    uint64_t count, cursor;
    uint64_t *timestamps, *values;
    DataPoints* dataPoints;

    // Get the number of data points and pre-allocate momery space for it
    if (fscanf(inputFile, "%llu", &count) == EOF) {
        //fclose(inputFile);
        exit(EXIT_FAILURE);
    };
    timestamps = (uint64_t *)malloc(sizeof(uint64_t)*count);
    values = (uint64_t *)malloc(sizeof(uint64_t)*count);
    assert(timestamps != NULL && values != NULL);

    // Parse every line of file
    cursor = 0;
    if (timestampType == _LONG_LONG && valueType == _LONG_LONG) {
        while (cursor < count &&
            // Format the string and convert to the values in corresponding 
            // type, then write the bits of values into the specific address
            fscanf(inputFile, "%llu %lld", &timestamps[cursor], &values[cursor]) != EOF) {
            cursor++;
        }
    }
    else if (timestampType == _LONG_LONG && valueType == _DOUBLE) {
        while (cursor < count &&
            fscanf(inputFile, "%lld %lf", &timestamps[cursor], &values[cursor]) != EOF) {
            cursor++;
        }
    }
    // Timestamp must be long long type
    // else if (timestampType == _DOUBLE_ && valueType == _LONG_LONG_)
    // else if (timestampType == _DOUBLE_ && valueType == _DOUBLE_)

    // Return the uncompressed data points
    dataPoints = (DataPoints*)malloc(sizeof(DataPoints));
    assert(dataPoints != NULL);
    dataPoints->count = count;
    dataPoints->timestamps = timestamps;
    dataPoints->timestampType = timestampType;
    dataPoints->values = values;
    dataPoints->valueType = valueType;
    return dataPoints;
}

void writeCompressedData(FILE *outputFile, CompressedData *compressedData) {

    // Write metadata as the header of compressed file
    fwrite(compressedData->metadata, sizeof(Metadata), 1, outputFile);
    // Write the compressed data into file
    fwrite(compressedData->timestamps,
        sizeof(byte)*compressedData->metadata->tsLength, 1, outputFile);
    fwrite(compressedData->values,
        sizeof(byte)*compressedData->metadata->valLength, 1, outputFile);
}

CompressedData *readCompressedFile(FILE *inputFile) {
    // Declare variables
    Metadata *metadata;
    byte *timestamps, *values;
    CompressedData* compressedData;

    // Get the metadata of compressed data
    metadata = (Metadata *)malloc(sizeof(Metadata));
    assert(metadata != NULL);
    fread(metadata, sizeof(Metadata), 1, inputFile);
    // Get the compressed data
    timestamps = (byte *)malloc(sizeof(byte)*metadata->tsLength);
    values = (byte *)malloc(sizeof(byte)*metadata->valLength);
    assert(timestamps != NULL && values != NULL);
    fread(timestamps, sizeof(byte)*metadata->tsLength, 1, inputFile);
    fread(values, sizeof(byte)*metadata->valLength, 1, inputFile);

    // Return the compressed data points and metadata
    compressedData = (CompressedData*)malloc(sizeof(CompressedData));
    assert(compressedData != NULL);
    compressedData->metadata = metadata;
    compressedData->timestamps = timestamps;
    compressedData->values = values;
    return compressedData;
}

void writeDecompressedData(FILE *outputFile, DataPoints *decompressedData) {
    // Write the number of data points into file
    fprintf(outputFile, "%lld\n", decompressedData->count);

    // Write the data points into file
    uint64_t cursor = 0;
    if (decompressedData->timestampType == _LONG_LONG
        && decompressedData->valueType == _LONG_LONG) {
        while (cursor < decompressedData->count) {
            fprintf(
                outputFile, "%lld %lld\n",
                decompressedData->timestamps[cursor],
                decompressedData->values[cursor]
            );
            cursor++;
        }
    }
    else if (decompressedData->timestampType == _LONG_LONG
        && decompressedData->valueType == _DOUBLE) {
        while (cursor < decompressedData->count) {
            fprintf(
                outputFile, "%lld %lf\n",
                decompressedData->timestamps[cursor],
                decompressedData->values[cursor]
            );
            cursor++;
        }
    }
    // Timestamp must be long long type
    // else if (timestampType == _DOUBLE_ && valueType == _LONG_LONG_)
    // else if (timestampType == _DOUBLE_ && valueType == _DOUBLE_)
}