#include "io_utils.h"

DataPoints *readUncompressedFile(
    FILE *inputFile, ValueType tsType, ValueType valType
) {
    // Declare variables
    uint64_t count, cursor;
    uint64_t *timestamps, *values;
    DataPoints* dataPoints;

    // Get the number of data points and pre-allocate momery space for it
    if (fscanf(inputFile, "%llu", &count) == EOF) {
        //fclose(inputFile);
        exit(EXIT_FAILURE);
    };
    timestamps = malloc(sizeof(uint64_t)*count);
    values = malloc(sizeof(uint64_t)*count);
    assert(timestamps != NULL && values != NULL);

    // Parse every line of file
    cursor = 0;
    if (tsType == _LONG_LONG && valType == _LONG_LONG) {
        while (cursor < count &&
            // Format the string and convert to the values in corresponding 
            // type, then write the bits of values into the specific address
            fscanf(inputFile, "%llu %lld", &timestamps[cursor], &values[cursor]) != EOF) {
            cursor++;
        }
    }
    else if (tsType == _LONG_LONG && valType == _DOUBLE) {
        while (cursor < count &&
            fscanf(inputFile, "%lld %lf", &timestamps[cursor], &values[cursor]) != EOF) {
            cursor++;
        }
    }
    // Timestamp must be long long type
    // else if (timestampType == _DOUBLE_ && valueType == _LONG_LONG_)
    // else if (timestampType == _DOUBLE_ && valueType == _DOUBLE_)

    // Return the uncompressed data points
    dataPoints = malloc(sizeof(DataPoints));
    assert(dataPoints != NULL);
    dataPoints->count = count;
    dataPoints->timestamps = timestamps;
    dataPoints->timestampType = tsType;
    dataPoints->values = values;
    dataPoints->valueType = valType;
    return dataPoints;
}

DataPoints *readUncompressedFile_b(
    const char *const input
) {
    // declare
    uint64_t
        count, *timestamps, *values;
    ValueType
        tsType, valType;
    FILE 
        *inputFile;
    DataPoints
        *dataPoints;

    // read dataset in binary format
    inputFile = fopen(input, "rb");
    // read the header of dataset
    assert(fread(&count, sizeof(uint64_t), 1, inputFile) == 1);
    assert(fread(&tsType, sizeof(ValueType), 1, inputFile) == 1);
    assert(fread(&valType, sizeof(ValueType), 1, inputFile) == 1);
    // read the data
    timestamps = (uint64_t*)malloc(sizeof(uint64_t)*count);
    values = (uint64_t*)malloc(sizeof(uint64_t)*count);
    assert(timestamps != NULL && values != NULL);
    assert(fread(timestamps, sizeof(uint64_t)*count, 1, inputFile) == 1);
    assert(fread(values, sizeof(uint64_t)*count, 1, inputFile) == 1);

    // construct result
    dataPoints = (DataPoints*)malloc(sizeof(DataPoints));
    assert(dataPoints != NULL);
    dataPoints->count = count;
    dataPoints->timestampType = tsType;
    dataPoints->valueType = valType;
    dataPoints->timestamps = timestamps;
    dataPoints->values = values;

    // free memory
    fclose(inputFile);

    // return
    return dataPoints;
}

void writeCompressedData(
    FILE *outputFile, CompressedData *compressedData
) {

    // Write metadata as the header of compressed file
    fwrite(compressedData->metadata, sizeof(Metadata), 1, outputFile);
    // Write the compressed data into file
    fwrite(compressedData->timestamps,
        sizeof(byte)*compressedData->metadata->tsLength, 1, outputFile);
    fwrite(compressedData->values,
        sizeof(byte)*compressedData->metadata->valLength, 1, outputFile);
}

CompressedData *readCompressedFile(
    FILE *inputFile
) {
    // Declare variables
    Metadata *metadata;
    byte *timestamps, *values;
    CompressedData* compressedData;

    // Get the metadata of compressed data
    metadata = malloc(sizeof(Metadata));
    assert(metadata != NULL);
    fread(metadata, sizeof(Metadata), 1, inputFile);
    // Get the compressed data
    timestamps = malloc(sizeof(byte)*metadata->tsLength);
    values = malloc(sizeof(byte)*metadata->valLength);
    assert(timestamps != NULL && values != NULL);
    fread(timestamps, sizeof(byte)*metadata->tsLength, 1, inputFile);
    fread(values, sizeof(byte)*metadata->valLength, 1, inputFile);

    // Return the compressed data points and metadata
    compressedData = malloc(sizeof(CompressedData));
    assert(compressedData != NULL);
    compressedData->metadata = metadata;
    compressedData->timestamps = timestamps;
    compressedData->values = values;
    return compressedData;
}

void writeDecompressedData(
    FILE *outputFile, DataPoints *decompressedData
) {
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

// Transform dataset from text format into binary
void textToBinary(
    char const *const input, char const *const output,
    ValueType tsType, ValueType valType
) {
    // declare
    FILE 
        *inputFile, *outputFile;
    DataPoints 
        *dataPoints;

    // open file
    inputFile = fopen(input, "r");
    outputFile = fopen(output, "wb");
    assert(inputFile != NULL && outputFile != NULL);

    // read and parse file to get data points
    dataPoints = readUncompressedFile(
        inputFile, tsType, valType
    );

    // write data points into specific file in binary format
    // write the header of dataset
    assert(fwrite(&dataPoints->count, sizeof(uint64_t), 1, outputFile) == 1);
    assert(fwrite(&dataPoints->timestampType, sizeof(ValueType), 1, outputFile) == 1);
    assert(fwrite(&dataPoints->valueType, sizeof(ValueType), 1, outputFile) == 1);
    // write the data
    assert(fwrite(dataPoints->timestamps, sizeof(uint64_t)*dataPoints->count, 1, outputFile) == 1);
    assert(fwrite(dataPoints->values, sizeof(uint64_t)*dataPoints->count, 1, outputFile) == 1);

    // free memory
    freeDataPoints(dataPoints);
    fclose(inputFile);
    fclose(outputFile);
}
