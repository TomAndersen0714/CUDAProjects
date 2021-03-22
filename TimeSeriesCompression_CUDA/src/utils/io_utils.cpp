#include "io_utils.h"

DataPoints *readUncompressedFile(
    FILE *inputFile, ValueType timestampType, ValueType valueType
) {
    // Declare variables
    uint64_t count, cursor;
    uint64_t *timestamps, *values;
    DataPoints* dataPoints;

    // Get the number of data points and pre-allocate momery space for it
    if (fscanf(inputFile, "%llu", &count) != 1) exit(EXIT_FAILURE);

    timestamps = (uint64_t *)malloc(sizeof(uint64_t)*count);
    values = (uint64_t *)malloc(sizeof(uint64_t)*count);
    assert(timestamps != NULL && values != NULL);

    // Parse every line of file
    cursor = 0;
    if (timestampType == _LONG_LONG && valueType == _LONG_LONG) {
        while (cursor < count) {
            // Format the string and convert to the values in corresponding 
            // type, then write the bits of values into the specific address
            assert(fscanf(inputFile, "%llu %lld", &timestamps[cursor], &values[cursor]) == 2);
            cursor++;
        }
    }
    else if (timestampType == _LONG_LONG && valueType == _DOUBLE) {
        while (cursor < count) {
            assert(fscanf(inputFile, "%llu %lf", &timestamps[cursor], &values[cursor]) == 2);
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
    FILE *outputFile, CompressedDPs *compressedDPs
) {
    uint16_t
        frame = compressedDPs->metadata->frame;
    uint64_t
        count = compressedDPs->metadata->count,
        frame_b = frame*BYTES_OF_LONG_LONG,
        thd = (count + frame - 1) / frame,
        start = 0;

    // write metadata as the header of compressed file
    assert(fwrite(compressedDPs->metadata, sizeof(Metadata), 1, outputFile) == 1);

    // write the size of compressed frames
    assert(fwrite(compressedDPs->tsLens, BYTES_OF_SHORT*thd, 1, outputFile) == 1);
    assert(fwrite(compressedDPs->valLens, BYTES_OF_SHORT*thd, 1, outputFile) == 1);

    // compact and write the compressed data frame-by-frame
    // compact and write the compressed timestamps
    for (int i = 0; i < thd; i++) {
        assert(fwrite(compressedDPs->timestamps + start,
            compressedDPs->tsLens[i], 1, outputFile) == 1);
        start += frame_b;
    }

    // compact and write the compressed values
    start = 0;
    for (int i = 0; i < thd; i++) {
        assert(fwrite(compressedDPs->values + start,
            compressedDPs->valLens[i], 1, outputFile) == 1);
        start += frame_b;
    }

}

CompressedDPs *readCompressedFile(
    FILE *inputFile
) {
    // declare variables
    CompressedDPs *compressedDPs;
    Metadata *metadata;
    uint16_t
        *tsLens, *valLens;
    byte
        *timestamps, *values;
    uint64_t
        thd, tsSize = 0, valSize = 0; // the byte size of compressed timestamps and values

    // get the metadata of compressed data
    metadata = (Metadata *)malloc(sizeof(Metadata));
    assert(metadata != NULL);
    assert(fread(metadata, sizeof(Metadata), 1, inputFile) == 1);

    // get the compressed data
    thd = (metadata->count + metadata->frame - 1) / metadata->frame;
    tsLens = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    valLens = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    assert(tsLens != NULL);
    assert(valLens != NULL);
    // get the size of compressed frames
    assert(fread(tsLens, BYTES_OF_SHORT*thd, 1, inputFile) == 1);
    assert(fread(valLens, BYTES_OF_SHORT*thd, 1, inputFile) == 1);

    // get the compacted and compressed values
    for (int i = 0; i < thd; i++) { // get the size of compressed data
        tsSize += tsLens[i];
        valSize += valLens[i];
    }
    timestamps = (byte*)malloc(tsSize);
    values = (byte*)malloc(valSize);
    assert(timestamps != NULL && values != NULL);
    assert(fread(timestamps, tsSize, 1, inputFile) == 1);
    assert(fread(values, valSize, 1, inputFile) == 1);

    // return the compressed data points and metadata
    compressedDPs = (CompressedDPs*)malloc(sizeof(CompressedDPs));
    assert(compressedDPs != NULL);
    compressedDPs->metadata = metadata;
    compressedDPs->tsLens = tsLens;
    compressedDPs->valLens = valLens;
    compressedDPs->timestamps = timestamps;
    compressedDPs->values = values;

    return compressedDPs;
}

void writeDecompressedData(
    FILE *outputFile, DataPoints *decompressedDPs
) {
    // Write the number of data points into file
    fprintf(outputFile, "%lld\n", decompressedDPs->count);

    // Write the data points into file
    uint64_t cursor = 0;
    if (decompressedDPs->timestampType == _LONG_LONG
        && decompressedDPs->valueType == _LONG_LONG) {
        while (cursor < decompressedDPs->count) {
            fprintf(
                outputFile, "%lld %lld\n",
                decompressedDPs->timestamps[cursor],
                decompressedDPs->values[cursor]
            );
            cursor++;
        }
    }
    else if (decompressedDPs->timestampType == _LONG_LONG
        && decompressedDPs->valueType == _DOUBLE) {
        while (cursor < decompressedDPs->count) {
            fprintf(
                outputFile, "%lld %lf\n",
                decompressedDPs->timestamps[cursor],
                decompressedDPs->values[cursor]
            );
            cursor++;
        }
    }
    // Timestamp must be long long type
    // else if (timestampType == _DOUBLE_ && valueType == _LONG_LONG_)
    // else if (timestampType == _DOUBLE_ && valueType == _DOUBLE_)
}
