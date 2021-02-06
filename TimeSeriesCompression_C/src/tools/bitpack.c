#include "compressors.h"
#include "decompressors.h"

ByteBuffer * value_compress_bitpack(DataBuffer * values) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter* bitWriter;
    int64_t value, prevValue = 0, frame[DEFAULT_FRAME_SIZE];
    int32_t pos = 0, maxLeastSignificantBits = 0;
    uint64_t diff, cursor = 0;

    // Allocate memory space
    compressedValues = malloc(sizeof(ByteBuffer));
    assert(compressedValues != NULL);
    compressedValues->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedValues->buffer != NULL);
    compressedValues->capacity = DEFAULT_BUFFER_SIZE;
    compressedValues->length = 0;

    bitWriter = bitWriterConstructor(compressedValues);

    // Read each value and compress it into byte byffer.
    while (cursor < values->length) {
        value = values->buffer[cursor++];
        // If current frame is full, then flush it.
        if (pos == DEFAULT_BUFFER_SIZE) {
            //flush();
            // Write the minimum leading zero into buffer as the header of current frame.
            bitWriterWriteBits(bitWriter, maxLeastSignificantBits, 6);

            // Write the significant bits of every value in current frame into buffer.
            for (int i = 0; i < pos; i++) {
                bitWriterWriteBits(bitWriter, frame[i], maxLeastSignificantBits);
            }
            // Reset the pos and the maximum number of least significant bit in the frame.
            pos = 0;
            maxLeastSignificantBits = 0;
        }

        // Calculate the difference between current value and previous value.
        diff = encodeZigZag64(value - prevValue);
        prevValue = value;

        // Try to update the maximum number of least significant bit.
        maxLeastSignificantBits =
            max(
                maxLeastSignificantBits,
                BITS_OF_LONG_LONG - leadingZerosCount64(diff)
            );

        // Store value into the current frame.
        frame[pos++] = diff;
    }

    // Flush the left value in current frame into buffer.
    // flush()
    if (pos != 0) {
        //flush();
        // Write the minimum leading zero into buffer as the header of current frame.
        bitWriterWriteBits(bitWriter, maxLeastSignificantBits, 6);

        // Write the significant bits of every value in current frame into buffer.
        for (int i = 0; i < pos; i++) {
            bitWriterWriteBits(bitWriter, frame[i], maxLeastSignificantBits);
        }
        // Reset the pos and the maximum number of least significant bit in the frame.
        pos = 0;
        maxLeastSignificantBits = 0;
    }

    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // Return byte buffer.
    return compressedValues;
}

DataBuffer* value_decompress_bitpack(ByteBuffer* values, uint64_t length) {
    // Declare variables
    DataBuffer* dataBuffer;
    BitReader* bitReader;
    int64_t value, prevValue = 0;
    uint32_t pos = DEFAULT_FRAME_SIZE, maxLeastSignificantBits = 0;
    uint64_t diff, cursor = 0;

    // Allocate memory space
    dataBuffer = malloc(sizeof(DataBuffer));
    assert(dataBuffer != NULL);
    dataBuffer->buffer = malloc(length * sizeof(uint64_t));
    assert(dataBuffer->buffer != NULL);
    dataBuffer->length = length;

    bitReader = bitReaderConstructor(values);

    // Decompress each timestamp from byte buffer
    while (cursor < length) {

        // If current compressed frame reach the end, read next maximum number of least
        // significant bit.
        if (pos == DEFAULT_FRAME_SIZE) {
            maxLeastSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 6);
            pos = 0;
        }
        // Decompress the difference in current frame according to the value of maxLeastSignificantBits
        diff = decodeZigZag64(bitReaderNextLong(bitReader, maxLeastSignificantBits));
        // Restore the value.
        value = diff + prevValue;

        // update predictor and position.
        prevValue = value;
        pos++;

        // return value;
        // Store current value into data buffer
        dataBuffer->buffer[cursor++] = value;
    }

    return dataBuffer;
}