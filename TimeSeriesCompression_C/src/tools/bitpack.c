#include "compressors.h"
#include "decompressors.h"

ByteBuffer* value_compress_bitpack(ByteBuffer* valByteBuffer) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter* bitWriter;
    int64_t value, prevValue = 0, frame[DEFAULT_FRAME_SIZE];
    int32_t pos = 0, maxLeastSignificantBits = 0;
    uint64_t diff, cursor = 0,
        valCount = valByteBuffer->length / sizeof(uint64_t),
        *valBuffer = (uint64_t*)valByteBuffer->buffer;

    // Allocate memory space
    compressedValues = malloc(sizeof(ByteBuffer));
    assert(compressedValues != NULL);
    compressedValues->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedValues->buffer != NULL);
    compressedValues->capacity = DEFAULT_BUFFER_SIZE;
    compressedValues->length = 0;

    bitWriter = bitWriterConstructor(compressedValues);

    // Read each value and compress it into byte byffer.
    while (cursor < valCount) {
        value = valBuffer[cursor++];
        // If current frame is full, then flush it.
        if (pos == DEFAULT_FRAME_SIZE) {
            // If all values in the frame equals zero(i.e. 'maxLeastSignificantBits' equals 0)
            // we just store the 0b0 in next 6 bits and clear frame
            if (maxLeastSignificantBits == 0) {
                bitWriterWriteBits(bitWriter, 0b0, 6);
                pos = 0;
            }
            else {
                //flush();
                // Since 'maxLeastSignificantBits' could not equals to '0',
                // we leverage this point to cover range [1~64] by storing
                // 'maxLeastSignificantBits-1'
                bitWriterWriteBits(bitWriter, maxLeastSignificantBits - 1, 6);

                // Write the significant bits of every value in current frame into buffer.
                for (int i = 0; i < pos; i++) {
                    bitWriterWriteBits(bitWriter, frame[i], maxLeastSignificantBits);
                }
                // Reset the pos and the maximum number of least significant bit in the frame.
                pos = 0;
                maxLeastSignificantBits = 0;
            }

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
        // If all values in the frame equals zero(i.e. 'maxLeastSignificantBits' equals 0)
        // we just store the 0b0 in next 6 bits
        if (maxLeastSignificantBits == 0) {
            bitWriterWriteBits(bitWriter, 0b0, 6);
        }
        else {
            // Since 'maxLeastSignificantBits' could not equals to '0',
            // we leverage this point to cover range [1~64] by storing
            // 'maxLeastSignificantBits-1'
            bitWriterWriteBits(bitWriter, maxLeastSignificantBits - 1, 6);

            // Write the significant bits of every value in current frame into buffer.
            for (int i = 0; i < pos; i++) {
                bitWriterWriteBits(bitWriter, frame[i], maxLeastSignificantBits);
            }
        }
        // Reset the pos and the maximum number of least significant bit in the frame.
        pos = 0;
        maxLeastSignificantBits = 0;
    }

    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // Free the allocated memory
    free(bitWriter);

    // Return byte buffer.
    return compressedValues;
}

ByteBuffer* value_decompress_bitpack(ByteBuffer* values, uint64_t count) {
    // Declare variables
    ByteBuffer* byteBuffer;
    BitReader* bitReader;
    int64_t value, prevValue = 0;
    uint32_t pos = DEFAULT_FRAME_SIZE, maxLeastSignificantBits = 0;
    uint64_t diff = 0, cursor = 0, *valBuffer;;

    // Allocate memory space
    byteBuffer = malloc(sizeof(ByteBuffer));
    assert(byteBuffer != NULL);
    byteBuffer->length = count * sizeof(uint64_t);
    byteBuffer->capacity = byteBuffer->length;
    byteBuffer->buffer = malloc(byteBuffer->length);
    assert(byteBuffer->buffer != NULL);

    valBuffer = (uint64_t*)byteBuffer->buffer;
    bitReader = bitReaderConstructor(values);

    // Decompress each timestamp from byte buffer
    while (cursor < count) {

        // If current compressed frame reach the end, read next maximum number of least
        // significant bit.
        if (pos == DEFAULT_FRAME_SIZE) {
            maxLeastSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 6);
            pos = 0;
        }

        // If maxLeastSignificantBits equals zero, the all diff value in
        // current frame is zero.(i.e. current value and previous is same)
        if (maxLeastSignificantBits == 0) {
            // Restore the value.
            value = diff + prevValue;
        }
        else {
            // Decompress the difference in current frame according to the value of maxLeastSignificantBits
            diff = decodeZigZag64(
                bitReaderNextLong(
                    // Since we compressed 'maxLeastSignificantBits-1' into buffer,
                    // we restore it here
                    bitReader, maxLeastSignificantBits + 1
                )
            );
            // Restore the value.
            value = diff + prevValue;
        }
        // update predictor and position.
        prevValue = value;
        // Store current value into data buffer
        pos++;
        valBuffer[cursor++] = value;
    }

    // Free the allocated memory
    free(bitReader);

    return byteBuffer;
}