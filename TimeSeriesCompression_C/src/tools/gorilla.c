#include "compressors.h"

static const int32_t DELTA_7_MASK = 0b10 << 7;
static const int32_t DELTA_9_MASK = 0b110 << 9;
static const int32_t DELTA_12_MASK = 0b1110 << 12;

ByteBuffer* timestamp_compress_gorilla(UncompressedData* timestamps) {
    // Declare variables
    ByteBuffer *compressedTimestamps;
    BitWriter* bitWriter;
    int64_t timestamp, prevTimestamp = 0;
    int32_t newDelta, deltaOfDelta, prevDelta = 0;
    uint32_t leastBitLength;
    uint64_t cursor = 0;

    // Allocate memory space for byte buffer
    compressedTimestamps = malloc(sizeof(ByteBuffer));
    assert(compressedTimestamps != NULL);
    compressedTimestamps->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedTimestamps->buffer != NULL);
    compressedTimestamps->capacity = DEFAULT_BUFFER_SIZE;
    compressedTimestamps->length = 0;

    bitWriter = bitWriterConstructor(compressedTimestamps);

    // Read each timestamp and compress it into byte byffer.
    while (cursor < timestamps->length) {

        // Calculate the delta of delta of timestamp.
        timestamp = timestamps->buffer[cursor++];

        // PS: Since original implementation in gorilla paper requires that delta-of-delta
        // of timestamps can be stored by a signed 32-bit value, it doesn't support
        // compression timestamps in millisecond as good as second.
        newDelta = (int32_t)(timestamp - prevTimestamp);
        deltaOfDelta = newDelta - prevDelta;

        // If current delta and previous delta is same
        if (deltaOfDelta == 0) {
            // Write '0' bit as control bit(i.e. previous and current delta value is same).
            bitWriterWriteZeroBit(bitWriter);
        }
        else {
            // Tips: since deltaOfDelta == 0 is unoccupied, we can utilize it to cover a larger range.
            if (deltaOfDelta > 0) deltaOfDelta--;
            // Convert signed value to unsigned value for compression.
            deltaOfDelta = encodeZigZag32(deltaOfDelta);

            leastBitLength = BITS_OF_INT - leadingZerosCount32(deltaOfDelta);
            // Match the deltaOfDelta to the these case as follow.
            switch (leastBitLength) {
            case 0:
            case 1:
            case 2:
            case 3:
            case 4:
            case 5:
            case 6:
            case 7:
                ////
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_7_MASK, 9);
                break;
            case 8:
            case 9:
                ////
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_9_MASK, 12);
                break;
            case 10:
            case 11:
            case 12:
                ////
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_12_MASK, 16);
                break;
            default:
                // Write '1111' control bits.
                bitWriterWriteBits(bitWriter, 0b1111, 4);
                // Since it only takes 4 bytes(i.e. 32 bits) to save a unix timestamp input second, we write
                // delta-of-delta using 32 bits.
                bitWriterWriteBits(bitWriter, deltaOfDelta, 32);
                break;
            }

            // update previous delta of timestamp
            prevDelta = newDelta;
        }
        // update previous timestamp
        prevTimestamp = timestamp;
    }
    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // Return byte buffer.
    return compressedTimestamps;
}

ByteBuffer * value_compress_gorilla(UncompressedData * values) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter* bitWriter;
    int64_t value, prevValue = 0;
    uint32_t leadingZeros, trailingZeros, significantBits;
    uint32_t prevLeadingZeros = UINT32_MAX;
    uint32_t prevTrailingZeros = UINT32_MAX;
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

        // Calculate the XOR difference between prediction and current value to be compressed.
        value = values->buffer[cursor++];
        diff = prevValue^value;
        prevValue = value;

        // If previous value and current value is same
        if (diff == 0) {
            // Write '0' bit as entire control bit(i.e. prediction and current value is same).
            bitWriterWriteZeroBit(bitWriter);
        }
        else {
            leadingZeros = leadingZerosCount64(diff);
            trailingZeros = trailingZerosCount64(diff);

            // Write '1' bit as first control bit.
            bitWriterWriteOneBit(bitWriter);

            // If the scope of meaningful bits falls within the scope of previous meaningful bits,
            // i.e. there are at least as many leading zeros and as many trailing zeros as with
            // the previous value.
            if (leadingZeros >= prevLeadingZeros && trailingZeros >= prevTrailingZeros) {
                // Write current value into previous scope
                //writeInPrevScope(diff);

                // Write '0' bit as second control bit.
                bitWriterWriteZeroBit(bitWriter);

                // Write significant bits of difference value input the scope.
                significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;

                bitWriterWriteBits(bitWriter, diff >> prevTrailingZeros, significantBits);
            }
            else {
                // Write current value into new scope
                //writeInNewScope(diff, leadingZeros, trailingZeros);

                // Write '1' bit as second control bit.
                bitWriterWriteOneBit(bitWriter);
                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;

                /*
                Different from original implementation, 5 -> 6 bits to store the number of leading zeros,
                avoids the special situation when high precision xor value appears.
                In original implementation,when the leading zeros of xor residual is more than 32,
                you need to store the excess part in the meaningful bits, which cost more bits.
                Actually you need calculate the distribution of the leading zeros of the xor residual first,
                and then decide whether it needs 5 bits or 6 bits to save the leading zeros.
                */
                bitWriterWriteBits(bitWriter, leadingZeros, 6);// Write the number of leading zeros input the next 6 bits
                // Since 'significantBits == 0' is unoccupied, we can just store 'significantBits - 1' to
                // cover a larger range and avoid the situation when 'significantBits == 64'.

                // Write the length of meaningful bits input the next 6 bits
                bitWriterWriteBits(bitWriter, significantBits - 1, 6);

                // Write the meaningful bits of XOR
                bitWriterWriteBits(bitWriter, diff >> trailingZeros, significantBits);
            }
            // Update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
        }
    }
    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // Return byte buffer.
    return compressedValues;
}
