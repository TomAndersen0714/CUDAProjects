#include "compressors.h"

static const uint32_t DELTA_3_MASK = 0b10 << 3;
static const uint32_t DELTA_5_MASK = 0b110 << 5;
static const uint32_t DELTA_9_MASK = 0b1110 << 9;

static inline void flushZeros(BitWriter* bitWriter, int32_t* storedZeros) {
    while ((*storedZeros) > 0) {
        // Tips: since storedZeros == 0 is unoccupied, we can utilize it to cover a larger range
        (*storedZeros)--;

        // Write '0' control bit
        bitWriterWriteZeroBit(bitWriter);

        if ((*storedZeros) < 8) {
            // Tips: if there is too much case, you can use the number of leading zeros as
            // the condition for using switch-case code block.

            // Write '0' control bit
            bitWriterWriteZeroBit(bitWriter);
            // Write the number of cached zeros using 3 bits
            bitWriterWriteBits(bitWriter, *storedZeros, 3);
            *storedZeros = 0;
        }
        else if ((*storedZeros) < 32) {
            // Write '1' control bit
            bitWriterWriteOneBit(bitWriter);
            // Write the number of cached zeros using 5 bits
            bitWriterWriteBits(bitWriter, *storedZeros, 5);
            *storedZeros = 0;
        }
        else {
            // Write '1' control bit
            bitWriterWriteOneBit(bitWriter);
            // Write 32 cached zeros
            bitWriterWriteBits(bitWriter, 0b11111, 5);
            *storedZeros -= 31;
        }
    }
}

ByteBuffer * timestamp_compress_rle(UncompressedData * timestamps) {
    // Declare variables
    ByteBuffer *compressedTimestamps;
    BitWriter* bitWriter;
    int64_t timestamp, prevTimestamp = 0;
    int32_t newDelta, deltaOfDelta, prevDelta = 0;
    uint32_t leastBitLength, storedZeros = 0;
    uint64_t cursor = 0;

    // Allocate memory space
    compressedTimestamps = malloc(sizeof(ByteBuffer));
    assert(compressedTimestamps != NULL);
    compressedTimestamps->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedTimestamps->buffer != NULL);
    compressedTimestamps->capacity = DEFAULT_BUFFER_SIZE;
    compressedTimestamps->length = 0;

    bitWriter = bitWriterConstructor(compressedTimestamps);

    // Write the header of current block for supporting millisecond.
    timestamp = timestamps->buffer[cursor++];
    bitWriterWriteLong(bitWriter, timestamp);
    prevTimestamp = timestamp;

    // Read each timestamp and compress it into byte byffer.
    while (cursor < timestamps->length) {

        // Calculate the delta-of-delta of timestamps.
        timestamp = timestamps->buffer[cursor++];
        newDelta = (int32_t)(timestamp - prevTimestamp);
        deltaOfDelta = newDelta - prevDelta;

        // If current delta and previous one is same
        if (deltaOfDelta == 0)
            storedZeros++;// Counting the continuous and same delta of timestamps
        else {
            // Write the privious stored zeros to the buffer.
            flushZeros(bitWriter, &storedZeros);

            // Tips: since deltaOfDelta == 0 is unoccupied, we can utilize it to cover a larger range.
            if (deltaOfDelta > 0) deltaOfDelta--;
            // Convert signed value to unsigned value for compression.
            deltaOfDelta = encodeZigZag32(deltaOfDelta);

            leastBitLength = BITS_OF_INT - leadingZerosCount32(deltaOfDelta);
            // Match the deltaOfDelta to the three case as follow.
            switch (leastBitLength) {
            case 0:
            case 1:
            case 2:
            case 3:
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_3_MASK, 5);
                break;
            case 4:
            case 5:
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_5_MASK, 8);
                break;
            case 6:
            case 7:
            case 8:
            case 9:
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_9_MASK, 13);
                break;
            case 10:
            case 11:
            case 12:
            default:
                bitWriterWriteBits(bitWriter, 0b1111, 4); // Write '1111' control bits.
                // Since it only takes 4 bytes(i.e. 32 bits) to save a unix timestamp input second, we write
                // delta-of-delta using 32 bits.
                bitWriterWriteBits(bitWriter, deltaOfDelta, 32);
                break;
            }
            prevDelta = newDelta;
        }
        prevTimestamp = timestamp;
    }

    // Write all stored zeros into the buffer.
    flushZeros(bitWriter, &storedZeros);

    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // Return byte buffer.
    return compressedTimestamps;
}

