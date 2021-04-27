#include "compressors.h"
#include "decompressors.h"
#include "bucket_counter.h"

static const uint32_t DELTA_MASK_3 = 0b10 << 3;
static const uint32_t DELTA_MASK_5 = 0b110 << 5;
static const uint32_t DELTA_MASK_9 = 0b1110 << 9;

static inline void flushZeros(BitWriter *bitWriter, int32_t *storedZeros) {
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
            // Write the number of cached zeros using 3 bits(i.e. '00'+3)
            bitWriterWriteBits(bitWriter, *storedZeros, 3);
            *storedZeros = 0;
        }
        else if ((*storedZeros) < 32) {
            // Write '1' control bit(i.e. '01'+5)
            bitWriterWriteOneBit(bitWriter);
            // Write the number of cached zeros using 5 bits
            bitWriterWriteBits(bitWriter, *storedZeros, 5);
            *storedZeros = 0;
        }
        else {
            // Write '1' control bit(i.e. '01'+5)
            bitWriterWriteOneBit(bitWriter);
            // Write 32 cached zeros
            bitWriterWriteBits(bitWriter, 0b11111, 5);
            *storedZeros -= 31;
        }
    }
}

ByteBuffer * timestamp_compress_rle(ByteBuffer * tsByteBuffer) {
    // Declare variables
    ByteBuffer *compressedTimestamps;
    BitWriter* bitWriter;
    int64_t timestamp, prevTimestamp = 0;
    int32_t newDelta, deltaOfDelta, prevDelta = 0;
    uint32_t leastBitLength, storedZeros = 0;
    uint64_t cursor = 0, tsCount = tsByteBuffer->length / sizeof(uint64_t),
        *tsBuffer = (uint64_t*)tsByteBuffer->buffer;;

    // Allocate memory space
    compressedTimestamps = malloc(sizeof(ByteBuffer));
    assert(compressedTimestamps != NULL);
    compressedTimestamps->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedTimestamps->buffer != NULL);
    compressedTimestamps->capacity = DEFAULT_BUFFER_SIZE;
    compressedTimestamps->length = 0;

    bitWriter = bitWriterConstructor(compressedTimestamps);

    // Write the header of current block for supporting millisecond.
    timestamp = tsBuffer[cursor++];
    bitWriterWriteLong(bitWriter, timestamp);
    prevTimestamp = timestamp;

    // Read each timestamp and compress it into byte byffer.
    while (cursor < tsCount) {

        // Calculate the delta-of-delta of timestamps.
        timestamp = tsBuffer[cursor++];
        newDelta = (int32_t)(timestamp - prevTimestamp);
        deltaOfDelta = newDelta - prevDelta;

        // If current delta and previous one is same
        if (deltaOfDelta == 0) {
            storedZeros++;// Counting the continuous and same delta of timestamps
            //
            bukAdd(0);
        }
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
                //
                bukAdd(1);
                // '10'+3
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_3, 5);
                break;
            case 4:
            case 5:
                //
                bukAdd(2);
                // '110'+5
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_5, 8);
                break;
            case 6:
            case 7:
            case 8:
            case 9:
                //
                bukAdd(3);
                // '1110'+9
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_9, 13);
                break;
            case 10:
            case 11:
            case 12:
            default:
                //
                bukAdd(4);
                // '1111'+32
                bitWriterWriteBits(bitWriter, 0b1111, 4); // Write '1111' control bits.
                // Since it only takes 4 bytes(i.e. 32 bits) to save a unix timestamp in seconds, we write
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

ByteBuffer* timestamp_decompress_rle(ByteBuffer* timestamps, uint64_t count) {
    // Declare variables
    ByteBuffer* byteBuffer;
    BitReader* bitReader;
    int64_t timestamp, prevTimestamp = 0;
    int64_t newDelta, deltaOfDelta = 0, prevDelta = 0;
    uint64_t cursor = 0, *tsBuffer;
    uint32_t controlBits, storedZeros = 0;

    // Allocate memory space
    byteBuffer = malloc(sizeof(ByteBuffer));
    assert(byteBuffer != NULL);
    byteBuffer->length = count * sizeof(uint64_t);
    byteBuffer->capacity = byteBuffer->length;
    byteBuffer->buffer = malloc(byteBuffer->length);
    assert(byteBuffer->buffer != NULL);

    tsBuffer = (uint64_t*)byteBuffer->buffer;
    bitReader = bitReaderConstructor(timestamps);

    // Get the head of current block.
    prevTimestamp = bitReaderNextLong(bitReader, BITS_OF_LONG_LONG);
    tsBuffer[cursor++] = prevTimestamp;

    // Decompress each timestamp from byte buffer
    while (cursor < count) {
        // If storedZeros != 0, previous and current timestamp interval/delta is same,
        // just update prevTimestamp and storedZeros, and return prevTimestamp.
        if (storedZeros > 0) {
            storedZeros--;
            prevTimestamp = prevDelta + prevTimestamp;
            //return prevTimestamp;
            tsBuffer[cursor++] = prevTimestamp;
            continue;
        }

        // Read timestamp control bits.
        controlBits = bitReaderNextControlBits(bitReader, 4);
        switch (controlBits)
        {
        case 0b0:
            // '0' bit (i.e. previous and current timestamp interval(delta) is same).
            // Get next the number of consecutive zeros
            //getConsecutiveZeros();
            // Read consecutive zeros control bits.
            controlBits = bitReaderNextBit(bitReader);

            switch (controlBits) {
            case 0:
                storedZeros = (uint32_t)bitReaderNextLong(bitReader, 3);
                break;
            case 1:
                storedZeros = (uint32_t)bitReaderNextLong(bitReader, 5);
                break;
            }
            // Since we have reduced the 'storedZeros' by 1 when we
            // compress it, we need to restore it's value here.
            storedZeros++;

            // Continue decompression
            storedZeros--;
            prevTimestamp = prevDelta + prevTimestamp;
            //return prevTimestamp;
            tsBuffer[cursor++] = prevTimestamp;
            continue;
        case 0b10:
            // '10' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 3 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 3);
            break;
        case 0b110:
            // '110' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 5 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 5);
            break;
        case 0b1110:
            // '1110' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 9 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 9);
            break;
        case 0b1111:
            // '1111' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 32 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 32);
            // If current deltaOfDelta value is the special end sign, set the isClosed value to true
            // (i.e. this buffer reach the end).
            break;
        default:
            break;
        }
        // Decode the deltaOfDelta value.
        deltaOfDelta = decodeZigZag32((int32_t)deltaOfDelta);

        // Since we have reduced the 'delta-of-delta' by 1 when we compress the 'delta-of-delta',
        // we restore the value here.
        if (deltaOfDelta >= 0) deltaOfDelta++;

        // Calculate the new delta and timestamp.
        newDelta = prevDelta + deltaOfDelta;
        timestamp = prevTimestamp + newDelta;

        // update prevDelta and prevTimestamp
        prevDelta = newDelta;
        prevTimestamp = timestamp;

        // return prevTimestamp;
        // Store current timestamp into data buffer
        tsBuffer[cursor++] = timestamp;
    }

    return byteBuffer;
}
