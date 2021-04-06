#include "compressors.h"
#include "decompressors.h"

static const uint32_t DELTA_MASK_7 = 0b10 << 7;
static const uint32_t DELTA_MASK_9 = 0b110 << 9;
static const uint32_t DELTA_MASK_12 = 0b1110 << 12;

ByteBuffer *timestamp_compress_gorilla(ByteBuffer *tsByteBuffer) {
    // Declare variables
    ByteBuffer *compressedTimestamps;
    BitWriter *bitWriter;
    int64_t timestamp, prevTimestamp = 0;
    int32_t newDelta, deltaOfDelta, prevDelta = 0;
    uint32_t leastBitLength;
    uint64_t cursor = 0, count = tsByteBuffer->length / sizeof(uint64_t),
        *uncompressed_t = (uint64_t*)tsByteBuffer->buffer;

    // Allocate memory space for byte buffer
    compressedTimestamps = malloc(sizeof(ByteBuffer));
    assert(compressedTimestamps != NULL);
    compressedTimestamps->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedTimestamps->buffer != NULL);
    compressedTimestamps->capacity = DEFAULT_BUFFER_SIZE;
    compressedTimestamps->length = 0;

    bitWriter = bitWriterConstructor(compressedTimestamps);

    // write the header in big-endian mode
    timestamp = uncompressed_t[cursor++];
    bitWriterWriteBits(bitWriter, timestamp, BITS_OF_LONG_LONG);
    prevTimestamp = timestamp;
    prevDelta = 0;

    // Read each timestamp and compress it into byte byffer.
    //while (cursor < timestamps->length) {
    while (cursor < count) {

        // Calculate the delta of delta of timestamp.
        timestamp = uncompressed_t[cursor++];

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
                // '10'+7
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_7, 9);
                break;
            case 8:
            case 9:
                // '110'+9
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_9, 12);
                break;
            case 10:
            case 11:
            case 12:
                // '1110'+12
                bitWriterWriteBits(bitWriter, deltaOfDelta | DELTA_MASK_12, 16);
                break;
            default:
                // '1111'+32
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

ByteBuffer *value_compress_gorilla(ByteBuffer *valByteBuffer) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter *bitWriter;
    int64_t value, prevValue = 0;
    uint32_t leadingZeros, trailingZeros, significantBits;
    uint32_t prevLeadingZeros = BITS_OF_LONG_LONG;
    uint32_t prevTrailingZeros = BITS_OF_LONG_LONG;
    uint64_t 
        diff, cursor = 0,
        count = valByteBuffer->length / sizeof(uint64_t),
        *uncompressed_v = (uint64_t*)valByteBuffer->buffer;

    // Allocate memory space
    compressedValues = malloc(sizeof(ByteBuffer));
    assert(compressedValues != NULL);
    compressedValues->buffer = malloc(DEFAULT_BUFFER_SIZE);
    assert(compressedValues->buffer != NULL);
    compressedValues->capacity = DEFAULT_BUFFER_SIZE;
    compressedValues->length = 0;

    bitWriter = bitWriterConstructor(compressedValues);

    // write the header in big-endian mode
    value = uncompressed_v[cursor++];
    bitWriterWriteBits(bitWriter, value, BITS_OF_LONG_LONG);
    prevValue = value;
    if (prevValue == 0) {
        prevLeadingZeros = 0;
        prevTrailingZeros = 0;
    }
    else {
        prevLeadingZeros = leadingZerosCount64(prevValue);
        prevTrailingZeros = trailingZerosCount64(prevValue);
    }

    // Read each value and compress it into byte byffer.
    while (cursor < count) {

        // Calculate the XOR difference between prediction and current value to be compressed.
        value = uncompressed_v[cursor++];
        diff = prevValue^value;

        // updata previous value
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
                for special situation in which high precision xor value occurred.
                In original implementation, when leading zeros of xor residual is more than 32,
                you need to store the excess part in the meaningful bits, which cost more bits.
                Actually you need calculate the distribution of the leading zeros of the xor residual first,
                and then decide whether it needs 5 bits or 6 bits to save the leading zeros for best compression ratio.
                */
                bitWriterWriteBits(bitWriter, leadingZeros, 6);// Write the number of leading zeros input the next 6 bits
                // Since 'significantBits == 0' is unoccupied, we can just store 'significantBits - 1' to
                // cover a larger range and avoid the situation when 'significantBits == 64
                bitWriterWriteBits(bitWriter, significantBits - 1, 6);// Write the length of meaningful bits input the next 6 bits

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

ByteBuffer *timestamp_decompress_gorilla(ByteBuffer *timestamps, uint64_t count) {
    // Declare variables
    ByteBuffer *byteBuffer;
    BitReader *bitReader;
    int64_t timestamp, prevTimestamp = 0;
    int64_t newDelta, deltaOfDelta = 0, prevDelta = 0;
    uint64_t cursor = 0, *tsBuffer;
    uint32_t controlBits;

    // Allocate memory space
    byteBuffer = malloc(sizeof(ByteBuffer));
    assert(byteBuffer != NULL);
    byteBuffer->length = count  *sizeof(uint64_t);
    byteBuffer->capacity = byteBuffer->length;
    byteBuffer->buffer = malloc(byteBuffer->length);
    assert(byteBuffer->buffer != NULL);

    tsBuffer = (uint64_t*)byteBuffer->buffer;
    bitReader = bitReaderConstructor(timestamps);

    // get the header in bit-endian mode
    timestamp = bitReaderNextLong(bitReader, BITS_OF_LONG_LONG);
    tsBuffer[cursor++] = timestamp;
    prevTimestamp = timestamp;
    prevDelta = 0;

    // Decompress each timestamp from byte buffer
    while (cursor < count) {
        controlBits = bitReaderNextControlBits(bitReader, 4);

        switch (controlBits)
        {
        case 0b0:
            // '0' bit (i.e. previous and current timestamp interval(delta) is same).
            prevTimestamp = prevDelta + prevTimestamp;
            // Store current timestamp into data buffer
            //dataBuffer->buffer[cursor++] = prevTimestamp;
            tsBuffer[cursor++] = prevTimestamp;
            continue;
        case 0b10:
            // '10' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 7 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 7);
            break;
        case 0b110:
            // '110' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 9 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 9);
            break;
        case 0b1110:
            // '1110' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 12 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 12);
            break;
        case 0b1111:
            // '1111' bits (i.e. deltaOfDelta value encoded by zigzag32 is stored input next 32 bits).
            deltaOfDelta = bitReaderNextLong(bitReader, 32);
            break;
        default:
            break;
        }

        // Decode the deltaOfDelta value.
        deltaOfDelta = decodeZigZag32((int32_t)deltaOfDelta);
        // Since we have decreased the 'delta-of-delta' by 1 when we compress the it,
        // we restore it's value here.
        if (deltaOfDelta >= 0) deltaOfDelta++;

        // Calculate the new delta and timestamp.
        //prevDelta += deltaOfDelta;
        newDelta = prevDelta + deltaOfDelta;
        //prevTimestamp += prevDelta;
        timestamp = prevTimestamp + newDelta;

        // update prevDelta and prevTimestamp
        prevDelta = newDelta;
        prevTimestamp = timestamp;

        // return prevTimestamp;
        // Store current timestamp into data buffer
        //dataBuffer->buffer[cursor++] = prevTimestamp;
        tsBuffer[cursor++] = prevTimestamp;
    }

    return byteBuffer;
}

ByteBuffer *value_decompress_gorilla(ByteBuffer *values, uint64_t count) {
    // Declare variables
    ByteBuffer *byteBuffer;
    BitReader *bitReader;
    int64_t 
        value = 0, prevValue = 0;
    uint64_t
        diff, cursor = 0, *decompressed_v;
    uint32_t 
        prevLeadingZeros = 0, prevTrailingZeros = 0,
        leadingZeros, trailingZeros,
        controlBits, significantBitLength;

    // Allocate memory space
    byteBuffer = malloc(sizeof(ByteBuffer));
    assert(byteBuffer != NULL);
    byteBuffer->length = count  *sizeof(uint64_t);
    byteBuffer->capacity = byteBuffer->length;
    byteBuffer->buffer = malloc(byteBuffer->length);
    assert(byteBuffer->buffer != NULL);

    decompressed_v = (uint64_t*)byteBuffer->buffer;
    bitReader = bitReaderConstructor(values);

    // get the header in bit-endian mode
    value = bitReaderNextLong(bitReader, BITS_OF_LONG_LONG);
    decompressed_v[cursor++] = value;
    prevValue = value;
    if (prevValue == 0) {
        prevLeadingZeros = 0;
        prevTrailingZeros = 0;
    }
    else {
        prevLeadingZeros = leadingZerosCount64(prevValue);
        prevTrailingZeros = trailingZerosCount64(prevValue);
    }

    // Decompress each value from byte buffer and write it into data byffer.
    while (cursor < count) {
        // Read next value's control bits.
        controlBits = bitReaderNextControlBits(bitReader, 2);

        // Match the case corresponding to the control bits.
        switch (controlBits)
        {
        case 0b0:
            // '0' bit (i.e. prediction(previous) and current value is same)
            value = prevValue;
            break;

        case 0b10:
            // '10' bits (i.e. the block of current value meaningful bits falls within
            // the scope of prediction(previous) meaningful bits)

            // Read the significant bits and restore the xor value.
            significantBitLength = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
            diff = bitReaderNextLong(bitReader, significantBitLength) << prevTrailingZeros;
            value = prevValue ^ diff;
            prevValue = value;

            // Update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZerosCount64(diff);
            prevTrailingZeros = trailingZerosCount64(diff);
            break;

        case 0b11:
            // '11' bits (i.e. the block of current value meaningful bits doesn't falls within
            // the scope of previous meaningful bits)
            // Update the number of leading and trailing zeros.
            //prevLeadingZeros = (uint32_t)bitReaderNextLong(bitReader, 6);
            leadingZeros = (uint32_t)bitReaderNextLong(bitReader, 6);
            significantBitLength = (uint32_t)bitReaderNextLong(bitReader, 6);
            // Since we have decreased the length of significant bits by 1 for larger compression range
            // when we compress it, we restore it's value here.
            significantBitLength++;

            // Read the significant bits and restore the xor value.
            //prevTrailingZeros = BITS_OF_LONG_LONG - leadingZeros - significantBitLength;
            trailingZeros = BITS_OF_LONG_LONG - leadingZeros - significantBitLength;
            diff = bitReaderNextLong(bitReader, significantBitLength) << trailingZeros;
            value = prevValue ^ diff;
            prevValue = value;

            // Update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;

            break;
        default:
            break;
        }
        // return value;
        // Store current value into data buffer
        decompressed_v[cursor++] = value;
    }

    return byteBuffer;
}