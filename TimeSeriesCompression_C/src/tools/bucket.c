#include "compressors.h"
#include "decompressors.h"

static const uint32_t DIFF_MASK_2 = 0b0 << 2;
static const uint32_t DIFF_MASK_4 = 0b10 << 4;
static const uint32_t DIFF_MASK_6 = 0b11 << 6;

ByteBuffer * value_compress_bucket(DataBuffer * values) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter* bitWriter;
    int64_t value, prevValue = 0;
    uint32_t leadingZeros, trailingZeros, significantBits,
        diffLeadingZeros, diffSignificantBits, leastSignificantBits;
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
        diff = prevValue ^ value;
        prevValue = value;

        if (diff == 0) {
            // Write '0' bit as entire control bit(i.e. prediction and current value is same).
            bitWriterWriteBits(bitWriter, 0b11, 2);
        }
        else {
            leadingZeros = leadingZerosCount64(diff);
            trailingZeros = trailingZerosCount64(diff);


            // If the scope of meaningful bits falls within the scope of previous meaningful bits,
            // i.e. there are at least as many leading zeros and as many trailing zeros as with
            // the previous value.
            if (leadingZeros >= prevLeadingZeros && trailingZeros >= prevTrailingZeros) {
                //writeInPrevScope(diff);
                // Write '10' as control bit
                bitWriterWriteBits(bitWriter, 0b10, 2);

                // Write significant bits of difference value input the scope.
                significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;

                // Since the first bit of significant bits must be '1', we can utilize it to store less bits.
                bitWriterWriteBits(bitWriter, diff >> prevTrailingZeros, significantBits - 1);
            }
            else {
                //writeInNewScope(diff, leadingZeros, trailingZeros);
                // Write '0' bit as second control bit.
                bitWriterWriteBits(bitWriter, 0b0, 1);

                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                diffLeadingZeros = encodeZigZag32(leadingZeros - prevLeadingZeros);
                diffSignificantBits = encodeZigZag32(leadingZeros + trailingZeros - prevLeadingZeros - prevTrailingZeros);
                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffLeadingZeros);

                switch (leastSignificantBits) {// [0,32]
                case 0:
                case 1:
                case 2:// diffLeadingZeros:[0,4)
                    // '0'+2
                    //bitWriterWriteZeroBit(bitWriter);
                    //bitWriterWriteBits(bitWriter, diffLeadingZeros, 2);
                    bitWriterWriteBits(bitWriter, diffLeadingZeros | DIFF_MASK_2, 3);
                    break;
                case 3:
                case 4:// diffLeadingZeros:[4,16)
                    // '10'+4
                    //bitWriterWriteBits(bitWriter, 0b10, 2);
                    //bitWriterWriteBits(bitWriter, diffLeadingZeros, 4);
                    bitWriterWriteBits(bitWriter, diffLeadingZeros | DIFF_MASK_4, 6);

                    break;
                default:// diffLeadingZeros:[16,32]
                    // '11'+6
                    //bitWriterWriteBits(bitWriter, 0b11, 2);
                    //bitWriterWriteBits(bitWriter, leadingZeros, 6);
                    bitWriterWriteBits(bitWriter, leadingZeros | DIFF_MASK_6, 8);
                    break;
                }

                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffSignificantBits);
                switch (leastSignificantBits) {
                case 0:
                case 1:
                case 2:// diffSignificantBits:[0,4)
                    // '0'+2
                    /*bitWriterWriteZeroBit(bitWriter);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 2);*/
                    bitWriterWriteBits(bitWriter, diffSignificantBits | DIFF_MASK_2, 3);
                    break;
                case 3:
                case 4:// diffSignificantBits:[4,16)
                    // '10'+4
                    /*bitWriterWriteBits(bitWriter, 0b10, 2);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 4);*/
                    bitWriterWriteBits(bitWriter, diffSignificantBits | DIFF_MASK_4, 6);
                    break;
                default:// diffSignificantBits:[16,32]
                    // '11'+6
                    /*bitWriterWriteBits(bitWriter, 0b11, 2);
                    bitWriterWriteBits(bitWriter, significantBits, 6);*/
                    bitWriterWriteBits(bitWriter, significantBits | DIFF_MASK_6, 8);
                    break;
                }

                // Since the first bit of significant bits must be '1', we can utilize it to store less bits.
                bitWriterWriteBits(
                    bitWriter, diff >> trailingZeros, significantBits - 1
                ); // Write the meaningful bits of XOR
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

DataBuffer* value_decompress_bucket(ByteBuffer* values, uint64_t length) {

    // Declare variables
    DataBuffer* dataBuffer;
    BitReader* bitReader;
    uint32_t prevLeadingZeros = 0, prevTrailingZeros = 0,
        leadingZeros, trailingZeros, significantBits,
        controlBits, diffLeadingZeros, diffSignificantBits;
    int64_t value = 0, prevValue = 0, diff;
    uint64_t cursor = 0;

    // Allocate memory space
    dataBuffer = malloc(sizeof(DataBuffer));
    assert(dataBuffer != NULL);
    dataBuffer->buffer = malloc(length * sizeof(uint64_t));
    assert(dataBuffer->buffer != NULL);
    dataBuffer->length = length;

    bitReader = bitReaderConstructor(values);

    // Read each value and compress it into byte byffer.
    while (cursor < length) {
        // Read next value's control bits.
        controlBits = bitReaderNextControlBits(bitReader, 2);

        // Match the case corresponding to the control bits.
        switch (controlBits) {
        case 0b11:
            // '11' bits (i.e. prediction(previous) and current value is same)
            value = prevValue;
            break;

        case 0b10:
            // '10' bits (i.e. the block of current value meaningful bits falls within
            // the scope of prediction(previous) meaningful bits)

            // Read the significant bits and restore the xor value.
            significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;

            // Since the first bit of significant bits must be '1', we reduce 'significantBitLength' by 1
            // when we compress the xor value(i.e. 'diff'), and we need to restore it here
            diff = bitReaderNextLong(bitReader, significantBits - 1) << prevTrailingZeros
                | 1ULL << (significantBits - 1 + prevTrailingZeros);
            value = prevValue ^ diff;
            prevValue = value;

            // Update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZerosCount64(diff);
            prevTrailingZeros = trailingZerosCount64(diff);
            break;

            /*case 0b0:*/
        default:
            // '0' bits (i.e. the block of current value meaningful bits doesn't falls within
            // the scope of previous meaningful bits)

            controlBits = bitReaderNextControlBits(bitReader, 2);
            switch (controlBits)
            {
            case 0b0:// diffLeadingZeros:[0,4)
                // '0'+2
                diffLeadingZeros = (uint32_t)bitReaderNextLong(bitReader, 2);
                leadingZeros = decodeZigZag32(diffLeadingZeros) + prevLeadingZeros;
                break;
            case 0b10:// diffLeadingZeros:[4,16)
                // '10'+4
                diffLeadingZeros = (uint32_t)bitReaderNextLong(bitReader, 4);
                leadingZeros = decodeZigZag32(diffLeadingZeros) + prevLeadingZeros;
                break;
                /*case 0b11:
                    break;*/
            default:// diffLeadingZeros:[16,32]
                // '11'+6
                leadingZeros = (uint32_t)bitReaderNextLong(bitReader, 6);
                break;
            }

            controlBits = bitReaderNextControlBits(bitReader, 2);
            switch (controlBits)
            {
            case 0b0:// diffSignificantBits:[0,4)
                // '0'+2
                diffSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 2);
                significantBits =
                    BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros
                    - decodeZigZag32(diffSignificantBits);
                break;
            case 0b10:// diffSignificantBits:[4,16)
                // '10'+4
                diffSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 4);
                // significantBits == prevSignificantBits - decodeZigZag32(diffSignificantBits)
                significantBits =
                    BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros - decodeZigZag32(diffSignificantBits);
                break;
            default:// diffSignificantBits:[16,32]
                // '11'+6
                significantBits = (uint32_t)bitReaderNextLong(bitReader, 6);
                break;
            }
            trailingZeros = BITS_OF_LONG_LONG - significantBits - leadingZeros;

            // Since the first bit of significant bits must be '1', we reduce 'significantBitLength' by 1
            // when we compress the xor value(i.e. 'diff'), and we need to restore it
            diff = bitReaderNextLong(bitReader, significantBits - 1) << trailingZeros
                | 1ULL << (significantBits - 1 + trailingZeros);
            
            value = prevValue ^ diff;
            prevValue = value;

            // Update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
            break;
        }

        // return value;
        // Store current value into data buffer
        dataBuffer->buffer[cursor++] = value;
    }

    return dataBuffer;
}