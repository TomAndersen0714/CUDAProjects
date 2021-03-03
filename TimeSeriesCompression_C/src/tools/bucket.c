#include "compressors.h"
#include "decompressors.h"

static const uint32_t DIFF_MASK_2 = 0b0 << 2;
static const uint32_t DIFF_MASK_4 = 0b10 << 4;
static const uint32_t DIFF_MASK_6 = 0b11 << 6;

ByteBuffer * value_compress_bucket(ByteBuffer * valByteBuffer) {
    // Declare variables
    ByteBuffer *compressedValues;
    BitWriter* bitWriter;
    int64_t value, prevValue = 0;
    uint32_t leadingZeros, trailingZeros, significantBits,
        diffLeadingZeros, diffSignificantBits, leastSignificantBits;
    uint32_t prevLeadingZeros = BITS_OF_LONG_LONG;
    uint32_t prevTrailingZeros = BITS_OF_LONG_LONG;
    uint64_t xor, cursor = 0,
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

        // Calculate the XOR difference between prediction and current value to be compressed.
        value = valBuffer[cursor++];
        xor = prevValue ^ value;
        prevValue = value;

        if (xor == 0) {// case A:
            // Write '11' bit as entire control bit(i.e. prediction and current value is same).
            // According the the distribution of values(i.e. entropy code).
            bitWriterWriteBits(bitWriter, 0b11, 2);
        }
        else {
            leadingZeros = leadingZerosCount64(xor);
            trailingZeros = trailingZerosCount64(xor);


            // If the scope of meaningful bits falls within the scope of previous meaningful bits,
            // i.e. there are at least as many leading zeros and as many trailing zeros as with
            // the previous value.
            if (leadingZeros >= prevLeadingZeros && trailingZeros >= prevTrailingZeros) {
                // case B:

                //writeInPrevScope(xor);
                // Write '10' as control bit
                bitWriterWriteBits(bitWriter, 0b10, 2);

                // Write significant bits of difference value input the scope.
                significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
                bitWriterWriteBits(bitWriter, xor >> prevTrailingZeros, significantBits);
            }
            else {// case C:
                //writeInNewScope(diff, leadingZeros, trailingZeros);
                // Write '0' bit as second control bit.
                bitWriterWriteBits(bitWriter, 0b0, 1);

                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                diffLeadingZeros = encodeZigZag32(leadingZeros - prevLeadingZeros);
                diffSignificantBits = encodeZigZag32(
                    leadingZeros + trailingZeros -
                    prevLeadingZeros - prevTrailingZeros
                );
                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffLeadingZeros);

                switch (leastSignificantBits) {// [0,32]
                case 0:
                case 1:
                case 2:// diffLeadingZeros:[0,4)
                    // '0'+2
                    // '0' as entire control bit meaning the number of least significant bits of 
                    // encoded 'diffLeadingZeros' equals 2
                    bitWriterWriteZeroBit(bitWriter);
                    // write the least significant bits of encoded 'diffLeadingZeros'
                    bitWriterWriteBits(bitWriter, diffLeadingZeros, 2);
                    break;
                case 3:
                case 4:// diffLeadingZeros:[4,16)
                    // '10'+4
                    // '10' as entire control bit meaning the number of least significant bits of 
                    // encoded 'diffLeadingZeros' equals 4
                    bitWriterWriteBits(bitWriter, 0b10, 2);
                    // write the least significant bits of encoded 'diffLeadingZeros'
                    bitWriterWriteBits(bitWriter, diffLeadingZeros, 4);
                    break;
                default:// diffLeadingZeros:[16,32]
                    // '11'+6
                    // '11' as entire control bit meaning just write the number of leading zeros in 6 bits
                    bitWriterWriteBits(bitWriter, 0b11, 2);
                    bitWriterWriteBits(bitWriter, leadingZeros, 6);
                    break;
                }

                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffSignificantBits);
                switch (leastSignificantBits) {
                case 0:
                case 1:
                case 2:// diffSignificantBits:[0,4)
                    // '0'+2
                    // '0' as entire control bit meaning the number of least significant bits of 
                    // encoded 'diffSignificantBits' equals 2
                    bitWriterWriteZeroBit(bitWriter);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 2);
                    break;
                case 3:
                case 4:// diffSignificantBits:[4,16)
                    // '10'+4
                    // '10' as entire control bit meaning the number of least significant bits of 
                    // encoded 'diffSignificantBits' equals 4
                    bitWriterWriteBits(bitWriter, 0b10, 2);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 4);
                    break;
                default:// diffSignificantBits:[16,32]
                    // '11'+6
                    // '11' as entire control bit meaning just write the number of significant bits in 6 bits
                    bitWriterWriteBits(bitWriter, 0b11, 2);
                    // In this case xor value don't equal to zero, so 'significantBits' will not be '0'
                    // which we can leverage to reduce 'significantBits' by 1 to cover scope [1,64]
                    bitWriterWriteBits(bitWriter, significantBits - 1, 6);
                    break;
                }

                // Since the first bit of significant bits must be '1', we can utilize it to store less bits.
                bitWriterWriteBits(
                    bitWriter, xor >> trailingZeros, significantBits - 1
                ); // Write the meaningful bits of XOR
            }
            // Update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
        }
    }

    // Write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    free(bitWriter);

    // Return byte buffer.
    return compressedValues;
}

ByteBuffer* value_decompress_bucket(ByteBuffer* values, uint64_t count) {

    // Declare variables
    ByteBuffer* byteBuffer;
    BitReader* bitReader;
    uint32_t prevLeadingZeros = 0, prevTrailingZeros = 0,
        leadingZeros, trailingZeros, significantBits,
        controlBits, diffLeadingZeros, diffSignificantBits;
    int64_t value = 0, prevValue = 0, xor;
    uint64_t cursor = 0, *valBuffer;

    // Allocate memory space
    byteBuffer = malloc(sizeof(ByteBuffer));
    assert(byteBuffer != NULL);
    byteBuffer->length = count * sizeof(uint64_t);
    byteBuffer->capacity = byteBuffer->length;
    byteBuffer->buffer = malloc(byteBuffer->length);
    assert(byteBuffer->buffer != NULL);

    valBuffer = (uint64_t*)byteBuffer->buffer;
    bitReader = bitReaderConstructor(values);

    // Read each value and compress it into byte byffer.
    while (cursor < count) {
        // Read next value's control bits.
        controlBits = bitReaderNextControlBits(bitReader, 2);

        // Match the case corresponding to the control bits.
        switch (controlBits) {
        case 0b0: // '0' as entire control bit(i.e. next value is in a new scope).

            controlBits = bitReaderNextControlBits(bitReader, 2);
            switch (controlBits)
            {
            case 0b0:// '0' as entire control bit meaning the number of least significant bits of
                // encoded 'diffLeadingZeros' equals 2
                diffLeadingZeros = (uint32_t)bitReaderNextLong(bitReader, 2);
                diffLeadingZeros = decodeZigZag32(diffLeadingZeros);
                leadingZeros = diffLeadingZeros + prevLeadingZeros;
                break;
            case 0b10:// '10' as entire control bit meaning the number of least significant bits of
                // encoded 'diffLeadingZeros' equals 4
                diffLeadingZeros = (uint32_t)bitReaderNextLong(bitReader, 4);
                diffLeadingZeros = decodeZigZag32(diffLeadingZeros);
                leadingZeros = diffLeadingZeros + prevLeadingZeros;
                break;
            case 0b11:// '11' as entire control bit meaning just write the number of leading zeros
                // in 6 bits
                leadingZeros = (uint32_t)bitReaderNextLong(bitReader, 6);
                break;
            default:// Do nothing
                break;
            }

            controlBits = bitReaderNextControlBits(bitReader, 2);
            switch (controlBits)
            {
            case 0b0:// '0' as entire control bit meaning the number of least significant bits of
                // encoded 'diffSignificantBits' equals 2
                diffSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 2);
                diffSignificantBits = decodeZigZag32(diffSignificantBits);
                trailingZeros = diffSignificantBits +
                    prevLeadingZeros + prevTrailingZeros - leadingZeros;
                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                break;
            case 0b10:// '10' as entire control bit meaning the number of least significant bits of
                // encoded 'diffSignificantBits' equals 4
                diffSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 4);
                diffSignificantBits = decodeZigZag32(diffSignificantBits);
                trailingZeros = diffSignificantBits +
                    prevLeadingZeros + prevTrailingZeros - leadingZeros;
                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                break;
            case 0b11:
                // '11' as entire control bit meaning just write the number of significant bits
                // in 6 bits
                // Since we write 'significantBits-1' to cover scope [1,64], we
                // restore 'significantBits' here
                significantBits = (uint32_t)bitReaderNextLong(bitReader, 6) + 1;
                trailingZeros = BITS_OF_LONG_LONG - leadingZeros - significantBits;
                break;
            default:
                // Do nothing
                break;
            }

            // Read the next xor value according to the 'trailingZeros' and 'significantBits'
            // Since we reduce the 'significantBitLength' by 1 when we write it, we need
            // to restore it here.
            xor = (bitReaderNextLong(bitReader, significantBits - 1) | (1 << (significantBits - 1)))
                << trailingZeros;
            value = prevValue ^ xor;
            prevValue = value;

            // Update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
            break;

        case 0b10:
            // '10' bits (i.e. the block of next value meaningful bits falls within
            // the scope of prediction(previous value) meaningful bits)

            // Read the significant bits and restore the xor value.
            significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
            xor = bitReaderNextLong(bitReader, significantBits) << prevTrailingZeros;
            value = prevValue ^ xor;
            prevValue = value;

            // Update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZerosCount64(xor);
            prevTrailingZeros = trailingZerosCount64(xor);
            break;

        case 0b11:
            // '11' bits (i.e. prediction(previous) and current value is same)
            value = prevValue;
            break;

        default: // Do nothing
            break;
        }

        // return value;
        // Store current value into data buffer
        valBuffer[cursor++] = value;
    }

    return byteBuffer;
}