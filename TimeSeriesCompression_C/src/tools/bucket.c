#include "compressors.h"

static const int32_t MASK_OFFSET_4 = 0b10 << 4;
static const int32_t MASK_OFFSET_6 = 0b11 << 6;

ByteBuffer * value_compress_bucket(UncompressedData * values) {
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
                // Write '1' bit as second control bit.
                bitWriterWriteBits(bitWriter, 0b1, 1);

                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                diffLeadingZeros = encodeZigZag32(leadingZeros - prevLeadingZeros);
                diffSignificantBits = encodeZigZag32(leadingZeros + trailingZeros - prevLeadingZeros - prevTrailingZeros);
                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffLeadingZeros);

                switch (leastSignificantBits) {
                case 0:
                case 1:
                case 2:
                    bitWriterWriteZeroBit(bitWriter);
                    bitWriterWriteBits(bitWriter, diffLeadingZeros, 2);
                    break;
                case 3:
                case 4:
                    bitWriterWriteBits(bitWriter, 0b10, 2);
                    bitWriterWriteBits(bitWriter, diffLeadingZeros, 4);

                    break;
                default:
                    bitWriterWriteBits(bitWriter, 0b11, 2);
                    bitWriterWriteBits(bitWriter, leadingZeros, 6);
                    break;
                }
                leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffSignificantBits);
                switch (leastSignificantBits) {
                case 0:
                case 1:
                case 2:
                    bitWriterWriteZeroBit(bitWriter);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 2);
                    break;
                case 3:
                case 4:
                    bitWriterWriteBits(bitWriter, 0b10, 2);
                    bitWriterWriteBits(bitWriter, diffSignificantBits, 4);
                    break;
                default:
                    bitWriterWriteBits(bitWriter, 0b11, 2);
                    bitWriterWriteBits(bitWriter, significantBits, 6);
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