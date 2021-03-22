#include "compressors.h"
#include "decompressors.h"
#include "utils/encode_utils.h"
#include "utils/cuda_common_utils.h"
#include "utils/bit_writer.cuh"
#include "utils/bit_reader.cuh"

__constant__ uint64_t *c_uncompressed_v; // uncompressed values on device
__constant__ uint64_t *c_offs; // offset of each frame in compacted data
__constant__ byte *c_compressed_v; // compressed values on device
__constant__ uint16_t *c_len_v; // length of compressed values on device
__constant__ uint64_t *c_decompressed_v; // decompressed values on device


// Compress timestamps in specific scope of uncompressed timestamps
__device__ static inline void value_compress_device(
    uint32_t start, uint32_t end, BitWriter *bitWriter, uint32_t thdIdx
) {
    // avoid acessing out of bound
    //assert(start <= end);
    if (start >= end) return;

    // declare
    int64_t
        value, prevValue;
    uint32_t
        leadingZeros, trailingZeros, significantBits,
        prevLeadingZeros, prevTrailingZeros,
        diffLeadingZeros, diffSignificantBits, leastSignificantBits;
    uint64_t
        xor,
        *uncompressed_v = c_uncompressed_v;

    // since the header of current frame has been written(i.e. 
    // 'start' must >=1), we can get the previous value and 
    // 'diff' as follow
    prevValue = uncompressed_v[start - 1];
    if (prevValue == 0) {
        prevLeadingZeros = 0;
        prevTrailingZeros = 0;
    }
    else {
        /*prevLeadingZeros = leadingZerosCount64(prevValue);
        prevTrailingZeros = trailingZerosCount64(prevValue);*/
        prevLeadingZeros = __clzll(prevValue);
        prevTrailingZeros = __ffsll(prevValue) - 1;
    }

    // compress every value in the specific scope of uncompressed buffer
    for (int cur = start; cur < end; cur++) {
        // get next value and calculate the xor value with previous one
        value = uncompressed_v[cur];
        xor = prevValue^value;

        if (xor == 0) {// case A:
                       // write '11' bit as entire control bit(i.e. prediction and current value is same).
            bitWriterWriteBits(bitWriter, 0b11, 2);
        }
        else {
            /*leadingZeros = leadingZerosCount64(xor);
            trailingZeros = trailingZerosCount64(xor);*/
            leadingZeros = __clzll(xor);
            trailingZeros = __ffsll(xor) - 1;

            // if the scope of meaningful bits falls within the scope of previous meaningful bits,
            // i.e. there are at least as many leading zeros and as many trailing zeros as with
            // the previous value.
            if (leadingZeros >= prevLeadingZeros && trailingZeros >= prevTrailingZeros) {
                // case B
                // Write '10' as control bit
                bitWriterWriteBits(bitWriter, 0b10, 2);

                // Write significant bits of difference value input the scope.
                significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
                bitWriterWriteBits(bitWriter, xor >> prevTrailingZeros, significantBits);
            }
            else {// case C:
                  // Write '0' bit as second control bit.
                bitWriterWriteBits(bitWriter, 0b0, 1);

                significantBits = BITS_OF_LONG_LONG - leadingZeros - trailingZeros;
                diffLeadingZeros = encodeZigZag32(leadingZeros - prevLeadingZeros);
                diffSignificantBits = encodeZigZag32(
                    leadingZeros + trailingZeros - prevLeadingZeros - prevTrailingZeros
                );

                //leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffLeadingZeros);
                leastSignificantBits = BITS_OF_INT - __clz(diffLeadingZeros);
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

                //leastSignificantBits = BITS_OF_INT - leadingZerosCount32(diffSignificantBits);
                leastSignificantBits = BITS_OF_INT - __clz(diffSignificantBits);
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


                // since the first bit of significant bits must be '1', we can utilize it to store less bits.
                // write the meaningful bits of XOR
                bitWriterWriteBits(bitWriter, xor >> trailingZeros, significantBits - 1);
            }
            // update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
        }

        // update previous value
        prevValue = value;
    }

    // write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // write the byte length of compressed values(include the header of current frame)
    c_len_v[thdIdx] = BYTES_OF_LONG_LONG + bitWriter->byteBuffer->length;
}


// Timestamps compression kernal
__global__ static void value_compress_kernal(
    uint16_t const frame, uint32_t const maxThd, uint32_t const count
) {
    // declare
    uint64_t
        *uncompressed_v = c_uncompressed_v;
    byte
        *compressed_v = c_compressed_v;
    uint32_t
        thdIdx, // thread index within grid
        start, // start offset of uncompressed data in current thread
        end; // end offset of uncompressed data in current thread

             // thread index of grid
    thdIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if (thdIdx >= maxThd) return;

    // calculate the scope of current frame
    start = frame*thdIdx;
    end = min(start + frame, count);
    // avoid acessing out of bound
    if (start >= end) {
        c_len_v[thdIdx] = 0;
        return;
    }

    // store the first value in little-endian mode as header of current frame
    *((uint64_t*)compressed_v + start) = uncompressed_v[start];
    start++;

    // construct
    ByteBuffer byteBuffer;
    byteBuffer.buffer = (byte*)((uint64_t*)compressed_v + start); // start pos for compression
    byteBuffer.length = 0; // the size of compressed data of current frame
    BitWriter bitWriter;
    bitWriter.byteBuffer = &byteBuffer;
    bitWriter.cacheByte = 0;
    bitWriter.leftBits = BITS_OF_BYTE;

    // compress the timestamps within current frame
    value_compress_device(start, end, &bitWriter, thdIdx);
}


// Compress values on GPU using Bucket, return compressed data which is not compacted
CompressedData *value_compress_bucket_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
) {
    // declare
    uint32_t
        // the number of data points
        const count = (uint32_t)uncompressedBuffer->length / BYTES_OF_LONG_LONG;
    uint32_t
        thdPB, // the number of threads within per block
        thd; // the total number of needed threads
    uint16_t
        frame; // the length of data that each thread will compress

               // divide the uncompressed data into frames according to the 
               // total number of threads
    thdPB = WARPSIZE*warp;
    thd = thdPB*block;
    frame = (count + thd - 1) / thd;
    verifyParams(count, &frame, &block, &thdPB, &thd);

    // allocate device memory and tranport data to device
    uint64_t
        *d_uncompressed_v; // uncompressed timestamps on device
    byte
        *d_compressed_v, // compressed timestamps on device
        *compressed_v; // compressed timestamps on host
    uint16_t
        *d_len_v, // length of compressed timestamps on device
        *len_v; // length of compressed timestamps on host

    checkCudaError(cudaMalloc((void**)&d_uncompressed_v, uncompressedBuffer->length));
    // pre-allocate as much memory for compressed data as uncompressed data
    // assuming that compression will work well
    checkCudaError(cudaMalloc((void**)&d_compressed_v, uncompressedBuffer->length));
    checkCudaError(cudaMalloc((void**)&d_len_v, BYTES_OF_SHORT*thd));
    checkCudaError(cudaMemcpy(
        d_uncompressed_v, uncompressedBuffer->buffer,
        uncompressedBuffer->length, cudaMemcpyHostToDevice
    ));

    // use global __constant__ variables to pass the params to avoid passing common params between functions
    checkCudaError(cudaMemcpyToSymbol(c_uncompressed_v, &d_uncompressed_v, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_compressed_v, &d_compressed_v, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_len_v, &d_len_v, sizeof(void *)));

    // initiate kernal
    value_compress_kernal << <block, thdPB >> > (frame, thd, count);
    checkCudaError(cudaDeviceSynchronize());

    // allocate cpu memory for compressed data, and copy data from GPU to CPU
    compressed_v = (byte*)malloc(uncompressedBuffer->length);
    len_v = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    assert(compressed_v != NULL && len_v != NULL);
    checkCudaError(cudaMemcpy(
        compressed_v, d_compressed_v, uncompressedBuffer->length,
        cudaMemcpyDeviceToHost)
    );
    checkCudaError(cudaMemcpy(
        len_v, d_len_v, BYTES_OF_SHORT*thd,
        cudaMemcpyDeviceToHost)
    );

    // free memory
    checkCudaError(cudaFree(d_uncompressed_v));
    checkCudaError(cudaFree(d_compressed_v));
    checkCudaError(cudaFree(d_len_v));

    // packing and return compressed data
    CompressedData *compressedData =
        (CompressedData*)malloc(sizeof(CompressedData));
    assert(compressedData != NULL);
    compressedData->buffer = compressed_v;
    compressedData->lens = len_v;
    compressedData->count = count;
    compressedData->frame = frame;

    return compressedData;
}

// Decompress values from compressed and compacted data, then write them into
// specific scope of decompressed buffer
__device__ static inline void value_decompress_device(
    uint32_t start, uint32_t end, BitReader *bitReader
) {
    // avoid acessing out of bound
    if (start >= end) return;

    // declare
    int64_t
        value, prevValue, xor;
    uint64_t
        *decompressed_v = c_decompressed_v;
    uint32_t
        leadingZeros, trailingZeros, significantBits,
        prevLeadingZeros, prevTrailingZeros,
        diffLeadingZeros, diffSignificantBits, controlBits;

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can set the previous value and 
    // 'diff' value as follow
    prevValue = decompressed_v[start - 1];
    if (prevValue == 0) {
        prevLeadingZeros = 0;
        prevTrailingZeros = 0;
    }
    else {
        /*prevLeadingZeros = leadingZerosCount64(prevValue);
        prevTrailingZeros = trailingZerosCount64(prevValue);*/
        prevLeadingZeros = __clzll(prevValue);
        prevTrailingZeros = __ffsll(prevValue) - 1;
    }

    // read and decompress each value
    for (uint32_t cursor = start; cursor < end; cursor++) {
        // read next value's control bits.
        controlBits = bitReaderNextControlBits(bitReader, 2);

        // match the case corresponding to the control bits.
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
            default:// do nothing
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

            // update the number of leading and trailing zeros.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;
            break;

        case 0b10:
            // '10' bits (i.e. the block of next value meaningful bits falls within
            // the scope of prediction(previous value) meaningful bits)

            // read the significant bits and restore the xor value.
            significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
            xor = bitReaderNextLong(bitReader, significantBits) << prevTrailingZeros;
            value = prevValue ^ xor;
            prevValue = value;

            // update the number of leading and trailing zeros of xor residual.
            /*prevLeadingZeros = leadingZerosCount64(xor);
            prevTrailingZeros = trailingZerosCount64(xor);*/
            prevLeadingZeros = __clzll(xor);
            prevTrailingZeros = __ffsll(xor) - 1;
            break;

        case 0b11:
            // '11' bits (i.e. prediction(previous) and current value is same)
            value = prevValue;
            break;

        default: // do nothing
            break;
        }

        // return value
        decompressed_v[cursor] = value;
    }
}

// Value decompression kernal
__global__ static void value_decompress_kernal(
    uint16_t frame, uint32_t maxThd, uint32_t count
) {
    // declare
    uint64_t
        thdIdx, // thread index within grid
        start, // start offset of decompressed data in current thread/frame
        end, // end offset of decompressed data in current thread/frame
        *offs = c_offs,
        *decompressed_v = c_decompressed_v;
    byte
        *compressed_v = c_compressed_v;

    thdIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if (thdIdx >= maxThd) return;

    // get the scope of current frame
    start = frame*thdIdx;
    end = min(count, frame*(thdIdx + 1));
    if (start >= end) return;

    // get the header of current frame
    // since the 'Size and Alignment Requirement', we have to read the 
    // the compressed data byte-by-byte
    decompressed_v[start] = readDeviceMem64(compressed_v + offs[thdIdx], 8);
    start += 1;

    // construct 'BitReader'
    ByteBuffer byteBuffer;
    byteBuffer.buffer = compressed_v + offs[thdIdx] + BYTES_OF_LONG_LONG;
    byteBuffer.length = offs[thdIdx + 1] - offs[thdIdx] - BYTES_OF_LONG_LONG;
    BitReader bitReader;
    bitReader.byteBuffer = &byteBuffer;
    bitReader.leftBits = BITS_OF_BYTE;
    bitReader.cacheByte = bitReader.byteBuffer->buffer[0];
    bitReader.cursor = 1;

    // decompress timestamps within current thread
    value_decompress_device(start, end, &bitReader);
}


// Decompress compressed and compacted values data on GPU using Bucket
ByteBuffer *value_decompress_bucket_gpu(
    CompressedData *compactedData, uint32_t block, uint32_t warp
) {
    // declare
    uint64_t
        len = 0, // the total size of compressed data
        *offs; // offsets of each compressed frame
    uint32_t
        thdPB, // the number of threads within each block
        thd; // the sum number of needed threads
    uint32_t const
        count = compactedData->count;
    uint16_t const
        frame = compactedData->frame,
        *lens = compactedData->lens;
    byte const
        *compressed_v = compactedData->buffer;

    thd = (count + frame - 1) / frame;
    thdPB = WARPSIZE*warp;

    // modify the input params
    if (thd <= MAX_THREADS_PER_BLOCK) {
        // use just one block to decompress
        block = 1; thdPB = thd;
    }
    else block = (thd + thdPB - 1) / thdPB;

    // get offsets of each compressed frame for decompression
    offs = (uint64_t*)malloc(BYTES_OF_LONG_LONG*(thd + 1));
    offs[0] = 0;
    for (uint32_t i = 1; i <= thd; i++) {
        len += lens[i - 1];
        offs[i] = lens[i - 1] + offs[i - 1];
    }

    // allocate memory and transport data to device
    uint64_t
        *decompressed_v, // decompressed values on host
        *d_decompressed_v, // decompressed values on device
        *d_offs; // offsets of each compressed frame on device
    byte
        *d_compressed_v; // compressed values on device

    decompressed_v = (uint64_t*)malloc(BYTES_OF_LONG_LONG*count);
    checkCudaError(cudaMalloc((void**)&d_decompressed_v, BYTES_OF_LONG_LONG*count));
    checkCudaError(cudaMalloc((void**)&d_compressed_v, len));
    checkCudaError(cudaMalloc((void**)&d_offs, BYTES_OF_LONG_LONG*(thd + 1)));

    checkCudaError(cudaMemcpy(d_compressed_v, compressed_v, len, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_offs, offs, BYTES_OF_LONG_LONG*(thd + 1), cudaMemcpyHostToDevice));

    // use global __constant__ variables to pass the params to avoid passing common params between functions
    checkCudaError(cudaMemcpyToSymbol(c_compressed_v, &d_compressed_v, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_decompressed_v, &d_decompressed_v, sizeof(void *)));

    // initiate decompressed kernal
    value_decompress_kernal <<<block, thdPB >>> (frame, thd, count);
    checkCudaError(cudaDeviceSynchronize());

    // copy decompressed data from device to host
    checkCudaError(
        cudaMemcpy(
            decompressed_v, d_decompressed_v,
            BYTES_OF_LONG_LONG*count, cudaMemcpyDeviceToHost
        )
    );

    // free device memory
    free(offs);
    checkCudaError(cudaFree(d_decompressed_v));
    checkCudaError(cudaFree(d_compressed_v));
    checkCudaError(cudaFree(d_offs));

    // pack and return decompressed data
    ByteBuffer* byteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    byteBuffer->buffer = (byte*)decompressed_v;
    byteBuffer->length = BYTES_OF_LONG_LONG*count;

    return byteBuffer;
}