#include "compressors.h"
#include "decompressors.h"
#include "utils/encode_utils.h"
#include "utils/cuda_common_utils.h"
#include "utils/bit_writer.cuh"
#include "utils/bit_reader.cuh"

#define DEFAULT_SUBFRAME_SIZE 8

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
    uint16_t
        pos = 0, maxLeastSignificantBits = 0;
    int64_t 
        value, prevValue;
    uint64_t
        diff, subframe[DEFAULT_SUBFRAME_SIZE],
        *uncompressed_v = c_uncompressed_v;

    // since the header of current frame has been written(i.e. 
    // 'start' must >=1), we can get the previous value and 
    // 'diff' as follow
    prevValue = uncompressed_v[start - 1];

    // compress every value in the specific scope of uncompressed buffer
    for (uint32_t cursor = start; cursor < end; cursor++) {
        // get next value and calculate the xor value with previous one
        value = uncompressed_v[cursor];
        
        // if current sub-frame is full, then flush it
        if (pos == DEFAULT_SUBFRAME_SIZE) {
            // if all values in the frame equals zero(i.e. 'maxLeastSignificantBits' equals 0)
            // we just store the 0b0 in next 6 bits and clear frame
            if (maxLeastSignificantBits == 0) {
                bitWriterWriteBits(bitWriter, 0b0, 6);
                pos = 0;
            }
            else {
                // since 'maxLeastSignificantBits' could not equals to '0',
                // we leverage this point to cover range [1~64] by storing
                // 'maxLeastSignificantBits-1' using 6 bits
                bitWriterWriteBits(bitWriter, maxLeastSignificantBits - 1, 6);

                // write the significant bits of every value in current sub-frame into buffer
                for (int i = 0; i < pos; i++) {
                    bitWriterWriteBits(bitWriter, subframe[i], maxLeastSignificantBits);
                }

                // reset the pos and the maximum number of least significant bit in the sub-frame
                pos = 0;
                maxLeastSignificantBits = 0;
            }
        }

        // calculate the difference between current and previous value
        diff = encodeZigZag64(value - prevValue);
        // update previous value
        prevValue = value;

        // update the maximum number of least significant bit.
        /*maxLeastSignificantBits =
            max(maxLeastSignificantBits, 
                BITS_OF_LONG_LONG - leadingZerosCount64(diff));*/
        maxLeastSignificantBits =
            max(maxLeastSignificantBits, BITS_OF_LONG_LONG - __clzll(diff));

        // store encoded diff value in sub-frame.
        subframe[pos++] = diff;
    }

    // flush the left value in sub-frame into buffer
    if (pos != 0) {
        // if all value in this sub-frame equals zero, we just store the 
        // 0b0 in next 6 bits
        if (maxLeastSignificantBits == 0) {
            bitWriterWriteBits(bitWriter, 0b0, 6);
        }
        else {
            // Since 'maxLeastSignificantBits' could vary within [1,64], and the possibility 
            // when 'maxLeastSignificantBits' equals 1 is very small, we merge this situation
            // into the othor one that 'maxLeastSignificantBits' equals 2 to cover [0~64] using
            // 6 bits
            if (maxLeastSignificantBits == 1) maxLeastSignificantBits++;
            bitWriterWriteBits(bitWriter, maxLeastSignificantBits - 1, 6);

            // write the significant bits of every value in current sub-frame into buffer.
            for (int i = 0; i < pos; i++) {
                bitWriterWriteBits(bitWriter, subframe[i], maxLeastSignificantBits);
            }
        }

        // reset the pos and the maximum number of least significant bit in the sub-frame.
        pos = 0;
        maxLeastSignificantBits = 0;
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


// Compress values on GPU using Bitpack, return compressed data which is not compacted
CompressedData *value_compress_bitpack_gpu(
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
    value_compress_kernal <<<block, thdPB >>> (frame, thd, count);
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
        value, prevValue, diff;
    uint64_t
        *decompressed_v = c_decompressed_v;
    uint32_t
        pos = DEFAULT_FRAME_SIZE, maxLeastSignificantBits = 0;
    
    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can get the previous value as follow
    prevValue = decompressed_v[start - 1];
    diff = 0;

    // decompress each timestamp from byte buffer
    for (uint32_t cursor = start; cursor < end; cursor++) {
        // if current compressed subframe reach the end, read maximum number of least
        // significant bit in next subframe
        if (pos == DEFAULT_FRAME_SIZE) {
            maxLeastSignificantBits = (uint32_t)bitReaderNextLong(bitReader, 6);
            pos = 0;
            // since we compressed 'maxLeastSignificantBits-1' into buffer before,
            // we restore it's value here
            if (maxLeastSignificantBits > 0) maxLeastSignificantBits++;
        }

        // if 'maxLeastSignificantBits' equals zero, the all diff value in
        // current subframe is zero.(i.e. current value and previous is same)
        if (maxLeastSignificantBits == 0) {
            // Restore the value.
            value = diff + prevValue;
        }
        else {

            // decompress the difference in current subframe according to the 
            // value of 'maxLeastSignificantBits'
            diff = decodeZigZag64(bitReaderNextLong(bitReader, maxLeastSignificantBits));
            // restore the value.
            value = diff + prevValue;
        }
        // update predictor and position.
        prevValue = value;
        // Store current value into data buffer
        pos++;
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


// Decompress compressed and compacted values data on GPU using Bitpack
ByteBuffer *value_decompress_bitpack_gpu(
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