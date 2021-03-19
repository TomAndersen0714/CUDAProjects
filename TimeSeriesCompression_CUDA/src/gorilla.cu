#include "compressors.h"
#include "decompressors.h"
#include "utils/encode_utils.h"
#include "utils/cuda_common_utils.h"
#include "utils/bit_writer.cuh"
#include "utils/bit_reader.cuh"

//__constant__ uint64_t c_count; // the number of data points
//__constant__ uint16_t c_frame; // the length of frame
__constant__ uint64_t *c_uncompressed_t; // uncompressed timestamps on device
__constant__ uint64_t *c_uncompressed_v; // uncompressed values on device
__constant__ uint64_t *c_offs; // offset of each frame in compacted data
__constant__ byte *c_compressed_t; // compressed timestamps on device
__constant__ byte *c_compressed_v; // compressed values on device
__constant__ uint16_t *c_len_t; // length of compressed timestamps on device
__constant__ uint16_t *c_len_v; // length of compressed values on device
__constant__ uint64_t *c_decompressed_t; // decompressed timestamps on device
__constant__ uint64_t *c_decompressed_v; // decompressed values on device

// Compress timestamps in specific scope of uncompressed timestamps
__device__ static inline void timestamp_compress_device(
    uint32_t start, uint32_t end, BitWriter *bitWriter, uint32_t thdIdx
) {
    // avoid acessing out of bound
    //assert(start <= end);
    if (start >= end) return;

    // declaration
    int64_t timestamp, prevTimestamp;
    int32_t newDelta, deltaOfDelta, prevDelta;
    uint32_t leastBitLength;
    uint64_t *tsBuffer = c_uncompressed_t;

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can set the previous timestamp and 
    // 'delta' value as follow
    prevTimestamp = tsBuffer[start - 1];
    prevDelta = 0;

    // compress every timestamp in the specific scope of uncompressed buffer
    for (int cur = start; cur < end; cur++) {
        // Calculate the delta of delta of timestamp.
        timestamp = tsBuffer[cur];

        // PS: since original implementation in gorilla paper requires that delta-of-delta
        // of timestamps can be stored by a signed 32-bit value, it doesn't support
        // compression timestamps in millisecond as good as second.
        newDelta = (int32_t)(timestamp - prevTimestamp);
        deltaOfDelta = newDelta - prevDelta;

        // if current delta and previous delta is same
        if (deltaOfDelta == 0) {
            // Write '0' bit as control bit(i.e. previous and current delta value is same).
            bitWriterWriteZeroBit(bitWriter);
        }
        else {
            // Tips: since deltaOfDelta == 0 is unoccupied, we can utilize it to cover a larger range.
            if (deltaOfDelta > 0) deltaOfDelta--;
            // convert signed value to unsigned value for compression.
            deltaOfDelta = encodeZigZag32(deltaOfDelta);

            //leastBitLength = BITS_OF_INT - leadingZerosCount32(deltaOfDelta);
            leastBitLength = BITS_OF_INT - __clz(deltaOfDelta);
            // match the deltaOfDelta to the these case as follow.
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
                bitWriterWriteBits(bitWriter, 0b10, 2);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 7);
                break;
            case 8:
            case 9:
                // '110'+9
                bitWriterWriteBits(bitWriter, 0b110, 3);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 9);
                break;
            case 10:
            case 11:
            case 12:
                // '1110'+12
                bitWriterWriteBits(bitWriter, 0b1110, 4);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 12);
                break;
            default:
                // '1111'+32
                // Write '1111' control bits.
                bitWriterWriteBits(bitWriter, 0b1111, 4);
                // since it only takes 4 bytes(i.e. 32 bits) to save a unix timestamp in second, we write
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

    // write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // write the byte length of compressed timestamps(include the header of current frame)
    c_len_t[thdIdx] = BYTES_OF_LONG_LONG + bitWriter->byteBuffer->length;
}

// Timestamps compression kernal
__global__ static void timestamp_compress_kernal(
    uint16_t const frame, uint32_t const maxThd, uint32_t const count
) {
    // declare
    uint64_t 
        *uncompressed_t = c_uncompressed_t;
    byte 
        *compressed_t = c_compressed_t;
    uint32_t
        thdIdx, // thread index within grid
        start, // start offset of uncompressed data in current thread
        end; // end offset of uncompressed data in current thread

    thdIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if (thdIdx >= maxThd) return;

    /*start = c_offs[thdIdx];
    end = c_offs[thdIdx + 1];*/
    start = frame*thdIdx;
    end = min(start + frame, count);
    // avoid acessing out of bound
    if (start >= end) {
        c_len_t[thdIdx] = 0;
        return;
    }

    // store the first timestamp as header of current frame
    *((uint64_t*)compressed_t + start) = uncompressed_t[start];
    start += 1;

    // construct
    ByteBuffer byteBuffer;
    byteBuffer.buffer = (byte*)((uint64_t*)compressed_t + start); // start pos for compression
    byteBuffer.length = 0; // the size of compressed data of current frame
    BitWriter bitWriter;
    bitWriter.byteBuffer = &byteBuffer;
    bitWriter.cacheByte = 0;
    bitWriter.leftBits = BITS_OF_BYTE;

    // compress the timestamps within this thread
    timestamp_compress_device(start, end, &bitWriter, thdIdx);

}

// Compress timestamps on GPU
CompressedData* timestamp_compress_gorilla_gpu(
    ByteBuffer* uncompressedBuffer, uint32_t block, uint32_t warp
) {

    // divide the uncompressed data into frames according to the 
    // total number of threads
    uint32_t
        // the number of data points
        const count = (uint32_t)uncompressedBuffer->length / BYTES_OF_LONG_LONG;
    uint32_t
        thdPB, // the number of threads within per block
        thd; // the total number of needed threads
    uint16_t 
        frame; // the length of data that each thread will compress
        //padding = 0, // unused pos in the used last frame
        //left = 0, // unused threads in the last block
        //*offs; // start offsets of data that each thread will compress

    thdPB = WARPSIZE*warp;
    thd = thdPB*block;
    frame = (count + thd - 1) / thd;
    verifyParams(count, &frame, &block, &thdPB, &thd);

    // allocate device memory and tranport data to device
    uint64_t
        *d_uncompressed_t; // uncompressed timestamps on device
    byte
        *d_compressed_t, // compressed timestamps on device
        *compressed_t; // compressed timestamps on host
    //uint32_t
    //    *d_offs; // offsets of each frame on device
    uint16_t
        *d_len_t, // length of compressed timestamps on device
        *len_t; // length of compressed timestamps on host

    checkCudaError(cudaMalloc((void**)&d_uncompressed_t, uncompressedBuffer->length));
    // pre-allocate as much memory for compressed data as uncompressed data
    // assuming that compression will work well
    checkCudaError(cudaMalloc((void**)&d_compressed_t, uncompressedBuffer->length));
    checkCudaError(cudaMalloc((void**)&d_len_t, BYTES_OF_SHORT*thd));
    //checkCudaError(cudaMalloc((void**)&d_offs, BYTES_OF_INT*(thds + 1)));
    checkCudaError(cudaMemcpy(
        d_uncompressed_t, uncompressedBuffer->buffer,
        uncompressedBuffer->length, cudaMemcpyHostToDevice
    ));
    //checkCudaError(cudaMemcpy(
    //    d_offs, offs,
    //    BYTES_OF_INT*(thds + 1), cudaMemcpyHostToDevice
    //));
    checkCudaError(cudaMemcpyToSymbol(c_uncompressed_t, &d_uncompressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_compressed_t, &d_compressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_len_t, &d_len_t, sizeof(void *)));
    //checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    //checkCudaError(cudaMemcpyToSymbol(c_frame, &frame, BYTES_OF_SHORT));
    //checkCudaError(cudaMemcpyToSymbol(c_count, &count, BYTES_OF_LONG_LONG));

    // initiate kernal
    timestamp_compress_kernal <<<block, thdPB >>>(frame,thd,count);

    checkCudaError(cudaDeviceSynchronize());

    // allocate cpu memory for compressed data, and copy data from GPU to CPU
    compressed_t = (byte*)malloc(uncompressedBuffer->length);
    len_t = (uint16_t*)malloc(BYTES_OF_SHORT*thd);
    checkCudaError(cudaMemcpy(
        compressed_t, d_compressed_t, uncompressedBuffer->length,
        cudaMemcpyDeviceToHost)
    );
    checkCudaError(cudaMemcpy(
        len_t, d_len_t, BYTES_OF_SHORT*thd,
        cudaMemcpyDeviceToHost)
    );
    
    // free memory
    checkCudaError(cudaFree(d_uncompressed_t));
    checkCudaError(cudaFree(d_compressed_t));
    checkCudaError(cudaFree(d_len_t));

    // packing and return compressed data
    CompressedData *compressedData =
        (CompressedData*)malloc(sizeof(CompressedData));
    assert(compressedData != NULL);
    compressedData->buffer = compressed_t;
    compressedData->lens = len_t;
    compressedData->count = count;
    compressedData->frame = frame;

    return compressedData;
}


/**
 * Decompress timestamps from compressed and compacted data, then write them into
   specific scope of decompressed buffer
 * start: start offset of decompressed timestamp 
 * end: end offset of decompressed timestamp
 */
__device__ static inline void timestamp_decompress_device(
    uint32_t start, uint32_t end, BitReader *bitReader
) {
    // avoid acessing out of bound
    //assert(start <= end);
    if (start >= end) return;

    // decalre
    uint32_t 
        controlBits;
    int64_t 
        timestamp, prevTimestamp = 0,
        newDelta, deltaOfDelta = 0, prevDelta;
    uint64_t
        cursor = start,
        *tsBuffer = c_decompressed_t;

    /*// get previous timestamp
    if (start == 0) {// if current timestamp is the first one
        prevTimestamp = 0;
        prevDelta = 0;
    }
    else { // or not 
        prevTimestamp = tsBuffer[start - 1];
        prevDelta = prevTimestamp - tsBuffer[start - 2];
    }*/

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can set the previous timestamp and 
    // 'delta' value as follow
    prevTimestamp = tsBuffer[start - 1];
    prevDelta = 0;

    // decompressed timestamps from the compressed and compacted data, and write
    // them into sepecific scope of decompressed buffer
    while (cursor < end) {
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
}

// Timestamps decompression kernal
__global__ static void timestamp_decompress_kernal(
    uint16_t frame, uint32_t maxThd, uint32_t count
) {
    // declare
    uint64_t
        thdIdx, // thread index within grid
        start, // start offset of decompressed data in current thread/frame
        end, // end offset of decompressed data in current thread/frame
        *offs = c_offs,
        *decompressed_t = c_decompressed_t;
    byte 
        *compressed_t = c_compressed_t;

    thdIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if (thdIdx >= maxThd) return;

    // get the scope of current frame
    start = frame*thdIdx;
    end = min(count, frame*(thdIdx + 1));
    if (start >= end) return;

    // get the header of current frame
    /*decompressed_t[start] = *((uint64_t*)(compressed_t + offs[thdIdx]));
    start += 1;*/

    // since the 'Size and Alignment Requirement', we have to read the 
    // the compressed data byte-by-byte
    // PS: lettle-endian
    decompressed_t[start] = readDeviceMem64(compressed_t + offs[thdIdx], 8);
    start += 1;

    // construct 'BitReader'
    ByteBuffer byteBuffer;
    byteBuffer.buffer = compressed_t + offs[thdIdx] + BYTES_OF_LONG_LONG;
    byteBuffer.length = offs[thdIdx + 1] - offs[thdIdx] - BYTES_OF_LONG_LONG;
    BitReader bitReader;
    bitReader.byteBuffer = &byteBuffer;
    bitReader.leftBits = BITS_OF_BYTE;
    bitReader.cacheByte = bitReader.byteBuffer->buffer[0];
    bitReader.cursor = 1;

    // decompress timestamps within current thread
    timestamp_decompress_device(start, end, &bitReader);
}


// Decompress timestamps on GPU using Gorilla
ByteBuffer *timestamp_decompress_gorilla_gpu(
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
        *tsBuffer = compactedData->buffer;

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
        *decompressed_t, // decompressed timestamps on host
        *d_decompressed_t, // decompressed timestamps on device
        *d_offs; // offsets of each compressed frame on device
    byte
        *d_compressed_t; // compressed timestamps on device

    decompressed_t = (uint64_t*)malloc(BYTES_OF_LONG_LONG*count);
    checkCudaError(cudaMalloc((void**)&d_decompressed_t, BYTES_OF_LONG_LONG*count));
    checkCudaError(cudaMalloc((void**)&d_compressed_t, len));
    checkCudaError(cudaMalloc((void**)&d_offs, BYTES_OF_LONG_LONG*(thd + 1)));

    checkCudaError(cudaMemcpy(d_compressed_t, tsBuffer, len, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_offs, offs, BYTES_OF_LONG_LONG*(thd + 1), cudaMemcpyHostToDevice));

    checkCudaError(cudaMemcpyToSymbol(c_compressed_t, &d_compressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    //checkCudaError(cudaMemcpyToSymbol(c_count, &count, BYTES_OF_LONG_LONG));
    checkCudaError(cudaMemcpyToSymbol(c_decompressed_t, &d_decompressed_t, sizeof(void *)));

    // initiate decompressed kernal
    timestamp_decompress_kernal<<<block,thdPB>>>(frame,thd,count);
    checkCudaError(cudaDeviceSynchronize());

    // copy decompressed data from device to host
    checkCudaError(
        cudaMemcpy(
            decompressed_t, d_decompressed_t,
            BYTES_OF_LONG_LONG*count, cudaMemcpyDeviceToHost
        ));

    // free device memory
    free(offs);
    checkCudaError(cudaFree(d_decompressed_t));
    checkCudaError(cudaFree(d_compressed_t));
    checkCudaError(cudaFree(d_offs));

    // pack and return decompressed data
    ByteBuffer* byteBuffer = (ByteBuffer*)malloc(sizeof(ByteBuffer));
    byteBuffer->buffer = (byte*)decompressed_t;
    byteBuffer->length = BYTES_OF_LONG_LONG*count;

    return byteBuffer;
}



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
        prevLeadingZeros, prevTrailingZeros;
    uint64_t
        diff,
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
        prevLeadingZeros = __clzll(prevValue);
        prevTrailingZeros = __ffsll(prevValue) - 1;
    }

    // compress every value in the specific scope of uncompressed buffer
    for (int cur = start; cur < end; cur++) {
        // calculate the delta of delta of timestamp.
        value = uncompressed_v[cur];
        diff = prevValue^value;

        // if previous value and current value is same(Case A)
        if(diff == 0)
            // write '0' bit as entire control bit
            // (i.e. prediction and current value is same).
            bitWriterWriteZeroBit(bitWriter);
        else {
            leadingZeros = __clzll(diff);
            trailingZeros = __ffsll(diff) - 1;

            // write '1' bit as first control bit.
            bitWriterWriteOneBit(bitWriter);

            // if the scope of meaningful bits falls within the scope of previous meaningful bits,
            // i.e. there are at least as many leading zeros and as many trailing zeros as with
            // the previous value.(Case B)
            if (leadingZeros >= prevLeadingZeros && trailingZeros >= prevTrailingZeros) {
                // write current value into previous scope
                //writeInPrevScope(diff);

                // write '0' bit as second control bit.
                bitWriterWriteZeroBit(bitWriter);

                // Write significant bits of difference value input the scope.
                significantBits = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;

                bitWriterWriteBits(bitWriter, diff >> prevTrailingZeros, significantBits);
            }
            else {
                // write current value into new scope(Case C)
                //writeInNewScope(diff, leadingZeros, trailingZeros);

                // write '1' bit as second control bit.
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
                // write the number of leading zeros into the next 6 bits
                bitWriterWriteBits(bitWriter, leadingZeros, 6);

                /* 
                 since 'significantBits == 0' is unoccupied, we can just store 'significantBits - 1' to
                 cover a larger range and avoid the situation when 'significantBits == 64
                */
                // write the length of meaningful bits input the next 6 bits
                bitWriterWriteBits(bitWriter, significantBits - 1, 6);// Write the length of meaningful bits input the next 6 bits

                // Write the meaningful bits of XOR
                bitWriterWriteBits(bitWriter, diff >> trailingZeros, significantBits);
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

// Compress values on GPU using Gorilla, return compressed data which is not compacted
CompressedData *value_compress_gorilla_gpu(
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
        cursor = start,
        *decompressed_v = c_decompressed_v;
    uint32_t
        prevLeadingZeros, prevTrailingZeros, 
        leadingZeros, trailingZeros,
        controlBits, significantBitLength;

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can set the previous value and 
    // 'diff' value as follow
    prevValue = decompressed_v[start - 1];
    if (prevValue == 0) {
        prevLeadingZeros = 0;
        prevTrailingZeros = 0;
    }
    else {
        prevLeadingZeros = __clzll(prevValue);
        prevTrailingZeros = __ffsll(prevValue) - 1;
    }

    // decompressed values from the compressed and compacted data, and write
    // them into sepecific scope of decompressed buffer
    while (cursor < end) {
        // read next control bits.
        controlBits = bitReaderNextControlBits(bitReader, 2);

        // match the case corresponding to the control bits.
        switch (controlBits)
        {
        case 0b0:
            // '0' bit (i.e. prediction(previous) and current value is same)
            value = prevValue;
            break;

        case 0b10:
            // '10' bits (i.e. the block of current value meaningful bits falls within
            // the scope of prediction(previous) meaningful bits)

            // read the significant bits and restore the xor value.
            significantBitLength = BITS_OF_LONG_LONG - prevLeadingZeros - prevTrailingZeros;
            diff = bitReaderNextLong(bitReader, significantBitLength) << prevTrailingZeros;
            value = prevValue ^ diff;
            prevValue = value;

            // update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZerosCount64(diff);
            prevTrailingZeros = trailingZerosCount64(diff);
            break;

        case 0b11:
            // '11' bits (i.e. the block of current value meaningful bits doesn't falls within
            // the scope of previous meaningful bits)

            // update the number of leading and trailing zeros.
            leadingZeros = (uint32_t)bitReaderNextLong(bitReader, 6);
            significantBitLength = (uint32_t)bitReaderNextLong(bitReader, 6);
            
            // Since we have decreased the length of significant bits by 1 for larger compression range
            // when we compress it, we restore it's value here.
            significantBitLength++;

            // read the significant bits and restore the xor value.
            trailingZeros = BITS_OF_LONG_LONG - leadingZeros - significantBitLength;
            diff = bitReaderNextLong(bitReader, significantBitLength) << trailingZeros;
            value = prevValue ^ diff;
            prevValue = value;

            // update the number of leading and trailing zeros of xor residual.
            prevLeadingZeros = leadingZeros;
            prevTrailingZeros = trailingZeros;

            break;
        default:
            break;
        }
        // store current value into decompressed buffer
        decompressed_v[cursor++] = value;
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

// Decompress compressed and compacted values data on GPU using Gorilla
ByteBuffer *value_decompress_gorilla_gpu(
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

    checkCudaError(cudaMemcpyToSymbol(c_compressed_v, &d_compressed_v, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_decompressed_v, &d_decompressed_v, sizeof(void *)));

    // initiate decompressed kernal
    value_decompress_kernal <<<block, thdPB >>>(frame, thd, count);
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

// trigger the lazy creation of the CUDA context
void warmUp()
{
    cudaFree(0);
}