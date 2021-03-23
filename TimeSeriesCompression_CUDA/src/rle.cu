#include "compressors.h"
#include "decompressors.h"
#include "utils/encode_utils.h"
#include "utils/cuda_common_utils.h"
#include "utils/bit_writer.cuh"
#include "utils/bit_reader.cuh"

__constant__ uint64_t *c_uncompressed_t; // uncompressed timestamps on device
__constant__ uint64_t *c_offs; // offset of each frame in compacted data
__constant__ byte *c_compressed_t; // compressed timestamps on device
__constant__ uint16_t *c_len_t; // length of compressed timestamps on device
__constant__ uint64_t *c_decompressed_t; // decompressed timestamps on device

// Write the cached zeros into buffer
__device__ inline void flushZeros(BitWriter *bitWriter, uint32_t *storedZeros) {
    // get the number of cached zeros
    uint32_t count = *storedZeros;
    
    // ouput the cached zeros
    while (count > 0) {
        // tips: since storedZeros == 0 is unoccupied, we can utilize it to cover a larger range
        count--;

        // write '0' control bit
        bitWriterWriteZeroBit(bitWriter);

        // 0<=count<8 (i.e. '0'+'0'+3)
        if (count < 8) {
            // tips: if there is too much case, switch-case code block may be better
            // write '0' control bit
            bitWriterWriteZeroBit(bitWriter);
            // write the number of cached zeros using 3 bits
            bitWriterWriteBits(bitWriter, count, 3);
            // clear zeros
            count = 0;
        }
        // 8<=count<32 (i.e. '0'+'1'+5)
        else if (count < 32) {
            // write '1' control bit
            bitWriterWriteOneBit(bitWriter);
            // write the number of cached zeros using 5 bits
            bitWriterWriteBits(bitWriter, count, 5);
            // clear zeros
            count = 0;
        }
        // count>=32 (i.e. '0'+'1'+'11111')
        else {
            // write '1' control bit(i.e. '01'+5)
            bitWriterWriteOneBit(bitWriter);
            // write 32 cached zeros
            bitWriterWriteBits(bitWriter, 0b11111, 5);
            // reduce the the number of cached zeros by 31
            // ps: 'count' has been reduce by 1 before
            count -= 31;
        }
    }
    
    // clear the cached zeros
    *storedZeros = 0;
}

// Compress timestamps in specific scope of uncompressed timestamps
__device__ static inline void timestamp_compress_device(
    uint32_t start, uint32_t end, BitWriter *bitWriter, uint32_t thdIdx
) {
    // avoid acessing out of bound
    //assert(start <= end);
    if (start >= end) return;

    // declare
    int64_t timestamp, prevTimestamp;
    int32_t newDelta, deltaOfDelta, prevDelta;
    uint32_t leastBitLength, storedZeros = 0;
    uint64_t *tsBuffer = c_uncompressed_t;

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can set the previous timestamp and 
    // 'delta' value as follow
    prevTimestamp = tsBuffer[start - 1];
    prevDelta = 0;

    // compress every timestamp in the specific scope of uncompressed buffer
    for (uint32_t cur = start; cur < end; cur++) {

        // calculate the delta of delta of timestamp.
        timestamp = tsBuffer[cur];
        newDelta = (int32_t)(timestamp - prevTimestamp);
        deltaOfDelta = newDelta - prevDelta;

        // If current delta and previous one is same
        if (deltaOfDelta == 0) {
            // counting the continuous and same delta of timestamps
            storedZeros++;
        }
        else {
            // write the privious stored zeros to the buffer
            flushZeros(bitWriter, &storedZeros);

            // tips: since deltaOfDelta == 0 is unoccupied, we can utilize it 
            // to cover a larger range.
            if (deltaOfDelta > 0) deltaOfDelta--;
            // convert signed value to unsigned value for compression
            deltaOfDelta = encodeZigZag32(deltaOfDelta);

            // match the deltaOfDelta to the three case as follow
            leastBitLength = BITS_OF_INT - __clz(deltaOfDelta);
            switch (leastBitLength) {
            case 0:
            case 1:
            case 2:
            case 3:
                // '10'+3
                bitWriterWriteBits(bitWriter, 0b10, 2);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 3);
                break;
            case 4:
            case 5:
                // '110'+5
                bitWriterWriteBits(bitWriter, 0b110, 3);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 5);
                break;
            case 6:
            case 7:
            case 8:
            case 9:
                // '1110'+9
                bitWriterWriteBits(bitWriter, 0b1110, 4);
                bitWriterWriteBits(bitWriter, deltaOfDelta, 9);
                break;
            case 10:
            case 11:
            case 12:
            default:
                // '1111'+32
                bitWriterWriteBits(bitWriter, 0b1111, 4); // Write '1111' control bits.
                // since it only takes 4 bytes(i.e. 32 bits) to save a unix timestamp input second, we write
                // delta-of-delta using 32 bits assuming that value won't above this range
                bitWriterWriteBits(bitWriter, deltaOfDelta, 32);
                break;
            }
            // update previous delta
            prevDelta = newDelta;
        }
        // update previous timestamp
        prevTimestamp = timestamp;
    }

    // write left and cached zeros into the buffer
    flushZeros(bitWriter, &storedZeros);

    // write the left bits in cached byte into the buffer
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


// Compress timestamps on GPU using RLE, return compressed but not compacted data
CompressedData *timestamp_compress_rle_gpu(
    ByteBuffer *uncompressedBuffer, uint32_t block, uint32_t warp
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
    uint16_t
        *d_len_t, // length of compressed timestamps on device
        *len_t; // length of compressed timestamps on host

    checkCudaError(cudaMalloc((void**)&d_uncompressed_t, uncompressedBuffer->length));
    // pre-allocate as much memory for compressed data as uncompressed data
    // assuming that compression will work well
    checkCudaError(cudaMalloc((void**)&d_compressed_t, uncompressedBuffer->length));
    checkCudaError(cudaMalloc((void**)&d_len_t, BYTES_OF_SHORT*thd));
    checkCudaError(cudaMemcpy(
        d_uncompressed_t, uncompressedBuffer->buffer,
        uncompressedBuffer->length, cudaMemcpyHostToDevice
    ));

    // use global __constant__ variables to pass the params to avoid passing common params between functions
    checkCudaError(cudaMemcpyToSymbol(c_uncompressed_t, &d_uncompressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_compressed_t, &d_compressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_len_t, &d_len_t, sizeof(void *)));

    // initiate kernal
    timestamp_compress_kernal <<<block, thdPB >>>(frame, thd, count);
    // wait for kernal
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

    // packing and return compressed data which is incompact yet
    CompressedData *compressedData =
        (CompressedData*)malloc(sizeof(CompressedData));
    assert(compressedData != NULL);
    compressedData->buffer = compressed_t;
    compressedData->lens = len_t;
    compressedData->count = count;
    compressedData->frame = frame;

    return compressedData;
}

// Decompress timestamps from compressed and compacted data, then write them into
// specific scope of buffer for decompression
__device__ static inline void timestamp_decompress_device(
    uint32_t start, uint32_t end, BitReader *bitReader
) {
    // avoid acessing out of bound
    //assert(start <= end);
    if (start >= end) return;

    // decalre
    uint32_t
        controlBits, storedZeros = 0;
    int64_t
        timestamp, prevTimestamp,
        newDelta, deltaOfDelta, prevDelta;
    uint64_t
        *decompressed_t = c_decompressed_t;

    // cause the header of current frame has been decided(i.e. 
    // 'start' must >=1), we can get the previous timestamp and 
    // 'delta' value as follow
    prevTimestamp = decompressed_t[start - 1];
    prevDelta = 0;
    deltaOfDelta = 0;

    for (uint32_t cursor = start; cursor < end; cursor++) {
        // if storedZeros != 0, previous and current timestamp interval/delta is same,
        // just update prevTimestamp and storedZeros, and return prevTimestamp.
        if (storedZeros > 0) {
            storedZeros--;
            prevTimestamp = prevDelta + prevTimestamp;
            // return prevTimestamp;
            decompressed_t[cursor] = prevTimestamp;
            continue;
        }

        // read control bits and match the cases
        controlBits = bitReaderNextControlBits(bitReader, 4);
        switch (controlBits) {
        case 0b0:
            // '0' bit (i.e. previous and current timestamp interval(delta) is same).
            // read consecutive zeros control bits.
            controlBits = bitReaderNextBit(bitReader);
            switch (controlBits) {
            case 0b0:// '0'+'0'+3
                storedZeros = (uint32_t)bitReaderNextLong(bitReader, 3);
                break;
            case 0b1:// '0'+'1'+5
                storedZeros = (uint32_t)bitReaderNextLong(bitReader, 5);
                break;
            }
            // since we have reduce the 'storedZeros' by 1 when we
            // compress it, we need to restore it's value here.
            storedZeros++;

            // decompress and return timestamp
            storedZeros--;
            prevTimestamp = prevDelta + prevTimestamp;
            decompressed_t[cursor] = prevTimestamp;
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
            // if current deltaOfDelta value is the special end sign, set the isClosed value to true
            // (i.e. this buffer reach the end).
            break;
        default:
            break;
        }
        // decode the deltaOfDelta value.
        deltaOfDelta = decodeZigZag32((int32_t)deltaOfDelta);

        // since we have reduce the 'delta-of-delta' by 1 when we compress the 'delta-of-delta',
        // we restore it's value here.
        if (deltaOfDelta >= 0) deltaOfDelta++;

        // calculate the new delta and timestamp
        newDelta = prevDelta + deltaOfDelta;
        timestamp = prevTimestamp + newDelta;

        // update prevDelta and prevTimestamp
        prevDelta = newDelta;
        prevTimestamp = timestamp;

        // decompress and return timestamp
        decompressed_t[cursor] = timestamp;
    }
}

// Timestamp decompression kernal
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

    // since the 'Size and Alignment Requirement', we have to read the 
    // the compressed data byte-by-byte in lettle-endian mode
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

    // decompress timestamps within current frame
    timestamp_decompress_device(start, end, &bitReader);
}


// Decompress compressed and compacted timestamps data on GPU using RLE
ByteBuffer *timestamp_decompress_rle_gpu(
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

    // use global __constant__ variables to pass the params to avoid passing common params between functions
    checkCudaError(cudaMemcpyToSymbol(c_compressed_t, &d_compressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_decompressed_t, &d_decompressed_t, sizeof(void *)));

    // initiate decompressed kernal
    timestamp_decompress_kernal <<<block, thdPB >>> (frame, thd, count);
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