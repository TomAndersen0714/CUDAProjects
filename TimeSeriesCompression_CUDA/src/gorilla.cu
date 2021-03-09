#include "compressors.h"
#include "utils/encode_utils.h"
#include "utils/cuda_common_utils.h"
#include "utils/bit_writer.cuh"

__constant__ uint32_t c_count; // the number of data points
__constant__ uint64_t* c_uncompressed_t; // uncompressed timestamps on device
__constant__ uint64_t* c_uncompressed_v; // uncompressed values on device
__constant__ uint32_t* c_offs; // offset of uncompressed data divided by lane
__constant__ byte* c_compressed_t; // compressed timestamps on device
__constant__ byte* c_compressed_v; // compressed values on device
__constant__ uint32_t* c_len_t; // length of compressed timestamps on device
__constant__ uint32_t* c_len_v; // length of compressed values on device
__constant__ uint64_t* c_decompressed_t; // decompressed timestamps on device
__constant__ uint64_t* c_decompressed_v; // decompressed values on device

// Timestamps compression kernal 
__global__ static void timestamp_compress_kernal();

// Compress timestamps in specific scope of uncompressed timestamps
__device__ static inline void timestamp_compress_device(
    int start, int end, BitWriter *bitWriter, uint32_t thdIdx
);

// Compress timestamps on GPU
void timestamp_compress_gorilla_gpu(
    ByteBuffer* tsByteBuffer,
    uint32_t blocks,
    uint32_t warps
) {
    // divide the uncompressed data into frames according to the 
    // total number of threads
    uint32_t
        count, // the number of data points
        thdOfBlock, // the number of threads within per block
        thd, // the total number of needed threads
        *offs, // start offsets of data that each thread will compress
        frame, // the length of data that each thread will compress
        padding = 0, // unused pos in the used last frame
        left = 0; // unused threads in the last block

    count = tsByteBuffer->length / BYTES_OF_LONG_LONG;
    thdOfBlock = WARPSIZE*warps;
    thd = thdOfBlock*blocks;
    frame = (count + thd - 1) / thd;

    // if count <= MIN_FRAME_SIZE, use just one thread to compress
    if (count <= MIN_FRAME_SIZE) {
        frame = count; blocks = 1; thdOfBlock = 1; thd = 1;
    }
    else if (frame < MIN_FRAME_SIZE) {// else if frame is too small
        frame = MIN_FRAME_SIZE;
        // recalculatre the number of needed threads
        thd = (count + frame - 1) / frame;
        padding = frame - count % frame;
        if (thd < MAX_THREADS_PER_BLOCK) {
            // use just one block to compress
            blocks = 1;
            //warps = (thd + WARPSIZE - 1) / WARPSIZE;
            thdOfBlock = thd;
        }
        else {
            // use block as less as possible to compress according to the 
            // number of threads within per block
            blocks = (thd + thdOfBlock - 1) / thdOfBlock;
            left = blocks*thdOfBlock - thd;
            thd = blocks*thdOfBlock;
        }
    }

    // construct the offsets array, each scope of frame is [offs[i],offs[i+1])
    offs = (uint32_t*)malloc(BYTES_OF_INT*(thd + 1));
    for (int i = 0; i <= thd - left - 1; i++)
        offs[i] = i*frame;
    offs[thd - left] = (thd - left)*frame - padding;
    for (int i = thd - left + 1; i <= thd; i++)
        offs[i] = offs[thd - left];

    // allocate device memory and tranport data to GPU
    uint64_t* d_uncompressed_t; // uncompressed timestamps
    byte* d_compressed_t; // compressed timestamps
    uint32_t* d_len_t; // length of compressed timestamps on device
    uint32_t* d_offs; // data offset of threads
    checkCudaError(cudaMalloc((void**)&d_uncompressed_t, tsByteBuffer->length));
    checkCudaError(cudaMalloc((void**)&d_compressed_t, tsByteBuffer->length));
    checkCudaError(cudaMalloc((void**)&d_len_t, BYTES_OF_INT*thd));
    checkCudaError(cudaMalloc((void**)&d_offs, BYTES_OF_INT*(thd + 1)));
    checkCudaError(cudaMemcpy(
        d_uncompressed_t, tsByteBuffer->buffer,
        tsByteBuffer->length, cudaMemcpyHostToDevice
    ));
    checkCudaError(cudaMemcpy(
        d_offs, offs,
        BYTES_OF_INT*(thd + 1), cudaMemcpyHostToDevice
    ));
    checkCudaError(cudaMemcpyToSymbol(c_uncompressed_t, &d_uncompressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_compressed_t, &d_compressed_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_len_t, &d_len_t, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_offs, &d_offs, sizeof(void *)));
    checkCudaError(cudaMemcpyToSymbol(c_count, &count, sizeof(uint32_t)));

    // initiate kernal
    timestamp_compress_kernal <<<blocks, thdOfBlock >>> ();

    checkCudaError(cudaDeviceSynchronize());

    // allocate cpu memory for compressed data, and copy data from GPU to CPU
    byte* compressed_t = (byte*)malloc(tsByteBuffer->length);
    uint32_t* len_t = (uint32_t*)malloc(BYTES_OF_INT*thd);
    checkCudaError(cudaMemcpy(
        compressed_t, d_compressed_t, tsByteBuffer->length,
        cudaMemcpyDeviceToHost)
    );
    checkCudaError(cudaMemcpy(
        len_t, d_len_t, BYTES_OF_INT*thd,
        cudaMemcpyDeviceToHost)
    );

    // Print the compressed data(just for test)
    printCompressedData(compressed_t, offs, len_t, thd);

    // return compressed data and it's length
}


// Timestamps compression kernal
__global__ static void timestamp_compress_kernal() {
    // declare
    uint32_t
        thdIdx, // thread index within grid
        start, // start offset of uncompressed data in current thread
        end; // end offset of uncompressed data in current thread

    thdIdx = threadIdx.x + blockIdx.x*blockDim.x;
    start = c_offs[thdIdx];
    end = c_offs[thdIdx + 1];

    // construct
    ByteBuffer byteBuffer;
    byteBuffer.buffer = (byte*)(((uint64_t*)c_compressed_t)+start); // start pos for compression
    byteBuffer.length = 0;
    BitWriter bitWriter;
    bitWriter.byteBuffer = &byteBuffer;
    bitWriter.cacheByte = 0;
    bitWriter.leftBits = BITS_OF_BYTE;

    // compress the timestamps within this thread
    timestamp_compress_device(start, end, &bitWriter, thdIdx);

}

// Compress timestamps in specific scope of uncompressed timestamps
__device__ static inline void timestamp_compress_device(
    int start, int end, BitWriter *bitWriter, uint32_t thdIdx
) {
    // declaration
    int64_t timestamp, prevTimestamp;
    int32_t newDelta, deltaOfDelta, prevDelta;
    uint32_t leastBitLength;
    uint64_t *tsBuffer = c_uncompressed_t;

    if (start == 0) {// If current timestamp is the first one
        prevTimestamp = 0;
        prevDelta = 0;
    }
    else {// else 'start'>=32
        prevTimestamp = tsBuffer[start - 1];
        prevDelta = prevTimestamp - tsBuffer[start - 2];
    }

    // compress every timestamp in the scope into the compressed buffer
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

    // write the left bits in cached byte into the buffer.
    bitWriterFlush(bitWriter);

    // write the byte length of compressed timestamps
    c_len_t[thdIdx] = bitWriter->byteBuffer->length;
}