//
// Created by depaulsmiller on 10/4/20.
//

#ifndef GPUHASHMAPS_GPUDATA_CUH
#define GPUHASHMAPS_GPUDATA_CUH
namespace megakv {
    struct GPUData {


        GPUData() {
            gpuErrchk(cudaMallocHost(&keys_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
            gpuErrchk(cudaMallocHost(&values_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
            gpuErrchk(cudaMallocHost(&requests_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
            gpuErrchk(cudaMalloc(&keys_k, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
            gpuErrchk(cudaMalloc(&values_k, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
            gpuErrchk(cudaMalloc(&requests_k, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS));
        }

        inline void moveToGPU(cudaStream_t stream) {
            gpuErrchk(cudaMemcpyAsync(keys_k, keys_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS,
                                      cudaMemcpyHostToDevice,
                                      stream));
            gpuErrchk(cudaMemcpyAsync(values_k, values_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS,
                                      cudaMemcpyHostToDevice, stream));
            gpuErrchk(cudaMemcpyAsync(requests_k, requests_h, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS,
                                      cudaMemcpyHostToDevice, stream));
        }

        inline void moveValuesBack(cudaStream_t stream) {
            gpuErrchk(cudaMemcpyAsync(values_h, values_k, sizeof(unsigned) * THREADS_PER_BLOCK * BLOCKS,
                                      cudaMemcpyDeviceToHost, stream));
        }

        unsigned *keys_h;
        unsigned *values_h;
        int *requests_h;

        unsigned *keys_k;
        unsigned *values_k;
        int *requests_k;

    };

}
#endif //GPUHASHMAPS_GPUDATA_CUH
