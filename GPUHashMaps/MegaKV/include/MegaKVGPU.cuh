/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>

#include "stdrequestHandler.cuh"

#ifndef MegaKVGPU_CUH
#define MegaKVGPU_CUH

namespace megakv {

    class MegaKVGPU {
    public:

        explicit MegaKVGPU(int size) {
            slab = setUp(size);
        }

        void batch(unsigned *keys, unsigned *values, int *requests, cudaStream_t stream, float &ms) {
            cudaEvent_t start, end;
            gpuErrchk(cudaEventCreate(&start));
            gpuErrchk(cudaEventCreate(&end));

            gpuErrchk(cudaEventRecord(start, stream));
            requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(slab->slabs, slab->num_of_buckets, keys, values,
                                                                     requests);
            gpuErrchk(cudaEventRecord(end, stream));
            gpuErrchk(cudaEventSynchronize(end));
            gpuErrchk(cudaEventElapsedTime(&ms, start, end));
            gpuErrchk(cudaEventDestroy(start));
            gpuErrchk(cudaEventDestroy(end));
        }

        void exec_async(unsigned *keys, unsigned *values, int *requests, cudaStream_t stream) {
            requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(slab->slabs, slab->num_of_buckets, keys, values,
                                                                     requests);
        }

    private:
        SlabCtx *slab;
    };

}
#endif // SLAB_SLAB_CUH
