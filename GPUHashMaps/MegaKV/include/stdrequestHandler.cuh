//
// Created by depaulsmiller on 7/24/20.
//

#include "Operations.cuh"
#include "gpuErrchk.cuh"
#include "Request.cuh"

#ifndef MEGAKV_STDREQUESTHANDLER_CUH
#define MEGAKV_STDREQUESTHANDLER_CUH

namespace megakv {

    int getBlocks() noexcept {
        cudaDeviceProp prop;
        prop.multiProcessorCount = 68;

        auto code = cudaGetDeviceProperties(&prop, 0);
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), __FILE__,
                    __LINE__);
        }
        return 2 * prop.multiProcessorCount;
    }

    const int BLOCKS = getBlocks() / 4; // TODO implement some tuning to run on any machine
    const int THREADS_PER_BLOCK = 512;

    /**
     *
     * @param slabs
     * @param num_of_buckets
     * @param myKey
     * @param myValue
     * @param request
     */
    __global__ void requestHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                                   unsigned *myKey,
                                   unsigned *myValue, int *request) {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        unsigned key = myKey[tid];
        unsigned value = myValue[tid];
        bool activity = (request[tid] == REQUEST_GET);

        warp_operation_search(activity, key, value, slabs, num_of_buckets);

        activity = (request[tid] == REQUEST_INSERT);
        warp_operation_replace(activity, key, value, slabs,
                               num_of_buckets);
        activity = (request[tid] == REQUEST_REMOVE);
        warp_operation_delete(activity, key, value, slabs, num_of_buckets);
        myValue[tid] = value;
    }

}

#endif//GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
