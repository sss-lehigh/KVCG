#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpuErrchk.cuh"
#include <iostream>
#include <nvfunctional>

#ifndef MEGAKV_OPERATIONS_CUH
#define MEGAKV_OPERATIONS_CUH

#define SEARCH_NOT_FOUND 0
#define DELETED_KEY 0

namespace megakv {

    const unsigned OP_SUCCESS = 2;
    const unsigned OP_NONSUCESS = 3;

    const unsigned long long EMPTY = 0;
    const unsigned long long DELETED = ~0ull;
    const unsigned long long EMPTY_POINTER = 0;

    struct SlabData {

        unsigned long long keyValue[32];

        // the 32nd element is next
        //unsigned long long *next;
    };

    std::ostream &operator<<(std::ostream &output, const SlabData &s) {
        output << s.keyValue;
        return output;
    }

    struct SlabCtx {
        SlabCtx() : slabs(nullptr), num_of_buckets(0) {}

        volatile SlabData **slabs;
        unsigned num_of_buckets;
    };

    __host__ __device__ unsigned hash(unsigned src_key, unsigned num_of_buckets);

    __forceinline__ __device__ unsigned long long
    ReadSlab(const unsigned &src_bucket,
             const unsigned laneId, volatile SlabData **slabs) {
        return slabs[src_bucket]->keyValue[laneId];
    }

    __forceinline__ __device__ unsigned long long *
    SlabAddress(const unsigned &src_bucket,
                const unsigned laneId, volatile SlabData **slabs) {
        return (unsigned long long *) slabs[src_bucket]->keyValue + laneId;
    }

    __forceinline__ __host__ __device__ unsigned hash(unsigned src_key, unsigned num_of_buckets) {
        return src_key % num_of_buckets;
    }

    __forceinline__ __device__ void warp_operation_search(bool &is_active, const unsigned &myKey,
                                                          unsigned &myValue,
                                                          volatile SlabData **__restrict__ slabs, unsigned num_of_buckets) {
        const unsigned laneId = threadIdx.x & 0x1Fu;
        unsigned work_queue = __ballot_sync(~0u, is_active);

        const unsigned threadKey = myKey;

        //if (laneId == 0)
        //    printf("%ld %d %d\n", next, blockIdx.x, threadIdx.x);

        while (work_queue != 0) {

            unsigned src_lane = __ffs((int) work_queue) - 1;
            unsigned src_key = __shfl_sync(~0u, threadKey, (int) src_lane);
            unsigned src_bucket = hash(src_key, num_of_buckets);
            unsigned long long read_data = ReadSlab(src_bucket, laneId, slabs);

            unsigned map_key = (read_data >> 32u) & 0xffffffffu;
            auto masked_ballot = (unsigned) (__ballot_sync(~0u, map_key == src_key));

            if (masked_ballot != 0) {
                auto found_lane = (unsigned) (__ffs(masked_ballot) - 1);
                unsigned long long found_value = __shfl_sync(~0u, read_data, found_lane);
                if (laneId == src_lane) {
                    myValue = found_value & 0xffffffff;
                    is_active = false;
                }
            } else {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            }

            work_queue = __ballot_sync(~0u, is_active);
        }
    }


    __forceinline__ __device__ void
    warp_operation_delete(bool &is_active, const unsigned &myKey,
                          unsigned &myValue,
                          volatile SlabData **__restrict__ slabs, unsigned num_of_buckets) {
        const unsigned laneId = threadIdx.x & 0x1Fu;
        unsigned work_queue = __ballot_sync(~0u, is_active);

        while (work_queue != 0) {
            auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
            unsigned src_key = __shfl_sync(~0u, myKey, src_lane);
            unsigned src_bucket = hash(src_key, num_of_buckets);
            unsigned long long read_data = ReadSlab(src_bucket, laneId, slabs);

            auto key = (unsigned) ((read_data >> 32u) & 0xffffffff);

            int masked_ballot = (int) (__ballot_sync(~0u, key == src_key));

            if (masked_ballot != 0) {

                if (src_lane == laneId) {
                    unsigned dest_lane = __ffs(masked_ballot) - 1;

                    read_data = ReadSlab(src_bucket, dest_lane, slabs);

                    key = (unsigned) ((read_data >> 32u) & 0xffffffff);

                    if (key == myKey) {
                        //TODO switch to deleted
                        if (atomicCAS(SlabAddress(src_bucket, dest_lane, slabs), read_data, EMPTY) == read_data) {
                            myValue = (unsigned) (read_data & 0xffffffff);
                        } else {
                            myValue = 0;
                        }
                    }
                    is_active = false;
                }
            } else {
                is_active = false;
                myValue = 0;
            }

            work_queue = __ballot_sync(~0u, is_active);
        }
    }

    __forceinline__ __device__ void
    warp_operation_replace(bool &is_active, const unsigned &myKey,
                           unsigned &myValue,
                           volatile SlabData **__restrict__ slabs, unsigned num_of_buckets) {
        const unsigned laneId = threadIdx.x & 0x1Fu;
        unsigned work_queue = __ballot_sync(~0u, is_active);

        while (work_queue != 0) {

            auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
            unsigned src_key = __shfl_sync(~0u, myKey, src_lane);
            unsigned src_bucket = hash(src_key, num_of_buckets);

            unsigned long long read_data = ReadSlab(src_bucket, laneId, slabs);

            auto key = (unsigned) ((read_data >> 32u) & 0xffffffff);
            bool to_share = (key == src_key || key == EMPTY);
            int masked_ballot = (int) (__ballot_sync(~0u, to_share)); // will be operating on the desired key before empty

            if (masked_ballot != 0) {
                if (src_lane == laneId) {
                    unsigned dest_lane = (unsigned) __ffs(masked_ballot) - 1;
                    unsigned long long newPair = ((unsigned long long) myKey << 32u) | (unsigned long long) myValue;
                    unsigned long long *addr = SlabAddress(src_bucket, dest_lane, slabs);
                    unsigned long long old_pair = atomicCAS(addr, 0, newPair);
                    if (old_pair == 0) {
                        myValue = 0;
                        is_active = false;
                    } else if ((unsigned) ((old_pair >> 32u) & 0xffffffff) == myKey) {
                        if (atomicCAS(addr, old_pair, newPair) == old_pair) {
                            myValue = old_pair & 0xffffffff;
                            is_active = false;
                        }
                    }
                }
            } else {
                // evict
                unsigned long long *addr = SlabAddress(src_bucket, laneId, slabs);
                unsigned long long newPair = ((unsigned long long) myKey << 32u) | (unsigned long long) myValue;

                unsigned long long old_pair = atomicExch(addr, newPair);
                myValue = old_pair & 0xffffffff;
                is_active = false;
            }

            work_queue = __ballot_sync(~0u, is_active);
        }
    }

    SlabCtx *setUp(unsigned size) {
        auto *sctx = new SlabCtx();
        sctx->num_of_buckets = size;
        volatile SlabData **slabs_tmp_h = new volatile SlabData *[size];
        volatile SlabData zeroedSlab;

        for (int i = 0; i < 32; i++) {
            zeroedSlab.keyValue[i] = 0;
        }

        for (int i = 0; i < sctx->num_of_buckets; i++) {
            gpuErrchk(cudaMalloc(&(slabs_tmp_h[i]), sizeof(SlabData)));
            gpuErrchk(cudaMemcpy((SlabData *) slabs_tmp_h[i], (SlabData *) &zeroedSlab, sizeof(SlabData), cudaMemcpyHostToDevice));
        }

        gpuErrchk(cudaMalloc(&(sctx->slabs), sizeof(SlabData *) * sctx->num_of_buckets));
        gpuErrchk(cudaMemcpy((SlabData **) sctx->slabs, (SlabData **) slabs_tmp_h, sizeof(SlabData *) * sctx->num_of_buckets, cudaMemcpyHostToDevice));

        delete[] slabs_tmp_h;

        std::cerr << "Size allocated so far: " << (double) ((sizeof(SlabData *) + sizeof(SlabData)) * sctx->num_of_buckets) / 1024.0 / 1024.0 / 1024.0 << "GB"
                  << std::endl;
        return sctx;
    }

}

#endif
