#include <groupallocator>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <gpuErrchk.cuh>
#include <gpumemory.cuh>
#include <iostream>
#include <nvfunctional>

#ifndef OPERATIONS_CUH
#define OPERATIONS_CUH

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 32
#define VALID_KEY_MASK 0x7fffffffu
#define DELETED_KEY UINT32_MAX

const unsigned OP_SUCCESS = 2;
const unsigned OP_NONSUCESS = 3;


const unsigned long long EMPTY = 0;
const unsigned long long EMPTY_POINTER = 0;
#define BASE_SLAB 0

struct SlabData {

    unsigned long long *keyValue;

    // the 32nd element is next
    //unsigned long long *next;
};

struct MemoryBlock {
    MemoryBlock() : bitmap(~0u), slab(nullptr) {
    }

    unsigned bitmap;
    SlabData *slab;// 32 slabs
};

struct SuperBlock {
    MemoryBlock *memblocks;// 32 memblocks
};

struct WarpAllocCtx {
    WarpAllocCtx() : blocks(nullptr) {
        // creates default context
    }

    SuperBlock *blocks;
    // there should be a block per warp ie threadsPerBlock * blocks / 32 superblocks
};

std::ostream &operator<<(std::ostream &output, const SlabData &s) {
    output << s.keyValue;
    return output;
}

WarpAllocCtx setupWarpAllocCtxGroup(groupallocator::GroupAllocator &gAlloc, int threadsPerBlock, int blocks, int gpuid = 0, cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid))
    WarpAllocCtx actx;
    gAlloc.allocate(&actx.blocks, (size_t) ceil(threadsPerBlock * blocks / 32.0) * sizeof(SuperBlock), false);
    for (size_t i = 0; i < (size_t) ceil(threadsPerBlock * blocks / 32.0); i++) {
        gAlloc.allocate(&(actx.blocks[i].memblocks), sizeof(MemoryBlock) * 32, false);
        for (int j = 0; j < 32; j++) {
            actx.blocks[i].memblocks[j].bitmap = ~0u;
            gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab), sizeof(SlabData) * 32, false);
            for (int k = 0; k < 32; k++) {
                gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab[k].keyValue), sizeof(unsigned long long) * 32, false);
                for (int w = 0; w < 32; w++) {
                    actx.blocks[i].memblocks[j].slab[k].keyValue[w] = 0;
                }
            }
        }
    }
    gAlloc.moveToDevice(gpuid, stream);
    gpuErrchk(cudaDeviceSynchronize())
    return actx;
}

struct SlabCtx {
    SlabCtx() : slabs(nullptr), num_of_buckets(0) {}

    volatile SlabData **slabs;
    unsigned num_of_buckets;
};

__host__ __device__ unsigned hash(unsigned src_key, unsigned num_of_buckets);

__forceinline__ __device__ unsigned long long
ReadSlab(const unsigned long long &next, const unsigned &src_bucket,
         const unsigned laneId, volatile SlabData **slabs) {
        return next == BASE_SLAB ? slabs[src_bucket]->keyValue[laneId] : ((SlabData*)next)->keyValue[laneId];
}

__forceinline__ __device__ unsigned long long *
SlabAddress(const unsigned long long &next, const unsigned &src_bucket,
            const unsigned laneId, volatile SlabData **slabs,
            unsigned num_of_buckets) {
    return (next == BASE_SLAB ? slabs[src_bucket]->keyValue : ((SlabData*)next)->keyValue) + laneId;
}

// just doing parallel shared-nothing allocation
__forceinline__ __device__ unsigned long long warp_allocate(WarpAllocCtx ctx) {

    const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
    const unsigned laneId = threadIdx.x & 0x1Fu;
    if (ctx.blocks == nullptr) {
        return 0;
    }

    MemoryBlock *blocks = ctx.blocks[warpIdx].memblocks;
    unsigned bitmap = blocks[laneId].bitmap;
    int index = __ffs((int) bitmap) - 1;
    int ballotThread = __ffs((int) __ballot_sync(~0u, (index != -1))) - 1;
    if (ballotThread == -1) {
        return 0;
    }
    auto location = (unsigned long long) (blocks[laneId].slab + index);
    if (ballotThread == laneId) {
        //unsigned oldbitmap = bitmap;
        bitmap = bitmap ^ (1u << (unsigned) index);
        blocks[laneId].bitmap = bitmap;
    }
    location = __shfl_sync(~0u, location, ballotThread);

    return location;
}

__forceinline__ __device__ void deallocate(WarpAllocCtx ctx, unsigned long long l) {

    const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
    const unsigned laneId = threadIdx.x & 0x1Fu;
    if (ctx.blocks == nullptr) {
        return;
    }

    MemoryBlock *blocks = ctx.blocks[warpIdx].memblocks;
    if ((unsigned long long) blocks[laneId].slab <= l && (unsigned long long) (blocks[laneId].slab + 32) > l) {
        unsigned diff = l - (unsigned long long) blocks[laneId].slab;
        unsigned idx = diff / sizeof(SlabData);
        blocks[laneId].bitmap = blocks[laneId].bitmap | (1u << idx);
    }
}

__forceinline__ __host__ __device__ unsigned hash(unsigned src_key, unsigned num_of_buckets) {
    return src_key % num_of_buckets;
}

// manually inlined
__forceinline__ __device__ void warp_operation_search(bool &is_active, const unsigned &myKey,
                                                      unsigned &myValue,
                                                      volatile SlabData **__restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    const unsigned threadKey = myKey;

    //if (laneId == 0)
    //    printf("%ld %d %d\n", next, blockIdx.x, threadIdx.x);

    unsigned last_work_queue = work_queue;

    while (work_queue != 0) {

        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

        unsigned src_lane = __ffs((int) work_queue) - 1;
        unsigned src_key = __shfl_sync(~0u, threadKey, (int) src_lane);
        unsigned src_bucket = hash(src_key, num_of_buckets);
        unsigned long long read_data =
                ReadSlab(next, src_bucket, laneId, slabs);

        unsigned map_key = (read_data >> 32u) & 0xffffffffu;
        auto masked_ballot = (unsigned) (__ballot_sync(~0u, map_key == src_key) & VALID_KEY_MASK);

        if (masked_ballot != 0) {
            auto found_lane = (unsigned) (__ffs(masked_ballot) - 1);
            unsigned long long found_value = __shfl_sync(~0u, read_data, found_lane);
            if (laneId == src_lane) {
                myValue = found_value & 0xffffffff;
                is_active = false;
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, read_data, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            } else {
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;


        work_queue = __ballot_sync(~0u, is_active);
    }
}


__forceinline__ __device__ void
warp_operation_delete(bool &is_active, const unsigned &myKey,
                      unsigned &myValue,
                      volatile SlabData **__restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = work_queue;

    while (work_queue != 0) {
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
        auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
        unsigned src_key = __shfl_sync(~0u, myKey, src_lane);
        unsigned src_bucket = hash(src_key, num_of_buckets);
        unsigned long long read_data =
                ReadSlab(next, src_bucket, laneId, slabs);

        auto key = (unsigned) ((read_data >> 32u) & 0xffffffff);

        int masked_ballot = (int) (__ballot_sync(~0u, key == src_key) & VALID_KEY_MASK);

        if (masked_ballot != 0) {

            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                *(SlabAddress(next, src_bucket, dest_lane, slabs, num_of_buckets)) =
                        DELETED_KEY;
                is_active = false;
                myValue = OP_SUCCESS;
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, read_data, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                is_active = false;
                myValue = OP_NONSUCESS;
            } else {
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
    }
}

__forceinline__ __device__ void
warp_operation_replace(bool &is_active, const unsigned &myKey,
                       unsigned &myValue,
                       volatile SlabData **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx ctx) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = work_queue;

    while (work_queue != 0) {
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
        //if (laneId == 0)
        //    printf("%ld %d %d\n", next, blockIdx.x, threadIdx.x);
        auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
        unsigned src_key = __shfl_sync(~0u, myKey, src_lane);
        unsigned src_bucket = hash(src_key, num_of_buckets);
        // if (laneId == 0)
        //  printf("src_lane %d from %d: %d -> %d\n", src_lane, work_queue, src_key,
        //  src_bucket);
        unsigned long long read_data =
                ReadSlab(next, src_bucket, laneId, slabs);

        auto key = (unsigned) ((read_data >> 32u) & 0xffffffff);
        bool to_share = (key == EMPTY || key == src_key);
        int masked_ballot = (int) (__ballot_sync(~0u, to_share) & VALID_KEY_MASK);

        if (masked_ballot != 0) {
            if (src_lane == laneId) {
                unsigned dest_lane = (unsigned) __ffs(masked_ballot) - 1;
                unsigned long long newPair = ((unsigned long long) myKey << 32u) | (unsigned long long) myValue;
                unsigned long long *addr =
                        SlabAddress(next, src_bucket, dest_lane, slabs, num_of_buckets);
                unsigned long long old_pair = atomicCAS(addr, 0, newPair);
                if (old_pair == 0) {
                    myValue = OP_SUCCESS;
                    is_active = false;
                } else if ((unsigned) ((old_pair >> 32u) & 0xffffffff) == myKey) {
                    myValue = OP_NONSUCESS;
                    is_active = false;
                }
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, read_data, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                __threadfence_system();
                unsigned long long new_slab_ptr = warp_allocate(ctx);
                if (laneId == ADDRESS_LANE - 1) {
                    unsigned long long temp = atomicCAS(
                            SlabAddress(next, src_bucket, ADDRESS_LANE - 1, slabs, num_of_buckets),
                            EMPTY_POINTER, new_slab_ptr);
                    if (temp != EMPTY_POINTER) {
                        deallocate(ctx, new_slab_ptr);
                    }
                }
            } else {
                next = next_ptr;
            }
        }
        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
    }
}

SlabCtx *setUp(unsigned size, unsigned numberOfSlabsPerBucket) {
    auto *sctx = new SlabCtx();
    sctx->num_of_buckets = size;
    auto *slabs_tmp = new GPUCPU2DArray<SlabData>(sctx->num_of_buckets, numberOfSlabsPerBucket);

    for (int i = 0; i < sctx->num_of_buckets; i++) {
        std::cerr << i << std::endl;
        for (int k = 0; k < numberOfSlabsPerBucket; k++) {

            auto *keyValue_tmp = new GPUCPUMemory<unsigned long long>(32);
            //GPUCPUMemory<unsigned long long> *next_tmp = new GPUCPUMemory<unsigned long long>(1);

            (*slabs_tmp)[i][k].keyValue = keyValue_tmp->getDevice();
            //(*slabs_tmp)[i][k].next = next_tmp->getDevice();


            for (int j = 0; j < 31; j++) {
                keyValue_tmp->host[j] = 0;// EMPTY_PAIR;
            }
            if (k < numberOfSlabsPerBucket - 1) {
                keyValue_tmp->host[31] = (unsigned long long) k + 2ull;
            } else {
                keyValue_tmp->host[31] = 0;// EMPTY_POINTER;
            }
            keyValue_tmp->movetoGPU();
            //next_tmp->movetoGPU();
        }
    }
    //slabs_tmp->print();
    //exit(1);
    slabs_tmp->movetoGPU();
    sctx->slabs = (volatile SlabData **) slabs_tmp->getDevice2DArray();
    std::cerr << "Size allocated so far: " << (double) (sizeof(SlabData) * sctx->num_of_buckets * numberOfSlabsPerBucket + sizeof(unsigned long long) * 32 * sctx->num_of_buckets * numberOfSlabsPerBucket) / 1024.0 / 1024.0 / 1024.0 << "GB"
              << std::endl;
    return sctx;
}

SlabCtx *setUpUnified(unsigned size, unsigned numberOfSlabsPerBucket) {
    auto sctx = new SlabCtx();
    sctx->num_of_buckets = size;
    gpuErrchk(cudaMallocManaged(&(sctx->slabs), sizeof(void *) * sctx->num_of_buckets));

    for (int i = 0; i < sctx->num_of_buckets; i++) {
        gpuErrchk(cudaMallocManaged(&sctx->slabs[i], sizeof(SlabData) * numberOfSlabsPerBucket));
        for (int k = 0; k < numberOfSlabsPerBucket; k++) {

            gpuErrchk(cudaMallocManaged((unsigned long long **) &(sctx->slabs[i][k].keyValue),
                                        sizeof(unsigned long long) * 32));

            for (int j = 0; j < 31; j++) {
                sctx->slabs[i][k].keyValue[j] = 0;// EMPTY_PAIR;
            }
            if (k < numberOfSlabsPerBucket - 1) {
                sctx->slabs[i][k].keyValue[31] = (unsigned long long) k + 2ull;
            } else {
                sctx->slabs[i][k].keyValue[31] = 0;// EMPTY_POINTER;
            }
            gpuErrchk(cudaMemAdvise((unsigned long long *) sctx->slabs[i][k].keyValue, sizeof(unsigned long long) * 32,
                                    cudaMemAdviseSetPreferredLocation, 0));
            gpuErrchk(cudaMemPrefetchAsync((unsigned long long *) sctx->slabs[i][k].keyValue,
                                           sizeof(unsigned long long) * 32,
                                           0));
        }
        gpuErrchk(cudaMemAdvise((SlabData *) sctx->slabs[i], sizeof(SlabData) * numberOfSlabsPerBucket,
                                cudaMemAdviseSetPreferredLocation, 0));
        gpuErrchk(cudaMemPrefetchAsync((SlabData *) sctx->slabs[i], sizeof(SlabData) * numberOfSlabsPerBucket, 0));
    }
    //slabs_tmp->print();
    //exit(1);
    gpuErrchk(cudaMemAdvise(sctx->slabs, sizeof(void *) * sctx->num_of_buckets, cudaMemAdviseSetPreferredLocation, 0));
    gpuErrchk(cudaMemPrefetchAsync(sctx->slabs, sizeof(void *) * sctx->num_of_buckets, 0));
    gpuErrchk(cudaDeviceSynchronize());
    std::cerr << "Size allocated so far: " << (double) (sizeof(SlabData) * sctx->num_of_buckets * numberOfSlabsPerBucket + sizeof(unsigned long long) * 32 * sctx->num_of_buckets * numberOfSlabsPerBucket) / 1024.0 / 1024.0 / 1024.0 << "GB" << std::endl;
    return sctx;
}

SlabCtx *setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, unsigned numberOfSlabsPerBucket, int gpuid = 0, cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid));

    auto sctx = new SlabCtx();
    sctx->num_of_buckets = size;

    gAlloc.allocate(&(sctx->slabs), sizeof(void *) * sctx->num_of_buckets, false);

    for (int i = 0; i < sctx->num_of_buckets; i++) {
        gAlloc.allocate(&sctx->slabs[i], sizeof(SlabData) * numberOfSlabsPerBucket, false);

        for (int k = 0; k < numberOfSlabsPerBucket; k++) {

            gAlloc.allocate((unsigned long long **) &(sctx->slabs[i][k].keyValue),sizeof(unsigned long long) * 32, false);

            for (int j = 0; j < 31; j++) {
                sctx->slabs[i][k].keyValue[j] = 0;// EMPTY_PAIR;
            }
            if (k < numberOfSlabsPerBucket - 1) {
                sctx->slabs[i][k].keyValue[31] = (unsigned long long) k + 2ull;
            } else {
                sctx->slabs[i][k].keyValue[31] = 0;// EMPTY_POINTER;
            }
        }
    }

    gAlloc.moveToDevice(gpuid, stream);

    gpuErrchk(cudaDeviceSynchronize())

    std::cerr << "Size allocated so far: " << (double)(sizeof(SlabData) * sctx->num_of_buckets * numberOfSlabsPerBucket + sizeof(unsigned long long) * 32 * sctx->num_of_buckets * numberOfSlabsPerBucket) / 1024.0 / 1024.0 / 1024.0 << "GB" << std::endl;
    return sctx;
}


#endif
