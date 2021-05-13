#include <groupallocator>
#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <gpuErrchk.cuh>
#include <gpumemory.cuh>
#include <iostream>
#include <nvfunctional>
#include "ImportantDefinitions.cuh"

#ifndef OPERATIONS_CUH
#define OPERATIONS_CUH

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 32
#define VALID_KEY_MASK 0x7fffffffu
#define DELETED_KEY 0

const unsigned OP_SUCCESS = 2;
const unsigned OP_NONSUCESS = 3;


//const unsigned long long EMPTY = 0;
const unsigned long long EMPTY_POINTER = 0;
#define BASE_SLAB 0

template<typename K, typename V>
struct SlabData {

    typedef K KSub;

    union {
        int ilock;
        char p[128];
    }; // 128 bytes

    KSub key[32]; // 256 byte

    V value[32]; // 256 byte

    // the 32nd element is next
    //unsigned long long *next;
};

template<typename V>
struct SlabData<char, V> {

    typedef unsigned long long KSub;


    union {
        int ilock;
        char p[128];
    }; // 128 bytes

    KSub key[32]; // 256 byte

    V value[32]; // 256 byte

    // the 32nd element is next
    //unsigned long long *next;
};

template<typename V>
struct SlabData<short, V> {

    typedef unsigned long long KSub;

    union {
        int ilock;
        char p[128];
    }; // 128 bytes

    KSub key[32]; // 256 byte

    V value[32]; // 256 byte

    // the 32nd element is next
    //unsigned long long *next;
};

template<typename V>
struct SlabData<unsigned, V> {

    typedef unsigned long long KSub;

    union {
        int ilock;
        char p[128];
    }; // 128 bytes

    KSub key[32]; // 256 byte

    V value[32]; // 256 byte

    // the 32nd element is next
    //unsigned long long *next;
};

template<typename K, typename V>
struct MemoryBlock {
    MemoryBlock() : bitmap(~0u), slab(nullptr) {
    }

    unsigned long long bitmap;
    SlabData<K, V> *slab;// 64 slabs
};

template<typename K, typename V>
struct SuperBlock {
    MemoryBlock<K, V> *memblocks;// 32 memblocks
};

template<typename K, typename V>
struct WarpAllocCtx {
    WarpAllocCtx() : blocks(nullptr) {
        // creates default context
    }

    SuperBlock<K, V> *blocks;
    // there should be a block per warp ie threadsPerBlock * blocks / 32 superblocks
};

template<typename K, typename V>
std::ostream &operator<<(std::ostream &output, const SlabData<K, V> &s) {
    output << s.keyValue;
    return output;
}

template<typename K, typename V>
WarpAllocCtx<K, V>
setupWarpAllocCtxGroup(groupallocator::GroupAllocator &gAlloc, int threadsPerBlock, int blocks, int gpuid = 0,
                       cudaStream_t stream = cudaStreamDefault) {
    gpuErrchk(cudaSetDevice(gpuid))
    WarpAllocCtx<K, V> actx;
    gAlloc.allocate(&actx.blocks, (size_t) ceil(threadsPerBlock * blocks / 32.0) * sizeof(SuperBlock<K, V>), false);
    for (size_t i = 0; i < (size_t) ceil(threadsPerBlock * blocks / 32.0); i++) {
        gAlloc.allocate(&(actx.blocks[i].memblocks), sizeof(MemoryBlock<K, V>) * 32, false);
        for (int j = 0; j < 32; j++) {
            actx.blocks[i].memblocks[j].bitmap = ~0ull;
            gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab), sizeof(SlabData<K, V>) * 64, false);
            for (int k = 0; k < 64; k++) {
                //gAlloc.allocate(&(actx.blocks[i].memblocks[j].slab[k].keyValue), sizeof(unsigned long long) * 32, false);
                actx.blocks[i].memblocks[j].slab[k].ilock = 0;
                for (int w = 0; w < 32; w++) {
                    actx.blocks[i].memblocks[j].slab[k].key[w] = EMPTY<K>::value;
                    actx.blocks[i].memblocks[j].slab[k].value[j] = EMPTY<V>::value;
                }
            }
        }
    }
    gAlloc.moveToDevice(gpuid, stream);
    gpuErrchk(cudaDeviceSynchronize())
    std::cerr << "Size allocated for warp alloc: "
              << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
              << std::endl;
    return actx;
}

template<typename K, typename V>
struct SlabCtx {
    SlabCtx() : slabs(nullptr), num_of_buckets(0) {}

    volatile SlabData<K, V> **slabs;
    unsigned num_of_buckets;
};

//template<typename T>
//__host__ __device__ unsigned hash(T src_key, unsigned num_of_buckets);

/**
 * There is a barrier after this locking
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
__forceinline__ __device__ void
LockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
         volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        while (atomicCAS(ilock, 0, -1) != 0);
    }
    __syncwarp();

}

/**
 * There is a barrier after this locking
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
__forceinline__ __device__ void
SharedLockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
               volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        while (true) {
            auto pred = *ilock;

            if (pred != -1 && atomicCAS(ilock, pred, pred + 1) == pred) {
                break;
            }
        }
    }
    __syncwarp();
}


/**
 * Note there is no barrier before or after, pay attention to reordering
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
__forceinline__ __device__ void
UnlockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
           volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *)  &(slabs[src_bucket]->ilock);
        atomicExch(ilock, 0);
    }

}

/**
 * Note there is no barrier before or after, pay attention to reordering
 * @param next
 * @param src_bucket
 * @param laneId
 * @param slabs
 */
template<typename K, typename V>
__forceinline__ __device__ void
SharedUnlockSlab(const unsigned long long &next, const unsigned &src_bucket, const unsigned &laneId,
                 volatile SlabData<K, V> **slabs) {

    if (laneId == 0) {
        auto ilock = (int *) &(slabs[src_bucket]->ilock);
        atomicAdd(ilock, -1);
    }

}


template<typename K, typename V>
__forceinline__ __device__ typename SlabData<K, V>::KSub
ReadSlabKey(const unsigned long long &next, const unsigned &src_bucket,
            const unsigned laneId, volatile SlabData<K, V> **slabs) {
    static_assert(sizeof(typename SlabData<K, V>::KSub) >= sizeof(void*), "Need to be able to substitute pointers for values");
    return next == BASE_SLAB ? slabs[src_bucket]->key[laneId] : ((SlabData<K, V> *) next)->key[laneId];
}

template<typename K, typename V>
__forceinline__ __device__ V
ReadSlabValue(const unsigned long long &next, const unsigned &src_bucket,
              const unsigned laneId, volatile SlabData<K, V> **slabs) {
    return (next == BASE_SLAB ? slabs[src_bucket]->value[laneId] : ((SlabData<K, V> *) next)->value[laneId]);
}


template<typename K, typename V>
__forceinline__ __device__ volatile typename SlabData<K, V>::KSub *
SlabAddressKey(const unsigned long long &next, const unsigned &src_bucket,
               const unsigned laneId, volatile SlabData<K, V> **slabs,
               unsigned num_of_buckets) {
    return (volatile typename SlabData<K, V>::KSub *) ((next == BASE_SLAB ? slabs[src_bucket]->key : ((SlabData<K, V> *) next)->key) + laneId);
}

template<typename K, typename V>
__forceinline__ __device__ volatile V *
SlabAddressValue(const unsigned long long &next, const unsigned &src_bucket,
                 const unsigned laneId, volatile SlabData<K, V> **slabs,
                 unsigned num_of_buckets) {
    return (next == BASE_SLAB ? slabs[src_bucket]->value : ((SlabData<K, V> *) next)->value) + laneId;
}

// just doing parallel shared-nothing allocation
template<typename K, typename V>
__forceinline__ __device__ unsigned long long warp_allocate(WarpAllocCtx<K, V> ctx) {

    const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
    const unsigned laneId = threadIdx.x & 0x1Fu;
    if (ctx.blocks == nullptr) {
        return 0;
    }

    MemoryBlock<K, V> *blocks = ctx.blocks[warpIdx].memblocks;
    unsigned bitmap = blocks[laneId].bitmap;
    int index = __ffs((int) bitmap) - 1;
    int ballotThread = __ffs((int) __ballot_sync(~0u, (index != -1))) - 1;
    if (ballotThread == -1) {
        if(laneId == 0)
            printf("Ran out of memory\n");
        __threadfence_system();
        __syncwarp();
        __trap();
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

template<typename K, typename V>
__forceinline__ __device__ void deallocate(WarpAllocCtx<K, V> ctx, unsigned long long l) {

    const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
    const unsigned laneId = threadIdx.x & 0x1Fu;
    if (ctx.blocks == nullptr) {
        return;
    }

    MemoryBlock<K, V> *blocks = ctx.blocks[warpIdx].memblocks;
    if ((unsigned long long) blocks[laneId].slab <= l && (unsigned long long) (blocks[laneId].slab + 32) > l) {
        unsigned diff = l - (unsigned long long) blocks[laneId].slab;
        unsigned idx = diff / sizeof(SlabData<K, V>);
        blocks[laneId].bitmap = blocks[laneId].bitmap | (1u << idx);
    }
}


// manually inlined
template<typename K, typename V>
__forceinline__ __device__ void warp_operation_search(bool &is_active, const K &myKey,
                                                      V &myValue, const unsigned &modhash,
                                                      volatile SlabData<K, V> **__restrict__ slabs,
                                                      unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    const auto threadKey = (unsigned long long) myKey;

    unsigned last_work_queue = work_queue;

    while (work_queue != 0) {

        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

        unsigned src_lane = __ffs((int) work_queue) - 1;
        unsigned long long src_key = __shfl_sync(~0u, threadKey, (int) src_lane);
        unsigned src_bucket = __shfl_sync(~0u, modhash, (int) src_lane);
        SharedLockSlab(next, src_bucket, laneId, slabs);
        unsigned long long read_key =
                (unsigned long long) ReadSlabKey(next, src_bucket, laneId, slabs);

        auto masked_ballot = (unsigned) (__ballot_sync(~0u, compare((K) read_key, (K) src_key) == 0) & VALID_KEY_MASK);

        if (masked_ballot != 0) {
            V read_value = ReadSlabValue(next, src_bucket, laneId, slabs);

            auto found_lane = (unsigned) (__ffs(masked_ballot) - 1);
            auto found_value = (V) __shfl_sync(~0u, (unsigned long long) read_value, found_lane);
            if (laneId == src_lane) {
                myValue = found_value;
                is_active = false;
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, (unsigned long long) read_key, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            } else {
                SharedUnlockSlab(next, src_bucket, laneId, slabs);
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;


        work_queue = __ballot_sync(~0u, is_active);

        SharedUnlockSlab(next, src_bucket, laneId, slabs);
    }
}

/**
 * Returns value when removed or empty on removal
 * @tparam K
 * @tparam V
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash
 * @param slabs
 * @param num_of_buckets
 */
template<typename K, typename V>
__forceinline__ __device__ void
warp_operation_delete(bool &is_active, const K &myKey,
                      V &myValue, const unsigned &modhash,
                      volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = work_queue;

    while (work_queue != 0) {
        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
        auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
        auto src_key = (K) __shfl_sync(~0u, (unsigned long long) myKey, src_lane);
        unsigned src_bucket = __shfl_sync(~0u, modhash, (int) src_lane);
        LockSlab(next, src_bucket, laneId, slabs);

        K read_key =
                ReadSlabKey(next, src_bucket, laneId, slabs);

        auto masked_ballot = (unsigned) (__ballot_sync(~0u, compare(read_key, src_key) == 0) & VALID_KEY_MASK);

        if (masked_ballot != 0) {

            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                *(SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets)) = EMPTY<K>::value;
                is_active = false;
                myValue = ReadSlabValue(next, src_bucket, dest_lane, slabs);
                //success = true;
                __threadfence();
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, (unsigned long long) read_key, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                is_active = false;
                myValue = EMPTY<V>::value;
                //success = false;
            } else {
                UnlockSlab(next, src_bucket, laneId, slabs);
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
        UnlockSlab(next, src_bucket, laneId, slabs);
    }
}

template<typename K, typename V>
__forceinline__ __device__ void
warp_operation_replace(bool &is_active, const K &myKey,
                       V &myValue, const unsigned &modhash,
                       volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    unsigned long long next = BASE_SLAB;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = 0;

    bool foundEmptyNext = false;
    unsigned long long empty_next = BASE_SLAB;

    while (work_queue != 0) {
        //bool unlocked = false;

        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;

        //if (laneId == 0)
        //    printf("%ld %d %d\n", next, blockIdx.x, threadIdx.x);
        auto src_lane = (unsigned) (__ffs((int) work_queue) - 1);
        auto src_key = (K) __shfl_sync(~0u, (unsigned long long) myKey, src_lane);
        unsigned src_bucket = __shfl_sync(~0u, modhash, (int) src_lane);

        if(work_queue != last_work_queue){
            foundEmptyNext = false;
            LockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }

        // if (laneId == 0)
        //  printf("src_lane %d from %d: %d -> %d\n", src_lane, work_queue, src_key, src_bucket);

        //if(laneId == 0){
        //    printf("Locked src_bucket %d\n", src_bucket);
        //}

        K read_key =
                ReadSlabKey(next, src_bucket, laneId, slabs);


        bool to_share = (compare(read_key, src_key) == 0);
        int masked_ballot = (int) (__ballot_sync(~0u, to_share) & VALID_KEY_MASK);

        if(!foundEmptyNext && read_key == EMPTY<K>::value){
            foundEmptyNext = true;
            empty_next = next;
        }

        if (masked_ballot != 0) {
            if (src_lane == laneId) {
                unsigned dest_lane = (unsigned) __ffs(masked_ballot) - 1;
                volatile K *addrKey =
                        (volatile K*) SlabAddressKey(next, src_bucket, dest_lane, slabs, num_of_buckets);
                volatile V *addrValue =
                        SlabAddressValue(next, src_bucket, dest_lane, slabs, num_of_buckets);
                V tmpValue = EMPTY<V>::value;
                if (*addrKey == EMPTY<K>::value) {
                    *addrKey = myKey;
                } else {
                    tmpValue = *addrValue;
                }
                *addrValue = myValue;
                myValue = tmpValue;
                __threadfence_system();
                is_active = false;
            }
        } else {
            unsigned long long next_ptr = __shfl_sync(~0u, (unsigned long long) read_key, ADDRESS_LANE - 1);
            if (next_ptr == 0) {
                __threadfence_system();
                masked_ballot = (int) (__ballot_sync(~0u, foundEmptyNext) & VALID_KEY_MASK);
                if (masked_ballot != 0) {
                    unsigned dest_lane = (unsigned) __ffs(masked_ballot) - 1;
                    unsigned new_empty_next = __shfl_sync(~0u, empty_next, (int) dest_lane);
                    if (src_lane == laneId) {
                        volatile K *addrKey =
                                (volatile K*) SlabAddressKey(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                        volatile V *addrValue =
                                SlabAddressValue(new_empty_next, src_bucket, dest_lane, slabs, num_of_buckets);
                        V tmpValue = EMPTY<V>::value;
                        if (*addrKey == EMPTY<K>::value) {
                            *addrKey = src_key;
                        } else {
                            tmpValue = *addrValue;
                        }
                        *addrValue = myValue;
                        myValue = tmpValue;
                        __threadfence_system();
                        is_active = false;
                    }
                } else {
                    unsigned long long new_slab_ptr = warp_allocate(ctx);
                    if (laneId == ADDRESS_LANE - 1) {
                        auto *slabAddr = SlabAddressKey(next, src_bucket, ADDRESS_LANE - 1,
                                                                               slabs, num_of_buckets);
                        *((unsigned long long*)slabAddr) = new_slab_ptr;
                        __threadfence_system();
                    }
                    next = new_slab_ptr;
                }
            } else {
                next = next_ptr;
            }
        }
        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);

        if (work_queue != last_work_queue) {
            UnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
        }
    }
}

template<typename K, typename V>
SlabCtx<K, V> *setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid = 0,
                          cudaStream_t stream = cudaStreamDefault) {

    gpuErrchk(cudaSetDevice(gpuid));

    auto sctx = new SlabCtx<K, V>();
    sctx->num_of_buckets = size;
    std::cerr << "Size of index is " << size << std::endl;
    std::cerr << "Each slab is " << sizeof(SlabData<K, V>) << "B" << std::endl;


    gAlloc.allocate(&(sctx->slabs), sizeof(void *) * sctx->num_of_buckets, false);

    for (int i = 0; i < sctx->num_of_buckets; i++) {
        gAlloc.allocate(&(sctx->slabs[i]), sizeof(SlabData<K, V>), false);

        static_assert(sizeof(sctx->slabs[i]->key[0]) >= sizeof(void *),
                      "The key size needs to be greater or equal to the size of a memory address");

        //gAlloc.allocate((unsigned long long **) &(sctx->slabs[i][k].keyValue), sizeof(unsigned long long) * 32, false);

        memset((void *) (sctx->slabs[i]), 0, sizeof(SlabData<K, V>));

        for (int j = 0; j < 31; j++) {
            sctx->slabs[i]->key[j] = EMPTY<K>::value;// EMPTY_PAIR;
        }

        void **ptrs = (void **) sctx->slabs[i]->key;

        ptrs[31] = nullptr;// EMPTY_POINTER;

        for (int j = 0; j < 32; j++) {
            sctx->slabs[i]->value[j] = EMPTY<V>::value;
        }

    }

    gAlloc.moveToDevice(gpuid, stream);

    gpuErrchk(cudaDeviceSynchronize())

    std::cerr << "Size allocated for Slab: "
              << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
              << std::endl;
    return sctx;
}


#endif
