//
// Created by depaulsmiller on 7/24/20.
//

#include "Operations.cuh"

#ifndef GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
#define GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH

const int REQUEST_INSERT = 1;
const int REQUEST_GET = 2;
const int REQUEST_REMOVE = 3;
const int REQUEST_EMPTY = 0;

__global__ void requestHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request, WarpAllocCtx ctx = WarpAllocCtx()) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, slabs, num_of_buckets);

    activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, slabs,
                           num_of_buckets, ctx);
    activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}

__global__ void getHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}

__global__ void insertHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request, WarpAllocCtx ctx = WarpAllocCtx()) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];

    bool activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, slabs,
                           num_of_buckets, ctx);
    myValue[tid] = value;
}

__global__ void deleteHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}


#endif//GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
