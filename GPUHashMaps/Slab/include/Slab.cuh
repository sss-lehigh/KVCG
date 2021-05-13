/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

#include "stdrequestHandler.cuh"

#ifndef SLAB_SLAB_CUH
#define SLAB_SLAB_CUH

#define BLOCKS 20 // 134
#define THREADS_PER_BLOCK 512

#define USE_HOST

class Slab {
public:
    virtual void batch(unsigned *keys, unsigned *values, unsigned *requests) = 0;

protected:
    SlabCtx *slab;
    WarpAllocCtx ctx;
    cudaStream_t *_stream;
    int _gpu;
    int mapSize;
    std::thread *handler;
    std::atomic<bool> *signal;
    std::mutex mtx;
    int position;
};

class SlabUnified : public Slab {
public:
    SlabUnified(int size);

    SlabUnified(int size, int gpu);

    SlabUnified(int size, cudaStream_t *stream);

    SlabUnified(int size, int gpu, cudaStream_t *stream);

    ~SlabUnified();

    virtual void batch(unsigned *keys, unsigned *values, unsigned *requests);

    std::tuple<float, float, float> batch_bench(unsigned *keys, unsigned *values, unsigned *requests);

    std::tuple<float, float, float> batch_get(unsigned *keys, unsigned *values, unsigned *requests);

    std::tuple<float, float, float> batch_insert(unsigned *keys, unsigned *values, unsigned *requests);

    std::tuple<float, float, float> batch_delete(unsigned *keys, unsigned *values, unsigned *requests);

private:
    unsigned *batchKeys;
    unsigned *batchValues;
    int *batchRequests;

#ifdef USE_HOST
    unsigned *batchKeys_h;
    unsigned *batchValues_h;
    int *batchRequests_h;
#endif

    groupallocator::GroupAllocator *slabGAlloc;
    groupallocator::GroupAllocator *allocGAlloc;
    groupallocator::GroupAllocator *bufferGAlloc;
};

#endif // SLAB_SLAB_CUH
