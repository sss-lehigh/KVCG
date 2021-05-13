/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>

#include "stdrequestHandler.cuh"

#ifndef SLAB_SLAB_CUH
#define SLAB_SLAB_CUH

class Slab {
public:
  virtual void batch(unsigned *keys, unsigned *values, unsigned *requests) = 0;

protected:
  SlabCtx *slab;
  unsigned *batchKeys;
  unsigned *batchValues;
  int *batchRequests;
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

  void batch(unsigned *keys, unsigned *values, unsigned *requests);

private:
  groupallocator::GroupAllocator *slabGAlloc;
  groupallocator::GroupAllocator *allocGAlloc;
  groupallocator::GroupAllocator *bufferGAlloc;
};

#endif // SLAB_SLAB_CUH
