#include <Operations.cuh>
#include <Slab.cuh>

#define BLOCKS 134
#define THREADS_PER_BLOCK 512

SlabUnified::SlabUnified(int size) : SlabUnified(size, 0, nullptr) {}

SlabUnified::SlabUnified(int size, int gpu)
    : SlabUnified(size, gpu, nullptr) {}

SlabUnified::SlabUnified(int size, cudaStream_t *stream)
    : SlabUnified(size, 0, stream) {}

SlabUnified::SlabUnified(int size, int gpu, cudaStream_t *stream) {
  gpuErrchk(cudaSetDevice(gpu));

  if(stream == nullptr){
      _stream = new cudaStream_t();
      *_stream = cudaStreamDefault;
  } else {
      _stream = stream;
  }

  slabGAlloc = new groupallocator::GroupAllocator(0, 4096);
  allocGAlloc = new groupallocator::GroupAllocator(1, 4096);
  bufferGAlloc = new groupallocator::GroupAllocator(2, 4096);
  this->slab = setUpGroup(*slabGAlloc, size, 1, gpu, *_stream);
  bufferGAlloc->allocate(&batchKeys,
                         BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), false);
  bufferGAlloc->allocate(&batchValues,
                         BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), false);
  bufferGAlloc->allocate(&batchRequests,
                         BLOCKS * THREADS_PER_BLOCK * sizeof(int), false);
  this->ctx = setupWarpAllocCtxGroup(*allocGAlloc, THREADS_PER_BLOCK, BLOCKS,
                                     gpu, *_stream);

  _gpu = gpu;
  mapSize = size;
}

SlabUnified::~SlabUnified() {
  delete slabGAlloc;
  delete allocGAlloc;
  delete bufferGAlloc;
}

void SlabUnified::batch(unsigned *keys, unsigned *values, unsigned *requests) {

  gpuErrchk(cudaSetDevice(_gpu));

  for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
    batchKeys[i] = keys[i];
    batchValues[i] = values[i];
    batchRequests[i] = requests[i];
  }

  bufferGAlloc->moveToDevice(_gpu, *_stream);
  gpuErrchk(cudaStreamSynchronize(*_stream));

  requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, *_stream>>>(
      slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests,
      ctx);
  gpuErrchk(cudaStreamSynchronize(*_stream));
  bufferGAlloc->moveToDevice(cudaCpuDeviceId, *_stream);
  gpuErrchk(cudaStreamSynchronize(*_stream));

  for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
    keys[i] = batchKeys[i];
    values[i] = batchValues[i];
    requests[i] = batchRequests[i];
  }
}
