#include <Operations.cuh>
#include <Slab.cuh>

#define PAGE_SIZE 4096

SlabUnified::SlabUnified(int size) : SlabUnified(size, 0, nullptr) {}

SlabUnified::SlabUnified(int size, int gpu)
        : SlabUnified(size, gpu, nullptr) {}

SlabUnified::SlabUnified(int size, cudaStream_t *stream)
        : SlabUnified(size, 0, stream) {}

SlabUnified::SlabUnified(int size, int gpu, cudaStream_t *stream) {
    gpuErrchk(cudaSetDevice(gpu));
    slabGAlloc = new groupallocator::GroupAllocator(0, PAGE_SIZE);
    allocGAlloc = new groupallocator::GroupAllocator(1, PAGE_SIZE);
    bufferGAlloc = new groupallocator::GroupAllocator(2, 4096);
    this->slab = setUpGroup(*slabGAlloc, size, 1, gpu, (stream == nullptr ? cudaStreamDefault : *stream));

#ifdef USE_HOST
    gpuErrchk(cudaMalloc(&batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
    gpuErrchk(cudaMallocHost(&batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int)));

#else
    bufferGAlloc->allocate(&batchKeys,
                           BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), false);
    bufferGAlloc->allocate(&batchValues,
                           BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), false);
    bufferGAlloc->allocate(&batchRequests,
                           BLOCKS * THREADS_PER_BLOCK * sizeof(int), false);
#endif

    this->ctx = setupWarpAllocCtxGroup(*allocGAlloc, THREADS_PER_BLOCK, BLOCKS,
                                       gpu, (stream == nullptr ? cudaStreamDefault : *stream));
    _stream = stream;
    _gpu = gpu;
    mapSize = size;
}

SlabUnified::~SlabUnified() {
    delete slabGAlloc;
    delete allocGAlloc;
    delete bufferGAlloc;

#ifdef USE_HOST
    gpuErrchk(cudaFree(batchKeys));
    gpuErrchk(cudaFree(batchValues));
    gpuErrchk(cudaFree(batchRequests));
    gpuErrchk(cudaFreeHost(batchKeys_h));
    gpuErrchk(cudaFreeHost(batchValues_h));
    gpuErrchk(cudaFreeHost(batchRequests_h));

#endif

}

void SlabUnified::batch(unsigned *keys, unsigned *values, unsigned *requests) {

    gpuErrchk(cudaSetDevice(_gpu));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys_h[i] = keys[i];
        batchValues_h[i] = values[i];
        batchRequests_h[i] = requests[i];
    }

    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    gpuErrchk(cudaMemcpyAsync(batchKeys, batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues, batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests, batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

#else
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys[i] = keys[i];
        batchValues[i] = values[i];
        batchRequests[i] = requests[i];
    }

    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    bufferGAlloc->moveToDevice(_gpu, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
#endif

    requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
            slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests,
            ctx);
    gpuErrchk(cudaStreamSynchronize(stream));

#ifdef USE_HOST

    gpuErrchk(cudaMemcpyAsync(batchKeys_h, batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues_h, batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests_h, batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost, stream));

    gpuErrchk(cudaStreamSynchronize(stream));

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys_h[i];
        values[i] = batchValues_h[i];
        requests[i] = batchRequests_h[i];
    }

#else
    bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys[i];
        values[i] = batchValues[i];
        requests[i] = batchRequests[i];
    }
#endif
}

std::tuple<float, float, float> SlabUnified::batch_bench(unsigned *keys, unsigned *values, unsigned *requests) {
    gpuErrchk(cudaSetDevice(_gpu));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys_h[i] = keys[i];
        batchValues_h[i] = values[i];
        batchRequests_h[i] = requests[i];
    }

#else

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys[i] = keys[i];
        batchValues[i] = values[i];
        batchRequests[i] = requests[i];
    }

#endif
    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    cudaEvent_t start_m, start_k, end_k, end_m;

    gpuErrchk(cudaEventCreate(&start_m));
    gpuErrchk(cudaEventCreate(&start_k));

    gpuErrchk(cudaEventCreate(&end_k));
    gpuErrchk(cudaEventCreate(&end_m));

    cudaEventRecord(start_m, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys, batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues, batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests, batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice, stream));
#else
    bufferGAlloc->moveToDevice(_gpu, stream);
#endif
    cudaEventRecord(start_k, stream);
    requestHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
            slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests,
            ctx);
    cudaEventRecord(end_k, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys_h, batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues_h, batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests_h, batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost, stream));
#else
    bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);
#endif
    cudaEventRecord(end_m, stream);

    gpuErrchk(cudaEventSynchronize(end_m));

    float ms1, ms2, ms3;
    gpuErrchk(cudaEventElapsedTime(&ms1, start_k, end_k));
    gpuErrchk(cudaEventElapsedTime(&ms2, start_m, start_k));
    gpuErrchk(cudaEventElapsedTime(&ms3, end_k, end_m));


    gpuErrchk(cudaEventDestroy(start_m));
    gpuErrchk(cudaEventDestroy(start_k));

    gpuErrchk(cudaEventDestroy(end_k));
    gpuErrchk(cudaEventDestroy(end_m));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys_h[i];
        values[i] = batchValues_h[i];
        requests[i] = batchRequests_h[i];
    }
#else
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys[i];
        values[i] = batchValues[i];
        requests[i] = batchRequests[i];
    }
#endif
    return {ms1, ms2, ms3};
}

std::tuple<float, float, float> SlabUnified::batch_get(unsigned *keys, unsigned *values, unsigned *requests) {
    gpuErrchk(cudaSetDevice(_gpu));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys_h[i] = keys[i];
        batchValues_h[i] = values[i];
        batchRequests_h[i] = requests[i];
    }

#else

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys[i] = keys[i];
        batchValues[i] = values[i];
        batchRequests[i] = requests[i];
    }

#endif
    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    cudaEvent_t start_m, start_k, end_k, end_m;

    gpuErrchk(cudaEventCreate(&start_m));
    gpuErrchk(cudaEventCreate(&start_k));

    gpuErrchk(cudaEventCreate(&end_k));
    gpuErrchk(cudaEventCreate(&end_m));

    cudaEventRecord(start_m, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys, batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues, batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests, batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice, stream));
#else
    bufferGAlloc->moveToDevice(_gpu, stream);
#endif
    cudaEventRecord(start_k, stream);
    getHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
            slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests);
    cudaEventRecord(end_k, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys_h, batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues_h, batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests_h, batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost, stream));
#else
    bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);
#endif
    cudaEventRecord(end_m, stream);

    gpuErrchk(cudaEventSynchronize(end_m));

    float ms1, ms2, ms3;
    gpuErrchk(cudaEventElapsedTime(&ms1, start_k, end_k));
    gpuErrchk(cudaEventElapsedTime(&ms2, start_m, start_k));
    gpuErrchk(cudaEventElapsedTime(&ms3, end_k, end_m));


    gpuErrchk(cudaEventDestroy(start_m));
    gpuErrchk(cudaEventDestroy(start_k));

    gpuErrchk(cudaEventDestroy(end_k));
    gpuErrchk(cudaEventDestroy(end_m));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys_h[i];
        values[i] = batchValues_h[i];
        requests[i] = batchRequests_h[i];
    }
#else
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys[i];
        values[i] = batchValues[i];
        requests[i] = batchRequests[i];
    }
#endif
    return {ms1, ms2, ms3};
}

std::tuple<float, float, float> SlabUnified::batch_insert(unsigned *keys, unsigned *values, unsigned *requests) {
    gpuErrchk(cudaSetDevice(_gpu));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys_h[i] = keys[i];
        batchValues_h[i] = values[i];
        batchRequests_h[i] = requests[i];
    }

#else

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys[i] = keys[i];
        batchValues[i] = values[i];
        batchRequests[i] = requests[i];
    }

#endif
    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    cudaEvent_t start_m, start_k, end_k, end_m;

    gpuErrchk(cudaEventCreate(&start_m));
    gpuErrchk(cudaEventCreate(&start_k));

    gpuErrchk(cudaEventCreate(&end_k));
    gpuErrchk(cudaEventCreate(&end_m));

    cudaEventRecord(start_m, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys, batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues, batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests, batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice, stream));
#else
    bufferGAlloc->moveToDevice(_gpu, stream);
#endif
    cudaEventRecord(start_k, stream);
    insertHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
            slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests,
            ctx);
    cudaEventRecord(end_k, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys_h, batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues_h, batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests_h, batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost, stream));
#else
    bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);
#endif
    cudaEventRecord(end_m, stream);

    gpuErrchk(cudaEventSynchronize(end_m));

    float ms1, ms2, ms3;
    gpuErrchk(cudaEventElapsedTime(&ms1, start_k, end_k));
    gpuErrchk(cudaEventElapsedTime(&ms2, start_m, start_k));
    gpuErrchk(cudaEventElapsedTime(&ms3, end_k, end_m));


    gpuErrchk(cudaEventDestroy(start_m));
    gpuErrchk(cudaEventDestroy(start_k));

    gpuErrchk(cudaEventDestroy(end_k));
    gpuErrchk(cudaEventDestroy(end_m));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys_h[i];
        values[i] = batchValues_h[i];
        requests[i] = batchRequests_h[i];
    }
#else
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys[i];
        values[i] = batchValues[i];
        requests[i] = batchRequests[i];
    }
#endif
    return {ms1, ms2, ms3};
}

std::tuple<float, float, float> SlabUnified::batch_delete(unsigned *keys, unsigned *values, unsigned *requests) {
    gpuErrchk(cudaSetDevice(_gpu));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys_h[i] = keys[i];
        batchValues_h[i] = values[i];
        batchRequests_h[i] = requests[i];
    }

#else

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        batchKeys[i] = keys[i];
        batchValues[i] = values[i];
        batchRequests[i] = requests[i];
    }

#endif
    auto stream = (_stream == nullptr ? cudaStreamDefault : *_stream);

    cudaEvent_t start_m, start_k, end_k, end_m;

    gpuErrchk(cudaEventCreate(&start_m));
    gpuErrchk(cudaEventCreate(&start_k));

    gpuErrchk(cudaEventCreate(&end_k));
    gpuErrchk(cudaEventCreate(&end_m));

    cudaEventRecord(start_m, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys, batchKeys_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues, batchValues_h, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests, batchRequests_h, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyHostToDevice, stream));
#else
    bufferGAlloc->moveToDevice(_gpu, stream);
#endif
    cudaEventRecord(start_k, stream);
    deleteHandler<<<BLOCKS, THREADS_PER_BLOCK, 0, stream>>>(
            slab->slabs, slab->num_of_buckets, batchKeys, batchValues, batchRequests);
    cudaEventRecord(end_k, stream);
#ifdef USE_HOST
    gpuErrchk(cudaMemcpyAsync(batchKeys_h, batchKeys, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchValues_h, batchValues, BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
    gpuErrchk(cudaMemcpyAsync(batchRequests_h, batchRequests, BLOCKS * THREADS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost, stream));
#else
    bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);
#endif
    cudaEventRecord(end_m, stream);

    gpuErrchk(cudaEventSynchronize(end_m));

    float ms1, ms2, ms3;
    gpuErrchk(cudaEventElapsedTime(&ms1, start_k, end_k));
    gpuErrchk(cudaEventElapsedTime(&ms2, start_m, start_k));
    gpuErrchk(cudaEventElapsedTime(&ms3, end_k, end_m));


    gpuErrchk(cudaEventDestroy(start_m));
    gpuErrchk(cudaEventDestroy(start_k));

    gpuErrchk(cudaEventDestroy(end_k));
    gpuErrchk(cudaEventDestroy(end_m));

#ifdef USE_HOST
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys_h[i];
        values[i] = batchValues_h[i];
        requests[i] = batchRequests_h[i];
    }
#else
    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = batchKeys[i];
        values[i] = batchValues[i];
        requests[i] = batchRequests[i];
    }
#endif
    return {ms1, ms2, ms3};
}
