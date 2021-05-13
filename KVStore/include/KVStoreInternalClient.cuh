//
// Created by depaulsmiller on 8/28/20.
//

#include "KVStore.cuh"
#include <PartitionedSlabUnified.cuh>
#include <functional>
#include <chrono>
#include <tbb/concurrent_queue.h>

#ifndef KVGPU_KVSTOREINTERNALCLIENT_CUH
#define KVGPU_KVSTOREINTERNALCLIENT_CUH

int LOAD_THRESHOLD = BLOCKS * 10000;

template<typename K, typename V>
struct RequestWrapper {
    K key;
    V value;
    //std::shared_ptr<SharedResult<std::pair<bool, V>>> getPromise;
    //std::shared_ptr<SharedResult<bool>> otherPromise;
    unsigned requestInteger;
};

struct block_t {
    std::condition_variable cond;
    std::mutex mtx;
    int count;
    int crossing;

    block_t(int n) : count(n), crossing(0) {}

    void wait() {
        std::unique_lock<std::mutex> ulock(mtx);

        crossing++;

        // wait here

        cond.wait(ulock);
    }

    bool threads_blocked() {
        std::unique_lock<std::mutex> ulock(mtx);
        return crossing == count;
    }

    void wake() {
        std::unique_lock<std::mutex> ulock(mtx);
        cond.notify_all();
        crossing = 0;
    }

};

template<typename K, typename V>
void schedule_for_batch_helper(K *&keys, V *&values, unsigned *requests, unsigned *hashes,
                               std::unique_lock<kvgpu::mutex> *locks, unsigned *&correspondence,
                               int &index, const int &hash, const RequestWrapper<K, V> &req,
                               std::unique_lock<kvgpu::mutex> &&lock, int i) {

    keys[index] = req.key;
    values[index] = req.value;
    requests[index] = req.requestInteger;
    hashes[index] = hash;
    locks[index] = std::move(lock);
    correspondence[index] = i;
    index++;
}

/*
template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>, bool UseCache = true, bool UseGPU = true>
class KVStoreInternalClient {
public:
    KVStoreInternalClient(std::shared_ptr<Slab_t> s, std::shared_ptr<typename Cache<K, V>::type> c,
                          std::shared_ptr<M> m) : numslabs(s->numslabs),
                                                  slabs(s), cache(c), hits(0),
                                                  operations(0), start(std::chrono::high_resolution_clock::now()),
                                                  model(m) {

    }

    ~KVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        std::cerr << "This shouldnt be used now\n"; //TODO remove
        bool dontDoGPU = false;

        if (slabs->load >= LOAD_THRESHOLD) {
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (model->operator()(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = 0;//cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches; //0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key,
                            cache_batch_idx.second,
                            *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }

                    if (pair.first != nullptr)
                        pair.second.unlock();
                    times.push_back(std::chrono::high_resolution_clock::now());
                }


            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches2[i]->idx > 0) {
                    gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                    sizeForGPUBatches += gpu_batches2[i]->idx;
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches2[i]);
                } else {
                    delete gpu_batches2[i];
                }
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches2[i];
            }
            resBuf->retryGPU = true;
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<V> *log_values = cache->log_values;

        tbb::concurrent_vector<int> *tmp_log_requests = new tbb::concurrent_vector<int>(
                cache->getN() * cache->getSETS());
        tbb::concurrent_vector<unsigned> *tmp_log_hash = new tbb::concurrent_vector<unsigned>(
                cache->getN() * cache->getSETS());
        tbb::concurrent_vector<K> *tmp_log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        tbb::concurrent_vector<V> *tmp_log_values = new tbb::concurrent_vector<V>(cache->getN() * cache->getSETS());

        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = tmp_log_requests;
        cache->log_hash = tmp_log_hash;
        cache->log_keys = tmp_log_keys;
        cache->log_values = tmp_log_values;
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        //std::cerr << "Tmp size " << tmpSize << "\n";

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            for (int i = 0; i < numslabs; ++i) {
                gpu_batches[i] = new BatchData<K, V>(0, std::make_shared<ResultsBuffers<V>>(batchSizeUsed),
                                                     batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }
        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    M getModel() {
        return *model;
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};
*/

/**
 * K is the type of the Key
 * V is the type of the Value
 * M is the type of the Model
 * @tparam K
 * @tparam V
 * @tparam M
 */
template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>, bool UseCache = true, bool UseGPU = true>
class KVStoreInternalClient {
public:
    KVStoreInternalClient(std::shared_ptr<Slab_t> s,
                          std::shared_ptr<typename Cache<K, typename Slab_t::VType>::type> c, std::shared_ptr<M> m)
            : numslabs(
            UseGPU ? s->numslabs : 0), slabs(s), cache(c), hits(0),
              operations(0),
              start(std::chrono::high_resolution_clock::now()),
              model(m) {

    }

    ~KVStoreInternalClient() {}

    typedef RequestWrapper<K, typename Slab_t::VType> RW;

    /**
     * Performs the batch of operations given
     * @param req_vector
     */
    void batch(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
               std::shared_ptr<ResultsBuffers<V>> &resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        bool dontDoGPU = false;

        if (numslabs == 0 || slabs->load >= LOAD_THRESHOLD) {
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        //req_vector.size() % 512 == 0 &&
        assert(req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_correspondence;

        cache_batch_correspondence.reserve(req_vector.size());
        std::vector<BatchData<K, V> *> gpu_batches;

        gpu_batches.reserve(numslabs);

        int sizeForGPUBatches = route(req_vector, resBuf, cache_batch_correspondence, gpu_batches);

        sendGPUBatches(dontDoGPU, gpu_batches);

        cpuStage(req_vector, resBuf, times, sizeForGPUBatches, cache_batch_correspondence, dontDoGPU);

    }

    template<typename V1 = typename Slab_t::VType, std::enable_if_t<std::is_same<data_t *, V1>::value> * = nullptr>
    V1 handle_copy(std::pair<kvgpu::LockingPair<K, typename Slab_t::VType> *, kvgpu::sharedlocktype> &pair) {
        typename Slab_t::VType cpy = nullptr;
        if (pair.first->deleted == 0 && pair.first->value) {
            cpy = new data_t(pair.first->value->size);
            memcpy(cpy->data, pair.first->value->data, cpy->size);
        }
        return cpy;
    }

    template<typename V1 = typename Slab_t::VType, std::enable_if_t<!std::is_same<data_t *, V1>::value> * = nullptr>
    V1 handle_copy(std::pair<kvgpu::LockingPair<K, typename Slab_t::VType> *, kvgpu::sharedlocktype> &pair) {
        std::cerr << "Why is this being used?" << std::endl;
        _exit(1);
        if (pair.first->deleted == 0 && pair.first->value) {
            return pair.first->value;
        }
        return EMPTY<V1>::value;
    }


    void cpuStage(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
                  std::shared_ptr<ResultsBuffers<V>> &resBuf,
                  std::vector<std::chrono::high_resolution_clock::time_point> &times, int sizeForGPUBatches,
                  std::vector<std::pair<int, unsigned>> &cache_batch_correspondence, bool dontDoGPU) {
        std::vector<BatchData<K, V> *> gpu_batches2;
        gpu_batches2.reserve(numslabs);
        setUpMissBatches(req_vector, resBuf, gpu_batches2);

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches;

        for (auto &cache_batch_idx : cache_batch_correspondence) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, typename Slab_t::VType> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        onMiss(cache_batch_idx, gpu_batches2, req_vector_elm, resBuf, responseLocationInResBuf, times);
                    } else {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";
                        resBuf->resultValues[responseLocationInResBuf] = handle_copy(pair);
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());
                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, typename Slab_t::VType> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            if (pair.first->valid == 1) {
                                resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                                pair.first->value = nullptr;
                            }

                            pair.first->deleted = 1;
                            pair.first->valid = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;
        forwardMissBatch(dontDoGPU, gpu_batches2, sizeForGPUBatches, resBuf);
        // send gpu_batch2

        operations += req_vector.size();
    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<typename Slab_t::VType> *log_values = cache->log_values;

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = new tbb::concurrent_vector<int>(cache->getN() * cache->getSETS());
        cache->log_hash = new tbb::concurrent_vector<unsigned>(cache->getN() * cache->getSETS());
        cache->log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        cache->log_values = new tbb::concurrent_vector<typename Slab_t::VType>(cache->getN() * cache->getSETS());
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

            for (int i = 0; i < numslabs; ++i) {
                std::shared_ptr<ResultsBuffers<V>> resBuf = std::make_shared<ResultsBuffers<V>>(
                        batchSizeUsed);
                gpu_batches[i] = new BatchData<K, V>(0,
                                                     resBuf,
                                                     batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }

        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    M getModel() {
        return *model;
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        if (numslabs == 0)
            return 0;
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        if (numslabs != 0)
            slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:

    // normal
    /// returns response location in resbuf
    template<bool UseCache_ = UseCache, typename std::enable_if_t<(UseCache_ && UseGPU)> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
          std::shared_ptr<ResultsBuffers<V>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, V> *> &gpu_batches) {

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches.push_back(new BatchData<K, V>(0, resBuf, req_vector.size()));
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (model->operator()(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }
        return sizeForGPUBatches;
    }

    // just cache
    template<bool UseCache_ = UseCache, typename std::enable_if_t<(UseCache_ && !UseGPU)> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
          std::shared_ptr<ResultsBuffers<V>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, V> *> &gpu_batches) {

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                cache_batch_corespondance.push_back({i, h});
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        return sizeForGPUBatches;
    }

    // GPU only
    template<bool UseCache_ = UseCache, typename std::enable_if_t<!UseCache_> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
          std::shared_ptr<ResultsBuffers<V>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, V> *> &gpu_batches) {

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches.push_back(new BatchData<K, V>(0, resBuf, req_vector.size()));
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                int gpuToUse = h % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = req.key;
                gpu_batches[gpuToUse]->values[idx] = req.value;
                gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                gpu_batches[gpuToUse]->hashes[idx] = h;

            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }
        return sizeForGPUBatches;
    }

    // gpu only and normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseGPU_> * = nullptr>
    inline void sendGPUBatches(bool &dontDoGPU, std::vector<BatchData<K, V> *> &gpu_batches) {
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
        }
    }

    // cpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseGPU_> * = nullptr>
    inline void sendGPUBatches(bool &dontDoGPU, std::vector<BatchData<K, V> *> &gpu_batches) {}

    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void setUpMissBatches(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
                                 std::shared_ptr<ResultsBuffers<V>> &resBuf,
                                 std::vector<BatchData<K, V> *> &gpu_batches2) {
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2.push_back(new BatchData<K, V>(0, resBuf, req_vector.size()));
        }
    }

    // gpu only or cpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<
            (UseCache && !UseGPU_) || (!UseCache && UseGPU_)> * = nullptr>
    inline void setUpMissBatches(std::vector<RequestWrapper<K, typename Slab_t::VType>> &req_vector,
                                 std::shared_ptr<ResultsBuffers<V>> &resBuf,
                                 std::vector<BatchData<K, V> *> &gpu_batches2) {}

    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, V> *> &gpu_batches2,
                       RequestWrapper<K, typename Slab_t::VType> &req_vector_elm,
                       std::shared_ptr<ResultsBuffers<V>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        int gpuToUse = cache_batch_idx.second % numslabs;
        int idx = gpu_batches2[gpuToUse]->idx;
        gpu_batches2[gpuToUse]->idx++;
        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
        gpu_batches2[gpuToUse]->handleInCache[idx] = true;
    }

    // cache only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && !UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, V> *> &gpu_batches2,
                       RequestWrapper<K, typename Slab_t::VType> &req_vector_elm,
                       std::shared_ptr<ResultsBuffers<V>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        hits.fetch_add(1, std::memory_order_relaxed);
        resBuf->resultValues[responseLocationInResBuf] = nullptr;
        asm volatile("":: : "memory");
        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
        responseLocationInResBuf++;
        times.push_back(std::chrono::high_resolution_clock::now());
    }

    // gpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseCache && UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, V> *> &gpu_batches2,
                       RequestWrapper<K, typename Slab_t::VType> &req_vector_elm,
                       std::shared_ptr<ResultsBuffers<V>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        // should never be called
        assert(false);
    }


    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, V> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<V>> &resBuf) {
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches2[i]->idx > 0) {
                    gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                    sizeForGPUBatches += gpu_batches2[i]->idx;
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches2[i]);
                } else {
                    delete gpu_batches2[i];
                }
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches2[i];
            }
            resBuf->retryGPU = true;
        }
    }

    // cache only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<(UseCache && !UseGPU_)> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, V> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<V>> &resBuf) {}

    // gpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseCache && UseGPU_> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, V> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<V>> &resBuf) {
        if (dontDoGPU) {
            resBuf->retryGPU = true;
        }
    }

    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, typename Slab_t::VType>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};


/*
template<typename K, typename M, typename Slab_t, bool UseCache, bool UseGPU>
class KVStoreInternalClient<K, data_t, M, Slab_t, UseCache, UseGPU> {
    // Slab_t should be some Slab<K, data_t*, M>
    static_assert(std::is_same<data_t *, typename Slab_t::VType>::value, "GPU Hashmap needs to handle GPU stuff");
public:
    KVStoreInternalClient(std::shared_ptr<Slab_t> s,
                          std::shared_ptr<typename Cache<K, data_t *>::type> c, std::shared_ptr<M> m) : numslabs(
            UseGPU ? s->numslabs : 0), slabs(s), cache(c), hits(0),
                                                                                                        operations(0),
                                                                                                        start(std::chrono::high_resolution_clock::now()),
                                                                                                        model(m) {

    }

    ~KVStoreInternalClient() {}

    typedef RequestWrapper<K, data_t *> RW;

    void batch(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        bool dontDoGPU = false;

        if (numslabs == 0 || slabs->load >= LOAD_THRESHOLD) {
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        //req_vector.size() % 512 == 0 &&
        assert(req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_correspondence;

        cache_batch_correspondence.reserve(req_vector.size());
        std::vector<BatchData<K, data_t> *> gpu_batches;

        gpu_batches.reserve(numslabs);

        int sizeForGPUBatches = route(req_vector, resBuf, cache_batch_correspondence, gpu_batches);

        sendGPUBatches(dontDoGPU, gpu_batches);

        cpuStage(req_vector, resBuf, times, sizeForGPUBatches, cache_batch_correspondence, dontDoGPU);

    }

    void cpuStage(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                  std::vector<std::chrono::high_resolution_clock::time_point> &times, int sizeForGPUBatches,
                  std::vector<std::pair<int, unsigned>>& cache_batch_correspondence, bool dontDoGPU){
        std::vector<BatchData<K, data_t> *> gpu_batches2;
        gpu_batches2.reserve(numslabs);
        setUpMissBatches(req_vector, resBuf, gpu_batches2);

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches;

        for (auto &cache_batch_idx : cache_batch_correspondence) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, data_t *> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        onMiss(cache_batch_idx, gpu_batches2, req_vector_elm, resBuf, responseLocationInResBuf, times);
                    } else {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";
                        data_t *cpy = nullptr;
                        if (pair.first->deleted == 0 && pair.first->value) {
                            cpy = new data_t(pair.first->value->size);
                            memcpy(cpy->data, pair.first->value->data, cpy->size);
                        }
                        resBuf->resultValues[responseLocationInResBuf] = cpy;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());
                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, data_t *> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            if (pair.first->valid == 1) {
                                resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                                pair.first->value = nullptr;
                            }

                            pair.first->deleted = 1;
                            pair.first->valid = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;
        forwardMissBatch(dontDoGPU, gpu_batches2, sizeForGPUBatches, resBuf);
        // send gpu_batch2

        operations += req_vector.size();
    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<data_t *> *log_values = cache->log_values;

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = new tbb::concurrent_vector<int>(cache->getN() * cache->getSETS());
        cache->log_hash = new tbb::concurrent_vector<unsigned>(cache->getN() * cache->getSETS());
        cache->log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        cache->log_values = new tbb::concurrent_vector<data_t *>(cache->getN() * cache->getSETS());
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            auto gpu_batches = std::vector<BatchData<K, data_t> *>(numslabs);

            for (int i = 0; i < numslabs; ++i) {
                std::shared_ptr<ResultsBuffers<data_t>> resBuf = std::make_shared<ResultsBuffers<data_t>>(
                        batchSizeUsed);
                gpu_batches[i] = new BatchData<K, data_t>(0,
                                                          resBuf,
                                                          batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }

        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    M getModel() {
        return *model;
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        if (numslabs == 0)
            return 0;
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        if (numslabs != 0)
            slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:

    // normal
    /// returns response location in resbuf
    template<bool UseCache_ = UseCache, typename std::enable_if_t<(UseCache_ && UseGPU)> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, data_t> *> &gpu_batches) {

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches.push_back(new BatchData<K, data_t>(0, resBuf, req_vector.size()));
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (model->operator()(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }
        return sizeForGPUBatches;
    }

    // just cache
    template<bool UseCache_ = UseCache, typename std::enable_if_t<(UseCache_ && !UseGPU)> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, data_t> *> &gpu_batches) {

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                cache_batch_corespondance.push_back({i, h});
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        return sizeForGPUBatches;
    }

    // GPU only
    template<bool UseCache_ = UseCache, typename std::enable_if_t<!UseCache_> * = nullptr>
    inline int
    route(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, data_t> *> &gpu_batches) {

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches.push_back(new BatchData<K, data_t>(0, resBuf, req_vector.size()));
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                int gpuToUse = h % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = req.key;
                gpu_batches[gpuToUse]->values[idx] = req.value;
                gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                gpu_batches[gpuToUse]->hashes[idx] = h;

            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }
        return sizeForGPUBatches;
    }

    // gpu only and normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseGPU_> * = nullptr>
    inline void sendGPUBatches(bool &dontDoGPU, std::vector<BatchData<K, data_t> *> &gpu_batches) {
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
        }
    }

    // cpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseGPU_> * = nullptr>
    inline void sendGPUBatches(bool &dontDoGPU, std::vector<BatchData<K, data_t> *> &gpu_batches) {}

    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void setUpMissBatches(std::vector<RequestWrapper<K, data_t *>> &req_vector,
                                 std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                                 std::vector<BatchData<K, data_t> *> &gpu_batches2) {
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2.push_back(new BatchData<K, data_t>(0, resBuf, req_vector.size()));
        }
    }

    // gpu only or cpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<
            (UseCache && !UseGPU_) || (!UseCache && UseGPU_)> * = nullptr>
    inline void setUpMissBatches(std::vector<RequestWrapper<K, data_t *>> &req_vector,
                                 std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                                 std::vector<BatchData<K, data_t> *> &gpu_batches2) {}

    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, data_t> *> &gpu_batches2,
                       RequestWrapper<K, data_t *> &req_vector_elm, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        int gpuToUse = cache_batch_idx.second % numslabs;
        int idx = gpu_batches2[gpuToUse]->idx;
        gpu_batches2[gpuToUse]->idx++;
        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
        gpu_batches2[gpuToUse]->handleInCache[idx] = true;
    }

    // cache only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && !UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, data_t> *> &gpu_batches2,
                       RequestWrapper<K, data_t *> &req_vector_elm, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        hits.fetch_add(1, std::memory_order_relaxed);
        resBuf->resultValues[responseLocationInResBuf] = nullptr;
        asm volatile("":: : "memory");
        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
        responseLocationInResBuf++;
        times.push_back(std::chrono::high_resolution_clock::now());
    }

    // gpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseCache && UseGPU_> * = nullptr>
    inline void onMiss(std::pair<int, unsigned int> &cache_batch_idx, std::vector<BatchData<K, data_t> *> &gpu_batches2,
                       RequestWrapper<K, data_t *> &req_vector_elm, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
                       int &responseLocationInResBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        // should never be called
        assert(false);
    }


    // normal
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<UseCache && UseGPU_> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, data_t> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<data_t>> &resBuf) {
        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches2[i]->idx > 0) {
                    gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                    sizeForGPUBatches += gpu_batches2[i]->idx;
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches2[i]);
                } else {
                    delete gpu_batches2[i];
                }
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches2[i];
            }
            resBuf->retryGPU = true;
        }
    }

    // cache only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<(UseCache && !UseGPU_)> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, data_t> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<data_t>> &resBuf) {}

    // gpu only
    template<bool UseGPU_ = UseGPU, typename std::enable_if_t<!UseCache && UseGPU_> * = nullptr>
    inline void
    forwardMissBatch(bool &dontDoGPU, std::vector<BatchData<K, data_t> *> &gpu_batches2, int &sizeForGPUBatches,
                     std::shared_ptr<ResultsBuffers<data_t>> &resBuf) {
        if (dontDoGPU) {
            resBuf->retryGPU = true;
        }
    }

    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, data_t *>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};*/

template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>>
using NoCacheKVStoreInternalClient = KVStoreInternalClient<K, V, M, Slab_t, false, true>;

template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>>
using JustCacheKVStoreInternalClient = KVStoreInternalClient<K, V, M, Slab_t, true, false>;

/*
template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>>
class NoCacheKVStoreInternalClient {
    // TODO make sure it is working
public:
    NoCacheKVStoreInternalClient(std::shared_ptr<Slab_t> s, std::shared_ptr<typename Cache<K, V>::type> c,
                                 std::shared_ptr<M> m) : numslabs(s->numslabs), slabs(s), cache(c),
                                                         hits(0), operations(0),
                                                         start(std::chrono::high_resolution_clock::now()), model(m) {
        std::cerr << "This should not get used" << std::endl;
    }

    ~NoCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (mfn(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        for (int i = 0; i < numslabs; ++i) {
            slabs->gpu_qs[i].push(gpu_batches[i]);
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = 0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::locktype> pair = cache->get(req_vector_elm.key,
                                                                                             cache_batch_idx.second,
                                                                                             *model);
                    if (pair.first == nullptr) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        auto endTime = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        for (int i = 0; i < numslabs; ++i) {
            if (gpu_batches2[i]->idx > 0) {
                gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                sizeForGPUBatches += gpu_batches2[i]->idx;
                slabs->gpu_qs[i].push(gpu_batches2[i]);
            } else {
                delete gpu_batches2[i];
            }
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
};

template<typename K, typename M, typename Slab_t>
class NoCacheKVStoreInternalClient<K, data_t, M, Slab_t> {
public:
    NoCacheKVStoreInternalClient(std::shared_ptr<Slab_t> s,
                                 std::shared_ptr<typename Cache<K, data_t *>::type> c, std::shared_ptr<M> m) : numslabs(
            s->numslabs), slabs(s), cache(c), hits(0),
                                                                                                               operations(
                                                                                                                       0),
                                                                                                               start(std::chrono::high_resolution_clock::now()),
                                                                                                               model(m) {

    }

    ~NoCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, data_t *> RW;

    void batch(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        bool dontDoGPU = false;

        if (slabs->load >= LOAD_THRESHOLD) {
            std::cerr << "Dropped " << std::endl;
            dontDoGPU = true;
        }

        //std::cerr << req_vector.size() << std::endl;
        //req_vector.size() % 512 == 0 &&
        assert(req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        auto gpu_batches = std::vector<BatchData<K, data_t> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, data_t>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);

                int gpuToUse = h % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = req.key;
                gpu_batches[gpuToUse]->values[idx] = req.value;
                gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                gpu_batches[gpuToUse]->hashes[idx] = h;
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        if (!dontDoGPU) {
            for (int i = 0; i < numslabs; ++i) {
                slabs->load++;
                slabs->gpu_qs[i].push(gpu_batches[i]);
            }
        } else {
            for (int i = 0; i < numslabs; ++i) {
                delete gpu_batches[i];
            }
            resBuf->retryGPU = true;
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches;

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;

        // send gpu_batch2

        operations += req_vector.size();

    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<data_t *> *log_values = cache->log_values;

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = new tbb::concurrent_vector<int>(cache->getN() * cache->getSETS());
        cache->log_hash = new tbb::concurrent_vector<unsigned>(cache->getN() * cache->getSETS());
        cache->log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        cache->log_values = new tbb::concurrent_vector<data_t *>(cache->getN() * cache->getSETS());
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            auto gpu_batches = std::vector<BatchData<K, data_t> *>(numslabs);

            for (int i = 0; i < numslabs; ++i) {
                std::shared_ptr<ResultsBuffers<data_t>> resBuf = std::make_shared<ResultsBuffers<data_t>>(
                        batchSizeUsed);
                gpu_batches[i] = new BatchData<K, data_t>(0,
                                                          resBuf,
                                                          batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }

        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    M getModel() {
        return *model;
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, data_t *>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};

template<typename K, typename V, typename M, typename Slab_t = Slabs<K, V, M>>
class JustCacheKVStoreInternalClient {
    // TODO make sure it is working and correct logic
public:
    JustCacheKVStoreInternalClient(std::shared_ptr<Slab_t> s, std::shared_ptr<typename Cache<K, V>::type> c,
                                   std::shared_ptr<M> m) : numslabs(s->numslabs), slabs(s), cache(c),
                                                           hits(0), operations(0),
                                                           start(std::chrono::high_resolution_clock::now()), model(m) {
        std::cerr << "This should not get used" << std::endl;
    }

    ~JustCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, V> RW;

    void batch(std::vector<RequestWrapper<K, V>> &req_vector, std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {

        //std::cerr << req_vector.size() << std::endl;
        assert(req_vector.size() % 512 == 0 && req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        auto gpu_batches = std::vector<BatchData<K, V> *>(numslabs);

        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                if (mfn(req.key, h)) {
                    cache_batch_corespondance.push_back({i, h});
                } else {
                    int gpuToUse = h % numslabs;
                    int idx = gpu_batches[gpuToUse]->idx;
                    gpu_batches[gpuToUse]->idx++;
                    gpu_batches[gpuToUse]->keys[idx] = req.key;
                    gpu_batches[gpuToUse]->values[idx] = req.value;
                    gpu_batches[gpuToUse]->requests[idx] = req.requestInteger;
                    gpu_batches[gpuToUse]->hashes[idx] = h;
                }
            }
        }

        int sizeForGPUBatches = cache_batch_corespondance.size();
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches[i]->resBufStart = sizeForGPUBatches;
            sizeForGPUBatches += gpu_batches[i]->idx;
        }

        for (int i = 0; i < numslabs; ++i) {
            slabs->gpu_qs[i].push(gpu_batches[i]);
        }

        auto gpu_batches2 = std::vector<BatchData<K, V> *>(numslabs);
        for (int i = 0; i < numslabs; ++i) {
            gpu_batches2[i] = new BatchData<K, V>(0, resBuf, req_vector.size());
        }

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = 0;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, V> *, kvgpu::locktype> pair = cache->get(req_vector_elm.key,
                                                                                             cache_batch_idx.second,
                                                                                             *model);
                    if (pair.first == nullptr) {
                        int gpuToUse = cache_batch_idx.second % numslabs;
                        int idx = gpu_batches2[gpuToUse]->idx;
                        gpu_batches2[gpuToUse]->idx++;
                        gpu_batches2[gpuToUse]->keys[idx] = req_vector_elm.key;
                        gpu_batches2[gpuToUse]->requests[idx] = req_vector_elm.requestInteger;
                        gpu_batches2[gpuToUse]->hashes[idx] = cache_batch_idx.second;
                        gpu_batches2[gpuToUse]->handleInCache[idx] = true;

                    } else {
                        hits++;
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";

                        resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());

                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    std::pair<kvgpu::LockingPair<K, V> *, std::unique_lock<kvgpu::mutex>> pair = cache->get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            pair.first->deleted = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        auto endTime = std::chrono::high_resolution_clock::now();

        std::atomic_thread_fence(std::memory_order_seq_cst);

        for (int i = 0; i < numslabs; ++i) {
            if (gpu_batches2[i]->idx > 0) {
                gpu_batches2[i]->resBufStart = sizeForGPUBatches;
                sizeForGPUBatches += gpu_batches2[i]->idx;
                slabs->gpu_qs[i].push(gpu_batches2[i]);
            } else {
                delete gpu_batches2[i];
            }
        }

        // send gpu_batch2

        operations += req_vector.size();

    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return slabs->getOps();
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
        slabs->clearMops();
    }

    void stat() {
        for (int i = 0; i < numslabs; i++) {
            std::cout << "TABLE: GPU Info " << i << std::endl;
            std::cout
                    << "Time from start (s)\tTime spent responding (ms)\tTime in batch fn (ms)\tTime Dequeueing (ms)\tFraction that goes to cache\tDuration (ms)\tFill\tThroughput GPU "
                    << i << " (Mops)" << std::endl;
            for (auto &s : slabs->mops[i]) {

                std::cout << std::chrono::duration<double>(s.timestampEnd - start).count() << "\t"
                          << std::chrono::duration<double>(s.timestampEnd - s.timestampWriteBack).count() * 1e3 << "\t"
                          << std::chrono::duration<double>(s.timestampWriteBack - s.timestampStartBatch).count() * 1e3
                          << "\t"
                          << std::chrono::duration<double>(s.timestampStartBatch - s.timestampDequeueToBatch).count() *
                             1e3 << "\t"
                          << s.timesGoingToCache / (double) s.size << "\t"
                          << s.duration << "\t" << (double) s.size / THREADS_PER_BLOCK / BLOCKS << "\t"
                          << s.size / s.duration / 1e3 << std::endl;
            }
            std::cout << std::endl;
        }
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
};


template<typename K, typename M, typename Slab_t>
class JustCacheKVStoreInternalClient<K, data_t, M, Slab_t> {
    // Slab_t should be some Slab<K, data_t*, M>
    static_assert(std::is_same<data_t *, typename Slab_t::VType>::value, "GPU Hashmap needs to handle GPU stuff");
public:
    JustCacheKVStoreInternalClient(std::shared_ptr<Slab_t> s,
                                   std::shared_ptr<typename Cache<K, data_t *>::type> c, std::shared_ptr<M> m)
            : numslabs(0), slabs(nullptr), cache(c), hits(0),
              operations(0),
              start(std::chrono::high_resolution_clock::now()),
              model(m) {

    }

    ~JustCacheKVStoreInternalClient() {}

    typedef RequestWrapper<K, data_t *> RW;

    void batch(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {

        //std::cerr << req_vector.size() << std::endl;
        //req_vector.size() % 512 == 0 &&
        //assert(req_vector.size() <= THREADS_PER_BLOCK * BLOCKS * numslabs);

        std::vector<std::pair<int, unsigned>> cache_batch_corespondance;

        cache_batch_corespondance.reserve(req_vector.size());
        std::vector<BatchData<K, data_t> *> gpu_batches;

        gpu_batches.reserve(numslabs);

        int sizeForGPUBatches = route(req_vector, resBuf, cache_batch_corespondance, gpu_batches);

        //std::cerr << "Looking through cache now\n";
        int responseLocationInResBuf = sizeForGPUBatches;

        for (auto &cache_batch_idx : cache_batch_corespondance) {

            auto req_vector_elm = req_vector[cache_batch_idx.first];

            if (req_vector_elm.requestInteger != REQUEST_EMPTY) {

                if (req_vector_elm.requestInteger == REQUEST_GET) {
                    std::pair<kvgpu::LockingPair<K, data_t *> *, kvgpu::sharedlocktype> pair = cache->fast_get(
                            req_vector_elm.key, cache_batch_idx.second, *model);
                    if (pair.first == nullptr || pair.first->valid != 1) {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        resBuf->resultValues[responseLocationInResBuf] = nullptr;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());
                    } else {
                        hits.fetch_add(1, std::memory_order_relaxed);
                        //std::cerr << "Hit on get" << __FILE__ << ":" << __LINE__ << "\n";
                        data_t *cpy = nullptr;
                        if (pair.first->deleted == 0 && pair.first->value) {
                            cpy = new data_t(pair.first->value->size);
                            memcpy(cpy->data, pair.first->value->data, cpy->size);
                        }
                        resBuf->resultValues[responseLocationInResBuf] = cpy;
                        asm volatile("":: : "memory");
                        resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                        responseLocationInResBuf++;
                        times.push_back(std::chrono::high_resolution_clock::now());
                    }
                    if (pair.first != nullptr)
                        pair.second.unlock();
                } else {
                    size_t logLoc = 0;
                    std::pair<kvgpu::LockingPair<K, data_t *> *, std::unique_lock<kvgpu::mutex>> pair = cache->get_with_log(
                            req_vector_elm.key, cache_batch_idx.second, *model, logLoc);
                    switch (req_vector_elm.requestInteger) {
                        case REQUEST_INSERT:
                            //std::cerr << "Insert request\n";
                            hits++;
                            pair.first->value = req_vector_elm.value;
                            pair.first->deleted = 0;
                            pair.first->valid = 1;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;
                            cache->log_requests->operator[](logLoc) = REQUEST_INSERT;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;
                            cache->log_values->operator[](logLoc) = req_vector_elm.value;

                            break;
                        case REQUEST_REMOVE:
                            //std::cerr << "RM request\n";

                            if (pair.first->valid == 1) {
                                resBuf->resultValues[responseLocationInResBuf] = pair.first->value;
                                pair.first->value = nullptr;
                            }

                            pair.first->deleted = 1;
                            pair.first->valid = 1;
                            hits++;
                            resBuf->requestIDs[responseLocationInResBuf] = cache_batch_idx.first;
                            responseLocationInResBuf++;

                            cache->log_requests->operator[](logLoc) = REQUEST_REMOVE;
                            cache->log_hash->operator[](logLoc) = cache_batch_idx.second;
                            cache->log_keys->operator[](logLoc) = req_vector_elm.key;

                            break;
                    }
                    times.push_back(std::chrono::high_resolution_clock::now());

                    if (pair.first != nullptr)
                        pair.second.unlock();
                }
            }
        }

        //std::cerr << "Done looking through cache now\n";

        asm volatile("":: : "memory");
        sizeForGPUBatches = responseLocationInResBuf;

        // send gpu_batch2

        operations += req_vector.size();

    }

    /// returns response location in resbuf
    inline int
    route(std::vector<RequestWrapper<K, data_t *>> &req_vector, std::shared_ptr<ResultsBuffers<data_t>> &resBuf,
          std::vector<std::pair<int, unsigned>> &cache_batch_corespondance,
          std::vector<BatchData<K, data_t> *> &gpu_batches) {

        for (int i = 0; i < req_vector.size(); ++i) {
            RW req = req_vector[i];
            if (req.requestInteger != REQUEST_EMPTY) {
                unsigned h = hfn(req.key);
                cache_batch_corespondance.push_back({i, h});
            }
        }

        int sizeForGPUBatches = 0; //cache_batch_corespondance.size();
        return sizeForGPUBatches;
    }

    // single threaded
    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        std::unique_lock<std::mutex> modelLock(modelMtx);
        tbb::concurrent_vector<int> *log_requests = cache->log_requests;
        tbb::concurrent_vector<unsigned> *log_hash = cache->log_hash;
        tbb::concurrent_vector<K> *log_keys = cache->log_keys;
        tbb::concurrent_vector<data_t *> *log_values = cache->log_values;

        while (!block->threads_blocked());
        //std::cerr << "All threads at barrier\n";
        asm volatile("":: : "memory");
        auto start = std::chrono::high_resolution_clock::now();

        *model = newModel;
        cache->log_requests = new tbb::concurrent_vector<int>(cache->getN() * cache->getSETS());
        cache->log_hash = new tbb::concurrent_vector<unsigned>(cache->getN() * cache->getSETS());
        cache->log_keys = new tbb::concurrent_vector<K>(cache->getN() * cache->getSETS());
        cache->log_values = new tbb::concurrent_vector<data_t *>(cache->getN() * cache->getSETS());
        size_t tmpSize = cache->log_size;
        cache->log_size = 0;

        int batchSizeUsed = std::min(THREADS_PER_BLOCK * BLOCKS,
                                     (int) (tmpSize / THREADS_PER_BLOCK + 1) * THREADS_PER_BLOCK);

        for (int enqueued = 0; enqueued < tmpSize; enqueued += batchSizeUsed) {

            auto gpu_batches = std::vector<BatchData<K, data_t> *>(numslabs);

            for (int i = 0; i < numslabs; ++i) {
                std::shared_ptr<ResultsBuffers<data_t>> resBuf = std::make_shared<ResultsBuffers<data_t>>(
                        batchSizeUsed);
                gpu_batches[i] = new BatchData<K, data_t>(0,
                                                          resBuf,
                                                          batchSizeUsed);
                gpu_batches[i]->resBufStart = 0;
                gpu_batches[i]->flush = true;
            }

            for (int i = 0; i + enqueued < tmpSize && i < batchSizeUsed; ++i) {
                int gpuToUse = log_hash->operator[](i + enqueued) % numslabs;
                int idx = gpu_batches[gpuToUse]->idx;
                gpu_batches[gpuToUse]->idx++;
                gpu_batches[gpuToUse]->keys[idx] = log_keys->operator[](i + enqueued);
                gpu_batches[gpuToUse]->values[idx] = log_values->operator[](i + enqueued);
                gpu_batches[gpuToUse]->requests[idx] = log_requests->operator[](i + enqueued);
                gpu_batches[gpuToUse]->hashes[idx] = log_hash->operator[](i + enqueued);
            }

            for (int i = 0; i < numslabs; ++i) {
                if (gpu_batches[i]->idx == 0) {
                    delete gpu_batches[i];
                } else {
                    slabs->load++;
                    slabs->gpu_qs[i].push(gpu_batches[i]);
                }
            }

        }
        block->wake();
        auto end = std::chrono::high_resolution_clock::now();
        time = std::chrono::duration<double>(end - start).count();

        return std::async([this](std::unique_lock<std::mutex> l) {
            std::hash<K> h;
            cache->scan_and_evict(*(this->model), h, std::move(l));
        }, std::move(modelLock));
    }

    M getModel() {
        return *model;
    }

    float hitRate() {
        return (double) hits / operations;
    }

    size_t getOps() {
        return 0;
    }

    size_t getHits() {
        return hits;
    }

    void resetStats() {
        hits = 0;
        operations = 0;
    }

    void stat() {
        cache->stat();
    }

private:
    int numslabs;
    std::mutex mtx;
    std::shared_ptr<Slab_t> slabs;
    //SlabUnified<K,V> *slabs;
    std::shared_ptr<typename Cache<K, data_t *>::type> cache;
    std::hash<K> hfn;
    std::atomic_size_t hits;
    std::atomic_size_t operations;
    std::shared_ptr<M> model;
    std::chrono::high_resolution_clock::time_point start;
    std::mutex modelMtx;
};
*/

#endif //KVGPU_KVSTOREINTERNALCLIENT_CUH
