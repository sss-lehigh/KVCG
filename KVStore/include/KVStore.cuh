//
// Created by depaulsmiller on 9/2/20.
//

#include <utility>
#include <memory>
#include <atomic>
#include <KVCache.cuh>
#include <Slab.cuh>
#include <StandardSlabDefinitions.cuh>
#include <mutex>
#include <iostream>
#include <PartitionedSlabUnified.cuh>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_vector.h>
#include <MegaKV.cuh>

#ifndef KVGPU_KVSTORE_CUH
#define KVGPU_KVSTORE_CUH

int SLAB_SIZE = 1000000;

const int MAX_ATTEMPTS = 1;

const std::vector<PartitionedSlabUnifiedConfig> STANDARD_CONFIG = {{SLAB_SIZE, 0, cudaStreamDefault},
                                                                   {SLAB_SIZE, 1, cudaStreamDefault}};

template<typename V>
struct SubstituteType {
    using type = V;
};

template<>
struct SubstituteType<data_t> {
    using type = data_t *;
};

template<typename K, typename V>
class Cache {
public:
    typedef kvgpu::KVCache<K, V, 1000000, 8> type;
};

template<typename V>
struct ResultsBuffers {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new V[s]), size(s), retryGPU(false) {
        for (int i = 0; i < size; i++)
            requestIDs[i] = -1;
    }

    ResultsBuffers(const ResultsBuffers<V> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile V *resultValues;
    int size;
    bool retryGPU;
};

template<>
struct ResultsBuffers<data_t> {

    explicit ResultsBuffers(int s) : requestIDs(new int[s]), resultValues(new volatile data_t *[s]), size(s),
                                     retryGPU(false) {
        for (int i = 0; i < size; i++) {
            requestIDs[i] = -1;
            resultValues[i] = nullptr;
        }
    }

    ResultsBuffers(const ResultsBuffers<data_t> &) = delete;

    ~ResultsBuffers() {
        delete[] requestIDs;
        for (int i = 0; i < size; i++) {
            if (resultValues[i] && resultValues[i]->data)
                delete[] resultValues[i]->data;
        }
        delete[] resultValues;
    }

    volatile int *requestIDs;
    volatile data_t **resultValues;
    int size;
    bool retryGPU;
};

template<typename K, typename V>
struct BatchData {
    BatchData(int rbStart, std::shared_ptr<ResultsBuffers<V>> rb, int s) : keys(s), values(s), requests(s), hashes(s),
                                                                           requestID(s),
                                                                           handleInCache(s), resBuf(rb),
                                                                           resBufStart(rbStart), size(s), idx(0),
                                                                           flush(false) {
        for (int i = 0; i < s; i++) {
            handleInCache[i] = false;
        }
    }

    ~BatchData() = default;

    std::vector<K> keys;
    std::vector<V> values;
    std::vector<unsigned> requests;
    std::vector<unsigned> hashes;
    std::vector<int> requestID;
    std::vector<bool> handleInCache;
    std::shared_ptr<ResultsBuffers<V>> resBuf;
    int resBufStart;
    int size;
    int idx;
    bool flush;
};

template<typename K>
struct BatchData<K, data_t> {
    BatchData(int rbStart, std::shared_ptr<ResultsBuffers<data_t>> &rb, int s) : keys(s), values(s), requests(s),
                                                                                 hashes(s), requestID(s),
                                                                                 handleInCache(s), resBuf(rb),
                                                                                 resBufStart(rbStart), size(s), idx(0),
                                                                                 flush(false) {
        for (int i = 0; i < s; i++) {
            handleInCache[i] = false;
        }
    }

    ~BatchData() = default;

    std::vector<K> keys;
    std::vector<data_t *> values;
    std::vector<unsigned> requests;
    std::vector<unsigned> hashes;
    std::vector<int> requestID;
    std::vector<bool> handleInCache;
    std::shared_ptr<ResultsBuffers<data_t>> resBuf;
    int resBufStart;
    int size;
    int idx;
    bool flush;
};

struct StatData {
    std::chrono::high_resolution_clock::time_point timestampEnd;
    std::chrono::high_resolution_clock::time_point timestampWriteBack;
    std::chrono::high_resolution_clock::time_point timestampStartBatch;
    std::chrono::high_resolution_clock::time_point timestampDequeueToBatch;
    float duration;
    int size;
    int timesGoingToCache;
};

template<typename K, typename V, typename M>
struct Slabs {

    using VType = V;

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, V> *> q_t;

    Slabs(const std::vector<PartitionedSlabUnifiedConfig> &config, std::shared_ptr<typename Cache<K, V>::type> cache,
          std::shared_ptr<M> m) : numslabs(config.size()), slabs(new SlabUnified<K, V>[numslabs]),
                                  gpu_qs(new q_t[numslabs]), done(false),
                                  mops(new tbb::concurrent_vector<StatData>[numslabs]), _cache(cache), ops(0), load(0),
                                  model(m) {
        for (int i = 0; i < config.size(); i++) {
            cudaStream_t *stream = new cudaStream_t();
            *stream = config[i].stream;
            slabs[i] = std::move(SlabUnified<K, V>(config[i].size, config[i].gpu, stream));
        }
        for (int i = 0; i < numslabs; ++i) {
            threads.push_back(std::thread([this](int tid) {
                K *keys = slabs[tid].getBatchKeys();
                V *values = slabs[tid].getBatchValues();
                int *requests = slabs[tid].getBatchRequests();
                unsigned *hashes = slabs[tid].getHashValues();

                BatchData<K, V> *holdonto = nullptr;

                std::vector<std::pair<int, BatchData<K, V> *>> writeBack;
                writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                int index = THREADS_PER_BLOCK * BLOCKS;
                while (!done.load()) {
                    writeBack.clear();
                    for (int i = 0; i < index; i++) {
                        requests[i] = REQUEST_EMPTY;
                    }
                    index = 0;

                    BatchData<K, V> *res;

                    auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                    if (holdonto) {
                        //std::cerr << "Hold onto set " << tid << std::endl;
                        writeBack.push_back({index, holdonto});

                        for (int i = 0; i < holdonto->idx; i++) {
                            keys[index + i] = holdonto->keys[i];
                            values[index + i] = holdonto->values[i];
                            requests[index + i] = holdonto->requests[i];
                            hashes[index + i] = holdonto->hashes[i];
                        }
                        index += holdonto->idx;
                        holdonto = nullptr;
                    }

                    int attempts = 0;

                    while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                        if (this->gpu_qs[tid].try_pop(res)) {
                            load--;
                            //std::cerr << "Got a batch on handler thread " << tid << "\n";
                            if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                holdonto = res;
                                break;
                            }
                            for (int i = 0; i < res->idx; i++) {
                                keys[index + i] = res->keys[i];
                                values[index + i] = res->values[i];
                                requests[index + i] = res->requests[i];
                                hashes[index + i] = res->hashes[i];
                            }
                            writeBack.push_back({index, res});
                            index += res->idx;
                            if (res->flush) {
                                break;
                            }
                        } else {
                            attempts++;
                        }
                    }

                    if (index > 0) {

                        //std::cerr << "Batching " << tid << "\n";

                        auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                        float t;

                        this->slabs[tid].diy_batch(t, ceil(index / 512.0), 512);

                        auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                        int timesGoingToCache = 0;
                        for (auto &wb : writeBack) {

                            int rbLoc = wb.second->resBufStart;

                            for (int i = 0; i < wb.second->idx; ++i) {

                                if (wb.second->handleInCache[i]) {
                                    timesGoingToCache++;
                                    auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                *(this->model));
                                    if (cacheRes.first->valid == 1) {
                                        wb.second->resBuf->resultValues[rbLoc + i] = cacheRes.first->value;
                                    } else {
                                        cacheRes.first->valid = 1;
                                        cacheRes.first->value = values[wb.first + i];
                                        cacheRes.first->deleted = (values[wb.first + i] == EMPTY<V>::value);
                                    }
                                    asm volatile("":: : "memory");

                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                } else {
                                    wb.second->resBuf->resultValues[rbLoc + i] = values[wb.first + i];
                                    asm volatile("":: : "memory");
                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                }
                            }
                            delete wb.second;
                        }

                        mops[tid].push_back(
                                {std::chrono::high_resolution_clock::now(), timestampWriteBack, timestampStartBatch,
                                 timestampWriteToBatch, t, index, timesGoingToCache});

                        ops += index;
                        //std::cerr << "Batched " << tid << "\n";

                    }
                }
            }, i));
        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] slabs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    SlabUnified<K, V> *slabs;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};

template<typename K, typename M>
struct Slabs<K, data_t *, M> {

    using V = data_t *;
    using VType = data_t *;

    Slabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, data_t> *> q_t;

    Slabs(const std::vector<PartitionedSlabUnifiedConfig> &config,
          std::shared_ptr<typename Cache<K, V>::type> cache, std::shared_ptr<M> m) : done(false),
                                                                                     mops(new tbb::concurrent_vector<StatData>[config.size()]),
                                                                                     _cache(cache), ops(0),
                                                                                     load(0), model(m) {
        std::unordered_map<int, std::shared_ptr<SlabUnified<K, V>>> gpusToSlab;
        for (int i = 0; i < config.size(); i++) {
            if (gpusToSlab.find(config[i].gpu) == gpusToSlab.end())
                gpusToSlab[config[i].gpu] = std::make_shared<SlabUnified<K, V>>(config[i].size, config[i].gpu);
        }
        gpu_qs = new q_t[gpusToSlab.size()];
        numslabs = gpusToSlab.size();

        for (int i = 0; i < config.size(); i++) {
            //config[i].stream;
            threads.push_back(
                    std::thread([this](int tid, int gpu, std::shared_ptr<SlabUnified<K, V>> slab,
                                       cudaStream_t stream) {
                                    slab->setGPU();
                                    auto batchData = new BatchBuffer<K, V>();

                                    K *keys = batchData->getBatchKeys();
                                    V *values = batchData->getBatchValues();
                                    int *requests = batchData->getBatchRequests();
                                    unsigned *hashes = batchData->getHashValues();

                                    BatchData<K, data_t> *holdonto = nullptr;

                                    std::vector<std::pair<int, BatchData<K, data_t> *>> writeBack;
                                    writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                                    int index = THREADS_PER_BLOCK * BLOCKS;
                                    while (!done.load()) {
                                        writeBack.clear();
                                        for (int i = 0; i < index; i++) {
                                            requests[i] = REQUEST_EMPTY;
                                        }
                                        index = 0;

                                        BatchData<K, data_t> *res;

                                        auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                                        if (holdonto) {
                                            //std::cerr << "Hold onto set " << tid << std::endl;
                                            writeBack.push_back({index, holdonto});

                                            for (int i = 0; i < holdonto->idx; i++) {
                                                keys[index + i] = holdonto->keys[i];
                                                values[index + i] = holdonto->values[i];
                                                requests[index + i] = holdonto->requests[i];
                                                hashes[index + i] = holdonto->hashes[i];
                                            }
                                            index += holdonto->idx;
                                            holdonto = nullptr;
                                        }

                                        int attempts = 0;

                                        while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                                            if (this->gpu_qs[gpu].try_pop(res)) {
                                                load--;
                                                //std::cerr << "Got a batch on handler thread " << tid << "\n";
                                                if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                                    //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                                    holdonto = res;
                                                    break;
                                                }
                                                for (int i = 0; i < res->idx; i++) {
                                                    keys[index + i] = res->keys[i];
                                                    values[index + i] = res->values[i];
                                                    assert(res->requests[i] != REQUEST_INSERT || res->values[i]->size > 0);
                                                    requests[index + i] = res->requests[i];
                                                    hashes[index + i] = res->hashes[i];
                                                }
                                                writeBack.push_back({index, res});
                                                index += res->idx;
                                                if (res->flush) {
                                                    break;
                                                }
                                            } else {
                                                attempts++;
                                            }
                                        }

                                        if (index > 0) {

                                            //std::cerr << "Batching " << tid << "\n";

                                            auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                                            cudaEvent_t start, stop;

                                            gpuErrchk(cudaEventCreate(&start));
                                            gpuErrchk(cudaEventCreate(&stop));

                                            float t;

                                            slab->moveBufferToGPU(batchData, stream);
                                            gpuErrchk(cudaEventRecord(start, stream));
                                            slab->diy_batch(batchData, ceil(index / 512.0), 512, stream);
                                            gpuErrchk(cudaEventRecord(stop, stream));
                                            slab->moveBufferToCPU(batchData, stream);
                                            gpuErrchk(cudaStreamSynchronize(stream));

                                            auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                                            gpuErrchk(cudaEventElapsedTime(&t, start, stop));
                                            gpuErrchk(cudaEventDestroy(start));
                                            gpuErrchk(cudaEventDestroy(stop));
                                            int timesGoingToCache = 0;
                                            for (auto &wb : writeBack) {

                                                int rbLoc = wb.second->resBufStart;

                                                for (int i = 0; i < wb.second->idx; ++i) {

                                                    if (wb.second->handleInCache[i]) {
                                                        timesGoingToCache++;
                                                        auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                                    *(this->model));
                                                        if (cacheRes.first->valid == 1) {
                                                            V cpy = nullptr;
                                                            if (cacheRes.first->deleted == 0) {
                                                                cpy = new data_t(cacheRes.first->value->size);
                                                                memcpy(cpy->data, cacheRes.first->value->data, cpy->size);
                                                            }

                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            cacheRes.first->valid = 1;
                                                            cacheRes.first->value = values[wb.first + i];
                                                            cacheRes.first->deleted = (values[wb.first + i] ==
                                                                                       EMPTY<V>::value);
                                                        }
                                                        asm volatile("":: : "memory");

                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                                    } else {
                                                        if (requests[wb.first + i] == REQUEST_REMOVE) {
                                                            wb.second->resBuf->resultValues[rbLoc +
                                                                                            i] = values[wb.first + i];
                                                        } else if (requests[wb.first + i] == REQUEST_GET) {
                                                            V cpy = nullptr;
                                                            if (values[wb.first + i]) {
                                                                cpy = new data_t(values[wb.first + i]->size);
                                                                memcpy(cpy->data, values[wb.first + i]->data, cpy->size);
                                                            }
                                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                                        } else {
                                                            wb.second->resBuf->resultValues[rbLoc + i] = nullptr;
                                                        }

                                                        asm volatile("":: : "memory");
                                                        wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                                    }
                                                }
                                                delete wb.second;
                                            }

                                            mops[tid].push_back(
                                                    {std::chrono::high_resolution_clock::now(), timestampWriteBack,
                                                     timestampStartBatch,
                                                     timestampWriteToBatch, t, index, timesGoingToCache});

                                            ops += index;
                                            //std::cerr << "Batched " << tid << "\n";

                                        }
                                    }
                                    if (stream != cudaStreamDefault) gpuErrchk(cudaStreamDestroy(stream));
                                }, i, config[i].gpu,
                                gpusToSlab[config[i].gpu], config[i].stream));

        }
    }

    ~Slabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};

template<typename K, typename V, typename M>
struct MegaKVSlabs {

    using VType = V;

    MegaKVSlabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, V> *> q_t;

    MegaKVSlabs(const std::vector<PartitionedSlabUnifiedConfig> &config,
                std::shared_ptr<typename Cache<K, V>::type> cache,
                std::shared_ptr<M> m) : numslabs(1), slabs(new megakv::MegaKVGPU(config[0].size)),
                                        secondaryIndex(new megakv::SecondaryIndex<K, V>(megakv::SizeLog2)),
                                        gpu_qs(new q_t[numslabs]), done(false),
                                        mops(new tbb::concurrent_vector<StatData>[numslabs]), _cache(cache), ops(0),
                                        load(0),
                                        model(m) {

        std::cerr << "Dont use this version " << std::endl;

        for (int i = 0; i < numslabs; ++i) {
            threads.push_back(std::thread([this](int tid) {
                cudaStream_t stream;
                gpuErrchk(cudaStreamCreate(&stream));
                auto data = std::make_unique<megakv::GPUData>();

                for (int i = 0; i < megakv::THREADS_PER_BLOCK * megakv::BLOCKS; i++) {
                    data->requests_h[i] = megakv::EMPTY;
                }

                BatchData<K, V> *holdonto = nullptr;

                std::vector<std::pair<int, BatchData<K, V> *>> writeBack;
                writeBack.reserve(THREADS_PER_BLOCK * BLOCKS / 512);

                int index = THREADS_PER_BLOCK * BLOCKS;
                while (!done.load()) {
                    writeBack.clear();
                    for (int i = 0; i < index; i++) {
                        data->requests_h[i] = REQUEST_EMPTY;
                    }
                    index = 0;

                    BatchData<K, V> *res;

                    auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                    if (holdonto) {
                        //std::cerr << "Hold onto set " << tid << std::endl;
                        writeBack.push_back({index, holdonto});

                        for (int i = 0; i < holdonto->idx; i++) {
                            data->keys_h[index + i] = holdonto->hashes[i];
                            data->requests_h[index + i] = holdonto->requests[i];
                            uint8_t loc;
                            uint32_t loc_hash;
                            megakv::Bucket<K, V> *b = secondaryIndex->alloc(loc, loc_hash);

                            std::pair<K, V> p = {holdonto->keys[i], holdonto->values[i]};
                            (*b).set(loc, p);
                            data->values_h[index + i] = loc_hash;
                        }
                        index += holdonto->idx;
                        holdonto = nullptr;
                    }

                    int attempts = 0;

                    while (attempts < MAX_ATTEMPTS && index < THREADS_PER_BLOCK * BLOCKS) {
                        if (this->gpu_qs->try_pop(res)) {
                            load--;
                            //std::cerr << "Got a batch on handler thread " << tid << "\n";
                            if (res->idx + index > THREADS_PER_BLOCK * BLOCKS) {
                                //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                holdonto = res;
                                break;
                            }
                            for (int i = 0; i < res->idx; i++) {
                                data->keys_h[index + i] = res->hashes[i];
                                data->requests_h[index + i] = res->requests[i];
                                uint8_t loc;
                                uint32_t loc_hash;
                                megakv::Bucket<K, V> *b = secondaryIndex->alloc(loc, loc_hash);

                                std::pair<K, V> p = {res->keys[i], res->values[i]};
                                (*b).set(loc, p);
                                data->values_h[index + i] = loc_hash;
                            }

                            writeBack.push_back({index, res});
                            index += res->idx;
                            if (res->flush) {
                                break;
                            }
                        } else {
                            attempts++;
                        }
                    }

                    if (index > 0) {

                        //std::cerr << "Batching " << tid << "\n";

                        auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                        float t;

                        data->moveToGPU(stream);
                        this->slabs->exec_async(data->keys_k, data->values_k, data->requests_k, stream);
                        data->moveValuesBack(stream);
                        gpuErrchk(cudaStreamSynchronize(stream));

                        auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                        int timesGoingToCache = 0;
                        for (auto &wb : writeBack) {

                            int rbLoc = wb.second->resBufStart;

                            for (int i = 0; i < wb.second->idx; ++i) {

                                if (wb.second->handleInCache[i]) {
                                    timesGoingToCache++;
                                    auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                *(this->model));
                                    if (cacheRes.first->valid == 1) {
                                        wb.second->resBuf->resultValues[rbLoc + i] = cacheRes.first->value;
                                    } else {
                                        cacheRes.first->valid = 1;
                                        uint32_t val = data->values_h[wb.first +
                                                                      i];
                                        std::pair<K, V> p;
                                        megakv::Bucket<K, V> *b = secondaryIndex->getBucket(val);
                                        b->get(val & 0xFF, p);

                                        if (wb.second->keys[i] == p.first) {
                                            cacheRes.first->value = p.second;
                                            cacheRes.first->deleted = 0;
                                            wb.second->resBuf->resultValues[rbLoc + i] = p.second;
                                        } else {
                                            cacheRes.first->value = EMPTY<V>::value;
                                            cacheRes.first->deleted = 1;
                                            wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                        }
                                    }
                                    asm volatile("":: : "memory");

                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                } else {

                                    uint32_t val = data->values_h[wb.first +
                                                                  i];
                                    std::pair<K, V> p;
                                    megakv::Bucket<K, V> *b = secondaryIndex->getBucket(val);
                                    b->get(val & 0xFF, p);

                                    if (wb.second->keys[i] == p.first) {
                                        wb.second->resBuf->resultValues[rbLoc + i] = p.second;
                                    } else {
                                        wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                    }

                                    asm volatile("":: : "memory");
                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                }
                            }
                            delete wb.second;
                        }

                        mops[tid].push_back(
                                {std::chrono::high_resolution_clock::now(), timestampWriteBack, timestampStartBatch,
                                 timestampWriteToBatch, t, index, timesGoingToCache});

                        ops += index;
                        //std::cerr << "Batched " << tid << "\n";

                    }
                }
                gpuErrchk(cudaStreamDestroy(stream));
            }, i));
        }
    }

    ~MegaKVSlabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete slabs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    megakv::MegaKVGPU *slabs;
    megakv::SecondaryIndex<K, V> *secondaryIndex;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};

template<typename K, typename M>
struct MegaKVSlabs<K, data_t *, M> {

    size_t SizeLog2 = 8;

    using V = data_t *;
    using VType = data_t *;

    MegaKVSlabs() = delete;

    typedef tbb::concurrent_queue<BatchData<K, data_t> *> q_t;

    MegaKVSlabs(const std::vector<PartitionedSlabUnifiedConfig> &config,
                std::shared_ptr<typename Cache<K, V>::type> cache,
                std::shared_ptr<M> m) : numslabs(1), slabs(new megakv::MegaKVGPU(config[0].size)),
                                        secondaryIndex(new megakv::SecondaryIndex<K, V>(SizeLog2)),
                                        gpu_qs(new q_t[1]), done(false),
                                        mops(new tbb::concurrent_vector<StatData>[config.size()]), _cache(cache),
                                        ops(0),
                                        load(0),
                                        model(m) {

        for (int i = 0; i < config.size(); ++i) {
            threads.push_back(std::thread([this](int tid) {
                cudaStream_t stream;
                gpuErrchk(cudaStreamCreate(&stream));
                auto gpucpumem = std::make_unique<megakv::GPUData>();

                for (int i = 0; i < megakv::THREADS_PER_BLOCK * megakv::BLOCKS; i++) {
                    gpucpumem->requests_h[i] = megakv::EMPTY;
                }

                BatchData<K, data_t> *holdonto = nullptr;

                std::vector<std::pair<int, BatchData<K, data_t> *>> writeBack;
                writeBack.reserve(megakv::THREADS_PER_BLOCK * megakv::BLOCKS / 512);

                int index = megakv::THREADS_PER_BLOCK * megakv::BLOCKS;
                while (!done.load()) {
                    writeBack.clear();
                    for (int i = 0; i < index; i++) {
                        gpucpumem->requests_h[i] = REQUEST_EMPTY;
                    }
                    index = 0;

                    BatchData<K, data_t> *res;

                    auto timestampWriteToBatch = std::chrono::high_resolution_clock::now();

                    if (holdonto) {
                        //std::cerr << "Hold onto set " << tid << std::endl;
                        writeBack.push_back({index, holdonto});

                        for (int i = 0; i < holdonto->idx; i++) {
                            gpucpumem->keys_h[index + i] = holdonto->hashes[i];
                            gpucpumem->requests_h[index + i] = holdonto->requests[i];
                            uint8_t loc;
                            uint32_t loc_hash;
                            megakv::Bucket<K, V> *b = secondaryIndex->alloc(loc, loc_hash);

                            std::pair<K, V> p = {holdonto->keys[i], holdonto->values[i]};
                            (*b).set(loc, p);
                            gpucpumem->values_h[index + i] = loc_hash;
                        }
                        index += holdonto->idx;
                        holdonto = nullptr;
                    }

                    int attempts = 0;

                    while (attempts < MAX_ATTEMPTS && index < megakv::THREADS_PER_BLOCK * megakv::BLOCKS) {
                        if (this->gpu_qs->try_pop(res)) {
                            load--;
                            //std::cerr << "Got a batch on handler thread " << tid << "\n";
                            if (res->idx + index > megakv::THREADS_PER_BLOCK * megakv::BLOCKS) {
                                //std::cerr << "Cannot add any more to batch " << tid << "\n";
                                holdonto = res;
                                break;
                            }
                            for (int i = 0; i < res->idx; i++) {
                                gpucpumem->keys_h[index + i] = res->hashes[i];
                                gpucpumem->requests_h[index + i] = res->requests[i];
                                if (res->requestID[i] == REQUEST_INSERT) {
                                    uint8_t loc;
                                    uint32_t loc_hash;
                                    megakv::Bucket<K, V> *b = secondaryIndex->alloc(loc, loc_hash);

                                    std::pair<K, V> p = {res->keys[i], res->values[i]};
                                    (*b).set(loc, p);
                                    gpucpumem->values_h[index + i] = loc_hash;
                                }
                            }

                            writeBack.push_back({index, res});
                            index += res->idx;
                            if (res->flush) {
                                break;
                            }
                        } else {
                            attempts++;
                        }
                    }

                    if (index > 0) {

                        //std::cerr << "Batching " << tid << "\n";

                        cudaEvent_t start, end;
                        gpuErrchk(cudaEventCreate(&start));
                        gpuErrchk(cudaEventCreate(&end));

                        auto timestampStartBatch = std::chrono::high_resolution_clock::now();

                        float t;

                        gpucpumem->moveToGPU(stream);
                        gpuErrchk(cudaEventRecord(start));
                        this->slabs->exec_async(gpucpumem->keys_k, gpucpumem->values_k, gpucpumem->requests_k, stream);
                        gpuErrchk(cudaEventRecord(end));
                        gpucpumem->moveValuesBack(stream);
                        gpuErrchk(cudaStreamSynchronize(stream));
                        gpuErrchk(cudaEventElapsedTime(&t, start, end));

                        auto timestampWriteBack = std::chrono::high_resolution_clock::now();
                        int timesGoingToCache = 0;
                        std::vector<uint32_t> toDel;
                        for (auto &wb : writeBack) {

                            int rbLoc = wb.second->resBufStart;

                            for (int i = 0; i < wb.second->idx; ++i) {

                                if (wb.second->handleInCache[i]) {
                                    timesGoingToCache++;
                                    auto cacheRes = _cache->get(wb.second->keys[i], wb.second->hashes[i],
                                                                *(this->model));
                                    if (cacheRes.first->valid == 1) {

                                        data_t* cpy = nullptr;
                                        if(cacheRes.first->value){
                                            cpy = new data_t(cacheRes.first->value->size);
                                            memcpy(cpy->data, cacheRes.first->value->data, cacheRes.first->value->size);
                                        }

                                        wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                    } else {
                                        cacheRes.first->valid = 1;
                                        uint32_t val = gpucpumem->values_h[wb.first +
                                                                           i];
                                        std::pair<K, V> p;
                                        megakv::Bucket<K, V> *b = secondaryIndex->getBucket(val);
                                        b->get(val & 0xFF, p);

                                        if (wb.second->keys[i] == p.first) {
                                            if (p.second) {
                                                cacheRes.first->value = p.second;
                                                cacheRes.first->deleted = 0;
                                                data_t *cpy = new data_t(p.second->size);
                                                memcpy(cpy->data, p.second->data, p.second->size);
                                                wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                            } else {
                                                cacheRes.first->value = EMPTY<V>::value;
                                                cacheRes.first->deleted = 1;
                                                wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                            }
                                        } else {
                                            cacheRes.first->value = EMPTY<V>::value;
                                            cacheRes.first->deleted = 1;
                                            wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                        }
                                    }
                                    asm volatile("":: : "memory");

                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];

                                } else {

                                    uint32_t val = gpucpumem->values_h[wb.first +
                                                                       i];
                                    std::pair<K, V> p;
                                    megakv::Bucket<K, V> *b = secondaryIndex->getBucket(val);
                                    b->get(val & 0xFF, p);

                                    if (wb.second->keys[i] == p.first) {
                                        if (p.second) {
                                            data_t *cpy = new data_t(p.second->size);
                                            memcpy(cpy->data, p.second->data, p.second->size);
                                            wb.second->resBuf->resultValues[rbLoc + i] = cpy;
                                        } else {
                                            wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                        }
                                        if (wb.second->requests[i] == REQUEST_REMOVE ||
                                            wb.second->requests[i] == REQUEST_INSERT) {
                                            toDel.push_back(val);
                                        }
                                    } else {
                                        wb.second->resBuf->resultValues[rbLoc + i] = EMPTY<V>::value;
                                    }

                                    asm volatile("":: : "memory");
                                    wb.second->resBuf->requestIDs[rbLoc + i] = wb.second->requestID[i];
                                }
                            }
                            delete wb.second;
                        }
                        for (auto &d : toDel) {
                            megakv::Bucket<K, V> *b = secondaryIndex->getBucket(d);
                            b->free(d & 0xFF);
                        }

                        gpuErrchk(cudaEventDestroy(start));
                        gpuErrchk(cudaEventDestroy(end));

                        mops[tid].push_back(
                                {std::chrono::high_resolution_clock::now(), timestampWriteBack, timestampStartBatch,
                                 timestampWriteToBatch, t, index, timesGoingToCache});

                        ops += index;
                        //std::cerr << "Batched " << tid << "\n";

                    }
                }
                gpuErrchk(cudaStreamDestroy(stream));
            }, i));
        }
    }

    ~MegaKVSlabs() {
        std::cerr << "Slabs deleted\n";
        done = true;
        for (auto &t : threads) {
            if (t.joinable())
                t.join();
        }
        delete[] gpu_qs;
        delete slabs;
        delete[] mops;
    }

    void clearMops() {
        for (int i = 0; i < numslabs; i++) {
            mops[i].clear();
        }
        ops = 0;
    }

    size_t getOps() {
        return ops;
    }

    int numslabs;
    megakv::MegaKVGPU *slabs;
    megakv::SecondaryIndex<K, V> *secondaryIndex;
    q_t *gpu_qs;
    std::vector<std::thread> threads;
    std::atomic_bool done;
    tbb::concurrent_vector<StatData> *mops;
    std::shared_ptr<typename Cache<K, V>::type> _cache;
    std::atomic_size_t ops;
    std::atomic_int load;
    std::shared_ptr<M> model;
};

template<typename K, typename V, typename M, bool MegaKV>
class KVStore {
public:

    using Slab_t = typename std::conditional<MegaKV, MegaKVSlabs<K, V, M>, Slabs<K, V, M>>::type;

    KVStore() : cache(std::make_shared<typename Cache<K, V>::type>()), model(new M()) {
        slab = std::make_shared<Slab_t>(STANDARD_CONFIG, this->cache, model);
    }

    KVStore(const std::vector<PartitionedSlabUnifiedConfig> &conf) : cache(
            std::make_shared<typename Cache<K, V>::type>()), model(new M()) {
        if (!conf.empty())
            slab = std::make_shared<Slab_t>(conf, this->cache, model);
    }

    KVStore(const KVStore<K, V, M, MegaKV> &other) : slab(other.slab), cache(other.cache), model(other.model) {

    }

    ~KVStore() {

    }

    std::shared_ptr<Slab_t> getSlab() {
        return slab;
    }

    std::shared_ptr<typename Cache<K, V>::type> getCache() {
        return cache;
    }

    std::shared_ptr<M> getModel() {
        return model;
    }

private:

    std::shared_ptr<Slab_t> slab;
    std::shared_ptr<typename Cache<K, V>::type> cache;
    std::shared_ptr<M> model;
};

#endif //KVGPU_KVSTORE_CUH
