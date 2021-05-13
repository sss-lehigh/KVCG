//
// Created by depaulsmiller on 8/28/20.
//

#include <utility>
#include "KVStoreInternalClient.cuh"
#include <memory>
#include <atomic>
#include <exception>
#include "KVStoreCtx.cuh"

#ifndef KVGPU_KVSTORECLIENT_CUH
#define KVGPU_KVSTORECLIENT_CUH

template<typename K, typename V, typename M>
class GeneralClient {
public:
    GeneralClient() {}

    virtual ~GeneralClient() {}

    virtual void batch(std::vector<RequestWrapper<K, typename SubstituteType<V>::type>> &req_vector,
                       std::shared_ptr<ResultsBuffers<V>> resBuf,
                       std::vector<std::chrono::high_resolution_clock::time_point> &times) = 0;

    virtual float hitRate() = 0;

    virtual void resetStats() = 0;

    virtual size_t getHits() = 0;

    virtual size_t getOps() = 0;

    virtual void stat() = 0;

    virtual std::future<void> change_model(M &newModel, block_t *block, double &time) = 0;

    virtual M getModel() = 0;

};

template<typename K, typename V, typename M, typename Slab_t, bool UseCache = true, bool UseGPU = true>
struct ChooseInternalClient {
    static_assert(UseGPU || UseCache, "Need to use GPU or Use Cache");
    using type = KVStoreInternalClient<K, V, M, Slab_t>;

    template<bool MegaKV>
    static inline std::unique_ptr<type> getClient(KVStoreCtx<K, V, M, MegaKV>& ctx) {
        return ctx.getClient();
    };
};

template<typename K, typename V, typename M, typename Slab_t>
struct ChooseInternalClient<K, V, M, Slab_t, false, true> {
    using type = NoCacheKVStoreInternalClient<K, V, M, Slab_t>;

    template<bool MegaKV>
    static inline std::unique_ptr<type> getClient(KVStoreCtx<K, V, M, MegaKV>& ctx) {
        return ctx.getNoCacheClient();
    };

};

// true, true and false, true

template<typename K, typename V, typename M, typename Slab_t>
struct ChooseInternalClient<K, V, M, Slab_t, true, false> {
    using type = JustCacheKVStoreInternalClient<K, V, M, Slab_t>;

    template<bool MegaKV>
    static inline std::unique_ptr<type> getClient(KVStoreCtx<K, V, M, MegaKV>& ctx) {
        return ctx.getJustCacheClient();
    };

};

template<typename K, typename V, typename M, bool UseCache = true, bool UseGPU = true, bool MegaKV = DEFAULT_MEGAKV>
class KVStoreClient : public GeneralClient<K, V, M> {
private:
    using Slab_t = typename KVStoreCtx<K, V, M, MegaKV>::Slab_t;
    using ChooseType = ChooseInternalClient<K, V, M, Slab_t, UseCache, UseGPU>;
    using InternalType = typename ChooseType::type;
public:

    KVStoreClient() = delete;

    explicit KVStoreClient(KVStoreCtx<K, V, M, MegaKV> ctx) : client(std::move(ChooseType::getClient(ctx))){
    }

    KVStoreClient(const KVStoreClient<K, V, M, UseCache, UseGPU, MegaKV> &) = delete;

    KVStoreClient(KVStoreClient<K, V, M, UseCache, UseGPU, MegaKV> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~KVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, typename SubstituteType<V>::type>> &req_vector,
               std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        return client->change_model(newModel, block, time);
    }

    M getModel() {
        return client->getModel();
    }

private:
    std::unique_ptr<InternalType> client;
};

template<typename K, typename V, typename M>
using NoCacheKVStoreClient = KVStoreClient<K, V, M, false, true, DEFAULT_MEGAKV>;

template<typename K, typename V, typename M>
using JustCacheKVStoreClient = KVStoreClient<K, V, M, true, false, DEFAULT_MEGAKV>;

/*template<typename K, typename V, typename M, bool MegaKV = DEFAULT_MEGAKV>
class NoCacheKVStoreClient : public GeneralClient<K, V, M> {
private:
    using Slab_t = typename KVStoreCtx<K, V, M, MegaKV>::Slab_t;
public:

    NoCacheKVStoreClient() = delete;

    explicit NoCacheKVStoreClient(KVStoreCtx<K, V, M, MegaKV> ctx) : client(std::move(ctx.getNoCacheClient())) {

    }

    NoCacheKVStoreClient(const NoCacheKVStoreClient<K, V, M, MegaKV> &) = delete;

    NoCacheKVStoreClient(NoCacheKVStoreClient<K, V, M, MegaKV> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~NoCacheKVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, typename SubstituteType<V>::type>> &req_vector,
               std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        return client->change_model(newModel, block, time);
    }

    M getModel() {
        return client->getModel();
    }

private:
    std::unique_ptr<NoCacheKVStoreInternalClient<K, V, M, Slab_t>> client;
};*/

/*template<typename K, typename V, typename M, bool MegaKV = DEFAULT_MEGAKV>
class JustCacheKVStoreClient : public GeneralClient<K, V, M> {
private:
    using Slab_t = typename KVStoreCtx<K, V, M, MegaKV>::Slab_t;
public:

    JustCacheKVStoreClient() = delete;

    explicit JustCacheKVStoreClient(KVStoreCtx<K, V, M, MegaKV> ctx) : client(std::move(ctx.getJustCacheClient())) {

    }

    JustCacheKVStoreClient(const JustCacheKVStoreClient<K, V, M, MegaKV> &) = delete;

    JustCacheKVStoreClient(JustCacheKVStoreClient<K, V, M, MegaKV> &&other) {
        client = std::move(other.client);
        other.client = nullptr;
    }


    ~JustCacheKVStoreClient() {

    }

    void batch(std::vector<RequestWrapper<K, typename SubstituteType<V>::type>> &req_vector,
               std::shared_ptr<ResultsBuffers<V>> resBuf,
               std::vector<std::chrono::high_resolution_clock::time_point> &times) {
        client->batch(req_vector, resBuf, times);
    }

    float hitRate() {
        return client->hitRate();
    }

    void resetStats() {
        client->resetStats();
    }

    size_t getHits() {
        return client->getHits();
    }

    size_t getOps() {
        return client->getOps();
    }

    void stat() {
        return client->stat();
    }

    std::future<void> change_model(M &newModel, block_t *block, double &time) {
        return client->change_model(newModel, block, time);
    }

    M getModel() {
        return client->getModel();
    }

private:
    std::unique_ptr<JustCacheKVStoreInternalClient<K, V, M, Slab_t>> client;
};*/

#endif //KVGPU_KVSTORECLIENT_CUH
