//
// Created by depaulsmiller on 9/2/20.
//

#include "KVStoreInternalClient.cuh"

#ifndef KVGPU_KVSTORECTX_CUH
#define KVGPU_KVSTORECTX_CUH

#ifdef USE_MEGAKV
#define DEFAULT_MEGAKV true
#else
#define DEFAULT_MEGAKV false
#endif

template<typename K, typename V, typename M, bool MegaKV = DEFAULT_MEGAKV>
class KVStoreCtx {
private:
    using KVStore_t = KVStore<K, typename SubstituteType<V>::type, M, MegaKV>;
public:

    using Slab_t = typename KVStore_t::Slab_t;

    KVStoreCtx() : k() {

    }

    KVStoreCtx(const std::vector<PartitionedSlabUnifiedConfig> &conf) : k(conf) {

    }


    ~KVStoreCtx() {}

    std::unique_ptr<KVStoreInternalClient<K, V, M, Slab_t>> getClient() {
        return std::make_unique<KVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel());
    }

    std::unique_ptr<NoCacheKVStoreInternalClient<K, V, M, Slab_t>> getNoCacheClient() {
        return std::make_unique<NoCacheKVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel());
    }

    std::unique_ptr<JustCacheKVStoreInternalClient<K, V, M, Slab_t>> getJustCacheClient() {
        return std::make_unique<JustCacheKVStoreInternalClient<K, V, M, Slab_t>>(k.getSlab(), k.getCache(), k.getModel());
    }

private:
    KVStore_t k;
};

#endif //KVGPU_KVSTORECTX_CUH
