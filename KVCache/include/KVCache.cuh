//
// Created by depaulsmiller on 8/28/20.
//

#ifndef KVGPU_KVCACHE_CUH
#define KVGPU_KVCACHE_CUH

#include <mutex>
#include <atomic>
#include <functional>
#include <iostream>
#include <shared_mutex>
#include <ImportantDefinitions.cuh>
#include <tbb/concurrent_vector.h>
#include <immintrin.h>

namespace kvgpu {

    /*template<typename T>
    unsigned compare(T& rhs, T& lhs);

    template<>
    unsigned compare(unsigned& rhs, unsigned& lhs){
        return rhs - lhs;
    }*/


    using mutex = std::shared_mutex;
    typedef std::unique_lock<mutex> locktype;
    typedef std::shared_lock<mutex> sharedlocktype;

    template<typename K, typename V>
    struct LockingPair {
        LockingPair() : valid(0), deleted(0) {}

        ~LockingPair() {}

        char padding[40];
        unsigned long valid;
        unsigned long deleted;
        K key;
        V value;
    };

    template<typename K>
    struct Model {
        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        virtual bool operator()(K key, unsigned hash) const = 0;
    };

    template<typename K>
    struct SimplModel : public Model<K> {

        SimplModel() : value(16000) {

        }

        explicit SimplModel(int v) : value(v) {

        }

        SimplModel(const SimplModel<K> &other) {
            value = other.value;
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return hash < value;
        }

    private:
        int value;
    };

    template<typename K>
    struct AllGPUModel : public Model<K> {

        AllGPUModel() {
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return false;
        }

    };

    template<typename K>
    struct AllCPUModel : public Model<K> {

        AllCPUModel() {
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return true;
        }

    };

    template<typename K>
    struct APPModel : public Model<K> {

        APPModel() : value(100000) {

        }

        explicit APPModel(int v) : value(v) {

        }

        APPModel(const APPModel<K> &other) {
            value = other.value;
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return hash < value;
        }

    private:
        int value;
    };

    template<>
    struct APPModel<unsigned long long> : public Model<unsigned long long> {

        APPModel() : value(100000) {

        }

        explicit APPModel(int v) : value(v) {

        }

        APPModel(const APPModel<unsigned long long> &other) {
            value = other.value;
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(unsigned long long key, unsigned hash) const {
            return key <= value;
        }

    private:
        int value;
    };

    template<typename K>
    struct AnalyticalModel : public Model<K> {

        AnalyticalModel() : threshold(0.2), size(100000), pred(new std::atomic<float *>()) {
            *pred = new float[size];
            for (int i = 0; i < size; i++) {
                (*pred)[i] = 1.0 / size;
            }
        }

        explicit AnalyticalModel(float v) : threshold(v), size(100000), pred(new std::atomic<float *>()) {
            *pred = new float[size];
            for (int i = 0; i < size; i++) {
                (*pred)[i] = 1.0 / size;
            }
        }

        AnalyticalModel(const AnalyticalModel<K> &other) {
            threshold = other.threshold;
            size = other.size;
            pred = other.pred;
        }

        void setPred(float *p) {
            (*pred) = p;
        }

        /**
         * Return true if should be cached
         * @param key
         * @param hash
         * @return
         */
        bool operator()(K key, unsigned hash) const {
            return (*pred)[hash % size] >= threshold;
        }

    private:
        float threshold;
        size_t size;
        std::atomic<float *> *pred;
    };


    /**
     * KVCache caches keys and values
     * K is the key type
     * V is the value type
     * DSCaching is a data structure that is being cached by this cache
     * SETS is the number of SETs in the cache
     * N is the number of elements per set
     * @tparam K
     * @tparam V
     * @tparam DSCaching
     * @tparam SETS
     * @tparam N
     */
    template<typename K, typename V, unsigned SETS = 524288 / sizeof(LockingPair<K, V>) / 8, unsigned N = 8>
    class KVCache {
    private:
        struct Node_t {
            explicit Node_t(int startLoc) : loc(startLoc), set(new LockingPair<K, V>[N]), next(nullptr) {

                for (int j = 0; j < N; j++) {
                    set[j].valid = 0;
                }
            }

            ~Node_t() {
                delete[] set;
            }

            int loc;
            LockingPair<K, V> *set;
            std::atomic<Node_t *> next;
        };

    public:
        /**
         * Creates cache
         */
        KVCache() : log_requests(new tbb::concurrent_vector<int>(N * SETS)),
                    log_hash(new tbb::concurrent_vector<unsigned>(N * SETS)),
                    log_keys(new tbb::concurrent_vector<K>(N * SETS)),
                    log_values(new tbb::concurrent_vector<V>(N * SETS)),
                    log_size(N * SETS),
                    map(new LockingPair<K, V> *[SETS]),
                    mtx(new mutex[SETS]),
                    nodes(new std::atomic<Node_t *>[SETS]),
                    expansions(0) {
            for (int i = 0; i < SETS; i++) {
                nodes[i] = nullptr;
                map[i] = new LockingPair<K, V>[N];
                for (int j = 0; j < N; j++) {
                    std::unique_lock<mutex> ul(mtx[i]);
                    map[i][j].valid = 0;
                    map[i][j].value = 0;
                }
            }
        }

        /**
         * Removes cache
         */
        ~KVCache() {
            for (int i = 0; i < SETS; i++) {
                delete[] map[i];
            }
            delete[] map;
            delete[] nodes;
            delete[] mtx;
            delete log_requests;
            delete log_hash;
            delete log_keys;
            delete log_values;
        }


        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::pair<LockingPair<K, V> *, locktype> get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    return {&set[i], std::move(unique)};
                } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                    set[i].valid = 0;
                    firstInvalidPair = &set[i];
                }
            }
            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        return {&set[i], std::move(unique)};
                    } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                        set[i].valid = 0;
                        firstInvalidPair = &set[i];
                    }
                }
                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(N);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                node = prevNode->next;
                firstInvalidPair = &(node->set[0]);
            }

            firstInvalidPair->valid = 2;
            firstInvalidPair->key = key;

            return {firstInvalidPair, std::move(unique)};
        }

        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::pair<LockingPair<K, V> *, sharedlocktype> fast_get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            sharedlocktype sharedlock(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    return {&set[i], std::move(sharedlock)};
                }
            }
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        return {&set[i], std::move(sharedlock)};
                    }
                }
                node = node->next;
            }

            return {firstInvalidPair, std::move(std::shared_lock<std::shared_mutex>())};
        }


        std::pair<LockingPair<K, V> *, locktype>
        get_with_log(K key, unsigned hash, const Model<K> &mfn, size_t &logLoc) {
            unsigned setIdx = hash % SETS;
            LockingPair<K, V> *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            LockingPair<K, V> *firstInvalidPair = nullptr;

            for (unsigned i = 0; i < N; i++) {
                if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                    logLoc = setIdx * N + i;
                    return {&set[i], std::move(unique)};
                } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                    set[i].valid = 0;
                    firstInvalidPair = &set[i];
                }
            }
            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
                for (unsigned i = 0; i < N; i++) {
                    if (set[i].valid != 0 && compare(set[i].key, key) == 0) {
                        logLoc = setIdx * N + i;
                        return {&set[i], std::move(unique)};
                    } else if (!firstInvalidPair && (set[i].valid == 0 || !mfn(set[i].key, hash))) {
                        set[i].valid = 0;
                        firstInvalidPair = &set[i];
                    }
                }
                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(N);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                node = prevNode->next;
                firstInvalidPair = &(node->set[0]);
                logLoc = tmploc;
            }

            firstInvalidPair->valid = 2;
            firstInvalidPair->key = key;

            return {firstInvalidPair, std::move(unique)};
        }

        template<typename H>
        void scan_and_evict(const Model<K> &mfn, const H &hfn, std::unique_lock<std::mutex> modelLock) {

            for (int setIdx = 0; setIdx < SETS; setIdx++) {
                LockingPair<K, V> *set = map[setIdx];
                locktype unique(mtx[setIdx]);

                for (unsigned i = 0; i < N; i++) {
                    if (!mfn(set[i].key, hfn(set[i].key))) {
                        set[i].valid = 0;
                    }
                }
                Node_t *node = nodes[setIdx].load();
                while (node != nullptr) {
                    set = node->set;
                    for (unsigned i = 0; i < N; i++) {

                        if (!mfn(set[i].key, hfn(set[i].key))) {
                            set[i].valid = 0;
                        }
                    }
                    node = node->next;
                }
            }
        }


        constexpr size_t getN() {
            return N;
        }

        constexpr size_t getSETS() {
            return SETS;
        }

        void stat() {
            //std::cout << "Cache Expansions " << expansions << std::endl;
            //std::cout << "Footprint without expansions: " << (sizeof(*this) + sizeof(LockingPair<K,V>) * SETS * N) / 1024.0 / 1024.0 << " MB" << std::endl;
        }

        tbb::concurrent_vector<int> *log_requests;
        tbb::concurrent_vector<unsigned> *log_hash;
        tbb::concurrent_vector<K> *log_keys;
        tbb::concurrent_vector<V> *log_values;
        std::atomic_size_t log_size;

    private:
        LockingPair<K, V> **map;
        mutex *mtx;
        std::atomic<Node_t *> *nodes;
        std::atomic_size_t expansions;
    };

    /**
 * KVCache caches keys and values
 * K is the key type
 * V is the value type
 * DSCaching is a data structure that is being cached by this cache
 * SETS is the number of SETs in the cache
 * N is the number of elements per set
 * @tparam K
 * @tparam V
 * @tparam DSCaching
 * @tparam SETS
 * @tparam N
 */
    template<typename K, typename V, unsigned SETS = 524288 / sizeof(LockingPair<K, V>) / 8>
    class KVSimdCache {
        static_assert(sizeof(K) == sizeof(uint64_t), "SIMD cache needs 8B entries");
    private:

        struct Bucket {
            Bucket() {
                uint8_t *v = &valid;
                *v = 0;
                uint8_t *d = &deleted;
                *d = 0;
            }

            ~Bucket() {}

            K key[8];
            struct {
                unsigned bit: 1;
            } valid[8];

            struct {
                unsigned bit: 1;
            } deleted[8];

            V value[8];
        };


        struct Node_t {
            explicit Node_t(int startLoc) : loc(startLoc), set(), next(nullptr) {
            }

            ~Node_t() {
                delete[] set;
            }

            Bucket set;
            int loc;
            std::atomic<Node_t *> next;
        };

    public:
        /**
         * Creates cache
         */
        KVSimdCache() : log_requests(new tbb::concurrent_vector<int>(8 * SETS)),
                        log_hash(new tbb::concurrent_vector<unsigned>(8 * SETS)),
                        log_keys(new tbb::concurrent_vector<K>(8 * SETS)),
                        log_values(new tbb::concurrent_vector<V>(8 * SETS)),
                        log_size(8 * SETS),
                        map(new Bucket *[SETS]),
                        mtx(new mutex[SETS]),
                        nodes(new std::atomic<Node_t *>[SETS]),
                        expansions(0) {
            for (int i = 0; i < SETS; i++) {
                nodes[i] = nullptr;
                map[i] = new LockingPair<K, V>[8];
                for (int j = 0; j < 8; j++) {
                    std::unique_lock<mutex> ul(mtx[i]);
                    map[i][j].valid = 0;
                    map[i][j].value = 0;
                }
            }
        }

        /**
         * Removes cache
         */
        ~KVSimdCache() {
            for (int i = 0; i < SETS; i++) {
                delete[] map[i];
            }
            delete[] map;
            delete[] nodes;
            delete[] mtx;
            delete log_requests;
            delete log_hash;
            delete log_keys;
            delete log_values;
        }


        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::tuple<Bucket *, int, locktype> get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            Bucket *firstInvalidPair = nullptr;
            int firstInvalidPairIdx = 0;

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        return {set, i + 4 * k, std::move(unique)};
                    } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                        firstInvalidPair = set;
                        firstInvalidPairIdx = i + 4 * k;
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            return {set, i + 4 * k, std::move(unique)};
                        } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                            firstInvalidPair = set;
                            firstInvalidPairIdx = i + 4 * k;
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(8);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                node = prevNode->next;
                firstInvalidPair = &(node->set);
                firstInvalidPairIdx = 0;
            }

            firstInvalidPair->valid[firstInvalidPairIdx] = 2;
            firstInvalidPair->key[firstInvalidPairIdx] = key;

            return {firstInvalidPair, firstInvalidPairIdx, std::move(unique)};
        }

        /**
         * Gets a key returns {ptr, lock} if successful and {nullptr, ...} if not
         * @param key
         * @param hash
         * @return
         */
        std::tuple<Bucket *, int, locktype> fast_get(K key, unsigned hash, const Model<K> &mfn) {
            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            sharedlocktype unique(mtx[setIdx]);

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        return {set, i + 4 * k, std::move(unique)};
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            return {set, i + 4 * k, std::move(unique)};
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            return {nullptr, 0, std::move(sharedlocktype())};
        }


        std::tuple<Bucket *, int, locktype>
        get_with_log(K key, unsigned hash, const Model<K> &mfn, size_t &logLoc) {

            unsigned setIdx = hash % SETS;
            Bucket *set = map[setIdx];
            locktype unique(mtx[setIdx]);

            Bucket *firstInvalidPair = nullptr;
            int firstInvalidPairIdx = 0;

            // TODO use mfn

#pragma unroll
            for (int k = 0; k < 2; k++) {
                __m256i vec_key = _mm256_set1_epi64x(key);
                __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                uint64_t results[4];
                _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                        // found
                        logLoc = setIdx * 8 + (i + 4 * k);
                        return {set, i + 4 * k, std::move(unique)};
                    } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                        firstInvalidPair = set;
                        firstInvalidPairIdx = i + 4 * k;
                    }
                }
            }

            Node_t *prevNode = nullptr;
            Node_t *node = nodes[setIdx].load();
            while (node != nullptr) {
                set = node->set;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    __m256i vec_key = _mm256_set1_epi64x(key);
                    __m256i keys_found = _mm256_loadu_si256(&(set->key + 4 * k));
                    __m256i result = _mm256_cmpeq_epi64(keys_found, vec_key); // if not equal it is 0x0
                    uint64_t results[4];
                    _mm256_storeu_si256((__m256i *) (results), result);
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        if (results[i] != 0x0 && set->valid[i + 4 * k]) {
                            // found
                            logLoc = setIdx * 8 + (i + 4 * k);
                            return {set, i + 4 * k, std::move(unique)};
                        } else if (!firstInvalidPair && !(set->valid[i + 4 * k])) {
                            firstInvalidPair = set;
                            firstInvalidPairIdx = i + 4 * k;
                        }
                    }
                }

                prevNode = node;
                node = node->next;
            }

            if (!firstInvalidPair) {
                int tmploc = log_size.fetch_add(8);
                log_requests->grow_to_at_least(log_size);
                log_hash->grow_to_at_least(log_size);
                log_keys->grow_to_at_least(log_size);
                log_values->grow_to_at_least(log_size);
                prevNode->next = new Node_t(tmploc);
                expansions++;
                logLoc = tmploc;
                node = prevNode->next;
                firstInvalidPair = &(node->set);
                firstInvalidPairIdx = 0;
            }

            firstInvalidPair->valid[firstInvalidPairIdx] = 2;
            firstInvalidPair->key[firstInvalidPairIdx] = key;

            return {firstInvalidPair, firstInvalidPairIdx, std::move(unique)};
        }

        template<typename H>
        void scan_and_evict(const Model<K> &mfn, const H &hfn, std::unique_lock<std::mutex> modelLock) {

            for (int setIdx = 0; setIdx < SETS; setIdx++) {
                LockingPair<K, V> *set = map[setIdx];
                locktype unique(mtx[setIdx]);

                for (unsigned i = 0; i < 8; i++) {
                    if (!mfn(set[i].key, hfn(set[i].key))) {
                        set[i].valid = 0;
                    }
                }
                Node_t *node = nodes[setIdx].load();
                while (node != nullptr) {
                    set = node->set;
                    for (unsigned i = 0; i < 8; i++) {

                        if (!mfn(set[i].key, hfn(set[i].key))) {
                            set[i].valid = 0;
                        }
                    }
                    node = node->next;
                }
            }
        }


        constexpr size_t getN() {
            return 8;
        }

        constexpr size_t getSETS() {
            return SETS;
        }

        void stat() {
            //std::cout << "Cache Expansions " << expansions << std::endl;
            //std::cout << "Footprint without expansions: " << (sizeof(*this) + sizeof(LockingPair<K,V>) * SETS * N) / 1024.0 / 1024.0 << " MB" << std::endl;
        }

        tbb::concurrent_vector<int> *log_requests;
        tbb::concurrent_vector<unsigned> *log_hash;
        tbb::concurrent_vector<K> *log_keys;
        tbb::concurrent_vector<V> *log_values;
        std::atomic_size_t log_size;

    private:
        Bucket **map;
        mutex *mtx;
        std::atomic<Node_t *> *nodes;
        std::atomic_size_t expansions;
    };

}

#endif //KVGPU_KVCACHE_CUH
