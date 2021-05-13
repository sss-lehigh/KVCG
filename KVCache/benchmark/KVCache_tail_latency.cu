#include <KVCache.cuh>
#include <vector>
#include <thread>
#include <chrono>
#include <iostream>
#include <cassert>
#include <random>

using cache_t = kvgpu::KVCache<unsigned long long, unsigned long long, 10000000>;

struct barrier_t {
    std::condition_variable cond;
    std::mutex mtx;
    int count;
    int crossing;

    barrier_t(int n) : count(n), crossing(0) {}

    void wait() {
        std::unique_lock<std::mutex> ulock(mtx);
        /* One more thread through */
        crossing++;
        /* If not all here, wait */
        if (crossing < count) {
            cond.wait(ulock);
        } else {
            cond.notify_all();
            /* Reset for next time */
            crossing = 0;
        }
    }
};

template<>
__host__ __device__ __forceinline__ unsigned
compare<unsigned long long>(const unsigned long long &x, const unsigned long long &y) {
    return x - y;
}

double thread_func(barrier_t &b, int ops, int range,
                   cache_t &cache, std::vector<int> &foundValid, std::vector<int> &searchTotal, int tid) {

    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned long long> distribution(1, range);

    std::uniform_int_distribution<int> readWrite(0, 99);

    std::vector<unsigned long long> keys(ops);
    for (int i = 0; i < ops; i++) {
        keys[i] = distribution(generator);
    }

    kvgpu::SimplModel<unsigned long long> m;
    b.wait();
    auto start = std::chrono::high_resolution_clock::now();
    for (auto &k : keys) {
        if (readWrite(generator) <= 95) {
            auto ret = cache.fast_get(k, k, m);
            if (ret.first && ret.first->valid == 1) {
                foundValid[tid] = foundValid[tid] + 1;
            }
            searchTotal[tid] = searchTotal[tid] + 1;
        } else {
            size_t log_loc;

            auto ret = cache.get_with_log(k, k, m, log_loc);
            if (ret.first) {
                ret.first->valid = 1;
                ret.first->value = 4;
                foundValid[tid] = foundValid[tid] + 1;
            }
            searchTotal[tid] = searchTotal[tid] + 1;

        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char **argv) {
    int nthreads = 8;

    if (argc > 1) {
        nthreads = atoi(argv[1]);
    }

    int ops = 512;

    cache_t cache;
    std::vector<std::thread> threads;

    barrier_t b(nthreads + 1);

    std::vector<int> foundValid(nthreads);
    std::vector<int> searchTotal(nthreads);
    std::vector<double> times(nthreads);

    int range = 1000000;

    for (int i = 0; i < range; i++) {
        kvgpu::SimplModel<unsigned long long> m;
        size_t log_loc;
        auto ret = cache.get_with_log(i, i, m, log_loc);
        assert(ret.first != nullptr);
        ret.first->valid = 1;
        ret.first->key = (unsigned long long) i;
        ret.first->value = 2;
    }

    for (int i = 0; i < nthreads; i++) {
        threads.push_back(std::thread([&](int tid) {
            times[tid] = thread_func(b, ops, range, cache, foundValid, searchTotal, tid);
        }, i));
    }

    b.wait();

    for (auto &t : threads) {
        t.join();
    }

    for (auto &t : times) {
        std::cout << "Tail latency " << t * 1e3 << " ms" << std::endl;
    }

}