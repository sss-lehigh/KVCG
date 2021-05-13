//
// Created by depaulsmiller on 8/28/20.
//

#include <zipf.hh>
#include <iostream>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <kvcg.cuh>
#include <groupallocator>
#include <set>
#include <algorithm>

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

std::vector<RequestWrapper<unsigned, unsigned>> generateWorkloadZipf(int size, double theta, int n, double zetaN, unsigned *seed, int ratioOfReads);

std::vector<RequestWrapper<unsigned, unsigned>> generateWorkloadRand(int size, int range, unsigned *seed, int ratioOfReads);

std::vector<RequestWrapper<data_t *, data_t *>> generateWorkloadZipfAPP(double theta, int n, double zetaN, unsigned *seed, int ratioOfReads);

void populateUnsignedUnsigned(JustCacheKVStoreClient<unsigned, unsigned, kvgpu::SimplModel<unsigned>> &client, unsigned *seed, int n, int range);

std::vector<char> toBase256(unsigned x);

const int BATCHSIZE = 512;
const int NUM_THREADS = 24;

int main() {
    JustCacheKVStoreCtx<unsigned, unsigned, kvgpu::SimplModel<unsigned>> ctx;

    const double theta = 0.2;
    const int n = 100000;
    const double zetaN = betterstd::zeta(theta, n);
    unsigned seed = time(nullptr);

    JustCacheKVStoreClient<unsigned, unsigned, kvgpu::SimplModel<unsigned>> client(ctx);

    populateUnsignedUnsigned(client, &seed, 10000, 2 * n);

    std::cerr << "Populated\n";

    client.resetStats();

    std::atomic_bool done(false);

    std::vector<std::thread> threads;
    tbb::concurrent_vector<double> times;

    barrier_t *barrier = new barrier_t(NUM_THREADS + 1);

    int seconds = 10;

    for (int i = 0; i < NUM_THREADS; ++i) {
        threads.push_back(std::thread([&done, &client, &barrier, &theta, &zetaN, &times, &seconds]() {

            unsigned threadSeed = time(nullptr);

            std::vector<double> tmpTimes;

            barrier->wait();

            while (!done.load()) {
                auto req = generateWorkloadZipf(BATCHSIZE, theta, 2 * n, zetaN, &threadSeed, 95);
                //std::cerr << "Generated workload\n";
                double rt = 0;
                //std::cerr << "Batching\n";
                client.batch(req, rt);
                //std::cerr << "Batched\n";
                tmpTimes.push_back(rt);
            }
            for (auto &t : tmpTimes) {
                times.push_back(t);
            }
        }));
    }

    barrier->wait();

    sleep(seconds);
    done = true;
    //std::cerr << "Awake and joining\n";
    for (auto &t : threads) {
        t.join();
    }

    std::cout << "TABLE: Max latency of Hot Storage" << std::endl;
    for (auto &t : times) {
        std::cout << t << std::endl;
    }
    std::cout << std::endl;


    std::cout << "Max time: " << *std::max_element(times.begin(), times.end()) << std::endl;
    std::cout << "Min time: " << *std::min_element(times.begin(), times.end()) << std::endl;
    std::cout << std::endl;

    client.stat();

    std::cout << "Throughput (Mops) " << client.hitRate() * BATCHSIZE * times.size() / seconds / 1e6 << std::endl;

    /*std::cout << "Hit Rate\tHits" << std::endl;
    std::cout << client.hitRate() << "\t" << client.getHits() << std::endl;
    std::cout << std::endl; */


    return 0;
}

std::vector<char> toBase256(unsigned x) {

    std::vector<char> cvec;

    while (x != 0) {
        unsigned mod = x % 256;
        x = x / 256;
        cvec.insert(cvec.begin(), mod);
    }
    return cvec;
}

std::vector<RequestWrapper<unsigned, unsigned>> generateWorkloadZipf(int size, double theta, int n, double zetaN, unsigned *seed, int ratioOfReads) {

    std::vector<RequestWrapper<unsigned, unsigned >> vec;

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            vec.push_back({(unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta), 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_GET});
        } else {
            if (rand_r(seed) % 100 < 50) {
                vec.push_back({(unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta), 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_INSERT});
            } else {
                vec.push_back({(unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta), 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_REMOVE});
            }
        }

    }
    return vec;

}

std::vector<RequestWrapper<unsigned, unsigned>> generateWorkloadRand(int size, int range, unsigned *seed, int ratioOfReads) {

    std::vector<RequestWrapper<unsigned, unsigned >> vec;

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            vec.push_back({(unsigned) rand_r(seed) % range, 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_GET});
        } else {
            if (rand_r(seed) % 100 < 50) {
                vec.push_back({(unsigned) rand_r(seed) % range, 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_INSERT});
            } else {
                vec.push_back({(unsigned) rand_r(seed) % range, 1, SimplePromise<std::pair<bool, unsigned>>(), SimplePromise<bool>(), REQUEST_REMOVE});
            }
        }

    }
    return vec;

}

void populateUnsignedUnsigned(JustCacheKVStoreClient<unsigned, unsigned, kvgpu::SimplModel<unsigned>> &client, unsigned *seed, int n, int range) {

    std::set<unsigned> keys;
    while (keys.size() < n) {
        if (keys.size() % 1000 == 0) {
            std::cerr << "So far have " << keys.size() << " unique keys" << std::endl;
        }
        keys.insert(rand_r(seed) % range + 1);
    }


    std::vector<RequestWrapper<unsigned, unsigned >> vec;
    std::vector<SimpleFuture<bool>> fut;

    for (auto &k : keys) {
        auto prom = SimplePromise<bool>();
        fut.push_back(prom.getFuture());
        vec.push_back({k, 1, SimplePromise<std::pair<bool, unsigned>>(), std::move(prom), REQUEST_INSERT});
    }
    while (vec.size() % 512 != 0) {
        RequestWrapper<unsigned, unsigned> rw;
        rw.requestInteger = REQUEST_EMPTY;
        vec.push_back(rw);
    }

    double rt;
    client.batch(vec, rt);

}


std::vector<RequestWrapper<data_t *, data_t *>> generateWorkloadZipfAPP(double theta, int n, double zetaN, unsigned *seed, int ratioOfReads) {

    using namespace groupallocator;

    std::vector<RequestWrapper<data_t *, data_t *>> vec;

    Context ctx;

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS * 2; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {

            auto data_vec = toBase256((unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta));

            data_t *d;
            allocate(&d, sizeof(data_t), ctx);
            char *underlyingData;
            allocate(&underlyingData, sizeof(char) * 40, ctx);

            d->size = 40;
            d->data = underlyingData;

            int k = 0;
            for (; k < data_vec.size(); ++k) {
                underlyingData[k] = data_vec[k];
            }
            for (; k < 40; ++k) {
                underlyingData[k] = '\0';
            }

            vec.push_back({d, new data_t(), SimplePromise<std::pair<bool, data_t *>>(), SimplePromise<bool>(), REQUEST_GET});
        } else {
            if (rand_r(seed) % 100 < 50) {

                auto data_vec = toBase256((unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta));

                data_t *d;
                allocate(&d, sizeof(data_t), ctx);
                char *underlyingData;
                allocate(&underlyingData, sizeof(char) * 40, ctx);

                d->size = 40;
                d->data = underlyingData;

                int k = 0;
                for (; k < data_vec.size(); ++k) {
                    underlyingData[k] = data_vec[k];
                }
                for (; k < 40; ++k) {
                    underlyingData[k] = '\0';
                }

                vec.push_back({d, new data_t(), SimplePromise<std::pair<bool, data_t *>>(), SimplePromise<bool>(), REQUEST_INSERT});
            } else {

                auto data_vec = toBase256((unsigned) betterstd::rand_zipf_r(seed, n, zetaN, theta));

                data_t *d;
                allocate(&d, sizeof(data_t), ctx);
                char *underlyingData;
                allocate(&underlyingData, sizeof(char) * 40, ctx);

                d->size = 40;
                d->data = underlyingData;

                int k = 0;
                for (; k < data_vec.size(); ++k) {
                    underlyingData[k] = data_vec[k];
                }
                for (; k < 40; ++k) {
                    underlyingData[k] = '\0';
                }

                vec.push_back({d, new data_t(), SimplePromise<std::pair<bool, data_t *>>(), SimplePromise<bool>(), REQUEST_REMOVE});
            }
        }

    }
    return vec;

}
