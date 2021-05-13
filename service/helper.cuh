//
// Created by depaulsmiller on 9/22/20.
//

#include <iostream>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <kvcg.cuh>
#include <groupallocator>
#include <set>

#ifndef KVGPU_HELPER_CUH
#define KVGPU_HELPER_CUH

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

std::vector<char> toBase256(unsigned x) {

    std::vector<char> cvec;

    while (x != 0) {
        unsigned mod = x % 256;
        x = x / 256;
        cvec.insert(cvec.begin(), mod);
    }
    return cvec;
}

data_t *unsignedToData_t(unsigned x, size_t s) {
    using namespace groupallocator;
    Context ctx;
    auto v = toBase256(x);
    data_t *d;
    allocate(&d, sizeof(data_t), ctx);
    char *underlyingData;
    allocate(&underlyingData, sizeof(char) * s, ctx);

    d->size = s;
    d->data = underlyingData;

    int k = 0;
    for (; k < v.size(); ++k) {
        underlyingData[k] = v[k];
    }
    for (; k < s; ++k) {
        underlyingData[k] = '\0';
    }
    return d;
}
#endif //KVGPU_HELPER_CUH
