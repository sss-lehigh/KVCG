//
// Created by depaulsmiller on 9/9/20.
//

#include <chrono>
#include <algorithm>
#include <vector>
#include <iostream>
#include <functional>
#include <unordered_map>
#include <set>
#include <atomic>
#include <thread>

int main() {

    int size = 136 * 512;

    std::vector<std::pair<unsigned, unsigned>> vec;
    vec.reserve(size);
    for (int i = 0; i < size; i++) {
        vec.push_back({rand(), 1});
    }


    std::vector<std::pair<unsigned, unsigned>> vec2;
    vec2.reserve(size);

    float *prob = new float[10000];

    std::atomic_bool go{false};
    std::atomic_int done{0};
    std::vector<std::thread> threads;
    int num_threads = 12;
    threads.reserve(num_threads);
    std::atomic_int caught;
    for (int i = 0; i < num_threads; i++) {
        threads.push_back(std::thread([&go, &done, &prob, &caught](std::pair<unsigned, unsigned> *data, int startidx, int endidx) {

            while (!go);

            int tmp = 0;

            for (int i = startidx; i < endidx; i++) {
                if (prob[data[i].first % 10000] < 0.01)
                    data[i].second = 1;
            }

            //caught.fetch_add(tmp);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            done++;
            //std::cerr << tmp << std::endl;
        }, vec.data(), i * size / num_threads, i * size / num_threads + size / num_threads));
    }

    auto start = std::chrono::high_resolution_clock::now();

    go = true;
    while (done != num_threads);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;

    for(auto& t : threads){
        t.join();
    }

    std::cerr << dur.count() * 1e3 << " ms" << std::endl;

    std::cerr << vec.size() / dur.count() / 1e6 << "Mops" << std::endl;

}