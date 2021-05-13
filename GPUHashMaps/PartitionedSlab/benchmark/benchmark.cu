//
// Created by depaulsmiller on 9/3/20.
//

#include <StandardSlabDefinitions.cuh>
#include <vector>
#include <PartitionedSlabUnified.cuh>

int main() {

    const int size = 100000;

    std::vector<unsigned> keys(2 * THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> values(2 * THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> hashes(2 * THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> requests(2 * THREADS_PER_BLOCK * BLOCKS);

    std::hash<unsigned> hfn;

    auto s = new PartitionedSlabUnified<unsigned, unsigned>({{size, 0, cudaStreamDefault},
                                                             {size, 1, cudaStreamDefault}});

    int inserted = 0;
    while (inserted < size / 2 - 1) {
        int k = 0;
        for (; k < size / 2 - inserted && k < THREADS_PER_BLOCK * BLOCKS * 2; ++k) {
            keys[k] = rand() / (double) RAND_MAX * (2 * size);
            hashes[k] = hfn(keys[k]);
            requests[k] = REQUEST_INSERT;
        }
        for (; k < THREADS_PER_BLOCK * BLOCKS * 2; ++k) {
            requests[k] = REQUEST_EMPTY;
        }
        s->batch(keys.data(), values.data(), requests.data(), hashes.data());

        k = 0;
        int loopCond = size / 2 - inserted;
        for (; k < loopCond && k < THREADS_PER_BLOCK * BLOCKS * 2; ++k) {
            if (values[k] != EMPTY<unsigned>::value)
                inserted++;
        }
    }

    for (int i = 0; i < 2 * THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = rand() / (double) RAND_MAX * (2 * size);
        hashes[i] = hfn(keys[i]);
        requests[i] = REQUEST_GET;
    }

    std::vector<std::pair<int,float>> sizeAndTime;

    auto start = std::chrono::high_resolution_clock::now();
    s->batch(keys.data(), values.data(), requests.data(), hashes.data(), sizeAndTime);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end - start;

    std::cout << dur.count() << " s" << std::endl;
    for (auto &t : sizeAndTime) {
        std::cout << t.first << " to a GPU" << std::endl;
        std::cout << t.first / (t.second * 1e3) << " Mops" << std::endl;
    }
}