//
// Created by depaulsmiller on 1/15/21.
//

#include <iostream>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <kvcg.cuh>
#include <vector>
#include <zipf.hh>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <set>

namespace pt = boost::property_tree;


using BatchWrapper = std::vector<RequestWrapper<unsigned long long, data_t *>>;

struct UniformWorkloadConfig {
    UniformWorkloadConfig() {
        range = 1000000000;
        n = 10000;
        ops = 100000000;
        keysize = 8;
        ratio = 95;
    }

    UniformWorkloadConfig(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        range = root.get<int>("range", 10000000);
        n = root.get<int>("n", 10000);
        ops = root.get<int>("ops", 10000);
        keysize = root.get<size_t>("keysize", 8);
        ratio = root.get<int>("ratio", 95);
    }


    ~UniformWorkloadConfig() {}

    int range;
    int n;
    int ops;
    size_t keysize;
    int ratio;
};

UniformWorkloadConfig workloadConfig;

extern "C" int getBatchesToRun() {
    return workloadConfig.n;
}

extern "C" void initWorkload() {
}

extern "C" void initWorkloadFile(std::string filename) {
    workloadConfig = UniformWorkloadConfig(filename);
}

std::vector<RequestWrapper<unsigned long long, data_t *>>
generateWorkloadLargeKey(size_t keySize, int size, int n, unsigned *seed,
                         int ratioOfReads) {

    std::vector<RequestWrapper<unsigned long long, data_t *>> vec;

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            vec.push_back({(unsigned long long) (rand_r(seed) % n) + 1, nullptr, REQUEST_GET});
        } else {
            if (rand_r(seed) % 100 < 50) {
                vec.push_back({(unsigned long long) (rand_r(seed) % n) + 1, new data_t(keySize),
                               REQUEST_INSERT});
            } else {
                vec.push_back(
                        {(unsigned long long) (rand_r(seed) % n) + 1, nullptr, REQUEST_REMOVE});
            }
        }

    }
    return vec;

}

extern "C" BatchWrapper generateWorkloadBatch(unsigned int *seed, unsigned batchsize) {
    return generateWorkloadLargeKey(workloadConfig.keysize, batchsize,
                                    workloadConfig.range, seed, workloadConfig.ratio);
}

extern "C" std::vector<BatchWrapper> getPopulationBatches(unsigned int *seed, unsigned batchsize) {

    std::set<unsigned long long> keys;

    if (workloadConfig.range < workloadConfig.n) {
        exit(1);
    }

    while (keys.size() < workloadConfig.n) {
        keys.insert((rand_r(seed) % workloadConfig.range) + 1);
    }

    std::vector<BatchWrapper> batches;
    auto iter = keys.begin();

    while (iter != keys.end()) {
        std::vector<RequestWrapper<unsigned long long, data_t *>> vec;
        for (int i = 0; i < batchsize && iter != keys.end(); i++) {
            vec.push_back({*iter, new data_t(workloadConfig.keysize), REQUEST_INSERT});
            ++iter;
        }
        batches.push_back(vec);
    }

    return batches;
}
