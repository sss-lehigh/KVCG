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

double zetaN = 0.1;

struct ZipfianWorkloadConfig {
    ZipfianWorkloadConfig() {
        theta = 0.99;
        range = 1000000000;
        n = 10000;
        ops = 100000000;
        keysize = 8;
        ratio = 95;
    }

    ZipfianWorkloadConfig(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        theta = root.get<double>("theta", 0.99);
        range = root.get<int>("range", 10000000);
        n = root.get<int>("n", 10000);
        ops = root.get<int>("ops", 10000);
        keysize = root.get<size_t>("keysize", 8);
        ratio = root.get<int>("ratio", 95);
    }


    ~ZipfianWorkloadConfig() {}

    double theta;
    int range;
    int n;
    int ops;
    size_t keysize;
    int ratio;
};

ZipfianWorkloadConfig zipfianWorkloadConfig;

extern "C" int getBatchesToRun() {
    return zipfianWorkloadConfig.n;
}

extern "C" void initWorkload() {
    zetaN = sssrand::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

extern "C" void initWorkloadFile(std::string filename) {
    zipfianWorkloadConfig = ZipfianWorkloadConfig(filename);
    zetaN = sssrand::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

std::vector<RequestWrapper<unsigned long long, data_t *>>
generateWorkloadZipfLargeKey(size_t keySize, int size, double theta, int n, double zetaN, unsigned *seed,
                             int ratioOfReads) {

    std::vector<RequestWrapper<unsigned long long, data_t *>> vec;

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            vec.push_back({(unsigned long long) sssrand::rand_zipf_r(seed, n, zetaN, theta), nullptr, REQUEST_GET});
        } else {
            if (rand_r(seed) % 100 < 50) {
                vec.push_back({(unsigned long long) sssrand::rand_zipf_r(seed, n, zetaN, theta), new data_t(keySize),
                               REQUEST_INSERT});
            } else {
                vec.push_back(
                        {(unsigned long long) sssrand::rand_zipf_r(seed, n, zetaN, theta), nullptr, REQUEST_REMOVE});
            }
        }

    }
    return vec;

}

extern "C" BatchWrapper generateWorkloadBatch(unsigned int *seed, unsigned batchsize) {
    return generateWorkloadZipfLargeKey(zipfianWorkloadConfig.keysize, batchsize, zipfianWorkloadConfig.theta,
                                        zipfianWorkloadConfig.range, zetaN, seed, zipfianWorkloadConfig.ratio);
}

extern "C" std::vector<BatchWrapper> getPopulationBatches(unsigned int *seed, unsigned batchsize) {

    std::set<unsigned long long> keys;

    if (zipfianWorkloadConfig.range < zipfianWorkloadConfig.n) {
        exit(1);
    }

    while (keys.size() < zipfianWorkloadConfig.n) {
        keys.insert((rand_r(seed) % zipfianWorkloadConfig.range) + 1);
    }

    std::vector<BatchWrapper> batches;
    auto iter = keys.begin();

    while (iter != keys.end()) {
        std::vector<RequestWrapper<unsigned long long, data_t *>> vec;
        for (int i = 0; i < batchsize && iter != keys.end(); i++) {
            vec.push_back({*iter, new data_t(zipfianWorkloadConfig.keysize), REQUEST_INSERT});
            ++iter;
        }
        batches.push_back(vec);
    }

    return batches;
}
