//
// Created by depaulsmiller on 1/15/21.
//

#include <iostream>
#include <thread>
#include <unistd.h>
#include <atomic>
#include <Request.cuh>
#include <vector>
#include <zipf.hh>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <set>

namespace pt = boost::property_tree;

double zetaN = 0.1;

struct ZipfianWorkloadConfig {
    ZipfianWorkloadConfig() {
        theta = 0.99;
        range = 100000000;
        n = 1000000;
        keysize = 8;
        ratio = 100;
    }

    ZipfianWorkloadConfig(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        theta = root.get<double>("theta");
        range = root.get<int>("range");
        n = root.get<int>("n");
        keysize = root.get<size_t>("keysize");
        ratio = root.get<int>("ratio");
    }


    ~ZipfianWorkloadConfig() {}

    double theta;
    int range;
    int n;
    size_t keysize;
    int ratio;
};

ZipfianWorkloadConfig zipfianWorkloadConfig;

extern "C" void initWorkload() {
    zetaN = sssrand::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

extern "C" void initWorkloadFile(std::string filename) {
    zipfianWorkloadConfig = ZipfianWorkloadConfig(filename);
    zetaN = sssrand::zeta(zipfianWorkloadConfig.theta, zipfianWorkloadConfig.range);
}

std::shared_ptr<megakv::BatchOfRequests>
generateWorkloadZipfLargeKey(size_t keySize, unsigned size, double theta, int n, double zetaN, unsigned *seed,
                             int ratioOfReads) {

    std::shared_ptr<megakv::BatchOfRequests> batch = std::make_shared<megakv::BatchOfRequests>();

    std::string longValue;
    for (size_t i = 0; i < keySize; i++) {
        longValue += 'a';
    }

    for (int i = 0; i < size; i++) {
        if (rand_r(seed) % 100 < ratioOfReads) {
            batch->reqs[i].key = std::to_string(sssrand::rand_zipf_r(seed, n, zetaN, theta));
            batch->reqs[i].value = "";
            batch->reqs[i].requestInt = megakv::REQUEST_GET;
        } else {
            if (rand_r(seed) % 100 < 50) {
                batch->reqs[i].key = std::to_string(sssrand::rand_zipf_r(seed, n, zetaN, theta));
                batch->reqs[i].value = longValue;
                batch->reqs[i].requestInt = megakv::REQUEST_INSERT;
            } else {
                batch->reqs[i].key = std::to_string(sssrand::rand_zipf_r(seed, n, zetaN, theta));
                batch->reqs[i].value = "";
                batch->reqs[i].requestInt = megakv::REQUEST_REMOVE;
            }
        }

    }
    return batch;

}

extern "C" std::shared_ptr<megakv::BatchOfRequests> generateWorkloadBatch(unsigned int *seed, unsigned batchsize) {
    return generateWorkloadZipfLargeKey(zipfianWorkloadConfig.keysize, batchsize, zipfianWorkloadConfig.theta,
                                        zipfianWorkloadConfig.range, zetaN, seed, zipfianWorkloadConfig.ratio);
}

extern "C" std::vector<std::shared_ptr<megakv::BatchOfRequests>>
getPopulationBatches(unsigned int *seed, unsigned batchsize) {

    std::set<unsigned long long> keys;

    if (zipfianWorkloadConfig.range < zipfianWorkloadConfig.n) {
        exit(1);
    }

    while (keys.size() < zipfianWorkloadConfig.n) {
        keys.insert((rand_r(seed) % zipfianWorkloadConfig.range) + 1);
    }

    std::vector<std::shared_ptr<megakv::BatchOfRequests>> batches;
    auto iter = keys.begin();

    while (iter != keys.end()) {
        std::shared_ptr<megakv::BatchOfRequests> batch = std::make_shared<megakv::BatchOfRequests>();
        for (int i = 0; i < batchsize && iter != keys.end(); i++) {
            batch->reqs[i].key = std::to_string(*iter);
            batch->reqs[i].value = "";
            batch->reqs[i].requestInt = megakv::REQUEST_GET;
            ++iter;
        }
        batches.push_back(batch);
    }

    return batches;
}
