//
// Created by depaulsmiller on 1/15/21.
//

#include <unistd.h>
#include "helper.cuh"
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <dlfcn.h>

namespace pt = boost::property_tree;
using BatchWrapper = std::vector<RequestWrapper<unsigned long long, data_t *>>;
using Model = kvgpu::SimplModel<unsigned long long>;

int totalBatches = 10000;
int BATCHSIZE = 512;
int NUM_THREADS = std::thread::hardware_concurrency() - 4;

void usage(char *command);

struct ServerConf {
    int threads;
    int gpus;
    int streams;
    std::string modelFile;
    bool train;
    int size;
    int batchSize;
    bool cache;

    ServerConf() {
        batchSize = BATCHSIZE;
        modelFile = "";
        threads = NUM_THREADS;
        gpus = 1;
        streams = 10;
        size = 1000000;
        train = false;
        cache = true;
    }

    ServerConf(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        threads = root.get<int>("threads", NUM_THREADS);
        streams = root.get<int>("streams", 2);
        gpus = root.get<int>("gpus", 2);
        modelFile = root.get<std::string>("modelFile", "");
        train = root.get<bool>("train", false);
        size = root.get<int>("size", 1000000);
        batchSize = root.get<int>("batchSize", BATCHSIZE);
        cache = root.get<bool>("cache", true);
    }

    void persist(std::string filename) {
        pt::ptree root;
        root.put("threads", threads);
        root.put("streams", streams);
        root.put("gpus", gpus);
        root.put("modelFile", modelFile);
        root.put("train", train);
        root.put("size", size);
        root.put("batchSize", batchSize);
        root.put("cache", cache);
        pt::write_json(filename, root);
    }

    ~ServerConf() {

    }

};

int main(int argc, char **argv) {

    ServerConf sconf;

    bool workloadFilenameSet = false;
    std::string workloadFilename = "";
    std::string dllib = "./libzipfianWorkload.so";

    char c;
    while ((c = getopt(argc, argv, "f:w:l:")) != -1) {
        switch (c) {
            case 'f':
                sconf = ServerConf(std::string(optarg));
                // optarg is the file
                break;
            case 'w':
                workloadFilenameSet = true;
                workloadFilename = optarg;
                break;
            case 'l':
                dllib = optarg;
                break;
            case '?':
                usage(argv[0]);
                return 1;
        }
    }

    void (*initWorkload)() = nullptr;
    void (*initWorkloadFile)(std::string) = nullptr;
    BatchWrapper (*generateWorkloadBatch)(unsigned int *, unsigned) = nullptr;
    int (*getBatchesToRun)() = nullptr;
    std::vector<BatchWrapper> (*getPopulationBatches)(unsigned int *, unsigned) = nullptr;
    auto handler = dlopen(dllib.c_str(), RTLD_LAZY);
    if (!handler) {
        std::cerr << dlerror() << std::endl;
        return 1;
    }
    initWorkload = (void (*)()) dlsym(handler, "initWorkload");
    initWorkloadFile = (void (*)(std::string)) dlsym(handler, "initWorkloadFile");
    generateWorkloadBatch = (BatchWrapper(*)(unsigned *, unsigned)) dlsym(handler, "generateWorkloadBatch");
    getBatchesToRun = (int (*)()) dlsym(handler, "getBatchesToRun");
    getPopulationBatches = (std::vector<BatchWrapper> (*)(unsigned int *, unsigned)) dlsym(handler,
                                                                                           "getPopulationBatches");

    if (workloadFilenameSet) {
        initWorkloadFile(workloadFilename);
    } else {
        initWorkload();
    }

    totalBatches = getBatchesToRun();

    std::vector<PartitionedSlabUnifiedConfig> conf;
    for (int i = 0; i < sconf.gpus; i++) {
        for (int j = 0; j < sconf.streams; j++) {
            gpuErrchk(cudaSetDevice(i));
            cudaStream_t stream = cudaStreamDefault;
            if (j != 0) {
                gpuErrchk(cudaStreamCreate(&stream));
            }
            conf.push_back({sconf.size, i, stream});
        }
    }

    //TODO allow for this to be swappable.
    KVStoreCtx<unsigned long long, data_t, Model> ctx(conf);

    GeneralClient<unsigned long long, data_t, Model> *client = nullptr;
    if (sconf.cache) {
        if(sconf.gpus == 0){
            client = new JustCacheKVStoreClient<unsigned long long, data_t, Model>(ctx);
        } else {
            client = new KVStoreClient<unsigned long long, data_t, Model>(ctx);
        }
    } else {
        client = new NoCacheKVStoreClient<unsigned long long, data_t, Model>(ctx);
    }

    unsigned popSeed = time(nullptr);
    auto pop = getPopulationBatches(&popSeed, BATCHSIZE);

    for (auto &b : pop) {
        bool retry;
        int size = b.size();
        do {
            auto rb = std::make_shared<ResultsBuffers<data_t>>(sconf.batchSize);
            std::vector<std::chrono::high_resolution_clock::time_point> rt;
            rt.reserve(BATCHSIZE);
            client->batch(b, rb, rt);


            bool finished;

            do {
                finished = true;
                for (int i = 0; i < size; i++) {
                    if (rb->requestIDs[i] == -1) {
                        finished = false;
                        break;
                    }
                }
            } while (!finished && !rb->retryGPU);
            retry = rb->retryGPU;
        } while (retry);
    }

    std::cerr << "Populated" << std::endl;

    client->resetStats();

    std::vector<std::thread> threads;
    tbb::concurrent_vector<std::pair<std::chrono::high_resolution_clock::time_point, std::vector<std::chrono::high_resolution_clock::time_point>>> times;

    using RB = std::shared_ptr<ResultsBuffers<data_t>>;

    tbb::concurrent_queue<std::pair<BatchWrapper, RB>> *q = new tbb::concurrent_queue<std::pair<BatchWrapper, RB>>[sconf.threads];

    std::atomic_bool reclaim{false};

    std::atomic_bool changing{false};

    block_t *block = new block_t(sconf.threads);

    for (int i = 0; i < sconf.threads; ++i) {
        threads.push_back(std::thread([&client, &times, &reclaim, &q, &changing, &block, sconf](int tid) {

            std::vector<std::pair<std::chrono::high_resolution_clock::time_point, std::vector<std::chrono::high_resolution_clock::time_point>>> tmpTimes;

            std::shared_ptr<ResultsBuffers<data_t>> lastResBuf = nullptr;

            while (!reclaim) {

                if (changing) {
                    block->wait();
                    while (changing);
                }

                std::pair<BatchWrapper, RB> p;
                if (q[tid].try_pop(p)) {
                    auto start = std::chrono::high_resolution_clock::now();
                    std::vector<std::chrono::high_resolution_clock::time_point> rt;
                    rt.reserve(BATCHSIZE);
                    lastResBuf = p.second;
                    client->batch(p.first, p.second, rt);
                    tmpTimes.push_back({start, rt});
                }
            }

            std::pair<BatchWrapper, RB> p;
            while (q[tid].try_pop(p)) {
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<std::chrono::high_resolution_clock::time_point> rt;
                rt.reserve(BATCHSIZE);
                lastResBuf = p.second;
                client->batch(p.first, p.second, rt);
                tmpTimes.push_back({start, rt});
            }

            for (auto &t : tmpTimes) {
                times.push_back(t);
            }

            bool finished;

            do {
                finished = true;
                for (int i = 0; i < sconf.batchSize; i++) {
                    if (lastResBuf->requestIDs[i] == -1) {
                        finished = false;
                        break;
                    }
                }
            } while (!finished && !lastResBuf->retryGPU);

        }, i));
    }
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads2;
    int clients = 8;
    for (int j = 0; j < clients; j++) {
        threads2.push_back(std::thread(
                [clients, &q, &sconf, generateWorkloadBatch](int tid) {
                    unsigned tseed = time(nullptr);
                    for (int i = 0; i < totalBatches / clients; i++) {

                        /*if (tid == 0 && i == totalBatches / clients / 10) {
                            std::cerr << "Changing\n";
                            auto tmp = kvgpu::SimplModel<unsigned>(18000);
                            changing = true;
                            client->change_model(tmp, block);
                            changing = false;
                            std::cerr << "Changed\n";
                        }*/
                        std::pair<BatchWrapper, RB> p = {
                                generateWorkloadBatch(&tseed, sconf.batchSize),
                                std::make_shared<ResultsBuffers<data_t>>(sconf.batchSize)};
                        q[(tid + i) % NUM_THREADS].push(std::move(p));
                    }
                }, j
        ));
    }
    for (auto &t : threads2) {
        t.join();
    }
    auto endTimeArrival = std::chrono::high_resolution_clock::now();


    reclaim = true;


    std::cerr << "Awake and joining\n";
    for (auto &t : threads) {
        t.join();
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    size_t ops = client->getOps();

    std::sort(times.begin(), times.end(),
              [](const std::pair<std::chrono::high_resolution_clock::time_point, std::vector<std::chrono::high_resolution_clock::time_point>> &lhs,
                 const std::pair<std::chrono::high_resolution_clock::time_point, std::vector<std::chrono::high_resolution_clock::time_point>> &rhs) {
                  return lhs.first < rhs.first;
              });

    std::vector<std::pair<std::chrono::high_resolution_clock::time_point, std::vector<double>>> times2;

    for (auto &t : times) {
        std::vector<double> tmp;
        for (auto &t2 : t.second) {
            tmp.push_back(std::chrono::duration<double>(t2 - t.first).count());
        }
        times2.push_back({t.first, tmp});
    }

    std::chrono::duration<double> dur = endTime - startTime;
    std::chrono::duration<double> durArr = endTimeArrival - startTime;
    if (!times.empty()) {
        if (sconf.cache) {
            auto s = times[0].first;
            std::cout << "TABLE: Latency of Hot Storage" << std::endl;
            std::cout << "Timestamp\tAvg Latency\tMin Latency\tMax Latency" << std::endl;
            for (auto &t : times2) {
                if (!t.second.empty()) {
                    double avg = 0.0;
                    std::for_each(t.second.begin(), t.second.end(), [&avg](double d) {
                        avg += d;
                    });

                    avg /= t.second.size();

                    std::cout << std::chrono::duration<double>(t.first - s).count() << "\t" << avg * 1e3 << "\t"
                              << t.second[0] * 1e3 << "\t" << t.second[t.second.size() - 1] * 1e3 << std::endl;
                }
            }

            //delete barrier;

            std::cout << std::endl;

            std::cout << "TABLE: Hot Storage Latencies" << std::endl;
            std::cout << "Latency" << std::endl;
            for (auto &t : times2) {
                for (auto &t2 : t.second) {
                    std::cout << t2 * 1e3 << std::endl;
                }
            }

            std::cout << std::endl;
        }

        client->stat();

        std::cerr << "Arrival Rate (Mops) " << (sconf.batchSize * times.size()) / durArr.count() / 1e6 << std::endl;
        std::cerr << "Throughput (Mops) " << ((double) ops + client->getHits()) / dur.count() / 1e6 << std::endl;

        if (sconf.cache) {
            std::cerr << "Hit Rate\tHits" << std::endl;
            std::cerr << client->hitRate() << "\t" << client->getHits() << std::endl;
            std::cerr << std::endl;
        }

        std::cout << "TABLE: Throughput" << std::endl;
        std::cout << "Throughput" << std::endl;
        std::cout << ((double) ops + client->getHits()) / dur.count() / 1e6 << std::endl;
    }
    delete client;
    delete block;
    dlclose(handler);
    return 0;
}

void usage(char *command) {
    using namespace std;
    cout << command << " [-f <config file>]" << std::endl;
}
