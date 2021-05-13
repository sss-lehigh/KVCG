//
// Created by depaulsmiller on 1/15/21.
//

#include <unistd.h>
#include <MegaKV.cuh>
#include <algorithm>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <dlfcn.h>
#include <tbb/pipeline.h>
#include <tbb/concurrent_queue.h>
#include <tbb/task_scheduler_init.h>

namespace pt = boost::property_tree;

int totalBatches = 10000;
int BATCHSIZE = 512;
int NUM_THREADS = std::thread::hardware_concurrency() - 4;

void usage(char *command);

double stage1(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
              megakv::GPUData *data);

double
stage3(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
       megakv::GPUData *data,
       std::shared_ptr<megakv::Response> resp);

static void pipeline_exec(megakv::MegaKV<std::string, std::string> *s, int nstreams, std::atomic_bool &done,
                          std::atomic_long &processed,
                          tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>> *q);

void (*initWorkload)() = nullptr;

void (*initWorkloadFile)(std::string) = nullptr;

std::shared_ptr<megakv::BatchOfRequests> (*generateWorkloadBatch)(unsigned int *, unsigned) = nullptr;

std::vector<std::shared_ptr<megakv::BatchOfRequests>> (*getPopulationBatches)(unsigned *, unsigned) = nullptr;

struct ServerConf {
    int threads;
    int nstreams;
    int size;

    ServerConf() {
        threads = NUM_THREADS;
        nstreams = 10;
        size = 1000000;
    }

    ServerConf(std::string filename) {
        pt::ptree root;
        pt::read_json(filename, root);
        threads = root.get<int>("threads", NUM_THREADS);
        nstreams = root.get<int>("nstreams", 10);
        size = root.get<int>("size", 10000000);
    }

    void persist(std::string filename) {
        pt::ptree root;
        root.put("threads", threads);
        root.put("nstreams", 10);
        root.put("size", size);
        pt::write_json(filename, root);
    }

    ~ServerConf() {

    }

};

int main(int argc, char **argv) {

    ServerConf sconf;

    bool workloadFilenameSet = false;
    std::string workloadFilename = "";
    std::string dllib = "./libmkvzipfianWorkload.so";

    char c;
    while ((c = getopt(argc, argv, "f:w:")) != -1) {
        switch (c) {
            case 'f':
                sconf = ServerConf(std::string(optarg));
                // optarg is the file
                break;
            case 'w':
                workloadFilenameSet = true;
                workloadFilename = optarg;
            case '?':
                usage(argv[0]);
                return 1;
        }
    }

    NUM_THREADS = sconf.threads;
    auto *s = new megakv::MegaKV<std::string, std::string>(sconf.size);

    auto handler = dlopen(dllib.c_str(), RTLD_LAZY);
    if (!handler) {
        std::cerr << dlerror() << std::endl;
        return 1;
    }
    initWorkload = (void (*)()) dlsym(handler, "initWorkload");
    initWorkloadFile = (void (*)(std::string)) dlsym(handler, "initWorkloadFile");
    generateWorkloadBatch = (std::shared_ptr<megakv::BatchOfRequests>(*)(unsigned *, unsigned)) dlsym(handler,
                                                                                                      "generateWorkloadBatch");
    getPopulationBatches = (std::vector<std::shared_ptr<megakv::BatchOfRequests>>(*)(unsigned *, unsigned)) dlsym(
            handler, "getPopulationBatches");

    if (workloadFilenameSet) {
        initWorkloadFile(workloadFilename);
    } else {
        initWorkload();
    }

    std::cerr << "Starting populating" << std::endl;

    unsigned seed = time(nullptr);
    auto popBatches = getPopulationBatches(&seed, BATCHSIZE);
    auto tmpData = new megakv::GPUData();
    auto batchesIter = popBatches.begin();
    while (batchesIter != popBatches.end()) {
        std::vector<std::shared_ptr<megakv::BatchOfRequests>> batchToRun;
        for (int i = 0; i < megakv::BLOCKS && batchesIter != popBatches.end(); i++) {
            batchToRun.push_back(*batchesIter);
            ++batchesIter;
        }
        auto resp = std::make_shared<megakv::Response>(megakv::BLOCKS * megakv::THREADS_PER_BLOCK);
        s->preprocess_hashes(batchToRun, tmpData);
        s->preprocess_rest(batchToRun, tmpData);
        s->moveTo(tmpData, cudaStreamDefault);
        s->execute(tmpData, cudaStreamDefault);
        s->moveFrom(tmpData, cudaStreamDefault);
        cudaStreamSynchronize(cudaStreamDefault);
        s->postprocess(batchToRun, resp, tmpData);
    }
    delete tmpData;

    std::cerr << "Starting workload" << std::endl;

    auto *q = new tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>>();
    std::atomic_bool done = false;

    auto start = std::chrono::high_resolution_clock::now();

    int batches = 100000000;
    std::atomic_long processed{0};

    auto f = std::thread([&]() {
        pipeline_exec(s, sconf.nstreams, done, processed, q);
    });

    auto start_arrival = std::chrono::high_resolution_clock::now();

    const int clients = 8;
    std::vector<std::thread> threads;
    for (int i = 0; i < clients; i++) {
        threads.push_back(std::thread([&]() {
            unsigned s = time(nullptr);
            for (int j = 0; j < batches / clients; j++) {
                q->push(std::move(generateWorkloadBatch(&s, megakv::THREADS_PER_BLOCK)));
            }
        }));
    }

    for (auto &t : threads) {
        t.join();
    }
    auto end_arrival = std::chrono::high_resolution_clock::now();

    done = true;

    f.join();

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << processed.load() * megakv::THREADS_PER_BLOCK /
                 std::chrono::duration<double>(end - start).count() / 1e6
              << std::endl;
    std::cout << processed.load() * megakv::THREADS_PER_BLOCK /
                 std::chrono::duration<double>(end_arrival - start_arrival).count() / 1e6
              << std::endl;

    delete s;
    delete q;

    dlclose(handler);

    gpuErrchk(cudaDeviceReset());

    return 0;
}

void usage(char *command) {
    using namespace std;
    cout << command << " [-f <config file>]" << std::endl;
}

double stage1(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
              megakv::GPUData *data) {
    auto start = std::chrono::high_resolution_clock::now();
    s->preprocess_hashes(reqs, data);
    s->preprocess_rest(reqs, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

double
stage3(megakv::MegaKV<std::string, std::string> *s, std::vector<std::shared_ptr<megakv::BatchOfRequests>> &reqs,
       megakv::GPUData *data,
       std::shared_ptr<megakv::Response> resp) {
    auto start = std::chrono::high_resolution_clock::now();
    s->postprocess(reqs, resp, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

static void pipeline_exec(megakv::MegaKV<std::string, std::string> *s, int nstreams, std::atomic_bool &done,
                          std::atomic_long &processed,
                          tbb::concurrent_queue<std::shared_ptr<megakv::BatchOfRequests>> *q) {

    using vec_type = std::vector<std::shared_ptr<megakv::BatchOfRequests>>;

    megakv::GPUData **data = new megakv::GPUData *[nstreams];
    for (int i = 0; i < nstreams; i++) {
        data[i] = new megakv::GPUData();
    }

    std::vector<std::shared_ptr<megakv::Response>> resp;
    for (int i = 0; i < nstreams; i++) {
        resp.push_back(std::make_shared<megakv::Response>(megakv::BLOCKS * megakv::THREADS_PER_BLOCK));
    }
    cudaStream_t streams[nstreams];
    std::mutex streamMtx[nstreams];
    std::chrono::high_resolution_clock::time_point starts[nstreams];

    for (int i = 0; i < nstreams; i++) gpuErrchk(cudaStreamCreate(&streams[i]));
    std::atomic_int streamChoice{0};

    tbb::task_scheduler_init init(NUM_THREADS);
    tbb::parallel_pipeline(nstreams,
                           tbb::make_filter<void, std::pair<vec_type, int>>(tbb::filter::parallel,
                                                                            [&](tbb::flow_control &fc) -> std::pair<vec_type, int> {
                                                                                if (done && q->empty()) {
                                                                                    fc.stop();
                                                                                    return std::make_pair(
                                                                                            vec_type(),
                                                                                            0);
                                                                                }

                                                                                auto v = vec_type();
                                                                                int i = 0;
                                                                                for (; i < megakv::BLOCKS &&
                                                                                       !done;) {
                                                                                    std::shared_ptr<megakv::BatchOfRequests> b;
                                                                                    if (q->try_pop(b)) {
                                                                                        v.push_back(std::move(b));
                                                                                        i++;
                                                                                        processed.fetch_add(1,
                                                                                                            std::memory_order_relaxed);
                                                                                    }
                                                                                }

                                                                                if (done && i == 0) {
                                                                                    fc.stop();
                                                                                    return std::make_pair(
                                                                                            v,
                                                                                            0);
                                                                                }

                                                                                return std::make_pair(v,
                                                                                                      streamChoice.fetch_add(
                                                                                                              1));
                                                                            }) &
                           tbb::make_filter<std::pair<vec_type, int>, std::pair<vec_type, int>>(
                                   tbb::filter::parallel,
                                   [&](std::pair<vec_type, int> p) -> std::pair<vec_type, int> {
                                       streamMtx[p.second % nstreams].lock();
                                       starts[p.second % nstreams] = std::chrono::high_resolution_clock::now();
                                       stage1(s, p.first, data[p.second % nstreams]);
                                       return p;
                                   }) & tbb::make_filter<std::pair<vec_type, int>, std::pair<vec_type, int>>(
                                   tbb::filter::parallel,
                                   [&](std::pair<vec_type, int> p) -> std::pair<vec_type, int> {
                                       s->moveTo(data[p.second % nstreams], streams[p.second % nstreams]);
                                       s->execute(data[p.second % nstreams], streams[p.second % nstreams]);
                                       s->moveFrom(data[p.second % nstreams], streams[p.second % nstreams]);
                                       cudaStreamSynchronize(streams[p.second % nstreams]);
                                       return p;
                                   }) & tbb::make_filter<std::pair<vec_type, int>, void>(tbb::filter::parallel,
                                                                                         [&](std::pair<vec_type, int> p) {
                                                                                             stage3(s, p.first,
                                                                                                    data[p.second %
                                                                                                         nstreams],
                                                                                                    resp[p.second %
                                                                                                         nstreams]);
                                                                                             auto startTmp = starts[
                                                                                                     p.second %
                                                                                                     nstreams];
                                                                                             streamMtx[p.second %
                                                                                                       nstreams].unlock();
                                                                                             double dur = std::chrono::duration<double>(
                                                                                                     std::chrono::high_resolution_clock::now() -
                                                                                                     startTmp).count();
                                                                                             std::cerr << dur * 1e3
                                                                                                       << std::endl;
                                                                                         })
    );

    for (int i = 0; i < nstreams; i++) {
        delete data[i];
    }

    delete[] data;

    for (int i = 0; i < nstreams; i++) gpuErrchk(cudaStreamDestroy(streams[i]));


}