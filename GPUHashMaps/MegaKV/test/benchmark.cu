#include <MegaKV.cuh>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <tbb/pipeline.h>

using namespace megakv;

double stage1(MegaKV<data_t,data_t> *s, std::vector<std::shared_ptr<BatchOfRequests>> reqs, GPUData *data) {
    auto start = std::chrono::high_resolution_clock::now();
    s->preprocess_hashes(reqs, data);
    s->preprocess_rest(reqs, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

double stage2(MegaKV<data_t,data_t> *s, GPUData *data, cudaStream_t stream) {
    auto start = std::chrono::high_resolution_clock::now();
    s->moveTo(data, stream);
    s->execute(data, stream);
    s->moveFrom(data, stream);
    cudaStreamSynchronize(stream);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

double stage3(MegaKV<data_t,data_t> *s, std::vector<std::shared_ptr<BatchOfRequests>> reqs, GPUData *data, std::shared_ptr<Response> resp) {
    auto start = std::chrono::high_resolution_clock::now();
    s->postprocess(reqs, resp, data);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count() * 1e3;
}

struct stage1args {

    stage1args(MegaKV<data_t,data_t> *s, std::vector<std::shared_ptr<BatchOfRequests>> &&reqs, GPUData *data) : s(s), reqs(reqs), data(data) {

    }

    MegaKV<data_t,data_t> *s;
    std::vector<std::shared_ptr<BatchOfRequests>> reqs;
    GPUData *data;
};

void CUDART_CB stage1CallBack(cudaStream_t stream, cudaError_t status, void *userData) {
    auto args = (stage1args *) userData;
    stage1(args->s, args->reqs, args->data);
}

void asyncStage2(MegaKV<data_t,data_t> *s, GPUData *data, cudaStream_t stream) {
    s->moveTo(data, stream);
    s->execute(data, stream);
    s->moveFrom(data, stream);
}

struct stage3args {

    stage3args(MegaKV<data_t,data_t> *s, std::vector<std::shared_ptr<BatchOfRequests>> &&reqs, GPUData *data, std::shared_ptr<Response> resp) : s(s), reqs(reqs),
                                                                                               data(data), resp(resp) {}

    MegaKV<data_t,data_t> *s;
    std::vector<std::shared_ptr<BatchOfRequests>> reqs;
    GPUData *data;
    std::shared_ptr<Response> resp;
};

void CUDART_CB stage3CallBack(cudaStream_t stream, cudaError_t status, void *userData) {
    auto args = (stage3args *) userData;
    stage3(args->s, args->reqs, args->data, args->resp);
}

int main() {

    int batches = 100;
    int nstreams = 10;

    MegaKV<data_t,data_t> *s = new MegaKV<data_t,data_t>(1000000);

    GPUData **data = new GPUData *[nstreams];
    for (int i = 0; i < nstreams; i++) {
        data[i] = new GPUData();
    }

    auto b = new std::vector<std::shared_ptr<BatchOfRequests>>[batches];
    for (int i = 0; i < batches; i++) {
        for (int k = 0; k < BLOCKS; k++) {
            std::shared_ptr<BatchOfRequests> r = std::make_shared<BatchOfRequests>();
            for (int j = 0; j < 512; j++) {
                r->reqs[j] = {(i < 1) ? REQUEST_INSERT : REQUEST_GET, std::to_string((unsigned) (rand() % 100000)),
                             "test"};
            }
            b[i].push_back(r);
        }
    }
    std::vector<std::shared_ptr<Response>> resp;
    for (int i = 0; i < nstreams; i++) {
        resp.push_back(std::make_shared<Response>(BLOCKS * THREADS_PER_BLOCK));
    }
    cudaStream_t streams[nstreams];
    std::mutex streamMtx[nstreams];
    std::chrono::high_resolution_clock::time_point starts[nstreams];

    for (int i = 0; i < nstreams; i++) gpuErrchk(cudaStreamCreate(&streams[i]));

    int dataElement = 0;

    auto start = std::chrono::high_resolution_clock::now();

    tbb::parallel_pipeline(nstreams,
                           tbb::make_filter<void, int>(tbb::filter::serial_in_order, [&](tbb::flow_control &fc) {
                               if (dataElement == batches) {
                                   fc.stop();
                                   return 0;
                               }
                               return dataElement++;
                           }) & tbb::make_filter<int, int>(tbb::filter::parallel, [&](int i) {
                               streamMtx[i % nstreams].lock();
                               starts[i % nstreams] = std::chrono::high_resolution_clock::now();
                               stage1(s, std::move(b[i]), data[i % nstreams]);
                               return i;
                           }) & tbb::make_filter<int, int>(tbb::filter::parallel, [&](int i) {
                               s->moveTo(data[i % nstreams], streams[i % nstreams]);
                               s->execute(data[i % nstreams], streams[i % nstreams]);
                               s->moveFrom(data[i % nstreams], streams[i % nstreams]);
                               cudaStreamSynchronize(streams[i % nstreams]);
                               return i;
                           }) & tbb::make_filter<int, void>(tbb::filter::parallel, [&](int i) {
                               stage3(s, std::move(b[i]), data[i % nstreams], resp[i % nstreams]);
                               auto startTmp = starts[i % nstreams];
                               streamMtx[i % nstreams].unlock();
                               double dur = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTmp).count();
                               std::cerr << dur * 1e3 << std::endl;
                           })
    );

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << batches * BLOCKS * THREADS_PER_BLOCK / std::chrono::duration<double>(end - start).count() / 1e6
              << std::endl;

    delete s;
    for (int i = 0; i < nstreams; i++) {
        delete data[i];
    }

    delete[] data;

    return 0;
}