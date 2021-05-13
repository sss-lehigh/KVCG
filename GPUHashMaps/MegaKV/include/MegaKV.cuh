//
// Created by depaulsmiller on 10/4/20.
//

#include "MegaKVGPU.cuh"
#include <string>
#include <vector>
#include <atomic>
#include "GPUData.cuh"
#include "Response.cuh"
#include <condition_variable>
#include "Request.cuh"
#include <mutex>

#ifndef GPUHASHMAPS_MEGAKV_CUH
#define GPUHASHMAPS_MEGAKV_CUH

namespace megakv {


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

    template<typename K, typename V>
    struct Bucket {

        Bucket() : anyFree(true), availible(255), pairs(new std::pair<K, V>[255]){
            for(int i = 0; i < 255; i++){
                availible[i] = true;
            }
        }
        ~Bucket(){
            delete[] pairs;
        }

        uint8_t allocate(){
            std::lock_guard<std::mutex> lg(mtx);
            for(int i = 0; i < 255; i++){
                if(availible[i]){
                    availible[i] = false;
                    if(i == 254)
                        anyFree = false;
                    return i;
                }
            }
            return 255;
        } // returns 255 on error

        void free(uint8_t i){
            std::lock_guard<std::mutex> lg(mtx);
            anyFree = true;
            availible[i] = true;
            pairs[i] = {K(), V()}; // assuming shared_ptr or caller can delete
        }

        /// not necessarily safe, but it is fast
        inline void get(uint8_t i, std::pair<K, V>& p){
            p = pairs[i];
        }

        inline void set(uint8_t i, std::pair<K, V>& p){
            pairs[i] = p;
        }


        std::mutex mtx;
        std::atomic_bool anyFree;
        std::vector<bool> availible; // 255 locations are availible in the store this is a bitmap
        std::pair<K, V>* pairs;
    };

    template<typename K, typename V>
    struct SecondaryIndex {

        /// give the size log 2 up to 
        SecondaryIndex(uint32_t sizeLog2) : nbits(sizeLog2), buckets(new Bucket<K,V>[(1 << sizeLog2)]){

        }

        ~SecondaryIndex(){
            delete[] buckets;
        }

        Bucket<K,V>* getBucket(uint64_t hash){
            return &(buckets[(uint32_t)(hash >> (32 - nbits))]);
        }

        Bucket<K,V>* alloc(uint8_t& loc, uint32_t& hash){
            for(uint32_t i = 0; i < (1 << nbits); i++){
                if(buckets[i].anyFree){ // can only observe that it is free when it is not at worst
                    auto res = buckets[i].allocate();
                    if(res != 255){
                        loc = res;
                        hash = (i << (32 - nbits)) | (uint32_t)loc;
                        return &buckets[i];
                    }
                }
            }
            std::cerr << "Allocations full" << std::endl;
            exit(255);
            return nullptr;
        }

        Bucket<K,V>* buckets;
        const uint32_t nbits; // top number of bits to refer to a bucket
    };

    const uint32_t SizeLog2 = 8;

    template<typename K, typename V>
    class MegaKV {
    public:

        MegaKV(int first_index_size) : secondaryIndex(SizeLog2) {
            gpu_handle = new MegaKVGPU(first_index_size);
        }

        ~MegaKV(){
            delete gpu_handle;
        }

        void preprocess_hashes(std::vector<std::shared_ptr<BatchOfRequests>> &requests, GPUData *data) {
            unsigned *keys = data->keys_h;
            for (int i = 0; i < requests.size(); i++) {
#pragma unroll
                for (int j = 0; j < 512; j++) {
                    keys[i * 512 + j] = std::hash<K>{}(requests[i]->reqs[j].key);
                }
            }
        }

        void preprocess_rest(std::vector<std::shared_ptr<BatchOfRequests>> &requests, GPUData *data) {
            unsigned *values = data->values_h;
            int *requestType = data->requests_h;
            for (int i = 0; i < requests.size(); i++) {
#pragma unroll
                for (int j = 0; j < 512; j++) {
                    requestType[i * 512 + j] = requests[i]->reqs[j].requestInt;
                    if (requests[i]->reqs[j].requestInt == REQUEST_INSERT) {

                        uint8_t loc;
                        uint32_t hash;
                        Bucket<K,V>* b = secondaryIndex.alloc(loc, hash);

                        std::pair<K, V> p = {requests[i]->reqs[j].key, std::move(requests[i]->reqs[j].value)};
                        (*b).set(loc, p);
                        values[i * 512 + j] = hash;

                    }
                }
            }
        }

        void moveTo(GPUData *data, cudaStream_t stream) {
            data->moveToGPU(stream);

        }

        void moveFrom(GPUData *data, cudaStream_t stream) {
            data->moveValuesBack(stream);
        }

        void execute(GPUData *data, cudaStream_t stream) {
            gpu_handle->exec_async(data->keys_k, data->values_k, data->requests_k, stream);
        }

        void postprocess(std::vector<std::shared_ptr<BatchOfRequests>> &requests, const std::shared_ptr<Response>& resp, GPUData *data) {
            std::vector<uint32_t> dealloc;
            for (int i = 0; i < requests.size(); i++) {
                for (int j = 0; j < 512; j++){
                    if (data->requests_h[i * 512 + j] == REQUEST_GET){

                        uint32_t val = data->values_h[i * 512 + j];
                        std::pair<K, V> p;
                        Bucket<K,V>* b = secondaryIndex.getBucket(val);
                        b->get(val & 0xFF, p);
                        
                        if(requests[i]->reqs[j].key == p.first) {
                            resp->values[i] = p.second;
                        }
                    } else if(data->requests_h[i * 512 + j] == REQUEST_REMOVE){
                        dealloc.push_back(data->values_h[i * 512 + j]);
                    }
                }
            }
            for (auto &d : dealloc){
                Bucket<K,V>* b = secondaryIndex.getBucket(d);
                b->free(d & 0xFF);
            }
        }


        void batch(std::vector<std::shared_ptr<BatchOfRequests>> &requests, std::shared_ptr<Response> resp, double &duration, float &gpums, GPUData *data,
                   cudaStream_t stream) {
            auto start = std::chrono::high_resolution_clock::now();
            unsigned *keys = data->keys_h;
            unsigned *values = data->values_h;
            int *requestType = data->requests_h;

            int spawn = 4;
            barrier_t b(spawn);

            std::vector<std::thread> threadVec;
            for (int i = 0; i < spawn; i++) {
                threadVec.push_back(std::thread(
                        [&requests, &keys, &requestType, &values, &b, &resp, this, &data, &stream, &gpums](int tid,
                                                                                                           int threads) {
                            for (int i = requests.size() / threads * tid;
                                 i < requests.size() / threads * (tid + 1); i++) {
                                for (int j = 0; j < 512; j++) {
                                    keys[i * 512 + j] = std::hash<K>{}(requests[i]->reqs[j].key);
                                    requestType[i * 512 + j] = requests[i]->reqs[j].requestInt;
                                    if (requests[i]->reqs[j].requestInt == REQUEST_INSERT) {

                                        uint8_t loc;
                                        uint32_t hash;
                                        Bucket<K,V>* buc = secondaryIndex.alloc(loc, hash);
                
                                        std::pair<K, V> p = {requests[i]->reqs[j].key, std::move(requests[i]->reqs[j].value)};
                                        (*buc).set(loc, p);
                                
                                        values[i * 512 + j] = hash;
                
                                    }
                                }
                            }
                            b.wait();
                            if (tid == 0) {
                                data->moveToGPU(stream);
                                gpu_handle->batch(data->keys_k, data->values_k, data->requests_k, stream, gpums);
                                data->moveValuesBack(stream);
                            }
                            b.wait();
                            std::vector<uint32_t> dealloc;
                            for (int i = requests.size() / threads * tid;
                                 i < requests.size() / threads * (tid + 1); i++) {
                                    for (int j = 0; j < 512; j++){
                                        if (data->requests_h[i * 512 + j] == REQUEST_GET){
                    
                                            uint32_t val = data->values_h[i * 512 + j];
                                            std::pair<K, V> p;
                                            Bucket<K,V>* buc = secondaryIndex.getBucket(val);
                                            buc->get(val & 0xFF, p);
                                            
                                            if(requests[i]->reqs[j].key == p.first) {
                                                resp->values[i] = p.second;
                                            }
                                        } else if(data->requests_h[i * 512 + j] == REQUEST_REMOVE){
                                            dealloc.push_back(data->values_h[i * 512 + j]);
                                        }
                                    }                            
                                }
                                for (auto &d : dealloc){
                                    Bucket<K,V>* buc = secondaryIndex.getBucket(d);
                                    buc->free(d & 0xFF);
                                }
                    
                        }, i, spawn));
            }

            for (auto &t: threadVec) {
                t.join();
            }

            auto end = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration<double>(end - start).count();
        }

    private:

        MegaKVGPU *gpu_handle;
        SecondaryIndex<K,V> secondaryIndex;
    };
}


#endif //GPUHASHMAPS_MEGAKV_CUH
