/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <Slab.cuh>
#include <list>
#include <vector>

#ifndef PARTITIONED_SLAB_UNIFIED_CUH
#define PARTITIONED_SLAB_UNIFIED_CUH

/**
 * PartitionedSlabUnifiedConfig has 3 configurable parameters
 * size of the map, the gpu used, and the stream
 */
struct PartitionedSlabUnifiedConfig {
    int size;
    int gpu;
    cudaStream_t stream;
};

/*
template<typename K, typename V>
class PartitionedSlabUnified : public Slab<K, V> {
private:

    struct BatchData {

        BatchData() : keys_gpu(THREADS_PER_BLOCK * BLOCKS), values_gpu(THREADS_PER_BLOCK * BLOCKS),
                      index(0), requests_gpu(THREADS_PER_BLOCK * BLOCKS), hashes_gpu(THREADS_PER_BLOCK * BLOCKS), wbLocation(THREADS_PER_BLOCK * BLOCKS) {

        }

        std::vector<K> keys_gpu;
        std::vector<V> values_gpu;
        int index;
        std::vector<unsigned> requests_gpu;
        std::vector<unsigned> hashes_gpu;
        std::vector<int> wbLocation;
    };

public:

    explicit PartitionedSlabUnified(const std::vector<PartitionedSlabUnifiedConfig> &config) : numberOfSlabs(config.size()) {
        slabs = new SlabUnified<K, V>[numberOfSlabs];
        buffers = new BatchBuffer<K,V>*[numberOfSlabs];
        _streams = new cudaStream_t[numberOfSlabs];

        for (int i = 0; i < numberOfSlabs; i++) {
            buffers[i] = new BatchBuffer<K,V>();
            _streams[i] = config[i].stream;
            slabs[i] = std::move(SlabUnified<K, V>(config[i].size, config[i].gpu));
        }
    }

    ~PartitionedSlabUnified() {
        delete[] slabs;
        for (int i = 0; i < numberOfSlabs; i++)
            delete buffers;
        delete[] buffers;
        delete[] slabs;
    }

    void batch(K *keys, V *values, const unsigned *requests, const unsigned *hashes) {

        std::vector<std::list<BatchData>> batches(numberOfSlabs);

        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS * numberOfSlabs; i++) {
            if (requests[i] != REQUEST_EMPTY) {
                int gpu = hashes[i] % numberOfSlabs;

                if (batches[gpu].empty()) {
                    BatchData b;
                    batches[gpu].push_back(b);
                }

                BatchData &b = batches[gpu].back();

                int idx = b.index;

                b.keys_gpu[idx] = keys[i];
                b.values_gpu[idx] = values[i];
                b.requests_gpu[idx] = requests[i];
                b.hashes_gpu[idx] = hashes[i];
                b.wbLocation[idx] = i;
                b.index++;
                if (b.index == THREADS_PER_BLOCK * BLOCKS) {
                    BatchData b2;
                    batches[gpu].push_back(b2);
                }
            }
        }

        for (int gpu = 0; gpu < numberOfSlabs; gpu++) {
            for (BatchData &batch : batches[gpu]) {
                slabs[gpu].batch(batch.keys_gpu.data(), batch.values_gpu.data(), batch.requests_gpu.data(), batch.hashes_gpu.data());
            }
        }

        for (int gpu = 0; gpu < numberOfSlabs; gpu++) {
            for (BatchData &batch : batches[gpu]) {
                for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
                    values[batch.wbLocation[i]] = batch.values_gpu[i];
                }
            }
        }

    }

    void batch(K *keys, V *values, const unsigned *requests, const unsigned *hashes, std::vector<std::pair<int, float>> &sizeAndTime) {

        std::vector<std::list<BatchData>> batches(numberOfSlabs);

        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS * numberOfSlabs; i++) {
            if (requests[i] != REQUEST_EMPTY) {
                int gpu = hashes[i] % numberOfSlabs;

                if (batches[gpu].empty()) {
                    BatchData b;
                    batches[gpu].push_back(b);
                }

                BatchData &b = batches[gpu].back();

                int idx = b.index;

                b.keys_gpu[idx] = keys[i];
                b.values_gpu[idx] = values[i];
                b.requests_gpu[idx] = requests[i];
                b.hashes_gpu[idx] = hashes[i];
                b.wbLocation[idx] = i;
                b.index++;

                if (b.index == THREADS_PER_BLOCK * BLOCKS) {
                    BatchData b2;
                    batches[gpu].push_back(b2);
                }
            }
        }

        for (int gpu = 0; gpu < numberOfSlabs; gpu++) {
            for (BatchData &batch : batches[gpu]) {
                float tmptime = 0;
                slabs[gpu].batch(batch.keys_gpu.data(), batch.values_gpu.data(), batch.requests_gpu.data(), batch.hashes_gpu.data(), tmptime);
                sizeAndTime.push_back({batch.index, tmptime});
            }
        }

        for (int gpu = 0; gpu < numberOfSlabs; gpu++) {
            for (BatchData &batch : batches[gpu]) {
                for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
                    values[batch.wbLocation[i]] = batch.values_gpu[i];
                }
            }
        }


    }

    int getNumberOfSlabs() {
        return numberOfSlabs;
    }

private:
    int numberOfSlabs;
    SlabUnified<K, V> *slabs;
    cudaStream_t* _streams;
    BatchBuffer<K, V>** buffers;
};
*/

#endif // SLAB_SLAB_CUH
