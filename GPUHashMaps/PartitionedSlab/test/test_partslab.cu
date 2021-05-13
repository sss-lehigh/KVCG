#include <StandardSlabDefinitions.cuh>
#include <PartitionedSlabUnified.cuh>

int main() {

    std::cerr << "Starting" << std::endl;
    auto s = new PartitionedSlabUnified<unsigned, unsigned>({{100000, 0, cudaStreamDefault}, {100000, 1, cudaStreamDefault}});
    std::cerr << "Slab Created" << std::endl;

    std::vector<unsigned> keys(THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> values(THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> requests(THREADS_PER_BLOCK * BLOCKS);
    std::vector<unsigned> hashes(THREADS_PER_BLOCK * BLOCKS);
    std::hash<unsigned> hfn;


    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys[i] = (unsigned) rand();
        values[i] = 1;
        requests[i] = REQUEST_INSERT;
        hashes[i] = hfn(keys[i]);
    }

    std::cerr << "Batching" << std::endl;

    s->batch(keys.data(), values.data(), requests.data(), hashes.data());
    std::cerr << "Batched" << std::endl;

    delete s;
    std::cerr << "Deleted" << std::endl;

    return 0;
}