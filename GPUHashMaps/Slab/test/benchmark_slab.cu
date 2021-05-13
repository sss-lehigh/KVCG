#include <Slab.cuh>
#include <vector>

int main() {

    SlabUnified *s = new SlabUnified(1000000);

    std::vector<unsigned> keys;
    std::vector<unsigned> values;
    std::vector<unsigned> requests;

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys.push_back((unsigned) (rand() % 100000));
        values.push_back(1);
        requests.push_back(REQUEST_INSERT);
    }

    auto res = s->batch_insert(keys.data(), values.data(), requests.data());

    std::cout << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << std::endl;

    keys = std::vector<unsigned>();
    values = std::vector<unsigned>();
    requests = std::vector<unsigned>();

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys.push_back((unsigned) (rand() % 100000));
        values.push_back(1);
        requests.push_back(REQUEST_GET);
    }

    res = s->batch_get(keys.data(), values.data(), requests.data());

    std::cout << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << std::endl;


    delete s;

    return 0;
}