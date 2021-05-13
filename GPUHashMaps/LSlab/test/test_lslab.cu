#include <Slab.cuh>
#include <vector>

int main() {

    SlabUnified *s = new SlabUnified(100);

    std::vector<unsigned> keys;
    std::vector<unsigned> values;
    std::vector<unsigned> requests;

    for (int i = 0; i < 134 * 512; i++) {
        keys.push_back((unsigned) rand());
        values.push_back(1);
        requests.push_back(REQUEST_INSERT);
    }

    s->batch(keys.data(), values.data(), requests.data());

    delete s;

    return 0;
}