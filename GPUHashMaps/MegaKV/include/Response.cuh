//
// Created by depaulsmiller on 10/4/20.
//

#include <string>

#ifndef GPUHASHMAPS_RESPONSE_CUH
#define GPUHASHMAPS_RESPONSE_CUH
namespace megakv {

    typedef std::string data_t;

    struct Response {

        explicit Response(int size) {
            if (size == 0) {
                values = nullptr;
            } else {
                values = new data_t[size];
            }
        }

        ~Response() {
            delete[] values;
        }

        data_t *values;
    };
}

#endif //GPUHASHMAPS_RESPONSE_CUH
