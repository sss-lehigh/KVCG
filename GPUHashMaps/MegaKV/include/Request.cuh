#ifndef REQUEST_CUH
#define REQUEST_CUH
namespace megakv {

    const int REQUEST_INSERT = 1;
    const int REQUEST_GET = 2;
    const int REQUEST_REMOVE = 3;
    const int REQUEST_EMPTY = 0;

    typedef std::string data_t;

    struct Request {
        int requestInt;
        data_t key;
        data_t value;
    };

    struct BatchOfRequests {
        BatchOfRequests() : reqs(new Request[512]) {

        }

        ~BatchOfRequests() {
            delete[] reqs;
        }

        Request *reqs;
    };

}
#endif