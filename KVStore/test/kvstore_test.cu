#include <kvstore.cuh>
#include <iostream>
#include <cassert>

int main() {
    std::clog << "starting" << std::endl;
    KVStoreCtx<unsigned, unsigned> ctx;
    std::clog << "ctx created" << std::endl;
    KVStoreClient<unsigned, unsigned> c(ctx);
    std::clog << "client created" << std::endl;
    //auto res = c.get(1);
    //assert(res.get().first == false);
    return 0;
}