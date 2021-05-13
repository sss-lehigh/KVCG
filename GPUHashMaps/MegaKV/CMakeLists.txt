PROJECT(MegaKV LANGUAGES CUDA)

find_package(TBB REQUIRED)

add_library(libmegakv INTERFACE)
target_include_directories(libmegakv INTERFACE include)
target_link_libraries(libmegakv INTERFACE TBB::tbb)

add_executable(megakv_benchmark test/benchmark.cu)
target_link_libraries(megakv_benchmark PRIVATE libmegakv)
target_link_libraries(megakv_benchmark PRIVATE TBB::tbb)