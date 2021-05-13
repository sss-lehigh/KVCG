project(KVCache LANGUAGES CUDA)

add_library(kvcache INTERFACE)
target_include_directories(kvcache INTERFACE include)
target_link_libraries(kvcache INTERFACE lslabext TBB::tbb)

#add_executable(kvcache-benchmark benchmark/KVCache_benchmark.cc)
#target_link_libraries(kvcache-benchmark kvcache)
#target_link_libraries(kvcache-benchmark pthread)
#target_link_libraries(kvcache-benchmark multithreading)

add_executable(kvcache-bench benchmark/KVCache_benchmark.cu)
target_link_libraries(kvcache-bench kvcache pthread)

add_executable(kvcache-tail-latency benchmark/KVCache_tail_latency.cu)
target_link_libraries(kvcache-tail-latency kvcache pthread)

#target_link_libraries(kvcache-batched kvcache)
#target_link_libraries(kvcache-batched pthread)
#target_link_libraries(kvcache-batched multithreading)