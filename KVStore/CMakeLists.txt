project(KVStore LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_library(kvstore INTERFACE)
target_include_directories(kvstore INTERFACE include)
target_link_libraries(kvstore INTERFACE libmegakv)
target_link_libraries(kvstore INTERFACE pthread)
target_link_libraries(kvstore INTERFACE partslab)
target_link_libraries(kvstore INTERFACE kvcache)
target_link_libraries(kvstore INTERFACE tbb)

add_executable(KVStoreClientTest test/KVStoreClientTest.cu)
target_link_libraries(KVStoreClientTest gtest gmock gtest_main kvstore)
add_test(KVStoreClientTest KVStoreClientTest)

add_executable(KVStoreCtxTest test/KVStoreCtxTest.cu)
target_link_libraries(KVStoreCtxTest gtest gmock gtest_main kvstore)
add_test(KVStoreCtxTest KVStoreCtxTest)

add_executable(KVStoreInternalClientTest test/KVStoreInternalClientTest.cu)
target_link_libraries(KVStoreInternalClientTest gtest gmock gtest_main kvstore)
add_test(KVStoreInternalClientTest KVStoreInternalClientTest)

add_executable(KVStoreTest test/KVStoreTest.cu)
target_link_libraries(KVStoreTest gtest gmock gtest_main kvstore)
add_test(KVStoreTest KVStoreTest)

add_executable(SimpleFutureTest test/SimpleFutureTest.cu)
target_link_libraries(SimpleFutureTest gtest gmock gtest_main kvstore)
add_test(SimpleFutureTest SimpleFutureTest)

add_executable(model_service_rate benchmark/model_service_rate.cu)