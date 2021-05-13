cmake_minimum_required(VERSION 3.18)

project(KVCG LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES OFF)
message("USING ${CMAKE_TOOLCHAIN_FILE}")

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

enable_testing()

add_subdirectory(rand)

add_subdirectory(KVCache)

add_subdirectory(GPUHashMaps)

add_subdirectory(KVStore)

add_executable(kvcg service/server.cu)
target_link_libraries(kvcg PUBLIC partslab)
#target_link_libraries(kvcg PUBLIC multithreading)
target_link_libraries(kvcg PUBLIC pthread)
target_link_libraries(kvcg PUBLIC kvstore)
target_link_libraries(kvcg PUBLIC tbbmalloc_proxy)
target_link_libraries(kvcg PUBLIC Boost::boost)
target_link_libraries(kvcg PUBLIC libmegakv)

add_executable(kvcg_mkv service/server.cu)
target_link_libraries(kvcg_mkv PUBLIC partslab)
#target_link_libraries(kvcg_mkv PUBLIC multithreading)
target_link_libraries(kvcg_mkv PUBLIC pthread)
target_link_libraries(kvcg_mkv PUBLIC kvstore)
target_link_libraries(kvcg_mkv PUBLIC tbbmalloc_proxy)
target_link_libraries(kvcg_mkv PUBLIC Boost::boost)
target_link_libraries(kvcg_mkv PUBLIC libmegakv)
target_compile_options(kvcg_mkv PUBLIC -DUSE_MEGAKV)

add_executable(kvcg_asan service/server.cu)
target_link_libraries(kvcg_asan PUBLIC asan)
target_link_libraries(kvcg_asan PUBLIC partslab)
#target_link_libraries(kvcg_asan PUBLIC multithreading)
target_link_libraries(kvcg_asan PUBLIC pthread)
target_link_libraries(kvcg_asan PUBLIC kvstore)
target_link_libraries(kvcg_asan PUBLIC tbbmalloc_proxy)
target_link_libraries(kvcg_asan PUBLIC Boost::boost)
target_link_libraries(kvcg_asan PUBLIC libmegakv)
target_compile_options(kvcg_asan PRIVATE -Xcompiler -fsanitize=address)

add_executable(megakv service/megakv_server.cu)
target_link_libraries(megakv PUBLIC partslab)
#target_link_libraries(megakv PUBLIC multithreading)
target_link_libraries(megakv PUBLIC pthread)
target_link_libraries(megakv PUBLIC libmegakv)
target_link_libraries(megakv PUBLIC tbbmalloc_proxy)
target_link_libraries(megakv PUBLIC Boost::boost)
target_link_libraries(megakv PUBLIC TBB::tbb)

add_library(zipfianWorkload SHARED service/zipfianWorkload.cu)
target_link_libraries(zipfianWorkload PRIVATE kvstore)
target_link_libraries(zipfianWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(zipfianWorkload PRIVATE rand)
target_link_libraries(zipfianWorkload PRIVATE Boost::boost)
target_link_libraries(zipfianWorkload PRIVATE libmegakv)

add_library(uniformWorkload SHARED service/uniformWorkload.cu)
target_link_libraries(uniformWorkload PRIVATE kvstore)
target_link_libraries(uniformWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(uniformWorkload PRIVATE rand)
target_link_libraries(uniformWorkload PRIVATE Boost::boost)

add_library(selfsimilarWorkload SHARED service/selfsimilarWorkload.cu)
target_link_libraries(selfsimilarWorkload PRIVATE kvstore)
target_link_libraries(selfsimilarWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(selfsimilarWorkload PRIVATE rand)
target_link_libraries(selfsimilarWorkload PRIVATE Boost::boost)

add_library(mkvzipfianWorkload SHARED service/mkvzipfianWorkload.cu)
target_link_libraries(mkvzipfianWorkload PRIVATE libmegakv)
target_link_libraries(mkvzipfianWorkload PRIVATE tbbmalloc_proxy)
target_link_libraries(mkvzipfianWorkload PRIVATE rand)
target_link_libraries(mkvzipfianWorkload PRIVATE Boost::boost)

set(KVGPU_TARGETLIST ${KVGPU_TARGETLIST} kvstore rand)

install(TARGETS ${KVGPU_TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

set(CPACK_PACKAGE_VENDOR "dePaul Miller et. al.")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "dePaul Miller et. al.")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)
