cmake_minimum_required(VERSION 3.14)

project(Libgpuhash LANGUAGES CUDA CXX)

set(CXX_STANDARD 14)

message("BUILD TYPE: ${CMAKE_BUILD_TYPE}")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
    message("CUDA FLAGS: ${CMAKE_CUDA_FLAGS}")
endif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

if (NOT DEFINED CUDA_ARCH)
    set(CUDA_ARCH "70" "75")
    message("WARNING CUDA_ARCH IS GENERATING DEFAULT ${CUDA_ARCH}")
endif (NOT DEFINED CUDA_ARCH)

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda -Xptxas=-v -Xptxas -dlcm=cg")

foreach (ARCH_ELM ${CUDA_ARCH})
    message("${ARCH_ELM}")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH_ELM},code=sm_${ARCH_ELM}")
endforeach (ARCH_ELM)

string(APPEND CMAKE_CUDA_FLAGS " -Wreorder")

message(${CMAKE_CUDA_FLAGS})

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

enable_testing()

add_library(gpuhash SHARED gpu_hash.cu)
target_include_directories(gpuhash PUBLIC /usr/local/cuda/samples/common/inc/ .)

add_executable(gpuhash_test test/insert_test.cu)
target_link_libraries(gpuhash_test PRIVATE gpuhash)
add_test(gpuhash_test gpuhash_test)

set(TARGETLIST gpuhash)

install(TARGETS ${TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

set(CPACK_PACKAGE_VENDOR "dePaul Miller et. al.")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "dePaul Miller")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)
