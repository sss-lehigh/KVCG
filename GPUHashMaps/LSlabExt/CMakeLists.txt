cmake_minimum_required(VERSION 3.14)

project(LSlabExt LANGUAGES CUDA CXX)

include(FetchContent)

FetchContent_Declare(
        UnifiedMemoryGroupAllocation
        GIT_REPOSITORY git@github.com:sss-lehigh/UnifiedMemoryGroupAllocation.git
        GIT_TAG main
)

# CMake 3.14+
FetchContent_MakeAvailable(UnifiedMemoryGroupAllocation)

set(CXX_STANDARD 14)

find_package(GTest CONFIG REQUIRED)

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

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda -Xptxas=-v")

foreach (ARCH_ELM ${CUDA_ARCH})
    message("${ARCH_ELM}")
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${ARCH_ELM},code=sm_${ARCH_ELM}")
endforeach (ARCH_ELM)

string(APPEND CMAKE_CUDA_FLAGS " -Wreorder")

message(${CMAKE_CUDA_FLAGS})

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

enable_testing()

add_library(lslabext INTERFACE)
target_include_directories(lslabext INTERFACE include)
target_link_libraries(lslabext INTERFACE allocator)

add_executable(lslabext_benchmark benchmark/benchmark.cu)
target_link_libraries(lslabext_benchmark lslabext)

add_executable(lslabext_memory_benchmark benchmark/benchmark_mem.cu)
target_link_libraries(lslabext_memory_benchmark lslabext)

add_executable(slabunified_test test/slabunified_test.cu)
target_link_libraries(slabunified_test PRIVATE GTest::gtest GTest::gmock GTest::gtest_main lslabext)
add_test(slabunified_test slabunified_test)

set(TARGETLIST lslabext)

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
