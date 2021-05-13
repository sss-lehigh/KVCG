//
// Created by depaulsmiller on 9/23/20.
//

#ifndef GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH
#define GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH

template<typename T>
struct EMPTY {
    static constexpr T value{};
};

template<typename T>
__forceinline__ __host__ __device__ unsigned compare(const T &lhs, const T &rhs);

#endif //GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH
