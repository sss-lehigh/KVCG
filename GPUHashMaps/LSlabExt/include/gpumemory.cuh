#include "gpuErrchk.cuh"
#include <iostream>

#ifndef GPUMEMORY_CUH
#define GPUMEMORY_CUH

template<typename T>
struct GPUCPUMemory {

    GPUCPUMemory() : host(nullptr), size(0), device(nullptr) {
    }

    GPUCPUMemory(size_t size) : GPUCPUMemory(new T[size], size) {}

    GPUCPUMemory(T *h, size_t size) : host(h), size(size), device(new T *[1]) {
        gpuErrchk(cudaMalloc(&device[0], sizeof(T) * size))
    }

    GPUCPUMemory(GPUCPUMemory<T> &&ref) noexcept {
        host = ref.host;
        device = ref.device;
        size = ref.size;
        ref.host = nullptr;
        ref.device = nullptr;
    }

    ~GPUCPUMemory() {
        if (device != nullptr) {
            std::cerr << "Deleting memory\n";
            gpuErrchk(cudaFree(*device))
            delete[] device;
        }
    }

    GPUCPUMemory<T> &operator=(GPUCPUMemory<T> &&other) {
        if (&other != this) {
            if (device != nullptr) {
                gpuErrchk(cudaFree(*device))
                delete[] device;
            }
            host = other.host;
            device = other.device;
            size = other.size;
            other.host = nullptr;
            other.device = nullptr;
        }
        return *this;
    }

    void movetoGPU() {
        gpuErrchk(
                cudaMemcpy(*device, host, sizeof(T) * size, cudaMemcpyHostToDevice))
    }

    void movetoCPU() {
        gpuErrchk(
                cudaMemcpy(host, *device, sizeof(T) * size, cudaMemcpyDeviceToHost))
    }

    T *getDevice() {
        return *device;
    }

    T *host;
    size_t size;
private:
    T **device;
};

template<typename T>
struct GPUCPU2DArray {

    GPUCPU2DArray(size_t dim1, size_t dim2) : dim1(dim1), dim2(dim2), outer(dim1), inner(new GPUCPUMemory<T>[dim1]) {
        for (size_t i = 0; i < dim1; i++) {
            inner[i] = GPUCPUMemory<T>(dim2);
            outer.host[i] = inner[i].getDevice();
        }
    }

    ~GPUCPU2DArray() {
    }

    void movetoGPU() {
        for (int i = 0; i < dim1; i++) {
            inner[i].movetoGPU();
        }
        outer.movetoGPU();
    }

    void print() {
        for (int i = 0; i < dim1; i++) {
            std::cerr << outer.host[i] << std::endl;
            for (int j = 0; j < dim2; j++)
                std::cerr << "\t" << inner[i].host[j] << std::endl;
        }
    }


    void movetoCPU() {
        for (int i = 0; i < dim1; i++) {
            inner[i].movetoCPU();
        }
        outer.movetoCPU();
    }

    T **getDevice2DArray() {
        return outer.getDevice();
    }

    T *&operator[](int idx) {
        return inner[idx].host;
    }

    size_t dim1;
    size_t dim2;
    GPUCPUMemory<T *> outer;
    GPUCPUMemory<T> *inner;
};

#endif