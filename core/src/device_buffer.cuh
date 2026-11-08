// Owning device buffer (RAII). Allocation reports a Status instead of throwing.
#pragma once

#include <cstddef>
#include <cstdio>
#include <utility>

#include "ombra/core/cuda_check.hpp"

namespace ombra::core {

template <typename T>
class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(std::exchange(o.ptr_, nullptr)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
        }
        return *this;
    }

    Status allocate(std::size_t count) {
        release();
        OMBRA_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(T)));
        return {};
    }
    T* get() const { return ptr_; }

  private:
    void release() noexcept {
        if (ptr_ == nullptr) return;
        // A destructor cannot return a Status; report a failed free on stderr (no data in it).
        const cudaError_t err = cudaFree(ptr_);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "ombra: cudaFree failed: %s\n", cudaGetErrorString(err));
        }
        ptr_ = nullptr;
    }
    T* ptr_ = nullptr;
};

}  // namespace ombra::core
