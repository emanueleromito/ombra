// Small CUDA RAII helpers for the cache: stream, device array, bandwidth probe.
// Internal code returns core::Status; the public functions throw at the boundary.

#include <algorithm>
#include <cstdlib>
#include <new>
#include <stdexcept>

#include "cuda_throw.cuh"
#include "ombra/cache/gpu_cache.hpp"

namespace ombra::cache {

Stream::Stream() {
  cudaStream_t s = nullptr;
  OMBRA_CUDA_THROW(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  handle_ = s;
}

Stream::~Stream() {
  // A destructor cannot report; a failed destroy leaks one stream handle and nothing else.
  if (handle_ != nullptr) (void)cudaStreamDestroy(static_cast<cudaStream_t>(handle_));
}

void Stream::synchronize() {
  OMBRA_CUDA_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(handle_)));
}

DeviceArray::DeviceArray(std::size_t words) : words_(words) {
  void* p = nullptr;
  OMBRA_CUDA_THROW(cudaMalloc(&p, words * sizeof(std::uint64_t)));
  ptr_ = static_cast<std::uint64_t*>(p);
}

DeviceArray::~DeviceArray() {
  if (ptr_ != nullptr) (void)cudaFree(ptr_);
}

void DeviceArray::upload(const std::uint64_t* host, std::size_t words, std::size_t offset_words) {
  if (offset_words + words > words_) throw std::invalid_argument("device array: upload out of range");
  OMBRA_CUDA_THROW(cudaMemcpy(ptr_ + offset_words, host, words * sizeof(std::uint64_t),
                              cudaMemcpyHostToDevice));
}

void DeviceArray::download(std::uint64_t* host, std::size_t words, std::size_t offset_words) const {
  if (offset_words + words > words_) throw std::invalid_argument("device array: download out of range");
  OMBRA_CUDA_THROW(cudaMemcpy(host, ptr_ + offset_words, words * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToHost));
}

double measure_h2d_gbps(std::size_t bytes, bool pinned, int reps) {
  if (bytes == 0 || reps < 1) throw std::invalid_argument("measure_h2d_gbps: bad arguments");
  void* dev = nullptr;
  void* host = nullptr;
  OMBRA_CUDA_THROW(cudaMalloc(&dev, bytes));
  const auto cleanup = [&]() {
    if (dev != nullptr) (void)cudaFree(dev);
    if (host != nullptr) {
      if (pinned) {
        (void)cudaFreeHost(host);
      } else {
        std::free(host);
      }
    }
  };
  cudaEvent_t start = nullptr, stop = nullptr;
  double best = 0;
  try {
    if (pinned) {
      OMBRA_CUDA_THROW(cudaHostAlloc(&host, bytes, cudaHostAllocDefault));
    } else {
      host = std::malloc(bytes);
      if (host == nullptr) throw std::bad_alloc();
    }
    std::fill_n(static_cast<unsigned char*>(host), bytes, static_cast<unsigned char>(0x5a));
    OMBRA_CUDA_THROW(cudaEventCreate(&start));
    OMBRA_CUDA_THROW(cudaEventCreate(&stop));
    OMBRA_CUDA_THROW(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice));  // warm-up
    for (int i = 0; i < reps; ++i) {
      OMBRA_CUDA_THROW(cudaEventRecord(start, nullptr));
      OMBRA_CUDA_THROW(cudaMemcpyAsync(dev, host, bytes, cudaMemcpyHostToDevice, nullptr));
      OMBRA_CUDA_THROW(cudaEventRecord(stop, nullptr));
      OMBRA_CUDA_THROW(cudaEventSynchronize(stop));
      float ms = 0;
      OMBRA_CUDA_THROW(cudaEventElapsedTime(&ms, start, stop));
      best = std::max(best, static_cast<double>(bytes) / (ms / 1000.0) / 1e9);
    }
  } catch (...) {
    if (start != nullptr) (void)cudaEventDestroy(start);
    if (stop != nullptr) (void)cudaEventDestroy(stop);
    cleanup();
    throw;
  }
  (void)cudaEventDestroy(start);
  (void)cudaEventDestroy(stop);
  cleanup();
  return best;
}

}  // namespace ombra::cache
