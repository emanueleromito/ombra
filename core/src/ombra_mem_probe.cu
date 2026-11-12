// ombra_mem_probe: measures what the driver really reserves for many ciphertext-sized allocations.
//
// Usage: ombra_mem_probe <bytes_per_buffer> <count>
//
// Allocates `count` separate buffers of `bytes_per_buffer` (cudaMalloc each) and reports the
// change in free device memory (cudaMemGetInfo) per buffer, then does the same with the buffers
// carved from one slab of the same total nominal size. Prints sizes only; no data. Used by
// tests/test_gpu_mem_probe.py to document the nominal-versus-actual footprint.
// Exit code: 0 ok, 2 error.

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>

#include "ombra/core/cuda_check.hpp"

namespace {

using ombra::core::Status;

Status free_bytes(std::size_t& out) {
  std::size_t total = 0;
  OMBRA_CUDA_CHECK(cudaMemGetInfo(&out, &total));
  return {};
}

Status run(std::size_t bytes, std::size_t count) {
  OMBRA_CUDA_CHECK(cudaFree(nullptr));  // create the context before the first reading
  std::size_t before = 0, after = 0;
  OMBRA_TRY(free_bytes(before));
  std::vector<void*> ptrs(count, nullptr);
  for (std::size_t i = 0; i < count; ++i) {
    OMBRA_CUDA_CHECK(cudaMalloc(&ptrs[i], bytes));
  }
  OMBRA_TRY(free_bytes(after));
  const double separate = static_cast<double>(before - after) / static_cast<double>(count);
  for (void* p : ptrs) OMBRA_CUDA_CHECK(cudaFree(p));

  OMBRA_TRY(free_bytes(before));
  void* slab = nullptr;
  OMBRA_CUDA_CHECK(cudaMalloc(&slab, bytes * count));
  OMBRA_TRY(free_bytes(after));
  const double slabbed = static_cast<double>(before - after) / static_cast<double>(count);
  OMBRA_CUDA_CHECK(cudaFree(slab));

  std::printf("nominal_bytes_per_buffer=%zu\n", bytes);
  std::printf("separate_cudamalloc_actual_bytes_per_buffer=%.1f\n", separate);
  std::printf("slab_actual_bytes_per_buffer=%.1f\n", slabbed);
  std::printf("separate_over_nominal=%.3f\n", separate / static_cast<double>(bytes));
  std::printf("slab_over_nominal=%.3f\n", slabbed / static_cast<double>(bytes));
  return {};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: ombra_mem_probe <bytes_per_buffer> <count>\n");
    return 2;
  }
  try {
    run(std::stoull(argv[1]), std::stoull(argv[2])).throw_if_error();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
