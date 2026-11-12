#include "device_util.hpp"
#include "ombra/core/cuda_check.hpp"

namespace ombra::search {
namespace {

using ombra::core::Status;

Status copy_impl(std::uint64_t* dst, const std::uint64_t* src, std::size_t words) {
  OMBRA_CUDA_CHECK(cudaMemcpy(dst, src, words * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice));
  return {};
}

Status sync_impl() {
  OMBRA_CUDA_CHECK(cudaDeviceSynchronize());
  return {};
}

}  // namespace

void device_copy_words(std::uint64_t* dst, const std::uint64_t* src, std::size_t words) {
  copy_impl(dst, src, words).throw_if_error();
}

void device_sync() { sync_impl().throw_if_error(); }

}  // namespace ombra::search
