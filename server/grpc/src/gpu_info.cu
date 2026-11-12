// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// GPU memory totals for /metrics: two numbers, nothing else.
#include <cuda_runtime_api.h>

#include <cstddef>

#include "gpu_info.hpp"

namespace ombra::server {

bool gpu_memory(std::uint64_t* used_bytes, std::uint64_t* total_bytes) {
  std::size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return false;
  *total_bytes = total_b;
  *used_bytes = total_b - free_b;
  return true;
}

}  // namespace ombra::server
