// Runs the (unchanged) column search kernel over blocks served by the GPU cluster cache.
// Header only, no CUDA types, no OpenFHE: shared by the paging test driver and the benchmark.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ombra/cache/gpu_cache.hpp"
#include "ombra/core/column_search.hpp"

namespace ombra::cache {

// Scores every block of `order` against `query` (shape.query_words() device words), one kernel
// launch per block on `compute`, blocks paged in by `cache` (copy of block i+1 overlaps the kernel
// of block i). Result slot i (3 * poly_words words at out.get() + i * 3 * poly_words) is the score
// of order[i]. With `serial` the compute stream is drained after each block, which removes the
// overlap (benchmark baseline). Returns after all kernels have finished.
inline void run_through_cache(GpuClusterCache& cache, const core::ColumnSearchLauncher& launcher,
                              const core::ColumnSearchShape& shape, const DeviceArray& query,
                              const std::vector<BlockKey>& order, Stream& compute,
                              DeviceArray& out, bool serial = false) {
  const std::size_t out_words = 3 * shape.poly_words();
  cache.plan(order);
  for (std::size_t i = 0; i < order.size(); ++i) {
    GpuClusterCache::Lease lease = cache.next(compute);
    launcher.launch(lease.device_ptr(), query.get(), out.get() + i * out_words, compute.get());
    // lease goes out of scope: records the last use of the block on `compute`
    lease.release();
    if (serial) compute.synchronize();
  }
  compute.synchronize();
}

}  // namespace ombra::cache
