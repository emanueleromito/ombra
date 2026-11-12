// Cache counters. Counters only: no ids, no segment keys, no data.
#pragma once

#include <cstdint>

namespace ombra::cache {

struct CacheMetrics {
  std::uint64_t hits = 0;            // blocks found on the GPU when requested
  std::uint64_t misses = 0;          // blocks that had to be copied to the GPU
  std::uint64_t evictions = 0;       // GPU blocks dropped to make room
  std::uint64_t bytes_paged = 0;     // bytes copied host to device
  std::uint64_t host_loads = 0;      // segments loaded from storage into the pinned pool
  std::uint64_t host_hits = 0;       // GPU misses served from the pinned pool
  std::uint64_t host_evictions = 0;  // segments dropped from the pinned pool
  std::uint64_t bytes_loaded = 0;    // bytes produced by the loader

  // hits / (hits + misses); 0 when nothing was requested.
  double hit_rate() const {
    const std::uint64_t total = hits + misses;
    return total == 0 ? 0.0 : static_cast<double>(hits) / static_cast<double>(total);
  }
};

}  // namespace ombra::cache
