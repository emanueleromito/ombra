// GPU cluster cache. Unit of caching = one segment (cluster block) as device-ready RNS words
// (the layout the column search kernel reads), keyed by (collection, segment).
//
//   storage --loader--> pinned host pool (LRU) --async H2D on a copy stream--> GPU cache (LRU)
//
// Search code announces the blocks it will need with plan(), then calls next() once per block.
// next() returns a pinned Lease for the next block of the plan and, before returning, starts the
// copy of the following `lookahead` blocks on the copy stream, so that copy and compute overlap
// (double buffering at lookahead 1). A leased block is never evicted. The compute stream waits on
// the block's copy event inside next(); the lease's release records an event on the compute stream,
// and any later copy into that memory waits for it, so a kernel still running on an evicted block
// is never overwritten.
//
// Threading: public calls are serialized by an internal mutex; a Lease must be destroyed before the
// cache. Metrics are counters only (no keys, no data). The header has no CUDA types, so it can be
// included from plain C++; streams are passed as the opaque `Stream` wrapper.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "ombra/cache/arena_lru.hpp"
#include "ombra/cache/cache_metrics.hpp"

namespace ombra::cache {

// Budget problems the caller can act on: one block larger than a budget, or every byte of the GPU
// budget pinned by running work. CUDA failures throw ombra::core::CudaError instead.
class CacheError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// RAII CUDA stream (cudaStreamCreateWithFlags, non-blocking).
class Stream {
 public:
  Stream();
  ~Stream();
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  void* get() const { return handle_; }  // cudaStream_t
  void synchronize();

 private:
  void* handle_ = nullptr;
};

// RAII device array of 64-bit words, for queries and results.
class DeviceArray {
 public:
  explicit DeviceArray(std::size_t words);
  ~DeviceArray();
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  std::uint64_t* get() const { return ptr_; }
  std::size_t words() const { return words_; }
  void upload(const std::uint64_t* host, std::size_t words, std::size_t offset_words = 0);
  void download(std::uint64_t* host, std::size_t words, std::size_t offset_words = 0) const;

 private:
  std::uint64_t* ptr_ = nullptr;
  std::size_t words_ = 0;
};

// Host-to-device bandwidth probe: copies `bytes` `reps` times and returns the best-case GB/s
// (1e9 bytes per second) from CUDA events. `pinned` selects cudaHostAlloc or pageable memory.
double measure_h2d_gbps(std::size_t bytes, bool pinned, int reps);

// The loader callback: how big a block is and how to produce its host form. Called with the cache
// lock held. `load` writes exactly `words` words into `dst` (pinned memory). Typically it reads the
// segment from storage and turns each column envelope into device words with the existing core
// code (flatten_ciphertexts).
struct BlockLoader {
  std::function<std::size_t(const BlockKey&)> size_words;
  std::function<void(const BlockKey&, std::uint64_t* dst, std::size_t words)> load;
};

struct CacheConfig {
  std::size_t gpu_budget_bytes = 0;
  std::size_t host_budget_bytes = 0;  // pinned pool, cudaHostAlloc'd once
  std::size_t lookahead = 1;          // blocks to copy ahead of the one being returned; 0 = none
};

class GpuClusterCache {
 public:
  GpuClusterCache(const CacheConfig& config, BlockLoader loader);
  ~GpuClusterCache();
  GpuClusterCache(const GpuClusterCache&) = delete;
  GpuClusterCache& operator=(const GpuClusterCache&) = delete;

  // A block pinned on the GPU. Movable. Destroying it (or release()) records a "last use" event on
  // the compute stream it was obtained for and unpins the block.
  class Lease {
   public:
    Lease() = default;
    Lease(Lease&& o) noexcept { swap(o); }
    Lease& operator=(Lease&& o) noexcept {
      if (this != &o) {
        release();
        swap(o);
      }
      return *this;
    }
    ~Lease() { release(); }
    Lease(const Lease&) = delete;
    Lease& operator=(const Lease&) = delete;

    const std::uint64_t* device_ptr() const { return ptr_; }
    std::size_t words() const { return words_; }
    bool valid() const { return cache_ != nullptr; }
    void release() noexcept;

   private:
    friend class GpuClusterCache;
    void swap(Lease& o) noexcept {
      std::swap(cache_, o.cache_);
      std::swap(key_, o.key_);
      std::swap(ptr_, o.ptr_);
      std::swap(words_, o.words_);
      std::swap(stream_, o.stream_);
    }
    GpuClusterCache* cache_ = nullptr;
    BlockKey key_;
    const std::uint64_t* ptr_ = nullptr;
    std::size_t words_ = 0;
    void* stream_ = nullptr;
  };

  // "I will need these blocks, in this order." Replaces any earlier plan (blocks staged for it are
  // unpinned). Nothing is copied until next().
  void plan(const std::vector<BlockKey>& order);
  // The next block of the plan, resident and pinned; `compute` waits for its copy. Starts the copy
  // of the following `lookahead` blocks. Throws CacheError if the block alone exceeds the GPU or
  // host budget, or if the GPU budget cannot hold it next to the blocks that are pinned.
  Lease next(Stream& compute);
  // Blocks left in the plan.
  std::size_t remaining() const;

  CacheMetrics metrics() const;
  void reset_metrics();
  // Drops every unpinned block from the GPU cache (not counted as evictions): a cold GPU, with the
  // pinned pool untouched. For benchmarks.
  void clear_gpu();

 private:
  void release_lease(const BlockKey& key, void* compute_stream) noexcept;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ombra::cache
