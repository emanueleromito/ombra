// GPU cluster cache: pinned host pool + GPU cache + asynchronous paging. See
// include/ombra/cache/gpu_cache.hpp for the contract. The eviction policy and the offset
// bookkeeping are in arena_lru.cpp / range_allocator.cpp (pure C++, unit tested on the CPU).
//
// Ordering rules (all device work for the cache is on `copy_stream`, the caller's kernels on its
// own compute stream):
//   * copy -> compute: next() makes the compute stream wait on the block's `ready` event.
//   * compute -> copy: releasing a lease records `last_use` on the compute stream; when that block
//   is
//     evicted, the copy stream waits on `last_use` before the memory is reused.
//   * host pool: a pinned host block stays pinned until the event recorded after its copy has
//     completed, so a segment is never overwritten while the DMA reads it.

#include <cstring>
#include <mutex>
#include <unordered_map>

#include "cuda_throw.cuh"
#include "ombra/cache/gpu_cache.hpp"

namespace ombra::cache {
namespace {

constexpr std::size_t kAlign = 256;
constexpr std::size_t kWordBytes = sizeof(std::uint64_t);

class Event {
 public:
  Event() = default;
  ~Event() {
    if (e_ != nullptr) (void)cudaEventDestroy(e_);
  }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&& o) noexcept : e_(std::exchange(o.e_, nullptr)) {}
  Event& operator=(Event&& o) noexcept {
    if (this != &o) {
      if (e_ != nullptr) (void)cudaEventDestroy(e_);
      e_ = std::exchange(o.e_, nullptr);
    }
    return *this;
  }
  void create() { OMBRA_CUDA_THROW(cudaEventCreateWithFlags(&e_, cudaEventDisableTiming)); }
  cudaEvent_t get() const { return e_; }

 private:
  cudaEvent_t e_ = nullptr;
};

class PinnedBuffer {
 public:
  explicit PinnedBuffer(std::size_t bytes) {
    void* p = nullptr;
    OMBRA_CUDA_THROW(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
    ptr_ = static_cast<unsigned char*>(p);
  }
  ~PinnedBuffer() {
    if (ptr_ != nullptr) (void)cudaFreeHost(ptr_);
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  unsigned char* get() const { return ptr_; }

 private:
  unsigned char* ptr_ = nullptr;
};

class DeviceBytes {
 public:
  explicit DeviceBytes(std::size_t bytes) {
    void* p = nullptr;
    OMBRA_CUDA_THROW(cudaMalloc(&p, bytes));
    ptr_ = static_cast<unsigned char*>(p);
  }
  ~DeviceBytes() {
    if (ptr_ != nullptr) (void)cudaFree(ptr_);
  }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  unsigned char* get() const { return ptr_; }

 private:
  unsigned char* ptr_ = nullptr;
};

std::size_t round_down(std::size_t x) { return x & ~(kAlign - 1); }

}  // namespace

struct GpuClusterCache::Impl {
  struct GpuEntry {
    Event ready;     // recorded on the copy stream after the block's copy
    Event last_use;  // recorded on the compute stream when the last lease was released
    bool used = false;
    std::size_t offset = 0;
    std::size_t bytes = 0;
  };
  struct Inflight {
    BlockKey key;  // host block to unpin once the copy is done
    Event done;
  };

  CacheConfig cfg;
  BlockLoader loader;
  mutable std::mutex mu;
  ArenaLru host_lru, gpu_lru;
  PinnedBuffer host_mem;
  DeviceBytes dev_mem;
  Stream copy_stream;
  std::unordered_map<BlockKey, GpuEntry, BlockKeyHash> entries;
  std::vector<Inflight> inflight;
  std::vector<BlockKey> order;
  std::vector<char> staged;  // per plan position: a pin is held for it already
  std::size_t cursor = 0;
  CacheMetrics m;

  Impl(const CacheConfig& c, BlockLoader l)
      : cfg(c),
        loader(std::move(l)),
        host_lru(round_down(c.host_budget_bytes), kAlign),
        gpu_lru(round_down(c.gpu_budget_bytes), kAlign),
        host_mem(check_budget(round_down(c.host_budget_bytes), "host")),
        dev_mem(check_budget(round_down(c.gpu_budget_bytes), "gpu")) {
    if (!loader.size_words || !loader.load) throw std::invalid_argument("cache: loader not set");
  }

  ~Impl() {
    // Finish the copies before the buffers and events go away (errors cannot be reported here).
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(copy_stream.get()));
  }

  static std::size_t check_budget(std::size_t bytes, const char* what) {
    if (bytes == 0) throw std::invalid_argument(std::string("cache: ") + what + " budget is zero");
    return bytes;
  }

  cudaStream_t copy() const { return static_cast<cudaStream_t>(copy_stream.get()); }

  // Unpins host blocks whose copy has finished. block = wait for all of them.
  void reap(bool block) {
    for (std::size_t i = 0; i < inflight.size();) {
      bool done = block;
      if (block) {
        OMBRA_CUDA_THROW(cudaEventSynchronize(inflight[i].done.get()));
      } else {
        const cudaError_t q = cudaEventQuery(inflight[i].done.get());
        if (q == cudaSuccess) {
          done = true;
        } else if (q == cudaErrorNotReady) {
          (void)cudaGetLastError();  // clear the "not ready" status
        } else {
          OMBRA_CUDA_THROW(cudaEventQuery(inflight[i].done.get()));  // throws the real error
        }
      }
      if (!done) {
        ++i;
        continue;
      }
      host_lru.unpin(inflight[i].key);
      inflight[i] = std::move(inflight.back());
      inflight.pop_back();
    }
  }

  void drop_gpu(const std::vector<Evicted>& ev, bool count) {
    for (const Evicted& e : ev) {
      auto it = entries.find(e.key);
      if (it == entries.end()) continue;
      // The memory is reused by a later copy on the copy stream: it must not start before the last
      // kernel that read this block has finished.
      if (it->second.used) {
        OMBRA_CUDA_THROW(cudaStreamWaitEvent(copy(), it->second.last_use.get(), 0));
      }
      entries.erase(it);
      if (count) ++m.evictions;
    }
  }

  // Pinned host copy of a block, loading it from storage if needed (one pin added).
  const unsigned char* host_acquire(const BlockKey& key, std::size_t words) {
    if (auto off = host_lru.find_and_pin(key)) {
      ++m.host_hits;
      return host_mem.get() + *off;
    }
    const std::size_t bytes = words * kWordBytes;
    std::vector<Evicted> ev;
    std::size_t off = 0;
    AdmitResult r = host_lru.admit(key, bytes, off, ev);
    m.host_evictions += ev.size();
    if (r == AdmitResult::kBusy) {
      reap(true);  // blocks still being copied hold the pool; wait for them
      ev.clear();
      r = host_lru.admit(key, bytes, off, ev);
      m.host_evictions += ev.size();
    }
    if (r == AdmitResult::kTooLarge)
      throw CacheError("cache: a block is larger than the host pool");
    if (r == AdmitResult::kBusy)
      throw CacheError("cache: host pool too small for the blocks in use");
    try {
      loader.load(key, reinterpret_cast<std::uint64_t*>(host_mem.get() + off), words);
    } catch (...) {
      host_lru.abandon(key);
      throw;
    }
    ++m.host_loads;
    m.bytes_loaded += bytes;
    return host_mem.get() + off;
  }

  // Makes `key` resident on the GPU and adds one pin. false only for a prefetch (demand = false)
  // that does not fit next to the pinned blocks. Hits and misses are counted here, once per call.
  bool acquire(const BlockKey& key, bool demand) {
    reap(false);
    if (gpu_lru.find_and_pin(key)) {
      ++m.hits;
      return true;
    }
    const std::size_t words = loader.size_words(key);
    if (words == 0) throw CacheError("cache: empty block");
    const std::size_t bytes = words * kWordBytes;
    if (bytes > gpu_lru.capacity())
      throw CacheError("cache: a block is larger than the GPU budget");

    const unsigned char* host_ptr = host_acquire(key, words);
    std::size_t goff = 0;
    std::vector<Evicted> ev;
    const AdmitResult r = gpu_lru.admit(key, bytes, goff, ev);
    drop_gpu(ev, true);
    if (r != AdmitResult::kOk) {
      host_lru.unpin(key);
      if (r == AdmitResult::kTooLarge)
        throw CacheError("cache: a block is larger than the GPU budget");
      if (demand) throw CacheError("cache: GPU budget too small for the blocks in use");
      return false;
    }

    GpuEntry entry;
    entry.ready.create();
    entry.last_use.create();
    entry.offset = goff;
    entry.bytes = bytes;
    Inflight fl{key, {}};
    fl.done.create();
    OMBRA_CUDA_THROW(
        cudaMemcpyAsync(dev_mem.get() + goff, host_ptr, bytes, cudaMemcpyHostToDevice, copy()));
    OMBRA_CUDA_THROW(cudaEventRecord(entry.ready.get(), copy()));
    OMBRA_CUDA_THROW(cudaEventRecord(fl.done.get(), copy()));
    entries.emplace(key, std::move(entry));
    inflight.push_back(std::move(fl));
    ++m.misses;
    m.bytes_paged += bytes;
    return true;
  }
};

GpuClusterCache::GpuClusterCache(const CacheConfig& config, BlockLoader loader)
    : impl_(std::make_unique<Impl>(config, std::move(loader))) {}

GpuClusterCache::~GpuClusterCache() = default;

void GpuClusterCache::Lease::release() noexcept {
  if (cache_ == nullptr) return;
  cache_->release_lease(key_, stream_);
  cache_ = nullptr;
  ptr_ = nullptr;
}

void GpuClusterCache::release_lease(const BlockKey& key, void* compute_stream) noexcept {
  // A destructor path: nothing can be reported. A failed event record only means a later eviction
  // does not wait for this lease's kernels, which is why record errors are not ignored silently in
  // next(); here the stream was valid when the lease was created.
  try {
    std::lock_guard<std::mutex> lock(impl_->mu);
    auto it = impl_->entries.find(key);
    if (it != impl_->entries.end()) {
      (void)cudaEventRecord(it->second.last_use.get(), static_cast<cudaStream_t>(compute_stream));
      it->second.used = true;
    }
    impl_->gpu_lru.unpin(key);
  } catch (...) {
  }
}

void GpuClusterCache::plan(const std::vector<BlockKey>& order) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  Impl& s = *impl_;
  for (std::size_t p = s.cursor; p < s.order.size(); ++p) {
    if (s.staged[p]) s.gpu_lru.unpin(s.order[p]);
  }
  s.order = order;
  s.staged.assign(order.size(), 0);
  s.cursor = 0;
}

GpuClusterCache::Lease GpuClusterCache::next(Stream& compute) {
  std::lock_guard<std::mutex> lock(impl_->mu);
  Impl& s = *impl_;
  if (s.cursor >= s.order.size()) throw CacheError("cache: the plan is exhausted");
  const std::size_t pos = s.cursor++;
  const BlockKey key = s.order[pos];
  if (s.staged[pos]) {
    s.staged[pos] = 0;  // the pin taken by the prefetch becomes the lease's pin
  } else {
    s.acquire(key, true);
  }
  Lease lease;  // from here the pin is released on any exception
  {
    const auto& e = s.entries.at(key);
    lease.cache_ = this;
    lease.key_ = key;
    lease.ptr_ = reinterpret_cast<const std::uint64_t*>(s.dev_mem.get() + e.offset);
    lease.words_ = e.bytes / kWordBytes;
    lease.stream_ = compute.get();
    OMBRA_CUDA_THROW(
        cudaStreamWaitEvent(static_cast<cudaStream_t>(compute.get()), e.ready.get(), 0));
  }
  for (std::size_t i = 1; i <= s.cfg.lookahead && pos + i < s.order.size(); ++i) {
    const std::size_t p = pos + i;
    if (s.staged[p]) continue;
    if (!s.acquire(s.order[p], false)) break;
    s.staged[p] = 1;
  }
  return lease;
}

std::size_t GpuClusterCache::remaining() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->order.size() - impl_->cursor;
}

CacheMetrics GpuClusterCache::metrics() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->m;
}

void GpuClusterCache::reset_metrics() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  impl_->m = CacheMetrics{};
}

void GpuClusterCache::clear_gpu() {
  std::lock_guard<std::mutex> lock(impl_->mu);
  std::vector<Evicted> ev;
  impl_->gpu_lru.evict_all_unpinned(ev);
  impl_->drop_gpu(ev, false);
}

}  // namespace ombra::cache
