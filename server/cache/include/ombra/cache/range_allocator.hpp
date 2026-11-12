// Offset allocator over a fixed arena. Pure bookkeeping: it hands out byte offsets inside a
// buffer of known capacity (a pinned host pool or a device buffer) and never touches memory, so it
// is tested on the CPU. First fit, with coalescing of neighbouring free ranges on free().
#pragma once

#include <cstddef>
#include <map>
#include <optional>

namespace ombra::cache {

class RangeAllocator {
 public:
  // `alignment` must be a power of two; every offset and size is rounded up to it.
  explicit RangeAllocator(std::size_t capacity, std::size_t alignment = 256);

  // Returns the offset of a free range of at least `bytes` bytes, or nullopt if no single free
  // range is large enough (the arena may be fragmented). bytes == 0 is rejected (nullopt).
  std::optional<std::size_t> allocate(std::size_t bytes);
  // Throws std::invalid_argument if `offset` is not the start of a live allocation.
  void free(std::size_t offset);

  std::size_t capacity() const { return capacity_; }
  std::size_t used_bytes() const { return used_bytes_; }
  std::size_t largest_free() const;
  std::size_t round_up(std::size_t bytes) const {
    return (bytes + alignment_ - 1) & ~(alignment_ - 1);
  }

 private:
  std::size_t capacity_;  // rounded down to the alignment
  std::size_t alignment_;
  std::size_t used_bytes_ = 0;
  std::map<std::size_t, std::size_t> free_;  // offset -> length
  std::map<std::size_t, std::size_t> used_;  // offset -> length
};

}  // namespace ombra::cache
