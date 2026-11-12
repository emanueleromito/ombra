// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/cache/range_allocator.hpp"

#include <stdexcept>

namespace ombra::cache {

RangeAllocator::RangeAllocator(std::size_t capacity, std::size_t alignment)
    : capacity_(0), alignment_(alignment) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("range allocator: alignment must be a power of two");
  }
  capacity_ = capacity & ~(alignment_ - 1);
  if (capacity_ > 0) free_[0] = capacity_;
}

std::optional<std::size_t> RangeAllocator::allocate(std::size_t bytes) {
  if (bytes == 0 || bytes > capacity_) return std::nullopt;
  const std::size_t need = round_up(bytes);
  if (need > capacity_) return std::nullopt;
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < need) continue;
    const std::size_t off = it->first;
    const std::size_t len = it->second;
    free_.erase(it);
    if (len > need) free_[off + need] = len - need;
    used_[off] = need;
    used_bytes_ += need;
    return off;
  }
  return std::nullopt;
}

void RangeAllocator::free(std::size_t offset) {
  auto u = used_.find(offset);
  if (u == used_.end()) throw std::invalid_argument("range allocator: free of unknown offset");
  std::size_t off = u->first;
  std::size_t len = u->second;
  used_.erase(u);
  used_bytes_ -= len;
  // Merge with the following free range.
  auto next = free_.find(off + len);
  if (next != free_.end()) {
    len += next->second;
    free_.erase(next);
  }
  // Merge with the preceding free range.
  auto after = free_.lower_bound(off);
  if (after != free_.begin()) {
    auto prev = std::prev(after);
    if (prev->first + prev->second == off) {
      off = prev->first;
      len += prev->second;
      free_.erase(prev);
    }
  }
  free_[off] = len;
}

std::size_t RangeAllocator::largest_free() const {
  std::size_t best = 0;
  for (const auto& kv : free_) best = kv.second > best ? kv.second : best;
  return best;
}

}  // namespace ombra::cache
