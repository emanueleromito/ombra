#include "ombra/cache/arena_lru.hpp"

#include <stdexcept>

namespace ombra::cache {

ArenaLru::ArenaLru(std::size_t capacity, std::size_t alignment) : alloc_(capacity, alignment) {}

std::optional<std::size_t> ArenaLru::find_and_pin(const BlockKey& key) {
  auto it = map_.find(key);
  if (it == map_.end()) return std::nullopt;
  Node& n = it->second;
  if (n.pins++ == 0) pinned_bytes_ += alloc_.round_up(n.bytes);
  order_.splice(order_.begin(), order_, n.pos);
  return n.offset;
}

void ArenaLru::evict_node(std::unordered_map<BlockKey, Node, BlockKeyHash>::iterator it,
                          std::vector<Evicted>& evicted) {
  Node& n = it->second;
  evicted.push_back({it->first, n.offset, n.bytes});
  alloc_.free(n.offset);
  order_.erase(n.pos);
  map_.erase(it);
}

AdmitResult ArenaLru::admit(const BlockKey& key, std::size_t bytes, std::size_t& offset,
                            std::vector<Evicted>& evicted) {
  if (map_.count(key) != 0) throw std::invalid_argument("arena lru: key already resident");
  if (bytes == 0) throw std::invalid_argument("arena lru: empty block");
  const std::size_t need = alloc_.round_up(bytes);
  if (need > alloc_.capacity()) return AdmitResult::kTooLarge;
  if (need > alloc_.capacity() - pinned_bytes_) return AdmitResult::kBusy;

  std::optional<std::size_t> off = alloc_.allocate(bytes);
  // Evict from the least recently used end, skipping pinned blocks, until a range fits. The check
  // above guarantees enough unpinned bytes, but fragmentation may need more than one eviction.
  while (!off) {
    // Least recently used unpinned block (scan from the tail).
    auto victim = map_.end();
    for (auto rit = order_.rbegin(); rit != order_.rend(); ++rit) {
      auto mit = map_.find(*rit);
      if (mit->second.pins == 0) {
        victim = mit;
        break;
      }
    }
    if (victim == map_.end()) break;
    evict_node(victim, evicted);
    off = alloc_.allocate(bytes);
  }
  if (!off) return AdmitResult::kBusy;  // fragmentation by pinned blocks

  order_.push_front(key);
  map_[key] = Node{*off, bytes, 1, order_.begin()};
  pinned_bytes_ += need;
  offset = *off;
  return AdmitResult::kOk;
}

void ArenaLru::pin(const BlockKey& key) {
  auto it = map_.find(key);
  if (it == map_.end()) throw std::invalid_argument("arena lru: pin of a non-resident block");
  if (it->second.pins++ == 0) pinned_bytes_ += alloc_.round_up(it->second.bytes);
}

void ArenaLru::unpin(const BlockKey& key) {
  auto it = map_.find(key);
  if (it == map_.end() || it->second.pins == 0) {
    throw std::invalid_argument("arena lru: unpin of a block that is not pinned");
  }
  if (--it->second.pins == 0) pinned_bytes_ -= alloc_.round_up(it->second.bytes);
}

void ArenaLru::abandon(const BlockKey& key) {
  auto it = map_.find(key);
  if (it == map_.end() || it->second.pins != 1) {
    throw std::invalid_argument("arena lru: abandon needs a block with exactly one pin");
  }
  pinned_bytes_ -= alloc_.round_up(it->second.bytes);
  alloc_.free(it->second.offset);
  order_.erase(it->second.pos);
  map_.erase(it);
}

std::size_t ArenaLru::evict_all_unpinned(std::vector<Evicted>& evicted) {
  std::size_t n = 0;
  for (auto it = map_.begin(); it != map_.end();) {
    if (it->second.pins == 0) {
      auto cur = it++;
      evict_node(cur, evicted);
      ++n;
    } else {
      ++it;
    }
  }
  return n;
}

std::size_t ArenaLru::pins(const BlockKey& key) const {
  auto it = map_.find(key);
  return it == map_.end() ? 0 : it->second.pins;
}

}  // namespace ombra::cache
