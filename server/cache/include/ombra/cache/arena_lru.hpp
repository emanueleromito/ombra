// LRU policy with pinning over an arena. Pure bookkeeping, no memory is touched: used for
// the pinned host pool and for the GPU cache. A key is one segment (cluster block) of a collection.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ombra/cache/range_allocator.hpp"

namespace ombra::cache {

struct BlockKey {
  std::string collection;
  std::uint32_t segment = 0;
  bool operator==(const BlockKey& o) const {
    return segment == o.segment && collection == o.collection;
  }
  bool operator!=(const BlockKey& o) const { return !(*this == o); }
};

struct BlockKeyHash {
  std::size_t operator()(const BlockKey& k) const {
    return std::hash<std::string>()(k.collection) * 1000003u ^ std::hash<std::uint32_t>()(k.segment);
  }
};

// A block removed to make room.
struct Evicted {
  BlockKey key;
  std::size_t offset = 0;
  std::size_t bytes = 0;
};

enum class AdmitResult {
  kOk,        // inserted, pinned once
  kTooLarge,  // the block alone exceeds the arena: can never fit
  kBusy       // it fits the arena but pinned blocks leave too little room right now
};

class ArenaLru {
 public:
  explicit ArenaLru(std::size_t capacity, std::size_t alignment = 256);

  // Resident: pins it (one more pin), makes it most recently used, returns its offset.
  std::optional<std::size_t> find_and_pin(const BlockKey& key);
  // Not resident: makes room by evicting least recently used unpinned blocks (appended to
  // `evicted`, in order) and inserts the block pinned once at `offset`. kTooLarge never evicts and
  // kBusy evicts nothing unless pinned blocks fragment the arena (then `evicted` lists what was
  // dropped on the way: the caller must process it whatever the result). The key must not be
  // resident.
  AdmitResult admit(const BlockKey& key, std::size_t bytes, std::size_t& offset,
                    std::vector<Evicted>& evicted);
  void pin(const BlockKey& key);    // throws std::invalid_argument if not resident
  void unpin(const BlockKey& key);  // throws if not resident or not pinned
  // Removes a block that has exactly one pin (just admitted, e.g. its load failed). Throws
  // std::invalid_argument otherwise.
  void abandon(const BlockKey& key);
  // Drops every unpinned block (appended to `evicted`); returns how many were dropped.
  std::size_t evict_all_unpinned(std::vector<Evicted>& evicted);

  bool contains(const BlockKey& key) const { return map_.count(key) != 0; }
  std::size_t pins(const BlockKey& key) const;
  std::size_t resident_blocks() const { return map_.size(); }
  std::size_t used_bytes() const { return alloc_.used_bytes(); }
  std::size_t pinned_bytes() const { return pinned_bytes_; }
  std::size_t capacity() const { return alloc_.capacity(); }

 private:
  struct Node {
    std::size_t offset, bytes, pins;
    std::list<BlockKey>::iterator pos;
  };
  void evict_node(std::unordered_map<BlockKey, Node, BlockKeyHash>::iterator it,
                  std::vector<Evicted>& evicted);

  RangeAllocator alloc_;
  std::list<BlockKey> order_;  // front = most recently used
  std::unordered_map<BlockKey, Node, BlockKeyHash> map_;
  std::size_t pinned_bytes_ = 0;  // aligned bytes of blocks with pins > 0
};

}  // namespace ombra::cache
