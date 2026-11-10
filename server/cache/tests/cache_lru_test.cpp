#include <vector>

#include "check.hpp"
#include "ombra/cache/arena_lru.hpp"
#include "ombra/cache/cache_metrics.hpp"

using namespace ombra::cache;

static BlockKey K(std::uint32_t s, const char* c = "col") { return BlockKey{c, s}; }

int main() {
  ArenaLru lru(4 * 1024, 1024);  // room for four 1 KiB blocks
  std::vector<Evicted> ev;
  std::size_t off = 0;
  for (std::uint32_t i = 0; i < 4; ++i) {
    CHECK(lru.admit(K(i), 1024, off, ev) == AdmitResult::kOk);
    lru.unpin(K(i));
  }
  CHECK(ev.empty() && lru.resident_blocks() == 4);

  // Touch 0 so that 1 is now the least recently used; admitting 4 evicts exactly 1.
  CHECK(lru.find_and_pin(K(0)).has_value());
  lru.unpin(K(0));
  CHECK(lru.admit(K(4), 1024, off, ev) == AdmitResult::kOk);
  CHECK(ev.size() == 1 && ev[0].key == K(1));
  CHECK(!lru.contains(K(1)) && lru.contains(K(0)));
  lru.unpin(K(4));

  // Different collection, same segment number: a different key.
  CHECK(!lru.contains(K(0, "other")));

  // Pinned blocks are never evicted. Pin 0, 2, 3; only 4 is evictable.
  ev.clear();
  lru.pin(K(0));
  lru.pin(K(2));
  lru.pin(K(3));
  CHECK(lru.admit(K(10), 1024, off, ev) == AdmitResult::kOk);
  CHECK(ev.size() == 1 && ev[0].key == K(4));
  CHECK(lru.contains(K(0)) && lru.contains(K(2)) && lru.contains(K(3)));
  // Everything is pinned now: busy, nothing evicted.
  ev.clear();
  CHECK(lru.admit(K(11), 1024, off, ev) == AdmitResult::kBusy);
  CHECK(ev.empty() && lru.contains(K(10)));
  lru.unpin(K(10));
  CHECK(lru.admit(K(11), 1024, off, ev) == AdmitResult::kOk && ev.size() == 1);

  // A block larger than the arena can never fit.
  CHECK(lru.admit(K(99), 5 * 1024, off, ev) == AdmitResult::kTooLarge);
  CHECK_THROWS(lru.unpin(K(12345)));
  CHECK_THROWS(lru.admit(K(11), 1024, off, ev));  // already resident

  // Pin counts add up.
  lru.pin(K(11));
  CHECK(lru.pins(K(11)) == 2);
  lru.unpin(K(11));
  lru.unpin(K(11));
  CHECK(lru.pins(K(11)) == 0);
  CHECK_THROWS(lru.unpin(K(11)));

  // Variable sizes: a big block evicts as many small ones as needed (fragmentation included).
  ArenaLru v(4096, 1024);
  for (std::uint32_t i = 0; i < 4; ++i) {
    CHECK(v.admit(K(i), 1024, off, ev) == AdmitResult::kOk);
    v.unpin(K(i));
  }
  ev.clear();
  CHECK(v.admit(K(50), 3000, off, ev) == AdmitResult::kOk);  // needs 3 KiB contiguous
  CHECK(ev.size() >= 3);

  // evict_all_unpinned keeps pinned blocks.
  ev.clear();
  v.unpin(K(50));
  CHECK(v.admit(K(60), 1024, off, ev) == AdmitResult::kOk);  // pinned
  ev.clear();
  CHECK(v.evict_all_unpinned(ev) >= 1);
  CHECK(v.contains(K(60)));

  // abandon removes a freshly admitted block and frees its space.
  ev.clear();
  CHECK(v.admit(K(70), 1024, off, ev) == AdmitResult::kOk);
  v.abandon(K(70));
  CHECK(!v.contains(K(70)));
  CHECK_THROWS(v.abandon(K(70)));

  CacheMetrics m;
  CHECK(m.hit_rate() == 0.0);
  m.hits = 1;
  m.misses = 3;
  CHECK(m.hit_rate() == 0.25);
  std::puts("cache_lru: PASS");
  return 0;
}
