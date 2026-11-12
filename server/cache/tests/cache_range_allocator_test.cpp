// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include <vector>

#include "check.hpp"
#include "ombra/cache/range_allocator.hpp"

using ombra::cache::RangeAllocator;

int main() {
  RangeAllocator a(1024, 256);
  CHECK(a.capacity() == 1024);
  auto x = a.allocate(300);  // rounds to 512
  auto y = a.allocate(256);
  auto z = a.allocate(256);
  CHECK(x && y && z);
  CHECK(*x == 0 && *y == 512 && *z == 768);
  CHECK(a.used_bytes() == 1024);
  CHECK(!a.allocate(1));
  CHECK(!a.allocate(0));

  a.free(*x);
  a.free(*z);
  CHECK(a.largest_free() == 512);
  CHECK(!a.allocate(768));  // fragmented: 512 + 256 not contiguous
  a.free(*y);               // everything merges back
  CHECK(a.largest_free() == 1024 && a.used_bytes() == 0);
  CHECK(a.allocate(1024).has_value());

  CHECK_THROWS(a.free(12345));
  CHECK_THROWS(RangeAllocator(1024, 100));  // not a power of two
  RangeAllocator tiny(100, 256);            // capacity rounds down to 0
  CHECK(!tiny.allocate(1));

  // Churn: allocate and free in a scrambled order, the arena always returns to one range.
  RangeAllocator b(1 << 20, 256);
  std::vector<std::size_t> offs;
  for (int i = 0; i < 64; ++i) offs.push_back(*b.allocate(16384));
  CHECK(!b.allocate(1));
  for (int i = 0; i < 64; i += 2) b.free(offs[i]);
  for (int i = 1; i < 64; i += 2) b.free(offs[i]);
  CHECK(b.largest_free() == (1u << 20));
  std::puts("cache_range_allocator: PASS");
  return 0;
}
