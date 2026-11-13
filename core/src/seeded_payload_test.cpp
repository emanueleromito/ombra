// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Unit test of the seeded payload format (kind 4). No OpenFHE, no GPU. Synthetic bytes only.
#include <iostream>

#include "ombra/core/seeded_ciphertext.hpp"

using namespace ombra::core;
using Bytes = std::vector<std::uint8_t>;

static int fails = 0;
#define CHECK(c)                                             \
  do {                                                       \
    if (!(c)) {                                              \
      std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; \
      ++fails;                                               \
    }                                                        \
  } while (0)

template <class F>
static bool refuses(F f) {
  try {
    f();
  } catch (const std::runtime_error&) {
    return true;
  }
  return false;
}

static void put_at(Bytes& b, std::size_t off, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) b[off + i] = static_cast<std::uint8_t>(v >> (8 * i));
}

int main() {
  SeededPayload p;
  for (std::size_t i = 0; i < 32; ++i) p.seed[i] = static_cast<std::uint8_t>(i);
  p.ring_dim = 8;
  p.towers = 2;
  p.level = 0;
  p.noise_scale_deg = 1;
  p.slots = 4;
  p.scaling_factor = 1099511627776.0;
  p.key_tag = "tag";
  p.moduli = {97, 193};
  for (std::uint64_t i = 0; i < 16; ++i) p.c0.push_back(i);
  const Bytes good = write_seeded_payload(p);
  auto back = read_seeded_payload(good);
  CHECK(back.c0 == p.c0 && back.moduli == p.moduli && back.key_tag == "tag" && back.seed == p.seed);
  CHECK(check_seeded_header(good).ring_dim == 8);

  const std::size_t tag_at = 2 + 2 + 32 + 4 + 2 + 4 + 4 + 4 + 8;
  const std::size_t mod_at = tag_at + 2 + 3, word_at = mod_at + 16;
  auto bad = [&](std::size_t off, std::uint64_t v, int n) {
    Bytes b = good;
    put_at(b, off, v, n);
    return b;
  };
  CHECK(refuses([&] { read_seeded_payload(bad(0, 2, 2)); }));            // payload version
  CHECK(refuses([&] { read_seeded_payload(bad(2, 2, 2)); }));            // generator
  CHECK(refuses([&] { read_seeded_payload(bad(36, 0, 4)); }));           // N zero
  CHECK(refuses([&] { read_seeded_payload(bad(36, 12, 4)); }));          // N not a power of two
  CHECK(refuses([&] { read_seeded_payload(bad(40, 0, 2)); }));           // T zero
  CHECK(refuses([&] { read_seeded_payload(bad(40, 3, 2)); }));           // size mismatch (T)
  CHECK(refuses([&] { read_seeded_payload(bad(36, 16, 4)); }));          // size mismatch (N)
  CHECK(refuses([&] { read_seeded_payload(bad(36, 1ull << 31, 4)); }));  // huge N, no allocation
  CHECK(refuses([&] { read_seeded_payload(bad(tag_at, 65535, 2)); }));   // tag longer than bytes
  Bytes extra = good;
  extra.push_back(0);
  CHECK(refuses([&] { read_seeded_payload(extra); }));
  CHECK(refuses([&] { check_seeded_header(extra); }));
  for (std::uint64_t q : {0ull, 1ull, 98ull, 1ull << 60, (1ull << 60) + 1, ~0ull})
    CHECK(refuses([&] { read_seeded_payload(bad(mod_at, q, 8)); }));
  CHECK(refuses([&] { read_seeded_payload(bad(word_at, 97, 8)); }));  // unreduced word
  CHECK(!check_seeded_header(bad(word_at, 97, 8)).moduli.empty());    // header check skips words
  for (std::size_t n = 0; n < good.size(); ++n)
    CHECK(refuses([&] { read_seeded_payload(Bytes(good.begin(), good.begin() + n)); }));
  SeededPayload w = p;
  w.key_tag.assign(65536, 'x');
  bool threw = false;
  try {
    write_seeded_payload(w);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::cout << (fails ? "FAILED" : "seeded_payload ok") << "\n";
  return fails ? 1 : 0;
}
