// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Generator 1 of seeded ciphertexts (file-formats.md): BLAKE2xb as shipped in OpenFHE
// (utils/prng/blake2.h, exported by libOPENFHEcore). Needs only that one C header, not OpenFHE's
// C++ headers. For tower t with modulus q of bit length n and block c = 0, 1, ...:
//   BLAKE2xb(outlen 4096, in = "ombra-seeded-v1" || seed(32) || u32 LE t || u64 LE c, no key)
// read as little-endian u64 words, each masked to n bits, kept if < q; the first N kept words.
// A seed must NEVER be used for two ciphertexts under one key.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ombra/core/seeded_ciphertext.hpp"
#include "utils/prng/blake2.h"

namespace ombra::core {

inline constexpr std::size_t kSeededBlockBytes = 4096;

inline std::vector<std::uint64_t> expand_seed_tower(
    const std::array<std::uint8_t, kSeedBytes>& seed, std::uint32_t tower, std::uint64_t q,
    std::size_t n_coeffs) {
  if (q < 2 || (q & 1) == 0 || q >= (1ull << 60))
    throw std::invalid_argument("expand_seed_tower: bad modulus");
  int bits = 0;
  for (std::uint64_t v = q; v; v >>= 1) ++bits;
  const std::uint64_t mask = (1ull << bits) - 1;
  static constexpr char kDomain[] = "ombra-seeded-v1";
  constexpr std::size_t kDomainLen = sizeof(kDomain) - 1;
  constexpr std::size_t kTowerAt = kDomainLen + kSeedBytes, kCounterAt = kTowerAt + 4;
  std::vector<std::uint64_t> out;
  out.reserve(n_coeffs);
  std::vector<std::uint8_t> in(kCounterAt + 8);
  std::memcpy(in.data(), kDomain, kDomainLen);
  std::memcpy(in.data() + kDomainLen, seed.data(), kSeedBytes);
  for (int i = 0; i < 4; ++i) in[kTowerAt + i] = static_cast<std::uint8_t>(tower >> (8 * i));
  std::vector<std::uint8_t> buf(kSeededBlockBytes);
  for (std::uint64_t c = 0; out.size() < n_coeffs; ++c) {
    for (int i = 0; i < 8; ++i) in[kCounterAt + i] = static_cast<std::uint8_t>(c >> (8 * i));
    if (blake2xb(buf.data(), buf.size(), in.data(), in.size(), nullptr, 0) != 0)
      throw std::runtime_error("expand_seed_tower: blake2xb failed");
    for (std::size_t o = 0; o < kSeededBlockBytes && out.size() < n_coeffs; o += 8) {
      std::uint64_t w = seeded_detail::get(buf.data() + o, 8) & mask;
      if (w < q) out.push_back(w);
    }
  }
  return out;
}

// [2][T][N] words: c0' then u (what the server copies to the device). No OpenFHE objects.
inline std::vector<std::uint64_t> seeded_words(const SeededPayload& p) {
  std::vector<std::uint64_t> w;
  w.reserve(2 * p.c0.size());
  w.insert(w.end(), p.c0.begin(), p.c0.end());
  for (std::uint32_t t = 0; t < p.towers; ++t) {
    auto u = expand_seed_tower(p.seed, t, p.moduli[t], p.ring_dim);
    w.insert(w.end(), u.begin(), u.end());
  }
  return w;
}

}  // namespace ombra::core
