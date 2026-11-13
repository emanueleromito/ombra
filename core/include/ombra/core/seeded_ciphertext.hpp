// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Seeded stored ciphertext, payload of envelope kind 4 (docs/decisions/0023). Header only, NO
// OpenFHE: the server's storage layer includes it. Little-endian layout:
//   2 payload version (u16, 1) | 2 generator (u16, 1 = BLAKE2xb, see seeded_ciphertext_prg.hpp)
//   32 seed | 4 ring dimension N | 2 towers T | 4 level | 4 noise scale degree | 4 slots
//   8 scaling factor (IEEE double) | 2 key tag length K | K key tag | 8*T moduli
//   8*T*N words of c0', [tower][coefficient], evaluation form, reduced
// The second ciphertext component is not stored: it is expand_seed_tower(seed, t, q_t, N).
#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace ombra::core {

inline constexpr std::uint16_t kSeededPayloadVersion = 1;
inline constexpr std::uint16_t kSeededGeneratorBlake2xb = 1;
inline constexpr std::size_t kSeedBytes = 32;

struct SeededPayload {
  std::uint16_t payload_version = kSeededPayloadVersion;
  std::uint16_t generator = kSeededGeneratorBlake2xb;
  std::array<std::uint8_t, kSeedBytes> seed{};
  std::uint32_t ring_dim = 0;
  std::uint16_t towers = 0;
  std::uint32_t level = 0;
  std::uint32_t noise_scale_deg = 0;
  std::uint32_t slots = 0;
  double scaling_factor = 0;
  std::string key_tag;
  std::vector<std::uint64_t> moduli;  // T
  std::vector<std::uint64_t> c0;      // [tower][coeff], T * N words
};

namespace seeded_detail {
inline void put(std::vector<std::uint8_t>& o, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
inline std::uint64_t get(const std::uint8_t* p, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  return v;
}
inline bool is_pow2(std::uint64_t n) { return n != 0 && (n & (n - 1)) == 0; }
}  // namespace seeded_detail

inline std::vector<std::uint8_t> write_seeded_payload(const SeededPayload& p) {
  using namespace seeded_detail;
  if (p.key_tag.size() > 65535) throw std::invalid_argument("seeded: key tag too long");
  if (!is_pow2(p.ring_dim) || p.towers == 0 || p.moduli.size() != p.towers ||
      p.c0.size() != static_cast<std::size_t>(p.towers) * p.ring_dim)
    throw std::invalid_argument("seeded: inconsistent shape");
  std::vector<std::uint8_t> o;
  o.reserve(64 + p.key_tag.size() + 8 * (p.moduli.size() + p.c0.size()));
  put(o, p.payload_version, 2);
  put(o, p.generator, 2);
  o.insert(o.end(), p.seed.begin(), p.seed.end());
  put(o, p.ring_dim, 4);
  put(o, p.towers, 2);
  put(o, p.level, 4);
  put(o, p.noise_scale_deg, 4);
  put(o, p.slots, 4);
  std::uint64_t sf;
  std::memcpy(&sf, &p.scaling_factor, 8);
  put(o, sf, 8);
  put(o, p.key_tag.size(), 2);
  o.insert(o.end(), p.key_tag.begin(), p.key_tag.end());
  for (auto q : p.moduli) put(o, q, 8);
  for (auto w : p.c0) put(o, w, 8);
  return o;
}

// Parses and validates; sizes are checked against the bytes present before anything is allocated.
// Throws std::runtime_error. `with_words` false skips copying the words; `check_words` false skips
// the range check (the cheap check for the storage layer at insert).
inline SeededPayload parse_seeded_payload(const std::uint8_t* d, std::size_t size, bool with_words,
                                          bool check_words) {
  using namespace seeded_detail;
  std::size_t pos = 0;
  auto need = [&](std::size_t n) {
    if (size - pos < n) throw std::runtime_error("seeded: truncated payload");
  };
  constexpr std::size_t kFixed = 2 + 2 + kSeedBytes + 4 + 2 + 4 + 4 + 4 + 8 + 2;
  need(kFixed);
  SeededPayload p;
  p.payload_version = static_cast<std::uint16_t>(get(d + pos, 2));
  pos += 2;
  if (p.payload_version != kSeededPayloadVersion)
    throw std::runtime_error("seeded: unknown payload version " +
                             std::to_string(p.payload_version));
  p.generator = static_cast<std::uint16_t>(get(d + pos, 2));
  pos += 2;
  if (p.generator != kSeededGeneratorBlake2xb)
    throw std::runtime_error("seeded: unknown generator " + std::to_string(p.generator));
  std::memcpy(p.seed.data(), d + pos, kSeedBytes);
  pos += kSeedBytes;
  p.ring_dim = static_cast<std::uint32_t>(get(d + pos, 4));
  pos += 4;
  p.towers = static_cast<std::uint16_t>(get(d + pos, 2));
  pos += 2;
  if (!is_pow2(p.ring_dim)) throw std::runtime_error("seeded: ring dimension not a power of two");
  if (p.towers == 0) throw std::runtime_error("seeded: zero towers");
  p.level = static_cast<std::uint32_t>(get(d + pos, 4));
  pos += 4;
  p.noise_scale_deg = static_cast<std::uint32_t>(get(d + pos, 4));
  pos += 4;
  p.slots = static_cast<std::uint32_t>(get(d + pos, 4));
  pos += 4;
  std::uint64_t sf = get(d + pos, 8);
  pos += 8;
  std::memcpy(&p.scaling_factor, &sf, 8);
  std::size_t k = get(d + pos, 2);
  pos += 2;
  need(k);
  p.key_tag.assign(reinterpret_cast<const char*>(d + pos), k);
  pos += k;
  // Exact size, in 64-bit arithmetic that cannot overflow (N < 2^32, T < 2^16).
  const std::uint64_t words = static_cast<std::uint64_t>(p.towers) * p.ring_dim;
  const std::uint64_t want = 8ull * p.towers + 8ull * words;
  if (size - pos < want) throw std::runtime_error("seeded: payload size does not match N and T");
  if (size - pos > want) throw std::runtime_error("seeded: trailing bytes after payload");
  p.moduli.resize(p.towers);
  for (std::size_t t = 0; t < p.towers; ++t) {
    std::uint64_t q = get(d + pos, 8);
    pos += 8;
    if (q < 2 || (q & 1) == 0 || q >= (1ull << 60))
      throw std::runtime_error("seeded: modulus must be odd and in [2, 2^60)");
    p.moduli[t] = q;
  }
  if (with_words) {
    p.c0.resize(words);
    for (std::size_t t = 0; t < p.towers; ++t)
      for (std::size_t i = 0; i < p.ring_dim; ++i) {
        std::uint64_t w = get(d + pos, 8);
        pos += 8;
        if (check_words && w >= p.moduli[t])
          throw std::runtime_error("seeded: word not reduced modulo its modulus");
        p.c0[t * p.ring_dim + i] = w;
      }
  }
  return p;
}

inline SeededPayload read_seeded_payload(const std::uint8_t* d, std::size_t size) {
  return parse_seeded_payload(d, size, true, true);
}
inline SeededPayload read_seeded_payload(const std::vector<std::uint8_t>& b) {
  return read_seeded_payload(b.data(), b.size());
}

// Everything except the word range (words are not copied).
inline SeededPayload check_seeded_header(const std::uint8_t* d, std::size_t size) {
  return parse_seeded_payload(d, size, false, false);
}
inline SeededPayload check_seeded_header(const std::vector<std::uint8_t>& b) {
  return check_seeded_header(b.data(), b.size());
}

}  // namespace ombra::core
