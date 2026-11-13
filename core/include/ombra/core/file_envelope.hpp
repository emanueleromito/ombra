// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Ombra file envelope v1/v2. Header only, no OpenFHE. Little-endian layout:
//   0  4  magic "OMBR"
//   4  2  format version (u16: 1, or 2 which adds kind 4)
//   6  2  kind (u16: 1 context, 2 ciphertext, 3 public eval key, 4 seeded ciphertext)
// Writers emit version 1 for kinds 1-3 (files unchanged) and version 2 for kind 4. Readers accept
// version 1 (kinds 1-3 only) and version 2 (kinds 1-4). Kind 4 payload: seeded_ciphertext.hpp.
//   8  2  preset name length P (u16, 1..255)
//   .  P  preset name, ASCII/UTF-8
//   .  2  OpenFHE version length V (u16, 1..255)
//   .  V  OpenFHE version string
//   .  8  payload length L (u64)
//   .  L  payload (OpenFHE BINARY serialization); the file ends here
// There is deliberately no kind for a secret key (the secret key never leaves the client).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ombra::core {

inline constexpr std::uint16_t kEnvelopeVersion = 1;        // kinds 1-3
inline constexpr std::uint16_t kEnvelopeVersionSeeded = 2;  // adds kind 4
inline constexpr std::size_t kEnvelopeMaxName = 255;

enum class EnvelopeKind : std::uint16_t {
  Context = 1,
  Ciphertext = 2,
  PublicEvalKey = 3,
  SeededCiphertext = 4
};

struct Envelope {
  EnvelopeKind kind;
  std::string preset;
  std::string openfhe_version;
  std::vector<std::uint8_t> payload;
};

namespace detail {
inline void put(std::vector<std::uint8_t>& o, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
inline std::uint64_t get(const std::uint8_t* p, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  return v;
}
}  // namespace detail

inline std::vector<std::uint8_t> write_envelope(const Envelope& e) {
  if (e.preset.empty() || e.preset.size() > kEnvelopeMaxName || e.openfhe_version.empty() ||
      e.openfhe_version.size() > kEnvelopeMaxName)
    throw std::invalid_argument("envelope: preset and OpenFHE version must be 1..255 bytes");
  std::vector<std::uint8_t> o{'O', 'M', 'B', 'R'};
  detail::put(
      o, e.kind == EnvelopeKind::SeededCiphertext ? kEnvelopeVersionSeeded : kEnvelopeVersion, 2);
  detail::put(o, static_cast<std::uint16_t>(e.kind), 2);
  detail::put(o, e.preset.size(), 2);
  o.insert(o.end(), e.preset.begin(), e.preset.end());
  detail::put(o, e.openfhe_version.size(), 2);
  o.insert(o.end(), e.openfhe_version.begin(), e.openfhe_version.end());
  detail::put(o, e.payload.size(), 8);
  o.insert(o.end(), e.payload.begin(), e.payload.end());
  return o;
}

// Refuses (std::runtime_error) on bad magic, unknown version, kind != expected_kind, preset or
// OpenFHE version != expected, truncation, or trailing bytes. The payload length is checked
// against the bytes actually present before anything is allocated.
inline Envelope read_envelope_kinds(const std::uint8_t* data, std::size_t size, bool accept_plain,
                                    bool accept_seeded, const std::string& expected_preset,
                                    const std::string& expected_openfhe_version,
                                    EnvelopeKind plain_kind) {
  std::size_t pos = 0;
  auto need = [&](std::size_t n) {
    if (size - pos < n) throw std::runtime_error("envelope: truncated file");
  };
  need(8);
  if (data[0] != 'O' || data[1] != 'M' || data[2] != 'B' || data[3] != 'R')
    throw std::runtime_error("envelope: bad magic, not an Ombra file");
  auto version = static_cast<std::uint16_t>(detail::get(data + 4, 2));
  if (version != kEnvelopeVersion && version != kEnvelopeVersionSeeded)
    throw std::runtime_error("envelope: unknown format version " + std::to_string(version));
  auto kind = static_cast<std::uint16_t>(detail::get(data + 6, 2));
  const bool is_seeded = kind == static_cast<std::uint16_t>(EnvelopeKind::SeededCiphertext);
  if (is_seeded ? !accept_seeded
                : (!accept_plain || kind != static_cast<std::uint16_t>(plain_kind)))
    throw std::runtime_error("envelope: wrong kind " + std::to_string(kind));
  if (kind < 1 || kind > 4 || (is_seeded && version != kEnvelopeVersionSeeded))
    throw std::runtime_error("envelope: kind " + std::to_string(kind) +
                             " is not valid in format version " + std::to_string(version));
  const EnvelopeKind expected_kind = static_cast<EnvelopeKind>(kind);
  pos = 8;
  auto read_name = [&](const char* what) {
    need(2);
    std::size_t n = detail::get(data + pos, 2);
    pos += 2;
    if (n == 0 || n > kEnvelopeMaxName)
      throw std::runtime_error(std::string("envelope: bad ") + what + " length");
    need(n);
    std::string s(reinterpret_cast<const char*>(data + pos), n);
    pos += n;
    return s;
  };
  Envelope e{expected_kind, read_name("preset"), read_name("OpenFHE version"), {}};
  if (e.preset != expected_preset)
    throw std::runtime_error("envelope: preset mismatch (file has '" + e.preset + "')");
  if (e.openfhe_version != expected_openfhe_version)
    throw std::runtime_error("envelope: OpenFHE version mismatch (file has '" + e.openfhe_version +
                             "')");
  need(8);
  std::uint64_t len = detail::get(data + pos, 8);
  pos += 8;
  if (len > size - pos) throw std::runtime_error("envelope: payload length exceeds file");
  if (len < size - pos) throw std::runtime_error("envelope: trailing bytes after payload");
  e.payload.assign(data + pos, data + pos + len);
  return e;
}

// Exactly `expected_kind`. Kind 4 is accepted only when it is the expected kind, so queries, score
// blocks and keys cannot silently be seeded.
inline Envelope read_envelope(const std::uint8_t* data, std::size_t size,
                              EnvelopeKind expected_kind, const std::string& expected_preset,
                              const std::string& expected_openfhe_version) {
  return read_envelope_kinds(data, size, expected_kind != EnvelopeKind::SeededCiphertext,
                             expected_kind == EnvelopeKind::SeededCiphertext, expected_preset,
                             expected_openfhe_version, expected_kind);
}

inline Envelope read_envelope(const std::vector<std::uint8_t>& b, EnvelopeKind k,
                              const std::string& preset, const std::string& ofhe) {
  return read_envelope(b.data(), b.size(), k, preset, ofhe);
}

// A stored column: kind Ciphertext (2) or SeededCiphertext (4); check `kind` of the result.
inline Envelope read_column_envelope(const std::uint8_t* data, std::size_t size,
                                     const std::string& preset, const std::string& ofhe) {
  return read_envelope_kinds(data, size, true, true, preset, ofhe, EnvelopeKind::Ciphertext);
}
inline Envelope read_column_envelope(const std::vector<std::uint8_t>& b, const std::string& preset,
                                     const std::string& ofhe) {
  return read_column_envelope(b.data(), b.size(), preset, ofhe);
}

}  // namespace ombra::core
