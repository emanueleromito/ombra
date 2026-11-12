// Ombra file envelope v1 (decision 0016). Header only, no OpenFHE. Little-endian layout:
//   0  4  magic "OMBR"
//   4  2  format version (u16, 1)
//   6  2  kind (u16: 1 context, 2 ciphertext, 3 public eval key)
//   8  2  preset name length P (u16, 1..255)
//   .  P  preset name, ASCII/UTF-8
//   .  2  OpenFHE version length V (u16, 1..255)
//   .  V  OpenFHE version string
//   .  8  payload length L (u64)
//   .  L  payload (OpenFHE BINARY serialization); the file ends here
// There is deliberately no kind for a secret key (the secret key stays on the client).
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ombra::core {

inline constexpr std::uint16_t kEnvelopeVersion = 1;
inline constexpr std::size_t kEnvelopeMaxName = 255;

enum class EnvelopeKind : std::uint16_t { Context = 1, Ciphertext = 2, PublicEvalKey = 3 };

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
  detail::put(o, kEnvelopeVersion, 2);
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
inline Envelope read_envelope(const std::uint8_t* data, std::size_t size,
                              EnvelopeKind expected_kind, const std::string& expected_preset,
                              const std::string& expected_openfhe_version) {
  std::size_t pos = 0;
  auto need = [&](std::size_t n) {
    if (size - pos < n) throw std::runtime_error("envelope: truncated file");
  };
  need(8);
  if (data[0] != 'O' || data[1] != 'M' || data[2] != 'B' || data[3] != 'R')
    throw std::runtime_error("envelope: bad magic, not an Ombra file");
  auto version = static_cast<std::uint16_t>(detail::get(data + 4, 2));
  if (version != kEnvelopeVersion)
    throw std::runtime_error("envelope: unknown format version " + std::to_string(version));
  auto kind = static_cast<std::uint16_t>(detail::get(data + 6, 2));
  if (kind != static_cast<std::uint16_t>(expected_kind))
    throw std::runtime_error("envelope: wrong kind " + std::to_string(kind));
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
    throw std::runtime_error("envelope: OpenFHE version mismatch (file has '" +
                             e.openfhe_version + "')");
  need(8);
  std::uint64_t len = detail::get(data + pos, 8);
  pos += 8;
  if (len > size - pos) throw std::runtime_error("envelope: payload length exceeds file");
  if (len < size - pos) throw std::runtime_error("envelope: trailing bytes after payload");
  e.payload.assign(data + pos, data + pos + len);
  return e;
}

inline Envelope read_envelope(const std::vector<std::uint8_t>& b, EnvelopeKind k,
                              const std::string& preset, const std::string& ofhe) {
  return read_envelope(b.data(), b.size(), k, preset, ofhe);
}

}  // namespace ombra::core
