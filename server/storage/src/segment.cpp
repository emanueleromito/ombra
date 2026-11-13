// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/storage/segment.hpp"

#include <cmath>
#include <stdexcept>

#include "ombra/core/file_envelope.hpp"
#include "ombra/core/seeded_ciphertext.hpp"
#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/fsutil.hpp"

namespace ombra::storage {

void check_column_envelope(const std::uint8_t* data, std::size_t size,
                           const std::string& expected_preset,
                           const std::string& expected_openfhe_version) {
  const auto env =
      ombra::core::read_column_envelope(data, size, expected_preset, expected_openfhe_version);
  // Kind 4: only the cheap structural check; the word range is checked when the block is loaded.
  if (env.kind == ombra::core::EnvelopeKind::SeededCiphertext)
    ombra::core::check_seeded_header(env.payload);
}

void check_column_for_insert(const std::uint8_t* data, std::size_t size,
                             const std::string& expected_preset,
                             const std::string& expected_openfhe_version,
                             const ColumnExpect& expect) {
  const auto env =
      ombra::core::read_column_envelope(data, size, expected_preset, expected_openfhe_version);
  if (env.kind != ombra::core::EnvelopeKind::SeededCiphertext) return;  // kind 2: opaque payload
  const auto p = ombra::core::read_seeded_payload(env.payload);         // also checks the words
  if (p.key_tag != expect.key_tag)
    throw std::runtime_error("seeded column was made under another key than the collection's");
  if ((expect.ring_dim != 0 && p.ring_dim != expect.ring_dim) ||
      (expect.slots != 0 && p.slots != expect.slots) || p.level != 0 || p.noise_scale_deg != 1 ||
      !std::isfinite(p.scaling_factor) || !(p.scaling_factor > 1.0))
    throw std::runtime_error("seeded column has an unexpected ring, slots, level or scale");
}

namespace {
constexpr std::size_t kHeader = 10;
void before_rename_hook() { maybe_crash(CrashPoint::InsertBeforeSegmentRename); }
}  // namespace

std::vector<std::uint8_t> encode_segment(const SegmentData& s) {
  std::vector<std::uint8_t> body;
  put_le(body, s.cluster, 4);
  put_le(body, s.ids.size(), 4);
  put_le(body, s.column_envelopes.size(), 4);
  for (const auto& id : s.ids) {
    put_le(body, id.size(), 4);
    body.insert(body.end(), id.begin(), id.end());
  }
  for (const auto& c : s.column_envelopes) {
    put_le(body, c.size(), 8);
    body.insert(body.end(), c.begin(), c.end());
  }
  std::vector<std::uint8_t> out{'O', 'S', 'E', 'G'};
  put_le(out, kSegmentVersion, 2);
  put_le(out, crc32(body.data(), body.size()), 4);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

SegmentInfo write_segment(const std::string& path, const SegmentData& s) {
  auto bytes = encode_segment(s);
  SegmentInfo info;
  info.n_vectors = static_cast<std::uint32_t>(s.ids.size());
  info.n_columns = static_cast<std::uint32_t>(s.column_envelopes.size());
  info.crc = static_cast<std::uint32_t>(get_le(bytes.data() + 6, 4));
  write_file_durable(path, bytes, before_rename_hook);
  maybe_crash(CrashPoint::InsertAfterSegmentRename);
  return info;
}

SegmentData read_segment(const std::string& path, const std::string& expected_preset,
                         const std::string& expected_openfhe_version, SegmentInfo* info) {
  auto b = read_file(path);
  if (b.size() < kHeader) throw std::runtime_error("segment: truncated header");
  if (b[0] != 'O' || b[1] != 'S' || b[2] != 'E' || b[3] != 'G')
    throw std::runtime_error("segment: bad magic");
  auto version = get_le(b.data() + 4, 2);
  if (version != kSegmentVersion)
    throw std::runtime_error("segment: unknown container version " + std::to_string(version));
  auto crc = static_cast<std::uint32_t>(get_le(b.data() + 6, 4));
  const std::uint8_t* p = b.data() + kHeader;
  const std::size_t n = b.size() - kHeader;
  if (crc32(p, n) != crc) throw std::runtime_error("segment: CRC mismatch");

  std::size_t pos = 0;
  auto need = [&](std::size_t k) {
    if (n - pos < k) throw std::runtime_error("segment: truncated body");
  };
  SegmentData s;
  need(12);
  s.cluster = static_cast<std::uint32_t>(get_le(p, 4));
  auto n_ids = get_le(p + 4, 4);
  auto n_cols = get_le(p + 8, 4);
  pos = 12;
  // Every id costs at least 4 bytes and every column at least 8: bound counts before reserving.
  if (n_ids > n / 4 || n_cols > n / 8) throw std::runtime_error("segment: bad counts");
  s.ids.reserve(n_ids);
  for (std::uint64_t i = 0; i < n_ids; ++i) {
    need(4);
    auto len = get_le(p + pos, 4);
    pos += 4;
    need(len);
    s.ids.emplace_back(reinterpret_cast<const char*>(p + pos), len);
    pos += len;
  }
  s.column_envelopes.reserve(n_cols);
  for (std::uint64_t i = 0; i < n_cols; ++i) {
    need(8);
    auto len = get_le(p + pos, 8);
    pos += 8;
    need(len);
    // Throws on unknown envelope version, wrong kind, wrong preset or OpenFHE version, or a
    // malformed seeded payload.
    check_column_envelope(p + pos, len, expected_preset, expected_openfhe_version);
    s.column_envelopes.emplace_back(p + pos, p + pos + len);
    pos += len;
  }
  if (pos != n) throw std::runtime_error("segment: trailing bytes");
  if (info) {
    info->n_vectors = static_cast<std::uint32_t>(n_ids);
    info->n_columns = static_cast<std::uint32_t>(n_cols);
    info->crc = crc;
  }
  return s;
}

}  // namespace ombra::storage
