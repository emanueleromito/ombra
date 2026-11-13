// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Segment file: one cluster block, immutable once written. Little-endian container, version 1:
//   0   4  magic "OSEG"
//   4   2  container version (u16, 1)
//   6   4  CRC32 of the body (everything after this header)
//   10  .  body: u32 cluster, u32 n_ids, u32 n_columns,
//          n_ids x (u32 length, id bytes), n_columns x (u64 length, versioned envelope)
// Each column is a complete versioned envelope of kind ciphertext (2) or seeded ciphertext (4); a
// kind-2 payload is opaque here, a kind-4 payload gets the cheap structural check (no OpenFHE).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ombra::storage {

inline constexpr std::uint16_t kSegmentVersion = 1;

struct SegmentData {
  std::uint32_t cluster = 0;
  std::vector<std::string> ids;                             // opaque byte strings
  std::vector<std::vector<std::uint8_t>> column_envelopes;  // as given to the writer
};

struct SegmentInfo {
  std::uint32_t n_vectors = 0;
  std::uint32_t n_columns = 0;
  std::uint32_t crc = 0;
};

// Validates one stored column (kind 2 or 4) and refuses with std::runtime_error: wrong kind, preset
// or OpenFHE version, unknown envelope version, and for kind 4 a malformed seeded payload header.
void check_column_envelope(const std::uint8_t* data, std::size_t size,
                           const std::string& expected_preset,
                           const std::string& expected_openfhe_version);

// What a seeded column must match at insert (audit A-71): the collection's key tag and, when the
// server knows the preset, its ring dimension and slot count. Level 0, noise degree 1, a finite
// scale and reduced words are always required. No OpenFHE needed.
struct ColumnExpect {
  std::string key_tag;
  std::uint32_t ring_dim = 0;  // 0: not checked
  std::uint32_t slots = 0;     // 0: not checked
};

// check_column_envelope plus, for a seeded column, the full header comparison with `expect`.
void check_column_for_insert(const std::uint8_t* data, std::size_t size,
                             const std::string& expected_preset,
                             const std::string& expected_openfhe_version,
                             const ColumnExpect& expect);

std::vector<std::uint8_t> encode_segment(const SegmentData& s);

// Durable write (temp, fsync, rename, fsync dir). Crash points fire inside.
SegmentInfo write_segment(const std::string& path, const SegmentData& s);

// Refuses (std::runtime_error) on bad magic, unknown container version, bad CRC, truncation, and
// any column whose envelope has an unknown version, wrong kind, or a preset/OpenFHE version
// other than expected.
SegmentData read_segment(const std::string& path, const std::string& expected_preset,
                         const std::string& expected_openfhe_version, SegmentInfo* info = nullptr);

}  // namespace ombra::storage
