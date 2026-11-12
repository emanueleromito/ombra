// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Segment file: one cluster block, immutable once written. Little-endian container, version 1:
//   0   4  magic "OSEG"
//   4   2  container version (u16, 1)
//   6   4  CRC32 of the body (everything after this header)
//   10  .  body: u32 cluster, u32 n_ids, u32 n_columns,
//          n_ids x (u32 length, id bytes), n_columns x (u64 length, versioned envelope)
// Each column is a complete versioned envelope of kind ciphertext; its payload is opaque here.
#pragma once

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

std::vector<std::uint8_t> encode_segment(const SegmentData& s);

// Durable write (temp, fsync, rename, fsync dir). Crash points fire inside.
SegmentInfo write_segment(const std::string& path, const SegmentData& s);

// Refuses (std::runtime_error) on bad magic, unknown container version, bad CRC, truncation, and
// any column whose envelope has an unknown version, wrong kind, or a preset/OpenFHE version
// other than expected.
SegmentData read_segment(const std::string& path, const std::string& expected_preset,
                         const std::string& expected_openfhe_version, SegmentInfo* info = nullptr);

}  // namespace ombra::storage
