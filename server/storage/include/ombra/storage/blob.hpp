// Blob file: one encrypted blob of the blob store, immutable once written. Little-endian
// container, version 1:
//   0   4  magic "OBLB"
//   4   2  container version (u16, 1)
//   6   4  CRC32 of the body
//   10  .  body: u32 kind (proto BlobKind), u64 payload length, payload (opaque client bytes)
// The server never parses the payload. A blob's size is one of the bucket sizes (kBlobBuckets).
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ombra::storage {

inline constexpr std::uint16_t kBlobVersion = 1;
// Proto LIMIT_BLOB_BUCKET_0..3.
inline constexpr std::array<std::uint64_t, 4> kBlobBuckets{16384, 262144, 2097152, 16777216};
// Proto BlobKind values RECORD and DOCUMENT.
inline constexpr std::uint32_t kBlobKindRecord = 1;
inline constexpr std::uint32_t kBlobKindDocument = 2;

inline bool is_blob_bucket(std::uint64_t n) {
  for (auto b : kBlobBuckets)
    if (b == n) return true;
  return false;
}
inline bool is_blob_kind(std::uint32_t k) { return k == kBlobKindRecord || k == kBlobKindDocument; }

std::vector<std::uint8_t> encode_blob(std::uint32_t kind, const std::vector<std::uint8_t>& payload);
// Durable write (temp, fsync, rename, fsync dir). Crash points fire inside.
void write_blob_file(const std::string& path, std::uint32_t kind,
                     const std::vector<std::uint8_t>& payload);
// Refuses (std::runtime_error) on bad magic, unknown version, bad CRC, truncation, a kind other
// than expected, or a payload that is not a bucket size.
std::vector<std::uint8_t> read_blob_file(const std::string& path, std::uint32_t expected_kind);

}  // namespace ombra::storage
