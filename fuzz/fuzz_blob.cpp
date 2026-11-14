// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: blob file reading (server/storage/src/blob.cpp). Input: one mode byte, then the
// file bytes. Mode bit 0 set: the harness rewrites magic, version and CRC (see fuzz_segment.cpp).
// A blob is only accepted at a bucket size (16 KiB and up), which a short input cannot reach, so
// bit 1 additionally makes the harness build a valid container around the input as the payload
// of the smallest bucket (zero padded), to exercise the accept path.
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "fuzz_util.hpp"
#include "ombra/storage/blob.hpp"
#include "ombra/storage/fsutil.hpp"

namespace st = ombra::storage;

static ombra::fuzz::TempDir* g_dir = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  if (n < 1) return 0;
  std::vector<std::uint8_t> f;
  if ((d[0] & 2) != 0) {
    std::vector<std::uint8_t> payload(d + 1, d + n);
    payload.resize(st::kBlobBuckets[0]);
    f = st::encode_blob(1 + ((d[0] >> 2) & 1), payload);
  } else {
    f.assign(d + 1, d + n);
    if ((d[0] & 1) != 0 && f.size() >= 10) {
      f[0] = 'O';
      f[1] = 'B';
      f[2] = 'L';
      f[3] = 'B';
      ombra::fuzz::put_le(f.data() + 4, st::kBlobVersion, 2);
      ombra::fuzz::put_le(f.data() + 6, st::crc32(f.data() + 10, f.size() - 10), 4);
    }
  }
  const std::string path = g_dir->path() + "/blob.oblob";
  ombra::fuzz::write_file(path, f.data(), f.size());
  for (std::uint32_t kind : {st::kBlobKindRecord, st::kBlobKindDocument}) {
    try {
      const auto p = st::read_blob_file(path, kind);
      if (!st::is_blob_bucket(p.size()) || st::encode_blob(kind, p) != f) std::abort();
    } catch (const std::exception&) {
    }
  }
  return 0;
}
