// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: segment file reading (server/storage/src/segment.cpp), which also runs the column
// envelope and seeded payload checks. Input: one mode byte, then the file bytes. Mode bit 0 set:
// the harness rewrites magic, version and CRC so the fuzzer gets past the checksum into the body.
// A segment that reads back must encode to the same bytes. The insert-time column check
// (check_column_for_insert) gets the same bytes as a bare envelope.
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "fuzz_util.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/segment.hpp"

namespace st = ombra::storage;

static ombra::fuzz::TempDir* g_dir = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  if (n < 1) return 0;
  const bool fix = (d[0] & 1) != 0;
  std::vector<std::uint8_t> f(d + 1, d + n);
  if (fix && f.size() >= 10) {
    f[0] = 'O';
    f[1] = 'S';
    f[2] = 'E';
    f[3] = 'G';
    ombra::fuzz::put_le(f.data() + 4, st::kSegmentVersion, 2);
    ombra::fuzz::put_le(f.data() + 6, st::crc32(f.data() + 10, f.size() - 10), 4);
  }
  const std::string path = g_dir->path() + "/seg.oseg";
  ombra::fuzz::write_file(path, f.data(), f.size());
  try {
    st::SegmentInfo info;
    const auto s = st::read_segment(path, ombra::fuzz::kPreset, ombra::fuzz::kOfhe, &info);
    if (st::encode_segment(s) != f) std::abort();
    if (info.n_vectors != s.ids.size() || info.n_columns != s.column_envelopes.size()) std::abort();
  } catch (const std::exception&) {
  }
  try {
    st::ColumnExpect expect;
    expect.key_tag = "tag-synthetic";
    st::check_column_for_insert(f.data(), f.size(), ombra::fuzz::kPreset, ombra::fuzz::kOfhe,
                                expect);
  } catch (const std::exception&) {
  }
  return 0;
}
