// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: write-ahead log reading (server/storage/src/wal.cpp). Input: one mode byte, then the
// bytes of wal.log. Mode bit 0 set: the harness rewrites magic and version and recomputes the CRC
// of every frame whose length fits, so the fuzzer reaches the record decoder.
// Checks: recover() either throws or returns records; a second recover() on the repaired file
// returns the same count and drops nothing; an append after recovery is read back.
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "fuzz_util.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/wal.hpp"

namespace st = ombra::storage;

static ombra::fuzz::TempDir* g_dir = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  return 0;
}

static void fix_frames(std::vector<std::uint8_t>& f) {
  if (f.size() < 6) return;
  f[0] = 'O';
  f[1] = 'W';
  f[2] = 'A';
  f[3] = 'L';
  ombra::fuzz::put_le(f.data() + 4, st::kWalVersion, 2);
  std::size_t pos = 6;
  while (f.size() - pos >= 8) {
    const std::uint64_t len = st::get_le(f.data() + pos, 4);
    if (len > f.size() - pos - 8) break;
    ombra::fuzz::put_le(f.data() + pos + 4, st::crc32(f.data() + pos + 8, len), 4);
    pos += 8 + len;
  }
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  if (n < 1) return 0;
  std::vector<std::uint8_t> f(d + 1, d + n);
  if ((d[0] & 1) != 0) fix_frames(f);
  const std::string path = g_dir->path() + "/wal.log";
  ombra::fuzz::write_file(path, f.data(), f.size());
  try {
    std::size_t count = 0;
    {
      st::Wal w(path);
      count = w.recover().size();
    }
    {
      st::Wal w(path);
      std::size_t dropped = 0;
      if (w.recover(&dropped).size() != count || dropped != 0) std::abort();
      st::WalRecord r{st::WalType::Delete, "c", 0, "", {"i1"}};
      w.append(r);
      if (w.recover().size() != count + 1) std::abort();
    }
  } catch (const std::exception&) {
  }
  return 0;
}
