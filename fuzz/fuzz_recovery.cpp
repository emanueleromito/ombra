// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: crash recovery replay (Store::recover in server/storage/src/store.cpp). A template
// data dir with one collection and one segment is built once; each run copies it, replaces wal.log
// with the input (mode byte as in fuzz_wal.cpp) and opens a Store, which replays the log against
// the catalog and the segment files. The Store's own log lines go to stderr; they are silenced
// for the open so the fuzzer output stays readable.
#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include "fuzz_util.hpp"
#include "ombra/core/file_envelope.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/store.hpp"
#include "ombra/storage/wal.hpp"

namespace st = ombra::storage;
namespace fs = std::filesystem;

static ombra::fuzz::TempDir* g_dir = nullptr;
static std::string g_template;
static std::string g_segment;  // name of the segment in the template

static std::vector<std::uint8_t> column(std::uint8_t fill) {
  ombra::core::Envelope e{ombra::core::EnvelopeKind::Ciphertext, ombra::fuzz::kPreset,
                          ombra::fuzz::kOfhe, std::vector<std::uint8_t>(32, fill)};
  return ombra::core::write_envelope(e);
}

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  g_template = g_dir->path() + "/template";
  const int saved = ::dup(2);
  const int null = ::open("/dev/null", O_WRONLY);
  ::dup2(null, 2);
  {
    st::Store s(g_template, ombra::fuzz::kOfhe);
    st::CollectionSpec spec;
    spec.name = "docs";
    spec.preset = ombra::fuzz::kPreset;
    spec.dimension = 8;
    spec.security_level = 1;
    spec.filter_fields = {{0, 3}};
    spec.key_tag = "tag-synthetic";
    spec.cluster_count = 4;
    s.create_collection(spec);
    g_segment = s.insert("docs", 1, {"a1", "a2"}, {column(1), column(2)});
  }
  ::dup2(saved, 2);
  ::close(saved);
  ::close(null);
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
  const std::string work = g_dir->path() + "/work";
  std::error_code ec;
  fs::remove_all(work, ec);
  fs::copy(g_template, work, fs::copy_options::recursive, ec);
  if (ec) std::abort();
  ombra::fuzz::write_file(work + "/wal.log", f.data(), f.size());

  const int saved = ::dup(2);
  const int null = ::open("/dev/null", O_WRONLY);
  ::dup2(null, 2);
  try {
    st::Store s(work, ombra::fuzz::kOfhe);
    // After recovery the log is checkpointed and the collection is still readable.
    st::CollectionSpec spec;
    if (!s.get_collection("docs", &spec)) std::abort();
    (void)s.segments("docs");
    (void)s.tombstones("docs");
  } catch (const std::exception&) {
  }
  ::dup2(saved, 2);
  ::close(saved);
  ::close(null);
  return 0;
}
