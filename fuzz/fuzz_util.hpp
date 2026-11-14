// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Helpers shared by the fuzz targets. Synthetic data only; nothing here touches a real data dir.
#pragma once

#include <sanitizer/common_interface_defs.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ombra::fuzz {

// A private directory under the temp dir, removed at exit and when libFuzzer reports a crash
// (the death callback), so a run never leaves files behind and never writes anywhere else.
class TempDir {
 public:
  TempDir() {
    std::string t = (std::filesystem::temp_directory_path() / "ombra_fuzz_XXXXXX").string();
    if (!::mkdtemp(t.data())) std::exit(2);
    path_ = t;
    registered() = t;
    std::atexit([] { cleanup(); });
    __sanitizer_set_death_callback([] { cleanup(); });
  }
  ~TempDir() { cleanup(); }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
  const std::string& path() const { return path_; }

 private:
  // The directory to remove lives in a static, so cleanup is safe after the object is gone.
  static std::string& registered() {
    static std::string* p = new std::string();
    return *p;
  }
  static void cleanup() {
    std::string& r = registered();
    if (r.empty()) return;
    std::error_code ec;
    std::filesystem::remove_all(r, ec);
    r.clear();
  }
  std::string path_;
};

inline void write_file(const std::string& path, const std::uint8_t* d, std::size_t n) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(d), static_cast<std::streamsize>(n));
}

inline void put_le(std::uint8_t* p, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) p[i] = static_cast<std::uint8_t>(v >> (8 * i));
}

// Same synthetic names the storage tests use. A target that wants to get past the preset and
// OpenFHE version checks of the envelope reader expects these, and the corpus carries them.
inline constexpr const char* kPreset = "m0_compact_n8192";
inline constexpr const char* kOfhe = "1.4.0-test";

}  // namespace ombra::fuzz
