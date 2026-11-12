// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Shared helpers for the storage tests. Synthetic data only.
#pragma once

#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "ombra/core/file_envelope.hpp"
#include "ombra/storage/catalog.hpp"

#define CHECK(cond)                                                                \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                                \
    }                                                                              \
  } while (0)

#define CHECK_THROWS(expr)                                                           \
  do {                                                                               \
    bool threw_ = false;                                                             \
    try {                                                                            \
      expr;                                                                          \
    } catch (const std::exception&) {                                                \
      threw_ = true;                                                                 \
    }                                                                                \
    if (!threw_) {                                                                   \
      std::fprintf(stderr, "expected throw %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      std::exit(1);                                                                  \
    }                                                                                \
  } while (0)

namespace ombra::storage::testing {

inline constexpr const char* kPreset = "m0_compact_n8192";
inline constexpr const char* kOfhe = "1.4.0-test";

struct TempDir {
  std::string path;
  TempDir() {
    std::string t = (std::filesystem::temp_directory_path() / "ombra_storage_XXXXXX").string();
    if (!::mkdtemp(t.data())) std::exit(2);
    path = t;
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;
};

inline std::vector<std::uint8_t> column(std::uint8_t fill, std::size_t n = 64,
                                        const std::string& preset = kPreset) {
  ombra::core::Envelope e{ombra::core::EnvelopeKind::Ciphertext, preset, kOfhe,
                          std::vector<std::uint8_t>(n, fill)};
  return ombra::core::write_envelope(e);
}

inline CollectionSpec spec(const std::string& name = "docs") {
  CollectionSpec s;
  s.name = name;
  s.preset = kPreset;
  s.dimension = 8;
  s.security_level = 1;
  s.filter_fields = {{0, 3}, {1, 5}};
  s.key_tag = "tag-synthetic";
  s.cluster_count = 4;
  return s;
}

}  // namespace ombra::storage::testing
