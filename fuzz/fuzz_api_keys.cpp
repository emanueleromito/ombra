// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: the API key file parser (ApiKeyStore::load and reload, server/grpc/src/auth.cpp).
// The input is the whole file, written with mode 0600 into the target's private temp dir. Tokens
// used here are synthetic.
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

#include "fuzz_util.hpp"
#include "ombra/server/auth.hpp"

namespace srv = ombra::server;

static ombra::fuzz::TempDir* g_dir = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  const std::string path = g_dir->path() + "/" + srv::ApiKeyStore::kFileName;
  ombra::fuzz::write_file(path, d, n);
  ::chmod(path.c_str(), 0600);
  try {
    srv::ApiKeyStore store = srv::ApiKeyStore::load(path);
    const std::size_t count = store.size();
    (void)store.verify("tok-synthetic");
    (void)srv::authorize(&store, "Bearer tok-synthetic", srv::Scope::Read);
    store.reload(path);
    if (store.size() != count) std::abort();
  } catch (const std::exception&) {
  }
  // A key file readable by others is refused, whatever it contains.
  ::chmod(path.c_str(), 0644);
  try {
    (void)srv::ApiKeyStore::load(path);
    std::abort();
  } catch (const std::exception&) {
  }
  return 0;
}
