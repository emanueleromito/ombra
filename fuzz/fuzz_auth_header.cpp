// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: the Authorization header check (authorize in server/grpc/src/auth.cpp) and the
// small flag parsers of server/grpc/src/limits.cpp, on arbitrary bytes. A fixed synthetic key makes
// "Bearer tok-synthetic" a valid header, which the dictionary carries.
#include <sys/stat.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "fuzz_util.hpp"
#include "ombra/server/auth.hpp"
#include "ombra/server/limits.hpp"

namespace srv = ombra::server;

static ombra::fuzz::TempDir* g_dir = nullptr;
static srv::ApiKeyStore* g_store = nullptr;

extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  std::array<std::uint8_t, 16> salt{};
  salt.fill(0x42);
  const std::string tok =
      std::string(reinterpret_cast<const char*>(salt.data()), 16) + "tok-synthetic";
  const auto h = srv::sha256(reinterpret_cast<const std::uint8_t*>(tok.data()), tok.size());
  const std::string content = std::string(srv::ApiKeyStore::kHeader) + "\n" +
                              srv::to_hex(salt.data(), salt.size()) + " " +
                              srv::to_hex(h.data(), h.size()) + " write\n";
  const std::string path = g_dir->path() + "/api_keys.v1";
  ombra::fuzz::write_file(path, reinterpret_cast<const std::uint8_t*>(content.data()),
                          content.size());
  ::chmod(path.c_str(), 0600);
  g_store = new srv::ApiKeyStore(srv::ApiKeyStore::load(path));
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  const std::string_view s(reinterpret_cast<const char*>(d), n);
  for (auto need : {srv::Scope::Read, srv::Scope::Write, srv::Scope::Admin}) {
    const auto r = srv::authorize(g_store, s, need);
    if (r == srv::AuthResult::Ok && need == srv::Scope::Admin) std::abort();    // key is write only
    if (srv::authorize(nullptr, s, need) != srv::AuthResult::Ok) std::abort();  // --no-auth
  }
  // Flag parsers.
  const std::string str(s);
  srv::Scope scope;
  (void)srv::parse_scope(str, &scope);
  std::uint64_t v = 0;
  if (srv::parse_uint(str, 1, 1000000, &v) && (v < 1 || v > 1000000)) std::abort();
  if (srv::parse_uint(str, 0, UINT64_MAX, &v) && std::to_string(v).size() > str.size())
    std::abort();
  std::size_t bytes;
  if (n >= 8) {
    std::uint64_t mb = 0;
    for (int i = 0; i < 8; ++i) mb |= std::uint64_t{d[i]} << (8 * i);
    (void)srv::mb_to_bytes(mb, &bytes);
  }
  (void)srv::is_loopback_listen(str);
  (void)srv::is_loopback_host(str);
  return 0;
}
