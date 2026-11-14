// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Fuzz target: the REST port's request parser (server/grpc/src/http.cpp) and the Authorization
// header check (authorize in server/grpc/src/auth.cpp) on what the parser extracts. The parser
// runs on the full input and on every prefix at a few cut points, as the server feeds it a growing
// buffer. A fixed synthetic key file gives the header check real entries to compare against.
#include <sys/stat.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include "fuzz_util.hpp"
#include "ombra/server/auth.hpp"
#include "ombra/server/http.hpp"

namespace srv = ombra::server;

static ombra::fuzz::TempDir* g_dir = nullptr;
static srv::ApiKeyStore* g_store = nullptr;
static constexpr const char* kToken = "tok-synthetic";

// A key file with one entry per scope, salts fixed so a run is reproducible.
extern "C" int LLVMFuzzerInitialize(int*, char***) {
  g_dir = new ombra::fuzz::TempDir();
  std::string content = std::string(srv::ApiKeyStore::kHeader) + "\n";
  int i = 0;
  for (const char* scope : {"read", "write", "admin"}) {
    std::array<std::uint8_t, 16> salt{};
    salt.fill(static_cast<std::uint8_t>(0x10 + i++));
    std::string buf(reinterpret_cast<const char*>(salt.data()), salt.size());
    buf += kToken;
    const auto h = srv::sha256(reinterpret_cast<const std::uint8_t*>(buf.data()), buf.size());
    content += srv::to_hex(salt.data(), salt.size()) + " " + srv::to_hex(h.data(), h.size()) + " " +
               scope + "\n";
  }
  const std::string path = g_dir->path() + "/api_keys.v1";
  ombra::fuzz::write_file(path, reinterpret_cast<const std::uint8_t*>(content.data()),
                          content.size());
  ::chmod(path.c_str(), 0600);
  g_store = new srv::ApiKeyStore(srv::ApiKeyStore::load(path));
  if (g_store->size() != 3) std::abort();
  return 0;
}

static void check_one(std::string_view buf) {
  srv::HttpRequest req;
  const auto r = srv::parse_http_request(buf, &req);
  if (r != srv::HttpParse::Ok) return;
  if (req.path.empty() || req.path[0] != '/' || req.method.empty()) std::abort();
  if (buf.size() > srv::kMaxHttpHeaderBytes && buf.find("\r\n\r\n") > srv::kMaxHttpHeaderBytes)
    std::abort();
  for (auto need : {srv::Scope::Read, srv::Scope::Write, srv::Scope::Admin}) {
    std::string id;
    const auto a = srv::authorize(g_store, req.authorization, need, &id);
    if (a == srv::AuthResult::Ok && req.authorization.substr(0, 7) != "Bearer ") std::abort();
  }
}

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* d, std::size_t n) {
  const std::string_view buf(reinterpret_cast<const char*>(d), n);
  check_one(buf);
  for (std::size_t cut : {n / 4, n / 2, n - n / 4}) check_one(buf.substr(0, cut));
  if (n > 0) (void)srv::http_response(100 + d[0], buf.substr(1));
  return 0;
}
