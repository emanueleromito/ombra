// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CPU unit tests of SHA-256, API key file (format, mode, hashing, verification), scope table,
// HTTP request parsing limits, metrics JSON (allow-listed keys only).
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "ombra/server/auth.hpp"
#include "ombra/server/http.hpp"
#include "ombra/server/metrics.hpp"

#define CHECK(c)                                                                \
  do {                                                                          \
    if (!(c)) {                                                                 \
      std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

using namespace ombra::server;
namespace fs = std::filesystem;

static std::string sha_hex(const std::string& s) {
  const auto h = sha256(reinterpret_cast<const std::uint8_t*>(s.data()), s.size());
  return to_hex(h.data(), h.size());
}

static std::string slurp(const std::string& p) {
  std::ifstream in(p);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static void test_sha256() {
  CHECK(sha_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  CHECK(sha_hex(std::string(1000000, 'a')) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  // Padding edge cases around the 55/56/64 byte boundaries must not crash and must differ.
  CHECK(sha_hex(std::string(55, 'x')) != sha_hex(std::string(56, 'x')));
  CHECK(sha_hex(std::string(63, 'x')) != sha_hex(std::string(64, 'x')));
  const std::uint8_t a[3] = {1, 2, 3}, b[3] = {1, 2, 3}, c[3] = {1, 2, 4};
  CHECK(constant_time_equal(a, b, 3) && !constant_time_equal(a, c, 3));
}

static void test_scopes() {
  Scope s;
  CHECK(parse_scope("read", &s) && s == Scope::Read);
  CHECK(parse_scope("write", &s) && s == Scope::Write);
  CHECK(parse_scope("admin", &s) && s == Scope::Admin);
  CHECK(!parse_scope("root", &s) && !parse_scope("", &s));
  CHECK(required_scope(Rpc::GetCollection) == Scope::Read);
  CHECK(required_scope(Rpc::ListCollections) == Scope::Read);
  CHECK(required_scope(Rpc::ListBlocks) == Scope::Read);
  CHECK(required_scope(Rpc::GetBlock) == Scope::Read);
  CHECK(required_scope(Rpc::Search) == Scope::Read);
  CHECK(required_scope(Rpc::CreateCollection) == Scope::Write);
  CHECK(required_scope(Rpc::Insert) == Scope::Write);
  CHECK(required_scope(Rpc::Delete) == Scope::Write);
  CHECK(required_scope(Rpc::UploadKeys) == Scope::Write);
  CHECK(required_scope(Rpc::ReplaceBlock) == Scope::Write);
  CHECK(required_scope(Rpc::PutBlob) == Scope::Write);
  CHECK(required_scope(Rpc::DeleteBlobs) == Scope::Write);
  CHECK(required_scope(Rpc::GetBlob) == Scope::Read);
  CHECK(required_scope(Rpc::DropCollection) == Scope::Admin);
  CHECK(scope_allows(Scope::Admin, Scope::Write) && scope_allows(Scope::Write, Scope::Read));
  CHECK(!scope_allows(Scope::Read, Scope::Write) && !scope_allows(Scope::Write, Scope::Admin));
  for (std::size_t i = 0; i < kRpcCount; ++i)
    CHECK(std::string(rpc_name(static_cast<Rpc>(i))) != "?");
}

static void test_key_file() {
  const fs::path dir =
      fs::temp_directory_path() / ("ombra_admin_test_" + std::to_string(::getpid()));
  fs::remove_all(dir);
  const std::string d = (dir / "data").string();  // does not exist yet: created

  std::string id_r, id_w, id_a;
  const std::string tr = create_api_key(d, Scope::Read, &id_r);
  const std::string tw = create_api_key(d, Scope::Write, &id_w);
  const std::string ta = create_api_key(d, Scope::Admin, &id_a);
  CHECK(tr.size() == 64 && tr != tw && tw != ta);
  CHECK(id_r.size() == 8);

  const std::string path = ApiKeyStore::path_in(d);
  struct stat st {};
  CHECK(::stat(path.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  CHECK(!fs::exists(path + ".tmp"));

  // Versioned header, and only salted hashes on disk: no token appears in the file.
  const std::string text = slurp(path);
  CHECK(text.rfind("OMBRA-API-KEYS v1\n", 0) == 0);
  CHECK(text.find(tr) == std::string::npos && text.find(tw) == std::string::npos &&
        text.find(ta) == std::string::npos);

  const ApiKeyStore store = ApiKeyStore::load(path);
  CHECK(store.size() == 3);
  auto v = store.verify(tw);
  CHECK(v && v->scope == Scope::Write && v->id == id_w);
  CHECK(store.verify(tr)->scope == Scope::Read && store.verify(ta)->scope == Scope::Admin);
  CHECK(!store.verify("") && !store.verify(tr + "0") && !store.verify(tr.substr(1)));

  // authorize(): header parsing and scope checks.
  CHECK(authorize(&store, "", Scope::Read) == AuthResult::Unauthenticated);
  CHECK(authorize(&store, tr, Scope::Read) == AuthResult::Unauthenticated);  // no "Bearer "
  CHECK(authorize(&store, "Bearer nope", Scope::Read) == AuthResult::Unauthenticated);
  CHECK(authorize(&store, "Bearer " + tr, Scope::Read) == AuthResult::Ok);
  CHECK(authorize(&store, "Bearer " + tr, Scope::Write) == AuthResult::PermissionDenied);
  CHECK(authorize(&store, "Bearer " + tw, Scope::Write) == AuthResult::Ok);
  CHECK(authorize(&store, "Bearer " + tw, Scope::Admin) == AuthResult::PermissionDenied);
  CHECK(authorize(&store, "Bearer " + ta, Scope::Admin) == AuthResult::Ok);
  CHECK(authorize(nullptr, "", Scope::Admin) == AuthResult::Ok);  // --no-auth

  // Refused files: wrong version, group-readable, malformed line.
  const std::string bad = (dir / "bad").string();
  { std::ofstream(bad) << "OMBRA-API-KEYS v2\n"; }
  ::chmod(bad.c_str(), 0600);
  bool threw = false;
  try {
    ApiKeyStore::load(bad);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  ::chmod(path.c_str(), 0644);
  threw = false;
  try {
    ApiKeyStore::load(path);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  ::chmod(path.c_str(), 0600);
  { std::ofstream(bad) << "OMBRA-API-KEYS v1\nzz zz admin\n"; }
  threw = false;
  try {
    ApiKeyStore::load(bad);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  fs::remove_all(dir);
}

static void test_http() {
  HttpRequest r;
  CHECK(parse_http_request("GET /metrics HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer abc\r\n\r\n",
                           &r) == HttpParse::Ok);
  CHECK(r.method == "GET" && r.path == "/metrics" && r.authorization == "Bearer abc");
  CHECK(parse_http_request("GET /healthz HTTP/1.0\r\nauthorization:   Bearer q  \r\n\r\n", &r) ==
            HttpParse::Ok &&
        r.authorization == "Bearer q");
  CHECK(parse_http_request("GET /healthz HTTP/1.1\r\n\r\n", &r) == HttpParse::Ok &&
        r.authorization.empty());
  CHECK(parse_http_request("GET /healthz HTTP/1.1\r\nHost: x\r\n", &r) == HttpParse::Incomplete);
  CHECK(parse_http_request("", &r) == HttpParse::Incomplete);
  CHECK(parse_http_request("GET /x HTTP/2\r\n\r\n", &r) == HttpParse::Bad);
  CHECK(parse_http_request("GET x HTTP/1.1\r\n\r\n", &r) == HttpParse::Bad);
  CHECK(parse_http_request("GET\r\n\r\n", &r) == HttpParse::Bad);
  CHECK(parse_http_request("GET /x HTTP/1.1\r\nnocolon\r\n\r\n", &r) == HttpParse::Bad);
  CHECK(parse_http_request("POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello", &r) ==
        HttpParse::Bad);
  CHECK(parse_http_request("POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n", &r) ==
        HttpParse::Bad);
  // Header cap: no terminator within 8 KiB is refused without waiting; a block over 8 KiB too.
  CHECK(parse_http_request(std::string(kMaxHttpHeaderBytes, 'a'), &r) == HttpParse::TooLarge);
  CHECK(parse_http_request(std::string(kMaxHttpHeaderBytes - 1, 'a'), &r) == HttpParse::Incomplete);
  const std::string big =
      "GET / HTTP/1.1\r\nX: " + std::string(kMaxHttpHeaderBytes, 'a') + "\r\n\r\n";
  CHECK(parse_http_request(big, &r) == HttpParse::TooLarge);

  const std::string resp = http_response(401, "{}");
  CHECK(resp.rfind("HTTP/1.1 401 Unauthorized\r\n", 0) == 0);
  CHECK(resp.find("Content-Length: 2\r\n") != std::string::npos);
  CHECK(resp.find("Connection: close") != std::string::npos);
}

static void test_metrics() {
  Metrics m;
  m.record(Rpc::Search, false, 2.5);
  m.record(Rpc::Search, true, 7.0);
  m.count_unauthenticated();
  m.count_permission_denied();
  m.count_permission_denied();
  const auto s = m.snapshot(Rpc::Search);
  CHECK(s.requests == 2 && s.errors == 1 && s.latency_count == 2);
  CHECK(s.latency_sum_us == 9500 && s.latency_max_us == 7000);
  Gauges g;
  g.collections = 2;
  g.vectors_live = 10;
  g.cache_hits = 3;
  g.cache_misses = 1;
  const std::string j = metrics_json(m, g);
  CHECK(j.find("\"collections\":2") != std::string::npos);
  CHECK(j.find("\"hit_rate\":0.7500") != std::string::npos);
  CHECK(j.find("\"permission_denied\":2") != std::string::npos);
  CHECK(j.find("\"Search\":{\"requests\":2,\"errors\":1") != std::string::npos);
  CHECK(j.front() == '{' && j.back() == '}');
}

int main() {
  test_sha256();
  test_scopes();
  test_key_file();
  test_http();
  test_metrics();
  return 0;
}
