#include "ombra/server/auth.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ombra::server {

namespace {

constexpr std::uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void compress(std::uint32_t h[8], const std::uint8_t* p) {
  std::uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] = (std::uint32_t{p[4 * i]} << 24) | (std::uint32_t{p[4 * i + 1]} << 16) |
           (std::uint32_t{p[4 * i + 2]} << 8) | std::uint32_t{p[4 * i + 3]};
  for (int i = 16; i < 64; ++i) {
    const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
    const std::uint32_t ch = (e & f) ^ (~e & g);
    const std::uint32_t t1 = hh + S1 + ch + kK[i] + w[i];
    const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
    const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t t2 = S0 + mj;
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;
}

bool from_hex(std::string_view s, std::uint8_t* out, std::size_t n) {
  if (s.size() != 2 * n) return false;
  auto nib = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
  };
  for (std::size_t i = 0; i < n; ++i) {
    const int hi = nib(s[2 * i]), lo = nib(s[2 * i + 1]);
    if (hi < 0 || lo < 0) return false;
    out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
  }
  return true;
}

std::array<std::uint8_t, 32> salted_hash(const std::array<std::uint8_t, 16>& salt,
                                         std::string_view token) {
  std::vector<std::uint8_t> buf(salt.begin(), salt.end());
  buf.insert(buf.end(), token.begin(), token.end());
  return sha256(buf.data(), buf.size());
}

void random_bytes(std::uint8_t* out, std::size_t n) {
  const int fd = ::open("/dev/urandom", O_RDONLY | O_CLOEXEC);
  if (fd < 0) throw std::runtime_error("cannot open /dev/urandom");
  std::size_t got = 0;
  while (got < n) {
    const ssize_t r = ::read(fd, out + got, n - got);
    if (r <= 0) {
      ::close(fd);
      throw std::runtime_error("cannot read /dev/urandom");
    }
    got += static_cast<std::size_t>(r);
  }
  ::close(fd);
}

}  // namespace

std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t len) {
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::size_t i = 0;
  for (; i + 64 <= len; i += 64) compress(h, data + i);
  std::uint8_t tail[128] = {};
  const std::size_t rem = len - i;
  if (rem > 0) std::memcpy(tail, data + i, rem);
  tail[rem] = 0x80;
  const std::size_t tail_len = rem < 56 ? 64 : 128;
  const std::uint64_t bits = static_cast<std::uint64_t>(len) * 8;
  for (int k = 0; k < 8; ++k) tail[tail_len - 1 - k] = static_cast<std::uint8_t>(bits >> (8 * k));
  compress(h, tail);
  if (tail_len == 128) compress(h, tail + 64);
  std::array<std::uint8_t, 32> out{};
  for (int k = 0; k < 8; ++k)
    for (int j = 0; j < 4; ++j) out[4 * k + j] = static_cast<std::uint8_t>(h[k] >> (24 - 8 * j));
  return out;
}

std::string to_hex(const std::uint8_t* data, std::size_t len) {
  static const char* kDigits = "0123456789abcdef";
  std::string s;
  s.reserve(2 * len);
  for (std::size_t i = 0; i < len; ++i) {
    s.push_back(kDigits[data[i] >> 4]);
    s.push_back(kDigits[data[i] & 15]);
  }
  return s;
}

bool constant_time_equal(const std::uint8_t* a, const std::uint8_t* b, std::size_t len) {
  std::uint8_t diff = 0;
  for (std::size_t i = 0; i < len; ++i) diff |= static_cast<std::uint8_t>(a[i] ^ b[i]);
  return diff == 0;
}

bool parse_scope(std::string_view s, Scope* out) {
  if (s == "read")
    *out = Scope::Read;
  else if (s == "write")
    *out = Scope::Write;
  else if (s == "admin")
    *out = Scope::Admin;
  else
    return false;
  return true;
}

const char* scope_name(Scope s) {
  switch (s) {
    case Scope::Read:
      return "read";
    case Scope::Write:
      return "write";
    case Scope::Admin:
      return "admin";
  }
  return "?";
}

const char* rpc_name(Rpc r) {
  switch (r) {
    case Rpc::CreateCollection:
      return "CreateCollection";
    case Rpc::GetCollection:
      return "GetCollection";
    case Rpc::ListCollections:
      return "ListCollections";
    case Rpc::DropCollection:
      return "DropCollection";
    case Rpc::UploadKeys:
      return "UploadKeys";
    case Rpc::Insert:
      return "Insert";
    case Rpc::Delete:
      return "Delete";
    case Rpc::Search:
      return "Search";
    case Rpc::ListBlocks:
      return "ListBlocks";
    case Rpc::GetBlock:
      return "GetBlock";
    case Rpc::ReplaceBlock:
      return "ReplaceBlock";
    case Rpc::PutBlob:
      return "PutBlob";
    case Rpc::GetBlob:
      return "GetBlob";
    case Rpc::DeleteBlobs:
      return "DeleteBlobs";
    case Rpc::kCount:
      break;
  }
  return "?";
}

Scope required_scope(Rpc r) {
  switch (r) {
    case Rpc::GetCollection:
    case Rpc::ListCollections:
    case Rpc::ListBlocks:
    case Rpc::GetBlock:
    case Rpc::GetBlob:
    case Rpc::Search:
      return Scope::Read;
    case Rpc::CreateCollection:
    case Rpc::UploadKeys:
    case Rpc::Insert:
    case Rpc::Delete:
    case Rpc::ReplaceBlock:
    case Rpc::PutBlob:
    case Rpc::DeleteBlobs:
      return Scope::Write;
    case Rpc::DropCollection:
    case Rpc::kCount:
      break;
  }
  return Scope::Admin;
}

std::string ApiKeyStore::path_in(const std::string& data_dir) {
  return (std::filesystem::path(data_dir) / kFileName).string();
}

ApiKeyStore ApiKeyStore::load(const std::string& path) {
  struct stat st {};
  if (::stat(path.c_str(), &st) != 0) throw std::runtime_error("cannot read the API key file");
  if ((st.st_mode & 077) != 0)
    throw std::runtime_error(
        "the API key file must not be accessible by group or others (chmod 600)");
  std::ifstream in(path);
  std::string line;
  if (!std::getline(in, line) || line != kHeader)
    throw std::runtime_error("the API key file has an unknown version header");
  ApiKeyStore store;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::istringstream ls(line);
    std::string salt_hex, hash_hex, scope_s, extra;
    if (!(ls >> salt_hex >> hash_hex >> scope_s) || (ls >> extra))
      throw std::runtime_error("malformed line in the API key file");
    Entry e;
    if (!from_hex(salt_hex, e.salt.data(), e.salt.size()) ||
        !from_hex(hash_hex, e.hash.data(), e.hash.size()) || !parse_scope(scope_s, &e.scope))
      throw std::runtime_error("malformed line in the API key file");
    store.entries_.push_back(e);
  }
  return store;
}

std::optional<Verified> ApiKeyStore::verify(std::string_view token) const {
  std::optional<Verified> found;
  for (const Entry& e : entries_) {
    const auto h = salted_hash(e.salt, token);
    const bool match = constant_time_equal(h.data(), e.hash.data(), h.size());
    if (match && !found) found = Verified{to_hex(e.hash.data(), 4), e.scope};
  }
  return found;
}

std::string create_api_key(const std::string& data_dir, Scope scope, std::string* key_id) {
  namespace fs = std::filesystem;
  fs::create_directories(data_dir);
  const std::string path = ApiKeyStore::path_in(data_dir);
  std::string content;
  if (fs::exists(path)) {
    std::ifstream in(path);
    std::string first;
    if (!std::getline(in, first) || first != ApiKeyStore::kHeader)
      throw std::runtime_error("the existing API key file has an unknown version header");
    content = first + "\n";
    std::string line;
    while (std::getline(in, line))
      if (!line.empty()) content += line + "\n";
  } else {
    content = std::string(ApiKeyStore::kHeader) + "\n";
  }

  std::uint8_t raw[32];
  std::array<std::uint8_t, 16> salt{};
  random_bytes(raw, sizeof raw);
  random_bytes(salt.data(), salt.size());
  const std::string token = to_hex(raw, sizeof raw);
  const auto hash = salted_hash(salt, token);
  content += to_hex(salt.data(), salt.size()) + " " + to_hex(hash.data(), hash.size()) + " " +
             scope_name(scope) + "\n";
  if (key_id) *key_id = to_hex(hash.data(), 4);

  const std::string tmp = path + ".tmp";
  ::unlink(tmp.c_str());
  const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) throw std::runtime_error("cannot create the API key file");
  std::size_t off = 0;
  bool ok = true;
  while (off < content.size()) {
    const ssize_t w = ::write(fd, content.data() + off, content.size() - off);
    if (w <= 0) {
      ok = false;
      break;
    }
    off += static_cast<std::size_t>(w);
  }
  ok = ok && ::fchmod(fd, 0600) == 0 && ::fsync(fd) == 0;
  ::close(fd);
  if (!ok || ::rename(tmp.c_str(), path.c_str()) != 0) {
    ::unlink(tmp.c_str());
    throw std::runtime_error("cannot write the API key file");
  }
  const int dfd = ::open(data_dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dfd >= 0) {
    ::fsync(dfd);
    ::close(dfd);
  }
  return token;
}

AuthResult authorize(const ApiKeyStore* store, std::string_view authorization, Scope need,
                     std::string* key_id) {
  if (store == nullptr) return AuthResult::Ok;
  constexpr std::string_view kPrefix = "Bearer ";
  if (authorization.substr(0, kPrefix.size()) != kPrefix) return AuthResult::Unauthenticated;
  const auto v = store->verify(authorization.substr(kPrefix.size()));
  if (!v) return AuthResult::Unauthenticated;
  if (key_id) *key_id = v->id;
  return scope_allows(v->scope, need) ? AuthResult::Ok : AuthResult::PermissionDenied;
}

}  // namespace ombra::server
