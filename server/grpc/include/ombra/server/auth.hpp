// API keys and scopes. An API key is an access token for the data API, not an FHE key; the server
// never sees a secret key. Tokens are random, shown once at creation and stored only as salted
// SHA-256 in <data-dir>/api_keys.v1 (mode 0600). Nothing here logs a token.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ombra::server {

// SHA-256 (FIPS 180-4). Self-contained so the hashing is unit tested on the CPU without OpenSSL.
std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::size_t len);
std::string to_hex(const std::uint8_t* data, std::size_t len);
// Equal-length byte strings compared without an early exit.
bool constant_time_equal(const std::uint8_t* a, const std::uint8_t* b, std::size_t len);

enum class Scope : int { Read = 1, Write = 2, Admin = 3 };
bool parse_scope(std::string_view s, Scope* out);
const char* scope_name(Scope s);

// Every RPC of the service.
enum class Rpc : int {
  CreateCollection,
  GetCollection,
  ListCollections,
  DropCollection,
  UploadKeys,
  Insert,
  Delete,
  Search,
  ListBlocks,
  GetBlock,
  ReplaceBlock,
  PutBlob,
  GetBlob,
  DeleteBlobs,
  kCount
};
inline constexpr std::size_t kRpcCount = static_cast<std::size_t>(Rpc::kCount);
const char* rpc_name(Rpc r);
// read: Get/List collections, ListBlocks, GetBlock, Search. write: read + Insert, Delete,
// UploadKeys, ReplaceBlock, PutBlob, DeleteBlobs, CreateCollection (GetBlob is read). admin:
// everything (DropCollection, REST /metrics).
Scope required_scope(Rpc r);
inline bool scope_allows(Scope have, Scope need) {
  return static_cast<int>(have) >= static_cast<int>(need);
}

enum class AuthResult { Ok, Unauthenticated, PermissionDenied };

struct Verified {
  std::string id;  // first 8 hex of the stored hash; safe to log
  Scope scope = Scope::Read;
};

class ApiKeyStore {
 public:
  static constexpr const char* kFileName = "api_keys.v1";
  static constexpr const char* kHeader = "OMBRA-API-KEYS v1";

  // Throws std::runtime_error on a missing file, a wrong version header, a malformed line or a file
  // readable by group or others.
  static ApiKeyStore load(const std::string& path);
  static std::string path_in(const std::string& data_dir);

  // Hashes the token with every entry's salt and compares in constant time; no early exit.
  std::optional<Verified> verify(std::string_view token) const;
  std::size_t size() const { return entries_.size(); }

 private:
  struct Entry {
    std::array<std::uint8_t, 16> salt{};
    std::array<std::uint8_t, 32> hash{};
    Scope scope = Scope::Read;
  };
  std::vector<Entry> entries_;
};

// Makes a random 32-byte token (64 hex characters), appends its salted hash to the key file of
// `data_dir` (created if needed) with an atomic write (tmp, fsync, rename, mode 0600) and returns
// the token. The caller prints it once. Throws std::runtime_error.
std::string create_api_key(const std::string& data_dir, Scope scope, std::string* key_id = nullptr);

// `authorization` is the value of the `authorization` header ("Bearer <token>"), empty if absent.
// store == nullptr means authentication is off (--no-auth): everything is allowed.
AuthResult authorize(const ApiKeyStore* store, std::string_view authorization, Scope need,
                     std::string* key_id = nullptr);

}  // namespace ombra::server
