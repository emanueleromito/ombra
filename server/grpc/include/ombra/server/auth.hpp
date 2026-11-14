// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// API keys and scopes. An API key is an access token for the data API, not an FHE key; the server
// never sees a secret key. Tokens are random, shown once at creation and stored only as salted
// SHA-256 in <data-dir>/api_keys.v1 or api_keys.v2 (mode 0600). v2 adds an optional collection per
// key; the server reads both, writes v1 until the first collection-scoped key is created and v2
// from then on. Nothing here logs a token.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
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
  std::string collection;  // empty: the key works on every collection (always for v1 files)
  // True if the key may act on `name`. An unscoped key may act on any collection.
  bool allows_collection(std::string_view name) const {
    return collection.empty() || collection == name;
  }
};

// A collection name that may be written to a v2 key file: 1 to 128 printable ASCII bytes, no
// space (the file is split on whitespace).
bool valid_key_collection(std::string_view name);

class ApiKeyStore {
 public:
  static constexpr const char* kFileName = "api_keys.v1";
  static constexpr const char* kHeader = "OMBRA-API-KEYS v1";
  static constexpr const char* kFileNameV2 = "api_keys.v2";
  static constexpr const char* kHeaderV2 = "OMBRA-API-KEYS v2";

  // Reads a v1 or a v2 file (the header says which). Throws std::runtime_error on a missing file,
  // an unknown version header, a malformed line (v1: exactly `salt hash scope`; v2: the same plus
  // an optional collection; an admin key with a collection is malformed) or a file readable by
  // group or others.
  static ApiKeyStore load(const std::string& path);
  // api_keys.v2 if it exists, else api_keys.v1. A v2 file is never ignored in favour of a v1 file.
  static std::string path_in(const std::string& data_dir);

  // Hashes the token with every entry's salt and compares in constant time; no early exit.
  // Thread-safe, also against reload().
  std::optional<Verified> verify(std::string_view token) const;
  std::size_t size() const;
  // Re-reads the key file (SIGHUP) so that revoked keys stop working without a restart. On any
  // error (missing file, bad mode, malformed line) throws and keeps the keys that are loaded.
  void reload(const std::string& path);

 private:
  struct Entry {
    std::array<std::uint8_t, 16> salt{};
    std::array<std::uint8_t, 32> hash{};
    Scope scope = Scope::Read;
    std::string collection;
  };
  using Entries = std::vector<Entry>;
  std::shared_ptr<const Entries> snapshot() const;
  std::shared_ptr<const Entries> entries_ = std::make_shared<const Entries>();
  std::unique_ptr<std::mutex> mu_ = std::make_unique<std::mutex>();
};

// Makes a random 32-byte token (64 hex characters), appends its salted hash to the key file of
// `data_dir` (created if needed) with an atomic write (tmp, fsync, rename, mode 0600) under an
// exclusive lock on <data_dir>/api_keys.v1.lock, so two concurrent creations cannot lose a key, and
// returns the token. The caller prints it once. Throws std::runtime_error.
// `collection` non-empty: the key is scoped to that collection (not allowed with Scope::Admin).
// The first scoped key migrates the v1 file to api_keys.v2 (the v1 file is renamed to
// api_keys.v1.migrated); an existing v2 file is always kept as v2. Without a scoped key the file
// stays v1.
std::string create_api_key(const std::string& data_dir, Scope scope, std::string* key_id = nullptr,
                           const std::string& collection = {});

// `authorization` is the value of the `authorization` header ("Bearer <token>"), empty if absent.
// store == nullptr means authentication is off (--no-auth): everything is allowed.
AuthResult authorize(const ApiKeyStore* store, std::string_view authorization, Scope need,
                     std::string* key_id = nullptr);
// Same, and fills `who` (id, scope, collection scope) whenever the token is valid, also when the
// scope is too low. With store == nullptr `who` is an unscoped admin with an empty id.
AuthResult authorize_who(const ApiKeyStore* store, std::string_view authorization, Scope need,
                         Verified* who);

}  // namespace ombra::server
