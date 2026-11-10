// Write-ahead log. File: "OWAL", u16 version (1), then framed records: u32 body length, u32 CRC32
// of the body, body. Body: u8 type, then fields (strings are u32 length + bytes).
//   InsertIntent: collection, u32 cluster, segment name, u32 n, n ids
//   InsertCommit: collection, segment name
//   Delete:       collection, u32 n, n ids
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ombra::storage {

inline constexpr std::uint16_t kWalVersion = 1;

enum class WalType : std::uint8_t { InsertIntent = 1, InsertCommit = 2, Delete = 3 };

struct WalRecord {
  WalType type = WalType::InsertIntent;
  std::string collection;
  std::uint32_t cluster = 0;
  std::string segment;
  std::vector<std::string> ids;
};

class Wal {
 public:
  explicit Wal(std::string path);  // creates the file if missing; refuses an unknown version
  ~Wal();
  Wal(const Wal&) = delete;
  Wal& operator=(const Wal&) = delete;

  // Reads all valid records. A torn tail (short or bad CRC) is truncated away and fsynced.
  // dropped (optional) receives the number of bytes truncated.
  std::vector<WalRecord> recover(std::size_t* dropped = nullptr);
  void append(const WalRecord& r);  // write + fsync before returning
  void reset();                     // truncate to the header (checkpoint), fsync

 private:
  std::string path_;
  int fd_ = -1;
};

}  // namespace ombra::storage
