// Storage engine: segments + write-ahead log + catalog, with crash recovery.
//
// Layout under root: catalog.db, wal.log, segments/<name>.oseg, keys/<name>.
//
// Guarantee: an acknowledged write (insert() or remove() returned) survives a crash or power
// loss. A write that was not acknowledged is either fully there (segment, all its ids, catalog
// rows) or fully absent (no rows, no segment file); never partial. Single writer: methods are
// serialised by a mutex. Logs carry sizes, counts and timings only (hard rule 4).
#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "ombra/storage/catalog.hpp"
#include "ombra/storage/segment.hpp"
#include "ombra/storage/wal.hpp"

namespace ombra::storage {

struct RecoveryStats {
  std::size_t records = 0, rolled_forward = 0, discarded = 0, deletes_applied = 0,
              orphans_removed = 0, torn_bytes_dropped = 0;
};

class Store {
 public:
  // Opens (creating if empty) and runs recovery. openfhe_version is what column envelopes must
  // carry; it is checked when segments are read.
  Store(std::string root, std::string openfhe_version);

  void create_collection(const CollectionSpec& spec);
  bool get_collection(const std::string& name, CollectionSpec* out) const;

  // Appends one cluster block as a new immutable segment. Returns the segment name once the
  // write is acknowledged (durable). Throws on duplicate or tombstoned ids, unknown collection or
  // cluster, or a column envelope that does not match the collection's preset.
  std::string insert(const std::string& collection, std::uint32_t cluster,
                     const std::vector<std::string>& ids,
                     const std::vector<std::vector<std::uint8_t>>& column_envelopes);
  // Marks ids deleted (unknown ids ignored). Acknowledged on return.
  void remove(const std::string& collection, const std::vector<std::string>& ids);

  // Stores a public key / context envelope file and registers it. kind: "context" or "eval_key".
  void put_key_file(const std::string& collection, const std::string& kind,
                    const std::vector<std::uint8_t>& envelope);

  SegmentData read_segment(const std::string& collection, const std::string& segment) const;
  const Catalog& catalog() const { return *catalog_; }
  const RecoveryStats& recovery_stats() const { return stats_; }
  std::string segment_path(const std::string& name) const;
  std::string wal_path() const { return root_ + "/wal.log"; }
  std::string segments_dir() const { return root_ + "/segments"; }

 private:
  void recover();
  std::string root_, ofhe_;
  std::unique_ptr<Catalog> catalog_;
  std::unique_ptr<Wal> wal_;
  mutable std::mutex mu_;
  RecoveryStats stats_;
};

}  // namespace ombra::storage
