// SQLite catalog (schema_version 1, api_version 1). journal_mode=WAL, synchronous=FULL,
// foreign_keys=ON. Opening a catalog with another stored api_version or schema_version is refused.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct sqlite3;

namespace ombra::storage {

inline constexpr int kCatalogSchemaVersion = 1;
inline constexpr int kCatalogApiVersion = 1;

struct FilterField {
  std::uint32_t field_id = 0;
  std::uint32_t cardinality = 0;
};

// Mirrors proto CollectionSpec (proto/ombra/v1/ombra.proto).
struct CollectionSpec {
  std::uint32_t api_version = 1;
  std::string name;
  std::string preset;
  std::uint32_t dimension = 0;
  std::uint32_t security_level = 0;  // proto SecurityLevel value
  std::vector<FilterField> filter_fields;
  std::string key_tag;
  std::uint32_t cluster_count = 0;
};

struct SegmentRow {
  std::uint32_t cluster = 0;
  std::string file;
  std::uint32_t n_vectors = 0, n_columns = 0, crc = 0;
};

class Catalog {
 public:
  explicit Catalog(const std::string& path);
  ~Catalog();
  Catalog(const Catalog&) = delete;
  Catalog& operator=(const Catalog&) = delete;

  void create_collection(const CollectionSpec& spec);  // also creates the cluster rows
  bool get_collection(const std::string& name, CollectionSpec* out) const;
  std::vector<std::string> list_collections() const;

  std::uint64_t next_segment_id();  // durable counter
  bool has_segment(const std::string& file) const;
  std::vector<SegmentRow> segments(const std::string& collection) const;
  std::vector<std::string> all_segment_files() const;

  // One transaction: the segment row plus one vectors row per id. Fails (and rolls back) on a
  // duplicate id. Idempotent when the segment is already registered.
  void add_segment(const std::string& collection, const SegmentRow& seg,
                   const std::vector<std::string>& ids);
  bool id_exists(const std::string& collection, const std::string& id) const;
  // Tombstone the ids that exist; unknown ids are ignored; idempotent.
  void tombstone(const std::string& collection, const std::vector<std::string>& ids);
  std::vector<std::string> tombstones(const std::string& collection) const;
  std::vector<std::string> vector_ids(const std::string& collection) const;

  void put_key_file(const std::string& collection, const std::string& kind,
                    const std::string& file);
  std::vector<std::pair<std::string, std::string>> key_files(const std::string& collection) const;

  sqlite3* raw() const { return db_; }  // for tests that tamper with versions

 private:
  sqlite3* db_ = nullptr;
};

}  // namespace ombra::storage
