#include "ombra/storage/catalog.hpp"

#include <sqlite3.h>

#include <stdexcept>

namespace ombra::storage {

namespace {

[[noreturn]] void fail(sqlite3* db, const char* what) {
  // SQLite messages name tables and columns, never row values.
  throw std::runtime_error(std::string("catalog: ") + what + ": " + sqlite3_errmsg(db));
}

class Stmt {
 public:
  Stmt(sqlite3* db, const char* sql) : db_(db) {
    if (sqlite3_prepare_v2(db, sql, -1, &s_, nullptr) != SQLITE_OK) fail(db, "prepare");
  }
  ~Stmt() { sqlite3_finalize(s_); }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;

  Stmt& text(int i, const std::string& v) {
    if (sqlite3_bind_text(s_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT) != SQLITE_OK)
      fail(db_, "bind");
    return *this;
  }
  Stmt& blob(int i, const std::string& v) {
    if (sqlite3_bind_blob(s_, i, v.data(), static_cast<int>(v.size()), SQLITE_TRANSIENT) !=
        SQLITE_OK)
      fail(db_, "bind");
    return *this;
  }
  Stmt& num(int i, std::int64_t v) {
    if (sqlite3_bind_int64(s_, i, v) != SQLITE_OK) fail(db_, "bind");
    return *this;
  }
  bool step() {
    int rc = sqlite3_step(s_);
    if (rc == SQLITE_ROW) return true;
    if (rc == SQLITE_DONE) return false;
    fail(db_, "step");
  }
  void run() { step(); }
  void reset() { sqlite3_reset(s_); sqlite3_clear_bindings(s_); }
  std::string col_text(int i) const {
    auto p = sqlite3_column_blob(s_, i);
    return std::string(static_cast<const char*>(p), static_cast<std::size_t>(sqlite3_column_bytes(s_, i)));
  }
  std::int64_t col_num(int i) const { return sqlite3_column_int64(s_, i); }

 private:
  sqlite3* db_;
  sqlite3_stmt* s_ = nullptr;
};

void exec(sqlite3* db, const char* sql) {
  char* msg = nullptr;
  if (sqlite3_exec(db, sql, nullptr, nullptr, &msg) != SQLITE_OK) {
    std::string m = msg ? msg : "unknown";
    sqlite3_free(msg);
    throw std::runtime_error("catalog: " + m);
  }
}

// RAII transaction: rolls back unless commit() is called.
class Tx {
 public:
  explicit Tx(sqlite3* db) : db_(db) { exec(db_, "BEGIN IMMEDIATE"); }
  ~Tx() {
    if (open_) sqlite3_exec(db_, "ROLLBACK", nullptr, nullptr, nullptr);
  }
  void commit() {
    exec(db_, "COMMIT");
    open_ = false;
  }

 private:
  sqlite3* db_;
  bool open_ = true;
};

constexpr const char* kSchema = R"sql(
CREATE TABLE meta(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE collections(
  name TEXT PRIMARY KEY, api_version INTEGER NOT NULL, preset TEXT NOT NULL,
  dimension INTEGER NOT NULL, security_level INTEGER NOT NULL, key_tag TEXT NOT NULL,
  cluster_count INTEGER NOT NULL, filter_fields TEXT NOT NULL);
CREATE TABLE clusters(
  collection TEXT NOT NULL REFERENCES collections(name), cluster INTEGER NOT NULL,
  PRIMARY KEY(collection, cluster));
CREATE TABLE segments(
  file TEXT PRIMARY KEY, collection TEXT NOT NULL, cluster INTEGER NOT NULL,
  n_vectors INTEGER NOT NULL, n_columns INTEGER NOT NULL, crc INTEGER NOT NULL,
  FOREIGN KEY(collection, cluster) REFERENCES clusters(collection, cluster));
CREATE TABLE vectors(
  collection TEXT NOT NULL, id BLOB NOT NULL, segment TEXT NOT NULL REFERENCES segments(file),
  PRIMARY KEY(collection, id));
CREATE TABLE tombstones(
  collection TEXT NOT NULL, id BLOB NOT NULL, PRIMARY KEY(collection, id),
  FOREIGN KEY(collection, id) REFERENCES vectors(collection, id));
CREATE TABLE key_files(
  collection TEXT NOT NULL REFERENCES collections(name), kind TEXT NOT NULL,
  file TEXT NOT NULL, PRIMARY KEY(collection, kind));
INSERT INTO meta VALUES('schema_version', 1), ('api_version', 1), ('next_segment', 1);
)sql";

std::string encode_filters(const std::vector<FilterField>& f) {
  std::string s;
  for (const auto& x : f) {
    if (!s.empty()) s += ',';
    s += std::to_string(x.field_id) + ':' + std::to_string(x.cardinality);
  }
  return s;
}
std::vector<FilterField> decode_filters(const std::string& s) {
  std::vector<FilterField> out;
  std::size_t pos = 0;
  while (pos < s.size()) {
    auto colon = s.find(':', pos);
    auto comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    if (colon == std::string::npos || colon > comma)
      throw std::runtime_error("catalog: bad filter field text");
    out.push_back({static_cast<std::uint32_t>(std::stoul(s.substr(pos, colon - pos))),
                   static_cast<std::uint32_t>(std::stoul(s.substr(colon + 1, comma - colon - 1)))});
    pos = comma + 1;
  }
  return out;
}

}  // namespace

Catalog::Catalog(const std::string& path) {
  if (sqlite3_open_v2(path.c_str(), &db_, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr) !=
      SQLITE_OK) {
    std::string m = db_ ? sqlite3_errmsg(db_) : "out of memory";
    sqlite3_close(db_);
    throw std::runtime_error("catalog: cannot open: " + m);
  }
  try {
    exec(db_, "PRAGMA journal_mode=WAL");
    exec(db_, "PRAGMA synchronous=FULL");
    exec(db_, "PRAGMA foreign_keys=ON");
    sqlite3_busy_timeout(db_, 5000);
    bool has_meta;
    {
      Stmt q(db_, "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'");
      has_meta = q.step();
    }
    if (!has_meta) {
      Tx tx(db_);
      exec(db_, kSchema);
      tx.commit();
    }
    Stmt q(db_, "SELECT value FROM meta WHERE key=?");
    auto get = [&](const char* k) {
      q.reset();
      q.text(1, k);
      if (!q.step()) throw std::runtime_error(std::string("catalog: missing ") + k);
      return q.col_num(0);
    };
    if (get("schema_version") != kCatalogSchemaVersion)
      throw std::runtime_error("catalog: unsupported schema_version");
    if (get("api_version") != kCatalogApiVersion)
      throw std::runtime_error("catalog: unsupported api_version");
  } catch (...) {
    sqlite3_close(db_);
    db_ = nullptr;
    throw;
  }
}

Catalog::~Catalog() { sqlite3_close(db_); }

void Catalog::create_collection(const CollectionSpec& s) {
  if (s.api_version != kCatalogApiVersion)
    throw std::runtime_error("catalog: collection api_version must be 1");
  Tx tx(db_);
  Stmt(db_, "INSERT INTO collections VALUES(?,?,?,?,?,?,?,?)")
      .text(1, s.name).num(2, s.api_version).text(3, s.preset).num(4, s.dimension)
      .num(5, s.security_level).text(6, s.key_tag).num(7, s.cluster_count)
      .text(8, encode_filters(s.filter_fields)).run();
  Stmt c(db_, "INSERT INTO clusters VALUES(?,?)");
  // Cluster ids 0 .. cluster_count - 1 are data clusters; the id cluster_count itself is reserved
  // for the block(s) of cluster centers (stage 1 of the search).
  for (std::uint32_t i = 0; i <= s.cluster_count; ++i) {
    c.reset();
    c.text(1, s.name).num(2, i).run();
  }
  tx.commit();
}

bool Catalog::get_collection(const std::string& name, CollectionSpec* out) const {
  Stmt q(db_,
         "SELECT api_version,preset,dimension,security_level,key_tag,cluster_count,filter_fields "
         "FROM collections WHERE name=?");
  q.text(1, name);
  if (!q.step()) return false;
  if (out) {
    out->name = name;
    out->api_version = static_cast<std::uint32_t>(q.col_num(0));
    out->preset = q.col_text(1);
    out->dimension = static_cast<std::uint32_t>(q.col_num(2));
    out->security_level = static_cast<std::uint32_t>(q.col_num(3));
    out->key_tag = q.col_text(4);
    out->cluster_count = static_cast<std::uint32_t>(q.col_num(5));
    out->filter_fields = decode_filters(q.col_text(6));
  }
  return true;
}

std::vector<std::string> Catalog::list_collections() const {
  Stmt q(db_, "SELECT name FROM collections ORDER BY name");
  std::vector<std::string> v;
  while (q.step()) v.push_back(q.col_text(0));
  return v;
}

bool Catalog::drop_collection(const std::string& name, std::vector<std::string>* segment_files,
                              std::vector<std::string>* key_files) {
  Tx tx(db_);
  if (!get_collection(name, nullptr)) return false;
  if (segment_files) {
    segment_files->clear();
    for (const auto& r : segments(name)) segment_files->push_back(r.file);
  }
  if (key_files) {
    key_files->clear();
    for (const auto& kv : this->key_files(name)) key_files->push_back(kv.second);
  }
  // Children first (foreign keys are on).
  for (const char* sql : {"DELETE FROM tombstones WHERE collection=?",
                          "DELETE FROM vectors WHERE collection=?",
                          "DELETE FROM segments WHERE collection=?",
                          "DELETE FROM clusters WHERE collection=?",
                          "DELETE FROM key_files WHERE collection=?",
                          "DELETE FROM collections WHERE name=?"})
    Stmt(db_, sql).text(1, name).run();
  tx.commit();
  return true;
}

std::uint64_t Catalog::next_segment_id() {
  Tx tx(db_);
  std::uint64_t id;
  {
    Stmt q(db_, "SELECT value FROM meta WHERE key='next_segment'");
    q.step();
    id = static_cast<std::uint64_t>(q.col_num(0));
  }
  Stmt(db_, "UPDATE meta SET value=? WHERE key='next_segment'").num(1, static_cast<std::int64_t>(id + 1)).run();
  tx.commit();
  return id;
}

bool Catalog::has_segment(const std::string& file) const {
  Stmt q(db_, "SELECT 1 FROM segments WHERE file=?");
  q.text(1, file);
  return q.step();
}

std::vector<SegmentRow> Catalog::segments(const std::string& collection) const {
  Stmt q(db_,
         "SELECT cluster,file,n_vectors,n_columns,crc FROM segments WHERE collection=? ORDER BY file");
  q.text(1, collection);
  std::vector<SegmentRow> v;
  while (q.step())
    v.push_back({static_cast<std::uint32_t>(q.col_num(0)), q.col_text(1),
                 static_cast<std::uint32_t>(q.col_num(2)), static_cast<std::uint32_t>(q.col_num(3)),
                 static_cast<std::uint32_t>(q.col_num(4))});
  return v;
}

std::vector<std::string> Catalog::all_segment_files() const {
  Stmt q(db_, "SELECT file FROM segments");
  std::vector<std::string> v;
  while (q.step()) v.push_back(q.col_text(0));
  return v;
}

void Catalog::add_segment(const std::string& collection, const SegmentRow& seg,
                          const std::vector<std::string>& ids) {
  Tx tx(db_);
  if (has_segment(seg.file)) {
    tx.commit();
    return;
  }
  Stmt(db_, "INSERT INTO segments VALUES(?,?,?,?,?,?)")
      .text(1, seg.file).text(2, collection).num(3, seg.cluster).num(4, seg.n_vectors)
      .num(5, seg.n_columns).num(6, seg.crc).run();
  Stmt v(db_, "INSERT INTO vectors VALUES(?,?,?)");
  for (const auto& id : ids) {
    v.reset();
    v.text(1, collection).blob(2, id).text(3, seg.file).run();
  }
  tx.commit();
}

bool Catalog::id_exists(const std::string& collection, const std::string& id) const {
  Stmt q(db_, "SELECT 1 FROM vectors WHERE collection=? AND id=?");
  q.text(1, collection).blob(2, id);
  return q.step();
}

void Catalog::tombstone(const std::string& collection, const std::vector<std::string>& ids) {
  Tx tx(db_);
  Stmt s(db_,
         "INSERT OR IGNORE INTO tombstones SELECT collection,id FROM vectors "
         "WHERE collection=? AND id=?");
  for (const auto& id : ids) {
    s.reset();
    s.text(1, collection).blob(2, id).run();
  }
  tx.commit();
}

std::vector<std::string> Catalog::tombstones(const std::string& collection) const {
  Stmt q(db_, "SELECT id FROM tombstones WHERE collection=? ORDER BY id");
  q.text(1, collection);
  std::vector<std::string> v;
  while (q.step()) v.push_back(q.col_text(0));
  return v;
}

std::vector<std::string> Catalog::vector_ids(const std::string& collection) const {
  Stmt q(db_, "SELECT id FROM vectors WHERE collection=? ORDER BY id");
  q.text(1, collection);
  std::vector<std::string> v;
  while (q.step()) v.push_back(q.col_text(0));
  return v;
}

bool Catalog::is_live_in_other_cluster(const std::string& collection, const std::string& id,
                                       std::uint32_t excluded_cluster) const {
  Stmt q(db_,
         "SELECT 1 FROM vectors v JOIN segments s ON s.file=v.segment "
         "WHERE v.collection=? AND v.id=? AND s.cluster<>? AND NOT EXISTS("
         "SELECT 1 FROM tombstones t WHERE t.collection=v.collection AND t.id=v.id)");
  q.text(1, collection).blob(2, id).num(3, excluded_cluster);
  return q.step();
}

void Catalog::put_key_file(const std::string& collection, const std::string& kind,
                           const std::string& file) {
  Stmt(db_, "INSERT OR REPLACE INTO key_files VALUES(?,?,?)")
      .text(1, collection).text(2, kind).text(3, file).run();
}

std::vector<std::pair<std::string, std::string>> Catalog::key_files(
    const std::string& collection) const {
  Stmt q(db_, "SELECT kind,file FROM key_files WHERE collection=? ORDER BY kind");
  q.text(1, collection);
  std::vector<std::pair<std::string, std::string>> v;
  while (q.step()) v.emplace_back(q.col_text(0), q.col_text(1));
  return v;
}

std::vector<std::string> Catalog::all_key_files() const {
  Stmt q(db_, "SELECT file FROM key_files");
  std::vector<std::string> v;
  while (q.step()) v.push_back(q.col_text(0));
  return v;
}

}  // namespace ombra::storage
