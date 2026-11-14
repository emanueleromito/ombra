// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/storage/verify.hpp"

#include <dirent.h>
#include <sqlite3.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <exception>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#include "ombra/core/file_envelope.hpp"
#include "ombra/storage/blob.hpp"
#include "ombra/storage/catalog.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/segment.hpp"
#include "ombra/storage/store.hpp"
#include "ombra/storage/wal.hpp"

namespace ombra::storage {

namespace {

namespace core = ombra::core;

struct Db {
  sqlite3* db = nullptr;
  ~Db() {
    if (db) sqlite3_close(db);
  }
};

struct Stmt {
  sqlite3_stmt* s = nullptr;
  Stmt(sqlite3* db, const char* sql) {
    if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK)
      throw std::runtime_error(std::string("catalog: ") + sqlite3_errmsg(db));
  }
  ~Stmt() { sqlite3_finalize(s); }
  Stmt(const Stmt&) = delete;
  Stmt& operator=(const Stmt&) = delete;
  bool step() { return sqlite3_step(s) == SQLITE_ROW; }
  std::string text(int i) const {
    auto p = sqlite3_column_blob(s, i);
    return std::string(static_cast<const char*>(p),
                       static_cast<std::size_t>(sqlite3_column_bytes(s, i)));
  }
  long long num(int i) const { return sqlite3_column_int64(s, i); }
};

bool exists(const std::string& p) {
  struct stat st {};
  return ::stat(p.c_str(), &st) == 0;
}

std::vector<std::string> list_dir_names(const std::string& p) {
  std::vector<std::string> v;
  DIR* d = ::opendir(p.c_str());
  if (!d) return v;
  while (auto* e = ::readdir(d)) {
    const std::string n = e->d_name;
    if (n != "." && n != "..") v.push_back(n);
  }
  ::closedir(d);
  return v;
}

struct SegRow {
  std::string file, collection;
  long long cluster, n_vectors, n_columns, crc;
};
struct KeyRow {
  std::string collection, kind, file;
};
struct BlobRow {
  std::string collection, file;
  long long kind, size;
};

void add(VerifyReport& r, Severity sv, const char* kind, const std::string& file,
         const std::string& what) {
  r.findings.push_back({sv, kind, file, what});
}

// PRAGMA integrity_check / quick_check / foreign_key_check: report the number of problems only
// (SQLite's messages can quote row keys).
void pragma_check(VerifyReport& r, sqlite3* db, const std::string& pragma) {
  Stmt q(db, ("PRAGMA " + pragma).c_str());
  std::size_t bad = 0;
  while (q.step()) {
    if (pragma == "foreign_key_check" || q.text(0) != "ok") ++bad;
  }
  if (bad)
    add(r, Severity::Corrupt, "catalog", "catalog.db",
        pragma + " reported " + std::to_string(bad) + " problem(s)");
}

core::EnvelopeKind key_kind(const std::string& kind, bool* ok) {
  *ok = true;
  if (kind == kKeyContext) return core::EnvelopeKind::Context;
  if (kind == kKeyEvalMult || kind == kKeyAutomorphism) return core::EnvelopeKind::PublicEvalKey;
  *ok = false;
  return core::EnvelopeKind::Context;
}

bool is_missing_error(const std::exception& e) {
  return std::string(e.what()).find("No such file") != std::string::npos;
}

}  // namespace

std::size_t VerifyReport::corrupt_count() const {
  std::size_t n = 0;
  for (const auto& f : findings) n += f.severity == Severity::Corrupt;
  return n;
}
std::size_t VerifyReport::warning_count() const { return findings.size() - corrupt_count(); }

sqlite3* open_catalog_readonly(const std::string& root) {
  const std::string path = root + "/catalog.db";
  if (!exists(path)) throw std::runtime_error("catalog.db does not exist");
  std::string uri = "file:";
  for (char c : path) {
    if (c == '?' || c == '#' || c == '%') {
      char b[4];
      std::snprintf(b, sizeof b, "%%%02X", static_cast<unsigned char>(c));
      uri += b;
    } else {
      uri += c;
    }
  }
  uri += exists(path + "-wal") ? "?mode=ro" : "?mode=ro&immutable=1";
  sqlite3* db = nullptr;
  if (sqlite3_open_v2(uri.c_str(), &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nullptr) !=
      SQLITE_OK) {
    const std::string m = db ? sqlite3_errmsg(db) : "out of memory";
    sqlite3_close(db);
    throw std::runtime_error("cannot open catalog.db read-only: " + m);
  }
  sqlite3_busy_timeout(db, 5000);
  return db;
}

void verify_data_dir(const std::string& root, VerifyReport& r, const VerifyOptions& opt) {
  struct stat st {};
  if (::stat(root.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
    throw std::runtime_error("not a readable directory: " + root);

  const std::string seg_dir = root + "/segments", key_dir = root + "/keys",
                    blob_dir = root + "/blobs";
  std::map<std::string, std::string> preset;  // collection -> preset
  std::vector<SegRow> segs;
  std::vector<KeyRow> keys;
  std::vector<BlobRow> blobs;
  std::map<std::string, long long> vec_per_seg;
  bool have_catalog = false;

  // ---- Catalog: integrity, then every row, all in one read transaction (one instant). ----
  try {
    Db d;
    d.db = open_catalog_readonly(root);
    sqlite3* db = d.db;
    if (sqlite3_exec(db, "BEGIN", nullptr, nullptr, nullptr) != SQLITE_OK)
      throw std::runtime_error(std::string("catalog: ") + sqlite3_errmsg(db));
    pragma_check(r, db, "integrity_check");
    pragma_check(r, db, "quick_check");
    pragma_check(r, db, "foreign_key_check");
    long long schema = 0, api = 0;
    {
      Stmt q(db, "SELECT key, value FROM meta");
      while (q.step()) {
        if (q.text(0) == "schema_version") schema = q.num(1);
        if (q.text(0) == "api_version") api = q.num(1);
      }
    }
    if (schema != kCatalogSchemaVersion && schema != 1)
      add(r, Severity::Corrupt, "catalog", "catalog.db", "unsupported schema_version");
    if (schema == 1)
      add(r, Severity::Warning, "catalog", "catalog.db",
          "schema_version 1: the server migrates it at the next start");
    if (api != kCatalogApiVersion)
      add(r, Severity::Corrupt, "catalog", "catalog.db", "unsupported api_version");
    {
      Stmt q(db, "SELECT name, preset FROM collections");
      while (q.step()) preset[q.text(0)] = q.text(1);
    }
    {
      Stmt q(db, "SELECT file, collection, cluster, n_vectors, n_columns, crc FROM segments");
      while (q.step())
        segs.push_back({q.text(0), q.text(1), q.num(2), q.num(3), q.num(4), q.num(5)});
    }
    {
      Stmt q(db, "SELECT collection, kind, file FROM key_files");
      while (q.step()) keys.push_back({q.text(0), q.text(1), q.text(2)});
    }
    if (schema >= 2) {
      Stmt q(db, "SELECT collection, file, kind, size FROM blobs");
      while (q.step()) blobs.push_back({q.text(0), q.text(1), q.num(2), q.num(3)});
    }
    {
      Stmt q(db, "SELECT segment, COUNT(*) FROM vectors GROUP BY segment");
      while (q.step()) {
        vec_per_seg[q.text(0)] = q.num(1);
        r.vectors += static_cast<std::size_t>(q.num(1));
      }
    }
    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    have_catalog = true;
  } catch (const std::exception& e) {
    add(r, Severity::Corrupt, "catalog", "catalog.db", e.what());
  }
  r.collections = preset.size();
  r.segments = segs.size();
  r.blobs = blobs.size();
  r.key_files = keys.size();
  if (!have_catalog) {
    // Without a catalog nothing can be cross-checked; the files are listed as unmatched.
    const std::tuple<std::string, const char*> dirs[] = {
        {seg_dir, "segments"}, {blob_dir, "blobs"}, {key_dir, "keys"}};
    for (const auto& [dir, label] : dirs)
      for (const auto& n : list_dir_names(dir))
        add(r, Severity::Warning, "orphan", n,
            std::string("file in ") + label + "/ cannot be matched: no readable catalog");
  }

  auto vcheck = [&](const std::string& seen) {
    if (!opt.expected_openfhe_version.empty() && !seen.empty() &&
        seen != opt.expected_openfhe_version)
      throw std::runtime_error("OpenFHE version differs from the expected one (file has '" + seen +
                               "')");
  };
  auto preset_of = [&](const std::string& collection) {
    auto it = preset.find(collection);
    return it == preset.end() ? std::string() : it->second;
  };

  // ---- Segments ----
  std::set<std::string> seg_known, key_known, blob_known;
  for (const auto& s : segs) {
    seg_known.insert(s.file);
    const std::string path = seg_dir + "/" + s.file;
    if (!is_valid_segment_name(s.file)) {
      add(r, Severity::Corrupt, "segment", s.file, "catalog row has an invalid segment name");
      continue;
    }
    if (!exists(path)) {
      add(r, Severity::Corrupt, "segment", s.file, "file missing");
      continue;
    }
    try {
      SegmentInfo info;
      std::string seen;
      auto sd = inspect_segment(path, preset_of(s.collection), &seen, &info);
      vcheck(seen);
      if (info.n_vectors != s.n_vectors || info.n_columns != s.n_columns)
        throw std::runtime_error("vector or column count differs from the catalog row");
      if (static_cast<long long>(info.crc) != s.crc)
        throw std::runtime_error("CRC differs from the catalog row");
      if (static_cast<long long>(sd.cluster) != s.cluster)
        throw std::runtime_error("cluster differs from the catalog row");
      auto it = vec_per_seg.find(s.file);
      const long long n = it == vec_per_seg.end() ? 0 : it->second;
      if (n != s.n_vectors)
        throw std::runtime_error("catalog has " + std::to_string(n) + " vector rows, block has " +
                                 std::to_string(s.n_vectors));
    } catch (const std::exception& e) {
      add(r, Severity::Corrupt, "segment", s.file, is_missing_error(e) ? "file missing" : e.what());
    }
  }

  // ---- Blobs ----
  for (const auto& b : blobs) {
    blob_known.insert(b.file);
    const std::string path = blob_dir + "/" + b.file;
    if (!exists(path)) {
      add(r, Severity::Corrupt, "blob", b.file, "file missing");
      continue;
    }
    try {
      if (!is_blob_kind(static_cast<std::uint32_t>(b.kind)))
        throw std::runtime_error("catalog row has an unknown blob kind");
      auto payload = read_blob_file(path, static_cast<std::uint32_t>(b.kind));
      if (static_cast<long long>(payload.size()) != b.size)
        throw std::runtime_error("size differs from the catalog row");
    } catch (const std::exception& e) {
      add(r, Severity::Corrupt, "blob", b.file, is_missing_error(e) ? "file missing" : e.what());
    }
  }

  // ---- Key files. The envelope has a header but no CRC: magic, version, kind, preset, OpenFHE
  //      version and lengths are checked; a flipped payload byte is not detectable here. ----
  for (const auto& k : keys) {
    key_known.insert(k.file);
    const std::string path = key_dir + "/" + k.file;
    if (!exists(path)) {
      add(r, Severity::Corrupt, "key", k.file, "file missing");
      continue;
    }
    try {
      bool ok;
      const auto kind = key_kind(k.kind, &ok);
      if (!ok) throw std::runtime_error("catalog row has an unknown key kind");
      auto bytes = read_file(path);
      const std::string ofhe = envelope_openfhe_version(bytes.data(), bytes.size());
      vcheck(ofhe);
      core::read_envelope(bytes, kind, preset_of(k.collection), ofhe);
    } catch (const std::exception& e) {
      add(r, Severity::Corrupt, "key", k.file, is_missing_error(e) ? "file missing" : e.what());
    }
  }

  // ---- Orphans: files no catalog row names (recovery sweeps them at the next start) ----
  if (have_catalog) {
    const std::tuple<std::string, const std::set<std::string>*, const char*> dirs[] = {
        {seg_dir, &seg_known, "segments"},
        {blob_dir, &blob_known, "blobs"},
        {key_dir, &key_known, "keys"}};
    for (const auto& [dir, known, label] : dirs)
      for (const auto& n : list_dir_names(dir))
        if (!known->count(n))
          add(r, Severity::Warning, "orphan", n,
              std::string("file in ") + label + "/ has no catalog row");
  }

  // ---- Write-ahead log (scanned, never truncated) ----
  {
    const std::string wal = root + "/wal.log";
    if (!exists(wal)) {
      add(r, Severity::Warning, "wal", "wal.log", "file missing (the server creates it)");
    } else {
      try {
        const WalScan sc = scan_wal_file(wal);
        r.wal_records = sc.records;
        if (!sc.header_ok) {
          add(r, Severity::Warning, "wal", "wal.log", "shorter than its header (torn creation)");
        } else if (sc.bad_magic) {
          add(r, Severity::Corrupt, "wal", "wal.log", "bad magic");
        } else if (sc.version != kWalVersion) {
          add(r, Severity::Corrupt, "wal", "wal.log",
              "unknown version " + std::to_string(sc.version));
        } else {
          if (sc.mid_corruption)
            add(r, Severity::Corrupt, "wal", "wal.log",
                "damaged record before valid ones: the server refuses to start");
          else if (sc.tail_bytes > 0)
            add(r, Severity::Warning, "wal", "wal.log",
                "torn tail of " + std::to_string(sc.tail_bytes) +
                    " byte(s): recovery drops it, nothing was repaired here");
          if (sc.bad_names)
            add(r, Severity::Corrupt, "wal", "wal.log",
                std::to_string(sc.bad_names) +
                    " record(s) with an invalid segment name: the server refuses to start");
        }
      } catch (const std::exception& e) {
        add(r, Severity::Corrupt, "wal", "wal.log", e.what());
      }
    }
  }

  // ---- A live server may have removed a file after our snapshot: ask the catalog again and drop
  //      "file missing" findings whose row is gone. ----
  bool any_missing = false;
  for (const auto& f : r.findings) any_missing |= f.what == "file missing" && f.kind != "wal";
  if (any_missing && have_catalog) {
    try {
      Db d;
      d.db = open_catalog_readonly(root);
      std::vector<Finding> kept;
      for (auto& f : r.findings) {
        if (f.what == "file missing" && f.kind != "wal") {
          const char* sql = f.kind == "segment" ? "SELECT 1 FROM segments WHERE file=?"
                            : f.kind == "blob"  ? "SELECT 1 FROM blobs WHERE file=?"
                                                : "SELECT 1 FROM key_files WHERE file=?";
          Stmt q(d.db, sql);
          sqlite3_bind_text(q.s, 1, f.file.c_str(), -1, SQLITE_TRANSIENT);
          if (!q.step()) continue;  // row gone: the server removed the file meanwhile
        }
        kept.push_back(std::move(f));
      }
      r.findings = std::move(kept);
    } catch (const std::exception&) {
      // keep the findings as they are
    }
  }
}

std::string report_text(const VerifyReport& r) {
  std::string o;
  for (const auto& f : r.findings) {
    o += f.severity == Severity::Corrupt ? "CORRUPT " : "warning ";
    o += f.kind;
    if (!f.file.empty()) o += " " + f.file;
    o += ": " + f.what + "\n";
  }
  o += "checked: " + std::to_string(r.collections) + " collections, " + std::to_string(r.segments) +
       " segments, " + std::to_string(r.blobs) + " blobs, " + std::to_string(r.key_files) +
       " key files, " + std::to_string(r.vectors) + " vectors, " + std::to_string(r.wal_records) +
       " log records\n";
  o += r.clean() ? "result: clean"
                 : "result: CORRUPT (" + std::to_string(r.corrupt_count()) + " problem(s))";
  if (r.warning_count()) o += ", " + std::to_string(r.warning_count()) + " warning(s)";
  o += "\n";
  return o;
}

namespace {
std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += static_cast<char>(c);
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      o += b;
    } else {
      o += static_cast<char>(c);
    }
  }
  return o + "\"";
}
}  // namespace

std::string report_json(const VerifyReport& r) {
  std::string o = std::string("{\"clean\":") + (r.clean() ? "true" : "false") +
                  ",\"corrupt\":" + std::to_string(r.corrupt_count()) +
                  ",\"warnings\":" + std::to_string(r.warning_count()) +
                  ",\"counts\":{\"collections\":" + std::to_string(r.collections) +
                  ",\"segments\":" + std::to_string(r.segments) +
                  ",\"blobs\":" + std::to_string(r.blobs) +
                  ",\"key_files\":" + std::to_string(r.key_files) +
                  ",\"vectors\":" + std::to_string(r.vectors) +
                  ",\"wal_records\":" + std::to_string(r.wal_records) + "},\"findings\":[";
  bool first = true;
  for (const auto& f : r.findings) {
    if (!first) o += ",";
    first = false;
    o += std::string("{\"severity\":") +
         (f.severity == Severity::Corrupt ? "\"corrupt\"" : "\"warning\"") +
         ",\"kind\":" + jstr(f.kind) + ",\"file\":" + jstr(f.file) + ",\"what\":" + jstr(f.what) +
         "}";
  }
  return o + "]}\n";
}

}  // namespace ombra::storage
