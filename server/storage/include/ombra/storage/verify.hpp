// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Read-only integrity check of a data directory (ombra_verify), and the read-only catalog open it
// shares with ombra_backup.
//
// verify_data_dir never writes, truncates, repairs or runs recovery, and takes no lock, so it is
// safe next to a running server. The server holds no lock on the directory, so verify cannot tell
// whether one is running. On a live directory a write in flight can show up as a warning (an
// orphan file, a torn log tail); a file that a later catalog commit removed during the run is
// dropped from the findings again. For a result you can rely on, verify a backup or a stopped
// directory.
//
// Findings name files, counts and kinds only: never ids, vectors or other contents.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct sqlite3;

namespace ombra::storage {

enum class Severity { Warning, Corrupt };

struct Finding {
  Severity severity = Severity::Corrupt;
  std::string kind;  // catalog, segment, blob, key, wal, orphan
  std::string file;  // file name inside the data directory, "" if not about one file
  std::string what;
};

struct VerifyReport {
  std::size_t collections = 0, segments = 0, blobs = 0, key_files = 0, vectors = 0;
  std::size_t wal_records = 0;
  std::vector<Finding> findings;
  std::size_t corrupt_count() const;
  std::size_t warning_count() const;
  bool clean() const { return corrupt_count() == 0; }  // warnings do not count
};

struct VerifyOptions {
  // If set, every segment column and key file must carry this OpenFHE version (what the server
  // would require). Empty: each file only has to agree with itself.
  std::string expected_openfhe_version;
};

// Throws std::runtime_error only if `root` is not a readable directory. Everything wrong inside is
// a finding.
void verify_data_dir(const std::string& root, VerifyReport& report, const VerifyOptions& opt = {});

std::string report_text(const VerifyReport& r);
std::string report_json(const VerifyReport& r);

// Opens <root>/catalog.db read-only without creating or changing any file: through SQLite's
// immutable mode when no catalog.db-wal exists (nobody has the database open), else normally
// read-only (a server holds it, SQLite coordinates through the shared-memory file). The caller
// closes it with sqlite3_close. Throws std::runtime_error.
sqlite3* open_catalog_readonly(const std::string& root);

}  // namespace ombra::storage
