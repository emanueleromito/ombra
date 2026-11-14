// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Online backup of a data directory (ombra_backup): a copy of one instant, taken while a server
// may be writing, with no cooperation from the server and no change to the source.
//
// Why the copy is consistent
//   1. The catalog is the single source of truth. SQLite's backup API copies catalog.db from a
//      read-only connection in one step, inside one read snapshot: the copy is the catalog as of
//      one instant T. The server's writers cannot interleave with it (WAL mode readers never block
//      writers and never see half a commit).
//   2. The store only ever adds a catalog row for a file that is already durable (segment, blob,
//      key file: write, fsync, rename, then the catalog commit), and never reuses a file name
//      (the name counter is in the catalog). So every file the copy of the catalog names existed
//      at T, and its content cannot change afterwards: files are immutable once named.
//   3. The only way a file named by the copy can be gone later is a catalog commit that came after
//      T and removed it (replace, compaction, drop, blob or key replacement or deletion), followed
//      by an unlink. If a file has vanished when we come to copy it, the copy of the catalog is
//      stale: the whole run starts again with a new snapshot (up to 5 attempts), else it fails.
//      A file we already opened stays readable after an unlink, so a copy is never torn.
//   4. The write-ahead log is not copied. A write is acknowledged only after its catalog commit
//      (insert: catalog, then log commit record; delete: log record, then catalog tombstone,
//      acknowledged after both), so every acknowledged write is in the catalog copy. What the log
//      adds beyond that is in-flight writes at T, which were not acknowledged and are therefore
//      allowed to be absent. Files without a row at T (in-flight inserts, leftovers) are not
//      copied, so the backup has no orphans.
//   5. The result is verified (verify_data_dir) before success is reported: every copied file's
//      CRC, header and catalog cross-references. A corrupt source file (disk rot) fails the run.
//
// Also copied when present: api_keys.v1 (written atomically by the server). Not copied: wal.log,
// tls/, admin_api_key (the Docker volume holds these; see docs/site/operations/backups.md).
// The output directory must not exist or be empty and must not lie inside the source.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ombra::storage {

struct BackupReport {
  std::size_t segments = 0, blobs = 0, key_files = 0;
  std::uint64_t bytes = 0;  // bytes copied, files only
  int attempts = 0;         // 1 unless a file vanished and the run restarted
  bool api_keys = false;
};

// Throws std::runtime_error (a message with file names and counts only). On failure the output
// directory is emptied again, so a failed run leaves no backup that could be mistaken for one.
void backup_data_dir(const std::string& root, const std::string& out, BackupReport& report);

}  // namespace ombra::storage
