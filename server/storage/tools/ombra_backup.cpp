// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_backup: consistent copy of a data directory, safe while the server runs. CPU only.
//   ombra_backup --data-dir DIR --out DIR2
// Exit 0 done (the copy was verified), 1 the backup failed, 2 usage error. Reads the source
// without changing it. Output: counts only. See ombra/storage/backup.hpp for why it is consistent.
#include <cstdio>
#include <exception>
#include <string>

#include "ombra/storage/backup.hpp"

namespace {
void usage(FILE* f) {
  std::fputs(
      "usage: ombra_backup --data-dir DIR --out DIR2\n"
      "  Copies the data directory as of one instant, also while a server is running. DIR2 must\n"
      "  not exist or be empty. To restore, start the server on DIR2 (or copy DIR2 into place).\n"
      "  Exit 0: done and verified, 1: failed, 2: usage error.\n",
      f);
}
}  // namespace

int main(int argc, char** argv) {
  std::string dir, out;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(stdout);
      return 0;
    } else if (a == "--data-dir" && i + 1 < argc) {
      dir = argv[++i];
    } else if (a == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else {
      usage(stderr);
      return 2;
    }
  }
  if (dir.empty() || out.empty()) {
    usage(stderr);
    return 2;
  }
  try {
    ombra::storage::BackupReport r;
    ombra::storage::backup_data_dir(dir, out, r);
    std::printf("backup done: %zu segments, %zu blobs, %zu key files, %llu bytes, %d attempt(s)\n",
                r.segments, r.blobs, r.key_files, static_cast<unsigned long long>(r.bytes),
                r.attempts);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ombra_backup: %s\n", e.what());
    return 1;
  }
}
