// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Online backup: a copy taken while the store is open and keeps being written is exactly the state
// at backup time, restores with recovery, and verifies clean. Also under a concurrent writer, with
// a source that changes under the copy, and the refusals. Synthetic data only.
#include "ombra/storage/backup.hpp"

#include <sys/wait.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#include "fill_util.hpp"
#include "ombra/storage/verify.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;

namespace {

void verify_clean(const std::string& root, bool allow_warnings) {
  VerifyReport r;
  verify_data_dir(root, r);
  for (auto& f : r.findings)
    std::fprintf(stderr, "finding %s %s: %s\n", f.kind.c_str(), f.file.c_str(), f.what.c_str());
  CHECK(r.clean());
  if (!allow_warnings) CHECK(r.findings.empty());
}

std::size_t count_files(const std::string& root) {
  std::size_t n = 0;
  for (auto& e : fs::recursive_directory_iterator(root)) n += e.is_regular_file();
  return n;
}

}  // namespace

int main() {
  {  // Backup while the store is open, keep writing, restore, compare.
    TempDir src, dst, restored;
    fill_store(src.path);
    Store live(src.path, kOfhe);
    // Stray files in the source (an in-flight write, leftovers) must not be copied.
    std::ofstream(src.path + "/segments/s0000000000000fff.oseg") << "x";
    std::ofstream(src.path + "/blobs/stray.tmp") << "x";
    const std::string at_backup = dump_state(live, true);

    const std::string out = dst.path + "/b";
    BackupReport rep;
    backup_data_dir(src.path, out, rep);
    CHECK(rep.attempts == 1 && rep.segments == 4 && rep.blobs == 2 && rep.key_files == 3);
    CHECK(!fs::exists(out + "/segments/s0000000000000fff.oseg") &&
          !fs::exists(out + "/blobs/stray.tmp"));
    CHECK(!fs::exists(out + "/wal.log"));  // not needed: see backup.hpp
    verify_clean(out, true);               // only the missing log file is noted

    // The source keeps changing after the backup.
    live.insert("docs", 1, {"vec-later-1", "vec-later-2"}, cols2(50));
    live.remove("docs", {"vec-alpha-1"});
    live.put_blob("docs", 1, "blob-rec-1", blob_bytes(99));
    live.put_key_files("docs", key_set(60));
    CHECK(live.drop_collection("other"));
    CHECK(dump_state(live, true) != at_backup);

    // Restore: copy the backup into a new directory, open it with Store (recovery runs).
    fs::copy(out, restored.path, fs::copy_options::recursive);
    {
      Store s(restored.path, kOfhe);
      CHECK(dump_state(s, true) == at_backup);  // exactly the state at backup time, names included
      CHECK(s.recovery_stats().records == 0);
      s.insert("docs", 1, {"vec-after-restore"}, cols2(70));  // and it takes writes
    }
    verify_clean(restored.path, false);
    // The backup directory itself was not changed by being restored from.
    verify_clean(out, true);
    CHECK(count_files(out) == 1 + 4 + 2 + 3);
  }

  {  // The log is not needed, but an unacknowledged tail in it must not matter either.
    TempDir src, dst;
    fill_store(src.path);
    Store live(src.path, kOfhe);
    const std::string before = dump_state(live, false);
    BackupReport rep;
    backup_data_dir(src.path, dst.path + "/b", rep);
    Store s(dst.path + "/b", kOfhe);
    CHECK(dump_state(s, false) == before);
  }

  {  // api_keys.v1 is carried over when present.
    TempDir src, dst;
    fill_store(src.path);
    std::ofstream(src.path + "/api_keys.v1") << "OMBRA-API-KEYS-1\n";
    BackupReport rep;
    backup_data_dir(src.path, dst.path + "/b", rep);
    CHECK(rep.api_keys && fs::exists(dst.path + "/b/api_keys.v1"));
  }

  {  // Refusals: not empty, inside the source, not a data directory, corrupt source file.
    TempDir src, dst;
    fill_store(src.path);
    BackupReport rep;
    fs::create_directories(dst.path + "/full");
    std::ofstream(dst.path + "/full/keep.txt") << "x";
    CHECK_THROWS(backup_data_dir(src.path, dst.path + "/full", rep));
    CHECK(fs::exists(dst.path + "/full/keep.txt"));  // never deleted
    CHECK_THROWS(backup_data_dir(src.path, src.path + "/inside", rep));
    CHECK_THROWS(backup_data_dir(src.path, src.path, rep));
    CHECK_THROWS(backup_data_dir(dst.path, dst.path + "-b", rep));  // no catalog.db
    fs::remove_all(dst.path + "-b");

    // A source file with a flipped byte (disk rot): the run fails and leaves nothing behind.
    const std::string seg = (*fs::directory_iterator(src.path + "/segments")).path().string();
    {
      std::fstream f(seg, std::ios::in | std::ios::out | std::ios::binary);
      f.seekp(40);
      f.put('\x7f');
    }
    CHECK_THROWS(backup_data_dir(src.path, dst.path + "/rot", rep));
    CHECK(fs::exists(dst.path + "/rot") && fs::is_empty(dst.path + "/rot"));
  }

  {  // A writer that replaces, compacts and drops while backups run: every backup is a consistent
     // state (verify clean, restores, every block readable, blobs readable).
    TempDir src, dst;
    {
      Store s(src.path, kOfhe);
      s.create_collection(spec("docs"));
    }
    Store live(src.path, kOfhe);
    std::atomic<bool> stop{false};
    std::atomic<int> rounds{0};
    std::thread writer([&] {
      int k = 0;
      while (!stop) {
        const std::string a = "w" + std::to_string(k) + "-a", b = "w" + std::to_string(k) + "-b";
        live.insert("docs", 0, {a, b}, cols2(static_cast<std::uint8_t>(k)));
        live.put_blob("docs", 1, "blob-rec-1", blob_bytes(static_cast<std::uint8_t>(k)));
        if (k % 3 == 1) live.put_key_files("docs", key_set(static_cast<std::uint8_t>(k)));
        if (k % 4 == 3) {
          live.remove("docs", {a, b});
          live.drop_dead_blocks("docs");
        }
        if (k % 5 == 4) {
          auto segs = live.segments("docs");
          std::vector<std::string> files;
          for (auto& r : segs)
            if (r.cluster == 0) files.push_back(r.file);
          std::vector<std::string> ids;
          for (auto& f : files)
            for (auto& id : live.catalog().live_ids("docs", f)) ids.push_back(id);
          if (!ids.empty()) live.replace_segments("docs", files, SegmentData{0, ids, cols2(7)});
        }
        ++k;
        rounds = k;
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
      }
    });
    for (int i = 0; i < 12; ++i) {
      const std::string out = dst.path + "/b" + std::to_string(i);
      BackupReport rep;
      backup_data_dir(src.path, out, rep);
      verify_clean(out, true);
      Store s(out, kOfhe);  // opens and recovers
      for (auto& r : s.segments("docs")) CHECK(!s.read_segment("docs", r.file).ids.empty());
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    stop = true;
    writer.join();
    CHECK(rounds > 2);  // the writer really ran next to the backups
  }
  {  // CLI: exit codes, output without ids.
    TempDir src, dst;
    fill_store(src.path);
    auto cli = [&](const std::string& args) {
      const std::string cmd =
          std::string(OMBRA_BACKUP_BIN) + " " + args + " > " + dst.path + "/out.txt 2>&1";
      int st = std::system(cmd.c_str());
      CHECK(WIFEXITED(st));
      return WEXITSTATUS(st);
    };
    auto out_text = [&] {
      std::ifstream in(dst.path + "/out.txt");
      return std::string(std::istreambuf_iterator<char>(in), {});
    };
    CHECK(cli("--data-dir " + src.path + " --out " + dst.path + "/b") == 0);
    auto t = out_text();
    CHECK(t.find("backup done: 4 segments, 2 blobs, 3 key files") != std::string::npos);
    CHECK(t.find("vec-") == std::string::npos && t.find("docs") == std::string::npos);
    verify_clean(dst.path + "/b", true);
    CHECK(cli("--data-dir " + src.path + " --out " + dst.path + "/b") == 1);  // not empty now
    CHECK(cli("--data-dir " + src.path) == 2);
    CHECK(cli("") == 2);
    CHECK(cli("--out x") == 2);
    CHECK(cli("--bogus") == 2);
  }
  std::fprintf(stderr, "backup_test: ok\n");
  return 0;
}
