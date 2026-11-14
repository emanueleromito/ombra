// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Disk full: ENOSPC is injected at every instrumented I/O call (write, fsync, rename, create, the
// log append, the catalog commit) of every write operation of the store. For each call index the
// operation must either throw DiskFullError or have been applied completely, the live store must
// match the state before or after it (nothing half-way), the same operation must succeed after the
// fault is cleared, acknowledged data must be intact, and after the next open no temp or orphan
// file is left and verify is clean. Synthetic data only.
#include <cerrno>
#include <filesystem>
#include <functional>

#include "fill_util.hpp"
#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/verify.hpp"
#include "ombra/storage/wal.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;

namespace {

constexpr long kBig = 1'000'000'000;

void base_store(const std::string& root) {
  Store s(root, kOfhe);
  s.create_collection(spec("docs"));
  s.create_collection(spec("other"));
  s.insert("docs", 0, kA, cols2(1));
  s.insert("docs", 1, kB, cols2(5));
  s.insert("other", 0, {"vec-other-1"}, cols2(11));
  s.put_blob("docs", 1, "blob-rec-1", blob_bytes(1));
  s.put_blob("docs", 2, "blob-doc-1", blob_bytes(2));
  s.put_key_files("docs", key_set(20));
}

struct Op {
  const char* name;
  std::function<void(Store&)> prep;  // not injected
  std::function<void(Store&)> act;   // the operation under test (may be retried)
};

std::vector<std::string> block_files(Store& s, std::uint32_t cluster) {
  std::vector<std::string> v;
  for (auto& r : s.segments("docs"))
    if (r.cluster == cluster) v.push_back(r.file);
  return v;
}

void no_tmp_files(const std::string& root) {
  for (auto& e : fs::recursive_directory_iterator(root)) {
    const std::string n = e.path().filename().string();
    CHECK(n.size() < 4 || n.compare(n.size() - 4, 4, ".tmp") != 0);
  }
}

}  // namespace

int main() {
  const std::vector<Op> ops = {
      {"insert", [](Store&) {},
       [](Store& s) { s.insert("docs", 1, {"vec-new-1", "vec-new-2"}, cols2(40)); }},
      {"insert-center", [](Store&) {},
       [](Store& s) { s.insert("docs", 4, {"vec-center-1"}, cols2(41)); }},
      {"remove", [](Store&) {}, [](Store& s) { s.remove("docs", {"vec-alpha-1", "vec-beta-1"}); }},
      {"replace",
       [](Store& s) {
         s.insert("docs", 0, {"vec-alpha-4", "vec-alpha-5"}, cols2(3));
         s.remove("docs", {"vec-alpha-2", "vec-alpha-5"});
       },
       [](Store& s) {
         auto old = block_files(s, 0);
         s.replace_segments(
             "docs", old, SegmentData{0, {"vec-alpha-1", "vec-alpha-3", "vec-alpha-4"}, cols2(9)});
       }},
      {"drop_dead_blocks", [](Store& s) { s.remove("docs", {"vec-beta-1", "vec-beta-2"}); },
       [](Store& s) { s.drop_dead_blocks("docs"); }},
      {"put_blob_new", [](Store&) {},
       [](Store& s) { s.put_blob("docs", 1, "blob-rec-new", blob_bytes(7)); }},
      {"put_blob_replace", [](Store&) {},
       [](Store& s) { s.put_blob("docs", 1, "blob-rec-1", blob_bytes(8)); }},
      {"delete_blobs", [](Store&) {}, [](Store& s) { s.delete_blobs("docs", 2, {"blob-doc-1"}); }},
      {"put_key_files", [](Store&) {}, [](Store& s) { s.put_key_files("docs", key_set(50)); }},
      {"put_key_file", [](Store&) {},
       [](Store& s) { s.put_key_file("docs", kKeyContext, key_envelope(true, 60)); }},
      {"create_collection", [](Store&) {}, [](Store& s) { s.create_collection(spec("third")); }},
      {"drop_collection", [](Store&) {}, [](Store& s) { s.drop_collection("docs"); }},
  };

  for (const auto& op : ops) {
    // The state after a clean run, and the number of instrumented calls the operation makes.
    std::string after;
    long n_calls;
    {
      TempDir d;
      base_store(d.path);
      Store s(d.path, kOfhe);
      op.prep(s);
      set_fs_fault(kBig, ENOSPC);
      op.act(s);
      n_calls = kBig - fs_fault_remaining();
      clear_fs_fault();
      after = dump_state(s, false);
    }
    CHECK(n_calls >= 1);
    int threw = 0, applied_anyway = 0;

    for (long k = 1; k <= n_calls; ++k) {
      TempDir d;
      base_store(d.path);
      std::string before, mid;
      bool did_throw = false;
      {
        Store s(d.path, kOfhe);
        op.prep(s);
        before = dump_state(s, false);
        set_fs_fault(k, ENOSPC);
        try {
          op.act(s);
        } catch (const DiskFullError&) {
          did_throw = true;  // the clear error the service maps to RESOURCE_EXHAUSTED
        } catch (const std::exception& e) {
          std::fprintf(stderr, "%s k=%ld: wrong exception type: %s\n", op.name, k, e.what());
          CHECK(false);
        }
        clear_fs_fault();
        CHECK(fs_fault_remaining() == 0);
        // The live store is exactly the state before or after, never in between.
        mid = dump_state(s, false);
        if (!did_throw) {
          // Only calls whose failure is harmless are swallowed (directory sync after a commit,
          // the log's commit record): the operation is then applied in full.
          CHECK(mid == after);
          ++applied_anyway;
        } else {
          ++threw;
          if (mid != before && mid != after) {
            std::fprintf(stderr, "%s k=%ld: state is neither before nor after\n", op.name, k);
            CHECK(false);
          }
          if (mid == before) {
            op.act(s);  // the same call succeeds once the fault is gone
            CHECK(dump_state(s, false) == after);
          }
        }
      }
      // Next open: recovery sweeps what the failed call left; nothing temporary or unknown stays.
      {
        Store s(d.path, kOfhe);
        CHECK(dump_state(s, false) == after);
      }
      no_tmp_files(d.path);
      VerifyReport r;
      verify_data_dir(d.path, r);
      for (auto& f : r.findings)
        std::fprintf(stderr, "%s k=%ld: finding %s %s: %s\n", op.name, k, f.kind.c_str(),
                     f.file.c_str(), f.what.c_str());
      CHECK(r.findings.empty());
    }
    std::fprintf(stderr, "[diskfull] %-18s %2ld injection points: %d threw, %d applied anyway\n",
                 op.name, n_calls, threw, applied_anyway);
    CHECK(threw >= 1);
  }

  {  // The write-ahead log: a partial record is neither acknowledged nor left in the file.
    for (long k : {1L, 2L}) {  // 1: the write (half the frame lands), 2: the fsync
      TempDir d;
      const std::string path = d.path + "/wal.log";
      Wal w(path);
      WalRecord r1;
      r1.type = WalType::Delete;
      r1.collection = "docs";
      r1.ids = {"vec-alpha-1"};
      w.append(r1);
      const auto size1 = fs::file_size(path);
      WalRecord r2 = r1;
      r2.ids = {"vec-alpha-2", "vec-alpha-3"};
      set_fs_fault(k, ENOSPC);
      bool threw = false;
      try {
        w.append(r2);
      } catch (const DiskFullError&) {
        threw = true;
      }
      clear_fs_fault();
      CHECK(threw);
      CHECK(fs::file_size(path) == size1);  // cut back to the last acknowledged record
      WalRecord r3 = r1;
      r3.ids = {"vec-alpha-9"};
      w.append(r3);  // later appends work and are not preceded by a partial frame
      std::size_t dropped = 0;
      auto recs = w.recover(&dropped);
      CHECK(dropped == 0 && recs.size() == 2);
      CHECK(recs[0].ids == r1.ids && recs[1].ids == r3.ids);
    }
    {  // A frame left half-written (as after a crash) is dropped by recovery, not applied.
      TempDir d;
      const std::string path = d.path + "/wal.log";
      {
        Wal w(path);
        WalRecord r1;
        r1.type = WalType::Delete;
        r1.collection = "docs";
        r1.ids = {"vec-alpha-1"};
        w.append(r1);
      }
      const auto size1 = fs::file_size(path);
      {
        Wal w(path);
        WalRecord r2;
        r2.type = WalType::Delete;
        r2.collection = "docs";
        r2.ids = {"vec-alpha-2"};
        w.append(r2);
      }
      fs::resize_file(path, size1 + 5);  // the disk filled up after 5 bytes of the second frame
      Wal w(path);
      std::size_t dropped = 0;
      auto recs = w.recover(&dropped);
      CHECK(recs.size() == 1 && dropped == 5 && fs::file_size(path) == size1);
    }
  }

  {  // ENOSPC from fsutil directly: DiskFullError, with the temp file left for the caller to sweep.
    TempDir d;
    set_fs_fault(1, ENOSPC);
    CHECK_THROWS(write_file_durable(d.path + "/x", {1, 2, 3}));
    clear_fs_fault();
    set_fs_fault(1, EIO);  // another error is not a DiskFullError
    bool other = false;
    try {
      write_file_durable(d.path + "/y", {1, 2, 3});
    } catch (const DiskFullError&) {
    } catch (const std::runtime_error&) {
      other = true;
    }
    clear_fs_fault();
    CHECK(other);
    write_file_durable(d.path + "/z", {1, 2, 3});  // off again: works
  }
  std::fprintf(stderr, "diskfull_test: ok\n");
  return 0;
}
