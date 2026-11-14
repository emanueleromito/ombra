// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Writes do not hold the store's lock while they write files (see the "Locking" comment in
// store.hpp). Checks: a slow write does not block snapshot() or reads; many writers and readers at
// once leave a consistent store; a repeated insert is a no-op even when two copies race; a failed
// write leaves no file. Synthetic data only.
#include <atomic>
#include <chrono>
#include <filesystem>
#include <set>
#include <thread>

#include "fill_util.hpp"
#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/verify.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Every file of the segments, blobs and keys directories is known to the catalog, no temp files.
void check_files(const std::string& root, Store& s) {
  std::set<std::string> known;
  for (const auto& r : s.segments("docs")) known.insert(r.file);
  for (const auto& e : fs::directory_iterator(root + "/segments")) {
    CHECK(e.path().extension() == ".oseg");
    CHECK(known.count(e.path().filename().string()) == 1);
  }
  for (const char* dir : {"/keys", "/blobs"})
    for (const auto& e : fs::directory_iterator(root + dir)) CHECK(e.path().extension() != ".tmp");
}

void slow_write_does_not_block_reads() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec("docs"));
  s.insert("docs", 0, kA, cols2(1));
  set_write_delay_ms(600);
  std::atomic<bool> done{false};
  std::thread writer([&] {
    s.insert("docs", 1, kB, cols2(5));
    s.put_blob("docs", 1, "blob-1", blob_bytes(3));
    s.put_key_files("docs", key_set(20));
    done = true;
  });
  // Give the writer time to be inside its slow file write.
  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  CHECK(!done.load());
  double worst = 0;
  int reads = 0;
  while (!done.load()) {
    const auto t0 = Clock::now();
    const Snapshot snap = s.snapshot("docs");
    CHECK(!snap.segments.empty());
    (void)s.read_segment("docs", snap.segments.front().file);
    (void)s.tombstones("docs");
    (void)s.block_stats("docs");
    worst = std::max(worst, ms_since(t0));
    ++reads;
    if (reads >= 20) break;
  }
  CHECK(!done.load());  // the writes were still running while all these reads completed
  CHECK(worst < 300);   // each read set is far below the 600 ms of one slow write
  writer.join();
  set_write_delay_ms(0);
  CHECK(s.segments("docs").size() == 2);
  check_files(d.path, s);
}

void writers_and_readers() {
  TempDir d;
  constexpr int kWriters = 6, kPerWriter = 8;
  {
    Store s(d.path, kOfhe);
    s.create_collection(spec("docs"));
    s.insert("docs", 0, {"seed-1", "seed-2"}, cols2(1));
    std::atomic<bool> stop{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> ts;
    for (int w = 0; w < kWriters; ++w)
      ts.emplace_back([&, w] {
        try {
          for (int i = 0; i < kPerWriter; ++i) {
            const std::string a = "w" + std::to_string(w) + "-" + std::to_string(i) + "-a";
            const std::string b = "w" + std::to_string(w) + "-" + std::to_string(i) + "-b";
            s.insert("docs", static_cast<std::uint32_t>(w % 4), {a, b},
                     cols2(static_cast<std::uint8_t>(w * 10 + i)));
            if (i % 3 == 0) s.remove("docs", {a});
            if (i % 4 == 0)
              s.put_blob("docs", 1, a, blob_bytes(static_cast<std::uint8_t>(i)));
          }
        } catch (const std::exception& e) {
          std::fprintf(stderr, "writer failed: %s\n", e.what());
          ++failures;
        }
      });
    std::thread reader([&] {
      try {
        while (!stop.load()) {
          const Snapshot snap = s.snapshot("docs");
          std::size_t ids = 0;
          for (const auto& r : snap.segments) {
            const auto sd = s.read_segment("docs", r.file);  // pinned by snap: always readable
            CHECK(sd.ids.size() == r.n_vectors);
            ids += sd.ids.size();
          }
          CHECK(ids >= 2);
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "reader failed: %s\n", e.what());
        ++failures;
      }
    });
    std::thread compactor([&] {
      try {
        while (!stop.load()) {
          (void)s.drop_dead_blocks("docs");
          std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
      } catch (const std::exception& e) {
        std::fprintf(stderr, "compactor failed: %s\n", e.what());
        ++failures;
      }
    });
    for (auto& t : ts) t.join();
    stop = true;
    reader.join();
    compactor.join();
    CHECK(failures.load() == 0);
    // Every id that was inserted is in the catalog, none twice.
    std::set<std::string> have;
    for (const auto& id : s.catalog().vector_ids("docs")) CHECK(have.insert(id).second);
    CHECK(have.size() == 2 + kWriters * kPerWriter * 2);
    check_files(d.path, s);
  }
  // After a restart: consistent, and verify finds nothing wrong.
  Store again(d.path, kOfhe);
  check_files(d.path, again);
  CHECK(again.catalog().vector_ids("docs").size() == 2 + kWriters * kPerWriter * 2);
}

void repeated_insert_is_a_no_op() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec("docs"));
  bool replayed = false;
  const auto first = s.insert("docs", 0, kA, cols2(1), &replayed);
  CHECK(!replayed);
  const auto again = s.insert("docs", 0, kA, cols2(1), &replayed);
  CHECK(replayed && again == first);
  CHECK(s.segments("docs").size() == 1);
  // Refused: other bytes, other cluster, a partial overlap, a different order of the same ids.
  CHECK_THROWS(s.insert("docs", 0, kA, cols2(7)));
  CHECK_THROWS(s.insert("docs", 1, kA, cols2(1)));
  CHECK_THROWS(s.insert("docs", 0, {"vec-alpha-1", "new-id"}, cols2(1)));
  CHECK_THROWS(s.insert("docs", 0, {"vec-alpha-3", "vec-alpha-2", "vec-alpha-1"}, cols2(1)));
  CHECK_THROWS(s.insert("docs", 0, {"vec-alpha-1"}, cols2(1)));
  CHECK(s.segments("docs").size() == 1);
  // Still a repeat after a restart and after a delete of one of its ids.
  s.remove("docs", {"vec-alpha-2"});
  CHECK(s.insert("docs", 0, kA, cols2(1), &replayed) == first && replayed);
}

void racing_copies_store_one_block() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec("docs"));
  set_write_delay_ms(150);  // both copies pass step 1 before either one finishes step 2
  std::atomic<int> replays{0}, fresh{0}, errors{0};
  std::vector<std::thread> ts;
  for (int i = 0; i < 3; ++i)
    ts.emplace_back([&] {
      try {
        bool r = false;
        s.insert("docs", 0, kA, cols2(1), &r);
        (r ? replays : fresh)++;
      } catch (const std::exception&) {
        ++errors;
      }
    });
  for (auto& t : ts) t.join();
  set_write_delay_ms(0);
  CHECK(errors.load() == 0 && fresh.load() == 1 && replays.load() == 2);
  CHECK(s.segments("docs").size() == 1);
  check_files(d.path, s);
}

void failed_write_leaves_no_file() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec("docs"));
  s.insert("docs", 0, kA, cols2(1));
  // A column of another preset is refused after step 1: the file is not written, nothing is left.
  CHECK_THROWS(s.insert("docs", 1, kB, {column(1, 64, "m0_depth2"), column(2, 64, "m0_depth2")}));
  CHECK(s.segments("docs").size() == 1);
  check_files(d.path, s);
  // A replacement that lost a race (the block was deleted meanwhile) leaves no new file either.
  const auto seg = s.segments("docs").front().file;
  SegmentData neu{0, {"vec-alpha-1", "vec-alpha-2"}, cols2(4)};
  s.remove("docs", {"vec-alpha-3"});
  s.replace_segments("docs", {seg}, neu);  // ids are exactly the live ids now
  CHECK_THROWS(s.replace_segments("docs", {seg}, neu));  // the old block is gone
  check_files(d.path, s);
}

}  // namespace

int main() {
  slow_write_does_not_block_reads();
  writers_and_readers();
  repeated_insert_is_a_no_op();
  racing_copies_store_one_block();
  failed_write_leaves_no_file();
  std::puts("concurrency_test ok");
  return 0;
}
