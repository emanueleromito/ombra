// Crash-recovery matrix: a forked child opens the store, sets a crash point and does one insert or
// delete; it _exit()s at the point. The parent reopens the store and checks consistency.
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <set>

#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/store.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;

static const std::vector<std::string> kIdsA{"a1", "a2", "a3"};
static const std::vector<std::string> kIdsB{"b1", "b2"};

static std::vector<std::vector<std::uint8_t>> cols(std::uint8_t f) { return {column(f), column(f + 1)}; }

// Runs fn in a child with the crash point set. Returns the child's exit code.
template <class F>
static int run_child(const std::string& root, CrashPoint p, F fn) {
  pid_t pid = ::fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    Store s(root, kOfhe);
    set_crash_point(p);
    fn(s);
    ::_exit(0);  // reached the end: the write was acknowledged
  }
  int st = 0;
  CHECK(::waitpid(pid, &st, 0) == pid);
  CHECK(WIFEXITED(st));
  return WEXITSTATUS(st);
}

// Collection-wide consistency: files == catalog, ids unique and equal to the segments' ids, every
// segment readable, tombstones only on known ids, no temp files, log is empty after recovery.
static std::set<std::string> check_consistent(const std::string& root, std::set<std::string>* tombs) {
  Store s(root, kOfhe);
  const Catalog& c = s.catalog();
  std::set<std::string> files;
  for (auto& e : fs::directory_iterator(s.segments_dir())) {
    CHECK(e.path().extension() == ".oseg");  // no .tmp leftovers
    files.insert(e.path().filename().string());
  }
  std::set<std::string> cat_files;
  std::vector<std::string> seg_ids;
  for (auto& row : c.segments("docs")) {
    cat_files.insert(row.file);
    auto sd = s.read_segment("docs", row.file);
    CHECK(sd.ids.size() == row.n_vectors && sd.column_envelopes.size() == row.n_columns);
    seg_ids.insert(seg_ids.end(), sd.ids.begin(), sd.ids.end());
  }
  CHECK(files == cat_files);
  auto vids = c.vector_ids("docs");
  CHECK(std::set<std::string>(vids.begin(), vids.end()).size() == vids.size());  // no duplicates
  std::sort(seg_ids.begin(), seg_ids.end());
  CHECK(seg_ids == vids);
  auto t = c.tombstones("docs");
  for (auto& id : t) CHECK(std::binary_search(vids.begin(), vids.end(), id));
  if (tombs) *tombs = std::set<std::string>(t.begin(), t.end());
  return std::set<std::string>(vids.begin(), vids.end());
}

static void setup(const std::string& root, bool with_b) {
  Store s(root, kOfhe);
  s.create_collection(spec());
  s.insert("docs", 0, kIdsA, cols(1));  // acknowledged
  if (with_b) s.insert("docs", 1, kIdsB, cols(5));
}

static bool has_all(const std::set<std::string>& have, const std::vector<std::string>& ids) {
  return std::all_of(ids.begin(), ids.end(), [&](auto& i) { return have.count(i) > 0; });
}
static bool has_none(const std::set<std::string>& have, const std::vector<std::string>& ids) {
  return std::none_of(ids.begin(), ids.end(), [&](auto& i) { return have.count(i) > 0; });
}

static void insert_case(CrashPoint p, bool expect_present, bool expect_crash) {
  TempDir d;
  setup(d.path, false);
  int code = run_child(d.path, p, [](Store& s) { s.insert("docs", 1, kIdsB, cols(5)); });
  CHECK(code == (expect_crash ? kCrashExitCode : 0));
  for (int round = 0; round < 2; ++round) {  // second open: recovery is idempotent
    auto have = check_consistent(d.path, nullptr);
    CHECK(has_all(have, kIdsA));  // acknowledged earlier write survives
    if (!expect_crash) CHECK(has_all(have, kIdsB));  // acknowledged now
    if (expect_present) CHECK(has_all(have, kIdsB));
    else CHECK(has_none(have, kIdsB));  // all-or-nothing
  }
  // The recovered store accepts the same ids again iff the insert was absent.
  Store s(d.path, kOfhe);
  if (expect_present) CHECK_THROWS(s.insert("docs", 1, kIdsB, cols(5)));
  else s.insert("docs", 1, kIdsB, cols(5));
}

static void delete_case(CrashPoint p, bool expect_deleted, bool expect_crash) {
  TempDir d;
  setup(d.path, true);
  int code = run_child(d.path, p, [](Store& s) { s.remove("docs", {"a1", "b2", "nonexistent"}); });
  CHECK(code == (expect_crash ? kCrashExitCode : 0));
  for (int round = 0; round < 2; ++round) {
    std::set<std::string> t;
    auto have = check_consistent(d.path, &t);
    CHECK(has_all(have, kIdsA) && has_all(have, kIdsB));  // deletes tombstone, never drop rows
    if (expect_deleted) CHECK(t == std::set<std::string>({"a1", "b2"}));
    else CHECK(t.empty());
    if (!expect_crash) CHECK(!t.empty());
  }
}

int main() {
  std::fprintf(stderr, "[test] insert matrix\n");
  insert_case(CrashPoint::None, true, false);
  insert_case(CrashPoint::InsertBeforeIntent, false, true);
  insert_case(CrashPoint::InsertAfterIntent, false, true);
  insert_case(CrashPoint::InsertBeforeSegmentRename, false, true);
  insert_case(CrashPoint::InsertAfterSegmentRename, false, true);
  insert_case(CrashPoint::InsertAfterCatalog, true, true);
  insert_case(CrashPoint::InsertAfterCommit, true, true);

  std::fprintf(stderr, "[test] delete matrix\n");
  delete_case(CrashPoint::None, true, false);
  delete_case(CrashPoint::DeleteBeforeRecord, false, true);
  delete_case(CrashPoint::DeleteAfterRecord, true, true);
  delete_case(CrashPoint::DeleteAfterCatalog, true, true);

  {  // Torn log record: crash after the intent, then tear the log tail. The insert is absent.
    TempDir d;
    setup(d.path, false);
    CHECK(run_child(d.path, CrashPoint::InsertAfterIntent,
                    [](Store& s) { s.insert("docs", 1, kIdsB, cols(5)); }) == kCrashExitCode);
    const std::string wal = d.path + "/wal.log";
    auto sz = fs::file_size(wal);
    fs::resize_file(wal, sz - 3);
    auto have = check_consistent(d.path, nullptr);
    CHECK(has_all(have, kIdsA) && has_none(have, kIdsB));
  }
  {  // Committed record whose catalog update is missing is applied on recovery (roll forward).
    TempDir d;
    setup(d.path, false);
    CHECK(run_child(d.path, CrashPoint::InsertAfterCommit,
                    [](Store& s) { s.insert("docs", 1, kIdsB, cols(5)); }) == kCrashExitCode);
    Store s(d.path, kOfhe);
    CHECK(s.catalog().id_exists("docs", "b1"));
  }
  {  // Key files: stored through the envelope check; a wrong preset is refused.
    TempDir d;
    Store s(d.path, kOfhe);
    s.create_collection(spec());
    ombra::core::Envelope e{ombra::core::EnvelopeKind::PublicEvalKey, kPreset, kOfhe, {1, 2, 3}};
    s.put_key_file("docs", "eval_key", ombra::core::write_envelope(e));
    CHECK(s.catalog().key_files("docs").size() == 1);
    e.preset = "m0_depth2";
    CHECK_THROWS(s.put_key_file("docs", "eval_key", ombra::core::write_envelope(e)));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {column(1, 8, "m0_depth2")}));  // wrong preset column
  }
  {  // A reader refuses a segment whose envelope version was changed on disk.
    TempDir d;
    setup(d.path, false);
    Store s(d.path, kOfhe);
    auto row = s.catalog().segments("docs").at(0);
    auto sd = s.read_segment("docs", row.file);
    sd.column_envelopes[0][4] = 9;
    write_segment(s.segment_path(row.file), sd);
    CHECK_THROWS(s.read_segment("docs", row.file));
  }
  return 0;
}
