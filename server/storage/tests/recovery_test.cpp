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

static std::vector<std::vector<std::uint8_t>> cols(std::uint8_t f) {
  return {column(f), column(f + 1)};
}

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
static std::set<std::string> check_consistent(const std::string& root,
                                              std::set<std::string>* tombs) {
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
    CHECK(has_all(have, kIdsA));                     // acknowledged earlier write survives
    if (!expect_crash) CHECK(has_all(have, kIdsB));  // acknowledged now
    if (expect_present)
      CHECK(has_all(have, kIdsB));
    else
      CHECK(has_none(have, kIdsB));  // all-or-nothing
  }
  // The recovered store accepts the same ids again iff the insert was absent.
  Store s(d.path, kOfhe);
  if (expect_present)
    CHECK_THROWS(s.insert("docs", 1, kIdsB, cols(5)));
  else
    s.insert("docs", 1, kIdsB, cols(5));
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
    if (expect_deleted)
      CHECK(t == std::set<std::string>({"a1", "b2"}));
    else
      CHECK(t.empty());
    if (!expect_crash) CHECK(!t.empty());
  }
}

static std::string segment_of(const std::string& root, std::uint32_t cluster) {
  Store s(root, kOfhe);
  for (auto& r : s.catalog().segments("docs"))
    if (r.cluster == cluster) return r.file;
  CHECK(false);
  return {};
}

// drop_dead_blocks with a crash at p: block B is fully deleted, block A is partly deleted.
static void drop_case(CrashPoint p, bool expect_dropped, bool expect_crash) {
  TempDir d;
  setup(d.path, true);
  {
    Store s(d.path, kOfhe);
    s.remove("docs", {"b1", "b2", "a1"});
  }
  int code = run_child(d.path, p, [](Store& s) { CHECK(s.drop_dead_blocks("docs") == 1); });
  CHECK(code == (expect_crash ? kCrashExitCode : 0));
  for (int round = 0; round < 2; ++round) {
    std::set<std::string> t;
    auto have = check_consistent(d.path, &t);
    CHECK(has_all(have, {"a1", "a2", "a3"}));  // the partly deleted block is never touched
    if (expect_dropped)
      CHECK(has_none(have, kIdsB) && t == std::set<std::string>({"a1"}));
    else
      CHECK(has_all(have, kIdsB) && t == std::set<std::string>({"a1", "b1", "b2"}));
  }
  Store s(d.path, kOfhe);  // idempotent: a second run finds the work done or does it now
  s.drop_dead_blocks("docs");
  CHECK(has_none(check_consistent(d.path, nullptr), kIdsB));
}

static std::vector<std::string> segments_of(const std::string& root, std::uint32_t cluster) {
  Store s(root, kOfhe);
  std::vector<std::string> v;
  for (auto& r : s.catalog().segments("docs"))
    if (r.cluster == cluster) v.push_back(r.file);
  return v;
}

// replace_segments (2 -> 1) with a crash at p: cluster 0 has block A {a1,a2,a3} and block C
// {a4,a5}; a2 and a5 are deleted; the client merges the live ids into one block {a3, a4, a1}.
static void replace_case(CrashPoint p, bool expect_new, bool expect_crash) {
  TempDir d;
  setup(d.path, true);
  {
    Store s(d.path, kOfhe);
    s.insert("docs", 0, {"a4", "a5"}, cols(3));
    s.remove("docs", {"a2", "a5"});
  }
  const auto old_files = segments_of(d.path, 0);
  CHECK(old_files.size() == 2);
  int code = run_child(d.path, p, [&](Store& s) {
    s.replace_segments("docs", old_files, SegmentData{0, {"a3", "a4", "a1"}, cols(9)});
  });
  CHECK(code == (expect_crash ? kCrashExitCode : 0));
  for (int round = 0; round < 2; ++round) {
    std::set<std::string> t;
    auto have = check_consistent(d.path, &t);
    CHECK(has_all(have, {"a1", "a3", "a4"}) && has_all(have, kIdsB));  // no live id lost
    if (expect_new)
      CHECK(!have.count("a2") && !have.count("a5") && t.empty());
    else
      CHECK(have.count("a2") && have.count("a5") && t == std::set<std::string>({"a2", "a5"}));
    Store s(d.path, kOfhe);
    const auto now = segments_of(d.path, 0);
    CHECK(now.size() == (expect_new ? 1u : 2u));
    CHECK((now == old_files) == !expect_new);
    std::size_t n = 0;
    for (const auto& f : now) n += s.read_segment("docs", f).ids.size();
    CHECK(n == (expect_new ? 3u : 5u));
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

  std::fprintf(stderr, "[test] dead-block drop matrix\n");
  drop_case(CrashPoint::None, true, false);
  drop_case(CrashPoint::CompactBeforeCatalog, false, true);
  drop_case(CrashPoint::CompactAfterCatalog, true, true);
  drop_case(CrashPoint::CompactAfterUnlink, true, true);

  std::fprintf(stderr, "[test] replace matrix\n");
  replace_case(CrashPoint::None, true, false);
  replace_case(CrashPoint::ReplaceBeforeNewFile, false, true);
  replace_case(CrashPoint::ReplaceAfterNewFile, false, true);
  replace_case(CrashPoint::ReplaceAfterCatalog, true, true);
  replace_case(CrashPoint::ReplaceAfterFirstUnlink, true, true);
  replace_case(CrashPoint::ReplaceAfterUnlink, true, true);

  {  // Stats, refusals, pins.
    TempDir d;
    setup(d.path, true);
    Store s(d.path, kOfhe);
    s.insert("docs", 4, {"ctr0"}, cols(1));  // a center block
    s.remove("docs", {"a2", "b1", "b2"});
    auto bs = s.block_stats("docs");
    CHECK(bs.size() == 3);
    for (auto& b : bs) {
      if (b.cluster == 0)
        CHECK(b.n_vectors == 3 && b.n_deleted == 1);
      else if (b.cluster == 1)
        CHECK(b.n_vectors == 2 && b.n_deleted == 2);
      else
        CHECK(b.cluster == 4 && b.n_vectors == 1 && b.n_deleted == 0);
    }
    const std::string a = segment_of(d.path, 0), b = segment_of(d.path, 1),
                      c = segment_of(d.path, 4);
    // Refusals, each leaving the block as it was.
    CHECK_THROWS(
        s.replace_segments("docs", {"s_nope.oseg"}, SegmentData{0, {"a1", "a3"}, cols(9)}));
    CHECK_THROWS(s.replace_segments("docs", {a},
                                    SegmentData{0, {"a1", "a2", "a3"}, cols(9)}));   // dead id kept
    CHECK_THROWS(s.replace_segments("docs", {a}, SegmentData{0, {"a1"}, cols(9)}));  // live id lost
    CHECK_THROWS(s.replace_segments("docs", {a},
                                    SegmentData{0, {"a1", "a3", "zz"}, cols(9)}));  // foreign id
    CHECK_THROWS(
        s.replace_segments("docs", {a}, SegmentData{0, {"a1", "a1"}, cols(9)}));  // duplicate
    CHECK_THROWS(
        s.replace_segments("docs", {a}, SegmentData{1, {"a1", "a3"}, cols(9)}));  // cluster
    CHECK_THROWS(
        s.replace_segments("docs", {a}, SegmentData{0, {"a1", "a3"}, {column(9)}}));  // columns
    CHECK_THROWS(s.replace_segments(
        "docs", {a}, SegmentData{0, {"a1", "a3"}, {column(9, 8, "m0_depth2"), column(9)}}));
    CHECK_THROWS(
        s.replace_segments("docs", {c}, SegmentData{4, {"ctr0"}, cols(2)}));  // center block
    CHECK_THROWS(
        s.replace_segments("docs", {a, b}, SegmentData{0, {"a1", "a3"}, cols(9)}));  // two clusters
    CHECK_THROWS(
        s.replace_segments("docs", {a, a}, SegmentData{0, {"a1", "a3"}, cols(9)}));  // listed twice
    CHECK_THROWS(s.replace_segments("docs", {}, SegmentData{0, {"a1", "a3"}, cols(9)}));  // none
    CHECK_THROWS(s.replace_segments("docs", {b}, SegmentData{1, {}, cols(9)}));  // nothing live
    CHECK(s.block_stats("docs").size() == 3 && fs::exists(s.segment_path(a)));
    bool typed = false;
    try {
      s.replace_segments("docs", {"s_nope.oseg"}, SegmentData{0, {"a1"}, cols(9)});
    } catch (const BlockNotFound&) {
      typed = true;
    }
    CHECK(typed);
    try {
      s.replace_segments("docs", {a}, SegmentData{0, {"a1"}, cols(9)});
      CHECK(false);
    } catch (const BlockMismatch&) {
    }
    // A pinned snapshot keeps the files of a removed block until it is released.
    {
      Snapshot snap = s.snapshot("docs");
      CHECK(snap.segments.size() == 3 && snap.tombstones.size() == 3);
      CHECK(s.drop_dead_blocks("docs") == 1);
      CHECK(fs::exists(s.segment_path(b)));  // deferred
      const std::string a2 = s.replace_segments("docs", {a}, SegmentData{0, {"a3", "a1"}, cols(9)});
      CHECK(fs::exists(s.segment_path(a)) && fs::exists(s.segment_path(a2)));
      CHECK(s.read_segment("docs", a).ids.size() == 3);  // old block still readable
    }
    CHECK(!fs::exists(s.segment_path(b)) && !fs::exists(s.segment_path(a)));
    auto after = s.block_stats("docs");
    CHECK(after.size() == 2);
    for (auto& x : after) CHECK(x.n_deleted == 0);
  }
  {  // Key files are stored all or nothing.
    TempDir d;
    Store s(d.path, kOfhe);
    s.create_collection(spec());
    using ombra::core::Envelope;
    using ombra::core::EnvelopeKind;
    Envelope ctx{EnvelopeKind::Context, kPreset, kOfhe, {1}};
    Envelope key{EnvelopeKind::PublicEvalKey, kPreset, kOfhe, {2}};
    Envelope bad{EnvelopeKind::PublicEvalKey, "m0_depth2", kOfhe, {3}};
    auto w = [](const Envelope& e) { return ombra::core::write_envelope(e); };
    CHECK_THROWS(s.put_key_files(
        "docs", {{kKeyContext, w(ctx)}, {kKeyEvalMult, w(key)}, {kKeyAutomorphism, w(bad)}}));
    CHECK(s.catalog().key_files("docs").empty());
    CHECK(fs::is_empty(d.path + "/keys"));
    s.put_key_files("docs",
                    {{kKeyContext, w(ctx)}, {kKeyEvalMult, w(key)}, {kKeyAutomorphism, w(key)}});
    CHECK(s.catalog().key_files("docs").size() == 3);
    s.put_key_files("docs",
                    {{kKeyContext, w(ctx)}, {kKeyEvalMult, w(key)}, {kKeyAutomorphism, w(key)}});
    CHECK(s.catalog().key_files("docs").size() == 3);
    std::size_t n = 0;
    for (auto& e : fs::directory_iterator(d.path + "/keys")) {
      (void)e;
      ++n;
    }
    CHECK(n == 3);  // the replaced files are gone
  }
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
    s.put_key_file("docs", kKeyEvalMult, ombra::core::write_envelope(e));
    s.put_key_file("docs", kKeyAutomorphism, ombra::core::write_envelope(e));
    CHECK(s.catalog().key_files("docs").size() == 2);
    std::vector<std::uint8_t> back;
    CHECK(s.get_key_file("docs", kKeyAutomorphism, &back) &&
          back == ombra::core::write_envelope(e));
    CHECK(!s.get_key_file("docs", kKeyContext, &back));
    CHECK_THROWS(s.put_key_file("docs", "eval_key", ombra::core::write_envelope(e)));  // old name
    CHECK_THROWS(
        s.put_key_file("docs", kKeyContext, ombra::core::write_envelope(e)));  // wrong kind
    e.preset = "m0_depth2";
    CHECK_THROWS(s.put_key_file("docs", kKeyEvalMult, ombra::core::write_envelope(e)));
    CHECK_THROWS(s.insert("docs", 0, {"x"}, {column(1, 8, "m0_depth2")}));  // wrong preset column
  }
  {  // The id cluster_count holds the center blocks; one above it is out of range.
    TempDir d;
    Store s(d.path, kOfhe);
    s.create_collection(spec());  // cluster_count = 4
    s.insert("docs", 4, {"ctr0"}, cols(1));
    CHECK_THROWS(s.insert("docs", 5, {"ctr1"}, cols(1)));
    auto rows = s.catalog().segments("docs");
    CHECK(rows.size() == 1 && rows[0].cluster == 4);
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
