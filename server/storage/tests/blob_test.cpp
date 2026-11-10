// T3.4 blob store: put/get/replace/delete, refusals, drop with the collection, file corruption,
// and a crash matrix (forked child _exit()s at a crash point; the parent reopens the store).
#include <sys/wait.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <set>

#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/store.hpp"
#include "test_util.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;

namespace {

constexpr std::uint32_t kRec = kBlobKindRecord, kDoc = kBlobKindDocument;

std::vector<std::uint8_t> blob(std::uint8_t fill, std::size_t n = 16384) {
  return std::vector<std::uint8_t>(n, fill);
}

std::set<std::string> files_in(const std::string& dir) {
  std::set<std::string> s;
  for (auto& e : fs::directory_iterator(dir)) s.insert(e.path().filename().string());
  return s;
}

template <class F>
int run_child(const std::string& root, CrashPoint p, F fn) {
  pid_t pid = ::fork();
  CHECK(pid >= 0);
  if (pid == 0) {
    Store s(root, kOfhe);
    set_crash_point(p);
    fn(s);
    ::_exit(0);
  }
  int st = 0;
  CHECK(::waitpid(pid, &st, 0) == pid);
  CHECK(WIFEXITED(st));
  return WEXITSTATUS(st);
}

void basic() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec());
  std::vector<std::uint8_t> out;
  CHECK(!s.get_blob("docs", kRec, "x", &out));
  for (auto b : kBlobBuckets) {
    s.put_blob("docs", kRec, "id-" + std::to_string(b), blob(7, b));
  }
  CHECK(s.get_blob("docs", kRec, "id-16384", &out) && out == blob(7, 16384));
  CHECK(s.get_blob("docs", kRec, "id-16777216", &out) && out.size() == 16777216);
  // Same id, other kind: separate blob.
  CHECK(!s.get_blob("docs", kDoc, "id-16384", &out));
  s.put_blob("docs", kDoc, "id-16384", blob(9));
  CHECK(s.get_blob("docs", kDoc, "id-16384", &out) && out == blob(9));
  CHECK(s.get_blob("docs", kRec, "id-16384", &out) && out == blob(7));

  // Replace: new bytes (and another bucket), old file gone.
  const auto before = files_in(d.path + "/blobs").size();
  s.put_blob("docs", kRec, "id-16384", blob(8, 262144));
  CHECK(s.get_blob("docs", kRec, "id-16384", &out) && out == blob(8, 262144));
  CHECK(files_in(d.path + "/blobs").size() == before);

  // Delete: counts existing only.
  CHECK(s.delete_blobs("docs", kRec, {"id-16384", "missing"}) == 1);
  CHECK(!s.get_blob("docs", kRec, "id-16384", &out));
  CHECK(s.get_blob("docs", kDoc, "id-16384", &out));
  CHECK(s.delete_blobs("docs", kRec, {"id-16384"}) == 0);
  CHECK(files_in(d.path + "/blobs").size() == before - 1);
}

void refusals() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec());
  for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{16383}, std::size_t{16385},
                        std::size_t{300000}})
    CHECK_THROWS(s.put_blob("docs", kRec, "a", blob(1, n)));
  bool size_err = false, kind_err = false;
  try { s.put_blob("docs", kRec, "a", blob(1, 100)); } catch (const BlobSizeError&) { size_err = true; }
  try { s.put_blob("docs", 0, "a", blob(1)); } catch (const BlobKindError&) { kind_err = true; }
  CHECK(size_err && kind_err);
  CHECK_THROWS(s.put_blob("docs", 3, "a", blob(1)));
  CHECK_THROWS(s.put_blob("nope", kRec, "a", blob(1)));
  CHECK_THROWS(s.delete_blobs("docs", 0, {"a"}));
  std::vector<std::uint8_t> out;
  CHECK(!s.get_blob("docs", kRec, "a", &out));
  CHECK(files_in(d.path + "/blobs").empty());  // refusals leave nothing behind
}

void drop_and_restart() {
  TempDir d;
  {
    Store s(d.path, kOfhe);
    s.create_collection(spec());
    s.put_blob("docs", kRec, "a", blob(1));
    s.put_blob("docs", kDoc, "a", blob(2));
  }
  {
    Store s(d.path, kOfhe);  // survives a restart
    std::vector<std::uint8_t> out;
    CHECK(s.get_blob("docs", kRec, "a", &out) && out == blob(1));
    CHECK(s.drop_collection("docs"));
    CHECK(files_in(d.path + "/blobs").empty());
    s.create_collection(spec());  // same name again: no blob of the old one
    CHECK(!s.get_blob("docs", kRec, "a", &out));
  }
}

void corrupt_file_refused() {
  TempDir d;
  Store s(d.path, kOfhe);
  s.create_collection(spec());
  s.put_blob("docs", kRec, "a", blob(5));
  const auto f = *files_in(d.path + "/blobs").begin();
  const std::string path = d.path + "/blobs/" + f;
  CHECK(f.size() > 6 && f.substr(f.size() - 6) == ".oblob");
  std::vector<std::uint8_t> out;
  {  // flip a payload byte: CRC mismatch
    std::fstream fs_(path, std::ios::in | std::ios::out | std::ios::binary);
    fs_.seekp(100);
    fs_.put('\x55');
  }
  CHECK_THROWS(s.get_blob("docs", kRec, "a", &out));
  {  // unknown container version
    std::fstream fs_(path, std::ios::in | std::ios::out | std::ios::binary);
    fs_.seekp(4);
    fs_.put('\x09');
  }
  CHECK_THROWS(s.get_blob("docs", kRec, "a", &out));
}

// After a crash and restart: the blob table and the blob files agree, nothing partial is visible.
void check_state(const std::string& root, const std::string& id, std::uint32_t kind,
                 const std::vector<std::uint8_t>* expect) {
  Store s(root, kOfhe);
  std::vector<std::uint8_t> out;
  const bool have = s.get_blob("docs", kind, id, &out);
  if (expect) CHECK(have && out == *expect);
  else CHECK(!have);
  const auto cf = s.catalog().all_blob_files();
  const std::set<std::string> cat(cf.begin(), cf.end());
  CHECK(files_in(root + "/blobs") == cat);  // no orphan, no .tmp, no missing file
}

void crashes() {
  const auto old_b = blob(1), new_b = blob(2);
  // Replace of an existing blob.
  for (auto p : {CrashPoint::BlobBeforeRename, CrashPoint::BlobAfterRename, CrashPoint::BlobAfterCatalog}) {
    TempDir d;
    {
      Store s(d.path, kOfhe);
      s.create_collection(spec());
      s.put_blob("docs", kRec, "k", old_b);
    }
    CHECK(run_child(d.path, p, [&](Store& s) { s.put_blob("docs", kRec, "k", new_b); }) == kCrashExitCode);
    // Before the catalog commit the old blob stays, after it the new one is there.
    check_state(d.path, "k", kRec, p == CrashPoint::BlobAfterCatalog ? &new_b : &old_b);
  }
  // First put of a new blob: absent before the commit, present after.
  for (auto p : {CrashPoint::BlobBeforeRename, CrashPoint::BlobAfterRename, CrashPoint::BlobAfterCatalog}) {
    TempDir d;
    {
      Store s(d.path, kOfhe);
      s.create_collection(spec());
    }
    CHECK(run_child(d.path, p, [&](Store& s) { s.put_blob("docs", kRec, "k", new_b); }) == kCrashExitCode);
    check_state(d.path, "k", kRec, p == CrashPoint::BlobAfterCatalog ? &new_b : nullptr);
  }
  // Delete: the catalog commit decides; the file is swept.
  {
    TempDir d;
    {
      Store s(d.path, kOfhe);
      s.create_collection(spec());
      s.put_blob("docs", kRec, "k", old_b);
    }
    CHECK(run_child(d.path, CrashPoint::BlobDeleteAfterCatalog,
                    [&](Store& s) { s.delete_blobs("docs", kRec, {"k"}); }) == kCrashExitCode);
    check_state(d.path, "k", kRec, nullptr);
  }
  // Acknowledged put survives (no crash point hit).
  {
    TempDir d;
    {
      Store s(d.path, kOfhe);
      s.create_collection(spec());
    }
    CHECK(run_child(d.path, CrashPoint::None, [&](Store& s) { s.put_blob("docs", kRec, "k", new_b); }) == 0);
    check_state(d.path, "k", kRec, &new_b);
  }
}

}  // namespace

int main() {
  basic();
  refusals();
  drop_and_restart();
  corrupt_file_refused();
  crashes();
  std::puts("blob_test ok");
  return 0;
}
