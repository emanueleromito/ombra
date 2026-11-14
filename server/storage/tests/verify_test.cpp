// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_verify: a clean store verifies clean; one corrupted byte in each kind of file is reported
// for exactly that file; verify never writes (also next to an open store); exit codes and output
// of the CLI. Synthetic data only.
#include "ombra/storage/verify.hpp"

#include <sqlite3.h>
#include <sys/stat.h>
#include <sys/wait.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>

#include "fill_util.hpp"
#include "ombra/storage/fsutil.hpp"

using namespace ombra::storage;
using namespace ombra::storage::testing;
namespace fs = std::filesystem;

namespace {

std::vector<std::string> files_of(const std::string& root, const char* sub) {
  std::vector<std::string> v;
  for (auto& e : fs::directory_iterator(root + "/" + sub))
    v.push_back(e.path().filename().string());
  std::sort(v.begin(), v.end());
  return v;
}

std::string pick(const std::string& root, const char* sub, std::size_t i = 0) {
  auto v = files_of(root, sub);
  CHECK(v.size() > i);
  return v[i];
}

void flip_byte(const std::string& path, std::size_t offset) {
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  CHECK(f.good());
  f.seekg(static_cast<std::streamoff>(offset));
  char c;
  f.get(c);
  c = static_cast<char>(c ^ 0x40);
  f.seekp(static_cast<std::streamoff>(offset));
  f.put(c);
}

VerifyReport run(const std::string& root) {
  VerifyReport r;
  verify_data_dir(root, r);
  return r;
}

// The one corrupt finding must be of `kind` and name `file`; nothing else is corrupt.
void expect_only(const VerifyReport& r, const char* kind, const std::string& file) {
  CHECK(!r.clean());
  CHECK(r.corrupt_count() == 1);
  for (const auto& f : r.findings)
    if (f.severity == Severity::Corrupt) {
      if (f.kind != kind || f.file != file)
        std::fprintf(stderr, "got %s %s: %s, want %s %s\n", f.kind.c_str(), f.file.c_str(),
                     f.what.c_str(), kind, file.c_str());
      CHECK(f.kind == kind && f.file == file);
    }
}

// path -> (size, mtime, content) of every file except SQLite's shared-memory file, which readers
// legitimately update.
std::map<std::string, std::string> snapshot(const std::string& root) {
  std::map<std::string, std::string> m;
  for (auto& e : fs::recursive_directory_iterator(root)) {
    const std::string name = e.path().string();
    if (name.size() >= 4 && name.compare(name.size() - 4, 4, "-shm") == 0) continue;
    std::string v = e.is_directory() ? "dir" : "";
    if (e.is_regular_file()) {
      std::ifstream in(name, std::ios::binary);
      v = std::string(std::istreambuf_iterator<char>(in), {}) +
          "|mtime=" + std::to_string(fs::last_write_time(e.path()).time_since_epoch().count());
    }
    m[name] = v;
  }
  return m;
}

std::string slurp(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

int run_cli(const std::string& args, const std::string& out_file) {
  const std::string cmd = std::string(OMBRA_VERIFY_BIN) + " " + args + " > " + out_file + " 2>&1";
  int st = std::system(cmd.c_str());
  CHECK(WIFEXITED(st));
  return WEXITSTATUS(st);
}

}  // namespace

int main() {
  {  // Clean store: no findings at all, counts match what fill_store made.
    TempDir d;
    fill_store(d.path);
    auto r = run(d.path);
    for (auto& f : r.findings)
      std::fprintf(stderr, "unexpected %s %s: %s\n", f.kind.c_str(), f.file.c_str(),
                   f.what.c_str());
    CHECK(r.findings.empty() && r.clean());
    CHECK(r.collections == 2 && r.blobs == 2 && r.key_files == 3);
    CHECK(r.segments == 4);  // docs: replaced block, center, beta-9; other: one
    CHECK(r.wal_records >= 3);
    // With the expected version it is clean, with another version every column is refused.
    VerifyReport r2;
    VerifyOptions o;
    o.expected_openfhe_version = kOfhe;
    verify_data_dir(d.path, r2, o);
    CHECK(r2.clean());
    VerifyReport r3;
    o.expected_openfhe_version = "9.9.9";
    verify_data_dir(d.path, r3, o);
    CHECK(!r3.clean());
  }

  {  // One flipped byte in each kind of file: exactly that file is reported.
    for (std::size_t i = 0; i < 4; ++i) {  // every segment, body byte
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "segments", i);
      flip_byte(d.path + "/segments/" + f, 40);
      expect_only(run(d.path), "segment", f);
    }
    {  // segment header: magic, then version
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "segments");
      flip_byte(d.path + "/segments/" + f, 0);
      expect_only(run(d.path), "segment", f);
    }
    {
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "segments");
      flip_byte(d.path + "/segments/" + f, 4);  // container version
      expect_only(run(d.path), "segment", f);
    }
    {  // segment CRC field itself
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "segments");
      flip_byte(d.path + "/segments/" + f, 7);
      expect_only(run(d.path), "segment", f);
    }
    for (std::size_t i = 0; i < 2; ++i) {  // blobs: payload byte, then header byte
      for (std::size_t off : {std::size_t{100}, std::size_t{5}}) {
        TempDir d;
        fill_store(d.path);
        const std::string f = pick(d.path, "blobs", i);
        flip_byte(d.path + "/blobs/" + f, off);
        expect_only(run(d.path), "blob", f);
      }
    }
    for (std::size_t i = 0; i < 3; ++i) {  // key files: magic, version, kind, preset name
      for (std::size_t off : {std::size_t{0}, std::size_t{4}, std::size_t{6}, std::size_t{12}}) {
        TempDir d;
        fill_store(d.path);
        const std::string f = pick(d.path, "keys", i);
        flip_byte(d.path + "/keys/" + f, off);
        expect_only(run(d.path), "key", f);
      }
    }
    {  // a truncated key file (its length field no longer matches)
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "keys");
      fs::resize_file(d.path + "/keys/" + f, fs::file_size(d.path + "/keys/" + f) - 1);
      expect_only(run(d.path), "key", f);
    }
    {  // a truncated segment
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, "segments");
      fs::resize_file(d.path + "/segments/" + f, fs::file_size(d.path + "/segments/" + f) - 1);
      expect_only(run(d.path), "segment", f);
    }
  }

  {  // Missing files are corrupt, stray files are warnings (recovery would sweep them).
    for (const char* sub : {"segments", "blobs", "keys"}) {
      TempDir d;
      fill_store(d.path);
      const std::string f = pick(d.path, sub);
      fs::remove(d.path + "/" + sub + "/" + f);
      auto r = run(d.path);
      expect_only(r,
                  std::string(sub) == "segments" ? "segment"
                  : std::string(sub) == "blobs"  ? "blob"
                                                 : "key",
                  f);
      CHECK(r.findings[0].what == "file missing");
    }
    TempDir d;
    fill_store(d.path);
    for (const char* sub : {"segments", "blobs", "keys"}) {
      std::ofstream(d.path + "/" + sub + "/stray.tmp") << "x";
      std::ofstream(d.path + "/" + sub + "/s0000000000000fff.oseg") << "x";
    }
    auto r = run(d.path);
    CHECK(r.clean());  // warnings only
    CHECK(r.warning_count() == 6);
    for (auto& f : r.findings) CHECK(f.kind == "orphan" && f.severity == Severity::Warning);
  }

  {    // Catalog damage.
    {  // not a database
      TempDir d;
      fill_store(d.path);
      flip_byte(d.path + "/catalog.db", 0);
      auto r = run(d.path);
      CHECK(!r.clean());
      for (auto& f : r.findings) CHECK(f.kind == "catalog" || f.kind == "orphan");
    }
    {  // cut in the middle: integrity problems
      TempDir d;
      fill_store(d.path);
      const auto size = fs::file_size(d.path + "/catalog.db");
      CHECK(size >= 8192);
      fs::resize_file(d.path + "/catalog.db", size / 2 / 4096 * 4096);
      auto r = run(d.path);
      CHECK(!r.clean());
      CHECK(r.findings.front().kind == "catalog");
    }
    {  // missing catalog
      TempDir d;
      fill_store(d.path);
      fs::remove(d.path + "/catalog.db");
      fs::remove(d.path + "/catalog.db-wal");
      auto r = run(d.path);
      CHECK(!r.clean() && r.findings.front().kind == "catalog");
    }
    // Row-level damage made with SQLite itself: the catalog is intact, the cross-references not.
    auto tamper = [](const std::string& root, const char* sql) {
      sqlite3* db = nullptr;
      CHECK(sqlite3_open((root + "/catalog.db").c_str(), &db) == SQLITE_OK);
      CHECK(sqlite3_exec(db, sql, nullptr, nullptr, nullptr) == SQLITE_OK);
      sqlite3_close(db);
    };
    {  // a block's vector count no longer matches its file
      TempDir d;
      fill_store(d.path);
      tamper(d.path,
             "UPDATE segments SET n_vectors = n_vectors + 1 WHERE file = (SELECT MIN(file) FROM "
             "segments)");
      auto r = run(d.path);
      CHECK(!r.clean());
      bool named = false;
      for (auto& f : r.findings) named |= f.kind == "segment" && f.file == pick(d.path, "segments");
      CHECK(named);
    }
    {  // a vectors row went missing
      TempDir d;
      fill_store(d.path);
      tamper(d.path,
             "DELETE FROM vectors WHERE rowid = (SELECT MIN(rowid) FROM vectors WHERE id NOT IN "
             "(SELECT id FROM tombstones))");
      auto r = run(d.path);
      CHECK(!r.clean());
      bool seg = false;
      for (auto& f : r.findings) seg |= f.kind == "segment";
      CHECK(seg);
    }
    {  // a catalog row was lost: the file is an orphan (warning), nothing else changes
      TempDir d;
      fill_store(d.path);
      tamper(d.path, "DELETE FROM key_files WHERE kind = 'automorphism'");
      auto r = run(d.path);
      CHECK(r.clean() && r.warning_count() == 1 && r.findings[0].kind == "orphan");
    }
    {  // a blob row points to a file that does not exist
      TempDir d;
      fill_store(d.path);
      tamper(d.path, "UPDATE blobs SET file = 'b0000000000000000.oblob' WHERE kind = 2");
      auto r = run(d.path);
      CHECK(!r.clean());
      bool blob = false;
      for (auto& f : r.findings) blob |= f.kind == "blob" && f.file == "b0000000000000000.oblob";
      CHECK(blob);
    }
  }

  {    // Write-ahead log.
    {  // torn tail: a warning, the file is not repaired
      TempDir d;
      fill_store(d.path);
      const auto size = fs::file_size(d.path + "/wal.log");
      std::ofstream(d.path + "/wal.log", std::ios::app | std::ios::binary).write("\x05\x00\x00", 3);
      const auto before = slurp(d.path + "/wal.log");
      auto r = run(d.path);
      CHECK(r.clean() && r.warning_count() == 1 && r.findings[0].kind == "wal");
      CHECK(slurp(d.path + "/wal.log") == before && fs::file_size(d.path + "/wal.log") == size + 3);
    }
    {  // a damaged first record with valid ones after it: the server would refuse to start
      TempDir d;
      fill_store(d.path);
      flip_byte(d.path + "/wal.log", 6 + 8 + 2);
      expect_only(run(d.path), "wal", "wal.log");
    }
    {  // bad magic and unknown version
      TempDir d;
      fill_store(d.path);
      flip_byte(d.path + "/wal.log", 0);
      expect_only(run(d.path), "wal", "wal.log");
      flip_byte(d.path + "/wal.log", 0);
      flip_byte(d.path + "/wal.log", 4);
      expect_only(run(d.path), "wal", "wal.log");
    }
    {  // a record with a path-like segment name
      TempDir d;
      fill_store(d.path);
      // Build one valid frame with a bad segment name.
      std::vector<std::uint8_t> body{2};  // InsertCommit
      auto put_str = [&](const std::string& s) {
        for (int i = 0; i < 4; ++i) body.push_back(static_cast<std::uint8_t>(s.size() >> (8 * i)));
        body.insert(body.end(), s.begin(), s.end());
      };
      put_str("docs");
      put_str("../x");
      std::vector<std::uint8_t> frame;
      for (int i = 0; i < 4; ++i)
        frame.push_back(static_cast<std::uint8_t>(body.size() >> (8 * i)));
      const auto crc = ombra::storage::crc32(body.data(), body.size());
      for (int i = 0; i < 4; ++i) frame.push_back(static_cast<std::uint8_t>(crc >> (8 * i)));
      frame.insert(frame.end(), body.begin(), body.end());
      std::ofstream(d.path + "/wal.log", std::ios::app | std::ios::binary)
          .write(reinterpret_cast<const char*>(frame.data()),
                 static_cast<std::streamsize>(frame.size()));
      expect_only(run(d.path), "wal", "wal.log");
    }
  }

  {  // Read-only: nothing changes, with the store closed (no -wal file) and while it is open.
    TempDir d;
    fill_store(d.path);
    {
      const auto before = snapshot(d.path);
      auto r = run(d.path);
      CHECK(r.clean());
      CHECK(snapshot(d.path) == before);
      CHECK(!fs::exists(d.path + "/catalog.db-wal"));  // verify did not create one
    }
    {  // next to a running "server": the store keeps working during and after verify
      Store live(d.path, kOfhe);
      const auto before = snapshot(d.path);
      auto r = run(d.path);
      CHECK(r.clean());
      CHECK(snapshot(d.path) == before);
      live.insert("docs", 1, {"vec-live-1"}, cols2(40));
      auto r2 = run(d.path);
      CHECK(r2.clean() && r2.vectors == r.vectors + 1);
      live.remove("docs", {"vec-live-1"});
    }
  }

  {  // CLI: exit codes and output (no ids, no collection names).
    TempDir d, out;
    fill_store(d.path);
    const std::string o = out.path + "/out.txt";
    CHECK(run_cli("--data-dir " + d.path, o) == 0);
    auto text = slurp(o);
    CHECK(text.find("result: clean") != std::string::npos);
    CHECK(text.find("vec-") == std::string::npos && text.find("docs") == std::string::npos &&
          text.find("blob-") == std::string::npos);
    CHECK(run_cli("--data-dir " + d.path + " --json", o) == 0);
    text = slurp(o);
    CHECK(text.find("\"clean\":true") != std::string::npos);
    CHECK(text.find("vec-") == std::string::npos);
    CHECK(run_cli("--data-dir " + d.path + " --openfhe-version " + kOfhe, o) == 0);
    CHECK(run_cli("--data-dir " + d.path + " --openfhe-version 9.9.9", o) == 1);

    const std::string f = pick(d.path, "segments");
    flip_byte(d.path + "/segments/" + f, 40);
    CHECK(run_cli("--data-dir " + d.path, o) == 1);
    text = slurp(o);
    CHECK(text.find(f) != std::string::npos && text.find("CORRUPT") != std::string::npos);
    CHECK(text.find("vec-") == std::string::npos);
    CHECK(run_cli("--data-dir " + d.path + " --json", o) == 1);
    CHECK(slurp(o).find("\"clean\":false") != std::string::npos);

    CHECK(run_cli("", o) == 2);
    CHECK(run_cli("--bogus", o) == 2);
    CHECK(run_cli("--data-dir", o) == 2);
    CHECK(run_cli("--data-dir " + d.path + "/does-not-exist", o) == 2);
    CHECK(run_cli("--help", o) == 0);
  }
  std::fprintf(stderr, "verify_test: ok\n");
  return 0;
}
