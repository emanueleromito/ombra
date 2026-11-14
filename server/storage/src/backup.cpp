// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/storage/backup.hpp"

#include <fcntl.h>
#include <sqlite3.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>
#include <vector>

#include "ombra/storage/fsutil.hpp"
#include "ombra/storage/verify.hpp"

namespace fs = std::filesystem;

namespace ombra::storage {

namespace {

constexpr int kMaxAttempts = 5;

// A file the catalog copy names is gone from the source: the snapshot is stale.
class Vanished : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

struct Fd {
  int fd;
  explicit Fd(int f) : fd(f) {}
  Fd(Fd&& o) noexcept : fd(o.fd) { o.fd = -1; }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};

struct Db {
  sqlite3* db = nullptr;
  ~Db() {
    if (db) sqlite3_close(db);
  }
};

// Opens a source file for reading; a missing file means the snapshot is stale.
int open_source(const std::string& from) {
  int fd = ::open(from.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    if (errno == ENOENT) throw Vanished("source file vanished: " + from);
    throw_io_error("open", from);
  }
  return fd;
}

// Copies from an already open descriptor (which stays readable even if the file is unlinked).
std::uint64_t copy_fd(int in, const std::string& from, const std::string& to) {
  Fd out{::open(to.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)};
  if (out.fd < 0) throw_io_error("create", to);
  std::vector<std::uint8_t> buf(1 << 20);
  std::uint64_t total = 0;
  for (;;) {
    ssize_t r = ::read(in, buf.data(), buf.size());
    if (r < 0) {
      if (errno == EINTR) continue;
      throw_io_error("read", from);
    }
    if (r == 0) break;
    std::size_t off = 0;
    while (off < static_cast<std::size_t>(r)) {
      ssize_t w = ::write(out.fd, buf.data() + off, static_cast<std::size_t>(r) - off);
      if (w < 0) {
        if (errno == EINTR) continue;
        throw_io_error("write", to);
      }
      off += static_cast<std::size_t>(w);
    }
    total += static_cast<std::uint64_t>(r);
  }
  if (::fsync(out.fd) != 0) throw_io_error("fsync", to);
  return total;
}

std::uint64_t copy_file(const std::string& from, const std::string& to) {
  Fd in(open_source(from));
  return copy_fd(in.fd, from, to);
}

void make_private_dir(const std::string& p) {
  if (::mkdir(p.c_str(), 0700) != 0 && errno != EEXIST) throw_io_error("create directory", p);
}

void empty_dir(const std::string& p) {
  std::error_code ec;
  for (auto& e : fs::directory_iterator(p, ec)) fs::remove_all(e.path(), ec);
}

std::vector<std::string> one_column(sqlite3* db, const char* sql) {
  sqlite3_stmt* s = nullptr;
  if (sqlite3_prepare_v2(db, sql, -1, &s, nullptr) != SQLITE_OK)
    throw std::runtime_error(std::string("catalog: ") + sqlite3_errmsg(db));
  std::vector<std::string> v;
  while (sqlite3_step(s) == SQLITE_ROW)
    v.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(s, 0)));
  sqlite3_finalize(s);
  return v;
}

// One attempt: snapshot, then the files the snapshot names. Throws Vanished if the snapshot is
// stale.
void attempt(const std::string& root, const std::string& out, BackupReport& rep) {
  make_private_dir(out + "/segments");
  make_private_dir(out + "/keys");
  make_private_dir(out + "/blobs");
  rep.segments = rep.blobs = rep.key_files = 0;
  rep.bytes = 0;
  rep.api_keys = false;

  {  // 1. the catalog, as of one instant
    Db src, dst;
    src.db = open_catalog_readonly(root);
    const std::string dpath = out + "/catalog.db";
    if (sqlite3_open_v2(dpath.c_str(), &dst.db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                        nullptr) != SQLITE_OK)
      throw std::runtime_error("cannot create " + dpath);
    sqlite3_busy_timeout(src.db, 10000);
    sqlite3_backup* b = sqlite3_backup_init(dst.db, "main", src.db, "main");
    if (!b) throw std::runtime_error(std::string("catalog backup: ") + sqlite3_errmsg(dst.db));
    int rc;
    do {
      rc = sqlite3_backup_step(b, -1);  // all pages in one step: one read snapshot
      if (rc == SQLITE_BUSY || rc == SQLITE_LOCKED)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    } while (rc == SQLITE_BUSY || rc == SQLITE_LOCKED);
    sqlite3_backup_finish(b);
    if (rc != SQLITE_DONE)
      throw std::runtime_error(std::string("catalog backup failed: ") + sqlite3_errmsg(dst.db));
  }
  // 2. the files that copy names. Open every one of them first, right after the snapshot: this
  //    is the only window in which a concurrent removal can make the snapshot stale, and an open
  //    descriptor keeps a file readable after the server unlinks it. The copying then takes as long
  //    as it takes. (Batches if there are more files than descriptors.)
  struct Item {
    std::string dir, name;
  };
  std::vector<Item> items;
  {
    Db cat;
    cat.db = open_catalog_readonly(out);
    const std::pair<const char*, const char*> groups[] = {{"SELECT file FROM segments", "segments"},
                                                          {"SELECT file FROM blobs", "blobs"},
                                                          {"SELECT file FROM key_files", "keys"}};
    for (const auto& [sql, dir] : groups)
      for (auto& name : one_column(cat.db, sql)) {
        // Names come from a catalog copy that may be damaged: only plain file names are followed.
        if (name.empty() || name.find('/') != std::string::npos || name == "." || name == "..")
          throw std::runtime_error(std::string("catalog row with an invalid file name in ") + dir);
        items.push_back({dir, name});
      }
  }
  rlimit lim{};
  std::size_t batch = 256;
  if (::getrlimit(RLIMIT_NOFILE, &lim) == 0) {
    if (lim.rlim_cur < lim.rlim_max) {
      lim.rlim_cur = lim.rlim_max;
      ::setrlimit(RLIMIT_NOFILE, &lim);
      ::getrlimit(RLIMIT_NOFILE, &lim);
    }
    if (lim.rlim_cur > 128) batch = std::min<std::size_t>(lim.rlim_cur - 64, 4096);
  }
  bool first_batch = true;
  for (std::size_t i = 0; i < items.size(); i += batch) {
    const std::size_t end = std::min(items.size(), i + batch);
    std::vector<Fd> fds;
    fds.reserve(end - i);
    for (std::size_t j = i; j < end; ++j)
      fds.emplace_back(open_source(root + "/" + items[j].dir + "/" + items[j].name));
    if (first_batch) {
      fsync_path(out + "/catalog.db");
      first_batch = false;
    }
    for (std::size_t j = i; j < end; ++j) {
      const auto& it = items[j];
      rep.bytes += copy_fd(fds[j - i].fd, root + "/" + it.dir + "/" + it.name,
                           out + "/" + it.dir + "/" + it.name);
      if (it.dir == std::string("segments"))
        ++rep.segments;
      else if (it.dir == std::string("blobs"))
        ++rep.blobs;
      else
        ++rep.key_files;
    }
  }
  if (first_batch) fsync_path(out + "/catalog.db");
  for (const char* dir : {"segments", "blobs", "keys"}) fsync_path(out + "/" + dir);

  // 3. the API key file (atomically replaced by the server, so any read sees a whole file)
  // v1, or v2 (written once a collection-scoped key exists; the server reads v2 first).
  for (const char* name : {"api_keys.v1", "api_keys.v2"}) {
    if (fs::exists(root + "/" + name)) {
      rep.bytes += copy_file(root + "/" + name, out + "/" + name);
      rep.api_keys = true;
    }
  }
  fsync_path(out);
}

}  // namespace

void backup_data_dir(const std::string& root, const std::string& out, BackupReport& rep) {
  std::error_code ec;
  if (!fs::is_directory(root, ec)) throw std::runtime_error("not a readable directory: " + root);
  if (!fs::exists(root + "/catalog.db"))
    throw std::runtime_error("not a data directory: catalog.db is missing");
  const fs::path rroot = fs::weakly_canonical(root, ec);
  fs::create_directories(out, ec);
  if (ec) throw std::runtime_error("cannot create the output directory: " + out);
  const fs::path rout = fs::weakly_canonical(out, ec);
  const std::string a = rroot.string() + "/", b = rout.string() + "/";
  if (b.compare(0, a.size(), a) == 0 || a.compare(0, b.size(), b) == 0)
    throw std::runtime_error("the output directory and the data directory must be separate");
  if (!fs::is_empty(out, ec)) throw std::runtime_error("the output directory is not empty: " + out);
  ::chmod(out.c_str(), 0700);

  try {
    for (rep.attempts = 1;; ++rep.attempts) {
      try {
        attempt(root, out, rep);
        break;
      } catch (const Vanished&) {
        if (rep.attempts >= kMaxAttempts)
          throw std::runtime_error(
              "the data directory changed under the backup " + std::to_string(kMaxAttempts) +
              " times in a row (blocks replaced or dropped); try again when it is quieter");
        empty_dir(out);
      }
    }
    VerifyReport vr;
    verify_data_dir(out, vr);
    if (!vr.clean()) {
      std::string m = "the copy does not verify (is the source damaged? run ombra_verify on it):";
      for (const auto& f : vr.findings)
        if (f.severity == Severity::Corrupt) m += " [" + f.kind + " " + f.file + "]";
      throw std::runtime_error(m);
    }
  } catch (...) {
    empty_dir(out);  // the directory was empty when we started
    throw;
  }
}

}  // namespace ombra::storage
