#include "ombra/storage/wal.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include "ombra/storage/fsutil.hpp"

namespace ombra::storage {

namespace {
constexpr std::size_t kHeader = 6;
constexpr std::size_t kMaxRecord = 64u << 20;  // sanity bound before allocating

[[noreturn]] void fail(const std::string& what, const std::string& path) {
  throw std::runtime_error("wal: " + what + " '" + path + "': " + std::strerror(errno));
}

void put_str(std::vector<std::uint8_t>& o, const std::string& s) {
  put_le(o, s.size(), 4);
  o.insert(o.end(), s.begin(), s.end());
}

std::vector<std::uint8_t> encode_body(const WalRecord& r) {
  std::vector<std::uint8_t> b;
  b.push_back(static_cast<std::uint8_t>(r.type));
  put_str(b, r.collection);
  if (r.type == WalType::InsertIntent) put_le(b, r.cluster, 4);
  if (r.type != WalType::Delete) put_str(b, r.segment);
  if (r.type != WalType::InsertCommit) {
    put_le(b, r.ids.size(), 4);
    for (const auto& id : r.ids) put_str(b, id);
  }
  return b;
}

// Returns false on any malformed body (treated as a torn record).
bool decode_body(const std::uint8_t* p, std::size_t n, WalRecord* r) {
  std::size_t pos = 0;
  auto u32 = [&](std::uint32_t* v) {
    if (n - pos < 4) return false;
    *v = static_cast<std::uint32_t>(get_le(p + pos, 4));
    pos += 4;
    return true;
  };
  auto str = [&](std::string* s) {
    std::uint32_t len;
    if (!u32(&len) || n - pos < len) return false;
    s->assign(reinterpret_cast<const char*>(p + pos), len);
    pos += len;
    return true;
  };
  if (n < 1) return false;
  auto t = p[0];
  if (t < 1 || t > 3) return false;
  r->type = static_cast<WalType>(t);
  pos = 1;
  if (!str(&r->collection)) return false;
  if (r->type == WalType::InsertIntent && !u32(&r->cluster)) return false;
  if (r->type != WalType::Delete && !str(&r->segment)) return false;
  if (r->type != WalType::InsertCommit) {
    std::uint32_t cnt;
    if (!u32(&cnt) || cnt > n) return false;
    r->ids.resize(cnt);
    for (auto& id : r->ids)
      if (!str(&id)) return false;
  }
  return pos == n;
}

void write_all(int fd, const std::uint8_t* d, std::size_t n, const std::string& path) {
  while (n > 0) {
    ssize_t w = ::write(fd, d, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      fail("write", path);
    }
    d += w;
    n -= static_cast<std::size_t>(w);
  }
}
}  // namespace

Wal::Wal(std::string path) : path_(std::move(path)) {
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0644);
  if (fd_ < 0) fail("open", path_);
  struct stat st{};
  if (::fstat(fd_, &st) != 0) fail("stat", path_);
  if (static_cast<std::size_t>(st.st_size) < kHeader) {
    // New file, or a header torn during creation (no records can exist yet).
    if (::ftruncate(fd_, 0) != 0) fail("truncate", path_);
    std::vector<std::uint8_t> h{'O', 'W', 'A', 'L'};
    put_le(h, kWalVersion, 2);
    if (::pwrite(fd_, h.data(), h.size(), 0) != static_cast<ssize_t>(h.size())) fail("write", path_);
    if (::fsync(fd_) != 0) fail("fsync", path_);
    fsync_path(parent_dir(path_));
    return;
  }
  std::uint8_t h[kHeader];
  if (::pread(fd_, h, kHeader, 0) != static_cast<ssize_t>(kHeader)) fail("read", path_);
  if (std::memcmp(h, "OWAL", 4) != 0) {
    ::close(fd_);
    throw std::runtime_error("wal: bad magic");
  }
  auto v = get_le(h + 4, 2);
  if (v != kWalVersion) {
    ::close(fd_);
    throw std::runtime_error("wal: unknown version " + std::to_string(v));
  }
}

Wal::~Wal() {
  if (fd_ >= 0) ::close(fd_);
}

std::vector<WalRecord> Wal::recover(std::size_t* dropped) {
  auto b = read_file(path_);
  std::vector<WalRecord> out;
  std::size_t pos = kHeader;
  while (b.size() - pos >= 8) {
    auto len = get_le(b.data() + pos, 4);
    auto crc = static_cast<std::uint32_t>(get_le(b.data() + pos + 4, 4));
    if (len > kMaxRecord || b.size() - pos - 8 < len) break;
    const std::uint8_t* body = b.data() + pos + 8;
    if (crc32(body, len) != crc) break;
    WalRecord r;
    if (!decode_body(body, len, &r)) break;
    out.push_back(std::move(r));
    pos += 8 + len;
  }
  if (pos < b.size()) {
    if (dropped) *dropped = b.size() - pos;
    if (::ftruncate(fd_, static_cast<off_t>(pos)) != 0) fail("truncate", path_);
    if (::fsync(fd_) != 0) fail("fsync", path_);
  }
  return out;
}

void Wal::append(const WalRecord& r) {
  auto body = encode_body(r);
  std::vector<std::uint8_t> frame;
  put_le(frame, body.size(), 4);
  put_le(frame, crc32(body.data(), body.size()), 4);
  frame.insert(frame.end(), body.begin(), body.end());
  off_t end = ::lseek(fd_, 0, SEEK_END);
  if (end < 0) fail("seek", path_);
  write_all(fd_, frame.data(), frame.size(), path_);
  if (::fsync(fd_) != 0) fail("fsync", path_);
}

void Wal::reset() {
  if (::ftruncate(fd_, static_cast<off_t>(kHeader)) != 0) fail("truncate", path_);
  if (::fsync(fd_) != 0) fail("fsync", path_);
}

}  // namespace ombra::storage
