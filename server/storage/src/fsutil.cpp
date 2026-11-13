// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/storage/fsutil.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ombra::storage {

namespace {
std::string err(const std::string& what, const std::string& path) {
  return what + " '" + path + "': " + std::strerror(errno);
}

const std::array<std::uint32_t, 256>& crc_table() {
  static const auto t = [] {
    std::array<std::uint32_t, 256> a{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      a[i] = c;
    }
    return a;
  }();
  return t;
}

struct Fd {
  int fd;
  ~Fd() {
    if (fd >= 0) ::close(fd);
  }
};
}  // namespace

std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t seed) {
  std::uint32_t c = ~seed;
  const auto& t = crc_table();
  for (std::size_t i = 0; i < n; ++i) c = t[(c ^ data[i]) & 0xFF] ^ (c >> 8);
  return ~c;
}

void put_le(std::vector<std::uint8_t>& o, std::uint64_t v, int n) {
  for (int i = 0; i < n; ++i) o.push_back(static_cast<std::uint8_t>(v >> (8 * i)));
}
std::uint64_t get_le(const std::uint8_t* p, int n) {
  std::uint64_t v = 0;
  for (int i = 0; i < n; ++i) v |= static_cast<std::uint64_t>(p[i]) << (8 * i);
  return v;
}

std::string parent_dir(const std::string& path) {
  auto pos = path.find_last_of('/');
  if (pos == std::string::npos) return ".";
  return pos == 0 ? "/" : path.substr(0, pos);
}

void fsync_path(const std::string& path) {
  Fd f{::open(path.c_str(), O_RDONLY)};
  if (f.fd < 0) throw std::runtime_error(err("open for fsync", path));
  if (::fsync(f.fd) != 0) throw std::runtime_error(err("fsync", path));
}

void write_file_durable(const std::string& path, const std::vector<std::uint8_t>& bytes,
                        void (*before_rename)()) {
  const std::string tmp = path + ".tmp";
  {
    // Exclusive and never through a symlink (audit A-56); a stale temp file of a crashed write is
    // removed first (recovery sweeps them too). Mode 0600: the data directory is private (A-22).
    ::unlink(tmp.c_str());
    Fd f{::open(tmp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600)};
    if (f.fd < 0) throw std::runtime_error(err("create", tmp));
    std::size_t off = 0;
    while (off < bytes.size()) {
      ssize_t w = ::write(f.fd, bytes.data() + off, bytes.size() - off);
      if (w < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(err("write", tmp));
      }
      off += static_cast<std::size_t>(w);
    }
    if (::fsync(f.fd) != 0) throw std::runtime_error(err("fsync", tmp));
  }
  if (before_rename) before_rename();
  if (::rename(tmp.c_str(), path.c_str()) != 0) throw std::runtime_error(err("rename", path));
  fsync_path(parent_dir(path));
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  Fd f{::open(path.c_str(), O_RDONLY)};
  if (f.fd < 0) throw std::runtime_error(err("open", path));
  struct stat st {};
  if (::fstat(f.fd, &st) != 0) throw std::runtime_error(err("stat", path));
  std::vector<std::uint8_t> b(static_cast<std::size_t>(st.st_size));
  std::size_t off = 0;
  while (off < b.size()) {
    ssize_t r = ::read(f.fd, b.data() + off, b.size() - off);
    if (r < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(err("read", path));
    }
    if (r == 0) throw std::runtime_error("file shrank while reading: " + path);
    off += static_cast<std::size_t>(r);
  }
  return b;
}

}  // namespace ombra::storage
