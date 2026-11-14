// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Small durable-file helpers and CRC32 (IEEE). Errors are std::runtime_error.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ombra::storage {

// The disk or quota is full (ENOSPC, EDQUOT, or SQLITE_FULL). A std::runtime_error, so existing
// handlers still work; the gRPC service should map it to RESOURCE_EXHAUSTED: the call failed
// cleanly, nothing half-written is left, and the client may retry once space is freed.
class DiskFullError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// Throws DiskFullError if errno is ENOSPC or EDQUOT, else std::runtime_error. Message:
// "<what> '<path>': <strerror(errno)>" (paths are file names of the data directory).
[[noreturn]] void throw_io_error(const std::string& what, const std::string& path);

std::uint32_t crc32(const std::uint8_t* data, std::size_t n, std::uint32_t seed = 0);

void fsync_path(const std::string& path);  // file or directory
// Write to "<path>.tmp", fsync it, rename over path, fsync the directory. The optional hook runs
// after the temp file is durable and before the rename (used by crash tests).
void write_file_durable(const std::string& path, const std::vector<std::uint8_t>& bytes,
                        void (*before_rename)() = nullptr);
std::vector<std::uint8_t> read_file(const std::string& path);
std::string parent_dir(const std::string& path);

// Little-endian helpers shared by the segment and log formats.
void put_le(std::vector<std::uint8_t>& o, std::uint64_t v, int n);
std::uint64_t get_le(const std::uint8_t* p, int n);

}  // namespace ombra::storage
