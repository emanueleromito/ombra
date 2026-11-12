// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Small durable-file helpers and CRC32 (IEEE). Errors are std::runtime_error.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ombra::storage {

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
