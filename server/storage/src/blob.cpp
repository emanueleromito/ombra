#include "ombra/storage/blob.hpp"

#include <stdexcept>

#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/fsutil.hpp"

namespace ombra::storage {

namespace {
constexpr std::size_t kHeader = 10;
void before_rename_hook() { maybe_crash(CrashPoint::BlobBeforeRename); }
}  // namespace

std::vector<std::uint8_t> encode_blob(std::uint32_t kind,
                                      const std::vector<std::uint8_t>& payload) {
  std::vector<std::uint8_t> body;
  body.reserve(12 + payload.size());
  put_le(body, kind, 4);
  put_le(body, payload.size(), 8);
  body.insert(body.end(), payload.begin(), payload.end());
  std::vector<std::uint8_t> out{'O', 'B', 'L', 'B'};
  put_le(out, kBlobVersion, 2);
  put_le(out, crc32(body.data(), body.size()), 4);
  out.insert(out.end(), body.begin(), body.end());
  return out;
}

void write_blob_file(const std::string& path, std::uint32_t kind,
                     const std::vector<std::uint8_t>& payload) {
  write_file_durable(path, encode_blob(kind, payload), before_rename_hook);
  maybe_crash(CrashPoint::BlobAfterRename);
}

std::vector<std::uint8_t> read_blob_file(const std::string& path, std::uint32_t expected_kind) {
  auto b = read_file(path);
  if (b.size() < kHeader + 12) throw std::runtime_error("blob: truncated header");
  if (b[0] != 'O' || b[1] != 'B' || b[2] != 'L' || b[3] != 'B')
    throw std::runtime_error("blob: bad magic");
  auto version = get_le(b.data() + 4, 2);
  if (version != kBlobVersion)
    throw std::runtime_error("blob: unknown container version " + std::to_string(version));
  auto crc = static_cast<std::uint32_t>(get_le(b.data() + 6, 4));
  const std::uint8_t* p = b.data() + kHeader;
  const std::size_t n = b.size() - kHeader;
  if (crc32(p, n) != crc) throw std::runtime_error("blob: CRC mismatch");
  if (get_le(p, 4) != expected_kind) throw std::runtime_error("blob: kind mismatch");
  const auto len = get_le(p + 4, 8);
  if (len != n - 12) throw std::runtime_error("blob: length mismatch");
  if (!is_blob_bucket(len)) throw std::runtime_error("blob: size is not a bucket size");
  return std::vector<std::uint8_t>(p + 12, p + 12 + len);
}

}  // namespace ombra::storage
