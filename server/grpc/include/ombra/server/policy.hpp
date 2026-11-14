// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Request policy of the gRPC server: limits and collection-spec validation. Pure C++, no
// gRPC, no OpenFHE and no GPU, so it is unit-tested on the CPU.
//
// The preset numbers come from core/params/ckks_presets.json through a header that CMake generates
// at configure time (preset_table.hpp); nothing is hand-picked here.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "ombra/storage/catalog.hpp"

namespace ombra::server {

// Numeric values equal to proto ombra.v1.ErrorCode (checked by static_assert in the service).
enum class Err : int {
  Ok = 0,
  UnsupportedVersion = 1,
  CollectionNotFound = 2,
  CollectionExists = 3,
  InvalidArgument = 4,
  SizeLimit = 5,
  KeyTagMismatch = 6,
  PresetMismatch = 7,
  InvalidFilterLayout = 8,
  KeysMissing = 9,
  BadSequence = 10,
  Internal = 11,
  BlockNotFound = 12,
  BlockMismatch = 13,
  BlobNotFound = 14,
  StorageFull = 15,
};

struct Status {
  Err code = Err::Ok;
  std::string message;  // fixed text; never ids, bytes or values
  bool ok() const { return code == Err::Ok; }
};

// Proto Limit enum values.
inline constexpr std::uint32_t kApiVersion = 1;
inline constexpr std::size_t kMaxChunkBytes = 4194304;
inline constexpr std::size_t kMaxNameBytes = 128;
inline constexpr std::size_t kMaxIdBytes = 256;
inline constexpr std::size_t kMaxIdsPerRequest = 100000;
inline constexpr std::size_t kMaxClustersPerRound = 4096;
// Not in the proto Limit enum: the catalog creates one row per cluster, so cap the count.
inline constexpr std::uint32_t kMaxClusterCount = 1u << 20;
inline constexpr std::size_t kMaxKeyTagBytes = 256;
// Payload caps per envelope kind (audit A-16), applied before OpenFHE reads anything. A ciphertext
// of the largest preset is far below these; they only stop a hostile size.
inline constexpr std::size_t kMaxContextBytes = std::size_t{64} << 20;  // kind 1, crypto context
inline constexpr std::size_t kMaxQueryBytes = std::size_t{64} << 20;    // kind 2, search query
inline constexpr std::size_t kMaxColumnBytes = std::size_t{64} << 20;  // kind 2 or 4, stored column
inline constexpr std::size_t kMaxEvalKeyBytes = std::size_t{1} << 30;  // kind 3, per key file

struct PresetInfo {
  const char* name;
  std::size_t ring_dim;
  std::size_t batch_size;  // slots per block
  std::size_t first_mod_bits;
  std::size_t scale_bits;
  std::size_t depth;  // multiplicative depth: a context has depth + 1 Q towers
};

const PresetInfo* find_preset(const std::string& name);  // nullptr if unknown

// Largest filter field count the preset's headroom allows (includes the tombstone
// penalty, capped at the proto limit of 16).
std::size_t max_filter_fields_for(const PresetInfo& p);

// Checks a CollectionSpec of CreateCollection: api_version, name, preset, dimension, key tag,
// cluster count, filter layout (field ids, cardinalities, preset headroom, D <= N/2).
Status validate_spec(const storage::CollectionSpec& spec);

}  // namespace ombra::server
