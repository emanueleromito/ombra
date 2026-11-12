// The 128-bit security floor, enforced on the C++ side (review finding F7).
//
// `check_security_floor` is a pure function on plain numbers (no OpenFHE), so the negative tests can
// run without ever building an insecure CryptoContext. `require_security_floor` (in
// security_floor_context.hpp) reads the same three numbers from a CryptoContext and is called by
// every core entry point that takes one.
//
// Limits are the HomomorphicEncryption.org 128-bit classic table (maximum log2 of the full modulus
// QP, special primes included, per ring dimension). A context declared at 192 or 256 bits is held
// to the same 128-bit limit, which is the floor. Ring dimensions above 32768 are not in the table
// and are refused until a verified value is added.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace ombra::core {

enum class StdLevel {
  kNotSet,
  kClassic128,
  kClassic192,
  kClassic256,
  kQuantum128,
  kQuantum192,
  kQuantum256
};

struct SecurityLimit {
  std::size_t ring_dim;
  std::size_t max_log2_qp;
};

inline constexpr SecurityLimit kSecurityTable128Classic[] = {
    {1024, 27}, {2048, 54}, {4096, 109}, {8192, 218}, {16384, 438}, {32768, 881},
};

// Throws std::invalid_argument unless the level is declared and log2_qp is within the limit for
// ring_dim. A ring dimension that is not in the table is refused. `extra_limit` (0 = none) can only
// lower the limit; it exists for the wiring tests (OMBRA_TEST_LOG2QP_LIMIT) and never accepts more.
inline void check_security_floor(StdLevel level, std::size_t ring_dim, std::size_t log2_qp,
                                 std::size_t extra_limit = 0) {
  if (level == StdLevel::kNotSet) {
    throw std::invalid_argument("security floor: security level is HEStd_NotSet; refusing");
  }
  std::size_t limit = 0;
  for (const auto& e : kSecurityTable128Classic) {
    if (e.ring_dim == ring_dim) limit = e.max_log2_qp;
  }
  if (limit == 0) {
    throw std::invalid_argument("security floor: ring dimension " + std::to_string(ring_dim) +
                                " is not in the 128-bit table; refusing");
  }
  if (extra_limit != 0 && extra_limit < limit) limit = extra_limit;
  if (log2_qp > limit) {
    throw std::invalid_argument("security floor: log2(QP) = " + std::to_string(log2_qp) +
                                " exceeds the limit " + std::to_string(limit) +
                                " for ring dimension " + std::to_string(ring_dim) + "; refusing");
  }
}

}  // namespace ombra::core
