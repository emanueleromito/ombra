// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// CryptoContext wrapper for the security floor check (see security_floor.hpp).
#pragma once

#include <cstddef>

#include "openfhe.h"

namespace ombra::core {

// log2(QP) as the check computes it: the sum of the bit lengths of every Q and P tower.
std::size_t context_log2_qp(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

// Reads the security level, ring dimension and log2(QP) (sum of the bit lengths of every Q and P
// tower) from the context and throws std::invalid_argument if it is below the 128-bit floor.
// The environment variable OMBRA_TEST_LOG2QP_LIMIT, if set, can only lower the limit (used by the
// wiring tests to prove every entry point calls this check).
void require_security_floor(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc);

}  // namespace ombra::core
