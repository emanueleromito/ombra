#include "ombra/core/security_floor.hpp"

#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "ombra/core/security_floor_context.hpp"

namespace ombra::core {
namespace {

using namespace lbcrypto;

StdLevel to_level(SecurityLevel l) {
  switch (l) {
    case HEStd_128_classic:
      return StdLevel::kClassic128;
    case HEStd_192_classic:
      return StdLevel::kClassic192;
    case HEStd_256_classic:
      return StdLevel::kClassic256;
    case HEStd_128_quantum:
      return StdLevel::kQuantum128;
    case HEStd_192_quantum:
      return StdLevel::kQuantum192;
    case HEStd_256_quantum:
      return StdLevel::kQuantum256;
    default:
      return StdLevel::kNotSet;
  }
}

std::size_t sum_bits(const std::shared_ptr<ILDCRTParams<BigInteger>>& p) {
  std::size_t bits = 0;
  if (!p) return 0;
  for (const auto& tower : p->GetParams()) bits += tower->GetModulus().GetMSB();
  return bits;
}

}  // namespace

std::size_t context_log2_qp(const CryptoContext<DCRTPoly>& cc) {
  if (!cc) throw std::invalid_argument("security floor: null crypto context");
  const auto rns = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
  if (!rns) throw std::invalid_argument("security floor: not an RNS context; refusing");
  return sum_bits(rns->GetElementParams()) + sum_bits(rns->GetParamsP());
}

void require_security_floor(const CryptoContext<DCRTPoly>& cc) {
  const std::size_t log2_qp = context_log2_qp(cc);
  const auto rns = std::dynamic_pointer_cast<CryptoParametersRNS>(cc->GetCryptoParameters());
  std::size_t extra = 0;
  if (const char* env = std::getenv("OMBRA_TEST_LOG2QP_LIMIT")) {
    extra = std::strtoull(env, nullptr, 10);
  }
  check_security_floor(to_level(rns->GetStdLevel()), rns->GetElementParams()->GetRingDimension(),
                       log2_qp, extra);
}

}  // namespace ombra::core
