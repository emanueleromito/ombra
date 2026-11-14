// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// The OpenFHE-only part of loading a collection's keys. Moved out of session.cpp unchanged (only
// the names of the variables differ) so that the CPU-only build runs the same checks.
#include <cmath>
#include <set>
#include <sstream>

#include "ciphertext-ser.h"
#include "coll_keys.hpp"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/file_envelope.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

namespace ombra::search::detail {

using namespace lbcrypto;
namespace core = ombra::core;

namespace {

std::string to_string_bytes(const std::vector<std::uint8_t>& b) {
  return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

}  // namespace

void prepare_collection_keys(CollKeys& c, storage::Store& store, const EngineConfig& cfg,
                             const std::string& name, const std::vector<std::uint8_t>& ctx_b,
                             const std::vector<std::uint8_t>& mult_b,
                             const std::vector<std::uint8_t>& auto_b) {
  if (!store.get_collection(name, &c.spec)) throw CollectionNotFound("unknown collection");
  const std::string ofhe = store.openfhe_version();

  const auto payload = [&](const std::vector<std::uint8_t>& b, core::EnvelopeKind k) {
    return to_string_bytes(core::read_envelope(b, k, c.spec.preset, ofhe).payload);
  };
  {
    std::istringstream in(payload(ctx_b, core::EnvelopeKind::Context),
                          std::ios::in | std::ios::binary);
    Serial::Deserialize(c.cc, in, SerType::BINARY);
    if (!c.cc) throw SearchError("cannot read the crypto context");
  }
  core::require_security_floor(c.cc);
  // The uploaded context must be the collection's preset, not merely a secure one (audit A-16).
  if (cfg.preset_check) {
    const auto rns = std::dynamic_pointer_cast<CryptoParametersRNS>(c.cc->GetCryptoParameters());
    if (!rns) throw SearchError("context is not an RNS scheme");
    const auto& towers = rns->GetElementParams()->GetParams();
    // OpenFHE picks the scaling primes near 2^scale on both sides, so their bit lengths differ by
    // one (for example 50 and 51). Pass the smallest; towers that differ by more never match.
    std::size_t lo = 0, hi = 0;
    for (std::size_t i = 1; i < towers.size(); ++i) {
      const std::size_t b = towers[i]->GetModulus().GetMSB();
      lo = (lo == 0 || b < lo) ? b : lo;
      hi = b > hi ? b : hi;
    }
    const std::size_t scale_bits = hi > lo + 1 ? 1 : lo;
    if (towers.size() < 2 ||
        !cfg.preset_check(c.spec.preset, c.cc->GetRingDimension(),
                          c.cc->GetEncodingParams()->GetBatchSize(), towers.size() - 1,
                          towers[0]->GetModulus().GetMSB(), scale_bits))
      throw InvalidQuery("the uploaded context does not match the collection's preset");
  }

  // The key stores of OpenFHE are process-wide, by key tag: drop this tag's entries, load, and
  // require that the collection's tag is now present (a file made under another key is refused).
  const std::string& tag = c.spec.key_tag;
  CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys(tag);
  CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(tag);
  // The key files carry their own key tag and are inserted under it. Remember which tags exist,
  // so that keys of another tag that this call inserts can be removed again when they are refused.
  std::set<std::string> mult_before;
  std::set<std::string> auto_before;
  for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalMultKeys())
    mult_before.insert(kv.first);
  for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys())
    auto_before.insert(kv.first);
  const auto drop_foreign = [&] {
    std::vector<std::string> foreign;
    for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalMultKeys())
      if (!mult_before.count(kv.first)) foreign.push_back(kv.first);
    for (const auto& t : foreign) CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys(t);
    foreign.clear();
    for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys())
      if (!auto_before.count(kv.first)) foreign.push_back(kv.first);
    for (const auto& t : foreign) CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(t);
  };
  try {
    {
      std::istringstream in(payload(mult_b, core::EnvelopeKind::PublicEvalKey),
                            std::ios::in | std::ios::binary);
      if (!CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(in, SerType::BINARY))
        throw SearchError("cannot read the EvalMult key");
    }
    {
      std::istringstream in(payload(auto_b, core::EnvelopeKind::PublicEvalKey),
                            std::ios::in | std::ios::binary);
      if (!CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(in, SerType::BINARY))
        throw SearchError("cannot read the automorphism keys");
    }
  } catch (const lbcrypto::OpenFHEException&) {
    // OpenFHE refuses to insert keys under a tag that already holds keys: the files belong to a key
    // other than this collection's (a collection with the same files and another tag).
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }
  try {
    const auto& v = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag);
    if (v.empty()) throw std::runtime_error("empty");
    c.relin = v[0];
    c.akeys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag);
  } catch (const std::exception&) {
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }
  if (c.akeys.empty()) {
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }

  const auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(c.cc->GetCryptoParameters());
  if (!cp) throw SearchError("context is not an RNS scheme");
  c.ring = c.cc->GetRingDimension();
  c.batch = c.cc->GetEncodingParams()->GetBatchSize();
  c.column_scale = cp->GetScalingFactorReal(0);

  // F_max from the headroom of this context.
  const auto first_bits = cp->GetElementParams()->GetParams()[0]->GetModulus().GetMSB();
  const auto scale_bits = static_cast<std::size_t>(std::llround(std::log2(c.column_scale)));
  try {
    c.layout = make_layout(c.spec, c.ring, max_filter_fields(first_bits, scale_bits));
  } catch (const LayoutError& ex) {
    throw InvalidQuery(ex.what());
  }

}

}  // namespace ombra::search::detail
