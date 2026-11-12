#include "ombra/core/expansion_gpu.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>

#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"

namespace ombra::core {
namespace {

using namespace lbcrypto;
using u64 = std::uint64_t;

static_assert(sizeof(NativeInteger) == sizeof(u64), "OpenFHE must be built with NATIVE_SIZE=64");

[[noreturn]] void fail(const std::string& what) {
  throw std::invalid_argument("expansion tables: " + what);
}

std::uint32_t bit_reverse(std::uint32_t x, std::uint32_t bits) {
  std::uint32_t r = 0;
  for (std::uint32_t i = 0; i < bits; ++i) r |= ((x >> i) & 1u) << (bits - 1 - i);
  return r;
}

u64 mulmod(u64 a, u64 b, u64 q) {
  return static_cast<u64>((static_cast<unsigned __int128>(a) * b) % q);
}

u64 to_u64(const NativeInteger& v) { return v.ConvertToInt<u64>(); }

std::vector<u64> to_u64(const std::vector<NativeInteger>& v) {
  std::vector<u64> out;
  for (const auto& x : v) out.push_back(to_u64(x));
  return out;
}

std::shared_ptr<CryptoParametersRNS> rns_params(const CryptoContext<DCRTPoly>& cc) {
  require_security_floor(cc);
  auto p = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
  if (!p) fail("only CKKS is supported");
  if (p->GetKeySwitchTechnique() != HYBRID) fail("only HYBRID key switching is supported");
  if (p->GetNoiseScale() != 1) fail("noise scale must be 1 (t = 0 in ModDown)");
  return p;
}

}  // namespace

ExpansionTables extract_expansion_tables(const CryptoContext<DCRTPoly>& cc,
                                         const Ciphertext<DCRTPoly>& sample) {
  const auto cp = rns_params(cc);
  if (sample->GetElements().size() != 2) fail("ciphertext must have 2 components");
  const DCRTPoly& c0 = sample->GetElements()[0];
  const auto paramsQ = c0.GetParams();
  const auto paramsP = cp->GetParamsP();

  ExpansionTables t;
  t.ring_dim = c0.GetRingDimension();
  t.num_q = c0.GetNumOfElements();
  t.num_p = paramsP->GetParams().size();
  const std::size_t n = t.ring_dim, nq = t.num_q, np = t.num_p, qp = nq + np;
  std::uint32_t logn = 0;
  while ((std::size_t{1} << logn) < n) ++logn;

  // Moduli and NTT tables: OpenFHE's ChineseRemainderTransformFTTNat::GetTables builds
  // rootReverse[bitrev(i)] = psi^i with psi = the tower's root of unity (transformnat-impl.h:743).
  std::vector<std::shared_ptr<ILNativeParams>> towers;
  for (const auto& p : paramsQ->GetParams()) towers.push_back(p);
  for (const auto& p : paramsP->GetParams()) towers.push_back(p);
  if (towers.size() != qp) fail("tower count");
  for (std::size_t x = 0; x < qp; ++x) {
    const u64 q = to_u64(towers[x]->GetModulus());
    const u64 psi = to_u64(towers[x]->GetRootOfUnity());
    if (towers[x]->GetRingDimension() != n) fail("tower ring dimension");
    const u64 psi_inv = to_u64(NativeInteger(psi).ModInverse(NativeInteger(q)));
    t.moduli.push_back(q);
    t.n_inv.push_back(to_u64(NativeInteger(n).ModInverse(NativeInteger(q))));
    const std::size_t base = x * n;
    t.psi_rev.resize(base + n);
    t.psi_inv_rev.resize(base + n);
    u64 a = 1, b = 1;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint32_t pos = bit_reverse(static_cast<std::uint32_t>(i), logn);
      t.psi_rev[base + pos] = a;
      t.psi_inv_rev[base + pos] = b;
      a = mulmod(a, psi, q);
      b = mulmod(b, psi_inv, q);
    }
  }

  // Digits, as in KeySwitchHYBRID::EvalKeySwitchPrecomputeCore (keyswitch-hybrid.cpp:300).
  const std::size_t alpha = cp->GetNumPerPartQ();
  std::size_t num_parts = (nq + alpha - 1) / alpha;
  num_parts = std::min<std::size_t>(num_parts, cp->GetNumberOfQPartitions());
  for (std::size_t part = 0; part < num_parts; ++part) {
    const std::size_t first = alpha * part;
    const std::size_t count =
        (part == num_parts - 1) ? nq - alpha * part : cp->GetParamsPartQ(part)->GetParams().size();
    t.digit_first.push_back(first);
    t.digit_count.push_back(count);
    const auto hat_inv = to_u64(cp->GetPartQlHatInvModq(part, count - 1));
    if (hat_inv.size() != count) fail("PartQlHatInvModq size");
    t.part_hat_inv.insert(t.part_hat_inv.end(), hat_inv.begin(), hat_inv.end());
    const auto& hat_mod = cp->GetPartQlHatModp(nq - 1, part);
    const std::size_t ncomp = qp - count;
    if (hat_mod.size() != count) fail("PartQlHatModp rows");
    const auto& compl_params = cp->GetParamsComplPartQ(nq - 1, part)->GetParams();
    if (compl_params.size() != ncomp) fail("complement basis size");
    std::vector<u64> flat;
    for (std::size_t i = 0; i < count; ++i) {
      if (hat_mod[i].size() != ncomp) fail("PartQlHatModp columns");
      for (std::size_t j = 0; j < ncomp; ++j) flat.push_back(to_u64(hat_mod[i][j]));
    }
    // The complement basis is the Q towers outside the digit, then P, in this order.
    for (std::size_t j = 0; j < ncomp; ++j) {
      const std::size_t tower = j < first ? j : j + count;
      if (to_u64(compl_params[j]->GetModulus()) != t.moduli[tower]) fail("complement order");
    }
    t.part_hat_mod.push_back(std::move(flat));
  }
  if (t.part_hat_inv.size() != nq) fail("digits do not cover the Q towers");

  // ModDown (dcrtpoly-impl.h:1020) and the P multiplications.
  const auto p_mod_q = to_u64(cp->GetPModq());
  const auto p_inv_mod_q = to_u64(cp->GetPInvModq());
  const auto p_hat_inv = to_u64(cp->GetPHatInvModp());
  const auto& p_hat_mod = cp->GetPHatModq();
  if (p_mod_q.size() < nq || p_inv_mod_q.size() < nq || p_hat_inv.size() != np ||
      p_hat_mod.size() != np) {
    fail("P table sizes");
  }
  t.p_mod_q.assign(p_mod_q.begin(), p_mod_q.begin() + nq);
  t.p_inv_mod_q.assign(p_inv_mod_q.begin(), p_inv_mod_q.begin() + nq);
  t.p_hat_inv_mod_p = p_hat_inv;
  for (std::size_t j = 0; j < np; ++j) {
    if (p_hat_mod[j].size() < nq) fail("PHatModq columns");
    for (std::size_t i = 0; i < nq; ++i) t.p_hat_mod_q.push_back(to_u64(p_hat_mod[j][i]));
  }

  // Rescale: q_l^-1 mod q_i for the last tower of this level (ModReduceInternalInPlace uses
  // GetqlInvModq(diffQl) with diffQl = full Q towers - towers of the ciphertext).
  const std::size_t full_q = cp->GetElementParams()->GetParams().size();
  if (nq >= 2 && full_q >= nq) {
    t.ql_inv_mod_q = to_u64(cp->GetqlInvModq(full_q - nq));
    if (t.ql_inv_mod_q.size() != nq - 1) fail("qlInvModq size");
  }
  return t;
}

std::vector<u64> extract_key_words(const EvalKey<DCRTPoly>& key, const ExpansionTables& t,
                                   std::size_t full_q_towers) {
  const std::size_t n = t.ring_dim, nq = t.num_q, qp = nq + t.num_p, nd = t.digit_first.size();
  if (full_q_towers < nq) fail("key has fewer Q towers than the ciphertext");
  const std::size_t delta = full_q_towers - nq;
  const auto& av = key->GetAVector();
  const auto& bv = key->GetBVector();
  if (av.size() < nd || bv.size() < nd) fail("key has fewer digits than the ciphertext needs");
  std::vector<u64> words;
  words.reserve(nd * 2 * qp * n);
  for (std::size_t p = 0; p < nd; ++p) {
    for (const DCRTPoly* poly : {&bv[p], &av[p]}) {  // b first, then a
      if (poly->GetFormat() != Format::EVALUATION) fail("key is not in evaluation form");
      for (std::size_t x = 0; x < qp; ++x) {
        const std::size_t idx = x < nq ? x : x + delta;
        const auto& tower = poly->GetElementAtIndex(idx);
        if (to_u64(tower.GetModulus()) != t.moduli[x]) fail("key tower modulus");
        const auto& v = tower.GetValues();
        for (std::size_t i = 0; i < n; ++i) words.push_back(to_u64(v[i]));
      }
    }
  }
  return words;
}

std::vector<u64> extract_monomial_words(const DCRTPoly& like, std::size_t shift) {
  DCRTPoly m(like.GetParams(), Format::COEFFICIENT, true);
  const std::size_t n = like.GetRingDimension();
  if (shift == 0 || shift >= n) fail("monomial shift out of range");
  for (std::size_t x = 0; x < m.GetNumOfElements(); ++x) {
    NativePoly tower = m.GetElementAtIndex(x);
    const NativeInteger q = tower.GetModulus();
    NativeVector v(n, q);
    for (std::size_t i = 0; i < n; ++i) v[i] = NativeInteger(0);
    v[n - shift] = q - NativeInteger(1);  // X^-shift = -X^(N-shift)
    tower.SetValues(std::move(v), Format::COEFFICIENT);
    m.SetElementAtIndex(x, std::move(tower));
  }
  m.SetFormat(Format::EVALUATION);
  std::vector<u64> words;
  for (std::size_t x = 0; x < m.GetNumOfElements(); ++x) {
    const auto& v = m.GetElementAtIndex(x).GetValues();
    for (std::size_t i = 0; i < n; ++i) words.push_back(to_u64(v[i]));
  }
  return words;
}

namespace {
// Refuses keys of another secret key before any table or device buffer is built.
const Ciphertext<DCRTPoly>& checked_sample(const Ciphertext<DCRTPoly>& sample,
                                           const GpuExpansion::KeyMap& keys) {
  for (const auto& entry : keys) {
    require_eval_key_matches(entry.second, sample, "expansion automorphism key");
  }
  return sample;
}
}  // namespace

GpuExpansion::GpuExpansion(const CryptoContext<DCRTPoly>& cc, const Ct& sample, const KeyMap& keys,
                           std::size_t d)
    : tables_(extract_expansion_tables(cc, checked_sample(sample, keys))),
      device_(tables_),
      sample_(sample) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto cp = rns_params(cc);
  const std::size_t full_q = cp->GetElementParams()->GetParams().size();
  const std::size_t n = tables_.ring_dim;
  const std::size_t depth = expansion_depth(d);
  for (std::size_t l = 0; l < depth; ++l) {
    const std::uint32_t galois = static_cast<std::uint32_t>(n / (std::size_t{1} << l) + 1);
    const auto it = keys.find(galois);
    if (it == keys.end()) {
      throw std::invalid_argument("expansion: no automorphism key for Galois element " +
                                  std::to_string(galois));
    }
    std::vector<std::uint32_t> map(n);
    PrecomputeAutoMap(static_cast<std::uint32_t>(n), galois, &map);
    device_.add_key(galois, map, extract_key_words(it->second, tables_, full_q));
    device_.add_monomial(l, extract_monomial_words(sample->GetElements()[0], std::size_t{1} << l));
    ++key_count_;
  }
  setup_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

GpuCt GpuExpansion::upload(const Ct& ct) {
  require_same_key(ct, sample_, "expansion upload");
  const FlatCiphertexts f = flatten_ciphertexts({ct});
  if (f.level != sample_->GetLevel() || f.noise_scale_deg != sample_->GetNoiseScaleDeg() ||
      f.scaling_factor != sample_->GetScalingFactor()) {
    throw std::invalid_argument(
        "expansion: ciphertext level, noise degree or scale differ from the sample");
  }
  if (f.ring_dim != tables_.ring_dim || f.num_towers != tables_.num_q) {
    throw std::invalid_argument(
        "expansion: ciphertext ring dimension or tower count differ from the tables");
  }
  if (f.moduli !=
      std::vector<u64>(tables_.moduli.begin(), tables_.moduli.begin() + tables_.num_q)) {
    throw std::invalid_argument("expansion: ciphertext moduli differ from the tables");
  }
  return {device_.upload(f.words)};
}

GpuExpansion::Ct GpuExpansion::download(const GpuCt& ct) {
  return unflatten_ciphertext(sample_, device_.download(ct.dev));
}

std::pair<GpuCt, GpuCt> GpuExpansion::split(const GpuCt& c, std::size_t level, std::uint32_t galois,
                                            std::size_t /*shift*/, bool want_odd) {
  auto r = device_.split(c.dev, galois, level, want_odd);
  return {GpuCt{std::move(r.first)}, GpuCt{std::move(r.second)}};
}

std::vector<std::pair<GpuCt, GpuCt>> GpuExpansion::split_level(const std::vector<GpuCt>& nodes,
                                                               std::size_t level,
                                                               std::uint32_t galois,
                                                               std::size_t /*shift*/,
                                                               const std::vector<char>& want_odd) {
  std::vector<CtHandle> in;
  in.reserve(nodes.size());
  for (const GpuCt& c : nodes) in.push_back(c.dev);
  auto r = device_.split_level(in, galois, level, want_odd);
  std::vector<std::pair<GpuCt, GpuCt>> out;
  out.reserve(r.size());
  for (auto& p : r) out.push_back({GpuCt{std::move(p.first)}, GpuCt{std::move(p.second)}});
  return out;
}

}  // namespace ombra::core
