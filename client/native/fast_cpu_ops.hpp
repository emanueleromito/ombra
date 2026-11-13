// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Faster CPU operations of local mode (PERF-B, experiments 0003 and 0004). Bit-identical to the
// reference code: core/src/openfhe_expand_ops.hpp (expansion split) and the EvalMultNoRelin /
// EvalAdd loop (score). The reference stays untouched: it is what the GPU oracle tests compare
// against. Header only; included by ombra_native.cpp and by the exactness test.
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include "openfhe.h"
#include "openfhe_expand_ops.hpp"

namespace ombra::native {

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

static_assert(sizeof(NativeInteger) == sizeof(u64), "NativeInteger must be one 64-bit word");

// ---------------------------------------------------------------------------------------------
// Expansion: X^-k applied in evaluation form (one pointwise product with NTT(X^-k)) instead of
// the coefficient form of cpu_ref::mul_by_neg_monomial (inverse and forward NTT of every tower).
// ---------------------------------------------------------------------------------------------

// NTT of X^(-k) = -X^(N-k) over the towers of `like`.
inline DCRTPoly make_monomial(const DCRTPoly& like, std::size_t k) {
  DCRTPoly m(like.GetParams(), Format::COEFFICIENT, true);
  const std::size_t n = like.GetRingDimension();
  for (std::size_t t = 0; t < m.GetNumOfElements(); ++t) {
    NativePoly tower = m.GetElementAtIndex(t);
    NativeVector v(n, tower.GetModulus());
    v[n - k] = tower.GetModulus() - NativeInteger(1);
    tower.SetValues(std::move(v), Format::COEFFICIENT);
    m.SetElementAtIndex(t, std::move(tower));
  }
  m.SetFormat(Format::EVALUATION);
  return m;
}

// One monomial per tree level (the shift of level l is 2^l, fixed by the tree), computed on first
// use under a mutex. std::map references stay valid, so a returned reference can be used without
// the lock.
class MonomialCache {
 public:
  const DCRTPoly& get(const DCRTPoly& like, std::size_t shift) {
    std::lock_guard<std::mutex> g(mu_);
    auto it = map_.find(shift);
    if (it == map_.end() || it->second.GetNumOfElements() != like.GetNumOfElements() ||
        it->second.GetRingDimension() != like.GetRingDimension()) {
      map_.erase(shift);
      it = map_.emplace(shift, make_monomial(like, shift)).first;
    }
    return it->second;
  }

 private:
  std::mutex mu_;
  std::map<std::size_t, DCRTPoly> map_;
};

class FastSplitOps {
 public:
  FastSplitOps(CryptoContext<DCRTPoly> cc, const std::map<uint32_t, EvalKey<DCRTPoly>>& keys,
               MonomialCache& cache)
      : cc_(cc), keys_(keys), cache_(cache), ref_(std::move(cc), keys) {}

  // (c + tau(c), (c - tau(c)) * X^-shift); same contract as cpu_ref::OpenFheOps::split.
  std::pair<Ct, Ct> split(const Ct& c, std::size_t /*level*/, std::uint32_t galois,
                          std::size_t shift, bool want_odd) {
    const auto& ce = c->GetElements();
    bool eval_form = !ce.empty();
    for (const auto& e : ce) eval_form = eval_form && e.GetFormat() == Format::EVALUATION;
    if (!eval_form) return ref_.split(c, 0, galois, shift, want_odd);
    Ct sub = cc_->EvalAutomorphism(c, galois, keys_);
    Ct even = cc_->EvalAdd(c, sub);
    if (!want_odd) return {std::move(even), Ct()};
    const DCRTPoly& mono = cache_.get(ce[0], shift);
    auto& se = sub->GetElements();
    for (std::size_t r = 0; r < se.size(); ++r) {
      se[r] = ce[r] - se[r];
      se[r] *= mono;
    }
    return {std::move(even), std::move(sub)};
  }

 private:
  CryptoContext<DCRTPoly> cc_;
  const std::map<uint32_t, EvalKey<DCRTPoly>>& keys_;
  MonomialCache& cache_;
  ombra::core::cpu_ref::OpenFheOps ref_;
};

// ---------------------------------------------------------------------------------------------
// Score: sum_j leaf_j * column_j without relinearization, as one pass over the raw RNS words.
// ---------------------------------------------------------------------------------------------

namespace detail {

inline const u64* words(const DCRTPoly& p, std::size_t tower) {
  return reinterpret_cast<const u64*>(&p.GetAllElements()[tower].GetValues()[0]);
}

// Montgomery constants of one tower (R = 2^64), as in core/kernels/modarith.cuh.
struct Mod {
  u64 q;
  u64 qinv;  // q^-1 mod 2^64
  u64 r2;    // 2^128 mod q
  u64 fold;  // columns that may be summed before a reduction, see fold_interval
};

// REDC: x * R^-1 mod q for x < q * 2^64.
inline u64 redc(u128 x, const Mod& m) {
  const u64 mm = static_cast<u64>(x) * m.qinv;
  const u64 hi = static_cast<u64>(x >> 64);
  const u64 mq = static_cast<u64>((static_cast<u128>(mm) * m.q) >> 64);
  return hi >= mq ? hi - mq : hi + m.q - mq;
}

// Fold bound. One column adds at most (2q)^2 = 4 q^2 to an accumulator (the Karatsuba cross term
// (s0 + s1)(q0 + q1), each factor < 2q; the other two terms add less). REDC needs its input below
// q * 2^64, so K columns are safe when K * 4 q^2 < q * 2^64, i.e. K < 2^62 / q. For q < 2^60 that
// is at least 4 (for a 50-bit tower 4096). The same bound keeps the sum below 2^124, inside the
// 128-bit accumulator. The result is also capped at 256.
inline u64 fold_interval(u64 q) {
  const u64 k = (static_cast<u64>(1) << 62) / q;
  const u64 safe = k * q == (static_cast<u64>(1) << 62) ? k - 1 : k;
  return safe > 256 ? 256 : safe;
}

inline Mod make_mod(u64 q) {
  Mod m;
  m.q = q;
  m.qinv = q;
  for (int i = 0; i < 6; ++i) m.qinv *= 2 - q * m.qinv;
  const u64 r1 = static_cast<u64>((static_cast<u128>(1) << 64) % q);
  m.r2 = static_cast<u64>(static_cast<u128>(r1) * r1 % q);
  m.fold = fold_interval(q);
  return m;
}

inline u64 add_mod(u64 a, u64 b, u64 q) {
  const u64 r = a + b;
  return r >= q ? r - q : r;
}

constexpr std::size_t kChunk = 1024;  // words per task: 2 * d * 8 KB of input stays in L2

// True when the fused pass may be used (run-time check of its requirements): every ciphertext in
// EVALUATION format with 2 components, the same tower count, ring dimension and moduli, every
// modulus odd and < 2^60, ring dimension a multiple of kChunk.
inline bool fused_applicable(const std::vector<Ct>& leaves, const std::vector<Ct>& cols,
                             std::size_t n_cols) {
  if (n_cols == 0 || leaves.size() < n_cols || cols.size() < n_cols) return false;
  const auto& e0 = cols[0]->GetElements();
  if (e0.size() != 2) return false;
  const std::size_t towers = e0[0].GetNumOfElements(), ring = e0[0].GetRingDimension();
  if (towers == 0 || ring == 0 || ring % kChunk != 0) return false;
  std::vector<u64> q(towers);
  for (std::size_t t = 0; t < towers; ++t) {
    q[t] = e0[0].GetAllElements()[t].GetModulus().ConvertToInt<u64>();
    if ((q[t] & 1) == 0 || q[t] >= (static_cast<u64>(1) << 60)) return false;
  }
  const auto ok = [&](const Ct& c) {
    const auto& e = c->GetElements();
    if (e.size() != 2) return false;
    for (const auto& p : e) {
      if (p.GetFormat() != Format::EVALUATION || p.GetNumOfElements() != towers ||
          p.GetRingDimension() != ring)
        return false;
      for (std::size_t t = 0; t < towers; ++t)
        if (p.GetAllElements()[t].GetModulus().ConvertToInt<u64>() != q[t]) return false;
    }
    return true;
  };
  for (std::size_t j = 0; j < n_cols; ++j)
    if (!ok(cols[j]) || !ok(leaves[j])) return false;
  return true;
}

// Fused Karatsuba pass (experiment 0003, fused<true>). out: [3][towers][N] standard residues of
// the three components (s0 q0, s0 q1 + s1 q0, s1 q1) summed over the first d columns.
inline std::vector<u64> fused_words(const std::vector<Ct>& leaves, const std::vector<Ct>& cols,
                                    std::size_t d) {
  const auto& e0 = cols[0]->GetElements()[0];
  const std::size_t towers = e0.GetNumOfElements(), n = e0.GetRingDimension();
  std::vector<u64> out(3 * towers * n);
  std::vector<Mod> mods;
  for (std::size_t t = 0; t < towers; ++t)
    mods.push_back(make_mod(e0.GetAllElements()[t].GetModulus().ConvertToInt<u64>()));
  const long tasks = static_cast<long>(towers * (n / kChunk));
#pragma omp parallel for schedule(static)
  for (long task = 0; task < tasks; ++task) {
    const std::size_t t = static_cast<std::size_t>(task) / (n / kChunk);
    const std::size_t c0 = (static_cast<std::size_t>(task) % (n / kChunk)) * kChunk;
    const Mod m = mods[t];
    u128 a0[kChunk], a1[kChunk], a2[kChunk];
    u64 r0[kChunk], r1[kChunk], r2[kChunk];
    for (std::size_t k = 0; k < kChunk; ++k) {
      a0[k] = a1[k] = a2[k] = 0;
      r0[k] = r1[k] = r2[k] = 0;
    }
    std::size_t pending = 0;
    // r* hold R^-1 * (sum so far) mod q; the factor R is restored once at the end.
    const auto fold = [&]() {
      for (std::size_t k = 0; k < kChunk; ++k) {
        r0[k] = add_mod(r0[k], redc(a0[k], m), m.q);
        r1[k] = add_mod(r1[k], redc(a1[k], m), m.q);
        r2[k] = add_mod(r2[k], redc(a2[k], m), m.q);
        a0[k] = a1[k] = a2[k] = 0;
      }
      pending = 0;
    };
    for (std::size_t j = 0; j < d; ++j) {
      const u64* s0 = words(cols[j]->GetElements()[0], t) + c0;
      const u64* s1 = words(cols[j]->GetElements()[1], t) + c0;
      const u64* q0 = words(leaves[j]->GetElements()[0], t) + c0;
      const u64* q1 = words(leaves[j]->GetElements()[1], t) + c0;
      for (std::size_t k = 0; k < kChunk; ++k) {
        a0[k] += static_cast<u128>(s0[k]) * q0[k];
        a2[k] += static_cast<u128>(s1[k]) * q1[k];
        a1[k] += static_cast<u128>(s0[k] + s1[k]) * (q0[k] + q1[k]);  // each sum < 2^61
      }
      if (++pending == m.fold) fold();
    }
    if (pending != 0) fold();
    u64* o = out.data() + t * n + c0;
    for (std::size_t k = 0; k < kChunk; ++k) {
      r0[k] = redc(static_cast<u128>(r0[k]) * m.r2, m);  // times R: the standard residue
      r1[k] = redc(static_cast<u128>(r1[k]) * m.r2, m);
      r2[k] = redc(static_cast<u128>(r2[k]) * m.r2, m);
      o[k] = r0[k];
      o[2 * towers * n + k] = r2[k];
      // cross = K - a0 - a2 (mod q)
      u64 x = r1[k] >= r0[k] ? r1[k] - r0[k] : r1[k] + m.q - r0[k];
      x = x >= r2[k] ? x - r2[k] : x + m.q - r2[k];
      o[towers * n + k] = x;
    }
  }
  return out;
}

}  // namespace detail

// The old loop: per column EvalMultNoRelin then EvalAdd, per OpenMP thread, partial sums added.
inline Ct score_sum_loop(const CryptoContext<DCRTPoly>& cc, const std::vector<Ct>& leaves,
                         const std::vector<Ct>& cols, std::size_t n_cols) {
  const long n = static_cast<long>(n_cols);
  int threads = 1;
#ifdef _OPENMP
  threads = omp_get_max_threads();
#endif
  std::vector<Ct> partial(static_cast<std::size_t>(threads));
  std::exception_ptr err;
#pragma omp parallel num_threads(threads)
  {
    int me = 0;
#ifdef _OPENMP
    me = omp_get_thread_num();
#endif
    Ct acc;
    try {
#pragma omp for schedule(static)
      for (long j = 0; j < n; ++j) {
        Ct t = cc->EvalMultNoRelin(leaves[j], cols[j]);
        acc = acc ? cc->EvalAdd(acc, t) : t;
      }
    } catch (...) {
#pragma omp critical(ombra_err)
      err = std::current_exception();
    }
    partial[me] = acc;
  }
  if (err) std::rethrow_exception(err);
  Ct acc;
  for (auto& p : partial)
    if (p) acc = acc ? cc->EvalAdd(acc, p) : p;
  return acc;
}

// sum_j leaf_j * column_j as a 3-component ciphertext (before relinearization), by the fused
// pass when its requirements hold at run time, else by the old loop.
inline Ct score_sum(const CryptoContext<DCRTPoly>& cc, const std::vector<Ct>& leaves,
                    const std::vector<Ct>& cols, std::size_t n_cols) {
  if (!detail::fused_applicable(leaves, cols, n_cols))
    return score_sum_loop(cc, leaves, cols, n_cols);
  // The metadata (level, noise scale degree, scaling factor, key tag, slots, encoding) of the
  // product of two such ciphertexts comes from one real EvalMultNoRelin; its words are replaced.
  Ct res = cc->EvalMultNoRelin(leaves[0], cols[0]);
  auto& elems = res->GetElements();
  const std::size_t towers = cols[0]->GetElements()[0].GetNumOfElements();
  const std::size_t n = cols[0]->GetElements()[0].GetRingDimension();
  if (elems.size() != 3 || elems[0].GetNumOfElements() != towers ||
      elems[0].GetFormat() != Format::EVALUATION)
    return score_sum_loop(cc, leaves, cols, n_cols);
  const std::vector<u64> w = detail::fused_words(leaves, cols, n_cols);
  for (std::size_t r = 0; r < 3; ++r) {
    for (std::size_t t = 0; t < towers; ++t) {
      NativePoly tower = elems[r].GetElementAtIndex(t);
      NativeVector v(n, tower.GetModulus());
      const u64* src = w.data() + (r * towers + t) * n;
      for (std::size_t i = 0; i < n; ++i) v[i] = NativeInteger(src[i]);
      tower.SetValues(std::move(v), Format::EVALUATION);
      elems[r].SetElementAtIndex(t, std::move(tower));
    }
  }
  return res;
}

// Relinearize, rescale, add the plaintext penalty (one value per slot, or empty).
inline Ct score_finish(const CryptoContext<DCRTPoly>& cc, const Ct& sum,
                       const std::vector<double>& penalty) {
  Ct res = cc->Rescale(cc->Relinearize(sum));
  if (!penalty.empty()) {
    Plaintext pt = cc->MakeCKKSPackedPlaintext(penalty, 1, res->GetLevel());
    res = cc->EvalAdd(res, pt);
  }
  return res;
}

}  // namespace ombra::native
