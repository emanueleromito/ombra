// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Experiment 0003: the local-mode (CPU) score, sum_j leaf_j * column_j, three ways.
//
//   baseline : what client/native/ombra_native.cpp does today: per column EvalMultNoRelin then
//              EvalAdd, per OpenMP thread, then the partial sums added.
//   fused    : one pass over the raw RNS words. Per word position the three components
//              (s0 q0, s0 q1 + s1 q0, s1 q1) are accumulated over j as 128-bit sums and reduced
//              (Montgomery REDC) once every `fold` columns, `fold` being the largest count that
//              keeps the sum in REDC's domain for that tower's modulus.
//   karatsuba: as fused, with the cross term from (s0 + s1)(q0 + q1) - s0 q0 - s1 q1, 3 wide
//              products per column instead of 4.
// Exactness: every word of the 3-component result is compared with the baseline's (OpenFHE's own
// arithmetic), and the process exits non-zero on any difference. Relinearize and rescale are left
// to OpenFHE in all three, so they are not part of the comparison.
//
// Usage: fused_score <preset.json fields...> see run.py. Arguments:
//   <ring_dim> <depth> <scaling_bits> <first_bits> <batch> <num_large_digits or 0> <d> <reps>
//   <seed>
// Synthetic data, keys made in-process and never written.

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"

using namespace lbcrypto;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
using Ct = Ciphertext<DCRTPoly>;
using Clock = std::chrono::steady_clock;

namespace {

static_assert(sizeof(NativeInteger) == sizeof(u64));

const u64* words(const DCRTPoly& p, std::size_t tower) {
  return reinterpret_cast<const u64*>(&p.GetAllElements()[tower].GetValues()[0]);
}

// Montgomery constants of one tower (R = 2^64), as in core/kernels/modarith.cuh.
struct Mod {
  u64 q;
  u64 qinv;  // q^-1 mod 2^64
  u64 r2;    // 2^128 mod q
  u64 fold;  // columns that may be summed before a reduction, see fold_interval
};
// REDC: x * R^-1 mod q for x < q * 2^64 (proof in modarith.cuh, mont_mul).
inline u64 redc(u128 x, const Mod& m) {
  const u64 mm = (u64)x * m.qinv;
  const u64 hi = (u64)(x >> 64);
  const u64 mq = (u64)(((u128)mm * m.q) >> 64);
  return hi >= mq ? hi - mq : hi + m.q - mq;
}
// One column adds at most (2q)^2 = 4 q^2 to an accumulator (the Karatsuba term; the plain cross
// term adds 2 q^2). REDC needs the sum below q * 2^64, so K columns are safe when
// K * 4 q^2 < q * 2^64, i.e. K < 2^62 / q. For q < 2^60 that is at least 4; for a 50-bit tower
// 4096.
u64 fold_interval(u64 q) {
  const u64 k = ((u64)1 << 62) / q;  // floor; K * 4 q^2 <= 2^62 * 4 q = q * 2^64 needs K < 2^62/q
  const u64 safe = k * q == ((u64)1 << 62) ? k - 1 : k;
  return safe > 256 ? 256 : safe;
}
Mod make_mod(u64 q) {
  Mod m;
  m.q = q;
  m.qinv = q;
  for (int i = 0; i < 6; ++i) m.qinv *= 2 - q * m.qinv;
  const u64 r1 = (u64)(((u128)1 << 64) % q);
  m.r2 = (u64)((u128)r1 * r1 % q);
  m.fold = fold_interval(q);
  return m;
}
inline u64 add_mod(u64 a, u64 b, u64 q) {
  const u64 r = a + b;
  return r >= q ? r - q : r;
}

std::vector<u64> baseline_words(const Ct& c) {
  std::vector<u64> out;
  for (const auto& e : c->GetElements()) {
    for (std::size_t t = 0; t < e.GetNumOfElements(); ++t) {
      const u64* w = words(e, t);
      out.insert(out.end(), w, w + e.GetRingDimension());
    }
  }
  return out;
}

Ct baseline(const CryptoContext<DCRTPoly>& cc, const std::vector<Ct>& leaves,
            const std::vector<Ct>& cols) {
  const long n = (long)cols.size();
  const int threads = omp_get_max_threads();
  std::vector<Ct> partial((std::size_t)threads);
#pragma omp parallel num_threads(threads)
  {
    Ct acc;
#pragma omp for schedule(static)
    for (long j = 0; j < n; ++j) {
      Ct t = cc->EvalMultNoRelin(leaves[j], cols[j]);
      acc = acc ? cc->EvalAdd(acc, t) : t;
    }
    partial[(std::size_t)omp_get_thread_num()] = acc;
  }
  Ct acc;
  for (auto& p : partial)
    if (p) acc = acc ? cc->EvalAdd(acc, p) : p;
  return acc;
}

// out: [3][towers][N]. kFold products of q^2 < 2^120 fit a 128-bit sum when kFold <= 2^8; the
// cross term adds two products per column, so it folds twice as often.
template <bool kKaratsuba>
std::vector<u64> fused(const std::vector<Ct>& leaves, const std::vector<Ct>& cols) {
  const auto& e0 = cols[0]->GetElements()[0];
  const std::size_t towers = e0.GetNumOfElements(), n = e0.GetRingDimension(), d = cols.size();
  std::vector<u64> out(3 * towers * n);
  std::vector<Mod> mods;
  for (std::size_t t = 0; t < towers; ++t)
    mods.push_back(make_mod(e0.GetAllElements()[t].GetModulus().ConvertToInt<u64>()));
  // Pointers [j][comp] per tower are fetched once per (tower, chunk).
  constexpr std::size_t kChunk = 1024;  // words per task: 2 * d * 8 KB of input stays in L2
#pragma omp parallel for collapse(2) schedule(static)
  for (std::size_t t = 0; t < towers; ++t) {
    for (std::size_t c0 = 0; c0 < n; c0 += kChunk) {
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
        if constexpr (kKaratsuba) {
          for (std::size_t k = 0; k < kChunk; ++k) {
            a0[k] += (u128)s0[k] * q0[k];
            a2[k] += (u128)s1[k] * q1[k];
            a1[k] += (u128)(s0[k] + s1[k]) * (q0[k] + q1[k]);  // each sum < 2^61
          }
        } else {
          for (std::size_t k = 0; k < kChunk; ++k) {
            a0[k] += (u128)s0[k] * q0[k];
            a2[k] += (u128)s1[k] * q1[k];
            a1[k] += (u128)s0[k] * q1[k] + (u128)s1[k] * q0[k];
          }
        }
        if (++pending == m.fold) fold();
      }
      if (pending != 0) fold();
      u64* o = out.data() + t * n + c0;
      for (std::size_t k = 0; k < kChunk; ++k) {
        r0[k] = redc((u128)r0[k] * m.r2, m);  // times R: back to the standard residue
        r1[k] = redc((u128)r1[k] * m.r2, m);
        r2[k] = redc((u128)r2[k] * m.r2, m);
        o[k] = r0[k];
        o[2 * towers * n + k] = r2[k];
        if constexpr (kKaratsuba) {
          // cross = K - a0 - a2 (mod q)
          u64 x = r1[k] >= r0[k] ? r1[k] - r0[k] : r1[k] + m.q - r0[k];
          x = x >= r2[k] ? x - r2[k] : x + m.q - r2[k];
          o[towers * n + k] = x;
        } else {
          o[towers * n + k] = r1[k];
        }
      }
    }
  }
  return out;
}

template <class F>
double median_ms(F&& f, int reps) {
  std::vector<double> s;
  for (int r = 0; r < reps + 1; ++r) {
    const auto t0 = Clock::now();
    f();
    if (r > 0) s.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  }
  std::sort(s.begin(), s.end());
  return s[s.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 10) {
    std::fprintf(stderr, "usage: fused_score N depth scale first batch digits d reps seed\n");
    return 2;
  }
  const u64 ring = std::stoull(argv[1]);
  CCParams<CryptoContextCKKSRNS> p;
  p.SetSecurityLevel(HEStd_128_classic);
  p.SetRingDim(ring);
  p.SetMultiplicativeDepth(std::stoul(argv[2]));
  p.SetScalingModSize(std::stoul(argv[3]));
  p.SetFirstModSize(std::stoul(argv[4]));
  p.SetBatchSize(std::stoul(argv[5]));
  p.SetScalingTechnique(FIXEDMANUAL);
  p.SetKeySwitchTechnique(HYBRID);
  if (std::stoul(argv[6]) != 0) p.SetNumLargeDigits(std::stoul(argv[6]));
  const std::size_t d = std::stoul(argv[7]);
  const int reps = std::stoi(argv[8]);
  const u64 seed = std::stoull(argv[9]);
  auto cc = GenCryptoContext(p);
  ombra::core::require_security_floor(cc);  // hard rule 1: never below the 128-bit floor
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();

  // d distinct columns and leaves would take minutes to encrypt at d = 1536; the arithmetic does
  // not depend on the values, so a pool of kPool distinct ciphertexts is cycled.
  constexpr std::size_t kPool = 16;
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<Ct> pool_c, pool_l;
  for (std::size_t i = 0; i < 2 * kPool; ++i) {
    std::vector<double> v(ring / 2);
    for (auto& x : v) x = dist(rng);
    auto ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
    (i < kPool ? pool_c : pool_l).push_back(ct);
  }
  std::vector<Ct> cols(d), leaves(d);
  for (std::size_t j = 0; j < d; ++j) {
    cols[j] = pool_c[j % kPool];
    leaves[j] = pool_l[(j * 7 + 3) % kPool];
  }

  const Ct ref = baseline(cc, leaves, cols);
  const std::vector<u64> want = baseline_words(ref);
  const std::vector<u64> got_f = fused<false>(leaves, cols);
  const std::vector<u64> got_k = fused<true>(leaves, cols);
  const bool ok_f = got_f == want, ok_k = got_k == want;

  const double t_base = median_ms([&] { baseline(cc, leaves, cols); }, reps);
  const double t_fused = median_ms([&] { fused<false>(leaves, cols); }, reps);
  const double t_kara = median_ms([&] { fused<true>(leaves, cols); }, reps);
  std::printf(
      "{\"ring_dim\": %llu, \"d\": %zu, \"threads\": %d, \"towers\": %zu, \"words_compared\": %zu, "
      "\"fused_exact\": %s, \"karatsuba_exact\": %s, \"baseline_ms\": %.3f, \"fused_ms\": %.3f, "
      "\"karatsuba_ms\": %.3f}\n",
      (unsigned long long)ring, d, omp_get_max_threads(),
      (std::size_t)cols[0]->GetElements()[0].GetNumOfElements(), want.size(),
      ok_f ? "true" : "false", ok_k ? "true" : "false", t_base, t_fused, t_kara);
  return ok_f && ok_k ? 0 : 1;
}
