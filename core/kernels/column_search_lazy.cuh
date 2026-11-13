// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Arithmetic of the lazy column-search kernel (column_search_lazy_kernel in column_search.cu),
// as host + device functions so that core/src/column_search_lazy_test.cpp runs the very same source
// lines on the CPU against a plain 128-bit reference. Header only; includes modarith.cuh.
//
// One (stored block, word position) accumulates, over the dimensions j, the three components of the
// tensor product of s = (s0, s1) and q = (q0, q1):
//   a0 += s0 * q0                      (component 0)
//   a2 += s1 * q1                      (component 2)
//   a1 += (s0 + s1) * (q0 + q1)        (Karatsuba: = s0q0 + s0q1 + s1q0 + s1q1)
// as raw 128-bit sums, with NO modular reduction in the inner loop. Every `fold` dimensions (and
// once at the end) each a_i is reduced by Montgomery REDC and added to a 64-bit residue:
//   r_i = (r_i + redc(a_i)) mod q;  a_i = 0           (redc(x) = x * 2^-64 mod q)
// At the end cross = r1 - r0 - r2 = 2^-64 * sum_j (s0 q1 + s1 q0), and the outputs are
// mont_finish(r0), mont_finish(cross), mont_finish(r2): exactly sum_j s0 q0, sum_j (s0 q1 + s1 q0),
// sum_j s1 q1 mod q, the same residues as the Montgomery kernel and as OpenFHE.
//
// Why the 128-bit sums cannot overflow and REDC stays exact (the fold interval). Let q < 2^n with n
// = c.n, all inputs reduced (< q). One dimension adds at most (2q)^2 = 4 q^2 to an accumulator
// (the Karatsuba term; its factors are s0 + s1 < 2q and q0 + q1 < 2q; the others add q^2).
// mont_redc needs the accumulated value < q * 2^64. After K dimensions an accumulator is below K *
// 4 q^2. With K = 2^(62 - n):
//   K * 4 q^2 = 2^(64 - n) * q^2 < 2^(64 - n) * q * 2^n = q * 2^64        (strict, q < 2^n),
// so REDC is exact, and the sum is below q * 2^64 < 2^124, which fits 128 bits. make_tower_consts
// guarantees n <= 60, hence 62 - n >= 2: K >= 4 for the largest towers. K is capped at 256 (more
// dimensions per fold would save almost nothing). For n = 60, K = 4; for n = 50, K = 256.
#pragma once

#include "modarith.cuh"

namespace ombra::core {

// Dimensions that may be accumulated before a fold: min(2^(62 - n), 256), see above. c.n <= 60.
__host__ __device__ inline u64 lazy_fold_interval(const TowerConst& c) {
  const u64 shift = 62 - c.n;
  return shift >= 8 ? (u64)256 : ((u64)1 << shift);
}

struct LazyAcc {
  u128 a0, a1, a2;  // raw sums of products, below q * 2^64 between folds
  u64 r0, r1, r2;   // 2^-64 * (folded part of the sums) mod q
};

__host__ __device__ inline LazyAcc lazy_zero() { return LazyAcc{0, 0, 0, 0, 0, 0}; }

// One dimension: s0 + s1 and q0 + q1 are below 2q < 2^61, so the product is below 2^122.
__host__ __device__ inline void lazy_mac(LazyAcc& a, u64 s0, u64 s1, u64 q0, u64 q1) {
  a.a0 += (u128)s0 * q0;
  a.a2 += (u128)s1 * q1;
  a.a1 += (u128)(s0 + s1) * (q0 + q1);
}

__host__ __device__ inline void lazy_fold(LazyAcc& a, const TowerConst& c) {
  a.r0 = add_mod(a.r0, mont_redc(a.a0, c), c.q);
  a.r1 = add_mod(a.r1, mont_redc(a.a1, c), c.q);
  a.r2 = add_mod(a.r2, mont_redc(a.a2, c), c.q);
  a.a0 = a.a1 = a.a2 = 0;
}

// The three output words (standard-form residues). Call lazy_fold first (the sums must be empty).
__host__ __device__ inline void lazy_finish(const LazyAcc& a, const TowerConst& c, u64& out0,
                                            u64& out1, u64& out2) {
  const u64 cross = sub_mod(sub_mod(a.r1, a.r0, c.q), a.r2, c.q);
  out0 = mont_finish(a.r0, c);
  out1 = mont_finish(cross, c);
  out2 = mont_finish(a.r2, c);
}

}  // namespace ombra::core
