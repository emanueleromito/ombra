// Shared modular arithmetic for the RNS kernels (CUDA headers only; no OpenFHE).
//
// Representation (OpenFHE v1.6.1, NATIVE_SIZE=64 build): one coefficient = one uint64_t word,
// every tower has its own modulus q < 2^60 taken from the ciphertext, never hard-coded here.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ombra::core {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// Per-tower constants for Barrett reduction. For q with bit length n (2 <= q < 2^n, n <= 60):
// mu = floor(2^(2n) / q) < 2^(n+1) <= 2^61, so it fits a 64-bit word.
struct TowerConst {
  u64 q;
  u64 mu;
  u64 n;     // bit length of q
  u64 qinv;  // q^-1 mod 2^64 (Montgomery); 0 if q is even (Montgomery then unusable)
  u64 r2;    // 2^128 mod q (Montgomery); 0 if q is even
};

// x mod q for x < q^2, by Barrett reduction (HAC 14.42, b=2, k=n):
//   q1 = x >> (n-1); q3 = (q1 * mu) >> (n+1); r = x - q3*q  (0 <= r < 3q), then up to 2 subtracts.
// Correct for any q < 2^60; r < 3q < 2^62 so the low 64 bits of x - q3*q are exact.
__host__ __device__ inline u64 barrett_reduce(u128 x, const TowerConst& c) {
  const u64 q1 = (u64)(x >> (c.n - 1));
  const u64 q3 = (u64)(((u128)q1 * c.mu) >> (c.n + 1));
  u64 r = (u64)x - q3 * c.q;
  if (r >= c.q) r -= c.q;
  if (r >= c.q) r -= c.q;
  return r;
}

// (a * b) mod q for reduced a, b < q.
__host__ __device__ inline u64 mul_mod(u64 a, u64 b, const TowerConst& c) {
  return barrett_reduce((u128)a * b, c);
}

// Montgomery product with R = 2^64: returns a * b * R^-1 mod q, for a * b < q * 2^64 (so for any
// a < q, b < 2^64, and in particular a, b < q), q odd, q < 2^63, c.qinv = q^-1 mod 2^64.
// Proof of exactness: t = a * b < q * R. m = (t mod R) * qinv mod R gives m * q == t (mod R), hence
// t - m * q is divisible by R and equals R * (hi(t) - hi(m * q)) where hi(x) = floor(x / R) (the low
// halves are equal). hi(t) < q because t < q * R, and hi(m * q) < q because m < R, so the difference
// lies in (-q, q) and one conditional add of q gives the unique residue in [0, q) of t * R^-1.
// Cost: 2 full 64x64 products (a*b: lo and hi; m*q: hi only) and one low product, against 5 products
// plus two conditional subtractions for Barrett.
__host__ __device__ inline u64 mont_mul(u64 a, u64 b, const TowerConst& c) {
  const u128 t = (u128)a * b;
  const u64 m = (u64)t * c.qinv;
  const u64 hi = (u64)(t >> 64);
  const u64 mq = (u64)(((u128)m * c.q) >> 64);
  return hi >= mq ? hi - mq : hi + c.q - mq;
}

// Sum-of-products convention used by the search kernels: they accumulate
//   acc = sum_j mont_mul(x_j, y_j) mod q = R^-1 * sum_j x_j y_j   (mod q)
// with add_mod, then multiply once by R: mont_mul(acc, c.r2) = acc * R^2 * R^-1 = acc * R, so the
// output is exactly sum_j x_j y_j mod q in standard form. Inputs and outputs of the kernels are
// therefore unchanged (no Montgomery-form data is ever stored or transferred).
__host__ __device__ inline u64 mont_finish(u64 acc, const TowerConst& c) {
  return mont_mul(acc, c.r2, c);
}

// Shoup multiplication by a fixed multiplier w < q < 2^63 with wq = floor(w * 2^64 / q)
// (shoup_quotient). For ANY x < 2^64: hi = floor(x * wq / 2^64) satisfies
// x*w/q - 1 - x/2^64 < hi <= x*w/q, so r = x*w - hi*q (computed mod 2^64, exact because
// r < 2q < 2^64) lies in [0, 2q), and after one conditional subtraction r = x*w mod q. The result
// is the unique residue, identical to Barrett's; x does not need to be reduced.
__host__ __device__ inline u64 mul_mod_shoup(u64 x, u64 w, u64 wq, u64 q) {
  const u64 hi = (u64)(((u128)x * wq) >> 64);
  const u64 r = x * w - hi * q;
  return r >= q ? r - q : r;
}

// Host: floor(w * 2^64 / q) for w < q.
inline u64 shoup_quotient(u64 w, u64 q) { return (u64)(((u128)w << 64) / q); }

// (a + b) mod q for reduced a, b < q < 2^60 (a + b < 2^61, one conditional subtraction).
__host__ __device__ inline u64 add_mod(u64 a, u64 b, u64 q) {
  const u64 r = a + b;
  return r >= q ? r - q : r;
}

// (a - b) mod q for reduced a, b < q.
__host__ __device__ inline u64 sub_mod(u64 a, u64 b, u64 q) { return a >= b ? a - b : a + q - b; }

// ---- 32-bit limbs (T1.5) ----
// For moduli q < 2^31 (odd): residues fit a u32 and the sum of two residues does not overflow it.
// Montgomery with R = 2^32.
struct TowerConst32 {
  u32 q;
  u32 qinv;  // q^-1 mod 2^32
  u32 r2;    // 2^64 mod q
};

// a * b * R^-1 mod q, R = 2^32, for a * b < q * 2^32 (any reduced a, b). Same argument as mont_mul:
// t = a * b < q * R, m = (t mod R) * qinv mod R, t - m * q is divisible by R, hi(t) < q and
// hi(m * q) < q, so one conditional add of q gives the residue in [0, q).
__host__ __device__ inline u32 mont_mul32(u32 a, u32 b, const TowerConst32& c) {
  const u64 t = (u64)a * b;
  const u32 m = (u32)t * c.qinv;
  const u32 hi = (u32)(t >> 32);
  const u32 mq = (u32)(((u64)m * c.q) >> 32);
  return hi >= mq ? hi - mq : hi + c.q - mq;
}

// acc * R mod q: turns sum_j mont_mul32(x_j, y_j) = R^-1 * sum_j x_j y_j into the standard residue.
__host__ __device__ inline u32 mont_finish32(u32 acc, const TowerConst32& c) {
  return mont_mul32(acc, c.r2, c);
}

// (a + b) mod q for reduced a, b < q < 2^31.
__host__ __device__ inline u32 add_mod32(u32 a, u32 b, u32 q) {
  const u32 r = a + b;
  return r >= q ? r - q : r;
}

// Host: constants for 32-bit limbs; throws std::invalid_argument for an even modulus or one >= 2^31.
inline std::vector<TowerConst32> make_tower_consts32(const std::vector<u64>& moduli) {
  std::vector<TowerConst32> out;
  for (u64 q : moduli) {
    if (q < 3 || q >= (u64(1) << 31) || (q & 1) == 0) {
      throw std::invalid_argument("rns32: modulus must be odd and satisfy 3 <= q < 2^31");
    }
    u32 qinv = (u32)q;  // Newton iteration, each step doubles the correct low bits
    for (int i = 0; i < 5; ++i) qinv *= 2 - (u32)q * qinv;
    const u64 r1 = (u64(1) << 32) % q;
    out.push_back({(u32)q, qinv, (u32)(r1 * r1 % q)});
  }
  return out;
}

// Host: builds the constants; throws std::invalid_argument for a modulus outside [2, 2^60).
inline std::vector<TowerConst> make_tower_consts(const std::vector<u64>& moduli) {
  std::vector<TowerConst> out;
  for (u64 q : moduli) {
    if (q < 2 || q >= (u64(1) << 60)) {
      throw std::invalid_argument("rns: modulus must satisfy 2 <= q < 2^60");
    }
    u64 n = 0;
    while ((u64(1) << n) <= q) ++n;  // q < 2^n
    const u64 mu = (u64)((u128(1) << (2 * n)) / q);
    u64 qinv = 0, r2 = 0;
    if (q & 1) {
      qinv = q;  // Newton iteration: q * q == 1 (mod 8), each step doubles the correct low bits
      for (int i = 0; i < 6; ++i) qinv *= 2 - q * qinv;
      const u64 r1 = (u64)((u128(1) << 64) % q);
      r2 = (u64)((u128)r1 * r1 % q);
    }
    out.push_back({q, mu, n, qinv, r2});
  }
  return out;
}

}  // namespace ombra::core
