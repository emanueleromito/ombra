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
    u64 n;  // bit length of q
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

// (a + b) mod q for reduced a, b < q < 2^60 (a + b < 2^61, one conditional subtraction).
__host__ __device__ inline u64 add_mod(u64 a, u64 b, u64 q) {
    const u64 r = a + b;
    return r >= q ? r - q : r;
}

// (a - b) mod q for reduced a, b < q.
__host__ __device__ inline u64 sub_mod(u64 a, u64 b, u64 q) { return a >= b ? a - b : a + q - b; }

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
        out.push_back({q, mu, n});
    }
    return out;
}

}  // namespace ombra::core
