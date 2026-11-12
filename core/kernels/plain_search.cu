// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Kernel family: ciphertext x plaintext column-packed search.
//
// Packing as in column_search.cu: one plaintext per dimension j per block, one vector per slot;
// the pre-expanded query is d two-component ciphertexts (c0_j, c1_j), the j-th holding q_j in
// every slot. Per stored block b, in evaluation form, tower by tower (moduli from the ciphertexts):
//   out_b[0] = sum_j p_{b,j} * c0_j
//   out_b[1] = sum_j p_{b,j} * c1_j
// which is OpenFHE's EvalMult(ciphertext, plaintext) (LeveledSHERNS::EvalMultInPlace,
// rns-leveledshe.cpp:263: both components multiplied by the plaintext) accumulated with EvalAdd.
// The result has 2 components, so no relinearization is needed; the scale is the product of the
// two scales (rescale left to the caller).
//
// Memory layout (device, 64-bit words; P = num_towers * ring_dim words per polynomial):
//   plain: [block][dim][tower][coeff]           blocks * dims * P words
//   query: [dim][component 0..1][tower][coeff]  dims * 2 * P words
//   out:   [block][component 0..1][tower][coeff] blocks * 2 * P words
// All input words must be reduced (< their modulus); moduli < 2^60 and odd.
// Products are Montgomery products accumulated and corrected once at the end (mont_finish), exactly
// as in column_search.cu; the output is the standard-form residue.

#include <algorithm>

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr int kBlockThreads = 256;
constexpr int kBlockTile = 4;  // stored blocks per thread: the query words are loaded once

// Launch shape: grid.x = ceil(P / kBlockThreads), grid.y = ceil(blocks / kBlockTile) (<= 65535),
// block = kBlockThreads, no shared memory. 2 * kBlockTile accumulators per thread; per dimension
// and block it does 2 modular multiplications and 2 additions.
__global__ void __launch_bounds__(kBlockThreads, 4)
    plain_search_kernel(const u64* __restrict__ plain, const u64* __restrict__ query,
                        u64* __restrict__ out, const TowerConst* __restrict__ tc, u64 dims,
                        u64 blocks, u64 ring_dim, u64 poly_words) {
  const u64 e = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (e >= poly_words) return;
  const TowerConst c = tc[e / ring_dim];
  const u64 b0 = (u64)blockIdx.y * kBlockTile;
  const u64 nb = min((u64)kBlockTile, blocks - b0);

  u64 acc0[kBlockTile], acc1[kBlockTile];
#pragma unroll
  for (int k = 0; k < kBlockTile; ++k) acc0[k] = acc1[k] = 0;

  for (u64 j = 0; j < dims; ++j) {
    const u64 q0 = query[(j * 2) * poly_words + e];
    const u64 q1 = query[(j * 2 + 1) * poly_words + e];
#pragma unroll
    for (int k = 0; k < kBlockTile; ++k) {
      if ((u64)k < nb) {
        const u64 p = plain[((b0 + k) * dims + j) * poly_words + e];
        acc0[k] = add_mod(acc0[k], mont_mul(p, q0, c), c.q);
        acc1[k] = add_mod(acc1[k], mont_mul(p, q1, c), c.q);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kBlockTile; ++k) {
    if ((u64)k < nb) {
      const u64 o = ((b0 + k) * 2) * poly_words + e;
      out[o] = mont_finish(acc0[k], c);
      out[o + poly_words] = mont_finish(acc1[k], c);
    }
  }
}

}  // namespace

cudaError_t launch_plain_search(const u64* plain, const u64* query, u64* out, const TowerConst* tc,
                                u64 dims, u64 blocks, u64 num_towers, u64 ring_dim,
                                cudaStream_t stream) {
  const u64 poly_words = num_towers * ring_dim;
  const u64 grid_y = (blocks + kBlockTile - 1) / kBlockTile;
  if (poly_words == 0 || dims == 0 || blocks == 0 || grid_y > 65535) return cudaErrorInvalidValue;
  const dim3 grid((unsigned)((poly_words + kBlockThreads - 1) / kBlockThreads), (unsigned)grid_y);
  plain_search_kernel<<<grid, kBlockThreads, 0, stream>>>(plain, query, out, tc, dims, blocks,
                                                          ring_dim, poly_words);
  return cudaGetLastError();
}

}  // namespace ombra::core
