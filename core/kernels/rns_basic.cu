// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Kernel family: pointwise modular add / multiply on RNS towers in evaluation (NTT) form.
//
// In evaluation form, polynomial multiplication mod (X^N+1) is a pointwise product per tower.
// Host wrappers (copies, validation, error conversion) are in core/src/rns_basic_host.cu.

#include <algorithm>

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr int kBlock = 256;

// Computes out[i] = (a[i] + b[i % b_period]) mod q(i) for i in [0, total).
// q(i) = tc[(i / ring_dim) % num_towers].q.  Inputs must be reduced (< q), so a + b < 2^61 and
// one conditional subtraction suffices; no overflow is possible for q < 2^60.
// Launch shape: 1-D grid of ceil(total / kBlock) blocks (capped, grid-stride loop), kBlock threads
// per block, no shared memory. Memory: a, b, out are device arrays of 64-bit words; tc has
// num_towers entries (read-only, global memory).
__global__ void rns_add_kernel(const u64* __restrict__ a, const u64* __restrict__ b,
                               u64* __restrict__ out, const TowerConst* __restrict__ tc, u64 total,
                               u64 ring_dim, u64 num_towers, u64 b_period) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < total;
       i += (u64)gridDim.x * blockDim.x) {
    const u64 q = tc[(i / ring_dim) % num_towers].q;
    out[i] = add_mod(a[i], b[i % b_period], q);
  }
}

// Computes out[i] = (a[i] * b[i % b_period]) mod q(i) for i in [0, total), q(i) as above.
// The 128-bit product is reduced with Barrett reduction (see modarith.cuh).
// Launch shape: as rns_add_kernel. Memory: as rns_add_kernel.
__global__ void rns_mul_kernel(const u64* __restrict__ a, const u64* __restrict__ b,
                               u64* __restrict__ out, const TowerConst* __restrict__ tc, u64 total,
                               u64 ring_dim, u64 num_towers, u64 b_period) {
  for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < total;
       i += (u64)gridDim.x * blockDim.x) {
    const TowerConst c = tc[(i / ring_dim) % num_towers];
    out[i] = mul_mod(a[i], b[i % b_period], c);
  }
}

unsigned grid_for(u64 total) {
  return (unsigned)std::min<u64>((total + kBlock - 1) / kBlock, 65535);
}

}  // namespace

cudaError_t launch_rns_add(const u64* a, const u64* b, u64* out, const TowerConst* tc, u64 total,
                           u64 ring_dim, u64 num_towers, u64 b_period, cudaStream_t stream) {
  rns_add_kernel<<<grid_for(total), kBlock, 0, stream>>>(a, b, out, tc, total, ring_dim, num_towers,
                                                         b_period);
  return cudaGetLastError();
}

cudaError_t launch_rns_mul(const u64* a, const u64* b, u64* out, const TowerConst* tc, u64 total,
                           u64 ring_dim, u64 num_towers, u64 b_period, cudaStream_t stream) {
  rns_mul_kernel<<<grid_for(total), kBlock, 0, stream>>>(a, b, out, tc, total, ring_dim, num_towers,
                                                         b_period);
  return cudaGetLastError();
}

}  // namespace ombra::core
