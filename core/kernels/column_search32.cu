// Kernel family: column-packed encrypted search with 32-bit limbs (T1.5).
//
// The same computation as column_search.cu (see the packing and the three output polynomials
// there), for parameter sets whose RNS primes all fit a 32-bit limb: OpenFHE composite scaling with
// register word size 32 (preset m1_composite_n16384), where every Q prime is below 2^31. The stored
// and query words are held as 32-bit words on the device (half the bytes per word), the products
// use native 32 x 32 -> 64-bit multiplication, and the output stays 64-bit words so that the
// relinearization, rescale and mask steps (64-bit words holding the same residues) read it
// unchanged. The result is therefore word for word the 64-bit kernel's result on the same residues
// and OpenFHE's.
//
// Memory layout (device, P = num_towers * ring_dim words per polynomial):
//   stored: u32 [block][dim][component 0..1][tower][coeff]    blocks * dims * 2 * P words
//   query:  u32 [dim][component 0..1][tower][coeff]           dims * 2 * P words
//   out:    u64 [block][component 0..2][tower][coeff]         blocks * 3 * P words
// All input words must be reduced (< their modulus); moduli odd and below 2^31.
//
// Arithmetic: Montgomery with R = 2^32 (mont_mul32 in modarith.cuh): the kernel accumulates
// acc = sum_j mont_mul32(x_j, y_j) = R^-1 * sum_j x_j y_j (mod q) with a reduced addition each time,
// then multiplies once by R (mont_finish32), so inputs and outputs are standard residues.

#include <algorithm>

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr int kBlockThreads = 256;
// Stored blocks per thread: the query words are loaded once and reused for each, as in the 64-bit
// kernel.
constexpr int kBlockTile = 4;

// Launch shape: grid.x = ceil(P / kBlockThreads) (one thread per output word of a polynomial),
// grid.y = ceil(blocks / kBlockTile) (<= 65535), block = kBlockThreads threads, no shared memory.
// Each thread keeps 3 * kBlockTile 32-bit accumulators and loops over the d dimensions: per
// dimension and block 4 Montgomery products and 3 modular additions. The tower constant (12 bytes)
// is read once per thread.
template <int MinBlocks>
__global__ void __launch_bounds__(kBlockThreads, MinBlocks)
    column_search32_kernel(const u32* __restrict__ stored, const u32* __restrict__ query,
                           u64* __restrict__ out, const TowerConst32* __restrict__ tc, u64 dims,
                           u64 blocks, u64 ring_dim, u64 poly_words) {
  const u64 e = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (e >= poly_words) return;
  const TowerConst32 c = tc[e / ring_dim];
  const u64 b0 = (u64)blockIdx.y * kBlockTile;
  const u64 nb = min((u64)kBlockTile, blocks - b0);

  u32 acc0[kBlockTile], acc1[kBlockTile], acc2[kBlockTile];
#pragma unroll
  for (int k = 0; k < kBlockTile; ++k) acc0[k] = acc1[k] = acc2[k] = 0;

  for (u64 j = 0; j < dims; ++j) {
    const u32 q0 = query[(j * 2) * poly_words + e];
    const u32 q1 = query[(j * 2 + 1) * poly_words + e];
#pragma unroll
    for (int k = 0; k < kBlockTile; ++k) {
      if ((u64)k < nb) {
        const u64 base = (((b0 + k) * dims + j) * 2) * poly_words + e;
        const u32 s0 = stored[base];
        const u32 s1 = stored[base + poly_words];
        acc0[k] = add_mod32(acc0[k], mont_mul32(s0, q0, c), c.q);
        const u32 cross = add_mod32(mont_mul32(s0, q1, c), mont_mul32(s1, q0, c), c.q);
        acc1[k] = add_mod32(acc1[k], cross, c.q);
        acc2[k] = add_mod32(acc2[k], mont_mul32(s1, q1, c), c.q);
      }
    }
  }
#pragma unroll
  for (int k = 0; k < kBlockTile; ++k) {
    if ((u64)k < nb) {
      const u64 o = ((b0 + k) * 3) * poly_words + e;
      out[o] = mont_finish32(acc0[k], c);
      out[o + poly_words] = mont_finish32(acc1[k], c);
      out[o + 2 * poly_words] = mont_finish32(acc2[k], c);
    }
  }
}

// dst[i] = (u32)src[i] for i < count. Launch shape: grid = ceil(count / 256), block = 256. The
// caller guarantees every word is below 2^32 (reduced residues of 32-bit moduli).
__global__ void narrow_kernel(const u64* __restrict__ src, u32* __restrict__ dst, u64 count) {
  const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i < count) dst[i] = (u32)src[i];
}

}  // namespace

cudaError_t launch_column_search32(const u32* stored, const u32* query, u64* out,
                                   const TowerConst32* tc, const ColumnSearchDims& d,
                                   int min_blocks, cudaStream_t stream) {
  const u64 poly_words = d.num_towers * d.ring_dim;
  const u64 grid_y = (d.blocks + kBlockTile - 1) / kBlockTile;
  if (poly_words == 0 || d.dims == 0 || d.blocks == 0 || grid_y > 65535) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((unsigned)((poly_words + kBlockThreads - 1) / kBlockThreads), (unsigned)grid_y);
  switch (min_blocks) {
    case 4:
      column_search32_kernel<4><<<grid, kBlockThreads, 0, stream>>>(
          stored, query, out, tc, d.dims, d.blocks, d.ring_dim, poly_words);
      break;
    case 6:
      column_search32_kernel<6><<<grid, kBlockThreads, 0, stream>>>(
          stored, query, out, tc, d.dims, d.blocks, d.ring_dim, poly_words);
      break;
    case 8:
      column_search32_kernel<8><<<grid, kBlockThreads, 0, stream>>>(
          stored, query, out, tc, d.dims, d.blocks, d.ring_dim, poly_words);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_narrow_u64_to_u32(const u64* src, u32* dst, u64 count, cudaStream_t stream) {
  if (count == 0) return cudaSuccess;
  narrow_kernel<<<(unsigned)((count + 255) / 256), 256, 0, stream>>>(src, dst, count);
  return cudaGetLastError();
}

}  // namespace ombra::core
