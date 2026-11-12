// Kernel family: CKKS rescale and ciphertext x ciphertext tensor product.
//
// Mirrors OpenFHE v1.6.1, FIXEDMANUAL scaling, evaluation-form RNS words (exact modular
// arithmetic, so every output word equals OpenFHE's):
//
//  Rescale = LeveledSHECKKSRNS::ModReduceInternalInPlace (ckksrns-leveledshe.cpp:188), which calls
//  DCRTPoly::DropLastElementAndScale(qlInvModq) on each component (dcrtpoly-impl.h:737):
//    last = INTT_{q_l}(c[l])                                    (coefficient form)
//    for i < l:  tmp_i = NTT_{q_i}( centered(last) mod q_i )     centered(x) = x - q_l if x > q_l /
//    2
//                out_i = (c[i] - tmp_i) * q_l^-1 mod q_i
//  The NTTs run in ntt.cu; the two element-wise steps are the kernels below. The level, noise-scale
//  degree and scaling factor metadata are the host's business (expansion_gpu / gpu_search).
//
//  Tensor product = LeveledSHEBase::EvalMultCore (base-leveledshe.cpp:621) for two 2-component
//  ciphertexts: out0 = a0 b0, out1 = a0 b1 + a1 b0, out2 = a1 b1. Relinearization (3 -> 2
//  components) reuses the key-switching kernels in keyswitch.cu and ntt.cu.
//
// All inputs are reduced words, moduli < 2^60. Layouts (device, 64-bit words):
//   ciphertext: [component][tower][coeff].  Ciphertexts of a batch are reached through pointer
//   arrays; scratch is contiguous per (node, component).

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr u32 kThreads = 256;

// tmp[z][i][k] = centered(last[z][k]) mod q_i for i < num_q - 1, where q_l = tc[num_q - 1].q and
// `last` holds the coefficient-form words of the dropped tower (each < q_l).
// Launch shape: grid = (ceil(N / kThreads), num_q - 1, nodes2), block = kThreads, no shared memory;
// one thread per (z, tower i, coefficient k). nodes2 = 2 * nodes (z = node * 2 + component).
__global__ void rescale_switch_kernel(const u64* __restrict__ last, u64* __restrict__ tmp,
                                      const TowerConst* __restrict__ tc, u64 n, u64 num_q) {
  const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const u64 i = blockIdx.y, z = blockIdx.z;
  const u64 ql = tc[num_q - 1].q, qi = tc[i].q;
  const u64 x = last[z * n + k];
  u64 r = x % qi;
  if (x > (ql >> 1)) r = sub_mod(r, ql % qi, qi);  // x - q_l
  tmp[(z * (num_q - 1) + i) * n + k] = r;
}

// out[node][r][i][k] = (in[node][r][i][k] - tmp[z][i][k]) * ql_inv[i] mod q_i, z = node * 2 + r,
// i < num_q - 1. in: [2][num_q][N] per node; out: [2][num_q - 1][N] per node.
// Launch shape: grid = (ceil(N / kThreads), num_q - 1, 2 * nodes), block = kThreads, no shared
// memory; one thread per output word. `out` must not alias `in`.
__global__ void rescale_combine_kernel(const u64* const* __restrict__ in,
                                       const u64* __restrict__ tmp, u64* const* __restrict__ out,
                                       const u64* __restrict__ ql_inv,
                                       const TowerConst* __restrict__ tc, u64 n, u64 num_q) {
  const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const u64 i = blockIdx.y, z = blockIdx.z;
  const u64 node = z >> 1, r = z & 1;
  const TowerConst c = tc[i];
  const u64 v = in[node][(r * num_q + i) * n + k];
  const u64 t = tmp[(z * (num_q - 1) + i) * n + k];
  out[node][(r * (num_q - 1) + i) * n + k] = mul_mod(sub_mod(v, t, c.q), ql_inv[i], c);
}

// out[node] = [a0 b0, a0 b1 + a1 b0, a1 b1] tower by tower. a, b: [2][num_q][N]; out:
// [3][num_q][N]. Launch shape: grid = (ceil(N / kThreads), num_q, nodes), block = kThreads, no
// shared memory; one thread per (node, tower, coefficient), 4 modular multiplications each.
__global__ void ct_tensor_kernel(const u64* const* __restrict__ a, const u64* const* __restrict__ b,
                                 u64* const* __restrict__ out, const TowerConst* __restrict__ tc,
                                 u64 n, u64 num_q) {
  const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  const u64 i = blockIdx.y, z = blockIdx.z;
  const TowerConst c = tc[i];
  const u64 a0 = a[z][i * n + k], a1 = a[z][(num_q + i) * n + k];
  const u64 b0 = b[z][i * n + k], b1 = b[z][(num_q + i) * n + k];
  u64* o = out[z];
  o[i * n + k] = mul_mod(a0, b0, c);
  o[(num_q + i) * n + k] = add_mod(mul_mod(a0, b1, c), mul_mod(a1, b0, c), c.q);
  o[(2 * num_q + i) * n + k] = mul_mod(a1, b1, c);
}

bool valid_dims(const RescaleDims& d, u32 nodes, u64 min_q) {
  return d.ring_dim != 0 && d.num_q >= min_q && d.num_q <= 65535 && nodes != 0 && nodes <= 65535;
}

dim3 grid_for(const RescaleDims& d, u64 towers, u32 z) {
  return dim3((unsigned)((d.ring_dim + kThreads - 1) / kThreads), (unsigned)towers, z);
}

}  // namespace

cudaError_t launch_rescale_switch(const u64* last, u64* tmp, const TowerConst* tc,
                                  const RescaleDims& dims, u32 nodes2, cudaStream_t stream) {
  if (!valid_dims(dims, nodes2, 2)) return cudaErrorInvalidValue;
  rescale_switch_kernel<<<grid_for(dims, dims.num_q - 1, nodes2), kThreads, 0, stream>>>(
      last, tmp, tc, dims.ring_dim, dims.num_q);
  return cudaGetLastError();
}

cudaError_t launch_rescale_combine(const u64* const* in, const u64* tmp, u64* const* out,
                                   const u64* ql_inv, const TowerConst* tc, const RescaleDims& dims,
                                   u32 nodes, cudaStream_t stream) {
  if (!valid_dims(dims, nodes, 2) || nodes > 32767) return cudaErrorInvalidValue;
  rescale_combine_kernel<<<grid_for(dims, dims.num_q - 1, 2 * nodes), kThreads, 0, stream>>>(
      in, tmp, out, ql_inv, tc, dims.ring_dim, dims.num_q);
  return cudaGetLastError();
}

cudaError_t launch_ct_tensor(const u64* const* a, const u64* const* b, u64* const* out,
                             const TowerConst* tc, const RescaleDims& dims, u32 nodes,
                             cudaStream_t stream) {
  if (!valid_dims(dims, nodes, 1)) return cudaErrorInvalidValue;
  ct_tensor_kernel<<<grid_for(dims, dims.num_q, nodes), kThreads, 0, stream>>>(
      a, b, out, tc, dims.ring_dim, dims.num_q);
  return cudaGetLastError();
}

}  // namespace ombra::core
