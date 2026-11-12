// Kernel family: hybrid key-switched automorphism and the expansion split.
//
// Mirrors OpenFHE v1.6.1 LeveledSHEBase::EvalAutomorphismCore (base-leveledshe.cpp:400), HYBRID
// branch, followed by the tree's EvalAdd / EvalSub and the multiplication by X^-k. All arithmetic
// is exact modular arithmetic on evaluation-form RNS words, so every output word equals OpenFHE's.
// Moduli, tables and key words come from the host (extracted from OpenFHE objects); nothing is
// hard-coded. "QP" below is the extended basis: the num_q Q towers of the ciphertext followed by
// the num_p special P towers. N is the ring dimension.
//
// The basis conversions (ModUp's and ModDown's ApproxSwitchCRTBasis, dcrtpoly-impl.h:938) and the
// final combine (ModDown's last line, EvalAdd, EvalSub, X^-k) are fused into the NTT kernels in
// ntt.cu: the conversion is computed while the forward NTT loads its input, the combine runs as the
// epilogue of the last forward NTT pass. This file holds the inner product and test aids.
//
// Every kernel works on a batch of nodes (blockIdx.z): all nodes of one tree level share the key,
// the tables and the monomial. Per-node data is contiguous with a fixed stride, ciphertexts are
// reached through pointer arrays.
//
// Layouts (device, 64-bit words), per node:
//   ct:     [component 0..1][Q tower][coeff]                   2 * num_q * N
//   digits: [digit][QP tower][coeff]                           digits * num_qp * N
//   key:    [digit][0 = b, 1 = a][QP tower][coeff]             digits * 2 * num_qp * N (shared)
//   ext:    [component 0..1][QP tower][coeff]                  2 * num_qp * N
// All inputs are reduced words, moduli < 2^60.

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr u32 kThreads = 256;

// Copies component 1 of each node's ciphertext into the digit slot that holds the tower itself (a
// digit's own towers are its input, unchanged). Oracle tests only.
// Launch shape: grid = (ceil(N / kThreads), num_q, batch), block = kThreads, no shared memory.
__global__ void ks_copy_own_kernel(const u64* const* __restrict__ ct, u64* __restrict__ digits,
                                   const u32* __restrict__ digit_of_tower, KsDims d) {
  const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= d.ring_dim) return;
  const u64 n = d.ring_dim, i = blockIdx.y, z = blockIdx.z;
  digits[((z * d.digits + digit_of_tower[i]) * d.num_qp + i) * n + k] =
      ct[z][(d.num_q + i) * n + k];
}

// Key-switch inner product with the evaluation key, the "+ c0 * P" term and the automorphism
// permutation, in one pass (EvalFastKeySwitchCoreExt, keyswitch-hybrid.cpp:389; then
// ba[0] += cv0 * P on the Q towers; then AutomorphismTransform with out[j] = in[auto_map[j]]).
// All three are pointwise in the coefficient index, so evaluating the pointwise expression at
// s = auto_map[j] and storing at j equals "compute, then permute".
// Relinearization (KeySwitchCore on component 2 of a 3-component ciphertext) uses the same
// kernel with auto_map == nullptr (s = j), KsDims::add_c0_p == 0 (no "+ c0 * P": the caller adds
// the ModDown result to c0, c1 afterwards) and KsDims::src_comp == 2 (the polynomial being
// switched is component 2, so a digit's own towers are read from there).
//   ext[0][i][j] = sum_p digit_p[i][s] * b_p[i][s]  (+ c0[i][s] * P mod q_i if i < num_q)
//   ext[1][i][j] = sum_p digit_p[i][s] * a_p[i][s]
// where digit_p[i] is the converted tower from the digits buffer, except for the tower's own digit
// (p == digit_of_tower[i]) where it is component src_comp (1 by default) of the ciphertext itself.
// Launch shape: grid = (ceil(N / kThreads), num_qp, batch), block = kThreads, no shared memory;
// one thread per (node, tower, output coefficient) looping over the digits. The reads at s are
// gathers; the key (digits * 2 * num_qp * N words) is shared by all nodes and stays in L2.
__global__ void ks_inner_product_kernel(const u64* __restrict__ digits, const u64* __restrict__ key,
                                        const u64* const* __restrict__ ct,
                                        const u32* __restrict__ auto_map,
                                        const u64* __restrict__ p_mod_q,
                                        const u32* __restrict__ digit_of_tower,
                                        u64* __restrict__ ext, const TowerConst* __restrict__ tc,
                                        KsDims d) {
  const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d.ring_dim) return;
  const u64 n = d.ring_dim;
  const u64 i = blockIdx.y, z = blockIdx.z;
  const u64 s = auto_map != nullptr ? auto_map[j] : j;
  const TowerConst c = tc[i];
  const u32 own = digit_of_tower[i];
  const u64* dg = digits + z * d.digits * d.num_qp * n;
  u64 acc_b = 0, acc_a = 0;
  for (u64 p = 0; p < d.digits; ++p) {
    const u64 dv =
        p == own ? ct[z][(d.src_comp * d.num_q + i) * n + s] : dg[(p * d.num_qp + i) * n + s];
    const u64 kb = key[((p * 2 + 0) * d.num_qp + i) * n + s];
    const u64 ka = key[((p * 2 + 1) * d.num_qp + i) * n + s];
    acc_b = add_mod(acc_b, mul_mod(dv, kb, c), c.q);
    acc_a = add_mod(acc_a, mul_mod(dv, ka, c), c.q);
  }
  if (i < d.num_q && d.add_c0_p != 0) {
    acc_b = add_mod(acc_b, mul_mod(ct[z][i * n + s], p_mod_q[i], c), c.q);
  }
  u64* e = ext + z * 2 * d.num_qp * n;
  e[(0 * d.num_qp + i) * n + j] = acc_b;
  e[(1 * d.num_qp + i) * n + j] = acc_a;
}

// Does nothing. Launch shape: 1 block of 1 thread. Used to measure the cost of a launch.
__global__ void noop_kernel() {}

bool valid_batch(u32 batch) { return batch != 0 && batch <= 65535; }

}  // namespace

cudaError_t launch_noop(cudaStream_t stream) {
  noop_kernel<<<1, 1, 0, stream>>>();
  return cudaGetLastError();
}

cudaError_t launch_ks_copy_own_digits(const u64* const* ct, u64* digits, const u32* digit_of_tower,
                                      const KsDims& dims, u32 batch, cudaStream_t stream) {
  if (dims.num_q == 0 || dims.num_q > 65535 || dims.ring_dim == 0 || !valid_batch(batch)) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((unsigned)((dims.ring_dim + kThreads - 1) / kThreads), (unsigned)dims.num_q,
                  batch);
  ks_copy_own_kernel<<<grid, kThreads, 0, stream>>>(ct, digits, digit_of_tower, dims);
  return cudaGetLastError();
}

cudaError_t launch_ks_inner_product(const u64* digits, const u64* key, const u64* const* ct,
                                    const u32* auto_map, const u64* p_mod_q,
                                    const u32* digit_of_tower, u64* ext, const TowerConst* tc,
                                    const KsDims& dims, u32 batch, cudaStream_t stream) {
  if (dims.num_qp == 0 || dims.num_qp > 65535 || dims.ring_dim == 0 || !valid_batch(batch)) {
    return cudaErrorInvalidValue;
  }
  const dim3 grid((unsigned)((dims.ring_dim + kThreads - 1) / kThreads), (unsigned)dims.num_qp,
                  batch);
  ks_inner_product_kernel<<<grid, kThreads, 0, stream>>>(digits, key, ct, auto_map, p_mod_q,
                                                         digit_of_tower, ext, tc, dims);
  return cudaGetLastError();
}

}  // namespace ombra::core
