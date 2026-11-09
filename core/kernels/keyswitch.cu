// Kernel family: hybrid key-switched automorphism and the expansion split (T0.5, decision 0010).
//
// Mirrors OpenFHE v1.6.1 LeveledSHEBase::EvalAutomorphismCore (base-leveledshe.cpp:400), HYBRID
// branch, followed by the tree's EvalAdd / EvalSub and the multiplication by X^-k. All arithmetic
// is exact modular arithmetic on evaluation-form RNS words, so every output word equals OpenFHE's.
// Moduli, tables and key words come from the host (extracted from OpenFHE objects); nothing is
// hard-coded. "QP" below is the extended basis: the num_q Q towers of the ciphertext followed by
// the num_p special P towers. N is the ring dimension.
//
// Layouts (device, 64-bit words):
//   ct:     [component 0..1][Q tower][coeff]                   2 * num_q * N
//   digits: [digit][QP tower][coeff]                           digits * num_qp * N
//   key:    [digit][0 = b, 1 = a][QP tower][coeff]             digits * 2 * num_qp * N
//   ext:    [component 0..1][QP tower][coeff]                  2 * num_qp * N
//   pq:     [component 0..1][Q tower][coeff]                   2 * num_q * N
//   mono:   [Q tower][coeff]  evaluation form of X^-shift      num_q * N
// All inputs are reduced words, moduli < 2^60.

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr u32 kThreads = 256;

// Basis conversion (ApproxSwitchCRTBasis, dcrtpoly-impl.h:938), after the source towers were
// multiplied by their (hat)^-1 constants (done inside the preceding inverse NTT's row scale):
//   out[k] = sum_i (src_i[k] mod q_dst) * mat[i] mod q_dst
// Launch shape: grid = (ceil(N / kThreads), num_jobs), block = kThreads, no shared memory.
// Each thread produces one output word of one job and loops over the job's source towers.
__global__ void basis_conv_kernel(const u64* __restrict__ src, u64* __restrict__ out,
                                  const u64* __restrict__ mat, const ConvJob* __restrict__ jobs,
                                  const TowerConst* __restrict__ tc, u64 n) {
    const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    const ConvJob job = jobs[blockIdx.y];
    const TowerConst c = tc[job.dst_tower];
    u64 acc = 0;
    for (u32 i = 0; i < job.src_count; ++i) {
        const u64 x = src[job.src_off + (u64)i * n + k] % c.q;
        acc = add_mod(acc, mul_mod(x, mat[job.mat_off + i], c), c.q);
    }
    out[job.dst_off + k] = acc;
}

// Key-switch inner product with the evaluation key, the "+ c0 * P" term and the automorphism
// permutation, in one pass (EvalFastKeySwitchCoreExt, keyswitch-hybrid.cpp:389; then
// ba[0] += cv0 * P on the Q towers; then AutomorphismTransform with out[j] = in[auto_map[j]]).
// All three are pointwise in the coefficient index, so evaluating the pointwise expression at
// s = auto_map[j] and storing at j equals "compute, then permute".
//   ext[0][i][j] = sum_p digit_p[i][s] * b_p[i][s]  (+ c0[i][s] * P mod q_i if i < num_q)
//   ext[1][i][j] = sum_p digit_p[i][s] * a_p[i][s]
// Launch shape: grid = (ceil(N / kThreads), num_qp), block = kThreads, no shared memory; one thread
// per (tower, output coefficient) looping over the digits. The reads at s are gathers.
__global__ void ks_inner_product_kernel(const u64* __restrict__ digits, const u64* __restrict__ key,
                                        const u64* __restrict__ c0, const u32* __restrict__ auto_map,
                                        const u64* __restrict__ p_mod_q, u64* __restrict__ ext,
                                        const TowerConst* __restrict__ tc, KsDims d) {
    const u64 j = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d.ring_dim) return;
    const u64 n = d.ring_dim;
    const u64 i = blockIdx.y;
    const u64 s = auto_map[j];
    const TowerConst c = tc[i];
    u64 acc_b = 0, acc_a = 0;
    for (u64 p = 0; p < d.digits; ++p) {
        const u64 dv = digits[(p * d.num_qp + i) * n + s];
        const u64 kb = key[((p * 2 + 0) * d.num_qp + i) * n + s];
        const u64 ka = key[((p * 2 + 1) * d.num_qp + i) * n + s];
        acc_b = add_mod(acc_b, mul_mod(dv, kb, c), c.q);
        acc_a = add_mod(acc_a, mul_mod(dv, ka, c), c.q);
    }
    if (i < d.num_q) {
        acc_b = add_mod(acc_b, mul_mod(c0[i * n + s], p_mod_q[i], c), c.q);
    }
    ext[(0 * d.num_qp + i) * n + j] = acc_b;
    ext[(1 * d.num_qp + i) * n + j] = acc_a;
}

// Final step of ApproxModDown (dcrtpoly-impl.h:1020, t = 0) fused with the tree's combine:
//   sub = (ext - pq) * P^-1 mod q_i          (the automorphism image tau(ct), per component)
//   even = ct + sub                          (EvalAdd)
//   odd  = (ct - sub) * mono mod q_i         (EvalSub, then X^-shift in evaluation form)
// `odd` may be null, then only `even` is written.
// Launch shape: grid = (ceil(N / kThreads), 2 * num_q), block = kThreads, no shared memory.
__global__ void ks_combine_kernel(const u64* __restrict__ ct, const u64* __restrict__ ext,
                                  const u64* __restrict__ pq, const u64* __restrict__ p_inv_mod_q,
                                  const u64* __restrict__ mono, u64* __restrict__ even,
                                  u64* __restrict__ odd, const TowerConst* __restrict__ tc,
                                  KsDims d) {
    const u64 k = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= d.ring_dim) return;
    const u64 n = d.ring_dim;
    const u64 r = blockIdx.y / d.num_q;
    const u64 i = blockIdx.y % d.num_q;
    const TowerConst c = tc[i];
    const u64 o = (r * d.num_q + i) * n + k;
    const u64 sub = mul_mod(sub_mod(ext[(r * d.num_qp + i) * n + k], pq[o], c.q), p_inv_mod_q[i], c);
    const u64 v = ct[o];
    even[o] = add_mod(v, sub, c.q);
    if (odd != nullptr) {
        odd[o] = mul_mod(sub_mod(v, sub, c.q), mono[i * n + k], c);
    }
}

}  // namespace

cudaError_t launch_basis_conv(const u64* src, u64* out, const u64* mat, const ConvJob* jobs,
                              u32 num_jobs, const TowerConst* tc, u64 ring_dim,
                              cudaStream_t stream) {
    if (num_jobs == 0 || num_jobs > 65535 || ring_dim == 0) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)((ring_dim + kThreads - 1) / kThreads), num_jobs);
    basis_conv_kernel<<<grid, kThreads, 0, stream>>>(src, out, mat, jobs, tc, ring_dim);
    return cudaGetLastError();
}

cudaError_t launch_ks_inner_product(const u64* digits, const u64* key, const u64* c0,
                                    const u32* auto_map, const u64* p_mod_q, u64* ext,
                                    const TowerConst* tc, const KsDims& dims, cudaStream_t stream) {
    if (dims.num_qp == 0 || dims.num_qp > 65535 || dims.ring_dim == 0) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)((dims.ring_dim + kThreads - 1) / kThreads), (unsigned)dims.num_qp);
    ks_inner_product_kernel<<<grid, kThreads, 0, stream>>>(digits, key, c0, auto_map, p_mod_q, ext,
                                                           tc, dims);
    return cudaGetLastError();
}

cudaError_t launch_ks_combine(const u64* ct, const u64* ext, const u64* pq, const u64* p_inv_mod_q,
                              const u64* mono, u64* even, u64* odd, const TowerConst* tc,
                              const KsDims& dims, cudaStream_t stream) {
    if (dims.num_q == 0 || 2 * dims.num_q > 65535 || dims.ring_dim == 0) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)((dims.ring_dim + kThreads - 1) / kThreads),
                    (unsigned)(2 * dims.num_q));
    ks_combine_kernel<<<grid, kThreads, 0, stream>>>(ct, ext, pq, p_inv_mod_q, mono, even, odd, tc,
                                                     dims);
    return cudaGetLastError();
}

}  // namespace ombra::core
