// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Kernel family: GPU CKKS encoding of real plaintext columns (plaintext-DB mode).
//
// Mirrors OpenFHE v1.6.1 CKKSPackedEncoding::Encode (pke/lib/encoding/ckkspackedencoding.cpp) for
// real input, full packing (slots = N/2), noise-scale degree 1, NATIVE_SIZE = 64:
//   1. inverse = values (imaginary parts 0), length S = slots;
//   2. DiscreteFourierTransform::FFTSpecialInv(inverse, 2N)   (core/lib/math/dftransform.cpp):
//        for len = S, S/2 .. 2, lenh = len/2, for each block of len entries, for j < lenh:
//          u = lo + hi; v = lo - hi; lo = u; hi = v * w[lenh - 1 + j]       (complex arithmetic)
//        then bit-reverse the S entries, then multiply every entry by 1/S;
//   3. multiply every entry (both parts) by the scaling factor, round each part to the nearest
//      integer (std::llround: half away from zero): coefficient k = round(re_k), coefficient
//      k + S = round(im_k);
//   4. reduce each integer into every RNS tower (negative values q - |r| mod q);
//   5. forward NTT of every tower (ntt.cu, the same NTT the ciphertext kernels use).
// The butterflies use the round-to-nearest intrinsics __dadd_rn/__dmul_rn so the compiler cannot
// fuse them into FMAs: every operation is a separate IEEE double operation, in the same order as
// OpenFHE's loop, and the twiddle table is the one OpenFHE builds (computed on the host in
// plain_search_host.cu). Differences from OpenFHE can then only come from the host compiler's own
// contraction in OpenFHE; the tolerance is tested, not assumed.
//
// Layouts (device): values/codes [col][slot]; FFT buffer double2 [col][slot]; output
// [col][tower][coeff] 64-bit words.

#include <cuda_fp16.h>

#include "kernel_launch.hpp"

namespace ombra::core {
namespace {

constexpr int kThreads = 256;

// Launch shape: one thread per value, grid = ceil(cols * slots / kThreads), block = kThreads.
__global__ void decode_kernel(CodeFormat fmt, const void* codes, u64 total, double code_scale,
                              double2* buf) {
  const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i >= total) return;
  double v;
  if (fmt == CodeFormat::kDouble) {
    v = static_cast<const double*>(codes)[i];
  } else if (fmt == CodeFormat::kFp16) {
    v = static_cast<double>(__half2float(static_cast<const __half*>(codes)[i]));
  } else {
    v = static_cast<double>(static_cast<const signed char*>(codes)[i]) / code_scale;
  }
  buf[i] = make_double2(v, 0.0);
}

// One stage of FFTSpecialInv for every column. Launch shape: one thread per butterfly,
// grid = ceil(cols * slots / 2 / kThreads), block = kThreads; no shared memory (the buffer is read
// and written in place, each butterfly owns its two entries).
__global__ void fft_stage_kernel(double2* buf, const double2* tw, u64 total, u64 slots, u64 len) {
  const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (t >= total) return;
  const u64 half_slots = slots / 2;
  const u64 col = t / half_slots, b = t % half_slots;
  const u64 lenh = len / 2;
  const u64 j = b % lenh, grp = b / lenh;
  double2* lo = buf + col * slots + grp * len + j;
  double2* hi = lo + lenh;
  const double2 l = *lo, h = *hi;
  const double ur = __dadd_rn(l.x, h.x), ui = __dadd_rn(l.y, h.y);
  const double vr = __dsub_rn(l.x, h.x), vi = __dsub_rn(l.y, h.y);
  const double2 w = tw[lenh - 1 + j];
  *lo = make_double2(ur, ui);
  *hi = make_double2(__dsub_rn(__dmul_rn(vr, w.x), __dmul_rn(vi, w.y)),
                     __dadd_rn(__dmul_rn(vr, w.y), __dmul_rn(vi, w.x)));
}

__device__ inline u64 reduce_signed(long long r, u64 q) {
  if (r >= 0) return (u64)r % q;
  const u64 m = (u64)(-r) % q;
  return m == 0 ? 0 : q - m;
}

// Launch shape: one thread per (col, k), grid = ceil(cols * slots / kThreads), block = kThreads.
// slots is a power of two; entry k is read from bit_reverse(k). Each thread writes 2 words in each
// tower (coefficients k and k + slots).
__global__ void finish_kernel(const double2* buf, u64 total, u64 slots, u32 log_slots,
                              double inv_len, double scaling_factor, const TowerConst* tc,
                              u64 num_towers, u64 ring_dim, u64* out, int* err) {
  const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i >= total) return;
  const u64 col = i / slots;
  const u32 k = (u32)(i % slots);
  const u32 src = log_slots == 0 ? 0u : (__brev(k) >> (32 - log_slots));
  const double2 v = buf[col * slots + src];
  const double re = __dmul_rn(__dmul_rn(v.x, inv_len), scaling_factor);
  const double im = __dmul_rn(__dmul_rn(v.y, inv_len), scaling_factor);
  // OpenFHE throws when the scaled value overflows 64 bits; here: flag it, do not trap.
  if (!(fabs(re) < 4.0e18) || !(fabs(im) < 4.0e18)) {
    atomicExch(err, 1);
    return;
  }
  const long long r = llround(re), m = llround(im);
  for (u64 t = 0; t < num_towers; ++t) {
    const u64 q = tc[t].q;
    u64* poly = out + (col * num_towers + t) * ring_dim;
    poly[k] = reduce_signed(r, q);
    poly[k + slots] = reduce_signed(m, q);
  }
}

// Test aid for the exact oracle: the reduction step of finish_kernel on given integers.
// Launch shape: one thread per (col, coefficient), grid = ceil(cols * ring_dim / kThreads),
// block = kThreads. in is int64 [col][ring_dim]; out is [col][tower][ring_dim].
__global__ void reduce_ints_kernel(const long long* in, u64 total, u64 ring_dim,
                                   const TowerConst* tc, u64 num_towers, u64* out) {
  const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
  if (i >= total) return;
  const u64 col = i / ring_dim, k = i % ring_dim;
  for (u64 t = 0; t < num_towers; ++t) {
    out[(col * num_towers + t) * ring_dim + k] = reduce_signed(in[i], tc[t].q);
  }
}

}  // namespace

cudaError_t launch_reduce_ints(const long long* in, u64 cols, u64 ring_dim, const TowerConst* tc,
                               u64 num_towers, u64* out, cudaStream_t stream) {
  const u64 total = cols * ring_dim;
  if (total == 0) return cudaErrorInvalidValue;
  reduce_ints_kernel<<<(unsigned)((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      in, total, ring_dim, tc, num_towers, out);
  return cudaGetLastError();
}

cudaError_t launch_encode_decode(CodeFormat fmt, const void* codes, u64 cols, u64 slots,
                                 double code_scale, double2* buf, cudaStream_t stream) {
  const u64 total = cols * slots;
  if (total == 0) return cudaErrorInvalidValue;
  decode_kernel<<<(unsigned)((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      fmt, codes, total, code_scale, buf);
  return cudaGetLastError();
}

cudaError_t launch_encode_fft_stage(double2* buf, const double2* tw, u64 cols, u64 slots, u64 len,
                                    cudaStream_t stream) {
  const u64 total = cols * (slots / 2);
  if (total == 0 || len < 2 || len > slots) return cudaErrorInvalidValue;
  fft_stage_kernel<<<(unsigned)((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      buf, tw, total, slots, len);
  return cudaGetLastError();
}

cudaError_t launch_encode_finish(const double2* buf, u64 cols, u64 slots, double inv_len,
                                 double scaling_factor, const TowerConst* tc, u64 num_towers,
                                 u64 ring_dim, u64* out, int* err, cudaStream_t stream) {
  const u64 total = cols * slots;
  if (total == 0 || ring_dim != 2 * slots || (slots & (slots - 1)) != 0) {
    return cudaErrorInvalidValue;
  }
  u32 lg = 0;
  while ((u64(1) << lg) < slots) ++lg;
  finish_kernel<<<(unsigned)((total + kThreads - 1) / kThreads), kThreads, 0, stream>>>(
      buf, total, slots, lg, inv_len, scaling_factor, tc, num_towers, ring_dim, out, err);
  return cudaGetLastError();
}

}  // namespace ombra::core
