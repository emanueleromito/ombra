// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Kernel family: batched negacyclic NTT, in OpenFHE v1.6.1's convention.
//
// Forward (NumberTheoreticTransformNat::ForwardTransformToBitReverseInPlace, transformnat-impl.h):
// Cooley-Tukey. For m = 1, 2, 4 .. N/2 and t = N/(2m): for each group i < m and j < t, with
// j1 = i*2t + j, omega = psi_rev[m + i]:
//     hi = a[j1+t] * omega;   a[j1] = a[j1] + hi;   a[j1+t] = a[j1] - hi          (mod q)
// Input natural order, output bit-reversed. Inverse (...InverseTransformFromBitReverseInPlace):
// Gentleman-Sande, t = 1, 2 .. N/2 and m = N/(2t), omega = psi_inv_rev[m + i]:
//     a[j1] = lo + hi;   a[j1+t] = (lo - hi) * omega
// followed by multiplication of every word by N^-1 (here: by the row's `scale`, which the caller
// sets to N^-1 times any extra constant). Both are exact modular transforms, so outputs are the
// unique residues OpenFHE computes (its lazy reductions do not change the final residues).
// Twiddle multiplications use Shoup's method with precomputed quotients (modarith.cuh), which gives
// the same residue as a full reduction.
//
// Structure: with chunk = min(N, 2048) words (16 KB shared memory per block) and R = N / chunk,
//  - "strided" pass: the R-fold stages whose butterflies span more than one chunk (t >= chunk
//    words). A block takes W = chunk / R adjacent columns of the row and all R words at stride
//    `chunk` in each, which is closed under those log2(R) stages, and does them in shared memory.
//  - "shared" pass: the remaining log2(chunk) stages, on one contiguous chunk per block.
// Forward = strided then shared; inverse = shared then strided. So an NTT is two kernels.
//
// Rows: every launch transforms `num_rows` rows addressed by NttRow (offset, tower, scale), times
// `batch.count` copies. Words must be reduced (< q) and q < 2^60.

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr u32 kThreads = 256;
constexpr u32 kMaxChunk = 2048;

__device__ inline u32 lg2(u32 x) { return 31u - __clz(x); }  // x a power of two

// Forward butterfly on shared/global words (a, b) with a twiddle (w, wq).
__device__ inline void fwd_bf(u64& lo, u64& hi, u64 w, u64 wq, u64 q) {
  const u64 h = mul_mod_shoup(hi, w, wq, q);
  const u64 l = lo;
  lo = add_mod(l, h, q);
  hi = sub_mod(l, h, q);
}
__device__ inline void inv_bf(u64& lo, u64& hi, u64 w, u64 wq, u64 q) {
  const u64 l = lo, h = hi;
  lo = add_mod(l, h, q);
  hi = mul_mod_shoup(sub_mod(l, h, q), w, wq, q);
}

// One word of a basis-conversion job computed on the fly (the row's input of a forward NTT):
//   sum_i src[z][job.src_off + i*N + k] * mat[job.mat_off + i] mod q.
// The source words need not be reduced mod q (Shoup multiplication takes any 64-bit input).
__device__ inline u64 conv_word(const ConvLoad& cl, const NttRow& row, u64 z, u64 k, u64 q, u32 n) {
  const ConvJob job = cl.jobs[row.conv_job];
  const u64* src = cl.src + z * cl.src_stride + job.src_off + k;
  u64 acc = 0;
  for (u32 i = 0; i < job.src_count; ++i) {
    acc = add_mod(
        acc,
        mul_mod_shoup(src[(u64)i * n], cl.mat[job.mat_off + i], cl.mat_shoup[job.mat_off + i], q),
        q);
  }
  return acc;
}

// Forward strided pass. Launch shape: grid = (R, num_rows, batch), block = kThreads, dynamic
// shared memory chunk * 8 bytes (R * W words). Block (x, row, z) loads words
// [r * chunk + x * W + c] for r < R, c < W, runs the first log2(R) forward stages (m = 1 .. R/2)
// on them in shared memory and stores them back. If kConv the loaded words are not read from
// `data` but computed by conv_word (the row's basis-conversion job).
template <bool kConv>
__global__ void ntt_fwd_strided_kernel(u64* data, const NttRow* rows, const u64* psi,
                                       const u64* psi_shoup, const TowerConst* tc, u32 n, u32 chunk,
                                       u32 lg_r, u64 batch_stride, ConvLoad cl) {
  extern __shared__ u64 s[];
  const NttRow row = rows[blockIdx.y];
  const u64 q = tc[row.tower].q;
  const u64* w = psi + (u64)row.tower * n;
  const u64* wq = psi_shoup + (u64)row.tower * n;
  const u32 R = 1u << lg_r, lg_w = lg2(chunk) - lg_r, W = 1u << lg_w;
  u64* a = data + (u64)blockIdx.z * batch_stride + row.off + (u64)blockIdx.x * W;
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    const u64 off = (u64)(x >> lg_w) * chunk + (x & (W - 1));
    if constexpr (kConv) {
      s[x] = conv_word(cl, row, blockIdx.z, off + (u64)blockIdx.x * W, q, n);
    } else {
      s[x] = a[off];
    }
  }
  for (u32 k = 0; k < lg_r; ++k) {
    __syncthreads();
    const u32 lt = lg_r - 1 - k;  // log2 of the half-distance in rows
    for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
      const u32 cc = idx & (W - 1), bf = idx >> lg_w;
      const u32 i = bf >> lt, j = bf & ((1u << lt) - 1);
      const u32 r1 = (i << (lt + 1)) + j;
      const u32 o = (1u << k) + i;
      fwd_bf(s[(r1 << lg_w) + cc], s[((r1 + (1u << lt)) << lg_w) + cc], w[o], wq[o], q);
    }
  }
  __syncthreads();
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    a[(u64)(x >> lg_w) * chunk + (x & (W - 1))] = s[x];
  }
  (void)R;
}

// Inverse strided pass: same launch shape and tile as ntt_fwd_strided_kernel; runs the last
// log2(R) inverse stages (t = 1 .. R/2 in rows, m = R/(2t)) and multiplies every word by the row's
// scale on the way out.
__global__ void ntt_inv_strided_kernel(u64* data, const NttRow* rows, const u64* psi_inv,
                                       const u64* psi_inv_shoup, const TowerConst* tc, u32 n,
                                       u32 chunk, u32 lg_r, u64 batch_stride) {
  extern __shared__ u64 s[];
  const NttRow row = rows[blockIdx.y];
  const u64 q = tc[row.tower].q;
  const u64* w = psi_inv + (u64)row.tower * n;
  const u64* wq = psi_inv_shoup + (u64)row.tower * n;
  const u32 lg_w = lg2(chunk) - lg_r, W = 1u << lg_w;
  u64* a = data + (u64)blockIdx.z * batch_stride + row.off + (u64)blockIdx.x * W;
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    s[x] = a[(u64)(x >> lg_w) * chunk + (x & (W - 1))];
  }
  for (u32 k = 0; k < lg_r; ++k) {
    __syncthreads();
    const u32 lt = k;  // log2 of the half-distance in rows
    for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
      const u32 cc = idx & (W - 1), bf = idx >> lg_w;
      const u32 i = bf >> lt, j = bf & ((1u << lt) - 1);
      const u32 r1 = (i << (lt + 1)) + j;
      const u32 o = (1u << (lg_r - 1 - k)) + i;
      inv_bf(s[(r1 << lg_w) + cc], s[((r1 + (1u << lt)) << lg_w) + cc], w[o], wq[o], q);
    }
  }
  __syncthreads();
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    a[(u64)(x >> lg_w) * chunk + (x & (W - 1))] =
        mul_mod_shoup(s[x], row.scale, row.scale_shoup, q);
  }
}

// Forward shared pass. Launch shape: grid = (N / chunk, num_rows, batch), block = kThreads, dynamic
// shared memory chunk * 8 bytes. Block (x, row, z) runs the last log2(chunk) forward stages on
// words [x*chunk, (x+1)*chunk) of the row, entirely in shared memory. If kCombine, the final words
// are not stored but fed to the combine epilogue described at launch_ntt_forward_combine. If
// kConv the input words come from conv_word instead of `data` (only when N <= chunk, i.e. when
// there is no strided pass).
template <bool kCombine, bool kConv>
__global__ void ntt_fwd_shared_kernel(u64* data, const NttRow* rows, const u64* psi,
                                      const u64* psi_shoup, const TowerConst* tc, u32 n, u32 chunk,
                                      u64 batch_stride, CombineArgs ca, ConvLoad cl) {
  extern __shared__ u64 s[];
  const NttRow row = rows[blockIdx.y];
  const u64 q = tc[row.tower].q;
  const u64* w = psi + (u64)row.tower * n;
  const u64* wq = psi_shoup + (u64)row.tower * n;
  u64* a = data + (u64)blockIdx.z * batch_stride + row.off + (u64)blockIdx.x * chunk;
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    if constexpr (kConv) {
      s[x] = conv_word(cl, row, blockIdx.z, (u64)blockIdx.x * chunk + x, q, n);
    } else {
      s[x] = a[x];
    }
  }
  // Two stages per pass (radix 4, in registers): stages lt (half-distance t = 2^lt) and lt - 1
  // (h = t/2) act on the four words base + {0, h, t, t + h}; one synchronization per two stages.
  int lt = (int)lg2(chunk) - 1;
  for (; lt >= 1; lt -= 2) {
    __syncthreads();
    const u32 t = 1u << lt, h = t >> 1;
    for (u32 g = threadIdx.x; g < chunk / 4; g += blockDim.x) {
      const u32 i = g >> (lt - 1), j = g & (h - 1);
      const u32 base = (i << (lt + 1)) + j;
      u64 x0 = s[base], x1 = s[base + h], x2 = s[base + t], x3 = s[base + t + h];
      const u32 oa = (n >> (lt + 1)) + blockIdx.x * (chunk >> (lt + 1)) + i;
      fwd_bf(x0, x2, w[oa], wq[oa], q);
      fwd_bf(x1, x3, w[oa], wq[oa], q);
      const u32 ob = (n >> lt) + blockIdx.x * (chunk >> lt) + 2 * i;
      fwd_bf(x0, x1, w[ob], wq[ob], q);
      fwd_bf(x2, x3, w[ob + 1], wq[ob + 1], q);
      s[base] = x0;
      s[base + h] = x1;
      s[base + t] = x2;
      s[base + t + h] = x3;
    }
  }
  if (lt == 0) {  // an odd number of stages: the last one alone
    __syncthreads();
    for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
      const u32 o = (n >> 1) + blockIdx.x * (chunk >> 1) + idx;
      fwd_bf(s[2 * idx], s[2 * idx + 1], w[o], wq[o], q);
    }
  }
  __syncthreads();
  if constexpr (!kCombine) {
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) a[x] = s[x];
  } else {
    const TowerConst c = tc[row.tower];
    const u64 per_comp = ca.num_q * (u64)n;
    const u64 r = row.off / per_comp;  // component
    const u64 i = row.tower;           // Q tower
    const u64 base = row.off + (u64)blockIdx.x * chunk;
    const u64* ct = ca.ct[blockIdx.z];
    u64* even = ca.even[blockIdx.z];
    u64* odd = ca.odd != nullptr ? ca.odd[blockIdx.z] : nullptr;
    const u64* ext = ca.ext + (u64)blockIdx.z * ca.ext_stride + (r * ca.num_qp + i) * n +
                     (u64)blockIdx.x * chunk;
    const u64 pinv = ca.p_inv_mod_q[i], pinv_q = ca.p_inv_shoup[i];
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
      const u64 sub = mul_mod_shoup(sub_mod(ext[x], s[x], q), pinv, pinv_q, q);
      const u64 v = ct[base + x];
      even[base + x] = add_mod(v, sub, q);
      if (odd != nullptr) {
        odd[base + x] =
            mul_mod(sub_mod(v, sub, q), ca.mono[i * n + (u64)blockIdx.x * chunk + x], c);
      }
    }
  }
}

// Inverse shared pass: same launch shape; runs the first log2(chunk) inverse stages (t = 1 ..
// chunk/2). If apply_scale (chunk == N, so these are all the stages) the words are multiplied by
// the row's scale on the way out. If src_ptrs != nullptr the input is read from
// src_ptrs[z] + src_add + row.off (and written to `data`), else the pass is in place.
__global__ void ntt_inv_shared_kernel(u64* data, const NttRow* rows, const u64* psi_inv,
                                      const u64* psi_inv_shoup, const TowerConst* tc, u32 n,
                                      u32 chunk, int apply_scale, u64 batch_stride,
                                      const u64* const* src_ptrs, u64 src_add) {
  extern __shared__ u64 s[];
  const NttRow row = rows[blockIdx.y];
  const u64 q = tc[row.tower].q;
  const u64* w = psi_inv + (u64)row.tower * n;
  const u64* wq = psi_inv_shoup + (u64)row.tower * n;
  u64* a = data + (u64)blockIdx.z * batch_stride + row.off + (u64)blockIdx.x * chunk;
  const u64* in =
      src_ptrs != nullptr ? src_ptrs[blockIdx.z] + src_add + row.off + (u64)blockIdx.x * chunk : a;
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) s[x] = in[x];
  const u32 lg_chunk = lg2(chunk);
  // Two stages per pass (radix 4, in registers): stages lt (half-distance h = 2^lt) and lt + 1
  // (2h) act on the four words base + {0, h, 2h, 3h}; one synchronization per two stages.
  u32 lt = 0;
  for (; lt + 1 < lg_chunk; lt += 2) {
    __syncthreads();
    const u32 h = 1u << lt;
    for (u32 g = threadIdx.x; g < chunk / 4; g += blockDim.x) {
      const u32 i = g >> lt, j = g & (h - 1);
      const u32 base = (i << (lt + 2)) + j;
      u64 x0 = s[base], x1 = s[base + h], x2 = s[base + 2 * h], x3 = s[base + 3 * h];
      const u32 oa = (n >> (lt + 1)) + blockIdx.x * (chunk >> (lt + 1)) + 2 * i;
      inv_bf(x0, x1, w[oa], wq[oa], q);
      inv_bf(x2, x3, w[oa + 1], wq[oa + 1], q);
      const u32 ob = (n >> (lt + 2)) + blockIdx.x * (chunk >> (lt + 2)) + i;
      inv_bf(x0, x2, w[ob], wq[ob], q);
      inv_bf(x1, x3, w[ob], wq[ob], q);
      s[base] = x0;
      s[base + h] = x1;
      s[base + 2 * h] = x2;
      s[base + 3 * h] = x3;
    }
  }
  if (lt < lg_chunk) {  // an odd number of stages: the last one alone
    __syncthreads();
    const u32 h = 1u << lt;
    for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
      const u32 il = idx >> lt;
      const u32 j1 = (il << (lt + 1)) + (idx & (h - 1));
      const u32 o = (n >> (lt + 1)) + blockIdx.x * (chunk >> (lt + 1)) + il;
      inv_bf(s[j1], s[j1 + h], w[o], wq[o], q);
    }
  }
  __syncthreads();
  for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
    a[x] = apply_scale ? mul_mod_shoup(s[x], row.scale, row.scale_shoup, q) : s[x];
  }
}

u32 lg2_host(u32 x) {  // x a power of two
  u32 l = 0;
  while ((1u << l) < x) ++l;
  return l;
}

bool valid_ring(u64 n) { return n >= 2 && (n & (n - 1)) == 0 && n <= (u64(1) << 22); }

bool valid_launch(u32 num_rows, const NttBatch& b) {
  return num_rows != 0 && num_rows <= 65535 && b.count != 0 && b.count <= 65535;
}

}  // namespace

namespace {

// Forward chain: [strided pass] then shared pass; the first kernel of the chain does the
// basis-conversion load if `conv` is given. kCombine selects the combine epilogue.
template <bool kCombine>
cudaError_t forward_chain(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                          const CombineArgs& args, cudaStream_t stream, NttBatch batch,
                          const ConvLoad* conv) {
  if (!valid_ring(t.ring_dim) || !valid_launch(num_rows, batch)) return cudaErrorInvalidValue;
  const u32 n = (u32)t.ring_dim;
  const u32 chunk = n < kMaxChunk ? n : kMaxChunk;
  const u32 r = n / chunk;
  const dim3 grid(r, num_rows, batch.count);
  const ConvLoad cl = conv != nullptr ? *conv : ConvLoad{};
  const size_t smem = chunk * sizeof(u64);
  if (r > 1) {
    if (conv != nullptr) {
      ntt_fwd_strided_kernel<true><<<grid, kThreads, smem, stream>>>(
          data, rows, t.psi, t.psi_shoup, t.tc, n, chunk, lg2_host(r), batch.stride, cl);
    } else {
      ntt_fwd_strided_kernel<false><<<grid, kThreads, smem, stream>>>(
          data, rows, t.psi, t.psi_shoup, t.tc, n, chunk, lg2_host(r), batch.stride, cl);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
  }
  if (r == 1 && conv != nullptr) {
    ntt_fwd_shared_kernel<kCombine, true><<<grid, kThreads, smem, stream>>>(
        data, rows, t.psi, t.psi_shoup, t.tc, n, chunk, batch.stride, args, cl);
  } else {
    ntt_fwd_shared_kernel<kCombine, false><<<grid, kThreads, smem, stream>>>(
        data, rows, t.psi, t.psi_shoup, t.tc, n, chunk, batch.stride, args, cl);
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_ntt_forward(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream, NttBatch batch, const ConvLoad* conv) {
  return forward_chain<false>(data, rows, num_rows, t, CombineArgs{}, stream, batch, conv);
}

cudaError_t launch_ntt_forward_combine(u64* data, const NttRow* rows, u32 num_rows,
                                       const NttTables& t, const CombineArgs& args,
                                       cudaStream_t stream, NttBatch batch, const ConvLoad* conv) {
  return forward_chain<true>(data, rows, num_rows, t, args, stream, batch, conv);
}

cudaError_t launch_ntt_inverse(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream, NttBatch batch, NttSource src) {
  if (!valid_ring(t.ring_dim) || !valid_launch(num_rows, batch)) return cudaErrorInvalidValue;
  const u32 n = (u32)t.ring_dim;
  const u32 chunk = n < kMaxChunk ? n : kMaxChunk;
  const u32 r = n / chunk;
  const dim3 grid(r, num_rows, batch.count);
  ntt_inv_shared_kernel<<<grid, kThreads, chunk * sizeof(u64), stream>>>(
      data, rows, t.psi_inv, t.psi_inv_shoup, t.tc, n, chunk, r == 1 ? 1 : 0, batch.stride,
      src.ptrs, src.add);
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess || r == 1) return err;
  ntt_inv_strided_kernel<<<grid, kThreads, chunk * sizeof(u64), stream>>>(
      data, rows, t.psi_inv, t.psi_inv_shoup, t.tc, n, chunk, lg2_host(r), batch.stride);
  return cudaGetLastError();
}

}  // namespace ombra::core
