// Kernel family: batched negacyclic NTT (T0.5), in OpenFHE v1.6.1's convention.
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
// The twiddle tables psi_rev[bitrev(i)] = psi^i come from the towers' roots of unity (host side).
//
// Structure: stages whose butterflies span more than one chunk (t >= chunk) run one global-memory
// kernel per stage; the remaining log2(chunk) stages of a chunk run in one shared-memory kernel.
// chunk = min(N, 2048) words (16 KB shared memory per block).
//
// Rows: every launch transforms `num_rows` rows addressed by NttRow (offset, tower, scale).
// Words must be reduced (< q) and q < 2^60.

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr u32 kThreads = 256;
constexpr u32 kMaxChunk = 2048;

// Launch shape: grid = (ceil(N/2 / kThreads), num_rows), block = kThreads, no shared memory.
// One thread per butterfly of the stage with half-distance t (operates in place in global memory).
__global__ void ntt_fwd_global_kernel(u64* data, const NttRow* rows, const u64* psi,
                                      const TowerConst* tc, u32 n, u32 t) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    const NttRow row = rows[blockIdx.y];
    const TowerConst c = tc[row.tower];
    u64* a = data + row.off;
    const u32 i = tid / t;
    const u32 j1 = i * 2 * t + (tid - i * t);
    const u64 omega = psi[(u64)row.tower * n + n / (2 * t) + i];
    const u64 hi = mul_mod(a[j1 + t], omega, c);
    const u64 lo = a[j1];
    a[j1] = add_mod(lo, hi, c.q);
    a[j1 + t] = sub_mod(lo, hi, c.q);
}

// Launch shape: grid = (N / chunk, num_rows), block = kThreads, dynamic shared memory
// chunk * 8 bytes. Block (x, row) runs the last log2(chunk) forward stages on words
// [x*chunk, (x+1)*chunk) of the row, entirely in shared memory.
__global__ void ntt_fwd_shared_kernel(u64* data, const NttRow* rows, const u64* psi,
                                      const TowerConst* tc, u32 n, u32 chunk) {
    extern __shared__ u64 s[];
    const NttRow row = rows[blockIdx.y];
    const TowerConst c = tc[row.tower];
    const u64* w = psi + (u64)row.tower * n;
    u64* a = data + row.off + (u64)blockIdx.x * chunk;
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) s[x] = a[x];
    for (u32 t = chunk / 2; t >= 1; t >>= 1) {
        __syncthreads();
        const u32 groups = chunk / (2 * t);  // groups of this stage inside the chunk
        for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
            const u32 il = idx / t;
            const u32 j1 = il * 2 * t + (idx - il * t);
            const u64 omega = w[n / (2 * t) + blockIdx.x * groups + il];
            const u64 hi = mul_mod(s[j1 + t], omega, c);
            const u64 lo = s[j1];
            s[j1] = add_mod(lo, hi, c.q);
            s[j1 + t] = sub_mod(lo, hi, c.q);
        }
    }
    __syncthreads();
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) a[x] = s[x];
}

// Same launch shape as ntt_fwd_shared_kernel; runs the first log2(chunk) inverse stages (t = 1 ..
// chunk/2). If apply_scale (chunk == N, so these are all the stages) the words are multiplied by
// the row's scale on the way out.
__global__ void ntt_inv_shared_kernel(u64* data, const NttRow* rows, const u64* psi_inv,
                                      const TowerConst* tc, u32 n, u32 chunk, int apply_scale) {
    extern __shared__ u64 s[];
    const NttRow row = rows[blockIdx.y];
    const TowerConst c = tc[row.tower];
    const u64* w = psi_inv + (u64)row.tower * n;
    u64* a = data + row.off + (u64)blockIdx.x * chunk;
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) s[x] = a[x];
    for (u32 t = 1; t < chunk; t <<= 1) {
        __syncthreads();
        const u32 groups = chunk / (2 * t);
        for (u32 idx = threadIdx.x; idx < chunk / 2; idx += blockDim.x) {
            const u32 il = idx / t;
            const u32 j1 = il * 2 * t + (idx - il * t);
            const u64 omega = w[n / (2 * t) + blockIdx.x * groups + il];
            const u64 lo = s[j1];
            const u64 hi = s[j1 + t];
            s[j1] = add_mod(lo, hi, c.q);
            s[j1 + t] = mul_mod(sub_mod(lo, hi, c.q), omega, c);
        }
    }
    __syncthreads();
    for (u32 x = threadIdx.x; x < chunk; x += blockDim.x) {
        a[x] = apply_scale ? mul_mod(s[x], row.scale, c) : s[x];
    }
}

// Launch shape as ntt_fwd_global_kernel. One inverse stage with half-distance t; if apply_scale
// (the last stage, t = N/2) both outputs are multiplied by the row's scale.
__global__ void ntt_inv_global_kernel(u64* data, const NttRow* rows, const u64* psi_inv,
                                      const TowerConst* tc, u32 n, u32 t, int apply_scale) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n / 2) return;
    const NttRow row = rows[blockIdx.y];
    const TowerConst c = tc[row.tower];
    u64* a = data + row.off;
    const u32 i = tid / t;
    const u32 j1 = i * 2 * t + (tid - i * t);
    const u64 omega = psi_inv[(u64)row.tower * n + n / (2 * t) + i];
    const u64 lo = a[j1];
    const u64 hi = a[j1 + t];
    u64 x = add_mod(lo, hi, c.q);
    u64 y = mul_mod(sub_mod(lo, hi, c.q), omega, c);
    if (apply_scale) {
        x = mul_mod(x, row.scale, c);
        y = mul_mod(y, row.scale, c);
    }
    a[j1] = x;
    a[j1 + t] = y;
}

bool valid_ring(u64 n) { return n >= 2 && (n & (n - 1)) == 0 && n <= (u64(1) << 30); }

}  // namespace

cudaError_t launch_ntt_forward(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream) {
    if (!valid_ring(t.ring_dim) || num_rows == 0 || num_rows > 65535) return cudaErrorInvalidValue;
    const u32 n = (u32)t.ring_dim;
    const u32 chunk = n < kMaxChunk ? n : kMaxChunk;
    const dim3 gg((n / 2 + kThreads - 1) / kThreads, num_rows);
    for (u32 half = n / 2; half >= chunk; half >>= 1) {
        ntt_fwd_global_kernel<<<gg, kThreads, 0, stream>>>(data, rows, t.psi, t.tc, n, half);
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    ntt_fwd_shared_kernel<<<dim3(n / chunk, num_rows), kThreads, chunk * sizeof(u64), stream>>>(
        data, rows, t.psi, t.tc, n, chunk);
    return cudaGetLastError();
}

cudaError_t launch_ntt_inverse(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream) {
    if (!valid_ring(t.ring_dim) || num_rows == 0 || num_rows > 65535) return cudaErrorInvalidValue;
    const u32 n = (u32)t.ring_dim;
    const u32 chunk = n < kMaxChunk ? n : kMaxChunk;
    ntt_inv_shared_kernel<<<dim3(n / chunk, num_rows), kThreads, chunk * sizeof(u64), stream>>>(
        data, rows, t.psi_inv, t.tc, n, chunk, chunk == n ? 1 : 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    const dim3 gg((n / 2 + kThreads - 1) / kThreads, num_rows);
    for (u32 half = chunk; half < n; half <<= 1) {
        ntt_inv_global_kernel<<<gg, kThreads, 0, stream>>>(data, rows, t.psi_inv, t.tc, n, half,
                                                           half == n / 2 ? 1 : 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

}  // namespace ombra::core
