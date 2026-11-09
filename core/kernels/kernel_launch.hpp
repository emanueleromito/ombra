// Launchers for the CUDA kernels. Kernels live in one .cu file per family under core/kernels/;
// host wrappers in core/src/ call them through these plain functions, so no CUDA separable
// compilation (relocatable device code) is needed. Only cuda_runtime types appear here.
//
// Every launcher is asynchronous on `stream`, returns the launch error (cudaGetLastError) and does
// not synchronize. All pointers are device pointers.
#pragma once

#include <cuda_runtime.h>

#include "modarith.cuh"

namespace ombra::core {

// rns_basic.cu: out[i] = (a[i] op b[i % b_period]) mod q(i), with
// q(i) = tc[(i / ring_dim) % num_towers].q; op is + or *.
cudaError_t launch_rns_add(const u64* a, const u64* b, u64* out, const TowerConst* tc, u64 total,
                           u64 ring_dim, u64 num_towers, u64 b_period, cudaStream_t stream);
cudaError_t launch_rns_mul(const u64* a, const u64* b, u64* out, const TowerConst* tc, u64 total,
                           u64 ring_dim, u64 num_towers, u64 b_period, cudaStream_t stream);

// column_search.cu: see the layout comment above column_search_kernel.
struct ColumnSearchDims {
    u64 dims;        // d: number of dimensions (ciphertexts per block)
    u64 blocks;      // number of stored blocks scored in this launch
    u64 num_towers;  // RNS towers per polynomial
    u64 ring_dim;    // N
};
cudaError_t launch_column_search(const u64* stored, const u64* query, u64* out,
                                 const TowerConst* tc, const ColumnSearchDims& dims,
                                 cudaStream_t stream);

// ntt.cu: batched negacyclic NTT over rows of N words, one modulus per row, in OpenFHE's convention
// (Cooley-Tukey forward from natural to bit-reversed order, Gentleman-Sande inverse back, twiddle
// tables in bit-reversed order). A row is addressed by an element offset into `data`; `tower`
// selects the modulus and the twiddle tables. The inverse multiplies every output word by `scale`
// (the caller passes N^-1 times any extra per-row constant, mod q); the forward ignores `scale`.
struct NttRow {
    u64 off;    // offset of the row's first word in `data`, in words
    u64 scale;  // inverse only: output multiplier, reduced mod q
    u32 tower;  // index into tc and the twiddle tables
    u32 pad;
};
struct NttTables {
    const u64* psi;      // [tower][N] psi^i stored at the bit-reversed position
    const u64* psi_inv;  // [tower][N] psi^-i stored at the bit-reversed position
    const TowerConst* tc;
    u64 ring_dim;  // N, a power of two >= 2
};
cudaError_t launch_ntt_forward(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream);
cudaError_t launch_ntt_inverse(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream);

// keyswitch.cu: kernels of the hybrid key-switched automorphism (layouts documented there).
// One basis-conversion job: out[dst_off + k] = sum_{i < src_count} (src[src_off + i*N + k] mod q_dst)
// * mat[mat_off + i] mod q_dst, with q_dst = tc[dst_tower].q.
struct ConvJob {
    u64 src_off;
    u64 mat_off;
    u64 dst_off;
    u32 src_count;
    u32 dst_tower;
};
cudaError_t launch_basis_conv(const u64* src, u64* out, const u64* mat, const ConvJob* jobs,
                              u32 num_jobs, const TowerConst* tc, u64 ring_dim,
                              cudaStream_t stream);

struct KsDims {
    u64 ring_dim;  // N
    u64 num_q;     // Q towers of the ciphertext
    u64 num_qp;    // Q towers + P towers
    u64 digits;    // key-switch digits in use
};
cudaError_t launch_ks_inner_product(const u64* digits, const u64* key, const u64* c0,
                                    const u32* auto_map, const u64* p_mod_q, u64* ext,
                                    const TowerConst* tc, const KsDims& dims, cudaStream_t stream);
cudaError_t launch_ks_combine(const u64* ct, const u64* ext, const u64* pq, const u64* p_inv_mod_q,
                              const u64* mono, u64* even, u64* odd, const TowerConst* tc,
                              const KsDims& dims, cudaStream_t stream);

}  // namespace ombra::core
