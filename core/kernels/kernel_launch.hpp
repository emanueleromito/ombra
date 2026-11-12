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

// Batched search: `num_queries` queries over the same stored blocks. queries are
// [q][dim][comp][P], out is [q][block][comp 0..2][P]; see the comment in column_search.cu.
cudaError_t launch_column_search_batched(const u64* stored, const u64* queries, u64* out,
                                         const TowerConst* tc, const ColumnSearchDims& dims,
                                         u64 num_queries, cudaStream_t stream);
// Bytes requested from / written to global memory by one launch (from the kernel's loop structure).
struct BatchSearchTraffic {
  u64 stored_load_bytes = 0;
  u64 query_load_bytes = 0;
  u64 store_bytes = 0;
  u64 compulsory_bytes = 0;  // every stored, query and output word touched exactly once
  u64 load_bytes() const { return stored_load_bytes + query_load_bytes; }
};
BatchSearchTraffic column_search_batched_traffic(const ColumnSearchDims& dims, u64 num_queries);

// One basis-conversion job (ApproxSwitchCRTBasis): word k of the output tower is
//   sum_{i < src_count} src[src_off + i*N + k] * mat[mat_off + i]  mod q_dst.
// The source words need not be reduced mod q_dst (they are below 2^60); mat_shoup =
// shoup_quotient(mat, q_dst).
struct ConvJob {
  u64 src_off;
  u64 mat_off;
  u32 src_count;
  u32 pad;
};
// Makes the forward NTT take its input from basis-conversion jobs instead of from `data`: the row
// with NttRow::conv_job = j transforms the output tower of jobs[j] (modulus of NttRow::tower), for
// batch copy z computed from src + z * src_stride. The first pass of the NTT does the conversion
// as it loads, so the converted tower is never written to memory untransformed.
struct ConvLoad {
  const u64* src = nullptr;
  u64 src_stride = 0;
  const u64* mat = nullptr;
  const u64* mat_shoup = nullptr;
  const ConvJob* jobs = nullptr;
};
// Inverse NTT only: the first pass reads copy z of the row from ptrs[z] + add + row.off instead of
// from `data` (the output still goes to `data`), saving a separate copy.
struct NttSource {
  const u64* const* ptrs = nullptr;
  u64 add = 0;
};

// ntt.cu: batched negacyclic NTT over rows of N words, one modulus per row, in OpenFHE's convention
// (Cooley-Tukey forward from natural to bit-reversed order, Gentleman-Sande inverse back, twiddle
// tables in bit-reversed order). A row is addressed by an element offset into `data`; `tower`
// selects the modulus and the twiddle tables. The inverse multiplies every output word by `scale`
// (the caller passes N^-1 times any extra per-row constant, mod q); the forward ignores `scale`.
// scale_shoup = shoup_quotient(scale, q) (modarith.cuh). A launch can also be repeated over
// `NttBatch::count` independent copies of the row list, copy z shifted by z * stride words.
struct NttRow {
  u64 off;          // offset of the row's first word in `data`, in words
  u64 scale;        // inverse only: output multiplier, reduced mod q
  u64 scale_shoup;  // inverse only: shoup_quotient(scale, q)
  u32 tower;        // index into tc and the twiddle tables
  u32 conv_job;     // forward with ConvLoad only: index into ConvLoad::jobs
};
struct NttTables {
  const u64* psi;            // [tower][N] psi^i stored at the bit-reversed position
  const u64* psi_shoup;      // [tower][N] shoup_quotient of each entry of psi
  const u64* psi_inv;        // [tower][N] psi^-i stored at the bit-reversed position
  const u64* psi_inv_shoup;  // [tower][N] shoup_quotient of each entry of psi_inv
  const TowerConst* tc;
  u64 ring_dim;  // N, a power of two in [2, 2^22]
};
struct NttBatch {
  u32 count = 1;
  u64 stride = 0;
};
cudaError_t launch_ntt_forward(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream, NttBatch batch = {},
                               const ConvLoad* conv = nullptr);
cudaError_t launch_ntt_inverse(u64* data, const NttRow* rows, u32 num_rows, const NttTables& t,
                               cudaStream_t stream, NttBatch batch = {}, NttSource src = {});

// Forward NTT of the P -> Q converted rows (ConvLoad `conv`, required) fused with the end of
// ApproxModDown and the tree's combine (the epilogue of the last NTT pass). `data` is scratch of
// the layout pq[z][r][i] (r = 0..1, i < num_q; row.off = (r * num_q + i) * N, row.tower = i);
// the transformed words are not stored. Copy z reads ct[z] and ext + z * ext_stride and writes
// even[z] and, if odd != nullptr and odd[z] != nullptr, odd[z]:
//   sub  = (ext[z][r][i] - NTT(pq)) * P^-1 mod q_i       (the automorphism image tau(ct))
//   even = ct + sub;   odd = (ct - sub) * mono[i] mod q_i   (mono may be null if odd is null)
// ext is [2][num_qp][N] per copy; ct, even and odd are ciphertext-shaped [2][num_q][N].
struct CombineArgs {
  const u64* const* ct = nullptr;  // [batch] ciphertext pointers
  const u64* ext = nullptr;
  u64 ext_stride = 0;
  const u64* p_inv_mod_q = nullptr;  // [num_q]
  const u64* p_inv_shoup = nullptr;  // [num_q] shoup_quotient(p_inv_mod_q[i], q_i)
  const u64* mono = nullptr;         // [num_q][N] or null
  u64* const* even = nullptr;        // [batch]
  u64* const* odd = nullptr;         // [batch] or null
  u64 num_q = 0;
  u64 num_qp = 0;
};
cudaError_t launch_ntt_forward_combine(u64* data, const NttRow* rows, u32 num_rows,
                                       const NttTables& t, const CombineArgs& args,
                                       cudaStream_t stream, NttBatch batch, const ConvLoad* conv);

// keyswitch.cu: kernels of the hybrid key-switched automorphism (layouts documented there).
// Every launcher below handles `batch` independent nodes (one tree level, or a chunk of it): node
// z uses the same constants; its data sits z * stride words after node 0 or is addressed by a
// per-node pointer.
struct KsDims {
  u64 ring_dim;  // N
  u64 num_q;     // Q towers of the ciphertext
  u64 num_qp;    // Q towers + P towers
  u64 digits;    // key-switch digits in use
  // Which component of the ciphertext is being key-switched (1 for the automorphism, 2 for
  // relinearization) and whether the inner product adds c0 * P (automorphism only).
  u32 src_comp = 1;
  u32 add_c0_p = 1;
};
// Test/oracle aid, not on the hot path: writes component 1 of every node's ciphertext into the
// digit slot that holds the tower itself, digits[z][digit_of_tower[i]][i][k] (stride digits *
// num_qp * N per node). The hot path never materializes these words: the inner product reads
// them straight from the ciphertext.
cudaError_t launch_ks_copy_own_digits(const u64* const* ct, u64* digits, const u32* digit_of_tower,
                                      const KsDims& dims, u32 batch, cudaStream_t stream);
// digits: stride digits * num_qp * N per node (only the converted towers are read; a digit's own
// towers are read from component 1 of ct[z]); ext: stride 2 * num_qp * N; c0 of node z is
// component 0 of ct[z]. digit_of_tower has num_qp entries, 0xFFFFFFFF for the P towers. auto_map
// may be null (identity permutation); p_mod_q is only read if dims.add_c0_p != 0.
cudaError_t launch_ks_inner_product(const u64* digits, const u64* key, const u64* const* ct,
                                    const u32* auto_map, const u64* p_mod_q,
                                    const u32* digit_of_tower, u64* ext, const TowerConst* tc,
                                    const KsDims& dims, u32 batch, cudaStream_t stream);

// encode.cu: GPU CKKS encoding of plaintext columns, see the comment above the kernels.
enum class CodeFormat : int { kDouble = 0, kFp16 = 1, kInt8 = 2 };
// buf[col][k] = (decode(codes[col][k]), 0) for k < slots; int8 decodes as code / code_scale.
// Test aid: reduces int64 coefficients [col][ring_dim] into every tower, [col][tower][ring_dim].
cudaError_t launch_reduce_ints(const long long* in, u64 cols, u64 ring_dim, const TowerConst* tc,
                               u64 num_towers, u64* out, cudaStream_t stream);

cudaError_t launch_encode_decode(CodeFormat fmt, const void* codes, u64 cols, u64 slots,
                                 double code_scale, double2* buf, cudaStream_t stream);
// One stage (butterfly span `len`, len = slots .. 2) of the inverse special FFT on every column.
// tw = the inverse twiddle table [slots - 1] (OpenFHE's m_invTw layout).
cudaError_t launch_encode_fft_stage(double2* buf, const double2* tw, u64 cols, u64 slots, u64 len,
                                    cudaStream_t stream);
// Bit reversal, scaling by inv_len then scaling_factor, rounding, reduction into every tower:
// out[col][tower][k] = round(re), out[col][tower][k + slots] = round(im), coefficient form.
// *err is set to 1 if a scaled value does not fit in 62 bits.
cudaError_t launch_encode_finish(const double2* buf, u64 cols, u64 slots, double inv_len,
                                 double scaling_factor, const TowerConst* tc, u64 num_towers,
                                 u64 ring_dim, u64* out, int* err, cudaStream_t stream);

// plain_search.cu: ciphertext x plaintext column search, see the comment above the kernel.
cudaError_t launch_plain_search(const u64* plain, const u64* query, u64* out, const TowerConst* tc,
                                u64 dims, u64 blocks, u64 num_towers, u64 ring_dim,
                                cudaStream_t stream);

// rescale.cu: CKKS FIXEDMANUAL rescale and the ciphertext x ciphertext tensor product, see
// the comments above the kernels. Nodes are addressed through pointer arrays.
struct RescaleDims {
  u64 ring_dim;  // N
  u64 num_q;     // towers of the input ciphertext (the last one is dropped)
};
// Step 2 of the rescale. last: [z][N] coefficient-form words of the last tower (z = node * 2 +
// component); tmp: [z][num_q - 1][N] = last switched to modulus q_i with the centered representative
// (NativeVector(v, q_i) in DCRTPoly::DropLastElementAndScale). nodes2 = 2 * nodes.
cudaError_t launch_rescale_switch(const u64* last, u64* tmp, const TowerConst* tc,
                                  const RescaleDims& dims, u32 nodes2, cudaStream_t stream);
// Step 4: out[node][r][i][k] = (in[node][r][i][k] - tmp[z][i][k]) * ql_inv[i] mod q_i, i < num_q - 1;
// `in` is [2][num_q][N] and `out` is [2][num_q - 1][N] per node. tmp holds the forward NTT of
// the output of launch_rescale_switch.
cudaError_t launch_rescale_combine(const u64* const* in, const u64* tmp, u64* const* out,
                                   const u64* ql_inv, const TowerConst* tc, const RescaleDims& dims,
                                   u32 nodes, cudaStream_t stream);
// out[node] = tensor product of the 2-component ciphertexts a[node], b[node] (all [2][num_q][N]):
// out = [a0 b0, a0 b1 + a1 b0, a1 b1], [3][num_q][N] per node (LeveledSHEBase::EvalMultCore).
cudaError_t launch_ct_tensor(const u64* const* a, const u64* const* b, u64* const* out,
                             const TowerConst* tc, const RescaleDims& dims, u32 nodes,
                             cudaStream_t stream);

// An empty kernel, only to measure launch overhead when profiling.
cudaError_t launch_noop(cudaStream_t stream);

}  // namespace ombra::core
