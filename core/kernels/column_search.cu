// Kernel family: column-packed encrypted search (ciphertext x ciphertext multiply-accumulate).
//
// Packing (docs/architecture.md, decision 0003): stored vectors are column-packed, one ciphertext
// per dimension j, one vector per slot. The pre-expanded query is d ciphertexts, the j-th holding
// query value q_j in every slot. Score of vector s = sum_j stored_j[s] * query_j[s]; every slot of
// the result holds one complete score.
//
// What is computed, per stored block b, in evaluation (NTT) form, tower by tower (all moduli from
// the ciphertexts, never hard-coded):
//   out_b[0] = sum_j  s0 * q0
//   out_b[1] = sum_j (s0 * q1 + s1 * q0)
//   out_b[2] = sum_j  s1 * q1          (s = stored_{b,j} = (s0, s1), q = query_j = (q0, q1))
// This is OpenFHE's tensor product for two 2-component ciphertexts (LeveledSHEBase::EvalMultCore,
// base-leveledshe.cpp:621) accumulated over j with EvalAdd (EvalAddCoreInPlace, :575). The result
// has 3 components.
//
// LIMITATION (T0.4): no relinearization and no rescale. Relinearizing the 3-component sum once at
// the end needs key switching (T0.5 and later); rescale is likewise left to the caller. The
// result therefore has noise-scale degree 2 and scaling factor scale_stored * scale_query.
//
// Memory layout (device, 64-bit words, contiguous; P = num_towers * ring_dim words per polynomial):
//   stored: [block][dim][component 0..1][tower][coeff]    blocks * dims * 2 * P words
//   query:  [dim][component 0..1][tower][coeff]           dims * 2 * P words (shared by blocks)
//   out:    [block][component 0..2][tower][coeff]         blocks * 3 * P words
//   tc:     num_towers TowerConst entries
// Word e = tower * ring_dim + coeff is the offset inside a polynomial, so consecutive threads
// touch consecutive words (coalesced), and the tower modulus is tc[e / ring_dim].
// All input words must be reduced (< their modulus); moduli < 2^60.

#include <algorithm>

#include "kernel_launch.hpp"
#include "modarith.cuh"

namespace ombra::core {
namespace {

constexpr int kBlockThreads = 256;
// Stored blocks handled by one thread: the query words are loaded once and reused for each of them,
// which halves the global-memory traffic per product versus one block per thread.
constexpr int kBlockTile = 4;

// Launch shape: grid.x = ceil(P / kBlockThreads) (one thread per output word of a polynomial),
// grid.y = ceil(blocks / kBlockTile) (<= 65535), block = kBlockThreads threads, no shared memory.
// Each thread keeps 3 * kBlockTile accumulators in registers and loops over the d dimensions.
// Per dimension and block it does 4 modular multiplications (Barrett) and 3 modular additions.
__global__ void column_search_kernel(const u64* __restrict__ stored, const u64* __restrict__ query,
                                     u64* __restrict__ out, const TowerConst* __restrict__ tc,
                                     u64 dims, u64 blocks, u64 ring_dim, u64 poly_words) {
    const u64 e = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (e >= poly_words) return;
    const TowerConst c = tc[e / ring_dim];
    const u64 b0 = (u64)blockIdx.y * kBlockTile;
    const u64 nb = min((u64)kBlockTile, blocks - b0);

    u64 acc0[kBlockTile], acc1[kBlockTile], acc2[kBlockTile];
#pragma unroll
    for (int k = 0; k < kBlockTile; ++k) acc0[k] = acc1[k] = acc2[k] = 0;

    for (u64 j = 0; j < dims; ++j) {
        const u64 q0 = query[(j * 2) * poly_words + e];
        const u64 q1 = query[(j * 2 + 1) * poly_words + e];
#pragma unroll
        for (int k = 0; k < kBlockTile; ++k) {
            if ((u64)k < nb) {
                const u64 base = (((b0 + k) * dims + j) * 2) * poly_words + e;
                const u64 s0 = stored[base];
                const u64 s1 = stored[base + poly_words];
                acc0[k] = add_mod(acc0[k], mul_mod(s0, q0, c), c.q);
                const u64 cross = add_mod(mul_mod(s0, q1, c), mul_mod(s1, q0, c), c.q);
                acc1[k] = add_mod(acc1[k], cross, c.q);
                acc2[k] = add_mod(acc2[k], mul_mod(s1, q1, c), c.q);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kBlockTile; ++k) {
        if ((u64)k < nb) {
            const u64 o = ((b0 + k) * 3) * poly_words + e;
            out[o] = acc0[k];
            out[o + poly_words] = acc1[k];
            out[o + 2 * poly_words] = acc2[k];
        }
    }
}

}  // namespace

cudaError_t launch_column_search(const u64* stored, const u64* query, u64* out,
                                 const TowerConst* tc, const ColumnSearchDims& d,
                                 cudaStream_t stream) {
    const u64 poly_words = d.num_towers * d.ring_dim;
    const u64 grid_y = (d.blocks + kBlockTile - 1) / kBlockTile;
    if (poly_words == 0 || d.dims == 0 || d.blocks == 0 || grid_y > 65535) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((unsigned)((poly_words + kBlockThreads - 1) / kBlockThreads), (unsigned)grid_y);
    column_search_kernel<<<grid, kBlockThreads, 0, stream>>>(stored, query, out, tc, d.dims,
                                                             d.blocks, d.ring_dim, poly_words);
    return cudaGetLastError();
}

}  // namespace ombra::core
