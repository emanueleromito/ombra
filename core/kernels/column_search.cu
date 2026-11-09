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

// ---- Batched search (T0.7): Q queries in one pass over the same stored blocks ----
//
// Computes, for every query q < Q and stored block b, the same 3-component sum as above:
//   out[q][b][c] = column_search of stored block b with query q   (c = 0..2)
// Layout (device, 64-bit words, P = num_towers * ring_dim):
//   stored:  [block][dim][comp 0..1][P]      as above
//   queries: [q][dim][comp 0..1][P]          Q * dims * 2 * P words
//   out:     [q][block][comp 0..2][P]        Q * blocks * 3 * P words
// Each thread owns one word position e of a tile of kBatchBlockTile blocks x kBatchQueryTile
// queries and keeps the 3 * 4 * 4 accumulators in registers. For every dimension j it loads the
// stored words of its blocks once and the query words of its queries once and reuses each of them
// for the whole tile. Per product (one block, one query, one dimension) it requests
// (2 * Bt + 2 * Qt) / (Bt * Qt) = 1 word at tile 4 x 4, against 2.5 for the single-query kernel
// (2 + 8 words for 4 products). A stored word is requested ceil(Q / Qt) times, not Q times: the
// tile is the unit of reuse, so stored traffic grows once per query tile, not per query.
// Launch shape: grid.x = ceil(P / kBlockThreads), grid.y = ceil(blocks / kBatchBlockTile),
// grid.z = ceil(Q / kBatchQueryTile) (<= 65535 each), block = kBlockThreads, no shared memory.
constexpr int kBatchBlockTile = 4;
constexpr int kBatchQueryTile = 4;

__global__ void column_search_batched_kernel(const u64* __restrict__ stored,
                                             const u64* __restrict__ queries,
                                             u64* __restrict__ out, const TowerConst* __restrict__ tc,
                                             u64 dims, u64 blocks, u64 num_queries, u64 ring_dim,
                                             u64 poly_words) {
    const u64 e = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (e >= poly_words) return;
    const TowerConst c = tc[e / ring_dim];
    const u64 b0 = (u64)blockIdx.y * kBatchBlockTile;
    const u64 q0i = (u64)blockIdx.z * kBatchQueryTile;
    const u64 nb = min((u64)kBatchBlockTile, blocks - b0);
    const u64 nq = min((u64)kBatchQueryTile, num_queries - q0i);

    u64 acc0[kBatchQueryTile][kBatchBlockTile], acc1[kBatchQueryTile][kBatchBlockTile],
        acc2[kBatchQueryTile][kBatchBlockTile];
#pragma unroll
    for (int m = 0; m < kBatchQueryTile; ++m) {
#pragma unroll
        for (int k = 0; k < kBatchBlockTile; ++k) acc0[m][k] = acc1[m][k] = acc2[m][k] = 0;
    }

    const u64 query_words = dims * 2 * poly_words;
    for (u64 j = 0; j < dims; ++j) {
        u64 s0[kBatchBlockTile], s1[kBatchBlockTile];
#pragma unroll
        for (int k = 0; k < kBatchBlockTile; ++k) {
            if ((u64)k < nb) {
                const u64 base = (((b0 + k) * dims + j) * 2) * poly_words + e;
                s0[k] = stored[base];
                s1[k] = stored[base + poly_words];
            }
        }
#pragma unroll
        for (int m = 0; m < kBatchQueryTile; ++m) {
            if ((u64)m < nq) {
                const u64 qbase = (q0i + m) * query_words + (j * 2) * poly_words + e;
                const u64 q0 = queries[qbase];
                const u64 q1 = queries[qbase + poly_words];
#pragma unroll
                for (int k = 0; k < kBatchBlockTile; ++k) {
                    if ((u64)k < nb) {
                        acc0[m][k] = add_mod(acc0[m][k], mul_mod(s0[k], q0, c), c.q);
                        const u64 cross =
                            add_mod(mul_mod(s0[k], q1, c), mul_mod(s1[k], q0, c), c.q);
                        acc1[m][k] = add_mod(acc1[m][k], cross, c.q);
                        acc2[m][k] = add_mod(acc2[m][k], mul_mod(s1[k], q1, c), c.q);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < kBatchQueryTile; ++m) {
        if ((u64)m < nq) {
#pragma unroll
            for (int k = 0; k < kBatchBlockTile; ++k) {
                if ((u64)k < nb) {
                    const u64 o = (((q0i + m) * blocks + b0 + k) * 3) * poly_words + e;
                    out[o] = acc0[m][k];
                    out[o + poly_words] = acc1[m][k];
                    out[o + 2 * poly_words] = acc2[m][k];
                }
            }
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

cudaError_t launch_column_search_batched(const u64* stored, const u64* queries, u64* out,
                                         const TowerConst* tc, const ColumnSearchDims& d,
                                         u64 num_queries, cudaStream_t stream) {
    const u64 poly_words = d.num_towers * d.ring_dim;
    const u64 grid_y = (d.blocks + kBatchBlockTile - 1) / kBatchBlockTile;
    const u64 grid_z = (num_queries + kBatchQueryTile - 1) / kBatchQueryTile;
    if (poly_words == 0 || d.dims == 0 || d.blocks == 0 || num_queries == 0 || grid_y > 65535 ||
        grid_z > 65535) {
        return cudaErrorInvalidValue;
    }
    const dim3 grid((unsigned)((poly_words + kBlockThreads - 1) / kBlockThreads), (unsigned)grid_y,
                    (unsigned)grid_z);
    column_search_batched_kernel<<<grid, kBlockThreads, 0, stream>>>(
        stored, queries, out, tc, d.dims, d.blocks, num_queries, d.ring_dim, poly_words);
    return cudaGetLastError();
}

// Global-memory words requested by the threads of one launch, summed from the loops of the kernel
// above (one thread per word position, one thread tile per (block tile, query tile)):
//   loads  = P * dims * sum over tiles of (2 * nb + 2 * nq)
//          = P * dims * 2 * (blocks * ceil(Q / Qt) + Q * ceil(blocks / Bt))
//   stores = P * 3 * blocks * Q
// (the 24-byte tower constant each thread reads is not counted). Requests, not DRAM traffic: the
// L1 and L2 caches may serve repeated loads of the same word.
BatchSearchTraffic column_search_batched_traffic(const ColumnSearchDims& d, u64 num_queries) {
    const u64 p = d.num_towers * d.ring_dim;
    const u64 block_tiles = (d.blocks + kBatchBlockTile - 1) / kBatchBlockTile;
    const u64 query_tiles = (num_queries + kBatchQueryTile - 1) / kBatchQueryTile;
    BatchSearchTraffic t;
    t.stored_load_bytes = 8 * p * d.dims * 2 * d.blocks * query_tiles;
    t.query_load_bytes = 8 * p * d.dims * 2 * num_queries * block_tiles;
    t.store_bytes = 8 * p * 3 * d.blocks * num_queries;
    t.compulsory_bytes =
        8 * p * d.dims * 2 * (d.blocks + num_queries) + t.store_bytes;  // each word once
    return t;
}

}  // namespace ombra::core
