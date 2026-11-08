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

}  // namespace ombra::core
