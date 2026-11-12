// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Compact plaintext storage and ciphertext x plaintext column search.
//
// Plaintext-DB mode: the database holds the vectors as plain numbers (fp16 or int8, 2d or d bytes
// per vector), column-packed: dimension j of the vectors of a block is one array of `slots` values.
// Just before the multiply, the GPU turns each column into the RNS plaintext OpenFHE would build
// (CKKS encoding: inverse special FFT over the slots, scaling and rounding, reduction in each
// tower, forward NTT), see core/kernels/encode.cu. The search then computes, per block,
//   out[0] = sum_j p_j * c0_j,   out[1] = sum_j p_j * c1_j
// for a 2-component ciphertext query (c0_j, c1_j) (core/kernels/plain_search.cu).
//
// No OpenFHE or CUDA types in this header; no secret key can enter (public material only).
// Public functions throw std::invalid_argument for bad input and CudaError for CUDA failures.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ombra/core/expansion_device.hpp"

namespace ombra::core {

// Storage format of a column of plaintext values.
//  kDouble: IEEE double (testing and the exact-doubles oracle).
//  kFp16:   IEEE binary16 (raw 16-bit codes), decoded exactly.
//  kInt8:   signed 8-bit codes c; the value is c / code_scale (a double division, correctly
//  rounded).
//           For vectors normalized to unit length the caller picks code_scale so that
//           max|x| * code_scale <= 127 (e.g. 127 / max|x| over the collection, or 127 if only
//           |x| <= 1 is known) and stores c = round(x * code_scale), clamped to [-127, 127].
enum class PlainFormat { kDouble, kFp16, kInt8 };

std::size_t plain_format_bytes(PlainFormat f);  // bytes per stored value: 8, 2, 1

// What the encoder needs of the context (public data), at ciphertext level 0: the Q towers only.
struct EncodeTables {
  std::size_t ring_dim = 0;  // N; the slot count is N/2 (full packing)
  std::size_t num_towers = 0;
  std::vector<std::uint64_t> moduli;   // [num_towers]
  std::vector<std::uint64_t> psi_rev;  // [tower][N], psi^i at bitrev(i) (OpenFHE NTT tables)
};

// The first num_q towers of the expansion tables are the level-0 Q towers.
EncodeTables encode_tables_from(const ExpansionTables& t);

// CKKS encoder on the GPU, mirroring OpenFHE v1.6.1 CKKSPackedEncoding::Encode (real input, full
// packing, FIXEDMANUAL, noise-scale degree 1). Floating-point steps use the same operations in the
// same order as OpenFHE's FFTSpecialInv without fused multiply-adds, and the same twiddle tables
// (computed on the host exactly as DiscreteFourierTransform::PrecomputedValues does), so results
// are expected to be identical to OpenFHE's; the tested guarantee is a small tolerance (see
// tests/test_gpu_plain_search.py), since floating-point code cannot be promised word-identical.
class PlainEncoder {
 public:
  PlainEncoder(const EncodeTables& tables, double scaling_factor);
  ~PlainEncoder();
  PlainEncoder(const PlainEncoder&) = delete;
  PlainEncoder& operator=(const PlainEncoder&) = delete;

  std::size_t slots() const;
  std::size_t poly_words() const;  // num_towers * N

  // Encodes `cols` columns held on the device (cols * slots values of `fmt`) into `d_out`
  // (cols * poly_words() words, layout [col][tower][coeff], evaluation form). Runs on the
  // default stream and waits for completion; processes the columns in chunks bounded by an
  // internal scratch buffer. Throws if a scaled value does not fit in 62 bits.
  void encode_device(PlainFormat fmt, const void* d_codes, std::size_t cols, double code_scale,
                     std::uint64_t* d_out);

  // Test aid: host in, host out. If coefficients_only, the result is the coefficient-form
  // polynomial before the NTT, [col][tower][coeff].
  std::vector<std::uint64_t> encode_host(PlainFormat fmt, const void* codes, std::size_t cols,
                                         double code_scale, bool coefficients_only);

  // Test aid (exact oracle): the integer steps only. `coeffs` is [col][N] signed integers
  // (OpenFHE's rounded coefficients); returns them reduced into every tower, [col][tower][coeff],
  // and, if `ntt`, then forward-transformed to evaluation form. Both must match OpenFHE word for
  // word.
  std::vector<std::uint64_t> reduce_ntt_host(const std::int64_t* coeffs, std::size_t cols,
                                             bool ntt);

 private:
  friend class PlainSearchDevice;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct PlainSearchShape {
  std::size_t dims;        // d columns per block (and ciphertexts in the query)
  std::size_t blocks;      // stored blocks
  std::size_t num_towers;  // RNS towers per polynomial
  std::size_t ring_dim;    // N
  std::size_t poly_words() const { return num_towers * ring_dim; }
  std::size_t slots() const { return ring_dim / 2; }
  std::size_t encoded_block_words() const { return dims * poly_words(); }
  std::size_t query_words() const { return dims * 2 * poly_words(); }
  std::size_t out_words() const { return blocks * 2 * poly_words(); }
};

// Device-resident search over plaintext columns. Three stores, each allocated on first use:
//  * compact codes: [block][dim][slot] values of the format (what the database keeps);
//  * encoded plaintexts: [block][dim][tower][coeff] RNS evaluation form (for OpenFHE-encoded
//    plaintexts and for the "pre-encoded" baseline);
//  * a scratch of `chunk_blocks` blocks of encoded plaintexts for the compact path.
class PlainSearchDevice {
 public:
  PlainSearchDevice(const PlainSearchShape& shape, const EncodeTables& tables,
                    double scaling_factor, PlainFormat fmt, double code_scale,
                    std::size_t chunk_blocks);
  ~PlainSearchDevice();
  PlainSearchDevice(const PlainSearchDevice&) = delete;
  PlainSearchDevice& operator=(const PlainSearchDevice&) = delete;

  // Host values of one block, dims * slots entries of the format, [dim][slot].
  void upload_codes(std::size_t block, const void* codes);
  // Device-to-device copy of block `src` codes into block `dst` (synthetic benchmark data).
  void copy_codes(std::size_t src, std::size_t dst);
  // One block of RNS plaintexts (encoded_block_words() reduced words).
  void upload_encoded(std::size_t block, const std::uint64_t* words, std::size_t count);
  // Query: dims ciphertexts, [dim][component 0..1][tower][coeff] (query_words() reduced words).
  void upload_query(const std::uint64_t* words, std::size_t count);

  // Encodes every block's codes into the encoded store (the pre-encoded baseline). Returns the
  // GPU time in seconds.
  double encode_all();
  // Searches the encoded store. Returns the kernel time in seconds (CUDA events).
  double run_encoded();
  // Per chunk of blocks: encode from the compact codes into the scratch, then search. Returns the
  // total GPU time; encode_seconds / search_seconds (if not null) split it.
  double run_compact(double* encode_seconds, double* search_seconds);
  // Result [block][component 0..1][tower][coeff].
  void download_result(std::vector<std::uint64_t>& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ombra::core
