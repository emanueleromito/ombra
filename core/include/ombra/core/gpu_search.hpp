// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// GPU-only search pipeline: multiply-and-add over d dimensions, relinearize, rescale, mask
// multiply (with relinearization), rescale. After the uploads, run() executes every step on the
// GPU; no OpenFHE call is made between the search kernel and the final ciphertexts.
//
// Every step mirrors OpenFHE v1.6.1 (CKKS, FIXEDMANUAL, HYBRID key switching) word for word:
//   relinearize   LeveledSHEBase::RelinearizeInPlace (base-leveledshe.cpp:315) with the public
//                 EvalMult key (KeySwitchHYBRID::KeySwitchCore)
//   rescale       LeveledSHECKKSRNS::ModReduceInternalInPlace (ckksrns-leveledshe.cpp:188)
//   mask multiply LeveledSHEBase::EvalMult(ct, ct, evalKey) (base-leveledshe.cpp:202): tensor
//                 product (EvalMultCore) then relinearization of component 2
// Level, noise-scale degree and scaling factor are set as OpenFHE sets them (rules in the
// functions below). Relinearization and rescale happen once per output ciphertext, after the
// d-term sum. Public material only: ciphertexts and the public EvalMult key.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_device.hpp"
#include "openfhe.h"

namespace ombra::core {

// The metadata OpenFHE keeps next to the words of a ciphertext.
struct CtMeta {
  std::size_t level = 0;
  std::size_t noise_scale_deg = 0;
  double scaling_factor = 0;
};

// Relinearize / rescale / multiply for ciphertexts with a fixed tower count (one ciphertext level),
// on the GPU. The convenience methods upload an OpenFHE ciphertext, run the kernel and download
// the result (used by the oracle driver); the raw-pointer operations are on device().
class GpuCtOps {
 public:
  using Ct = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

  // `sample` (2 components) fixes the level and tower count; `relin_key` is the public EvalMult key
  // (CryptoContextImpl::GetEvalMultKeyVector(tag)[0]). Throws std::invalid_argument unless the
  // scheme is CKKS FIXEDMANUAL with HYBRID key switching.
  GpuCtOps(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const Ct& sample,
           const lbcrypto::EvalKey<lbcrypto::DCRTPoly>& relin_key);

  // 3 -> 2 components. Metadata unchanged.
  Ct relinearize(const Ct& ct3);
  // Drops the last tower: level + 1, noise-scale degree - 1, scaling factor / modReduceFactor.
  Ct rescale(const Ct& ct2);
  // EvalMult(a, b) with the EvalMult key: level of a, degrees added, scaling factors multiplied.
  // a and b must have the same level and tower count.
  Ct multiply(const Ct& a, const Ct& b);

  CtMeta rescaled(const CtMeta& m) const;
  static CtMeta multiplied(const CtMeta& a, const CtMeta& b);

  ExpansionDevice& device() { return device_; }
  const ExpansionTables& tables() const { return tables_; }
  std::size_t num_towers() const { return tables_.num_q; }
  std::size_t ring_dim() const { return tables_.ring_dim; }
  // Checks a ciphertext of `components` polynomials against this level (towers, moduli, format) and
  // returns its words ([component][tower][coeff]).
  std::vector<std::uint64_t> checked_words(const Ct& ct, std::size_t components) const;

 private:
  lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc_;
  ExpansionTables tables_;
  ExpansionDevice device_;
  std::string key_tag_;  // key of the sample; every ciphertext passed in must match
  double mod_reduce_factor_ = 0;
  std::size_t composite_degree_ = 1;
};

enum class SearchStage {
  kSum,       // after the search kernel: 3 components, level L, degree 2d
  kRelin,     // relinearized: 2 components
  kRescaled,  // rescaled once (query x stored): level L + 1
  kMasked,    // times the mask, relinearized: level L + 1
  kFinal      // rescaled again: level L + 2, the encrypted masked scores
};

struct GpuSearchTiming {
  double search_s = 0, relin_s = 0, rescale1_s = 0, mask_s = 0, rescale2_s = 0, total_s = 0;
};

class GpuSearch {
 public:
  using Ct = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;

  // `stored_sample` is a stored column (level L, 2 components); `mask_sample` is a mask ciphertext
  // at level L + 1. `dims` ciphertexts per block, `blocks` stored blocks.
  GpuSearch(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const Ct& stored_sample,
            const Ct& mask_sample, const lbcrypto::EvalKey<lbcrypto::DCRTPoly>& relin_key,
            std::size_t dims, std::size_t blocks);
  ~GpuSearch();
  GpuSearch(const GpuSearch&) = delete;
  GpuSearch& operator=(const GpuSearch&) = delete;

  void upload_block(std::size_t block, const std::vector<Ct>& columns);  // dims ciphertexts
  void upload_query(const std::vector<Ct>& columns);                     // dims ciphertexts
  // The same as upload_block for columns already flattened (flatten_ciphertexts), so the
  // host-to-device copy can be timed apart from the flattening.
  void upload_block_flat(std::size_t block, const FlatCiphertexts& columns);
  // Device-to-device copy of stored block `src` into `dst` (synthetic benchmark data only).
  void copy_block(std::size_t src, std::size_t dst);
  // Seconds of the host-to-device copy in the last upload_block / upload_block_flat.
  double last_upload_seconds() const;
  // Device bytes this object holds (stored blocks, query, search output, stage buffers).
  std::size_t device_bytes() const;
  // Device-to-device handoff of the query. column_ptrs[j] is a device pointer to dimension j
  // (2 * num_q * N words, [component][tower][coeff], e.g. an expansion leaf from
  // ExpansionDevice::device_ptr). `meta` is the level, noise-scale degree and scaling factor of the
  // columns; the caller has checked them (and the moduli) against the stored columns, as the
  // expansion's output is not an OpenFHE object. run() rejects a mismatch with the stored blocks.
  void set_query_from_device(const std::vector<const std::uint64_t*>& column_ptrs,
                             const CtMeta& meta);
  // The query buffer ([dim][component][tower][coeff], dims * 2 * num_q * N words) for a producer
  // that writes the columns in place (expand_tree_into); call set_query_meta afterwards with the
  // same checks as set_query_from_device. Writes must be ordered before run().
  std::uint64_t* query_device();
  void set_query_meta(const CtMeta& meta);
  void upload_mask(std::size_t block, const Ct& mask);  // level L + 1, same metadata for all

  // Runs all five steps on the GPU for every block; waits between steps to time them.
  GpuSearchTiming run();

  // Copies the result of `stage` for one block to the host as an OpenFHE ciphertext with the
  // metadata the GPU path derived (the only place OpenFHE objects are built).
  Ct download(SearchStage stage, std::size_t block);
  // The raw words ([component][tower][coeff]) of the same result, a device-to-host copy only (no
  // OpenFHE object is built). Not for kSum.
  std::vector<std::uint64_t> download_words(SearchStage stage, std::size_t block);
  CtMeta meta(SearchStage stage) const;

  GpuCtOps& level0();  // stored/query level L
  GpuCtOps& level1();  // mask level L + 1

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ombra::core
