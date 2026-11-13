// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// GPU operations for the expansion tree (expansion_tree.hpp) on OpenFHE ciphertexts.
//
// Host glue: extracts every table the kernels need from OpenFHE's crypto parameters, key and
// ciphertext objects (flat arrays; the kernels never see OpenFHE types) and exposes the `Ops`
// interface expand_tree expects. Public material only: ciphertexts and automorphism keys.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

#include "ombra/core/expansion_device.hpp"
#include "openfhe.h"

namespace ombra::core {

// Tables for ciphertexts shaped like `sample` (its tower count fixes the Q basis). Throws
// std::invalid_argument unless the scheme is CKKS with HYBRID key switching and noise scale 1.
ExpansionTables extract_expansion_tables(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                         const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& sample);

// Key words [digit][b, a][QP tower][N] of one automorphism key, in the device layout.
std::vector<std::uint64_t> extract_key_words(const lbcrypto::EvalKey<lbcrypto::DCRTPoly>& key,
                                             const ExpansionTables& tables,
                                             std::size_t full_q_towers);

// Evaluation form of X^-shift in the Q towers of `like`, computed with OpenFHE's own NTT.
std::vector<std::uint64_t> extract_monomial_words(const lbcrypto::DCRTPoly& like,
                                                  std::size_t shift);

struct GpuCt {
  CtHandle dev;
};

class GpuExpansion {
 public:
  using Ct = lbcrypto::Ciphertext<lbcrypto::DCRTPoly>;
  using KeyMap = std::map<std::uint32_t, lbcrypto::EvalKey<lbcrypto::DCRTPoly>>;

  // Uploads the keys for the Galois elements N/2^l + 1 (l < expansion depth of d) and the
  // monomials X^-2^l. `sample` fixes the level and shape and is the metadata template for
  // download(). Throws if a key is missing.
  GpuExpansion(const lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc, const Ct& sample,
               const KeyMap& keys, std::size_t d);

  GpuCt upload(const Ct& ct);
  Ct download(const GpuCt& ct);
  void synchronize() { device_.synchronize(); }
  void release_pool() { device_.release_pool(); }

  // The Ops interface of expand_tree.
  std::pair<GpuCt, GpuCt> split(const GpuCt& c, std::size_t level, std::uint32_t galois,
                                std::size_t shift, bool want_odd);

  // Optional level-at-once operation of expand_tree: all splits of a level in batched launches.
  std::vector<std::pair<GpuCt, GpuCt>> split_level(const std::vector<GpuCt>& nodes,
                                                   std::size_t level, std::uint32_t galois,
                                                   std::size_t shift,
                                                   const std::vector<char>& want_odd);

  // Destination of expand_tree_into: leaf j is written to base + j * leaf_words, for j < leaves.
  // `base` is a device pointer; leaf_words must be the words of one ciphertext (2 * num_q * N).
  struct LeafBuffer {
    std::uint64_t* base = nullptr;
    std::size_t leaf_words = 0;
    std::size_t leaves = 0;
  };
  // Metadata every expansion leaf carries (the split never changes it): that of the `sample`
  // ciphertext, i.e. what download() puts on a leaf. Moduli are tables().moduli[0 .. num_q).
  struct LeafMeta {
    std::size_t level = 0;
    std::size_t noise_scale_deg = 0;
    double scaling_factor = 0;
  };
  LeafMeta leaf_meta() const;

  // Optional last-level operation of expand_tree_into: like split_level, but the even child of
  // nodes[k] is written to leaf residues[k] and, if want_odd[k], the odd child to leaf
  // residues[k] + shift, directly in `sink`. Asynchronous (synchronize() before reading).
  void split_level_into(const std::vector<GpuCt>& nodes, std::size_t level, std::uint32_t galois,
                        std::size_t shift, const std::vector<char>& want_odd,
                        const std::vector<std::size_t>& residues, const LeafBuffer& sink);

  ExpansionDevice& device() { return device_; }
  const ExpansionTables& tables() const { return tables_; }
  std::size_t key_count() const { return key_count_; }
  double setup_seconds() const { return setup_seconds_; }  // table extraction + key upload

 private:
  ExpansionTables tables_;
  ExpansionDevice device_;
  Ct sample_;
  std::size_t key_count_ = 0;
  double setup_seconds_ = 0;
};

}  // namespace ombra::core
