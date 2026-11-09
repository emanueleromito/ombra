// GPU versions of OpenFHE ciphertext operations (CKKS, FIXEDMANUAL, evaluation form).
//
// Public material only: these functions take ciphertexts and an encoded plaintext, never keys.
// Results are new OpenFHE ciphertexts whose metadata (level, noise-scale degree, scaling factor)
// is derived from the operation, mirroring OpenFHE v1.6.1:
//   add:  LeveledSHEBase::EvalAddCoreInPlace  (base-leveledshe.cpp:575) - metadata is that of the
//         operands, which must agree.
//   mult by plaintext: LeveledSHERNS::EvalMultInPlace (rns-leveledshe.cpp:263) and
//         LeveledSHECKKSRNS::EvalMultInPlace (ckksrns-leveledshe.cpp:162): level unchanged,
//         noise-scale degree = sum of the degrees, scaling factor = product of the factors.
// Operand mismatches that OpenFHE would repair by rescaling or level adjustment are rejected.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openfhe.h"

namespace ombra::core {

// A list of ciphertexts with identical shape, moduli and metadata, flattened to RNS words laid out
// [ciphertext][component][tower][coeff] (the layout the search kernel reads).
struct FlatCiphertexts {
    std::size_t count = 0;
    std::size_t num_towers = 0;
    std::size_t ring_dim = 0;
    std::vector<std::uint64_t> words;
    std::vector<std::uint64_t> moduli;  // one per tower, from the ciphertexts
    std::size_t level = 0;
    std::size_t noise_scale_deg = 0;
    double scaling_factor = 0;
};

// Flattens 2-component, evaluation-form CKKS FIXEDMANUAL ciphertexts. Throws
// std::invalid_argument if they differ in level, noise degree, scale, shape or moduli.
FlatCiphertexts flatten_ciphertexts(const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& cts);

// Inverse of flatten_ciphertexts for one ciphertext: a copy of `like` (context, key tag, level,
// noise degree, scaling factor) whose RNS words are replaced by `words` ([component][tower][coeff],
// evaluation form, same shape and moduli as `like`).
lbcrypto::Ciphertext<lbcrypto::DCRTPoly> unflatten_ciphertext(
    const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& like, const std::vector<std::uint64_t>& words);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_eval_add(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& a,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& b);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_eval_mult_plain(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& ct, lbcrypto::ConstPlaintext& pt);

// Column-packed search (kernel: core/kernels/column_search.cu). stored_blocks[b][j] is dimension j
// of block b; query[j] holds query value j in every slot. Returns one 3-component ciphertext per
// block holding sum_j stored[b][j] * query[j] (no relinearization, no rescale). Metadata, derived
// as in LeveledSHEBase::EvalMultCore (base-leveledshe.cpp:621): level unchanged, noise-scale degree
// = sum of the degrees, scaling factor = product of the factors. Stored and query ciphertexts must
// agree in level, degree, scale and moduli. Public material only: no keys.
std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> gpu_eval_column_search(
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& stored_blocks,
    const std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>& query);

// Batched column search (T0.7): several queries in one pass over the stored blocks. queries[q][j]
// is dimension j of query q. Returns results[q][b], each what gpu_eval_column_search gives for
// query q (same metadata rules).
std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> gpu_eval_column_search_batched(
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& stored_blocks,
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& queries);

}  // namespace ombra::core
