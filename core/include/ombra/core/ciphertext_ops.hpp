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

#include "openfhe.h"

namespace ombra::core {

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_eval_add(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& a,
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& b);

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> gpu_eval_mult_plain(
    lbcrypto::ConstCiphertext<lbcrypto::DCRTPoly>& ct, lbcrypto::ConstPlaintext& pt);

}  // namespace ombra::core
