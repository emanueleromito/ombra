// ombra_column_search: oracle driver for the column-packed search kernel.
//
// Usage: ombra_column_search <context.bin> <in_dir> <out_dir> <dims> <blocks>
//
// Inputs (written by the Python test with OpenFHE's BINARY serialization, public material only):
// the CryptoContext, and in <in_dir> the ciphertexts stored_<b>_<j>.bin (dimension j of block b)
// and query_<j>.bin (query value j in every slot), j in [0, dims), b in [0, blocks). Computes, per
// block, sum_j stored_j * query_j with the GPU kernel and, as the oracle, with OpenFHE
// EvalMultNoRelin (cryptocontext.h:2107 -> LeveledSHERNS::EvalMult, rns-leveledshe.cpp:211, ->
// EvalMultCore, base-leveledshe.cpp:621) accumulated with EvalAdd (EvalAddCoreInPlace,
// base-leveledshe.cpp:575). Compares all RNS words, moduli and metadata exactly and writes
// gpu_<b>.bin and cpu_<b>.bin (OpenFHE ciphertext serialization) to <out_dir>.
// Never reads or writes a secret key. Prints only PASS/FAIL lines, never data values.
// Exit code: 0 exact match, 1 mismatch, 2 error.

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/oracle_compare.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

namespace {

Ciphertext<DCRTPoly> load_ct(const std::string& path) {
  Ciphertext<DCRTPoly> ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot read ciphertext " + path);
  }
  return ct;
}

void save_ct(const std::string& path, const Ciphertext<DCRTPoly>& ct) {
  if (!Serial::SerializeToFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot write ciphertext " + path);
  }
}

bool identical(const Ciphertext<DCRTPoly>& x, const Ciphertext<DCRTPoly>& y) {
  return ombra::core::identical_ciphertexts(x, y);  // shared oracle comparator
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: ombra_column_search <context> <in_dir> <out_dir> <dims> <blocks>\n");
    return 2;
  }
  try {
    const std::string in_dir = argv[2], out_dir = argv[3];
    const std::size_t dims = std::stoul(argv[4]);
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);
    const std::size_t blocks = std::stoul(argv[5]);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> stored(blocks);
    std::vector<Ciphertext<DCRTPoly>> query;
    for (std::size_t j = 0; j < dims; ++j) {
      query.push_back(load_ct(in_dir + "/query_" + std::to_string(j) + ".bin"));
    }
    for (std::size_t b = 0; b < blocks; ++b) {
      for (std::size_t j = 0; j < dims; ++j) {
        stored[b].push_back(
            load_ct(in_dir + "/stored_" + std::to_string(b) + "_" + std::to_string(j) + ".bin"));
      }
    }

    const auto gpu = ombra::core::gpu_eval_column_search(stored, query);

    bool ok = true;
    for (std::size_t b = 0; b < blocks; ++b) {
      Ciphertext<DCRTPoly> cpu = cc->EvalMultNoRelin(stored[b][0], query[0]);
      for (std::size_t j = 1; j < dims; ++j) {
        cpu = cc->EvalAdd(cpu, cc->EvalMultNoRelin(stored[b][j], query[j]));
      }
      const bool block_ok = identical(gpu[b], cpu);
      ok = ok && block_ok;
      std::printf("column search block %zu exact RNS match: %s\n", b, block_ok ? "PASS" : "FAIL");
      save_ct(out_dir + "/gpu_" + std::to_string(b) + ".bin", gpu[b]);
      save_ct(out_dir + "/cpu_" + std::to_string(b) + ".bin", cpu);
    }
    std::printf("column search exact RNS match: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
