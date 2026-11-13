// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_column_search_batch: oracle driver for the batched column-packed search kernel.
//
// Usage: ombra_column_search_batch <context.bin> <in_dir> <out_dir> <dims> <blocks> <queries>
//
// Inputs (OpenFHE BINARY serialization, public material only): the CryptoContext, and in <in_dir>
// stored_<b>_<j>.bin (dimension j of block b) and query_<q>_<j>.bin (value j of query q in every
// slot). Computes, for every query q and block b, sum_j stored_j * query_j with the batched GPU
// kernel (all queries in one pass) and, as the oracle, with OpenFHE EvalMultNoRelin accumulated
// with EvalAdd, exactly as ombra_column_search does. Compares all RNS words, moduli and metadata
// exactly and writes gpu_<q>_<b>.bin and cpu_<q>_<b>.bin to <out_dir>.
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
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: ombra_column_search_batch <context> <in_dir> <out_dir> <dims> "
                 "<blocks> <queries>\n");
    return 2;
  }
  try {
    const std::string in_dir = argv[2], out_dir = argv[3];
    const std::size_t dims = std::stoul(argv[4]);
    const std::size_t blocks = std::stoul(argv[5]);
    const std::size_t nq = std::stoul(argv[6]);
    if (dims == 0 || blocks == 0 || nq == 0) {
      throw std::invalid_argument("dims, blocks and queries must be positive");
    }
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);
    std::vector<std::vector<Ciphertext<DCRTPoly>>> stored(blocks), queries(nq);
    for (std::size_t q = 0; q < nq; ++q) {
      for (std::size_t j = 0; j < dims; ++j) {
        queries[q].push_back(
            load_ct(in_dir + "/query_" + std::to_string(q) + "_" + std::to_string(j) + ".bin"));
      }
    }
    for (std::size_t b = 0; b < blocks; ++b) {
      for (std::size_t j = 0; j < dims; ++j) {
        stored[b].push_back(
            load_ct(in_dir + "/stored_" + std::to_string(b) + "_" + std::to_string(j) + ".bin"));
      }
    }

    const auto gpu = ombra::core::gpu_eval_column_search_batched(stored, queries);

    bool ok = true;
    for (std::size_t q = 0; q < nq; ++q) {
      for (std::size_t b = 0; b < blocks; ++b) {
        Ciphertext<DCRTPoly> cpu = cc->EvalMultNoRelin(stored[b][0], queries[q][0]);
        for (std::size_t j = 1; j < dims; ++j) {
          cpu = cc->EvalAdd(cpu, cc->EvalMultNoRelin(stored[b][j], queries[q][j]));
        }
        const bool one_ok = identical(gpu[q][b], cpu);
        ok = ok && one_ok;
        std::printf("batched search query %zu block %zu exact RNS match: %s\n", q, b,
                    one_ok ? "PASS" : "FAIL");
        const std::string tag = std::to_string(q) + "_" + std::to_string(b) + ".bin";
        save_ct(out_dir + "/gpu_" + tag, gpu[q][b]);
        save_ct(out_dir + "/cpu_" + tag, cpu);
      }
    }
    std::printf("batched column search exact RNS match: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
