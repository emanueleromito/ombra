// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// ombra_security_floor_wiring: shows that each library entry point enforces the security floor by
// itself. It deliberately never calls require_security_floor; the test sets
// OMBRA_TEST_LOG2QP_LIMIT=1 so a valid preset context falls below the limit.
//
// Usage: ombra_security_floor_wiring <context.bin> <evalmultkey.bin> <ciphertext.bin>
// Prints "log2qp: <n>" and one line per entry point: "<name>: PASS" when it threw
// std::invalid_argument containing "security floor", else "<name>: FAIL (<reason>)".
// Exit code: 0 all refused, 1 some did not, 2 error. No data values are printed.

#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/gpu_search.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;

namespace {

int g_failures = 0;

void expect_refused(const std::string& name, const std::function<void()>& fn) {
  try {
    fn();
    std::printf("%s: FAIL (did not throw)\n", name.c_str());
  } catch (const std::invalid_argument& e) {
    if (std::string(e.what()).find("security floor") != std::string::npos) {
      std::printf("%s: PASS\n", name.c_str());
      return;
    }
    std::printf("%s: FAIL (other error: %s)\n", name.c_str(), e.what());
  } catch (const std::exception& e) {
    std::printf("%s: FAIL (other error: %s)\n", name.c_str(), e.what());
  }
  ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: ombra_security_floor_wiring <context> <evalmultkey> <ciphertext>\n");
    return 2;
  }
  try {
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    std::ifstream keyfile(argv[2], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read the EvalMult key");
    }
    Ct ct;
    if (!Serial::DeserializeFromFile(argv[3], ct, SerType::BINARY)) {
      throw std::runtime_error("cannot read ciphertext");
    }
    const auto& key_vec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(ct->GetKeyTag());
    if (key_vec.empty()) throw std::runtime_error("no EvalMult key for this key tag");

    std::printf("log2qp: %zu\n", ombra::core::context_log2_qp(cc));

    expect_refused("extract_expansion_tables",
                   [&] { (void)ombra::core::extract_expansion_tables(cc, ct); });
    expect_refused("GpuExpansion", [&] {
      ombra::core::GpuExpansion::KeyMap keys;
      ombra::core::GpuExpansion e(cc, ct, keys, 2);
    });
    expect_refused("GpuCtOps", [&] { ombra::core::GpuCtOps ops(cc, ct, key_vec[0]); });
    expect_refused("GpuSearch", [&] { ombra::core::GpuSearch gs(cc, ct, ct, key_vec[0], 1, 1); });
    expect_refused("gpu_eval_add (require_fixedmanual)",
                   [&] { (void)ombra::core::gpu_eval_add(ct, ct); });
    return g_failures ? 1 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
