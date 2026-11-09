// ombra_expand: CPU reference for query expansion on OpenFHE (T0.5, decision 0010).
//
// Usage: ombra_expand <context.bin> <query.bin> <autokeys.bin> <out_dir> <d>
//
// Inputs are OpenFHE BINARY serializations of public material only: the CryptoContext, one query
// ciphertext (symmetric coefficient-packed, see client/ombra/query.py) and the automorphism keys from
// EvalAutomorphismKeyGen. Runs the SealPIR-style tree of expansion_tree.hpp with
//   - EvalAutomorphism (base-leveledshe.cpp:382, key switch then permute, :411-436) for tau_t,
//   - EvalAdd / EvalSub,
//   - an exact multiplication by X^(-k) done on the coefficients of every RNS tower (no level,
//     no scale change),
// The tree puts a factor 2^L on every value; the client has already multiplied the query by 2^-L
// before encoding (client/ombra/query.py), so the leaves are q_j with no server-side EvalMult, no
// scale change and no level used. The driver checks that: every leaf must have NoiseScaleDeg 1 and
// Level 0, else it exits 2.
// Writes expanded_<j>.bin for j in [0, d) to <out_dir>. Never takes a secret key and prints no
// data values: only depth, node counts and timing.
// Exit code: 0 ok, 2 error.

#include <chrono>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/expansion_tree.hpp"
#include "openfhe.h"
#include "openfhe_expand_ops.hpp"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

namespace {

using ombra::core::cpu_ref::OpenFheOps;
using Ct = Ciphertext<DCRTPoly>;

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: ombra_expand <context> <query> <autokeys> <out_dir> <d>\n");
        return 2;
    }
    try {
        const std::string out_dir = argv[4];
        const std::size_t d = std::stoul(argv[5]);

        CryptoContext<DCRTPoly> cc;
        if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
            throw std::runtime_error("cannot read crypto context");
        }
        std::ifstream keyfile(argv[3], std::ios::in | std::ios::binary);
        if (!keyfile.is_open() ||
            !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
            throw std::runtime_error("cannot read automorphism keys");
        }
        Ct query;
        if (!Serial::DeserializeFromFile(argv[2], query, SerType::BINARY)) {
            throw std::runtime_error("cannot read query ciphertext");
        }
        const auto& keys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(query->GetKeyTag());

        OpenFheOps ops(cc, keys);
        ombra::core::ExpansionStats stats;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<Ct> out = ombra::core::expand_tree(query, d, cc->GetRingDimension(), ops, &stats);
        const auto t1 = std::chrono::steady_clock::now();

        for (std::size_t j = 0; j < out.size(); ++j) {
            if (out[j]->GetNoiseScaleDeg() != 1 || out[j]->GetLevel() != 0) {
                throw std::runtime_error("expansion leaf is not at NoiseScaleDeg 1 and Level 0");
            }
            const std::string path = out_dir + "/expanded_" + std::to_string(j) + ".bin";
            if (!Serial::SerializeToFile(path, out[j], SerType::BINARY)) {
                throw std::runtime_error("cannot write " + path);
            }
        }
        std::printf("d=%zu depth=%zu outputs=%zu nodes=%zu automorphisms=%zu seconds=%.3f\n",
                    d, stats.depth, out.size(), stats.nodes, stats.automorphisms,
                    std::chrono::duration<double>(t1 - t0).count());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
