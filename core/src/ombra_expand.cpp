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
// then removes the factor 2^L the tree put on every value by multiplying each output by the exact
// constant 2^-L on the server (EvalMult by a double: scale degree +1, no rescale). Adjusting the
// ciphertext scaling factor instead does nothing under FIXEDMANUAL, whose decode ignores it
// (ckkspackedencoding.cpp:315-363), so the leaves would decrypt to 2^L * q_j.
// Writes expanded_<j>.bin for j in [0, d) to <out_dir>. Never takes a secret key and prints no
// data values: only depth, node counts and timing.
// Exit code: 0 ok, 2 error.

#include <chrono>
#include <cmath>
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
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

namespace {

using Ct = Ciphertext<DCRTPoly>;

// Multiply every element of the ciphertext by X^(-k) in Z[X]/(X^N+1), exactly: coefficient i moves
// to i - k, negated when it wraps around. No level is used and the scaling factor is unchanged.
Ct mul_by_neg_monomial(const Ct& ct, std::size_t k) {
    Ct out = ct->Clone();
    std::vector<DCRTPoly> elems = ct->GetElements();
    for (DCRTPoly& poly : elems) {
        const Format original = poly.GetFormat();
        poly.SetFormat(Format::COEFFICIENT);
        for (std::size_t t = 0; t < poly.GetNumOfElements(); ++t) {
            NativePoly tower = poly.GetElementAtIndex(t);
            const NativeInteger q = tower.GetModulus();
            const std::size_t n = tower.GetLength();
            const NativeVector& src = tower.GetValues();
            NativeVector dst(n, q);
            for (std::size_t i = 0; i < n; ++i) {
                if (i >= k) {
                    dst[i - k] = src[i];
                } else {
                    dst[i - k + n] = (src[i] == NativeInteger(0)) ? NativeInteger(0) : q - src[i];
                }
            }
            tower.SetValues(std::move(dst), Format::COEFFICIENT);
            poly.SetElementAtIndex(t, std::move(tower));
        }
        poly.SetFormat(original);
    }
    out->SetElements(std::move(elems));
    return out;
}

class OpenFheOps {
public:
    OpenFheOps(CryptoContext<DCRTPoly> cc, const std::map<uint32_t, EvalKey<DCRTPoly>>& keys)
        : cc_(std::move(cc)), keys_(keys) {}

    std::pair<Ct, Ct> split(const Ct& c, std::size_t /*level*/, std::uint32_t galois,
                            std::size_t shift, bool want_odd) {
        Ct sub = cc_->EvalAutomorphism(c, galois, keys_);
        Ct even = cc_->EvalAdd(c, sub);
        if (!want_odd) {
            return {std::move(even), Ct()};
        }
        return {std::move(even), mul_by_neg_monomial(cc_->EvalSub(c, sub), shift)};
    }

private:
    CryptoContext<DCRTPoly> cc_;
    const std::map<uint32_t, EvalKey<DCRTPoly>>& keys_;
};

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

        const double factor = std::ldexp(1.0, static_cast<int>(stats.depth));
        for (std::size_t j = 0; j < out.size(); ++j) {
            out[j] = cc->EvalMult(out[j], 1.0 / factor);
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
