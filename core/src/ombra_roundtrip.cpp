// ombra_roundtrip: oracle driver for the GPU RNS kernels (T0.3).
//
// Usage: ombra_roundtrip <context.bin> <ct_a.bin> <ct_b.bin> <pt_values.bin> <out_dir>
//
// Inputs (all written by the Python test with OpenFHE's own BINARY serialization, public material
// only): the CryptoContext, two ciphertexts, and a test-only plaintext vector file (uint32
// version = 1, uint32 count, then `count` little-endian float64; synthetic test data).
// Computes ct_a + ct_b and ct_a * encode(values) both with the GPU kernels and with OpenFHE EvalAdd /
// EvalMult, compares the raw RNS polynomials and metadata exactly, and writes
// add_gpu.bin, add_cpu.bin, mul_gpu.bin, mul_cpu.bin (OpenFHE ciphertext serialization) to out_dir.
// Never reads or writes a secret key. Prints only PASS/FAIL lines, never data values.
// Exit code: 0 all exact matches, 1 mismatch, 2 error.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;

namespace {

std::vector<double> read_values(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::uint32_t version = 0, count = 0;
    in.read(reinterpret_cast<char*>(&version), sizeof version);
    in.read(reinterpret_cast<char*>(&count), sizeof count);
    if (!in || version != 1) throw std::runtime_error("bad plaintext input file header");
    std::vector<double> v(count);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!in) throw std::runtime_error("truncated plaintext input file");
    return v;
}

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

// Exact comparison of all RNS words, moduli and metadata.
bool identical(const Ciphertext<DCRTPoly>& x, const Ciphertext<DCRTPoly>& y) {
    if (x->GetLevel() != y->GetLevel() || x->GetNoiseScaleDeg() != y->GetNoiseScaleDeg() ||
        x->GetScalingFactor() != y->GetScalingFactor() ||
        x->GetElements().size() != y->GetElements().size()) {
        return false;
    }
    for (std::size_t c = 0; c < x->GetElements().size(); ++c) {
        const DCRTPoly& px = x->GetElements()[c];
        const DCRTPoly& py = y->GetElements()[c];
        if (px.GetFormat() != py.GetFormat() || px.GetNumOfElements() != py.GetNumOfElements()) {
            return false;
        }
        for (std::size_t t = 0; t < px.GetNumOfElements(); ++t) {
            const auto& tx = px.GetElementAtIndex(t);
            const auto& ty = py.GetElementAtIndex(t);
            if (tx.GetModulus() != ty.GetModulus() || !(tx.GetValues() == ty.GetValues())) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: ombra_roundtrip <context> <ct_a> <ct_b> <pt_values> <out_dir>\n");
        return 2;
    }
    try {
        const std::string out_dir = argv[5];
        CryptoContext<DCRTPoly> cc;
        if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
            throw std::runtime_error("cannot read crypto context");
        }
        ombra::core::require_security_floor(cc);
        const auto a = load_ct(argv[2]);
        const auto b = load_ct(argv[3]);
        // Same encoding call the Python side's expected values assume: scale degree 1, level 0.
        Plaintext pt = cc->MakeCKKSPackedPlaintext(read_values(argv[4]));

        const auto add_gpu = ombra::core::gpu_eval_add(a, b);
        const auto add_cpu = cc->EvalAdd(a, b);
        const auto mul_gpu = ombra::core::gpu_eval_mult_plain(a, pt);
        const auto mul_cpu = cc->EvalMult(a, pt);

        const bool add_ok = identical(add_gpu, add_cpu);
        const bool mul_ok = identical(mul_gpu, mul_cpu);
        std::printf("add exact RNS match: %s\n", add_ok ? "PASS" : "FAIL");
        std::printf("mult-by-plaintext exact RNS match: %s\n", mul_ok ? "PASS" : "FAIL");

        save_ct(out_dir + "/add_gpu.bin", add_gpu);
        save_ct(out_dir + "/add_cpu.bin", add_cpu);
        save_ct(out_dir + "/mul_gpu.bin", mul_gpu);
        save_ct(out_dir + "/mul_cpu.bin", mul_cpu);
        return (add_ok && mul_ok) ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
