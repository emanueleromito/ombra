// ombra_encoding_check: reads the polynomial coefficients OpenFHE produces when it encodes given
// real slot values (T0.5 encoding test).
//
// Usage: ombra_encoding_check <context.bin> <slots.bin> <q.bin> <d>
//
// <slots.bin> holds N/2 real slot values as little-endian doubles; <q.bin> holds d doubles. The
// plaintext is encoded with the real MakeCKKSPackedPlaintext, its DCRTPoly is interpolated over the
// CRT basis to centered big integers, and the maximum of |coefficient / scale - target| over all N
// coefficients is printed, where target is q_j at index j, -q_j at index N - j (j >= 1) and 0
// elsewhere. Needs no keys and makes no CUDA calls. Prints only that maximum.
// Exit code: 0 ok, 2 error.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "cryptocontext-ser.h"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

#include "ombra/core/security_floor_context.hpp"

using namespace lbcrypto;

namespace {

std::vector<double> read_doubles(const std::string& path, std::size_t count) {
    std::ifstream f(path, std::ios::in | std::ios::binary);
    std::vector<double> v(count);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(double)));
    if (!f || f.gcount() != static_cast<std::streamsize>(count * sizeof(double))) {
        throw std::runtime_error("cannot read " + path);
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "usage: ombra_encoding_check <context> <slots> <q> <d>\n");
        return 2;
    }
    try {
        CryptoContext<DCRTPoly> cc;
        if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
            throw std::runtime_error("cannot read crypto context");
        }
        ombra::core::require_security_floor(cc);
        const std::size_t d = std::stoul(argv[4]);
        const std::size_t ring = cc->GetRingDimension();
        if (d == 0 || d > ring / 2) {
            throw std::runtime_error("d must be in [1, N/2]");
        }
        const std::vector<double> values = read_doubles(argv[2], ring / 2);
        const std::vector<double> q = read_doubles(argv[3], d);

        Plaintext pt = cc->MakeCKKSPackedPlaintext(values);
        DCRTPoly poly = pt->GetElement<DCRTPoly>();
        poly.SetFormat(Format::COEFFICIENT);
        const auto big = poly.CRTInterpolate();
        const BigInteger modulus = big.GetModulus();
        const BigInteger half = modulus.DividedBy(BigInteger(2));
        const double scale = pt->GetScalingFactor();

        double worst = 0.0;
        for (std::size_t j = 0; j < ring; ++j) {
            const BigInteger& v = big.GetValues()[j];
            const double c = (v > half) ? -(modulus - v).ConvertToDouble() : v.ConvertToDouble();
            double want = 0.0;
            if (j < d) {
                want = q[j];
            } else if (j > ring - d) {  // index N - i holds -q_i for 1 <= i < d
                want = -q[ring - j];
            }
            worst = std::max(worst, std::abs(c / scale - want));
        }
        std::printf("encoding check: d=%zu max_coeff_error_in_scale_units=%.3e\n", d, worst);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
