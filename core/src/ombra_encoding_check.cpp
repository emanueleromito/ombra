// ombra_encoding_check: reads the polynomial coefficients OpenFHE produces when it encodes given
// complex slot values (T0.5 encoding test).
//
// Usage: ombra_encoding_check <context.bin> <slots.bin> <q.bin> <d>
//
// <slots.bin> holds N/2 complex slot values as interleaved little-endian doubles (re, im); <q.bin>
// holds d doubles. The plaintext is encoded with MakeCKKSPackedPlaintext (complex overload), its
// DCRTPoly is interpolated over the CRT basis to centered big integers, and the maximum of
// |coefficient / scale - target| over all N coefficients is printed, where target is q_j for
// j < d and 0 otherwise. Needs no keys. Prints only that maximum.
// Exit code: 0 ok, 2 error.

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "cryptocontext-ser.h"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

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
        const std::size_t d = std::stoul(argv[4]);
        const std::size_t ring = cc->GetRingDimension();
        const std::size_t slots = ring / 2;
        const std::vector<double> raw = read_doubles(argv[2], 2 * slots);
        const std::vector<double> q = read_doubles(argv[3], d);
        std::vector<std::complex<double>> values(slots);
        for (std::size_t k = 0; k < slots; ++k) {
            values[k] = {raw[2 * k], raw[2 * k + 1]};
        }

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
            const double want = (j < d) ? q[j] : 0.0;
            worst = std::max(worst, std::abs(c / scale - want));
        }
        std::printf("encoding check: d=%zu max_coeff_error_in_scale_units=%.3e\n", d, worst);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
