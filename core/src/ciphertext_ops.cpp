#include "ombra/core/ciphertext_ops.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ombra/core/rns_basic.hpp"

namespace ombra::core {
namespace {

using namespace lbcrypto;

// Words are copied as 64-bit values. Valid only when OpenFHE's native integer is 64 bits wide
// (NATIVE_SIZE=64, the default; CMake also checks OpenFHE_NATIVE_SIZE).
static_assert(sizeof(NativeInteger) == sizeof(std::uint64_t),
              "OpenFHE must be built with NATIVE_SIZE=64");

// Flattened view of a polynomial set: [poly][tower][coeff] words plus the moduli of the towers.
struct Flat {
    RnsShape shape{};
    std::vector<std::uint64_t> words;
    std::vector<std::uint64_t> moduli;  // one per tower, from the ciphertext itself
};

Flat flatten(const std::vector<DCRTPoly>& polys) {
    if (polys.empty()) throw std::invalid_argument("ciphertext has no components");
    Flat f;
    f.shape.num_polys = polys.size();
    f.shape.num_towers = polys[0].GetNumOfElements();
    f.shape.ring_dim = polys[0].GetRingDimension();
    f.words.reserve(f.shape.total_words());
    for (std::size_t p = 0; p < polys.size(); ++p) {
        const DCRTPoly& poly = polys[p];
        if (poly.GetFormat() != Format::EVALUATION) {
            throw std::invalid_argument("polynomial is not in evaluation form");
        }
        if (poly.GetNumOfElements() != f.shape.num_towers ||
            poly.GetRingDimension() != f.shape.ring_dim) {
            throw std::invalid_argument("components differ in shape");
        }
        const auto& towers = poly.GetAllElements();
        for (std::size_t t = 0; t < towers.size(); ++t) {
            const std::uint64_t q = towers[t].GetModulus().ConvertToInt<std::uint64_t>();
            if (p == 0) {
                f.moduli.push_back(q);
            } else if (f.moduli[t] != q) {
                throw std::invalid_argument("components differ in moduli");
            }
            const auto& v = towers[t].GetValues();
            for (std::size_t i = 0; i < f.shape.ring_dim; ++i) {
                f.words.push_back(v[i].ConvertToInt<std::uint64_t>());
            }
        }
    }
    return f;
}

void write_back(std::vector<DCRTPoly>& polys, const Flat& shape_src,
                const std::vector<std::uint64_t>& words) {
    const RnsShape& s = shape_src.shape;
    for (std::size_t p = 0; p < s.num_polys; ++p) {
        auto& towers = polys[p].GetAllElements();
        for (std::size_t t = 0; t < s.num_towers; ++t) {
            const NativeInteger modulus(shape_src.moduli[t]);
            NativeVector v(s.ring_dim, modulus);
            const std::uint64_t* src = words.data() + (p * s.num_towers + t) * s.ring_dim;
            for (std::size_t i = 0; i < s.ring_dim; ++i) v[i] = NativeInteger(src[i]);
            towers[t].SetValues(std::move(v), Format::EVALUATION);
        }
    }
}

void require_fixedmanual(const CiphertextImpl<DCRTPoly>& ct) {
    const auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ct.GetCryptoParameters());
    if (!params || params->GetScalingTechnique() != FIXEDMANUAL) {
        throw std::invalid_argument("only CKKS with FIXEDMANUAL scaling is supported");
    }
}

}  // namespace

Ciphertext<DCRTPoly> gpu_eval_add(ConstCiphertext<DCRTPoly>& a, ConstCiphertext<DCRTPoly>& b) {
    require_fixedmanual(*a);
    if (a->GetLevel() != b->GetLevel() || a->GetNoiseScaleDeg() != b->GetNoiseScaleDeg() ||
        a->GetScalingFactor() != b->GetScalingFactor()) {
        throw std::invalid_argument("add: operands differ in level, noise degree or scale");
    }
    const Flat fa = flatten(a->GetElements());
    const Flat fb = flatten(b->GetElements());
    if (fa.shape.num_polys != fb.shape.num_polys || fa.shape.num_towers != fb.shape.num_towers ||
        fa.shape.ring_dim != fb.shape.ring_dim || fa.moduli != fb.moduli) {
        throw std::invalid_argument("add: operands differ in shape or moduli");
    }
    std::vector<std::uint64_t> out;
    rns_add(fa.words, fb.words, out, fa.shape, fa.moduli);

    Ciphertext<DCRTPoly> result = a->Clone();  // carries context, key tag, encoding, slots
    write_back(result->GetElements(), fa, out);
    result->SetLevel(a->GetLevel());
    result->SetNoiseScaleDeg(a->GetNoiseScaleDeg());
    result->SetScalingFactor(a->GetScalingFactor());
    return result;
}

Ciphertext<DCRTPoly> gpu_eval_mult_plain(ConstCiphertext<DCRTPoly>& ct, ConstPlaintext& pt) {
    require_fixedmanual(*ct);
    if (ct->GetLevel() != pt->GetLevel()) {
        throw std::invalid_argument("mult: ciphertext and plaintext differ in level");
    }
    const Flat fc = flatten(ct->GetElements());
    DCRTPoly pe = pt->GetElement<DCRTPoly>();  // copy: the plaintext element is immutable
    pe.SetFormat(Format::EVALUATION);
    const Flat fp = flatten({pe});
    if (fp.shape.num_towers != fc.shape.num_towers || fp.shape.ring_dim != fc.shape.ring_dim ||
        fp.moduli != fc.moduli) {
        throw std::invalid_argument("mult: plaintext differs in shape or moduli");
    }
    std::vector<std::uint64_t> out;
    rns_mul_by_poly(fc.words, fp.words, out, fc.shape, fc.moduli);

    Ciphertext<DCRTPoly> result = ct->Clone();
    write_back(result->GetElements(), fc, out);
    result->SetLevel(ct->GetLevel());
    result->SetNoiseScaleDeg(ct->GetNoiseScaleDeg() + pt->GetNoiseScaleDeg());
    result->SetScalingFactor(ct->GetScalingFactor() * pt->GetScalingFactor());
    return result;
}

}  // namespace ombra::core
