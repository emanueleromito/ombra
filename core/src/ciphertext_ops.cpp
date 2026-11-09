#include "ombra/core/ciphertext_ops.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ombra/core/column_search.hpp"
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

void write_back(std::vector<DCRTPoly>& polys, const Flat& shape_src, const std::uint64_t* words) {
    const RnsShape& s = shape_src.shape;
    for (std::size_t p = 0; p < s.num_polys; ++p) {
        auto& towers = polys[p].GetAllElements();
        for (std::size_t t = 0; t < s.num_towers; ++t) {
            const NativeInteger modulus(shape_src.moduli[t]);
            NativeVector v(s.ring_dim, modulus);
            const std::uint64_t* src = words + (p * s.num_towers + t) * s.ring_dim;
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
    write_back(result->GetElements(), fa, out.data());
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
    write_back(result->GetElements(), fc, out.data());
    result->SetLevel(ct->GetLevel());
    result->SetNoiseScaleDeg(ct->GetNoiseScaleDeg() + pt->GetNoiseScaleDeg());
    result->SetScalingFactor(ct->GetScalingFactor() * pt->GetScalingFactor());
    return result;
}

Ciphertext<DCRTPoly> unflatten_ciphertext(const Ciphertext<DCRTPoly>& like,
                                          const std::vector<std::uint64_t>& words) {
    const Flat shape = flatten(like->GetElements());
    if (words.size() != shape.shape.total_words()) {
        throw std::invalid_argument("unflatten: word count does not match the template");
    }
    Ciphertext<DCRTPoly> result = like->Clone();
    write_back(result->GetElements(), shape, words.data());
    return result;
}

FlatCiphertexts flatten_ciphertexts(const std::vector<Ciphertext<DCRTPoly>>& cts) {
    if (cts.empty()) throw std::invalid_argument("no ciphertexts to flatten");
    FlatCiphertexts out;
    out.count = cts.size();
    out.level = cts[0]->GetLevel();
    out.noise_scale_deg = cts[0]->GetNoiseScaleDeg();
    out.scaling_factor = cts[0]->GetScalingFactor();
    for (std::size_t i = 0; i < cts.size(); ++i) {
        const auto& ct = cts[i];
        require_fixedmanual(*ct);
        if (ct->GetElements().size() != 2) {
            throw std::invalid_argument("only 2-component ciphertexts are supported");
        }
        if (ct->GetLevel() != out.level || ct->GetNoiseScaleDeg() != out.noise_scale_deg ||
            ct->GetScalingFactor() != out.scaling_factor) {
            throw std::invalid_argument("ciphertexts differ in level, noise degree or scale");
        }
        Flat f = flatten(ct->GetElements());
        if (i == 0) {
            out.num_towers = f.shape.num_towers;
            out.ring_dim = f.shape.ring_dim;
            out.moduli = f.moduli;
            out.words.reserve(cts.size() * f.words.size());
        } else if (f.shape.num_towers != out.num_towers || f.shape.ring_dim != out.ring_dim ||
                   f.moduli != out.moduli) {
            throw std::invalid_argument("ciphertexts differ in shape or moduli");
        }
        out.words.insert(out.words.end(), f.words.begin(), f.words.end());
    }
    return out;
}

std::vector<Ciphertext<DCRTPoly>> gpu_eval_column_search(
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& stored_blocks,
    const std::vector<Ciphertext<DCRTPoly>>& query) {
    if (stored_blocks.empty() || query.empty()) {
        throw std::invalid_argument("column search: no stored blocks or empty query");
    }
    const FlatCiphertexts fq = flatten_ciphertexts(query);
    const ColumnSearchShape shape{query.size(), stored_blocks.size(), fq.num_towers, fq.ring_dim};

    std::vector<std::uint64_t> stored;
    stored.reserve(shape.stored_words());
    std::size_t noise_deg = 0;
    double scale = 0;
    for (const auto& block : stored_blocks) {
        if (block.size() != query.size()) {
            throw std::invalid_argument("column search: block and query differ in dimensions");
        }
        const FlatCiphertexts fb = flatten_ciphertexts(block);
        if (fb.level != fq.level || fb.scaling_factor != fq.scaling_factor ||
            fb.noise_scale_deg != fq.noise_scale_deg || fb.num_towers != fq.num_towers ||
            fb.ring_dim != fq.ring_dim || fb.moduli != fq.moduli) {
            throw std::invalid_argument(
                "column search: stored and query differ in level, noise degree, scale or moduli");
        }
        stored.insert(stored.end(), fb.words.begin(), fb.words.end());
    }
    noise_deg = fq.noise_scale_deg * 2;  // equal degrees on both sides: sum of the degrees
    scale = fq.scaling_factor * fq.scaling_factor;

    std::vector<std::uint64_t> out;
    column_search(stored, fq.words, out, shape, fq.moduli);

    Flat meta;
    meta.shape = RnsShape{3, shape.num_towers, shape.ring_dim};
    meta.moduli = fq.moduli;
    std::vector<Ciphertext<DCRTPoly>> results;
    for (std::size_t b = 0; b < stored_blocks.size(); ++b) {
        Ciphertext<DCRTPoly> r = stored_blocks[b][0]->Clone();  // context, key tag, encoding, slots
        r->GetElements().push_back(r->GetElements()[0]);        // third component (same shape)
        write_back(r->GetElements(), meta, out.data() + b * 3 * shape.poly_words());
        r->SetLevel(fq.level);
        r->SetNoiseScaleDeg(noise_deg);
        r->SetScalingFactor(scale);
        results.push_back(std::move(r));
    }
    return results;
}

}  // namespace ombra::core
