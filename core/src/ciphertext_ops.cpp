#include "ombra/core/ciphertext_ops.hpp"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "ombra/core/column_search.hpp"
#include "ombra/core/rns_basic.hpp"
#include "ombra/core/security_floor_context.hpp"

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
    require_security_floor(ct.GetCryptoContext());
    const auto params = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ct.GetCryptoParameters());
    if (!params || params->GetScalingTechnique() != FIXEDMANUAL) {
        throw std::invalid_argument("only CKKS with FIXEDMANUAL scaling is supported");
    }
}

}  // namespace

void require_same_key(ConstCiphertext<DCRTPoly> a, ConstCiphertext<DCRTPoly> b,
                      const std::string& what) {
    if (a->GetKeyTag() != b->GetKeyTag()) {
        throw std::invalid_argument(what + ": ciphertexts are encrypted under different keys");
    }
    if (a->GetEncodingType() != b->GetEncodingType()) {
        throw std::invalid_argument(what + ": ciphertexts differ in encoding type");
    }
}

void require_eval_key_matches(const EvalKey<DCRTPoly>& key, ConstCiphertext<DCRTPoly> ct,
                              const std::string& what) {
    if (!key || key->GetKeyTag() != ct->GetKeyTag()) {
        throw std::invalid_argument(what + ": evaluation key belongs to a different key");
    }
}

Ciphertext<DCRTPoly> gpu_eval_add(ConstCiphertext<DCRTPoly>& a, ConstCiphertext<DCRTPoly>& b) {
    require_fixedmanual(*a);
    require_fixedmanual(*b);
    require_same_key(a, b, "add");
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
    if (ct->GetEncodingType() != pt->GetEncodingType()) {
        throw std::invalid_argument("mult: ciphertext and plaintext differ in encoding type");
    }
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

Ciphertext<DCRTPoly> gpu_eval_plain_search(const std::vector<Ciphertext<DCRTPoly>>& query,
                                           const std::vector<Plaintext>& plaintexts,
                                           const std::vector<std::uint64_t>& words) {
    if (query.empty() || query.size() != plaintexts.size()) {
        throw std::invalid_argument("plain search: query and plaintexts differ in count or empty");
    }
    const auto& q0 = query[0];
    require_fixedmanual(*q0);
    for (std::size_t j = 0; j < query.size(); ++j) {
        require_fixedmanual(*query[j]);
        require_same_key(query[j], q0, "plain search");
        if (plaintexts[j]->GetEncodingType() != q0->GetEncodingType()) {
            throw std::invalid_argument(
                "plain search: query and plaintext differ in encoding type");
        }
        if (query[j]->GetLevel() != q0->GetLevel() ||
            query[j]->GetNoiseScaleDeg() != q0->GetNoiseScaleDeg() ||
            query[j]->GetScalingFactor() != q0->GetScalingFactor() ||
            plaintexts[j]->GetLevel() != q0->GetLevel() ||
            plaintexts[j]->GetNoiseScaleDeg() != plaintexts[0]->GetNoiseScaleDeg() ||
            plaintexts[j]->GetScalingFactor() != plaintexts[0]->GetScalingFactor()) {
            throw std::invalid_argument("plain search: operands differ in level, degree or scale");
        }
    }
    Ciphertext<DCRTPoly> result = unflatten_ciphertext(q0, words);  // context and shape only
    result->SetLevel(q0->GetLevel());
    result->SetNoiseScaleDeg(q0->GetNoiseScaleDeg() + plaintexts[0]->GetNoiseScaleDeg());
    result->SetScalingFactor(q0->GetScalingFactor() * plaintexts[0]->GetScalingFactor());
    return result;
}

FlatOne flatten_one(const Ciphertext<DCRTPoly>& ct) {
    require_fixedmanual(*ct);
    Flat f = flatten(ct->GetElements());
    FlatOne out;
    out.components = f.shape.num_polys;
    out.num_towers = f.shape.num_towers;
    out.ring_dim = f.shape.ring_dim;
    out.words = std::move(f.words);
    out.moduli = std::move(f.moduli);
    out.level = ct->GetLevel();
    out.noise_scale_deg = ct->GetNoiseScaleDeg();
    out.scaling_factor = ct->GetScalingFactor();
    out.key_tag = ct->GetKeyTag();
    out.encoding = ct->GetEncodingType();
    return out;
}

Ciphertext<DCRTPoly> make_ciphertext_like(const Ciphertext<DCRTPoly>& like, std::size_t components,
                                          std::size_t towers,
                                          const std::vector<std::uint64_t>& words,
                                          std::size_t level, std::size_t noise_scale_deg,
                                          double scaling_factor) {
    const std::size_t like_towers = like->GetElements()[0].GetNumOfElements();
    if (components < 2 || towers == 0 || towers > like_towers) {
        throw std::invalid_argument("make_ciphertext_like: bad component or tower count");
    }
    Ciphertext<DCRTPoly> result = like->Clone();
    auto& elements = result->GetElements();
    const DCRTPoly first = elements[0];
    while (elements.size() < components) elements.push_back(first);
    elements.resize(components);
    for (auto& e : elements) {
        e.SetFormat(Format::EVALUATION);
        if (towers < like_towers) e.DropLastElements(like_towers - towers);
    }
    Flat shape = flatten(elements);
    if (words.size() != shape.shape.total_words()) {
        throw std::invalid_argument("make_ciphertext_like: word count does not match the shape");
    }
    write_back(elements, shape, words.data());
    // Degree before level: SetLevel checks the tower count against the current degree.
    result->SetNoiseScaleDeg(noise_scale_deg);
    result->SetLevel(level);
    result->SetScalingFactor(scaling_factor);
    return result;
}

FlatCiphertexts flatten_ciphertexts(const std::vector<Ciphertext<DCRTPoly>>& cts) {
    if (cts.empty()) throw std::invalid_argument("no ciphertexts to flatten");
    FlatCiphertexts out;
    out.count = cts.size();
    out.level = cts[0]->GetLevel();
    out.noise_scale_deg = cts[0]->GetNoiseScaleDeg();
    out.scaling_factor = cts[0]->GetScalingFactor();
    out.key_tag = cts[0]->GetKeyTag();
    out.encoding = cts[0]->GetEncodingType();
    for (std::size_t i = 0; i < cts.size(); ++i) {
        const auto& ct = cts[i];
        require_fixedmanual(*ct);
        require_same_key(ct, cts[0], "flatten");
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
        if (fb.key_tag != fq.key_tag || fb.encoding != fq.encoding) {
            throw std::invalid_argument(
                "column search: stored and query are encrypted under different keys");
        }
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

std::vector<std::vector<Ciphertext<DCRTPoly>>> gpu_eval_column_search_batched(
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& stored_blocks,
    const std::vector<std::vector<Ciphertext<DCRTPoly>>>& queries) {
    if (stored_blocks.empty() || queries.empty() || queries[0].empty()) {
        throw std::invalid_argument("column search: no stored blocks or empty queries");
    }
    const std::size_t dims = queries[0].size();
    std::vector<std::uint64_t> qwords;
    FlatCiphertexts fq0;
    for (std::size_t q = 0; q < queries.size(); ++q) {
        if (queries[q].size() != dims) {
            throw std::invalid_argument("column search: queries differ in dimensions");
        }
        FlatCiphertexts fq = flatten_ciphertexts(queries[q]);
        if (q == 0) {
            qwords.reserve(queries.size() * fq.words.size());
            fq0 = fq;
        } else {
            // Ring dimension and tower count first: they size the buffers read below.
            if (fq.ring_dim != fq0.ring_dim || fq.num_towers != fq0.num_towers) {
                throw std::invalid_argument(
                    "column search: queries differ in ring dimension or tower count");
            }
            if (fq.key_tag != fq0.key_tag || fq.encoding != fq0.encoding) {
                throw std::invalid_argument(
                    "column search: queries are encrypted under different keys");
            }
            if (fq.level != fq0.level || fq.scaling_factor != fq0.scaling_factor ||
                fq.noise_scale_deg != fq0.noise_scale_deg || fq.moduli != fq0.moduli) {
                throw std::invalid_argument(
                    "column search: queries differ in level, degree or scale");
            }
        }
        if (fq.words.size() != fq.count * 2 * fq.num_towers * fq.ring_dim) {
            throw std::invalid_argument("column search: query buffer has an unexpected size");
        }
        qwords.insert(qwords.end(), fq.words.begin(), fq.words.end());
    }
    const ColumnSearchShape shape{dims, stored_blocks.size(), fq0.num_towers, fq0.ring_dim};

    BatchedColumnSearchDevice dev(shape, fq0.moduli, queries.size());
    for (std::size_t b = 0; b < stored_blocks.size(); ++b) {
        if (stored_blocks[b].size() != dims) {
            throw std::invalid_argument("column search: block and query differ in dimensions");
        }
        const FlatCiphertexts fb = flatten_ciphertexts(stored_blocks[b]);
        if (fb.key_tag != fq0.key_tag || fb.encoding != fq0.encoding) {
            throw std::invalid_argument(
                "column search: stored and query are encrypted under different keys");
        }
        if (fb.level != fq0.level || fb.scaling_factor != fq0.scaling_factor ||
            fb.noise_scale_deg != fq0.noise_scale_deg || fb.num_towers != fq0.num_towers ||
            fb.ring_dim != fq0.ring_dim || fb.moduli != fq0.moduli) {
            throw std::invalid_argument(
                "column search: stored and query differ in level, noise degree, scale or moduli");
        }
        dev.upload_block(b, fb.words.data(), fb.words.size());
    }
    if (qwords.size() != queries.size() * shape.query_words()) {
        throw std::invalid_argument("column search: query buffer has an unexpected size");
    }
    for (std::size_t q = 0; q < queries.size(); ++q) {
        dev.upload_query(q, qwords.data() + q * shape.query_words(), shape.query_words());
    }
    dev.run(queries.size());
    std::vector<std::uint64_t> out;
    dev.download_result(queries.size(), out);

    Flat meta;
    meta.shape = RnsShape{3, shape.num_towers, shape.ring_dim};
    meta.moduli = fq0.moduli;
    std::vector<std::vector<Ciphertext<DCRTPoly>>> results(queries.size());
    for (std::size_t q = 0; q < queries.size(); ++q) {
        for (std::size_t b = 0; b < stored_blocks.size(); ++b) {
            Ciphertext<DCRTPoly> r = stored_blocks[b][0]->Clone();
            r->GetElements().push_back(r->GetElements()[0]);
            write_back(r->GetElements(), meta,
                       out.data() + (q * shape.blocks + b) * 3 * shape.poly_words());
            r->SetLevel(fq0.level);
            r->SetNoiseScaleDeg(fq0.noise_scale_deg * 2);
            r->SetScalingFactor(fq0.scaling_factor * fq0.scaling_factor);
            results[q].push_back(std::move(r));
        }
    }
    return results;
}

}  // namespace ombra::core
