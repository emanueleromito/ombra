// ombra_plain_search: oracle driver for the GPU CKKS encoder and the ciphertext x plaintext search
// (T0.11 step 2).
//
// Usage: ombra_plain_search <context.bin> <dir> <dims> <format: f64|fp16|int8> <code_scale>
//                           <coef_tolerance>
//
// Inputs in <dir> (public material and synthetic data only; no secret key is read or needed):
//   query_<j>.bin   OpenFHE BINARY ciphertexts, query value j in every slot, j < dims (level 0)
//   values.bin      float64 [dims][slots]: the exact doubles of the stored columns (for fp16/int8
//                   the DECODED values, so the OpenFHE oracle sees what the GPU decodes)
//   codes.bin       the raw stored codes, [dims][slots], in <format> (f64: same as values.bin)
// Checks (each prints PASS/FAIL, never values; exit 0 only if all pass):
//   1. Encoding. For every column, the GPU encoder's coefficient-form polynomial is compared with
//      OpenFHE's MakeCKKSPackedPlaintext (same doubles) tower by tower: the centered difference of
//      every coefficient must be at most <coef_tolerance> units (1 unit = 1 / scale of the value).
//      The evaluation-form words are compared too; if every coefficient is identical they must be
//      identical word for word (the NTT is exact).
//   2. Search with OpenFHE-encoded plaintexts: the GPU result equals OpenFHE
//      sum_j EvalMult(query_j, plaintext_j) (EvalAdd accumulation) EXACTLY (every RNS word and the
//      metadata).
//   3. Search with GPU-encoded plaintexts (compact path), and with a separate encode-all then
//      search-encoded pass: both must be identical to each other.
// Writes gpu_coef.bin / cpu_coef.bin (int64 [dims][N], coefficients centered with the first
// modulus, for the decode check in the Python test) and result_openfhe_pt.bin /
// result_gpu_pt.bin / result_cpu.bin (ciphertexts for the decrypt check).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/plain_search.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using u64 = std::uint64_t;
using Ct = Ciphertext<DCRTPoly>;

namespace {

std::vector<char> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

void write_file(const std::string& path, const void* data, std::size_t bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  if (!f) throw std::runtime_error("cannot write " + path);
}

Ct load_ct(const std::string& path) {
  Ct ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot read ciphertext " + path);
  }
  return ct;
}

void save_ct(const std::string& path, const Ct& ct) {
  if (!Serial::SerializeToFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot write ciphertext " + path);
  }
}

std::int64_t center(u64 a, u64 q) {
  return a > q / 2 ? static_cast<std::int64_t>(a) - static_cast<std::int64_t>(q)
                   : static_cast<std::int64_t>(a);
}

// [tower][coeff] words of one polynomial in the given format.
std::vector<u64> poly_words(const DCRTPoly& poly, Format fmt) {
  DCRTPoly p = poly;
  p.SetFormat(fmt);
  std::vector<u64> w;
  for (const auto& tower : p.GetAllElements()) {
    const auto& v = tower.GetValues();
    for (std::size_t i = 0; i < p.GetRingDimension(); ++i) {
      w.push_back(v[i].ConvertToInt<u64>());
    }
  }
  return w;
}

bool identical(const Ct& x, const Ct& y) {
  if (x->GetLevel() != y->GetLevel() || x->GetNoiseScaleDeg() != y->GetNoiseScaleDeg() ||
      x->GetScalingFactor() != y->GetScalingFactor() ||
      x->GetElements().size() != y->GetElements().size()) {
    return false;
  }
  for (std::size_t c = 0; c < x->GetElements().size(); ++c) {
    if (poly_words(x->GetElements()[c], Format::EVALUATION) !=
        poly_words(y->GetElements()[c], Format::EVALUATION)) {
      return false;
    }
  }
  return true;
}

ombra::core::PlainFormat parse_format(const std::string& s) {
  if (s == "f64") return ombra::core::PlainFormat::kDouble;
  if (s == "fp16") return ombra::core::PlainFormat::kFp16;
  if (s == "int8") return ombra::core::PlainFormat::kInt8;
  throw std::invalid_argument("format must be f64, fp16 or int8");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: ombra_plain_search <context> <dir> <dims> <f64|fp16|int8> "
                 "<code_scale> <coef_tolerance>\n");
    return 2;
  }
  try {
    const std::string dir = argv[2];
    const std::size_t dims = std::stoul(argv[3]);
    const auto fmt = parse_format(argv[4]);
    const double code_scale = std::stod(argv[5]);
    const std::int64_t tol = std::stoll(argv[6]);
    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();

    std::vector<Ct> query;
    for (std::size_t j = 0; j < dims; ++j) {
      query.push_back(load_ct(dir + "/query_" + std::to_string(j) + ".bin"));
    }
    const auto vbytes = read_file(dir + "/values.bin");
    const auto cbytes = read_file(dir + "/codes.bin");
    if (vbytes.size() != dims * slots * sizeof(double) ||
        cbytes.size() != dims * slots * ombra::core::plain_format_bytes(fmt)) {
      throw std::runtime_error("values.bin or codes.bin has the wrong size");
    }
    const double* values = reinterpret_cast<const double*>(vbytes.data());

    const auto qflat = ombra::core::flatten_ciphertexts(query);
    const auto tables =
        ombra::core::encode_tables_from(ombra::core::extract_expansion_tables(cc, query[0]));
    const double scaling_factor = query[0]->GetScalingFactor();
    const std::size_t n = tables.ring_dim, nt = tables.num_towers, poly = nt * n;
    if (slots != n / 2) throw std::runtime_error("only full packing is supported");

    // OpenFHE's plaintexts.
    std::vector<Plaintext> pts;
    std::vector<u64> cpu_eval, cpu_coef;  // [dim][tower][coeff]
    for (std::size_t j = 0; j < dims; ++j) {
      std::vector<double> col(values + j * slots, values + (j + 1) * slots);
      pts.push_back(cc->MakeCKKSPackedPlaintext(col));
      const DCRTPoly& e = pts.back()->GetElement<DCRTPoly>();
      const auto ew = poly_words(e, Format::EVALUATION);
      const auto cw = poly_words(e, Format::COEFFICIENT);
      cpu_eval.insert(cpu_eval.end(), ew.begin(), ew.end());
      cpu_coef.insert(cpu_coef.end(), cw.begin(), cw.end());
    }

    bool ok = true;
    ombra::core::PlainEncoder enc(tables, scaling_factor);

    // 1. Encoding.
    const auto gpu_coef = enc.encode_host(fmt, cbytes.data(), dims, code_scale, true);
    const auto gpu_eval = enc.encode_host(fmt, cbytes.data(), dims, code_scale, false);
    std::int64_t max_diff = 0;
    std::size_t exact_coeffs = 0, total_coeffs = 0;
    for (std::size_t c = 0; c < dims; ++c) {
      for (std::size_t t = 0; t < nt; ++t) {
        for (std::size_t i = 0; i < n; ++i) {
          const std::size_t k = c * poly + t * n + i;
          const u64 q = tables.moduli[t];
          const u64 diff = (gpu_coef[k] + q - cpu_coef[k]) % q;
          const std::int64_t dd = center(diff, q);
          max_diff = std::max(max_diff, dd < 0 ? -dd : dd);
          exact_coeffs += diff == 0;
          ++total_coeffs;
        }
      }
    }
    std::size_t exact_eval = 0;
    for (std::size_t k = 0; k < gpu_eval.size(); ++k) exact_eval += gpu_eval[k] == cpu_eval[k];
    const bool enc_ok = max_diff <= tol && (max_diff != 0 || exact_eval == gpu_eval.size());
    std::printf(
        "encoding: max coefficient difference %lld units (tolerance %lld); %zu of %zu "
        "coefficient words and %zu of %zu evaluation words identical: %s\n",
        static_cast<long long>(max_diff), static_cast<long long>(tol), exact_coeffs, total_coeffs,
        exact_eval, gpu_eval.size(), enc_ok ? "PASS" : "FAIL");
    ok = ok && enc_ok;
    std::vector<std::int64_t> gc(dims * n), cc_(dims * n);
    for (std::size_t c = 0; c < dims; ++c) {
      for (std::size_t i = 0; i < n; ++i) {  // tower 0 holds the largest modulus
        gc[c * n + i] = center(gpu_coef[c * poly + i], tables.moduli[0]);
        cc_[c * n + i] = center(cpu_coef[c * poly + i], tables.moduli[0]);
      }
    }
    write_file(dir + "/gpu_coef.bin", gc.data(), gc.size() * sizeof(std::int64_t));
    write_file(dir + "/cpu_coef.bin", cc_.data(), cc_.size() * sizeof(std::int64_t));

    // OpenFHE oracle for the search.
    Ct cpu = cc->EvalMult(query[0], pts[0]);
    for (std::size_t j = 1; j < dims; ++j) cpu = cc->EvalAdd(cpu, cc->EvalMult(query[j], pts[j]));
    save_ct(dir + "/result_cpu.bin", cpu);

    ombra::core::PlainSearchShape shape{dims, 1, nt, n};
    ombra::core::PlainSearchDevice dev(shape, tables, scaling_factor, fmt, code_scale, 1);
    dev.upload_query(qflat.words.data(), qflat.words.size());
    std::vector<u64> words;

    // 2. Search with OpenFHE-encoded plaintexts: exact.
    dev.upload_encoded(0, cpu_eval.data(), cpu_eval.size());
    dev.run_encoded();
    dev.download_result(words);
    const Ct gpu_ofhe_pt = ombra::core::unflatten_ciphertext(cpu, words);
    const bool s1 = identical(gpu_ofhe_pt, cpu);
    std::printf("search with OpenFHE-encoded plaintexts, exact RNS match with EvalMult: %s\n",
                s1 ? "PASS" : "FAIL");
    ok = ok && s1;
    save_ct(dir + "/result_openfhe_pt.bin", gpu_ofhe_pt);

    // 3. Search with GPU-encoded plaintexts: compact path vs encode-all then search.
    dev.upload_codes(0, cbytes.data());
    dev.run_compact(nullptr, nullptr);
    dev.download_result(words);
    const Ct gpu_compact = ombra::core::unflatten_ciphertext(cpu, words);
    dev.encode_all();
    dev.run_encoded();
    dev.download_result(words);
    const Ct gpu_split = ombra::core::unflatten_ciphertext(cpu, words);
    const bool s2 = identical(gpu_compact, gpu_split);
    std::printf("search with GPU-encoded plaintexts, fused and two-pass identical: %s\n",
                s2 ? "PASS" : "FAIL");
    ok = ok && s2;
    const bool same_as_cpu = identical(gpu_compact, cpu);
    std::printf("info: GPU-encoded search identical to OpenFHE-encoded search: %s\n",
                same_as_cpu ? "yes" : "no");
    save_ct(dir + "/result_gpu_pt.bin", gpu_compact);

    std::printf("plain search oracle: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
