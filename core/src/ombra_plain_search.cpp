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
//   1. Encoding, in three checks (decision 0015): (a) FFT and rounding: the centered difference of
//      every GPU coefficient from OpenFHE's MakeCKKSPackedPlaintext is at most <coef_tolerance>
//      units (1 unit = 1 / scale of the value); (b) tower reduction and NTT of OpenFHE's own
//      integer coefficients equal OpenFHE's words EXACTLY; (c) the GPU evaluation form equals
//      OpenFHE's NTT of the GPU's own coefficients EXACTLY, always; (d) the GPU's coefficient
//      words are one integer reduced into every tower, EXACTLY. An optional last argument
//      (none|reduction|ntt|tower|meta_level|meta_degree|meta_scale) injects a fault, for tests of the oracle only.
//   2. Search with OpenFHE-encoded plaintexts: the GPU result equals OpenFHE
//      sum_j EvalMult(query_j, plaintext_j) (EvalAdd accumulation) EXACTLY (every RNS word and the
//      metadata, derived by gpu_eval_plain_search from the query and plaintexts, not copied
//      from the OpenFHE result).
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
#include "ombra/core/oracle_compare.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/plain_search.hpp"
#include "ombra/core/security_floor_context.hpp"
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
  return ombra::core::identical_ciphertexts(x, y);  // shared oracle comparator
}

ombra::core::PlainFormat parse_format(const std::string& s) {
  if (s == "f64") return ombra::core::PlainFormat::kDouble;
  if (s == "fp16") return ombra::core::PlainFormat::kFp16;
  if (s == "int8") return ombra::core::PlainFormat::kInt8;
  throw std::invalid_argument("format must be f64, fp16 or int8");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7 && argc != 8) {
    std::fprintf(stderr,
                 "usage: ombra_plain_search <context> <dir> <dims> <f64|fp16|int8> "
                 "<code_scale> <coef_tolerance> [none|reduction|ntt|tower]\n");
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
    ombra::core::require_security_floor(cc);
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

    // 1. Encoding, in separate checks (T1.9, decision 0015). The FFT and rounding are floating
    //    point and get a tolerance; every integer step after them is compared exactly.
    auto gpu_coef = enc.encode_host(fmt, cbytes.data(), dims, code_scale, true);
    auto gpu_eval = enc.encode_host(fmt, cbytes.data(), dims, code_scale, false);

    // Test-only fault injection (argv[7]): a one-unit coefficient difference that is consistent in
    // every tower (so the GPU evaluation form is a correct NTT of it), plus one wrong word in the
    // reduction ("reduction") or in the NTT ("ntt"). It lets the tests show that the checks below
    // fail on such a defect; it is not a product feature and changes no kernel.
    const std::string inject = argc == 8 ? argv[7] : "none";
    if (inject != "none" && inject != "reduction" && inject != "ntt" && inject != "tower" &&
        inject.rfind("meta_", 0) != 0) {
      throw std::invalid_argument(
          "inject must be none, reduction, ntt, tower, meta_level, meta_degree or meta_scale");
    }
    const bool meta_fault = inject.rfind("meta_", 0) == 0;
    const auto bump = [](u64& w, u64 q) { w = (w + 1) % q; };
    if (inject != "none" && !meta_fault) {
      for (std::size_t t = 0; t < nt; ++t) bump(gpu_coef[t * n], tables.moduli[t]);
      std::vector<std::int64_t> ints(dims * n);
      for (std::size_t c = 0; c < dims; ++c) {
        for (std::size_t i = 0; i < n; ++i) {
          ints[c * n + i] = center(gpu_coef[c * poly + i], tables.moduli[0]);
        }
      }
      gpu_eval = enc.reduce_ntt_host(ints.data(), dims, true);
    }
    // One tower of one coefficient off by one unit: the production path's reduction went wrong.
    if (inject == "tower") bump(gpu_coef[n + 3], tables.moduli[1]);

    // 1a. FFT and rounding: centered coefficient difference within the tolerance.
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
    const bool fft_ok = max_diff <= tol;
    std::printf(
        "encoding FFT and rounding: max coefficient difference %lld units (tolerance %lld); "
        "%zu of %zu coefficient words identical: %s\n",
        static_cast<long long>(max_diff), static_cast<long long>(tol), exact_coeffs, total_coeffs,
        fft_ok ? "PASS" : "FAIL");
    ok = ok && fft_ok;

    // 1b. Tower reduction and NTT of OpenFHE's own integer coefficients: exact. The integers are
    //     OpenFHE's coefficients centered with the largest modulus; the exact comparison also
    //     proves they reduce to OpenFHE's words in every other tower.
    std::vector<std::int64_t> ofhe_ints(dims * n);
    for (std::size_t c = 0; c < dims; ++c) {
      for (std::size_t i = 0; i < n; ++i) {
        ofhe_ints[c * n + i] = center(cpu_coef[c * poly + i], tables.moduli[0]);
      }
    }
    auto red = enc.reduce_ntt_host(ofhe_ints.data(), dims, false);
    auto ntt = enc.reduce_ntt_host(ofhe_ints.data(), dims, true);
    if (inject == "reduction") bump(red[5], tables.moduli[0]);
    if (inject == "ntt") bump(ntt[7], tables.moduli[0]);
    const bool red_ok = red == cpu_coef;
    const bool ntt_ok = ntt == cpu_eval;
    std::printf("encoding tower reduction of OpenFHE's integer coefficients, exact words: %s\n",
                red_ok ? "PASS" : "FAIL");
    std::printf("encoding NTT of OpenFHE's integer coefficients, exact words: %s\n",
                ntt_ok ? "PASS" : "FAIL");
    ok = ok && red_ok && ntt_ok;

    // 1c. The GPU evaluation form against OpenFHE's NTT of the GPU's own coefficients: exact, and
    //     never skipped, whatever the coefficient differences are.
    std::vector<u64> own_eval;
    for (std::size_t c = 0; c < dims; ++c) {
      DCRTPoly own = pts[0]->GetElement<DCRTPoly>();
      own.SetFormat(Format::COEFFICIENT);
      for (std::size_t t = 0; t < nt; ++t) {
        NativePoly tp = own.GetElementAtIndex(t);
        for (std::size_t i = 0; i < n; ++i) {
          tp[i] = NativeInteger(gpu_coef[c * poly + t * n + i]);
        }
        own.SetElementAtIndex(t, tp);
      }
      const auto w = poly_words(own, Format::EVALUATION);
      own_eval.insert(own_eval.end(), w.begin(), w.end());
    }
    if (inject == "ntt") bump(gpu_eval[9], tables.moduli[0]);
    const bool eval_ok = gpu_eval == own_eval;
    std::printf("encoding GPU evaluation form vs OpenFHE NTT of the GPU coefficients, exact: %s\n",
                eval_ok ? "PASS" : "FAIL");
    ok = ok && eval_ok;

    // 1d. The production path's coefficient words, in every tower, are one integer. Without this a
    //     one-unit error in a single tower would pass 1a, and 1c would take it as its input.
    std::vector<std::int64_t> own_ints(dims * n);
    for (std::size_t c = 0; c < dims; ++c) {
      for (std::size_t i = 0; i < n; ++i) {
        own_ints[c * n + i] = center(gpu_coef[c * poly + i], tables.moduli[0]);
      }
    }
    const bool towers_ok = enc.reduce_ntt_host(own_ints.data(), dims, false) == gpu_coef;
    std::printf("encoding GPU coefficient words are one integer in every tower, exact: %s\n",
                towers_ok ? "PASS" : "FAIL");
    ok = ok && towers_ok;

    std::size_t exact_eval = 0;
    for (std::size_t k = 0; k < gpu_eval.size(); ++k) exact_eval += gpu_eval[k] == cpu_eval[k];
    std::printf("info: %zu of %zu evaluation words identical to OpenFHE's plaintext\n", exact_eval,
                gpu_eval.size());
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
    // The result's metadata comes from the wrapper (query and plaintexts), never from `cpu`.
    // Test-only metadata fault: corrupt what the wrapper derived before the comparison.
    const auto derive = [&](const std::vector<u64>& w) {
      Ct r = ombra::core::gpu_eval_plain_search(query, pts, w);
      if (inject == "meta_level") r->SetLevel(r->GetLevel() + 1);
      if (inject == "meta_degree") r->SetNoiseScaleDeg(r->GetNoiseScaleDeg() + 1);
      if (inject == "meta_scale") r->SetScalingFactor(r->GetScalingFactor() * 2);
      return r;
    };
    const Ct gpu_ofhe_pt = derive(words);
    const bool s1 = identical(gpu_ofhe_pt, cpu);
    std::printf("search with OpenFHE-encoded plaintexts, exact RNS match with EvalMult: %s\n",
                s1 ? "PASS" : "FAIL");
    ok = ok && s1;
    save_ct(dir + "/result_openfhe_pt.bin", gpu_ofhe_pt);

    // 3. Search with GPU-encoded plaintexts: compact path vs encode-all then search.
    dev.upload_codes(0, cbytes.data());
    dev.run_compact(nullptr, nullptr);
    dev.download_result(words);
    const Ct gpu_compact = derive(words);
    dev.encode_all();
    dev.run_encoded();
    dev.download_result(words);
    const Ct gpu_split = derive(words);
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
