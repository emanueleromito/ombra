// ombra_bench_plain_search: search throughput of compact plaintext storage (T0.11 step 2).
//
// Usage: ombra_bench_plain_search <context.bin> <public_key.bin> <dims> <blocks> <warmup> <iters>
//                                 <out.json>
//
// Inputs are OpenFHE BINARY files written by core/bench/run.py: the CryptoContext (a preset from
// core/params) and the PUBLIC key; one sample ciphertext is encrypted with the public key only to
// read the tower layout. No secret key is read. All data is seeded synthetic (random reduced RNS
// words, random fp16 codes in [2^-7, 1) and random int8 codes: the kernels' speed does not depend
// on the values), so nothing real is stored or printed. Four modes are timed on the same shape
// (d dimensions, `blocks` blocks of N/2 vectors), each with CUDA events, after `warmup` runs:
//   ct_x_ct          the T0.4 kernel (stored ciphertexts x query ciphertexts, 3 components)
//   pt_preencoded    the ciphertext x plaintext kernel on RNS plaintexts already encoded
//   fp16, int8       compact codes encoded on the GPU in chunks of blocks right before the multiply
//                    (encode + search; the two parts are reported separately as well)
// Writes one JSON object with sizes and timings only.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/plain_search.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using u64 = std::uint64_t;

namespace {

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.size() % 2 == 1 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

// Random words below the modulus of their tower; `polys` polynomials of [tower][coeff].
std::vector<u64> random_words(std::size_t polys, const std::vector<u64>& moduli, std::size_t n,
                              std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<u64> w;
  w.reserve(polys * moduli.size() * n);
  for (std::size_t p = 0; p < polys; ++p) {
    for (u64 q : moduli) {
      for (std::size_t i = 0; i < n; ++i) w.push_back(rng() % q);
    }
  }
  return w;
}

std::vector<std::uint16_t> random_fp16(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::uint16_t> v(count);
  for (auto& x : v) {
    const std::uint32_t r = static_cast<std::uint32_t>(rng());
    const std::uint32_t exp = 8 + (r >> 10) % 7;  // 2^-7 .. 2^-1
    x = static_cast<std::uint16_t>(((r >> 20) & 1u) << 15 | exp << 10 | (r & 0x3FFu));
  }
  return v;
}

std::vector<std::int8_t> random_int8(std::size_t count, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::vector<std::int8_t> v(count);
  for (auto& x : v) x = static_cast<std::int8_t>(static_cast<int>(rng() % 255) - 127);
  return v;
}

struct Entry {
  std::string mode;
  std::size_t storage_bytes_per_vector = 0;
  double seconds = 0, encode_seconds = -1, search_seconds = -1, one_time_encode_seconds = -1;
  std::size_t device_bytes = 0;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 8) {
    std::fprintf(stderr,
                 "usage: ombra_bench_plain_search <context> <public_key> <dims> <blocks> "
                 "<warmup> <iters> <out.json>\n");
    return 2;
  }
  try {
    const std::size_t dims = std::stoul(argv[3]);
    const std::size_t blocks = std::stoul(argv[4]);
    const int warmup = std::stoi(argv[5]);
    const int iters = std::stoi(argv[6]);
    if (dims == 0 || blocks == 0 || warmup < 0 || iters < 1) {
      throw std::invalid_argument("dims, blocks and iters must be positive");
    }
    CryptoContext<DCRTPoly> cc;
    PublicKey<DCRTPoly> pk;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    if (!Serial::DeserializeFromFile(argv[2], pk, SerType::BINARY)) {
      throw std::runtime_error("cannot read public key");
    }
    const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();
    const Ciphertext<DCRTPoly> sample =
        cc->Encrypt(pk, cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.25)));
    const auto tables =
        ombra::core::encode_tables_from(ombra::core::extract_expansion_tables(cc, sample));
    const double scaling_factor = sample->GetScalingFactor();
    const std::size_t n = tables.ring_dim, nt = tables.num_towers, poly = nt * n;
    if (slots != n / 2) throw std::runtime_error("only full packing is supported");
    const std::size_t vectors = blocks * slots;

    std::vector<Entry> entries;
    const auto timed = [&](auto&& run) {
      for (int i = 0; i < warmup; ++i) run();
      std::vector<double> t;
      for (int i = 0; i < iters; ++i) t.push_back(run());
      return median(t);
    };

    // ct_x_ct: the T0.4 kernel on random ciphertext words.
    {
      ombra::core::ColumnSearchShape shape{dims, blocks, nt, n};
      ombra::core::ColumnSearchDevice dev(shape, tables.moduli);
      {
        const auto block = random_words(dims * 2, tables.moduli, n, 11);
        dev.upload_block(0, block.data(), block.size());
      }
      for (std::size_t b = 1; b < blocks; ++b) dev.copy_block(0, b);
      {
        const auto q = random_words(dims * 2, tables.moduli, n, 12);
        dev.upload_query(q.data(), q.size());
      }
      Entry e;
      e.mode = "ct_x_ct";
      e.storage_bytes_per_vector = dims * 2 * poly * sizeof(u64) / slots;
      e.seconds = timed([&] { return dev.run(); });
      e.device_bytes =
          (shape.stored_words() + shape.query_words() + shape.out_words()) * sizeof(u64);
      entries.push_back(e);
    }

    const auto query_words = random_words(dims * 2, tables.moduli, n, 13);
    const std::size_t scratch_budget = std::size_t{3} << 29;  // 1.5 GiB of encoded plaintexts
    const std::size_t chunk =
        std::max<std::size_t>(1, std::min(blocks, scratch_budget / (dims * poly * 8)));

    // pt_preencoded: plaintexts encoded once (int8 codes), then the ct x pt kernel.
    {
      ombra::core::PlainSearchShape shape{dims, blocks, nt, n};
      ombra::core::PlainSearchDevice dev(shape, tables, scaling_factor,
                                         ombra::core::PlainFormat::kInt8, 100.0, 1);
      const auto codes = random_int8(dims * slots, 21);
      dev.upload_codes(0, codes.data());
      for (std::size_t b = 1; b < blocks; ++b) dev.copy_codes(0, b);
      dev.upload_query(query_words.data(), query_words.size());
      Entry e;
      e.mode = "pt_preencoded";
      e.storage_bytes_per_vector = dims * poly * sizeof(u64) / slots;
      e.one_time_encode_seconds = dev.encode_all();
      e.seconds = timed([&] { return dev.run_encoded(); });
      e.device_bytes =
          (shape.blocks * shape.encoded_block_words() + shape.query_words() + shape.out_words()) *
          sizeof(u64);
      entries.push_back(e);
    }

    // Compact modes.
    for (const auto fmt : {ombra::core::PlainFormat::kFp16, ombra::core::PlainFormat::kInt8}) {
      ombra::core::PlainSearchShape shape{dims, blocks, nt, n};
      ombra::core::PlainSearchDevice dev(shape, tables, scaling_factor, fmt, 100.0, chunk);
      if (fmt == ombra::core::PlainFormat::kFp16) {
        const auto codes = random_fp16(dims * slots, 22);
        dev.upload_codes(0, codes.data());
      } else {
        const auto codes = random_int8(dims * slots, 23);
        dev.upload_codes(0, codes.data());
      }
      for (std::size_t b = 1; b < blocks; ++b) dev.copy_codes(0, b);
      dev.upload_query(query_words.data(), query_words.size());
      Entry e;
      e.mode = fmt == ombra::core::PlainFormat::kFp16 ? "fp16" : "int8";
      e.storage_bytes_per_vector = dims * ombra::core::plain_format_bytes(fmt);
      std::vector<double> total, enc, srch;
      for (int i = 0; i < warmup + iters; ++i) {
        double a = 0, s = 0;
        const double t = dev.run_compact(&a, &s);
        if (i >= warmup) {
          total.push_back(t);
          enc.push_back(a);
          srch.push_back(s);
        }
      }
      e.seconds = median(total);
      e.encode_seconds = median(enc);
      e.search_seconds = median(srch);
      e.device_bytes = (blocks * dims * slots * ombra::core::plain_format_bytes(fmt) +
                        chunk * shape.encoded_block_words() * sizeof(u64) +
                        shape.query_words() * sizeof(u64) + shape.out_words() * sizeof(u64));
      entries.push_back(e);
    }

    std::ostringstream js;
    js.precision(9);
    js << "{\n  \"dims\": " << dims << ",\n  \"blocks\": " << blocks
       << ",\n  \"vectors_per_block\": " << slots << ",\n  \"total_vectors_scored\": " << vectors
       << ",\n  \"num_towers\": " << nt << ",\n  \"ring_dim\": " << n
       << ",\n  \"chunk_blocks\": " << chunk << ",\n  \"warmup_iterations\": " << warmup
       << ",\n  \"timed_iterations\": " << iters << ",\n  \"modes\": [\n";
    for (std::size_t i = 0; i < entries.size(); ++i) {
      const Entry& e = entries[i];
      js << "    {\"mode\": \"" << e.mode
         << "\", \"storage_bytes_per_vector\": " << e.storage_bytes_per_vector
         << ", \"seconds_median\": " << e.seconds
         << ", \"vectors_per_second\": " << static_cast<double>(vectors) / e.seconds
         << ", \"device_memory_bytes\": " << e.device_bytes;
      if (e.encode_seconds >= 0) {
        js << ", \"encode_seconds_median\": " << e.encode_seconds
           << ", \"search_seconds_median\": " << e.search_seconds;
      }
      if (e.one_time_encode_seconds >= 0) {
        js << ", \"one_time_encode_seconds\": " << e.one_time_encode_seconds;
      }
      js << "}" << (i + 1 < entries.size() ? "," : "") << "\n";
    }
    js << "  ],\n  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\"\n}\n";
    std::ofstream out(argv[7]);
    out << js.str();
    if (!out) throw std::runtime_error("cannot write output file");
    std::printf("plain search bench done: dims=%zu blocks=%zu\n", dims, blocks);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
