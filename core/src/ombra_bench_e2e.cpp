// ombra_bench_e2e: server side of the end-to-end query latency measurement (T1.3).
//
// Usage: ombra_bench_e2e <context.bin> <autokeys.bin> <evalmultkey.bin> <dir> <d>
//
// A long-lived process driven line by line over stdin/stdout by core/bench/e2e.py (which is the
// client: it holds the secret key, encrypts, decrypts, picks the clusters and ranks). Everything
// read here is PUBLIC material (context, automorphism keys, EvalMult key, ciphertexts); no secret
// key can be read or is needed. Files in <dir>: centers_<j>.bin (encrypted cluster-center block,
// dimension j < d), stored_<j>.bin (one encrypted stored block, dimension j), mask_ones.bin (an
// all-ones mask at level 1), and per case a partial mask named in the CASE command.
//
// Commands (one per line) and replies (one line each, key=value pairs, sizes and seconds only):
//   WARM <query.bin>             expand once so the expansion's device pool is allocated before the
//                                memory plan is made -> "OK"
//   CASE <blocks> <last_mask>    build stage 2 for `blocks` stored blocks of d columns: block 0 is
//                                the real uploaded block, the others are device-to-device copies of
//                                it (synthetic stand-in for the chosen clusters; kernel timing does
//                                not depend on the values) -> "CASE ..." (memory plan, H2D time)
//   QUERY <query.bin>            expansion -> device-to-device handoff -> stage 1 (centers), writes
//                                s1_result.bin -> "S1 ..."; then waits for the next command
//   GO                           stage 2 over the resident blocks, writes s2_result_<b>.bin for
//                                each distinct resident block -> "S2 ..."
//   QUIT
// Server timings (steady clock with the device synchronized, or CUDA events inside the kernels'
// wrappers) cover: query deserialization, expansion, the leaf handoff, the GPU stages, and the
// device-to-host copy of the result words. Writing result files (OpenFHE serialization) is not
// timed: it stands in for the network encoding. Prints sizes and timings only, never data values.
// Exit code: 0 ok, 2 error.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/gpu_search.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using ombra::core::CtMeta;
using ombra::core::GpuSearch;
using ombra::core::SearchStage;
using Clock = std::chrono::steady_clock;
using Ct = Ciphertext<DCRTPoly>;
using u64 = std::uint64_t;

namespace {

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

Ct load_ct(const std::string& path) {
  Ct ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot read ciphertext " + path);
  }
  return ct;
}

std::vector<Ct> load_columns(const std::string& dir, const std::string& prefix, std::size_t d) {
  std::vector<Ct> cols;
  cols.reserve(d);
  for (std::size_t j = 0; j < d; ++j) {
    cols.push_back(load_ct(dir + "/" + prefix + "_" + std::to_string(j) + ".bin"));
  }
  return cols;
}

void save_ct(const std::string& path, const Ct& ct) {
  if (!Serial::SerializeToFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot write " + path);
  }
}

constexpr std::size_t kHeadroomBytes = std::size_t{1536} << 20;  // CUDA context, fragmentation

void reply(const std::string& line) {
  std::printf("%s\n", line.c_str());
  std::fflush(stdout);
}

std::string kv(const char* key, double v) {
  char buf[96];
  std::snprintf(buf, sizeof buf, " %s=%.9g", key, v);
  return buf;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr, "usage: ombra_bench_e2e <context> <autokeys> <evalmultkey> <dir> <d>\n");
    return 2;
  }
  try {
    const std::string dir = argv[4];
    const std::size_t d = std::stoul(argv[5]);

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);
    {
      std::ifstream f(argv[2], std::ios::in | std::ios::binary);
      if (!f.is_open() ||
          !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(f, SerType::BINARY)) {
        throw std::runtime_error("cannot read automorphism keys");
      }
    }
    {
      std::ifstream f(argv[3], std::ios::in | std::ios::binary);
      if (!f.is_open() ||
          !CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(f, SerType::BINARY)) {
        throw std::runtime_error("cannot read the EvalMult key");
      }
    }
    const std::size_t ring = cc->GetRingDimension();

    // Stored block 0 (kept flattened on the host for the per-case upload) and the centers block.
    std::vector<Ct> stored_cols = load_columns(dir, "stored", d);
    const Ct stored_sample = stored_cols[0];
    const auto key_vec =
        CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(stored_sample->GetKeyTag());
    if (key_vec.empty()) throw std::runtime_error("no EvalMult key for this key tag");
    const Ct mask_ones = load_ct(dir + "/mask_ones.bin");
    const auto& akeys =
        CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(stored_sample->GetKeyTag());

    ombra::core::FlatCiphertexts stored_flat = ombra::core::flatten_ciphertexts(stored_cols);
    stored_cols.clear();
    const std::size_t block_bytes = stored_flat.words.size() * sizeof(u64);

    // Stage 1: one block of centers, resident for the life of the process.
    const auto mem0 = ombra::core::cuda_memory_info();
    GpuSearch stage1(cc, stored_sample, mask_ones, key_vec[0], d, 1);
    {
      const auto cols = load_columns(dir, "centers", d);
      stage1.upload_block(0, cols);
    }
    stage1.upload_mask(0, mask_ones);

    // Expansion: needs a query-shaped ciphertext for the metadata template; the first WARM/QUERY
    // file provides it.
    std::unique_ptr<ombra::core::GpuExpansion> gx;
    std::unique_ptr<GpuSearch> stage2;
    std::size_t case_blocks = 0, resident = 0, passes = 1;
    bool leaves_checked = false;

    const auto expand_and_hand_off = [&](const std::string& path, double& deser_s, double& exp_s,
                                         double& hand_s) {
      auto t0 = Clock::now();
      const Ct q = load_ct(path);
      deser_s = seconds_since(t0);
      if (!gx) gx = std::make_unique<ombra::core::GpuExpansion>(cc, q, akeys, d);

      t0 = Clock::now();
      const ombra::core::GpuCt root = gx->upload(q);
      std::vector<ombra::core::GpuCt> lv = ombra::core::expand_tree(root, d, ring, *gx);
      gx->synchronize();
      exp_s = seconds_since(t0);

      if (!leaves_checked) {  // the leaves must carry the stored columns' metadata and moduli
        const auto leaf = ombra::core::flatten_one(gx->download(lv[0]));
        const auto col = ombra::core::flatten_one(stored_sample);
        if (leaf.level != col.level || leaf.noise_scale_deg != col.noise_scale_deg ||
            leaf.scaling_factor != col.scaling_factor || leaf.moduli != col.moduli) {
          throw std::runtime_error("expanded leaves and stored columns differ in metadata");
        }
        leaves_checked = true;
      }
      const auto lf = ombra::core::flatten_one(stored_sample);
      const CtMeta meta{lf.level, lf.noise_scale_deg, lf.scaling_factor};
      std::vector<const u64*> ptrs;
      ptrs.reserve(d);
      for (const auto& l : lv) ptrs.push_back(gx->device().device_ptr(l.dev));

      t0 = Clock::now();
      stage1.set_query_from_device(ptrs, meta);
      if (stage2) stage2->set_query_from_device(ptrs, meta);
      gx->synchronize();
      hand_s = seconds_since(t0);
    };

    reply("READY" + kv("d", static_cast<double>(d)) + kv("block_bytes", block_bytes) +
          kv("stage1_device_bytes", static_cast<double>(stage1.device_bytes())));

    std::string line;
    while (std::getline(std::cin, line)) {
      std::istringstream in(line);
      std::string cmd;
      in >> cmd;
      if (cmd == "QUIT") break;

      if (cmd == "WARM") {
        std::string path;
        in >> path;
        double a, b, c;
        expand_and_hand_off(path, a, b, c);
        reply("OK");
      } else if (cmd == "CASE") {
        std::size_t blocks;
        std::string last_mask_path;
        in >> blocks >> last_mask_path;
        if (blocks == 0) throw std::invalid_argument("blocks must be positive");
        stage2.reset();  // free the previous case before planning
        const Ct mask_last = load_ct(last_mask_path);

        // Memory plan: what a real query needs resident is the centers (stage1, already there),
        // the chosen blocks, a query buffer, the mask and the working buffers. The expansion pool
        // is already allocated (WARM). Largest block count that fits in the free memory.
        const auto mem = ombra::core::cuda_memory_info();
        const std::size_t per_block_extra = 4u
                                            << 20;  // stage buffers and masks per block (< 3.5 MB)
        const std::size_t query_bytes = d * (block_bytes / d);
        std::size_t avail = mem.free_bytes > kHeadroomBytes + query_bytes
                                ? mem.free_bytes - kHeadroomBytes - query_bytes
                                : 0;
        std::size_t fit = avail / (block_bytes + per_block_extra);
        if (fit == 0) throw std::runtime_error("not even one chosen block fits on the device");
        resident = std::min(blocks, fit);
        passes = (blocks + resident - 1) / resident;
        case_blocks = blocks;

        stage2 = std::make_unique<GpuSearch>(cc, stored_sample, mask_ones, key_vec[0], d, resident);
        stage2->upload_block_flat(0, stored_flat);
        const double h2d_wall = stage2->last_upload_seconds();
        auto t0 = Clock::now();
        for (std::size_t b = 1; b < resident; ++b) stage2->copy_block(0, b);
        const double d2d_s = seconds_since(t0);
        for (std::size_t b = 0; b < resident; ++b) {
          stage2->upload_mask(b, (b + 1 == resident && passes == 1) ? mask_last : mask_ones);
        }
        const auto after = ombra::core::cuda_memory_info();
        reply("CASE" + kv("blocks", static_cast<double>(blocks)) +
              kv("resident_blocks", static_cast<double>(resident)) +
              kv("passes", static_cast<double>(passes)) + kv("h2d_one_block_seconds", h2d_wall) +
              kv("d2d_fill_seconds", d2d_s) +
              kv("stage2_device_bytes", static_cast<double>(stage2->device_bytes())) +
              kv("free_bytes_before", static_cast<double>(mem.free_bytes)) +
              kv("free_bytes_after", static_cast<double>(after.free_bytes)) +
              kv("total_bytes", static_cast<double>(after.total_bytes)) +
              kv("device_used_since_start_bytes",
                 static_cast<double>(mem0.free_bytes - after.free_bytes)));
      } else if (cmd == "QUERY") {
        if (!stage2) throw std::runtime_error("QUERY before CASE");
        std::string path;
        in >> path;
        double deser_s, exp_s, hand_s;
        expand_and_hand_off(path, deser_s, exp_s, hand_s);

        const ombra::core::GpuSearchTiming t = stage1.run();
        auto t0 = Clock::now();
        const std::vector<u64> words = stage1.download_words(SearchStage::kFinal, 0);
        const double d2h_s = seconds_since(t0);
        // Result file: not timed (network encoding stand-in).
        const CtMeta m = stage1.meta(SearchStage::kFinal);
        save_ct(dir + "/s1_result.bin",
                ombra::core::make_ciphertext_like(mask_ones, 2, words.size() / (2 * ring), words,
                                                  m.level, m.noise_scale_deg, m.scaling_factor));
        reply("S1" + kv("deserialize_s", deser_s) + kv("expansion_s", exp_s) +
              kv("handoff_s", hand_s) + kv("search_s", t.search_s) + kv("relin_s", t.relin_s) +
              kv("rescale1_s", t.rescale1_s) + kv("mask_s", t.mask_s) +
              kv("rescale2_s", t.rescale2_s) + kv("gpu_total_s", t.total_s) + kv("d2h_s", d2h_s));
      } else if (cmd == "GO") {
        if (!stage2) throw std::runtime_error("GO before CASE");
        ombra::core::GpuSearchTiming sum;
        double d2h_s = 0;
        for (std::size_t p = 0; p < passes; ++p) {
          const auto t = stage2->run();
          sum.search_s += t.search_s;
          sum.relin_s += t.relin_s;
          sum.rescale1_s += t.rescale1_s;
          sum.mask_s += t.mask_s;
          sum.rescale2_s += t.rescale2_s;
          sum.total_s += t.total_s;
          if (p + 1 < passes) {  // results of an earlier pass would be downloaded here
            const auto t0 = Clock::now();
            for (std::size_t b = 0; b < resident; ++b) {
              (void)stage2->download_words(SearchStage::kFinal, b);
            }
            d2h_s += seconds_since(t0);
          }
        }
        std::vector<std::vector<u64>> words(resident);
        auto t0 = Clock::now();
        for (std::size_t b = 0; b < resident; ++b) {
          words[b] = stage2->download_words(SearchStage::kFinal, b);
        }
        d2h_s += seconds_since(t0);
        const CtMeta m = stage2->meta(SearchStage::kFinal);
        for (std::size_t b = 0; b < resident; ++b) {  // not timed
          save_ct(dir + "/s2_result_" + std::to_string(b) + ".bin",
                  ombra::core::make_ciphertext_like(mask_ones, 2, words[b].size() / (2 * ring),
                                                    words[b], m.level, m.noise_scale_deg,
                                                    m.scaling_factor));
        }
        reply("S2" + kv("blocks", static_cast<double>(case_blocks)) +
              kv("resident_blocks", static_cast<double>(resident)) +
              kv("passes", static_cast<double>(passes)) + kv("search_s", sum.search_s) +
              kv("relin_s", sum.relin_s) + kv("rescale1_s", sum.rescale1_s) +
              kv("mask_s", sum.mask_s) + kv("rescale2_s", sum.rescale2_s) +
              kv("gpu_total_s", sum.total_s) + kv("d2h_s", d2h_s));
      } else {
        throw std::invalid_argument("unknown command");
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
