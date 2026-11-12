// ombra_recall_search: end-to-end GPU search driver for the accuracy measurement.
//
// Usage: ombra_recall_search <context.bin> <autokeys.bin> <data_dir> <out_dir> <d> <blocks>
//                            <queries> <stats.json>
//
// Inputs (OpenFHE BINARY serializations of PUBLIC material only, written by core/bench/recall.py):
// the CryptoContext, the automorphism keys, and in <data_dir> the encrypted column-packed blocks
// stored_<b>_<j>.bin (dimension j of block b, b < blocks, j < d) and the encoded queries
// query_<i>.bin (symmetric coefficient packing, i < queries). No secret key is read or needed.
//
// All blocks are uploaded to the GPU once. Per query: GPU expansion into d leaves (GpuExpansion),
// then the column-packed search kernel over every block, then one 3-component
// degree-2 result ciphertext per block is written to <out_dir>/scores_<i>_<b>.bin. Relinearize,
// rescale, masking and decryption happen in the Python client. The kernels are the ones whose
// outputs are compared exactly with OpenFHE in ombra_expand_gpu and ombra_column_search; this
// driver adds no kernel. Writes <stats.json> with sizes and timings only; prints aggregates only,
// never ciphertext or plaintext values. Exit code: 0 ok, 2 error.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
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

// A 3-component result ciphertext from the kernel's words [component][tower][coeff], with the
// metadata EvalMultNoRelin gives (level unchanged, degree and scale squared; see
// gpu_eval_column_search). `like` is a stored ciphertext (context, key tag, shape).
Ct make_result(const Ct& like, const u64* words, std::size_t poly_words) {
  std::vector<u64> first(words, words + 2 * poly_words);
  std::vector<u64> second(2 * poly_words, 0);
  std::copy(words + 2 * poly_words, words + 3 * poly_words, second.begin());
  Ct r = ombra::core::unflatten_ciphertext(like, first);
  Ct extra = ombra::core::unflatten_ciphertext(like, second);
  r->GetElements().push_back(extra->GetElements()[0]);
  r->SetNoiseScaleDeg(like->GetNoiseScaleDeg() * 2);
  r->SetScalingFactor(like->GetScalingFactor() * like->GetScalingFactor());
  return r;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::fprintf(
        stderr,
        "usage: ombra_recall_search <context> <autokeys> <data_dir> <out_dir> <d> <blocks> "
        "<queries> <stats.json>\n");
    return 2;
  }
  try {
    const std::string data_dir = argv[3], out_dir = argv[4];
    const std::size_t d = std::stoul(argv[5]);
    const std::size_t blocks = std::stoul(argv[6]);
    const std::size_t queries = std::stoul(argv[7]);
    if (d == 0 || blocks == 0 || queries == 0) {
      throw std::invalid_argument("d, blocks and queries must be positive");
    }

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    ombra::core::require_security_floor(cc);
    std::ifstream keyfile(argv[2], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read automorphism keys");
    }
    const std::size_t ring = cc->GetRingDimension();

    const Ct first_query = load_ct(data_dir + "/query_0.bin");
    const auto& keys =
        CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(first_query->GetKeyTag());
    ombra::core::GpuExpansion gx(cc, first_query, keys, d);

    // Stored blocks: loaded one at a time (host memory stays at one block), kept on the device.
    std::unique_ptr<ombra::core::ColumnSearchDevice> dev;
    ombra::core::ColumnSearchShape shape{};
    std::vector<u64> moduli;
    Ct like;
    auto t0 = Clock::now();
    for (std::size_t b = 0; b < blocks; ++b) {
      std::vector<Ct> cols;
      cols.reserve(d);
      for (std::size_t j = 0; j < d; ++j) {
        cols.push_back(
            load_ct(data_dir + "/stored_" + std::to_string(b) + "_" + std::to_string(j) + ".bin"));
      }
      const auto flat = ombra::core::flatten_ciphertexts(cols);
      if (b == 0) {
        shape = {d, blocks, flat.num_towers, flat.ring_dim};
        moduli = flat.moduli;
        like = cols[0];
        dev = std::make_unique<ombra::core::ColumnSearchDevice>(shape, moduli);
      } else if (flat.moduli != moduli || flat.num_towers != shape.num_towers) {
        throw std::runtime_error("stored blocks differ in shape or moduli");
      }
      dev->upload_block(b, flat.words.data(), flat.words.size());
    }
    const double load_seconds = seconds_since(t0);
    if (shape.ring_dim != ring) throw std::runtime_error("ring dimension mismatch");
    std::printf("loaded %zu blocks of %zu columns in %.1f s\n", blocks, d, load_seconds);

    double expand_total = 0, search_total = 0, convert_total = 0, write_total = 0;
    double expand_max = 0;
    std::vector<u64> query_words, result;
    query_words.reserve(shape.query_words());
    for (std::size_t i = 0; i < queries; ++i) {
      const Ct q = load_ct(data_dir + "/query_" + std::to_string(i) + ".bin");

      t0 = Clock::now();
      // upload() refuses a query whose key, level, degree, scale or shape differ from the first
      // query's; the stored columns must be under the same key as well.
      ombra::core::require_same_key(q, like, "recall search");
      const ombra::core::GpuCt root = gx.upload(q);
      std::vector<ombra::core::GpuCt> leaves = ombra::core::expand_tree(root, d, ring, gx);
      gx.synchronize();
      const double expand_s = seconds_since(t0);
      expand_total += expand_s;
      if (expand_s > expand_max) expand_max = expand_s;

      t0 = Clock::now();
      query_words.clear();
      for (std::size_t j = 0; j < d; ++j) {
        const auto w = gx.device().download(leaves[j].dev);
        if (w.size() != 2 * shape.poly_words()) throw std::runtime_error("leaf size mismatch");
        query_words.insert(query_words.end(), w.begin(), w.end());
      }
      if (i == 0) {  // the leaves must carry the stored columns' level, degree, scale and moduli
        const auto leaf = ombra::core::flatten_ciphertexts({gx.download(leaves[0])});
        const auto col = ombra::core::flatten_ciphertexts({like});
        if (leaf.level != col.level || leaf.noise_scale_deg != col.noise_scale_deg ||
            leaf.scaling_factor != col.scaling_factor || leaf.moduli != moduli) {
          throw std::runtime_error("expanded leaves and stored columns differ in metadata");
        }
      }
      leaves.clear();
      dev->upload_query(query_words.data(), query_words.size());
      convert_total += seconds_since(t0);

      search_total += dev->run();
      dev->download_result(result);

      t0 = Clock::now();
      for (std::size_t b = 0; b < blocks; ++b) {
        const Ct r =
            make_result(like, result.data() + b * 3 * shape.poly_words(), shape.poly_words());
        const std::string path =
            out_dir + "/scores_" + std::to_string(i) + "_" + std::to_string(b) + ".bin";
        // Written under a temporary name and renamed, so the client can decrypt a query's results
        // while the driver works on later queries (it waits for the file of the last block).
        const std::string tmp = path + ".tmp";
        if (!Serial::SerializeToFile(tmp, r, SerType::BINARY)) {
          throw std::runtime_error("cannot write " + path);
        }
        std::filesystem::rename(tmp, path);
      }
      write_total += seconds_since(t0);
    }

    const double n = static_cast<double>(queries);
    std::ofstream out(argv[8]);
    out.precision(9);
    out << "{\n"
        << "  \"dims\": " << d << ",\n"
        << "  \"blocks\": " << blocks << ",\n"
        << "  \"queries\": " << queries << ",\n"
        << "  \"num_towers\": " << shape.num_towers << ",\n"
        << "  \"ring_dim\": " << shape.ring_dim << ",\n"
        << "  \"load_blocks_seconds\": " << load_seconds << ",\n"
        << "  \"expansion_setup_seconds\": " << gx.setup_seconds() << ",\n"
        << "  \"expansion_seconds_mean\": " << expand_total / n << ",\n"
        << "  \"expansion_seconds_max\": " << expand_max << ",\n"
        << "  \"leaf_download_and_query_upload_seconds_mean\": " << convert_total / n << ",\n"
        << "  \"search_kernel_seconds_mean\": " << search_total / n << ",\n"
        << "  \"result_write_seconds_mean\": " << write_total / n << ",\n"
        << "  \"stored_device_bytes\": " << shape.stored_words() * sizeof(u64) << ",\n"
        << "  \"gpu_name\": \"" << ombra::core::cuda_device_name() << "\"\n"
        << "}\n";
    if (!out) throw std::runtime_error("cannot write stats file");
    std::printf("recall search done: queries=%zu expansion_mean=%.3f s search_mean=%.4f s\n",
                queries, expand_total / n, search_total / n);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
