// ombra_search_gpu: oracle driver for GPU relinearize, rescale and ciphertext mask multiply, and
// for the GPU-only search pipeline (T1.2).
//
// Usage: ombra_search_gpu <context.bin> <evalmultkey.bin> <in_dir> <out_dir> <dims> <mask> [mask..]
//
// Inputs (OpenFHE BINARY serializations written by the Python test; public material only, no
// secret key): the CryptoContext, the public EvalMult key, and in <in_dir> stored_0_<j>.bin
// (dimension j of the one stored block), query_<j>.bin (pre-expanded query, value j in every
// slot) for j < dims, and mask_<name>.bin (an encrypted 0/1 mask, one level above the stored
// columns) for every <mask> name given.
//
// Stage A - one kernel at a time, each fed the same OpenFHE inputs and compared EXACTLY (every RNS
// word, moduli, level, noise-scale degree, scaling factor) with OpenFHE's CPU result:
//   relinearize at the stored level and at the mask level   vs  CryptoContext::Relinearize
//   rescale at both levels                                  vs  CryptoContext::Rescale
//   ciphertext x ciphertext multiply at the mask level      vs  CryptoContext::EvalMult
// Stage B - the GPU-only pipeline (search kernel, relinearize, rescale, mask, rescale, with no
// OpenFHE call between the first and the last device step) against the OpenFHE chain
// EvalMultNoRelin/EvalAdd -> Relinearize -> Rescale -> EvalMult(mask) -> Rescale, exact at every
// stage. Writes gpu_final_<mask>.bin and cpu_final_<mask>.bin to <out_dir>.
// Prints PASS/FAIL lines, indices and timings only, never data values.
// Exit code: 0 all exact, 1 mismatch, 2 error.

#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/gpu_search.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using ombra::core::GpuSearch;
using ombra::core::SearchStage;
using Ct = Ciphertext<DCRTPoly>;

namespace {

int g_failures = 0;

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

// Exact comparison; prints the first difference as indices only.
void check(const std::string& stage, const Ct& gpu, const Ct& cpu) {
  const auto a = ombra::core::flatten_one(gpu);
  const auto b = ombra::core::flatten_one(cpu);
  bool ok = true;
  if (a.components != b.components || a.num_towers != b.num_towers || a.ring_dim != b.ring_dim ||
      a.moduli != b.moduli) {
    std::printf("%s: FAIL (shape or moduli differ)\n", stage.c_str());
    ok = false;
  } else if (a.level != b.level || a.noise_scale_deg != b.noise_scale_deg ||
             a.scaling_factor != b.scaling_factor) {
    std::printf("%s: FAIL (level %zu/%zu, noise degree %zu/%zu, scaling factor %s)\n",
                stage.c_str(), a.level, b.level, a.noise_scale_deg, b.noise_scale_deg,
                a.scaling_factor == b.scaling_factor ? "equal" : "differs");
    ok = false;
  } else if (gpu->GetScalingFactorInt() != cpu->GetScalingFactorInt()) {
    std::printf("%s: FAIL (integer scaling factor differs)\n", stage.c_str());
    ok = false;
  } else if (a.words != b.words) {
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < a.words.size(); ++i) {
      if (a.words[i] != b.words[i] && bad++ == 0) first = i;
    }
    std::printf(
        "%s: FAIL (%zu of %zu words differ; first at component %zu, tower %zu, index %zu)\n",
        stage.c_str(), bad, a.words.size(), first / (a.num_towers * a.ring_dim),
        (first / a.ring_dim) % a.num_towers, first % a.ring_dim);
    ok = false;
  }
  if (ok) {
    std::printf("%s: PASS (level %zu, noise degree %zu, %zu components x %zu towers)\n",
                stage.c_str(), a.level, a.noise_scale_deg, a.components, a.num_towers);
  } else {
    ++g_failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 7) {
    std::fprintf(stderr,
                 "usage: ombra_search_gpu <context> <evalmultkey> <in_dir> <out_dir> <dims> "
                 "<mask> [mask..]\n");
    return 2;
  }
  try {
    const std::string in_dir = argv[3], out_dir = argv[4];
    const std::size_t d = std::stoul(argv[5]);
    std::vector<std::string> mask_names;
    for (int i = 6; i < argc; ++i) mask_names.push_back(argv[i]);

    CryptoContext<DCRTPoly> cc;
    if (!Serial::DeserializeFromFile(argv[1], cc, SerType::BINARY)) {
      throw std::runtime_error("cannot read crypto context");
    }
    std::ifstream keyfile(argv[2], std::ios::in | std::ios::binary);
    if (!keyfile.is_open() ||
        !CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(keyfile, SerType::BINARY)) {
      throw std::runtime_error("cannot read the EvalMult key");
    }

    std::vector<Ct> stored(d), query(d);
    for (std::size_t j = 0; j < d; ++j) {
      stored[j] = load_ct(in_dir + "/stored_0_" + std::to_string(j) + ".bin");
      query[j] = load_ct(in_dir + "/query_" + std::to_string(j) + ".bin");
    }
    std::vector<Ct> masks;
    for (const auto& name : mask_names) masks.push_back(load_ct(in_dir + "/mask_" + name + ".bin"));

    const auto& key_vec = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(stored[0]->GetKeyTag());
    if (key_vec.empty()) throw std::runtime_error("no EvalMult key for this key tag");

    GpuSearch gs(cc, stored[0], masks[0], key_vec[0], d, 1);
    std::printf("device: %s; d=%zu towers=%zu/%zu\n", ombra::core::cuda_device_name().c_str(), d,
                gs.level0().num_towers(), gs.level1().num_towers());

    // Stage A: one kernel at a time.
    const Ct x3 = cc->EvalMultNoRelin(stored[0], query[0]);  // 3 components, stored level
    const Ct cpu_rel = cc->Relinearize(x3);
    check("relinearize, level 0", gs.level0().relinearize(x3), cpu_rel);
    const Ct cpu_res = cc->Rescale(cpu_rel);
    check("rescale, level 0 -> 1", gs.level0().rescale(cpu_rel), cpu_res);
    const Ct m3 = cc->EvalMultNoRelin(cpu_res, masks[0]);
    check("relinearize, level 1", gs.level1().relinearize(m3), cc->Relinearize(m3));
    const Ct cpu_masked = cc->EvalMult(cpu_res, masks[0]);
    check("ciphertext multiply with relinearization, level 1",
          gs.level1().multiply(cpu_res, masks[0]), cpu_masked);
    check("rescale, level 1 -> 2", gs.level1().rescale(cpu_masked), cc->Rescale(cpu_masked));

    // Stage B: the GPU-only pipeline against the OpenFHE chain.
    Ct sum = cc->EvalMultNoRelin(stored[0], query[0]);
    for (std::size_t j = 1; j < d; ++j) {
      cc->EvalAddInPlace(sum, cc->EvalMultNoRelin(stored[j], query[j]));
    }
    const Ct ref_rel = cc->Relinearize(sum);
    const Ct ref_res = cc->Rescale(ref_rel);

    gs.upload_block(0, stored);
    gs.upload_query(query);
    for (std::size_t m = 0; m < masks.size(); ++m) {
      const std::string& name = mask_names[m];
      gs.upload_mask(0, masks[m]);
      ombra::core::GpuSearchTiming t = gs.run();
      if (m == 0) t = gs.run();  // second run: kernels loaded, no first-touch costs
      const Ct ref_masked = cc->EvalMult(ref_res, masks[m]);
      const Ct ref_final = cc->Rescale(ref_masked);
      const std::string tag = " (d=" + std::to_string(d) + ", mask " + name + ")";
      if (m == 0) check("pipeline stage: search sum" + tag, gs.download(SearchStage::kSum, 0), sum);
      check("pipeline stage: relinearize" + tag, gs.download(SearchStage::kRelin, 0), ref_rel);
      check("pipeline stage: rescale" + tag, gs.download(SearchStage::kRescaled, 0), ref_res);
      check("pipeline stage: mask" + tag, gs.download(SearchStage::kMasked, 0), ref_masked);
      const Ct gpu_final = gs.download(SearchStage::kFinal, 0);
      check("pipeline final: rescale" + tag, gpu_final, ref_final);
      std::printf(
          "timing d=%zu mask=%s search_s=%.5f relin_s=%.5f rescale1_s=%.5f mask_s=%.5f "
          "rescale2_s=%.5f total_s=%.5f\n",
          d, name.c_str(), t.search_s, t.relin_s, t.rescale1_s, t.mask_s, t.rescale2_s, t.total_s);
      save_ct(out_dir + "/gpu_final_" + name + ".bin", gpu_final);
      save_ct(out_dir + "/cpu_final_" + name + ".bin", ref_final);
    }

    if (g_failures != 0) {
      std::printf("GPU relinearize/rescale/mask exact match with OpenFHE: FAIL (%d checks)\n",
                  g_failures);
      return 1;
    }
    std::printf("GPU relinearize/rescale/mask exact match with OpenFHE: PASS\n");
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
