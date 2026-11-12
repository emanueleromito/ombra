// ombra_refusal_check: shows that the GPU entry points refuse inputs that do not belong together
// (review findings F4, F5, F6). Public material only, no secret key.
//
// Usage:
//   ombra_refusal_check keys  <context> <dir> <d>
//   ombra_refusal_check shape <context1> <context2> <dir>
//
// keys: <dir> holds ciphertexts under two key pairs A and B of the same context (a_<j>.bin,
//   b_<j>.bin, j < 2, level 0; a_mask.bin, b_mask.bin, level 1), the expansion roots a_root.bin and
//   b_root.bin, autokeys.bin (automorphism keys of both) and evalmult_a.bin, evalmult_b.bin.
//   Mixed-key, wrong-degree, wrong-scale, wrong-level and wrong-encoding inputs are sent to every
//   GPU entry point, plus valid controls.
// shape: queries q1_<j>.bin under context1 and q2_<j>.bin under context2 (a different ring
//   dimension), stored_<j>.bin under context1.
//
// Prints one line per case: "CASE <name>: ACCEPTED" or "CASE <name>: REFUSED <message>". The
// calling test asserts the message of every refusal. Exit code 0, or 2 on a setup error.

#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/gpu_search.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

using namespace lbcrypto;
using Ct = Ciphertext<DCRTPoly>;
using ombra::core::GpuCtOps;
using ombra::core::GpuExpansion;
using ombra::core::GpuSearch;

namespace {

Ct load_ct(const std::string& path) {
  Ct ct;
  if (!Serial::DeserializeFromFile(path, ct, SerType::BINARY)) {
    throw std::runtime_error("cannot read ciphertext " + path);
  }
  return ct;
}

CryptoContext<DCRTPoly> load_cc(const std::string& path) {
  CryptoContext<DCRTPoly> cc;
  if (!Serial::DeserializeFromFile(path, cc, SerType::BINARY)) {
    throw std::runtime_error("cannot read crypto context " + path);
  }
  ombra::core::require_security_floor(cc);
  return cc;
}

void run_case(const std::string& name, const std::function<void()>& fn) {
  try {
    fn();
    std::printf("CASE %s: ACCEPTED\n", name.c_str());
  } catch (const std::exception& e) {
    std::printf("CASE %s: REFUSED %s\n", name.c_str(), e.what());
  }
}

void load_evalmult(const std::string& path) {
  std::ifstream f(path, std::ios::in | std::ios::binary);
  if (!f.is_open() || !CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(f, SerType::BINARY)) {
    throw std::runtime_error("cannot read the EvalMult key " + path);
  }
}

EvalKey<DCRTPoly> relin_of(const Ct& ct) {
  const auto& v = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(ct->GetKeyTag());
  if (v.empty()) throw std::runtime_error("no EvalMult key for this key tag");
  return v[0];
}

using Cols = std::vector<Ct>;

int run_keys(const std::string& ctx, const std::string& dir, std::size_t d) {
  const auto cc = load_cc(ctx);
  std::ifstream af(dir + "/autokeys.bin", std::ios::in | std::ios::binary);
  if (!af.is_open() ||
      !CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(af, SerType::BINARY)) {
    throw std::runtime_error("cannot read automorphism keys");
  }
  load_evalmult(dir + "/evalmult_a.bin");
  load_evalmult(dir + "/evalmult_b.bin");

  const Cols a = {load_ct(dir + "/a_0.bin"), load_ct(dir + "/a_1.bin")};
  const Cols b = {load_ct(dir + "/b_0.bin"), load_ct(dir + "/b_1.bin")};
  const Ct a_mask = load_ct(dir + "/a_mask.bin"), b_mask = load_ct(dir + "/b_mask.bin");
  const Ct a_root = load_ct(dir + "/a_root.bin"), b_root = load_ct(dir + "/b_root.bin");
  if (a[0]->GetKeyTag() == b[0]->GetKeyTag()) throw std::runtime_error("key pairs share a tag");
  const auto relin_a = relin_of(a[0]), relin_b = relin_of(b[0]);
  const auto& keys_a = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(a_root->GetKeyTag());
  const auto& keys_b = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(b_root->GetKeyTag());

  // ---- Metadata on expansion upload ----
  {
    GpuExpansion gx(cc, a_root, keys_a, d);
    run_case("expansion_upload_valid", [&] { gx.upload(a_root); });
    Ct deg = a_root->Clone();
    deg->SetNoiseScaleDeg(a_root->GetNoiseScaleDeg() + 1);
    run_case("expansion_upload_degree", [&] { gx.upload(deg); });
    Ct scale = a_root->Clone();
    scale->SetScalingFactor(a_root->GetScalingFactor() * 2.0);
    run_case("expansion_upload_scale", [&] { gx.upload(scale); });
    // A real ciphertext one level down (a mask): different level, tower count and scale.
    run_case("expansion_upload_level", [&] { gx.upload(a_mask); });

    // ---- Keys ----
    run_case("expansion_upload_other_key", [&] { gx.upload(b_root); });
    run_case("expansion_upload_other_encoding", [&] {
      Ct enc = a_root->Clone();
      enc->SetEncodingType(COEF_PACKED_ENCODING);
      gx.upload(enc);
    });
  }
  run_case("expansion_keys_valid", [&] { GpuExpansion g(cc, a_root, keys_a, d); });
  run_case("expansion_keys_of_other_key", [&] { GpuExpansion g(cc, a_root, keys_b, d); });

  // ---- add / multiply by plaintext ----
  const std::size_t slots = cc->GetEncodingParams()->GetBatchSize();
  const Plaintext pt = cc->MakeCKKSPackedPlaintext(std::vector<double>(slots, 0.5));
  run_case("add_valid", [&] { ombra::core::gpu_eval_add(a[0], a[1]); });
  run_case("add_other_key", [&] { ombra::core::gpu_eval_add(a[0], b[0]); });
  run_case("add_other_encoding", [&] {
    Ct enc = a[1]->Clone();
    enc->SetEncodingType(COEF_PACKED_ENCODING);
    ombra::core::gpu_eval_add(a[0], enc);
  });
  run_case("mult_plain_valid", [&] { ombra::core::gpu_eval_mult_plain(a[0], pt); });
  run_case("mult_plain_other_encoding", [&] {
    Ct enc = a[0]->Clone();
    enc->SetEncodingType(COEF_PACKED_ENCODING);
    ombra::core::gpu_eval_mult_plain(enc, pt);
  });

  // ---- plain search ----
  const auto fa = ombra::core::flatten_ciphertexts({a[0]});
  const std::vector<std::uint64_t> zeros(2 * fa.num_towers * fa.ring_dim, 0);
  const std::vector<Plaintext> pts = {pt, pt};
  run_case("plain_search_valid", [&] { ombra::core::gpu_eval_plain_search(a, pts, zeros); });
  run_case("plain_search_other_key",
           [&] { ombra::core::gpu_eval_plain_search({a[0], b[1]}, pts, zeros); });

  // ---- column search, single and batched ----
  const std::vector<Cols> stored_a = {a};
  run_case("column_search_valid", [&] { ombra::core::gpu_eval_column_search(stored_a, a); });
  run_case("column_search_other_key", [&] { ombra::core::gpu_eval_column_search(stored_a, b); });
  run_case("column_search_mixed_query", [&] {
    ombra::core::gpu_eval_column_search(stored_a, {a[0], b[1]});
  });
  run_case("column_search_batched_valid",
           [&] { ombra::core::gpu_eval_column_search_batched(stored_a, {a, a}); });
  run_case("column_search_batched_other_key",
           [&] { ombra::core::gpu_eval_column_search_batched(stored_a, {b, b}); });
  run_case("column_search_batched_mixed_queries",
           [&] { ombra::core::gpu_eval_column_search_batched(stored_a, {a, b}); });

  // ---- GpuCtOps ----
  run_case("ops_ctor_valid", [&] { GpuCtOps o(cc, a[0], relin_a); });
  run_case("ops_ctor_relin_key_of_other_key", [&] { GpuCtOps o(cc, a[0], relin_b); });
  {
    GpuCtOps ops(cc, a[0], relin_a);
    run_case("ops_multiply_valid", [&] { ops.multiply(a[0], a[1]); });
    run_case("ops_multiply_other_key", [&] { ops.multiply(a[0], b[0]); });
    run_case("ops_rescale_valid", [&] { ops.rescale(a[0]); });
    run_case("ops_rescale_other_key", [&] { ops.rescale(b[0]); });
    run_case("ops_relinearize_other_key", [&] { ops.relinearize(b[0]); });
  }

  // ---- GpuSearch ----
  run_case("search_ctor_valid", [&] { GpuSearch g(cc, a[0], a_mask, relin_a, 2, 1); });
  run_case("search_ctor_mask_of_other_key", [&] { GpuSearch g(cc, a[0], b_mask, relin_a, 2, 1); });
  run_case("search_ctor_relin_key_of_other_key",
           [&] { GpuSearch g(cc, a[0], a_mask, relin_b, 2, 1); });
  {
    GpuSearch gs(cc, a[0], a_mask, relin_a, 2, 1);
    run_case("search_upload_block_valid", [&] { gs.upload_block(0, a); });
    run_case("search_upload_block_mixed", [&] { gs.upload_block(0, {a[0], b[1]}); });
    run_case("search_upload_block_other_key", [&] { gs.upload_block(0, b); });
    run_case("search_upload_block_flat_other_key",
             [&] { gs.upload_block_flat(0, ombra::core::flatten_ciphertexts(b)); });
    run_case("search_upload_query_valid", [&] { gs.upload_query(a); });
    run_case("search_upload_query_other_key", [&] { gs.upload_query(b); });
    run_case("search_upload_mask_valid", [&] { gs.upload_mask(0, a_mask); });
    run_case("search_upload_mask_other_key", [&] { gs.upload_mask(0, b_mask); });
  }
  return 0;
}

int run_shape(const std::string& ctx1, const std::string& ctx2, const std::string& dir) {
  const auto cc1 = load_cc(ctx1);
  const auto cc2 = load_cc(ctx2);
  if (cc1->GetRingDimension() == cc2->GetRingDimension()) {
    throw std::runtime_error("the two contexts must differ in ring dimension");
  }
  const Cols q1 = {load_ct(dir + "/q1_0.bin"), load_ct(dir + "/q1_1.bin")};
  const Cols q2 = {load_ct(dir + "/q2_0.bin"), load_ct(dir + "/q2_1.bin")};
  const Cols stored = {load_ct(dir + "/stored_0.bin"), load_ct(dir + "/stored_1.bin")};
  run_case("batched_search_valid",
           [&] { ombra::core::gpu_eval_column_search_batched({stored}, {q1, q1}); });
  run_case("batched_search_smaller_ring_query",
           [&] { ombra::core::gpu_eval_column_search_batched({stored}, {q1, q2}); });
  run_case("batched_search_smaller_ring_first_query",
           [&] { ombra::core::gpu_eval_column_search_batched({stored}, {q2, q1}); });
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "keys" && argc == 5) return run_keys(argv[2], argv[3], std::stoul(argv[4]));
    if (mode == "shape" && argc == 5) return run_shape(argv[2], argv[3], argv[4]);
    std::fprintf(stderr,
                 "usage: ombra_refusal_check keys <context> <dir> <d>\n"
                 "       ombra_refusal_check shape <context1> <context2> <dir>\n");
    return 2;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}
