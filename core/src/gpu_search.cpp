// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "ombra/core/gpu_search.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>

#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/security_floor_context.hpp"

namespace ombra::core {
namespace {

using namespace lbcrypto;
using u64 = std::uint64_t;
using Clock = std::chrono::steady_clock;

[[noreturn]] void fail(const std::string& what) {
  throw std::invalid_argument("gpu search: " + what);
}

double seconds_since(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

CtMeta meta_of(const FlatOne& f) { return {f.level, f.noise_scale_deg, f.scaling_factor}; }

bool same_meta(const CtMeta& a, const CtMeta& b) {
  return a.level == b.level && a.noise_scale_deg == b.noise_scale_deg &&
         a.scaling_factor == b.scaling_factor;
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// GpuCtOps

GpuCtOps::GpuCtOps(const CryptoContext<DCRTPoly>& cc, const Ct& sample,
                   const EvalKey<DCRTPoly>& relin_key)
    : cc_((require_security_floor(cc),
           require_eval_key_matches(relin_key, sample, "relinearization key"), cc)),
      tables_(extract_expansion_tables(cc, sample)),
      device_(tables_) {
  const auto cp = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(cc->GetCryptoParameters());
  if (!cp || cp->GetScalingTechnique() != FIXEDMANUAL) {
    fail("only CKKS with FIXEDMANUAL scaling is supported");
  }
  key_tag_ = sample->GetKeyTag();
  composite_degree_ = cp->GetCompositeDegree();
  if (tables_.num_q >= 2) mod_reduce_factor_ = cp->GetModReduceFactor(tables_.num_q - 1);
  const std::size_t full_q = cp->GetElementParams()->GetParams().size();
  device_.set_relin_key(extract_key_words(relin_key, tables_, full_q));
}

std::vector<u64> GpuCtOps::checked_words(const Ct& ct, std::size_t components) const {
  if (ct->GetKeyTag() != key_tag_) fail("ciphertext is encrypted under a different key");
  FlatOne f = flatten_one(ct);
  if (f.components != components) fail("unexpected number of components");
  if (f.num_towers != tables_.num_q || f.ring_dim != tables_.ring_dim) {
    fail("ciphertext has a different level or ring dimension than this object");
  }
  if (f.moduli !=
      std::vector<u64>(tables_.moduli.begin(), tables_.moduli.begin() + tables_.num_q)) {
    fail("ciphertext moduli differ from the tables");
  }
  return std::move(f.words);
}

CtMeta GpuCtOps::rescaled(const CtMeta& m) const {
  // ModReduceInternalInPlace with levels = 1: degree - 1 / compositeDegree, level + 1, scaling
  // factor / GetModReduceFactor(sizeQl - 1).
  if (tables_.num_q < 2) fail("nothing to drop");
  return {m.level + 1, m.noise_scale_deg - 1 / composite_degree_,
          m.scaling_factor / mod_reduce_factor_};
}

CtMeta GpuCtOps::multiplied(const CtMeta& a, const CtMeta& b) {
  // EvalMultCore: result = ctxt1->CloneEmpty(), degrees added, scaling factors multiplied.
  return {a.level, a.noise_scale_deg + b.noise_scale_deg, a.scaling_factor * b.scaling_factor};
}

GpuCtOps::Ct GpuCtOps::relinearize(const Ct& ct3) {
  const std::size_t nq = tables_.num_q, n = tables_.ring_dim;
  const std::vector<u64> words = checked_words(ct3, 3);
  DeviceWords in(words.size()), out(2 * nq * n);
  in.upload(words);
  device_.relinearize({in.data()}, {out.data()});
  device_.synchronize();
  const FlatOne f = flatten_one(ct3);
  return make_ciphertext_like(ct3, 2, nq, out.download(0, out.size()), f.level, f.noise_scale_deg,
                              f.scaling_factor);
}

GpuCtOps::Ct GpuCtOps::rescale(const Ct& ct2) {
  const std::size_t nq = tables_.num_q, n = tables_.ring_dim;
  const std::vector<u64> words = checked_words(ct2, 2);
  DeviceWords in(words.size()), out(2 * (nq - 1) * n);
  in.upload(words);
  device_.rescale({in.data()}, {out.data()});
  device_.synchronize();
  const CtMeta m = rescaled(meta_of(flatten_one(ct2)));
  return make_ciphertext_like(ct2, 2, nq - 1, out.download(0, out.size()), m.level,
                              m.noise_scale_deg, m.scaling_factor);
}

GpuCtOps::Ct GpuCtOps::multiply(const Ct& a, const Ct& b) {
  const std::size_t nq = tables_.num_q, n = tables_.ring_dim;
  const std::vector<u64> wa = checked_words(a, 2), wb = checked_words(b, 2);
  const CtMeta ma = meta_of(flatten_one(a)), mb = meta_of(flatten_one(b));
  if (ma.level != mb.level) fail("multiply: operands differ in level");
  DeviceWords da(wa.size()), db(wb.size()), prod(3 * nq * n), out(2 * nq * n);
  da.upload(wa);
  db.upload(wb);
  device_.multiply({da.data()}, {db.data()}, {prod.data()});
  device_.relinearize({prod.data()}, {out.data()});
  device_.synchronize();
  const CtMeta m = multiplied(ma, mb);
  Ct result = make_ciphertext_like(a, 2, nq, out.download(0, out.size()), m.level,
                                   m.noise_scale_deg, m.scaling_factor);
  result->SetScalingFactorInt(a->GetScalingFactorInt().ModMul(
      b->GetScalingFactorInt(), cc_->GetCryptoParameters()->GetPlaintextModulus()));
  return result;
}

// ---------------------------------------------------------------------------------------------
// GpuSearch

struct GpuSearch::Impl {
  CryptoContext<DCRTPoly> cc;
  std::size_t dims, blocks;
  std::unique_ptr<GpuCtOps> ops0, ops1;
  std::unique_ptr<ColumnSearchDevice> search;
  Ct stored_template, mask_template;
  std::size_t n = 0, nq0 = 0, nq1 = 0;

  // Device buffers (see the stage list in the header): s1 relinearized, s2 rescaled, masks, m3 the
  // masked product, m2 relinearized, s3 final.
  std::unique_ptr<DeviceWords> s1, s2, masks, m3, m2, s3;

  CtMeta stored_meta{}, query_meta{}, mask_meta{};
  bool have_stored = false, have_query = false;
  std::vector<char> have_mask;
  CtMeta sum_meta, rescaled_meta, masked_meta, final_meta;

  std::size_t words3_0() const { return 3 * nq0 * n; }
  std::size_t words2_0() const { return 2 * nq0 * n; }
  std::size_t words2_1() const { return 2 * nq1 * n; }
  std::size_t words3_1() const { return 3 * nq1 * n; }
  std::size_t words2_2() const { return 2 * (nq1 - 1) * n; }
};

GpuSearch::GpuSearch(const CryptoContext<DCRTPoly>& cc, const Ct& stored_sample,
                     const Ct& mask_sample, const EvalKey<DCRTPoly>& relin_key, std::size_t dims,
                     std::size_t blocks)
    : impl_(std::make_unique<Impl>()) {
  Impl& s = *impl_;
  if (dims == 0 || blocks == 0) fail("empty shape");
  require_same_key(stored_sample, mask_sample, "search samples");
  s.cc = cc;
  s.dims = dims;
  s.blocks = blocks;
  s.ops0 = std::make_unique<GpuCtOps>(cc, stored_sample, relin_key);
  s.ops1 = std::make_unique<GpuCtOps>(cc, mask_sample, relin_key);
  s.n = s.ops0->ring_dim();
  s.nq0 = s.ops0->num_towers();
  s.nq1 = s.ops1->num_towers();
  if (s.nq1 + 1 != s.nq0 || s.nq1 < 2) {
    fail("the mask must be one level above the stored columns, with a level left to rescale");
  }
  s.stored_template = stored_sample;
  s.mask_template = mask_sample;
  const FlatCiphertexts fs = flatten_ciphertexts({stored_sample});
  s.search =
      std::make_unique<ColumnSearchDevice>(ColumnSearchShape{dims, blocks, s.nq0, s.n}, fs.moduli);
  s.s1 = std::make_unique<DeviceWords>(blocks * s.words2_0());
  s.s2 = std::make_unique<DeviceWords>(blocks * s.words2_1());
  s.masks = std::make_unique<DeviceWords>(blocks * s.words2_1());
  s.m3 = std::make_unique<DeviceWords>(blocks * s.words3_1());
  s.m2 = std::make_unique<DeviceWords>(blocks * s.words2_1());
  s.s3 = std::make_unique<DeviceWords>(blocks * s.words2_2());
  s.have_mask.assign(blocks, 0);
}

GpuSearch::~GpuSearch() = default;

GpuCtOps& GpuSearch::level0() { return *impl_->ops0; }
GpuCtOps& GpuSearch::level1() { return *impl_->ops1; }

void GpuSearch::upload_block(std::size_t block, const std::vector<Ct>& columns) {
  Impl& s = *impl_;
  if (block >= s.blocks || columns.size() != s.dims) fail("bad block index or dimension count");
  s.ops0->checked_words(columns[0], 2);  // level, tower count and moduli of the first column
  upload_block_flat(block, flatten_ciphertexts(columns));
}

void GpuSearch::upload_block_flat(std::size_t block, const FlatCiphertexts& f) {
  Impl& s = *impl_;
  if (block >= s.blocks || f.count != s.dims) fail("bad block index or dimension count");
  if (f.key_tag != s.stored_template->GetKeyTag() ||
      f.encoding != s.stored_template->GetEncodingType()) {
    fail("block is encrypted under a different key");
  }
  if (f.num_towers != s.nq0 || f.ring_dim != s.n) fail("block has a different level or ring dim");
  const CtMeta m{f.level, f.noise_scale_deg, f.scaling_factor};
  if (s.have_stored && !same_meta(m, s.stored_meta))
    fail("blocks differ in level, degree or scale");
  s.stored_meta = m;
  s.have_stored = true;
  s.search->upload_block(block, f.words.data(), f.words.size());
}

void GpuSearch::upload_query(const std::vector<Ct>& columns) {
  Impl& s = *impl_;
  if (columns.size() != s.dims) fail("bad dimension count");
  const FlatCiphertexts f = flatten_ciphertexts(columns);
  s.ops0->checked_words(columns[0], 2);
  s.query_meta = {f.level, f.noise_scale_deg, f.scaling_factor};
  s.have_query = true;
  s.search->upload_query(f.words.data(), f.words.size());
}

void GpuSearch::copy_block(std::size_t src, std::size_t dst) {
  impl_->search->copy_block(src, dst);
}

double GpuSearch::last_upload_seconds() const { return impl_->search->last_upload_seconds(); }

std::size_t GpuSearch::device_bytes() const {
  const Impl& s = *impl_;
  const ColumnSearchShape sh{s.dims, s.blocks, s.nq0, s.n};
  return (sh.stored_words() + sh.query_words() + sh.out_words() +
          s.blocks *
              (s.words2_0() + 2 * s.words2_1() + s.words3_1() + s.words2_1() + s.words2_2())) *
         sizeof(u64);
}

void GpuSearch::set_query_from_device(const std::vector<const u64*>& column_ptrs,
                                      const CtMeta& meta) {
  Impl& s = *impl_;
  if (column_ptrs.size() != s.dims) fail("bad dimension count");
  for (std::size_t j = 0; j < s.dims; ++j)
    s.search->set_query_column_from_device(j, column_ptrs[j]);
  s.query_meta = meta;
  s.have_query = true;
}

void GpuSearch::upload_mask(std::size_t block, const Ct& mask) {
  Impl& s = *impl_;
  if (block >= s.blocks) fail("bad block index");
  const std::vector<u64> words = s.ops1->checked_words(mask, 2);
  const CtMeta m = meta_of(flatten_one(mask));
  if (std::count(s.have_mask.begin(), s.have_mask.end(), 1) != 0 && !same_meta(m, s.mask_meta)) {
    fail("masks differ in level, degree or scale");
  }
  s.mask_meta = m;
  s.have_mask[block] = 1;
  s.masks->upload(words, block * s.words2_1());
}

CtMeta GpuSearch::meta(SearchStage stage) const {
  const Impl& s = *impl_;
  switch (stage) {
    case SearchStage::kSum:
    case SearchStage::kRelin:
      return s.sum_meta;
    case SearchStage::kRescaled:
      return s.rescaled_meta;
    case SearchStage::kMasked:
      return s.masked_meta;
    case SearchStage::kFinal:
      return s.final_meta;
  }
  fail("bad stage");
}

GpuSearchTiming GpuSearch::run() {
  Impl& s = *impl_;
  if (!s.have_stored || !s.have_query) fail("upload the stored blocks and the query first");
  for (char h : s.have_mask) {
    if (!h) fail("upload a mask for every block first");
  }
  if (!same_meta(s.stored_meta, s.query_meta))
    fail("stored and query differ in level, degree or scale");
  if (s.mask_meta.level != s.stored_meta.level + 1) fail("mask must be one level above the scores");

  // Metadata, derived exactly as OpenFHE derives it at each step.
  s.sum_meta = GpuCtOps::multiplied(s.stored_meta, s.query_meta);
  s.rescaled_meta = s.ops0->rescaled(s.sum_meta);
  s.masked_meta = GpuCtOps::multiplied(s.rescaled_meta, s.mask_meta);
  s.final_meta = s.ops1->rescaled(s.masked_meta);

  const std::size_t nb = s.blocks;
  const auto ptrs_in = [nb](const u64* base, std::size_t stride) {
    std::vector<const u64*> v(nb);
    for (std::size_t b = 0; b < nb; ++b) v[b] = base + b * stride;
    return v;
  };
  const auto ptrs_out = [nb](u64* base, std::size_t stride) {
    std::vector<u64*> v(nb);
    for (std::size_t b = 0; b < nb; ++b) v[b] = base + b * stride;
    return v;
  };

  GpuSearchTiming t;
  const auto t_start = Clock::now();
  auto t0 = t_start;
  const auto lap = [&](double& field) {
    field = seconds_since(t0);
    t0 = Clock::now();
  };
  s.search->run();  // the multiply-and-add kernel; waits
  lap(t.search_s);

  const u64* sum = s.search->result_device();
  s.ops0->device().relinearize(ptrs_in(sum, s.words3_0()), ptrs_out(s.s1->data(), s.words2_0()));
  s.ops0->device().synchronize();
  lap(t.relin_s);

  s.ops0->device().rescale(ptrs_in(s.s1->data(), s.words2_0()),
                           ptrs_out(s.s2->data(), s.words2_1()));
  s.ops0->device().synchronize();
  lap(t.rescale1_s);

  s.ops1->device().multiply(ptrs_in(s.s2->data(), s.words2_1()),
                            ptrs_in(s.masks->data(), s.words2_1()),
                            ptrs_out(s.m3->data(), s.words3_1()));
  s.ops1->device().relinearize(ptrs_in(s.m3->data(), s.words3_1()),
                               ptrs_out(s.m2->data(), s.words2_1()));
  s.ops1->device().synchronize();
  lap(t.mask_s);

  s.ops1->device().rescale(ptrs_in(s.m2->data(), s.words2_1()),
                           ptrs_out(s.s3->data(), s.words2_2()));
  s.ops1->device().synchronize();
  lap(t.rescale2_s);
  t.total_s = seconds_since(t_start);
  return t;
}

std::vector<u64> GpuSearch::download_words(SearchStage stage, std::size_t block) {
  Impl& s = *impl_;
  if (block >= s.blocks) fail("bad block index");
  const auto out = [&](const DeviceWords& buf, std::size_t comps, std::size_t towers) {
    const std::size_t w = comps * towers * s.n;
    return buf.download(block * w, w);
  };
  switch (stage) {
    case SearchStage::kSum:
      fail("download_words does not support the search sum");
    case SearchStage::kRelin:
      return out(*s.s1, 2, s.nq0);
    case SearchStage::kRescaled:
      return out(*s.s2, 2, s.nq1);
    case SearchStage::kMasked:
      return out(*s.m2, 2, s.nq1);
    case SearchStage::kFinal:
      return out(*s.s3, 2, s.nq1 - 1);
  }
  fail("bad stage");
}

GpuSearch::Ct GpuSearch::download(SearchStage stage, std::size_t block) {
  Impl& s = *impl_;
  if (block >= s.blocks) fail("bad block index");
  const CtMeta m = meta(stage);
  const auto out = [&](const DeviceWords& buf, std::size_t comps, std::size_t towers,
                       const Ct& like) {
    const std::size_t w = comps * towers * s.n;
    return make_ciphertext_like(like, comps, towers, buf.download(block * w, w), m.level,
                                m.noise_scale_deg, m.scaling_factor);
  };
  switch (stage) {
    case SearchStage::kSum: {  // test aid: copies the whole search output to the host
      const std::size_t w = s.words3_0();
      std::vector<u64> all;
      s.search->download_result(all);
      const std::vector<u64> words(all.begin() + block * w, all.begin() + (block + 1) * w);
      return make_ciphertext_like(s.stored_template, 3, s.nq0, words, m.level, m.noise_scale_deg,
                                  m.scaling_factor);
    }
    case SearchStage::kRelin:
      return out(*s.s1, 2, s.nq0, s.stored_template);
    case SearchStage::kRescaled:
      return out(*s.s2, 2, s.nq1, s.mask_template);
    case SearchStage::kMasked:
      return out(*s.m2, 2, s.nq1, s.mask_template);
    case SearchStage::kFinal:
      return out(*s.s3, 2, s.nq1 - 1, s.mask_template);
  }
  fail("bad stage");
}

}  // namespace ombra::core
