#include "ombra/search/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "device_util.hpp"
#include "key/key-ser.h"
#include "ombra/cache/gpu_cache.hpp"
#include "ombra/core/ciphertext_ops.hpp"
#include "ombra/core/column_search.hpp"
#include "ombra/core/expansion_gpu.hpp"
#include "ombra/core/expansion_tree.hpp"
#include "ombra/core/file_envelope.hpp"
#include "ombra/core/gpu_search.hpp"
#include "ombra/core/security_floor_context.hpp"
#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-ser.h"

namespace ombra::search {

using namespace lbcrypto;

namespace {

using u64 = std::uint64_t;
using Ct = Ciphertext<DCRTPoly>;
namespace core = ombra::core;

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Counts and timings only (no plaintext in logs or errors).
void log_line(const char* what, std::size_t blocks, double ms) {
  std::fprintf(stderr, "[ombra.search] %s blocks=%zu ms=%.2f\n", what, blocks, ms);
}

std::string to_string_bytes(const std::vector<std::uint8_t>& b) {
  return std::string(reinterpret_cast<const char*>(b.data()), b.size());
}

}  // namespace

namespace detail {

// Everything loaded for one collection. Shared between the engine and its sessions.
struct Coll {
  storage::CollectionSpec spec;
  QueryLayout layout;
  CryptoContext<DCRTPoly> cc;
  std::size_t ring = 0, batch = 0;
  double column_scale = 0;  // scaling factor of a fresh level-0 ciphertext
  EvalKey<DCRTPoly> relin;
  core::GpuExpansion::KeyMap akeys;

  // GPU state, created by the first session (under the engine's GPU mutex).
  bool ready = false;
  Ct sample;  // level-0 ciphertext of this collection: the template for results
  std::unique_ptr<core::GpuCtOps> ops;
  std::unique_ptr<core::GpuExpansion> gx;
  std::unique_ptr<core::ColumnSearchLauncher> launch_centers, launch_data;
  std::vector<u64> moduli;  // the num_q Q moduli at level 0
  std::size_t nq = 0;
  std::unique_ptr<core::DeviceWords> search_out, relin_out, rescale_out;  // one batch of blocks
};

struct EngineCore {
  storage::Store& store;
  EngineConfig cfg;
  std::mutex gpu_mu;  // serialises all GPU work and the maps below
  std::map<std::string, std::shared_ptr<Coll>> colls;
  std::unique_ptr<cache::Stream> compute;
  std::unique_ptr<cache::GpuClusterCache> gpu_cache;

  struct Seg {
    std::string coll, file;
    std::uint32_t cluster = 0, n_columns = 0;
  };
  std::vector<Seg> segs;
  std::unordered_map<std::string, std::uint32_t> seg_index;  // "coll/file" -> index
  std::unordered_map<std::uint32_t, std::vector<std::string>> seg_ids;

  EngineCore(storage::Store& s, const EngineConfig& c) : store(s), cfg(c) {}

  std::uint32_t segment_index(const std::string& coll, const storage::SegmentRow& row) {
    auto key = coll + "/" + row.file;
    auto it = seg_index.find(key);
    if (it != seg_index.end()) return it->second;
    const auto idx = static_cast<std::uint32_t>(segs.size());
    segs.push_back({coll, row.file, row.cluster, row.n_columns});
    seg_index.emplace(std::move(key), idx);
    return idx;
  }

  // Ids of a segment in slot order, read once (segments are immutable).
  const std::vector<std::string>& segment_ids(std::uint32_t idx) {
    auto it = seg_ids.find(idx);
    if (it != seg_ids.end()) return it->second;
    const Seg& s = segs.at(idx);
    return seg_ids.emplace(idx, store.read_segment(s.coll, s.file).ids).first->second;
  }

  std::shared_ptr<Coll> find(const std::string& name) {
    auto it = colls.find(name);
    return it == colls.end() ? nullptr : it->second;
  }

  // The cache loader: one segment -> device words, a column at a time (the whole block as OpenFHE
  // objects would be hundreds of MB). Every column is checked against the collection.
  std::size_t block_words(const cache::BlockKey& key) {
    auto c = find(key.collection);
    if (!c || !c->ready) throw SearchError("loader: collection not ready");
    return static_cast<std::size_t>(segs.at(key.segment).n_columns) * 2 * c->nq * c->ring;
  }

  void load_block(const cache::BlockKey& key, u64* dst, std::size_t words) {
    auto c = find(key.collection);
    if (!c || !c->ready) throw SearchError("loader: collection not ready");
    const Seg& seg = segs.at(key.segment);
    const storage::SegmentData sd = store.read_segment(seg.coll, seg.file);
    const std::size_t col_words = 2 * c->nq * c->ring;
    if (sd.column_envelopes.size() * col_words != words) throw SearchError("loader: block shape");
    for (std::size_t j = 0; j < sd.column_envelopes.size(); ++j) {
      const core::Envelope e = core::read_envelope(sd.column_envelopes[j], core::EnvelopeKind::Ciphertext,
                                                   c->spec.preset, store.openfhe_version());
      std::istringstream in(to_string_bytes(e.payload), std::ios::in | std::ios::binary);
      Ct ct;
      Serial::Deserialize(ct, in, SerType::BINARY);
      if (!ct) throw SearchError("loader: cannot read a stored column");
      if (ct->GetKeyTag() != c->spec.key_tag)
        throw KeyTagMismatch("stored column was made under another key than the collection's");
      const core::FlatCiphertexts f = core::flatten_ciphertexts({ct});
      if (f.num_towers != c->nq || f.ring_dim != c->ring || f.moduli != c->moduli ||
          f.level != 0 || f.noise_scale_deg != 1 || f.scaling_factor != c->column_scale)
        throw SearchError("loader: stored column has an unexpected level, scale or shape");
      std::copy(f.words.begin(), f.words.end(), dst + j * col_words);
    }
  }

  // First use: GPU objects that need a ciphertext of the collection as a template.
  void ensure_gpu(Coll& c, const Ct& sample) {
    if (c.ready) return;
    c.sample = sample;
    c.column_scale = sample->GetScalingFactor();  // the scale every stored column must carry
    c.ops = std::make_unique<core::GpuCtOps>(c.cc, sample, c.relin);
    try {
      c.gx = std::make_unique<core::GpuExpansion>(c.cc, sample, c.akeys, c.layout.leaves());
    } catch (const std::invalid_argument&) {
      throw KeysMissing("automorphism keys do not cover the expansion of d + sum(cardinality)");
    }
    c.nq = c.ops->num_towers();
    c.moduli.assign(c.ops->tables().moduli.begin(), c.ops->tables().moduli.begin() + c.nq);
    c.launch_centers = std::make_unique<core::ColumnSearchLauncher>(
        core::ColumnSearchShape{c.layout.center_columns(), 1, c.nq, c.ring}, c.moduli);
    c.launch_data = std::make_unique<core::ColumnSearchLauncher>(
        core::ColumnSearchShape{c.layout.data_columns(), 1, c.nq, c.ring}, c.moduli);
    const std::size_t b = std::max<std::size_t>(1, cfg.blocks_per_batch);
    c.search_out = std::make_unique<core::DeviceWords>(b * 3 * c.nq * c.ring);
    c.relin_out = std::make_unique<core::DeviceWords>(b * 2 * c.nq * c.ring);
    c.rescale_out = std::make_unique<core::DeviceWords>(b * 2 * (c.nq - 1) * c.ring);
    c.ready = true;
  }
};

}  // namespace detail

using detail::Coll;
using detail::EngineCore;

// ---------------------------------------------------------------------------------------------
// SearchSession state (defined first: the engine creates sessions)

struct SearchSession::Impl {
  EngineCore* eng = nullptr;
  std::shared_ptr<Coll> coll;
  std::unique_ptr<cache::DeviceArray> query;  // D leaves, [leaf][comp][tower][coeff]

  std::vector<BlockScores> run(const std::vector<std::uint32_t>& clusters, bool centers);
};

// ---------------------------------------------------------------------------------------------
// SearchEngine

SearchEngine::SearchEngine(storage::Store& store, const EngineConfig& config)
    : impl_(std::make_unique<EngineCore>(store, config)) {
  EngineCore& e = *impl_;
  e.compute = std::make_unique<cache::Stream>();
  cache::BlockLoader loader;
  loader.size_words = [&e](const cache::BlockKey& k) { return e.block_words(k); };
  loader.load = [&e](const cache::BlockKey& k, u64* dst, std::size_t w) { e.load_block(k, dst, w); };
  cache::CacheConfig cc;
  cc.gpu_budget_bytes = config.gpu_budget_bytes;
  cc.host_budget_bytes = config.host_budget_bytes;
  cc.lookahead = config.lookahead;
  e.gpu_cache = std::make_unique<cache::GpuClusterCache>(cc, std::move(loader));
}

SearchEngine::~SearchEngine() = default;

cache::CacheMetrics SearchEngine::cache_metrics() const { return impl_->gpu_cache->metrics(); }

void SearchEngine::load_collection(const std::string& name) {
  EngineCore& e = *impl_;
  std::vector<std::uint8_t> ctx_b, mult_b, auto_b;
  if (!e.store.get_collection(name, nullptr)) throw CollectionNotFound("unknown collection");
  if (!e.store.get_key_file(name, storage::kKeyContext, &ctx_b) ||
      !e.store.get_key_file(name, storage::kKeyEvalMult, &mult_b) ||
      !e.store.get_key_file(name, storage::kKeyAutomorphism, &auto_b))
    throw KeysMissing("context, EvalMult key or automorphism keys not uploaded");
  install_keys(prepare_keys(name, ctx_b, mult_b, auto_b));
}

void SearchEngine::install_keys(PreparedKeys&& prepared) {
  EngineCore& e = *impl_;
  if (!prepared.c_) throw SearchError("no prepared keys");
  std::lock_guard g(e.gpu_mu);
  e.colls[prepared.c_->spec.name] = std::move(prepared.c_);
}

PreparedKeys SearchEngine::prepare_keys(const std::string& name,
                                        const std::vector<std::uint8_t>& ctx_b,
                                        const std::vector<std::uint8_t>& mult_b,
                                        const std::vector<std::uint8_t>& auto_b) {
  EngineCore& e = *impl_;
  auto c = std::make_shared<Coll>();
  if (!e.store.get_collection(name, &c->spec)) throw CollectionNotFound("unknown collection");
  const std::string ofhe = e.store.openfhe_version();

  const auto payload = [&](const std::vector<std::uint8_t>& b, core::EnvelopeKind k) {
    return to_string_bytes(core::read_envelope(b, k, c->spec.preset, ofhe).payload);
  };
  {
    std::istringstream in(payload(ctx_b, core::EnvelopeKind::Context), std::ios::in | std::ios::binary);
    Serial::Deserialize(c->cc, in, SerType::BINARY);
    if (!c->cc) throw SearchError("cannot read the crypto context");
  }
  core::require_security_floor(c->cc);

  // The key stores of OpenFHE are process-wide, by key tag: drop this tag's entries, load, and
  // require that the collection's tag is now present (a file made under another key is refused).
  const std::string& tag = c->spec.key_tag;
  CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys(tag);
  CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(tag);
  // The key files carry their own key tag and are inserted under it. Remember which tags exist,
  // so that keys of another tag that this call inserts can be removed again when they are refused.
  std::set<std::string> mult_before;
  std::set<std::string> auto_before;
  for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalMultKeys()) mult_before.insert(kv.first);
  for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys())
    auto_before.insert(kv.first);
  const auto drop_foreign = [&] {
    std::vector<std::string> foreign;
    for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalMultKeys())
      if (!mult_before.count(kv.first)) foreign.push_back(kv.first);
    for (const auto& t : foreign) CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys(t);
    foreign.clear();
    for (const auto& kv : CryptoContextImpl<DCRTPoly>::GetAllEvalAutomorphismKeys())
      if (!auto_before.count(kv.first)) foreign.push_back(kv.first);
    for (const auto& t : foreign) CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(t);
  };
  try {
    {
      std::istringstream in(payload(mult_b, core::EnvelopeKind::PublicEvalKey),
                            std::ios::in | std::ios::binary);
      if (!CryptoContextImpl<DCRTPoly>::DeserializeEvalMultKey(in, SerType::BINARY))
        throw SearchError("cannot read the EvalMult key");
    }
    {
      std::istringstream in(payload(auto_b, core::EnvelopeKind::PublicEvalKey),
                            std::ios::in | std::ios::binary);
      if (!CryptoContextImpl<DCRTPoly>::DeserializeEvalAutomorphismKey(in, SerType::BINARY))
        throw SearchError("cannot read the automorphism keys");
    }
  } catch (const lbcrypto::OpenFHEException&) {
    // OpenFHE refuses to insert keys under a tag that already holds keys: the files belong to a key
    // other than this collection's (a collection with the same files and another tag).
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }
  try {
    const auto& v = CryptoContextImpl<DCRTPoly>::GetEvalMultKeyVector(tag);
    if (v.empty()) throw std::runtime_error("empty");
    c->relin = v[0];
    c->akeys = CryptoContextImpl<DCRTPoly>::GetEvalAutomorphismKeyMap(tag);
  } catch (const std::exception&) {
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }
  if (c->akeys.empty()) {
    drop_foreign();
    throw KeyTagMismatch("evaluation keys were made under another key than the collection's");
  }

  const auto cp = std::dynamic_pointer_cast<CryptoParametersRNS>(c->cc->GetCryptoParameters());
  if (!cp) throw SearchError("context is not an RNS scheme");
  c->ring = c->cc->GetRingDimension();
  c->batch = c->cc->GetEncodingParams()->GetBatchSize();
  c->column_scale = cp->GetScalingFactorReal(0);

  // F_max from the headroom of this context.
  const auto first_bits = cp->GetElementParams()->GetParams()[0]->GetModulus().GetMSB();
  const auto scale_bits = static_cast<std::size_t>(std::llround(std::log2(c->column_scale)));
  try {
    c->layout = make_layout(c->spec, c->ring, max_filter_fields(first_bits, scale_bits));
  } catch (const LayoutError& ex) {
    throw InvalidQuery(ex.what());
  }

  PreparedKeys out;
  out.c_ = std::move(c);
  return out;
}

std::unique_ptr<SearchSession> SearchEngine::open_session(
    const std::string& collection, const std::string& key_tag,
    const std::vector<std::uint8_t>& query_envelope, std::uint32_t filter_leaves) {
  EngineCore& e = *impl_;
  std::lock_guard g(e.gpu_mu);
  auto c = e.find(collection);
  if (!c) {
    storage::CollectionSpec s;
    if (!e.store.get_collection(collection, &s)) throw CollectionNotFound("unknown collection");
    throw KeysMissing("collection not loaded: upload the keys first");
  }
  if (key_tag != c->spec.key_tag) throw KeyTagMismatch("request key tag differs from the collection's");
  if (filter_leaves != c->layout.filter_leaves())
    throw InvalidQuery("filter_leaves differs from sum(cardinality) of the collection");

  Ct query;
  try {
    const core::Envelope env = core::read_envelope(query_envelope, core::EnvelopeKind::Ciphertext,
                                                   c->spec.preset, e.store.openfhe_version());
    std::istringstream in(to_string_bytes(env.payload), std::ios::in | std::ios::binary);
    Serial::Deserialize(query, in, SerType::BINARY);
  } catch (const std::exception& ex) {
    throw InvalidQuery(std::string("bad query envelope: ") + ex.what());
  }
  if (!query) throw InvalidQuery("bad query ciphertext");
  if (query->GetKeyTag() != c->spec.key_tag)
    throw KeyTagMismatch("query ciphertext was made under another key than the collection's");
  if (query->GetElements().size() != 2 || query->GetLevel() != 0 || query->GetNoiseScaleDeg() != 1 ||
      std::abs(query->GetScalingFactor() / c->column_scale - 1.0) > 1e-9 ||
      query->GetElements()[0].GetRingDimension() != c->ring)
    throw InvalidQuery("query ciphertext has an unexpected shape, level or scale");

  const auto t0 = Clock::now();
  e.ensure_gpu(*c, query);
  const std::size_t leaves = c->layout.leaves();
  const std::size_t leaf_words = 2 * c->nq * c->ring;

  // Expand once; the leaves are copied into one contiguous device buffer (the column search kernel
  // reads the query as D consecutive ciphertexts), the first d of them serve stage 1.
  auto sess = std::unique_ptr<SearchSession::Impl>(new SearchSession::Impl);
  sess->eng = &e;
  sess->coll = c;
  sess->query = std::make_unique<cache::DeviceArray>(leaves * leaf_words);
  {
    core::GpuExpansion& gx = *c->gx;
    const core::GpuCt root = gx.upload(query);
    std::vector<core::GpuCt> lv = core::expand_tree(root, leaves, c->ring, gx);
    gx.synchronize();
    for (std::size_t j = 0; j < leaves; ++j)
      device_copy_words(sess->query->get() + j * leaf_words, gx.device().device_ptr(lv[j].dev),
                        leaf_words);
    device_sync();
    lv.clear();
  }
  log_line("expand", leaves, ms_since(t0));
  return std::unique_ptr<SearchSession>(new SearchSession(std::move(sess)));
}

// ---------------------------------------------------------------------------------------------
// SearchSession

SearchSession::SearchSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SearchSession::~SearchSession() = default;  // frees the device leaves; the engine must outlive it

const QueryLayout& SearchSession::layout() const { return impl_->coll->layout; }

std::vector<BlockScores> SearchSession::score_centers() {
  return impl_->run({static_cast<std::uint32_t>(impl_->coll->spec.cluster_count)}, true);
}

std::vector<BlockScores> SearchSession::score_clusters(const std::vector<std::uint32_t>& clusters) {
  for (auto cl : clusters)
    if (cl >= impl_->coll->spec.cluster_count) throw InvalidQuery("cluster id out of range");
  return impl_->run(clusters, false);
}

std::vector<BlockScores> SearchSession::Impl::run(const std::vector<std::uint32_t>& clusters,
                                                  bool centers) {
  EngineCore& e = *eng;
  Coll& c = *coll;
  std::lock_guard g(e.gpu_mu);
  const auto t0 = Clock::now();

  struct Item {
    std::uint32_t idx;
    storage::SegmentRow row;
  };
  std::vector<Item> items;
  // One consistent view of the blocks and tombstones; the pin keeps every block file of the view on
  // disk even if a compaction or a client repack removes it from the catalog meanwhile.
  const storage::Snapshot snap = e.store.snapshot(c.spec.name);
  {
    const auto& rows = snap.segments;
    for (auto cl : clusters)
      for (const auto& r : rows)
        if (r.cluster == cl) items.push_back({e.segment_index(c.spec.name, r), r});
  }
  const std::size_t cols = centers ? c.layout.center_columns() : c.layout.data_columns();
  for (const auto& it : items)
    if (it.row.n_columns != cols) throw SearchError("segment has an unexpected column count");

  std::unordered_set<std::string> tombstones;
  if (!centers) {
    for (const auto& t : snap.tombstones) tombstones.insert(t);
  }

  const std::size_t nq = c.nq, n = c.ring, pw = nq * n;
  const core::ColumnSearchLauncher& launcher = centers ? *c.launch_centers : *c.launch_data;

  // Metadata as OpenFHE derives it: query x stored (level 0, degree 1 each), then one rescale.
  const core::CtMeta col_meta{0, 1, c.column_scale};
  const core::CtMeta sum_meta = core::GpuCtOps::multiplied(col_meta, col_meta);
  const core::CtMeta out_meta = c.ops->rescaled(sum_meta);

  std::vector<cache::BlockKey> order;
  for (const auto& it : items) order.push_back({c.spec.name, it.idx});
  e.gpu_cache->plan(order);

  std::vector<BlockScores> results;
  results.reserve(items.size());
  const std::size_t batch = std::max<std::size_t>(1, e.cfg.blocks_per_batch);
  for (std::size_t base = 0; base < items.size(); base += batch) {
    const std::size_t cnt = std::min(batch, items.size() - base);
    for (std::size_t k = 0; k < cnt; ++k) {
      cache::GpuClusterCache::Lease lease = e.gpu_cache->next(*e.compute);
      launcher.launch(lease.device_ptr(), query->get(), c.search_out->data() + k * 3 * pw,
                      e.compute->get());
      lease.release();  // records the last use on the compute stream
    }
    e.compute->synchronize();  // the search kernels (non-blocking stream) before the default stream

    std::vector<const u64*> in3(cnt), in2(cnt);
    std::vector<u64*> out2a(cnt), out2b(cnt);
    for (std::size_t k = 0; k < cnt; ++k) {
      in3[k] = c.search_out->data() + k * 3 * pw;
      out2a[k] = c.relin_out->data() + k * 2 * pw;
      in2[k] = out2a[k];
      out2b[k] = c.rescale_out->data() + k * 2 * (nq - 1) * n;
    }
    c.ops->device().relinearize(in3, out2a);
    c.ops->device().rescale(in2, out2b);
    c.ops->device().synchronize();

    for (std::size_t k = 0; k < cnt; ++k) {
      const Item& it = items[base + k];
      const std::size_t w = 2 * (nq - 1) * n;
      Ct ct = core::make_ciphertext_like(c.sample, 2, nq - 1, c.rescale_out->download(k * w, w),
                                         out_meta.level, out_meta.noise_scale_deg,
                                         out_meta.scaling_factor);
      BlockScores bs;
      bs.cluster = it.row.cluster;
      bs.segment = it.row.file;
      bs.ids = e.segment_ids(it.idx);
      if (!centers) {
        // Deleted vectors: a plaintext penalty at their slots, added at the ciphertext's level and
        // scale (no level used). Only blocks with at least one tombstone need it.
        const std::vector<double> pen =
            penalty_vector(bs.ids, tombstones, c.batch, c.layout.tombstone_penalty());
        if (!pen.empty()) {
          Plaintext pt = c.cc->MakeCKKSPackedPlaintext(pen, 1, 1);
          ct = c.cc->EvalAdd(ct, pt);
          bs.tombstone_penalty_applied = true;
        }
      }
      std::ostringstream out(std::ios::out | std::ios::binary);
      Serial::Serialize(ct, out, SerType::BINARY);
      const std::string s = out.str();
      core::Envelope env{core::EnvelopeKind::Ciphertext, c.spec.preset, e.store.openfhe_version(),
                         std::vector<std::uint8_t>(s.begin(), s.end())};
      bs.envelope = core::write_envelope(env);
      results.push_back(std::move(bs));
    }
  }
  log_line(centers ? "stage1" : "stage2", items.size(), ms_since(t0));
  return results;
}

}  // namespace ombra::search
