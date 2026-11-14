// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// The search engine of the CPU-only build (OMBRA_ENABLE_CUDA=OFF), for tests and development.
//
// Key upload and collection load run the same code as the GPU engine (key_prep.cpp): security
// floor, preset check, key tags, layout limits. Opening a search session needs the GPU and throws
// GpuUnavailable. Nothing here is used by the GPU build.
#include <map>
#include <mutex>
#include <set>

#include "coll_keys.hpp"
#include "ombra/search/session.hpp"

namespace ombra::search {

using namespace lbcrypto;

namespace detail {

struct Coll : CollKeys {};

struct EngineCore {
  storage::Store& store;
  EngineConfig cfg;
  std::mutex mu;  // guards colls
  std::map<std::string, std::shared_ptr<Coll>> colls;
  EngineCore(storage::Store& s, const EngineConfig& c) : store(s), cfg(c) {}
};

}  // namespace detail

using detail::Coll;
using detail::EngineCore;

struct SearchSession::Impl {};

SearchSession::SearchSession(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SearchSession::~SearchSession() = default;

namespace {
[[noreturn]] void no_gpu() {
  throw GpuUnavailable("this server was built without GPU support: it cannot run Search");
}
}  // namespace

const QueryLayout& SearchSession::layout() const { no_gpu(); }
std::vector<BlockScores> SearchSession::score_centers() { no_gpu(); }
std::vector<BlockScores> SearchSession::score_clusters(const std::vector<std::uint32_t>&) {
  no_gpu();
}

SearchEngine::SearchEngine(storage::Store& store, const EngineConfig& config)
    : impl_(std::make_unique<EngineCore>(store, config)) {}

SearchEngine::~SearchEngine() = default;

cache::CacheMetrics SearchEngine::cache_metrics() const { return {}; }

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

void SearchEngine::forget_collection(const std::string& collection) {
  EngineCore& e = *impl_;
  std::lock_guard g(e.mu);
  auto it = e.colls.find(collection);
  std::string tag;
  if (it != e.colls.end()) {
    tag = it->second->spec.key_tag;
    e.colls.erase(it);
  }
  // OpenFHE's key stores are process-wide by key tag: clear the tag unless another loaded
  // collection was made under the same key.
  if (!tag.empty()) {
    bool shared = false;
    for (const auto& kv : e.colls) shared = shared || kv.second->spec.key_tag == tag;
    if (!shared) {
      CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys(tag);
      CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys(tag);
    }
  }
}

void SearchEngine::install_keys(PreparedKeys&& prepared) {
  EngineCore& e = *impl_;
  if (!prepared.c_) throw SearchError("no prepared keys");
  std::lock_guard g(e.mu);
  e.colls[prepared.c_->spec.name] = std::move(prepared.c_);
}

PreparedKeys SearchEngine::prepare_keys(const std::string& name,
                                        const std::vector<std::uint8_t>& ctx_b,
                                        const std::vector<std::uint8_t>& mult_b,
                                        const std::vector<std::uint8_t>& auto_b) {
  EngineCore& e = *impl_;
  auto c = std::make_shared<Coll>();
  detail::prepare_collection_keys(*c, e.store, e.cfg, name, ctx_b, mult_b, auto_b);
  PreparedKeys out;
  out.c_ = std::move(c);
  return out;
}

std::unique_ptr<SearchSession> SearchEngine::open_session(const std::string&, const std::string&,
                                                          const std::vector<std::uint8_t>&,
                                                          std::uint32_t) {
  no_gpu();
}

}  // namespace ombra::search
