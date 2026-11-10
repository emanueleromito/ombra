// Search engine API (T2.7a), what the gRPC service (part b) calls. Needs OpenFHE and CUDA.
//
//   SearchEngine engine(store, config);        // GPU cluster cache, one compute stream
//   engine.load_collection("docs");            // context + EvalMult + automorphism keys from the store
//   auto s = engine.open_session("docs", key_tag, query_envelope, filter_leaves);
//   auto centers = s->score_centers();         // round 1: stage 1, first d leaves, per center block
//   auto scores  = s->score_clusters({3, 7});  // round 2 (callable again: more rounds, same stream)
//
// The query is expanded ONCE on the GPU into D = d + sum(cardinality) leaves that stay on the device
// for the life of the session. Stage 1 multiplies the d first leaves with the center block columns;
// stage 2 multiplies all D leaves with the d + sum(V) columns of each block of a chosen cluster,
// then relinearizes and rescales on the GPU (existing pipeline, exact against OpenFHE). A result
// ciphertext is at level 1; slot i decrypts to score + lambda * c (decision 0019, lambda = 4, c =
// number of filter fields that vector matches). Tombstoned slots get -lambda * (F + 1) added with
// an OpenFHE plaintext (no level), so they decrypt below every live slot. No mask multiply.
//
// Key handling (hard rule 2): the engine reads the public context and the public EvalMult and
// automorphism keys. It never accepts or derives a secret key. Every request carries a key tag that
// is checked against the collection's, and so is the key tag inside every ciphertext (query and
// stored columns); a mismatch throws KeyTagMismatch.
//
// Threading: GPU work of all sessions is serialised by one mutex inside the engine. Logs carry
// counts and timings only.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "ombra/cache/cache_metrics.hpp"
#include "ombra/search/layout.hpp"
#include "ombra/storage/store.hpp"

namespace ombra::search {

namespace detail {
struct EngineCore;  // defined in session.cpp
struct Coll;
}

// Error classes the service maps to proto ErrorCode.
class SearchError : public std::runtime_error {  // ERROR_INTERNAL unless a subclass says otherwise
 public:
  using std::runtime_error::runtime_error;
};
class KeyTagMismatch : public SearchError {  // ERROR_KEY_TAG_MISMATCH
 public:
  using SearchError::SearchError;
};
class KeysMissing : public SearchError {  // ERROR_KEYS_MISSING
 public:
  using SearchError::SearchError;
};
class InvalidQuery : public SearchError {  // ERROR_INVALID_ARGUMENT / ERROR_INVALID_FILTER_LAYOUT
 public:
  using SearchError::SearchError;
};
class CollectionNotFound : public SearchError {  // ERROR_COLLECTION_NOT_FOUND
 public:
  using SearchError::SearchError;
};

struct EngineConfig {
  std::size_t gpu_budget_bytes = 0;   // cluster cache on the GPU
  std::size_t host_budget_bytes = 0;  // pinned pool
  std::size_t lookahead = 1;          // blocks copied ahead
  std::size_t blocks_per_batch = 8;   // blocks searched before one relinearize/rescale launch
};

// The result for one stored block (one segment).
struct BlockScores {
  std::uint32_t cluster = 0;
  std::string segment;                   // segment file name in the store
  std::vector<std::uint8_t> envelope;    // decision-0016 envelope, kind ciphertext (level 1)
  std::vector<std::string> ids;          // one per occupied slot, in slot order, tombstoned included
  bool tombstone_penalty_applied = false;
};

class SearchSession {
 public:
  ~SearchSession();
  SearchSession(const SearchSession&) = delete;
  SearchSession& operator=(const SearchSession&) = delete;

  const QueryLayout& layout() const;

  // Stage 1: one result per block of the center cluster (cluster id == spec.cluster_count), scored
  // with the first d leaves only. No tombstone penalty (centers are not vectors).
  std::vector<BlockScores> score_centers();
  // Stage 2: one result per block of every listed cluster, in the order given (and within a
  // cluster, in storage order). Callable any number of times on the same session. Throws
  // InvalidQuery for a cluster id >= spec.cluster_count.
  std::vector<BlockScores> score_clusters(const std::vector<std::uint32_t>& clusters);

 private:
  friend class SearchEngine;
  struct Impl;
  explicit SearchSession(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

// Keys of one collection that were read and checked (key tag, security floor, layout) but are not
// installed in the engine yet. Lets the service validate an upload before anything is stored.
class PreparedKeys {
 public:
  PreparedKeys() = default;

 private:
  friend class SearchEngine;
  std::shared_ptr<detail::Coll> c_;
};

class SearchEngine {
 public:
  SearchEngine(storage::Store& store, const EngineConfig& config);
  ~SearchEngine();
  SearchEngine(const SearchEngine&) = delete;
  SearchEngine& operator=(const SearchEngine&) = delete;

  // Reads the context and the EvalMult and automorphism key files of the collection from the store
  // and loads them into OpenFHE. Calling it again (after a key re-upload) replaces the keys; open
  // sessions keep the ones they started with. Throws CollectionNotFound, KeysMissing (a key file is
  // absent), KeyTagMismatch (a key file was made under another key than spec.key_tag),
  // InvalidQuery (layout above the preset's limits).
  void load_collection(const std::string& collection);
  // The same checks on key files held in memory, without touching the store or the engine's
  // collections (a refused upload leaves the loaded keys of the collection as they were). Throws
  // what load_collection throws. install_keys() then makes the result the collection's keys.
  PreparedKeys prepare_keys(const std::string& collection, const std::vector<std::uint8_t>& context,
                            const std::vector<std::uint8_t>& eval_mult,
                            const std::vector<std::uint8_t>& automorphism);
  void install_keys(PreparedKeys&& prepared);

  // Opens a session for one search stream. `key_tag` is the request's tag; `query_envelope` is a
  // 0016 envelope (kind ciphertext) holding the packed query of D = d + sum(V) entries;
  // `filter_leaves` is the request's sum(V). Throws KeyTagMismatch if the request tag or the tag
  // inside the ciphertext differs from the collection's, InvalidQuery for a bad envelope, shape,
  // level or filter_leaves, KeysMissing if load_collection was not called.
  std::unique_ptr<SearchSession> open_session(const std::string& collection,
                                              const std::string& key_tag,
                                              const std::vector<std::uint8_t>& query_envelope,
                                              std::uint32_t filter_leaves);

  cache::CacheMetrics cache_metrics() const;

 private:
  std::unique_ptr<detail::EngineCore> impl_;
};

}  // namespace ombra::search
