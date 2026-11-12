// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
#include "service.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>

#include "ombra/server/policy.hpp"

namespace ombra::server {

namespace {

using Clock = std::chrono::steady_clock;

// The policy limits must be the proto's Limit values.
static_assert(static_cast<std::size_t>(v1::LIMIT_MAX_CHUNK_BYTES) == kMaxChunkBytes);
static_assert(static_cast<std::size_t>(v1::LIMIT_MAX_NAME_BYTES) == kMaxNameBytes);
static_assert(static_cast<std::size_t>(v1::LIMIT_MAX_ID_BYTES) == kMaxIdBytes);
static_assert(static_cast<std::size_t>(v1::LIMIT_MAX_IDS_PER_REQUEST) == kMaxIdsPerRequest);
static_assert(static_cast<std::size_t>(v1::LIMIT_MAX_CLUSTERS_PER_ROUND) == kMaxClustersPerRound);
static_assert(static_cast<std::uint32_t>(v1::LIMIT_API_VERSION) == kApiVersion);
static_assert(static_cast<int>(v1::ERROR_UNSUPPORTED_VERSION) ==
              static_cast<int>(Err::UnsupportedVersion));
static_assert(static_cast<int>(v1::ERROR_COLLECTION_NOT_FOUND) ==
              static_cast<int>(Err::CollectionNotFound));
static_assert(static_cast<int>(v1::ERROR_COLLECTION_EXISTS) ==
              static_cast<int>(Err::CollectionExists));
static_assert(static_cast<int>(v1::ERROR_INVALID_ARGUMENT) ==
              static_cast<int>(Err::InvalidArgument));
static_assert(static_cast<int>(v1::ERROR_SIZE_LIMIT) == static_cast<int>(Err::SizeLimit));
static_assert(static_cast<int>(v1::ERROR_KEY_TAG_MISMATCH) ==
              static_cast<int>(Err::KeyTagMismatch));
static_assert(static_cast<int>(v1::ERROR_PRESET_MISMATCH) == static_cast<int>(Err::PresetMismatch));
static_assert(static_cast<int>(v1::ERROR_INVALID_FILTER_LAYOUT) ==
              static_cast<int>(Err::InvalidFilterLayout));
static_assert(static_cast<int>(v1::ERROR_KEYS_MISSING) == static_cast<int>(Err::KeysMissing));
static_assert(static_cast<int>(v1::ERROR_BAD_SEQUENCE) == static_cast<int>(Err::BadSequence));
static_assert(static_cast<int>(v1::ERROR_INTERNAL) == static_cast<int>(Err::Internal));
static_assert(static_cast<int>(v1::ERROR_BLOCK_NOT_FOUND) == static_cast<int>(Err::BlockNotFound));
static_assert(static_cast<int>(v1::ERROR_BLOCK_MISMATCH) == static_cast<int>(Err::BlockMismatch));
static_assert(static_cast<int>(v1::ERROR_BLOB_NOT_FOUND) == static_cast<int>(Err::BlobNotFound));
static_assert(static_cast<std::uint64_t>(v1::LIMIT_BLOB_BUCKET_0) == storage::kBlobBuckets[0]);
static_assert(static_cast<std::uint64_t>(v1::LIMIT_BLOB_BUCKET_1) == storage::kBlobBuckets[1]);
static_assert(static_cast<std::uint64_t>(v1::LIMIT_BLOB_BUCKET_2) == storage::kBlobBuckets[2]);
static_assert(static_cast<std::uint64_t>(v1::LIMIT_BLOB_BUCKET_3) == storage::kBlobBuckets[3]);
static_assert(static_cast<std::uint32_t>(v1::BLOB_KIND_RECORD) == storage::kBlobKindRecord);
static_assert(static_cast<std::uint32_t>(v1::BLOB_KIND_DOCUMENT) == storage::kBlobKindDocument);

constexpr std::size_t kMaxKeyFileBytes = std::size_t{1} << 30;  // per key kind
constexpr std::size_t kClustersPerGroup = 16;                   // clusters scored per write burst

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Logs: counts, sizes and timings only.
void log_line(const char* what, std::size_t a, std::size_t b, double ms) {
  std::fprintf(stderr, "[ombra.server] %s count=%zu bytes=%zu ms=%.2f\n", what, a, b, ms);
}

void fill(v1::Error* e, const Status& s) {
  e->set_code(static_cast<v1::ErrorCode>(static_cast<int>(s.code)));
  e->set_message(s.message);
}

Status err(Err c, const char* m) { return Status{c, m}; }

// Maps the exception being handled to a Status. Call only inside a catch block.
Status classify_current() {
  try {
    throw;
  } catch (const storage::BlockNotFound&) {
    return {Err::BlockNotFound, "block not found"};
  } catch (const storage::BlobSizeError&) {
    return {Err::SizeLimit, "blob size is not a bucket size"};
  } catch (const storage::BlobKindError&) {
    return {Err::InvalidArgument, "unknown blob kind"};
  } catch (const storage::BlockMismatch& e) {
    return {Err::BlockMismatch, e.what()};
  } catch (const search::KeyTagMismatch& e) {
    return {Err::KeyTagMismatch, e.what()};
  } catch (const search::KeysMissing& e) {
    return {Err::KeysMissing, e.what()};
  } catch (const search::CollectionNotFound& e) {
    return {Err::CollectionNotFound, e.what()};
  } catch (const search::InvalidQuery& e) {
    return {Err::InvalidArgument, e.what()};
  } catch (const search::SearchError& e) {
    std::fprintf(stderr, "[ombra.server] internal error: %s\n", e.what());
    return {Err::Internal, "internal error"};
  } catch (const std::exception& e) {
    const std::string w = e.what();
    const auto has = [&](const char* s) { return w.find(s) != std::string::npos; };
    if (has("unknown format version"))
      return {Err::UnsupportedVersion, "unsupported envelope version"};
    if (has("preset mismatch"))
      return {Err::PresetMismatch, "envelope preset differs from the collection's"};
    if (has("envelope:")) return {Err::InvalidArgument, "invalid envelope"};
    if (has("duplicate id")) return {Err::InvalidArgument, "duplicate id"};
    if (has("unknown collection")) return {Err::CollectionNotFound, "unknown collection"};
    std::fprintf(stderr, "[ombra.server] internal error: %s\n", e.what());
    return {Err::Internal, "internal error"};
  } catch (...) {
    return {Err::Internal, "internal error"};
  }
}

storage::CollectionSpec to_storage(const v1::CollectionSpec& p) {
  storage::CollectionSpec s;
  s.api_version = p.api_version();
  s.name = p.name();
  s.preset = p.preset();
  s.dimension = p.dimension();
  s.security_level = static_cast<std::uint32_t>(p.security_level());
  for (const auto& f : p.filter_fields())
    s.filter_fields.push_back({f.field_id(), f.cardinality()});
  s.key_tag = p.key_tag();
  s.cluster_count = p.cluster_count();
  return s;
}

void to_proto(const storage::CollectionSpec& s, v1::CollectionSpec* p) {
  p->set_api_version(s.api_version);
  p->set_name(s.name);
  p->set_preset(s.preset);
  p->set_dimension(s.dimension);
  p->set_security_level(static_cast<v1::SecurityLevel>(s.security_level));
  for (const auto& f : s.filter_fields) {
    auto* q = p->add_filter_fields();
    q->set_field_id(f.field_id);
    q->set_cardinality(f.cardinality);
  }
  p->set_key_tag(s.key_tag);
  p->set_cluster_count(s.cluster_count);
}

std::size_t sum_cardinality(const storage::CollectionSpec& s) {
  std::size_t n = 0;
  for (const auto& f : s.filter_fields) n += f.cardinality;
  return n;
}

const char* kind_name(int kind) {
  switch (kind) {
    case v1::KEY_KIND_CONTEXT:
      return storage::kKeyContext;
    case v1::KEY_KIND_EVAL_MULT:
      return storage::kKeyEvalMult;
    case v1::KEY_KIND_AUTOMORPHISM:
      return storage::kKeyAutomorphism;
    default:
      return nullptr;
  }
}

std::string to_str(const std::vector<std::uint8_t>& v) { return std::string(v.begin(), v.end()); }

}  // namespace

// ---------------------------------------------------------------------------------------------
// Engine key state

bool OmbraService::keys_present(const std::string& c) const {
  for (const char* k : {storage::kKeyContext, storage::kKeyEvalMult, storage::kKeyAutomorphism})
    if (!store_.get_key_file(c, k, nullptr)) return false;
  return true;
}

void OmbraService::ensure_loaded(const std::string& c) {
  std::lock_guard lk(load_mu_);
  if (loaded_.count(c)) return;
  engine_.load_collection(c);
  loaded_.insert(c);
}

void OmbraService::forget(const std::string& c) {
  // The engine keeps its (stale) entry for a dropped collection, but nothing reaches it again: a
  // search needs the name in loaded_, and a re-created collection is loaded afresh.
  std::lock_guard lk(load_mu_);
  loaded_.erase(c);
}

// ---------------------------------------------------------------------------------------------
// Collections

grpc::Status OmbraService::run_CreateCollection(grpc::ServerContext*,
                                                const v1::CreateCollectionRequest* req,
                                                v1::CreateCollectionResponse* resp) {
  const auto spec = to_storage(req->spec());
  Status st = validate_spec(spec);
  if (!st.ok()) {
    fill(resp->mutable_error(), st);
    return grpc::Status::OK;
  }
  try {
    if (store_.get_collection(spec.name, nullptr)) {
      fill(resp->mutable_error(), err(Err::CollectionExists, "collection exists"));
      return grpc::Status::OK;
    }
    store_.create_collection(spec);
    forget(spec.name);
    to_proto(spec, resp->mutable_spec());
    log_line("create_collection", spec.filter_fields.size(), 0, 0);
  } catch (...) {
    // A concurrent create of the same name loses at the catalog's primary key.
    if (store_.get_collection(spec.name, nullptr))
      fill(resp->mutable_error(), err(Err::CollectionExists, "collection exists"));
    else
      fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

static void fill_info(storage::Store& store, const storage::CollectionSpec& spec,
                      v1::CollectionInfo* info) {
  to_proto(spec, info->mutable_spec());
  std::uint64_t n = 0;
  for (const auto& r : store.segments(spec.name))
    if (r.cluster != spec.cluster_count) n += r.n_vectors;  // the center block is not a vector
  info->set_vector_count(n);  // includes tombstoned vectors until compaction
  info->set_deleted_count(store.tombstones(spec.name).size());
  bool keys = true;
  for (const char* k : {storage::kKeyContext, storage::kKeyEvalMult, storage::kKeyAutomorphism})
    keys = keys && store.get_key_file(spec.name, k, nullptr);
  info->set_keys_uploaded(keys);
}

grpc::Status OmbraService::run_GetCollection(grpc::ServerContext*,
                                             const v1::GetCollectionRequest* req,
                                             v1::GetCollectionResponse* resp) {
  try {
    storage::CollectionSpec spec;
    if (!store_.get_collection(req->name(), &spec))
      fill(resp->mutable_error(), err(Err::CollectionNotFound, "collection not found"));
    else
      fill_info(store_, spec, resp->mutable_info());
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_ListCollections(grpc::ServerContext*,
                                               const v1::ListCollectionsRequest*,
                                               v1::ListCollectionsResponse* resp) {
  try {
    for (const auto& n : store_.list_collections()) {
      storage::CollectionSpec spec;
      if (store_.get_collection(n, &spec)) fill_info(store_, spec, resp->add_collections());
    }
  } catch (...) {
    return grpc::Status(grpc::StatusCode::INTERNAL, "internal error");
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_DropCollection(grpc::ServerContext*,
                                              const v1::DropCollectionRequest* req,
                                              v1::DropCollectionResponse* resp) {
  try {
    if (!store_.drop_collection(req->name()))
      fill(resp->mutable_error(), err(Err::CollectionNotFound, "collection not found"));
    else
      forget(req->name());
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------------------------
// UploadKeys

// All three key kinds must arrive in one stream. They are collected in memory, checked as a unit
// against the collection (envelopes, key tag, security floor, layout: SearchEngine::prepare_keys),
// and only then stored (one catalog transaction) and installed. A refused upload leaves no key
// file, no catalog row and the collection's loaded keys as they were.
grpc::Status OmbraService::run_UploadKeys(grpc::ServerContext*,
                                          grpc::ServerReader<v1::UploadKeysChunk>* in,
                                          v1::UploadKeysResponse* resp) {
  const auto t0 = Clock::now();
  const auto fail = [&](Err c, const char* m) {
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  std::string coll;
  int cur_kind = 0;  // kind of the file being assembled, 0 if none
  std::vector<std::uint8_t> buf;
  std::map<std::string, std::vector<std::uint8_t>> staged;  // kind name -> complete file
  std::size_t chunks = 0, total = 0;
  try {
    v1::UploadKeysChunk c;
    while (in->Read(&c)) {
      ++chunks;
      if (c.api_version() != kApiVersion)
        return fail(Err::UnsupportedVersion, "unsupported api_version");
      if (chunks == 1) {
        if (c.collection().empty())
          return fail(Err::InvalidArgument, "collection is required on the first chunk");
        coll = c.collection();
        if (!store_.get_collection(coll, nullptr))
          return fail(Err::CollectionNotFound, "collection not found");
      } else if (!c.collection().empty() && c.collection() != coll) {
        return fail(Err::InvalidArgument, "collection changed within one upload");
      }
      const int kind = static_cast<int>(c.kind());
      if (!kind_name(kind)) return fail(Err::InvalidArgument, "unknown key kind");
      if (cur_kind != 0 && kind != cur_kind)
        return fail(Err::BadSequence, "key kind changed before last");
      if (cur_kind == 0) {
        if (c.offset() != 0)
          return fail(Err::BadSequence, "first chunk of a key file must have offset 0");
        if (staged.count(kind_name(kind)))
          return fail(Err::InvalidArgument, "key kind uploaded twice");
        cur_kind = kind;
        buf.clear();
      }
      if (c.offset() != buf.size())
        return fail(Err::BadSequence, "chunk offset does not follow the previous chunk");
      if (c.data().size() > kMaxChunkBytes)
        return fail(Err::SizeLimit, "chunk is larger than the limit");
      if (buf.size() + c.data().size() > kMaxKeyFileBytes)
        return fail(Err::SizeLimit, "key file is too large");
      buf.insert(buf.end(), c.data().begin(), c.data().end());
      total += c.data().size();
      if (c.last()) {
        staged[kind_name(cur_kind)] = std::move(buf);
        cur_kind = 0;
        buf.clear();
      }
    }
    if (chunks == 0) return fail(Err::BadSequence, "no chunks");
    if (cur_kind != 0)
      return fail(Err::BadSequence, "stream ended before the last chunk of a key file");
    for (const char* k : {storage::kKeyContext, storage::kKeyEvalMult, storage::kKeyAutomorphism})
      if (!staged.count(k))
        return fail(Err::InvalidArgument, "an upload must contain all three key kinds");

    std::lock_guard lk(load_mu_);
    search::PreparedKeys prepared =
        engine_.prepare_keys(coll, staged[storage::kKeyContext], staged[storage::kKeyEvalMult],
                             staged[storage::kKeyAutomorphism]);
    store_.put_key_files(
        coll, {{storage::kKeyContext, std::move(staged[storage::kKeyContext])},
               {storage::kKeyEvalMult, std::move(staged[storage::kKeyEvalMult])},
               {storage::kKeyAutomorphism, std::move(staged[storage::kKeyAutomorphism])}});
    engine_.install_keys(std::move(prepared));
    loaded_.insert(coll);
    log_line("upload_keys", staged.size(), total, ms_since(t0));
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------------------------
// Insert and Delete

grpc::Status OmbraService::run_Insert(grpc::ServerContext*, grpc::ServerReader<v1::InsertBlock>* in,
                                      v1::InsertResponse* resp) {
  const auto t0 = Clock::now();
  std::uint64_t inserted = 0;
  const auto fail = [&](Err c, const char* m) {
    resp->set_inserted(inserted);
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  std::string coll;
  storage::CollectionSpec spec;
  const PresetInfo* preset = nullptr;
  std::size_t blocks = 0, bytes = 0;
  try {
    v1::InsertBlock b;
    while (in->Read(&b)) {
      if (b.api_version() != kApiVersion)
        return fail(Err::UnsupportedVersion, "unsupported api_version");
      if (blocks == 0) {
        coll = b.collection();
        if (!store_.get_collection(coll, &spec))
          return fail(Err::CollectionNotFound, "collection not found");
        preset = find_preset(spec.preset);
        if (!preset) return fail(Err::Internal, "internal error");
        if (!keys_present(coll)) return fail(Err::KeysMissing, "evaluation keys are not uploaded");
      } else if (b.collection() != coll) {
        return fail(Err::InvalidArgument, "collection changed within one insert stream");
      }
      ++blocks;
      if (b.cluster_id() > spec.cluster_count)
        return fail(Err::InvalidArgument, "cluster id out of range");
      if (b.is_center() != (b.cluster_id() == spec.cluster_count))
        return fail(Err::InvalidArgument,
                    "is_center must be set exactly for the center cluster id");
      if (b.ids_size() == 0) return fail(Err::InvalidArgument, "block has no ids");
      if (static_cast<std::size_t>(b.ids_size()) > preset->batch_size)
        return fail(Err::SizeLimit, "block has more ids than slots");
      std::unordered_set<std::string> seen;
      for (const auto& id : b.ids()) {
        if (id.empty()) return fail(Err::InvalidArgument, "empty id");
        if (id.size() > kMaxIdBytes) return fail(Err::SizeLimit, "id is too long");
        if (!seen.insert(id).second) return fail(Err::InvalidArgument, "duplicate id");
      }
      const std::size_t want_cols = spec.dimension + (b.is_center() ? 0 : sum_cardinality(spec));
      if (static_cast<std::size_t>(b.columns_size()) != want_cols)
        return fail(Err::InvalidArgument, "wrong number of columns for the collection's layout");

      std::vector<std::string> ids(b.ids().begin(), b.ids().end());
      std::vector<std::vector<std::uint8_t>> cols;
      cols.reserve(want_cols);
      for (const auto& c : b.columns()) {
        cols.emplace_back(c.begin(), c.end());
        bytes += c.size();
      }
      store_.insert(coll, b.cluster_id(), ids, cols);
      inserted += ids.size();
    }
    if (blocks == 0) return fail(Err::BadSequence, "no blocks");
    resp->set_inserted(inserted);
    log_line("insert", inserted, bytes, ms_since(t0));
  } catch (...) {
    resp->set_inserted(inserted);
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_Delete(grpc::ServerContext*, const v1::DeleteRequest* req,
                                      v1::DeleteResponse* resp) {
  const auto t0 = Clock::now();
  const auto fail = [&](Err c, const char* m) {
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  if (req->api_version() != kApiVersion)
    return fail(Err::UnsupportedVersion, "unsupported api_version");
  if (static_cast<std::size_t>(req->ids_size()) > kMaxIdsPerRequest)
    return fail(Err::SizeLimit, "too many ids");
  try {
    if (!store_.get_collection(req->collection(), nullptr))
      return fail(Err::CollectionNotFound, "collection not found");
    std::set<std::string> uniq;
    for (const auto& id : req->ids()) {
      if (id.empty() || id.size() > kMaxIdBytes) return fail(Err::InvalidArgument, "invalid id");
      uniq.insert(id);
    }
    // Unknown, already deleted and cluster-center ids are ignored: a center is never tombstoned.
    std::vector<std::string> todo;
    for (const auto& id : uniq)
      if (store_.is_live_data_id(req->collection(), id)) todo.push_back(id);
    if (!todo.empty()) store_.remove(req->collection(), todo);
    resp->set_deleted(todo.size());
    log_line("delete", todo.size(), 0, ms_since(t0));
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------------------------
// Search

grpc::Status OmbraService::run_Search(
    grpc::ServerContext*, grpc::ServerReaderWriter<v1::SearchResponse, v1::SearchRequest>* stream) {
  // Sends an error message and ends the stream (status OK: the error is in the message).
  const auto send_error = [&](const Status& s) {
    v1::SearchResponse out;
    fill(out.mutable_error(), s);
    stream->Write(out);
    return grpc::Status::OK;
  };

  v1::SearchRequest req;
  if (!stream->Read(&req)) return grpc::Status::OK;  // client closed without a request
  if (req.step_case() != v1::SearchRequest::kStart)
    return send_error(err(Err::BadSequence, "the first message must be SearchStart"));

  std::unique_ptr<search::SearchSession> session;
  storage::CollectionSpec spec;
  const auto t_open = Clock::now();
  try {
    const auto& st = req.start();
    if (!store_.get_collection(st.collection(), &spec))
      return send_error(err(Err::CollectionNotFound, "collection not found"));
    if (st.key_tag() != spec.key_tag)
      return send_error(err(Err::KeyTagMismatch, "request key tag differs from the collection's"));
    if (!keys_present(st.collection()))
      return send_error(err(Err::KeysMissing, "evaluation keys are not uploaded"));
    if (st.filter_leaves() != sum_cardinality(spec))
      return send_error(
          err(Err::InvalidFilterLayout, "filter_leaves differs from sum(cardinality)"));
    if (st.query().empty()) return send_error(err(Err::InvalidArgument, "query is empty"));
    ensure_loaded(st.collection());
    const std::string& q = st.query();
    session =
        engine_.open_session(st.collection(), st.key_tag(),
                             std::vector<std::uint8_t>(q.begin(), q.end()), st.filter_leaves());
  } catch (...) {
    return send_error(classify_current());
  }

  // Round 1: stage 1 over the center blocks.
  try {
    auto centers = session->score_centers();
    v1::SearchResponse out;
    auto* c = out.mutable_centers();
    std::size_t n = 0, bytes = 0;
    for (const auto& b : centers) {
      c->add_score_blocks(to_str(b.envelope));
      n += b.ids.size();
      bytes += b.envelope.size();
    }
    c->set_center_count(static_cast<std::uint32_t>(n));
    if (!stream->Write(out)) return grpc::Status::OK;
    log_line("search_centers", centers.size(), bytes, ms_since(t_open));
  } catch (...) {
    return send_error(classify_current());
  }

  // Rounds 2..n: every SearchChosen is answered by one SearchScores per cluster.
  while (stream->Read(&req)) {
    if (req.step_case() != v1::SearchRequest::kChosen)
      return send_error(err(Err::BadSequence, "expected SearchChosen"));
    const auto t0 = Clock::now();
    try {
      const auto& ids = req.chosen().cluster_ids();
      if (ids.empty()) return send_error(err(Err::InvalidArgument, "no clusters chosen"));
      if (static_cast<std::size_t>(ids.size()) > kMaxClustersPerRound)
        return send_error(err(Err::SizeLimit, "too many clusters in one round"));
      std::vector<std::uint32_t> clusters;
      std::set<std::uint32_t> seen;
      for (auto id : ids) {
        if (id >= spec.cluster_count)
          return send_error(err(Err::InvalidArgument, "cluster id out of range"));
        if (seen.insert(id).second) clusters.push_back(id);  // each cluster once, order kept
      }
      std::size_t blocks = 0, bytes = 0;
      for (std::size_t g = 0; g < clusters.size(); g += kClustersPerGroup) {
        const std::vector<std::uint32_t> group(
            clusters.begin() + static_cast<std::ptrdiff_t>(g),
            clusters.begin() +
                static_cast<std::ptrdiff_t>(std::min(clusters.size(), g + kClustersPerGroup)));
        const auto results = session->score_clusters(group);
        for (std::size_t i = 0; i < group.size(); ++i) {
          v1::SearchResponse out;
          auto* s = out.mutable_scores();
          s->set_cluster_id(group[i]);
          for (const auto& b : results) {
            if (b.cluster != group[i]) continue;
            s->add_score_blocks(to_str(b.envelope));
            for (const auto& id : b.ids) s->add_ids(id);
            ++blocks;
            bytes += b.envelope.size();
          }
          s->set_last(g + i + 1 == clusters.size());
          if (!stream->Write(out)) return grpc::Status::OK;
        }
      }
      log_line("search_round", blocks, bytes, ms_since(t0));
    } catch (...) {
      return send_error(classify_current());
    }
  }
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------------------------
// Blocks

grpc::Status OmbraService::run_ListBlocks(grpc::ServerContext*, const v1::ListBlocksRequest* req,
                                          v1::ListBlocksResponse* resp) {
  resp->set_api_version(kApiVersion);
  try {
    if (req->api_version() != kApiVersion) {
      fill(resp->mutable_error(), err(Err::UnsupportedVersion, "unsupported api_version"));
    } else if (!store_.get_collection(req->collection(), nullptr)) {
      fill(resp->mutable_error(), err(Err::CollectionNotFound, "collection not found"));
    } else {
      for (const auto& b : store_.block_stats(req->collection())) {
        auto* o = resp->add_blocks();
        o->set_segment(b.segment);
        o->set_cluster_id(b.cluster);
        o->set_n_vectors(b.n_vectors);
        o->set_n_deleted(b.n_deleted);
      }
    }
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_GetBlock(grpc::ServerContext*, const v1::GetBlockRequest* req,
                                        grpc::ServerWriter<v1::GetBlockResponse>* out) {
  const auto t0 = Clock::now();
  const auto send_error = [&](const Status& s) {
    v1::GetBlockResponse r;
    r.set_api_version(kApiVersion);
    fill(r.mutable_error(), s);
    out->Write(r);
    return grpc::Status::OK;
  };
  if (req->api_version() != kApiVersion)
    return send_error(err(Err::UnsupportedVersion, "unsupported api_version"));
  try {
    if (!store_.get_collection(req->collection(), nullptr))
      return send_error(err(Err::CollectionNotFound, "collection not found"));
    // The pin keeps the file readable even if the block is replaced while it is sent.
    const storage::Snapshot snap = store_.snapshot(req->collection());
    bool found = false;
    for (const auto& r : snap.segments) found = found || r.file == req->segment();
    if (!found) return send_error(err(Err::BlockNotFound, "block not found"));
    const storage::SegmentData sd = store_.read_segment(req->collection(), req->segment());
    const std::set<std::string> dead(snap.tombstones.begin(), snap.tombstones.end());
    v1::GetBlockResponse head;
    head.set_api_version(kApiVersion);
    auto* h = head.mutable_header();
    h->set_cluster_id(sd.cluster);
    h->set_n_columns(static_cast<std::uint32_t>(sd.column_envelopes.size()));
    for (const auto& id : sd.ids) {
      h->add_ids(id);
      h->add_deleted(dead.count(id) > 0);
    }
    if (!out->Write(head)) return grpc::Status::OK;
    std::size_t bytes = 0;
    for (const auto& c : sd.column_envelopes) {
      v1::GetBlockResponse col;
      col.set_api_version(kApiVersion);
      col.set_column(to_str(c));
      bytes += c.size();
      if (!out->Write(col)) return grpc::Status::OK;
    }
    log_line("get_block", sd.column_envelopes.size(), bytes, ms_since(t0));
  } catch (...) {
    return send_error(classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_ReplaceBlock(grpc::ServerContext*,
                                            grpc::ServerReader<v1::ReplaceBlockRequest>* in,
                                            v1::ReplaceBlockResponse* resp) {
  const auto t0 = Clock::now();
  resp->set_api_version(kApiVersion);
  const auto fail = [&](Err c, const char* m) {
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  try {
    v1::ReplaceBlockRequest m;
    std::string coll;
    std::vector<std::string> segments;
    storage::SegmentData nd;
    bool have_header = false;
    std::size_t bytes = 0;
    while (in->Read(&m)) {
      if (m.api_version() != kApiVersion)
        return fail(Err::UnsupportedVersion, "unsupported api_version");
      if (m.part_case() == v1::ReplaceBlockRequest::kHeader) {
        if (have_header) return fail(Err::BadSequence, "second header");
        have_header = true;
        coll = m.header().collection();
        for (const auto& s : m.header().segments()) segments.push_back(s);
        if (segments.empty() || segments.size() > kMaxClustersPerRound)
          return fail(Err::InvalidArgument, "between 1 and 4096 blocks must be listed");
        for (const auto& id : m.header().ids()) {
          if (id.empty() || id.size() > kMaxIdBytes)
            return fail(Err::InvalidArgument, "invalid id");
          nd.ids.push_back(id);
        }
        if (nd.ids.size() > kMaxIdsPerRequest) return fail(Err::SizeLimit, "too many ids");
      } else if (m.part_case() == v1::ReplaceBlockRequest::kColumn) {
        if (!have_header) return fail(Err::BadSequence, "a column before the header");
        nd.column_envelopes.emplace_back(m.column().begin(), m.column().end());
        bytes += m.column().size();
      } else {
        return fail(Err::InvalidArgument, "empty message");
      }
    }
    if (!have_header) return fail(Err::BadSequence, "no header");
    storage::CollectionSpec spec;
    if (!store_.get_collection(coll, &spec))
      return fail(Err::CollectionNotFound, "collection not found");
    const PresetInfo* preset = find_preset(spec.preset);
    if (!preset) return fail(Err::Internal, "internal error");
    if (nd.ids.size() > preset->batch_size)
      return fail(Err::SizeLimit, "block has more ids than slots");
    // The cluster is the old blocks' (the store checks that they all share it), so take it from
    // the catalog row of the first one.
    bool found = false;
    for (const auto& r : store_.segments(coll))
      if (r.file == segments.front()) {
        nd.cluster = r.cluster;
        found = true;
      }
    if (!found) return fail(Err::BlockNotFound, "block not found");
    resp->set_segment(store_.replace_segments(coll, segments, nd));
    log_line("replace_block", nd.ids.size(), bytes, ms_since(t0));
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

// ---------------------------------------------------------------------------------------------
// Blob store. Blobs are opaque bytes; ids and bytes are never logged.

grpc::Status OmbraService::run_PutBlob(grpc::ServerContext*,
                                       grpc::ServerReader<v1::PutBlobRequest>* in,
                                       v1::PutBlobResponse* resp) {
  const auto t0 = Clock::now();
  resp->set_api_version(kApiVersion);
  const auto fail = [&](Err c, const char* m) {
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  try {
    v1::PutBlobRequest m;
    bool have_header = false;
    std::string coll, id;
    std::uint32_t kind = 0;
    std::uint64_t size = 0;
    std::vector<std::uint8_t> payload;
    while (in->Read(&m)) {
      if (m.api_version() != kApiVersion)
        return fail(Err::UnsupportedVersion, "unsupported api_version");
      if (m.part_case() == v1::PutBlobRequest::kHeader) {
        if (have_header) return fail(Err::BadSequence, "second header");
        have_header = true;
        const auto& h = m.header();
        coll = h.collection();
        id = h.id();
        kind = static_cast<std::uint32_t>(h.kind());
        size = h.size();
        if (coll.empty() || coll.size() > kMaxNameBytes)
          return fail(Err::InvalidArgument, "invalid collection name");
        if (id.empty() || id.size() > kMaxIdBytes) return fail(Err::InvalidArgument, "invalid id");
        if (!storage::is_blob_kind(kind)) return fail(Err::InvalidArgument, "unknown blob kind");
        if (!storage::is_blob_bucket(size))
          return fail(Err::SizeLimit, "blob size is not a bucket size");
        payload.reserve(static_cast<std::size_t>(size));
      } else if (m.part_case() == v1::PutBlobRequest::kData) {
        if (!have_header) return fail(Err::BadSequence, "data before the header");
        if (m.data().size() > kMaxChunkBytes) return fail(Err::SizeLimit, "chunk too large");
        if (payload.size() + m.data().size() > size)
          return fail(Err::InvalidArgument, "more data than the header's size");
        payload.insert(payload.end(), m.data().begin(), m.data().end());
      } else {
        return fail(Err::InvalidArgument, "empty message");
      }
    }
    if (!have_header) return fail(Err::BadSequence, "no header");
    if (payload.size() != size)
      return fail(Err::InvalidArgument, "less data than the header's size");
    if (!store_.get_collection(coll, nullptr))
      return fail(Err::CollectionNotFound, "collection not found");
    store_.put_blob(coll, kind, id, payload);
    log_line("put_blob", 1, payload.size(), ms_since(t0));
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_GetBlob(grpc::ServerContext*, const v1::GetBlobRequest* req,
                                       grpc::ServerWriter<v1::GetBlobResponse>* out) {
  const auto t0 = Clock::now();
  const auto send_error = [&](const Status& s) {
    v1::GetBlobResponse r;
    r.set_api_version(kApiVersion);
    fill(r.mutable_error(), s);
    out->Write(r);
    return grpc::Status::OK;
  };
  if (req->api_version() != kApiVersion)
    return send_error(err(Err::UnsupportedVersion, "unsupported api_version"));
  try {
    const auto kind = static_cast<std::uint32_t>(req->kind());
    if (!storage::is_blob_kind(kind))
      return send_error(err(Err::InvalidArgument, "unknown blob kind"));
    if (req->id().empty() || req->id().size() > kMaxIdBytes)
      return send_error(err(Err::InvalidArgument, "invalid id"));
    if (!store_.get_collection(req->collection(), nullptr))
      return send_error(err(Err::CollectionNotFound, "collection not found"));
    std::vector<std::uint8_t> bytes;
    if (!store_.get_blob(req->collection(), kind, req->id(), &bytes))
      return send_error(err(Err::BlobNotFound, "blob not found"));
    v1::GetBlobResponse head;
    head.set_api_version(kApiVersion);
    head.set_size(bytes.size());
    if (!out->Write(head)) return grpc::Status::OK;
    for (std::size_t pos = 0; pos < bytes.size(); pos += kMaxChunkBytes) {
      const std::size_t n = std::min(kMaxChunkBytes, bytes.size() - pos);
      v1::GetBlobResponse chunk;
      chunk.set_api_version(kApiVersion);
      chunk.set_data(reinterpret_cast<const char*>(bytes.data() + pos), n);
      if (!out->Write(chunk)) return grpc::Status::OK;
    }
    log_line("get_blob", 1, bytes.size(), ms_since(t0));
  } catch (...) {
    return send_error(classify_current());
  }
  return grpc::Status::OK;
}

grpc::Status OmbraService::run_DeleteBlobs(grpc::ServerContext*, const v1::DeleteBlobsRequest* req,
                                           v1::DeleteBlobsResponse* resp) {
  const auto t0 = Clock::now();
  const auto fail = [&](Err c, const char* m) {
    fill(resp->mutable_error(), err(c, m));
    return grpc::Status::OK;
  };
  try {
    if (req->api_version() != kApiVersion)
      return fail(Err::UnsupportedVersion, "unsupported api_version");
    const auto kind = static_cast<std::uint32_t>(req->kind());
    if (!storage::is_blob_kind(kind)) return fail(Err::InvalidArgument, "unknown blob kind");
    if (static_cast<std::size_t>(req->ids_size()) > kMaxIdsPerRequest)
      return fail(Err::SizeLimit, "too many ids");
    std::vector<std::string> ids;
    for (const auto& id : req->ids()) {
      if (id.empty() || id.size() > kMaxIdBytes) return fail(Err::InvalidArgument, "invalid id");
      ids.push_back(id);
    }
    if (!store_.get_collection(req->collection(), nullptr))
      return fail(Err::CollectionNotFound, "collection not found");
    resp->set_deleted(store_.delete_blobs(req->collection(), kind, ids));
    log_line("delete_blobs", ids.size(), 0, ms_since(t0));
  } catch (...) {
    fill(resp->mutable_error(), classify_current());
  }
  return grpc::Status::OK;
}

namespace {

std::string bearer_of(const grpc::ServerContext* ctx) {
  const auto it = ctx->client_metadata().find("authorization");
  if (it == ctx->client_metadata().end()) return {};
  return std::string(it->second.data(), it->second.size());
}

}  // namespace

template <class Resp, class Fn>
grpc::Status OmbraService::guarded(grpc::ServerContext* ctx, Rpc rpc, Resp* resp, Fn&& fn) {
  std::string key_id;
  const AuthResult a = authorize(keys_, bearer_of(ctx), required_scope(rpc), &key_id);
  if (a == AuthResult::Unauthenticated) {
    metrics_.count_unauthenticated();
    std::fprintf(stderr, "[ombra.server] refused rpc=%s reason=unauthenticated\n", rpc_name(rpc));
    return grpc::Status(grpc::StatusCode::UNAUTHENTICATED, "missing or invalid API key");
  }
  if (a == AuthResult::PermissionDenied) {
    metrics_.count_permission_denied();
    std::fprintf(stderr, "[ombra.server] refused rpc=%s reason=scope key=%s\n", rpc_name(rpc),
                 key_id.c_str());
    return grpc::Status(grpc::StatusCode::PERMISSION_DENIED, "the API key's scope is too low");
  }
  const auto t0 = Clock::now();
  grpc::Status st = fn();
  bool failed = !st.ok();
  if constexpr (requires(const Resp* r) { r->has_error(); }) failed = failed || resp->has_error();
  metrics_.record(rpc, failed, ms_since(t0));
  return st;
}

// ---------------------------------------------------------------------------------------------
// Public RPCs: API key check and metrics around the bodies above

grpc::Status OmbraService::CreateCollection(grpc::ServerContext* ctx,
                                            const v1::CreateCollectionRequest* req,
                                            v1::CreateCollectionResponse* resp) {
  return guarded(ctx, Rpc::CreateCollection, resp,
                 [&] { return run_CreateCollection(ctx, req, resp); });
}

grpc::Status OmbraService::GetCollection(grpc::ServerContext* ctx,
                                         const v1::GetCollectionRequest* req,
                                         v1::GetCollectionResponse* resp) {
  return guarded(ctx, Rpc::GetCollection, resp, [&] { return run_GetCollection(ctx, req, resp); });
}

grpc::Status OmbraService::ListCollections(grpc::ServerContext* ctx,
                                           const v1::ListCollectionsRequest* req,
                                           v1::ListCollectionsResponse* resp) {
  return guarded(ctx, Rpc::ListCollections, resp,
                 [&] { return run_ListCollections(ctx, req, resp); });
}

grpc::Status OmbraService::DropCollection(grpc::ServerContext* ctx,
                                          const v1::DropCollectionRequest* req,
                                          v1::DropCollectionResponse* resp) {
  return guarded(ctx, Rpc::DropCollection, resp,
                 [&] { return run_DropCollection(ctx, req, resp); });
}

grpc::Status OmbraService::UploadKeys(grpc::ServerContext* ctx,
                                      grpc::ServerReader<v1::UploadKeysChunk>* req,
                                      v1::UploadKeysResponse* resp) {
  return guarded(ctx, Rpc::UploadKeys, resp, [&] { return run_UploadKeys(ctx, req, resp); });
}

grpc::Status OmbraService::Insert(grpc::ServerContext* ctx,
                                  grpc::ServerReader<v1::InsertBlock>* req,
                                  v1::InsertResponse* resp) {
  return guarded(ctx, Rpc::Insert, resp, [&] { return run_Insert(ctx, req, resp); });
}

grpc::Status OmbraService::Delete(grpc::ServerContext* ctx, const v1::DeleteRequest* req,
                                  v1::DeleteResponse* resp) {
  return guarded(ctx, Rpc::Delete, resp, [&] { return run_Delete(ctx, req, resp); });
}

grpc::Status OmbraService::Search(
    grpc::ServerContext* ctx,
    grpc::ServerReaderWriter<v1::SearchResponse, v1::SearchRequest>* req) {
  return guarded(ctx, Rpc::Search, static_cast<void*>(nullptr),
                 [&] { return run_Search(ctx, req); });
}

grpc::Status OmbraService::ListBlocks(grpc::ServerContext* ctx, const v1::ListBlocksRequest* req,
                                      v1::ListBlocksResponse* resp) {
  return guarded(ctx, Rpc::ListBlocks, resp, [&] { return run_ListBlocks(ctx, req, resp); });
}

grpc::Status OmbraService::GetBlock(grpc::ServerContext* ctx, const v1::GetBlockRequest* req,
                                    grpc::ServerWriter<v1::GetBlockResponse>* resp) {
  return guarded(ctx, Rpc::GetBlock, resp, [&] { return run_GetBlock(ctx, req, resp); });
}

grpc::Status OmbraService::ReplaceBlock(grpc::ServerContext* ctx,
                                        grpc::ServerReader<v1::ReplaceBlockRequest>* req,
                                        v1::ReplaceBlockResponse* resp) {
  return guarded(ctx, Rpc::ReplaceBlock, resp, [&] { return run_ReplaceBlock(ctx, req, resp); });
}

grpc::Status OmbraService::PutBlob(grpc::ServerContext* ctx,
                                   grpc::ServerReader<v1::PutBlobRequest>* req,
                                   v1::PutBlobResponse* resp) {
  return guarded(ctx, Rpc::PutBlob, resp, [&] { return run_PutBlob(ctx, req, resp); });
}

grpc::Status OmbraService::GetBlob(grpc::ServerContext* ctx, const v1::GetBlobRequest* req,
                                   grpc::ServerWriter<v1::GetBlobResponse>* resp) {
  return guarded(ctx, Rpc::GetBlob, resp, [&] { return run_GetBlob(ctx, req, resp); });
}

grpc::Status OmbraService::DeleteBlobs(grpc::ServerContext* ctx, const v1::DeleteBlobsRequest* req,
                                       v1::DeleteBlobsResponse* resp) {
  return guarded(ctx, Rpc::DeleteBlobs, resp, [&] { return run_DeleteBlobs(ctx, req, resp); });
}

}  // namespace ombra::server
