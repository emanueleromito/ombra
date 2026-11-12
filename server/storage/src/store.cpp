#include "ombra/storage/store.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <map>
#include <set>
#include <stdexcept>

#include "ombra/core/file_envelope.hpp"
#include "ombra/storage/failpoint.hpp"
#include "ombra/storage/fsutil.hpp"

namespace ombra::storage {

namespace {

namespace core = ombra::core;

void make_dir(const std::string& p) {
  if (::mkdir(p.c_str(), 0755) != 0 && errno != EEXIST)
    throw std::runtime_error("cannot create directory " + p);
}

std::vector<std::string> list_dir(const std::string& p) {
  std::vector<std::string> v;
  DIR* d = ::opendir(p.c_str());
  if (!d) throw std::runtime_error("cannot read directory " + p);
  while (auto* e = ::readdir(d)) {
    std::string n = e->d_name;
    if (n != "." && n != "..") v.push_back(n);
  }
  ::closedir(d);
  return v;
}

bool ends_with(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

std::string hex_name(char prefix, std::uint64_t id, const char* ext) {
  char buf[40];
  std::snprintf(buf, sizeof buf, "%c%016llx%s", prefix, static_cast<unsigned long long>(id), ext);
  return buf;
}

// Logs carry counts and timings only: never ids, names of vectors or bytes.
void log_line(const char* what, std::size_t a, std::size_t b, double ms) {
  std::fprintf(stderr, "[ombra.storage] %s count=%zu bytes=%zu ms=%.2f\n", what, a, b, ms);
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Store::Store(std::string root, std::string openfhe_version)
    : root_(std::move(root)), ofhe_(std::move(openfhe_version)) {
  make_dir(root_);
  make_dir(segments_dir());
  make_dir(root_ + "/keys");
  make_dir(blob_dir());
  catalog_ = std::make_unique<Catalog>(root_ + "/catalog.db");
  wal_ = std::make_unique<Wal>(wal_path());
  recover();
}

std::string Store::segment_path(const std::string& name) const {
  return segments_dir() + "/" + name;
}

void Store::create_collection(const CollectionSpec& spec) {
  std::lock_guard lk(mu_);
  catalog_->create_collection(spec);
}

bool Store::get_collection(const std::string& name, CollectionSpec* out) const {
  std::lock_guard lk(mu_);
  return catalog_->get_collection(name, out);
}

std::vector<std::string> Store::list_collections() const {
  std::lock_guard lk(mu_);
  return catalog_->list_collections();
}

bool Store::is_live_data_id(const std::string& collection, const std::string& id) const {
  std::lock_guard lk(mu_);
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) return false;
  return catalog_->is_live_in_other_cluster(collection, id, spec.cluster_count);
}

bool Store::drop_collection(const std::string& name) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::string> segs, keys, blobs;
  maybe_crash(CrashPoint::DropBeforeCatalog);
  if (!catalog_->drop_collection(name, &segs, &keys, &blobs)) return false;
  maybe_crash(CrashPoint::DropAfterCatalog);
  wal_->reset();  // everything in the log is in the (durable) catalog or belongs to this collection
  maybe_crash(CrashPoint::DropAfterWalReset);
  bool first = true;
  for (const auto& f : segs) {
    ::unlink(segment_path(f).c_str());
    if (first) maybe_crash(CrashPoint::DropAfterFirstFile);
    first = false;
  }
  for (const auto& f : keys) ::unlink((root_ + "/keys/" + f).c_str());
  for (const auto& f : blobs) ::unlink((blob_dir() + "/" + f).c_str());
  fsync_path(blob_dir());
  fsync_path(segments_dir());
  fsync_path(root_ + "/keys");
  log_line("drop", segs.size() + keys.size() + blobs.size(), 0, ms_since(t0));
  return true;
}

void SegmentPin::release() {
  if (store_) store_->unpin(files_);
  store_ = nullptr;
  files_.clear();
}

void Store::unlink_or_defer(const std::string& file) {
  auto it = pins_.find(file);
  if (it != pins_.end() && it->second > 0)
    deferred_.insert(file);
  else
    ::unlink(segment_path(file).c_str());
}

void Store::unpin(const std::vector<std::string>& files) const {
  std::lock_guard lk(mu_);
  bool any = false;
  for (const auto& f : files) {
    auto it = pins_.find(f);
    if (it == pins_.end()) continue;
    if (--it->second > 0) continue;
    pins_.erase(it);
    if (deferred_.erase(f)) {
      ::unlink(segment_path(f).c_str());
      any = true;
    }
  }
  if (any) fsync_path(segments_dir());
}

Snapshot Store::snapshot(const std::string& collection) const {
  std::lock_guard lk(mu_);
  Snapshot s;
  s.segments = catalog_->segments(collection);
  s.tombstones = catalog_->tombstones(collection);
  std::vector<std::string> files;
  for (const auto& r : s.segments) {
    ++pins_[r.file];
    files.push_back(r.file);
  }
  s.pin = SegmentPin(this, std::move(files));
  return s;
}

std::vector<BlockStat> Store::block_stats(const std::string& collection) const {
  std::lock_guard lk(mu_);
  return catalog_->block_stats(collection);
}

std::size_t Store::drop_dead_blocks(const std::string& collection) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) return 0;
  std::vector<std::string> dead;
  for (const auto& b : catalog_->block_stats(collection))
    if (b.cluster != spec.cluster_count && b.n_vectors > 0 && b.n_deleted == b.n_vectors)
      dead.push_back(b.segment);
  if (dead.empty()) return 0;
  wal_->reset();  // see the header: nothing in the log is newer than the catalog
  maybe_crash(CrashPoint::CompactBeforeCatalog);
  catalog_->drop_segments(collection, dead);
  maybe_crash(CrashPoint::CompactAfterCatalog);
  for (const auto& f : dead) unlink_or_defer(f);
  fsync_path(segments_dir());
  maybe_crash(CrashPoint::CompactAfterUnlink);
  log_line("drop_dead_blocks", dead.size(), 0, ms_since(t0));
  return dead.size();
}

std::string Store::replace_segments(const std::string& collection,
                                    const std::vector<std::string>& old_segments,
                                    const SegmentData& neu) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) throw std::runtime_error("unknown collection");
  if (old_segments.empty()) throw BlockMismatch("no blocks to replace");
  if (std::set<std::string>(old_segments.begin(), old_segments.end()).size() != old_segments.size())
    throw BlockMismatch("a block is listed twice");
  std::vector<std::string> live;
  for (const auto& seg : old_segments) {
    SegmentRow old;
    if (!catalog_->segment_row(collection, seg, &old)) throw BlockNotFound("block not found");
    if (old.cluster == spec.cluster_count) throw BlockMismatch("center blocks cannot be replaced");
    if (neu.cluster != old.cluster) throw BlockMismatch("cluster differs from the old block's");
    if (neu.column_envelopes.size() != old.n_columns)
      throw BlockMismatch("column count differs from the old block's");
    auto l = catalog_->live_ids(collection, seg);
    live.insert(live.end(), l.begin(), l.end());
  }
  if (neu.ids.empty())
    throw BlockMismatch("no ids: a block with no live vector is dropped instead");
  if (std::set<std::string>(neu.ids.begin(), neu.ids.end()).size() != neu.ids.size() ||
      neu.ids.size() != live.size() ||
      std::set<std::string>(neu.ids.begin(), neu.ids.end()) !=
          std::set<std::string>(live.begin(), live.end()))
    throw BlockMismatch("ids are not exactly the live ids of the old blocks");
  std::size_t bytes = 0;
  for (const auto& c : neu.column_envelopes) {
    core::read_envelope(c, core::EnvelopeKind::Ciphertext, spec.preset, ofhe_);
    bytes += c.size();
  }

  wal_->reset();  // see the header: nothing in the log is newer than the catalog
  const std::string name = hex_name('s', catalog_->next_segment_id(), ".oseg");
  const std::string path = segment_path(name);
  maybe_crash(CrashPoint::ReplaceBeforeNewFile);
  try {
    SegmentInfo info = write_segment(path, neu);
    maybe_crash(CrashPoint::ReplaceAfterNewFile);
    catalog_->replace_segments(collection, old_segments,
                               {neu.cluster, name, info.n_vectors, info.n_columns, info.crc},
                               neu.ids);
  } catch (...) {
    if (!catalog_->has_segment(name)) {
      ::unlink(path.c_str());
      ::unlink((path + ".tmp").c_str());
    }
    throw;
  }
  maybe_crash(CrashPoint::ReplaceAfterCatalog);
  bool first = true;
  for (const auto& f : old_segments) {
    unlink_or_defer(f);
    if (first) maybe_crash(CrashPoint::ReplaceAfterFirstUnlink);  // some old files gone
    first = false;
  }
  fsync_path(segments_dir());
  maybe_crash(CrashPoint::ReplaceAfterUnlink);
  log_line("replace_block", neu.ids.size(), bytes, ms_since(t0));
  return name;
}

std::string Store::insert(const std::string& collection, std::uint32_t cluster,
                          const std::vector<std::string>& ids,
                          const std::vector<std::vector<std::uint8_t>>& cols) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) throw std::runtime_error("unknown collection");
  if (cluster > spec.cluster_count) throw std::runtime_error("cluster out of range");
  if (ids.empty()) throw std::runtime_error("insert: no ids");
  std::set<std::string> seen;
  for (const auto& id : ids)
    if (!seen.insert(id).second || catalog_->id_exists(collection, id))
      throw std::runtime_error("insert: duplicate id");
  std::size_t bytes = 0;
  for (const auto& c : cols) {
    core::read_envelope(c, core::EnvelopeKind::Ciphertext, spec.preset, ofhe_);
    bytes += c.size();
  }

  const std::string name = hex_name('s', catalog_->next_segment_id(), ".oseg");
  const std::string path = segment_path(name);
  WalRecord intent;
  intent.type = WalType::InsertIntent;
  intent.collection = collection;
  intent.cluster = cluster;
  intent.segment = name;
  intent.ids = ids;

  maybe_crash(CrashPoint::InsertBeforeIntent);
  wal_->append(intent);
  maybe_crash(CrashPoint::InsertAfterIntent);
  try {
    SegmentData sd{cluster, ids, cols};
    SegmentInfo info = write_segment(path, sd);  // rename crash points fire inside
    catalog_->add_segment(collection, {cluster, name, info.n_vectors, info.n_columns, info.crc},
                          ids);
  } catch (...) {
    // Not acknowledged: drop the file now; the dangling intent is discarded at next recovery.
    if (!catalog_->has_segment(name)) {
      ::unlink(path.c_str());
      ::unlink((path + ".tmp").c_str());
    }
    throw;
  }
  maybe_crash(CrashPoint::InsertAfterCatalog);
  WalRecord commit;
  commit.type = WalType::InsertCommit;
  commit.collection = collection;
  commit.segment = name;
  wal_->append(commit);
  maybe_crash(CrashPoint::InsertAfterCommit);
  log_line("insert", ids.size(), bytes, ms_since(t0));
  return name;
}

void Store::remove(const std::string& collection, const std::vector<std::string>& ids) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  if (!catalog_->get_collection(collection, nullptr))
    throw std::runtime_error("unknown collection");
  WalRecord rec;
  rec.type = WalType::Delete;
  rec.collection = collection;
  rec.ids = ids;
  maybe_crash(CrashPoint::DeleteBeforeRecord);
  wal_->append(rec);
  maybe_crash(CrashPoint::DeleteAfterRecord);
  catalog_->tombstone(collection, ids);
  maybe_crash(CrashPoint::DeleteAfterCatalog);
  log_line("delete", ids.size(), 0, ms_since(t0));
}

void Store::put_key_file(const std::string& collection, const std::string& kind,
                         const std::vector<std::uint8_t>& envelope) {
  std::lock_guard lk(mu_);
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) throw std::runtime_error("unknown collection");
  core::EnvelopeKind k;
  if (kind == kKeyContext)
    k = core::EnvelopeKind::Context;
  else if (kind == kKeyEvalMult || kind == kKeyAutomorphism)
    k = core::EnvelopeKind::PublicEvalKey;
  else
    throw std::runtime_error("unknown key file kind");
  core::read_envelope(envelope, k, spec.preset, ofhe_);  // also refuses unknown versions
  std::string old;
  for (auto& [kk, f] : catalog_->key_files(collection))
    if (kk == kind) old = f;
  const std::string file = hex_name('k', catalog_->next_segment_id(), ".env");
  write_file_durable(root_ + "/keys/" + file, envelope);
  catalog_->put_key_file(collection, kind, file);
  if (!old.empty()) ::unlink((root_ + "/keys/" + old).c_str());
}

void Store::put_key_files(
    const std::string& collection,
    const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& files) {
  std::lock_guard lk(mu_);
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) throw std::runtime_error("unknown collection");
  for (const auto& [kind, envelope] : files) {
    core::EnvelopeKind k;
    if (kind == kKeyContext)
      k = core::EnvelopeKind::Context;
    else if (kind == kKeyEvalMult || kind == kKeyAutomorphism)
      k = core::EnvelopeKind::PublicEvalKey;
    else
      throw std::runtime_error("unknown key file kind");
    core::read_envelope(envelope, k, spec.preset, ofhe_);
  }
  std::map<std::string, std::string> old;
  for (auto& [kk, f] : catalog_->key_files(collection)) old[kk] = f;
  std::vector<std::pair<std::string, std::string>> rows;
  std::vector<std::string> written;
  try {
    for (const auto& [kind, envelope] : files) {
      const std::string file = hex_name('k', catalog_->next_segment_id(), ".env");
      write_file_durable(key_dir() + "/" + file, envelope);  // temp + fsync + rename
      written.push_back(file);
      rows.emplace_back(kind, file);
    }
    catalog_->put_key_files(collection, rows);
  } catch (...) {
    for (const auto& f : written) {
      ::unlink((key_dir() + "/" + f).c_str());
      ::unlink((key_dir() + "/" + f + ".tmp").c_str());
    }
    throw;
  }
  for (const auto& [kind, file] : rows) {
    auto it = old.find(kind);
    if (it != old.end()) ::unlink((key_dir() + "/" + it->second).c_str());
  }
}

void Store::put_blob(const std::string& collection, std::uint32_t kind, const std::string& id,
                     const std::vector<std::uint8_t>& payload) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  if (!is_blob_kind(kind)) throw BlobKindError("unknown blob kind");
  if (!is_blob_bucket(payload.size())) throw BlobSizeError("blob size is not a bucket size");
  if (!catalog_->get_collection(collection, nullptr))
    throw std::runtime_error("unknown collection");
  const std::string file = hex_name('b', catalog_->next_segment_id(), ".oblob");
  const std::string path = blob_dir() + "/" + file;
  std::string old;
  bool had = false;
  try {
    write_blob_file(path, kind, payload);  // BlobBeforeRename / BlobAfterRename fire inside
    had = catalog_->put_blob(collection, kind, id, file, payload.size(), &old);
  } catch (...) {
    std::string f;
    if (!catalog_->get_blob(collection, kind, id, &f, nullptr) || f != file) {
      ::unlink(path.c_str());
      ::unlink((path + ".tmp").c_str());
    }
    throw;
  }
  maybe_crash(CrashPoint::BlobAfterCatalog);
  if (had) ::unlink((blob_dir() + "/" + old).c_str());
  fsync_path(blob_dir());
  log_line("put_blob", 1, payload.size(), ms_since(t0));
}

bool Store::get_blob(const std::string& collection, std::uint32_t kind, const std::string& id,
                     std::vector<std::uint8_t>* out) const {
  std::lock_guard lk(mu_);
  std::string file;
  if (!catalog_->get_blob(collection, kind, id, &file, nullptr)) return false;
  auto bytes = read_blob_file(blob_dir() + "/" + file, kind);
  if (out) *out = std::move(bytes);
  return true;
}

std::size_t Store::delete_blobs(const std::string& collection, std::uint32_t kind,
                                const std::vector<std::string>& ids) {
  std::lock_guard lk(mu_);
  const auto t0 = std::chrono::steady_clock::now();
  if (!is_blob_kind(kind)) throw BlobKindError("unknown blob kind");
  if (!catalog_->get_collection(collection, nullptr))
    throw std::runtime_error("unknown collection");
  const auto files = catalog_->delete_blobs(collection, kind, ids);
  maybe_crash(CrashPoint::BlobDeleteAfterCatalog);
  for (const auto& f : files) ::unlink((blob_dir() + "/" + f).c_str());
  fsync_path(blob_dir());
  log_line("delete_blobs", files.size(), 0, ms_since(t0));
  return files.size();
}

bool Store::get_key_file(const std::string& collection, const std::string& kind,
                         std::vector<std::uint8_t>* out) const {
  std::lock_guard lk(mu_);
  for (auto& [kk, f] : catalog_->key_files(collection)) {
    if (kk != kind) continue;
    if (out) *out = read_file(root_ + "/keys/" + f);
    return true;
  }
  return false;
}

std::vector<SegmentRow> Store::segments(const std::string& collection) const {
  std::lock_guard lk(mu_);
  return catalog_->segments(collection);
}

std::vector<std::string> Store::tombstones(const std::string& collection) const {
  std::lock_guard lk(mu_);
  return catalog_->tombstones(collection);
}

SegmentData Store::read_segment(const std::string& collection, const std::string& segment) const {
  std::lock_guard lk(mu_);
  CollectionSpec spec;
  if (!catalog_->get_collection(collection, &spec)) throw std::runtime_error("unknown collection");
  return ombra::storage::read_segment(segment_path(segment), spec.preset, ofhe_);
}

void Store::recover() {
  const auto t0 = std::chrono::steady_clock::now();
  std::size_t dropped = 0;
  auto recs = wal_->recover(&dropped);
  stats_.records = recs.size();
  stats_.torn_bytes_dropped = dropped;

  std::set<std::string> committed;
  for (const auto& r : recs)
    if (r.type == WalType::InsertCommit) committed.insert(r.segment);

  for (const auto& r : recs) {
    if (r.type == WalType::Delete) {
      if (catalog_->get_collection(r.collection, nullptr)) catalog_->tombstone(r.collection, r.ids);
      ++stats_.deletes_applied;
      continue;
    }
    if (r.type != WalType::InsertIntent) continue;
    const bool in_catalog = catalog_->has_segment(r.segment);
    const bool is_committed = committed.count(r.segment) > 0;
    const std::string path = segment_path(r.segment);
    CollectionSpec spec;
    if (!in_catalog && is_committed && catalog_->get_collection(r.collection, &spec)) {
      // Committed means the segment was durable; finish the catalog update (idempotent).
      SegmentInfo info;
      auto sd = ombra::storage::read_segment(path, spec.preset, ofhe_, &info);
      catalog_->add_segment(
          r.collection, {sd.cluster, r.segment, info.n_vectors, info.n_columns, info.crc}, sd.ids);
      ++stats_.rolled_forward;
    } else if (in_catalog && !is_committed) {
      // The catalog already has it, so the whole write is there: complete it by logging the commit.
      WalRecord c;
      c.type = WalType::InsertCommit;
      c.collection = r.collection;
      c.segment = r.segment;
      wal_->append(c);
      ++stats_.rolled_forward;
    } else if (!in_catalog && !is_committed) {
      // Never acknowledged and not applied: discard, together with any orphan file.
      if (::unlink(path.c_str()) == 0) ++stats_.orphans_removed;
      ::unlink((path + ".tmp").c_str());
      ++stats_.discarded;
    }
  }

  // Sweep: temp files and any segment file the catalog does not know are leftovers of aborted
  // writes.
  auto files = catalog_->all_segment_files();
  std::set<std::string> known(files.begin(), files.end());
  for (const auto& n : list_dir(segments_dir())) {
    if (ends_with(n, ".tmp") || !known.count(n)) {
      if (::unlink(segment_path(n).c_str()) == 0) ++stats_.orphans_removed;
    }
  }
  // Key files: temp files and files no catalog row points to (aborted upload, or a dropped
  // collection whose files were not yet unlinked).
  {
    auto kf = catalog_->all_key_files();
    std::set<std::string> known_keys(kf.begin(), kf.end());
    for (const auto& n : list_dir(root_ + "/keys"))
      if (ends_with(n, ".tmp") || !known_keys.count(n)) {
        if (::unlink((root_ + "/keys/" + n).c_str()) == 0) ++stats_.orphans_removed;
      }
  }

  // Blob files: temp files and files no catalog row points to (aborted put, replaced or deleted
  // blob whose old file was not yet unlinked, dropped collection).
  {
    auto bf = catalog_->all_blob_files();
    std::set<std::string> known_blobs(bf.begin(), bf.end());
    for (const auto& n : list_dir(blob_dir()))
      if (ends_with(n, ".tmp") || !known_blobs.count(n)) {
        if (::unlink((blob_dir() + "/" + n).c_str()) == 0) ++stats_.orphans_removed;
      }
  }

  // Everything in the log is now in the catalog, which is durable (synchronous=FULL): checkpoint.
  wal_->reset();
  log_line("recovery", stats_.records, stats_.torn_bytes_dropped, ms_since(t0));
}

}  // namespace ombra::storage
