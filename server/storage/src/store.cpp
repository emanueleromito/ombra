#include "ombra/storage/store.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
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

// Logs carry counts and timings only: never ids, names of vectors or bytes (hard rule 4).
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
  catalog_ = std::make_unique<Catalog>(root_ + "/catalog.db");
  wal_ = std::make_unique<Wal>(wal_path());
  recover();
}

std::string Store::segment_path(const std::string& name) const { return segments_dir() + "/" + name; }

void Store::create_collection(const CollectionSpec& spec) {
  std::lock_guard lk(mu_);
  catalog_->create_collection(spec);
}

bool Store::get_collection(const std::string& name, CollectionSpec* out) const {
  std::lock_guard lk(mu_);
  return catalog_->get_collection(name, out);
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
    catalog_->add_segment(collection, {cluster, name, info.n_vectors, info.n_columns, info.crc}, ids);
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
  if (!catalog_->get_collection(collection, nullptr)) throw std::runtime_error("unknown collection");
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
  if (kind == kKeyContext) k = core::EnvelopeKind::Context;
  else if (kind == kKeyEvalMult || kind == kKeyAutomorphism) k = core::EnvelopeKind::PublicEvalKey;
  else throw std::runtime_error("unknown key file kind");
  core::read_envelope(envelope, k, spec.preset, ofhe_);  // also refuses unknown versions
  std::string old;
  for (auto& [kk, f] : catalog_->key_files(collection))
    if (kk == kind) old = f;
  const std::string file = hex_name('k', catalog_->next_segment_id(), ".env");
  write_file_durable(root_ + "/keys/" + file, envelope);
  catalog_->put_key_file(collection, kind, file);
  if (!old.empty()) ::unlink((root_ + "/keys/" + old).c_str());
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
      catalog_->add_segment(r.collection, {sd.cluster, r.segment, info.n_vectors, info.n_columns, info.crc},
                            sd.ids);
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

  // Sweep: temp files and any segment file the catalog does not know are leftovers of aborted writes.
  auto files = catalog_->all_segment_files();
  std::set<std::string> known(files.begin(), files.end());
  for (const auto& n : list_dir(segments_dir())) {
    if (ends_with(n, ".tmp") || !known.count(n)) {
      if (::unlink(segment_path(n).c_str()) == 0) ++stats_.orphans_removed;
    }
  }
  for (const auto& n : list_dir(root_ + "/keys"))
    if (ends_with(n, ".tmp")) ::unlink((root_ + "/keys/" + n).c_str());

  // Everything in the log is now in the catalog, which is durable (synchronous=FULL): checkpoint.
  wal_->reset();
  log_line("recovery", stats_.records, stats_.torn_bytes_dropped, ms_since(t0));
}

}  // namespace ombra::storage
