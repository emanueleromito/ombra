// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 OR LicenseRef-Ombra-Commercial
// Storage engine: segments + write-ahead log + catalog, with crash recovery.
//
// Layout under root: catalog.db, wal.log, segments/<name>.oseg, keys/<name>, blobs/<name>.oblob.
//
// Guarantee: an acknowledged write (insert() or remove() returned) survives a crash or power
// loss. A write that was not acknowledged is either fully there (segment, all its ids, catalog
// rows) or fully absent (no rows, no segment file); never partial. Single writer: methods are
// serialised by a mutex. Logs carry sizes, counts and timings only.
//
// Searches vs. block removal: a search takes snapshot() once per round. It sees the block list and
// the tombstones of one instant and pins those segment files. drop_dead_blocks() and
// replace_segments() change the catalog under the same mutex, so a snapshot is either wholly before
// or wholly after; the files they remove are unlinked only when no pin holds them (deferred_), so
// a running search can still read every block of its snapshot. Segment names are never reused
// (durable counter), so a cache keyed by segment name never serves a stale block.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ombra/storage/blob.hpp"
#include "ombra/storage/catalog.hpp"
#include "ombra/storage/segment.hpp"
#include "ombra/storage/wal.hpp"

namespace ombra::storage {

struct RecoveryStats {
  std::size_t records = 0, rolled_forward = 0, discarded = 0, deletes_applied = 0,
              orphans_removed = 0, torn_bytes_dropped = 0;
};

// Typed refusals of replace_segments / pins (the service maps them to proto error codes).
class BlockNotFound : public std::runtime_error {  // the collection has no such segment
 public:
  using std::runtime_error::runtime_error;
};
class BlockMismatch : public std::runtime_error {  // replacement does not satisfy the precondition
 public:
  using std::runtime_error::runtime_error;
};

class BlobSizeError : public std::runtime_error {  // blob size is not a bucket size
 public:
  using std::runtime_error::runtime_error;
};
class BlobKindError : public std::runtime_error {  // blob kind is not a known kind
 public:
  using std::runtime_error::runtime_error;
};

class Store;

// Keeps a set of segment files on disk while alive. A segment removed meanwhile (drop_dead_blocks,
// replace_segments) is unlinked when the last pin on it is released. Move-only RAII.
class SegmentPin {
 public:
  SegmentPin() = default;
  SegmentPin(SegmentPin&& o) noexcept : store_(o.store_), files_(std::move(o.files_)) {
    o.store_ = nullptr;
  }
  SegmentPin& operator=(SegmentPin&& o) noexcept {
    if (this != &o) {
      release();
      store_ = o.store_;
      files_ = std::move(o.files_);
      o.store_ = nullptr;
    }
    return *this;
  }
  SegmentPin(const SegmentPin&) = delete;
  SegmentPin& operator=(const SegmentPin&) = delete;
  ~SegmentPin() { release(); }

 private:
  friend class Store;
  SegmentPin(const Store* s, std::vector<std::string> f) : store_(s), files_(std::move(f)) {}
  void release();
  const Store* store_ = nullptr;
  std::vector<std::string> files_;
};

// A consistent view for one search round: the block list and the tombstones as of the same
// instant, with the segment files pinned for as long as the view lives.
struct Snapshot {
  std::vector<SegmentRow> segments;
  std::vector<std::string> tombstones;
  SegmentPin pin;
};

// Key file kinds of put_key_file (proto KeyKind). The context is an envelope of kind Context; the
// EvalMult and automorphism keys are envelopes of kind PublicEvalKey. There is no secret-key kind.
inline constexpr const char* kKeyContext = "context";
inline constexpr const char* kKeyEvalMult = "eval_mult";
inline constexpr const char* kKeyAutomorphism = "automorphism";

class Store {
 public:
  // Opens (creating if empty) and runs recovery. openfhe_version is what column envelopes must
  // carry; it is checked when segments are read.
  Store(std::string root, std::string openfhe_version);

  // Optional: resolves a preset name to its ring dimension and slot count, so that insert can
  // compare a seeded column's header with the collection (audit A-71). Set before serving.
  using ShapeProvider = std::function<bool(const std::string& preset, std::uint32_t* ring_dim,
                                           std::uint32_t* slots)>;
  void set_shape_provider(ShapeProvider p) { shape_ = std::move(p); }

  void create_collection(const CollectionSpec& spec);
  bool get_collection(const std::string& name, CollectionSpec* out) const;

  // Appends one cluster block as a new immutable segment. cluster == spec.cluster_count is the
  // reserved id of the cluster-center blocks (d columns, no filter columns). Returns the segment
  // name once the write is acknowledged (durable). Throws on duplicate or tombstoned ids, unknown
  // collection or cluster, or a column envelope that does not match the collection's preset.
  std::string insert(const std::string& collection, std::uint32_t cluster,
                     const std::vector<std::string>& ids,
                     const std::vector<std::vector<std::uint8_t>>& column_envelopes);
  // Names of all collections, sorted.
  std::vector<std::string> list_collections() const;
  // Removes a collection with all its rows, segment files and key files. Returns false if it does
  // not exist. Order (crash safe): (1) one catalog transaction removes every row; that commit is
  // the point of no return. (2) The write-ahead log is checkpointed, so no old record can touch a
  // later collection of the same name. (3) The segment and key files are unlinked. A crash after
  // (1) leaves only unreferenced files, which recovery sweeps; a crash before (1) leaves the
  // collection untouched. Same-name re-creation is therefore safe at any point.
  bool drop_collection(const std::string& name);
  // True if `id` exists in a data cluster (not the center cluster spec.cluster_count) and is not
  // tombstoned yet. Used by the server to filter a delete request.
  bool is_live_data_id(const std::string& collection, const std::string& id) const;
  // Marks ids deleted (unknown ids ignored). Acknowledged on return.
  void remove(const std::string& collection, const std::vector<std::string>& ids);

  // Stores a public key / context envelope file and registers it. kind: kKeyContext,
  // kKeyEvalMult or kKeyAutomorphism.
  void put_key_file(const std::string& collection, const std::string& kind,
                    const std::vector<std::uint8_t>& envelope);
  // Stores several key files as one unit: every envelope is checked first, then all files are
  // written (temp, fsync, rename) and registered in ONE catalog transaction. If anything fails, no
  // file and no catalog row is left behind (temp and unregistered files are removed).
  void put_key_files(const std::string& collection,
                     const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& files);

  // The stored envelope of a key kind (false if none was uploaded).
  bool get_key_file(const std::string& collection, const std::string& kind,
                    std::vector<std::uint8_t>* out) const;
  // Locked snapshots for the search path (the catalog() accessor is not serialised with writers).
  std::vector<SegmentRow> segments(const std::string& collection) const;
  std::vector<std::string> tombstones(const std::string& collection) const;
  // Consistent view for a search (see "Blocks" below).
  Snapshot snapshot(const std::string& collection) const;

  // ---- Blocks: dead-block drop, replacement, metrics -------------------------------------------
  // Counts per block, no ids: segment, cluster, n_vectors (tombstoned included), n_deleted. The
  // center blocks (cluster == cluster_count) are listed too and never have deletes.
  std::vector<BlockStat> block_stats(const std::string& collection) const;
  // Removes every data block whose vectors are ALL tombstoned (never a center block): one catalog
  // transaction removes the segment rows with their vectors and tombstone rows; the files are
  // unlinked after the commit. Returns the number of blocks removed. A crash before the commit
  // changes nothing; after it the files are unreferenced and recovery sweeps them. The log is
  // checkpointed first (all its records are in the catalog while the writer mutex is held), so
  // recovery can never roll a removed segment forward again.
  std::size_t drop_dead_blocks(const std::string& collection);
  // Repack by the client (it decrypts 1..N blocks of one cluster, drops deleted slots, merges,
  // re-encrypts). Precondition, else BlockMismatch: all old blocks (distinct) are data blocks of
  // one cluster with the same column count as `neu`, and `neu.ids` are exactly the union of their
  // LIVE ids (any order: the client chooses the slot order; at least one), dead ids are dropped.
  // BlockNotFound if an old segment is not a block of the collection; center blocks are refused.
  // The caller checks that the new block fits the preset's slot count. Order: checkpoint the log;
  // write the new file durably; one catalog transaction (new row, live vectors repointed, dead
  // vectors and tombstones removed, old rows removed); unlink the old files (deferred while a
  // search pins them). Atomic: after a crash either the old set or the new block is in the
  // catalog, and the other files are swept. Returns the new segment name.
  std::string replace_segments(const std::string& collection,
                               const std::vector<std::string>& old_segments,
                               const SegmentData& neu);

  // ---- Blobs: opaque encrypted bytes keyed by (collection, kind, id) --------------------
  // put_blob: payload size must be a bucket size (BlobSizeError) and kind a known kind
  // (BlobKindError); unknown collection throws "unknown collection". Order: write the file
  // durably (temp, fsync, rename), one catalog transaction inserts or replaces the row, then the
  // replaced file is unlinked. A crash leaves the old blob or the new one, never a part; files
  // without a catalog row are swept at recovery. Blobs are removed with their collection.
  void put_blob(const std::string& collection, std::uint32_t kind, const std::string& id,
                const std::vector<std::uint8_t>& payload);
  // False if there is no such blob. Throws on a corrupt file.
  bool get_blob(const std::string& collection, std::uint32_t kind, const std::string& id,
                std::vector<std::uint8_t>* out) const;
  // Removes the blobs that exist (missing ids are ignored); returns how many were removed.
  // Number and payload bytes of the collection's blobs (quota checks).
  void blob_totals(const std::string& collection, std::uint64_t* count, std::uint64_t* bytes) const;
  std::size_t delete_blobs(const std::string& collection, std::uint32_t kind,
                           const std::vector<std::string>& ids);

  const std::string& openfhe_version() const { return ofhe_; }

  SegmentData read_segment(const std::string& collection, const std::string& segment) const;
  const Catalog& catalog() const { return *catalog_; }
  const RecoveryStats& recovery_stats() const { return stats_; }
  std::string segment_path(const std::string& name) const;
  std::string wal_path() const { return root_ + "/wal.log"; }
  std::string segments_dir() const { return root_ + "/segments"; }

 private:
  friend class SegmentPin;
  void recover();
  // Unlinks `file` now, or when the last pin on it is released. Caller holds mu_.
  void unlink_or_defer(const std::string& file);
  void unpin(const std::vector<std::string>& files) const;
  std::string blob_dir() const { return root_ + "/blobs"; }
  std::string key_dir() const { return root_ + "/keys"; }
  ColumnExpect expect_for(const CollectionSpec& spec) const;
  std::string root_, ofhe_;
  ShapeProvider shape_;
  std::unique_ptr<Catalog> catalog_;
  std::unique_ptr<Wal> wal_;
  mutable std::mutex mu_;
  // Pins (see Snapshot): segment file -> number of live pins; files removed from the catalog while
  // pinned wait in deferred_ (memory only: after a crash recovery sweeps them as unknown files).
  mutable std::map<std::string, int> pins_;
  mutable std::set<std::string> deferred_;
  RecoveryStats stats_;
};

}  // namespace ombra::storage
