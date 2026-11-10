// Crash points for the recovery tests. A test sets one point; when the writer reaches it the
// process _exit()s at once (no destructors, no flush), like a kill -9. Off in production: the
// point is None and the check is one relaxed atomic load.
#pragma once

#include <unistd.h>

#include <atomic>

namespace ombra::storage {

enum class CrashPoint : int {
  None = 0,
  InsertBeforeIntent,
  InsertAfterIntent,
  InsertBeforeSegmentRename,
  InsertAfterSegmentRename,
  InsertAfterCatalog,
  InsertAfterCommit,
  DeleteBeforeRecord,
  DeleteAfterRecord,
  DeleteAfterCatalog,
  DropBeforeCatalog,
  DropAfterCatalog,   // collection gone from the catalog, files and log not yet cleaned
  DropAfterWalReset,  // log checkpointed, files not yet unlinked
  DropAfterFirstFile, // some files unlinked
  // Block replacement (replace_segment), in order.
  ReplaceBeforeNewFile,  // log checkpointed, nothing written
  ReplaceAfterNewFile,   // new file durable, catalog untouched
  ReplaceAfterCatalog,   // catalog switched, old file still on disk
  ReplaceAfterUnlink,    // old file gone
  // Dropping fully tombstoned blocks (drop_dead_blocks), in order.
  CompactBeforeCatalog,
  CompactAfterCatalog,   // rows gone, files still on disk
  CompactAfterUnlink,
};

inline constexpr int kCrashExitCode = 42;

inline std::atomic<int>& crash_point_storage() {
  static std::atomic<int> p{0};
  return p;
}
inline void set_crash_point(CrashPoint p) { crash_point_storage().store(static_cast<int>(p)); }
inline void maybe_crash(CrashPoint p) {
  if (crash_point_storage().load(std::memory_order_relaxed) == static_cast<int>(p))
    _exit(kCrashExitCode);
}

}  // namespace ombra::storage
