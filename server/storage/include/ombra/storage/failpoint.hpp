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
