#pragma once

// =========================================================================
// RESIDUE WALL — Main Facade
// Integrates CacheTopology + AlignedGateway + PrefetchEngine into a
// single component that EntropyControllerV3 owns.
//
// Usage:
//   ResidueWall wall;  // detects cache, configures prefetcher
//   const float* aligned = wall.prepare_input(raw_ptr, count);
//   wall.prefetch_next_frame(aligned, frame_size, i, num_frames);
// =========================================================================

#include "aligned_gateway.h"
#include "cache_topology.h"
#include "prefetch_engine.h"

#include <cstddef>

#ifdef _MSC_VER
#define RESIDUE_WALL_FORCEINLINE __forceinline
#else
#define RESIDUE_WALL_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace residue_wall {

class alignas(64) ResidueWall {
public:
  // Cold path: detect topology, configure prefetcher with default frame size.
  // Frame size is re-configured lazily on first prepare_input call.
  ResidueWall() noexcept;

  // -----------------------------------------------------------------------
  // COLD PATH: Prepare input buffer for aligned, prefetch-ready processing.
  //
  // If `raw` is already 64-byte aligned → returns raw directly (zero-copy).
  // Otherwise → copies into internal AlignedBuffer using non-temporal stores,
  //             returns aligned pointer.
  //
  // The internal buffer is reused across calls (amortized zero allocation).
  // Re-configures prefetcher if frame_size changes.
  // -----------------------------------------------------------------------
  const float *prepare_input(const float *raw, size_t count,
                             size_t frame_size) noexcept;

  // -----------------------------------------------------------------------
  // HOT PATH: Inject prefetch hints for next + lookahead frames.
  // Must be called with the aligned pointer returned by prepare_input().
  // -----------------------------------------------------------------------
  RESIDUE_WALL_FORCEINLINE void
  prefetch_next_frame(const float *aligned_base, size_t frame_size,
                      size_t current_frame,
                      size_t total_frames) const noexcept {
    prefetcher_.prefetch_frame(aligned_base, frame_size, current_frame,
                               total_frames);
  }

  // -----------------------------------------------------------------------
  // COLD PATH: Warm up L1d with the first frame's data before processing.
  // -----------------------------------------------------------------------
  void warmup(const float *aligned_base, size_t frame_size) const noexcept {
    prefetcher_.warmup_prefetch(aligned_base, frame_size);
  }

  // -----------------------------------------------------------------------
  // Accessors (for diagnostics / tests)
  // -----------------------------------------------------------------------
  const CacheTopology &topology() const noexcept { return topology_; }
  bool last_input_was_zero_copy() const noexcept { return last_zero_copy_; }

private:
  CacheTopology topology_;
  PrefetchEngine prefetcher_;
  AlignedBuffer<float, 64> staging_buf_;
  size_t last_frame_size_;
  bool last_zero_copy_;
};

} // namespace residue_wall
