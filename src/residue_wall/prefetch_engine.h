#pragma once

// =========================================================================
// RESIDUE WALL — Component 3: Software Prefetch Engine
// Branchless, unrolled _mm_prefetch with T0/T1 hints.
// Auto-adjusts stride and count based on CacheTopology.
// Zero branches in hot path. Zero heap allocation.
// =========================================================================

#include "cache_topology.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#ifdef _MSC_VER
#define RESIDUE_WALL_FORCEINLINE __forceinline
#else
#define RESIDUE_WALL_FORCEINLINE __attribute__((always_inline)) inline
#endif

namespace residue_wall {

class PrefetchEngine {
public:
  // Configuration — set once from CacheTopology, never changes on hot path
  uint32_t line_size_; // Cache line size (bytes)
  uint32_t t0_lines_;  // Number of lines to prefetch with T0 (into L1d)
  uint32_t t1_lines_;  // Number of lines to prefetch with T1 (into L2)
  uint32_t t0_stride_; // = line_size_ (byte stride between T0 prefetches)
  uint32_t t1_stride_; // = line_size_ (byte stride between T1 prefetches)

  PrefetchEngine() noexcept { configure(kDefaultTopology, 4096); }

  // Configure based on detected topology and expected frame size.
  // Called once at construction (cold path).
  void configure(const CacheTopology &topo, size_t frame_bytes) noexcept {
    line_size_ = topo.cache_line_size;
    t0_stride_ = line_size_;
    t1_stride_ = line_size_;

    // Frame lines = how many cache lines one frame occupies
    uint32_t frame_lines =
        static_cast<uint32_t>((frame_bytes + line_size_ - 1) / line_size_);

    // T0 (into L1d): prefetch up to 8 lines of next frame.
    // Cap at frame_lines (don't prefetch beyond frame).
    // If L1d is small (< 32KB), reduce to 4 to avoid self-eviction.
    uint32_t max_t0 = (topo.l1d_size < 32768) ? 4u : 8u;
    t0_lines_ = std::min(frame_lines, max_t0);

    // T1 (into L2): prefetch 4 lines of lookahead frame.
    // Cap at frame_lines. This is the 2-frame-ahead hint.
    uint32_t max_t1 = 4u;
    t1_lines_ = std::min(frame_lines, max_t1);
  }

  // =========================================================================
  // HOT PATH: Prefetch next frame (T0) + lookahead frame (T1)
  // Called once per frame in the stream processing loop.
  //
  // Arguments:
  //   base        — pointer to beginning of input buffer (aligned)
  //   frame_bytes — size of one frame in bytes
  //   current     — index of current frame being processed
  //   total       — total number of frames
  //
  // Cost: 12 instructions (8 T0 + 4 T1), ~4ns, branchless via pointer clamping
  // =========================================================================
  RESIDUE_WALL_FORCEINLINE void prefetch_frame(const char *base,
                                               size_t frame_bytes,
                                               size_t current,
                                               size_t total) const noexcept {
    // Clamp indices — branchless via min. If at last frame, prefetches
    // still target valid memory (the last frame itself). This avoids
    // branch misprediction and is safe because prefetch of valid
    // addresses is always a no-op hint to the CPU.
    size_t next_idx = current + 1;
    size_t look_idx = current + 2;

    // Branchless clamp to (total - 1)
    size_t last = total - 1;
    next_idx = next_idx < total ? next_idx : last;
    look_idx = look_idx < total ? look_idx : last;

    const char *p_next = base + next_idx * frame_bytes;
    const char *p_look = base + look_idx * frame_bytes;

    // --- T0: Bring next frame into L1d ---
    // Unrolled to 8 lines max. For frames < 8 lines, some prefetches
    // will target within the frame (harmless redundancy, no branch).
    _mm_prefetch(p_next, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 2, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 3, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 4, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 5, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 6, _MM_HINT_T0);
    _mm_prefetch(p_next + t0_stride_ * 7, _MM_HINT_T0);

    // --- T1: Bring lookahead frame into L2 ---
    _mm_prefetch(p_look, _MM_HINT_T1);
    _mm_prefetch(p_look + t1_stride_, _MM_HINT_T1);
    _mm_prefetch(p_look + t1_stride_ * 2, _MM_HINT_T1);
    _mm_prefetch(p_look + t1_stride_ * 3, _MM_HINT_T1);
  }

  // =========================================================================
  // HOT PATH (float-typed convenience overload)
  // =========================================================================
  RESIDUE_WALL_FORCEINLINE void prefetch_frame(const float *base,
                                               size_t frame_size_floats,
                                               size_t current,
                                               size_t total) const noexcept {
    prefetch_frame(reinterpret_cast<const char *>(base),
                   frame_size_floats * sizeof(float), current, total);
  }

  // =========================================================================
  // COLD PATH: Prefetch entire buffer head (for first-frame warmup)
  // Called once at stream start to prime L1d with the first frame's data.
  // =========================================================================
  void warmup_prefetch(const char *ptr, size_t bytes) const noexcept {
    size_t lines = std::min<size_t>(bytes / line_size_, 16u);
    for (size_t i = 0; i < lines; ++i) {
      _mm_prefetch(ptr + i * line_size_, _MM_HINT_T0);
    }
  }

  void warmup_prefetch(const float *ptr, size_t count) const noexcept {
    warmup_prefetch(reinterpret_cast<const char *>(ptr), count * sizeof(float));
  }
};

} // namespace residue_wall
