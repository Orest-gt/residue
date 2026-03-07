#include "residue_wall.h"

namespace residue_wall {

// =========================================================================
// COLD PATH: Construction — detect cache topology, configure prefetcher
// =========================================================================
ResidueWall::ResidueWall() noexcept
    : topology_(get_cache_topology()), prefetcher_(), staging_buf_(),
      last_frame_size_(0), last_zero_copy_(false) {
  // Default configuration for 1024-float frames (4096 bytes).
  // Will be reconfigured on first prepare_input if frame_size differs.
  prefetcher_.configure(topology_, 4096);
}

// =========================================================================
// COLD PATH: Prepare input — alignment check + optional staging copy
// =========================================================================
const float *ResidueWall::prepare_input(const float *raw, size_t count,
                                        size_t frame_size) noexcept {
  // Reconfigure prefetcher if frame size changed
  if (frame_size != last_frame_size_) {
    prefetcher_.configure(topology_, frame_size * sizeof(float));
    last_frame_size_ = frame_size;
  }

  // Fast path: already aligned → zero copy
  if (is_aligned(raw, 64)) {
    last_zero_copy_ = true;
    return raw;
  }

  // Slow path: copy into aligned staging buffer
  // ensure_capacity is a no-op if buffer is already large enough
  staging_buf_.ensure_capacity(count);
  nontemporal_copy_float(staging_buf_.data(), raw, count);
  last_zero_copy_ = false;
  return staging_buf_.data();
}

} // namespace residue_wall
