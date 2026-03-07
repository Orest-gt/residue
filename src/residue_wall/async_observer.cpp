#include "async_observer.h"
#include <chrono>

namespace residue_wall {

AsyncObserver::AsyncObserver(size_t frame_size, size_t buffer_capacity_frames)
    : frame_size_(frame_size),
      intake_queue_(frame_size * buffer_capacity_frames),
      output_queue_(buffer_capacity_frames) {
  engine_ = std::make_unique<EntropyControllerV3>();
}

AsyncObserver::~AsyncObserver() { stop(); }

void AsyncObserver::start() {
  bool expected = false;
  if (is_running_.compare_exchange_strong(expected, true)) {
    should_stop_.store(false, std::memory_order_release);
    worker_thread_ = std::thread(&AsyncObserver::worker_loop, this);
  }
}

void AsyncObserver::stop() {
  if (is_running_.load(std::memory_order_acquire)) {
    should_stop_.store(true, std::memory_order_release);
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
    is_running_.store(false, std::memory_order_release);
  }
}

size_t AsyncObserver::push_data(const float *data, size_t size) {
  if (!is_running_.load(std::memory_order_relaxed))
    return 0;
  size_t written = intake_queue_.push(data, size);
  // V4.1: Track dropped data for backpressure reporting
  size_t dropped = size - written;
  if (dropped > 0) {
    frames_dropped_.fetch_add(dropped / frame_size_, std::memory_order_relaxed);
  }
  return written;
}

size_t AsyncObserver::pull_output(float *dest, size_t max_count) {
  return output_queue_.pop(dest, max_count);
}

void AsyncObserver::reset_telemetry() {
  samples_ingested_.store(0, std::memory_order_relaxed);
  samples_processed_.store(0, std::memory_order_relaxed);
  samples_skipped_.store(0, std::memory_order_relaxed);
  frames_dropped_.store(0, std::memory_order_relaxed); // V4.1
  last_fps_update_ns_.store(0, std::memory_order_relaxed);
  current_fps_x100_.store(0, std::memory_order_relaxed);
}

TelemetrySnapshot AsyncObserver::poll_telemetry() const {
  TelemetrySnapshot snap;
  snap.total_samples_ingested =
      samples_ingested_.load(std::memory_order_relaxed);
  snap.total_samples_processed =
      samples_processed_.load(std::memory_order_relaxed);
  snap.total_samples_skipped = samples_skipped_.load(std::memory_order_relaxed);
  snap.total_frames_dropped = frames_dropped_.load(std::memory_order_relaxed);

  uint64_t total = snap.total_samples_processed;
  if (total > 0) {
    snap.sparsity_pct =
        (double)snap.total_samples_skipped / (double)total * 100.0;
  } else {
    snap.sparsity_pct = 0.0;
  }

  uint64_t fps_scaled = current_fps_x100_.load(std::memory_order_relaxed);
  snap.current_fps = (double)fps_scaled / 100.0;

  // V4.1: Buffer fill level and backpressure
  size_t fill = intake_queue_.fill_level();
  size_t cap = intake_queue_.max_capacity();
  snap.buffer_fill_pct = (cap > 0) ? ((double)fill / (double)cap * 100.0) : 0.0;
  snap.backpressure_active = (snap.buffer_fill_pct > 90.0);

  snap.is_running = is_running_.load(std::memory_order_relaxed);
  snap.isolation_active = true;

  return snap;
}

void AsyncObserver::worker_loop() {
  // V4.2: Enter the Isolation Zone (HIGH priority, SMT-aware core pin)
  IsolationZone guard;

  const size_t TILE_FRAMES = 64;
  const size_t tile_size_floats = TILE_FRAMES * frame_size_;

  // Thread-local aligned buffer for the C++ side to pull data into
#ifdef _MSC_VER
  float *aligned_tile = static_cast<float *>(
      _aligned_malloc(tile_size_floats * sizeof(float), 64));
#else
  float *aligned_tile =
      static_cast<float *>(aligned_alloc(64, tile_size_floats * sizeof(float)));
#endif

  float output_factors[TILE_FRAMES];

  // FPS tracking (local cache before atomic update)
  uint64_t frames_this_second = 0;
  auto last_fps_time = std::chrono::steady_clock::now();

  // V4.2: Exponential Backoff spin-wait state
  uint32_t idle_spins = 0;

  // Engine state
  engine_->reset_history();

  while (!should_stop_.load(std::memory_order_acquire)) {
    guard.reset_timer();

    // 1. Lock-free Pull
    size_t floats_read = intake_queue_.pop(aligned_tile, tile_size_floats);

    if (floats_read == 0) {
      // V4.2: Adaptive 3-Tier Exponential Backoff
      // Prevents thermal throttling and GC-pause CPU waste.
      idle_spins++;

      if (idle_spins < 64) {
        // Tier 1: Hot spin (~1us) — sub-microsecond wakeup latency
        _mm_pause();
      } else if (idle_spins < 4096) {
        // Tier 2: Warm pause (~50us) — yields HT sibling resources
        for (int k = 0; k < 8; ++k)
          _mm_pause();
      } else {
        // Tier 3: Cold yield — give up timeslice, saves TDP
#ifdef _WIN32
        SwitchToThread();
#else
        sched_yield();
#endif
      }
      continue;
    }

    // Data arrived — reset backoff state
    idle_spins = 0;

    size_t frames_read = floats_read / frame_size_;
    samples_ingested_.fetch_add(floats_read, std::memory_order_relaxed);

    // V4.2: Dead heuristic loop removed. Delegate directly to the
    // battle-tested process_stream_walled which contains the full-scan gate.
    engine_->process_stream_walled(aligned_tile, floats_read, frame_size_,
                                   output_factors);
    samples_processed_.store(engine_->get_total_samples_processed(),
                             std::memory_order_relaxed);
    samples_skipped_.store(engine_->get_total_samples_skipped(),
                           std::memory_order_relaxed);

    // Calculate FPS
    frames_this_second += frames_read;
    auto now = std::chrono::steady_clock::now();
    double elapsed_s =
        std::chrono::duration<double>(now - last_fps_time).count();

    if (elapsed_s >= 0.1) { // Update telemetry every 100ms
      double fps = (double)frames_this_second / elapsed_s;
      current_fps_x100_.store(static_cast<uint64_t>(fps * 100.0),
                              std::memory_order_relaxed);
      frames_this_second = 0;
      last_fps_time = now;
    }

    output_queue_.push(output_factors, frames_read);

    guard.check_safety();
  } // end while

#ifdef _MSC_VER
  _aligned_free(aligned_tile);
#else
  free(aligned_tile);
#endif
}

} // namespace residue_wall
