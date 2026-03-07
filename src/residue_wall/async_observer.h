#pragma once

// =========================================================================
// RESIDUE WALL — Level 5: Active Observer / Async Ingestion (V4.2)
//
// V4.1: Backpressure Architecture — fill-level monitoring + drop counting
// V4.2: Exponential Backoff spin-wait — prevents thermal throttling
//       Dead code removal — eliminated unused V4.0 heuristic loop
//       Batch size hinting — recommended_push_size() for producers
//
// Encapsulates the EntropyControllerV3 in a dedicated C++ background thread.
// Python (Producer) pushes data into a lock-free Ring Buffer.
// The Worker Thread (Consumer) runs in the IsolationZone (OS Bypass),
// pulling data, aligning it, and executing the Full-Scan Gate.
//
// Telemetry (FPS, Sparsity, Buffer Fill, Drops) is reported back to Python
// via std::atomic without blocking the hot path.
// =========================================================================

#include "../residue/core.h"
#include "aligned_gateway.h"
#include "isolation_zone.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

namespace residue_wall {

// -------------------------------------------------------------------------
// Lock-Free Single-Producer Single-Consumer (SPSC) Ring Buffer
// Designed for one producer (Python) and one consumer (C++ Worker).
// -------------------------------------------------------------------------
template <typename T> class SpscRingBuffer {
private:
  std::vector<T> buffer_;
  const size_t capacity_;
  alignas(64) std::atomic<size_t> head_{0}; // Written by Producer
  alignas(64) std::atomic<size_t> tail_{0}; // Written by Consumer

public:
  explicit SpscRingBuffer(size_t capacity)
      : buffer_(capacity + 1), capacity_(capacity + 1) {}

  // Produces 'count' elements. Returns number of elements actually written.
  size_t push(const T *data, size_t count) {
    size_t current_tail = tail_.load(std::memory_order_acquire);
    size_t current_head = head_.load(std::memory_order_relaxed);

    size_t available_space =
        capacity_ - 1 - ((current_head - current_tail + capacity_) % capacity_);
    size_t to_write = std::min(count, available_space);
    if (to_write == 0)
      return 0;

    // Write in one or two chunks (if wrapping around)
    size_t first_chunk = std::min(to_write, capacity_ - current_head);
    std::memcpy(&buffer_[current_head], data, first_chunk * sizeof(T));

    if (to_write > first_chunk) {
      std::memcpy(&buffer_[0], data + first_chunk,
                  (to_write - first_chunk) * sizeof(T));
    }

    head_.store((current_head + to_write) % capacity_,
                std::memory_order_release);
    return to_write;
  }

  // Consumes up to 'count' elements. Returns number actually read.
  size_t pop(T *dest, size_t count) {
    size_t current_head = head_.load(std::memory_order_acquire);
    size_t current_tail = tail_.load(std::memory_order_relaxed);

    size_t available = (current_head - current_tail + capacity_) % capacity_;
    size_t to_read = std::min(count, available);
    if (to_read == 0)
      return 0;

    size_t first_chunk = std::min(to_read, capacity_ - current_tail);
    std::memcpy(dest, &buffer_[current_tail], first_chunk * sizeof(T));

    if (to_read > first_chunk) {
      std::memcpy(dest + first_chunk, &buffer_[0],
                  (to_read - first_chunk) * sizeof(T));
    }

    tail_.store((current_tail + to_read) % capacity_,
                std::memory_order_release);
    return to_read;
  }

  // V4.1: Backpressure support — query current fill level
  size_t fill_level() const {
    size_t h = head_.load(std::memory_order_relaxed);
    size_t t = tail_.load(std::memory_order_relaxed);
    return (h - t + capacity_) % capacity_;
  }

  size_t max_capacity() const { return capacity_ - 1; }
};

// -------------------------------------------------------------------------
// Telemetry Snapshot
// -------------------------------------------------------------------------
struct TelemetrySnapshot {
  uint64_t total_samples_ingested;
  uint64_t total_samples_processed;
  uint64_t total_samples_skipped; // Passed the predicted gate
  uint64_t total_frames_dropped;  // V4.1: frames lost due to buffer overflow
  double current_fps;             // Output frames per second
  double sparsity_pct;            // Percentage of frames skipped
  double buffer_fill_pct;         // V4.1: 0.0-100.0 buffer saturation
  bool is_running;
  bool isolation_active;
  bool backpressure_active; // V4.1: true if fill > 90%
};

// -------------------------------------------------------------------------
// Async Observer
// -------------------------------------------------------------------------
class AsyncObserver {
private:
  std::unique_ptr<EntropyControllerV3> engine_;
  SpscRingBuffer<float> intake_queue_;

  // Output Ring Buffer (Consumer = Python, Producer = C++ Worker)
  // For now, Python just checks telemetry. Output buffer stores scaling
  // factors.
  SpscRingBuffer<float> output_queue_;

  // Thread control
  std::thread worker_thread_;
  std::atomic<bool> should_stop_{false};
  std::atomic<bool> is_running_{false};

  // Telemetry (Atomic)
  alignas(64) std::atomic<uint64_t> samples_ingested_{0};
  alignas(64) std::atomic<uint64_t> samples_processed_{0};
  alignas(64) std::atomic<uint64_t> samples_skipped_{0};
  alignas(64) std::atomic<uint64_t> frames_dropped_{0}; // V4.1: backpressure

  size_t frame_size_;

  // Timing for FPS
  std::atomic<uint64_t> last_fps_update_ns_{0};
  std::atomic<uint64_t> current_fps_x100_{
      0}; // scaled integer to avoid float atomics

  void worker_loop();

  // OS specific pause
  static inline void micro_pause() {
#ifdef _MSC_VER
    _mm_pause();
#else
    __builtin_ia32_pause();
#endif
  }

public:
  AsyncObserver(size_t frame_size = 1024,
                size_t buffer_capacity_frames = 10000);
  ~AsyncObserver();

  void start();
  void stop();

  // Producer (Python): Push raw float data into C++ background thread
  size_t push_data(const float *data, size_t size);

  // Consumer (Python): Read available output scaling factors
  size_t pull_output(float *dest, size_t max_count);

  TelemetrySnapshot poll_telemetry() const;
  void reset_telemetry();

  // V4.2: Optimal push size hint for Python producers.
  // Pushing in multiples of this value minimizes atomic coherency traffic.
  size_t recommended_push_size() const { return frame_size_ * 64; }
};

} // namespace residue_wall
