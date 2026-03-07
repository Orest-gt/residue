#ifndef ENTROPY_CONTROLLER_V3_H
#define ENTROPY_CONTROLLER_V3_H

#include "../residue_wall/residue_wall.h"
#include <cstddef>

/**
 * PROJECT RESIDUE V4.1 — Hardened Structural Intelligence Engine.
 * ZERO HEAP ALLOCATION. PURE HARDWARE EVENT. INDUSTRIAL-GRADE.
 *
 * V4.1 Changes:
 *   - Replaced indirect V-Table dispatch with [[likely]]/[[unlikely]]
 *     predicted branches (eliminates BTB misses)
 *   - Multi-probe heuristic gate (head/mid/tail sampling)
 *   - No function pointers in hot path
 */
class alignas(64) EntropyControllerV3 {
protected:
  // --- Residue Wall (Memory Prefetching & Alignment Engine) ---
  residue_wall::ResidueWall wall_;

  // --- Configuration (immutable after construction) ---
  int num_bins;
  float min_scaling_factor;
  float max_scaling_factor;
  float l1_sparsity_threshold;
  float ema_alpha;

  // --- Live State (mutated per frame) ---
  float current_scaling_factor;
  size_t total_samples_processed;
  size_t total_samples_skipped;

  // --- Temporal Coherence (inline EMA, no heap) ---
  float ema_scaling;
  bool ema_initialized;

  // --- Predicted Gating (V4.1: no indirect calls) ---
  float activity_threshold_;

public:
  EntropyControllerV3(int bins = 256, float entropy_threshold = 0.1f,
                      size_t temporal_buffer_size = 5,
                      float l1_threshold = 0.1f, float ema_alpha_param = 0.2f);

  // --- Hot Path ---
  float infer_single_sample_fast(const float *input, size_t size);
  float infer_single_sample_noop(const float *input,
                                 size_t size); // O(1) bypass
  void process_stream_fast(const float *input, size_t total_size,
                           size_t frame_size, float *output_scalings);
  void process_stream_walled(const float *input, size_t total_size,
                             size_t frame_size, float *output_scalings);

  // --- State Management ---
  void reset_history();
  float get_current_scaling_factor() const { return current_scaling_factor; }
  size_t get_total_samples_skipped() const { return total_samples_skipped; }
  size_t get_total_samples_processed() const { return total_samples_processed; }

protected:
  // --- Inline Temporal EMA (replaces V2 CircularBuffer) ---
  float compute_temporal_coherence(float current_scaling);

  // --- AVX2 Feature Extractors (pointer-only, zero heap) ---
  float calculate_zero_crossing_rate_ptr(const float *input, size_t size) const;
  float calculate_l1_norm_sparsity_ptr(const float *input, size_t size) const;
  float calculate_input_entropy_ptr(const float *input, size_t size);
  float calculate_complexity_score_ptr(const float *input, size_t size) const;
  float calculate_structure_score_ptr(const float *input, size_t size) const;
  float calculate_sparsity_ptr(const float *input, size_t size) const;
};

#endif // ENTROPY_CONTROLLER_V3_H
