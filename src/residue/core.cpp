#include "core.h"
#include "guardian.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

// =========================================================================
// SIMD UTILITIES
// =========================================================================

RESIDUE_FORCEINLINE float hsum_ps_avx2(__m256 v) {
  __m128 v_low = _mm256_castps256_ps128(v);
  __m128 v_high = _mm256_extractf128_ps(v, 1);
  __m128 v_sum4 = _mm_add_ps(v_low, v_high);
  __m128 v_sum2 = _mm_hadd_ps(v_sum4, v_sum4);
  __m128 v_sum1 = _mm_hadd_ps(v_sum2, v_sum2);
  return _mm_cvtss_f32(v_sum1);
}

// Vectorized exp(x) — Cephes polynomial with Cody-Waite reduction.
// Relative error < 2e-7 across full float range. Replaces 7x scalar std::exp.
RESIDUE_FORCEINLINE __m256 fast_exp_avx2(__m256 x) {
  const __m256 log2e = _mm256_set1_ps(1.44269504089f);
  const __m256 ln2_hi = _mm256_set1_ps(0.693359375f);
  const __m256 ln2_lo = _mm256_set1_ps(-2.12194440e-4f);
  const __m256 one = _mm256_set1_ps(1.0f);

  const __m256 c5 = _mm256_set1_ps(1.9875691500e-4f);
  const __m256 c4 = _mm256_set1_ps(1.3981999507e-3f);
  const __m256 c3 = _mm256_set1_ps(8.3334519073e-3f);
  const __m256 c2 = _mm256_set1_ps(4.1665795894e-2f);
  const __m256 c1 = _mm256_set1_ps(1.6666665459e-1f);
  const __m256 c0 = _mm256_set1_ps(5.0000001201e-1f);

  // x * log2(e) -> integer n + fraction f
  __m256 t = _mm256_mul_ps(x, log2e);
  __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

  // Cody-Waite reduction: r = x - n*ln(2)
  __m256 r = _mm256_fnmadd_ps(n, ln2_hi, x);
  r = _mm256_fnmadd_ps(n, ln2_lo, r);

  // Horner polynomial: 1 + r + r^2/2! + r^3/3! + r^4/4! + r^5/5!
  __m256 poly = _mm256_fmadd_ps(c5, r, c4);
  poly = _mm256_fmadd_ps(poly, r, c3);
  poly = _mm256_fmadd_ps(poly, r, c2);
  poly = _mm256_fmadd_ps(poly, r, c1);
  poly = _mm256_fmadd_ps(poly, r, c0);
  __m256 r2 = _mm256_mul_ps(r, r);
  poly = _mm256_fmadd_ps(poly, r2, _mm256_add_ps(r, one));

  // Reconstruct: exp(x) = poly * 2^n via IEEE 754 exponent injection
  __m256i ni = _mm256_cvtps_epi32(n);
  ni = _mm256_add_epi32(ni, _mm256_set1_epi32(127));
  ni = _mm256_slli_epi32(ni, 23);
  __m256 pow2n = _mm256_castsi256_ps(ni);

  return _mm256_mul_ps(poly, pow2n);
}

// =========================================================================
// CONSTRUCTOR / STATE MANAGEMENT
// =========================================================================

EntropyControllerV3::EntropyControllerV3(int bins, float /*entropy_threshold*/,
                                         size_t /*temporal_buffer_size*/,
                                         float l1_threshold,
                                         float ema_alpha_param)
    : num_bins(bins), min_scaling_factor(0.0f), max_scaling_factor(10.0f),
      l1_sparsity_threshold(l1_threshold), ema_alpha(ema_alpha_param),
      current_scaling_factor(1.0f), total_samples_processed(0),
      total_samples_skipped(0), ema_scaling(0.0f), ema_initialized(false),
      activity_threshold_(1e-5f) // Threshold for the L1 head heuristic
{
  dispatch_table_[0] = &EntropyControllerV3::infer_single_sample_noop;
  dispatch_table_[1] = &EntropyControllerV3::infer_single_sample_fast;
}

void EntropyControllerV3::reset_history() {
  current_scaling_factor = 1.0f;
  total_samples_processed = 0;
  total_samples_skipped = 0;
  ema_scaling = 0.0f;
  ema_initialized = false;
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::compute_temporal_coherence(float current_scaling) {
  if (!ema_initialized) {
    ema_scaling = current_scaling;
    ema_initialized = true;
  } else {
    ema_scaling =
        ema_alpha * current_scaling + (1.0f - ema_alpha) * ema_scaling;
  }
  return ema_scaling;
}

// =========================================================================
// HOT PATH: SINGLE FRAME INFERENCE
// =========================================================================

float EntropyControllerV3::infer_single_sample_fast(const float *input,
                                                    size_t size) {
  if (!input || size == 0)
    return 1.0f;

  // 1. EXTRACT FEATURES
  alignas(32) float features[8] = {0.0f};

  features[0] = this->calculate_zero_crossing_rate_ptr(input, size);
  features[1] = this->calculate_l1_norm_sparsity_ptr(input, size);
  features[2] = this->compute_temporal_coherence(this->current_scaling_factor);
  features[3] = this->calculate_input_entropy_ptr(input, size);
  features[4] = this->calculate_complexity_score_ptr(input, size);
  features[5] = this->calculate_structure_score_ptr(input, size);
  features[6] = this->calculate_sparsity_ptr(input, size);
  features[7] = 0.0f;

  // 2. SIMD CLIP [-88, 88]
  __m256 v_features = _mm256_load_ps(features);
  __m256 v_clipped = _mm256_max_ps(
      _mm256_set1_ps(-88.0f), _mm256_min_ps(v_features, _mm256_set1_ps(88.0f)));

  // 3. VECTORIZED EXP & SOFTMAX [H-1 FIX: eliminates 7x scalar std::exp]
  __m256 v_exp = fast_exp_avx2(v_clipped);
  // Zero lane 7 (padding) — exp(0) = 1.0, must be 0.0
  v_exp = _mm256_blend_ps(v_exp, _mm256_setzero_ps(), 0x80);

  float exp_sum = hsum_ps_avx2(v_exp) + 1e-6f;
  __m256 v_probs = _mm256_mul_ps(v_exp, _mm256_set1_ps(1.0f / exp_sum));

  // 4. FMA WEIGHTED DOT-PRODUCT [H-3 FIX: static weights, loaded once]
  static const alignas(32) float weights[8] = {9.0f, 8.0f, 7.0f, 5.0f,
                                               4.0f, 3.0f, 2.0f, 0.0f};
  __m256 v_weights = _mm256_load_ps(weights);
  __m256 v_acc = _mm256_mul_ps(v_probs, v_weights);

  // 5. HORIZONTAL SUM [H-2 FIX: single hsum call, no duplication]
  float scaling = hsum_ps_avx2(v_acc);

  // 6. STATE UPDATE (zero heap)
  this->current_scaling_factor = scaling;
  this->total_samples_processed++;

  return std::max(this->min_scaling_factor,
                  std::min(this->max_scaling_factor, scaling));
}

float EntropyControllerV3::infer_single_sample_noop(const float * /*input*/,
                                                    size_t /*size*/) {
  // O(1) Branchless Dispatch No-Op:
  // Maintain system state and return current scaling factor without math
  this->total_samples_processed++;
  this->total_samples_skipped++;
  return this->current_scaling_factor;
}

// =========================================================================
// BATCH STREAM PROCESSOR
// =========================================================================

void EntropyControllerV3::process_stream_fast(const float *input,
                                              size_t total_size,
                                              size_t frame_size,
                                              float *output_scalings) {
  if (!input || !output_scalings || frame_size == 0)
    return;

  const size_t num_frames = total_size / frame_size;
  const size_t TILE_SIZE = 64;

  // SOFT-LANDING STATE
  float last_good_scaling = this->current_scaling_factor;
  const float FADE_FACTOR = 0.95f;

  // [G-1 FIX] Single ThreadGuard for entire stream — NOT per tile.
  // Affinity + priority set once. Timer reset per tile.
  ThreadGuard guard;

  for (size_t tile_start = 0; tile_start < num_frames;
       tile_start += TILE_SIZE) {
    size_t tile_end = std::min(tile_start + TILE_SIZE, num_frames);

    guard.reset_timer();

    for (size_t i = tile_start; i < tile_end; ++i) {
      output_scalings[i] =
          this->infer_single_sample_fast(input + (i * frame_size), frame_size);
    }

    guard.check_safety();

    if (guard.was_stalled()) {
      for (size_t i = tile_start; i < tile_end; ++i) {
        last_good_scaling =
            last_good_scaling * FADE_FACTOR + (1.0f - FADE_FACTOR);
        output_scalings[i] = last_good_scaling;
      }
    } else {
      last_good_scaling = output_scalings[tile_end - 1];
    }
  }
}
// =========================================================================
// BATCH STREAM PROCESSOR — WALLED (Residue Wall prefetch + alignment)
// Identical to process_stream_fast but with:
//   1. Input alignment gateway (zero-copy or non-temporal staging)
//   2. L1d warmup on stream start
//   3. Per-frame T0/T1 software prefetch for next+lookahead frames
// =========================================================================

void EntropyControllerV3::process_stream_walled(const float *input,
                                                size_t total_size,
                                                size_t frame_size,
                                                float *output_scalings) {
  if (!input || !output_scalings || frame_size == 0)
    return;

  // --- COLD PATH: Align input and configure prefetcher ---
  const float *aligned = wall_.prepare_input(input, total_size, frame_size);
  const size_t num_frames = total_size / frame_size;
  const size_t TILE_SIZE = 64;

  // Warm up L1d with first frame's cache lines
  wall_.warmup(aligned, frame_size);

  // SOFT-LANDING STATE
  float last_good_scaling = this->current_scaling_factor;
  const float FADE_FACTOR = 0.95f;

  // Single IsolationZone for entire stream (timer + affinity + priority)
  ThreadGuard guard;

  // --- COLD PATH: Lock data pages in physical RAM ---
  // Prevents page faults during hot loop. Silently degrades if not admin.
  guard.lock_memory(aligned, total_size * sizeof(float));
  guard.lock_memory(output_scalings, num_frames * sizeof(float));

  for (size_t tile_start = 0; tile_start < num_frames;
       tile_start += TILE_SIZE) {
    size_t tile_end = std::min(tile_start + TILE_SIZE, num_frames);

    guard.reset_timer();

    for (size_t i = tile_start; i < tile_end; ++i) {
      const float *frame_ptr = aligned + (i * frame_size);

      // --- HOT PATH: Prefetch next + lookahead frames ---
      wall_.prefetch_next_frame(aligned, frame_size, i, num_frames);

      // --- BRANCHLESS DYNAMIC DISPATCH (Level 4.1) ---
      // 1. O(1) Fast Heuristic: L1 Norm of the first 256 bits (8 floats)
      __m256 v_frame = _mm256_load_ps(frame_ptr);
      __m256 v_absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
      __m256 v_abs = _mm256_and_ps(v_frame, v_absmask);
      float l1_head = hsum_ps_avx2(v_abs);

      // 2. Branchless Gate: cast boolean to 0 or 1 without jumps
      int is_signal = (int)(l1_head >= activity_threshold_);

      // 3. V-Table Dispatch: skip heavy math when is_signal == 0
      output_scalings[i] =
          (this->*dispatch_table_[is_signal])(frame_ptr, frame_size);
    }

    guard.check_safety();

    if (guard.was_stalled()) {
      for (size_t i = tile_start; i < tile_end; ++i) {
        last_good_scaling =
            last_good_scaling * FADE_FACTOR + (1.0f - FADE_FACTOR);
        output_scalings[i] = last_good_scaling;
      }
    } else {
      last_good_scaling = output_scalings[tile_end - 1];
    }
  }
}

// =========================================================================
// AVX2 FEATURE EXTRACTORS
// =========================================================================

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_zero_crossing_rate_ptr(const float *data,
                                                      size_t size) const {
  if (size < 2)
    return 0.0f;
  size_t zero_crossings = 0;
  size_t i = 0;
  __m256 v_zero = _mm256_setzero_ps();
  for (; i + 8 < size; i += 8) {
    __m256 v1 = _mm256_loadu_ps(data + i);
    __m256 v2 = _mm256_loadu_ps(data + i + 1);

    __m256 cmp1_ge = _mm256_cmp_ps(v1, v_zero, _CMP_GE_OQ);
    __m256 cmp2_lt = _mm256_cmp_ps(v2, v_zero, _CMP_LT_OQ);
    __m256 cross1 = _mm256_and_ps(cmp1_ge, cmp2_lt);

    __m256 cmp1_lt = _mm256_cmp_ps(v1, v_zero, _CMP_LT_OQ);
    __m256 cmp2_ge = _mm256_cmp_ps(v2, v_zero, _CMP_GE_OQ);
    __m256 cross2 = _mm256_and_ps(cmp1_lt, cmp2_ge);

    __m256 cross = _mm256_or_ps(cross1, cross2);
    int mask = _mm256_movemask_ps(cross);
    zero_crossings += RESIDUE_POPCNT(mask);
  }
  for (; i < size - 1; ++i) {
    if ((data[i] >= 0.0f && data[i + 1] < 0.0f) ||
        (data[i] < 0.0f && data[i + 1] >= 0.0f))
      zero_crossings++;
  }
  return static_cast<float>(zero_crossings) / (size - 1);
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_l1_norm_sparsity_ptr(const float *data,
                                                    size_t size) const {
  if (size == 0)
    return 0.0f;
  __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  __m256 v_sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(data + i);
    v_sum = _mm256_add_ps(v_sum, _mm256_and_ps(v, abs_mask));
  }
  float l1_norm = hsum_ps_avx2(v_sum);
  for (; i < size; ++i)
    l1_norm += std::abs(data[i]);

  return std::max(0.0f, 1.0f - (l1_norm / size));
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_input_entropy_ptr(const float *input,
                                                 size_t size) {
  if (!input || size < 2)
    return 0.0f;

  // AVX2 min/max reduction
  __m256 v_min = _mm256_set1_ps(input[0]);
  __m256 v_max = _mm256_set1_ps(input[0]);
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    v_min = _mm256_min_ps(v_min, v);
    v_max = _mm256_max_ps(v_max, v);
  }
  alignas(32) float min_arr[8], max_arr[8];
  _mm256_store_ps(min_arr, v_min);
  _mm256_store_ps(max_arr, v_max);
  float min_val = min_arr[0], max_val = max_arr[0];
  for (int j = 1; j < 8; ++j) {
    if (min_arr[j] < min_val)
      min_val = min_arr[j];
    if (max_arr[j] > max_val)
      max_val = max_arr[j];
  }
  for (; i < size; ++i) {
    if (input[i] < min_val)
      min_val = input[i];
    if (input[i] > max_val)
      max_val = input[i];
  }
  if (max_val - min_val < 1e-8f)
    return 0.0f;

  alignas(32) float local_histogram[256] = {0};
  const size_t local_num_bins = std::min((size_t)256, (size_t)num_bins);

  float range_inv = 1.0f / (max_val - min_val);
  float num_bins_minus_1 = static_cast<float>(local_num_bins - 1);

  for (size_t k = 0; k < size; k++) {
    float normalized = (input[k] - min_val) * range_inv;
    size_t bin = std::min(static_cast<size_t>(normalized * num_bins_minus_1),
                          static_cast<size_t>(local_num_bins - 1));
    local_histogram[bin] += 1.0f;
  }

  float total_samples_inv = 1.0f / static_cast<float>(size);
  float entropy = 0.0f;
  for (size_t k = 0; k < local_num_bins; k++) {
    float prob = local_histogram[k];
    if (prob > 0.0f) {
      float normalized_prob = prob * total_samples_inv;
      entropy -= normalized_prob * std::log2(normalized_prob);
    }
  }
  return entropy;
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_complexity_score_ptr(const float *input,
                                                    size_t size) const {
  if (size < 3)
    return 0.0f;

  __m256 v_sum = _mm256_setzero_ps();
  __m256 v_eps = _mm256_set1_ps(1e-8f);
  __m256 v_neg_eps = _mm256_set1_ps(-1e-8f);

  size_t i = 0;
  size_t zero_count = 0;

  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    v_sum = _mm256_add_ps(v_sum, v);

    __m256 mask1 = _mm256_cmp_ps(v, v_eps, _CMP_LT_OQ);
    __m256 mask2 = _mm256_cmp_ps(v, v_neg_eps, _CMP_GT_OQ);
    __m256 mask = _mm256_and_ps(mask1, mask2);
    int imask = _mm256_movemask_ps(mask);
    zero_count += RESIDUE_POPCNT(imask);
  }

  float sum = hsum_ps_avx2(v_sum);
  for (; i < size; ++i) {
    sum += input[i];
    if (std::abs(input[i]) < 1e-8f)
      zero_count++;
  }

  float mean = sum / size;
  __m256 v_mean = _mm256_set1_ps(mean);
  __m256 v_variance = _mm256_setzero_ps();

  i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    __m256 diff = _mm256_sub_ps(v, v_mean);
    v_variance = _mm256_fmadd_ps(diff, diff, v_variance);
  }

  float variance = hsum_ps_avx2(v_variance);
  for (; i < size; ++i) {
    float diff = input[i] - mean;
    variance += diff * diff;
  }
  variance /= size;
  float std_dev = std::sqrt(variance);
  float sparsity = static_cast<float>(zero_count) / size;

  float structure = 0.0f;
  if (size > 1 && variance > 1e-8f) {
    __m256 v_covariance = _mm256_setzero_ps();
    i = 0;
    for (; i + 8 <= size - 1; i += 8) {
      __m256 v1 = _mm256_loadu_ps(input + i);
      __m256 v2 = _mm256_loadu_ps(input + i + 1);
      __m256 d1 = _mm256_sub_ps(v1, v_mean);
      __m256 d2 = _mm256_sub_ps(v2, v_mean);
      v_covariance = _mm256_fmadd_ps(d1, d2, v_covariance);
    }
    float covariance = hsum_ps_avx2(v_covariance);
    for (; i < size - 1; ++i) {
      covariance += (input[i] - mean) * (input[i + 1] - mean);
    }
    covariance /= variance * (size - 1);
    structure = std::abs(covariance);
  }
  float complexity =
      0.4f * std_dev + 0.3f * (1.0f - sparsity) + 0.3f * structure;
  return std::max(0.0f, std::min(1.0f, complexity));
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_sparsity_ptr(const float *input,
                                            size_t size) const {
  if (size == 0)
    return 1.0f;

  __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  __m256 v_sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    v_sum = _mm256_add_ps(v_sum, _mm256_and_ps(v, abs_mask));
  }
  float l1_norm = hsum_ps_avx2(v_sum);
  for (; i < size; ++i)
    l1_norm += std::abs(input[i]);

  float normalized_l1 = l1_norm / size;
  if (normalized_l1 < l1_sparsity_threshold)
    return 1.0f;

  size_t zero_count = 0;
  __m256 v_eps = _mm256_set1_ps(1e-8f);
  __m256 v_neg_eps = _mm256_set1_ps(-1e-8f);

  i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    __m256 mask1 = _mm256_cmp_ps(v, v_eps, _CMP_LT_OQ);
    __m256 mask2 = _mm256_cmp_ps(v, v_neg_eps, _CMP_GT_OQ);
    __m256 mask = _mm256_and_ps(mask1, mask2);
    int imask = _mm256_movemask_ps(mask);
    zero_count += RESIDUE_POPCNT(imask);
  }
  for (; i < size; ++i) {
    if (std::abs(input[i]) < 1e-8f)
      zero_count++;
  }
  return static_cast<float>(zero_count) / size;
}

RESIDUE_FORCEINLINE float
EntropyControllerV3::calculate_structure_score_ptr(const float *input,
                                                   size_t size) const {
  if (size < 3)
    return 0.0f;
  float zcr = calculate_zero_crossing_rate_ptr(input, size);

  __m256 v_sum = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 8 <= size; i += 8)
    v_sum = _mm256_add_ps(v_sum, _mm256_loadu_ps(input + i));
  float sum = hsum_ps_avx2(v_sum);
  for (; i < size; ++i)
    sum += input[i];

  float mean = sum / size;
  __m256 v_mean = _mm256_set1_ps(mean);
  __m256 v_variance = _mm256_setzero_ps();

  i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 v = _mm256_loadu_ps(input + i);
    __m256 diff = _mm256_sub_ps(v, v_mean);
    v_variance = _mm256_fmadd_ps(diff, diff, v_variance);
  }

  float variance = hsum_ps_avx2(v_variance);
  for (; i < size; ++i) {
    float diff = input[i] - mean;
    variance += diff * diff;
  }
  variance /= size;
  if (variance <= 1e-8f)
    return 0.0f;

  __m256 v_autocorr = _mm256_setzero_ps();
  i = 0;
  for (; i + 8 <= size - 1; i += 8) {
    __m256 v1 = _mm256_loadu_ps(input + i);
    __m256 v2 = _mm256_loadu_ps(input + i + 1);
    __m256 d1 = _mm256_sub_ps(v1, v_mean);
    __m256 d2 = _mm256_sub_ps(v2, v_mean);
    v_autocorr = _mm256_fmadd_ps(d1, d2, v_autocorr);
  }
  float autocorr = hsum_ps_avx2(v_autocorr);
  for (; i < size - 1; ++i)
    autocorr += (input[i] - mean) * (input[i + 1] - mean);

  autocorr /= variance * (size - 1);
  float normalized_autocorr = std::abs(autocorr);
  float structure = 0.7f * normalized_autocorr + 0.3f * zcr;
  return std::max(0.0f, std::min(1.0f, structure));
}
