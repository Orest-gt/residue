#include "core.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Optimized constants
constexpr float EPSILON = 1e-8f;
constexpr float TEMPERATURE = 1.0f;
constexpr size_t MAX_HISTORY_SIZE = 1000;

EntropyControllerV2::EntropyControllerV2(int bins, float threshold, size_t temporal_buffer_size, float l1_threshold, float ema_alpha_param)
    : num_bins(bins),
      entropy_threshold(threshold),
      min_scaling_factor(0.1f),
      max_scaling_factor(10.0f),
      current_scaling_factor(1.0f),
      total_samples_processed(0),
      l1_sparsity_threshold(l1_threshold),
      zcr_window_size(std::max(size_t(2), temporal_buffer_size)),
      ema_alpha(ema_alpha_param) {
    
    // Validate parameters
    if (num_bins <= 0) {
        throw std::invalid_argument("Number of bins must be positive");
    }
    if (threshold <= 0.0f) {
        throw std::invalid_argument("Entropy threshold must be positive");
    }
    
    // Pre-allocate history for performance
    entropy_history.reserve(MAX_HISTORY_SIZE);
    scaling_history.reserve(MAX_HISTORY_SIZE);
    complexity_history.reserve(MAX_HISTORY_SIZE);
    
    // Pre-allocate histogram for entropy calculation
    histogram_buffer.resize(num_bins, 0.0f);
    
    // Initialize V3.0 structural heuristics
    temporal_buffer = std::make_unique<CircularBuffer>(temporal_buffer_size, ema_alpha);
}

float EntropyControllerV2::calculate_input_entropy(const std::vector<float>& input) {
    if (input.empty() || input.size() < 2) {
        return 0.0f; // No entropy in empty or single-element data
    }
    
    // Optimized min/max calculation
    const auto [min_it, max_it] = std::minmax_element(input.begin(), input.end());
    const float min_val = *min_it;
    const float max_val = *max_it;
    
    // Check for constant signal (zero variance)
    if (max_val - min_val < EPSILON) {
        return 0.0f; // No entropy in constant signal
    }
    
    // Clear histogram buffer
    std::fill(histogram_buffer.begin(), histogram_buffer.end(), 0.0f);
    
    // Optimized binning with range normalization
    const float range_inv = 1.0f / (max_val - min_val);
    const float num_bins_minus_1 = static_cast<float>(num_bins - 1);
    
    for (const float value : input) {
        // Normalize to [0, 1]
        const float normalized = (value - min_val) * range_inv;
        
        // Convert to bin index with bounds checking
        const size_t bin = std::min(static_cast<size_t>(normalized * num_bins_minus_1), 
                                   static_cast<size_t>(num_bins - 1));
        
        histogram_buffer[bin] += 1.0f;
    }
    
    // Normalize histogram to probabilities
    const float total_samples_inv = 1.0f / static_cast<float>(input.size());
    float entropy = 0.0f;
    
    for (const float prob : histogram_buffer) {
        if (prob > 0.0f) {
            const float normalized_prob = prob * total_samples_inv;
            entropy += -normalized_prob * std::log2(normalized_prob);
        }
    }
    
    return entropy;
}

float EntropyControllerV2::calculate_complexity_score(const std::vector<float>& input) {
    const size_t size = input.size();
    if (size < 3) {
        return 0.0f; // Cannot calculate complexity for very small arrays
    }
    
    // Optimized standard deviation calculation
    const float mean = std::accumulate(input.begin(), input.end(), 0.0f) / size;
    float variance = 0.0f;
    
    for (const float value : input) {
        const float diff = value - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    const float std_dev = std::sqrt(variance);
    
    // Optimized sparsity calculation
    size_t zero_count = 0;
    for (const float value : input) {
        if (std::abs(value) < EPSILON) {
            zero_count++;
        }
    }
    const float sparsity = static_cast<float>(zero_count) / size;
    
    // Optimized structure calculation (autocorrelation)
    float structure = 0.0f;
    if (size > 1) {
        float covariance = 0.0f;
        for (size_t i = 0; i < size - 1; ++i) {
            covariance += (input[i] - mean) * (input[i + 1] - mean);
        }
        covariance /= variance * (size - 1);
        structure = std::abs(covariance);
    }
    
    // Combine features with weights
    const float complexity = 0.4f * std_dev + 0.3f * (1.0f - sparsity) + 0.3f * structure;
    
    // Normalize to [0, 1] range
    return std::max(0.0f, std::min(1.0f, complexity));
}

float EntropyControllerV2::compute_softmax_scaling(float entropy, float complexity_score) {
    // Optimized softmax with numerical stability
    const float features[3] = {entropy, complexity_score, 1.0f};
    
    // Compute softmax weights with temperature control
    float exp_sum = EPSILON; // Start with epsilon to prevent division by zero
    float exp_features[3];
    
    for (int i = 0; i < 3; ++i) {
        exp_features[i] = std::exp(features[i] / TEMPERATURE);
        exp_sum += exp_features[i];
    }
    
    const float exp_sum_inv = 1.0f / exp_sum;
    float weights[3];
    for (int i = 0; i < 3; ++i) {
        weights[i] = exp_features[i] * exp_sum_inv;
    }
    
    // Optimized scaling calculation
    const float scaling = 0.1f * weights[2] + 1.0f * weights[1] + 
                         10.0f * (1.0f - weights[0] - weights[1]);
    
    // Ensure scaling is within bounds
    return std::max(min_scaling_factor, std::min(max_scaling_factor, scaling));
}

float EntropyControllerV2::compute_adaptive_scaling(const std::vector<float>& input) {
    const float entropy = calculate_input_entropy(input);
    const float complexity = calculate_complexity_score(input);
    
    // Use optimized softmax-based scaling
    const float scaling = compute_softmax_scaling(entropy, complexity);
    
    // Update performance tracking
    update_performance_history(entropy, scaling, complexity);
    total_samples_processed++;
    
    current_scaling_factor = scaling;
    return scaling;
}

EntropyControllerV2::FeatureVectorV3 EntropyControllerV2::extract_features_v3(const std::vector<float>& input) {
    FeatureVectorV3 features;
    features.entropy = calculate_input_entropy(input);
    features.complexity = calculate_complexity_score(input);
    features.sparsity = calculate_sparsity(input);
    features.structure = calculate_structure_score(input);
    
    // V3.0 structural heuristics
    features.temporal_coherence = compute_temporal_coherence(current_scaling_factor);
    features.zcr_rate = calculate_zero_crossing_rate(input);
    features.l1_sparsity = calculate_l1_norm_sparsity(input);
    
    return features;
}

float EntropyControllerV2::compute_multi_dimensional_scaling_v3(const FeatureVectorV3& features) {
    // V3.0 Enhanced multi-dimensional softmax with 7 features
    const float feature_array[7] = {
        features.entropy, 
        features.complexity, 
        features.sparsity, 
        features.structure,
        features.temporal_coherence,
        features.zcr_rate,
        features.l1_sparsity
    };
    
    // Compute softmax with numerical stability
    float exp_sum = EPSILON;
    float exp_features[7];
    
    for (int i = 0; i < 7; ++i) {
        exp_features[i] = std::exp(feature_array[i]);
        exp_sum += exp_features[i];
    }
    
    const float exp_sum_inv = 1.0f / exp_sum;
    float weights[7];
    for (int i = 0; i < 7; ++i) {
        weights[i] = exp_features[i] * exp_sum_inv;
    }
    
    // V3.0 Enhanced scaling calculation with structural heuristics
    const float scaling = 15.0f * weights[0] + 10.0f * weights[1] + 
                         8.0f * weights[2] + 5.0f * weights[3] +
                         3.0f * weights[4] + 2.0f * weights[5] +
                         1.0f * weights[6];
    
    return std::max(min_scaling_factor, std::min(max_scaling_factor, scaling));
}

float EntropyControllerV2::sigmoid_scaling(float x, float midpoint, float steepness) {
    // Optimized sigmoid function
    const float exponent = -steepness * (x - midpoint);
    const float sigmoid = 1.0f / (1.0f + std::exp(exponent));
    
    // Map sigmoid [0,1] to scaling range
    return min_scaling_factor + sigmoid * (max_scaling_factor - min_scaling_factor);
}

float EntropyControllerV2::linear_interpolation_scaling(float entropy, float min_entropy, float max_entropy) {
    // Optimized linear interpolation
    if (entropy <= min_entropy) {
        return max_scaling_factor;
    } else if (entropy >= max_entropy) {
        return min_scaling_factor;
    } else {
        const float ratio = (entropy - min_entropy) / (max_entropy - min_entropy);
        return max_scaling_factor + ratio * (min_scaling_factor - max_scaling_factor);
    }
}

float EntropyControllerV2::smoothstep_scaling(float edge0, float edge1, float x) {
    // Optimized smoothstep function
    if (x <= edge0) {
        return max_scaling_factor;
    } else if (x >= edge1) {
        return min_scaling_factor;
    } else {
        // Normalize to [0,1]
        const float t = (x - edge0) / (edge1 - edge0);
        // Smoothstep: 3t² - 2t³
        const float smooth = t * t * (3.0f - 2.0f * t);
        return max_scaling_factor + smooth * (min_scaling_factor - max_scaling_factor);
    }
}

void EntropyControllerV2::update_performance_history(float entropy, float scaling, float complexity) {
    entropy_history.push_back(entropy);
    scaling_history.push_back(scaling);
    complexity_history.push_back(complexity);
    
    // Maintain maximum history size
    if (entropy_history.size() > MAX_HISTORY_SIZE) {
        entropy_history.erase(entropy_history.begin());
        scaling_history.erase(scaling_history.begin());
        complexity_history.erase(complexity_history.begin());
    }
}

float EntropyControllerV2::calculate_standard_deviation(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f;
    }
    
    const float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float variance = 0.0f;
    
    for (const float value : data) {
        const float diff = value - mean;
        variance += diff * diff;
    }
    
    return std::sqrt(variance / data.size());
}

float EntropyControllerV2::calculate_sparsity(const std::vector<float>& data) {
    if (data.empty()) {
        return 1.0f; // Empty data is fully sparse
    }
    
    // V3.0: L1-norm sparsity detection
    float l1_norm = 0.0f;
    for (const float value : data) {
        l1_norm += std::abs(value);
    }
    
    // Normalize by vector length
    const float normalized_l1 = l1_norm / data.size();
    
    // Check against threshold
    if (normalized_l1 < l1_sparsity_threshold) {
        return 1.0f; // Sparse data
    }
    
    // Fallback to original sparsity calculation
    size_t zero_count = 0;
    for (const float value : data) {
        if (std::abs(value) < EPSILON) {
            zero_count++;
        }
    }
    
    return static_cast<float>(zero_count) / data.size();
}

float EntropyControllerV2::calculate_structure_score(const std::vector<float>& data) {
    const size_t size = data.size();
    if (size < 3) {
        return 0.0f;
    }
    
    // V3.0: Zero-crossing rate calculation
    float zcr = calculate_zero_crossing_rate(data);
    
    // Optimized autocorrelation calculation
    const float mean = std::accumulate(data.begin(), data.end(), 0.0f) / size;
    float variance = 0.0f;
    
    for (const float value : data) {
        const float diff = value - mean;
        variance += diff * diff;
    }
    variance /= size;
    
    if (variance <= EPSILON) {
        return 0.0f;
    }
    
    // Calculate autocorrelation with lag 1
    float autocorr = 0.0f;
    for (size_t i = 0; i < size - 1; ++i) {
        autocorr += (data[i] - mean) * (data[i + 1] - mean);
    }
    autocorr /= variance * (size - 1);
    
    // Combine ZCR with autocorrelation for enhanced structure score
    const float normalized_autocorr = std::abs(autocorr);
    const float structure = 0.7f * normalized_autocorr + 0.3f * zcr;
    
    return std::max(0.0f, std::min(1.0f, structure));
}

float EntropyControllerV2::calculate_zero_crossing_rate(const std::vector<float>& data) {
    if (data.size() < 2) {
        return 0.0f;
    }
    
    // Check for invalid values
    for (const float value : data) {
        if (std::isnan(value) || std::isinf(value)) {
            return 0.0f; // Invalid data, return zero ZCR
        }
    }
    
    // Count zero crossings
    size_t zero_crossings = 0;
    for (size_t i = 0; i < data.size() - 1; ++i) {
        const float current = data[i];
        const float next = data[i + 1];
        
        // Check for zero crossing (sign change)
        if ((current >= 0.0f && next < 0.0f) || 
            (current < 0.0f && next >= 0.0f)) {
            zero_crossings++;
        }
    }
    
    // Normalize by number of possible crossings
    const float zcr = static_cast<float>(zero_crossings) / (data.size() - 1);
    return zcr;
}

float EntropyControllerV2::compute_temporal_coherence(float current_scaling) {
    // Push current scaling to temporal buffer
    temporal_buffer->push(current_scaling);
    
    // Return EMA as temporal coherence measure
    return temporal_buffer->get_ema();
}

void EntropyControllerV2::set_entropy_threshold(float threshold) {
    if (threshold <= 0.0f) {
        throw std::invalid_argument("Entropy threshold must be positive");
    }
    entropy_threshold = threshold;
}

void EntropyControllerV2::set_scaling_range(float min_factor, float max_factor) {
    if (min_factor <= 0.0f || max_factor <= 0.0f || min_factor >= max_factor) {
        throw std::invalid_argument("Invalid scaling range");
    }
    min_scaling_factor = min_factor;
    max_scaling_factor = max_factor;
}

void EntropyControllerV2::set_num_bins(int bins) {
    if (bins <= 0) {
        throw std::invalid_argument("Number of bins must be positive");
    }
    num_bins = bins;
    histogram_buffer.resize(num_bins, 0.0f);
}

// V3.0 Structural heuristics configuration
void EntropyControllerV2::set_temporal_buffer_size(size_t buffer_size) {
    temporal_buffer = std::make_unique<CircularBuffer>(buffer_size, 0.2f);
}

void EntropyControllerV2::set_l1_sparsity_threshold(float threshold) {
    l1_sparsity_threshold = threshold;
}

void EntropyControllerV2::set_zcr_window_size(size_t window_size) {
    zcr_window_size = std::max(size_t(2), window_size);
}

void EntropyControllerV2::set_ema_alpha(float alpha) {
    if (alpha < 0.0f || alpha > 1.0f) {
        throw std::invalid_argument("EMA alpha must be between 0.0 and 1.0");
    }
    ema_alpha = alpha;
    // Reinitialize temporal buffer with new alpha
    if (temporal_buffer) {
        size_t buffer_size = temporal_buffer->get_size();
        temporal_buffer = std::make_unique<CircularBuffer>(buffer_size, alpha);
    }
}

float EntropyControllerV2::get_ema_alpha() const {
    return ema_alpha;
}

void EntropyControllerV2::reset_history() {
    entropy_history.clear();
    scaling_history.clear();
    complexity_history.clear();
    total_samples_processed = 0;
    current_scaling_factor = 1.0f;
}

std::unique_ptr<EntropyControllerV2> EntropyControllerV2::clone() const {
    return std::make_unique<EntropyControllerV2>(num_bins, entropy_threshold, 
                                               zcr_window_size, l1_sparsity_threshold, ema_alpha);
}

// Factory function
std::unique_ptr<EntropyControllerV2> create_entropy_controller_v2(
    int num_bins, float entropy_threshold,
    size_t temporal_buffer_size, float l1_threshold, float ema_alpha) {
    return std::make_unique<EntropyControllerV2>(num_bins, entropy_threshold, 
                                               temporal_buffer_size, l1_threshold, ema_alpha);
}
