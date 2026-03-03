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

EntropyControllerV2::EntropyControllerV2(int bins, float threshold)
    : num_bins(bins),
      entropy_threshold(threshold),
      min_scaling_factor(0.1f),
      max_scaling_factor(10.0f),
      current_scaling_factor(1.0f),
      total_samples_processed(0) {
    
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

EntropyControllerV2::FeatureVector EntropyControllerV2::extract_features(const std::vector<float>& input) {
    FeatureVector features;
    features.entropy = calculate_input_entropy(input);
    features.complexity = calculate_complexity_score(input);
    features.sparsity = calculate_sparsity(input);
    features.structure = calculate_structure_score(input);
    
    return features;
}

float EntropyControllerV2::compute_multi_dimensional_scaling(const FeatureVector& features) {
    // Optimized multi-dimensional softmax with 4 features
    const float feature_array[4] = {features.entropy, features.complexity, 
                                   features.sparsity, features.structure};
    
    // Compute softmax with numerical stability
    float exp_sum = EPSILON;
    float exp_features[4];
    
    for (int i = 0; i < 4; ++i) {
        exp_features[i] = std::exp(feature_array[i]);
        exp_sum += exp_features[i];
    }
    
    const float exp_sum_inv = 1.0f / exp_sum;
    float weights[4];
    for (int i = 0; i < 4; ++i) {
        weights[i] = exp_features[i] * exp_sum_inv;
    }
    
    // Optimized scaling calculation
    const float scaling = 10.0f * weights[0] + 5.0f * weights[1] + 
                         8.0f * weights[2] + 2.0f * weights[3];
    
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
    
    // Optimized sparsity calculation
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
    
    // Optimized autocorrelation calculation
    const float mean = std::accumulate(data.begin(), data.end(), 0.0f) / size;
    float variance = 0.0f;
    
    for (const float value : data) {
        const float diff = value - mean;
        variance += diff * diff;
    }
    
    if (variance <= EPSILON) {
        return 0.0f;
    }
    
    // Calculate autocorrelation with lag 1
    float autocorr = 0.0f;
    for (size_t i = 0; i < size - 1; ++i) {
        autocorr += (data[i] - mean) * (data[i + 1] - mean);
    }
    
    autocorr /= variance * (size - 1);
    return std::abs(autocorr);
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

void EntropyControllerV2::reset_history() {
    entropy_history.clear();
    scaling_history.clear();
    complexity_history.clear();
    total_samples_processed = 0;
    current_scaling_factor = 1.0f;
}

std::unique_ptr<EntropyControllerV2> EntropyControllerV2::clone() const {
    return std::make_unique<EntropyControllerV2>(num_bins, entropy_threshold);
}

// Factory function
std::unique_ptr<EntropyControllerV2> create_entropy_controller_v2(
    int num_bins, float entropy_threshold) {
    return std::make_unique<EntropyControllerV2>(num_bins, entropy_threshold);
}
