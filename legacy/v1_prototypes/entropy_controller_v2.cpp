#include "entropy_controller_v2.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    
    // Reserve history space for efficiency
    entropy_history.reserve(1000);
    scaling_history.reserve(1000);
    complexity_history.reserve(1000);
}

float EntropyControllerV2::calculate_input_entropy(const std::vector<float>& input) {
    if (input.empty() || input.size() < 2) {
        return 0.0f; // No entropy in empty or single-element data
    }
    
    // Find min and max values for normalization
    auto min_max = std::minmax_element(input.begin(), input.end());
    float min_val = *min_max.first;
    float max_val = *min_max.second;
    
    // Check for constant signal (zero variance)
    if (max_val - min_val < 1e-10f) {
        return 0.0f; // No entropy in constant signal
    }
    
    // Create histogram
    std::vector<float> histogram(num_bins, 0.0f);
    
    // Bin the data
    for (float value : input) {
        // Normalize to [0, 1]
        float normalized = (value - min_val) / (max_val - min_val);
        
        // Convert to bin index
        size_t bin = static_cast<size_t>(normalized * num_bins);
        bin = std::min(static_cast<size_t>(num_bins - 1), bin);
        
        histogram[bin] += 1.0f;
    }
    
    // Normalize histogram to probabilities
    float total_samples = static_cast<float>(input.size());
    for (float& count : histogram) {
        count /= total_samples;
    }
    
    // Calculate Shannon entropy
    float entropy = 0.0f;
    for (float prob : histogram) {
        if (prob > 0.0f) {
            entropy += -prob * std::log2(prob);
        }
    }
    
    return entropy;
}

float EntropyControllerV2::calculate_complexity_score(const std::vector<float>& input) {
    if (input.size() < 3) {
        return 0.0f; // Cannot calculate complexity for very small arrays
    }
    
    // Multi-dimensional complexity measure
    float std_dev = calculate_standard_deviation(input);
    float sparsity = calculate_sparsity(input);
    float structure = calculate_structure_score(input);
    
    // Combine features with weights
    float complexity = 0.4f * std_dev + 0.3f * (1.0f - sparsity) + 0.3f * structure;
    
    // Normalize to [0, 1] range
    return std::max(0.0f, std::min(1.0f, complexity));
}

float EntropyControllerV2::compute_softmax_scaling(float entropy, float complexity_score) {
    // Create feature vector for softmax
    float features[3] = {entropy, complexity_score, 1.0f}; // Add bias term
    
    // Compute softmax weights with numerical stability
    const float epsilon = 1e-8f;
    float exp_sum = std::exp(features[0]) + std::exp(features[1]) + std::exp(features[2]) + epsilon;
    float weights[3] = {
        std::exp(features[0]) / exp_sum,
        std::exp(features[1]) / exp_sum,
        std::exp(features[2]) / exp_sum
    };
    
    // Analog scaling based on weighted features
    // Map weights to scaling factors
    float scaling = 0.1f * weights[2] + 1.0f * weights[1] + 10.0f * (1.0f - weights[0] - weights[1]);
    
    // Ensure scaling is within bounds
    scaling = std::max(min_scaling_factor, std::min(max_scaling_factor, scaling));
    
    return scaling;
}

float EntropyControllerV2::compute_adaptive_scaling(const std::vector<float>& input) {
    float entropy = calculate_input_entropy(input);
    float complexity = calculate_complexity_score(input);
    
    // Use softmax-based scaling for smooth transitions
    float scaling = compute_softmax_scaling(entropy, complexity);
    
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
    // Multi-dimensional softmax with 4 features
    float feature_array[4] = {features.entropy, features.complexity, features.sparsity, features.structure};
    
    // Compute softmax with numerical stability
    const float epsilon = 1e-8f;
    float exp_sum = std::exp(feature_array[0]) + std::exp(feature_array[1]) + 
                   std::exp(feature_array[2]) + std::exp(feature_array[3]) + epsilon;
    
    float weights[4];
    for (int i = 0; i < 4; i++) {
        weights[i] = std::exp(feature_array[i]) / exp_sum;
    }
    
    // Map to scaling factors
    // High entropy → high scaling (less computation)
    // High complexity → medium scaling (balanced)
    // High sparsity → high scaling (less computation)
    // High structure → low scaling (more computation)
    float scaling = 10.0f * weights[0] + 5.0f * weights[1] + 8.0f * weights[2] + 2.0f * weights[3];
    
    return std::max(min_scaling_factor, std::min(max_scaling_factor, scaling));
}

float EntropyControllerV2::sigmoid_scaling(float x, float midpoint, float steepness) {
    // Sigmoid function for smooth transitions
    float sigmoid = 1.0f / (1.0f + std::exp(-steepness * (x - midpoint)));
    
    // Map sigmoid [0,1] to scaling range
    return min_scaling_factor + sigmoid * (max_scaling_factor - min_scaling_factor);
}

float EntropyControllerV2::linear_interpolation_scaling(float entropy, float min_entropy, float max_entropy) {
    // Linear interpolation for smooth scaling
    if (entropy <= min_entropy) {
        return max_scaling_factor;
    } else if (entropy >= max_entropy) {
        return min_scaling_factor;
    } else {
        // Linear interpolation between max and min scaling
        float ratio = (entropy - min_entropy) / (max_entropy - min_entropy);
        return max_scaling_factor + ratio * (min_scaling_factor - max_scaling_factor);
    }
}

float EntropyControllerV2::smoothstep_scaling(float edge0, float edge1, float x) {
    // Smoothstep function for very smooth transitions
    if (x <= edge0) {
        return max_scaling_factor;
    } else if (x >= edge1) {
        return min_scaling_factor;
    } else {
        // Normalize to [0,1]
        float t = (x - edge0) / (edge1 - edge0);
        // Smoothstep: 3t² - 2t³
        float smooth = t * t * (3.0f - 2.0f * t);
        return max_scaling_factor + smooth * (min_scaling_factor - max_scaling_factor);
    }
}

void EntropyControllerV2::update_performance_history(float entropy, float scaling, float complexity) {
    entropy_history.push_back(entropy);
    scaling_history.push_back(scaling);
    complexity_history.push_back(complexity);
    
    // Keep history size manageable
    const size_t max_history = 1000;
    if (entropy_history.size() > max_history) {
        entropy_history.erase(entropy_history.begin());
        scaling_history.erase(scaling_history.begin());
        complexity_history.erase(complexity_history.begin());
    }
}

float EntropyControllerV2::calculate_standard_deviation(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0f;
    }
    
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float variance = 0.0f;
    
    for (float value : data) {
        variance += (value - mean) * (value - mean);
    }
    
    variance /= data.size();
    return std::sqrt(variance);
}

float EntropyControllerV2::calculate_sparsity(const std::vector<float>& data) {
    if (data.empty()) {
        return 1.0f; // Empty data is fully sparse
    }
    
    // Count near-zero elements
    const float threshold = 1e-6f;
    size_t zero_count = 0;
    
    for (float value : data) {
        if (std::abs(value) < threshold) {
            zero_count++;
        }
    }
    
    return static_cast<float>(zero_count) / data.size();
}

float EntropyControllerV2::calculate_structure_score(const std::vector<float>& data) {
    if (data.size() < 3) {
        return 0.0f;
    }
    
    // Calculate autocorrelation as a measure of structure
    float autocorr = 0.0f;
    size_t n = data.size();
    
    // Mean for centering
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / n;
    
    // Autocorrelation with lag 1
    for (size_t i = 0; i < n - 1; i++) {
        autocorr += (data[i] - mean) * (data[i + 1] - mean);
    }
    
    // Normalize
    float variance = 0.0f;
    for (float value : data) {
        variance += (value - mean) * (value - mean);
    }
    
    if (variance > 0.0f) {
        autocorr /= variance;
    }
    
    // Return absolute value as structure score
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
