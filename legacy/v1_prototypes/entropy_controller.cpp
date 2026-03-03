#include "entropy_controller.h"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cassert>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float EntropyController::calculate_standard_deviation(const std::vector<float>& data) {
    if (data.size() < 2) return 0.0f;
    
    float mean = std::accumulate(data.begin(), data.end(), 0.0f) / data.size();
    float variance = 0.0f;
    
    for (float value : data) {
        variance += (value - mean) * (value - mean);
    }
    
    return std::sqrt(variance / data.size());
}

EntropyController::EntropyController(int bins, float threshold)
    : num_bins(bins), entropy_threshold(threshold), 
      min_scaling_factor(0.1f), max_scaling_factor(10.0f),
      compute_budget(1.0f), current_budget_usage(0.0f), budget_efficiency_target(0.8f),
      adaptation_rate(0.1f), smoothing_factor(0.9f), last_entropy(0.0f),
      current_scaling_factor(1.0f), max_history_size(100),
      average_processing_time(0.0f) {
    
    entropy_history.reserve(max_history_size);
    scaling_history.reserve(max_history_size);
    efficiency_history.reserve(max_history_size);
    
    last_update_time = std::chrono::high_resolution_clock::now();
}

float EntropyController::calculate_input_entropy(const std::vector<float>& input) {
    if (input.empty()) return 0.0f;
    
    // Create histogram for entropy calculation
    std::vector<float> histogram(num_bins, 0.0f);
    
    // Find min and max for normalization
    auto min_max = std::minmax_element(input.begin(), input.end());
    float min_val = *min_max.first;
    float max_val = *min_max.second;
    
    // Handle edge case of all equal values
    if (max_val - min_val < 1e-10f) {
        return 0.0f; // No entropy in constant signal
    }
    
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
    return calculate_shannon_entropy(histogram);
}

float EntropyController::calculate_shannon_entropy(const std::vector<float>& histogram) {
    float entropy = 0.0f;
    
    for (float probability : histogram) {
        if (probability > 1e-10f) { // Avoid log(0)
            entropy -= probability * std::log2(probability + 1e-10f);
        }
    }
    
    return entropy;
}

float EntropyController::calculate_differential_entropy(const std::vector<float>& input) {
    if (input.size() < 2) return 0.0f;
    
    // Estimate probability density using kernel density estimation
    float bandwidth = 1.06f * std::sqrt(static_cast<float>(input.size())) * 
                     calculate_standard_deviation(input) / 
                     std::pow(static_cast<float>(input.size()), 0.2f);
    
    float differential_entropy = 0.0f;
    float normalization = 0.0f;
    
    // Sample points for density estimation
    int num_samples = std::min(1000, static_cast<int>(input.size()));
    float step = static_cast<float>(input.size()) / num_samples;
    
    for (int i = 0; i < num_samples; i++) {
        int idx = static_cast<int>(i * step);
        float x = input[idx];
        
        // Estimate density at point x
        float density = 0.0f;
        for (float sample : input) {
            float diff = (x - sample) / bandwidth;
            density += std::exp(-0.5f * diff * diff);
        }
        density /= (input.size() * bandwidth * std::sqrt(2.0f * M_PI));
        
        if (density > 1e-10f) {
            differential_entropy -= std::log(density);
            normalization += 1.0f;
        }
    }
    
    return normalization > 0 ? differential_entropy / normalization : 0.0f;
}

float EntropyController::compute_scaling_factor(float entropy) {
    // Adaptive scaling based on entropy relative to threshold
    float entropy_ratio = entropy / entropy_threshold;
    
    if (entropy_ratio <= 1.0f) {
        // Low entropy: reduce computation
        float scaling = min_scaling_factor + 
                       (1.0f - min_scaling_factor) * entropy_ratio;
        return scaling;
    } else {
        // High entropy: increase computation, but cap at maximum
        float scaling = 1.0f + 
                       (max_scaling_factor - 1.0f) * 
                       std::tanh(entropy_ratio - 1.0f);
        return scaling;
    }
}

void EntropyController::update_scaling_factor(float new_entropy) {
    // Record current entropy
    last_entropy = new_entropy;
    
    // Compute new scaling factor
    float new_scaling = compute_scaling_factor(new_entropy);
    
    // Smooth transition to avoid oscillations
    smooth_scaling_transition();
    
    // Update history
    entropy_history.push_back(new_entropy);
    scaling_history.push_back(current_scaling_factor);
    
    // Trim history if too long
    if (entropy_history.size() > max_history_size) {
        entropy_history.erase(entropy_history.begin());
        scaling_history.erase(scaling_history.begin());
    }
}

void EntropyController::smooth_scaling_transition() {
    // Apply exponential smoothing to prevent rapid oscillations
    float target_scaling = compute_scaling_factor(last_entropy);
    current_scaling_factor = smoothing_factor * current_scaling_factor + 
                           (1.0f - smoothing_factor) * target_scaling;
    
    // Ensure scaling stays within bounds
    current_scaling_factor = std::max(min_scaling_factor, 
                                      std::min(max_scaling_factor, current_scaling_factor));
}

void EntropyController::set_compute_budget(float budget) {
    compute_budget = budget;
    current_budget_usage = 0.0f;
}

float EntropyController::get_remaining_budget() const {
    return compute_budget - current_budget_usage;
}

void EntropyController::allocate_processing_budget(float entropy) {
    float scaling = compute_scaling_factor(entropy);
    float budget_allocation = scaling / max_scaling_factor; // Normalize to [0, 1]
    
    current_budget_usage += budget_allocation;
    
    // If over budget, reduce scaling factor
    if (current_budget_usage > compute_budget) {
        float reduction_factor = compute_budget / current_budget_usage;
        current_scaling_factor *= reduction_factor;
        current_budget_usage = compute_budget;
    }
}

void EntropyController::optimize_for_efficiency() {
    if (efficiency_history.size() < 10) return; // Need sufficient history
    
    // Calculate recent efficiency trend
    float recent_efficiency = 0.0f;
    int count = std::min(10, static_cast<int>(efficiency_history.size()));
    
    for (int i = efficiency_history.size() - count; i < efficiency_history.size(); i++) {
        recent_efficiency += efficiency_history[i];
    }
    recent_efficiency /= count;
    
    // Adjust adaptation parameters based on efficiency
    if (recent_efficiency < budget_efficiency_target) {
        // Low efficiency: be more conservative
        adaptation_rate *= 0.9f;
        smoothing_factor = std::min(0.95f, smoothing_factor + 0.01f);
    } else if (recent_efficiency > budget_efficiency_target * 1.1f) {
        // High efficiency: can be more aggressive
        adaptation_rate = std::min(0.2f, adaptation_rate * 1.1f);
        smoothing_factor = std::max(0.8f, smoothing_factor - 0.01f);
    }
}

void EntropyController::monitor_performance(float processing_time, float accuracy) {
    // Calculate efficiency score (accuracy per unit time)
    float efficiency = accuracy / (processing_time + 1e-10f);
    efficiency_history.push_back(efficiency);
    
    // Trim history
    if (efficiency_history.size() > max_history_size) {
        efficiency_history.erase(efficiency_history.begin());
    }
    
    // Update average processing time
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - last_update_time).count() / 1000.0f; // Convert to milliseconds
    
    average_processing_time = 0.9f * average_processing_time + 0.1f * duration;
    last_update_time = now;
    
    // Optimize based on performance
    optimize_for_efficiency();
}

float EntropyController::get_efficiency_score() const {
    if (efficiency_history.empty()) return 0.0f;
    
    // Return average of recent efficiency scores
    float sum = std::accumulate(efficiency_history.begin(), efficiency_history.end(), 0.0f);
    return sum / efficiency_history.size();
}

void EntropyController::reset_history() {
    entropy_history.clear();
    scaling_history.clear();
    efficiency_history.clear();
    current_scaling_factor = 1.0f;
    last_entropy = 0.0f;
    current_budget_usage = 0.0f;
}

void EntropyController::set_adaptation_parameters(float rate, float smoothing) {
    adaptation_rate = std::max(0.01f, std::min(1.0f, rate));
    smoothing_factor = std::max(0.1f, std::min(0.99f, smoothing));
}

std::vector<float> EntropyController::get_entropy_statistics() const {
    if (entropy_history.empty()) {
        return {0.0f, 0.0f, 0.0f}; // mean, std, trend
    }
    
    // Calculate mean
    float mean = std::accumulate(entropy_history.begin(), entropy_history.end(), 0.0f) / 
                entropy_history.size();
    
    // Calculate standard deviation
    float variance = 0.0f;
    for (float entropy : entropy_history) {
        variance += (entropy - mean) * (entropy - mean);
    }
    float std_dev = std::sqrt(variance / entropy_history.size());
    
    // Calculate trend (linear regression slope)
    float trend = 0.0f;
    if (entropy_history.size() > 1) {
        float n = static_cast<float>(entropy_history.size());
        float sum_x = n * (n - 1) / 2.0f;
        float sum_y = std::accumulate(entropy_history.begin(), entropy_history.end(), 0.0f);
        float sum_xy = 0.0f;
        float sum_x2 = n * (n - 1) * (2 * n - 1) / 6.0f;
        
        for (size_t i = 0; i < entropy_history.size(); i++) {
            sum_xy += i * entropy_history[i];
        }
        
        trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
    }
    
    return {mean, std_dev, trend};
}

bool EntropyController::should_increase_resolution() const {
    if (entropy_history.size() < 3) return false;
    
    // Check if recent entropy is increasing and above threshold
    float recent_avg = 0.0f;
    for (int i = std::max(0, static_cast<int>(entropy_history.size()) - 3); 
         i < entropy_history.size(); i++) {
        recent_avg += entropy_history[i];
    }
    recent_avg /= 3.0f;
    
    return recent_avg > entropy_threshold && current_scaling_factor < max_scaling_factor;
}

bool EntropyController::should_decrease_resolution() const {
    if (entropy_history.size() < 3) return false;
    
    // Check if recent entropy is decreasing and below threshold
    float recent_avg = 0.0f;
    for (int i = std::max(0, static_cast<int>(entropy_history.size()) - 3); 
         i < entropy_history.size(); i++) {
        recent_avg += entropy_history[i];
    }
    recent_avg /= 3.0f;
    
    return recent_avg < entropy_threshold && current_scaling_factor > min_scaling_factor;
}

void EntropyController::set_entropy_threshold(float threshold) {
    entropy_threshold = std::max(0.01f, threshold);
}

void EntropyController::set_scaling_range(float min_factor, float max_factor) {
    min_scaling_factor = std::max(0.01f, min_factor);
    max_scaling_factor = std::max(min_scaling_factor, max_factor);
    
    // Ensure current scaling is within new bounds
    current_scaling_factor = std::max(min_scaling_factor, 
                                      std::min(max_scaling_factor, current_scaling_factor));
}

void EntropyController::set_budget_efficiency_target(float target) {
    budget_efficiency_target = std::max(0.1f, std::min(1.0f, target));
}

// Factory function implementation
std::unique_ptr<EntropyController> create_entropy_controller(
    int num_bins, float entropy_threshold) {
    return std::make_unique<EntropyController>(num_bins, entropy_threshold);
}
