#pragma once

#include <vector>
#include <cmath>
#include <memory>
#include <chrono>

// Entropy-Driven Temporal Scaling Controller
// Adapts computational resolution based on input entropy for efficient processing

class EntropyController {
private:
    // Entropy calculation parameters
    int num_bins;
    float entropy_threshold;
    float min_scaling_factor;
    float max_scaling_factor;
    
    // Computational budget
    float compute_budget;
    float current_budget_usage;
    float budget_efficiency_target;
    
    // Adaptive parameters
    float adaptation_rate;
    float smoothing_factor;
    float last_entropy;
    float current_scaling_factor;
    
    // Performance monitoring
    std::vector<float> entropy_history;
    std::vector<float> scaling_history;
    std::vector<float> efficiency_history;
    int max_history_size;
    
    // Timing
    std::chrono::high_resolution_clock::time_point last_update_time;
    float average_processing_time;
    
public:
    // Constructor
    EntropyController(int bins = 256, float threshold = 0.1f);
    
    // Core entropy operations
    float calculate_input_entropy(const std::vector<float>& input);
    float calculate_shannon_entropy(const std::vector<float>& histogram);
    float calculate_differential_entropy(const std::vector<float>& input);
    
    // Adaptive scaling operations
    float compute_scaling_factor(float entropy);
    void update_scaling_factor(float new_entropy);
    void smooth_scaling_transition();
    
    // Budget management
    void set_compute_budget(float budget);
    float get_remaining_budget() const;
    void allocate_processing_budget(float entropy);
    
    // Performance optimization
    void optimize_for_efficiency();
    void monitor_performance(float processing_time, float accuracy);
    float get_efficiency_score() const;
    
    // Utility functions
    void reset_history();
    void set_adaptation_parameters(float rate, float smoothing);
    std::vector<float> get_entropy_statistics() const;
    bool should_increase_resolution() const;
    bool should_decrease_resolution() const;
    
    // Helper function
    float calculate_standard_deviation(const std::vector<float>& data);
    
    // Accessors
    float get_current_scaling_factor() const { return current_scaling_factor; }
    float get_entropy_threshold() const { return entropy_threshold; }
    float get_compute_budget() const { return compute_budget; }
    
    // Configuration
    void set_entropy_threshold(float threshold);
    void set_scaling_range(float min_factor, float max_factor);
    void set_budget_efficiency_target(float target);
};

// Factory function for easy creation
std::unique_ptr<EntropyController> create_entropy_controller(
    int num_bins = 256, float entropy_threshold = 0.1f, float compute_budget = 1.0f
);
