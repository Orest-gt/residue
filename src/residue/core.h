#ifndef ENTROPY_CONTROLLER_V2_H
#define ENTROPY_CONTROLLER_V2_H

#include <vector>
#include <memory>
#include <cmath>

/**
 * PROJECT RESIDUE v2.0 - The Analog Scientist
 * 
 * Softmax-based scaling for smooth, multi-dimensional optimization
 * Moving beyond binary thresholds to analog control
 */

class EntropyControllerV2 {
private:
    // Configuration parameters
    int num_bins;
    float entropy_threshold;
    float min_scaling_factor;
    float max_scaling_factor;
    
    // Performance tracking
    std::vector<float> entropy_history;
    std::vector<float> scaling_history;
    std::vector<float> complexity_history;
    
    // Internal state
    float current_scaling_factor;
    size_t total_samples_processed;
    
    // Optimization buffers
    std::vector<float> histogram_buffer;
    
public:
    // Constructor
    EntropyControllerV2(int bins = 256, float threshold = 0.1f);
    
    // Core entropy operations
    float calculate_input_entropy(const std::vector<float>& input);
    float calculate_complexity_score(const std::vector<float>& input);
    
    // Softmax-based analog scaling
    float compute_softmax_scaling(float entropy, float complexity_score);
    float compute_adaptive_scaling(const std::vector<float>& input);
    
    // Multi-dimensional optimization
    struct FeatureVector {
        float entropy;
        float complexity;
        float sparsity;
        float structure;
    };
    
    FeatureVector extract_features(const std::vector<float>& input);
    float compute_multi_dimensional_scaling(const FeatureVector& features);
    
    // Advanced scaling functions
    float sigmoid_scaling(float x, float midpoint = 2.5f, float steepness = 1.0f);
    float linear_interpolation_scaling(float entropy, float min_entropy = 1.0f, float max_entropy = 5.0f);
    float smoothstep_scaling(float edge0, float edge1, float x);
    
    // Performance monitoring
    void update_performance_history(float entropy, float scaling, float complexity);
    std::vector<float> get_entropy_history() const { return entropy_history; }
    std::vector<float> get_scaling_history() const { return scaling_history; }
    std::vector<float> get_complexity_history() const { return complexity_history; }
    
    // Accessors
    float get_current_scaling_factor() const { return current_scaling_factor; }
    float get_entropy_threshold() const { return entropy_threshold; }
    size_t get_samples_processed() const { return total_samples_processed; }
    
    // Configuration
    void set_entropy_threshold(float threshold);
    void set_scaling_range(float min_factor, float max_factor);
    void set_num_bins(int bins);
    
    // Analysis functions
    float calculate_standard_deviation(const std::vector<float>& data);
    float calculate_sparsity(const std::vector<float>& data);
    float calculate_structure_score(const std::vector<float>& data);
    
    // Utility functions
    void reset_history();
    std::unique_ptr<EntropyControllerV2> clone() const;
    
    ~EntropyControllerV2() = default;
};

// Factory function for easy creation
std::unique_ptr<EntropyControllerV2> create_entropy_controller_v2(
    int num_bins = 256, float entropy_threshold = 0.1f
);

#endif // ENTROPY_CONTROLLER_V2_H
