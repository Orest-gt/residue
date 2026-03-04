#ifndef ENTROPY_CONTROLLER_V2_H
#define ENTROPY_CONTROLLER_V2_H

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>

/**
 * Circular buffer for temporal coherence tracking
 * Provides O(1) push and O(N) access to recent values
 */
class CircularBuffer {
private:
    std::vector<float> buffer;
    size_t head, tail, size, capacity;
    float ema_value;
    float alpha;  // EMA smoothing factor
    
public:
    CircularBuffer(size_t buffer_size, float ema_alpha = 0.2f)
        : buffer(buffer_size, 0.0f), head(0), tail(0), size(0), capacity(buffer_size),
          ema_value(0.0f), alpha(ema_alpha) {}
    
    void push(float value) {
        buffer[head] = value;
        head = (head + 1) % capacity;
        if (size < capacity) {
            size++;
        } else {
            tail = (tail + 1) % capacity;
        }
        
        // Update EMA
        if (size == 1) {
            ema_value = value;
        } else {
            ema_value = alpha * value + (1.0f - alpha) * ema_value;
        }
    }
    
    float get_ema() const { return ema_value; }
    
    std::vector<float> get_recent(size_t count) const {
        std::vector<float> result;
        count = std::min(count, size);
        
        for (size_t i = 0; i < count; ++i) {
            size_t idx = (tail + i) % capacity;
            result.push_back(buffer[idx]);
        }
        
        return result;
    }
    
    size_t get_size() const { return size; }
    void clear() {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        head = tail = size = 0;
        ema_value = 0.0f;
    }
};

/**
 * PROJECT RESIDUE v3.0 - The Structural Scientist
 * 
 * Advanced heuristics for temporal coherence and data structure analysis
 * Moving beyond softmax to intelligent scaling decisions
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
    
    // V3.0 Structural heuristics
    std::unique_ptr<CircularBuffer> temporal_buffer;
    float l1_sparsity_threshold;
    size_t zcr_window_size;
    float ema_alpha;  // Exposed EMA smoothing factor
    
public:
    // Constructor
    EntropyControllerV2(int bins = 256, float threshold = 0.1f, 
                     size_t temporal_buffer_size = 5, float l1_threshold = 0.1f, float ema_alpha_param = 0.2f);
    
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
    
    // V3.0 Enhanced feature vector with structural heuristics
    struct FeatureVectorV3 {
        float entropy;
        float complexity;
        float sparsity;
        float structure;
        float temporal_coherence;  // EMA of recent scalings
        float zcr_rate;          // Zero-crossing rate
        float l1_sparsity;       // L1-norm based sparsity
    };
    
    FeatureVector extract_features(const std::vector<float>& input);
    float compute_multi_dimensional_scaling(const FeatureVector& features);
    
    // V3.0 Structural heuristics
    FeatureVectorV3 extract_features_v3(const std::vector<float>& input);
    float compute_multi_dimensional_scaling_v3(const FeatureVectorV3& features);
    
    // Advanced scaling functions
    float sigmoid_scaling(float x, float midpoint = 2.5f, float steepness = 1.0f);
    float linear_interpolation_scaling(float entropy, float min_entropy = 1.0f, float max_entropy = 5.0f);
    float smoothstep_scaling(float edge0, float edge1, float x);
    
    // V3.0 Structural heuristic functions
    float calculate_l1_norm_sparsity(const std::vector<float>& input);
    float calculate_zero_crossing_rate(const std::vector<float>& input);
    float compute_temporal_coherence(float current_scaling);
    
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
    
    // V3.0 Structural heuristics configuration
    void set_temporal_buffer_size(size_t buffer_size);
    void set_l1_sparsity_threshold(float threshold);
    void set_zcr_window_size(size_t window_size);
    void set_ema_alpha(float alpha);  // Set EMA smoothing factor
    float get_ema_alpha() const;  // Get current EMA alpha
    
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
    int num_bins = 256, float entropy_threshold = 0.1f,
    size_t temporal_buffer_size = 5, float l1_threshold = 0.1f, float ema_alpha = 0.2f
);

#endif // ENTROPY_CONTROLLER_V2_H
