#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>  // Critical for std::vector conversions
#include <vector>
#include <string>
#include "core.h"

namespace py = pybind11;

// PROJECT RESIDUE V3.0 - Complete Structural Heuristics Bindings
PYBIND11_MODULE(residue_v3, m) {
    m.doc() = "PROJECT RESIDUE V3.0 - Structural Heuristics Implementation";
    
    // Bind EntropyControllerV2 class with V3.0 features
    py::class_<EntropyControllerV2>(m, "EntropyControllerV2")
        .def(py::init<int, float, size_t, float>(),
             py::arg("num_bins") = 256, py::arg("entropy_threshold") = 0.1f,
             py::arg("temporal_buffer_size") = 5, py::arg("l1_threshold") = 0.1f)
        
        // Core entropy operations
        .def("calculate_input_entropy", &EntropyControllerV2::calculate_input_entropy,
             "Calculate Shannon entropy of input array",
             py::arg("input"))
        .def("calculate_complexity_score", &EntropyControllerV2::calculate_complexity_score,
             "Calculate multi-dimensional complexity score",
             py::arg("input"))
        .def("compute_softmax_scaling", &EntropyControllerV2::compute_softmax_scaling,
             "Compute softmax-based analog scaling",
             py::arg("entropy"), py::arg("complexity_score"))
        .def("compute_adaptive_scaling", &EntropyControllerV2::compute_adaptive_scaling,
             "Compute adaptive scaling factor for input data",
             py::arg("input"))
        
        // V2.0 Feature extraction
        .def("extract_features", &EntropyControllerV2::extract_features,
             "Extract multi-dimensional feature vector",
             py::arg("input"))
        .def("compute_multi_dimensional_scaling", 
             py::overload_cast<const EntropyControllerV2::FeatureVector&>(
                 &EntropyControllerV2::compute_multi_dimensional_scaling),
             "Compute scaling from multi-dimensional features",
             py::arg("features"))
        
        // V3.0 Structural heuristics
        .def("extract_features_v3", &EntropyControllerV2::extract_features_v3,
             "Extract V3.0 enhanced features with structural heuristics",
             py::arg("input"))
        .def("compute_multi_dimensional_scaling_v3", &EntropyControllerV2::compute_multi_dimensional_scaling_v3,
             "Compute V3.0 multi-dimensional scaling with structural heuristics",
             py::arg("features"))
        .def("calculate_l1_norm_sparsity", &EntropyControllerV2::calculate_l1_norm_sparsity,
             "Calculate L1 norm sparsity",
             py::arg("input"))
        .def("calculate_zero_crossing_rate", &EntropyControllerV2::calculate_zero_crossing_rate,
             "Calculate zero crossing rate",
             py::arg("input"))
        .def("compute_temporal_coherence", &EntropyControllerV2::compute_temporal_coherence,
             "Compute temporal coherence using EMA",
             py::arg("current_scaling"))
        
        // V3.0 Configuration
        .def("set_temporal_buffer_size", &EntropyControllerV2::set_temporal_buffer_size,
             "Set temporal buffer size",
             py::arg("size"))
        .def("set_l1_sparsity_threshold", &EntropyControllerV2::set_l1_sparsity_threshold,
             "Set L1 sparsity threshold",
             py::arg("threshold"))
        .def("set_zcr_window_size", &EntropyControllerV2::set_zcr_window_size,
             "Set zero crossing rate window size",
             py::arg("size"))
        .def("set_ema_alpha", &EntropyControllerV2::set_ema_alpha,
             "Set EMA smoothing factor (0.0 to 1.0)",
             py::arg("alpha"))
        .def("get_ema_alpha", &EntropyControllerV2::get_ema_alpha,
             "Get current EMA smoothing factor")
        
        // Existing methods
        .def("sigmoid_scaling", &EntropyControllerV2::sigmoid_scaling,
             "Compute sigmoid-based smooth scaling",
             py::arg("x"), py::arg("midpoint") = 2.5f, py::arg("steepness") = 1.0f)
        .def("linear_interpolation_scaling", &EntropyControllerV2::linear_interpolation_scaling,
             "Compute linear interpolation scaling",
             py::arg("entropy"), py::arg("min_entropy") = 1.0f, py::arg("max_entropy") = 5.0f)
        .def("smoothstep_scaling", &EntropyControllerV2::smoothstep_scaling,
             "Compute smoothstep scaling",
             py::arg("edge0"), py::arg("edge1"), py::arg("x"))
        .def("update_performance_history", &EntropyControllerV2::update_performance_history,
             "Update performance history",
             py::arg("entropy"), py::arg("scaling"), py::arg("complexity"))
        .def("get_entropy_history", &EntropyControllerV2::get_entropy_history,
             "Get entropy history")
        .def("get_scaling_history", &EntropyControllerV2::get_scaling_history,
             "Get scaling history")
        .def("get_complexity_history", &EntropyControllerV2::get_complexity_history,
             "Get complexity history")
        .def("get_current_scaling_factor", &EntropyControllerV2::get_current_scaling_factor,
             "Get current adaptive scaling factor")
        .def("get_entropy_threshold", &EntropyControllerV2::get_entropy_threshold,
             "Get current entropy threshold")
        .def("get_samples_processed", &EntropyControllerV2::get_samples_processed,
             "Get total samples processed")
        .def("set_entropy_threshold", &EntropyControllerV2::set_entropy_threshold,
             "Set entropy threshold for scaling adaptation",
             py::arg("threshold"))
        .def("set_scaling_range", &EntropyControllerV2::set_scaling_range,
             "Set min/max scaling range",
             py::arg("min_factor"), py::arg("max_factor"))
        .def("set_num_bins", &EntropyControllerV2::set_num_bins,
             "Set number of histogram bins",
             py::arg("bins"))
        .def("calculate_standard_deviation", &EntropyControllerV2::calculate_standard_deviation,
             "Calculate standard deviation",
             py::arg("data"))
        .def("calculate_sparsity", &EntropyControllerV2::calculate_sparsity,
             "Calculate sparsity",
             py::arg("data"))
        .def("calculate_structure_score", &EntropyControllerV2::calculate_structure_score,
             "Calculate structure score",
             py::arg("data"))
        .def("reset_history", &EntropyControllerV2::reset_history,
             "Reset performance history")
        .def("clone", &EntropyControllerV2::clone,
             "Create a clone of this controller");
    
    // V2.0 Feature vector binding
    py::class_<EntropyControllerV2::FeatureVector>(m, "FeatureVector")
        .def(py::init<>())
        .def_readwrite("entropy", &EntropyControllerV2::FeatureVector::entropy)
        .def_readwrite("complexity", &EntropyControllerV2::FeatureVector::complexity)
        .def_readwrite("sparsity", &EntropyControllerV2::FeatureVector::sparsity)
        .def_readwrite("structure", &EntropyControllerV2::FeatureVector::structure)
        .def("__repr__", [](const EntropyControllerV2::FeatureVector& fv) {
            return py::str("FeatureVector(entropy={}, complexity={}, sparsity={}, structure={})")
                .format(fv.entropy, fv.complexity, fv.sparsity, fv.structure);
        });
    
    // V3.0 Enhanced feature vector binding
    py::class_<EntropyControllerV2::FeatureVectorV3>(m, "FeatureVectorV3")
        .def(py::init<>())
        .def_readwrite("entropy", &EntropyControllerV2::FeatureVectorV3::entropy)
        .def_readwrite("complexity", &EntropyControllerV2::FeatureVectorV3::complexity)
        .def_readwrite("sparsity", &EntropyControllerV2::FeatureVectorV3::sparsity)
        .def_readwrite("structure", &EntropyControllerV2::FeatureVectorV3::structure)
        .def_readwrite("temporal_coherence", &EntropyControllerV2::FeatureVectorV3::temporal_coherence)
        .def_readwrite("zcr_rate", &EntropyControllerV2::FeatureVectorV3::zcr_rate)
        .def_readwrite("l1_sparsity", &EntropyControllerV2::FeatureVectorV3::l1_sparsity)
        .def("__repr__", [](const EntropyControllerV2::FeatureVectorV3& fv) {
            return py::str("FeatureVectorV3(entropy={}, complexity={}, sparsity={}, structure={}, temporal_coherence={}, zcr_rate={}, l1_sparsity={})")
                .format(fv.entropy, fv.complexity, fv.sparsity, fv.structure, fv.temporal_coherence, fv.zcr_rate, fv.l1_sparsity);
        });
    
    // Factory function for V3.0 creation
    m.def("create_entropy_controller_v3", [](int num_bins, float entropy_threshold, 
                                                   size_t temporal_buffer_size, float l1_threshold, float ema_alpha) {
        return create_entropy_controller_v2(num_bins, entropy_threshold, temporal_buffer_size, l1_threshold, ema_alpha);
    }, py::arg("num_bins") = 256, py::arg("entropy_threshold") = 0.1f,
       py::arg("temporal_buffer_size") = 5, py::arg("l1_threshold") = 0.1f, py::arg("ema_alpha") = 0.2f,
       "Create V3.0 entropy controller with structural heuristics");
    
    // V3.0 Convenience functions
    m.def("compute_v3_structural_scaling", [](py::array_t<float> input) {
        py::buffer_info buf = input.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input must be 1D array");
        }
        
        std::vector<float> input_vec(buf.shape[0]);
        std::copy((float*)buf.ptr, (float*)buf.ptr + buf.shape[0], input_vec.begin());
        
        auto controller = create_entropy_controller_v2();
        auto features = controller->extract_features_v3(input_vec);
        return controller->compute_multi_dimensional_scaling_v3(features);
    }, py::arg("input"),
       "Compute V3.0 structural scaling - main production function");
    
    // Batch processing with proper STL conversions
    m.def("batch_compute_v3_scaling", [](std::vector<std::vector<float>> inputs) {
        std::vector<float> results;
        results.reserve(inputs.size());
        
        auto controller = create_entropy_controller_v2();
        
        for (const auto& input_vec : inputs) {
            auto features = controller->extract_features_v3(input_vec);
            float scaling = controller->compute_multi_dimensional_scaling_v3(features);
            results.push_back(scaling);
        }
        
        return results;
    }, py::arg("inputs"),
       "Batch compute V3.0 scaling for multiple inputs");
    
    // Batch decisions with proper type handling
    m.def("batch_v3_decisions", [](std::vector<std::vector<float>> inputs, float threshold = 0.1f) {
        std::vector<bool> decisions;
        decisions.reserve(inputs.size());
        
        auto controller = create_entropy_controller_v2(256, threshold);
        
        for (const auto& input_vec : inputs) {
            auto features = controller->extract_features_v3(input_vec);
            float scaling = controller->compute_multi_dimensional_scaling_v3(features);
            decisions.push_back(scaling > threshold);
        }
        
        return decisions;
    }, py::arg("inputs"), py::arg("threshold") = 0.1f,
       "Batch V3.0 decisions for multiple inputs");
    
    // V3.0 Analysis functions
    m.def("analyze_signal_structure", [](py::array_t<float> input) {
        py::buffer_info buf = input.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input must be 1D array");
        }
        
        std::vector<float> input_vec(buf.shape[0]);
        std::copy((float*)buf.ptr, (float*)buf.ptr + buf.shape[0], input_vec.begin());
        
        auto controller = create_entropy_controller_v2();
        
        float zcr = controller->calculate_zero_crossing_rate(input_vec);
        float l1_sparsity = controller->calculate_l1_norm_sparsity(input_vec);
        float entropy = controller->calculate_input_entropy(input_vec);
        
        return py::dict(
            "zcr_rate"_a=zcr,
            "l1_sparsity"_a=l1_sparsity,
            "entropy"_a=entropy,
            "signal_type"_a=std::string(zcr > 0.3 ? "high_frequency" : zcr > 0.1 ? "medium_frequency" : "low_frequency")
        );
    }, py::arg("input"),
       "Analyze signal structure using V3.0 heuristics");
    
    // Temporal coherence analysis
    m.def("analyze_temporal_stability", [](std::vector<std::vector<float>> input_sequence) {
        auto controller = create_entropy_controller_v2();
        std::vector<float> scaling_factors;
        scaling_factors.reserve(input_sequence.size());
        
        for (const auto& input_vec : input_sequence) {
            auto features = controller->extract_features_v3(input_vec);
            float scaling = controller->compute_multi_dimensional_scaling_v3(features);
            scaling_factors.push_back(scaling);
        }
        
        // Calculate stability metrics
        float mean_scaling = 0.0f;
        for (float sf : scaling_factors) {
            mean_scaling += sf;
        }
        mean_scaling /= scaling_factors.size();
        
        float variance = 0.0f;
        for (float sf : scaling_factors) {
            float diff = sf - mean_scaling;
            variance += diff * diff;
        }
        variance /= scaling_factors.size();
        float std_dev = std::sqrt(variance);
        
        return py::dict(
            "scaling_factors"_a=scaling_factors,
            "mean_scaling"_a=mean_scaling,
            "std_deviation"_a=std_dev,
            "stability_score"_a=std::string(std_dev < 0.1f ? "excellent" : std_dev < 0.5f ? "good" : "poor")
        );
    }, py::arg("input_sequence"),
       "Analyze temporal stability of scaling factors");
}
