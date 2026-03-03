#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include "core.h"

namespace py = pybind11;

// Optimized Python bindings for EntropyControllerV2
PYBIND11_MODULE(residue_v2, m) {
    m.doc() = "PROJECT RESIDUE v2.0 - The Analog Scientist with optimized performance";
    
    // Entropy Controller V2 bindings
    py::class_<EntropyControllerV2>(m, "EntropyControllerV2")
        .def(py::init<int, float>(),
             py::arg("num_bins") = 256,
             py::arg("entropy_threshold") = 0.1f)
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
             "Compute adaptive scaling with multi-dimensional features",
             py::arg("input"))
        .def("extract_features", &EntropyControllerV2::extract_features,
             "Extract multi-dimensional feature vector",
             py::arg("input"))
        .def("compute_multi_dimensional_scaling", 
             py::overload_cast<const EntropyControllerV2::FeatureVector&>(
                 &EntropyControllerV2::compute_multi_dimensional_scaling),
             "Compute scaling from multi-dimensional features",
             py::arg("features"))
        .def("sigmoid_scaling", &EntropyControllerV2::sigmoid_scaling,
             "Compute sigmoid-based smooth scaling",
             py::arg("x"), py::arg("midpoint") = 2.5f, py::arg("steepness") = 1.0f)
        .def("linear_interpolation_scaling", &EntropyControllerV2::linear_interpolation_scaling,
             "Compute linear interpolation scaling",
             py::arg("entropy"), py::arg("min_entropy") = 1.0f, py::arg("max_entropy") = 5.0f)
        .def("smoothstep_scaling", &EntropyControllerV2::smoothstep_scaling,
             "Compute smoothstep scaling",
             py::arg("edge0"), py::arg("edge1"), py::arg("x"))
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
             "Set min/max scaling factors",
             py::arg("min_factor"), py::arg("max_factor"))
        .def("set_num_bins", &EntropyControllerV2::set_num_bins,
             "Set number of histogram bins",
             py::arg("bins"))
        .def("reset_history", &EntropyControllerV2::reset_history,
             "Reset performance history")
        .def("get_entropy_history", &EntropyControllerV2::get_entropy_history,
             "Get entropy history")
        .def("get_scaling_history", &EntropyControllerV2::get_scaling_history,
             "Get scaling history")
        .def("get_complexity_history", &EntropyControllerV2::get_complexity_history,
             "Get complexity history");
    
    // FeatureVector binding
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
    
    // Factory function
    m.def("create_entropy_controller_v2", &create_entropy_controller_v2,
           "Create optimized entropy controller v2.0 instance",
           py::arg("num_bins") = 256, py::arg("entropy_threshold") = 0.1f);
    
    // Convenience functions for production use
    m.def("compute_analog_scaling", [](py::array_t<float> input, float threshold = 0.1f) {
        py::buffer_info buf = input.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Input must be 1D array");
        }
        
        std::vector<float> input_vec(buf.shape[0]);
        std::copy((float*)buf.ptr, (float*)buf.ptr + buf.shape[0], input_vec.begin());
        
        auto controller = create_entropy_controller_v2(256, threshold);
        auto features = controller->extract_features(input_vec);
        float scaling = controller->compute_multi_dimensional_scaling(features);
        
        return py::make_tuple(features.entropy, features.complexity, features.sparsity, 
                             features.structure, scaling);
    }, py::arg("input"), py::arg("threshold") = 0.1f,
       "Compute multi-dimensional analog scaling - main production function");
    
    m.def("batch_compute_analog_scaling", [](py::array_t<float> inputs, float threshold = 0.1f) {
        py::buffer_info buf = inputs.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Inputs must be 2D array (batch_size, features)");
        }
        
        const size_t batch_size = buf.shape[0];
        const size_t num_features = buf.shape[1];
        
        std::vector<float> entropies(batch_size);
        std::vector<float> complexities(batch_size);
        std::vector<float> sparsities(batch_size);
        std::vector<float> structures(batch_size);
        std::vector<float> scalings(batch_size);
        
        auto controller = create_entropy_controller_v2(256, threshold);
        
        for (size_t i = 0; i < batch_size; i++) {
            std::vector<float> input_vec(num_features);
            for (size_t j = 0; j < num_features; j++) {
                input_vec[j] = ((float*)buf.ptr)[i * num_features + j];
            }
            
            auto features = controller->extract_features(input_vec);
            entropies[i] = features.entropy;
            complexities[i] = features.complexity;
            sparsities[i] = features.sparsity;
            structures[i] = features.structure;
            scalings[i] = controller->compute_multi_dimensional_scaling(features);
        }
        
        return py::make_tuple(
            py::array_t<float>(batch_size, entropies.data()),
            py::array_t<float>(batch_size, complexities.data()),
            py::array_t<float>(batch_size, sparsities.data()),
            py::array_t<float>(batch_size, structures.data()),
            py::array_t<float>(batch_size, scalings.data())
        );
    }, py::arg("inputs"), py::arg("threshold") = 0.1f,
       "Batch compute multi-dimensional analog scaling for production workloads");
    
    // Semantic bridge functions
    m.def("compute_skip_predict_decision", [](float scaling, float confidence_threshold = 0.7f) {
        // Convert scaling to skip/predict decision
        float confidence = 1.0f / (1.0f + std::exp(-scaling)); // Sigmoid confidence
        bool should_skip = confidence > confidence_threshold;
        
        return py::make_tuple(should_skip, confidence);
    }, py::arg("scaling"), py::arg("confidence_threshold") = 0.7f,
       "Convert scaling factor to skip/predict decision");
    
    m.def("batch_skip_predict_decisions", [](py::array_t<float> scalings, float confidence_threshold = 0.7f) {
        py::buffer_info buf = scalings.request();
        if (buf.ndim != 1) {
            throw std::runtime_error("Scalings must be 1D array");
        }
        
        const size_t batch_size = buf.shape[0];
        std::vector<uint8_t> decisions_uint8(batch_size);
        std::vector<float> confidences(batch_size);
        
        float* scaling_ptr = static_cast<float*>(buf.ptr);
        for (size_t i = 0; i < batch_size; i++) {
            float confidence = 1.0f / (1.0f + std::exp(-scaling_ptr[i]));
            decisions_uint8[i] = confidence > confidence_threshold ? 1 : 0;
            confidences[i] = confidence;
        }
        
        // Convert uint8 to bool for Python
        std::vector<bool> decisions(batch_size);
        for (size_t i = 0; i < batch_size; i++) {
            decisions[i] = decisions_uint8[i] == 1;
        }
        
        return py::make_tuple(
            py::cast(decisions),
            py::array_t<float>(batch_size, confidences.data())
        );
    }, py::arg("scalings"), py::arg("confidence_threshold") = 0.7f,
       "Batch convert scaling factors to skip/predict decisions");
    
    // Version and metadata
    m.attr("__version__") = "2.0.0";
    m.attr("PROJECT") = "RESIDUE";
    m.attr("COMPONENT") = "Entropy Controller V2.0";
    m.attr("DESCRIPTION") = "The Analog Scientist - Multi-dimensional optimization with semantic bridge";
    m.attr("STATUS") = "Production Ready - Optimized";
    m.attr("OPTIMIZATIONS") = "NaN fix, C++ kernel optimization, semantic bridge";
}
