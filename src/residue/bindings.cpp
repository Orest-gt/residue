#include "../residue_wall/async_observer.h"
#include "../residue_wall/isolation_zone.h"
#include "../residue_wall/residue_wall.h"
#include "core.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(core, m) {
  m.doc() = "PROJECT RESIDUE V4.2 — Reality-Synchronized AVX2 Engine + "
            "Residue Wall";

  py::class_<EntropyControllerV3>(m, "EntropyControllerV3")
      .def(py::init<int, float, size_t, float, float>(),
           py::arg("num_bins") = 256, py::arg("entropy_threshold") = 0.1f,
           py::arg("temporal_buffer_size") = 5, py::arg("l1_threshold") = 0.1f,
           py::arg("ema_alpha") = 0.2f)

      .def(
          "infer_single_sample_fast",
          [](EntropyControllerV3 &self, py::array_t<float> input) {
            py::buffer_info buf = input.request();
            return self.infer_single_sample_fast((const float *)buf.ptr,
                                                 buf.shape[0]);
          },
          py::arg("input"))

      .def("reset_history", &EntropyControllerV3::reset_history)
      .def("get_total_samples_processed",
           &EntropyControllerV3::get_total_samples_processed)
      .def("get_total_samples_skipped",
           &EntropyControllerV3::get_total_samples_skipped)

      .def(
          "batch_infer_fast",
          [](EntropyControllerV3 &self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input,
             size_t frame_size) {
            py::buffer_info buf = input.request();

            size_t total_size = buf.shape[0];
            size_t num_frames = total_size / frame_size;

            auto result = py::array_t<float, py::array::c_style>(num_frames);
            py::buffer_info res_buf = result.request();

            self.process_stream_fast((const float *)buf.ptr, total_size,
                                     frame_size, (float *)res_buf.ptr);

            return result;
          },
          py::arg("input"), py::arg("frame_size"))

      // ---- Residue Wall: prefetch-aware batch inference ----
      .def(
          "batch_infer_walled",
          [](EntropyControllerV3 &self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input,
             size_t frame_size) {
            py::buffer_info buf = input.request();

            size_t total_size = buf.shape[0];
            size_t num_frames = total_size / frame_size;

            auto result = py::array_t<float, py::array::c_style>(num_frames);
            py::buffer_info res_buf = result.request();

            self.process_stream_walled((const float *)buf.ptr, total_size,
                                       frame_size, (float *)res_buf.ptr);

            return result;
          },
          py::arg("input"), py::arg("frame_size"));

  // ---- Residue Wall: Cache Topology Diagnostics ----
  m.def("get_cache_topology", []() {
    const auto &topo = residue_wall::get_cache_topology();
    py::dict d;
    d["l1d_size"] = topo.l1d_size;
    d["l2_size"] = topo.l2_size;
    d["l3_size"] = topo.l3_size;
    d["cache_line_size"] = topo.cache_line_size;
    d["l1d_associativity"] = topo.l1d_associativity;
    d["num_physical_cores"] = topo.num_physical_cores;
    d["l1d_lines"] = topo.l1d_lines;
    return d;
  });

  // ---- Thread Isolation Zone: Diagnostics ----
  m.def("print_isolation_report",
        []() { residue_wall::IsolationZone::print_isolation_report(); });

  m.def(
      "create_entropy_controller_v3",
      [](int num_bins, float entropy_threshold, size_t temporal_buffer_size,
         float l1_threshold, float ema_alpha) {
        return new EntropyControllerV3(num_bins, entropy_threshold,
                                       temporal_buffer_size, l1_threshold,
                                       ema_alpha);
      },
      py::return_value_policy::take_ownership, py::arg("num_bins") = 256,
      py::arg("entropy_threshold") = 0.1f, py::arg("temporal_buffer_size") = 5,
      py::arg("l1_threshold") = 0.1f, py::arg("ema_alpha") = 0.2f);

  // ---- Level 5: Async Observer & Telemetry ----
  py::class_<residue_wall::TelemetrySnapshot>(m, "TelemetrySnapshot")
      .def_readonly("total_samples_ingested",
                    &residue_wall::TelemetrySnapshot::total_samples_ingested)
      .def_readonly("total_samples_processed",
                    &residue_wall::TelemetrySnapshot::total_samples_processed)
      .def_readonly("total_samples_skipped",
                    &residue_wall::TelemetrySnapshot::total_samples_skipped)
      .def_readonly("total_frames_dropped",
                    &residue_wall::TelemetrySnapshot::total_frames_dropped)
      .def_readonly("current_fps",
                    &residue_wall::TelemetrySnapshot::current_fps)
      .def_readonly("sparsity_pct",
                    &residue_wall::TelemetrySnapshot::sparsity_pct)
      .def_readonly("buffer_fill_pct",
                    &residue_wall::TelemetrySnapshot::buffer_fill_pct)
      .def_readonly("is_running", &residue_wall::TelemetrySnapshot::is_running)
      .def_readonly("isolation_active",
                    &residue_wall::TelemetrySnapshot::isolation_active)
      .def_readonly("backpressure_active",
                    &residue_wall::TelemetrySnapshot::backpressure_active);

  py::class_<residue_wall::AsyncObserver>(m, "AsyncObserver")
      .def(py::init<size_t, size_t>(), py::arg("frame_size") = 1024,
           py::arg("buffer_capacity_frames") = 10000)
      .def("start", &residue_wall::AsyncObserver::start,
           py::call_guard<py::gil_scoped_release>())
      .def("stop", &residue_wall::AsyncObserver::stop,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "push_data",
          [](residue_wall::AsyncObserver &self,
             py::array_t<float, py::array::c_style | py::array::forcecast>
                 input) {
            py::buffer_info buf = input.request();
            // Push is highly optimized, but we still release GIL to untether
            // Python quickly
            py::gil_scoped_release release;
            return self.push_data((const float *)buf.ptr, buf.shape[0]);
          },
          py::arg("input"))
      .def(
          "pull_output",
          [](residue_wall::AsyncObserver &self, size_t max_count) {
            auto result = py::array_t<float, py::array::c_style>(max_count);
            py::buffer_info res_buf = result.request();
            size_t actual_read = 0;
            {
              py::gil_scoped_release release;
              actual_read = self.pull_output((float *)res_buf.ptr, max_count);
            }
            // Resize the numpy array to actual_read
            result.resize({actual_read});
            return result;
          },
          py::arg("max_count"))
      .def("reset_telemetry", &residue_wall::AsyncObserver::reset_telemetry)
      .def("poll_telemetry", &residue_wall::AsyncObserver::poll_telemetry)
      .def("recommended_push_size",
           &residue_wall::AsyncObserver::recommended_push_size);

  m.def("print_isolation_report",
        &residue_wall::IsolationZone::print_isolation_report,
        "Print the current hardware isolation status to stdout");
}
