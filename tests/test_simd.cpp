#include "../src/residue/core_v3.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>


class TestController : public EntropyControllerV3 {
public:
  float test_zcr(const float *data, size_t size) {
    return calculate_zero_crossing_rate_ptr(data, size);
  }
  float test_sparsity(const float *data, size_t size) {
    return calculate_sparsity_ptr(data, size);
  }
  float test_structure(const float *data, size_t size) {
    return calculate_structure_score_ptr(data, size);
  }

  // Original V2 scalar implementations for strict verification
  float ref_zcr(const float *data, size_t size) {
    if (size < 2)
      return 0.0f;
    size_t zero_crossings = 0;
    for (size_t i = 0; i < size - 1; ++i) {
      if ((data[i] >= 0.0f && data[i + 1] < 0.0f) ||
          (data[i] < 0.0f && data[i + 1] >= 0.0f))
        zero_crossings++;
    }
    return static_cast<float>(zero_crossings) / (size - 1);
  }

  float ref_sparsity(const float *data, size_t size) {
    if (size == 0)
      return 1.0f;
    float l1_norm = 0.0f;
    for (size_t i = 0; i < size; ++i)
      l1_norm += std::abs(data[i]);
    float normalized_l1 = l1_norm / size;
    if (normalized_l1 < l1_sparsity_threshold)
      return 1.0f;

    size_t zero_count = 0;
    for (size_t i = 0; i < size; ++i) {
      if (std::abs(data[i]) < 1e-8f)
        zero_count++;
    }
    return static_cast<float>(zero_count) / size;
  }
};

int main() {
  TestController tc;
  std::vector<size_t> sizes = {1,  2,  3,  7,    8,    9,   15,
                               16, 17, 64, 1024, 1027, 2055};
  bool all_pass = true;

  std::cout << "=== V3 SIMD MATH VERIFICATION ===" << std::endl;
  for (size_t sz : sizes) {
    std::vector<float> data(sz);
    for (size_t i = 0; i < sz; i++) {
      // Mix of positive, negative, and zeroes
      data[i] = std::sin(i * 0.5f) * ((i % 5 == 0) ? 0.0f : 1.0f);
    }

    float simd_zcr = tc.test_zcr(data.data(), sz);
    float ref_zcr_val = tc.ref_zcr(data.data(), sz);
    if (std::abs(simd_zcr - ref_zcr_val) > 1e-6f) {
      std::cout << "❌ ZCR MISMATCH at size " << sz << ": SIMD=" << simd_zcr
                << " REF=" << ref_zcr_val << "\n";
      all_pass = false;
    }

    float simd_spar = tc.test_sparsity(data.data(), sz);
    float ref_spar_val = tc.ref_sparsity(data.data(), sz);
    if (std::abs(simd_spar - ref_spar_val) > 1e-6f) {
      std::cout << "❌ Sparsity MISMATCH at size " << sz
                << ": SIMD=" << simd_spar << " REF=" << ref_spar_val << "\n";
      all_pass = false;
    }
  }

  if (all_pass)
    std::cout << "✅ ALL SIMD MATH TESTS PASSED (Tolerance: 1e-6)\n";
  else
    std::cout << "❌ SIMD MATH TESTS FAILED\n";

  return all_pass ? 0 : 1;
}
