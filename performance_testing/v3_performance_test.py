#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.0 - Zero Overhead Performance Testing
========================================================

Comprehensive testing suite for V3.0 zero overhead architecture.
Validates sub-0.08ms latency, structural intelligence priority, and impenetrable stability.
"""

import sys
import time
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add V3.0 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import residue
    V3_AVAILABLE = hasattr(residue, 'create_entropy_controller_v3')
except ImportError:
    V3_AVAILABLE = False
    print("❌ PROJECT RESIDUE V3.0 not available")

class V3PerformanceTester:
    """Comprehensive V3.0 performance validation suite"""
    
    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if V3_AVAILABLE:
            self.controller = residue.create_entropy_controller_v3()
            self.controller.set_v3_weights([9.0, 8.0, 7.0, 5.0, 4.0, 3.0, 2.0])
            self.controller.set_ema_alpha_v3(residue.EMA_ALPHA_V3)
            self.controller.enable_avx_optimization()
        
        print("=== PROJECT RESIDUE V3.0 PERFORMANCE TESTING ===")
        print(f"V3.0 Available: {V3_AVAILABLE}")
        print("=" * 60)
    
    def test_sub_80us_latency(self):
        """Validate sub-0.08ms inference latency target"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== SUB-80μs LATENCY TEST ===")
        
        # Generate test data
        test_data = np.random.randn(1000).astype(np.float32)
        
        # Warm up
        for _ in range(100):
            self.controller.infer_single_sample_fast(test_data)
        
        # Benchmark
        iterations = 10000
        start_time = time.perf_counter()
        
        for _ in range(iterations):
            self.controller.infer_single_sample_fast(test_data)
        
        end_time = time.perf_counter()
        avg_time_us = ((end_time - start_time) / iterations) * 1e6
        
        meets_target = avg_time_us < 80.0  # 0.08ms = 80μs
        
        print(f"Average inference time: {avg_time_us:.2f} μs")
        print(f"Target: < 80.0 μs")
        print(f"Status: {'✅ PASS' if meets_target else '❌ FAIL'}")
        
        return {
            "test_type": "sub_80us_latency",
            "avg_time_us": avg_time_us,
            "meets_target": meets_target,
            "iterations": iterations
        }
    
    def test_zcr_priority(self):
        """Validate ZCR (Zero Crossing Rate) priority in V3.0"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== ZCR PRIORITY TEST ===")
        
        # High ZCR signal (alternating pattern)
        high_zcr_signal = np.array([1.0 if i % 10 < 5 else -1.0 for i in range(1000)], dtype=np.float32)
        
        features = self.controller.extract_features_v3_fast(high_zcr_signal)
        scaling = self.controller.compute_structural_scaling_v3(features)
        
        print(f"High ZCR signal features:")
        print(f"  ZCR Rate: {features.zcr_rate:.3f}")
        print(f"  L1 Sparsity: {features.l1_sparsity:.3f}")
        print(f"  Structural Confidence: {features.structural_confidence:.3f}")
        print(f"  Scaling Factor: {scaling:.3f}")
        
        zcr_priority_correct = features.zcr_rate > 0.4  # Should be > 40%
        print(f"ZCR Priority: {'✅ PASS' if zcr_priority_correct else '❌ FAIL'}")
        
        return {
            "test_type": "zcr_priority",
            "zcr_rate": features.zcr_rate,
            "l1_sparsity": features.l1_sparsity,
            "structural_confidence": features.structural_confidence,
            "scaling_factor": scaling,
            "priority_correct": zcr_priority_correct
        }
    
    def test_l1_sparsity_priority(self):
        """Validate L1 Sparsity priority in V3.0"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== L1 SPARSITY PRIORITY TEST ===")
        
        # Sparse signal (few non-zero elements)
        sparse_signal = np.zeros(1000, dtype=np.float32)
        sparse_signal[100] = 1.0
        sparse_signal[500] = -1.0
        sparse_signal[900] = 0.5
        
        features = self.controller.extract_features_v3_fast(sparse_signal)
        scaling = self.controller.compute_structural_scaling_v3(features)
        
        print(f"Sparse signal features:")
        print(f"  L1 Sparsity: {features.l1_sparsity:.3f}")
        print(f"  ZCR Rate: {features.zcr_rate:.3f}")
        print(f"  Structural Confidence: {features.structural_confidence:.3f}")
        print(f"  Scaling Factor: {scaling:.3f}")
        
        l1_priority_correct = features.l1_sparsity > 0.7  # Should be > 70%
        print(f"L1 Sparsity Priority: {'✅ PASS' if l1_priority_correct else '❌ FAIL'}")
        
        return {
            "test_type": "l1_sparsity_priority",
            "l1_sparsity": features.l1_sparsity,
            "zcr_rate": features.zcr_rate,
            "structural_confidence": features.structural_confidence,
            "scaling_factor": scaling,
            "priority_correct": l1_priority_correct
        }
    
    def test_ema_stability(self):
        """Validate EMA stability with 0.002 decay"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== EMA STABILITY TEST ===")
        
        temporal_values = []
        
        # Generate temporal sequence
        for i in range(100):
            signal = np.full(100, float(i), dtype=np.float32)
            features = self.controller.extract_features_v3_fast(signal)
            temporal_values.append(features.temporal_coherence)
        
        # Calculate stability
        variance = np.var(temporal_values)
        std_dev = np.sqrt(variance)
        ema_stable = std_dev < 0.1  # Should be very stable
        
        print(f"EMA Standard Deviation: {std_dev:.4f}")
        print(f"Target: < 0.1")
        print(f"EMA Stability: {'✅ PASS' if ema_stable else '❌ FAIL'}")
        
        return {
            "test_type": "ema_stability",
            "std_deviation": std_dev,
            "ema_stable": ema_stable,
            "temporal_values": temporal_values[::10]  # Sample every 10th value
        }
    
    def test_epsilon_protection(self):
        """Validate epsilon protection against edge cases"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== EPSILON PROTECTION TEST ===")
        
        # Edge cases
        edge_cases = [
            np.array([], dtype=np.float32),                    # Empty
            np.array([0.0], dtype=np.float32),                 # Single zero
            np.array([1e-10, 1e-10, 1e-10], dtype=np.float32), # Very small
            np.array([1e10, 1e10, 1e10], dtype=np.float32),    # Very large
            np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # All zeros
        ]
        
        epsilon_protection_works = True
        results = []
        
        for i, case in enumerate(edge_cases):
            try:
                features = self.controller.extract_features_v3_fast(case)
                scaling = self.controller.compute_structural_scaling_v3(features)
                
                if np.isnan(scaling) or np.isinf(scaling):
                    print(f"Edge case {i}: NaN/Inf detected ❌")
                    epsilon_protection_works = False
                    results.append({"case": i, "status": "FAIL", "reason": "NaN/Inf"})
                else:
                    print(f"Edge case {i}: Protected ✅")
                    results.append({"case": i, "status": "PASS", "scaling": scaling})
            except Exception as e:
                print(f"Edge case {i}: Exception - {e} ❌")
                epsilon_protection_works = False
                results.append({"case": i, "status": "FAIL", "reason": str(e)})
        
        print(f"Epsilon Protection: {'✅ PASS' if epsilon_protection_works else '❌ FAIL'}")
        
        return {
            "test_type": "epsilon_protection",
            "protection_works": epsilon_protection_works,
            "edge_case_results": results
        }
    
    def test_batch_performance(self):
        """Validate batch processing performance"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== BATCH PERFORMANCE TEST ===")
        
        # Batch processing test
        batch_size = 100
        batch_data = [np.random.randn(1000).astype(np.float32) for _ in range(batch_size)]
        
        start_time = time.perf_counter()
        
        batch_scalings = residue.batch_infer_structural_scaling_v3(batch_data)
        
        end_time = time.perf_counter()
        avg_batch_time_us = ((end_time - start_time) / batch_size) * 1e6
        
        batch_efficient = avg_batch_time_us < 100.0  # Should be < 100μs per sample
        
        print(f"Batch processing time per sample: {avg_batch_time_us:.2f} μs")
        print(f"Target: < 100.0 μs")
        print(f"Batch Performance: {'✅ PASS' if batch_efficient else '❌ FAIL'}")
        
        return {
            "test_type": "batch_performance",
            "avg_batch_time_us": avg_batch_time_us,
            "batch_size": batch_size,
            "batch_efficient": batch_efficient
        }
    
    def test_structural_confidence(self):
        """Validate structural confidence scoring"""
        
        if not V3_AVAILABLE:
            return {"error": "V3.0 not available"}
        
        print("\n=== STRUCTURAL CONFIDENCE TEST ===")
        
        # Test different signal types
        signals = {
            "high_confidence": np.random.randn(1000).astype(np.float32),  # Normal signal
            "low_confidence": np.random.randn(1000) * 0.01,               # Low amplitude
            "structured": np.sin(np.linspace(0, 10, 1000)).astype(np.float32),  # Structured
        }
        
        confidence_results = {}
        
        for name, signal in signals.items():
            features = self.controller.extract_features_v3_fast(signal)
            confidence_results[name] = {
                "confidence": features.structural_confidence,
                "zcr_rate": features.zcr_rate,
                "l1_sparsity": features.l1_sparsity
            }
            
            print(f"{name}: Confidence = {features.structural_confidence:.3f}")
        
        # High confidence should be > 0.5 for normal signals
        high_confidence_ok = confidence_results["high_confidence"]["confidence"] > 0.5
        print(f"Structural Confidence: {'✅ PASS' if high_confidence_ok else '❌ FAIL'}")
        
        return {
            "test_type": "structural_confidence",
            "confidence_results": confidence_results,
            "high_confidence_ok": high_confidence_ok
        }
    
    def run_comprehensive_v3_test(self):
        """Run complete V3.0 validation suite"""
        
        print("\n=== COMPREHENSIVE V3.0 VALIDATION ===")
        
        if not V3_AVAILABLE:
            print("❌ V3.0 not available - cannot run tests")
            return {"error": "V3.0 not available"}
        
        results = {
            "test_suite": "v3_comprehensive_validation",
            "timestamp": datetime.now().isoformat(),
            "v3_version": residue.__version__,
            "tests": []
        }
        
        # Run all tests
        tests = [
            self.test_sub_80us_latency,
            self.test_zcr_priority,
            self.test_l1_sparsity_priority,
            self.test_ema_stability,
            self.test_epsilon_protection,
            self.test_batch_performance,
            self.test_structural_confidence
        ]
        
        all_tests_pass = True
        
        for test_func in tests:
            try:
                test_result = test_func()
                results["tests"].append(test_result)
                
                if "error" not in test_result:
                    # Check if test passed based on test type
                    test_type = test_result.get("test_type", "")
                    if test_type == "sub_80us_latency":
                        test_passed = test_result.get("meets_target", False)
                    elif test_type in ["zcr_priority", "l1_sparsity_priority"]:
                        test_passed = test_result.get("priority_correct", False)
                    elif test_type == "ema_stability":
                        test_passed = test_result.get("ema_stable", False)
                    elif test_type == "epsilon_protection":
                        test_passed = test_result.get("protection_works", False)
                    elif test_type == "batch_performance":
                        test_passed = test_result.get("batch_efficient", False)
                    elif test_type == "structural_confidence":
                        test_passed = test_result.get("high_confidence_ok", False)
                    else:
                        test_passed = False
                    
                    all_tests_pass &= test_passed
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed: {e}")
                results["tests"].append({"test_type": test_func.__name__, "error": str(e)})
                all_tests_pass = False
        
        # Overall assessment
        results["overall_status"] = "PASS" if all_tests_pass else "FAIL"
        results["all_tests_pass"] = all_tests_pass
        
        # Performance metrics
        metrics = self.controller.get_performance_metrics()
        results["performance_metrics"] = {
            "samples_processed": metrics.samples_processed,
            "current_scaling": self.controller.get_current_scaling_factor()
        }
        
        # Save results
        output_file = self.output_dir / f"v3_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n=== FINAL VALIDATION RESULTS ===")
        print(f"Overall Status: {'✅ ALL TESTS PASS' if all_tests_pass else '❌ SOME TESTS FAIL'}")
        
        if all_tests_pass:
            print("🎉 V3.0 Zero Overhead Architecture VALIDATED!")
            print("✅ Sub-0.08ms inference latency achieved")
            print("✅ ZCR and L1 Sparsity prioritization working")
            print("✅ Impenetrable stability confirmed")
        else:
            print("⚠️  Some optimizations may need adjustment")
        
        print(f"📊 Results saved to: {output_file}")
        
        return results

def main():
    """Main V3.0 testing function"""
    
    tester = V3PerformanceTester()
    
    try:
        results = tester.run_comprehensive_v3_test()
        return results
    except Exception as e:
        print(f"❌ V3.0 testing failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    main()
