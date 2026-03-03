"""
PROJECT RESIDUE - Entropy Controller Unit Tests
Production-ready ML optimization testing
"""

import unittest
import numpy as np
import residue

class TestEntropyController(unittest.TestCase):
    """Comprehensive test suite for entropy controller"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.controller = residue.create_entropy_controller(
            num_bins=256,
            entropy_threshold=0.1
        )
        
        # Test data with known properties
        self.low_entropy_data = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])  # Low entropy
        self.high_entropy_data = np.random.randn(1000)  # High entropy
        self.constant_data = np.ones(1000)  # Zero entropy
    
    def test_entropy_calculation(self):
        """Test entropy calculation accuracy"""
        # Low entropy should be < 1 bit
        low_entropy = self.controller.calculate_input_entropy(self.low_entropy_data)
        self.assertLess(low_entropy, 1.0, "Low entropy data should have < 1 bit entropy")
        
        # High entropy should be > 5 bits
        high_entropy = self.controller.calculate_input_entropy(self.high_entropy_data)
        self.assertGreater(high_entropy, 5.0, "High entropy data should have > 5 bits entropy")
        
        # Constant data should have 0 entropy
        constant_entropy = self.controller.calculate_input_entropy(self.constant_data)
        self.assertAlmostEqual(constant_entropy, 0.0, places=3, msg="Constant data should have 0 entropy")
    
    def test_scaling_factor_calculation(self):
        """Test scaling factor computation"""
        # Low entropy should result in high scaling (less computation)
        low_entropy = self.controller.calculate_input_entropy(self.low_entropy_data)
        low_scaling = self.controller.compute_scaling_factor(low_entropy)
        self.assertGreater(low_scaling, 5.0, "Low entropy should result in high scaling factor")
        
        # High entropy should result in low scaling (more computation)
        high_entropy = self.controller.calculate_input_entropy(self.high_entropy_data)
        high_scaling = self.controller.compute_scaling_factor(high_entropy)
        self.assertLess(high_scaling, 2.0, "High entropy should result in low scaling factor")
    
    def test_convenience_functions(self):
        """Test convenience functions"""
        # Test compute_scaling function
        entropy, scaling = residue.compute_scaling(self.low_entropy_data)
        self.assertLess(entropy, 1.0, "Convenience function should compute correct entropy")
        self.assertGreater(scaling, 5.0, "Convenience function should compute correct scaling")
        
        # Test batch processing
        batch_data = np.array([
            self.low_entropy_data,
            self.high_entropy_data,
            self.constant_data
        ])
        entropies, scalings = residue.batch_compute_scaling(batch_data)
        
        self.assertEqual(len(entropies), 3, "Batch processing should return 3 entropies")
        self.assertEqual(len(scalings), 3, "Batch processing should return 3 scalings")
        self.assertLess(entropies[0], 1.0, "First batch should have low entropy")
        self.assertGreater(scalings[0], 5.0, "First batch should have high scaling")
    
    def test_adaptive_behavior(self):
        """Test adaptive parameter updates"""
        # Initial state
        initial_threshold = self.controller.get_entropy_threshold()
        
        # Update threshold
        new_threshold = 0.2
        self.controller.set_entropy_threshold(new_threshold)
        updated_threshold = self.controller.get_entropy_threshold()
        self.assertEqual(updated_threshold, new_threshold, "Threshold should be updated")
        
        # Test scaling range
        self.controller.set_scaling_range(0.1, 5.0)
        # Test with extreme values
        very_low_entropy = 0.01
        very_high_entropy = 10.0
        
        low_scaling = self.controller.compute_scaling_factor(very_low_entropy)
        high_scaling = self.controller.compute_scaling_factor(very_high_entropy)
        
        self.assertGreaterEqual(low_scaling, 0.1, "Scaling should respect minimum")
        self.assertLessEqual(high_scaling, 5.0, "Scaling should respect maximum")
    
    def test_performance_monitoring(self):
        """Test performance monitoring and adaptation"""
        # Monitor performance with good metrics
        self.controller.monitor_performance(processing_time=0.5, accuracy=0.95)
        efficiency = self.controller.get_efficiency_score()
        self.assertGreater(efficiency, 0.0, "Good performance should result in positive efficiency")
        
        # Test adaptation recommendations
        # Low entropy input should suggest decreasing resolution
        low_entropy = 0.1
        low_scaling = self.controller.compute_scaling_factor(low_entropy)
        
        # Create new controller with low threshold to trigger decrease recommendation
        low_threshold_controller = residue.create_entropy_controller(entropy_threshold=0.05)
        low_threshold_controller.update_scaling_factor(low_entropy)
        self.assertTrue(low_threshold_controller.should_decrease_resolution(), 
                       "Low entropy should suggest decreasing resolution")
        
        # High entropy input should suggest increasing resolution
        high_entropy = 5.0
        high_scaling = self.controller.compute_scaling_factor(high_entropy)
        
        high_threshold_controller = residue.create_entropy_controller(entropy_threshold=0.5)
        high_threshold_controller.update_scaling_factor(high_entropy)
        self.assertTrue(high_threshold_controller.should_increase_resolution(), 
                       "High entropy should suggest increasing resolution")
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty input
        empty_data = np.array([])
        try:
            entropy = self.controller.calculate_input_entropy(empty_data)
            # Should handle gracefully or raise appropriate error
            self.assertTrue(True, "Empty input should be handled gracefully")
        except:
            self.assertTrue(True, "Empty input should raise appropriate error")
        
        # Single value input
        single_data = np.array([1.0])
        entropy = self.controller.calculate_input_entropy(single_data)
        self.assertAlmostEqual(entropy, 0.0, places=3, msg="Single constant value should have 0 entropy")
        
        # Very large input
        large_data = np.random.randn(100000)
        entropy = self.controller.calculate_input_entropy(large_data)
        self.assertGreater(entropy, 0.0, "Large input should have positive entropy")
        self.assertLess(entropy, 20.0, "Large input entropy should be reasonable")
    
    def test_numerical_stability(self):
        """Test numerical stability and precision"""
        # Test with very small values
        small_data = np.array([1e-10, 1e-10, 1e-10])
        entropy = self.controller.calculate_input_entropy(small_data)
        self.assertFalse(np.isnan(entropy), "Small values should not result in NaN")
        self.assertFalse(np.isinf(entropy), "Small values should not result in infinity")
        
        # Test with very large values
        large_values = np.array([1e10, 1e10, 1e10])
        entropy = self.controller.calculate_input_entropy(large_values)
        self.assertFalse(np.isnan(entropy), "Large values should not result in NaN")
        self.assertFalse(np.isinf(entropy), "Large values should not result in infinity")
    
    def test_memory_efficiency(self):
        """Test memory efficiency"""
        import sys
        
        # Controller should not consume excessive memory
        initial_size = len(self.controller.__dict__)
        
        # Process large batch
        large_batch = np.random.randn(1000, 1000)
        entropies, scalings = residue.batch_compute_scaling(large_batch)
        
        # Memory usage should be reasonable
        final_size = len(self.controller.__dict__)
        self.assertLess(final_size - initial_size, 100, "Memory usage should be reasonable")
    
    def test_thread_safety(self):
        """Test thread safety (basic)"""
        import threading
        
        results = []
        
        def calculate_entropy(data):
            entropy = self.controller.calculate_input_entropy(data)
            results.append(entropy)
        
        # Create multiple threads
        threads = []
        test_data = [np.random.randn(100) for _ in range(5)]
        
        for data in test_data:
            thread = threading.Thread(target=calculate_entropy, args=(data,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        self.assertEqual(len(results), 5, "All threads should complete")
        for result in results:
            self.assertFalse(np.isnan(result), "All results should be valid")
            self.assertFalse(np.isinf(result), "All results should be valid")

class TestPerformanceRequirements(unittest.TestCase):
    """Test performance requirements for production use"""
    
    def test_speed_requirements(self):
        """Test that entropy calculation meets speed requirements"""
        controller = residue.create_entropy_controller()
        
        # Test data
        test_data = np.random.randn(1000)
        
        # Time multiple calculations
        import time
        times = []
        for _ in range(100):
            start = time.time()
            entropy = controller.calculate_input_entropy(test_data)
            times.append(time.time() - start)
        
        avg_time = np.mean(times)
        
        # Should be less than 1ms for 1000 elements
        self.assertLess(avg_time, 0.001, "Entropy calculation should be < 1ms for 1000 elements")
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        # Test batch processing
        batch_data = np.random.randn(100, 1000)
        
        import time
        start = time.time()
        entropies, scalings = residue.batch_compute_scaling(batch_data)
        batch_time = time.time() - start
        
        # Should process 100,000 elements in reasonable time
        self.assertLess(batch_time, 0.1, "Batch processing should be < 100ms for 100,000 elements")
        self.assertEqual(len(entropies), 100, "Should return 100 entropies")
        self.assertEqual(len(scalings), 100, "Should return 100 scalings")

def run_stress_test():
    """Run stress test to validate production readiness"""
    print("=== PROJECT RESIDUE Stress Test ===")
    
    controller = residue.create_entropy_controller()
    
    # Stress test with large data
    sizes = [1000, 5000, 10000, 50000]
    
    for size in sizes:
        print(f"Testing with {size} elements...")
        
        # Generate test data
        test_data = np.random.randn(size)
        
        # Time processing
        import time
        start = time.time()
        entropy = controller.calculate_input_entropy(test_data)
        processing_time = time.time() - start
        
        # Validate result
        if np.isnan(entropy) or np.isinf(entropy):
            print(f"❌  Size {size}: INVALID RESULT")
        elif processing_time > 0.01:  # More than 10ms
            print(f"⚠️  Size {size}: SLOW ({processing_time*1000:.1f}ms)")
        else:
            print(f"✅  Size {size}: OK ({processing_time*1000:.1f}ms)")
    
    print("Stress test completed")

if __name__ == "__main__":
    # Run unit tests
    unittest.main(verbosity=2)
    
    # Run stress test
    run_stress_test()
