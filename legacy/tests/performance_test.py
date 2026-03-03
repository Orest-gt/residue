#!/usr/bin/env python3
"""
PROJECT RESIDUE - Comprehensive Performance Validation
Empirical evidence for all performance claims
"""

import sys
import time
import numpy as np

# Add src to path
sys.path.insert(0, 'src')
import residue

def test_entropy_scaling_behavior():
    """Test scaling behavior across entropy levels"""
    print("=== ENTROPY SCALING BEHAVIOR ===")
    
    # Zero entropy (constant data)
    zero_data = np.ones(100)
    entropy, scaling = residue.compute_scaling(zero_data)
    print(f"Zero Entropy:   {entropy:.3f} bits → {scaling:.2f}x scaling")
    
    # Low entropy (sparse data)
    low_data = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1] * 10)
    entropy, scaling = residue.compute_scaling(low_data)
    print(f"Low Entropy:    {entropy:.3f} bits → {scaling:.2f}x scaling")
    
    # Medium entropy (random small)
    med_data = np.random.randn(100)
    entropy, scaling = residue.compute_scaling(med_data)
    print(f"Medium Entropy: {entropy:.3f} bits → {scaling:.2f}x scaling")
    
    # High entropy (random large)
    high_data = np.random.randn(1000)
    entropy, scaling = residue.compute_scaling(high_data)
    print(f"High Entropy:   {entropy:.3f} bits → {scaling:.2f}x scaling")

def test_performance_overhead():
    """Test <1ms overhead claim"""
    print("\n=== PERFORMANCE OVERHEAD TEST ===")
    
    test_sizes = [10, 50, 100, 500, 1000]
    
    for size in test_sizes:
        data = np.random.randn(size)
        
        # Time single sample processing
        start = time.time()
        entropy, scaling = residue.compute_scaling(data)
        elapsed = (time.time() - start) * 1000
        
        print(f"Size {size:4d}: {elapsed:6.3f}ms, Entropy: {entropy:5.2f}, Scaling: {scaling:4.1f}x")
        
        # Check <1ms claim
        if elapsed < 1.0:
            print(f"  ✅ <1ms overhead achieved")
        else:
            print(f"  ❌ <1ms overhead FAILED")

def test_batch_efficiency():
    """Test batch processing efficiency"""
    print("\n=== BATCH PROCESSING EFFICIENCY ===")
    
    batch_sizes = [10, 50, 100, 500]
    
    for batch_size in batch_sizes:
        data = np.random.randn(batch_size, 1000)
        
        # Time batch processing
        start = time.time()
        entropies, scalings = residue.batch_compute_scaling(data)
        elapsed = (time.time() - start) * 1000
        
        per_sample_time = elapsed / batch_size
        throughput = (batch_size * 1000) / (elapsed / 1000)  # elements per second
        
        print(f"Batch {batch_size:3d}: {elapsed:6.3f}ms total, {per_sample_time:6.3f}ms per sample")
        print(f"           Throughput: {throughput:8.0f} elements/sec")
        print(f"           Avg Scaling: {np.mean(scalings):.2f}x")

def test_computational_savings():
    """Test 47% computational savings claim"""
    print("\n=== COMPUTATIONAL SAVINGS TEST ===")
    
    # Test different entropy levels
    test_cases = [
        ("Zero Entropy", np.ones(1000)),
        ("Low Entropy", np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1] * 100)),
        ("Medium Entropy", np.random.randn(1000)),
        ("High Entropy", np.random.randn(1000) * 2)
    ]
    
    for name, data in test_cases:
        entropy, scaling = residue.compute_scaling(data)
        
        # Calculate potential savings
        if scaling > 1.0:
            savings = (1 - 1/scaling) * 100
        else:
            savings = (scaling - 1) * 100  # Negative savings (more computation)
        
        print(f"{name:15s}: {entropy:5.2f} bits → {scaling:4.1f}x scaling → {savings:6.1f}% savings")
        
        # Check 47% claim
        if savings >= 47:
            print(f"  ✅ 47%+ savings achieved")
        else:
            print(f"  ❌ 47% savings NOT achieved")

def test_memory_usage():
    """Test memory footprint"""
    print("\n=== MEMORY USAGE TEST ===")
    
    try:
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create controller and process data
        controller = residue.create_entropy_controller()
        data = np.random.randn(10000, 1000)
        
        # Process large batch
        entropies, scalings = residue.batch_compute_scaling(data)
        
        # Peak memory
        peak = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak - baseline
        
        print(f"Baseline Memory: {baseline:.1f} MB")
        print(f"Peak Memory:    {peak:.1f} MB")
        print(f"Additional:     {memory_used:.1f} MB")
        print(f"Memory per sample: {memory_used/10000:.3f} KB")
        
        # Check <10MB claim
        if memory_used < 10:
            print(f"  ✅ <10MB memory usage")
        else:
            print(f"  ❌ <10MB memory usage FAILED")
            
    except ImportError:
        print("psutil not available - cannot measure memory usage")

def main():
    """Run all performance validation tests"""
    print("PROJECT RESIDUE - PERFORMANCE VALIDATION")
    print("=" * 50)
    
    test_entropy_scaling_behavior()
    test_performance_overhead()
    test_batch_efficiency()
    test_computational_savings()
    test_memory_usage()
    
    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()
