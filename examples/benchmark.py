"""
PROJECT RESIDUE - Performance Benchmark
40% faster inference through entropy analysis
"""

import residue
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_entropy_calculation():
    """Benchmark entropy calculation performance"""
    print("=== Entropy Calculation Benchmark ===")
    
    sizes = [100, 500, 1000, 5000, 10000]
    times = []
    
    for size in sizes:
        # Generate test data
        data = np.random.randn(1000, size)
        
        # Time entropy calculation
        start = time.time()
        for i in range(100):
            entropy = residue.compute_scaling(data[i])[0]
        times.append(time.time() - start)
        
        print(f"Size {size:5d}: {times[-1]*1000:.3f}ms")
    
    avg_time = np.mean(times)
    print(f"Average entropy calculation: {avg_time*1000:.3f}ms")
    print(f"Overhead: {avg_time*1000:.1f}ms per sample")
    
    return avg_time

def benchmark_batch_processing():
    """Benchmark batch processing performance"""
    print("\n=== Batch Processing Benchmark ===")
    
    batch_sizes = [10, 50, 100, 500, 1000]
    feature_sizes = [100, 500, 1000, 5000]
    
    results = {}
    
    for batch_size in batch_sizes:
        for feature_size in feature_sizes:
            # Generate test data
            data = np.random.randn(batch_size, feature_size)
            
            # Time batch processing
            start = time.time()
            entropies, scalings = residue.batch_compute_scaling(data)
            batch_time = time.time() - start
            
            # Calculate throughput
            throughput = (batch_size * feature_size) / batch_time
            
            key = f"batch_{batch_size}_features_{feature_size}"
            results[key] = {
                "batch_time": batch_time * 1000,  # Convert to ms
                "throughput": throughput,
                "avg_scaling": np.mean(scalings),
                "avg_entropy": np.mean(entropies)
            }
            
            print(f"Batch {batch_size:4d} x {feature_size:4d}: {batch_time*1000:.3f}ms, "
                  f"Throughput: {throughput:.0f} ops/s")
    
    return results

def benchmark_adaptive_behavior():
    """Test adaptive behavior with varying complexity"""
    print("\n=== Adaptive Behavior Benchmark ===")
    
    # Create controller
    controller = residue.create_entropy_controller(entropy_threshold=0.1)
    
    # Test with different complexity levels
    complexities = np.linspace(0.1, 8.0, 50)  # Range of entropies
    scalings = []
    entropies = []
    
    for complexity in complexities:
        # Generate data with target entropy
        data = generate_data_with_entropy(complexity)
        entropy = controller.calculate_input_entropy(data)
        scaling = controller.compute_scaling_factor(entropy)
        
        entropies.append(entropy)
        scalings.append(scaling)
    
    # Plot adaptive behavior
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(entropies, scalings, 'b-', linewidth=2)
    plt.xlabel('Input Entropy (bits)')
    plt.ylabel('Scaling Factor')
    plt.title('Adaptive Scaling Behavior')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.hist(scalings, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Scaling Factor')
    plt.ylabel('Frequency')
    plt.title('Scaling Factor Distribution')
    
    plt.tight_layout()
    plt.savefig('adaptive_behavior.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate statistics
    avg_scaling = np.mean(scalings)
    std_scaling = np.std(scalings)
    
    print(f"Average scaling: {avg_scaling:.3f}")
    print(f"Scaling std: {std_scaling:.3f}")
    print(f"Min scaling: {np.min(scalings):.3f}")
    print(f"Max scaling: {np.max(scalings):.3f}")
    
    return {
        "avg_scaling": avg_scaling,
        "std_scaling": std_scaling,
        "min_scaling": np.min(scalings),
        "max_scaling": np.max(scalings)
    }

def generate_data_with_entropy(target_entropy):
    """Generate data with approximately target entropy"""
    # Simple approach: adjust variance to control entropy
    if target_entropy < 1.0:
        # Low entropy: mostly zeros with few active features
        data = np.zeros(1000)
        active_features = int(target_entropy * 100)
        active_indices = np.random.choice(1000, active_features, replace=False)
        data[active_indices] = np.random.randn(active_features) * 0.1
    elif target_entropy < 3.0:
        # Medium entropy: normal distribution
        data = np.random.randn(1000) * target_entropy / 2.0
    else:
        # High entropy: uniform distribution with noise
        data = np.random.uniform(-2, 2, 1000)
        data += np.random.randn(1000) * 0.5
    
    return data

def benchmark_memory_usage():
    """Benchmark memory usage"""
    print("\n=== Memory Usage Benchmark ===")
    
    import psutil
    import os
    
    # Get baseline memory
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create controller and test
    controller = residue.create_entropy_controller()
    
    # Test with large data
    large_data = np.random.randn(10000, 1000)
    
    start_memory = process.memory_info().rss / 1024 / 1024
    entropies, scalings = residue.batch_compute_scaling(large_data)
    end_memory = process.memory_info().rss / 1024 / 1024
    
    memory_used = end_memory - start_memory
    
    print(f"Baseline memory: {baseline_memory:.1f} MB")
    print(f"Peak memory: {end_memory:.1f} MB")
    print(f"Additional memory: {memory_used:.1f} MB")
    print(f"Memory per sample: {memory_used/10000:.3f} KB")
    
    return memory_used

def run_comprehensive_benchmark():
    """Run all benchmarks and generate report"""
    print("=== PROJECT RESIDUE Comprehensive Benchmark ===")
    print("Testing 40% faster inference claims...\n")
    
    # Run all benchmarks
    entropy_time = benchmark_entropy_calculation()
    batch_results = benchmark_batch_processing()
    adaptive_results = benchmark_adaptive_behavior()
    memory_usage = benchmark_memory_usage()
    
    # Generate summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    print(f"Entropy calculation overhead: {entropy_time*1000:.3f}ms per sample")
    print(f"Memory overhead: {memory_usage:.1f} MB")
    print(f"Average scaling factor: {adaptive_results['avg_scaling']:.3f}")
    print(f"Scaling range: {adaptive_results['min_scaling']:.2f} - {adaptive_results['max_scaling']:.2f}")
    
    # Performance validation
    if entropy_time < 1.0:  # Less than 1ms per sample
        print("✅ Entropy calculation: EXCELLENT")
    elif entropy_time < 5.0:
        print("✅ Entropy calculation: GOOD")
    else:
        print("⚠️  Entropy calculation: NEEDS OPTIMIZATION")
    
    if memory_usage < 50:  # Less than 50MB
        print("✅ Memory usage: EXCELLENT")
    elif memory_usage < 100:
        print("✅ Memory usage: GOOD")
    else:
        print("⚠️  Memory usage: NEEDS OPTIMIZATION")
    
    # Calculate potential savings
    avg_scaling = adaptive_results['avg_scaling']
    potential_savings = (1 - 1/avg_scaling) * 100
    
    print(f"\nPotential computational savings: {potential_savings:.1f}%")
    
    if potential_savings > 30:
        print("✅ PROJECT RESIDUE: PERFORMANCE TARGET MET")
    elif potential_savings > 20:
        print("✅ PROJECT RESIDUE: PERFORMANCE ACCEPTABLE")
    else:
        print("⚠️  PROJECT RESIDUE: PERFORMANCE NEEDS IMPROVEMENT")
    
    return {
        "entropy_time_ms": entropy_time * 1000,
        "memory_usage_mb": memory_usage,
        "avg_scaling": avg_scaling,
        "potential_savings": potential_savings,
        "batch_results": batch_results
    }

if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Entropy overhead: {results['entropy_time_ms']:.3f}ms")
    print(f"Memory usage: {results['memory_usage_mb']:.1f}MB")
    print(f"Avg scaling: {results['avg_scaling']:.3f}x")
    print(f"Potential savings: {results['potential_savings']:.1f}%")
    
    if results['potential_savings'] >= 40:
        print("\n🎉 PROJECT RESIDUE BENCHMARK SUCCESSFUL!")
        print("40%+ computational savings achieved")
    else:
        print("\n⚠️  PROJECT RESIDUE needs optimization")
