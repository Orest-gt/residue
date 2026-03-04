#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.0 - Sanity Check Script
Verify EMA functionality and temporal coherence
"""

import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, 'src')

def test_ema_functionality():
    """Test that EMA reduces scaling jitter"""
    print("=== EMA FUNCTIONALITY TEST ===")
    
    try:
        import residue_v2
        print("✅ V2.0 module imported successfully")
        print("⚠️  Using V2.0 bindings with V3.0 features")
    except ImportError as e:
        print(f"❌ Failed to import residue_v2: {e}")
        print("⚠️  Try building with: python setup.py build_ext --inplace")
        return False
    
    # Create V3.0 controller with different EMA alphas
    controller_slow = residue_v2.create_entropy_controller_v2(256, 0.1, 5, 0.1)  # Slow EMA
    controller_fast = residue_v2.create_entropy_controller_v2(256, 0.1, 5, 0.1)  # Fast EMA
    
    # Set different EMA alphas
    controller_slow.set_ema_alpha(0.1)
    controller_fast.set_ema_alpha(0.5)
    
    # Generate similar consecutive samples (should produce similar scaling)
    base_signal = np.sin(np.linspace(0, 2*np.pi, 768)) * 0.1
    similar_samples = [base_signal + np.random.randn(768) * 0.01 for _ in range(10)]
    
    print("\nTesting EMA with different alpha values:")
    print(f"Slow EMA (alpha=0.1):")
    slow_scalings = []
    for i, sample in enumerate(similar_samples):
        features = controller_slow.extract_features_v3(sample)
        scaling = controller_slow.compute_multi_dimensional_scaling_v3(features)
        slow_scalings.append(scaling)
        if i % 3 == 0:
            print(f"  Sample {i:2d}: scaling = {scaling:.6f}")
    
    print(f"Fast EMA (alpha=0.5):")
    fast_scalings = []
    for i, sample in enumerate(similar_samples):
        features = controller_fast.extract_features_v3(sample)
        scaling = controller_fast.compute_multi_dimensional_scaling_v3(features)
        fast_scalings.append(scaling)
        if i % 3 == 0:
            print(f"  Sample {i:2d}: scaling = {scaling:.6f}")
    
    # Calculate jitter metrics
    slow_std = np.std(slow_scalings)
    fast_std = np.std(fast_scalings)
    slow_range = np.max(slow_scalings) - np.min(slow_scalings)
    fast_range = np.max(fast_scalings) - np.min(fast_scalings)
    
    print(f"\nJitter Analysis:")
    print(f"Slow EMA (alpha=0.1):")
    print(f"  Mean scaling: {np.mean(slow_scalings):.6f}")
    print(f"  Std deviation: {slow_std:.6f}")
    print(f"  Range: {slow_range:.6f}")
    
    print(f"Fast EMA (alpha=0.5):")
    print(f"  Mean scaling: {np.mean(fast_scalings):.6f}")
    print(f"  Std deviation: {fast_std:.6f}")
    print(f"  Range: {fast_range:.6f}")
    
    # EMA should reduce jitter (lower std dev)
    if slow_std < fast_std:
        print("✅ EMA working: Slow EMA produces less jitter than fast EMA")
        ema_working = True
    else:
        print("⚠️  EMA may not be working optimally")
        ema_working = False
    
    return ema_working

def test_dynamic_ema_adjustment():
    """Test dynamic EMA alpha adjustment"""
    print("\n=== DYNAMIC EMA ADJUSTMENT TEST ===")
    
    import residue_v2
    controller = residue_v2.create_entropy_controller_v2()
    
    # Test initial alpha
    initial_alpha = controller.get_ema_alpha()
    print(f"Initial EMA alpha: {initial_alpha:.3f}")
    
    # Test alpha adjustment
    new_alpha = 0.3
    controller.set_ema_alpha(new_alpha)
    adjusted_alpha = controller.get_ema_alpha()
    print(f"Adjusted EMA alpha: {adjusted_alpha:.3f}")
    
    if abs(adjusted_alpha - new_alpha) < 0.001:
        print("✅ Dynamic EMA adjustment working")
        return True
    else:
        print("❌ Dynamic EMA adjustment failed")
        return False

def test_zcr_normalization():
    """Test ZCR normalization (should be 0.0 to 1.0)"""
    print("\n=== ZCR NORMALIZATION TEST ===")
    
    import residue_v2
    controller = residue_v2.create_entropy_controller_v2()
    
    # Test signals with known ZCR characteristics
    test_signals = {
        'DC Signal': np.ones(768) * 0.5,  # ZCR = 0.0
        'Low Frequency': np.sin(np.linspace(0, 2*np.pi, 768)),  # Low ZCR
        'High Frequency': np.sin(np.linspace(0, 50*np.pi, 768)),  # High ZCR
        'White Noise': np.random.randn(768),  # Variable ZCR
    }
    
    print(f"{'Signal Type':15s} {'ZCR Rate':10s} {'Normalized':12s}")
    print("-" * 45)
    
    zcr_normalized = True
    for signal_name, signal in test_signals.items():
        zcr = controller.calculate_zero_crossing_rate(signal)
        
        # Check if ZCR is normalized (0.0 to 1.0)
        is_normalized = 0.0 <= zcr <= 1.0
        status = "✅" if is_normalized else "❌"
        
        print(f"{signal_name:15s} {zcr:10.6f} {status:12s}")
        
        if not is_normalized:
            zcr_normalized = False
    
    print("-" * 45)
    
    if zcr_normalized:
        print("✅ All ZCR values properly normalized")
    else:
        print("❌ ZCR normalization issues detected")
    
    return zcr_normalized

def test_l1_sparsity_detection():
    """Test L1-norm sparsity detection"""
    print("\n=== L1-NORM SPARSITY TEST ===")
    
    import residue_v2
    controller = residue_v2.create_entropy_controller_v2()
    
    # Test data with different sparsity levels
    test_cases = {
        'Dense': np.random.randn(768),
        'Medium Sparse': np.concatenate([np.random.randn(384), np.zeros(384)]),
        'Very Sparse': np.concatenate([np.random.randn(100), np.zeros(668)]),
        'All Zeros': np.zeros(768)
    }
    
    print(f"{'Case':12s} {'L1-Sparsity':12s} {'Expected':12s}")
    print("-" * 40)
    
    sparsity_working = True
    for case_name, data in test_cases.items():
        l1_sparsity = controller.calculate_l1_norm_sparsity(data)
        
        # Expected behavior: more zeros = higher sparsity
        if case_name == 'All Zeros':
            expected = "High (>0.8)"
            working = l1_sparsity > 0.8
        elif case_name == 'Very Sparse':
            expected = "High (>0.6)"
            working = l1_sparsity > 0.6
        elif case_name == 'Medium Sparse':
            expected = "Medium (0.3-0.6)"
            working = 0.3 <= l1_sparsity <= 0.6
        else:  # Dense
            expected = "Low (<0.3)"
            working = l1_sparsity < 0.3
        
        status = "✅" if working else "❌"
        print(f"{case_name:12s} {l1_sparsity:12.6f} {expected:12s} {status}")
        
        if not working:
            sparsity_working = False
    
    print("-" * 40)
    
    if sparsity_working:
        print("✅ L1-norm sparsity detection working correctly")
    else:
        print("❌ L1-norm sparsity detection issues")
    
    return sparsity_working

def test_v3_performance_overhead():
    """Test V3.0 performance overhead"""
    print("\n=== V3.0 PERFORMANCE OVERHEAD TEST ===")
    
    import residue_v2
    controller = residue_v2.create_entropy_controller_v2()
    
    # Test with different input sizes
    sizes = [100, 500, 1000, 5000]
    
    print(f"{'Size':6s} {'Time (ms)':10s} {'Status':8s}")
    print("-" * 30)
    
    performance_ok = True
    for size in sizes:
        data = np.random.randn(size)
        
        # Time V3.0 processing
        start_time = time.time()
        features = controller.extract_features_v3(data)
        scaling = controller.compute_multi_dimensional_scaling_v3(features)
        processing_time = (time.time() - start_time) * 1000
        
        status = "✅ OK" if processing_time < 0.01 else "⚠️ High"
        print(f"{size:6d} {processing_time:8.3f} {status:8s}")
        
        if processing_time >= 0.01:
            performance_ok = False
    
    print("-" * 30)
    
    if performance_ok:
        print("✅ V3.0 performance overhead < 0.01ms requirement met")
    else:
        print("⚠️  V3.0 performance overhead exceeds 0.01ms requirement")
    
    return performance_ok

def main():
    """Run comprehensive V3.0 sanity check"""
    print("PROJECT RESIDUE V3.0 - SANITY CHECK")
    print("=" * 50)
    
    # Run all tests
    ema_ok = test_ema_functionality()
    dynamic_ema_ok = test_dynamic_ema_adjustment()
    zcr_ok = test_zcr_normalization()
    l1_ok = test_l1_sparsity_detection()
    perf_ok = test_v3_performance_overhead()
    
    print("\n" + "=" * 50)
    print("V3.0 SANITY CHECK RESULTS")
    print("=" * 50)
    
    print(f"\n🎯 Test Results:")
    print(f"EMA Functionality: {'✅ PASS' if ema_ok else '❌ FAIL'}")
    print(f"Dynamic EMA: {'✅ PASS' if dynamic_ema_ok else '❌ FAIL'}")
    print(f"ZCR Normalization: {'✅ PASS' if zcr_ok else '❌ FAIL'}")
    print(f"L1 Sparsity: {'✅ PASS' if l1_ok else '❌ FAIL'}")
    print(f"Performance: {'✅ PASS' if perf_ok else '❌ FAIL'}")
    
    # Overall status
    all_passed = ema_ok and dynamic_ema_ok and zcr_ok and l1_ok and perf_ok
    
    if all_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print("✅ PROJECT RESIDUE V3.0 - STRUCTURAL HEURISTICS READY")
        print("✅ EMA reduces scaling jitter")
        print("✅ Dynamic configuration working")
        print("✅ ZCR properly normalized")
        print("✅ L1 sparsity detection accurate")
        print("✅ Performance requirements met")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print("❌ PROJECT RESIDUE V3.0 needs attention")
    
    print(f"\n🏁 Sanity Check Complete")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
