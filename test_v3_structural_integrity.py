#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.0 - STRUCTURAL INTEGRITY TEST
TRUTH-SEEKER VALIDATION - No hiding behind old V2 tests
"""

import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, 'src')

def test_ema_jitter_reduction():
    """EMA Jitter Test: 10 identical vectors should converge smoothly"""
    print("=== EMA JITTER Test ===")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        controller.set_ema_alpha(0.3)  # Moderate EMA
        
        # Generate 10 identical vectors
        base_signal = np.sin(np.linspace(0, 2*np.pi, 100)) * 0.1
        identical_vectors = [base_signal + np.random.randn(100) * 0.001 for _ in range(10)]
        
        print("Testing EMA convergence with 10 identical vectors:")
        scaling_factors = []
        
        for i, vector in enumerate(identical_vectors):
            features = controller.extract_features_v3(vector)
            scaling = controller.compute_multi_dimensional_scaling_v3(features)
            scaling_factors.append(scaling)
            print(f"  Vector {i+1:2d}: scaling = {scaling:.6f}")
        
        # Check for EMA behavior (should converge, not be identical from start)
        first_scaling = scaling_factors[0]
        last_scaling = scaling_factors[-1]
        convergence = abs(last_scaling - first_scaling)
        
        print(f"\nEMA Analysis:")
        print(f"  First scaling: {first_scaling:.6f}")
        print(f"  Last scaling: {last_scaling:.6f}")
        print(f"  Convergence: {convergence:.6f}")
        
        # EMA should show some convergence behavior
        if convergence > 0.0001:
            print("✅ EMA working: Shows convergence behavior")
            return True
        else:
            print("❌ EMA failed: No convergence detected (possible empty buffer)")
            return False
            
    except Exception as e:
        print(f"❌ EMA test failed: {e}")
        return False

def test_7_feature_extraction():
    """7-Feature Extraction: Must return exactly 7 features"""
    print("\n=== 7-Feature Extraction Test ===")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Test with sample data
        test_data = np.random.randn(100)
        features = controller.extract_features_v3(test_data)
        
        print(f"Feature extraction result type: {type(features)}")
        print(f"Feature extraction result: {features}")
        
        # Check if it's a struct/object with attributes
        if hasattr(features, '__dict__'):
            feature_attrs = [attr for attr in dir(features) if not attr.startswith('_')]
            print(f"Available attributes: {feature_attrs}")
            
            # Count actual features
            feature_count = len(feature_attrs)
            print(f"Feature count: {feature_count}")
            
            # Check for V3.0 specific features
            v3_features = ['entropy', 'complexity', 'sparsity', 'structure', 
                          'temporal_coherence', 'zcr_rate', 'l1_sparsity']
            missing_features = [f for f in v3_features if f not in feature_attrs]
            
            if missing_features:
                print(f"❌ Missing V3.0 features: {missing_features}")
                return False
            
            if feature_count == 7:
                print("✅ 7-Feature extraction: SUCCESS")
                # Print actual values
                for attr in v3_features:
                    value = getattr(features, attr)
                    print(f"  {attr}: {value:.6f}")
                return True
            else:
                print(f"❌ Feature count mismatch: Expected 7, got {feature_count}")
                return False
        else:
            print(f"❌ Features not returned as object: {type(features)}")
            return False
            
    except Exception as e:
        print(f"❌ 7-Feature test failed: {e}")
        return False

def test_zcr_discrimination():
    """ZCR Discrimination: Sine wave vs random noise"""
    print("\n=== ZCR Discrimination Test ===")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Generate structured signal (sine wave)
        sine_wave = np.sin(np.linspace(0, 10*np.pi, 1000))
        
        # Generate chaotic signal (random noise)
        random_noise = np.random.randn(1000) * 0.5
        
        # Calculate ZCR for both
        zcr_sine = controller.calculate_zero_crossing_rate(sine_wave)
        zcr_noise = controller.calculate_zero_crossing_rate(random_noise)
        
        print(f"Sine wave ZCR: {zcr_sine:.6f}")
        print(f"Random noise ZCR: {zcr_noise:.6f}")
        print(f"ZCR difference: {abs(zcr_noise - zcr_sine):.6f}")
        
        # Random noise should have significantly higher ZCR
        if zcr_noise > zcr_sine * 2:
            print("✅ ZCR discrimination: SUCCESS (noise > sine)")
            return True
        else:
            print("❌ ZCR discrimination: FAILED (noise not significantly higher)")
            return False
            
    except Exception as e:
        print(f"❌ ZCR test failed: {e}")
        return False

def test_l1_sparsity():
    """L1-Sparsity: 90% zeros should return ~1.0"""
    print("\n=== L1-Sparsity Test ===")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Create vector with 90% zeros
        sparse_vector = np.zeros(1000)
        sparse_vector[:100] = np.random.randn(100)  # 10% non-zeros
        
        l1_sparsity = controller.calculate_l1_norm_sparsity(sparse_vector)
        
        print(f"90% sparse vector L1 sparsity: {l1_sparsity:.6f}")
        
        # Should be close to 1.0 for sparse data
        if l1_sparsity > 0.8:
            print("✅ L1-Sparsity: SUCCESS (correctly identifies sparse data)")
            return True
        else:
            print("❌ L1-Sparsity: FAILED (doesn't identify sparse data)")
            return False
            
    except Exception as e:
        print(f"❌ L1-Sparsity test failed: {e}")
        return False

def main():
    """Run complete V3.0 structural integrity validation"""
    print("PROJECT RESIDUE V3.0 - STRUCTURAL INTEGRITY VALIDATION")
    print("=" * 60)
    print("TRUTH-SEEKER TEST - No hiding behind V2 tests")
    print("=" * 60)
    
    # Run all critical tests
    ema_ok = test_ema_jitter_reduction()
    features_ok = test_7_feature_extraction()
    zcr_ok = test_zcr_discrimination()
    l1_ok = test_l1_sparsity()
    
    print("\n" + "=" * 60)
    print("V3.0 STRUCTURAL INTEGRITY RESULTS")
    print("=" * 60)
    
    print(f"\n🔍 Test Results:")
    print(f"EMA Jitter Reduction: {'✅ PASS' if ema_ok else '❌ FAIL'}")
    print(f"7-Feature Extraction: {'✅ PASS' if features_ok else '❌ FAIL'}")
    print(f"ZCR Discrimination: {'✅ PASS' if zcr_ok else '❌ FAIL'}")
    print(f"L1-Sparsity Detection: {'✅ PASS' if l1_ok else '❌ FAIL'}")
    
    # Overall status
    all_passed = ema_ok and features_ok and zcr_ok and l1_ok
    
    if all_passed:
        print(f"\n🎉 ALL STRUCTURAL INTEGRITY TESTS PASSED!")
        print("✅ PROJECT RESIDUE V3.0 - STRUCTURAL INTELLIGENCE VALIDATED")
        print("✅ EMA buffer working correctly")
        print("✅ 7-feature extraction functional")
        print("✅ ZCR discrimination accurate")
        print("✅ L1-sparsity detection working")
        print("\n🚀 READY FOR PRODUCTION DEPLOYMENT")
    else:
        print(f"\n⚠️  STRUCTURAL INTEGRITY ISSUES DETECTED!")
        print("❌ PROJECT RESIDUE V3.0 NEEDS IMMEDIATE FIXES")
        print("\n🔧 REQUIRED ACTIONS:")
        if not ema_ok:
            print("  - Fix EMA buffer implementation")
        if not features_ok:
            print("  - Fix 7-feature extraction logic")
        if not zcr_ok:
            print("  - Fix ZCR calculation")
        if not l1_ok:
            print("  - Fix L1-sparsity detection")
    
    print(f"\n🏁 Structural Integrity Check Complete")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
