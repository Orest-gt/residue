#!/usr/bin/env python3
"""
PROJECT RESIDUE V2.1 - ULTIMATE STRESS & INTEGRITY TEST
Testing Structural Intelligence under extreme conditions
"""

import sys
import numpy as np
import json

# Add src to path
sys.path.insert(0, 'src')

def test_silence_test():
    """Test 1: The "Silence" Test - Near-zero signal"""
    print("=== TEST 1: SILENCE TEST ===")
    print("Vector: All values = 0.0001 (near-zero signal)")
    print("Expected: L1-sparsity explodes, Maximum scaling factor")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Create near-zero signal
        silence_vector = np.full(1000, 0.0001)
        
        # Extract features
        features = controller.extract_features_v3(silence_vector)
        scaling = controller.compute_multi_dimensional_scaling_v3(features)
        
        # Results
        results = {
            'entropy': features.entropy,
            'complexity': features.complexity,
            'sparsity': features.sparsity,
            'structure': features.structure,
            'temporal_coherence': features.temporal_coherence,
            'zcr_rate': features.zcr_rate,
            'l1_sparsity': features.l1_sparsity,
            'scaling': scaling
        }
        
        print(f"\nResults:")
        print(f"  L1 Sparsity: {results['l1_sparsity']:.6f} (Expected: >0.9)")
        print(f"  Scaling: {results['scaling']:.6f} (Expected: High)")
        print(f"  Entropy: {results['entropy']:.6f}")
        print(f"  ZCR: {results['zcr_rate']:.6f}")
        
        # Validation
        l1_ok = results['l1_sparsity'] > 0.9
        scaling_ok = results['scaling'] > 1.0
        
        print(f"\nValidation:")
        print(f"  L1 Sparsity > 0.9: {'✅ PASS' if l1_ok else '❌ FAIL'}")
        print(f"  Scaling High: {'✅ PASS' if scaling_ok else '❌ FAIL'}")
        
        return results, l1_ok and scaling_ok
        
    except Exception as e:
        print(f"❌ Silence test failed: {e}")
        return None, False

def test_chaos_test():
    """Test 2: The "Chaos" Test - Pure random noise"""
    print("\n=== TEST 2: CHAOS TEST ===")
    print("Vector: Random uniform(-1, 1) values")
    print("Expected: ZCR ≈ 0.5, Entropy at ceiling, High scaling due to noise")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Create pure chaos
        chaos_vector = np.random.uniform(-1, 1, 1000)
        
        # Extract features
        features = controller.extract_features_v3(chaos_vector)
        scaling = controller.compute_multi_dimensional_scaling_v3(features)
        
        # Results
        results = {
            'entropy': features.entropy,
            'complexity': features.complexity,
            'sparsity': features.sparsity,
            'structure': features.structure,
            'temporal_coherence': features.temporal_coherence,
            'zcr_rate': features.zcr_rate,
            'l1_sparsity': features.l1_sparsity,
            'scaling': scaling
        }
        
        print(f"\nResults:")
        print(f"  ZCR: {results['zcr_rate']:.6f} (Expected: ≈0.5)")
        print(f"  Entropy: {results['entropy']:.6f} (Expected: High)")
        print(f"  Scaling: {results['scaling']:.6f} (Expected: High)")
        print(f"  L1 Sparsity: {results['l1_sparsity']:.6f}")
        
        # Validation
        zcr_ok = 0.3 < results['zcr_rate'] < 0.7  # Around 0.5
        entropy_ok = results['entropy'] > 5.0  # High entropy
        scaling_ok = results['scaling'] > 1.0  # High scaling
        
        print(f"\nValidation:")
        print(f"  ZCR ≈ 0.5: {'✅ PASS' if zcr_ok else '❌ FAIL'}")
        print(f"  High Entropy: {'✅ PASS' if entropy_ok else '❌ FAIL'}")
        print(f"  High Scaling: {'✅ PASS' if scaling_ok else '❌ FAIL'}")
        
        return results, zcr_ok and entropy_ok and scaling_ok
        
    except Exception as e:
        print(f"❌ Chaos test failed: {e}")
        return None, False

def test_pattern_test():
    """Test 3: The "Pattern" Test - Alternating pattern"""
    print("\n=== TEST 3: PATTERN TEST ===")
    print("Vector: Alternating [1, -1, 1, -1, ...]")
    print("Expected: ZCR = 1.0 (maximum), System detects extreme frequency")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Create alternating pattern
        pattern_vector = np.array([1, -1] * 500)  # 1000 elements
        
        # Extract features
        features = controller.extract_features_v3(pattern_vector)
        scaling = controller.compute_multi_dimensional_scaling_v3(features)
        
        # Results
        results = {
            'entropy': features.entropy,
            'complexity': features.complexity,
            'sparsity': features.sparsity,
            'structure': features.structure,
            'temporal_coherence': features.temporal_coherence,
            'zcr_rate': features.zcr_rate,
            'l1_sparsity': features.l1_sparsity,
            'scaling': scaling
        }
        
        print(f"\nResults:")
        print(f"  ZCR: {results['zcr_rate']:.6f} (Expected: 1.0)")
        print(f"  Complexity: {results['complexity']:.6f} (Expected: High)")
        print(f"  Scaling: {results['scaling']:.6f} (Expected: High due to complexity)")
        print(f"  Entropy: {results['entropy']:.6f}")
        
        # Validation
        zcr_ok = abs(results['zcr_rate'] - 1.0) < 0.01  # Very close to 1.0
        complexity_ok = results['complexity'] > 0.5  # High complexity
        scaling_ok = results['scaling'] > 1.0  # High scaling
        
        print(f"\nValidation:")
        print(f"  ZCR = 1.0: {'✅ PASS' if zcr_ok else '❌ FAIL'}")
        print(f"  High Complexity: {'✅ PASS' if complexity_ok else '❌ FAIL'}")
        print(f"  High Scaling: {'✅ PASS' if scaling_ok else '❌ FAIL'}")
        
        return results, zcr_ok and complexity_ok and scaling_ok
        
    except Exception as e:
        print(f"❌ Pattern test failed: {e}")
        return None, False

def test_ema_lag_test():
    """Test 4: EMA Lag Test - Smooth transition"""
    print("\n=== TEST 4: EMA LAG TEST ===")
    print("Sequence: 5 Chaos vectors → 5 Sine vectors")
    print("Expected: Smooth decay (EMA), not sudden drop")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        controller.set_ema_alpha(0.3)  # Moderate EMA
        
        # Create test sequence
        chaos_vectors = [np.random.uniform(-1, 1, 1000) for _ in range(5)]
        sine_vectors = [np.sin(np.linspace(0, 4*np.pi, 1000)) * 0.5 for _ in range(5)]
        
        # Test sequence
        all_vectors = chaos_vectors + sine_vectors
        scalings = []
        
        print(f"\nScaling progression:")
        for i, vector in enumerate(all_vectors):
            features = controller.extract_features_v3(vector)
            scaling = controller.compute_multi_dimensional_scaling_v3(features)
            scalings.append(scaling)
            
            vector_type = "Chaos" if i < 5 else "Sine"
            print(f"  Step {i+1:2d} ({vector_type:6s}): {scaling:.6f}")
        
        # Analyze EMA behavior
        chaos_scalings = scalings[:5]
        sine_scalings = scalings[5:]
        
        # Check for smooth transition
        transition_drop = scalings[5] - scalings[4]  # Drop from chaos to sine
        avg_chaos = np.mean(chaos_scalings)
        avg_sine = np.mean(sine_scalings)
        
        print(f"\nEMA Analysis:")
        print(f"  Average Chaos scaling: {avg_chaos:.6f}")
        print(f"  Average Sine scaling: {avg_sine:.6f}")
        print(f"  Transition drop: {transition_drop:.6f}")
        print(f"  Smoothness (|drop| < 0.5): {'✅ PASS' if abs(transition_drop) < 0.5 else '❌ FAIL'}")
        
        # Validation
        smooth_ok = abs(transition_drop) < 0.5  # Smooth transition
        chaos_higher = avg_chaos > avg_sine  # Chaos should have higher scaling
        
        print(f"  Chaos > Sine: {'✅ PASS' if chaos_higher else '❌ FAIL'}")
        
        return scalings, smooth_ok and chaos_higher
        
    except Exception as e:
        print(f"❌ EMA lag test failed: {e}")
        return None, False

def generate_results_table(test_results):
    """Generate comprehensive results table"""
    print("\n" + "=" * 80)
    print("PROJECT RESIDUE V2.1 - ULTIMATE VALIDATION RESULTS")
    print("=" * 80)
    
    table_data = []
    
    # Test 1: Silence
    if test_results[0][0]:
        silence = test_results[0][0]
        table_data.append({
            'Test': 'Silence (Near-Zero)',
            'L1 Sparsity': f"{silence['l1_sparsity']:.3f}",
            'Expected': '>0.9',
            'Scaling': f"{silence['scaling']:.3f}",
            'Status': '✅ PASS' if test_results[0][1] else '❌ FAIL'
        })
    
    # Test 2: Chaos
    if test_results[1][0]:
        chaos = test_results[1][0]
        table_data.append({
            'Test': 'Chaos (Random)',
            'ZCR': f"{chaos['zcr_rate']:.3f}",
            'Expected': '≈0.5',
            'Scaling': f"{chaos['scaling']:.3f}",
            'Status': '✅ PASS' if test_results[1][1] else '❌ FAIL'
        })
    
    # Test 3: Pattern
    if test_results[2][0]:
        pattern = test_results[2][0]
        table_data.append({
            'Test': 'Pattern (Alternating)',
            'ZCR': f"{pattern['zcr_rate']:.3f}",
            'Expected': '1.0',
            'Scaling': f"{pattern['scaling']:.3f}",
            'Status': '✅ PASS' if test_results[2][1] else '❌ FAIL'
        })
    
    # Test 4: EMA
    if test_results[3][0]:
        ema_scalings = test_results[3][0]
        table_data.append({
            'Test': 'EMA Lag (Smoothness)',
            'Transition': f"{ema_scalings[5] - ema_scalings[4]:.3f}",
            'Expected': '<0.5',
            'Status': '✅ PASS' if test_results[3][1] else '❌ FAIL'
        })
    
    # Print table
    print(f"{'Test':<20} {'Metric':<12} {'Expected':<12} {'Scaling':<10} {'Status':<8}")
    print("-" * 70)
    
    for row in table_data:
        if 'Metric' in row:
            print(f"{row['Test']:<20} {row['Metric']:<12} {row['Expected']:<12} {row['Scaling']:<10} {row['Status']:<8}")
        else:
            print(f"{row['Test']:<20} {row['Transition']:<12} {row['Expected']:<12} {'':<10} {row['Status']:<8}")
    
    return table_data

def main():
    """Main ultimate validation test"""
    print("PROJECT RESIDUE V2.1 - ULTIMATE STRESS & INTEGRITY TEST")
    print("=" * 80)
    print("Testing Structural Intelligence under extreme conditions")
    print("=" * 80)
    
    # Run all tests
    test_results = []
    
    # Test 1: Silence
    silence_result, silence_ok = test_silence_test()
    test_results.append((silence_result, silence_ok))
    
    # Test 2: Chaos
    chaos_result, chaos_ok = test_chaos_test()
    test_results.append((chaos_result, chaos_ok))
    
    # Test 3: Pattern
    pattern_result, pattern_ok = test_pattern_test()
    test_results.append((pattern_result, pattern_ok))
    
    # Test 4: EMA Lag
    ema_result, ema_ok = test_ema_lag_test()
    test_results.append((ema_result, ema_ok))
    
    # Generate results table
    results_table = generate_results_table(test_results)
    
    # Overall assessment
    all_passed = all(result[1] for result in test_results)
    
    print("\n" + "=" * 80)
    print("FINAL ASSESSMENT")
    print("=" * 80)
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ PROJECT RESIDUE V2.1 - STRUCTURAL INTELLIGENCE: IMPENETRABLE")
        print("✅ Architecture validated under extreme conditions")
        print("✅ EMA smoothness confirmed")
        print("✅ Structural heuristics working optimally")
        print("\n🚀 V2.1 READY FOR PRODUCTION DEPLOYMENT")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("❌ Architecture needs refinement")
        
        failed_tests = []
        test_names = ['Silence', 'Chaos', 'Pattern', 'EMA']
        for i, (result, passed) in enumerate(test_results):
            if not passed:
                failed_tests.append(test_names[i])
        
        print(f"❌ Failed tests: {', '.join(failed_tests)}")
        print("\n🔧 REQUIRED ACTIONS:")
        if not test_results[0][1]:
            print("  - Fix L1 sparsity detection for near-zero signals")
        if not test_results[1][1]:
            print("  - Improve noise discrimination")
        if not test_results[2][1]:
            print("  - Enhance pattern recognition")
        if not test_results[3][1]:
            print("  - Fix EMA smoothness")
    
    # Save results
    results_json = {
        'version': 'V2.1',
        'timestamp': str(np.datetime64('now')),
        'all_passed': all_passed,
        'test_results': results_table
    }
    
    with open('v21_ultimate_validation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n💾 Results saved to 'v21_ultimate_validation_results.json'")
    print(f"\n🏁 Ultimate Validation Complete")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
