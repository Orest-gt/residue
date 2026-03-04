#!/usr/bin/env python3
"""
PROJECT RESIDUE V2.1 - VALIDATION COMPARISON
Compare V3.0 vs V2.1 weights impact on scaling behavior
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def simulate_v3_scaling(features):
    """Simulate old V3.0 scaling weights"""
    weights = [
        features.entropy * 15.0,      # High conservative weight
        features.complexity * 10.0,    # High conservative weight
        features.sparsity * 8.0,       # Medium weight
        features.structure * 5.0,      # Medium weight
        features.temporal_coherence * 3.0,  # Low structural weight
        features.zcr_rate * 2.0,       # Very low structural weight
        features.l1_sparsity * 1.0     # Very low structural weight
    ]
    return sum(weights) / 44.0  # Normalize by total old weight

def simulate_v2_1_scaling(features):
    """Simulate new V2.1 scaling weights"""
    weights = [
        features.entropy * 8.0,       # Reduced conservative weight
        features.complexity * 6.0,     # Reduced conservative weight
        features.sparsity * 8.0,       # Maintained weight
        features.structure * 5.0,      # Stable weight
        features.temporal_coherence * 4.0,  # Increased structural weight
        features.zcr_rate * 6.0,       # 3x increased structural weight
        features.l1_sparsity * 5.0     # 5x increased structural weight
    ]
    return sum(weights) / 42.0  # Normalize by total new weight

def test_weight_impact():
    """Test the impact of weight changes on different data types"""
    print("=== V2.1 vs V3.0 WEIGHT IMPACT COMPARISON ===")
    
    try:
        import residue_v2
        controller = residue_v2.create_entropy_controller_v2()
        
        # Test scenarios
        scenarios = {
            'structured_signal': np.sin(np.linspace(0, 4*np.pi, 1000)),
            'high_frequency_noise': np.random.randn(1000) * 0.5,
            'sparse_data': np.concatenate([np.random.randn(100), np.zeros(900)]),
            'mixed_complex': np.sin(np.linspace(0, 10*np.pi, 500)) + np.random.randn(500) * 0.2
        }
        
        print("\nScaling Comparison (V3.0 vs V2.1):")
        print("=" * 70)
        print(f"{'Scenario':20s} {'V3.0':8s} {'V2.1':8s} {'Change':8s} {'ZCR':8s} {'L1':8s}")
        print("=" * 70)
        
        for scenario_name, data in scenarios.items():
            features = controller.extract_features_v3(data)
            
            # Simulate both weight configurations
            v3_scaling = simulate_v3_scaling(features)
            v2_1_scaling = simulate_v2_1_scaling(features)
            
            # Calculate change
            change = ((v2_1_scaling - v3_scaling) / v3_scaling) * 100
            change_str = f"{change:+.1f}%"
            
            print(f"{scenario_name:20s} {v3_scaling:8.3f} {v2_1_scaling:8.3f} {change_str:8s} "
                  f"{features.zcr_rate:8.3f} {features.l1_sparsity:8.3f}")
        
        print("=" * 70)
        
        # Analyze specific improvements
        print("\n🔍 IMPROVEMENT ANALYSIS:")
        
        # Test noise discrimination
        noise_data = np.random.randn(1000) * 0.5
        structured_data = np.sin(np.linspace(0, 4*np.pi, 1000))
        
        noise_features = controller.extract_features_v3(noise_data)
        structured_features = controller.extract_features_v3(structured_data)
        
        noise_v3 = simulate_v3_scaling(noise_features)
        noise_v2_1 = simulate_v2_1_scaling(noise_features)
        structured_v3 = simulate_v3_scaling(structured_features)
        structured_v2_1 = simulate_v2_1_scaling(structured_features)
        
        # Noise penalty (higher = more conservative on noise)
        noise_penalty_v3 = noise_v3 - structured_v3
        noise_penalty_v2_1 = noise_v2_1 - structured_v2_1
        
        print(f"Noise Discrimination:")
        print(f"  V3.0 noise penalty: {noise_penalty_v3:+.3f}")
        print(f"  V2.1 noise penalty: {noise_penalty_v2_1:+.3f}")
        print(f"  Improvement: {((noise_penalty_v2_1 - noise_penalty_v3) / abs(noise_penalty_v3) * 100):+.1f}%")
        
        # Test sparse data handling
        sparse_data = np.concatenate([np.random.randn(100), np.zeros(900)])
        dense_data = np.random.randn(1000)
        
        sparse_features = controller.extract_features_v3(sparse_data)
        dense_features = controller.extract_features_v3(dense_data)
        
        sparse_v3 = simulate_v3_scaling(sparse_features)
        sparse_v2_1 = simulate_v2_1_scaling(sparse_features)
        dense_v3 = simulate_v3_scaling(dense_features)
        dense_v2_1 = simulate_v2_1_scaling(dense_features)
        
        print(f"\nSparse Data Handling:")
        print(f"  V3.0 sparse scaling: {sparse_v3:.3f}")
        print(f"  V2.1 sparse scaling: {sparse_v2_1:.3f}")
        print(f"  Sparse improvement: {((sparse_v2_1 - sparse_v3) / sparse_v3 * 100):+.1f}%")
        
        # Weight distribution analysis
        print(f"\n📊 WEIGHT DISTRIBUTION ANALYSIS:")
        print(f"V3.0 Distribution:")
        print(f"  Conservative (entropy+complexity): {(15+10)/44*100:.1f}%")
        print(f"  Structural (zcr+l1+temporal): {(2+1+3)/44*100:.1f}%")
        print(f"V2.1 Distribution:")
        print(f"  Conservative (entropy+complexity): {(8+6)/42*100:.1f}%")
        print(f"  Structural (zcr+l1+temporal): {(6+5+4)/42*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Comparison test failed: {e}")
        return False

def main():
    """Main V2.1 validation comparison"""
    print("PROJECT RESIDUE V2.1 - VALIDATION COMPARISON")
    print("=" * 60)
    print("Comparing V3.0 vs V2.1 weight configurations")
    print("=" * 60)
    
    success = test_weight_impact()
    
    print(f"\n🏁 V2.1 Validation Complete")
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
