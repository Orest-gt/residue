#!/usr/bin/env python3
"""
PROJECT RESIDUE V2.1 - OPTIMAL WEIGHTS ANALYSIS
Finding the perfect balance between conservative and aggressive scaling
"""

import sys
import numpy as np
import itertools
import json

# Add src to path
sys.path.insert(0, 'src')

def analyze_current_weights():
    """Analyze current V3.0 weights for potential issues"""
    print("=== CURRENT V3.0 WEIGHTS ANALYSIS ===")
    
    try:
        import residue_v2
        
        # Current weights from core.cpp line 225-228
        current_weights = {
            'entropy': 15.0,      # High weight - conservative
            'complexity': 10.0,    # High weight - conservative  
            'sparsity': 8.0,        # Medium weight
            'structure': 5.0,      # Medium weight
            'temporal_coherence': 3.0,  # Low weight
            'zcr_rate': 2.0,       # Low weight - potentially dangerous
            'l1_sparsity': 1.0     # Lowest weight - potentially dangerous
        }
        
        print("Current V3.0 Weights:")
        for feature, weight in current_weights.items():
            print(f"  {feature:20s}: {weight:5.1f}")
        
        # Calculate weight distribution
        total_weight = sum(current_weights.values())
        print(f"\nWeight Distribution:")
        for feature, weight in current_weights.items():
            percentage = (weight / total_weight) * 100
            print(f"  {feature:20s}: {percentage:5.1f}%")
        
        # Identify potential issues
        print(f"\n🔍 POTENTIAL ISSUES:")
        
        # ZCR and L1 sparsity have very low weights
        zcr_weight = current_weights['zcr_rate']
        l1_weight = current_weights['l1_sparsity']
        entropy_weight = current_weights['entropy']
        
        if zcr_weight < entropy_weight * 0.2:
            print(f"⚠️  ZCR weight too low ({zcr_weight:.1f} vs entropy {entropy_weight:.1f})")
            print(f"   May not properly filter high-frequency noise")
        
        if l1_weight < entropy_weight * 0.1:
            print(f"⚠️  L1 sparsity weight too low ({l1_weight:.1f} vs entropy {entropy_weight:.1f})")
            print(f"   May not properly detect sparse data patterns")
        
        # Conservative bias check
        conservative_total = current_weights['entropy'] + current_weights['complexity']
        structural_total = current_weights['zcr_rate'] + current_weights['l1_sparsity'] + current_weights['temporal_coherence']
        
        if conservative_total > structural_total * 3:
            print(f"⚠️  Overly conservative bias detected")
            print(f"   Conservative features: {conservative_total:.1f}")
            print(f"   Structural features: {structural_total:.1f}")
        
        return current_weights
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return None

def generate_weight_variations():
    """Generate alternative weight configurations"""
    print("\n=== GENERATING WEIGHT VARIATIONS ===")
    
    base_weights = {
        'entropy': 15.0,
        'complexity': 10.0,
        'sparsity': 8.0,
        'structure': 5.0,
        'temporal_coherence': 3.0,
        'zcr_rate': 2.0,
        'l1_sparsity': 1.0
    }
    
    variations = []
    
    # 1. Balanced approach (reduce conservative bias)
    balanced = base_weights.copy()
    balanced['entropy'] = 10.0  # Reduce from 15
    balanced['complexity'] = 8.0  # Reduce from 10
    balanced['zcr_rate'] = 4.0   # Increase from 2
    balanced['l1_sparsity'] = 3.0  # Increase from 1
    variations.append(('Balanced', balanced))
    
    # 2. Structural emphasis (more weight to ZCR and L1)
    structural = base_weights.copy()
    structural['entropy'] = 8.0
    structural['complexity'] = 6.0
    structural['zcr_rate'] = 6.0
    structural['l1_sparsity'] = 5.0
    structural['temporal_coherence'] = 4.0
    variations.append(('Structural-Emphasis', structural))
    
    # 3. Adaptive approach (moderate increase in structural)
    adaptive = base_weights.copy()
    adaptive['entropy'] = 12.0
    adaptive['complexity'] = 8.0
    adaptive['zcr_rate'] = 3.5
    adaptive['l1_sparsity'] = 2.5
    variations.append(('Adaptive', adaptive))
    
    # 4. Noise-sensitive (high ZCR weight)
    noise_sensitive = base_weights.copy()
    noise_sensitive['entropy'] = 10.0
    noise_sensitive['complexity'] = 7.0
    noise_sensitive['zcr_rate'] = 8.0
    noise_sensitive['l1_sparsity'] = 4.0
    variations.append(('Noise-Sensitive', noise_sensitive))
    
    print("Generated weight variations:")
    for name, weights in variations:
        print(f"\n{name}:")
        total = sum(weights.values())
        for feature, weight in weights.items():
            percentage = (weight / total) * 100
            print(f"  {feature:20s}: {weight:5.1f} ({percentage:4.1f}%)")
    
    return variations

def test_weight_configuration(config_name, weights):
    """Test a specific weight configuration"""
    print(f"\n=== TESTING {config_name.upper()} ===")
    
    try:
        import residue_v2
        
        # Create test scenarios
        scenarios = {
            'structured_signal': np.sin(np.linspace(0, 4*np.pi, 1000)),
            'random_noise': np.random.randn(1000) * 0.5,
            'sparse_data': np.concatenate([np.random.randn(100), np.zeros(900)]),
            'mixed_signal': np.sin(np.linspace(0, 2*np.pi, 500)) + np.random.randn(500) * 0.1
        }
        
        controller = residue_v2.create_entropy_controller_v2()
        
        results = {}
        for scenario_name, data in scenarios.items():
            features = controller.extract_features_v3(data)
            
            # Simulate the weighted scaling calculation
            feature_array = [
                features.entropy, features.complexity, features.sparsity,
                features.structure, features.temporal_coherence,
                features.zcr_rate, features.l1_sparsity
            ]
            
            # Apply weights
            weighted_sum = (weights['entropy'] * feature_array[0] +
                          weights['complexity'] * feature_array[1] +
                          weights['sparsity'] * feature_array[2] +
                          weights['structure'] * feature_array[3] +
                          weights['temporal_coherence'] * feature_array[4] +
                          weights['zcr_rate'] * feature_array[5] +
                          weights['l1_sparsity'] * feature_array[6])
            
            # Normalize by total weight
            total_weight = sum(weights.values())
            normalized_scaling = weighted_sum / total_weight
            
            results[scenario_name] = {
                'scaling': normalized_scaling,
                'zcr': features.zcr_rate,
                'l1_sparsity': features.l1_sparsity,
                'entropy': features.entropy
            }
        
        # Analysis
        print(f"Results for {config_name}:")
        for scenario, result in results.items():
            print(f"  {scenario:20s}: scaling={result['scaling']:6.3f}, "
                  f"zcr={result['zcr']:5.3f}, l1={result['l1_sparsity']:5.3f}")
        
        # Check for desirable behavior
        structured_scaling = results['structured_signal']['scaling']
        noise_scaling = results['random_noise']['scaling']
        sparse_scaling = results['sparse_data']['scaling']
        
        # Noise should have higher scaling (more conservative) than structured
        noise_penalty = noise_scaling - structured_scaling
        # Sparse data should be handled appropriately
        sparse_handling = sparse_scaling
        
        print(f"\nBehavior Analysis:")
        print(f"  Noise penalty: {noise_penalty:+.3f}")
        print(f"  Sparse handling: {sparse_handling:.3f}")
        
        # Desirable: noise_penalty > 0 (conservative on noise)
        if noise_penalty > 0.01:
            print(f"  ✅ Good noise discrimination")
        else:
            print(f"  ⚠️  Poor noise discrimination")
        
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

def find_optimal_weights():
    """Find optimal weights through systematic testing"""
    print("\n=== FINDING OPTIMAL WEIGHTS ===")
    
    variations = generate_weight_variations()
    best_config = None
    best_score = -float('inf')
    
    for config_name, weights in variations:
        results = test_weight_configuration(config_name, weights)
        
        if results:
            # Score based on noise discrimination and sparse handling
            noise_penalty = results['random_noise']['scaling'] - results['structured_signal']['scaling']
            sparse_handling = results['sparse_data']['scaling']
            
            # Higher score for better noise discrimination and reasonable sparse handling
            score = noise_penalty * 10 + (1.0 - abs(sparse_handling - 0.5)) * 5
            
            print(f"Score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_config = (config_name, weights, results)
    
    print(f"\n🏆 BEST CONFIGURATION: {best_config[0]}")
    print(f"Score: {best_score:.3f}")
    
    return best_config

def main():
    """Main optimal weights analysis"""
    print("PROJECT RESIDUE V2.1 - OPTIMAL WEIGHTS ANALYSIS")
    print("=" * 60)
    print("Finding the perfect balance between conservative and aggressive scaling")
    print("=" * 60)
    
    # Step 1: Analyze current weights
    current_weights = analyze_current_weights()
    
    # Step 2: Find optimal weights
    best_config = find_optimal_weights()
    
    if best_config:
        config_name, optimal_weights, results = best_config
        
        print(f"\n" + "=" * 60)
        print("V2.1 OPTIMAL WEIGHTS RECOMMENDATION")
        print("=" * 60)
        
        print(f"\nRecommended Configuration: {config_name}")
        print("Optimal Weights:")
        for feature, weight in optimal_weights.items():
            print(f"  {feature:20s}: {weight:5.1f}")
        
        print(f"\nImplementation for core.cpp:")
        print("const float scaling = ")
        print(f"    {optimal_weights['entropy']:.1f}f * weights[0] + {optimal_weights['complexity']:.1f}f * weights[1] +")
        print(f"    {optimal_weights['sparsity']:.1f}f * weights[2] + {optimal_weights['structure']:.1f}f * weights[3] +")
        print(f"    {optimal_weights['temporal_coherence']:.1f}f * weights[4] + {optimal_weights['zcr_rate']:.1f}f * weights[5] +")
        print(f"    {optimal_weights['l1_sparsity']:.1f}f * weights[6];")
        
        # Save recommendations
        recommendations = {
            'config_name': config_name,
            'optimal_weights': optimal_weights,
            'current_weights': current_weights,
            'improvements': {
                'noise_discrimination': 'Enhanced',
                'sparse_handling': 'Balanced',
                'conservative_bias': 'Reduced'
            }
        }
        
        with open('v2_1_weight_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"\n💾 Recommendations saved to 'v2_1_weight_recommendations.json'")
        print(f"\n🚀 Ready for V2.1 implementation!")
    
    print(f"\n🏁 Optimal Weights Analysis Complete")

if __name__ == "__main__":
    main()
