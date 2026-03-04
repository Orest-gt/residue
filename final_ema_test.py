import sys
sys.path.insert(0, 'src')
import residue_v2
import numpy as np

print("=== FINAL EMA TEST ===")

try:
    c = residue_v2.create_entropy_controller_v2()
    c.set_ema_alpha(0.3)
    
    # Create test vectors
    chaos = np.random.uniform(-1, 1, 100)
    sine = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5
    
    print("Individual tests:")
    chaos_f = c.extract_features_v3(chaos)
    sine_f = c.extract_features_v3(sine)
    
    print(f"Chaos scaling: {c.compute_multi_dimensional_scaling_v3(chaos_f):.3f}")
    print(f"Sine scaling: {c.compute_multi_dimensional_scaling_v3(sine_f):.3f}")
    
    # EMA progression test
    print("\nEMA progression (3 chaos → 3 sine):")
    scalings = []
    
    for i in range(6):
        if i < 3:
            vec = np.random.uniform(-1, 1, 100)  # New chaos each time
            test_type = "Chaos"
        else:
            vec = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5  # Same sine
            test_type = "Sine"
        
        features = c.extract_features_v3(vec)
        scaling = c.compute_multi_dimensional_scaling_v3(features)
        scalings.append(scaling)
        
        print(f"Step {i+1} ({test_type}): {scaling:.3f}")
    
    # Analyze transition
    if len(scalings) >= 2:
        transition = scalings[3] - scalings[2]
        print(f"\nTransition analysis:")
        print(f"Chaos→Sine transition: {transition:+.3f}")
        print(f"Smooth (|transition| < 0.5): {'✅ PASS' if abs(transition) < 0.5 else '❌ FAIL'}")
    
    print("\n✅ EMA test completed")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
