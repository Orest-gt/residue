import sys
sys.path.insert(0, 'src')
import residue_v2
import numpy as np

print("=== QUICK EMA TEST ===")
c = residue_v2.create_entropy_controller_v2()
c.set_ema_alpha(0.3)

# Test sequence
chaos = np.random.uniform(-1, 1, 100)
sine = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5

print("Chaos scaling:", c.compute_multi_dimensional_scaling_v3(c.extract_features_v3(chaos)))
print("Sine scaling:", c.compute_multi_dimensional_scaling_v3(c.extract_features_v3(sine)))

# EMA progression
print("\nEMA progression:")
for i in range(6):
    vec = chaos if i < 3 else sine
    scaling = c.compute_multi_dimensional_scaling_v3(c.extract_features_v3(vec))
    print(f"Step {i+1}: {scaling:.3f}")
