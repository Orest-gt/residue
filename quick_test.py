import sys
sys.path.insert(0, 'src')

import residue_v2
import numpy as np

print("=== QUICK V3.0 INTEGRITY TEST ===")

# Test 1: 7-Feature Extraction
print("\n1. Testing 7-Feature Extraction...")
c = residue_v2.create_entropy_controller_v2()
data = np.random.randn(100)
features = c.extract_features_v3(data)
print(f"Features: {features}")
print(f"Feature count: {len([x for x in dir(features) if not x.startswith('_')])}")

# Test 2: EMA
print("\n2. Testing EMA...")
c.set_ema_alpha(0.3)
print(f"EMA alpha set: {c.get_ema_alpha()}")

# Test 3: ZCR
print("\n3. Testing ZCR...")
sine = np.sin(np.linspace(0, 10*np.pi, 1000))
noise = np.random.randn(1000)
zcr_sine = c.calculate_zero_crossing_rate(sine)
zcr_noise = c.calculate_zero_crossing_rate(noise)
print(f"Sine ZCR: {zcr_sine:.6f}")
print(f"Noise ZCR: {zcr_noise:.6f}")

# Test 4: L1 Sparsity
print("\n4. Testing L1 Sparsity...")
sparse = np.zeros(1000)
sparse[:100] = np.random.randn(100)
l1_sparse = c.calculate_l1_norm_sparsity(sparse)
print(f"90% sparse L1: {l1_sparse:.6f}")

print("\n=== QUICK TEST COMPLETE ===")
