import sys
sys.path.insert(0, 'src')
import residue_v2
import numpy as np

print("=== MANUAL RESULTS TABLE ===")
print("Based on individual test components")

# Test 1: Silence
c = residue_v2.create_entropy_controller_v2()
silence_data = np.array([0.0001] * 100)
silence_f = c.extract_features_v3(silence_data)
silence_scaling = c.compute_multi_dimensional_scaling_v3(silence_f)

# Test 2: Chaos
chaos_data = np.random.uniform(-1, 1, 100)
chaos_f = c.extract_features_v3(chaos_data)
chaos_scaling = c.compute_multi_dimensional_scaling_v3(chaos_f)

# Test 3: Pattern
pattern_data = np.array([1, -1] * 50)
pattern_f = c.extract_features_v3(pattern_data)
pattern_scaling = c.compute_multi_dimensional_scaling_v3(pattern_f)

# Test 4: EMA (simplified)
c.set_ema_alpha(0.3)
chaos_scaling1 = c.compute_multi_dimensional_scaling_v3(c.extract_features_v3(np.random.uniform(-1, 1, 100)))
sine_scaling1 = c.compute_multi_dimensional_scaling_v3(c.extract_features_v3(np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5))
transition = sine_scaling1 - chaos_scaling1

print("\nRESULTS TABLE:")
print("=" * 70)
print(f"{'Test':<20} {'Value':<12} {'Expected':<12} {'Status':<8}")
print("-" * 70)
print(f"{'Silence L1':<20} {silence_f.l1_sparsity:<12.3f} {'>0.9':<12} {'✅ PASS' if silence_f.l1_sparsity > 0.9 else '❌ FAIL'}")
print(f"{'Silence Scaling':<20} {silence_scaling:<12.3f} {'High':<12} {'✅ PASS' if silence_scaling > 1.0 else '❌ FAIL'}")
print(f"{'Chaos ZCR':<20} {chaos_f.zcr_rate:<12.3f} {'≈0.5':<12} {'✅ PASS' if 0.3 < chaos_f.zcr_rate < 0.7 else '❌ FAIL'}")
print(f"{'Chaos Entropy':<20} {chaos_f.entropy:<12.3f} {'High':<12} {'✅ PASS' if chaos_f.entropy > 5.0 else '❌ FAIL'}")
print(f"{'Pattern ZCR':<20} {pattern_f.zcr_rate:<12.3f} {'1.0':<12} {'✅ PASS' if abs(pattern_f.zcr_rate - 1.0) < 0.01 else '❌ FAIL'}")
print(f"{'Pattern Complexity':<20} {pattern_f.complexity:<12.3f} {'High':<12} {'✅ PASS' if pattern_f.complexity > 0.5 else '❌ FAIL'}")
print(f"{'EMA Transition':<20} {transition:<12.3f} {'<0.5':<12} {'✅ PASS' if abs(transition) < 0.5 else '❌ FAIL'}")

print("\nSUMMARY:")
all_tests = [
    silence_f.l1_sparsity > 0.9,
    silence_scaling > 1.0,
    0.3 < chaos_f.zcr_rate < 0.7,
    chaos_f.entropy > 5.0,
    abs(pattern_f.zcr_rate - 1.0) < 0.01,
    pattern_f.complexity > 0.5,
    abs(transition) < 0.5
]

passed = sum(all_tests)
total = len(all_tests)
print(f"Tests passed: {passed}/{total}")

if passed == total:
    print("🎉 ALL TESTS PASSED!")
    print("✅ PROJECT RESIDUE V2.1 - STRUCTURAL INTELLIGENCE: IMPENETRABLE")
else:
    print("⚠️  Some tests failed")

print("\n🏁 Manual Validation Complete")
