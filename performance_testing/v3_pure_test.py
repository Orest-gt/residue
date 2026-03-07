import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path so residue_v3 can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import residue.residue_v3 as residue_v3
    V3_AVAILABLE = True
except ImportError as e:
    V3_AVAILABLE = False
    print(f"❌ PROJECT RESIDUE V3.0 not available: {e}")
    print("Please run `python setup_v3.py build_ext --inplace` first.")
    sys.exit(1)

def main():
    print("=== PROJECT RESIDUE V3.0 PURE PERFORMANCE TESTING ===")
    print("=" * 60)
    
    controller = residue_v3.create_entropy_controller_v3()
    print("✅ V3.0 Controller Initialized")

    # Generate test data (simulated audio/sensor block)
    test_data = np.random.randn(1024).astype(np.float32)
    
    print("\n[ WARMUP ]")
    for _ in range(100):
        controller.infer_single_sample_fast(test_data)
        
    print("\n[ BENCHMARK: 10,000 Iterations ]")
    iterations = 10000
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        controller.infer_single_sample_fast(test_data)
        
    end_time = time.perf_counter()
    avg_time_us = ((end_time - start_time) / iterations) * 1e6
    
    meets_target = avg_time_us < 80.0
    
    print(f"Average pure inference time: {avg_time_us:.2f} μs")
    print(f"Target Sub-80μs: {'✅ PASS' if meets_target else '❌ FAIL'}")

    print("\n[ SIMD VERIFICATION ]")
    print("Sending extreme data to trigger clip(-88, 88) SIMD protections...")
    extreme_data = np.full(1024, 1e10, dtype=np.float32)
    try:
        scale = controller.infer_single_sample_fast(extreme_data)
        print(f"✅ Extreme Data Handled. Scale Output: {scale:.4f}")
    except Exception as e:
        print(f"❌ Failed on Extreme Data: {e}")

if __name__ == "__main__":
    main()
