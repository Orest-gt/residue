import numpy as np
import sys
import os

# Ensure residue module is mapped correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from residue import core

def test_odd_sizes():
    controller = core.create_entropy_controller_v3()
    
    sizes = [1, 2, 3, 7, 8, 9, 15, 16, 17, 1024, 1027, 2055, 9999]
    print("=== PROJECT RESIDUE V3.0 BOUNDARY TAIL STRESS TEST ===")
    
    all_passed = True
    for sz in sizes:
        data = np.random.randn(sz).astype(np.float32)
        try:
            # Test single infer
            res = controller.infer_single_sample_fast(data)
            
            # Test batch infer with odd size (ensuring Tiled Processing handles remainders)
            batch_data = np.random.randn(sz * 3).astype(np.float32)
            res_batch = controller.batch_infer_fast(batch_data, sz)
            print(f"✅ Size {sz} passed without segfault.")
        except Exception as e:
            print(f"❌ Size {sz} FAILED: {e}")
            all_passed = False
            
    if all_passed:
        print("\n✅ ALL BOUNDARY TESTS PASSED. NO SEGFAULTS.")
    else:
        print("\n❌ BOUNDARY TESTS FAILED.")
        sys.exit(1)

if __name__ == '__main__':
    test_odd_sizes()
