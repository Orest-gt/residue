import time
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from residue.core import EntropyControllerV3, get_cache_topology

def print_topology():
    print("========================================")
    print("RESIDUE WALL: CACHE TOPOLOGY DETECTED")
    print("========================================")
    topo = get_cache_topology()
    for k, v in topo.items():
        if "size" in k:
            print(f"  {k:20}: {v // 1024} KB")
        else:
            print(f"  {k:20}: {v}")
    print("========================================\n")

def test_aligner_and_prefetch():
    print("Initializing Walled Engine...")
    engine = EntropyControllerV3()
    
    # 1. Test fast path (already aligned numpy buffer)
    # Using posix_memalign / mmap equivalent in numpy is tricky in portable python,
    # but we can allocate a large array and slice it to find a 64-byte aligned boundary.
    raw_buffer = np.zeros(1024 * 1024, dtype=np.float32)
    # ptr address is raw_buffer.__array_interface__['data'][0]
    
    # Find aligned sub-array
    ptr = raw_buffer.__array_interface__['data'][0]
    offset = (64 - (ptr % 64)) % 64
    floats_offset = int(offset // 4)
    
    aligned_buffer = raw_buffer[floats_offset:floats_offset + 10240]
    aligned_ptr = aligned_buffer.__array_interface__['data'][0]
    assert aligned_ptr % 64 == 0, f"Failed to manually align buffer: {aligned_ptr}"
    
    print(f"Testing Zero-Copy Path with 64-byte aligned buffer (ptr: {hex(aligned_ptr)})...")
    
    # Run the walled inference
    start = time.perf_counter()
    out1 = engine.batch_infer_walled(aligned_buffer, 1024)
    end = time.perf_counter()
    print(f"  -> Walled inference completed in {(end - start)*1000:.3f} ms")
    
    # 2. Test slow path (misaligned buffer)
    misaligned_buffer = raw_buffer[floats_offset + 1 : floats_offset + 1 + 10240]
    misaligned_ptr = misaligned_buffer.__array_interface__['data'][0]
    assert misaligned_ptr % 64 != 0, "Buffer is accidentally aligned"
    
    print(f"Testing Non-Temporal Copy Path with misaligned buffer (ptr: {hex(misaligned_ptr)})...")
    
    # RESET HISTORY BEFORE SECOND PASS
    engine.reset_history()
    
    start = time.perf_counter()
    out2 = engine.batch_infer_walled(misaligned_buffer, 1024)
    end = time.perf_counter()
    print(f"  -> Walled inference (with copy) completed in {(end - start)*1000:.3f} ms")
    
    # Verify outputs match
    # Since the input is all zeros, output scaling should be identical
    np.testing.assert_array_almost_equal(out1, out2)
    print("\n[SUCCESS] Residue Wall tests passed!")

if __name__ == "__main__":
    print_topology()
    test_aligner_and_prefetch()
