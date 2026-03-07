"""V4.2 Verification: Full-Scan Gate Accuracy + NUMA + Batch Hint"""
import numpy as np
from residue import core

def test_fullscan_gate():
    print("\n--- V4.2 Test 1: Vectorized Full-Scan Gate (100% Accuracy) ---")
    ctrl = core.EntropyControllerV3(num_bins=256, entropy_threshold=1e-5, l1_threshold=1e-5)

    # Spike at position 500 (invisible to old 3-probe and 8-float gates)
    data_spike_500 = np.zeros(1024, dtype=np.float32)
    data_spike_500[500] = 100.0

    # Spike at position 900 (invisible to old 3-probe head/mid check)
    data_spike_900 = np.zeros(1024, dtype=np.float32)
    data_spike_900[900] = 50.0

    # Pure silence -> must be skipped
    data_silence = np.zeros(1024, dtype=np.float32)

    # process_stream_walled is exposed as batch_infer_walled in bindings, and returns the output array
    ctrl.batch_infer_walled(data_spike_500, frame_size=1024)
    skip1 = ctrl.get_total_samples_skipped()
    print(f"  Spike@500 Skipped: {skip1 > 0} (Expected: False) {'✅' if skip1 == 0 else '❌'}")

    ctrl.batch_infer_walled(data_spike_900, frame_size=1024)
    skip2 = ctrl.get_total_samples_skipped() - skip1
    print(f"  Spike@900 Skipped: {skip2 > 0} (Expected: False) {'✅' if skip2 == 0 else '❌'}")

    ctrl.batch_infer_walled(data_silence, frame_size=1024)
    skip3 = ctrl.get_total_samples_skipped() - skip1 - skip2
    print(f"  Silence   Skipped: {skip3 > 0} (Expected: True)  {'✅' if skip3 > 0 else '❌'}")

def test_numa_report():
    print("\n--- V4.2 Test 2: NUMA-Aware Isolation Report ---")
    core.print_isolation_report()

def test_batch_hint():
    print("\n--- V4.2 Test 3: Batch Size Hint ---")
    observer = core.AsyncObserver(frame_size=1024, buffer_capacity_frames=100)
    hint = observer.recommended_push_size()
    print(f"  Recommended push size: {hint} floats ({hint // 1024} frames)")
    print(f"  Expected: {1024 * 64} floats (64 frames) {'✅' if hint == 1024 * 64 else '❌'}")

if __name__ == "__main__":
    test_fullscan_gate()
    test_numa_report()
    test_batch_hint()
    print("\n=== ALL V4.2 VERIFICATIONS COMPLETE ===")
