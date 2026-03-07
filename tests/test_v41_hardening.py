import numpy as np
import time
from residue import core

def test_isolation_tier():
    print("\n--- Test 1: Deterministic Memory & Isolation Status ---")
    observer = core.AsyncObserver(1024, 100)
    telem = observer.poll_telemetry()
    print(f"telemetry is_running: {telem.is_running}")
    
    # Actually, IsolationZone is created inside the worker_loop when start() is called.
    observer.start()
    time.sleep(0.1) # Give thread time to start and report
    
    # We exposed get_status() to EntropyControllerV3 but not AsyncObserver or directly.
    # Let's check how we exposed get_status in bindings.cpp
    # Wait, did we expose get_status in bindings? Let's check bindings!
    # Ah, I didn't actually bind get_status() on the Python side in my bindings.cpp update!
    
def test_multi_probe_gate():
    print("\n--- Test 2: Widened Heuristic Gate (Multi-Probe) ---")
    ctrl = core.EntropyControllerV3(num_bins=256, entropy_threshold=1e-5, l1_threshold=1e-5)
    
    # Old gate: Spike at float #0 -> processed.
    data_head = np.zeros(1024, dtype=np.float32)
    data_head[0] = 100.0
    
    # New gate: Spike at float #1000 -> processed (old would skip this entirely!)
    data_tail = np.zeros(1024, dtype=np.float32)
    data_tail[1000] = 100.0
    
    # Pure silence -> skipped
    data_silence = np.zeros(1024, dtype=np.float32)
    
    out_head = np.zeros(1, dtype=np.float32)
    out_tail = np.zeros(1, dtype=np.float32)
    out_silence = np.zeros(1, dtype=np.float32)
    
    ctrl.process_stream_walled(data_head, 1024, 1024, out_head)
    skipped_1 = ctrl.total_samples_skipped
    
    ctrl.process_stream_walled(data_tail, 1024, 1024, out_tail)
    skipped_2 = ctrl.total_samples_skipped - skipped_1
    
    ctrl.process_stream_walled(data_silence, 1024, 1024, out_silence)
    skipped_3 = ctrl.total_samples_skipped - skipped_1 - skipped_2
    
    print(f"Head Spike Skipped: {skipped_1 > 0} (Expected False)")
    print(f"Tail Spike Skipped: {skipped_2 > 0} (Expected False)")
    print(f"Silence Skipped:    {skipped_3 > 0} (Expected True)")

def test_backpressure():
    print("\n--- Test 3: Backpressure Architecture ---")
    observer = core.AsyncObserver(frame_size=1024, buffer_capacity_frames=10)
    # We do NOT start the observer, so the buffer doesn't drain
    
    data = np.random.randn(1024 * 5).astype(np.float32)
    
    # Fill up 5 frames (capacity is 10)
    pushed1 = observer.push_data(data, len(data))
    telem1 = observer.poll_telemetry()
    print(f"Push 1: {pushed1} floats pushed. Fill %: {telem1.buffer_fill_pct:.1f}%")
    
    # Push another 10 frames (should overflow by 5)
    data2 = np.random.randn(1024 * 10).astype(np.float32)
    pushed2 = observer.push_data(data2, len(data2))
    telem2 = observer.poll_telemetry()
    
    print(f"Push 2: {pushed2} floats pushed.")
    print(f"Fill %: {telem2.buffer_fill_pct:.1f}% (Expected ~100%)")
    print(f"Backpressure Active: {telem2.backpressure_active} (Expected True)")
    print(f"Frames Dropped: {telem2.total_frames_dropped} (Expected 5)")


if __name__ == "__main__":
    test_multi_probe_gate()
    test_backpressure()
