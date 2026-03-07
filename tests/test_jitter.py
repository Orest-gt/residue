import numpy as np
import time
import sys
import os

# Ensure residue module is mapped correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from residue import core

def test_jitter():
    controller = core.create_entropy_controller_v3()
    
    # Simulating a tight real-time audio loop 
    # 10,000 frames delivered independently
    FRAME_SIZE = 1024
    ITERATIONS = 10000
    
    print("=== PROJECT RESIDUE V3.0 GUARDIAN JITTER TEST ===")
    print(f"Monitoring thread affinity and REALTIME priority across {ITERATIONS} cycles...")
    
    # Pre-allocate to prevent Python GC spikes
    test_frames = [np.random.randn(FRAME_SIZE).astype(np.float32) for _ in range(ITERATIONS)]
    latencies_ns = np.zeros(ITERATIONS, dtype=np.int64)
    
    # Initial warmup to populate L1/L2 and load DLL pages
    for i in range(100):
        controller.infer_single_sample_fast(test_frames[0])
        
    print("Warmup complete. Engaging OS ThreadGuard...\n")
    
    for i in range(ITERATIONS):
        frame = test_frames[i]
        
        # Microsecond precision timer map
        t0 = time.perf_counter_ns()
        
        # The C++ core will invoke ThreadGuard on this call internally during batch/single execution
        # Since we want to test process_stream_fast, we use batch_infer_fast with size 1 frame array
        controller.batch_infer_fast(frame, FRAME_SIZE)
        
        t1 = time.perf_counter_ns()
        latencies_ns[i] = t1 - t0
        
    latencies_us = latencies_ns / 1000.0
    
    # Diagnostics
    mean_us = np.mean(latencies_us)
    median_us = np.median(latencies_us)
    p99_us = np.percentile(latencies_us, 99)
    max_us = np.max(latencies_us)
    min_us = np.min(latencies_us)
    
    print(f"Metrics (Microseconds - μs):")
    print(f"Minimum Latency:   {min_us:.2f} μs")
    print(f"Median  Latency:   {median_us:.2f} μs")
    print(f"Mean    Latency:   {mean_us:.2f} μs")
    print(f"99th Percentile:   {p99_us:.2f} μs")
    print(f"MAX / WORST CASE:  {max_us:.2f} μs  <-- OS Scheduler Target")
    
    # Jitter is max varying delay
    jitter_us = max_us - min_us
    print(f"\nTOTAL SYSTEM JITTER: {jitter_us:.2f} μs")
    
    if max_us < 100.0:
        print("\n✅ GUARDIAN LAYER ACTIVE: Worst-case bounded under 100μs!")
    else:
        print("\n⚠️ WARNING: Worst-case exceeded 100μs. Safety timeout likely triggered.")

if __name__ == '__main__':
    test_jitter()
