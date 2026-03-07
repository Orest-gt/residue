import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import residue.residue_v3 as residue_v3
    V3_AVAILABLE = True
except ImportError as e:
    V3_AVAILABLE = False
    print(f"❌ PROJECT RESIDUE V3.0 not available: {e}")
    sys.exit(1)

def main():
    print("=== PROJECT RESIDUE V3.0 BATCH VS SINGLE PERFORMANCE ===")
    print("=" * 60)
    
    controller = residue_v3.create_entropy_controller_v3()
    
    frame_size = 1024
    num_frames = 100000
    
    # Generate flat stream data
    stream_data = np.random.randn(num_frames * frame_size).astype(np.float32)
    
    print(f"Processing {num_frames} frames of size {frame_size}...")

    # 1. Single Call Benchmark (Context Switching Torture)
    print("\n[ SINGLE-CALL LOOP ]")
    controller.reset_history() # RESET STATE TO ENSURE CONSISTENCY
    start_single = time.perf_counter()
    single_results = []
    for i in range(num_frames):
        frame = stream_data[i*frame_size : (i+1)*frame_size]
        single_results.append(controller.infer_single_sample_fast(frame))
    end_single = time.perf_counter()
    single_duration = end_single - start_single
    
    # 2. Batch Call Benchmark (The Anti-Overhead Solution)
    print("[ BATCH-INFER FAST ]")
    controller.reset_history() # RESET STATE TO ENSURE CONSISTENCY
    start_batch = time.perf_counter()
    batch_results = controller.batch_infer_fast(stream_data, frame_size)
    end_batch = time.perf_counter()
    batch_duration = end_batch - start_batch

    # Verify results match
    diff = np.abs(np.array(single_results) - batch_results).max()
    print(f"\nResult Consistency Check: {'✅ OK' if diff < 1e-4 else '❌ MISMATCH'}")
    
    print(f"\nTotal Time (Single): {single_duration*1000:.2f} ms")
    print(f"Total Time (Batch):  {batch_duration*1000:.2f} ms")
    
    speedup = single_duration / batch_duration
    overhead_saved = (1 - (batch_duration / single_duration)) * 100
    
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"Overhead Eliminated: {overhead_saved:.1f}%")

    if speedup > 2.0:
        print("\n✅ CONTEXT SWITCHING MITIGATED: The Python tax has been paid in full.")
    else:
        print("\n⚠️  Speedup subtle. Consider increasing num_frames for clearer profiling.")

if __name__ == "__main__":
    main()
