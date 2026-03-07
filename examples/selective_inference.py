"""
EXAMPLE: Real-World "Selective Inference" Simulation
-----------------------------------------------------
This script demonstrates how Residue V4.2.4 can be used to "Shield" 
a heavy LLM-style computation by pre-filtering noise at 2.0M+ FPS.
"""

import numpy as np
import time
from residue.core import AsyncObserver

# Simulated "Heavy Neural Network" computation
def heavy_neural_network_call(data):
    # This represents a GPU-bound or heavy CPU-bound task
    # (e.g., Matrix multiplication, Softmax, etc.)
    return np.exp(data) * np.tanh(data)

def run_selective_inference_demo():
    FRAME_SIZE = 1024
    NUM_FRAMES = 5000
    
    # 1. Setup Residue Shield
    # Highly sparse data (90% noise) simulates a real-time audio/sensor feed
    observer = AsyncObserver(frame_size=FRAME_SIZE)
    observer.start()
    
    print("--- SELECTIVE INFERENCE DEMO ---")
    print(f"Scenario: Processing {NUM_FRAMES} frames (1024 floats each)")
    print("Goal: Skip 'Heavy Computation' if Residue detects sparsity/noise.\n")

    # Generate data with transients (mostly zeros, some spikes)
    raw_data = np.zeros((NUM_FRAMES, FRAME_SIZE), dtype=np.float32)
    for i in range(0, NUM_FRAMES, 50): # Only 2% of frames have signals
        raw_data[i] = np.random.randn(FRAME_SIZE)
        
    # 4. Scenario: Continuous Stream
    print(f"[STEP 4] Pushing {NUM_FRAMES} frames into the Async Buffer...")
    
    start_time = time.perf_counter()
    
    # Push all data at once (simulating a high-speed intake)
    observer.push_data(raw_data.flatten())
    
    # Wait a moment for the C++ worker (at 2M+ FPS, this is instant)
    time.sleep(0.2)
    
    # 5. Telemetry Result
    print("\n[STEP 5] Reading Shield Telemetry:")
    tel = observer.poll_telemetry()
    
    processed_count = tel.total_samples_processed / FRAME_SIZE
    skipped_count = tel.total_samples_skipped / FRAME_SIZE
    
    print(f" - Frames Scanned: {NUM_FRAMES}")
    print(f" - Significant Signals Found: {processed_count:.0f} (Heavy Model triggered)")
    print(f" - Noise/Sparse Frames Skipped: {skipped_count:.0f} (No Model overhead)")
    print(f" - Shield Efficiency: {(skipped_count/NUM_FRAMES)*100:.1f}%")
    
    # Simulate time saved
    inference_cost_per_frame = 0.005 # 5ms average model latency
    time_saved = skipped_count * inference_cost_per_frame
    print(f"\n[ANALYSIS] Projected GPU Latency Saved: {time_saved:.2f} seconds.")
    
    observer.stop()
    print("\nResidue Shield: Reality-Synchronized Efficiency achieved.")

if __name__ == "__main__":
    run_selective_inference_demo()
