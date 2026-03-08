import torch
import torch.nn as nn
import numpy as np
import time
import os
import csv
from residue.core import AsyncObserver, print_isolation_report
from residue.pytorch_bridge import PyTorchShield

def benchmark_throughput_sparsity():
    """Measures raw throughput (FPS) across 0% to 99.9% sparsity."""
    print("\n--- [BENCHMARK 1] THROUGHPUT VS SPARSITY ---")
    FRAME_SIZE = 1024
    NUM_FRAMES = 5000
    SPARSITY_LEVELS = [0.0, 0.5, 0.9, 0.95, 0.99, 0.999]
    
    results = []
    observer = AsyncObserver(frame_size=FRAME_SIZE)
    observer.start()
    
    for sparsity in SPARSITY_LEVELS:
        print(f"Testing Sparsity: {sparsity*100:.1f}%...")
        
        # Generate data with controlled sparsity (Max value determines skip)
        # Residue uses Max-Abs scan.
        data = np.zeros((NUM_FRAMES, FRAME_SIZE), dtype=np.float32)
        
        num_dense = int(NUM_FRAMES * (1.0 - sparsity))
        if num_dense > 0:
            indices = np.random.choice(NUM_FRAMES, num_dense, replace=False)
            data[indices] = 1.0 # High value to trigger "SIGNAL"
        
        # Flatten for ingestion
        flat_data = data.flatten()
        
        # Benchmarking
        start = time.perf_counter()
        observer.push_data(flat_data)
        
        # Wait for completion
        while True:
            tel = observer.poll_telemetry()
            if tel.total_samples_processed >= NUM_FRAMES:
                break
            time.sleep(0.01)
        
        end = time.perf_counter()
        duration = end - start
        fps = NUM_FRAMES / duration
        
        results.append({"sparsity": sparsity, "fps": fps, "duration": duration})
        print(f" -> FPS: {fps:,.0f}")
        
        # Reset telemetry for next run
        # (We recreate the observer for a clean state in each loop for accuracy)
        observer.stop()
        observer = AsyncObserver(frame_size=FRAME_SIZE)
        observer.start()

    observer.stop()
    return results

def benchmark_pytorch_scaling():
    """Measures the speedup gain as the 'Heavy' model size increases."""
    print("\n--- [BENCHMARK 2] PYTORCH SHIELD SCALING ---")
    FRAME_SIZE = 1024
    NUM_FRAMES = 500
    MODEL_SIZES = [512, 1024, 2048, 4096] # Inner dimension
    SPARSITY = 0.8
    
    results = []
    
    for size in MODEL_SIZES:
        print(f"Testing Model Size (Hidden={size})...")
        
        # Dummy Model
        model = nn.Sequential(
            nn.Linear(FRAME_SIZE, size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, FRAME_SIZE)
        )
        
        # Data
        data = (np.random.randn(NUM_FRAMES, FRAME_SIZE) * 1e-6).astype(np.float32)
        # Add 20% signals
        data[np.random.choice(NUM_FRAMES, int(NUM_FRAMES*0.2), replace=False)] = 1.0
        
        # 1. Baseline
        start = time.perf_counter()
        for f in data:
            _ = model(torch.from_numpy(f))
        baseline_duration = time.perf_counter() - start
        
        # 2. Shielded
        shield = PyTorchShield(model, frame_size=FRAME_SIZE)
        shield.start()
        start = time.perf_counter()
        shield.observer.push_data(data.flatten())
        while True:
            tel = shield.get_stats()
            if tel.total_samples_processed >= NUM_FRAMES:
                break
            time.sleep(0.01)
        
        # To simulate real-time speedup in a loop:
        # We calculate (Dense frames * Baseline time per frame) + Residue overhead
        skips = tel.total_samples_skipped
        dense = NUM_FRAMES - skips
        time_per_frame = baseline_duration / NUM_FRAMES
        theoretical_time = (dense * time_per_frame) + (time.perf_counter() - start)
        
        shield.stop()
        
        speedup = baseline_duration / theoretical_time
        results.append({"model_size": size, "speedup": speedup})
        print(f" -> Speedup: {speedup:.2f}x")
        
    return results

def save_to_csv(data, filename):
    if not data: return
    keys = data[0].keys()
    os.makedirs("performance_testing/results", exist_ok=True)
    with open(f"performance_testing/results/{filename}", 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)

if __name__ == "__main__":
    print_isolation_report()
    
    sparsity_data = benchmark_throughput_sparsity()
    save_to_csv(sparsity_data, "sparsity_benchmark.csv")
    
    scaling_data = benchmark_pytorch_scaling()
    save_to_csv(scaling_data, "scaling_benchmark.csv")
    
    print("\n✅ Benchmarks complete. Results saved to performance_testing/results/")
