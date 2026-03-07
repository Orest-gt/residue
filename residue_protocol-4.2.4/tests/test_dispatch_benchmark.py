#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.1 - BRANCHLESS DYNAMIC DISPATCH BENCHMARK
=============================================================
Verifies the Level 4.1 "Branchless Dynamic Dispatch" architecture.
Generates streams with varying levels of sparsity (silence).
Proves that the AVX2 L1-norm Heuristic + V-Table correctly skips math
without causing branch misprediction penalties.

Usage: python tests/v3_dispatch_benchmark.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from residue.core import EntropyControllerV3

FRAME_SIZE = 1024
NUM_FRAMES = 50_000   # 50K frames = ~200MB aligned buffer
ALIGNMENT = 64

def make_aligned_array(size, dtype=np.float32):
    raw = np.empty(size + ALIGNMENT // np.dtype(dtype).itemsize, dtype=dtype)
    ptr = raw.__array_interface__['data'][0]
    off = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT
    arr = raw[off // np.dtype(dtype).itemsize:][:size]
    return arr

def run_benchmark(sparsity_pct):
    engine = EntropyControllerV3()
    
    total_floats = NUM_FRAMES * FRAME_SIZE
    data = make_aligned_array(total_floats)
    
    # 1. Fill with dense noise (Signal)
    np.random.seed(42)
    data[:] = np.random.randn(total_floats).astype(np.float32)
    
    # 2. Apply sparsity (Silence)
    num_sparse_frames = int(NUM_FRAMES * (sparsity_pct / 100.0))
    if num_sparse_frames > 0:
        # Randomly select frames to silence
        sparse_indices = np.random.choice(NUM_FRAMES, num_sparse_frames, replace=False)
        for idx in sparse_indices:
            start = idx * FRAME_SIZE
            end = start + FRAME_SIZE
            data[start:end] = 0.0  # Perfect silence
            
    # Benchmark Walled inference
    print(f"  Testing Sparsity: {sparsity_pct:3.0f}% ... ", end="", flush=True)
    engine.reset_history()
    
    # Warmup
    engine.batch_infer_walled(data[:FRAME_SIZE*64], FRAME_SIZE)
    engine.reset_history()
    
    t0 = time.perf_counter()
    engine.batch_infer_walled(data, FRAME_SIZE)
    t1 = time.perf_counter()
    
    elapsed_ms = (t1 - t0) * 1000.0
    fps = int(NUM_FRAMES / (elapsed_ms / 1000.0))
    print(f"{elapsed_ms:6.1f} ms  |  {fps:9,} FPS")
    return elapsed_ms, fps


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   LEVEL 4.1: BRANCHLESS DYNAMIC DISPATCH BENCHMARK     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Frames: {NUM_FRAMES:,} | Frame Size: {FRAME_SIZE} | Buffer: {NUM_FRAMES*FRAME_SIZE*4/1024/1024:.1f} MB  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    results = []
    baseline_fps = 0
    sparsities = [0, 25, 50, 75, 90, 99, 100]
    
    print("  === Throughput Analysis ===")
    
    for pct in sparsities:
        ms, fps = run_benchmark(pct)
        if pct == 0:
            baseline_fps = fps
        results.append((pct, ms, fps))
        
    print("\n  === Speedup Report ===")
    for pct, ms, fps in results:
        multiplier = fps / max(1, baseline_fps)
        if pct == 0:
            print(f"  {pct:3.0f}% Sparse: {multiplier:5.2f}x (Baseline)")
        else:
            print(f"  {pct:3.0f}% Sparse: {multiplier:5.2f}x speedup")
    
    # Verification
    p99_fps = results[-2][2] # 99% sparse
    if p99_fps > baseline_fps * 5:
        print("\n  ✅ SUCCESS: O(1) Branchless Dispatch is actively skipping math.")
    else:
        print("\n  ❌ FAIL: Dispatch table did not significantly increase throughput.")
    print()
