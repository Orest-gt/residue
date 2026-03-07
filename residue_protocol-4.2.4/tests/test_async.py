#!/usr/bin/env python3
"""
PROJECT RESIDUE V4.0 - ASYNCHRONOUS BUFFER INGESTION TEST
=========================================================
Tests the Level 5 'AsyncObserver' architecture.
Demonstrates that Python can asynchronously feed a massive stream of data 
without waiting for computation, while reading lock-free telemetry.
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from residue.core import AsyncObserver, print_isolation_report

FRAME_SIZE = 1024
NUM_CHUNKS = 100
FRAMES_PER_CHUNK = 1000  # Total 100,000 frames (~400MB)
SPARSITY_PCT = 90.0

def make_chunk(sparsity_pct):
    # Generates a chunk of frames, applying sparsity
    size = FRAMES_PER_CHUNK * FRAME_SIZE
    data = np.empty(size, dtype=np.float32)
    
    # 1. Dense noise (Signal)
    data[:] = np.random.randn(size).astype(np.float32)
    
    # 2. Sparsity (Silence)
    num_sparse_frames = int(FRAMES_PER_CHUNK * (sparsity_pct / 100.0))
    if num_sparse_frames > 0:
        sparse_indices = np.random.choice(FRAMES_PER_CHUNK, num_sparse_frames, replace=False)
        for idx in sparse_indices:
            start = idx * FRAME_SIZE
            end = start + FRAME_SIZE
            data[start:end] = 0.0
            
    return data

def run_async_test():
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   LEVEL 5: ASYNC OBSERVER & LOCK-FREE TELEMETRY TEST   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Init Background Thread & Lock-Free Ring Buffers...    ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    observer = AsyncObserver(frame_size=FRAME_SIZE, buffer_capacity_frames=20_000)
    
    # Pre-generate chunks so we don't benchmark Python's random generator
    print("  [Python] Pre-generating 100 chunks of 1000 frames...")
    chunks = [make_chunk(SPARSITY_PCT) for _ in range(NUM_CHUNKS)]
    
    print("\n  [C++] Starting AsyncObserver Thread (Entering Isolation Zone)...")
    observer.start()
    
    # Wait for thread to spin up
    time.sleep(0.1) 
    
    print("  [Python] Rapidly feeding 100,000 frames to SPSC Queue...")
    t_start = time.perf_counter()
    
    total_pushed = 0
    chunk_idx = 0
    
    while chunk_idx < NUM_CHUNKS:
        size = len(chunks[chunk_idx])
        pushed = observer.push_data(chunks[chunk_idx])
        
        if pushed == size:
            # Whole chunk pushed
            total_pushed += pushed
            chunk_idx += 1
        elif pushed > 0:
            # Partial push, slice the remainder
            chunks[chunk_idx] = chunks[chunk_idx][pushed:]
            total_pushed += pushed
        else:
            # Buffer full, back off slightly
            time.sleep(0.001)
            
        # Poll Telemetry asynchronously every 20 chunks
        if chunk_idx % 20 == 0 and pushed > 0:
            telemetry = observer.poll_telemetry()
            print(f"    [Telemetry] Ingested: {telemetry.total_samples_ingested/FRAME_SIZE:6.0f} / 100000 | "
                  f"Processed Engine: {telemetry.total_samples_processed:6.0f} | "
                  f"Skipped Gates: {telemetry.total_samples_skipped:6.0f} | "
                  f"Sparsity: {telemetry.sparsity_pct:5.1f}% | "
                  f"FPS: {telemetry.current_fps:9,.0f}")
            
    t_feed_end = time.perf_counter()
    feed_ms = (t_feed_end - t_start) * 1000
    print(f"\n  [Python] Finished feeding {total_pushed/FRAME_SIZE:,.0f} frames in {feed_ms:.2f} ms!")
    print(f"  [Python] Python thread is free! Meanwhile, C++ is still running...\n")
    
    # Wait for completion
    timeout = 5.0
    start_wait = time.time()
    while time.time() - start_wait < timeout:
        telemetry = observer.poll_telemetry()
        if telemetry.total_samples_processed >= 100_000:
            break
        time.sleep(0.01)
        
    final_telemetry = observer.poll_telemetry()
    t_end = time.perf_counter()
    
    observer.stop()
    print("  [C++] AsyncObserver Thread stopped cleanly.")
    
    print("\n  === Final Results ===")
    print(f"  Total Processed: {final_telemetry.total_samples_processed:,}")
    print(f"  Total Skipped:   {final_telemetry.total_samples_skipped:,}")
    print(f"  Final Sparsity:  {final_telemetry.sparsity_pct:.1f}%")
    print(f"  Peak FPS:        {final_telemetry.current_fps:,.0f}")
    
    if final_telemetry.total_samples_processed == 100_000:
        print("\n  ✅ SUCCESS: SPSC Lock-Free ingestion flawlessly swallowed 400MB asynchronously.")
    else:
        print(f"\n  ❌ FAIL: Dropped frames! Processed {final_telemetry.total_samples_processed} out of 100k.")


if __name__ == "__main__":
    np.random.seed(42)
    run_async_test()
