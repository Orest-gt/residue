#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.1 — THERMAL STRESS TEST
============================================
100,000 frames on a 400MB aligned buffer.
Measures per-tile latency to expose:
  - Thermal throttling (AVX2 frequency drop after sustained load)
  - L3 eviction pressure (buffer >> L3 cache)
  - Jitter distribution (P50/P95/P99/MAX)

Usage: python tests/v3_thermal_stress.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from residue.core import EntropyControllerV3, get_cache_topology

# =========================================================================
# CONFIG
# =========================================================================

FRAME_SIZE   = 1024       # floats per frame
NUM_FRAMES   = 100_000    # 100K frames = 400 MB
TILE_SIZE    = 64         # frames per tile (matches CoreV3 internal)
NUM_TILES    = NUM_FRAMES // TILE_SIZE
ALIGNMENT    = 64         # bytes


def make_aligned_array(size, dtype=np.float32):
    """Allocate a 64-byte aligned numpy array."""
    raw = np.empty(size + ALIGNMENT // np.dtype(dtype).itemsize, dtype=dtype)
    ptr = raw.__array_interface__['data'][0]
    off = (ALIGNMENT - (ptr % ALIGNMENT)) % ALIGNMENT
    arr = raw[off // np.dtype(dtype).itemsize:][:size]
    assert arr.__array_interface__['data'][0] % ALIGNMENT == 0
    return arr


def percentile_report(values_ms, label):
    """Print P50/P95/P99/MAX for a list of latencies."""
    arr = np.array(values_ms)
    p50  = np.percentile(arr, 50)
    p95  = np.percentile(arr, 95)
    p99  = np.percentile(arr, 99)
    pmax = np.max(arr)
    mean = np.mean(arr)
    print(f"  {label:12s}  mean={mean:8.3f}  P50={p50:8.3f}  P95={p95:8.3f}  P99={p99:8.3f}  MAX={pmax:8.3f} ms")


# =========================================================================
# STRESS TEST
# =========================================================================

def run_stress():
    topo = get_cache_topology()
    total_bytes = NUM_FRAMES * FRAME_SIZE * 4

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   RESIDUE WALL — THERMAL STRESS TEST (100K FRAMES)      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Frames      : {NUM_FRAMES:>10,}                              ║")
    print(f"║  Frame Size  : {FRAME_SIZE:>10,} floats                       ║")
    print(f"║  Buffer      : {total_bytes / (1024*1024):>10.1f} MB                           ║")
    print(f"║  Tiles       : {NUM_TILES:>10,} (×{TILE_SIZE} frames)               ║")
    print(f"║  L3 Cache    : {topo['l3_size'] / (1024*1024):>10.1f} MB                           ║")
    print(f"║  Buffer/L3   : {total_bytes / topo['l3_size']:>10.1f}x                            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # --- ALLOCATE ---
    print("[1/4] Allocating 64-byte aligned buffer...")
    data = make_aligned_array(NUM_FRAMES * FRAME_SIZE)
    np.random.seed(1337)
    # Fill in chunks to avoid massive temp array
    chunk = 1024 * 1024  # 1M floats at a time
    for i in range(0, len(data), chunk):
        end = min(i + chunk, len(data))
        data[i:end] = np.random.randn(end - i).astype(np.float32)
    print(f"       Done. ptr={hex(data.__array_interface__['data'][0])}")

    engine = EntropyControllerV3()

    # ─────────────────────────────────────────────
    # PASS 1: batch_infer_fast (baseline)
    # ─────────────────────────────────────────────
    print("\n[2/4] Running batch_infer_fast (baseline)...")
    engine.reset_history()

    fast_tile_times = []
    t_total_start = time.perf_counter()

    for tile in range(NUM_TILES):
        start_frame = tile * TILE_SIZE
        end_frame = start_frame + TILE_SIZE
        tile_data = data[start_frame * FRAME_SIZE : end_frame * FRAME_SIZE]

        t0 = time.perf_counter()
        engine.batch_infer_fast(tile_data, FRAME_SIZE)
        t1 = time.perf_counter()
        fast_tile_times.append((t1 - t0) * 1000)

    t_total_end = time.perf_counter()
    fast_total = (t_total_end - t_total_start) * 1000
    print(f"       Total: {fast_total:.1f} ms  ({NUM_FRAMES / (fast_total / 1000):,.0f} frames/sec)")

    # ─────────────────────────────────────────────
    # PASS 2: batch_infer_walled
    # ─────────────────────────────────────────────
    print("\n[3/4] Running batch_infer_walled (Residue Wall)...")
    engine.reset_history()

    walled_tile_times = []
    t_total_start = time.perf_counter()

    for tile in range(NUM_TILES):
        start_frame = tile * TILE_SIZE
        end_frame = start_frame + TILE_SIZE
        tile_data = data[start_frame * FRAME_SIZE : end_frame * FRAME_SIZE]

        t0 = time.perf_counter()
        engine.batch_infer_walled(tile_data, FRAME_SIZE)
        t1 = time.perf_counter()
        walled_tile_times.append((t1 - t0) * 1000)

    t_total_end = time.perf_counter()
    walled_total = (t_total_end - t_total_start) * 1000
    print(f"       Total: {walled_total:.1f} ms  ({NUM_FRAMES / (walled_total / 1000):,.0f} frames/sec)")

    # ─────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────
    print("\n[4/4] RESULTS")
    print("═" * 80)
    print("  Per-Tile Latency Distribution (ms per 64-frame tile):")
    percentile_report(fast_tile_times, "FAST")
    percentile_report(walled_tile_times, "WALLED")

    delta_total = ((walled_total - fast_total) / fast_total) * 100
    print(f"\n  Total Time Delta: {delta_total:+.2f}%")

    # Jitter analysis: coefficient of variation
    fast_cv = np.std(fast_tile_times) / np.mean(fast_tile_times) * 100
    walled_cv = np.std(walled_tile_times) / np.mean(walled_tile_times) * 100
    print(f"  Jitter CV (σ/μ): FAST={fast_cv:.1f}%  WALLED={walled_cv:.1f}%")

    # Thermal ramp check: compare first 100 tiles vs last 100 tiles
    fast_first = np.mean(fast_tile_times[:100])
    fast_last  = np.mean(fast_tile_times[-100:])
    walled_first = np.mean(walled_tile_times[:100])
    walled_last  = np.mean(walled_tile_times[-100:])

    print(f"\n  Thermal Ramp (first 100 tiles vs last 100 tiles):")
    print(f"    FAST:   {fast_first:.3f} → {fast_last:.3f} ms  (Δ {((fast_last - fast_first) / fast_first) * 100:+.1f}%)")
    print(f"    WALLED: {walled_first:.3f} → {walled_last:.3f} ms  (Δ {((walled_last - walled_first) / walled_first) * 100:+.1f}%)")

    # Verdict
    print("\n" + "═" * 80)
    jitter_ok = abs(delta_total) < 5.0
    thermal_ok = abs((walled_last - walled_first) / walled_first * 100) < 15.0

    if jitter_ok and thermal_ok:
        print("  🔒 VERDICT: RESIDUE WALL IS THERMALLY STABLE — LOCKED FOR PRODUCTION")
    elif not jitter_ok:
        print(f"  ⚠️  VERDICT: Jitter overhead {delta_total:+.1f}% exceeds 5% — consider Option B (Software Pipelining)")
    else:
        print(f"  ⚠️  VERDICT: Thermal ramp detected — CPU may be throttling under sustained AVX2 load")

    print("═" * 80)
    print()


if __name__ == "__main__":
    run_stress()
