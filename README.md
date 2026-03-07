# PROJECT RESIDUE: Bare-Metal AVX2 Inference Shield for LLMs

[![Version](https://img.shields.io/badge/version-4.2.0-blue.svg)](https://github.com/project-residue/residue)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **The ultimate real-time inference optimization tool, dropping LLM pre-filtering overhead to near-zero by completely bypassing the OS kernel and exploiting predicted AVX2 gating.**  
> **STATUS:** V4.2 PRODUCTION READY - BARE METAL ISOLATION - REALITY-SYNCHRONIZED

---

## The Origin: The Memory Wall
When processing massive sensor streams or high-frequency sparse data before they reach the GPU, traditional Python/NumPy logic suffers from catastrophic memory latency. Modern CPUs execute computations faster than the RAM can feed them, leading to L1 Cache starvation and rendering the execution pipeline useless.

**Project Residue solves this by operating as a "Shield" right before the neural network.** 

By analyzing the structure, complexity, and sparsity of raw data via heuristics, Residue dictates if an input block is "dense enough" to wake up the GPU, or if it is "sparse/noise" and should just bypass execution entirely. To do this without becoming a bottleneck itself, Residue V4.2 was forged directly in C++ AVX2 with techniques usually reserved for High-Frequency Trading.

---

## V4.2 Architecture Features (Reality-Synchronized Engine)

V4.2 closes the gap between the lab and production. By taking "real-world" bottlenecks like NUMA architecture, thermal throttling, and Python GC pauses into account, Residue is now an industrial-grade engine.

### 1. The Hardened Isolation Zone (OS Bypass + NUMA)
Residue completely removes the Operating System's scheduler from the hot path with safe degradation.
* **Deterministic Memory Cascade:** 3-Tier memory locking strategy (`VirtualLock` -> `Huge Pages` -> `PrefetchVirtualMemory`) guarantees the highest possible memory priority without crashing if admin privileges are missing.
* **SMT-Aware Core Pinning:** The `AsyncObserver` thread detects Hyperthreading and locks itself to a specific *physical* core, actively avoiding contention with hardware thread siblings.
* **NUMA Topology Hinting:** Automatically detects multi-socket layouts and logs a `CRITICAL WARNING` if the Python memory allocations map to a different CPU node than the worker thread, avoiding latency penalties over the Infinity Fabric/QPI bus.

### 2. Predicted Gating (Vectorized Full-Scan)
Traditional `if/else` statements for detecting noise severely punish instruction pipelines due to branch mispredictions. Residue V4.2 utilizes a purely predictive approach:
* **Vectorized Full-Scan Gate:** Instead of relying on heuristic sampling probes, the engine uses `_mm256_max_ps` to compute the Max-Abs value of **every single float** in the frame (1024 floats) in ~71 cycles. Zero false negatives.
* **Static Branch Prediction:** Uses C++20 `[[likely]]`/`[[unlikely]]` attributes to dictate static layout. The Direct Branch trains the CPU's dynamic Branch Target Buffer (BTB) instantly, eliminating the 20-cycle penalty of indirect V-Table calls.
* **The Result:** 2,178,336 FPS throughput on highly sparse data (a **14x** performance boost over baseline heavy compute).

### 3. Asynchronous Lock-Free Ingestion & Intelligent Wait
Python acts purely as a data pipe, completely decoupled from the C++ worker logic.
* **SPSC Ring Buffers:** Python pushes data via a lock-free Single-Producer Single-Consumer queue. `recommended_push_size()` ensures optimal cache line MESI coherency.
* **Atomic Telemetry & Backpressure:** The background C++ thread reports real-time metrics (FPS, incoming sparsity %, buffer fill level %, frames dropped). Python can dynamically back off if `backpressure_active` turns true.
* **Adaptive Exponential Backoff:** If the buffer empties (e.g. during a Python GC pause), the C++ worker gracefully decays its spin-wait (`_mm_pause` -> `SwitchToThread`), avoiding Thermal Throttling from 100% idle spinning, ensuring Turbo Boost headroom remains available.

---

## Quick Start

### Installation

Requires a C++17/C++20 compiler with AVX2 support (MSVC on Windows, GCC/Clang on Linux).

```bash
git clone https://github.com/project-residue/residue.git
cd residue

# Build and Install the V4.2 Engine
python setup.py build_ext --inplace
python setup.py install
```

### Advanced Usage (Async Active Observer Mode)
For separating Python ingestion from the C++ processing thread (useful for pipelining before PyTorch runs):

```python
import numpy as np
import time
from residue.core import AsyncObserver, print_isolation_report

# 1. Check OS Bypass Telemetry (SMT detection, Memory Tiers)
print_isolation_report()

# 2. Spawn Background Worker C++ Thread
observer = AsyncObserver(frame_size=1024, buffer_capacity_frames=10_000)
observer.start()  # Enters Isolation Zone

# 3. Python pushes data Non-Blocking in optimal batches
data = np.random.randn(500 * 1024).astype(np.float32)
push_size = observer.recommended_push_size()

# Push data in chunks that minimize MESI Coherency traffic
for i in range(0, len(data), push_size):
    chunk = data[i:i + push_size]
    observer.push_data(chunk, len(chunk))

# 4. Read Lock-Free Telemetry with Backpressure
telemetry = observer.poll_telemetry()
print(f"Processed: {telemetry.total_samples_processed}")
print(f"Skipped: {telemetry.total_samples_skipped} ({telemetry.sparsity_pct:.1f}%)")
print(f"FPS: {telemetry.current_fps:.1f}")

if telemetry.backpressure_active:
    print(f"WARNING: Buffer {telemetry.buffer_fill_pct:.1f}% full!")
if telemetry.total_frames_dropped > 0:
    print(f"FATAL: Dropped {telemetry.total_frames_dropped} frames!")

# 5. Stop Worker
observer.stop()
```

---

## Performance Validation

Tested on an AMD Ryzen 9 5900X (DDR4 3600MHz).
Framework: `tests/test_dispatch_benchmark.py`

| Sparsity (Silence) | Mode                  | Peak Throughput | Execution Speedup |
|--------------------|-----------------------|----------------|-------------------|
| **0% (Dense)**     | Baseline AVX2 Math    | **148,523 FPS** | 1.00x             |
| **50% (Mixed)**    | Predicted Gating      | **437,130 FPS** | 2.94x             |
| **90% (Sparse)**   | Predicted Gating      | **1,370,433 FPS** | 9.23x             |
| **99% (Extreme)**  | Predicted Gating      | **2,178,336 FPS**| **14.67x**        |

Residue absorbs extreme inputs, skipping mathematical processing on irrelevant/sparse segments in O(1) time without stalling the pipeline.

---

## License

MIT License - Free for commercial and research use.
See `LICENSE` for details.
