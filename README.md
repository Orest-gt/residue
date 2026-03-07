# PROJECT RESIDUE: Bare-Metal AVX2 Inference Shield for LLMs

[![Version](https://img.shields.io/badge/version-4.0.0-blue.svg)](https://github.com/project-residue/residue)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg)](#)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **The ultimate real-time inference optimization tool, dropping LLM pre-filtering overhead to near-zero by completely bypassing the OS kernel and exploiting branchless AVX2 dispatch.**  
> **STATUS:** V4.0 PRODUCTION READY - BARE METAL ISOLATION - ASYNC INGESTION

---

## The Origin: The Memory Wall
When processing massive sensor streams or high-frequency sparse data before they reach the GPU, traditional Python/NumPy logic suffers from catastrophic memory latency. Modern CPUs execute computations faster than the RAM can feed them, leading to L1 Cache starvation and rendering the execution pipeline useless.

**Project Residue solves this by operating as a "Shield" right before the neural network.** 

By analyzing the structure, complexity, and sparsity of raw data via heuristics, Residue dictates if an input block is "dense enough" to wake up the GPU, or if it is "sparse/noise" and should just bypass execution entirely. To do this without becoming a bottleneck itself, Residue V4.0 was forged directly in C++ AVX2 with techniques usually reserved for High-Frequency Trading.

---

## V4.0 Architecture Features

### 1. The Isolation Zone (OS Bypass)
Residue completely removes the Operating System's scheduler from the hot path.
* **VirtualLock (RAM Pinning):** Memory pages are locked into physical RAM (`VirtualLock` on Windows, `mlock` on Linux) guaranteeing zero page-faults.
* **Core Pinning & C-States:** The `AsyncObserver` thread locks itself to a specific physical core and requests max frequency allocation, preventing thread-migration and sleep states.
* **Kernel Timer Suppression:** OS timer interrupts are actively suppressed (1ms granularity) via `timeBeginPeriod` to prevent preemption.

### 2. Branchless Dynamic Dispatch
Traditional `if/else` statements for detecting noise severely punish instruction pipelines due to branch mispredictions. Residue V4.1 utilizes a purely mathematical approach:
* **The Heuristic Gate:** Uses AVX2 intrinsics to extract the absolute sum (L1-norm) of the first 8 floats in ~3 cycles.
* **V-Table Routing:** Instead of branching, the heuristic directly indexes a statically compiled V-Table (function pointer array), instantly routing the CPU to either the intensive `infer_single_sample_fast` or the instant `infer_single_sample_noop`.
* **The Result:** 2,360,000 FPS throughput on sparse data (a **19x** performance boost).

### 3. Asynchronous Lock-Free Ingestion
Python acts purely as a data pipe, completely decoupled from the C++ worker logic.
* **SPSC Ring Buffers:** Python pushes data and pulls results via a lock-free Single-Producer Single-Consumer queue.
* **Atomic Telemetry:** The background C++ thread reports real-time metrics (FPS, incoming sparsity %, skips) without a single mutex, allowing Python to poll diagnostics at 0 overhead.

---

## Quick Start

### Installation

Requires a C++20 compiler with AVX2 support (MSVC on Windows, GCC/Clang on Linux).

```bash
git clone https://github.com/your-repo/project-residue.git
cd project-residue

# Build and Install the V4 Engine
python setup.py build_ext --inplace
python setup.py install
```

### Basic Usage (Sync Mode)
For classic synchronous batch processing directly over NumPy arrays:

```python
import numpy as np
import residue.core as core

# 1. Initialize Controller (256 Bins, 0.1 Threshold, 1024 floats per frame)
controller = core.create_entropy_controller_v3(256, 0.1, 5, 0.1, 0.2)

# 2. Prepare Memory (Pinned)
data = np.random.randn(10_000 * 1024).astype(np.float32)

# 3. Process Stream Walled (AVX2 + Branchless Dispatch)
result_factors = controller.batch_infer_walled(data, frame_size=1024)
```

### Advanced Usage (Async Active Observer Mode)
For separating Python ingestion from the C++ processing thread (useful for pipelining before PyTorch runs):

```python
import numpy as np
import time
from residue.core import AsyncObserver, print_isolation_report

# 1. Check OS Bypass Telemetry (Must run as Administrator/Root for full memory pinning)
print_isolation_report()

# 2. Spawn Background Worker C++ Thread
observer = AsyncObserver(frame_size=1024, buffer_capacity_frames=10_000)
observer.start()  # Enters Isolation Zone

# 3. Python pushes data Non-Blocking
data = np.random.randn(500 * 1024).astype(np.float32)
observer.push_data(data)

# 4. Read Lock-Free Telemetry 
telemetry = observer.poll_telemetry()
print(f"Skipped Frames: {telemetry.total_samples_skipped}")
print(f"Real-Time FPS: {telemetry.current_fps}")

# 5. Stop Worker
observer.stop()
```

---

## Performance Validation

Tested on an AMD Ryzen 9 5900X (DDR4 3600MHz).
Framework: `tests/test_dispatch_benchmark.py`

| Sparsity (Silence) | Mode                  | Peak Throughput | Execution Speedup |
|--------------------|-----------------------|----------------|-------------------|
| **0% (Dense)**     | Heavy Compute (AVX2)  | **123,010 FPS** | 1.00x             |
| **50% (Mixed)**    | Branchless V-Table    | **221,418 FPS** | 1.80x             |
| **90% (Sparse)**   | Branchless V-Table    | **684,064 FPS** | 5.56x             |
| **99% (Extreme)**  | Pure V-Table Bypassing| **2,367,637 FPS**| **19.24x**        |

Residue absorbs extreme inputs, skipping mathematical processing on irrelevant/sparse segments in O(1) time without stalling the pipeline.

---

## License

MIT License - Free for commercial and research use.
See `LICENSE` for details.
