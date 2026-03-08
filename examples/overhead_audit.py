import torch
import torch.nn as nn
import numpy as np
import time
from residue.pytorch_bridge import PyTorchShield

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1024, 1024)
    def forward(self, x):
        return self.lin(x)

def run_overhead_audit():
    print("--- RESIDUE OVERHEAD AUDIT (WORST CASE: 0% SPARSITY) ---")
    
    FRAME_SIZE = 1024
    NUM_FRAMES = 1000
    # 100% DENSE DATA (Nothing to skip)
    data = np.random.randn(NUM_FRAMES, FRAME_SIZE).astype(np.float32)
    
    model = SimpleModel()
    shield = PyTorchShield(model, frame_size=FRAME_SIZE)
    
    # 1. Baseline
    start = time.perf_counter()
    for frame in data:
        _ = model(torch.from_numpy(frame))
    baseline_time = time.perf_counter() - start
    print(f"Baseline Time (0% skips): {baseline_time:.4f}s")
    
    # 2. Shielded (Worst Case)
    shield.start()
    start = time.perf_counter()
    for frame in data:
        # Pushing and Polling has a cost
        shield.observer.push_data(frame)
        _ = model(torch.from_numpy(frame))
    shield_time = time.perf_counter() - start
    shield.stop()
    
    print(f"Shielded Time (0% skips): {shield_time:.4f}s")
    overhead_pct = ((shield_time - baseline_time) / baseline_time) * 100
    print(f"Measured Overhead: {overhead_pct:.2f}%")
    
    print("\nCONCLUSION:")
    if overhead_pct < 5:
        print("✅ The overhead is negligible (Residue is efficient even when doing nothing).")
    else:
        print(f"⚠️ The shield adds {overhead_pct:.1f}% overhead. It needs to skip at least that much noise to be profitable.")

if __name__ == "__main__":
    run_overhead_audit()
