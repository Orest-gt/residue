import torch
import torch.nn as nn
import numpy as np
from residue.core import AsyncObserver
import time

class PyTorchShield(nn.Module):
    """
    A high-performance wrapper for PyTorch models that uses Project Residue
    to selectively bypass inference on sparse/noisy input frames.
    """
    def __init__(self, model, frame_size=1024, buffer_capacity=10000, bypass_threshold=100.0):
        super().__init__()
        self.model = model
        self.frame_size = frame_size
        self.observer = AsyncObserver(frame_size=frame_size, buffer_capacity_frames=buffer_capacity)
        self.bypass_threshold = bypass_threshold # 100.0 means "only skip 100% silence"
        self.is_active = False

    def start(self):
        """Activates the C++ background worker (Isolation Zone)."""
        self.observer.start()
        self.is_active = True

    def stop(self):
        """Shuts down the worker."""
        self.observer.stop()
        self.is_active = False

    def forward(self, x):
        """
        Forward pass with selective inference.
        Input x: torch.Tensor of shape (batch, frame_size) or (frame_size,)
        """
        if not self.is_active:
            return self.model(x)

        # Convert to numpy for Residue ingestion (zero-copy if possible)
        # We assume float32 here as Residue is optimized for it
        if x.is_cuda:
            x_cpu = x.detach().cpu().numpy()
        else:
            x_cpu = x.detach().numpy()

        # Handle batching
        if x_cpu.ndim == 1:
            frames = [x_cpu]
        else:
            frames = x_cpu

        results = []
        for frame in frames:
            # 1. Push to Residue Shield
            self.observer.push_data(frame)
            
            # 2. Check Telemetry for this specific frame
            # In a real batch pipeline, we'd process telemetry asynchronously.
            # Here we simulate the gating decision.
            tel = self.observer.poll_telemetry()
            
            if tel.sparsity_pct < self.bypass_threshold:
                # Dense/Signal detected: Run the heavy model
                res = self.model(torch.from_numpy(frame).to(x.device))
                results.append(res)
            else:
                # Noise detected: Bypass!
                # Return a zero-tensor or a "no-op" result
                # Note: The shape depends on the model's output
                results.append(None) # Flag for skipped inference

        return results

    def get_stats(self):
        """Returns the efficiency metrics from the Residue Shield."""
        return self.observer.poll_telemetry()
