"""
PROJECT RESIDUE V2.0 - The Analog Scientist
Multi-dimensional ML optimization with semantic bridge
"""

from .residue_v2 import *

__version__ = "2.0.0"
__author__ = "PROJECT RESIDUE"
__description__ = "The Analog Scientist - Multi-dimensional optimization with semantic bridge"
__status__ = "Production Ready - Optimized"

# Convenience imports for production use
__all__ = [
    "EntropyControllerV2",
    "create_entropy_controller_v2",
    "compute_analog_scaling",
    "batch_compute_analog_scaling",
    "compute_skip_predict_decision",
    "batch_skip_predict_decisions",
    "FeatureVector"
]
