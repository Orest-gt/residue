"""
PROJECT RESIDUE V2.1 - Structural Intelligence Optimizer
Multi-dimensional ML optimization with structural heuristics and optimal weight configuration
"""

from .residue_v2 import *

__version__ = "2.1.0"
__author__ = "PROJECT RESIDUE"
__description__ = "Structural Intelligence Optimizer - Multi-dimensional optimization with structural heuristics"
__status__ = "Production Ready - Structural-Emphasis"

# Convenience imports for production use
__all__ = [
    "EntropyControllerV2",
    "create_entropy_controller_v2",
    "compute_analog_scaling",
    "batch_compute_analog_scaling",
    "compute_skip_predict_decision",
    "batch_skip_predict_decisions",
    "FeatureVector",
    "FeatureVectorV3"  # V2.1 structural features
]
