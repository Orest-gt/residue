"""
PROJECT RESIDUE - 40% faster inference through entropy analysis
Production-ready ML optimization library
"""

from .residue import *

__version__ = "1.0.0"
__author__ = "PROJECT RESIDUE"
__description__ = "40% faster inference through input entropy analysis"
__status__ = "Production Ready"

# Convenience imports for production use
__all__ = [
    "compute_scaling",
    "batch_compute_scaling", 
    "create_entropy_controller",
    "EntropyController"
]
