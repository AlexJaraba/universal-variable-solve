"""
Universal-variable propagator.


"""

from importlib import metadata

__all__ = [
    "propagate",
    "stumpff_C", "stumpff_S", "solve_chi"
]

# version from pyproject.toml (requires Python ≥3.8)
__version__ = metadata.version(__package__ or __name__)

from .core.stumpff import stumpff_C, stumpff_S
from .api.propagator import UniversalPropagator
