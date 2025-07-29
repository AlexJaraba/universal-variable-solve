"""
Universal-variable propagator.


"""

from importlib import metadata

__all__ = [
    "UniversalPropagator",
    "stumpff_C", "stumpff_S", "solve_chi"
]

# version from pyproject.toml (requires Python ≥3.8)
__version__ = metadata.version(__package__ or __name__)

from .pyuniversal.stumpff import stumpff_C, stumpff_S
from .api.propagator import UniversalPropagator
