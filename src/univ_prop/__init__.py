"""
Universal-variable propagator.


"""

from importlib import metadata

__all__ = [
    "propagate",
    "C", "S", "solve_chi"
]

# version from pyproject.toml (requires Python ≥3.8)
__version__ = metadata.version(__package__ or __name__)
