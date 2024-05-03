"Mean-field tight-binding solver"

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "unknown"
    __version_tuple__ = (0, 0, "unknown", "unknown")

from .mf import construct_density_matrix

__all__ = [
    "construct_density_matrix",
    "__version__",
    "__version_tuple__",
]
