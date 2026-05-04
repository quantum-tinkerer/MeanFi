import numpy as np

from meanfi.tb.ops import _tb_type

try:
    from meanfi._zero_temp_ext import TightBindingModel, VertexCache
except ImportError:  # pragma: no cover - exercised when the extension is unavailable
    TightBindingModel = None
    VertexCache = None


def tb_to_tight_binding_model(tb: _tb_type):
    """Convert a tight-binding dictionary to the compiled model representation."""

    if TightBindingModel is None:  # pragma: no cover - extension is required
        raise ImportError("meanfi._zero_temp_ext is not available")

    keys = np.asarray(list(tb.keys()), dtype=np.int64, order="C")
    matrices = np.asarray(list(tb.values()), dtype=np.complex128, order="C")
    return TightBindingModel(keys, matrices)


def tb_to_vertex_cache(tb: _tb_type, tol: float = 1e-14):
    """Create a compiled vertex cache for a tight-binding dictionary."""

    if VertexCache is None:  # pragma: no cover - extension is required
        raise ImportError("meanfi._zero_temp_ext is not available")

    return VertexCache(tb_to_tight_binding_model(tb), float(tol))
