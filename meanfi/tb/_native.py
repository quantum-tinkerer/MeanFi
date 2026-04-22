import numpy as np

from meanfi.tb.tb import _tb_type

try:
    from meanfi._zero_temp_native import TightBindingModel, VertexCache
except ImportError:  # pragma: no cover - exercised when native extension is unavailable
    TightBindingModel = None
    VertexCache = None


def tb_to_tight_binding_model(tb: _tb_type):
    """Convert a tight-binding dictionary to the native model representation."""

    if TightBindingModel is None:  # pragma: no cover - native extension is required
        raise ImportError("meanfi._zero_temp_native is not available")

    keys = np.asarray(list(tb.keys()), dtype=np.int64, order="C")
    matrices = np.asarray(list(tb.values()), dtype=np.complex128, order="C")
    return TightBindingModel(keys, matrices)


def tb_to_vertex_cache(tb: _tb_type, tol: float = 1e-14):
    """Create a native vertex cache for a tight-binding dictionary."""

    if VertexCache is None:  # pragma: no cover - native extension is required
        raise ImportError("meanfi._zero_temp_native is not available")

    return VertexCache(tb_to_tight_binding_model(tb), float(tol))
