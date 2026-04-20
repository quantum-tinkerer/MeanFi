import numpy as np

from meanfi.tb.tb import _tb_type

try:
    from meanfi._zero_temp_native import NativeSpectralCache, NativeTightBindingModel
except ImportError:  # pragma: no cover - exercised when native extension is unavailable
    NativeSpectralCache = None
    NativeTightBindingModel = None


def tb_to_native_model(tb: _tb_type):
    """Convert a tight-binding dictionary to the native model representation."""

    if (
        NativeTightBindingModel is None
    ):  # pragma: no cover - native extension is required
        raise ImportError("meanfi._zero_temp_native is not available")

    keys = np.asarray(list(tb.keys()), dtype=np.int64, order="C")
    matrices = np.asarray(list(tb.values()), dtype=np.complex128, order="C")
    return NativeTightBindingModel(keys, matrices)


def tb_to_native_spectral_cache(tb: _tb_type, tol: float = 1e-14):
    """Create a native spectral cache for a tight-binding dictionary."""

    if NativeSpectralCache is None:  # pragma: no cover - native extension is required
        raise ImportError("meanfi._zero_temp_native is not available")

    return NativeSpectralCache(tb_to_native_model(tb), float(tol))
