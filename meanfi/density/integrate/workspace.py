from __future__ import annotations

import numpy as np

from meanfi.density.integrate.methods import (
    AdaptiveSimplex,
    IntegrationMethod,
    PeriodicGrid,
)


def workspace_complex_dtype(integration: IntegrationMethod) -> np.dtype:
    precision = getattr(integration, "workspace_precision", 128)
    if int(precision) == 64:
        return np.dtype(np.complex64)
    if int(precision) == 128:
        return np.dtype(np.complex128)
    raise ValueError("workspace_precision must be 64 or 128")


def require_supported_workspace_precision(integration: IntegrationMethod) -> None:
    if isinstance(integration, (PeriodicGrid, AdaptiveSimplex)):
        return
    raise TypeError("integration must be an IntegrationMethod instance")
