from __future__ import annotations

import numpy as np

from meanfi.density.integrate.methods import (
    AdaptiveQuadrature,
    AdaptiveSimplex,
    IntegrationMethod,
    UniformGrid,
)


def workspace_complex_dtype(integration: IntegrationMethod) -> np.dtype:
    precision = getattr(integration, "workspace_precision", 128)
    if int(precision) == 64:
        return np.dtype(np.complex64)
    if int(precision) == 128:
        return np.dtype(np.complex128)
    raise ValueError("workspace_precision must be 64 or 128")


def workspace_real_dtype(integration: IntegrationMethod) -> np.dtype:
    complex_dtype = workspace_complex_dtype(integration)
    return np.dtype(
        np.float32 if complex_dtype == np.dtype(np.complex64) else np.float64
    )


def require_supported_workspace_precision(integration: IntegrationMethod) -> None:
    if (
        isinstance(integration, AdaptiveSimplex)
        and int(integration.workspace_precision) != 128
    ):
        raise ValueError(
            "AdaptiveSimplex currently supports only workspace_precision=128 because the "
            "compiled zero-temperature backend does not yet implement complex64 workspaces"
        )
    if isinstance(integration, (AdaptiveQuadrature, UniformGrid, AdaptiveSimplex)):
        return
    raise TypeError("integration must be an IntegrationMethod instance")
