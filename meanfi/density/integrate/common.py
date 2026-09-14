"""Validate the integration family against the requested physics."""

import numpy as np

from meanfi.density.integrate.methods import (
    AdaptiveSimplex,
    IntegrationMethod,
    PeriodicGrid,
)


def validate_integration_method(integration: IntegrationMethod, *, kT: float) -> None:
    if not np.isfinite(kT) or kT < 0:
        raise ValueError(
            "meanfi supports only finite non-negative temperatures (kT >= 0)"
        )
    if isinstance(integration, AdaptiveSimplex):
        if kT != 0:
            raise ValueError("AdaptiveSimplex requires kT == 0")
        return
    if isinstance(integration, PeriodicGrid):
        if kT == 0 and integration.nk is None:
            raise ValueError("Zero-temperature PeriodicGrid requires explicit nk")
        return
    raise TypeError("integration must be an IntegrationMethod instance")
