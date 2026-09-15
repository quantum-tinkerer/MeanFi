"""Validate the integration family against the requested physics."""

import numpy as np

from meanfi.density.integrate.methods import (
    FermiSimplex,
    IntegrationMethod,
    UniformGrid,
)


def validate_integration_method(integration: IntegrationMethod, *, kT: float) -> None:
    if not np.isfinite(kT) or kT < 0:
        raise ValueError(
            "meanfi supports only finite non-negative temperatures (kT >= 0)"
        )
    if isinstance(integration, FermiSimplex):
        if kT != 0:
            raise ValueError("FermiSimplex requires kT == 0")
        return
    if isinstance(integration, UniformGrid):
        if kT == 0 and integration.nk is None:
            raise ValueError("Zero-temperature UniformGrid requires explicit nk")
        return
    raise TypeError("integration must be an IntegrationMethod instance")
