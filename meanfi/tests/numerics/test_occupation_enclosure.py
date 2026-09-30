"""Exercise the experimental charge stage through the full filling/density API."""

import numpy as np
import pytest

import meanfi as mf


@pytest.mark.parametrize("method", ["legacy", "quadratic"])
def test_fixed_filling_cosine_density_and_fourier_moment(method):
    filling = 0.37
    hopping = {(1,): np.array([[0.5]]), (-1,): np.array([[0.5]])}
    result = mf.density_matrix(
        hopping,
        filling=filling,
        keys=[(0,), (1,)],
        integration=mf.FermiSimplex(charge_method=method, max_refinements=500),
        tol=mf.ErrorTolerances(
            scf_residual=1e-4,
            density_matrix_integration=1e-5,
            charge_integration=1e-5,
            filling_residual=1e-7,
            matrix_function_tol=1e-8,
        ),
    )
    density = result.to_tb()
    assert abs(result.filling - filling) < 1e-7
    assert abs(result.mu + np.cos(np.pi * filling)) < 5e-5
    assert abs(density[(0,)][0, 0] - filling) < 1e-5
    # Exact integral of exp(ik) on [acos(mu), 2*pi-acos(mu)].
    assert abs(density[(1,)][0, 0] + np.sin(np.pi * filling) / np.pi) < 2e-5
    assert result.errors.charge_integration <= 1e-5


def test_invalid_charge_method():
    with pytest.raises(ValueError, match="charge_method"):
        mf.FermiSimplex(charge_method="unknown")
