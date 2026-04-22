import numpy as np
import pytest

from meanfi import AdaptiveQuadrature, density_matrix, density_matrix_at_mu, fermi_dirac
from meanfi.tests.helpers import (
    assert_estimator_covers_actual,
    converged_dense_reference,
    local_two_band_2d,
    max_density_error,
    max_density_estimate,
    spinful_chain,
)


pytestmark = pytest.mark.numerics


def test_finite_temperature_local_two_band_matches_exact_reference_across_tolerance_ladder(
    density_tolerance_ladder,
    scalar_tolerance_ladder,
):
    tb = local_two_band_2d()
    keys = [(0, 0), (1, 0), (0, 1)]
    exact_rho = {
        (0, 0): np.diag(fermi_dirac(np.array([-1.0, 1.0]), 0.2, 0.0)),
        (1, 0): np.zeros((2, 2)),
        (0, 1): np.zeros((2, 2)),
    }

    for density_atol, scalar_tol in zip(
        density_tolerance_ladder,
        scalar_tolerance_ladder,
        strict=True,
    ):
        result = density_matrix(
            tb,
            filling=1.0,
            kT=0.2,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
            filling_tol=scalar_tol,
        )
        actual_density_error = max_density_error(result.density_matrix, exact_rho)

        assert abs(result.mu) <= scalar_tol
        assert abs(result.filling - 1.0) <= scalar_tol
        assert actual_density_error <= density_atol
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(result.density_matrix_error),
        )
        assert result.filling_residual is not None
        assert result.filling_residual <= scalar_tol


def test_finite_temperature_density_matrix_at_mu_matches_self_converged_reference_across_density_ladder(
    density_tolerance_ladder,
):
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.15,
        keys=keys,
        target_tol=min(density_tolerance_ladder) / 10.0,
        nk_start=2001,
        nk_max=16001,
    )

    for density_atol in density_tolerance_ladder:
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.15,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)

        assert actual_density_error <= density_atol
        assert result.info.error_estimate_available is True
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(result.density_matrix_error),
        )


def test_finite_temperature_fixed_filling_matches_self_converged_reference_across_tolerance_ladder(
    density_tolerance_ladder,
    scalar_tolerance_ladder,
):
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]
    filling = 0.7
    reference = converged_dense_reference(
        tb,
        filling=filling,
        kT=0.15,
        keys=keys,
        target_tol=min(
            min(density_tolerance_ladder),
            min(scalar_tolerance_ladder),
        )
        / 10.0,
        nk_start=2001,
        nk_max=16001,
    )

    for density_atol, scalar_tol in zip(
        density_tolerance_ladder,
        scalar_tolerance_ladder,
        strict=True,
    ):
        result = density_matrix(
            tb,
            filling=filling,
            kT=0.15,
            keys=keys,
            integration=AdaptiveQuadrature(density_matrix_tol=density_atol),
            filling_tol=scalar_tol,
            mu_tol=scalar_tol,
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)
        actual_charge_error = abs(result.filling - filling)

        assert actual_density_error <= density_atol
        assert actual_charge_error <= scalar_tol
        assert result.info.error_estimate_available is True
