from dataclasses import dataclass

import pytest

from meanfi import density_matrix, density_matrix_at_mu
from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE
from meanfi.tests.helpers import (
    assert_estimator_covers_actual,
    converged_dense_reference,
    dimerized_chain,
    max_density_error,
    max_density_estimate,
    qiwuzhang,
    shifted_spinful_chain,
)


pytestmark = pytest.mark.numerics
requires_native = pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)


@dataclass(frozen=True)
class ZeroTempCase:
    name: str
    builder: object
    keys: list[tuple[int, ...]]
    nk_start: int
    nk_max: int


ZERO_TEMP_CASES = (
    ZeroTempCase(
        name="dimerized_chain_1d",
        builder=dimerized_chain,
        keys=[(0,), (1,), (-1,)],
        nk_start=251,
        nk_max=4001,
    ),
    ZeroTempCase(
        name="qiwuzhang_2d",
        builder=qiwuzhang,
        keys=[(0, 0), (1, 0), (0, 1)],
        nk_start=41,
        nk_max=321,
    ),
)


@requires_native
@pytest.mark.parametrize("case", ZERO_TEMP_CASES, ids=lambda case: case.name)
def test_zero_temperature_density_matrix_at_mu_matches_self_converged_reference_across_density_ladder(
    case,
    density_tolerance_ladder,
):
    tb = case.builder()
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.0,
        keys=case.keys,
        target_tol=min(density_tolerance_ladder) / 10.0,
        nk_start=case.nk_start,
        nk_max=case.nk_max,
    )

    for density_atol in density_tolerance_ladder:
        rho, error, info = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.0,
            keys=case.keys,
            density_atol=density_atol,
            max_subdivisions=None,
        )
        actual_density_error = max_density_error(rho, reference.rho)

        assert actual_density_error <= density_atol
        assert info.error_estimate_available is True
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(error),
        )


@requires_native
@pytest.mark.parametrize("case", ZERO_TEMP_CASES, ids=lambda case: case.name)
def test_zero_temperature_fixed_filling_matches_self_converged_reference_across_tolerance_ladder(
    case,
    density_tolerance_ladder,
    scalar_tolerance_ladder,
):
    tb = case.builder()
    filling = 1.0
    reference = converged_dense_reference(
        tb,
        filling=filling,
        kT=0.0,
        keys=case.keys,
        target_tol=min(
            min(density_tolerance_ladder),
            min(scalar_tolerance_ladder),
        )
        / 10.0,
        nk_start=case.nk_start,
        nk_max=case.nk_max,
    )

    for density_atol, scalar_tol in zip(
        density_tolerance_ladder,
        scalar_tolerance_ladder,
        strict=True,
    ):
        rho, error, mu, info = density_matrix(
            tb,
            filling=filling,
            kT=0.0,
            keys=case.keys,
            charge_tol=scalar_tol,
            density_atol=density_atol,
            mu_xtol=scalar_tol,
            max_subdivisions=None,
        )
        actual_density_error = max_density_error(rho, reference.rho)
        actual_charge_error = abs(info.charge - filling)
        actual_mu_error = abs(mu - reference.mu)

        assert actual_density_error <= density_atol
        assert actual_charge_error <= scalar_tol
        assert actual_mu_error <= scalar_tol
        assert info.error_estimate_available is True


@requires_native
def test_zero_temperature_density_at_mu_matches_reference_near_brillouin_zone_seam():
    tb = shifted_spinful_chain()
    keys = [(0,), (1,), (-1,)]
    reference = converged_dense_reference(
        tb,
        mu=1.5,
        kT=0.0,
        keys=keys,
        target_tol=2e-4,
        nk_start=2001,
        nk_max=16001,
    )

    rho, error, info = density_matrix_at_mu(
        tb,
        mu=1.5,
        kT=0.0,
        keys=keys,
        density_atol=3e-3,
        max_subdivisions=None,
    )
    actual_density_error = max_density_error(rho, reference.rho)

    assert actual_density_error <= 2e-3
    assert info.error_estimate_available is True
    assert_estimator_covers_actual(
        actual_density_error,
        max_density_estimate(error),
    )
