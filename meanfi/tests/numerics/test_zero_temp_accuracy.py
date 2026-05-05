from dataclasses import dataclass

import numpy as np
import pytest

from meanfi import AdaptiveSimplex, UniformGrid, density_matrix, density_matrix_at_mu
from meanfi.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi._zero_temp_ext import Geometry
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
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
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


@requires_ext
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
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.0,
            keys=case.keys,
            integration=AdaptiveSimplex(
                density_matrix_tol=density_atol,
                max_refinements=None,
            ),
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)

        assert actual_density_error <= density_atol
        assert result.info.error_estimate_available is True
        assert_estimator_covers_actual(
            actual_density_error,
            max_density_estimate(result.density_matrix_error),
        )


@requires_ext
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
        result = density_matrix(
            tb,
            filling=filling,
            kT=0.0,
            keys=case.keys,
            integration=AdaptiveSimplex(
                density_matrix_tol=density_atol,
                max_refinements=None,
            ),
            filling_tol=scalar_tol,
            mu_tol=scalar_tol,
        )
        actual_density_error = max_density_error(result.density_matrix, reference.rho)
        actual_charge_error = abs(result.filling - filling)
        actual_mu_error = abs(result.mu - reference.mu)

        assert actual_density_error <= density_atol
        assert actual_charge_error <= scalar_tol
        assert actual_mu_error <= scalar_tol
        assert result.info.error_estimate_available is True


@requires_ext
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

    result = density_matrix_at_mu(
        tb,
        mu=1.5,
        kT=0.0,
        keys=keys,
        integration=AdaptiveSimplex(
            density_matrix_tol=3e-3,
            max_refinements=None,
        ),
    )
    actual_density_error = max_density_error(result.density_matrix, reference.rho)

    assert actual_density_error <= 2e-3
    assert result.info.error_estimate_available is True
    assert_estimator_covers_actual(
        actual_density_error,
        max_density_estimate(result.density_matrix_error),
    )


@requires_ext
@pytest.mark.parametrize("case", ZERO_TEMP_CASES, ids=lambda case: case.name)
def test_uniform_grid_density_at_mu_converges_against_dense_reference(case):
    tb = case.builder()
    reference = converged_dense_reference(
        tb,
        mu=0.0,
        kT=0.0,
        keys=case.keys,
        target_tol=2e-4,
        nk_start=case.nk_start,
        nk_max=case.nk_max,
    )

    records = []
    for nk in (17, 33, 65):
        result = density_matrix_at_mu(
            tb,
            mu=0.0,
            kT=0.0,
            keys=case.keys,
            integration=UniformGrid(nk=nk),
        )
        records.append(
            (
                result.info.unique_evals,
                max_density_error(result.density_matrix, reference.rho),
            )
        )

    assert records[0][0] < records[1][0] < records[2][0]
    assert records[0][1] > records[1][1] > records[2][1]


def test_adaptive_simplex_rejects_negative_refinement_depth():
    with pytest.raises(ValueError, match="refinement_depth must be non-negative"):
        AdaptiveSimplex(refinement_depth=-1)


@requires_ext
@pytest.mark.parametrize(
    ("depth", "expected_descendants"),
    [
        (0, 2),
        (1, 4),
        (2, 8),
    ],
)
def test_zero_temp_geometry_refine_uses_power_of_two_descendants(
    depth, expected_descendants
):
    geometry = Geometry.root(2)
    refinement = geometry.refine(np.array([0], dtype=np.int64), depth + 1)
    child_offsets = refinement[2]
    child_ids = refinement[3]

    assert child_offsets.tolist() == [0, expected_descendants]
    assert child_ids.shape == (expected_descendants,)
