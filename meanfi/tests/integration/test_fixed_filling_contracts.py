from dataclasses import replace
from meanfi import default_solver_tolerances
import numpy as np
import pytest
import scipy.sparse as sp

from meanfi import (
    UniformGrid,
    density_matrix,
)
from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.tb.ops import matrix_bound
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration


def test_sparse_mu_bracket_uses_conservative_row_sum_bound():
    size = 32
    offdiag = np.ones(size - 1, dtype=complex)
    path = sp.diags([offdiag, offdiag], offsets=[-1, 1], format="csr")
    tb = {tuple(): path}

    lower, upper = mu_bracket(tb, 0.2)
    exact_bound = float(np.max(np.abs(np.linalg.eigvalsh(path.toarray()))))
    fallback = matrix_bound(path)
    padding = max(1.0, 10.0 * 0.2)

    assert upper >= exact_bound + padding
    assert upper == pytest.approx(fallback + padding)
    assert lower == -upper


def test_derivative_free_fixed_filling_root_solves_monotone_charge():
    def evaluate_charge(mu: float) -> tuple[float, None]:
        return 1.0 / (1.0 + np.exp(-mu)), None

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: (-4.0, 4.0),
        filling=0.7,
        mu_guess=0.0,
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=200,
        use_derivative=False,
    )

    assert root.derivative is None
    assert abs(root.charge - 0.7) <= 1e-6
    assert abs(root.mu - np.log(0.7 / 0.3)) <= 1e-5


def test_nonpositive_derivative_fixed_filling_root_falls_back_to_bracketing():
    def evaluate_charge(mu: float) -> tuple[float, float]:
        charge = 1.0 / (1.0 + np.exp(-mu))
        return charge, -1.0

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: (-4.0, 4.0),
        filling=0.7,
        mu_guess=0.0,
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=200,
        use_derivative=True,
    )

    assert root.derivative == -1.0
    assert abs(root.charge - 0.7) <= 1e-6
    assert abs(root.mu - np.log(0.7 / 0.3)) <= 1e-5


def test_explicit_density_tolerance_sets_charge_but_preserves_filling_residual():
    tolerances = replace(
        default_solver_tolerances(1e-3),
        density_matrix_integration=1e-8,
        charge_integration=None,
    )
    assert tolerances.charge_integration == pytest.approx(1e-8)
    assert tolerances.filling_residual == pytest.approx(1e-4)


def test_periodic_grid_accepts_finite_temperature_fixed_filling_controls():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=UniformGrid(nk=8),
        max_charge_evaluations=80,
        tol=replace(
            default_solver_tolerances(1e-3), filling_residual=1e-2, mu_tol=1e-8
        ),
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 1e-2


@pytest.mark.parametrize("nk", [None, 9, 10])
def test_zero_temperature_grid_rejects_filling_search_on_any_mesh(nk):
    with pytest.raises(NotImplementedError, match="UniformGrid fixed-filling"):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,)],
            integration=UniformGrid(nk=nk),
        )


def test_periodic_grid_default_filling_tol_matches_explicit_default():
    integration = UniformGrid(
        nk=8,
    )
    explicit_tol = 1e-4
    implicit = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=integration,
        max_charge_evaluations=80,
        tol=replace(default_solver_tolerances(1e-3), mu_tol=1e-8),
    )
    explicit = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=integration,
        max_charge_evaluations=80,
        tol=replace(
            default_solver_tolerances(1e-3), filling_residual=explicit_tol, mu_tol=1e-8
        ),
    )

    assert implicit.mu == pytest.approx(explicit.mu)
    assert implicit.filling == pytest.approx(explicit.filling)
