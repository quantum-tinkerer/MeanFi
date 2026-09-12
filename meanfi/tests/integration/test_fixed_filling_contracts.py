import numpy as np
import pytest
import scipy.sparse as sp

from meanfi import (
    PeriodicGrid,
    density_matrix,
)
from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tb.ops import matrix_bound
from meanfi.tests.fixtures.models import spinful_chain

pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


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
    def evaluate_charge(mu: float) -> tuple[float, float, None]:
        return 1.0 / (1.0 + np.exp(-mu)), 0.0, None

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


def test_fixed_filling_root_accepts_explicit_charge_error_tolerance():
    def evaluate_charge(mu: float) -> tuple[float, float, None]:
        return 1.0 / (1.0 + np.exp(-mu)), 5e-4, None

    root = solve_mu(
        evaluate_charge=evaluate_charge,
        initial_bracket=lambda: (-4.0, 4.0),
        filling=0.7,
        mu_guess=0.0,
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=200,
        charge_error_tol=1e-3,
        use_derivative=False,
    )

    assert abs(root.charge - 0.7) <= 1e-6
    assert root.charge_error == pytest.approx(5e-4)


def test_nonpositive_derivative_fixed_filling_root_falls_back_to_bracketing():
    def evaluate_charge(mu: float) -> tuple[float, float, float]:
        charge = 1.0 / (1.0 + np.exp(-mu))
        return charge, 0.0, -1.0

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


def test_explicit_density_tolerance_does_not_redefine_other_error_budgets():
    from meanfi.errors import resolve_integration_tolerances, default_solver_tolerances

    integration, tolerances = resolve_integration_tolerances(
        PeriodicGrid(density_matrix_tol=1e-8), default_solver_tolerances(1e-3)
    )
    assert integration.charge_tol == pytest.approx(2e-4)
    assert tolerances.filling_residual == pytest.approx(1e-4)


def test_periodic_grid_accepts_finite_temperature_fixed_filling_controls():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=PeriodicGrid(nk=8),
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 1e-2


def test_periodic_grid_accepts_zero_temperature_fixed_filling_controls():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=PeriodicGrid(nk=10),
    )

    assert np.isfinite(result.mu)
    assert result.filling == pytest.approx(1.0)


def test_periodic_grid_default_filling_tol_matches_explicit_default():
    integration = PeriodicGrid(
        nk=8,
    )
    explicit_tol = 1e-4
    implicit = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=integration,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )
    explicit = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=integration,
        filling_tol=explicit_tol,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )

    assert implicit.mu == pytest.approx(explicit.mu)
    assert implicit.filling == pytest.approx(explicit.filling)


def test_prescribed_zero_temperature_mesh_rejects_unrepresentable_filling():
    with pytest.raises(RuntimeError, match="filling tolerance"):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,)],
            integration=PeriodicGrid(nk=9),
        )
