# ruff: noqa: F401
import importlib
import inspect
from types import SimpleNamespace

import meanfi
import numpy as np
import pytest
import scipy.sparse as sp

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveQuadratureInfo,
    AdaptiveSimplex,
    AndersonMixing,
    DensityMatrixResult,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    guess_tb,
    solver,
)
from meanfi.density.filling import mu_bracket, solve_mu
from meanfi.density.integrate.quadrature.normal import resolve_normal_matrix_function
from meanfi.density.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.density.integrate.uniform import resolve_uniform_grid_matrix_function
from meanfi.scf.engine import NoConvergence
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


def test_adaptive_methods_default_filling_tol_scales_with_density_matrix_tol(
    monkeypatch,
):
    import meanfi.density.integrate.normal as integration

    captured = {}
    original = integration._normal_zero_dim_fixed_filling

    def wrapped(*args, **kwargs):
        captured["charge_tol"] = kwargs["charge_tol"]
        return original(*args, **kwargs)

    monkeypatch.setattr(integration, "_normal_zero_dim_fixed_filling", wrapped)
    density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-8),
    )

    assert captured["charge_tol"] == 2e-9


def test_uniform_grid_accepts_finite_temperature_fixed_filling_controls():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
        integration=UniformGrid(nk=8),
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_charge_evaluations=80,
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 1e-2


def test_uniform_grid_accepts_zero_temperature_fixed_filling_controls():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=UniformGrid(nk=9),
    )

    assert isinstance(result.integration, UniformGrid)
    assert np.isfinite(result.mu)
    assert result.target_filling == 1.0


def test_uniform_grid_default_filling_tol_matches_explicit_default():
    integration = UniformGrid(nk=8, density_matrix_tol=1e-4)
    explicit_tol = 0.1 * 2 * integration.density_matrix_tol
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
