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


@pytest.mark.perf_slow
def test_sparse_rational_fixed_filling_matches_dense_reference():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    keys = [(0,), (1,), (-1,)]
    dense_result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )
    sparse_result = density_matrix(
        sparse_tb,
        filling=1.0,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=RationalFOE(initial_poles=4, max_poles=64),
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
    )

    assert abs(sparse_result.mu - dense_result.mu) <= 1e-8
    assert abs(sparse_result.filling - dense_result.filling) <= 1e-8
    for key in keys:
        assert (
            np.max(
                np.abs(
                    sparse_result.density_matrix[key] - dense_result.density_matrix[key]
                )
            )
            <= 5e-4
        )


@pytest.mark.perf_slow
def test_sparse_rational_fixed_mu_matches_dense_reference():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    keys = [(0,), (1,), (-1,)]
    dense_result = density_matrix_at_mu(
        spinful_chain(),
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=DirectDiagonalization(),
        ),
    )
    sparse_result = density_matrix_at_mu(
        sparse_tb,
        mu=0.05,
        kT=0.15,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=20,
            matrix_function=RationalFOE(initial_poles=4, max_poles=64),
        ),
    )
    for key in keys:
        assert (
            np.max(
                np.abs(
                    sparse_result.density_matrix[key] - dense_result.density_matrix[key]
                )
            )
            <= 1e-8
        )


@pytest.mark.perf_slow
def test_bdg_sparse_rational_mumps_prepared_node_matches_solve_backend():
    from meanfi.space.density_selection import full_density_selection
    from meanfi.density.kpoint.matrix_functions import (
        density_block,
        shift_by_mu,
    )
    from meanfi.density.kpoint.matrix_functions.rational import (
        PreparedMumpsRationalNode,
    )

    matrix = sp.csr_matrix(
        np.array(
            [[0.2, 0.15 + 0.05j], [0.15 - 0.05j, -0.2]],
            dtype=complex,
        )
    )
    options = RationalFOE(initial_poles=4, max_poles=64)
    q_diag = np.array([1.0, -1.0], dtype=float)
    trace_weights = np.array([1.0, 0.0], dtype=float)
    selection = full_density_selection([tuple()], size=2)

    mumps_node = PreparedMumpsRationalNode(
        matrix,
        kT=0.2,
        q_diag=q_diag,
        options=options,
        charge_tolerance=1e-3,
        density_selection=selection,
        density_tolerance=1e-3,
        trace_weights_diag=trace_weights,
    )

    direct_density = density_block(
        DirectDiagonalization(),
        shift_by_mu(matrix, 0.05, q_diag),
        np.eye(2, dtype=complex),
        kT=0.2,
        q_diag=q_diag,
        derivative=False,
        tolerance=0.0,
    ).block
    solve_charge = float(np.real(np.sum(trace_weights * np.diag(direct_density))))
    mumps_charge, mumps_derivative = mumps_node.charge_and_derivative(0.05)
    solve_density = selection.values_from_assembled_matrix(direct_density)
    mumps_density = mumps_node.density_values_from_charge_order(0.05)

    assert np.isnan(mumps_derivative)
    assert abs(mumps_charge - solve_charge) <= 1e-8
    assert np.max(np.abs(mumps_density - solve_density)) <= 1e-8
