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


def test_sparse_rational_dense_input_is_rejected():
    with pytest.raises(ValueError, match="supported only for sparse matrices"):
        density_matrix_at_mu(
            spinful_chain(),
            mu=0.0,
            kT=0.15,
            keys=[(0,), (1,), (-1,)],
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-2,
                matrix_function=RationalFOE(),
            ),
        )


def test_sparse_rational_sparse_input_uses_required_mumps_path():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    result = density_matrix_at_mu(
        sparse_tb,
        mu=0.0,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            matrix_function=RationalFOE(),
        ),
    )

    assert result.mu == 0.0
    assert result.info.error_estimate_available is True


def test_dense_rational_rejects_aaa_scheme():
    with pytest.raises(ValueError, match="supported only for sparse matrices"):
        density_matrix_at_mu(
            spinful_chain(),
            mu=0.0,
            kT=0.15,
            keys=[(0,), (1,), (-1,)],
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-2,
                matrix_function=RationalFOE(rational_scheme="aaa"),
            ),
        )


def test_mumps_selected_inverse_matches_dense_inverse_values():
    pytest.importorskip("mumps")
    from meanfi.density.kpoint.matrix_functions.mumps_backend import (
        SelectedInverseFactorization,
        build_selected_inverse_pattern,
    )

    matrix = sp.csc_matrix(
        np.array(
            [[2.0 + 0.0j, 1.0 - 0.2j], [1.0 + 0.2j, 3.0 + 0.0j]],
            dtype=complex,
        )
    )
    pattern = build_selected_inverse_pattern(
        size=2,
        rows=np.array([0, 1, 1]),
        cols=np.array([0, 0, 1]),
    )
    factorization = SelectedInverseFactorization()
    factorization.factor(matrix)
    selected = factorization.selected_inverse(pattern)
    inverse = np.linalg.inv(matrix.toarray())

    np.testing.assert_allclose(
        selected,
        np.array([inverse[0, 0], inverse[1, 0], inverse[1, 1]], dtype=complex),
        atol=1e-10,
        rtol=1e-10,
    )
