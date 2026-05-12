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


def test_rational_foe_defaults_to_ozaki():
    assert RationalFOE().rational_scheme == "ozaki"


def test_sparse_normal_backend_defaults_to_aaa():
    resolved = resolve_normal_matrix_function(
        None, {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    )
    assert isinstance(resolved, RationalFOE)
    assert resolved.rational_scheme == "aaa"


def test_sparse_uniform_grid_defaults_to_aaa_at_positive_temperature():
    resolved = resolve_uniform_grid_matrix_function(
        None,
        {key: sp.csr_matrix(value) for key, value in spinful_chain().items()},
        kT=0.15,
    )
    assert isinstance(resolved, RationalFOE)
    assert resolved.rational_scheme == "aaa"


def test_dense_normal_backend_defaults_to_direct_diagonalization():
    resolved = resolve_normal_matrix_function(None, spinful_chain())
    assert isinstance(resolved, DirectDiagonalization)


def test_dense_uniform_grid_defaults_to_direct_diagonalization():
    resolved = resolve_uniform_grid_matrix_function(None, spinful_chain(), kT=0.15)
    assert isinstance(resolved, DirectDiagonalization)


def test_dense_finite_temperature_defaults_to_adaptive_quadrature_with_exact_diagonalization():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
    )

    assert isinstance(result.integration, AdaptiveQuadrature)
    assert isinstance(result.integration.matrix_function, DirectDiagonalization)


def test_sparse_finite_temperature_defaults_to_adaptive_quadrature_with_sparse_rational():
    sparse_tb = {key: sp.csr_matrix(value) for key, value in spinful_chain().items()}
    result = density_matrix(
        sparse_tb,
        filling=1.0,
        kT=0.15,
        keys=[(0,)],
    )

    assert isinstance(result.integration, AdaptiveQuadrature)
    assert isinstance(result.integration.matrix_function, RationalFOE)
    assert result.integration.matrix_function.rational_scheme == "aaa"


@requires_ext
def test_zero_temperature_defaults_to_adaptive_simplex():
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        keys=[(0,)],
    )

    assert isinstance(result.integration, AdaptiveSimplex)
