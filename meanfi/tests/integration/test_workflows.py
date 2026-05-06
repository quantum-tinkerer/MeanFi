import numpy as np
import pytest

from meanfi import (
    AndersonMixing,
    LinearMixing,
    Model,
    add_tb,
    density_matrix,
    guess_tb,
    solver,
)
from meanfi.interop import kwant as utils
from meanfi.integrate.common import effective_filling_tol
from meanfi.integrate.defaults import select_default_integration
from meanfi.tests.fixtures import kwant_examples
from meanfi.integrate.simplex import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.helpers import spinful_chain


pytestmark = pytest.mark.integration
requires_ext = pytest.mark.skipif(
    not _ZERO_TEMP_EXT_AVAILABLE,
    reason="compiled zero-temperature extension is unavailable",
)


def test_graphene_kwant_end_to_end_regression():
    graphene_builder, int_builder = kwant_examples.graphene_extended_hubbard()
    h_0 = utils.builder_to_tb(graphene_builder)
    h_int = utils.builder_to_tb(int_builder, {"U": 1.0, "V": 0.0})
    np.random.seed(0)
    guess = guess_tb(frozenset(h_int), len(next(iter(h_0.values()))))

    model = Model(h_0, h_int, filling=2.0, kT=0.05)
    with pytest.warns(UserWarning, match="structurally allowed SCF support"):
        result = solver(
            model,
            guess,
            scf=AndersonMixing(M=0, max_iterations=40),
            scf_tol=5e-4,
        )
    density_result = density_matrix(
        add_tb(h_0, result.mf),
        filling=2.0,
        kT=model.kT,
        keys=list(h_int),
    )
    expected_filling_tol = effective_filling_tol(
        select_default_integration(h_0, kT=model.kT, superconducting=False),
        hamiltonian=add_tb(h_0, result.mf),
        filling_tol=None,
    )

    assert result.info.residual_norm <= 2.0 * 5e-4
    assert abs(density_result.filling - model.filling) <= expected_filling_tol
    for key, matrix in result.mf.items():
        opposite = tuple(-np.array(key))
        assert np.allclose(matrix, result.mf[opposite].conj().T)
        assert np.all(np.isfinite(density_result.density_matrix[key]))


def test_solver_supports_anderson_mixing():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(h_0, h_int, filling=1.0, kT=0.2)

    result = solver(
        model,
        guess,
        scf=AndersonMixing(M=0, line_search="wolfe", max_iterations=8),
        scf_tol=1e-8,
    )

    assert result.info.method == "anderson_mixing"
    assert result.info.residual_norm <= 1e-8
    assert np.allclose(
        result.mf[(0,)],
        -result.density_matrix_result.mu * np.eye(2),
        atol=1e-6,
    )


@requires_ext
def test_zero_temperature_model_solver_workflow_supports_zero_interaction():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(h_0, h_int, filling=1.0)

    result = solver(
        model,
        guess,
        scf=LinearMixing(),
        scf_tol=1e-3,
    )

    assert abs(result.density_matrix_result.mu) < 1e-3
    assert np.allclose(
        result.mf[(0,)],
        -result.density_matrix_result.mu * np.eye(2),
        atol=1e-3,
    )
