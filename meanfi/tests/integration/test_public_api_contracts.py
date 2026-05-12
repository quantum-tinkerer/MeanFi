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


def _base_model_kwargs():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    return {"h_0": h_0, "h_int": h_int, "filling": 1.0, "kT": 0.1}


def test_public_signatures_expose_documented_keyword_only_controls():
    model_params = inspect.signature(Model).parameters
    assert model_params["kT"].kind is inspect.Parameter.KEYWORD_ONLY
    assert model_params["kT"].default == 0.0
    for name in ("charge_tol", "density_atol", "scf_tol", "max_subdivisions"):
        assert name not in model_params
    assert model_params["superconducting"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "bdg_meanfield" not in model_params

    solver_params = inspect.signature(solver).parameters
    for name in (
        "integration",
        "scf",
        "scf_tol",
        "filling_tol",
        "mu_tol",
        "max_charge_evaluations",
    ):
        assert solver_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert solver_params["integration"].default is None
    assert "optimizer" not in solver_params
    assert "optimizer_kwargs" not in solver_params

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["kT"].default == 0.0
    assert density_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_params["integration"].default is None
    assert density_params["filling_tol"].default is None
    assert density_params["mu_tol"].default == 1e-10
    assert density_params["max_charge_evaluations"].default is None

    density_at_mu_params = inspect.signature(density_matrix_at_mu).parameters
    assert density_at_mu_params["kT"].default == 0.0
    assert density_at_mu_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_at_mu_params["integration"].default is None
    assert "filling_tol" not in density_at_mu_params


@pytest.mark.parametrize(
    "module_name",
    [
        "meanfi._bdg",
        "meanfi._finite_temp",
        "meanfi._info",
        "meanfi._validation",
        "meanfi._zero_dim",
        "meanfi.mf",
        "meanfi.zero_temp",
        "meanfi.bdg",
    ],
)
def test_removed_shim_modules_are_no_longer_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_top_level_exports_only_supported_diagonalization_names():
    assert DirectDiagonalization.__name__ == "DirectDiagonalization"
    assert not hasattr(meanfi, "ExactDiagonalization")
    assert not hasattr(meanfi, "ChebyshevFOE")
    assert not hasattr(meanfi, "guess_tb")


def test_guess_tb_is_removed_from_public_tb_api():
    import meanfi.tb as tb

    assert not hasattr(tb, "guess_tb")
    with pytest.raises(ImportError):
        exec("from meanfi import guess_tb")
    with pytest.raises(ImportError):
        exec("from meanfi.tb import guess_tb")


def test_removed_chebyshev_public_api_is_not_importable():
    with pytest.raises(ImportError):
        exec("from meanfi import ChebyshevFOE")


def test_internal_matrix_function_package_root_exposes_shared_symbols():
    import meanfi.density.kpoint.matrix_functions as matrix_functions

    assert matrix_functions.DirectDiagonalization is DirectDiagonalization
    assert not hasattr(matrix_functions, "ChebyshevFOE")
    assert hasattr(matrix_functions, "density_block")
    assert hasattr(matrix_functions, "shift_by_mu")


def test_density_matrix_result_uses_fully_explicit_field_names():
    result = density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=AdaptiveQuadrature(),
    )

    assert isinstance(result, DensityMatrixResult)
    assert hasattr(result, "density_matrix")
    assert hasattr(result, "density_matrix_error")
    assert not hasattr(result, "rho")
    assert not hasattr(result, "rho_error")
    assert result.info.unique_evals == result.info.n_kernel_evals


def test_public_info_exposes_unique_eval_counters():
    adaptive = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
    )
    uniform = density_matrix_at_mu(
        spinful_chain(),
        mu=0.0,
        kT=0.0,
        keys=[(0,)],
        integration=UniformGrid(nk=9),
    )

    assert adaptive.info.unique_evals == adaptive.info.n_kernel_evals
    assert adaptive.info.unique_evals > 0
    assert uniform.info.unique_evals == uniform.info.n_kpoints == 9


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"filling": 0.0}, "positive scalar"),
        ({"kT": -1.0}, "kT >= 0"),
    ],
)
def test_model_rejects_invalid_scalar_controls(overrides, match):
    kwargs = _base_model_kwargs()
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        Model(**kwargs)


def test_model_rejects_nonhermitian_inputs():
    kwargs = _base_model_kwargs()
    kwargs["h_0"] = {
        (0,): np.zeros((1, 1)),
        (1,): np.array([[1.0 + 0.0j]]),
        (-1,): np.array([[2.0 + 0.0j]]),
    }

    with pytest.raises(ValueError, match="hermitian"):
        Model(**kwargs)


def test_model_is_immutable_and_owns_scf_space():
    model = Model(**_base_model_kwargs())

    assert model.scf_space is model.scf_space
    with pytest.raises(AttributeError, match="immutable"):
        model.filling = 2.0


def test_model_random_meanfield_is_seeded_and_solver_ready():
    model = Model(**_base_model_kwargs())

    first = model.random_meanfield(rng=123, scale=0.25)
    second = model.random_meanfield(rng=123, scale=0.25)
    zero = model.random_meanfield(rng=123, scale=0.0)

    for key in first:
        np.testing.assert_allclose(first[key], second[key])
        np.testing.assert_allclose(zero[key], np.zeros_like(zero[key]))
        np.testing.assert_allclose(first[key], first[tuple(-np.asarray(key))].conj().T)
