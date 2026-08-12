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
    AdaptiveSimplex,
    AndersonMixing,
    DensityResult,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
    total_energy,
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
    assert model_params["reference"].kind is inspect.Parameter.KEYWORD_ONLY
    assert model_params["reference"].default is None
    assert (
        model_params["reference_density_matrix"].kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert model_params["reference_density_matrix"].default is None
    for name in ("charge_tol", "density_atol", "scf_tol", "max_subdivisions"):
        assert name not in model_params
    assert model_params["superconducting"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "bdg_meanfield" not in model_params

    solver_params = inspect.signature(solver).parameters
    for name in (
        "integration",
        "scf",
        "tol",
        "tolerance_policy",
        "scf_tol",
        "filling_tol",
        "mu_tol",
        "max_charge_evaluations",
    ):
        assert solver_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert solver_params["integration"].default is None
    assert isinstance(solver_params["scf"].default, AndersonMixing)
    assert "accuracy" not in solver_params
    assert solver_params["tol"].default == 1e-3
    assert solver_params["scf_tol"].default is None
    assert "optimizer" not in solver_params
    assert "optimizer_kwargs" not in solver_params

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["kT"].default == 0.0
    assert density_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_params["integration"].default is None
    assert density_params["tol"].default == 1e-3
    assert density_params["filling_tol"].default is None
    assert density_params["mu_tol"].default == 1e-10
    assert density_params["max_charge_evaluations"].default is None

    selected_density_params = inspect.signature(density_matrix).parameters
    for name in ("coordinates", "interaction", "spatial_symmetries"):
        assert selected_density_params[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert selected_density_params["keys"].default is None
    assert selected_density_params["coordinates"].default is None
    assert selected_density_params["interaction"].default is None

    density_at_mu_params = inspect.signature(density_matrix_at_mu).parameters
    assert density_at_mu_params["kT"].default == 0.0
    assert density_at_mu_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_at_mu_params["integration"].default is None
    assert "filling_tol" not in density_at_mu_params

    total_energy_params = inspect.signature(total_energy).parameters
    assert list(total_energy_params) == ["model", "density_matrix"]
    assert meanfi.total_energy is total_energy

    for method in (AdaptiveSimplex, AdaptiveQuadrature, UniformGrid):
        params = inspect.signature(method).parameters
        assert params["charge_tol"].default is None
        assert params["density_matrix_tol"].default is None


def test_solver_uses_default_scf_tol_when_not_provided(monkeypatch):
    import meanfi.scf.scf as scf_pipeline

    captured = {}

    def fake_run_scf_loop(guess, *, scf, problem, verbose=False):
        captured["guess"] = guess
        captured["scf"] = scf
        captured["problem"] = problem
        captured["verbose"] = verbose
        return SimpleNamespace()

    monkeypatch.setattr(scf_pipeline, "run_scf_loop", fake_run_scf_loop)

    model = Model(**_base_model_kwargs())
    guess = {(0,): np.zeros((2, 2))}
    integration = AdaptiveQuadrature(density_matrix_tol=5.4e-4)

    result = solver(model, guess, integration=integration)

    assert result == SimpleNamespace()
    tolerances = captured["problem"].runtime.tolerances
    assert tolerances.scf_residual == pytest.approx(1e-3)
    assert tolerances.density_matrix_integration == pytest.approx(5.4e-4)
    assert tolerances.filling_residual == pytest.approx(1e-4)
    assert tolerances.charge_integration == pytest.approx(2e-4)


@pytest.mark.parametrize(
    "module_name",
    [
        "meanfi._bdg",
        "meanfi._finite_temp",
        "meanfi._info",
        "meanfi._validation",
        "meanfi._zero_dim",
        "meanfi.mean_field",
        "meanfi.zero_temp",
        "meanfi.bdg",
        "meanfi.scf.accuracy",
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
    assert not hasattr(meanfi, "tb_to_vertex_cache")
    assert not hasattr(meanfi, "tb_to_tight_binding_model")
    assert not hasattr(meanfi, "FixedAccuracy")
    assert not hasattr(meanfi, "ResidualDrivenAccuracy")
    assert not hasattr(meanfi, "SCFAccuracy")


def test_guess_tb_is_removed_from_public_tb_api():
    import meanfi.tb as tb

    assert not hasattr(tb, "guess_tb")
    assert not hasattr(tb, "tb_to_vertex_cache")
    assert not hasattr(tb, "tb_to_tight_binding_model")
    with pytest.raises(ImportError):
        exec("from meanfi import guess_tb")
    with pytest.raises(ImportError):
        exec("from meanfi import tb_to_tight_binding_model")
    with pytest.raises(ImportError):
        exec("from meanfi.tb import guess_tb")
    with pytest.raises(ImportError):
        exec("from meanfi import tb_to_vertex_cache")
    with pytest.raises(ImportError):
        exec("from meanfi.tb import tb_to_vertex_cache")


def test_removed_chebyshev_public_api_is_not_importable():
    with pytest.raises(ImportError):
        exec("from meanfi import ChebyshevFOE")


def test_internal_matrix_function_package_root_exposes_shared_symbols():
    import meanfi.density.kpoint.matrix_functions as matrix_functions

    assert matrix_functions.DirectDiagonalization is DirectDiagonalization
    assert not hasattr(matrix_functions, "ChebyshevFOE")
    assert hasattr(matrix_functions, "density_block")
    assert hasattr(matrix_functions, "shift_by_mu")


def test_density_result_has_only_physical_values_and_achieved_errors():
    result = density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=AdaptiveQuadrature(),
    )

    assert isinstance(result, DensityResult)
    assert tuple(result.__dataclass_fields__) == (
        "coordinates",
        "values",
        "mu",
        "filling",
        "errors",
    )
    assert not hasattr(result, "density_matrix_error")
    assert not hasattr(result, "info")
    assert not hasattr(result, "integration")
    assert not hasattr(result, "tolerances")


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


def test_model_rejects_invalid_reference_density_matrix_shape():
    kwargs = _base_model_kwargs()
    kwargs["reference_density_matrix"] = {(0,): np.zeros((3, 3))}

    with pytest.raises(ValueError, match="reference_density_matrix matrices"):
        Model(**kwargs)


def test_model_reference_requires_a_density_result():
    kwargs = _base_model_kwargs()
    kwargs["reference"] = {(0,): np.eye(2)}

    with pytest.raises(TypeError, match="DensityResult"):
        Model(**kwargs)


def test_model_rejects_invalid_reference_density_matrix_dimension():
    kwargs = _base_model_kwargs()
    kwargs["reference_density_matrix"] = {(0, 0): np.zeros((2, 2))}

    with pytest.raises(ValueError, match="reference_density_matrix keys"):
        Model(**kwargs)


def test_model_rejects_reference_density_matrix_for_superconducting_models():
    kwargs = _base_model_kwargs()
    kwargs["superconducting"] = True
    kwargs["reference_density_matrix"] = {(0,): np.zeros((2, 2))}

    with pytest.raises(ValueError, match="normal-state models"):
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
