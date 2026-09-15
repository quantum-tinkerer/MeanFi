import importlib
import inspect
from types import SimpleNamespace

import meanfi
import numpy as np
import pytest

from meanfi import (
    AdaptiveSimplex,
    DensityResult,
    DirectDiagonalization,
    Model,
    PeriodicGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
    internal_energy,
)
from meanfi.tests.fixtures.models import spinful_chain, density_result_from_tb

pytestmark = pytest.mark.integration


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
    assert "reference_density_matrix" not in model_params
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
    assert solver_params["scf"].default is None
    assert "accuracy" not in solver_params
    assert solver_params["tol"].default == 1e-3
    assert solver_params["scf_tol"].default is None
    assert "optimizer" not in solver_params
    assert "optimizer_kwargs" not in solver_params

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["kT"].default is None
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
    assert density_at_mu_params["kT"].default is None
    assert density_at_mu_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_at_mu_params["integration"].default is None
    assert "filling_tol" not in density_at_mu_params

    internal_energy_params = inspect.signature(internal_energy).parameters
    assert list(internal_energy_params) == ["model", "density_matrix"]
    assert meanfi.internal_energy is internal_energy

    for method in (AdaptiveSimplex, PeriodicGrid, PeriodicGrid):
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
    integration = PeriodicGrid(density_matrix_tol=5.4e-4)

    result = solver(model, guess, integration=integration)

    assert result == SimpleNamespace()
    tolerances = captured["problem"].density_problem.tolerances
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
        "meanfi.density.integrate.uniform",
        "meanfi.density.integrate.quadrature.runtime",
    ],
)
def test_removed_shim_modules_are_no_longer_importable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


def test_top_level_exports_only_supported_diagonalization_names():
    assert DirectDiagonalization.__name__ == "DirectDiagonalization"
    assert not hasattr(meanfi, "AdaptiveQuadrature")
    assert not hasattr(meanfi, "UniformGrid")
    assert not hasattr(meanfi, "PeriodicQuadrature")
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
    assert not hasattr(matrix_functions, "density_block")
    assert matrix_functions.__all__ == ["DirectDiagonalization", "RationalFOE"]


def test_density_result_exposes_physical_values_errors_and_mesh_statistics():
    result = density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=PeriodicGrid(),
    )

    assert isinstance(result, DensityResult)
    assert result.coordinates is result.entries.coordinates
    assert result.values is result.entries.values
    assert result.entry_errors is result.entries.errors
    assert not result.values.flags.writeable
    assert result.filling == pytest.approx(np.trace(result.to_tb()[()]).real)
    assert result.statistics is not None
    assert not hasattr(result, "density_matrix_error")
    assert not hasattr(result, "info")
    assert not hasattr(result, "integration")
    assert not hasattr(result, "tolerances")


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"filling": -0.1}, "filling must be finite"),
        ({"filling": np.inf}, "filling must be finite"),
        ({"filling": np.nan}, "filling must be finite"),
        ({"filling": 2.1}, "filling must be finite"),
        ({"kT": np.nan}, "kT must be finite"),
        ({"kT": -1.0}, "kT must be finite"),
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
    kwargs["reference"] = density_result_from_tb({(0,): np.zeros((3, 3))})

    with pytest.raises(ValueError, match="matrix sizes do not match"):
        Model(**kwargs)


def test_model_reference_requires_a_density_result():
    kwargs = _base_model_kwargs()
    kwargs["reference"] = {(0,): np.eye(2)}

    with pytest.raises(TypeError, match="DensityResult"):
        Model(**kwargs)


def test_model_rejects_invalid_reference_density_matrix_dimension():
    kwargs = _base_model_kwargs()
    kwargs["h_int"] = {(0,): np.ones((2, 2))}
    kwargs["reference"] = density_result_from_tb({(0, 0): np.zeros((2, 2))})

    with pytest.raises(ValueError, match="missing .* required coordinate"):
        Model(**kwargs)


def test_model_rejects_reference_density_matrix_for_superconducting_models():
    kwargs = _base_model_kwargs()
    kwargs["superconducting"] = True
    kwargs["reference"] = density_result_from_tb({(0,): np.zeros((2, 2))})

    with pytest.raises(ValueError, match="normal models"):
        Model(**kwargs)


def test_model_is_immutable_and_owns_scf_space():
    model = Model(**_base_model_kwargs())

    assert model.scf_space is model.scf_space
    with pytest.raises(AttributeError, match="cannot assign"):
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


@pytest.mark.parametrize("use_sparse", [False, True])
def test_model_owns_readonly_copies_of_input_blocks(use_sparse):
    from scipy.sparse import csr_matrix

    block = np.diag([-1.0, 1.0]).astype(complex)
    if use_sparse:
        block = csr_matrix(block)
    model = Model({(): block}, {(): block * 0}, filling=1)
    if use_sparse:
        block.data[:] = 7
        np.testing.assert_allclose(model.h_0[()].diagonal(), [-1, 1])
        with pytest.raises(ValueError):
            model.h_0[()].data[0] = 7
    else:
        block[:] = 7
        np.testing.assert_allclose(model.h_0[()].diagonal(), [-1, 1])
        with pytest.raises(ValueError):
            model.h_0[()][0, 0] = 7
    with pytest.raises(TypeError):
        model.h_0[()] = block


@pytest.mark.parametrize("interaction", [{(): np.eye(1)}, {(0,): np.eye(2)}])
def test_model_rejects_interaction_size_or_dimension_mismatch(interaction):
    with pytest.raises(ValueError, match="same dimension and matrix size"):
        Model({(): np.eye(2)}, interaction, filling=1)


def test_spatial_symmetry_owns_its_inputs():
    lattice, unitary = np.eye(1), np.eye(2)
    symmetry = meanfi.SpatialSymmetry(lattice, {(0,): unitary})
    lattice[:] = 7
    unitary[:] = 7
    np.testing.assert_array_equal(symmetry.lattice_matrix, np.eye(1))
    np.testing.assert_array_equal(symmetry.unitaries_by_shift[(0,)], np.eye(2))


@pytest.mark.parametrize("superconducting", [False, True])
def test_model_density_api_matches_full_blocks_at_filling_and_mu(superconducting):
    model = Model(
        {(): np.array([[-0.3, 0.1j], [-0.1j, 0.2]])},
        {(): np.array([[0.0, 0.4], [0.4, 0.0]])},
        filling=0.8,
        kT=0.2,
        superconducting=superconducting,
    )
    correction = model.random_meanfield(rng=42, scale=0.1)
    selected = density_matrix(model, mean_field=correction, tol=1e-8)
    full = density_matrix(model, mean_field=correction, keys=[()], tol=1e-8)
    at_mu = density_matrix_at_mu(
        model, full.mu, mean_field=correction, keys=[()], tol=1e-8
    )
    np.testing.assert_allclose(
        selected.values, full.values_for(selected.coordinates), atol=1e-9
    )
    np.testing.assert_allclose(at_mu.values, full.values, atol=1e-9)
    assert full.filling == pytest.approx(0.8, abs=1e-9)
    dense, sparse = full.to_tb(), full.to_tb(sparse=True)
    np.testing.assert_allclose(sparse[()].toarray(), dense[()])
    np.testing.assert_allclose(
        model.hamiltonian_from_density(selected)[()],
        model.hamiltonian_from_density(full)[()],
        atol=1e-9,
    )
    if not superconducting:
        np.testing.assert_allclose(
            meanfi.meanfield(selected, model.h_int)[()],
            meanfi.meanfield(dense, model.h_int)[()],
        )


def test_initial_integration_failure_has_the_public_solver_exception():
    model = Model(spinful_chain(), {(0,): np.zeros((2, 2))}, filling=0.7)
    integration = AdaptiveSimplex(max_refinements=0, density_matrix_tol=1e-9)
    with pytest.raises(meanfi.ConvergenceError):
        density_matrix(model, integration=integration)
    with pytest.raises(meanfi.SolverFailure) as caught:
        solver(model, model.random_meanfield(rng=1), integration=integration)
    assert caught.value.result is None
    assert isinstance(caught.value.__cause__, meanfi.ConvergenceError)
