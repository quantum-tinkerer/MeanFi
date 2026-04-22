import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from meanfi import (
    AdaptiveQuadrature,
    AdaptiveQuadratureInfo,
    AdaptiveSimplex,
    DensityMatrixResult,
    LinearMixing,
    Model,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.solvers import NoConvergence
from meanfi.zero_temp import _ZERO_TEMP_EXT_AVAILABLE
from meanfi.tests.helpers import spinful_chain


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
    for name in ("charge_tol", "density_atol", "scf_tol", "max_subdivisions"):
        assert name not in model_params

    solver_params = inspect.signature(solver).parameters
    for name in (
        "integration",
        "scf",
        "scf_tol",
        "filling_tol",
        "mu_tol",
        "max_mu_iterations",
        "optimizer",
        "optimizer_kwargs",
    ):
        assert solver_params[name].kind is inspect.Parameter.KEYWORD_ONLY

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_params["filling_tol"].default is None
    assert density_params["mu_tol"].default == 1e-10
    assert density_params["max_mu_iterations"].default == 128

    density_at_mu_params = inspect.signature(density_matrix_at_mu).parameters
    assert density_at_mu_params["integration"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "filling_tol" not in density_at_mu_params


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


def test_density_matrix_requires_local_key_for_zero_dimensional_inputs():
    with pytest.raises(ValueError, match="local key"):
        density_matrix_at_mu(
            {(): np.diag([-1.0, 1.0]), (1,): np.ones((2, 2))},
            mu=0.0,
            kT=0.1,
            keys=[()],
            integration=AdaptiveQuadrature(),
        )


def test_adaptive_methods_default_filling_tol_from_density_matrix_tol(monkeypatch):
    import meanfi._zero_dim as zero_dim

    captured = {}
    original = zero_dim.solve_mu

    def wrapped(*args, **kwargs):
        captured["charge_tol"] = kwargs["charge_tol"]
        return original(*args, **kwargs)

    monkeypatch.setattr(zero_dim, "solve_mu", wrapped)
    density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-8),
    )

    assert captured["charge_tol"] == 2e-8


def test_uniform_grid_rejects_fixed_filling_root_controls():
    with pytest.raises(ValueError, match="does not support"):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,)],
            integration=UniformGrid(nk=8),
            filling_tol=1e-4,
        )


@requires_ext
@pytest.mark.parametrize("mode", ("at_mu", "fixed_filling"))
def test_root_mesh_only_zero_temperature_mode_reports_no_error_estimate(mode):
    tb = spinful_chain()
    integration = AdaptiveSimplex(density_matrix_tol=1e-6, max_refinements=0)
    keys = [(0,), (1,), (-1,)]

    if mode == "at_mu":
        result = density_matrix_at_mu(
            tb,
            mu=0.2,
            kT=0.0,
            keys=keys,
            integration=integration,
        )
        assert result.info.refinements == 0
        assert result.info.error_estimate_available is False
        assert result.density_matrix_error is None
        assert np.allclose(
            result.density_matrix[(-1,)],
            result.density_matrix[(1,)].conj().T,
            atol=1e-8,
        )
        return

    result = density_matrix(
        tb,
        filling=1.0,
        kT=0.0,
        keys=keys,
        integration=integration,
    )
    assert np.isfinite(result.mu)
    assert result.info.refinements == 0
    assert result.info.error_estimate_available is False
    assert result.density_matrix_error is None
    assert np.allclose(
        result.density_matrix[(-1,)],
        result.density_matrix[(1,)].conj().T,
        atol=1e-8,
    )


def test_positive_temperature_density_matrix_does_not_use_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.zero_temp as zero_temp

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError(
            "AdaptiveQuadrature should not call the zero-temperature backend"
        )

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", fail)
    result = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        integration=AdaptiveQuadrature(density_matrix_tol=1e-4),
    )

    assert np.isfinite(result.mu)
    assert abs(result.filling - 1.0) <= 2e-4
    assert np.allclose(
        result.density_matrix[(0,)],
        result.density_matrix[(0,)].conj().T,
        atol=1e-8,
    )


def test_zero_temperature_density_matrix_dispatches_to_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.zero_temp as zero_temp

    called = {}

    def fake_density_matrix_zero_temp(*args, **kwargs):
        called["kwargs"] = kwargs
        return (
            {(0,): np.array([[1.0]])},
            {(0,): np.array([[0.0]])},
            0.0,
            SimpleNamespace(
                mu=0.0,
                charge=1.0,
                n_kernel_evals=1,
                unique_evals=1,
                n_evaluator_evals=1,
                n_cached_nodes=1,
                n_leaves=1,
                n_leaf_nodes=1,
                subdivisions=0,
                error_estimate_available=True,
                charge_integration_calls=1,
                density_integration_calls=1,
            ),
        )

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", fake_density_matrix_zero_temp)
    result = density_matrix(
        {(0,): np.zeros((1, 1)), (1,): np.zeros((1, 1)), (-1,): np.zeros((1, 1))},
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        integration=AdaptiveSimplex(density_matrix_tol=1e-4),
    )

    assert called["kwargs"]["density_atol"] == 1e-4
    assert np.allclose(result.density_matrix[(0,)], np.array([[1.0]]))
    assert np.allclose(result.density_matrix_error[(0,)], np.array([[0.0]]))
    assert result.mu == 0.0
    assert result.filling == 1.0


def test_zero_temperature_runtime_error_when_extension_missing(monkeypatch):
    import meanfi.zero_temp as zero_temp

    monkeypatch.setattr(zero_temp, "_ZERO_TEMP_EXT_AVAILABLE", False)
    monkeypatch.setattr(zero_temp, "Geometry", None)

    with pytest.raises(
        RuntimeError,
        match="requires the compiled meanfi._zero_temp_ext extension",
    ):
        density_matrix(
            spinful_chain(),
            filling=1.0,
            kT=0.0,
            keys=[(0,)],
            integration=AdaptiveSimplex(density_matrix_tol=1e-4),
        )


@requires_ext
def test_zero_temperature_backend_supports_higher_dimensions():
    result = density_matrix_at_mu(
        {(0, 0, 0, 0): np.diag([-1.0, 1.0])},
        mu=0.0,
        kT=0.0,
        keys=[(0, 0, 0, 0)],
        integration=AdaptiveSimplex(density_matrix_tol=1e-12, max_refinements=10),
    )

    assert np.allclose(
        result.density_matrix[(0, 0, 0, 0)],
        np.diag([1.0, 0.0]),
        atol=1e-12,
    )
    assert result.info.n_leaves > 0


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


def test_solver_raises_no_convergence_when_scf_budget_is_exhausted():
    model = Model(
        spinful_chain(),
        {(0,): np.eye(2)},
        filling=1.0,
        kT=0.1,
    )

    with pytest.raises(NoConvergence) as exc_info:
        solver(
            model,
            {(0,): 0.2 * np.eye(2)},
            integration=AdaptiveQuadrature(density_matrix_tol=1e-6),
            scf=LinearMixing(max_iterations=1, alpha=0.1),
            scf_tol=1e-30,
        )

    assert exc_info.value.last_iterate.size > 0


def test_solver_info_residual_norm_uses_max_norm_and_is_not_extensive(monkeypatch):
    import meanfi.solvers as solvers

    def fake_tb_to_rparams(tb):
        return np.asarray(tb[(0,)], dtype=float)

    def fake_rparams_to_tb(params, keys, ndof):
        del keys, ndof
        return {(0,): np.asarray(params, dtype=float)}

    def fake_meanfield(rho, h_int):
        del rho, h_int
        return {}

    def fake_result(hamiltonian, step):
        params = np.asarray(hamiltonian[(0,)], dtype=float)
        return DensityMatrixResult(
            density_matrix={(0,): params + step},
            density_matrix_error=None,
            mu=0.0,
            filling=1.0,
            target_filling=1.0,
            filling_residual=0.0,
            integration=AdaptiveQuadrature(),
            info=AdaptiveQuadratureInfo(
                n_kernel_evals=0,
                unique_evals=0,
                n_evaluator_evals=0,
                n_cached_nodes=0,
                n_leaves=0,
                n_leaf_nodes=0,
                refinements=0,
                error_estimate_available=True,
                charge_integration_calls=0,
                density_integration_calls=1,
            ),
        )

    class FakeModel:
        def __init__(self, step):
            self.step = np.asarray(step, dtype=float)
            self.h_int = {(0,): np.zeros((1, 1))}
            self._ndof = 1
            self._local_key = (0,)
            self.filling = 1.0
            self.kT = 0.2

        def hamiltonian_from_meanfield(self, mf):
            return mf

        def hamiltonian_from_rho(self, rho):
            return rho

    def fake_density_for_hamiltonian(
        model, hamiltonian, *, keys, integration, filling_tol, mu_tol, max_mu_iterations, mu_guess
    ):
        del keys, integration, filling_tol, mu_tol, max_mu_iterations, mu_guess
        return fake_result(hamiltonian, model.step)

    monkeypatch.setattr(solvers, "tb_to_rparams", fake_tb_to_rparams)
    monkeypatch.setattr(solvers, "rparams_to_tb", fake_rparams_to_tb)
    monkeypatch.setattr(solvers, "meanfield", fake_meanfield)
    monkeypatch.setattr(solvers, "_density_for_hamiltonian", fake_density_for_hamiltonian)

    info_short = solvers.solver(
        FakeModel([0.1, -0.02]),
        {(0,): np.zeros(2)},
        integration=AdaptiveQuadrature(),
        scf=LinearMixing(max_iterations=1),
        scf_tol=0.2,
    ).info
    info_long = solvers.solver(
        FakeModel([0.1, -0.02, 0.1, -0.02, 0.1, -0.02]),
        {(0,): np.zeros(6)},
        integration=AdaptiveQuadrature(),
        scf=LinearMixing(max_iterations=1),
        scf_tol=0.2,
    ).info

    assert np.isclose(info_short.residual_norm, 0.1)
    assert np.isclose(info_long.residual_norm, 0.1)
    assert info_short.total_unique_evals == info_long.total_unique_evals == 0


def test_solver_info_exposes_total_unique_evals():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2))},
        filling=1.0,
        kT=0.1,
    )
    result = solver(
        model,
        {(0,): np.zeros((2, 2))},
        integration=AdaptiveQuadrature(density_matrix_tol=1e-5),
        scf=LinearMixing(max_iterations=3),
        scf_tol=1e-5,
    )

    assert result.info.total_unique_evals >= result.density_matrix_result.info.unique_evals
    assert result.info.total_unique_evals > 0
