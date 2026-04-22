import inspect
from types import SimpleNamespace

import numpy as np
import pytest

from meanfi import Model, density_matrix, density_matrix_at_mu, solver
from meanfi.solvers import NoConvergence
from meanfi.zero_temp import _NATIVE_ZERO_TEMP_AVAILABLE, density_matrix_zero_temp
from meanfi.tests.helpers import dimerized_chain, spinful_chain


pytestmark = pytest.mark.integration
requires_native = pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)


def _base_model_kwargs():
    h_0 = spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    return {"h_0": h_0, "h_int": h_int, "filling": 1.0, "kT": 0.1}


def test_public_signatures_expose_documented_keyword_only_controls():
    model_params = inspect.signature(Model).parameters
    for name in ("kT", "charge_tol", "density_atol", "scf_tol", "max_subdivisions"):
        assert model_params[name].kind is inspect.Parameter.KEYWORD_ONLY

    solver_params = inspect.signature(solver).parameters
    for name in (
        "mu_guess",
        "optimizer",
        "optimizer_kwargs",
        "max_scf_steps",
        "callback",
        "debug",
        "return_info",
    ):
        assert solver_params[name].kind is inspect.Parameter.KEYWORD_ONLY

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["charge_tol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_params["mu_xtol"].default is None
    assert density_params["max_subdivisions"].default is None

    density_at_mu_params = inspect.signature(density_matrix_at_mu).parameters
    assert density_at_mu_params["density_atol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_at_mu_params["max_subdivisions"].default is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"filling": 0.0}, "positive scalar"),
        ({"kT": -1.0}, "kT >= 0"),
        ({"charge_tol": 0.0}, "tolerances must be positive"),
        ({"density_atol": 0.0}, "tolerances must be positive"),
        ({"scf_tol": 0.0}, "tolerances must be positive"),
        ({"max_subdivisions": -1}, "max_subdivisions"),
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
        )


def test_density_matrix_defaults_mu_xtol_to_charge_tol(monkeypatch):
    import meanfi._zero_dim as zero_dim

    captured = {}
    original = zero_dim.solve_mu

    def wrapped(*args, **kwargs):
        captured["mu_xtol"] = kwargs["mu_xtol"]
        return original(*args, **kwargs)

    monkeypatch.setattr(zero_dim, "solve_mu", wrapped)
    density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1.0,
        kT=0.2,
        keys=[()],
        charge_tol=1e-7,
        density_atol=1e-8,
    )

    assert captured["mu_xtol"] == 1e-7


def test_fixed_filling_info_reports_requested_tolerances():
    _rho, _error, _mu, info = density_matrix(
        spinful_chain(),
        filling=0.7,
        kT=0.15,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-6,
        density_atol=1e-8,
        density_rtol=0.0,
    )

    assert info.charge_integral_atol == 2.5e-7
    assert info.density_atol == 1e-8
    assert info.density_rtol == 0.0
    assert info.error_estimate_available is True
    assert info.n_kernel_evals == (
        info.charge_n_kernel_evals + info.density_n_kernel_evals
    )
    assert info.n_evaluator_evals == (
        info.charge_n_evaluator_evals + info.density_n_evaluator_evals
    )


@requires_native
@pytest.mark.parametrize("mode", ("at_mu", "fixed_filling"))
def test_root_mesh_only_zero_temperature_mode_reports_no_error_estimate(mode):
    tb = spinful_chain()
    keys = [(0,), (1,), (-1,)]

    if mode == "at_mu":
        rho, error, info = density_matrix_at_mu(
            tb,
            mu=0.2,
            kT=0.0,
            keys=keys,
            density_atol=1e-6,
            max_subdivisions=0,
        )
        assert info.subdivisions == 0
        assert info.error_estimate_available is False
        assert all(np.isnan(matrix).all() for matrix in error.values())
        assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)
        return

    rho, error, mu, info = density_matrix(
        tb,
        filling=1.0,
        kT=0.0,
        keys=keys,
        charge_tol=1e-3,
        density_atol=1e-6,
        max_subdivisions=0,
    )
    assert np.isfinite(mu)
    assert info.subdivisions == 0
    assert info.error_estimate_available is False
    assert np.isnan(info.charge_error)
    assert all(np.isnan(matrix).all() for matrix in error.values())
    assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)


def test_positive_temperature_density_matrix_does_not_use_zero_temperature_backend(
    monkeypatch,
):
    import meanfi.zero_temp as zero_temp

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError(
            "finite-temperature solve should not call zero-temperature backend"
        )

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", fail)
    rho, _error, mu, info = density_matrix(
        spinful_chain(),
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        charge_tol=1e-4,
        density_atol=1e-4,
    )

    assert np.isfinite(mu)
    assert abs(info.charge - 1.0) < 1e-4
    assert np.allclose(rho[(0,)], rho[(0,)].conj().T, atol=1e-8)


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
            SimpleNamespace(charge=1.0),
        )

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", fake_density_matrix_zero_temp)
    rho, error, mu, info = density_matrix(
        {(0,): np.zeros((1, 1)), (1,): np.zeros((1, 1)), (-1,): np.zeros((1, 1))},
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-4,
        density_atol=1e-4,
    )

    assert called["kwargs"]["density_atol"] == 1e-4
    assert np.allclose(rho[(0,)], np.array([[1.0]]))
    assert np.allclose(error[(0,)], np.array([[0.0]]))
    assert mu == 0.0
    assert info.charge == 1.0


def test_zero_temperature_runtime_error_when_native_backend_missing(monkeypatch):
    import meanfi.zero_temp as zero_temp

    monkeypatch.setattr(zero_temp, "_NATIVE_ZERO_TEMP_AVAILABLE", False)
    monkeypatch.setattr(zero_temp, "Geometry", None)

    with pytest.raises(
        RuntimeError,
        match="requires the native meanfi._zero_temp_native extension",
    ):
        density_matrix_zero_temp(
            dimerized_chain(),
            filling=1.0,
            keys=[(0,), (1,)],
            charge_tol=1e-4,
            density_atol=1e-4,
            density_rtol=0.0,
            mu_guess=0.0,
            mu_xtol=1e-4,
            max_mu_iterations=64,
            max_subdivisions=100,
        )


@requires_native
def test_zero_temperature_backend_supports_higher_dimensions():
    rho, _error, info = density_matrix_at_mu(
        {(0, 0, 0, 0): np.diag([-1.0, 1.0])},
        mu=0.0,
        kT=0.0,
        keys=[(0, 0, 0, 0)],
        density_atol=1e-12,
        max_subdivisions=10,
    )

    assert np.allclose(rho[(0, 0, 0, 0)], np.diag([1.0, 0.0]), atol=1e-12)
    assert info.n_leaves > 0


def test_solver_raises_no_convergence_when_scf_budget_is_zero():
    model = Model(
        spinful_chain(),
        {(0,): np.zeros((2, 2))},
        filling=1.0,
        kT=0.1,
        charge_tol=1e-6,
        density_atol=1e-6,
        scf_tol=1e-6,
    )

    with pytest.raises(NoConvergence) as exc_info:
        solver(model, {(0,): np.zeros((2, 2))}, max_scf_steps=0)

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

    def fake_info():
        return SimpleNamespace(
            charge_integration_calls=0,
            density_integration_calls=1,
            n_kernel_evals=0,
            n_evaluator_evals=0,
            error_estimate_available=True,
        )

    class FakeModel:
        def __init__(self, step):
            self.step = np.asarray(step, dtype=float)
            self.h_int = {(0,): np.zeros((1, 1))}
            self._ndof = 1
            self._local_key = (0,)
            self.scf_tol = 0.2

        def hamiltonian_from_meanfield(self, mf):
            return mf

        def density_matrix(self, rho, *, keys, mu_guess):
            del keys, mu_guess
            params = np.asarray(rho[(0,)], dtype=float)
            return {(0,): params + self.step}, None, 0.0, fake_info()

    def fake_density_for_hamiltonian(model, hamiltonian, *, keys, mu_guess):
        del hamiltonian, keys, mu_guess
        return {(0,): np.zeros_like(model.step)}, None, 0.0, fake_info()

    monkeypatch.setattr(solvers, "tb_to_rparams", fake_tb_to_rparams)
    monkeypatch.setattr(solvers, "rparams_to_tb", fake_rparams_to_tb)
    monkeypatch.setattr(solvers, "meanfield", fake_meanfield)
    monkeypatch.setattr(solvers, "_density_for_hamiltonian", fake_density_for_hamiltonian)

    _tb_short, info_short = solvers.solver(
        FakeModel([0.1, -0.02]),
        {(0,): np.zeros((1, 1))},
        return_info=True,
        max_scf_steps=1,
    )
    _tb_long, info_long = solvers.solver(
        FakeModel([0.1, -0.02, 0.1, -0.02, 0.1, -0.02]),
        {(0,): np.zeros((1, 1))},
        return_info=True,
        max_scf_steps=1,
    )

    assert np.isclose(info_short.residual_norm, 0.1)
    assert np.isclose(info_long.residual_norm, 0.1)
