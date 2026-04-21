import inspect

import numpy as np

from meanfi import (
    Model,
    add_tb,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    meanfield,
)
from meanfi import solver


def _hubbard_chain_hamiltonian(U=2.0):
    hopping = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopping + hopping.T.conj(), (1,): hopping, (-1,): hopping.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    rho_trial = {(0,): np.diag([0.7, 0.3, 0.3, 0.7])}
    return add_tb(h_0, meanfield(rho_trial, h_int))


def _local_spinful_2d(energy=1.0):
    return {(0, 0): np.diag([-energy, energy])}


def test_fixed_filling_density_reports_separate_charge_and_density_tolerances():
    tb = _hubbard_chain_hamiltonian()
    rho, error, mu, info = density_matrix(
        tb,
        filling=2.0,
        kT=0.1,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-6,
        density_atol=1e-8,
        density_rtol=0.0,
    )

    assert np.isfinite(mu)
    assert abs(info.charge - 2.0) <= 1e-6
    assert info.charge_integral_atol == 2.5e-7
    assert info.density_atol == 1e-8
    assert info.density_rtol == 0.0
    assert info.error_estimate_available is True
    assert info.density_integration_calls == 1
    assert (
        info.n_kernel_evals == info.charge_n_kernel_evals + info.density_n_kernel_evals
    )
    assert set(rho) == {(0,), (1,), (-1,)}
    assert set(error) == {(0,), (1,), (-1,)}


def test_fixed_filling_density_matches_dense_reference_in_2d():
    tb = _local_spinful_2d()
    filling = 1.0
    kT = 0.2
    keys = [(0, 0), (1, 0), (0, 1)]

    rho, _, mu, info = density_matrix(
        tb,
        filling=filling,
        kT=kT,
        keys=keys,
        charge_tol=1e-9,
        density_atol=1e-8,
    )
    occupations = fermi_dirac(np.array([-1.0, 1.0]), kT, 0.0)
    rho_expected = np.diag(occupations)

    assert abs(mu) < 5e-6
    assert abs(info.charge - filling) < 1e-9
    assert np.allclose(rho[(0, 0)], rho_expected, atol=5e-7)
    assert np.allclose(rho[(1, 0)], np.zeros((2, 2)), atol=5e-7)
    assert np.allclose(rho[(0, 1)], np.zeros((2, 2)), atol=5e-7)


def test_public_signatures_match_the_lean_api():
    model_params = inspect.signature(Model).parameters
    assert list(model_params) == [
        "h_0",
        "h_int",
        "filling",
        "kT",
        "charge_tol",
        "density_atol",
        "scf_tol",
        "max_subdivisions",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in model_params.items()
        if name in {"kT", "charge_tol", "density_atol", "scf_tol", "max_subdivisions"}
    )

    solver_params = inspect.signature(solver).parameters
    assert list(solver_params) == [
        "model",
        "mf_guess",
        "mu_guess",
        "optimizer",
        "optimizer_kwargs",
        "max_scf_steps",
        "callback",
        "debug",
        "return_info",
    ]
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for name, parameter in solver_params.items()
        if name not in {"model", "mf_guess"}
    )

    density_params = inspect.signature(density_matrix).parameters
    assert density_params["charge_tol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_params["mu_xtol"].default is None
    assert density_params["max_subdivisions"].default is None

    density_at_mu_params = inspect.signature(density_matrix_at_mu).parameters
    assert density_at_mu_params["density_atol"].kind is inspect.Parameter.KEYWORD_ONLY
    assert density_at_mu_params["max_subdivisions"].default is None


def test_low_level_density_matrix_defaults_mu_xtol_to_charge_tol(monkeypatch):
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


def test_low_level_density_matrix_at_mu_leaves_subdivisions_unbounded_by_default(
    monkeypatch,
):
    import meanfi.mf as mf_module

    captured = {}

    class DummyResult:
        estimate = np.array([0.5 + 0.0j])
        error = np.array([0.0])
        n_kernel_evals = 1
        n_evaluator_evals = 1
        n_cached_nodes = 1
        n_leaves = 1
        n_leaf_nodes = 1
        subdivisions = 0

    monkeypatch.setattr(mf_module, "build_integrator", lambda *args, **kwargs: object())

    def fake_run_integrator(*args, **kwargs):
        captured["max_subdivisions"] = kwargs["max_subdivisions"]
        return DummyResult()

    monkeypatch.setattr(mf_module, "run_integrator", fake_run_integrator)

    rho, error, info = density_matrix_at_mu(
        {(0,): np.zeros((1, 1)), (1,): np.zeros((1, 1)), (-1,): np.zeros((1, 1))},
        mu=0.0,
        kT=0.1,
        keys=[(0,)],
    )

    assert captured["max_subdivisions"] is None
    assert np.allclose(rho[(0,)], np.array([[0.5]]))
    assert np.allclose(error[(0,)], np.array([[0.0]]))
    assert info.n_kernel_evals == 1


def test_model_forwards_max_subdivisions_to_low_level_density(monkeypatch):
    import meanfi.model as model_module

    captured = {}

    def fake_density_matrix(*args, **kwargs):
        captured["max_subdivisions"] = kwargs["max_subdivisions"]
        return ({(0,): np.eye(1)}, {(0,): np.zeros((1, 1))}, 0.0, object())

    monkeypatch.setattr(model_module, "density_matrix", fake_density_matrix)

    model = Model(
        {(0,): np.zeros((1, 1))},
        {(0,): np.zeros((1, 1))},
        filling=1.0,
        kT=0.0,
        max_subdivisions=7,
    )
    model.density_matrix({(0,): np.zeros((1, 1))})

    assert captured["max_subdivisions"] == 7
