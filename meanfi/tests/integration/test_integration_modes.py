"""One input-driven mesh contract for both public integration families."""

import numpy as np
import pytest

from meanfi import (
    FermiSimplex,
    LinearMixing,
    Model,
    UniformGrid,
    density_matrix,
    density_matrix_at_mu,
    solver,
)
from meanfi.errors import default_solver_tolerances, resolve_integration_tolerances
from meanfi.space.coordinates import full_density_coordinates


@pytest.mark.parametrize("method", [FermiSimplex, UniformGrid])
def test_mode_is_resolved_before_policy_targets(method):
    policy = default_solver_tolerances(1e-5)
    fixed = method(nk=17, max_refinements=0)
    tolerances = resolve_integration_tolerances(fixed, policy)
    assert fixed.density_matrix_tol is fixed.charge_tol is None
    assert tolerances == policy
    adaptive = method(max_refinements=0)
    assert resolve_integration_tolerances(adaptive, policy) == policy
    assert adaptive.nk is adaptive.density_matrix_tol is adaptive.charge_tol is None


@pytest.mark.parametrize("method", [FermiSimplex, UniformGrid])
@pytest.mark.parametrize("target", ["density_matrix_tol", "charge_tol"])
def test_explicit_size_conflicts_with_explicit_targets(method, target):
    with pytest.raises(ValueError, match="nk cannot be combined"):
        method(nk=17, **{target: 1e-5})


@pytest.mark.parametrize("method", [FermiSimplex, UniformGrid])
@pytest.mark.parametrize(
    "options",
    [
        {"nk": 0},
        {"nk": 2.5},
        {"nk": True},
        {"max_points": 0},
        {"max_refinements": -1},
        {"max_refinements": 1.5},
        {"density_matrix_tol": float("nan")},
        {"charge_tol": float("inf")},
    ],
)
def test_invalid_mesh_controls_fail_early(method, options):
    with pytest.raises(ValueError):
        method(**options)


@pytest.mark.parametrize("method,kT", [(FermiSimplex, 0.0), (UniformGrid, 0.2)])
def test_prescribed_result_statistics_and_selection(method, kT):
    h = {(0,): np.diag([-1.0, 1.0])}
    result = density_matrix(
        h, filling=1, kT=kT, keys=[(0,)], integration=method(nk=17), tol=1e-8
    )
    assert result.errors.filling_residual <= 1e-9
    assert result.errors.charge_integration is None
    assert result.errors.density_matrix_integration is None
    assert result.statistics.requested_nk == 17
    assert result.statistics.n_kpoints >= 17
    assert result.statistics.n_diagonalizations >= result.statistics.n_kpoints
    selected = result.select(full_density_coordinates([(0,)], size=2))
    assert selected.statistics == result.statistics


@pytest.mark.parametrize("method,kT", [(FermiSimplex, 0.0), (UniformGrid, 0.2)])
def test_interacting_scf_keeps_prescribed_mesh_and_tol(method, kT):
    model = Model(
        {(0,): np.diag([-1.0, 1.0])},
        {(0,): np.array([[0.0, 0.2], [0.2, 0.0]])},
        filling=1,
        kT=kT,
    )
    result = solver(
        model,
        {(0,): np.zeros((2, 2))},
        integration=method(nk=17),
        scf=LinearMixing(alpha=0.5, max_iterations=100),
        tol=1e-6,
    )
    assert result.converged
    assert result.errors.scf_residual <= 1e-6
    assert result.errors.density_matrix_integration is None
    assert result.density.statistics.requested_nk == 17
    assert result.density.statistics.refinements == 0


def test_prescribed_normal_zero_temperature_periodic_public_workflow():
    result = density_matrix_at_mu(
        {(0,): np.diag([-1.0, 1.0])},
        mu=0,
        kT=0,
        keys=[(0,)],
        integration=UniformGrid(nk=11),
    )
    np.testing.assert_allclose(result.to_tb()[(0,)], np.diag([1.0, 0.0]))
    assert result.errors.density_matrix_integration is None


def test_zero_dimensional_prescribed_statistics_do_not_invent_integration_error():
    result = density_matrix(
        {(): np.diag([-1.0, 1.0])},
        filling=1,
        keys=[()],
        integration=FermiSimplex(nk=20),
    )
    assert result.statistics.requested_nk == 20
    assert result.statistics.n_kpoints == 1
    assert result.statistics.n_diagonalizations == 1
    assert result.errors.charge_integration is None
    assert result.errors.charge_integration is None


def test_sparse_shape_validation_never_materializes_a_dense_matrix(monkeypatch):
    import scipy.sparse as sp
    from meanfi.tb.validate import tb_orbital_count, validate_tb_dict

    h = {(0,): sp.eye(100_000, format="csr")}

    def fail(*args, **kwargs):
        raise AssertionError("Shape validation must not densify sparse matrices")

    monkeypatch.setattr(sp.csr_matrix, "toarray", fail)
    assert tb_orbital_count(h) == 100_000
    validate_tb_dict(h)


@pytest.mark.parametrize("kT", [-1, float("nan"), float("inf")])
def test_invalid_temperatures_cannot_produce_a_prescribed_density(kT):
    with pytest.raises(ValueError, match="finite non-negative"):
        density_matrix_at_mu(
            {(): np.eye(2)}, 0.0, kT=kT, keys=[()], integration=UniformGrid(nk=1)
        )


def test_nonfinite_mu_rejected_for_a_finite_simplex_system():
    with pytest.raises(ValueError, match="mu must be finite"):
        density_matrix_at_mu(
            {(): np.eye(2)}, float("nan"), keys=[()], integration=FermiSimplex(nk=1)
        )


@pytest.mark.parametrize("method", [FermiSimplex, UniformGrid])
def test_starting_mesh_and_fixed_mesh_are_distinct(method):
    with pytest.raises(ValueError, match="mutually exclusive"):
        method(nk=16, initial_nk=8)
    with pytest.raises(ValueError, match="initial_nk"):
        method(initial_nk=0)
    with pytest.raises(TypeError):
        method(16)


def test_simplex_initial_mesh_matches_analytic_constant_density():
    h = {(0,): np.diag([-1.0, 1.0])}
    result = density_matrix_at_mu(
        h,
        0,
        keys=[(0,)],
        integration=FermiSimplex(initial_nk=17),
    )
    np.testing.assert_allclose(result.to_tb()[(0,)], np.diag([1.0, 0.0]), atol=1e-14)
    assert result.statistics.n_kpoints >= 17
    assert result.statistics.requested_nk is None
    assert result.errors.density_matrix_integration is not None
