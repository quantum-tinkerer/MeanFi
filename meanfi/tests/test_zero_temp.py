import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import anderson, brentq

from meanfi import (
    Model,
    add_tb,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    guess_tb,
    solver,
    tb_to_tight_binding_model,
    tb_to_kfunc,
    tb_to_vertex_cache,
)
from meanfi.zero_temp import (
    _NATIVE_ZERO_TEMP_AVAILABLE,
    density_matrix_zero_temp,
)

if _NATIVE_ZERO_TEMP_AVAILABLE:
    from meanfi._zero_temp_native import (
        AdaptiveIntegrator,
        DensityIntegrateOptions,
        Geometry,
    )
else:  # pragma: no cover - only exercised when native extension is unavailable
    AdaptiveIntegrator = None
    DensityIntegrateOptions = None
    Geometry = None


def _spinful_chain():
    hopping = -np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def _shifted_spinful_chain(phi=np.pi / 3.0):
    hopping = -np.exp(1j * phi) * np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


def _bipartite_hubbard_1d(U: float):
    hop = np.kron(np.array([[0, 1], [0, 0]], dtype=complex), np.eye(2))
    h_0 = {(0,): hop + hop.T.conj(), (1,): hop, (-1,): hop.T.conj()}
    h_int = {(0,): U * np.kron(np.eye(2), np.ones((2, 2)))}
    return h_0, h_int


def _qiwuzhang(m=0.5):
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return {
        (0, 0): m * sz,
        (1, 0): 0.5 * sz - 0.5j * sx,
        (-1, 0): 0.5 * sz + 0.5j * sx,
        (0, 1): 0.5 * sz - 0.5j * sy,
        (0, -1): 0.5 * sz + 0.5j * sy,
    }


def _cubic_band_3d(t=1.0):
    hop = -t * np.ones((1, 1))
    return {
        (0, 0, 0): np.zeros((1, 1)),
        (1, 0, 0): hop,
        (-1, 0, 0): hop.conj().T,
        (0, 1, 0): hop,
        (0, -1, 0): hop.conj().T,
        (0, 0, 1): hop,
        (0, 0, -1): hop.conj().T,
    }


def _square_band_2d(t=1.0):
    hop = -t * np.ones((1, 1))
    return {
        (0, 0): np.zeros((1, 1)),
        (1, 0): hop,
        (-1, 0): hop.conj().T,
        (0, 1): hop,
        (0, -1): hop.conj().T,
    }


def _local_two_band_3d(energy=1.0):
    return {(0, 0, 0): np.diag([-energy, energy])}


def _duplicated_local_two_band_1d(energy=1.0):
    primitive = {(0,): np.diag([-energy, energy])}
    doubled = {(0,): np.diag([-energy, energy, -energy, energy])}
    return primitive, doubled


def _dense_reference_data(tb, nk: int):
    ndim = len(next(iter(tb)))
    hkfunc = tb_to_kfunc(tb)
    axes = [np.linspace(-np.pi, np.pi, nk, endpoint=False) for _ in range(ndim)]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([axis.ravel() for axis in mesh], axis=-1)
    h_k = hkfunc(points)
    if h_k.ndim == 2:
        h_k = h_k[np.newaxis, ...]
    eigenvalues, eigenvectors = np.linalg.eigh(h_k)
    return points, eigenvalues, eigenvectors


def _dense_charge(data, mu: float) -> float:
    _, eigenvalues, _ = data
    return float(np.mean(np.sum(fermi_dirac(eigenvalues, 0.0, mu), axis=-1)))


def _dense_mu(data, filling: float) -> float:
    _, eigenvalues, _ = data
    lower = float(np.min(eigenvalues)) - 1.0
    upper = float(np.max(eigenvalues)) + 1.0
    return float(brentq(lambda mu: _dense_charge(data, mu) - filling, lower, upper))


def _dense_density(data, mu: float, keys: list[tuple[int, ...]]):
    points, eigenvalues, eigenvectors = data
    occupation = fermi_dirac(eigenvalues, 0.0, mu)
    density_matrix_k = (
        eigenvectors
        * occupation[:, np.newaxis, :]
        @ eigenvectors.conj().transpose(0, 2, 1)
    )
    rho = {}
    for key in keys:
        phase = np.exp(1j * np.dot(points, np.array(key, dtype=float)))
        rho[key] = np.einsum("k,kab->ab", phase / points.shape[0], density_matrix_k)
    return rho


def _assert_density_close(rho, rho_ref, *, atol: float):
    for key in rho_ref:
        assert np.allclose(rho[key], rho_ref[key], atol=atol)


def _tutorial_zero_temp_validation():
    tutorial_root = Path(__file__).resolve().parents[2] / "docs" / "source" / "tutorial"
    tutorial_root_str = str(tutorial_root)
    if tutorial_root_str not in sys.path:
        sys.path.insert(0, tutorial_root_str)
    from scripts import zero_temp_validation

    return zero_temp_validation


def _solve_low_u_hubbard_solution(U: float = 4.0 / 29.0):
    h_0, h_int = _bipartite_hubbard_1d(U)
    np.random.seed(0)
    guess = guess_tb(frozenset(h_int), 4)
    model = Model(
        h_0,
        h_int,
        filling=2.0,
        kT=0.0,
        charge_tol=1e-3,
        density_atol=1e-3,
        scf_tol=2e-3,
        max_subdivisions=128,
    )
    mf_sol, solver_info = solver(
        model,
        guess,
        optimizer=anderson,
        optimizer_kwargs={
            "M": 0,
            "line_search": "wolfe",
            "maxiter": 80,
            "f_tol": model.scf_tol,
        },
        max_scf_steps=80,
        return_info=True,
    )
    h_full = add_tb(h_0, mf_sol)
    rho, _, _, density_info = density_matrix(
        h_full,
        filling=2.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-3,
        density_atol=1e-3,
        max_subdivisions=128,
    )
    return h_full, rho[(0,)], solver_info, density_info


def _exact_spinful_chain_mu(filling: float) -> float:
    return float(-2.0 * np.cos(0.5 * np.pi * filling))


def _exact_spinful_chain_charge(mu: float) -> float:
    if mu <= -2.0:
        return 0.0
    if mu >= 2.0:
        return 2.0
    return float(2.0 * np.arccos(-0.5 * mu) / np.pi)


def test_zero_temperature_density_matrix_matches_dense_reference_in_1d():
    tb = _spinful_chain()
    keys = [(0,), (1,), (-1,)]
    filling = 0.7
    dense = _dense_reference_data(tb, nk=8001)
    rho_ref = _dense_density(dense, _dense_mu(dense, filling=filling), keys)

    rho, _, _, info = density_matrix(
        tb,
        filling=filling,
        kT=0.0,
        keys=keys,
        charge_tol=1e-4,
        density_atol=1e-4,
        max_subdivisions=400,
    )

    assert abs(info.charge - filling) <= 1e-4
    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=6e-2)


def test_zero_temperature_fixed_filling_tracks_exact_mu_on_analytic_chain():
    tb = _spinful_chain()
    keys = [(0,), (1,)]

    for filling in (0.1, 0.25, 0.5, 0.9, 1.1, 1.5, 1.75, 1.9):
        _, _, mu, info = density_matrix(
            tb,
            filling=filling,
            kT=0.0,
            keys=keys,
            charge_tol=5e-5,
            density_atol=1e-2,
            max_subdivisions=600,
        )

        assert abs(mu - _exact_spinful_chain_mu(filling)) <= 1e-4
        assert abs(_exact_spinful_chain_charge(mu) - filling) <= 3e-5
        assert info.subdivisions > 0


def test_zero_temperature_density_matrix_at_mu_matches_dense_reference_in_1d():
    tb = _spinful_chain()
    keys = [(0,), (1,), (-1,)]
    mu = 0.0
    dense = _dense_reference_data(tb, nk=8001)
    rho_ref = _dense_density(dense, mu, keys)

    rho, _, info = density_matrix_at_mu(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        density_atol=1e-4,
        max_subdivisions=400,
    )

    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=2e-2)


def test_zero_temperature_density_matrix_matches_dense_reference_in_2d():
    tb = _qiwuzhang()
    keys = [(0, 0), (1, 0), (0, 1)]
    filling = 1.0
    dense = _dense_reference_data(tb, nk=161)
    mu_ref = _dense_mu(dense, filling=filling)
    rho_ref = _dense_density(dense, mu_ref, keys)

    rho, _, mu, info = density_matrix(
        tb,
        filling=filling,
        kT=0.0,
        keys=keys,
        charge_tol=2e-3,
        density_atol=1e-3,
        max_subdivisions=4000,
    )

    assert abs(info.charge - filling) <= 2e-3
    assert abs(mu - mu_ref) < 1e-8
    _assert_density_close(rho, rho_ref, atol=1e-3)


def test_zero_temperature_density_matrix_at_mu_matches_dense_reference_in_2d():
    tb = _qiwuzhang()
    keys = [(0, 0), (1, 0), (0, 1)]
    mu = 0.2
    dense = _dense_reference_data(tb, nk=161)
    rho_ref = _dense_density(dense, mu, keys)

    rho, _, info = density_matrix_at_mu(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        density_atol=1e-3,
        max_subdivisions=4000,
    )

    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=1e-3)


def test_zero_temperature_density_matrix_matches_dense_reference_in_3d():
    tb = _local_two_band_3d()
    keys = [(0, 0, 0)]
    filling = 1.0
    dense = _dense_reference_data(tb, nk=11)
    mu_ref = _dense_mu(dense, filling=filling)
    rho_ref = _dense_density(dense, mu_ref, keys)

    rho, _, mu, info = density_matrix(
        tb,
        filling=filling,
        kT=0.0,
        keys=keys,
        charge_tol=1e-12,
        density_atol=1e-12,
        max_subdivisions=10,
    )

    assert abs(info.charge - filling) <= 1e-12
    assert abs(mu - mu_ref) < 1e-12
    _assert_density_close(rho, rho_ref, atol=1e-12)


def test_zero_temperature_density_matrix_at_mu_matches_dense_reference_in_3d():
    tb = _cubic_band_3d()
    keys = [(0, 0, 0), (1, 0, 0)]
    mu = -1.0
    dense = _dense_reference_data(tb, nk=41)
    rho_ref = _dense_density(dense, mu, keys)

    rho, _, info = density_matrix_at_mu(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        density_atol=2e-2,
        max_subdivisions=5000,
    )

    _assert_density_close(rho, rho_ref, atol=4e-2)


def test_zero_temperature_density_at_mu_matches_dense_reference_near_bz_seam():
    tb = _spinful_chain()
    keys = [(0,), (1,), (-1,)]
    mu = 1.5
    dense = _dense_reference_data(tb, nk=8001)
    rho_ref = _dense_density(dense, mu, keys)

    rho, _, info = density_matrix_at_mu(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        density_atol=1e-4,
        max_subdivisions=500,
    )

    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=2e-2)


def test_zero_temperature_density_at_mu_matches_dense_reference_near_2d_bz_seam_corner():
    tb = _square_band_2d()
    keys = [(0, 0), (1, 0), (0, 1)]
    mu = 3.0
    dense = _dense_reference_data(tb, nk=801)
    rho_ref = _dense_density(dense, mu, keys)

    rho, _, info = density_matrix_at_mu(
        tb,
        mu=mu,
        kT=0.0,
        keys=keys,
        density_atol=2e-3,
        max_subdivisions=3000,
    )

    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=1e-2)


def test_zero_temperature_density_matrix_is_exact_in_zero_dimensional_limit():
    h = {(): np.diag([-1.0, 2.0])}
    rho_at_mu, _, info_at_mu = density_matrix_at_mu(h, mu=0.0, kT=0.0, keys=[()])
    rho_fill, _, mu_fill, info_fill = density_matrix(
        h,
        filling=1.0,
        kT=0.0,
        keys=[()],
        charge_tol=1e-12,
        density_atol=1e-12,
    )

    assert np.allclose(rho_at_mu[()], np.diag([1.0, 0.0]), atol=1e-12)
    assert np.allclose(rho_fill[()], np.diag([1.0, 0.0]), atol=1e-12)
    assert abs(mu_fill - 0.5) < 1e-12
    assert info_at_mu.n_kernel_evals == 1
    assert info_fill.n_kernel_evals == 1


def test_zero_temperature_backend_supports_higher_dimensions():
    h = {(0, 0, 0, 0): np.diag([-1.0, 1.0])}

    rho, _, info = density_matrix_at_mu(
        h,
        mu=0.0,
        kT=0.0,
        keys=[(0, 0, 0, 0)],
        density_atol=1e-12,
        max_subdivisions=10,
    )

    assert np.allclose(rho[(0, 0, 0, 0)], np.diag([1.0, 0.0]), atol=1e-12)
    assert info.n_leaves > 0


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_root_mesh_only_at_mu_skips_preview_and_reports_no_errors(
    monkeypatch,
):
    import meanfi.zero_temp as zero_temp

    captured = {}
    original = zero_temp._build_native_runtime

    def wrapped(h):
        runtime = original(h)
        captured["geometry"] = runtime[0]
        captured["n_simplices_initial"] = runtime[0].n_simplices
        return runtime

    monkeypatch.setattr(zero_temp, "_build_native_runtime", wrapped)

    rho, error, info = density_matrix_at_mu(
        _spinful_chain(),
        mu=0.2,
        kT=0.0,
        keys=[(0,), (1,), (-1,)],
        density_atol=1e-6,
        max_subdivisions=0,
    )

    assert info.subdivisions == 0
    assert info.error_estimate_available is False
    assert captured["geometry"].n_simplices == captured["n_simplices_initial"]
    assert all(np.isnan(matrix).all() for matrix in error.values())
    assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_root_mesh_only_fixed_filling_skips_preview_and_reports_no_errors(
    monkeypatch,
):
    import meanfi.zero_temp as zero_temp

    captured = {}
    original = zero_temp._build_native_runtime

    def wrapped(h):
        runtime = original(h)
        captured["geometry"] = runtime[0]
        captured["n_simplices_initial"] = runtime[0].n_simplices
        return runtime

    monkeypatch.setattr(zero_temp, "_build_native_runtime", wrapped)

    rho, error, mu, info = density_matrix(
        _spinful_chain(),
        filling=1.0,
        kT=0.0,
        keys=[(0,), (1,), (-1,)],
        charge_tol=1e-3,
        density_atol=1e-6,
        max_subdivisions=0,
    )

    assert np.isfinite(mu)
    assert info.subdivisions == 0
    assert info.error_estimate_available is False
    assert np.isnan(info.charge_error)
    assert captured["geometry"].n_simplices == captured["n_simplices_initial"]
    assert all(np.isnan(matrix).all() for matrix in error.values())
    assert np.allclose(rho[(-1,)], rho[(1,)].conj().T, atol=1e-8)


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_2d_root_mesh_uses_four_physical_k_points():
    h = {(0, 0): np.array([[0.0]])}
    vertex_cache = tb_to_vertex_cache(h, tol=1e-14)
    geometry = Geometry.root(2, root_subcells_per_axis=2)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0, 0.0)], dtype=float),
        tol=1e-14,
    )
    integrator.evaluate_charge(0.0, 0)
    assert vertex_cache.n_kernel_evals == 4

    integrator.evaluate_density(0.0, 0)
    assert vertex_cache.n_kernel_evals == 4


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_density_is_invariant_under_equivalent_local_supercell():
    primitive, doubled = _duplicated_local_two_band_1d()

    rho_primitive, _, mu_primitive, info_primitive = density_matrix(
        primitive,
        filling=1.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-12,
        density_atol=1e-12,
        max_subdivisions=4,
    )
    rho_doubled, _, mu_doubled, info_doubled = density_matrix(
        doubled,
        filling=2.0,
        kT=0.0,
        keys=[(0,)],
        charge_tol=1e-12,
        density_atol=1e-12,
        max_subdivisions=4,
    )

    assert info_primitive.error_estimate_available is True
    assert info_doubled.error_estimate_available is True
    assert info_primitive.subdivisions == info_doubled.subdivisions == 0
    assert abs(mu_primitive - mu_doubled) < 1e-12
    assert np.allclose(rho_primitive[(0,)], rho_doubled[(0,)][:2, :2], atol=1e-12)
    assert np.allclose(rho_primitive[(0,)], rho_doubled[(0,)][2:, 2:], atol=1e-12)
    assert np.allclose(rho_doubled[(0,)][:2, 2:], np.zeros((2, 2)), atol=1e-12)


def test_zero_temperature_solver_supports_zero_interaction():
    h_0 = _spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.0,
        charge_tol=1e-3,
        density_atol=1e-3,
        scf_tol=1e-3,
    )

    solution, info = solver(model, guess, return_info=True)

    assert abs(info.mu) < 1e-3
    assert np.allclose(solution[(0,)], -info.mu * np.eye(2), atol=1e-3)


def test_zero_temperature_driver_builds_fresh_runtime_each_call(monkeypatch):
    import meanfi.zero_temp as zero_temp

    calls = []
    original = zero_temp._build_native_runtime

    def wrapped(h):
        runtime = original(h)
        calls.append(runtime[0])
        return runtime

    monkeypatch.setattr(zero_temp, "_build_native_runtime", wrapped)

    h_0 = _spinful_chain()
    h_int = {(0,): np.zeros((2, 2))}
    guess = {(0,): np.zeros((2, 2))}
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.0,
        charge_tol=1e-3,
        density_atol=1e-3,
        scf_tol=1e-3,
    )

    model.density_matrix(guess)
    model.density_matrix(guess)

    assert len(calls) == 2
    assert calls[0] is not calls[1]


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_hubbard_low_u_solution_has_negligible_staggered_order():
    zero_temp_validation = _tutorial_zero_temp_validation()

    _, local_density, solver_info, density_info = _solve_low_u_hubbard_solution()

    assert solver_info.residual_norm <= 5e-3
    assert abs(density_info.charge - 2.0) <= 1e-3
    assert zero_temp_validation.staggered_magnetization(local_density) < 5e-4


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_hubbard_low_u_direct_gap_shrinks_with_postprocessing_grid():
    zero_temp_validation = _tutorial_zero_temp_validation()

    h_full, _, _, _ = _solve_low_u_hubbard_solution()

    gap_100 = zero_temp_validation.band_gap(h_full, nk=100)
    gap_200 = zero_temp_validation.band_gap(h_full, nk=200)
    gap_400 = zero_temp_validation.band_gap(h_full, nk=400)

    assert gap_200 < 0.55 * gap_100
    assert gap_400 < 0.55 * gap_200


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_zero_temperature_hubbard_resolved_gap_helper_removes_grid_floor():
    zero_temp_validation = _tutorial_zero_temp_validation()

    h_full, local_density, _, _ = _solve_low_u_hubbard_solution()
    resolved_gap, info = zero_temp_validation.resolved_hubbard_gap(
        h_full,
        U=4.0 / 29.0,
        local_density=local_density,
        nk_initial=100,
        nk_max=400,
    )

    assert info["used_order_parameter_gap"] is True
    assert info["band_gap"] > 1e-2
    assert resolved_gap < 1e-4


def test_zero_temperature_runtime_error_when_native_backend_missing(monkeypatch):
    import meanfi.zero_temp as zero_temp

    monkeypatch.setattr(zero_temp, "_NATIVE_ZERO_TEMP_AVAILABLE", False)
    monkeypatch.setattr(zero_temp, "Geometry", None)

    with pytest.raises(
        RuntimeError, match="requires the native meanfi._zero_temp_native extension"
    ):
        density_matrix_zero_temp(
            _spinful_chain(),
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


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_geometry_root_mesh_is_seam_safe():
    for ndim in (1, 2, 3):
        geometry = Geometry.root(ndim)
        active_ids = geometry.active_simplex_ids()

        assert geometry.n_active == 2**ndim * math.factorial(ndim)
        assert active_ids.shape == (geometry.n_active,)
        for simplex_id in active_ids:
            points = geometry.simplex_points(int(simplex_id))
            for i in range(points.shape[0]):
                for j in range(i + 1, points.shape[0]):
                    delta = np.abs(points[i] - points[j])
                    assert np.all(delta <= 0.5 + 1e-14)


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_geometry_refine_returns_expected_descriptors():
    geometry = Geometry.root(2)
    parent_id = int(geometry.active_simplex_ids()[0])
    parent_vertex_ids = geometry.simplex_vertex_ids(parent_id)
    children = geometry.ensure_children(parent_id)

    (
        refinements,
        parent_ids,
        child_offsets,
        child_ids,
        refined_parent_vertex_ids,
        child_vertex_ids,
        midpoint_ids,
        bisected_edges,
    ) = geometry.refine(np.array([parent_id], dtype=np.int64))

    assert refinements == 1
    assert np.array_equal(parent_ids, np.array([parent_id], dtype=np.int64))
    assert np.array_equal(child_offsets, np.array([0, 2], dtype=np.int64))
    assert np.array_equal(child_ids, children)
    assert np.array_equal(refined_parent_vertex_ids[0], parent_vertex_ids)
    assert child_vertex_ids.shape == (2, parent_vertex_ids.shape[0])
    assert midpoint_ids[0] >= 0
    assert bisected_edges.shape == (1, 2)

    active_ids = geometry.active_simplex_ids()
    assert parent_id not in set(active_ids.tolist())
    assert set(children.tolist()).issubset(set(active_ids.tolist()))
    assert geometry.generation == 1


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_integrator_preview_depth_is_one_and_geometry_active_set_changes_only_after_refine():
    geometry = Geometry.root(1)
    vertex_cache = tb_to_vertex_cache(_spinful_chain())
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0,)], dtype=float),
    )

    assert integrator.preview_depth == 1

    parent_id = int(geometry.active_simplex_ids()[0])
    children = geometry.ensure_children(parent_id)
    assert parent_id in set(geometry.active_simplex_ids().tolist())

    geometry.refine(np.array([parent_id], dtype=np.int64))

    active_ids = set(geometry.active_simplex_ids().tolist())
    assert parent_id not in active_ids
    assert set(children.tolist()).issubset(active_ids)


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_density_preview_error_matches_preview_minus_coarse_vertex_estimates():
    geometry = Geometry.root(1, root_subcells_per_axis=1)
    vertex_cache = tb_to_vertex_cache(_spinful_chain())
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0,), (1,), (-1,)], dtype=float),
    )
    options = DensityIntegrateOptions()
    options.density_atol = 1e9
    options.density_rtol = 0.0
    options.max_subdivisions = 1

    coarse_total, coarse_owner_ids, coarse_owner_estimates, coarse_evals = integrator.evaluate_density(0.2, 0)
    preview_total, preview_owner_ids, preview_owner_estimates, preview_evals = integrator.evaluate_density(0.2, 1)
    result = integrator.integrate_density(0.2, options)

    assert np.array_equal(coarse_owner_ids, preview_owner_ids)
    assert result.error_estimate_available is True
    assert result.subdivisions == 0
    assert np.allclose(result.estimate_array(), preview_total)
    assert np.allclose(
        result.error_vector_array(),
        np.abs(np.asarray(preview_owner_estimates)[0] - np.asarray(coarse_owner_estimates)[0]),
    )
    assert result.evaluator_evals == coarse_evals + preview_evals


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_density_reuses_geometry_vertex_spectra_without_repeating_reduced_point_lookups():
    geometry = Geometry.root(1, root_subcells_per_axis=1)
    vertex_cache = tb_to_vertex_cache(_spinful_chain())
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0,), (1.0,)], dtype=float),
    )

    integrator.evaluate_density(3.0, 0)
    kernel_evals = vertex_cache.n_kernel_evals
    assert vertex_cache.size <= geometry.n_leaf_vertices
    assert integrator.phase_cache_size == geometry.n_leaf_vertices

    integrator.evaluate_density(2.5, 0)
    assert vertex_cache.n_kernel_evals == kernel_evals
    assert integrator.phase_cache_size == geometry.n_leaf_vertices


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_charge_reuses_geometry_vertex_spectra_across_mu_evaluations():
    geometry = Geometry.root(1, root_subcells_per_axis=1)
    vertex_cache = tb_to_vertex_cache(_spinful_chain())
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0,)], dtype=float),
    )

    integrator.evaluate_charge(0.1, 0)
    kernel_evals = vertex_cache.n_kernel_evals
    assert vertex_cache.size <= geometry.n_leaf_vertices

    integrator.evaluate_charge(0.3, 0)
    assert vertex_cache.n_kernel_evals == kernel_evals


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_density_incremental_refine_builds_only_new_children():
    geometry = Geometry.root(1, root_subcells_per_axis=2)
    vertex_cache = tb_to_vertex_cache(_shifted_spinful_chain())
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0,), (1.0,)], dtype=float),
    )
    options = DensityIntegrateOptions()
    options.density_atol = 0.15
    options.density_rtol = 0.0
    options.max_subdivisions = 4
    options.bulk_theta = 0.999

    result = integrator.integrate_density(0.2, options)

    assert result.subdivisions == 2
    assert integrator.leaf_build_count == 6
    assert geometry.n_active == 4
    assert integrator.cached_simplex_value_count == 14


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_native_tight_binding_model_matches_python_kfunc():
    tb = _qiwuzhang()
    native_model = tb_to_tight_binding_model(tb)
    hkfunc = tb_to_kfunc(tb)
    points = np.array(
        [
            [0.0, 0.0],
            [0.3, -0.7],
            [-np.pi / 2.0, np.pi / 3.0],
        ],
        dtype=float,
    )

    assert native_model.ndim == 2
    assert native_model.ndof == 2
    assert native_model.nterms == len(tb)
    assert np.array_equal(
        native_model.keys_array(), np.asarray(list(tb.keys()), dtype=np.int64)
    )
    assert np.allclose(
        native_model.matrices_array(), np.asarray(list(tb.values()), dtype=complex)
    )
    assert np.allclose(native_model.evaluate_point(points[0]), hkfunc(points[0]))
    assert np.allclose(native_model.evaluate_many(points), hkfunc(points))


@pytest.mark.skipif(
    not _NATIVE_ZERO_TEMP_AVAILABLE,
    reason="native zero-temperature backend is unavailable",
)
def test_vertex_cache_invalidates_after_geometry_vertex_evaluations():
    tb = _spinful_chain()
    geometry = Geometry.root(1, root_subcells_per_axis=1)
    vertex_cache = tb_to_vertex_cache(tb)
    integrator = AdaptiveIntegrator(
        geometry,
        vertex_cache,
        np.asarray([(0.0,)], dtype=float),
    )

    integrator.evaluate_charge(0.0, 0)
    assert vertex_cache.size > 0
    assert vertex_cache.n_kernel_evals > 0

    vertex_cache.invalidate()
    assert vertex_cache.generation == 1
    assert vertex_cache.size == 0


def test_positive_temperature_path_does_not_use_zero_temperature_backend(monkeypatch):
    import meanfi.zero_temp as zero_temp

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError(
            "finite-temperature solve should not call zero-temperature backend"
        )

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", fail)
    tb = _spinful_chain()
    rho, _, mu, info = density_matrix(
        tb,
        filling=1.0,
        kT=0.1,
        keys=[(0,)],
        charge_tol=1e-4,
        density_atol=1e-4,
    )

    assert np.isfinite(mu)
    assert abs(info.charge - 1.0) < 1e-4
    assert np.allclose(rho[(0,)], rho[(0,)].conj().T, atol=1e-8)


def test_native_subdivision_limit_uses_minus_one_for_unbounded_mode():
    import meanfi.zero_temp as zero_temp

    assert zero_temp._native_subdivision_limit(None) == -1
    assert zero_temp._native_subdivision_limit(12) == 12
