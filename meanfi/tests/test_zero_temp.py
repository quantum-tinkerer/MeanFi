import math

import numpy as np
import pytest
from scipy.optimize import brentq

from meanfi import (
    Model,
    density_matrix,
    density_matrix_at_mu,
    fermi_dirac,
    solver,
    tb_to_kfunc,
    tb_to_native_model,
    tb_to_native_spectral_cache,
)
from meanfi.zero_temp import (
    _NATIVE_ZERO_TEMP_AVAILABLE,
    density_matrix_zero_temp,
)

if _NATIVE_ZERO_TEMP_AVAILABLE:
    from meanfi._zero_temp_native import (
        NativeChargeEvaluator,
        NativeFrontier,
        NativeGeometry,
    )
else:  # pragma: no cover - only exercised when native extension is unavailable
    NativeChargeEvaluator = None
    NativeFrontier = None
    NativeGeometry = None


def _spinful_chain():
    hopping = -np.eye(2)
    return {(0,): np.zeros((2, 2)), (1,): hopping, (-1,): hopping.conj().T}


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
        eigenvectors * occupation[:, np.newaxis, :] @ eigenvectors.conj().transpose(0, 2, 1)
    )
    rho = {}
    for key in keys:
        phase = np.exp(1j * np.dot(points, np.array(key, dtype=float)))
        rho[key] = np.einsum("k,kab->ab", phase / points.shape[0], density_matrix_k)
    return rho


def _assert_density_close(rho, rho_ref, *, atol: float):
    for key in rho_ref:
        assert np.allclose(rho[key], rho_ref[key], atol=atol)


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
            charge_tol=1e-4,
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
        density_atol=2e-2,
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
        density_atol=2e-2,
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

    assert info.subdivisions > 0
    _assert_density_close(rho, rho_ref, atol=3e-2)


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
        density_atol=1e-2,
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

    solution, info = solver(model, guess, return_info=True, max_subdivisions=300)

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

    model.density_matrix(guess, max_subdivisions=200)
    model.density_matrix(guess, max_subdivisions=200)

    assert len(calls) == 2
    assert calls[0] is not calls[1]


def test_zero_temperature_runtime_error_when_native_backend_missing(monkeypatch):
    import meanfi.zero_temp as zero_temp

    monkeypatch.setattr(zero_temp, "_NATIVE_ZERO_TEMP_AVAILABLE", False)
    monkeypatch.setattr(zero_temp, "NativeGeometry", None)

    with pytest.raises(RuntimeError, match="requires the native meanfi._zero_temp_native extension"):
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


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_geometry_root_mesh_is_seam_safe():
    for ndim in (1, 2, 3):
        geometry = NativeGeometry.root(ndim)
        active_ids = geometry.active_simplex_ids()

        assert geometry.n_active == 2**ndim * math.factorial(ndim)
        assert active_ids.shape == (geometry.n_active,)
        for simplex_id in active_ids:
            points = geometry.simplex_points(int(simplex_id))
            for i in range(points.shape[0]):
                for j in range(i + 1, points.shape[0]):
                    delta = np.abs(points[i] - points[j])
                    assert np.all(delta <= 0.5 + 1e-14)


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_geometry_refine_returns_expected_descriptors():
    geometry = NativeGeometry.root(2)
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


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_frontier_apply_refinement_matches_geometry_frontier():
    geometry = NativeGeometry.root(1)
    frontier = NativeFrontier.from_geometry(geometry)
    original_active = frontier.active_simplex_ids().copy()
    parent_id = int(original_active[0])

    (
        refinements,
        parent_ids,
        child_offsets,
        child_ids,
        _parent_vertex_ids,
        _child_vertex_ids,
        _midpoint_ids,
        _bisected_edges,
    ) = geometry.refine(np.array([parent_id], dtype=np.int64))
    assert refinements == 1

    frontier.apply_refinement(parent_ids, child_offsets, child_ids)

    expected_active = geometry.active_simplex_ids()
    assert np.array_equal(frontier.active_simplex_ids(), expected_active)
    assert frontier.n_active == geometry.n_active
    assert frontier.generation == 1
    assert parent_id not in set(frontier.active_simplex_ids().tolist())


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_charge_preview_depth_is_one_and_children_replace_only_after_refine():
    geometry = NativeGeometry.root(1)
    frontier = NativeFrontier.from_geometry(geometry)
    spectral_cache = tb_to_native_spectral_cache(_spinful_chain())
    evaluator = NativeChargeEvaluator(geometry, spectral_cache)

    assert evaluator.preview_depth == 1

    parent_id = int(frontier.active_simplex_ids()[0])
    children = geometry.ensure_children(parent_id)
    assert parent_id in set(frontier.active_simplex_ids().tolist())

    (
        refinements,
        parent_ids,
        child_offsets,
        child_ids,
        _parent_vertex_ids,
        _child_vertex_ids,
        _midpoint_ids,
        _bisected_edges,
    ) = geometry.refine(np.array([parent_id], dtype=np.int64))
    assert refinements == 1
    frontier.apply_refinement(parent_ids, child_offsets, child_ids)

    active_ids = set(frontier.active_simplex_ids().tolist())
    assert parent_id not in active_ids
    assert set(children.tolist()).issubset(active_ids)


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_tight_binding_model_matches_python_kfunc():
    tb = _qiwuzhang()
    native_model = tb_to_native_model(tb)
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
    assert np.array_equal(native_model.keys_array(), np.asarray(list(tb.keys()), dtype=np.int64))
    assert np.allclose(native_model.matrices_array(), np.asarray(list(tb.values()), dtype=complex))
    assert np.allclose(native_model.evaluate_point(points[0]), hkfunc(points[0]))
    assert np.allclose(native_model.evaluate_many(points), hkfunc(points))


@pytest.mark.skipif(not _NATIVE_ZERO_TEMP_AVAILABLE, reason="native zero-temperature backend is unavailable")
def test_native_spectral_cache_matches_numpy_eigh_and_invalidates():
    tb = _spinful_chain()
    spectral_cache = tb_to_native_spectral_cache(tb)
    hkfunc = tb_to_kfunc(tb)
    points = np.array(
        [
            [0.0],
            [0.4],
            [-1.1],
            [0.4],
        ],
        dtype=float,
    )

    values, vectors = spectral_cache.get_many(points)
    reference_h = hkfunc(points)
    reference_values, _reference_vectors = np.linalg.eigh(reference_h)
    reconstructed_h = np.einsum("...ib,...b,...jb->...ij", vectors, values, vectors.conj())

    assert np.allclose(values, reference_values)
    assert np.allclose(reconstructed_h, reference_h)
    assert spectral_cache.size == 3
    assert spectral_cache.n_kernel_evals == 3

    values_repeat, vectors_repeat = spectral_cache.get_many(points)
    assert np.allclose(values_repeat, values)
    assert np.allclose(vectors_repeat, vectors)
    assert spectral_cache.size == 3
    assert spectral_cache.n_kernel_evals == 3

    spectral_cache.invalidate()
    assert spectral_cache.generation == 1
    assert spectral_cache.size == 0


def test_positive_temperature_path_does_not_use_zero_temperature_backend(monkeypatch):
    import meanfi.zero_temp as zero_temp

    def fail(*args, **kwargs):  # pragma: no cover - executed only on regression
        raise AssertionError("finite-temperature solve should not call zero-temperature backend")

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
