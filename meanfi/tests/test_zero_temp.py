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
    _ZeroTempGeometryCache,
    density_matrix_at_mu_zero_temp,
    density_matrix_zero_temp,
)


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


def test_zero_temperature_root_mesh_is_seam_safe():
    for ndim in (1, 2, 3):
        mesh = _ZeroTempGeometryCache.root(ndim)

        assert len(mesh.simplices) == 2**ndim * math.factorial(ndim)
        for simplex in mesh.simplices:
            points = mesh.simplex_points(simplex)
            for i in range(points.shape[0]):
                for j in range(i + 1, points.shape[0]):
                    delta = np.abs(points[i] - points[j])
                    assert np.all(delta <= 0.5 + 1e-14)


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


def test_zero_temperature_solver_restarts_from_root_mesh_on_new_run(monkeypatch):
    import meanfi.zero_temp as zero_temp

    calls: list[bool] = []
    original = zero_temp.density_matrix_zero_temp

    def wrapped(*args, **kwargs):
        calls.append(kwargs.get("geometry_cache") is None)
        return original(*args, **kwargs)

    monkeypatch.setattr(zero_temp, "density_matrix_zero_temp", wrapped)

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

    solver(model, guess, max_subdivisions=300)
    first_run_calls = len(calls)
    solver(model, guess, max_subdivisions=300)

    assert calls[0] is True
    assert calls[first_run_calls] is True


def test_zero_temperature_charge_preview_children_persist_in_tree():
    tb = _spinful_chain()
    _, _, _, _, mesh = density_matrix_zero_temp(
        tb,
        filling=1.1,
        keys=[(0,), (1,)],
        charge_tol=1e-4,
        density_atol=1e-2,
        density_rtol=0.0,
        mu_guess=0.0,
        mu_xtol=1e-4,
        max_mu_iterations=64,
        max_subdivisions=120,
    )

    assert len(mesh._simplex_records) > len(mesh.simplices)
    assert any(record.children for record in mesh._simplex_records)
    assert any(
        record.active and record.children
        for record in mesh._simplex_records
    )


def test_zero_temperature_refinement_activates_children_not_parent():
    mesh = _ZeroTempGeometryCache.root(1)
    parent = mesh.simplices[0]
    children = mesh.ensure_children(parent)

    assert mesh._simplex_records[parent].active is True
    assert all(mesh._simplex_records[child].active is False for child in children)

    refinements, descriptors = mesh.refine_with_children([parent])

    assert refinements == 1
    assert descriptors[parent].child_ids == children
    assert descriptors[parent].parent_id == parent
    assert descriptors[parent].new_midpoint_vertex_id is not None
    assert len(descriptors[parent].child_vertex_ids) == 2
    assert mesh._simplex_records[parent].active is False
    assert all(mesh._simplex_records[child].active is True for child in children)
    assert parent not in mesh.simplices
    assert set(children).issubset(set(mesh.simplices))


def test_zero_temperature_density_round_points_remain_transient():
    tb = _qiwuzhang()
    _, _, info, mesh = density_matrix_at_mu_zero_temp(
        tb,
        mu=0.2,
        keys=[(0, 0), (1, 0), (0, 1)],
        density_atol=2e-2,
        density_rtol=0.0,
        max_subdivisions=4000,
    )

    assert info.n_cached_nodes <= len(mesh.vertices)


def test_zero_temperature_geometry_cache_uses_native_runtime_when_available():
    import meanfi.zero_temp as zero_temp

    if not zero_temp._NATIVE_ZERO_TEMP_AVAILABLE:
        pytest.skip("native zero-temperature backend is unavailable")

    mesh = _ZeroTempGeometryCache.root(2)

    assert mesh._native_geometry is not None
    assert mesh._native_frontier is not None
    assert np.array_equal(np.asarray(mesh.simplices, dtype=int), mesh._native_frontier.active_simplex_ids())


def test_zero_temperature_live_backend_keeps_native_frontier_in_sync():
    import meanfi.zero_temp as zero_temp

    if not zero_temp._NATIVE_ZERO_TEMP_AVAILABLE:
        pytest.skip("native zero-temperature backend is unavailable")

    _, _, info, mesh = density_matrix_at_mu_zero_temp(
        _spinful_chain(),
        mu=0.0,
        keys=[(0,), (1,)],
        density_atol=1e-3,
        density_rtol=0.0,
        max_subdivisions=100,
    )

    assert info.subdivisions > 0
    assert mesh._native_geometry is not None
    assert mesh._native_frontier is not None
    assert np.array_equal(np.asarray(mesh.simplices, dtype=int), mesh._native_frontier.active_simplex_ids())
    assert mesh._native_frontier.n_active == len(mesh.simplices)


def test_native_geometry_root_mesh_is_seam_safe():
    from meanfi._zero_temp_native import NativeGeometry

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


def test_native_geometry_refine_returns_expected_descriptors():
    from meanfi._zero_temp_native import NativeGeometry

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


def test_native_frontier_apply_refinement_matches_geometry_frontier():
    from meanfi._zero_temp_native import NativeFrontier, NativeGeometry

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
    assert np.array_equal(frontier.active_simplex_ids()[2:], original_active[1:])


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


def test_zero_temp_internal_spectral_cache_uses_native_backend():
    import meanfi.zero_temp as zero_temp

    if not zero_temp._NATIVE_ZERO_TEMP_AVAILABLE:
        pytest.skip("native zero-temperature backend is unavailable")

    spectral_cache = zero_temp._SpectralCache(_spinful_chain())
    assert spectral_cache._native_cache is not None

    points = np.array([[0.5], [0.5]], dtype=float)
    values = spectral_cache.get_many_values(points)

    assert values.shape == (2, 2)
    assert spectral_cache.n_kernel_evals == 1
    assert spectral_cache.n_cached_points == 1


def test_zero_temp_native_charge_evaluator_matches_python_path(monkeypatch):
    import meanfi.zero_temp as zero_temp

    if not zero_temp._NATIVE_ZERO_TEMP_AVAILABLE or zero_temp._NativeChargeEvaluator is None:
        pytest.skip("native zero-temperature charge backend is unavailable")

    tb = _spinful_chain()
    mesh = zero_temp._ZeroTempGeometryCache.root(1)
    spectral_cache = zero_temp._SpectralCache(tb)
    mu = 0.3

    native_counters = zero_temp._StageCounters()
    native_preparation = zero_temp._ChargePreparationStore(mesh, spectral_cache)
    native_evaluator = zero_temp._ChargeEvaluator(native_preparation, 1e-8, native_counters, refine_levels=2)
    assert native_evaluator._native_evaluator is not None

    native_summary = native_evaluator.evaluate(mu)
    native_owner_ids, native_owner_charges = native_evaluator.owner_charges(mu)

    monkeypatch.setattr(zero_temp, "_NativeChargeEvaluator", None)
    python_counters = zero_temp._StageCounters()
    python_preparation = zero_temp._ChargePreparationStore(mesh, spectral_cache)
    python_evaluator = zero_temp._ChargeEvaluator(python_preparation, 1e-8, python_counters, refine_levels=2)
    assert python_evaluator._native_evaluator is None

    python_summary = python_evaluator.evaluate(mu)
    python_owner_ids, python_owner_charges = python_evaluator.owner_charges(mu)

    assert np.allclose(native_summary.charge, python_summary.charge, atol=1e-12)
    assert np.array_equal(native_owner_ids, python_owner_ids)
    assert np.allclose(native_owner_charges, python_owner_charges, atol=1e-12)


def test_zero_temp_native_density_evaluator_matches_python_path(monkeypatch):
    import meanfi.zero_temp as zero_temp

    if not zero_temp._NATIVE_ZERO_TEMP_AVAILABLE or zero_temp._NativeDensityEvaluator is None:
        pytest.skip("native zero-temperature density backend is unavailable")

    tb = _spinful_chain()
    kwargs = dict(
        mu=0.3,
        kT=0.0,
        keys=[(0,), (1,)],
        density_atol=1e-4,
        max_subdivisions=200,
    )

    rho_native, err_native, info_native = density_matrix_at_mu(tb, **kwargs)

    monkeypatch.setattr(zero_temp, "_NativeDensityEvaluator", None)
    rho_python, err_python, info_python = density_matrix_at_mu(tb, **kwargs)

    for key in kwargs["keys"]:
        assert np.allclose(rho_native[key], rho_python[key], atol=1e-10)
        assert np.allclose(err_native[key], err_python[key], atol=1e-10)
    assert info_native.n_leaves == info_python.n_leaves


def test_zero_temperature_native_and_python_fallback_match(monkeypatch):
    import meanfi.zero_temp as zero_temp

    tb = _spinful_chain()
    kwargs = dict(
        mu=0.3,
        kT=0.0,
        keys=[(0,), (1,)],
        density_atol=1e-4,
        max_subdivisions=200,
    )

    rho_native, err_native, info_native = density_matrix_at_mu(tb, **kwargs)

    monkeypatch.setattr(zero_temp, "_native_point_key_bytes", None)
    monkeypatch.setattr(zero_temp, "_native_accumulate_density_terms", None)
    monkeypatch.setattr(zero_temp, "_native_density_tables_from_eigenvectors", None)
    monkeypatch.setattr(zero_temp, "_native_prepare_charge_batch_metadata", None)
    monkeypatch.setattr(zero_temp, "_native_prepare_density_cells_metadata", None)
    monkeypatch.setattr(zero_temp, "_native_unique_first_indices_int64", None)
    monkeypatch.setattr(zero_temp, "_NATIVE_ZERO_TEMP_AVAILABLE", False)

    rho_python, err_python, info_python = density_matrix_at_mu(tb, **kwargs)

    for key in kwargs["keys"]:
        assert np.allclose(rho_native[key], rho_python[key], atol=1e-10)
        assert np.allclose(err_native[key], err_python[key], atol=1e-10)
    assert info_native.n_leaves == info_python.n_leaves


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
