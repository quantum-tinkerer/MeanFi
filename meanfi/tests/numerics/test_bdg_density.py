import numpy as np
import pytest
import scipy.sparse as sparse
import warnings

import meanfi.density.kpoint.matrix_functions.direct as bdg_matrix_direct
import meanfi.density.integrate.quadrature.bdg as bdg_quadrature
from meanfi import (
    AdaptiveQuadrature,
    DirectDiagonalization,
    Model,
    RationalFOE,
    UniformGrid,
    tb_to_kfunc,
)
from meanfi.space.particlehole import bdg_top_half_selection
from meanfi.space.bdg import bdg_density_to_rparams
from meanfi.density.filling import charge_diagonal
from meanfi.density.integrate.bdg import solve_bdg_density_fixed_filling


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]


def _square_lattice_2d(t: float = 0.25):
    return {
        (0, 0): np.array([[0.1]], dtype=complex),
        (1, 0): np.array([[-t]], dtype=complex),
        (-1, 0): np.array([[-t]], dtype=complex),
        (0, 1): np.array([[-t]], dtype=complex),
        (0, -1): np.array([[-t]], dtype=complex),
    }


def _pairing(delta: float, *, sparse=None):
    matrix = np.array([[0.0, delta], [delta, 0.0]], dtype=complex)
    if sparse is not None:
        matrix = sparse.csr_matrix(matrix)
    return {(0, 0): matrix}


def _sparsify_tb(tb, sparse):
    return {
        key: sparse.csr_matrix(np.asarray(matrix, dtype=complex))
        for key, matrix in tb.items()
    }


def _bdg_reference(model: Model, meanfield, keys, *, nk: int):
    hkfunc = tb_to_kfunc(model.bdg_hamiltonian_from_meanfield(meanfield))
    q_matrix = np.diag(charge_diagonal(model._ndof))
    axis = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    kx, ky = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack([kx.ravel(), ky.ravel()], axis=-1)
    hamiltonians = hkfunc(points)

    def density_at_mu(mu: float):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians - mu * q_matrix)
        occupation = 1.0 / (np.exp(eigenvalues / model.kT) + 1.0)
        return (
            eigenvectors
            * occupation[..., np.newaxis, :]
            @ eigenvectors.conj().swapaxes(-1, -2)
        )

    def charge(mu: float) -> float:
        density_k = density_at_mu(mu)
        electron_block = density_k[:, : model._ndof, : model._ndof]
        return float(np.mean(np.trace(electron_block, axis1=1, axis2=2).real))

    lower, upper = -4.0, 4.0
    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if charge(midpoint) < model.filling:
            lower = midpoint
        else:
            upper = midpoint
    mu = 0.5 * (lower + upper)
    density_k = density_at_mu(mu)

    rho = {}
    for key in keys:
        phase = np.exp(1j * np.dot(points, np.asarray(key, dtype=float)))
        rho[key] = np.einsum("k,kab->ab", phase / points.shape[0], density_k)
    return mu, charge(mu), rho


def _max_density_error(lhs, rhs) -> float:
    return max(float(np.max(np.abs(lhs[key] - rhs[key]))) for key in rhs)


def test_bdg_charge_evaluator_allows_negative_local_derivative_contributions(
    monkeypatch,
):
    def fake_density_block(*args, **kwargs):
        del args, kwargs
        return type(
            "DensityBlockResult",
            (),
            {
                "block": np.array([[0.25 + 0.0j]], dtype=complex),
                "derivative_block": np.array([[-0.5 + 0.0j]], dtype=complex),
            },
        )()

    monkeypatch.setattr(bdg_quadrature, "density_block", fake_density_block)
    evaluator = bdg_quadrature._charge_evaluator(
        ndim=1,
        kT=0.2,
        q_diag=np.array([1.0]),
        matrix_function=DirectDiagonalization(),
        tolerance=1e-3,
        filling_indices=[0],
        filling_weights=np.array([1.0]),
        matrix_from_payload=lambda payload: np.asarray([[payload[0]]], dtype=complex),
        workspace_dtype=np.dtype(np.complex128),
    )

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        values = evaluator(
            np.array([[0.0], [np.pi]], dtype=float),
            np.array([[0.0], [0.0]], dtype=float),
            0.1,
        )

    assert len(record) == 0
    assert np.all(values[:, 1] < 0.0)


def test_bdg_split_charge_warns_only_for_nonpositive_integrated_derivative():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        charge, charge_error, derivative = bdg_quadrature._split_charge(
            np.array([0.4, 0.25]),
            np.array([1e-3, 2e-3]),
        )

    assert len(record) == 0
    assert charge == pytest.approx(0.4)
    assert charge_error == pytest.approx(1e-3)
    assert derivative == pytest.approx(0.25)

    with pytest.warns(RuntimeWarning, match="integrated dN/dmu was non-positive"):
        charge, charge_error, derivative = bdg_quadrature._split_charge(
            np.array([0.4, -0.25]),
            np.array([1e-3, 2e-3]),
        )

    assert charge == pytest.approx(0.4)
    assert charge_error == pytest.approx(1e-3)
    assert derivative == 0.0

    with pytest.warns(RuntimeWarning, match="integrated dN/dmu was non-positive"):
        _charge, _charge_error, derivative = bdg_quadrature._split_charge(
            np.array([0.4, np.nan]),
            np.array([1e-3, 2e-3]),
        )

    assert derivative == 0.0


def test_bdg_exact_density_matches_dense_2d_reference():
    keys = [(0, 0), (1, 0), (0, 1)]
    meanfield = _pairing(0.3)
    model = Model(
        _square_lattice_2d(),
        {(0, 0): np.array([[1.0]], dtype=complex)},
        filling=0.6,
        kT=0.35,
        superconducting=True,
    )
    reference_mu, reference_filling, reference_density = _bdg_reference(
        model,
        meanfield,
        keys,
        nk=121,
    )

    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=5e-4,
            max_refinements=100,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=5e-4,
        mu_tol=5e-4,
        max_charge_evaluations=80,
        mu_guess=0.0,
    )

    assert abs(result.mu - reference_mu) <= 8e-4
    assert abs(result.filling - reference_filling) <= 8e-4
    assert _max_density_error(result.density_matrix, reference_density) <= 8e-4


@pytest.mark.parametrize(
    "matrix_function",
    [
        RationalFOE(initial_poles=4, max_poles=256, rational_scheme="ozaki"),
        RationalFOE(initial_poles=4, max_poles=256),
    ],
    ids=["ozaki", "default-ozaki"],
)
def test_bdg_dense_rational_is_rejected(matrix_function):
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25)
    model = Model(
        _square_lattice_2d(t=0.15),
        {(0, 0): np.array([[1.0]], dtype=complex)},
        filling=0.6,
        kT=0.5,
        superconducting=True,
    )
    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        solve_bdg_density_fixed_filling(
            model,
            meanfield,
            keys=keys,
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-3,
                max_refinements=40,
                matrix_function=matrix_function,
            ),
            filling_tol=1e-3,
            mu_tol=1e-3,
            max_charge_evaluations=80,
            mu_guess=0.0,
        )


@pytest.mark.parametrize(
    "matrix_function",
    [
        None,
        RationalFOE(initial_poles=4, max_poles=256, rational_scheme="aaa"),
        RationalFOE(initial_poles=4, max_poles=256, rational_scheme="ozaki"),
    ],
    ids=["default-sparse-aaa", "explicit-aaa", "explicit-ozaki"],
)
def test_bdg_sparse_rational_matches_exact_density_in_2d(matrix_function):
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25, sparse=sparse)
    model = Model(
        _sparsify_tb(_square_lattice_2d(t=0.15), sparse),
        {local_key: sparse.csr_matrix([[1.0]]) for local_key in [(0, 0)]},
        filling=0.6,
        kT=0.5,
        superconducting=True,
    )
    exact = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_charge_evaluations=80,
        mu_guess=0.0,
    )
    rational = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_charge_evaluations=80,
        mu_guess=0.0,
    )

    assert abs(rational.mu - exact.mu) <= 2e-3
    assert abs(rational.filling - exact.filling) <= 2e-3
    assert _max_density_error(rational.density_matrix, exact.density_matrix) <= 2e-3


def test_bdg_sparse_rational_accepts_sparse_matrices_when_scipy_is_available():
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix([[0.0]])}
    h_int = {local: sparse.csr_matrix([[0.0]])}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6
    assert np.allclose(result.density_matrix[local], 0.5 * np.eye(2), atol=1e-6)


def test_bdg_sparse_rational_does_not_fallback_to_exact_diagonalization(monkeypatch):
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix([[0.0]])}
    h_int = {local: sparse.csr_matrix([[0.0]])}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    def fail_if_exact(*args, **kwargs):
        raise AssertionError(
            "Sparse Rational path should not call exact diagonalization"
        )

    monkeypatch.setattr(bdg_matrix_direct, "_exact_density_block", fail_if_exact)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_sparse_rational_density_path_avoids_dense_conversion(monkeypatch):
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix([[0.0]])}
    h_int = {local: sparse.csr_matrix([[0.0]])}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )

    def fail_if_dense(*args, **kwargs):
        raise AssertionError("Sparse Rational density path should not densify matrices")

    monkeypatch.setattr(bdg_matrix_direct, "to_dense", fail_if_dense)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_charge_evaluations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_zero_dimensional_rational_density_rejects_dense_matrix():
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)
    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        solve_bdg_density_fixed_filling(
            model,
            meanfield,
            keys=[local],
            integration=AdaptiveQuadrature(
                density_matrix_tol=1e-8,
                max_refinements=8,
                matrix_function=RationalFOE(initial_poles=4, max_poles=256),
            ),
            filling_tol=1e-8,
            mu_tol=1e-10,
            max_charge_evaluations=40,
            mu_guess=0.0,
        )


def test_bdg_sparse_selected_density_matches_dense_reference():
    local = (0, 0)
    dense_h0 = {local: np.array([[0.0]], dtype=complex)}
    dense_hint = {local: np.array([[1.0]], dtype=complex)}
    meanfield = _pairing(0.05)
    sparse_h0 = {local: sparse.csr_matrix(dense_h0[local])}
    sparse_hint = {local: sparse.csr_matrix(dense_hint[local])}
    sparse_meanfield = _pairing(0.05, sparse=sparse)

    dense_result = solve_bdg_density_fixed_filling(
        Model(dense_h0, dense_hint, filling=0.5, kT=0.2, superconducting=True),
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=16,
        ),
        filling_tol=1e-3,
        mu_tol=1e-8,
        max_charge_evaluations=40,
        mu_guess=0.0,
    )
    sparse_result = solve_bdg_density_fixed_filling(
        Model(sparse_h0, sparse_hint, filling=0.5, kT=0.2, superconducting=True),
        sparse_meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=16,
        ),
        filling_tol=1e-3,
        mu_tol=1e-8,
        max_charge_evaluations=40,
        mu_guess=0.0,
    )

    selection = bdg_top_half_selection(
        keys=[local],
        local_key=local,
        interaction_tb=dense_hint,
        ndof=1,
    )
    np.testing.assert_allclose(
        bdg_density_to_rparams(dense_result.density_matrix, ndof=1, selection=selection),
        bdg_density_to_rparams(sparse_result.density_matrix, ndof=1, selection=selection),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "matrix_function",
    [
        None,
        RationalFOE(initial_poles=4, max_poles=128, rational_scheme="aaa"),
        RationalFOE(initial_poles=4, max_poles=128, rational_scheme="ozaki"),
    ],
    ids=["default-sparse-aaa", "explicit-aaa", "explicit-ozaki"],
)
def test_bdg_sparse_uniform_grid_selected_density_matches_dense_reference(
    matrix_function,
):
    local = (0, 0)
    dense_h0 = {local: np.array([[0.0]], dtype=complex)}
    dense_hint = {local: np.array([[1.0]], dtype=complex)}
    meanfield = _pairing(0.05)
    sparse_h0 = {local: sparse.csr_matrix(dense_h0[local])}
    sparse_hint = {local: sparse.csr_matrix(dense_hint[local])}
    sparse_meanfield = _pairing(0.05, sparse=sparse)

    dense_model = Model(dense_h0, dense_hint, filling=0.5, kT=0.2, superconducting=True)
    sparse_model = Model(
        sparse_h0, sparse_hint, filling=0.5, kT=0.2, superconducting=True
    )
    selection = bdg_top_half_selection(
        keys=[local],
        local_key=local,
        interaction_tb=dense_hint,
        ndof=1,
    )

    dense_result = solve_bdg_density_fixed_filling(
        dense_model,
        meanfield,
        keys=[local],
        integration=UniformGrid(
            nk=25,
            density_matrix_tol=1e-8,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
        max_charge_evaluations=80,
        mu_guess=0.0,
    )
    sparse_result = solve_bdg_density_fixed_filling(
        sparse_model,
        sparse_meanfield,
        keys=[local],
        integration=UniformGrid(
            nk=25,
            density_matrix_tol=1e-3,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-3,
        mu_tol=1e-8,
        max_charge_evaluations=80,
        mu_guess=0.0,
        density_selection=None,
    )

    np.testing.assert_allclose(
        bdg_density_to_rparams(dense_result.density_matrix, ndof=1, selection=selection),
        bdg_density_to_rparams(sparse_result.density_matrix, ndof=1, selection=selection),
        atol=2e-3,
    )
