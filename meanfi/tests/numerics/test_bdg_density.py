import numpy as np
import pytest
import scipy.sparse as sparse

import meanfi.density.integrate.periodic_grid as periodic

from meanfi import (
    DirectDiagonalization,
    Model,
    RationalFOE,
    UniformGrid,
    tb_to_kfunc,
)
from meanfi.density.filling import charge_diagonal
from meanfi import density_matrix
from meanfi.tb.bdg import assemble_bdg_tb


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]


def _square_lattice_2d(t: float = 0.25):
    return {
        (0, 0): 0.1 * np.eye(2, dtype=complex),
        (1, 0): -t * np.eye(2, dtype=complex),
        (-1, 0): -t * np.eye(2, dtype=complex),
        (0, 1): -t * np.eye(2, dtype=complex),
        (0, -1): -t * np.eye(2, dtype=complex),
    }


def _interaction_2d():
    return {(0, 0): np.ones((2, 2), dtype=complex)}


def _pairing(delta: float, *, sparse=None):
    normal = {(0, 0): np.zeros((2, 2), dtype=complex)}
    anomalous = {(0, 0): np.array([[0.0, delta], [-delta, 0.0]], dtype=complex)}
    pairing = assemble_bdg_tb(normal, anomalous, ndof=2)
    if sparse is None:
        return pairing
    return {key: sparse.csr_matrix(matrix) for key, matrix in pairing.items()}


def _sparsify_tb(tb, sparse):
    return {
        key: sparse.csr_matrix(np.asarray(matrix, dtype=complex))
        for key, matrix in tb.items()
    }


def _bdg_reference(model: Model, meanfield, keys, *, nk: int):
    hkfunc = tb_to_kfunc(model.hamiltonian_from_meanfield(meanfield))
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


def test_bdg_exact_density_matches_dense_2d_reference():
    keys = [(0, 0), (1, 0), (0, 1)]
    meanfield = _pairing(0.3)
    model = Model(
        _square_lattice_2d(),
        _interaction_2d(),
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

    result = density_matrix(
        model,
        mean_field=meanfield,
        keys=keys,
        integration=UniformGrid(nk=256, matrix_function=DirectDiagonalization()),
        filling_tol=0.0005,
        mu_tol=0.0005,
        max_charge_evaluations=80,
    )

    assert abs(result.mu - reference_mu) <= 8e-4
    assert abs(result.filling - reference_filling) <= 8e-4
    assert _max_density_error(result.to_tb(), reference_density) <= 8e-4


@pytest.mark.parametrize(
    "matrix_function",
    [
        RationalFOE(initial_poles=4, max_poles=256),
    ],
    ids=["default-aaa"],
)
def test_bdg_dense_rational_is_rejected(matrix_function):
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25)
    model = Model(
        _square_lattice_2d(t=0.15),
        _interaction_2d(),
        filling=0.6,
        kT=0.5,
        superconducting=True,
    )
    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        density_matrix(
            model,
            mean_field=meanfield,
            keys=keys,
            integration=UniformGrid(nk=256, matrix_function=matrix_function),
            filling_tol=0.001,
            mu_tol=0.001,
            max_charge_evaluations=80,
        )


@pytest.mark.parametrize(
    "matrix_function",
    [
        None,
        RationalFOE(initial_poles=4, max_poles=256),
    ],
    ids=["default-sparse-aaa", "explicit-aaa"],
)
@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_rational_matches_exact_density_in_2d(matrix_function):
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25, sparse=sparse)
    model = Model(
        _sparsify_tb(_square_lattice_2d(t=0.15), sparse),
        {local_key: sparse.csr_matrix(np.ones((2, 2))) for local_key in [(0, 0)]},
        filling=0.6,
        kT=0.5,
        superconducting=True,
    )
    exact = density_matrix(
        model,
        mean_field=meanfield,
        keys=keys,
        integration=UniformGrid(nk=256, matrix_function=DirectDiagonalization()),
        filling_tol=0.001,
        mu_tol=0.001,
        max_charge_evaluations=80,
    )
    rational = density_matrix(
        model,
        mean_field=meanfield,
        keys=keys,
        integration=UniformGrid(nk=256, matrix_function=matrix_function),
        filling_tol=0.001,
        mu_tol=0.001,
        max_charge_evaluations=80,
    )

    assert abs(rational.mu - exact.mu) <= 2e-3
    assert abs(rational.filling - exact.filling) <= 2e-3
    assert _max_density_error(rational.to_tb(), exact.to_tb()) <= 2e-3


@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_rational_accepts_sparse_matrices_when_scipy_is_available():
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    h_int = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    result = density_matrix(
        model,
        mean_field=meanfield,
        keys=[local],
        integration=UniformGrid(nk=256),
        filling_tol=1e-06,
        mu_tol=1e-08,
        max_charge_evaluations=40,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 1.0) <= 1e-6
    assert np.allclose(result.to_tb()[local], 0.5 * np.eye(4), atol=1e-6)


@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_rational_does_not_fallback_to_exact_diagonalization(monkeypatch):
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    h_int = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    def fail_if_exact(*args, **kwargs):
        raise AssertionError(
            "Sparse Rational path should not call exact diagonalization"
        )

    monkeypatch.setattr(np.linalg, "eigh", fail_if_exact)
    monkeypatch.setattr(np.linalg, "eigvalsh", fail_if_exact)
    result = density_matrix(
        model,
        mean_field=meanfield,
        keys=[local],
        integration=UniformGrid(nk=256),
        filling_tol=1e-06,
        mu_tol=1e-08,
        max_charge_evaluations=40,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 1.0) <= 1e-6


@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_rational_density_path_avoids_dense_conversion(monkeypatch):
    local = (0, 0)
    h_0 = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    h_int = {local: sparse.csr_matrix(np.zeros((2, 2)))}
    meanfield = _pairing(0.0, sparse=sparse)
    model = Model(
        h_0,
        h_int,
        filling=1.0,
        kT=0.2,
        superconducting=True,
    )

    def fail_if_dense(*args, **kwargs):
        raise AssertionError("Sparse Rational density path should not densify matrices")

    monkeypatch.setattr(periodic, "to_dense", fail_if_dense)
    result = density_matrix(
        model,
        mean_field=meanfield,
        keys=[local],
        integration=UniformGrid(nk=256),
        filling_tol=1e-06,
        mu_tol=1e-08,
        max_charge_evaluations=40,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 1.0) <= 1e-6


def test_bdg_zero_dimensional_rational_density_rejects_dense_matrix():
    local = (0, 0)
    model = Model(
        {local: np.zeros((2, 2), dtype=complex)},
        {local: np.zeros((2, 2), dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)
    with pytest.raises(ValueError, match="RationalFOE is supported only for sparse"):
        density_matrix(
            model,
            mean_field=meanfield,
            keys=[local],
            integration=UniformGrid(
                nk=256, matrix_function=RationalFOE(initial_poles=4, max_poles=256)
            ),
            filling_tol=1e-08,
            mu_tol=1e-10,
            max_charge_evaluations=40,
        )


@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_selected_density_matches_dense_reference():
    local = (0, 0)
    dense_h0 = {local: np.zeros((2, 2), dtype=complex)}
    dense_hint = _interaction_2d()
    meanfield = _pairing(0.05)
    sparse_h0 = {local: sparse.csr_matrix(dense_h0[local])}
    sparse_hint = {local: sparse.csr_matrix(dense_hint[local])}
    sparse_meanfield = _pairing(0.05, sparse=sparse)

    dense_result = density_matrix(
        Model(dense_h0, dense_hint, filling=0.5, kT=0.2, superconducting=True),
        mean_field=meanfield,
        keys=[local],
        integration=UniformGrid(nk=256),
        filling_tol=0.001,
        mu_tol=1e-08,
        max_charge_evaluations=40,
    )
    sparse_result = density_matrix(
        Model(sparse_h0, sparse_hint, filling=0.5, kT=0.2, superconducting=True),
        mean_field=sparse_meanfield,
        keys=[local],
        integration=UniformGrid(nk=256),
        filling_tol=0.001,
        mu_tol=1e-08,
        max_charge_evaluations=40,
    )

    space = Model(
        dense_h0,
        dense_hint,
        filling=0.5,
        kT=0.2,
        superconducting=True,
    ).scf_space
    np.testing.assert_allclose(
        space.params_from_meanfield_input(dense_result.to_tb()),
        space.params_from_meanfield_input(sparse_result.to_tb()),
        atol=1e-3,
    )


@pytest.mark.parametrize(
    "matrix_function",
    [
        None,
        RationalFOE(initial_poles=4, max_poles=128),
    ],
    ids=["default-sparse-aaa", "explicit-aaa"],
)
@pytest.mark.usefixtures("require_mumps")
def test_bdg_sparse_periodic_grid_selected_density_matches_dense_reference(
    matrix_function,
):
    local = (0, 0)
    dense_h0 = {local: np.zeros((2, 2), dtype=complex)}
    dense_hint = _interaction_2d()
    meanfield = _pairing(0.05)
    sparse_h0 = {local: sparse.csr_matrix(dense_h0[local])}
    sparse_hint = {local: sparse.csr_matrix(dense_hint[local])}
    sparse_meanfield = _pairing(0.05, sparse=sparse)

    dense_model = Model(dense_h0, dense_hint, filling=0.5, kT=0.2, superconducting=True)
    sparse_model = Model(
        sparse_h0, sparse_hint, filling=0.5, kT=0.2, superconducting=True
    )
    space = dense_model.scf_space

    dense_result = density_matrix(
        dense_model,
        mean_field=meanfield,
        keys=[local],
        integration=UniformGrid(nk=25, matrix_function=DirectDiagonalization()),
        filling_tol=1e-08,
        mu_tol=1e-10,
        max_charge_evaluations=80,
    )
    sparse_result = density_matrix(
        sparse_model,
        mean_field=sparse_meanfield,
        integration=UniformGrid(nk=25, matrix_function=matrix_function),
        filling_tol=0.001,
        mu_tol=1e-08,
        max_charge_evaluations=80,
        coordinates=space.required_coordinates,
    )

    np.testing.assert_allclose(
        space.required_coordinates.values_from_tb(dense_result.to_tb()),
        sparse_result.values,
        atol=2e-3,
    )
