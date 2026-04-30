import numpy as np
import pytest
from meanfi.integrate.density_support import bdg_top_half_support
import meanfi.integrate.matrix_functions.direct as bdg_matrix_direct
from meanfi.params.rparams import bdg_density_to_rparams
from meanfi import (
    AdaptiveQuadrature,
    ChebyshevFOE,
    DirectDiagonalization,
    LinearMixing,
    Model,
    RationalFOE,
    solver,
    tb_to_kfunc,
)
from meanfi.superconducting.bdg import charge_diagonal
from meanfi.superconducting.density import solve_bdg_density_fixed_filling


pytestmark = [pytest.mark.numerics, pytest.mark.perf_slow]


def _square_lattice_2d(t: float = 0.25):
    return {
        (0, 0): np.array([[0.1]], dtype=complex),
        (1, 0): np.array([[-t]], dtype=complex),
        (-1, 0): np.array([[-t]], dtype=complex),
        (0, 1): np.array([[-t]], dtype=complex),
        (0, -1): np.array([[-t]], dtype=complex),
    }


def _triangular_lattice_2d(t: float = 0.2):
    return {
        (0, 0): np.array([[0.0]], dtype=complex),
        (1, 0): np.array([[-t]], dtype=complex),
        (-1, 0): np.array([[-t]], dtype=complex),
        (0, 1): np.array([[-t]], dtype=complex),
        (0, -1): np.array([[-t]], dtype=complex),
        (-1, 1): np.array([[-t]], dtype=complex),
        (1, -1): np.array([[-t]], dtype=complex),
    }


def _pairing(delta: float, *, sparse=None):
    matrix = np.array([[0.0, delta], [delta, 0.0]], dtype=complex)
    if sparse is not None:
        matrix = sparse.csr_matrix(matrix)
    return {(0, 0): matrix}


def _sparsify_tb(tb, sparse):
    return {key: sparse.csr_matrix(np.asarray(matrix, dtype=complex)) for key, matrix in tb.items()}


def _chiral_pairing(delta: float):
    phase = np.exp(2j * np.pi / 3.0)
    return {
        (1, 0): np.array([[0.0, delta], [delta, 0.0]], dtype=complex),
        (-1, 0): np.array([[0.0, delta], [delta, 0.0]], dtype=complex),
        (0, 1): np.array([[0.0, delta * phase], [delta * phase, 0.0]], dtype=complex),
        (0, -1): np.array(
            [[0.0, delta * np.conj(phase)], [delta * np.conj(phase), 0.0]],
            dtype=complex,
        ),
        (-1, 1): np.array(
            [[0.0, delta * np.conj(phase)], [delta * np.conj(phase), 0.0]],
            dtype=complex,
        ),
        (1, -1): np.array([[0.0, delta * phase], [delta * phase, 0.0]], dtype=complex),
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
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(result.mu - reference_mu) <= 8e-4
    assert abs(result.filling - reference_filling) <= 8e-4
    assert _max_density_error(result.density_matrix, reference_density) <= 8e-4


def test_bdg_chebyshev_matches_exact_density_in_2d():
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25)
    model = Model(
        _square_lattice_2d(t=0.15),
        {(0, 0): np.array([[1.0]], dtype=complex)},
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
        max_mu_iterations=80,
        mu_guess=0.0,
    )
    chebyshev = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=ChebyshevFOE(initial_order=4, max_order=256),
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(chebyshev.mu - exact.mu) <= 2e-3
    assert abs(chebyshev.filling - exact.filling) <= 2e-3
    assert _max_density_error(chebyshev.density_matrix, exact.density_matrix) <= 2e-3


def test_bdg_rational_matches_exact_density_in_2d():
    keys = [(0, 0), (1, 0)]
    meanfield = _pairing(0.25)
    model = Model(
        _square_lattice_2d(t=0.15),
        {(0, 0): np.array([[1.0]], dtype=complex)},
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
        max_mu_iterations=80,
        mu_guess=0.0,
    )
    rational = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=RationalFOE(initial_poles=4, max_poles=256),
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(rational.mu - exact.mu) <= 2e-3
    assert abs(rational.filling - exact.filling) <= 2e-3
    assert _max_density_error(rational.density_matrix, exact.density_matrix) <= 2e-3


def test_bdg_chebyshev_accepts_sparse_matrices_when_scipy_is_available():
    sparse = pytest.importorskip("scipy.sparse")
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
            matrix_function=ChebyshevFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6
    assert np.allclose(result.density_matrix[local], 0.5 * np.eye(2), atol=1e-6)


def test_bdg_rational_accepts_sparse_matrices_when_scipy_is_available():
    sparse = pytest.importorskip("scipy.sparse")
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
            matrix_function=RationalFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6
    assert np.allclose(result.density_matrix[local], 0.5 * np.eye(2), atol=1e-6)


def test_bdg_sparse_chebyshev_matches_exact_density_in_2d():
    sparse = pytest.importorskip("scipy.sparse")
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
        max_mu_iterations=80,
        mu_guess=0.0,
    )
    chebyshev = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=ChebyshevFOE(initial_order=4, max_order=256),
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(chebyshev.mu - exact.mu) <= 2e-3
    assert abs(chebyshev.filling - exact.filling) <= 2e-3
    assert _max_density_error(chebyshev.density_matrix, exact.density_matrix) <= 2e-3


def test_bdg_sparse_rational_matches_exact_density_in_2d():
    sparse = pytest.importorskip("scipy.sparse")
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
        max_mu_iterations=80,
        mu_guess=0.0,
    )
    rational = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=keys,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=40,
            matrix_function=RationalFOE(initial_poles=4, max_poles=256),
        ),
        filling_tol=1e-3,
        mu_tol=1e-3,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(rational.mu - exact.mu) <= 2e-3
    assert abs(rational.filling - exact.filling) <= 2e-3
    assert _max_density_error(rational.density_matrix, exact.density_matrix) <= 2e-3


def test_bdg_sparse_chebyshev_does_not_fallback_to_exact_diagonalization(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
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
        raise AssertionError("Sparse Chebyshev path should not call exact diagonalization")

    monkeypatch.setattr(bdg_matrix_direct, "_exact_density_block", fail_if_exact)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=ChebyshevFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_sparse_rational_does_not_fallback_to_exact_diagonalization(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
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
        raise AssertionError("Sparse Rational path should not call exact diagonalization")

    monkeypatch.setattr(bdg_matrix_direct, "_exact_density_block", fail_if_exact)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=RationalFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_sparse_chebyshev_density_path_avoids_dense_conversion(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
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
        raise AssertionError("Sparse Chebyshev density path should not densify matrices")

    monkeypatch.setattr(bdg_matrix_direct, "to_dense", fail_if_dense)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=ChebyshevFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_sparse_rational_density_path_avoids_dense_conversion(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
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
            matrix_function=RationalFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_dense_chebyshev_does_not_fallback_to_exact_diagonalization(monkeypatch):
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)

    def fail_if_exact(*args, **kwargs):
        raise AssertionError("Dense Chebyshev path should not call exact diagonalization")

    monkeypatch.setattr(bdg_matrix_direct, "_exact_density_block", fail_if_exact)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=ChebyshevFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_dense_rational_does_not_fallback_to_exact_diagonalization(monkeypatch):
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)

    def fail_if_exact(*args, **kwargs):
        raise AssertionError("Dense Rational path should not call exact diagonalization")

    monkeypatch.setattr(bdg_matrix_direct, "_exact_density_block", fail_if_exact)
    result = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=RationalFOE(),
        ),
        filling_tol=1e-6,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(result.mu) <= 1e-8
    assert abs(result.filling - 0.5) <= 1e-6


def test_bdg_sparse_chebyshev_scf_path_avoids_dense_conversion(monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")
    local = (0, 0)
    model = Model(
        {local: sparse.csr_matrix([[0.0]])},
        {local: sparse.csr_matrix([[0.0]])},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    guess = {local: sparse.csr_matrix([[0.0, 0.0], [0.0, 0.0]])}

    def fail_if_dense(*args, **kwargs):
        raise AssertionError("Sparse Chebyshev SCF path should not densify matrices")

    monkeypatch.setattr(bdg_matrix_direct, "to_dense", fail_if_dense)
    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=ChebyshevFOE(),
        ),
        scf=LinearMixing(max_iterations=2, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-6,
    )

    assert abs(result.density_matrix_result.mu) <= 1e-8
    assert abs(result.density_matrix_result.filling - 0.5) <= 1e-6


def test_bdg_zero_dimensional_rational_density_matches_exact():
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)
    exact = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            max_refinements=8,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
        max_mu_iterations=40,
        mu_guess=0.0,
    )
    rational = solve_bdg_density_fixed_filling(
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
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(rational.mu - exact.mu) <= 1e-10
    assert abs(rational.filling - exact.filling) <= 1e-8
    assert np.allclose(rational.density_matrix[local], exact.density_matrix[local], atol=1e-8)


@pytest.mark.parametrize(
    "matrix_function",
    [
        pytest.param(
            ChebyshevFOE(
                initial_order=4,
                max_order=128,
                trace_estimator="hutchinson",
                trace_probes=512,
                trace_seed=5,
            ),
            id="chebyshev-hutchinson",
        ),
        pytest.param(
            RationalFOE(
                initial_poles=4,
                max_poles=128,
                trace_estimator="hutchinson",
                trace_probes=512,
                trace_seed=5,
            ),
            id="rational-hutchinson",
        ),
    ],
)
def test_bdg_zero_dimensional_hutchinson_matches_exact(matrix_function):
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    meanfield = _pairing(0.0)
    exact = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            max_refinements=8,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
        max_mu_iterations=40,
        mu_guess=0.0,
    )
    estimated = solve_bdg_density_fixed_filling(
        model,
        meanfield,
        keys=[local],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-3,
            max_refinements=8,
            matrix_function=matrix_function,
        ),
        filling_tol=1e-3,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(estimated.mu - exact.mu) <= 5e-3
    assert abs(estimated.filling - exact.filling) <= 5e-3
    assert _max_density_error(estimated.density_matrix, exact.density_matrix) <= 5e-3


def test_bdg_rational_scf_smoke():
    local = (0, 0)
    model = Model(
        {local: np.array([[0.0]], dtype=complex)},
        {local: np.array([[0.0]], dtype=complex)},
        filling=0.5,
        kT=0.2,
        superconducting=True,
    )
    guess = {local: np.zeros((2, 2), dtype=complex)}

    result = solver(
        model,
        guess,
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-6,
            max_refinements=8,
            matrix_function=RationalFOE(),
        ),
        scf=LinearMixing(max_iterations=2, alpha=0.5),
        scf_tol=1e-8,
        filling_tol=1e-6,
    )

    assert abs(result.density_matrix_result.mu) <= 1e-8
    assert abs(result.density_matrix_result.filling - 0.5) <= 1e-6


def test_bdg_sparse_supported_density_matches_dense_reference():
    sparse = pytest.importorskip("scipy.sparse")
    h0_dense = {
        (0,): np.diag([0.3, -0.1]).astype(complex),
    }
    h_int_dense = {
        (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
    }
    meanfield_dense = {(0,): np.zeros((4, 4), dtype=complex)}

    h0_sparse = _sparsify_tb(h0_dense, sparse)
    h_int_sparse = _sparsify_tb(h_int_dense, sparse)
    meanfield_sparse = _sparsify_tb(meanfield_dense, sparse)
    integration = AdaptiveQuadrature(
        density_matrix_tol=1e-2,
        max_refinements=12,
        matrix_function=RationalFOE(initial_poles=4, max_poles=128),
    )
    dense_result = solve_bdg_density_fixed_filling(
        Model(h0_dense, h_int_dense, filling=0.8, kT=0.2, superconducting=True),
        meanfield_dense,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )
    sparse_result = solve_bdg_density_fixed_filling(
        Model(h0_sparse, h_int_sparse, filling=0.8, kT=0.2, superconducting=True),
        meanfield_sparse,
        keys=[(0,)],
        integration=integration,
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(dense_result.mu - sparse_result.mu) <= 5e-4
    assert abs(dense_result.filling - sparse_result.filling) <= 5e-4
    support = bdg_top_half_support(
        keys=[(0,)],
        interaction_support=h_int_dense,
        ndof=2,
        local_key=(0,),
    )
    np.testing.assert_allclose(
        bdg_density_to_rparams(
            dense_result.density_matrix,
            support=support,
            ndof=2,
        ),
        bdg_density_to_rparams(
            sparse_result.density_matrix,
            support=support,
            ndof=2,
        ),
        atol=5e-4,
    )


def test_bdg_sparse_rational_aaa_supported_density_matches_dense_reference():
    sparse = pytest.importorskip("scipy.sparse")
    h0_dense = {
        (0,): np.diag([0.3, -0.1]).astype(complex),
    }
    h_int_dense = {
        (0,): np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex),
    }
    meanfield_dense = {(0,): np.zeros((4, 4), dtype=complex)}

    h0_sparse = _sparsify_tb(h0_dense, sparse)
    h_int_sparse = _sparsify_tb(h_int_dense, sparse)
    meanfield_sparse = _sparsify_tb(meanfield_dense, sparse)
    dense_result = solve_bdg_density_fixed_filling(
        Model(h0_dense, h_int_dense, filling=0.8, kT=0.2, superconducting=True),
        meanfield_dense,
        keys=[(0,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-8,
            max_refinements=12,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=1e-8,
        mu_tol=1e-10,
        max_mu_iterations=40,
        mu_guess=0.0,
    )
    sparse_result = solve_bdg_density_fixed_filling(
        Model(h0_sparse, h_int_sparse, filling=0.8, kT=0.2, superconducting=True),
        meanfield_sparse,
        keys=[(0,)],
        integration=AdaptiveQuadrature(
            density_matrix_tol=1e-2,
            max_refinements=12,
            matrix_function=RationalFOE(
                initial_poles=4,
                max_poles=512,
                rational_scheme="aaa",
            ),
        ),
        filling_tol=1e-2,
        mu_tol=1e-8,
        max_mu_iterations=40,
        mu_guess=0.0,
    )

    assert abs(dense_result.mu - sparse_result.mu) <= 5e-3
    assert abs(dense_result.filling - sparse_result.filling) <= 5e-3
    support = bdg_top_half_support(
        keys=[(0,)],
        interaction_support=h_int_dense,
        ndof=2,
        local_key=(0,),
    )
    np.testing.assert_allclose(
        bdg_density_to_rparams(
            dense_result.density_matrix,
            support=support,
            ndof=2,
        ),
        bdg_density_to_rparams(
            sparse_result.density_matrix,
            support=support,
            ndof=2,
        ),
        atol=5e-3,
    )


def test_bdg_complex_chiral_density_matches_dense_reference():
    keys = [(1, 0), (0, 1), (-1, 1)]
    meanfield = _chiral_pairing(0.12)
    model = Model(
        _triangular_lattice_2d(t=0.25),
        {
            (1, 0): np.array([[1.0]], dtype=complex),
            (-1, 0): np.array([[1.0]], dtype=complex),
            (0, 1): np.array([[1.0]], dtype=complex),
            (0, -1): np.array([[1.0]], dtype=complex),
            (-1, 1): np.array([[1.0]], dtype=complex),
            (1, -1): np.array([[1.0]], dtype=complex),
        },
        filling=0.55,
        kT=0.14,
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
            density_matrix_tol=6e-4,
            max_refinements=120,
            matrix_function=DirectDiagonalization(),
        ),
        filling_tol=6e-4,
        mu_tol=6e-4,
        max_mu_iterations=80,
        mu_guess=0.0,
    )

    assert abs(result.mu - reference_mu) <= 1e-3
    assert abs(result.filling - reference_filling) <= 1e-3
    assert _max_density_error(result.density_matrix, reference_density) <= 1e-3
