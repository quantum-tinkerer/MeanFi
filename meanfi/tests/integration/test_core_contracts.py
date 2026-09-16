"""Physical boundaries and reproducible SCF results, independent of storage."""

from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse
from scipy.special import expit

import meanfi as mf
from meanfi.density.integrate.methods import IntegrationMethod
from meanfi.scf.methods import SCFMethod
from meanfi.tb.bdg import assemble_bdg_tb
from meanfi.tb.ops import to_dense

pytestmark = pytest.mark.integration


def _model(*, superconducting=False, storage=np.asarray):
    return mf.Model(
        {(): storage([[-0.3, 0.07j], [-0.07j, 0.7]])},
        {(): storage([[0.0, 0.7], [0.7, 0.0]])},
        filling=0.8,
        kT=0.2,
        superconducting=superconducting,
    )


@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix, sparse.csr_array])
def test_interaction_coefficients_must_be_real(storage):
    h = {(): storage(np.diag([-0.3, 0.7]))}
    interaction = {(): storage([[0.0, 0.2j], [-0.2j, 0.0]])}
    with pytest.raises(ValueError, match="real.*density-density"):
        mf.Model(h, interaction, filling=1)
    with pytest.raises(ValueError, match="real.*density-density"):
        mf.density_matrix(h, filling=1, interaction=interaction)
    # Complex storage is valid when the physical coefficients are real.
    model = _model(storage=storage)
    np.testing.assert_array_equal(to_dense(model.h_int[()]), [[0, 0.7], [0.7, 0]])


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("invalid", ["size", "key", "dimension", "nan", "hermiticity"])
def test_invalid_corrections_fail_at_public_boundaries(superconducting, invalid):
    model = _model(superconducting=superconducting)
    size = 4 if superconducting else 2
    matrix = np.zeros((size, size), dtype=complex)
    correction = {(): matrix}
    if invalid == "size":
        correction = {(): np.ones((1, 1))}
    elif invalid == "key":
        correction = {(0.5,): matrix}
    elif invalid == "dimension":
        correction = {(0,): matrix}
    elif invalid == "nan":
        matrix[0, 0] = np.nan
    else:
        matrix[0, 1] = 1.0
    for evaluate in (
        lambda: model.hamiltonian_from_meanfield(correction),
        lambda: mf.density_matrix(model, mean_field=correction),
        lambda: mf.solver(model, correction),
    ):
        with pytest.raises(ValueError):
            evaluate()


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("referenced", [False, True])
@pytest.mark.parametrize("storage", [np.asarray, sparse.csr_matrix])
def test_partial_result_correction_reproduces_density_and_energy(
    superconducting, referenced, storage
):
    model = _model(superconducting=superconducting, storage=storage)
    method = mf.UniformGrid(matrix_function=mf.DirectDiagonalization())
    targets = replace(mf.default_solver_tolerances(1e-12), mu_tol=1e-14)
    if referenced:
        reference = mf.density_matrix(model, integration=method, keys=[()], tol=targets)
        model = replace(model, reference=reference)
    with pytest.raises(mf.NoConvergence) as failure:
        mf.solver(
            model,
            model.random_meanfield(rng=17, scale=0.2),
            integration=method,
            scf=mf.LinearMixing(max_iterations=1),
            tol=targets,
            compute_free_energy=False,
        )
    result = failure.value.result
    matrix = to_dense(model.hamiltonian_from_meanfield(result.mean_field)[()])
    charge = np.array([1, 1, -1, -1]) if superconducting else np.ones(2)
    energies, vectors = np.linalg.eigh(matrix - result.mu * np.diag(charge))
    exact = (vectors * expit(-energies / model.kT)) @ vectors.conj().T
    expected = result.density.coordinates.values_from_assembled_matrix(exact)
    error = np.max(abs(result.density.values - expected))
    assert error < 1e-12, f"Returned Hamiltonian/density mismatch: {error}"
    assert (
        abs(result.internal_energy - mf.evaluate_internal_energy(model, {(): exact}))
        < 1e-12
    )
    # The next correction is a distinct operation before convergence.
    next_correction = model.mean_field(result.density)
    assert (
        max(
            np.max(
                abs(to_dense(next_correction[key]) - to_dense(result.mean_field[key]))
            )
            for key in next_correction
        )
        > 1e-7
    )
    for key, value in next_correction.items():
        np.testing.assert_allclose(
            to_dense(value), to_dense(model.mean_field({(): exact})[key]), atol=1e-12
        )


@pytest.mark.parametrize("superconducting", [False, True])
def test_empty_correction_is_zero(superconducting):
    model = _model(superconducting=superconducting)
    np.testing.assert_array_equal(
        model.hamiltonian_from_meanfield({})[()], model.hamiltonian_from_meanfield()[()]
    )


def test_bdg_correction_must_have_particle_hole_structure():
    model = _model(superconducting=True)
    correction = assemble_bdg_tb(
        {(): np.diag([0.1, 0.2])}, {(): np.array([[0, 0.03j], [-0.03j, 0]])}, ndof=2
    )
    model.hamiltonian_from_meanfield(correction)
    correction[()][2, 2] += 0.1  # Remains Hermitian, but violates Nambu structure.
    with pytest.raises(ValueError, match="lower-right"):
        model.hamiltonian_from_meanfield(correction)


def test_unsupported_methods_fail_before_any_eigensolve(monkeypatch):
    model = _model()

    def fail(*args, **kwargs):
        raise AssertionError("unsupported method reached numerical work")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    monkeypatch.setattr(np.linalg, "eigvalsh", fail)
    with pytest.raises(TypeError, match="scf must be"):
        mf.solver(model, {}, scf=SCFMethod())
    with pytest.raises(TypeError, match="integration must be"):
        mf.density_matrix(model, integration=IntegrationMethod())


@pytest.mark.parametrize("superconducting", [False, True])
def test_zero_correction_blocks_can_omit_their_opposite_key(superconducting):
    finite = _model(superconducting=superconducting)
    model = replace(finite, h_0={(0,): finite.h_0[()]}, h_int={(0,): finite.h_int[()]})
    size = 4 if superconducting else 2
    correction = {(1,): np.zeros((size, size))}
    result = mf.density_matrix(
        model, mean_field=correction, integration=mf.UniformGrid(nk=4)
    )
    expected = mf.density_matrix(model, integration=mf.UniformGrid(nk=4))
    np.testing.assert_array_equal(result.values, expected.values)
    assert result.internal_energy == expected.internal_energy


@pytest.mark.parametrize("lattice", [np.zeros((1, 1)), np.diag([1, 2])])
def test_spatial_symmetry_rejects_noninvertible_lattice_maps(lattice):
    with pytest.raises(ValueError, match="determinant"):
        mf.SpatialSymmetry(lattice, {(0,) * len(lattice): np.eye(2)})


@pytest.mark.parametrize(
    "blocks",
    [
        {(0,): 2 * np.eye(2)},
        {(0,): np.eye(2), (1,): np.eye(2)},
        {(0,): np.zeros((2, 2))},
    ],
)
def test_spatial_symmetry_rejects_nonunitary_bloch_transformations(blocks):
    with pytest.raises(ValueError, match="unitary transformation"):
        mf.SpatialSymmetry(np.eye(1, dtype=int), blocks)


def test_spatial_symmetry_accepts_unitarity_shared_between_shifts():
    blocks = {(0,): np.diag([1.0, 0.0]), (1,): np.diag([0.0, 1.0])}
    symmetry = mf.SpatialSymmetry(np.eye(1, dtype=int), blocks)
    for k in np.linspace(-np.pi, np.pi, 7):
        unitary = sum(
            block * np.exp(1j * key[0] * k)
            for key, block in symmetry.unitaries_by_shift.items()
        )
        np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(2), atol=1e-14)


def test_global_phase_does_not_constrain_normal_density():
    model = _model()
    symmetry = mf.SpatialSymmetry(
        np.empty((0, 0), dtype=int), {(): ((1 + 1j) / np.sqrt(2)) * np.eye(2)}
    )
    constrained = replace(model, spatial_symmetries=(symmetry,))
    # U† rho U = rho for a scalar phase. Floating-point U† U differs from I
    # by 2e-16, which must not remove all four real density degrees of freedom.
    assert constrained._space.num_params == model._space.num_params == 4
    density = {(): np.array([[0.7, 0.1 + 0.2j], [0.1 - 0.2j, 0.3]])}
    np.testing.assert_allclose(
        constrained.mean_field(density)[()],
        model.mean_field(density)[()],
        atol=1e-14,
        rtol=0,
    )


@pytest.mark.parametrize("superconducting", [False, True])
@pytest.mark.parametrize("storage", [sparse.csr_matrix, sparse.csr_array])
def test_sparse_structural_mutations_cannot_change_model_or_cached_state(
    superconducting, storage
):
    # setdiag can replace a sparse matrix's arrays even when they are read-only.
    model = mf.Model(
        {(): storage((2, 2))},
        {(): storage(np.array([[0.0, 0.7], [0.7, 0.0]]))},
        filling=1,
        kT=0.2,
        superconducting=superconducting,
        reference={(): storage((2, 2))},
    )
    method = mf.UniformGrid(matrix_function=mf.DirectDiagonalization())
    before = mf.density_matrix(model, keys=[()], integration=method)
    correction = model.mean_field(before)
    for name in ("h_0", "h_int", "reference"):
        blocks = getattr(model, name)
        expected = blocks[()].toarray()
        exposed = blocks[()]
        # Public reads share storage; they do not copy large matrix arrays.
        assert exposed.data is blocks[()].data
        exposed.setdiag([1.0, 2.0])
        np.testing.assert_array_equal(blocks[()].toarray(), expected)
    after = mf.density_matrix(model, keys=[()], integration=method)
    np.testing.assert_array_equal(after.values, before.values)
    np.testing.assert_array_equal(
        model.mean_field(after)[()].toarray(), correction[()].toarray()
    )
    assert after.internal_energy == pytest.approx(
        mf.evaluate_internal_energy(model, after), abs=1e-14
    )


@pytest.mark.parametrize(
    "left_sparse,right_sparse", [(False, False), (True, True), (True, False)]
)
@pytest.mark.parametrize("overlap", [False, True])
def test_add_tb_rejects_incompatible_matrix_sizes(left_sparse, right_sparse, overlap):
    left, right = np.eye(2), np.ones((1, 1))
    if left_sparse:
        left = sparse.csr_matrix(left)
    if right_sparse:
        right = sparse.csr_matrix(right)
    with pytest.raises(ValueError, match="matching matrix sizes"):
        mf.add_tb({(0,): left}, {(0,) if overlap else (1,): right})


def test_add_tb_requires_matching_lattice_dimensions_and_accepts_empty_corrections():
    h = {(): np.eye(2)}
    with pytest.raises(ValueError, match="lattice dimensions"):
        mf.add_tb(h, {(0,): np.eye(2)})
    for left, right in ((h, {}), ({}, h)):
        np.testing.assert_array_equal(mf.add_tb(left, right)[()], h[()])
    assert mf.add_tb({}, {}) == {}
