"""Sparse SCF reconstruction must stay sparse through Hamiltonian assembly."""

import numpy as np
import pytest
from scipy import sparse

from meanfi import (
    DensityEntries,
    DensityResult,
    DirectDiagonalization,
    ErrorValues,
    LinearMixing,
    Model,
    PeriodicGrid,
    expectation_value,
    density_matrix,
    add_tb,
    solver,
)
from meanfi.meanfield import meanfield
from meanfi.tb.bdg import validate_bdg_tb
from meanfi.tb.ops import to_dense


@pytest.mark.parametrize("superconducting", [False, True])
def test_large_sparse_model_reconstruction_avoids_dense_blocks(
    monkeypatch, superconducting
):
    size = 10_000
    h = {(0,): sparse.eye(size, dtype=complex, format="csr")}
    interaction = {
        (0,): sparse.diags(
            [np.ones(size - 1), np.ones(size - 1)], [-1, 1], format="csr"
        )
    }

    def forbid_dense(*args, **kwargs):
        raise AssertionError("sparse SCF attempted to materialize a dense matrix")

    for matrix_type in (sparse.csr_matrix, sparse.csc_matrix):
        monkeypatch.setattr(matrix_type, "toarray", forbid_dense)
    original_zeros = np.zeros

    def bounded_zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple) and len(shape) == 2 and min(shape) >= size:
            forbid_dense()
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", bounded_zeros)
    model = Model(h, interaction, filling=size / 2, superconducting=superconducting)
    correction = model.random_meanfield(rng=3, scale=0.01)
    assert all(sparse.issparse(block) for block in correction.values())
    if superconducting:
        hamiltonian = model.bdg_hamiltonian_from_meanfield(correction)
        validate_bdg_tb(hamiltonian, ndof=size, ndim=1)
    else:
        coordinates = model.scf_space.required_coordinates
        result = DensityResult(
            entries=DensityEntries(coordinates, np.ones(coordinates.value_count)),
            mu=0.0,
            filling=size,
            errors=ErrorValues(),
        )
        assert expectation_value(result, h) == size
        hamiltonian = model.hamiltonian_from_rho(result)
        density = model.scf_space.meanfield_input_from_params(
            np.ones(model.scf_space.num_params)
        )
        assert np.isfinite(expectation_value(density, meanfield(density, interaction)))
    assert all(sparse.issparse(block) for block in hamiltonian.values())


@pytest.mark.parametrize("superconducting", [False, True])
def test_sparse_and_dense_scf_agree(superconducting):
    h = {
        (0,): np.array([[0.15, 0.08j], [-0.08j, -0.1]]),
        (1,): np.diag([-0.7, -0.5]).astype(complex),
        (-1,): np.diag([-0.7, -0.5]).astype(complex),
    }
    interaction = {(0,): np.array([[0.0, 0.25], [0.25, 0.0]])}
    results = []
    for use_sparse in (False, True):
        model = Model(
            {
                key: sparse.csr_matrix(value) if use_sparse else value
                for key, value in h.items()
            },
            {
                key: sparse.csr_matrix(value) if use_sparse else value
                for key, value in interaction.items()
            },
            filling=0.8,
            kT=0.2,
            superconducting=superconducting,
        )
        result = solver(
            model,
            model.random_meanfield(rng=12, scale=0.03),
            integration=PeriodicGrid(nk=32, matrix_function=DirectDiagonalization()),
            scf=LinearMixing(alpha=0.7, max_iterations=100),
            scf_tol=1e-8,
            filling_tol=1e-10,
        )
        assert result.converged
        assert all(
            sparse.issparse(block) == use_sparse for block in result.mean_field.values()
        )
        results.append(result)
    dense, selected = results
    np.testing.assert_allclose(
        dense.density.values, selected.density.values, atol=1e-10
    )
    assert dense.mu == pytest.approx(selected.mu, abs=1e-10)
    for key in dense.mean_field:
        np.testing.assert_allclose(
            dense.mean_field[key], to_dense(selected.mean_field[key]), atol=1e-10
        )


def test_observable_from_mixed_dense_sparse_hamiltonian():
    h = {(): np.diag([-0.5, 0.5])}
    model = Model(h, {(): sparse.eye(2, format="csr")}, filling=1.0, kT=0.2)
    density = density_matrix(h, filling=1.0, kT=0.2, keys=[()])
    hamiltonian = model.hamiltonian_from_rho(density)
    assert expectation_value(density, hamiltonian) == pytest.approx(
        expectation_value(density.to_matrix(), hamiltonian)
    )
    mixed_density = add_tb(density.to_matrix(), {(): sparse.csr_matrix((2, 2))})
    np.testing.assert_allclose(
        to_dense(meanfield(mixed_density, model.h_int)[()]),
        to_dense(meanfield(density.to_matrix(), model.h_int)[()]),
    )
