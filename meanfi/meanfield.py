from __future__ import annotations

import numpy as np
import scipy.sparse as sparse

from meanfi.tb.bdg import assemble_bdg_tb
from meanfi.tb.ops import (
    _tb_type,
    add_tb,
    as_sparse,
    elementwise_product,
    is_sparse_like,
)
from meanfi.tb.validate import tb_dimension, tb_orbital_count
from meanfi.space.coordinates import onsite_key
from meanfi.tb.expectation import expectation_value
from meanfi.tb.storage import prefers_sparse_storage
from meanfi.results import DensityResult
from meanfi.space.space import ActiveSCFSpace


def meanfield(density_matrix: _tb_type | DensityResult, h_int: _tb_type) -> _tb_type:
    """Compute the normal mean-field correction from a density matrix."""

    if isinstance(density_matrix, DensityResult):
        space = ActiveSCFSpace.from_interaction(
            h_int, sparse=prefers_sparse_storage(h_int)
        )
        values = density_matrix.values_for(space.required_coordinates)
        density_matrix = space.density_from_params(
            space.params_from_required_entries(values)
        )
    local = onsite_key(tb_dimension(density_matrix))
    diagonal_density = np.asarray(density_matrix[local].diagonal()).real.ravel()
    onsite_diagonal = np.zeros_like(diagonal_density, dtype=complex)
    sparse_present = prefers_sparse_storage(density_matrix, h_int)
    for interaction in h_int.values():
        onsite_diagonal += np.asarray(diagonal_density @ interaction).ravel()
    direct = {
        local: (
            sparse.diags(onsite_diagonal, format="csr")
            if sparse_present
            else np.diag(onsite_diagonal)
        )
    }
    exchange = {
        key: -elementwise_product(interaction, density_matrix[key])
        for key, interaction in h_int.items()
    }
    return add_tb(direct, exchange)


def _antisymmetrize_anomalous_block(anomalous_block: _tb_type, ndof: int) -> _tb_type:
    """Project pairing blocks onto Delta(R) = -Delta(-R).T."""

    zero = (
        sparse.csr_matrix((ndof, ndof), dtype=complex)
        if prefers_sparse_storage(anomalous_block)
        else np.zeros((ndof, ndof), dtype=complex)
    )
    keys = frozenset(anomalous_block) | {
        tuple(-np.asarray(key, dtype=int)) for key in anomalous_block
    }
    result = {}
    for key in keys:
        if key in result:
            continue
        opposite = tuple(-np.asarray(key, dtype=int))
        block = anomalous_block.get(key, zero)
        opposite_block = anomalous_block.get(opposite, zero)
        if is_sparse_like(block) or is_sparse_like(opposite_block):
            projected = 0.5 * (as_sparse(block) - as_sparse(opposite_block).T)
            projected = projected.tocsr()
        else:
            projected = 0.5 * (
                np.asarray(block, dtype=complex)
                - np.asarray(opposite_block, dtype=complex).T
            )
        result[key] = projected
        if opposite != key:
            result[opposite] = -projected.T
    return result


def interaction_correction(
    density_matrix: _tb_type,
    h_int: _tb_type,
    *,
    electron_ndof: int | None = None,
) -> _tb_type:
    """Apply the linear interaction map to normal or normal-and-pairing density."""
    if electron_ndof is None:
        return meanfield(density_matrix, h_int)
    ndof = electron_ndof
    electron_density = {
        key: matrix[:ndof, :ndof] for key, matrix in density_matrix.items()
    }
    anomalous_density = {
        key: matrix[:ndof, ndof:] for key, matrix in density_matrix.items()
    }
    normal_block = meanfield(electron_density, h_int)
    anomalous_block = {
        key: -elementwise_product(
            interaction,
            anomalous_density[key],
        )
        for key, interaction in h_int.items()
    }
    anomalous_block = _antisymmetrize_anomalous_block(anomalous_block, ndof)
    return assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)


def correction_expectation(
    density: _tb_type, correction: _tb_type, *, electron_ndof: int | None = None
) -> float:
    """Interaction contraction per physical orbital, with BdG counting applied once."""
    if electron_ndof is None:
        return float(
            np.real(expectation_value(density, correction))
        ) / tb_orbital_count(density)
    ndof = electron_ndof
    electron = {key: block[:ndof, :ndof] for key, block in density.items()}
    normal = {key: block[:ndof, :ndof] for key, block in correction.items()}
    energy = expectation_value(electron, normal)
    # Pairing contracts at the same displacement and conjugates the density.
    for key, block in correction.items():
        energy += elementwise_product(
            density[key][:ndof, ndof:].conj(), block[:ndof, ndof:]
        ).sum()
    return float(np.real(energy)) / ndof


def interaction_energy(
    difference: _tb_type, h_int: _tb_type, *, electron_ndof: int | None = None
) -> float:
    """Quadratic interaction energy of a density difference, per physical orbital."""
    correction = interaction_correction(difference, h_int, electron_ndof=electron_ndof)
    return 0.5 * correction_expectation(
        difference, correction, electron_ndof=electron_ndof
    )
