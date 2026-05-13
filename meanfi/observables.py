import numpy as np

from meanfi.meanfield import (
    bdg_correction_from_density,
    extract_anomalous_density,
    extract_electron_density,
    meanfield,
)
from meanfi.model import Model
from meanfi.tb.ops import _tb_type


def expectation_value(density_matrix: _tb_type, observable: _tb_type) -> complex:
    """Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix :
        Density matrix tight-binding dictionary.
    observable :
        Observable tight-binding dictionary.

    Returns
    -------
    :
        Expectation value.
    """
    return np.sum(
        [
            np.sum(observable[k].T * density_matrix[tuple(-np.array(k))])
            for k in frozenset(density_matrix) & frozenset(observable)
        ]
    )


def total_energy(model: Model, density_matrix: _tb_type) -> float:
    """Compute the total mean-field internal energy density.

    This evaluates the mean-field energy per unit cell using the same Hartree,
    Fock, and pairing conventions as the solver. The interaction correction is
    counted with a factor of ``1/2`` to avoid double-counting the state that
    generated the mean-field Hamiltonian.

    The provided density matrix is expected to contain all keys required by
    ``model.h_0`` and ``model.h_int``. For fixed-filling finite-temperature
    calculations, this is an internal energy, not a free energy with entropy.

    Parameters
    ----------
    model :
        MeanFi model defining the non-interacting Hamiltonian and interaction.
    density_matrix :
        Density matrix tight-binding dictionary for the state whose energy is
        evaluated.

    Returns
    -------
    :
        Real scalar total mean-field energy density.
    """

    if not model.superconducting:
        correction = meanfield(density_matrix, model.h_int)
        energy = expectation_value(density_matrix, model.h_0)
        energy += 0.5 * expectation_value(density_matrix, correction)
        return float(np.real(energy))

    electron_density = extract_electron_density(density_matrix, model)
    anomalous_density = extract_anomalous_density(density_matrix, model)
    correction = bdg_correction_from_density(density_matrix, model)
    normal_correction = {
        key: matrix[: model._ndof, : model._ndof] for key, matrix in correction.items()
    }
    pairing_correction = {
        key: matrix[: model._ndof, model._ndof :] for key, matrix in correction.items()
    }

    energy = expectation_value(electron_density, model.h_0)
    energy += 0.5 * expectation_value(electron_density, normal_correction)
    energy += 0.5 * expectation_value(anomalous_density, pairing_correction)
    return float(np.real(energy))
