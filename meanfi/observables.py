import numpy as np

from meanfi.meanfield import (
    bdg_correction_from_density,
    extract_anomalous_density,
    extract_electron_density,
    meanfield,
)
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.space.coordinates import opposite_key
from meanfi.tb.ops import _tb_type, elementwise_product, is_sparse_like


def expectation_value(
    density_matrix: _tb_type | DensityResult,
    observable: _tb_type,
) -> complex:
    """Compute the expectation value of an observable with respect to a density matrix.

    Parameters
    ----------
    density_matrix :
        Layout-aware density result or complete density-matrix tight-binding
        dictionary.
    observable :
        Observable tight-binding dictionary.

    Returns
    -------
    :
        Expectation value.
    """
    if isinstance(density_matrix, DensityResult):
        available = dict(
            zip(
                density_matrix.coordinates.entries,
                density_matrix.values,
                strict=True,
            )
        )
        total = 0.0j
        missing = []
        for key, block in observable.items():
            if is_sparse_like(block):
                coordinate_matrix = block.tocoo(copy=True)
                coordinate_matrix.sum_duplicates()
                coordinate_matrix.eliminate_zeros()
                rows, cols, weights = (
                    coordinate_matrix.row,
                    coordinate_matrix.col,
                    coordinate_matrix.data,
                )
            else:
                block = np.asarray(block)
                rows, cols = np.nonzero(block)
                weights = block[rows, cols]
            density_key = opposite_key(tuple(key))
            for row, col, weight in zip(rows, cols, weights, strict=True):
                entry = (density_key, int(col), int(row))
                value = available.get(entry)
                if value is None:
                    missing.append(entry)
                else:
                    total += weight * value
        if missing:
            unique_missing = tuple(dict.fromkeys(missing))
            preview = ", ".join(map(str, unique_missing[:3]))
            suffix = "" if len(unique_missing) <= 3 else ", ..."
            raise ValueError(
                "density is missing "
                f"{len(unique_missing)} coordinate(s) required by the observable: "
                f"{preview}{suffix}"
            )
        return complex(total)

    total = 0.0j
    missing_keys = []
    for key, block in observable.items():
        density_key = opposite_key(tuple(key))
        if density_key not in density_matrix:
            if (block != 0).sum():
                missing_keys.append(density_key)
            continue
        total += elementwise_product(block.T, density_matrix[density_key]).sum()
    if missing_keys:
        raise ValueError(
            "density_matrix is missing keys required by the observable: "
            f"{sorted(set(missing_keys))}"
        )
    return complex(total)


def _validate_total_energy_density(
    model: Model,
    density_matrix: _tb_type | DensityResult,
) -> None:
    if isinstance(density_matrix, DensityResult):
        density_matrix.values_for(model.scf_space.required_coordinates)
        return
    required = set(model.h_0) | set(model.scf_space.interaction_keys)
    missing = sorted(required - set(density_matrix))
    if missing:
        raise ValueError(
            f"density_matrix is missing keys required for total energy: {missing}"
        )


def total_energy(model: Model, density_matrix: _tb_type | DensityResult) -> float:
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
        Layout-aware density result or complete density-matrix tight-binding
        dictionary for the state whose energy is evaluated. It must cover all
        coordinates used by the one-body and interaction terms.

    Returns
    -------
    :
        Real scalar total mean-field energy density.
    """

    if not model.superconducting:
        _validate_total_energy_density(model, density_matrix)
        density_difference = model._active_density_from_state(
            model._reference_difference(model._density_state(density_matrix))
        )
        correction = meanfield(density_difference, model.h_int)
        energy = expectation_value(density_matrix, model.h_0)
        energy += 0.5 * expectation_value(density_difference, correction)
        return float(np.real(energy))

    if isinstance(density_matrix, DensityResult):
        density_matrix = density_matrix.to_tb()
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
