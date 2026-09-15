import numpy as np

from meanfi.meanfield import (
    bdg_correction_from_density,
    extract_electron_density,
    meanfield,
)
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.space.coordinates import opposite_key
from meanfi.tb.ops import _tb_type, block_diag, elementwise_product, is_sparse_like
from scipy import sparse


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
        Unnormalized trace per cell. Unlike the thermodynamic energy helpers,
        this general observable contraction is not divided by orbital count.
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


def _bdg_correction_expectation(
    density: _tb_type, correction: _tb_type, ndof: int
) -> float:
    """Contract the electron and pairing blocks without Nambu double counting."""
    electron = {key: block[:ndof, :ndof] for key, block in density.items()}
    normal = {key: block[:ndof, :ndof] for key, block in correction.items()}
    energy = expectation_value(electron, normal)
    # Pairing uses the same-key Frobenius product, including complex phases.
    for key, block in correction.items():
        energy += elementwise_product(
            density[key][:ndof, ndof:].conj(), block[:ndof, ndof:]
        ).sum()
    return float(np.real(energy))


def internal_energy(model: Model, density_matrix: _tb_type | DensityResult) -> float:
    """Compute mean-field internal energy per cell per physical orbital.

    The density must cover the one-body Hamiltonian and interaction. Selected
    results may use the model's reduced interaction coordinates. Interaction
    energy carries a factor of one half and uses the difference from the
    reference normal and pairing densities. The one-body term uses the actual
    density. BdG pairing uses the conjugate anomalous density difference.
    """
    if not isinstance(density_matrix, DensityResult):
        required = set(model.h_0) | set(model.scf_space.interaction_keys)
        missing = sorted(required - set(density_matrix))
        if missing:
            raise ValueError(
                "density_matrix is missing keys required for internal energy: "
                f"{missing}"
            )
    active = model._active_density_from_state(
        model._reference_difference(model._density_state(density_matrix))
    )
    if not model.superconducting:
        correction = meanfield(active, model.h_int)
        energy = expectation_value(density_matrix, model.h_0)
        energy += 0.5 * expectation_value(active, correction)
        return float(np.real(energy)) / model._ndof

    if isinstance(density_matrix, DensityResult):
        # Embed the observable sparsely, leaving unrequested Nambu entries alone.
        zero = sparse.csr_matrix((model._ndof, model._ndof), dtype=complex)
        observable = {key: block_diag(block, zero) for key, block in model.h_0.items()}
        energy = expectation_value(density_matrix, observable)
    else:
        energy = expectation_value(
            extract_electron_density(density_matrix, model), model.h_0
        )
    correction = bdg_correction_from_density(active, model)
    energy += 0.5 * _bdg_correction_expectation(active, correction, model._ndof)
    return float(np.real(energy)) / model._ndof


def free_energy(model: Model, density: DensityResult) -> float:
    """Compute Helmholtz free energy per cell per physical orbital as ``U - kT * entropy``.

    Entropy is in units of Boltzmann's constant per physical orbital and belongs
    to the complete state evaluated by the density solver, including selected results.
    A bare tight-binding dictionary does not retain that entropy.
    """
    if not isinstance(density, DensityResult):
        raise TypeError("free_energy requires a DensityResult with computed entropy")
    return internal_energy(model, density) - model.kT * density.entropy
