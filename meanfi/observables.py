"""Physical observables and energies per cell per physical orbital."""

import numpy as np
from scipy import sparse

from meanfi.meanfield import interaction_energy
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.tb.expectation import expectation_value
from meanfi.tb.ops import _tb_type, block_diag


def internal_energy(model: Model, density_matrix: _tb_type | DensityResult) -> float:
    """Compute mean-field internal energy per cell per physical orbital.

    The density must cover the one-body Hamiltonian and interaction. Selected
    results may use the model's reduced interaction coordinates. Interaction
    energy carries a factor of one half and uses the difference from the
    reference normal and pairing densities. The one-body term uses the actual
    density. BdG pairing uses the conjugate anomalous density difference.
    """
    active = model._active_density_from_state(
        model._reference_difference(model._density_state(density_matrix))
    )
    if model.superconducting:
        # Embedding the observable preserves selected density entries.
        zero = sparse.csr_matrix((model._ndof, model._ndof), dtype=complex)
        observable = {key: block_diag(block, zero) for key, block in model.h_0.items()}
    else:
        observable = model.h_0
    one_body = (
        float(np.real(expectation_value(density_matrix, observable))) / model._ndof
    )
    return one_body + interaction_energy(
        active, model.h_int, electron_ndof=model._electron_ndof
    )


def free_energy(model: Model, density: DensityResult) -> float:
    """Compute Helmholtz free energy per cell per physical orbital as ``U - kT * entropy``.

    Entropy is in units of Boltzmann's constant per physical orbital and belongs
    to the complete state evaluated by the density solver, including selected results.
    A bare tight-binding dictionary does not retain that entropy.
    """
    if not isinstance(density, DensityResult):
        raise TypeError("free_energy requires a DensityResult with computed entropy")
    return internal_energy(model, density) - model.kT * density.entropy
