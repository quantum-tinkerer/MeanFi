"""Physical observables and energies per cell per physical orbital."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from meanfi.meanfield import correction_expectation, interaction_energy
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.tb.expectation import expectation_value
from meanfi.tb.ops import _tb_type, block_diag
from meanfi.tb.storage import tb_entries_changed


def internal_energy(model: Model, density_matrix: _tb_type | DensityResult) -> float:
    """Compute mean-field internal energy per cell per physical orbital.

    Model-based results reuse their retained energy, including selected views.
    Otherwise the density must cover the one-body Hamiltonian and interaction.
    Selected results may use the model's reduced interaction coordinates. Interaction
    energy carries a factor of one half and uses the difference from the
    reference normal and pairing densities. The one-body term uses the actual
    density. BdG pairing uses the conjugate anomalous density difference.
    """
    if (
        isinstance(density_matrix, DensityResult)
        and density_matrix._model is model
        and density_matrix.internal_energy is not None
    ):
        return density_matrix.internal_energy
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


def _internal_energy_from_band(model, state, band_energy, correction):
    """Use a known interaction correction to recover the bare one-body energy."""
    one_body = band_energy - correction_expectation(
        model._active_density_from_state(state),
        correction,
        electron_ndof=model._electron_ndof,
    )
    difference = model._active_density_from_state(model._reference_difference(state))
    return one_body + interaction_energy(
        difference,
        model.h_int,
        electron_ndof=model._electron_ndof,
    )


def _with_model_energy(model, density, correction):
    """Attach the evaluated model energy without requesting more density entries."""
    if not density.covers(model.required_coordinates):
        return density
    correction = {} if correction is None else correction
    if correction and tb_entries_changed(
        correction, model._project_mean_field(correction)
    ):
        # An arbitrary external correction may require density entries outside
        # the interaction space. Only use complete one-body blocks in that case.
        if not density.is_complete or not set(model.h_0) <= set(
            density.coordinates.keys
        ):
            return density
        energy = internal_energy(model, density)
    else:
        energy = _internal_energy_from_band(
            model, model._density_state(density), density.band_energy, correction
        )
    return replace(density, internal_energy=energy, _model=model)
