"""Physical observables and energies per cell per physical orbital."""

from dataclasses import replace

import numpy as np
from scipy import sparse

from meanfi.meanfield import correction_expectation, interaction_energy
from meanfi.model import Model
from meanfi.results import DensityResult
from meanfi.space.coordinates import opposite_key
from meanfi.tb.expectation import expectation_value
from meanfi.tb.ops import _tb_type, block_diag
from meanfi.tb.storage import tb_entries_changed


def evaluate_internal_energy(
    model: Model, density_matrix: _tb_type | DensityResult
) -> float:
    """Evaluate a supplied density under a model, per cell per physical orbital.

    Requires the one-body and interaction entries for the supplied model.
    Read result.internal_energy for the energy already computed by a density
    calculation or SCF solve.
    Selected results may use the model's reduced interaction coordinates. Interaction
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


def evaluate_free_energy(model: Model, density: DensityResult) -> float:
    """Evaluate a supplied state's Helmholtz free energy per physical orbital.

    Requires the one-body and interaction entries used by evaluate_internal_energy.
    Read result.free_energy for an already calculated state.
    Entropy is in units of Boltzmann's constant per physical orbital and belongs
    to the complete state evaluated by the density solver, including selected results.
    A bare tight-binding dictionary does not retain that entropy.
    """
    if not isinstance(density, DensityResult):
        raise TypeError(
            "evaluate_free_energy requires a DensityResult with computed entropy"
        )
    if density.entropy is None:
        raise ValueError(
            "entropy was not computed; enable compute_free_energy for the calculation"
        )
    return evaluate_internal_energy(model, density) - model.kT * density.entropy


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
    if density.band_energy is not None:
        projected = model._project_mean_field(correction) if correction else correction
        if not tb_entries_changed(correction, projected):
            energy = _internal_energy_from_band(
                model, model._density_state(density), density.band_energy, projected
            )
            return replace(density, internal_energy=energy)
    # A direct contraction needs only actual one-body nonzeros. For BdG these
    # are already addressed in the electron block of the selected density.
    available = set(density.coordinates.entries)
    for key, block in model.h_0.items():
        rows, cols = block.nonzero()
        if any(
            (opposite_key(key), int(col), int(row)) not in available
            for row, col in zip(rows, cols, strict=True)
        ):
            return density
    return replace(density, internal_energy=evaluate_internal_energy(model, density))
