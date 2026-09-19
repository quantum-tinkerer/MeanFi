"""Relative thermodynamic-energy comparisons for EDIIS, without retaining meshes."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from meanfi.meanfield import correction_expectation, interaction_energy
from meanfi.model import Model
from meanfi.scf.problem import SCFEvaluation
from meanfi.scf.ediis import EDIISPoint
from meanfi.tb.ops import _tb_type


@dataclass(frozen=True)
class EnergySample:
    """One history density, its generating field and scalar thermodynamics."""

    params: np.ndarray
    field: _tb_type
    interaction: float
    filling: float
    mu: float
    internal_energy: float | None
    entropy: float | None = None


def energy_sample(model: Model, evaluation: SCFEvaluation) -> EnergySample:
    state = evaluation.output_state
    difference = model._active_density_from_state(model._reference_difference(state))
    density = evaluation.density
    filling = density.density_filling
    if filling is None:
        filling = density.entries.trace(model._electron_ndof)
    if filling is None:
        filling = density.filling
    return EnergySample(
        params=state.values,
        field=evaluation.mean_field,
        interaction=interaction_energy(
            difference, model._h_int, electron_ndof=model._electron_ndof
        ),
        filling=filling,
        mu=density.mu,
        internal_energy=evaluation.internal_energy,
        entropy=density.entropy,
    )


def relative_energy(model: Model, left: EnergySample, right: EnergySample) -> float:
    """Compare states at target filling, per physical orbital.

    Use integrated energies when available, subtracting T*S at finite
    temperature. Extrapolate each sample to target filling with -mu*dN.
    Otherwise use zero-temperature endpoint quadrature of the density response;
    its filling correction is already included.
    """
    size = model._ndof
    target = model.filling
    if left.internal_energy is not None and right.internal_energy is not None:
        difference = right.internal_energy - left.internal_energy
        if model.kT > 0:
            if left.entropy is None or right.entropy is None:
                raise ValueError("Finite-temperature EDIIS requires entropy")
            difference -= model.kT * (right.entropy - left.entropy)
        difference -= (
            right.mu * (right.filling - target) - left.mu * (left.filling - target)
        ) / size
        return float(difference)
    if model.kT != 0:
        raise ValueError(
            "Finite-temperature EDIIS requires integrated energies and entropy"
        )
    difference = model._space.density_from_params(right.params - left.params)
    response = (
        sum(
            correction_expectation(
                difference, field, electron_ndof=model._electron_ndof
            )
            for field in (left.field, right.field)
        )
        / 2
    )
    filling_term = (
        (target - (left.filling + right.filling) / 2) * (right.mu - left.mu) / size
    )
    return float(right.interaction - left.interaction - response + filling_term)


def comparison_points(
    model: Model, history: Sequence[EnergySample]
) -> list[EDIISPoint]:
    """Re-anchor the retained history instead of accumulating iteration errors."""
    anchor = history[0]
    return [
        EDIISPoint(sample.params, relative_energy(model, anchor, sample))
        for sample in history
    ]
