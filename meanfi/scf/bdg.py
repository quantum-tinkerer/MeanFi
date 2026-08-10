from __future__ import annotations

import numpy as np

from meanfi.density.integrate.bdg import solve_bdg_density_fixed_filling
from meanfi.density.internal import DensityEvaluation, DensitySlice
from meanfi.meanfield import bdg_correction_from_density_parts
from meanfi.model import Model
from meanfi.scf.engine import SCFProblem, SolverRuntime, warn_on_projection
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.space.support import active_tb_keys
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb, zero_bdg_array
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like


def build_bdg_scf_problem(model: Model, runtime: SolverRuntime) -> SCFProblem:
    """Build the superconducting map consumed by the generic SCF engine."""

    space = model.scf_space
    active_keys = active_tb_keys({space.onsite})

    def project_guess(guess: _tb_type) -> _tb_type:
        nonlocal active_keys
        validate_bdg_tb(
            guess,
            ndof=model._ndof,
            ndim=model._ndim,
            name="BdG correction",
        )
        active_keys = active_tb_keys(set(guess) | {space.onsite})
        projected = _assemble_bdg_active_tb(
            space.project_meanfield_input(guess),
            ndof=model._ndof,
            ndim=model._ndim,
            keys=space.active_coordinates.keys,
        )
        projected = _fill_bdg_keys(projected, active_keys=active_keys, ndof=model._ndof)
        warn_on_projection(guess, projected, label="BdG SCF guess")
        return projected

    def evaluate_meanfield(
        meanfield_guess: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityEvaluation:
        return solve_bdg_density_fixed_filling(
            model,
            meanfield_guess,
            keys=space.density_keys,
            integration=runtime.integration,
            filling_tol=runtime.tolerances.filling_residual,
            mu_tol=runtime.mu_tol,
            max_charge_evaluations=runtime.max_charge_evaluations,
            mu_guess=mu_guess,
            density_coordinates=space.required_coordinates,
        )

    def evaluate_projected_guess(projected_guess: _tb_type) -> DensityEvaluation:
        return evaluate_meanfield(projected_guess, mu_guess=0.0)

    def state_from_density(density: DensitySlice) -> ActiveDensityState:
        if density.coordinates.entries != space.required_coordinates.entries:
            raise ValueError("density slice does not match the BdG SCF space")
        return ActiveDensityState(
            space,
            space.params_from_required_entries(density.values),
        )

    def active_density(state: ActiveDensityState) -> _tb_type:
        require_same_space(state, space)
        return space.meanfield_input_from_params(state.values)

    def mean_field_from_state(state: ActiveDensityState) -> _tb_type:
        return _bdg_meanfield_from_active_density(
            active_density(state),
            model=model,
            active_keys=active_keys,
        )

    def evaluate_state(state: ActiveDensityState, mu_guess: float) -> DensityEvaluation:
        return evaluate_meanfield(
            mean_field_from_state(state),
            mu_guess=mu_guess,
        )

    return SCFProblem(
        runtime=runtime,
        state_space=space,
        project_guess=project_guess,
        evaluate_projected_guess=evaluate_projected_guess,
        state_from_density=state_from_density,
        evaluate_state=evaluate_state,
        mean_field_from_state=mean_field_from_state,
    )


def _bdg_meanfield_from_active_density(
    active_density: _tb_type,
    *,
    model: Model,
    active_keys: list[tuple[int, ...]],
) -> _tb_type:
    updated = _fill_bdg_keys(
        bdg_correction_from_density_parts(
            active_density,
            h_int=model.h_int,
            ndof=model._ndof,
            ndim=model._ndim,
        ),
        active_keys=active_keys,
        ndof=model._ndof,
    )
    validate_bdg_tb(
        updated,
        ndof=model._ndof,
        ndim=model._ndim,
        name="BdG correction",
    )
    return updated


def _assemble_bdg_active_tb(
    active_tb: _tb_type,
    *,
    ndof: int,
    ndim: int,
    keys: tuple[tuple[int, ...], ...],
) -> _tb_type:
    normal_block = {}
    anomalous_block = {}
    zero = np.zeros((2 * ndof, 2 * ndof), dtype=complex)
    for key in keys:
        block = np.asarray(active_tb.get(key, zero), dtype=complex)
        normal_block[key] = block[:ndof, :ndof]
        anomalous_block[key] = block[:ndof, ndof:]
    tb = assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)
    validate_bdg_tb(tb, ndof=ndof, ndim=ndim, name="BdG correction")
    return tb


def _fill_bdg_keys(
    tb: _tb_type,
    *,
    active_keys: list[tuple[int, ...]],
    ndof: int,
) -> _tb_type:
    zero = zero_bdg_array(ndof)
    use_sparse = any(is_sparse_like(value) for value in tb.values())
    if use_sparse:
        zero_sparse = as_sparse(zero)
        return {key: as_sparse(tb.get(key, zero_sparse)) for key in active_keys}
    return {key: np.asarray(tb.get(key, zero), dtype=complex) for key in active_keys}
