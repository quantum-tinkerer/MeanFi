from __future__ import annotations

import numpy as np

from meanfi.integrate.engines.bdg import solve_bdg_density_fixed_filling
from meanfi.model import Model
from meanfi.physics.bdg import (
    bdg_correction_from_density,
    bdg_density_keys,
    validate_bdg_tb,
    zero_bdg_array,
)
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import SolverRuntime, warn_on_projection
from meanfi.state.bdg import (
    bdg_density_to_rparams,
    bdg_tb_to_rparams,
    rparams_to_bdg_density,
    rparams_to_bdg_tb,
)
from meanfi.state.keys import canonical_tb_keys
from meanfi.state.support import bdg_density_entry_support, bdg_top_half_support
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like


class BdGFamilyAdapter:
    def __init__(self, model: Model, runtime: SolverRuntime) -> None:
        self.model = model
        self.runtime = runtime
        self.onsite = (0,) * model._ndim
        self.density_keys = bdg_density_keys(model, {})
        self.top_half_support = bdg_top_half_support(
            keys=self.density_keys,
            interaction_support=model.h_int,
            ndof=model._ndof,
            local_key=self.onsite,
        )
        self.density_support = bdg_density_entry_support(
            keys=self.density_keys,
            interaction_support=model.h_int,
            ndof=model._ndof,
            local_key=self.onsite,
        )
        self.support_keys: list[tuple[int, ...]] = []

    def _fill_bdg_support(self, tb: _tb_type) -> _tb_type:
        zero = zero_bdg_array(self.model._ndof)
        use_sparse = any(is_sparse_like(value) for value in tb.values())
        if use_sparse:
            zero_sparse = as_sparse(zero)
            return {
                key: as_sparse(tb.get(key, zero_sparse)) for key in self.support_keys
            }
        return {
            key: np.asarray(tb.get(key, zero), dtype=complex)
            for key in self.support_keys
        }

    def _validated_bdg_update(self, density_matrix: _tb_type) -> _tb_type:
        updated = self._fill_bdg_support(
            bdg_correction_from_density(density_matrix, self.model)
        )
        validate_bdg_tb(
            updated,
            ndof=self.model._ndof,
            ndim=self.model._ndim,
            name="BdG correction",
        )
        return updated

    def project_guess(self, guess: _tb_type) -> _tb_type:
        validate_bdg_tb(
            guess,
            ndof=self.model._ndof,
            ndim=self.model._ndim,
            name="BdG correction",
        )
        self.support_keys = canonical_tb_keys(set(guess) | {self.onsite})
        projected = rparams_to_bdg_tb(
            bdg_tb_to_rparams(guess, self.model._ndof, support=self.top_half_support),
            self.support_keys,
            self.model._ndof,
            support=self.top_half_support,
        )
        projected = self._fill_bdg_support(projected)
        warn_on_projection(guess, projected, label="BdG SCF guess")
        return projected

    def _sparse_density_support(self, meanfield: _tb_type):
        if self.density_support.output_size == 0:
            return None
        if any(is_sparse_like(value) for value in meanfield.values()):
            return self.density_support
        return None

    def _evaluate_meanfield(
        self,
        meanfield_guess: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityMatrixResult:
        return solve_bdg_density_fixed_filling(
            self.model,
            meanfield_guess,
            keys=self.density_keys,
            integration=self.runtime.integration,
            filling_tol=self.runtime.filling_tol,
            mu_tol=self.runtime.mu_tol,
            max_charge_evaluations=self.runtime.max_charge_evaluations,
            mu_guess=mu_guess,
            density_entry_support=self._sparse_density_support(meanfield_guess),
        )

    def evaluate_projected_guess(
        self, projected_guess: _tb_type
    ) -> DensityMatrixResult:
        return self._evaluate_meanfield(projected_guess, mu_guess=0.0)

    def params_from_density_result(
        self, density_result: DensityMatrixResult
    ) -> np.ndarray:
        return np.asarray(
            bdg_density_to_rparams(
                density_result.density_matrix,
                support=self.top_half_support,
                ndof=self.model._ndof,
            ),
            dtype=float,
        )

    def evaluate_params(
        self,
        params: np.ndarray,
        *,
        mu_guess: float,
    ) -> DensityMatrixResult:
        density_state = rparams_to_bdg_density(
            params,
            support=self.top_half_support,
            ndof=self.model._ndof,
        )
        return self._evaluate_meanfield(
            self._validated_bdg_update(density_state),
            mu_guess=mu_guess,
        )

    def finalize_meanfield(self, density_result: DensityMatrixResult) -> _tb_type:
        return self._validated_bdg_update(density_result.density_matrix)
