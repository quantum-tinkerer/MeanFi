from __future__ import annotations

import inspect

import numpy as np

from meanfi.integrate.dispatch import solve_density_matrix_fixed_filling
from meanfi.integrate.methods import IntegrationMethod
from meanfi.model import Model
from meanfi.physics.meanfield import meanfield
from meanfi.results import DensityMatrixResult
from meanfi.scf.engine import (
    SolverRuntime,
    _prefer_sparse,
    restore_tb_type,
    warn_on_projection,
)
from meanfi.state.normal import rparams_to_tb, tb_to_rparams
from meanfi.state.support import normal_density_entry_support
from meanfi.tb.ops import _tb_type, is_sparse_like


def _default_density_for_hamiltonian(
    model: Model,
    hamiltonian: _tb_type,
    *,
    keys: list[tuple[int, ...]],
    integration: IntegrationMethod,
    filling_tol: float | None,
    mu_tol: float,
    max_charge_evaluations: int | None,
    mu_guess: float,
    density_entry_support=None,
) -> DensityMatrixResult:
    return solve_density_matrix_fixed_filling(
        hamiltonian,
        filling=model.filling,
        kT=model.kT,
        keys=keys,
        integration=integration,
        filling_tol=filling_tol,
        mu_tol=mu_tol,
        max_charge_evaluations=max_charge_evaluations,
        mu_guess=mu_guess,
        density_entry_support=density_entry_support,
    )


class NormalFamilyAdapter:
    def __init__(
        self,
        model: Model,
        runtime: SolverRuntime,
        *,
        tb_to_rparams_fn=tb_to_rparams,
        rparams_to_tb_fn=rparams_to_tb,
        meanfield_fn=meanfield,
        density_for_hamiltonian_fn=None,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.keys = list(model.h_int)
        support_keys = (
            self.keys
            if model._local_key in self.keys
            else [*self.keys, model._local_key]
        )
        self.param_support = normal_density_entry_support(
            keys=support_keys,
            interaction_support=model.h_int,
            ndof=model._ndof,
            local_key=model._local_key,
            allow_empty=True,
        )
        self._tb_to_rparams = tb_to_rparams_fn
        self._rparams_to_tb = rparams_to_tb_fn
        self._meanfield = meanfield_fn
        self._density_for_hamiltonian = (
            _default_density_for_hamiltonian
            if density_for_hamiltonian_fn is None
            else density_for_hamiltonian_fn
        )
        self._prefer_sparse = False

    def project_guess(self, guess: _tb_type) -> _tb_type:
        self._prefer_sparse = _prefer_sparse(
            getattr(self.model, "h_0", None),
            self.model.h_int,
            guess,
        )
        projected = self._rparams_to_tb(
            self._tb_to_rparams(guess, support=self.param_support),
            list(self.param_support.keys),
            self.model._ndof,
            support=self.param_support,
        )
        warn_on_projection(guess, projected, label="Normal SCF guess")
        return restore_tb_type(projected, prefer_sparse=self._prefer_sparse)

    def _density_entry_support(self, hamiltonian: _tb_type):
        if self.param_support is None or self.param_support.output_size == 0:
            return None
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            return self.param_support
        return None

    def _evaluate_hamiltonian(
        self,
        hamiltonian: _tb_type,
        *,
        mu_guess: float,
    ) -> DensityMatrixResult:
        kwargs = dict(
            keys=self.keys,
            integration=self.runtime.integration,
            filling_tol=self.runtime.filling_tol,
            mu_tol=self.runtime.mu_tol,
            max_charge_evaluations=self.runtime.max_charge_evaluations,
            mu_guess=mu_guess,
        )
        if (
            "density_entry_support"
            in inspect.signature(self._density_for_hamiltonian).parameters
        ):
            kwargs["density_entry_support"] = self._density_entry_support(hamiltonian)
        return self._density_for_hamiltonian(self.model, hamiltonian, **kwargs)

    def evaluate_projected_guess(
        self, projected_guess: _tb_type
    ) -> DensityMatrixResult:
        return self._evaluate_hamiltonian(
            self.model.hamiltonian_from_meanfield(projected_guess),
            mu_guess=0.0,
        )

    def params_from_density_result(
        self, density_result: DensityMatrixResult
    ) -> np.ndarray:
        density_guess = {key: density_result.density_matrix[key] for key in self.keys}
        return np.asarray(
            self._tb_to_rparams(density_guess, support=self.param_support),
            dtype=float,
        )

    def evaluate_params(
        self,
        params: np.ndarray,
        *,
        mu_guess: float,
    ) -> DensityMatrixResult:
        rho = self._rparams_to_tb(
            params,
            self.keys,
            self.model._ndof,
            support=self.param_support,
        )
        return self._evaluate_hamiltonian(
            self.model.hamiltonian_from_rho(rho),
            mu_guess=mu_guess,
        )

    def finalize_meanfield(self, density_result: DensityMatrixResult) -> _tb_type:
        density_reduced = {key: density_result.density_matrix[key] for key in self.keys}
        mf_result = self._meanfield(density_reduced, self.model.h_int)
        tb_result = dict(mf_result)
        tb_result[self.model._local_key] = tb_result.get(
            self.model._local_key,
            np.zeros((self.model._ndof, self.model._ndof), dtype=complex),
        ) - density_result.mu * np.eye(self.model._ndof)
        return tb_result
