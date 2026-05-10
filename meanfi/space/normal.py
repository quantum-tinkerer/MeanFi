from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.meanfield import meanfield
from meanfi.model import Model
from meanfi.space.density_selection import DensitySelection
from meanfi.space.hermitian import (
    real_params_to_selected_hermitian_tb,
    selected_hermitian_tb_to_real_params,
)
from meanfi.space.interaction_selection import normal_density_selection
from meanfi.tb.ops import _tb_type, is_sparse_like
from meanfi.tb.storage import match_tb_storage, prefers_sparse_storage


@dataclass
class NormalMeanFieldDensitySpace:
    model: Model
    interaction_keys: list[tuple[int, ...]]
    density_selection: DensitySelection
    _prefer_sparse: bool = False

    @classmethod
    def from_model(cls, model: Model) -> NormalMeanFieldDensitySpace:
        interaction_keys = list(model.h_int)
        active_keys = (
            interaction_keys
            if model._local_key in interaction_keys
            else [*interaction_keys, model._local_key]
        )
        density_selection = normal_density_selection(
            keys=active_keys,
            interaction_tb=model.h_int,
            ndof=model._ndof,
            local_key=model._local_key,
            allow_empty=True,
        )
        if (
            density_selection is None
        ):  # pragma: no cover - allow_empty guarantees selection
            raise ValueError("Normal density space unexpectedly missing")
        return cls(
            model=model,
            interaction_keys=interaction_keys,
            density_selection=density_selection,
        )

    def project_guess(self, guess: _tb_type) -> _tb_type:
        self._prefer_sparse = prefers_sparse_storage(
            getattr(self.model, "h_0", None),
            self.model.h_int,
            guess,
        )
        projected = real_params_to_selected_hermitian_tb(
            selected_hermitian_tb_to_real_params(guess, self.density_selection),
            self.density_selection,
            self.model._ndof,
        )
        return match_tb_storage(projected, like_sparse=self._prefer_sparse)

    def density_selection_for(self, hamiltonian: _tb_type):
        if self.density_selection.value_count == 0:
            return None
        if any(is_sparse_like(matrix) for matrix in hamiltonian.values()):
            return self.density_selection
        return None

    def params_from_density(self, density_matrix: _tb_type) -> np.ndarray:
        zero = np.zeros((self.model._ndof, self.model._ndof), dtype=complex)
        density_reduced = {
            key: density_matrix.get(key, zero) for key in self.density_selection.keys
        }
        return np.asarray(
            selected_hermitian_tb_to_real_params(
                density_reduced,
                self.density_selection,
            ),
            dtype=float,
        )

    def density_from_params(self, params: np.ndarray) -> _tb_type:
        return real_params_to_selected_hermitian_tb(
            params,
            self.density_selection,
            self.model._ndof,
        )

    def meanfield_from_density(
        self, density_matrix: _tb_type, *, mu: float = 0.0
    ) -> _tb_type:
        density_reduced = {key: density_matrix[key] for key in self.interaction_keys}
        result = dict(meanfield(density_reduced, self.model.h_int))
        result[self.model._local_key] = result.get(
            self.model._local_key,
            np.zeros((self.model._ndof, self.model._ndof), dtype=complex),
        ) - float(mu) * np.eye(self.model._ndof)
        return result
