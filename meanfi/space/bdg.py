from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from meanfi.meanfield import bdg_correction_from_density, bdg_density_keys
from meanfi.model import Model
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb, zero_bdg_array
from meanfi.space.params import (
    canonical_tb_keys,
    complex_to_real,
    independent_hopping_keys,
    onsite_key,
    real_to_complex,
)
from meanfi.space.hermitian import (
    full_hermitian_tb_to_real_params,
    hermitian_param_count,
    real_params_to_full_hermitian_tb,
    real_params_to_selected_hermitian_tb,
    selected_hermitian_tb_to_real_params,
)
from meanfi.space.particlehole import (
    BdGElectronAnomalousSelection,
    bdg_density_selection_from_top_half,
    bdg_top_half_selection,
    electron_and_anomalous_blocks,
)
from meanfi.space.density_selection import (
    DensitySelection,
)
from meanfi.tb.ops import _tb_type, as_sparse, is_sparse_like
from meanfi.tb.validate import validate_bdg_state


def _bdg_top_half_blocks_to_real_params(
    tb: _tb_type,
    *,
    selection: BdGElectronAnomalousSelection,
    ndof: int,
) -> np.ndarray:
    """Pack selected electron and anomalous top-half BdG blocks as real params."""

    zero = np.zeros((2 * ndof, 2 * ndof), dtype=complex)
    electron_tb = {
        key: electron_and_anomalous_blocks(tb.get(key, zero), ndof)[0]
        for key in selection.electron.keys
    }
    anomalous_tb = {
        key: electron_and_anomalous_blocks(tb.get(key, zero), ndof)[1]
        for key in selection.anomalous.keys
    }
    electron_params = selected_hermitian_tb_to_real_params(
        electron_tb,
        selection.electron,
    )
    anomalous_params = complex_to_real(selection.anomalous.values_from_tb(anomalous_tb))
    if anomalous_params.size == 0:
        return electron_params
    return np.concatenate((electron_params, anomalous_params))


def _real_params_to_bdg_top_half_blocks(
    params: np.ndarray,
    *,
    selection: BdGElectronAnomalousSelection,
    ndof: int,
) -> tuple[_tb_type, _tb_type]:
    """Unpack real params into selected electron and anomalous TB blocks."""

    params = np.asarray(params, dtype=float).reshape(-1)
    electron_size = hermitian_param_count(selection.electron, ndof)
    electron_tb = real_params_to_selected_hermitian_tb(
        params[:electron_size],
        selection.electron,
        ndof,
    )
    anomalous_count = selection.anomalous.value_count
    anomalous_params = params[electron_size : electron_size + 2 * anomalous_count]
    anomalous_tb = selection.anomalous.values_to_tb(real_to_complex(anomalous_params))
    offset = electron_size + 2 * anomalous_count
    if offset != len(params):
        raise ValueError(
            "tb_params has the wrong length for the requested BdG selection"
        )
    return electron_tb, anomalous_tb


def _top_half_blocks_to_density_tb(
    electron_tb: _tb_type,
    anomalous_tb: _tb_type,
    *,
    keys: tuple[tuple[int, ...], ...],
    ndof: int,
) -> _tb_type:
    density: _tb_type = {}
    zero_e = np.zeros((ndof, ndof), dtype=complex)
    for key in keys:
        block = np.zeros((2 * ndof, 2 * ndof), dtype=complex)
        block[:ndof, :ndof] = electron_tb.get(key, zero_e)
        block[:ndof, ndof:] = anomalous_tb.get(key, zero_e)
        density[key] = block
    return density


def bdg_tb_to_rparams(
    tb: _tb_type,
    ndof: int,
    selection: BdGElectronAnomalousSelection | None = None,
) -> np.ndarray:
    validate_bdg_state(tb, ndof=ndof)
    if selection is not None:
        return _bdg_top_half_blocks_to_real_params(tb, selection=selection, ndof=ndof)

    ordered_keys = canonical_tb_keys(tb.keys())
    normal_block = {}
    anomalous_parts = []
    for key in ordered_keys:
        normal, anomalous = electron_and_anomalous_blocks(tb[key], ndof)
        normal_block[key] = normal
        anomalous_parts.append(complex_to_real(anomalous.reshape(-1)))

    normal_params = full_hermitian_tb_to_real_params(normal_block)
    return np.concatenate((normal_params, *anomalous_parts))


def rparams_to_bdg_tb(
    tb_params: np.ndarray,
    tb_keys: list[tuple[None] | tuple[int, ...]],
    ndof: int,
    selection: BdGElectronAnomalousSelection | None = None,
) -> _tb_type:
    if selection is not None:
        normal_block, anomalous_block = _real_params_to_bdg_top_half_blocks(
            tb_params,
            selection=selection,
            ndof=ndof,
        )
        tb = assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)
        validate_bdg_state(tb, ndof=ndof)
        return tb

    ordered_keys = canonical_tb_keys(tb_keys)
    n_onsite = ndof * ndof
    n_hopping = len(independent_hopping_keys(ordered_keys)) * 2 * ndof * ndof
    normal_size = n_onsite + n_hopping

    params = np.asarray(tb_params, dtype=float).reshape(-1)
    normal_block = real_params_to_full_hermitian_tb(
        params[:normal_size],
        ordered_keys,
        ndof,
    )

    block_size = 2 * ndof * ndof
    offset = normal_size
    anomalous_block = {}
    for key in ordered_keys:
        anomalous = real_to_complex(params[offset : offset + block_size]).reshape(
            ndof, ndof
        )
        anomalous_block[key] = anomalous
        offset += block_size

    if offset != len(params):
        raise ValueError(
            "tb_params has the wrong length for the requested BdG selection"
        )

    tb = assemble_bdg_tb(normal_block, anomalous_block, ndof=ndof)
    validate_bdg_state(tb, ndof=ndof)
    return tb


def bdg_density_to_rparams(
    density_matrix: _tb_type,
    *,
    selection: BdGElectronAnomalousSelection,
    ndof: int,
) -> np.ndarray:
    return _bdg_top_half_blocks_to_real_params(
        density_matrix,
        selection=selection,
        ndof=ndof,
    )


def rparams_to_bdg_density(
    params: np.ndarray,
    *,
    selection: BdGElectronAnomalousSelection,
    ndof: int,
) -> _tb_type:
    electron_density, anomalous_density = _real_params_to_bdg_top_half_blocks(
        params,
        selection=selection,
        ndof=ndof,
    )
    return _top_half_blocks_to_density_tb(
        electron_density,
        anomalous_density,
        keys=selection.electron.keys,
        ndof=ndof,
    )


@dataclass
class BdGMeanFieldDensitySpace:
    model: Model
    onsite: tuple[int, ...]
    density_keys: list[tuple[int, ...]]
    top_half_selection: BdGElectronAnomalousSelection
    density_selection: DensitySelection
    active_keys: list[tuple[int, ...]]

    @classmethod
    def from_model(cls, model: Model) -> BdGMeanFieldDensitySpace:
        onsite = onsite_key(model._ndim)
        density_keys = bdg_density_keys(model, {})
        top_half_selection = bdg_top_half_selection(
            keys=density_keys,
            interaction_tb=model.h_int,
            ndof=model._ndof,
            local_key=onsite,
        )
        density_selection = bdg_density_selection_from_top_half(
            top_half_selection,
            ndof=model._ndof,
        )
        return cls(
            model=model,
            onsite=onsite,
            density_keys=density_keys,
            top_half_selection=top_half_selection,
            density_selection=density_selection,
            active_keys=[],
        )

    def projected_active_keys(self, guess: _tb_type) -> list[tuple[int, ...]]:
        return canonical_tb_keys(set(guess) | {self.onsite})

    def fill_active_bdg_keys(self, tb: _tb_type) -> _tb_type:
        zero = zero_bdg_array(self.model._ndof)
        use_sparse = any(is_sparse_like(value) for value in tb.values())
        if use_sparse:
            zero_sparse = as_sparse(zero)
            return {
                key: as_sparse(tb.get(key, zero_sparse)) for key in self.active_keys
            }
        return {
            key: np.asarray(tb.get(key, zero), dtype=complex)
            for key in self.active_keys
        }

    def validated_meanfield(self, density_matrix: _tb_type) -> _tb_type:
        updated = self.fill_active_bdg_keys(
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
        self.active_keys = self.projected_active_keys(guess)
        projected = rparams_to_bdg_tb(
            bdg_tb_to_rparams(
                guess, self.model._ndof, selection=self.top_half_selection
            ),
            self.active_keys,
            self.model._ndof,
            selection=self.top_half_selection,
        )
        return self.fill_active_bdg_keys(projected)

    def density_selection_for(self, meanfield: _tb_type):
        if self.density_selection.value_count == 0:
            return None
        if any(is_sparse_like(value) for value in meanfield.values()):
            return self.density_selection
        return None

    def params_from_density(self, density_matrix: _tb_type) -> np.ndarray:
        return np.asarray(
            bdg_density_to_rparams(
                density_matrix,
                selection=self.top_half_selection,
                ndof=self.model._ndof,
            ),
            dtype=float,
        )

    def density_from_params(self, params: np.ndarray) -> _tb_type:
        return rparams_to_bdg_density(
            params,
            selection=self.top_half_selection,
            ndof=self.model._ndof,
        )

    def meanfield_from_density(self, density_matrix: _tb_type) -> _tb_type:
        return self.validated_meanfield(density_matrix)
