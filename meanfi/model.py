from __future__ import annotations

from types import MappingProxyType

import numpy as np

from meanfi.tb.validate import (
    matrix_array,
    tb_dimension,
    tb_orbital_count,
    validate_hermiticity,
    validate_tb_dict,
    zero_key,
)
from meanfi.meanfield import (
    bdg_correction_from_density_parts,
    meanfield,
)
from meanfi.results import DensityResult
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.tb.bdg import electron_to_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import add_tb, _tb_type


def _validate_reference_density_matrix(
    reference_density_matrix: _tb_type,
    *,
    ndim: int,
    ndof: int,
) -> None:
    for key, value in reference_density_matrix.items():
        if len(key) != ndim:
            raise ValueError(
                "reference_density_matrix keys must match the model dimension"
            )
        if matrix_array(value).shape != (ndof, ndof):
            raise ValueError(
                "reference_density_matrix matrices must match the model shape"
            )


def _validate_reference_density(
    reference: DensityResult,
    *,
    ndim: int,
    ndof: int,
) -> None:
    if not isinstance(reference, DensityResult):
        raise TypeError("reference must be a DensityResult")
    if reference.coordinates.size != ndof:
        raise ValueError("reference density coordinate size must match the model shape")
    if any(len(key) != ndim for key in reference.coordinates.keys):
        raise ValueError("reference density keys must match the model dimension")


class Model:
    """Interacting tight-binding problem at non-negative temperature.

    ``reference`` enables normal-state reference subtraction: the interaction
    correction is built from ``rho - rho_ref`` instead of ``rho``. A selected
    density is sufficient as long as it covers the interaction-required layout.
    """

    _frozen = False

    def __setattr__(self, name, value) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("Model is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        h_0: _tb_type,
        h_int: _tb_type,
        filling: float,
        *,
        kT: float = 0.0,
        superconducting: bool = False,
        spatial_symmetries=(),
        reference: DensityResult | None = None,
        reference_density_matrix: _tb_type | None = None,
    ) -> None:
        validate_tb_dict(h_0)
        validate_tb_dict(h_int)
        validate_hermiticity(h_0)
        validate_hermiticity(h_int)

        if not isinstance(filling, (float, int)) or filling <= 0:
            raise ValueError("filling must be a positive scalar")
        if kT < 0:
            raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
        if reference is not None and reference_density_matrix is not None:
            raise ValueError(
                "reference and reference_density_matrix are mutually exclusive"
            )
        if (
            reference is not None or reference_density_matrix is not None
        ) and superconducting:
            raise ValueError(
                "reference density is only supported for normal-state models"
            )

        object.__setattr__(self, "h_0", MappingProxyType(dict(h_0)))
        object.__setattr__(self, "h_int", MappingProxyType(dict(h_int)))
        object.__setattr__(self, "filling", float(filling))
        object.__setattr__(self, "kT", float(kT))
        object.__setattr__(self, "superconducting", bool(superconducting))
        object.__setattr__(self, "spatial_symmetries", tuple(spatial_symmetries))

        object.__setattr__(self, "_ndim", tb_dimension(h_0))
        object.__setattr__(self, "_ndof", tb_orbital_count(h_0))
        object.__setattr__(self, "_local_key", zero_key(self._ndim))
        if reference is not None:
            _validate_reference_density(
                reference,
                ndim=self._ndim,
                ndof=self._ndof,
            )
        if reference_density_matrix is not None:
            _validate_reference_density_matrix(
                reference_density_matrix,
                ndim=self._ndim,
                ndof=self._ndof,
            )

        from meanfi.space import ActiveSCFSpace

        object.__setattr__(self, "scf_space", ActiveSCFSpace.from_model(self))
        if reference is not None:
            required_values = reference.values_for(self.scf_space.required_coordinates)
            reference_state = ActiveDensityState(
                self.scf_space,
                self.scf_space.params_from_required_entries(required_values),
            )
        elif reference_density_matrix is not None:
            reference_state = ActiveDensityState(
                self.scf_space,
                self.scf_space.params_from_meanfield_input(reference_density_matrix),
            )
        else:
            reference_state = None
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "_reference_state", reference_state)
        object.__setattr__(self, "_frozen", True)

    def _density_state(self, rho: _tb_type | DensityResult) -> ActiveDensityState:
        if isinstance(rho, DensityResult):
            params = self.scf_space.params_from_required_entries(
                rho.values_for(self.scf_space.required_coordinates)
            )
        else:
            params = self.scf_space.params_from_meanfield_input(rho)
        return ActiveDensityState(
            self.scf_space,
            params,
        )

    def _active_density_from_state(self, state: ActiveDensityState) -> _tb_type:
        require_same_space(state, self.scf_space)
        return self.scf_space.meanfield_input_from_params(state.values)

    def _reference_difference(
        self,
        state: ActiveDensityState,
    ) -> ActiveDensityState:
        require_same_space(state, self.scf_space)
        return state.relative_to(self._reference_state)

    def hamiltonian_from_rho(self, rho: _tb_type | DensityResult) -> _tb_type:
        """Return the interacting Hamiltonian implied by a trial density matrix."""

        difference = self._reference_difference(self._density_state(rho))
        correction = meanfield(
            self._active_density_from_state(difference),
            self.h_int,
        )
        return add_tb(self.h_0, correction)

    def hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        """Return the full Hamiltonian for a trial mean-field correction."""

        return add_tb(self.h_0, mf)

    def bdg_hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        """Return the unshifted electron-first BdG Hamiltonian for a mean-field correction."""

        if not self.superconducting:
            raise ValueError(
                "bdg_hamiltonian_from_meanfield requires superconducting=True"
            )
        validate_bdg_tb(
            mf,
            ndof=self._ndof,
            ndim=self._ndim,
            name="BdG correction",
        )
        return add_tb(electron_to_bdg_tb(self.h_0, self._ndof), mf)

    def random_meanfield(self, rng=None, scale: float = 1.0) -> _tb_type:
        """Sample a solver-ready mean-field correction in this model's SCF space."""

        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        params = float(scale) * generator.standard_normal(self.scf_space.num_params)
        meanfield_input = self.scf_space.meanfield_input_from_params(params)
        if self.superconducting:
            return bdg_correction_from_density_parts(
                meanfield_input,
                h_int=self.h_int,
                ndof=self._ndof,
                ndim=self._ndim,
            )
        return meanfield(meanfield_input, self.h_int)
