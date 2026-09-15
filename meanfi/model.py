from __future__ import annotations

from dataclasses import dataclass, field, KW_ONLY

import numpy as np

from meanfi.meanfield import bdg_correction_from_density_parts, meanfield
from meanfi.results import DensityResult
from meanfi.space.space import ActiveSCFSpace
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.space.symmetry import SpatialSymmetry
from meanfi.tb.bdg import electron_to_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import add_tb, _tb_type
from meanfi.tb.storage import prefers_sparse_storage
from meanfi.tb.validate import freeze_tb, tb_dimension, tb_orbital_count, zero_key


@dataclass(frozen=True, eq=False)
class Model:
    """Owned, read-only tight-binding inputs and their reduced SCF space.

    ``reference`` subtracts a normal reference density from the complete
    Hartree/Fock correction. Filling counts electrons per unit cell.
    """

    h_0: _tb_type
    h_int: _tb_type
    filling: float
    _: KW_ONLY
    kT: float = 0.0
    superconducting: bool = False
    spatial_symmetries: tuple[SpatialSymmetry, ...] = ()
    reference: DensityResult | None = None
    scf_space: ActiveSCFSpace = field(init=False, repr=False)
    _reference_state: ActiveDensityState | None = field(init=False, repr=False)
    _ndim: int = field(init=False, repr=False)
    _ndof: int = field(init=False, repr=False)
    _local_key: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        h_0, h_int = freeze_tb(self.h_0), freeze_tb(self.h_int)
        ndim, ndof = tb_dimension(h_0), tb_orbital_count(h_0)
        if tb_dimension(h_int) != ndim or tb_orbital_count(h_int) != ndof:
            raise ValueError(
                "Hamiltonian and interaction must have the same dimension and matrix size"
            )
        filling, kT = float(self.filling), float(self.kT)
        if not np.isfinite(filling) or not 0 <= filling <= ndof:
            raise ValueError(
                "filling must be finite and between zero and the orbital count"
            )
        if not np.isfinite(kT) or kT < 0:
            raise ValueError("kT must be finite and non-negative")
        symmetries = tuple(self.spatial_symmetries)
        for symmetry in symmetries:
            if symmetry.lattice_matrix.shape != (ndim, ndim) or any(
                matrix.shape != (ndof, ndof)
                for matrix in symmetry.unitaries_by_shift.values()
            ):
                raise ValueError(
                    "Spatial symmetry must match the model dimension and orbital count"
                )
        for name, value in dict(
            h_0=h_0,
            h_int=h_int,
            filling=filling,
            kT=kT,
            spatial_symmetries=symmetries,
            _ndim=ndim,
            _ndof=ndof,
            _local_key=zero_key(ndim),
        ).items():
            object.__setattr__(self, name, value)
        space = ActiveSCFSpace.from_interaction(
            h_int,
            superconducting=self.superconducting,
            spatial_symmetries=symmetries,
            sparse=prefers_sparse_storage(h_0, h_int),
        )
        reference_state = None
        if self.reference is not None:
            if self.superconducting:
                raise ValueError(
                    "reference density is supported only for normal models"
                )
            if not isinstance(self.reference, DensityResult):
                raise TypeError("reference must be a DensityResult")
            reference_state = ActiveDensityState(
                space,
                space.params_from_required_entries(
                    self.reference.values_for(space.required_coordinates)
                ),
            )
        object.__setattr__(self, "scf_space", space)
        object.__setattr__(self, "_reference_state", reference_state)

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

    def _mean_field_from_state(self, state: ActiveDensityState) -> _tb_type:
        active = self._active_density_from_state(self._reference_difference(state))
        if self.superconducting:
            return bdg_correction_from_density_parts(
                active, h_int=self.h_int, ndof=self._ndof, ndim=self._ndim
            )
        return meanfield(active, self.h_int)

    def hamiltonian_from_density(self, density: _tb_type | DensityResult) -> _tb_type:
        """Build the normal or BdG Hamiltonian from a trial density.

        Selected results must cover this model's required coordinates.
        Normal models subtract their reference before computing the correction.
        """
        return self.hamiltonian_from_meanfield(
            self._mean_field_from_state(self._density_state(density))
        )

    def hamiltonian_from_meanfield(
        self, mean_field: _tb_type | None = None
    ) -> _tb_type:
        """Build the unshifted normal or electron-first BdG Hamiltonian.

        Omitting ``mean_field`` returns the noninteracting Hamiltonian.
        Chemical potential is applied during density evaluation.
        """
        if not self.superconducting:
            return add_tb(self.h_0, mean_field or {})
        if mean_field is not None:
            validate_bdg_tb(
                mean_field, ndof=self._ndof, ndim=self._ndim, name="BdG correction"
            )
        return add_tb(electron_to_bdg_tb(self.h_0, self._ndof), mean_field or {})

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
