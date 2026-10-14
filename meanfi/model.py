from __future__ import annotations

from dataclasses import dataclass, field, KW_ONLY
from collections.abc import Mapping

import numpy as np

from meanfi.hamiltonian import (
    BlochHamiltonian,
    Hamiltonian,
    add_correction,
    electron_to_bdg,
    hamiltonian_dimension,
    hamiltonian_size,
)
from meanfi.interaction import BilinearInteraction
from meanfi.meanfield import interaction_correction
from meanfi.results import DensityResult
from meanfi.space.coordinates import DensityCoordinates
from meanfi.space.space import ActiveSCFSpace
from meanfi.space.state import ActiveDensityState, require_same_space
from meanfi.space.symmetry import SpatialSymmetry
from meanfi.tb.bdg import assemble_bdg_tb, validate_bdg_tb
from meanfi.tb.ops import _tb_type
from meanfi.tb.storage import _MatrixView, prefers_sparse_storage
from meanfi.tb.validate import (
    freeze_tb,
    tb_dimension,
    tb_orbital_count,
    validate_tb_dict,
    validate_hermiticity,
)


@dataclass(frozen=True, eq=False)
class Model:
    """Owned physical inputs and their reduced SCF space.

    Public matrix containers share read-only arrays with owned storage; replacing
    their arrays cannot change the model. Use ``dataclasses.replace`` for changes.

    ``h_0`` accepts a tight-binding dictionary or ``BlochHamiltonian``.
    ``h_int`` accepts a density-density dictionary or ``BilinearInteraction``.
    Both support normal or superconducting states. Callable parameters must stay
    fixed throughout a calculation; MeanFi cannot own their captured state.

    Positive density-density coefficients are repulsive; negative ones are
    attractive, including in the superconducting pairing channel.

    ``reference`` subtracts the normal and pairing reference densities from
    the mean-field correction. A normal reference has zero pairing in a BdG
    model. Filling counts electrons per unit cell.
    """

    h_0: Hamiltonian
    h_int: _tb_type | BilinearInteraction
    filling: float
    _: KW_ONLY
    kT: float = 0.0
    superconducting: bool = False
    spatial_symmetries: tuple[SpatialSymmetry, ...] = ()
    reference: _tb_type | DensityResult | None = None
    _h_0: Hamiltonian = field(init=False, repr=False)
    _h_int: _tb_type | BilinearInteraction = field(init=False, repr=False)
    _space: ActiveSCFSpace = field(init=False, repr=False)
    _reference_state: ActiveDensityState | None = field(init=False, repr=False)
    _ndim: int = field(init=False, repr=False)
    _ndof: int = field(init=False, repr=False)
    _hamiltonian: Hamiltonian = field(init=False, repr=False)

    def __post_init__(self):
        callable_h = isinstance(self.h_0, BlochHamiltonian)
        bilinear = isinstance(self.h_int, BilinearInteraction)
        h_0 = self.h_0 if callable_h else freeze_tb(self.h_0)
        h_int = self.h_int if bilinear else freeze_tb(self.h_int, real=True)
        ndim, ndof = hamiltonian_dimension(h_0), hamiltonian_size(h_0)
        interaction_size = h_int.ndof if bilinear else tb_orbital_count(h_int)
        if interaction_size != ndof or (not bilinear and tb_dimension(h_int) != ndim):
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
            h_0=h_0 if callable_h else _MatrixView(h_0),
            h_int=h_int if bilinear else _MatrixView(h_int),
            _h_0=h_0,
            _h_int=h_int,
            filling=filling,
            kT=kT,
            spatial_symmetries=symmetries,
            _ndim=ndim,
            _ndof=ndof,
            _hamiltonian=electron_to_bdg(h_0) if self.superconducting else h_0,
        ).items():
            object.__setattr__(self, name, value)
        space = ActiveSCFSpace.from_interaction(
            h_int,
            superconducting=self.superconducting,
            spatial_symmetries=symmetries,
            ndim=ndim,
            sparse=(not callable_h and prefers_sparse_storage(h_0))
            or (not bilinear and prefers_sparse_storage(h_int)),
        )
        reference_state = None
        if self.reference is not None:
            reference = self.reference
            if isinstance(reference, DensityResult):
                size = reference.coordinates.size
                read_values = reference.values_for
            else:
                if not isinstance(reference, Mapping):
                    raise TypeError(
                        "reference must be a density dictionary or DensityResult"
                    )
                reference = freeze_tb(reference)
                size = tb_orbital_count(reference)

                def read_values(coordinates):
                    return coordinates.values_from_tb(reference)

                object.__setattr__(self, "reference", _MatrixView(reference))
            coordinates = space.required_coordinates
            if size not in ((ndof, 2 * ndof) if self.superconducting else (ndof,)):
                raise ValueError("density coordinate matrix sizes do not match")
            if self.superconducting and size == ndof:
                # An electron-space reference specifies zero pairing. Read only
                # its required normal entries, without building Nambu matrices.
                normal = DensityCoordinates.from_pairs(
                    size=ndof,
                    keys=list(coordinates.keys),
                    pairs_by_key={
                        key: (rows[cols < ndof], cols[cols < ndof])
                        for key, rows, cols, _ in coordinates.iter_key_coordinates()
                    },
                )
                values = np.zeros(coordinates.value_count, dtype=complex)
                values[coordinates.all_cols < ndof] = read_values(normal)
            else:
                values = read_values(coordinates)
            reference_state = ActiveDensityState(
                space, space.params_from_required_entries(values)
            )
        object.__setattr__(self, "_space", space)
        object.__setattr__(self, "_reference_state", reference_state)

    @property
    def required_coordinates(self) -> DensityCoordinates:
        """Density entries needed to evaluate this model's interaction."""
        return self._space.required_coordinates

    @property
    def _electron_ndof(self) -> int | None:
        return self._ndof if self.superconducting else None

    def _density_state(self, rho: _tb_type | DensityResult) -> ActiveDensityState:
        if isinstance(rho, DensityResult):
            params = self._space.params_from_required_entries(
                rho.values_for(self.required_coordinates)
            )
        else:
            params = self._space.params_from_density(rho)
        return ActiveDensityState(
            self._space,
            params,
        )

    def _active_density_from_state(self, state: ActiveDensityState) -> _tb_type:
        require_same_space(state, self._space)
        return self._space.density_from_params(state.values)

    def _reference_difference(
        self,
        state: ActiveDensityState,
    ) -> ActiveDensityState:
        require_same_space(state, self._space)
        return state.relative_to(self._reference_state)

    def _mean_field_from_state(self, state: ActiveDensityState) -> _tb_type:
        active = self._active_density_from_state(self._reference_difference(state))
        return interaction_correction(
            active, self._h_int, electron_ndof=self._electron_ndof
        )

    def mean_field(self, density: _tb_type | DensityResult) -> _tb_type:
        """Return the interaction correction, including reference and pairing terms.

        The density must cover the model's required coordinates. The correction
        excludes the bare Hamiltonian and chemical-potential shift.
        """
        return self._mean_field_from_state(self._density_state(density))

    def hamiltonian_from_density(
        self, density: _tb_type | DensityResult
    ) -> Hamiltonian:
        """Build the normal or BdG Hamiltonian from a trial density.

        Selected results must cover this model's required coordinates.
        Subtract the reference normal and pairing densities before computing
        the correction; a normal reference contributes no pairing.
        """
        return add_correction(self._hamiltonian, self.mean_field(density))

    def hamiltonian_from_meanfield(
        self, mean_field: _tb_type | None = None
    ) -> Hamiltonian:
        """Build the unshifted normal or electron-first BdG Hamiltonian.

        Omitting ``mean_field`` returns the noninteracting Hamiltonian.
        Chemical potential is applied during density evaluation.
        """
        if mean_field is not None:
            self._validate_mean_field(mean_field)
        return add_correction(self._hamiltonian, mean_field or {})

    def _validate_mean_field(self, correction: _tb_type) -> None:
        if not correction:
            return
        if self.superconducting:
            validate_bdg_tb(correction, ndof=self._ndof, ndim=self._ndim)
        else:
            validate_tb_dict(correction)
            if (
                tb_dimension(correction) != self._ndim
                or tb_orbital_count(correction) != self._ndof
            ):
                raise ValueError(
                    "Mean-field correction must match the model dimension and matrix size"
                )
            validate_hermiticity(correction)

    def _project_mean_field(self, correction: _tb_type) -> _tb_type:
        projected = self._space.project_correction(correction)
        if not self.superconducting:
            return projected
        return assemble_bdg_tb(
            {
                key: block[: self._ndof, : self._ndof]
                for key, block in projected.items()
            },
            {
                key: block[: self._ndof, self._ndof :]
                for key, block in projected.items()
            },
            ndof=self._ndof,
        )

    def random_meanfield(self, rng=None, scale: float = 1.0) -> _tb_type:
        """Sample a solver-ready mean-field correction in this model's SCF space."""

        generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )
        params = float(scale) * generator.standard_normal(self._space.num_params)
        meanfield_input = self._space.density_from_params(params)
        return interaction_correction(
            meanfield_input, self._h_int, electron_ndof=self._electron_ndof
        )
