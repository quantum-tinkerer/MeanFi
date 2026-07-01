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
    reference_subtracted_density,
)
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


class Model:
    """Interacting tight-binding problem at non-negative temperature.

    ``reference_density_matrix`` enables full normal-state reference subtraction:
    the interaction correction is built from ``rho - rho_ref`` instead of
    ``rho``. This subtracts both Hartree and exchange-like mean-field terms.
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
        if reference_density_matrix is not None and superconducting:
            raise ValueError(
                "reference_density_matrix is only supported for normal-state models"
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
        if reference_density_matrix is not None:
            _validate_reference_density_matrix(
                reference_density_matrix,
                ndim=self._ndim,
                ndof=self._ndof,
            )

        from meanfi.space import ActiveSCFSpace

        object.__setattr__(self, "scf_space", ActiveSCFSpace.from_model(self))
        if reference_density_matrix is None:
            reference = None
        else:
            reference = MappingProxyType(
                dict(self.scf_space.project_meanfield_input(reference_density_matrix))
            )
        object.__setattr__(self, "reference_density_matrix", reference)
        object.__setattr__(self, "_frozen", True)

    def hamiltonian_from_rho(self, rho: _tb_type) -> _tb_type:
        """Return the interacting Hamiltonian implied by a trial density matrix."""

        if self.reference_density_matrix is None:
            correction = meanfield(rho, self.h_int)
        else:
            active_density = self.scf_space.project_meanfield_input(rho)
            density_difference = reference_subtracted_density(
                active_density,
                self.reference_density_matrix,
                interaction_keys=self.scf_space.interaction_keys,
                onsite=self.scf_space.onsite,
                ndof=self._ndof,
            )
            correction = meanfield(density_difference, self.h_int)
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
