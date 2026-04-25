from __future__ import annotations

from meanfi._bdg import electron_to_bdg_tb, validate_bdg_tb
from meanfi._validation import (
    tb_dimension,
    tb_orbital_count,
    validate_hermiticity,
    validate_tb_dict,
    zero_key,
)
from meanfi.mf import meanfield
from meanfi.tb.tb import add_tb, _tb_type


class Model:
    """Interacting tight-binding problem at non-negative temperature."""

    def __init__(
        self,
        h_0: _tb_type,
        h_int: _tb_type,
        filling: float,
        *,
        kT: float,
        superconducting: bool = False,
    ) -> None:
        validate_tb_dict(h_0)
        validate_tb_dict(h_int)
        validate_hermiticity(h_0)
        validate_hermiticity(h_int)

        if not isinstance(filling, (float, int)) or filling <= 0:
            raise ValueError("filling must be a positive scalar")
        if kT < 0:
            raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")

        self.h_0 = h_0
        self.h_int = h_int
        self.filling = float(filling)
        self.kT = float(kT)
        self.superconducting = bool(superconducting)

        self._ndim = tb_dimension(h_0)
        self._ndof = tb_orbital_count(h_0)
        self._local_key = zero_key(self._ndim)

    def hamiltonian_from_rho(self, rho: _tb_type) -> _tb_type:
        """Return the interacting Hamiltonian implied by a trial density matrix."""

        return add_tb(self.h_0, meanfield(rho, self.h_int))

    def hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        """Return the full Hamiltonian for a trial mean-field correction."""

        return add_tb(self.h_0, mf)

    def bdg_hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        """Return the unshifted electron-first BdG Hamiltonian for a mean-field correction."""

        if not self.superconducting:
            raise ValueError("bdg_hamiltonian_from_meanfield requires superconducting=True")
        validate_bdg_tb(
            mf,
            ndof=self._ndof,
            ndim=self._ndim,
            name="BdG correction",
        )
        return add_tb(electron_to_bdg_tb(self.h_0, self._ndof), mf)
