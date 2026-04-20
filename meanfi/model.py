from __future__ import annotations

from dataclasses import dataclass

from meanfi.mf import density_matrix, density_matrix_at_mu, meanfield
from meanfi.tb.tb import add_tb, _tb_type
from meanfi._validation import (
    tb_dimension,
    tb_orbital_count,
    validate_hermiticity,
    validate_tb_dict,
    zero_key,
)


@dataclass(frozen=True)
class _ModelPolicy:
    """High-level numerical policy stored on the interacting problem."""

    kT: float
    charge_tol: float
    density_atol: float
    scf_tol: float
    density_rtol: float = 0.0

    @property
    def mu_xtol(self) -> float:
        return self.charge_tol


class Model:
    """Interacting tight-binding problem at non-negative temperature.

    Parameters
    ----------
    h_0 :
        Non-interacting Hermitian Hamiltonian in tight-binding-dictionary form.
    h_int :
        Interaction Hamiltonian in tight-binding-dictionary form.
    filling :
        Number of particles in a unit cell.
    kT :
        Temperature in energy units.
    charge_tol, density_atol, scf_tol :
        High-level accuracy controls for fixed-filling density updates and the
        self-consistent field solver.
    """

    def __init__(
        self,
        h_0: _tb_type,
        h_int: _tb_type,
        filling: float,
        *,
        kT: float,
        charge_tol: float = 1e-4,
        density_atol: float = 1e-5,
        scf_tol: float = 1e-5,
    ) -> None:
        validate_tb_dict(h_0)
        validate_tb_dict(h_int)
        validate_hermiticity(h_0)
        validate_hermiticity(h_int)

        if not isinstance(filling, (float, int)) or filling <= 0:
            raise ValueError("filling must be a positive scalar")
        if kT < 0:
            raise ValueError("meanfi supports only non-negative temperatures (kT >= 0)")
        if charge_tol <= 0 or density_atol <= 0 or scf_tol <= 0:
            raise ValueError("tolerances must be positive")

        self.h_0 = h_0
        self.h_int = h_int
        self.filling = float(filling)
        self._policy = _ModelPolicy(
            kT=float(kT),
            charge_tol=float(charge_tol),
            density_atol=float(density_atol),
            scf_tol=float(scf_tol),
        )

        self._ndim = tb_dimension(h_0)
        self._ndof = tb_orbital_count(h_0)
        self._local_key = zero_key(self._ndim)

    @property
    def kT(self) -> float:
        return self._policy.kT

    @property
    def charge_tol(self) -> float:
        return self._policy.charge_tol

    @property
    def density_atol(self) -> float:
        return self._policy.density_atol

    @property
    def density_rtol(self) -> float:
        return self._policy.density_rtol

    @property
    def mu_xtol(self) -> float:
        return self._policy.mu_xtol

    @property
    def scf_tol(self) -> float:
        return self._policy.scf_tol

    def hamiltonian_from_rho(self, rho: _tb_type) -> _tb_type:
        """Return the interacting Hamiltonian implied by a trial density matrix."""

        return add_tb(self.h_0, meanfield(rho, self.h_int))

    def hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        """Return the full Hamiltonian for a trial mean-field correction."""

        return add_tb(self.h_0, mf)

    def density_matrix(
        self,
        rho: _tb_type,
        *,
        keys: list | None = None,
        mu_guess: float = 0.0,
    ):
        """Compute the fixed-filling density matrix for a trial density.

        The model-level accuracy policy is used for the entire solve. Advanced
        backend knobs remain available only through :func:`meanfi.density_matrix`.
        """

        resolved_keys = list(self.h_int) if keys is None else keys
        hamiltonian = self.hamiltonian_from_rho(rho)
        return density_matrix(
            hamiltonian,
            filling=self.filling,
            kT=self.kT,
            keys=resolved_keys,
            charge_tol=self.charge_tol,
            density_atol=self.density_atol,
            density_rtol=self.density_rtol,
            mu_guess=mu_guess,
            mu_xtol=self.mu_xtol,
        )

    def density_matrix_at_mu(
        self,
        rho: _tb_type,
        *,
        mu: float,
        keys: list | None = None,
    ):
        """Compute the density matrix at an explicit chemical potential."""

        resolved_keys = list(self.h_int) if keys is None else keys
        hamiltonian = self.hamiltonian_from_rho(rho)
        return density_matrix_at_mu(
            hamiltonian,
            mu=mu,
            kT=self.kT,
            keys=resolved_keys,
            density_atol=self.density_atol,
            density_rtol=self.density_rtol,
        )
