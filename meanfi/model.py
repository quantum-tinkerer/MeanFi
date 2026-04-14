import numpy as np

from meanfi.mf import density_matrix, density_matrix_at_mu, meanfield
from meanfi.tb.tb import add_tb, _tb_type


def _check_hermiticity(h: _tb_type) -> None:
    for vector in h.keys():
        op_vector = tuple(-1 * np.array(vector))
        if not np.allclose(h[vector], h[op_vector].conj().T):
            raise ValueError("Tight-binding dictionary must be hermitian.")


def _tb_type_check(tb: _tb_type) -> None:
    for count, key in enumerate(tb):
        if not isinstance(tb[key], np.ndarray):
            raise ValueError("Values of the tight-binding dictionary must be numpy arrays")
        shape = tb[key].shape
        if count == 0:
            size = shape[0]
        if len(shape) != 2:
            raise ValueError("Values of the tight-binding dictionary must be square matrices")
        if size != shape[0]:
            raise ValueError(
                "Values of the tight-binding dictionary must have consistent shape"
            )


class Model:
    """Interacting tight-binding problem at finite temperature."""

    def __init__(
        self,
        h_0: _tb_type,
        h_int: _tb_type,
        filling: float,
        *,
        kT: float,
        charge_tol: float = 1e-8,
        density_atol: float = 1e-8,
        density_rtol: float = 0.0,
        mu_xtol: float = 1e-8,
        scf_tol: float = 1e-6,
    ) -> None:
        _tb_type_check(h_0)
        _tb_type_check(h_int)
        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

        if not isinstance(filling, (float, int)) or filling <= 0:
            raise ValueError("filling must be a positive scalar")
        if kT <= 0:
            raise ValueError("meanfi supports only finite temperature (kT > 0)")
        if charge_tol <= 0 or density_atol <= 0 or mu_xtol <= 0 or scf_tol <= 0:
            raise ValueError("tolerances must be positive")
        if density_rtol < 0:
            raise ValueError("density_rtol must be non-negative")

        self.h_0 = h_0
        self.h_int = h_int
        self.filling = float(filling)
        self.kT = float(kT)
        self.charge_tol = float(charge_tol)
        self.density_atol = float(density_atol)
        self.density_rtol = float(density_rtol)
        self.mu_xtol = float(mu_xtol)
        self.scf_tol = float(scf_tol)

        first_key = list(h_0)[0]
        self._ndim = len(first_key)
        self._ndof = h_0[first_key].shape[0]
        self._local_key = tuple(np.zeros((self._ndim,), dtype=int))

    def hamiltonian_from_rho(self, rho: _tb_type) -> _tb_type:
        return add_tb(self.h_0, meanfield(rho, self.h_int))

    def hamiltonian_from_meanfield(self, mf: _tb_type) -> _tb_type:
        return add_tb(self.h_0, mf)

    def density_matrix(
        self,
        rho: _tb_type,
        *,
        keys: list | None = None,
        mu_guess: float = 0.0,
        charge_tol: float | None = None,
        density_atol: float | None = None,
        density_rtol: float | None = None,
        mu_xtol: float | None = None,
        max_mu_iterations: int = 32,
        max_subdivisions: int | None = 10_000,
        rule: str = "auto",
        batch_size: int | None = None,
    ):
        """Compute the fixed-filling density matrix for a trial density."""
        keys = list(self.h_int) if keys is None else keys
        h = self.hamiltonian_from_rho(rho)
        return density_matrix(
            h,
            filling=self.filling,
            kT=self.kT,
            keys=keys,
            charge_tol=self.charge_tol if charge_tol is None else charge_tol,
            density_atol=self.density_atol if density_atol is None else density_atol,
            density_rtol=self.density_rtol if density_rtol is None else density_rtol,
            mu_guess=mu_guess,
            mu_xtol=self.mu_xtol if mu_xtol is None else mu_xtol,
            max_mu_iterations=max_mu_iterations,
            max_subdivisions=max_subdivisions,
            rule=rule,
            batch_size=batch_size,
        )

    def density_matrix_at_mu(
        self,
        rho: _tb_type,
        *,
        mu: float,
        keys: list | None = None,
        density_atol: float | None = None,
        density_rtol: float | None = None,
        max_subdivisions: int | None = 10_000,
        rule: str = "auto",
        batch_size: int | None = None,
    ):
        """Compute the density matrix at an explicit chemical potential."""
        keys = list(self.h_int) if keys is None else keys
        h = self.hamiltonian_from_rho(rho)
        return density_matrix_at_mu(
            h,
            mu=mu,
            kT=self.kT,
            keys=keys,
            density_atol=self.density_atol if density_atol is None else density_atol,
            density_rtol=self.density_rtol if density_rtol is None else density_rtol,
            max_subdivisions=max_subdivisions,
            rule=rule,
            batch_size=batch_size,
        )
