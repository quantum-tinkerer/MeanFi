import numpy as np

from mf import (
    density_matrix,
    meanfield,
)
from tb.tb import add_tb, _tb_type


def _check_hermiticity(h):
    for vector in h.keys():
        op_vector = tuple(-1 * np.array(vector))
        op_vector = tuple(-1 * np.array(vector))
        if not np.allclose(h[vector], h[op_vector].conj().T):
            raise ValueError("Tight-binding dictionary must be hermitian.")


def _charge_op_check(Q, ndof, target_charge):
    if not Q.shape == (ndof, ndof):
        raise ValueError(f"Operator shape does not match expected: ({ndof}, {ndof})")

    if not isinstance(target_charge, (float, int)):
        raise ValueError("Target charge must be a float or an integer.")

    Q_vals = np.linalg.eigvalsh(Q)
    Q_pos = Q_vals[Q_vals > 0].sum()
    Q_neg = Q_vals[Q_vals < 0].sum()

    if (target_charge > Q_pos - 0.01) or (
        target_charge < Q_neg + 0.01
    ):  # The 0.01 is somewhat arbitrary.
        raise ValueError(
            f"Target charge can not fall outside of possible range: ({Q_neg + 0.01}, {Q_pos - 0.01})"
        )


def _tb_type_check(tb):
    for count, key in enumerate(tb):
        if not isinstance(tb[key], np.ndarray):
            raise ValueError(
                "Values of the tight-binding dictionary must be numpy arrays"
            )
        shape = tb[key].shape
        if count == 0:
            size = shape[0]
        if not len(shape) == 2:
            raise ValueError(
                "Values of the tight-binding dictionary must be square matrices"
            )
        if not size == shape[0]:
            raise ValueError(
                "Values of the tight-binding dictionary must have consistent shape"
            )


class Model:
    """
    Data class which defines the interacting tight-binding problem.

    Parameters
    ----------
    h_0 :
        Non-interacting hermitian Hamiltonian tight-binding dictionary.
    h_int :
        Interaction hermitian Hamiltonian tight-binding dictionary.
    charge_op:
        Charge operator for the Model.
    target_charge :
        Charge of a unit cell.
        Used to determine the Fermi level.
    kT:
        Dimensionless temperature of the system.

    Notes
    -----

    The interaction h_int must be of density-density type.
    For example, h_int[(1,)][i, j] = V means a repulsive interaction
    of strength V between two particles with internal degrees of freedom i and j
    separated by 1 lattice vector.
    """

    def __init__(
        self,
        h_0: _tb_type,
        h_int: _tb_type,
        charge_op: np.ndarray,
        target_charge: float,
        kT: float,
    ) -> None:
        _tb_type_check(h_0)
        _check_hermiticity(h_0)
        self.h_0 = h_0

        _tb_type_check(h_int)
        _check_hermiticity(h_int)
        self.h_int = h_int

        if not kT >= 0:
            raise ValueError("Temperature must be a positive value.")
        self.kT = kT

        _first_key = list(h_0)[0]
        self._ndim = len(_first_key)
        self._ndof = h_0[_first_key].shape[0]
        self._local_key = tuple(np.zeros((self._ndim,), dtype=int))

        _charge_op_check(charge_op, self._ndof, target_charge)
        self.target_charge = target_charge
        self.charge_op = charge_op

    def density_matrix(self, rho: _tb_type, nk: int = 20) -> _tb_type:
        """Computes the density matrix from a given initial density matrix.

        Parameters
        ----------
        rho :
            Initial density matrix tight-binding dictionary.
        nk :
            Number of k-points in a grid to sample the Brillouin zone along each dimension.
            If the system is 0-dimensional (finite), this parameter is ignored.

        Returns
        -------
        :
            Density matrix tight-binding dictionary.
        """
        mf = meanfield(rho, self.h_int)

        return density_matrix(
            add_tb(self.h_0, mf), self.charge_op, self.target_charge, self.kT, nk
        )[0]

    def mfield(self, mf: _tb_type, nk: int = 20) -> _tb_type:
        """Computes a new mean-field correction from a given one.

        Parameters
        ----------
        mf :
            Initial mean-field correction tight-binding dictionary.
        nk :
            Number of k-points in a grid to sample the Brillouin zone along each dimension.
            If the system is 0-dimensional (finite), this parameter is ignored.

        Returns
        -------
        :
            new mean-field correction tight-binding dictionary.
        """
        rho, fermi_level = density_matrix(
            add_tb(self.h_0, mf), self.charge_op, self.target_charge, self.kT, nk
        )
        return add_tb(
            meanfield(rho, self.h_int),
            {self._local_key: -fermi_level * np.eye(self._ndof)},
        )
