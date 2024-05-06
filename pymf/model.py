import numpy as np

from pymf.mf import (
    construct_density_matrix,
    meanfield,
)
from pymf.tb.tb import add_tb, tb_type


def _check_hermiticity(h):
    for vector in h.keys():
        op_vector = tuple(-1 * np.array(vector))
        op_vector = tuple(-1 * np.array(vector))
        if not np.allclose(h[vector], h[op_vector].conj().T):
            raise ValueError("Hamiltonian is not Hermitian.")


def _tb_type_check(tb):
    for count, key in enumerate(tb):
        if not isinstance(tb[key], np.ndarray):
            raise ValueError("Inputted dictionary values are not np.ndarray's")
        shape = tb[key].shape
        if count == 0:
            size = shape[0]
        if not len(shape) == 2:
            raise ValueError("Inputted dictionary values are not square matrices")
        if not size == shape[0]:
            raise ValueError("Inputted dictionary elements shapes are not consistent")


class Model:
    """
    Data class which defines the mean-field tight-binding problem
    and computes the mean-field Hamiltonian.

    Parameters
    ----------
    h_0 :
        Non-interacting Hamiltonian.
    h_int :
        Interaction Hamiltonian.
    filling :
        Filling of the system.

    Attributes
    ----------
    h_0 :
        Non-interacting Hamiltonian.
    h_int :
        Interaction Hamiltonian.
    filling :
        Filling of the system.
    """

    def __init__(self, h_0: tb_type, h_int: tb_type, filling: float) -> None:
        _tb_type_check(h_0)
        self.h_0 = h_0
        _tb_type_check(h_int)
        self.h_int = h_int
        if not isinstance(filling, (float, int)):
            raise ValueError("Filling must be a float or an integer")
        if not filling > 0:
            raise ValueError("Filling must be a positive value")
        self.filling = filling

        _first_key = list(h_0)[0]
        self._ndim = len(_first_key)
        self._size = h_0[_first_key].shape[0]
        self._local_key = tuple(np.zeros((self._ndim,), dtype=int))

        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

    def mfield(self, mf_tb: tb_type, nk: int = 200) -> tb_type:
        """Compute single mean field iteration.

        Parameters
        ----------
        mf_tb :
            Mean-field tight-binding model.
        nk :
            Number of k-points in the grid along a single direction.
            If the system is 0-dimensional (finite), this parameter is ignored.

        Returns
        -------
        :
            New mean-field tight-binding model.
        """
        rho, fermi_energy = construct_density_matrix(
            add_tb(self.h_0, mf_tb), self.filling, nk
        )
        return add_tb(
            meanfield(rho, self.h_int),
            {self._local_key: -fermi_energy * np.eye(self._size)},
        )
