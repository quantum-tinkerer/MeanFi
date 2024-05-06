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
            raise ValueError("Tight-binding dictionary must be hermitian.")


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
    filling :
        Number of particles in a unit cell.
        Used to determine the Fermi level.

    Notes
    -----
    The interaction h_int must be of density-density type.
    For example in 1D system with ndof internal degrees of freedom,
    h_int[(2,)] = U * np.ones((ndof, ndof)) is a Coulomb repulsion interaction
    with strength U between unit cells separated by 2 lattice vectors, where
    the interaction is the same between all internal degrees of freedom.
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
        self._ndof = h_0[_first_key].shape[0]
        self._local_key = tuple(np.zeros((self._ndim,), dtype=int))

        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

    def mfield(self, mf: tb_type, nk: int = 200) -> tb_type:
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
        rho, fermi_energy = construct_density_matrix(
            add_tb(self.h_0, mf), self.filling, nk
        )
        return add_tb(
            meanfield(rho, self.h_int),
            {self._local_key: -fermi_energy * np.eye(self._ndof)},
        )
