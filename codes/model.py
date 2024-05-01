import numpy as np

from codes.mf import (
    density_matrix,
    meanfield,
)
from codes.tb.tb import add_tb


class Model:
    def __init__(self, h_0, h_int, filling):
        self.h_0 = h_0
        self.h_int = h_int
        self.filling = filling

        _first_key = list(h_0)[0]
        self._ndim = len(_first_key)
        self._size = h_0[_first_key].shape[0]
        self._local_key = tuple(np.zeros((self._ndim,), dtype=int))

        def _check_hermiticity(h):
            # assert hermiticity of the Hamiltonian
            # assert hermiticity of the Hamiltonian
            for vector in h.keys():
                op_vector = tuple(-1 * np.array(vector))
                op_vector = tuple(-1 * np.array(vector))
                assert np.allclose(h[vector], h[op_vector].conj().T)

        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

    def mfield(self, mf_tb, nk=200):  # method or standalone?
        rho, fermi_energy = density_matrix(
            add_tb(self.h_0, mf_tb), self.filling, nk
        )
        return add_tb(
            meanfield(rho, self.h_int),
            {self._local_key: -fermi_energy * np.eye(self._size)},
        )
