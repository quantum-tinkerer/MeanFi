# %%
from codes.tb.tb import add_tb
from codes.tb.transforms import tb_to_khamvector, ifftn_to_tb
from codes.mf import (
    density_matrix,
    fermi_on_grid,
    meanfield,
)
import numpy as np
from scipy.fftpack import ifftn


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
        density_matrix_tb, fermi_energy = rho(
            add_tb(self.h_0, mf_tb), self.filling, nk, self._ndim
        )
        return add_tb(
            meanfield(density_matrix_tb, self.h_int, n=self._ndim),
            {self._local_key: -fermi_energy * np.eye(self._size)},
        )


def rho(h, filling, nk, ndim):
    if ndmin > 0:
        kham = tb_to_khamvector(h, nk=nk, ndim=ndim)
        fermi = fermi_on_grid(kham, filling)
        ndim = len(kham.shape) - 2
        return (
            ifftn_to_tb(ifftn(density_matrix(kham, fermi), axes=np.arange(ndim))),
            fermi,
        )
    else:
        fermi = fermi_on_grid(h[()], filling)
        return {(): density_matrix(h[()], fermi)}
