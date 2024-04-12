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

    def calculate_EF(self):
        self.fermi_energy = fermi_on_grid(self.kham, self.filling)

    def make_density_matrix_tb(self, mf_tb, nk=200):
        self.kham = tb_to_khamvector(add_tb(self.h_0, mf_tb), nk=nk, ndim=self._ndim)
        self.calculate_EF()
        return ifftn_to_tb(
            ifftn(
                density_matrix(self.kham, self.fermi_energy), axes=np.arange(self._ndim)
            )
        )

    def mfield(self, mf_tb, nk=200):
        density_matrix_tb = self.make_density_matrix_tb(mf_tb, nk=nk)
        return add_tb(
            meanfield(density_matrix_tb, self.h_int, n=self._ndim),
            {self._local_key: -self.fermi_energy * np.eye(self._size)},
        )
