# %%
from codes.tb.tb import addTb
from codes.tb.transforms import tb2kham, ifftn2tb
from codes.mf import (
    densityMatrix,
    fermiOnGrid,
    meanField,
)
import numpy as np
from scipy.fftpack import ifftn


class Model:
    def __init__(self, h_0, h_int, filling):
        self.h_0 = h_0
        self.h_int = h_int
        self.filling = filling

        _firstKey = list(h_0)[0]
        self._ndim = len(_firstKey)
        self._size = h_0[_firstKey].shape[0]
        self._localKey = tuple(np.zeros((self._ndim,), dtype=int))

        def _check_hermiticity(h):
            # assert hermiticity of the Hamiltonian
            # assert hermiticity of the Hamiltonian
            for vector in h.keys():
                op_vector = tuple(-1 * np.array(vector))
                op_vector = tuple(-1 * np.array(vector))
                assert np.allclose(h[vector], h[op_vector].conj().T)

        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

    def calculateEF(self):
        self.EF = fermiOnGrid(self.kham, self.filling)

    def makeDensityMatrixTb(self, mf_model, nK=200):
        self.kham = tb2kham(addTb(self.h_0, mf_model), nK=nK, ndim=self._ndim)
        self.calculateEF()
        return ifftn2tb(
            ifftn(densityMatrix(self.kham, self.EF), axes=np.arange(self._ndim))
        )

    def mfield(self, mf_model, nK=200):
        densityMatrixTb = self.makeDensityMatrixTb(mf_model, nK=nK)
        return addTb(
            meanField(densityMatrixTb, self.h_int, n=self._ndim),
            {self._localKey: -self.EF * np.eye(self._size)},
        )
