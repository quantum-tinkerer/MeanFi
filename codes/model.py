from .tb.tb import addTb
from .tb.transforms import tb2kfunc, tb2kham, kdens2tbFFT
from .mf import (
    densityMatrixGenerator,
    densityMatrix,
    fermiOnGridkvector,
    meanFieldFFTkvector,
    meanFieldFFT,
    meanFieldQuad,
    fermiOnGrid,
)
import numpy as np


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
            for vector in h.keys():
                op_vector = tuple(-1 * np.array(vector))
                assert np.allclose(h[vector], h[op_vector].conj().T)

        _check_hermiticity(h_0)
        _check_hermiticity(h_int)

    def makeDensityMatrix(self, mf_model, nK=200):
        self.hkfunc = tb2kfunc(addTb(self.h_0, mf_model))
        self.calculateEF(nK=nK)
        return densityMatrixGenerator(self.hkfunc, self.EF)

    def calculateEF(self, nK=200):
        self.EF = fermiOnGrid(self.hkfunc, self.filling, nK=nK, ndim=self._ndim)

    def mfield(self, mf_model):
        self.densityMatrix = self.makeDensityMatrix(mf_model)
        return addTb(
            meanFieldQuad(self.densityMatrix, self.h_int),
            {self._localKey: -self.EF * np.eye(self._size)},
        )

    def mfieldFFT(self, mf_model, nK=200):
        self.densityMatrix = self.makeDensityMatrix(mf_model, nK=nK)
        return addTb(
            meanFieldFFT(self.densityMatrix, self.h_int, n=self._ndim, nK=nK),
            {self._localKey: -self.EF * np.eye(self._size)},
        )

    #######################
    def calculateEFkvector(self):
        self.EF = fermiOnGridkvector(self.kham, self.filling)

    def makeDensityMatrixkvector(self, mf_model, nK=200):
        self.kham = tb2kham(addTb(self.h_0, mf_model), nK=nK, ndim=self._ndim)
        self.calculateEF()
        return densityMatrix(self.kham, self.EF)

    def mfieldFFTkvector(self, mf_model, nK=200):
        densityMatrix = self.makeDensityMatrixkvector(mf_model, nK=nK)
        densityMatrixTb = kdens2tbFFT(densityMatrix, nK)
        return addTb(
            meanFieldFFTkvector(densityMatrixTb, self.h_int, n=self._ndim),
            {self._localKey: -self.EF * np.eye(self._size)},
        )
