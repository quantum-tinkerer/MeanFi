from .tb.tb import addTb
from .tb.transforms import tb2kfunc
from .mf import densityMatrixGenerator, meanFieldFFT, meanFieldQuad, fermiOnGrid
import numpy as np


class Model:
    def __init__(self, tb_model, int_model, filling):
        self.tb_model = tb_model
        self.int_model = int_model
        self.filling = filling

        _firstKey = list(tb_model)[0]
        self._ndim = len(_firstKey)
        self._size = tb_model[_firstKey].shape[0]
        self._localKey = tuple(np.zeros((self._ndim,), dtype=int))

    def makeDensityMatrix(self, mf_model, nK=200):
        self.hkfunc = tb2kfunc(addTb(self.tb_model, mf_model))
        self.calculateEF(nK=nK)
        return densityMatrixGenerator(self.hkfunc, self.EF)

    def calculateEF(self, nK=200):
        self.EF = fermiOnGrid(self.hkfunc, self.filling, nK=nK, ndim=self._ndim)

    def mfield(self, mf_model):
        self.densityMatrix = self.makeDensityMatrix(mf_model)
        return addTb(
            meanFieldQuad(self.densityMatrix, self.int_model),
            {self._localKey: -self.EF * np.eye(self._size)},
        )

    def mfieldFFT(self, mf_model, nK=200):
        self.densityMatrix = self.makeDensityMatrix(mf_model, nK=nK)
        return addTb(
            meanFieldFFT(self.densityMatrix, self.int_model, n=self._ndim, nK=nK),
            {self._localKey: -self.EF * np.eye(self._size)},
        )
