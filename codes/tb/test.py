# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %%
import numpy as np
from codes.params.test import compareDicts
from functools import partial
import itertools as it
from codes.tb.transforms import kfunc2tbFFT, tb2kfunc

# %%
ndim = 2
maxOrder = 5
matrixSize = 5
nK = 10

keys = [np.arange(-maxOrder+1, maxOrder) for i in range(ndim)]
keys = it.product(*keys)
tb_model = {key : (np.random.rand(matrixSize, matrixSize)-0.5)*2 for key in keys}
kfunc = tb2kfunc(tb_model)

tb_new = kfunc2tbFFT(kfunc, nK, ndim=ndim)
compareDicts(tb_model, tb_new)

# %%
