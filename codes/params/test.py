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
from codes.params.rparams import mf2rParams, rParams2mf
from codes.kwant_helper.utils import generate_guess

def compareDicts(dict1, dict2):
    for key in dict1.keys():
        assert np.allclose(dict1[key], dict2[key])


# %%
ndof = 10
vectors = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1))

mf_guess = generate_guess(vectors, ndof)
mf_params = mf2rParams(mf_guess)
mf_new = rParams2mf(mf_params, vectors, ndof)

compareDicts(mf_guess, mf_new)

# %%
