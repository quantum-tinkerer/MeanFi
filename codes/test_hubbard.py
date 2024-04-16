# %%
import numpy as np
from codes.solvers import solver
from codes.tb import utils
from codes.tb.tb import add_tb
from codes.model import Model
import xarray as xr
import pytest

repeat_number = 10


# %%
def gap_relation_hubbard(Us, nk, nk_dense, tol=1e-3):
    hopp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(2))
    h_0 = {(0,): hopp + hopp.T.conj(), (1,): hopp, (-1,): hopp.T.conj()}
    gaps = []
    for U in Us:
        h_int = {
            (0,): U * np.kron(np.ones((2, 2)), np.eye(2)),
        }
        guess = utils.generate_guess(frozenset(h_int), len(list(h_0.values())[0]))
        full_model = Model(h_0, h_int, filling=2)
        mf_sol = solver(full_model, guess, nk=nk)
        _gap = utils.compute_gap(add_tb(h_0, mf_sol), fermi_energy=0, nk=nk_dense)
        gaps.append(_gap)

    ds = xr.Dataset(
        data_vars=dict(gap=(["Us"], gaps)),
        coords=dict(
            Us=Us,
        ),
    )

    fit_gap = ds.gap.polyfit(dim="Us", deg=1).polyfit_coefficients[0].data
    assert np.abs(fit_gap - 1) < tol


@pytest.mark.repeat(repeat_number)
def test_gap_hubbard():
    Us = np.linspace(0.5, 10, 20, endpoint=True)
    gap_relation_hubbard(Us, nk=20, nk_dense=100, tol=1e-2)
